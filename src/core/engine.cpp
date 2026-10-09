#include "liyab/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>

#include "core/log.h"
#include "core/quant.h"
#include "core/sampling.h"
#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/direct_io.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"
#include "liyab/speculative_decoder.h"

#include "liyab/triple_buffer_loader.h"

#if defined(LIYAB_ENABLE_EXPERIMENTAL)
#include "liyab/experimental/early_exit.h"
#include "liyab/experimental/egls.h"
#include "liyab/experimental/tdss.h"
#include "liyab/experimental/kv_dedup.h"
#include "liyab/experimental/head_pruner.h"
#endif

namespace liyab {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Length of the longest prefix of `s` that ends on a UTF-8 character
// boundary, so callers (Swift String, Kotlin String) never see half a glyph.
size_t complete_utf8_prefix(std::string_view s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= std::min<size_t>(4, n); ++back) {
        const auto c = static_cast<unsigned char>(s[n - back]);
        if ((c & 0xC0) == 0x80) continue;  // continuation byte: keep looking for the lead
        size_t need = 1;
        if ((c & 0xE0) == 0xC0) need = 2;
        else if ((c & 0xF0) == 0xE0) need = 3;
        else if ((c & 0xF8) == 0xF0) need = 4;
        return back >= need ? n : n - back;
    }
    return n;  // only continuation bytes: malformed input, pass it through
}

// `repack_cpu`: the CPU computes every block (no GPU / NPU route), so its
// i8mm kernels may get their own weight layout (TransformerOptions::repack_cpu).
Result<std::unique_ptr<Transformer>> load_model(const std::string& path, const EngineConfig& config,
                                                int32_t max_batch, bool repack_cpu) {
    LoaderOptions loader_options;
    loader_options.streaming = config.streaming;
    loader_options.memory_budget_bytes =
        config.memory_budget_mb > 0 ? static_cast<uint64_t>(config.memory_budget_mb) << 20 : 0;
    auto file = MmapLoader::open(path, loader_options);
    if (!file) return file.status();
    TransformerOptions options;
    options.context_length = config.context_length;
    options.sliding_window = config.sliding_window;
    options.sink_tokens = config.kv_sink_tokens;
    options.kv_type = config.kv_cache_type;
    options.max_batch = max_batch;
    options.expert_cache_bytes = config.expert_cache_mb > 0 ? config.expert_cache_mb << 20 : config.expert_cache_mb;
    options.memory_budget_bytes = loader_options.memory_budget_bytes;
    options.requant_bits = config.requant_bits;
    options.expert_mass = config.moe_expert_mass;
    options.max_experts = config.moe_max_experts;
    options.skip_slow = config.moe_skip_slow;
    options.repack_cpu = repack_cpu && !config.triple_buffer_loading;
    return Transformer::load(std::move(file).value(), options);
}

}  // namespace

struct Engine::Impl {
    EngineConfig config;
    DeviceInfo device;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<Backend> cpu;
    std::unique_ptr<Backend> gpu;  // Metal today; Vulkan once implemented
    std::unique_ptr<Backend> npu;  // QNN / NeuroPilot once implemented
    std::unique_ptr<Transformer> target;
    std::unique_ptr<Transformer> draft;
    std::unique_ptr<DirectFile> direct_file;  // must outlive weight_loader
    std::atomic<uint64_t> tokens_generated{0};  // Engine::counters()
    // Tokens whose KV / recurrent state the target holds, in order
    // (target->n_past() == context.size()); context lookup drafts from it. Not
    // used with a draft model, whose cache moves in step with the target's
    // during speculation.
    std::vector<int32_t> context;
    // The text `context` encodes, when known (valid): a prompt given as text
    // followed by the pieces of the reply tokens generated after it.
    std::string context_text;
    bool context_text_valid = false;

    void clear_context() {
        target->reset();
        if (draft) draft->reset();
        context.clear();
        context_text_valid = false;
    }

    // Tokens for `text`. When it starts with the text the context holds, the
    // context's own tokens followed by the encoding of the rest: re-encoding
    // the previous reply need not give back the tokens that were generated,
    // and on a hybrid model any difference would recompute from the last
    // state snapshot.
    Result<std::vector<int32_t>> encode_continuation(std::string_view text, bool add_bos) const {
        const bool bos = add_bos && tokenizer->add_bos_default();
        if (!draft && context_text_valid && !context.empty() && text.size() > context_text.size() &&
            text.starts_with(context_text) &&
            (static_cast<unsigned char>(text[context_text.size()]) & 0xC0) != 0x80) {  // not mid-character
            auto rest = tokenizer->encode_continuation(text.substr(context_text.size()), context.back());
            if (!rest) return rest.status();
            std::vector<int32_t> tokens = context;
            tokens.insert(tokens.end(), rest->begin(), rest->end());
            return tokens;
        }
        return tokenizer->encode(text, bos);
    }

    // After a run over `prompt` (the encoding of `text`): records the text the
    // context now holds, if the context still starts with the whole prompt.
    void remember_text(std::span<const int32_t> prompt, std::string_view text) {
        context_text_valid = context.size() >= prompt.size() && std::equal(prompt.begin(), prompt.end(), context.begin());
        if (!context_text_valid) return;
        context_text.assign(text);
        for (size_t i = prompt.size(); i < context.size(); ++i) context_text += tokenizer->piece(context[i]);
    }

    // Keeps the longest usable prefix of `tokens` already in the context and
    // returns its length (at most `limit`); drops everything else.
    size_t reuse_prefix(std::span<const int32_t> tokens, size_t limit) {
        if (draft || context.size() != static_cast<size_t>(target->n_past())) {
            clear_context();
            return 0;
        }
        size_t common = 0;
        const size_t n = std::min({context.size(), tokens.size(), limit});
        while (common < n && context[common] == tokens[common]) ++common;
        if (common == context.size()) return common;  // pure continuation: nothing to drop
        // Attention-only models rewind to any position; hybrid ones to the
        // nearest recurrent-state checkpoint or snapshot below it.
        const auto keep = static_cast<size_t>(target->restorable_prefix(static_cast<int32_t>(common)));
        if (keep > 0 && target->truncate(static_cast<int32_t>(keep)).is_ok()) {
            context.resize(keep);
            context_text_valid = false;
            return keep;
        }
        clear_context();
        return 0;
    }
    std::unique_ptr<TripleBufferLoader> weight_loader;
    std::unique_ptr<SpeculativeDecoder> speculative;
    std::optional<Tokenizer> tokenizer;
    std::unique_ptr<PowerManager> power;
    std::atomic<bool> cancel{false};
    std::mutex busy;
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
    std::unique_ptr<experimental::EarlyExit> early_exit;
    std::optional<experimental::HeadPruner> head_pruner;
    std::unique_ptr<experimental::Egls> egls;
    std::unique_ptr<experimental::Tdss> tdss;
    std::unique_ptr<experimental::KvDedup> kv_dedup;
#endif

    // Experimental hooks for one forward pass given the current power policy.
    ForwardHooks hooks(const PowerPolicy& policy, bool decoding) {
        ForwardHooks h;
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
        if (decoding && early_exit) h.early_exit = early_exit.get();
        if (decoding && egls) h.ffn_skip = egls.get();
        if (tdss && tdss->update(policy)) h.ffn_matmul = tdss.get();
        if (head_pruner && head_pruner->update(policy, power->profile())) h.head_mask = &*head_pruner;
#else
        (void)policy;
        (void)decoding;
#endif
        return h;
    }

    // Heterogeneous routing: attention projections on the GPU, FFN + output
    // head on the NPU (highest TOPS/W for large integer matmuls), each
    // falling back down the NPU → GPU → CPU chain.
    [[nodiscard]] Route normal_route() const {
        Backend* attention = gpu ? gpu.get() : cpu.get();
        Backend* ffn = npu ? npu.get() : attention;
        return {attention, ffn, cpu.get()};
    }
    // Thermal guard: take the GPU out of the loop (it is the hottest block on
    // phone SoCs); heavy layers go to the NPU if present, else the CPU, which
    // also runs with fewer cores under the power policy.
    [[nodiscard]] Route throttled_route() const {
        Backend* target_backend = npu ? npu.get() : cpu.get();
        return {target_backend, target_backend, cpu.get()};
    }
    [[nodiscard]] Route draft_route() const { return {cpu.get(), cpu.get(), cpu.get()}; }

    Status select_backends() {
        std::vector<BackendKind> order = device.backend_order;
        if (config.backend) order = {*config.backend, BackendKind::Cpu};
        for (const BackendKind kind : order) {
            switch (kind) {
                case BackendKind::Cpu:
                    break;
                case BackendKind::Metal: {
                    if (gpu) break;
                    auto metal = make_metal_backend();
                    if (metal) {
                        gpu = std::move(metal).value();
                    } else if (config.backend == BackendKind::Metal) {
                        return metal.status();
                    } else {
                        LIYAB_LOG_WARN("metal unavailable: %s", metal.status().to_string().c_str());
                    }
                    break;
                }
                case BackendKind::Vulkan: {
                    if (gpu) break;
                    auto vulkan = make_vulkan_backend();
                    if (vulkan) {
                        gpu = std::move(vulkan).value();
                    } else if (config.backend == BackendKind::Vulkan) {
                        return vulkan.status();
                    } else {
                        LIYAB_LOG_WARN("vulkan unavailable: %s", vulkan.status().to_string().c_str());
                    }
                    break;
                }
                case BackendKind::Qnn:
                case BackendKind::NeuroPilot:
                    // Runtime present (probe succeeded) but Liyab has no compute
                    // kernels for it yet: fall through to the next backend.
                    if (config.backend == kind) {
                        return Status(ErrorCode::Unsupported,
                                      std::string(backend_kind_name(kind)) + " compute path is not implemented yet");
                    }
                    LIYAB_LOG_INFO("%s detected; compute path not implemented, using next backend",
                                   backend_kind_name(kind));
                    break;
            }
        }
        return Status::ok();
    }

    // Streams dense blocks from storage through the 3-slot pipeline. `all`:
    // every block (EngineConfig::triple_buffer_loading). Otherwise, for a
    // dense model larger than RAM, as many blocks as fit stay resident (read
    // in place from the mapping, which direct I/O never evicts) and the rest
    // are streamed; the streamed blocks are spread evenly through the stack
    // so storage keeps reading while resident blocks compute.
    Status attach_layer_streaming(bool all) {
        const MmapLoader& file = target->file();
        const ModelConfig& c = target->config();
        if (target->expert_store() != nullptr) {
            if (all) {
                return Status(ErrorCode::Unsupported, "triple-buffer loading and expert streaming are exclusive (MoE "
                                                      "models stream their experts already)");
            }
            return Status::ok();
        }
        if (file.shard_count() > 1) {
            if (all) return Status(ErrorCode::Unsupported, "block streaming needs a single-file model (this one is split)");
            return Status::ok();  // split dense models keep the mmap streaming window
        }
        std::vector<size_t> bytes(static_cast<size_t>(c.n_layers));
        size_t block_total = 0;
        for (int32_t l = 0; l < c.n_layers; ++l) {
            const auto [begin, end] = file.layer_range(l);
            bytes[static_cast<size_t>(l)] = end - begin;
            block_total += end - begin;
        }
        const size_t largest = *std::max_element(bytes.begin(), bytes.end());
        int32_t streamed = c.n_layers;
        if (!all) {
            const uint64_t available = usable_memory_bytes(
                config.memory_budget_mb > 0 ? static_cast<uint64_t>(config.memory_budget_mb) << 20 : 0);
            if (available == 0 || static_cast<double>(file.file_size()) <= 0.8 * static_cast<double>(available)) {
                return Status::ok();  // fits: in-place reads from the page cache
            }
            // Resident budget: free RAM minus the non-block weights, the 3
            // pipeline slots and 1 GiB for KV cache, activations and the app.
            const size_t fixed = file.file_size() - block_total + 3 * largest + (size_t{1} << 30);
            const size_t budget = available > fixed ? static_cast<size_t>(available) - fixed : 0;
            const size_t average = block_total / static_cast<size_t>(c.n_layers);
            const auto resident = static_cast<int32_t>(std::min<size_t>(budget / average, static_cast<size_t>(c.n_layers)));
            streamed = c.n_layers - resident;
            if (streamed == 0) return Status::ok();
        }
        std::vector<int32_t> items(static_cast<size_t>(c.n_layers), -1);
        std::vector<TripleBufferLoader::Item> ranges;
        for (int32_t l = 0; l < c.n_layers; ++l) {
            // Evenly spaced: block l streams when floor((l+1)s/n) > floor(ls/n).
            if ((int64_t{l} + 1) * streamed / c.n_layers > int64_t{l} * streamed / c.n_layers) {
                items[static_cast<size_t>(l)] = static_cast<int32_t>(ranges.size());
                const auto [begin, end] = file.layer_range(l);
                ranges.push_back({begin, end - begin});
            }
        }
        auto reader = DirectFile::open(file.file().path());
        if (!reader) return reader.status();
        direct_file = std::move(reader).value();
        // Stage 1: a few concurrent large direct reads saturate UFS (see direct_io.h).
        TripleBufferLoader::FetchFn fetch = [f = direct_file.get()](uint64_t offset, size_t length, uint8_t* dst) {
            return f->read_parallel(offset, length, dst, 4);
        };
        // With a GPU that can share memory, the slots and every resident
        // block live in that memory: direct reads land where the GPU reads,
        // nothing is copied or cached per tensor, and streamed blocks run on
        // the GPU like resident ones.
        Backend* shared = nullptr;
        if (gpu) {
            if (uint8_t* probe = gpu->allocate_shared(DirectFile::kAlign)) {
                gpu->free_shared(probe);
                shared = gpu.get();
            }
        }
        TripleBufferLoader::SlotMemory slot_memory;
        if (shared != nullptr) {
            slot_memory.allocate = [shared](size_t bytes) { return shared->allocate_shared(bytes); };
            slot_memory.release = [shared](uint8_t* p) { shared->free_shared(p); };
        }
        // Stage 2 is a pass-through: the kernels consume packed blocks directly.
        auto loader = TripleBufferLoader::create(ranges, fetch, {}, 0, slot_memory);  // copies: retried below
        if (!loader && shared != nullptr) {
            LIYAB_LOG_WARN("streaming slots in GPU-shared memory failed (%s); using host memory",
                           loader.status().to_string().c_str());
            shared = nullptr;
            loader = TripleBufferLoader::create(std::move(ranges), std::move(fetch));
        }
        if (!loader) return loader.status();
        weight_loader = std::move(loader).value();
        std::vector<std::pair<size_t, size_t>> relocated;
        if (shared != nullptr) {
            for (int32_t l = 0; l <= c.n_layers; ++l) {  // resident blocks + the output head slot
                if (l < c.n_layers && items[static_cast<size_t>(l)] >= 0) continue;
                relocated.push_back(file.layer_range(l));
            }
            auto moved = target->mutable_file().relocate(
                relocated, [shared](size_t bytes) { return shared->allocate_shared(bytes); });
            if (!moved) return moved.status();
            LIYAB_LOG_INFO("block streaming: %.2f GiB of resident weights placed in GPU-shared memory",
                           static_cast<double>(moved.value()) / (1024.0 * 1024.0 * 1024.0));
        }
        auto in_ranges = [&](const TensorView& t, bool streamed_ranges) {
            if (streamed_ranges) {
                for (int32_t l = 0; l < c.n_layers; ++l) {
                    if (items[static_cast<size_t>(l)] < 0) continue;
                    const auto [begin, end] = file.layer_range(l);
                    if (t.file_offset >= begin && t.file_offset < end) return true;
                }
                return false;
            }
            for (const auto& [begin, end] : relocated) {
                if (t.file_offset >= begin && t.file_offset < end) return true;
            }
            return false;
        };
        // What still lives in the mapping (e.g. the embedding table, read a row
        // at a time) stays paged in; streamed and relocated bytes are dropped
        // from the page cache and never read through the mapping.
        const size_t resident_bytes = target->mutable_file().keep_resident(
            [&](const TensorView& t) { return !in_ranges(t, true) && !in_ranges(t, false); });
        target->set_layer_source(weight_loader.get(), std::move(items), shared != nullptr);
        LIYAB_LOG_INFO("block streaming: %d of %d blocks streamed (%s direct reads, 3 x %.0f MiB slots%s), %.2f GiB "
                       "left in the mapping",
                       streamed, c.n_layers, direct_file->direct() ? "O_DIRECT" : "buffered",
                       static_cast<double>(weight_loader->slot_bytes()) / (1024.0 * 1024.0),
                       shared != nullptr ? " in GPU-shared memory" : "",
                       static_cast<double>(resident_bytes) / (1024.0 * 1024.0 * 1024.0));
        return Status::ok();
    }

    // MoE expert streaming with a GPU: the resident (non-expert) weights move
    // into GPU-shared memory, read once with direct I/O. The GPU then reads
    // them in place with its native kernels instead of keeping a second,
    // repacked copy next to the mapping (which the expert cache budget, sized
    // at load, does not account for).
    Status share_resident_weights() {
        if (!gpu || target->expert_store() == nullptr) return Status::ok();
        uint8_t* probe = gpu->allocate_shared(DirectFile::kAlign);
        if (probe == nullptr) return Status::ok();
        gpu->free_shared(probe);
        auto is_expert = [](const TensorView& t) {
            const std::string_view n = t.name;
            return n.size() > 13 && n.substr(n.size() - 12) == "_exps.weight";
        };
        // Runs of consecutive non-expert tensors of the first file become one
        // range each (gaps under 1 MiB are read along).
        std::vector<const TensorView*> resident;
        for (const TensorView& t : target->file().tensors()) {
            // requantize()d tensors already live in memory of their own.
            if (t.shard == 0 && !is_expert(t) && !target->file().converted(t)) resident.push_back(&t);
        }
        std::sort(resident.begin(), resident.end(),
                  [](const TensorView* a, const TensorView* b) { return a->file_offset < b->file_offset; });
        std::vector<std::pair<size_t, size_t>> ranges;
        for (const TensorView* t : resident) {
            const auto begin = static_cast<size_t>(t->file_offset);
            const size_t end = begin + t->nbytes;
            if (!ranges.empty() && begin <= ranges.back().second + (size_t{1} << 20)) {
                ranges.back().second = std::max(ranges.back().second, end);
            } else {
                ranges.emplace_back(begin, end);
            }
        }
        Backend* shared = gpu.get();
        auto moved = target->mutable_file().relocate(ranges, [shared](size_t bytes) { return shared->allocate_shared(bytes); });
        if (!moved) return moved.status();
        LIYAB_LOG_INFO("expert streaming: %.2f GiB of resident weights placed in GPU-shared memory (%zu ranges)",
                       static_cast<double>(moved.value()) / (1024.0 * 1024.0 * 1024.0), ranges.size());
        return Status::ok();
    }

    // Tokens per prefill pass. With streamed MoE experts a pass reads the
    // union of the experts its tokens choose, and 64 tokens already choose
    // most of them: Qwen3.6-35B-A3B prefilled 1700 tokens in 180 s in passes
    // of 64, rereading nearly every expert from storage 27 times. Passes of
    // kStreamedPrefillChunk read each expert once per 1024 tokens: a 706-token
    // prompt read 26 MiB per token instead of 36 with passes of 512 and 70
    // with 64, for 81 MiB more peak memory than 512. Other models keep
    // max_batch passes.
    static constexpr int32_t kStreamedPrefillChunk = 1024;
    [[nodiscard]] int32_t prefill_chunk() const {
        return target->expert_store() != nullptr ? std::max(target->max_batch(), kStreamedPrefillChunk)
                                                 : target->max_batch();
    }

    void apply_policy(const PowerPolicy& policy) {
        const auto threads = static_cast<int32_t>(std::lround(pool->max_threads() * policy.thread_fraction));
        pool->set_active_threads(std::max(1, threads));
    }
};

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Engine::~Engine() = default;

Result<std::unique_ptr<Engine>> Engine::create(const EngineConfig& config) {
    if (config.model_path.empty()) return Status(ErrorCode::InvalidArgument, "model_path is required");
    if (config.sliding_window < 0 || config.context_length < 0 || config.draft_tokens < 0) {
        return Status(ErrorCode::InvalidArgument, "negative size in EngineConfig");
    }
#if !defined(LIYAB_ENABLE_EXPERIMENTAL)
    if (config.experimental.early_exit || config.experimental.head_pruning || config.experimental.egls ||
        config.experimental.tdss || !config.experimental.kv_dedup_dir.empty()) {
        return Status(ErrorCode::Unsupported, "experimental features need a build with LIYAB_ENABLE_EXPERIMENTAL=ON");
    }
#endif

    auto impl = std::make_unique<Impl>();
    impl->config = config;
    const auto t_create = Clock::now();
    impl->device = detect_device();
    LIYAB_LOG_INFO("device: %s (%s), %d cores, %.1f GiB available", impl->device.soc.name.c_str(),
                   soc_vendor_name(impl->device.soc.vendor), impl->device.cpu.cores,
                   static_cast<double>(impl->device.available_memory) / (1024.0 * 1024.0 * 1024.0));
    impl->pool = std::make_unique<ThreadPool>(config.n_threads);
    impl->cpu = make_cpu_backend(*impl->pool);
    LIYAB_RETURN_IF_ERROR(impl->select_backends());

    // Prefill chunks and speculative verification batches share max_batch.
    const int32_t max_batch = std::max(64, config.draft_tokens + 1);
    // Repacked weights are for the CPU's i8mm kernels only: no GPU / NPU route.
    const bool repack_cpu = !impl->gpu && !impl->npu && impl->device.cpu.i8mm && quant::repack_kernels_available();
    auto target = load_model(config.model_path, config, max_batch, repack_cpu);
    if (!target) return target.status();
    impl->target = std::move(target).value();

    LIYAB_RETURN_IF_ERROR(impl->attach_layer_streaming(config.triple_buffer_loading));
    LIYAB_RETURN_IF_ERROR(impl->share_resident_weights());

    auto tokenizer = Tokenizer::load(impl->target->file());
    if (!tokenizer) return tokenizer.status();
    impl->tokenizer.emplace(std::move(tokenizer).value());

    if (!config.draft_model_path.empty() && config.draft_tokens > 0) {
        auto draft = load_model(config.draft_model_path, config, max_batch, repack_cpu);
        if (!draft) return draft.status();
        if (draft.value()->config().n_vocab != impl->target->config().n_vocab) {
            return Status(ErrorCode::InvalidArgument, "draft and target models must share a vocabulary");
        }
        // Rejected drafts are rolled back; recurrent (DeltaNet) states need a
        // checkpoint per position of a verification batch (k + 1).
        impl->target->set_rollback_window(config.draft_tokens + 1);
        draft.value()->set_rollback_window(config.draft_tokens + 1);
        impl->draft = std::move(draft).value();
        impl->speculative = std::make_unique<SpeculativeDecoder>(*impl->target, *impl->draft, config.draft_tokens);
    } else if (config.lookup_drafts && config.draft_tokens > 0) {
        impl->target->set_rollback_window(config.draft_tokens + 1);
        impl->speculative = std::make_unique<SpeculativeDecoder>(*impl->target, config.draft_tokens);
    }
    if (impl->speculative) impl->speculative->set_adaptive(config.adaptive_drafts);

#if defined(LIYAB_ENABLE_EXPERIMENTAL)
    const ExperimentalConfig& x = config.experimental;
    if (x.early_exit) {
        impl->early_exit = std::make_unique<experimental::EarlyExit>(
            experimental::EarlyExitConfig{x.early_exit_threshold, x.early_exit_min_layer, x.early_exit_interval});
    }
    if (x.tdss) {
        if (impl->target->config().n_expert > 0) {
            return Status(ErrorCode::Unsupported, "TDSS sparsifies dense FFNs; this model uses mixture-of-experts FFNs");
        }
        std::vector<std::array<const TensorView*, 3>> ffn;
        for (int32_t l = 0; l < impl->target->config().n_layers; ++l) ffn.push_back(impl->target->ffn_weights(l));
        auto tdss = experimental::Tdss::create(ffn, *impl->pool, {x.tdss_force});
        if (!tdss) return tdss.status();
        impl->tdss = std::move(tdss).value();
        LIYAB_LOG_INFO("TDSS: 2:4 sparse FFN copy %.1f MiB (dense %.1f MiB)",
                       static_cast<double>(impl->tdss->sparse_bytes()) / (1024.0 * 1024.0),
                       static_cast<double>(impl->tdss->dense_bytes()) / (1024.0 * 1024.0));
    }
    if (!x.kv_dedup_dir.empty()) {
        auto dedup = experimental::KvDedup::create(
            x.kv_dedup_dir, experimental::model_fingerprint(impl->target->file().file()));
        if (!dedup) return dedup.status();
        impl->kv_dedup = std::move(dedup).value();
    }
    if (x.egls) impl->egls = std::make_unique<experimental::Egls>(experimental::EglsConfig{x.egls_threshold});
    if (x.head_pruning) {
        if (impl->target->config().hybrid()) {
            return Status(ErrorCode::Unsupported, "head pruning needs attention in every block (not a hybrid model)");
        }
        std::vector<const TensorView*> wo;
        for (int32_t l = 0; l < impl->target->config().n_layers; ++l) wo.push_back(impl->target->attn_output(l));
        auto pruner = experimental::HeadPruner::create(wo, impl->target->config().n_head,
                                                        impl->target->config().head_dim, {x.head_keep_ratio});
        if (!pruner) return pruner.status();
        impl->head_pruner.emplace(std::move(pruner).value());
    }
#endif

    impl->power = std::make_unique<PowerManager>(config.power);
    impl->power->poll_once();
    if (config.thermal_polling) impl->power->start();
    const Route route = impl->normal_route();
    LIYAB_LOG_INFO("engine ready in %.0f ms: attention on %s, FFN on %s, %d threads%s", ms_since(t_create),
                   route.attention->description().c_str(), route.ffn->description().c_str(),
                   impl->pool->max_threads(), impl->speculative ? ", speculative decoding" : "");

    return std::unique_ptr<Engine>(new Engine(std::move(impl)));
}

void Engine::cancel() noexcept { impl_->cancel.store(true, std::memory_order_relaxed); }

const DeviceInfo& Engine::device() const noexcept { return impl_->device; }

PowerManager& Engine::power() noexcept { return *impl_->power; }

Result<std::vector<int32_t>> Engine::tokenize(std::string_view text, bool add_bos) const {
    // BOS only when requested *and* the model uses one (Qwen does not).
    return impl_->tokenizer->encode(text, add_bos && impl_->tokenizer->add_bos_default());
}

std::string Engine::token_to_piece(int32_t token) const { return impl_->tokenizer->piece(token); }

namespace {
constexpr char kExpertProfileMagic[4] = {'L', 'X', 'H', 'L'};
}  // namespace

size_t Engine::warm_memory() {
    ExpertStore* store = impl_->target->expert_store();
    const size_t queued = store != nullptr ? store->warm() : 0;
    if (queued > 0) LIYAB_LOG_INFO("warming the expert cache: %zu experts queued", queued);
    return queued;
}

Status Engine::save_expert_profile(const std::string& path) const {
    const ExpertStore* store = impl_->target->expert_store();
    if (store == nullptr) return Status::ok();
    const std::vector<int32_t> keys = store->hot_keys();
    const ModelConfig& c = impl_->target->config();
    // Header: magic, model file size, blocks, experts per block, count.
    const int64_t header[] = {static_cast<int64_t>(impl_->target->file().file_size()), c.n_layers, c.n_expert,
                              static_cast<int64_t>(keys.size())};
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(kExpertProfileMagic, sizeof kExpertProfileMagic);
        out.write(reinterpret_cast<const char*>(header), sizeof header);
        out.write(reinterpret_cast<const char*>(keys.data()), static_cast<std::streamsize>(keys.size() * sizeof(int32_t)));
        if (!out) return Status(ErrorCode::IoError, "cannot write " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) return Status(ErrorCode::IoError, "cannot rename " + tmp);
    return Status::ok();
}

Status Engine::load_expert_profile(const std::string& path) {
    ExpertStore* store = impl_->target->expert_store();
    if (store == nullptr) return Status::ok();
    std::ifstream in(path, std::ios::binary);
    char magic[4] = {};
    int64_t header[4] = {};
    in.read(magic, sizeof magic);
    in.read(reinterpret_cast<char*>(header), sizeof header);
    const ModelConfig& c = impl_->target->config();
    if (!in || std::memcmp(magic, kExpertProfileMagic, sizeof magic) != 0) {
        return Status(ErrorCode::IoError, "not an expert profile: " + path);
    }
    if (header[0] != static_cast<int64_t>(impl_->target->file().file_size()) || header[1] != c.n_layers ||
        header[2] != c.n_expert || header[3] < 0 || header[3] > static_cast<int64_t>(c.n_layers) * c.n_expert) {
        return Status(ErrorCode::InvalidArgument, "expert profile of another model: " + path);
    }
    std::vector<int32_t> keys(static_cast<size_t>(header[3]));
    in.read(reinterpret_cast<char*>(keys.data()), static_cast<std::streamsize>(keys.size() * sizeof(int32_t)));
    if (!in) return Status(ErrorCode::IoError, "truncated expert profile: " + path);
    store->set_hot(std::move(keys));
    return Status::ok();
}

int32_t Engine::context_length() const noexcept { return impl_->target->context_length(); }

Engine::MemoryPlan Engine::memory_plan() const {
    const Transformer::MemoryPlan& p = impl_->target->memory_plan();
    return {p.resident_bytes, p.expert_bytes, p.expert_cache_bytes, p.recommended_bytes, p.requant_bits};
}

size_t Engine::trim_memory() {
    size_t released = 0;
    if (ExpertStore* store = impl_->target->expert_store()) released += store->trim();
    if (impl_->draft) {
        if (ExpertStore* store = impl_->draft->expert_store()) released += store->trim();
    }
    if (released > 0) LIYAB_LOG_INFO("trimmed %.2f GiB of cached experts", released / (1024.0 * 1024.0 * 1024.0));
    return released;
}

Engine::Counters Engine::counters() const {
    Counters c;
    if (impl_->gpu) c.accelerator_busy_ms = impl_->gpu->busy_ms();
    if (const ExpertStore* store = impl_->target->expert_store()) c.storage_bytes_read += store->stats().bytes_read;
    if (impl_->weight_loader) c.storage_bytes_read += impl_->weight_loader->stats().bytes_fetched;
    c.tokens_generated = impl_->tokens_generated.load(std::memory_order_relaxed);
    return c;
}

std::optional<std::string> Engine::model_metadata(std::string_view key) const {
    const GgufValue* v = impl_->target->file().metadata(key);
    if (v == nullptr) return std::nullopt;
    if (const auto s = v->as_string()) return std::string(*s);
    if (const auto b = v->as_bool()) return std::string(*b ? "true" : "false");
    if (const auto i = v->as_int()) return std::to_string(*i);
    if (const auto d = v->as_float()) {
        char buf[32];
        std::snprintf(buf, sizeof buf, "%.9g", *d);
        return std::string(buf);
    }
    return std::nullopt;
}

std::string Engine::describe() const {
    const ModelConfig& c = impl_->target->config();
    const KvCache& kv = impl_->target->kv_cache();
    char buf[768];
    std::string window = kv.window() > 0 ? " (window " + std::to_string(kv.window()) + " + " +
                                               std::to_string(kv.sink_tokens()) + " sink tokens)"
                                         : "";
    std::snprintf(buf, sizeof buf,
                  "Model:      %s, %d layers, d=%d, vocab=%d, %.2f GiB (%s)\n"
                  "Context:    %d tokens%s\n"
                  "KV cache:   %s, paged (%d tokens/page, %.1f KiB/token)\n"
                  "Weights:    %s\n",
                  c.arch.c_str(), c.n_layers, c.n_embd, c.n_vocab,
                  static_cast<double>(impl_->target->file().file_size()) / (1024.0 * 1024.0 * 1024.0),
                  impl_->target->file().streaming() ? "streaming" : "resident", impl_->target->context_length(),
                  window.c_str(), std::string(dtype_traits(kv.dtype()).name).c_str(), kv.config().page_tokens,
                  static_cast<double>(KvCache::bytes_per_token(kv.config().n_layers, c.n_head_kv, c.head_dim, kv.config().type)) /
                      1024.0,
                  impl_->weight_loader
                      ? ("triple-buffered blocks, 3 x " +
                         std::to_string(impl_->weight_loader->slot_bytes() / (1024 * 1024)) + " MiB slots")
                            .c_str()
                      : "mmap, read in place (zero-copy)");
    const Route route = impl_->normal_route();
    std::string out = describe_device(impl_->device) + buf;
    out += "Attention:  " + route.attention->description() + "\n";
    out += "FFN:        " + route.ffn->description() + "\n";
    if (c.hybrid()) {
        char mix[160];
        std::snprintf(mix, sizeof mix, "Mixers:     %d attention + %d DeltaNet blocks, recurrent state %.1f MiB\n",
                      c.n_attn_layers, c.n_layers - c.n_attn_layers,
                      static_cast<double>(impl_->target->recurrent_state_bytes()) / (1024.0 * 1024.0));
        out += mix;
    }
    if (impl_->draft) {
        out += "Draft:      " + impl_->draft->config().arch + ", " + std::to_string(impl_->draft->config().n_layers) +
               " layers, k=" + std::to_string(impl_->config.draft_tokens) + " (cpu)\n";
    }
    return out;
}

Status Engine::prefill(std::string_view text, bool add_bos) {
    std::unique_lock<std::mutex> lock(impl_->busy, std::try_to_lock);
    if (!lock.owns_lock()) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    Impl& s = *impl_;
    if (s.draft) return Status(ErrorCode::Unsupported, "prefill() is not available with a draft model");
    auto tokens = s.encode_continuation(text, add_bos);
    if (!tokens) return tokens.status();
    s.cancel.store(false, std::memory_order_relaxed);
    const std::vector<int32_t>& t = tokens.value();
    const size_t reused = s.reuse_prefix(t, t.size());
    s.context_text_valid = false;
    const PowerPolicy policy = s.power->policy();
    s.apply_policy(policy);
    const Route route = policy.throttled ? s.throttled_route() : s.normal_route();
    const ForwardHooks hooks = s.hooks(policy, false);
    const auto chunk = static_cast<size_t>(s.prefill_chunk());
    for (size_t i = reused; i < t.size(); i += chunk) {
        if (s.cancel.load(std::memory_order_relaxed)) return Status(ErrorCode::Cancelled, "prefill cancelled");
        const std::span<const int32_t> part(t.data() + i, std::min(chunk, t.size() - i));
        if (auto r = s.target->forward(part, Transformer::Logits::None, route, *s.pool, &hooks); !r) {
            s.clear_context();
            return r.status();
        }
        s.context.insert(s.context.end(), part.begin(), part.end());
    }
    s.target->snapshot_state();  // the next prompt starts with this text
    s.remember_text(t, text);
    return Status::ok();
}

namespace {
constexpr uint32_t kContextMagic = 0x5458434C;  // "LCXT": the engine's part after the model's state
}

Status Engine::save_state(const std::string& path) {
    std::unique_lock<std::mutex> lock(impl_->busy, std::try_to_lock);
    if (!lock.owns_lock()) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    Impl& s = *impl_;
    if (s.draft) return Status(ErrorCode::Unsupported, "save_state() is not available with a draft model");
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) return Status(ErrorCode::IoError, "cannot write " + tmp);
        LIYAB_RETURN_IF_ERROR(s.target->write_state(out));
        const auto n = static_cast<uint64_t>(s.context.size());
        const uint64_t text = s.context_text_valid ? s.context_text.size() : UINT64_MAX;
        out.write(reinterpret_cast<const char*>(&kContextMagic), sizeof kContextMagic);
        out.write(reinterpret_cast<const char*>(&n), sizeof n);
        out.write(reinterpret_cast<const char*>(s.context.data()), static_cast<std::streamsize>(n * sizeof(int32_t)));
        out.write(reinterpret_cast<const char*>(&text), sizeof text);
        if (s.context_text_valid) out.write(s.context_text.data(), static_cast<std::streamsize>(s.context_text.size()));
        if (!out.flush()) return Status(ErrorCode::IoError, "cannot write " + tmp);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return Status(ErrorCode::IoError, "cannot replace " + path);
    }
    return Status::ok();
}

Result<int32_t> Engine::load_state(const std::string& path) {
    std::unique_lock<std::mutex> lock(impl_->busy, std::try_to_lock);
    if (!lock.owns_lock()) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    Impl& s = *impl_;
    if (s.draft) return Status(ErrorCode::Unsupported, "load_state() is not available with a draft model");
    s.clear_context();
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status(ErrorCode::IoError, "cannot read " + path);
    LIYAB_RETURN_IF_ERROR(s.target->read_state(in));
    uint32_t magic = 0;
    uint64_t n = 0, text = 0;
    const auto damaged = [&s] {
        s.clear_context();
        return Status(ErrorCode::IoError, "the saved context is damaged");
    };
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof magic) || magic != kContextMagic ||
        !in.read(reinterpret_cast<char*>(&n), sizeof n) || n != static_cast<uint64_t>(s.target->n_past())) {
        return damaged();
    }
    s.context.resize(n);
    if (!in.read(reinterpret_cast<char*>(s.context.data()), static_cast<std::streamsize>(n * sizeof(int32_t))) ||
        !in.read(reinterpret_cast<char*>(&text), sizeof text)) {
        return damaged();
    }
    if (text != UINT64_MAX) {
        if (text > (uint64_t{1} << 30)) return damaged();
        s.context_text.resize(text);
        if (!in.read(s.context_text.data(), static_cast<std::streamsize>(text))) return damaged();
        s.context_text_valid = true;
    }
    return static_cast<int32_t>(n);
}

void Engine::reset_context() {
    std::lock_guard<std::mutex> lock(impl_->busy);
    impl_->clear_context();
}

Result<GenerationStats> Engine::generate(std::string_view prompt, const SamplingParams& params,
                                         const TokenCallback& on_token) {
    std::unique_lock<std::mutex> lock(impl_->busy, std::try_to_lock);
    if (!lock.owns_lock()) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    auto tokens = impl_->encode_continuation(prompt, params.add_bos);
    if (!tokens) return tokens.status();
    return generate_locked(*tokens, params, on_token, &prompt);
}

Result<GenerationStats> Engine::generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                                const TokenCallback& on_token) {
    std::unique_lock<std::mutex> lock(impl_->busy, std::try_to_lock);
    if (!lock.owns_lock()) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return generate_locked(prompt, params, on_token, nullptr);
}

Result<GenerationStats> Engine::generate_locked(std::span<const int32_t> prompt, const SamplingParams& params,
                                                const TokenCallback& on_token, const std::string_view* text) {
    if (prompt.empty()) return Status(ErrorCode::InvalidArgument, "empty prompt");
    if (params.max_tokens <= 0) return Status(ErrorCode::InvalidArgument, "max_tokens must be positive");

    Impl& s = *impl_;
    s.cancel.store(false, std::memory_order_relaxed);
    // The last prompt token seeds the decode loop, so it is never reused.
    const size_t reused = s.reuse_prefix(prompt, prompt.size() - 1);
    s.context_text_valid = false;  // set again once the run completes
    if (s.speculative) s.speculative->reset_stats();
    s.power->reset_pacing();
    {
        // The caller decodes too, and may be a different OS thread each run
        // (a Dart isolate moves between threads).
        std::vector<int32_t> decode_threads = s.pool->worker_thread_ids();
        decode_threads.push_back(ThreadPool::current_thread_id());
        s.power->set_hint_threads(std::move(decode_threads));
    }
    Sampler sampler(params);
    GenerationStats stats;
    const TripleBufferLoader::Stats loader_before =
        s.weight_loader ? s.weight_loader->stats() : TripleBufferLoader::Stats{};
    stats.prompt_tokens = static_cast<int32_t>(prompt.size());
    const ExpertStore::Stats experts_before =
        s.target->expert_store() != nullptr ? s.target->expert_store()->stats() : ExpertStore::Stats{};
    const Transformer::ExpertPredictions predictions_before = s.target->expert_predictions();

    // Prefill everything but the last prompt token, which seeds the decode
    // loop (the speculative decoder expects it uncached).
    const auto t_prefill = Clock::now();
    size_t first_uncached = reused;
    stats.cached_prefix_tokens = static_cast<int32_t>(reused);
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
    // Persistent prefix cache: attach snapshotted KV pages for the longest
    // known prefix (full attention, no draft model).
    // (Not for hybrid models: the snapshots hold KV pages, not recurrent states.)
    const bool use_dedup = reused == 0 && s.kv_dedup && !s.draft && s.target->kv_cache().window() == 0 &&
                           !s.target->config().hybrid();
    if (use_dedup) {
        const int32_t restored = s.kv_dedup->restore(prompt, static_cast<int32_t>(prompt.size()) - 1,
                                                     s.target->mutable_kv_cache());
        if (restored > 0 && s.target->adopt_cached_prefix(restored).is_ok()) {
            first_uncached = static_cast<size_t>(restored);
            stats.cached_prefix_tokens = restored;
            s.context.assign(prompt.begin(), prompt.begin() + restored);
        } else {
            s.clear_context();
        }
    }
#endif
    const PowerPolicy prefill_policy = s.power->policy();
    s.apply_policy(prefill_policy);
    const Route prefill_route = prefill_policy.throttled ? s.throttled_route() : s.normal_route();
    const ForwardHooks prefill_hooks = s.hooks(prefill_policy, false);
    const auto chunk = static_cast<size_t>(s.prefill_chunk());
    const std::span<const int32_t> prefix = prompt.first(prompt.size() - 1).subspan(first_uncached);
    for (size_t i = 0; i < prefix.size(); i += chunk) {
        if (s.cancel.load(std::memory_order_relaxed)) {
            stats.cancelled = true;
            return stats;
        }
        const auto part = prefix.subspan(i, std::min(chunk, prefix.size() - i));
        if (auto r = s.target->forward(part, Transformer::Logits::None, prefill_route, *s.pool, &prefill_hooks); !r) {
            s.clear_context();
            return r.status();
        }
        if (s.draft) {
            if (auto r = s.draft->forward(part, Transformer::Logits::None, s.draft_route(), *s.pool); !r) {
                s.clear_context();
                return r.status();
            }
        } else {
            s.context.insert(s.context.end(), part.begin(), part.end());
        }
    }
    stats.prefill_ms = ms_since(t_prefill);
    // The next chat turn re-sends this prompt: keep its recurrent state.
    if (!s.draft) s.target->snapshot_state(/*pin=*/true);
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
    if (use_dedup) {
        auto saved = s.kv_dedup->save(prompt, static_cast<int32_t>(prompt.size()) - 1, s.target->kv_cache());
        if (!saved) LIYAB_LOG_WARN("KV snapshot not saved: %s", saved.status().to_string().c_str());
    }
#endif

    const auto t_decode = Clock::now();
    const Transformer::PhaseTimes phases_before = s.target->phase_times();
    std::string pending_utf8;
    std::vector<float> probs;
    int32_t last = prompt.back();
    bool stop = false;
    std::chrono::microseconds idle{0};

    while (!stop) {
        if (s.cancel.load(std::memory_order_relaxed)) {
            stats.cancelled = true;
            break;
        }
        const PowerPolicy policy = s.power->policy();
        s.apply_policy(policy);
        const Route route = policy.throttled ? s.throttled_route() : s.normal_route();
        if (policy.throttled) ++stats.thermal_reroutes;

        std::vector<int32_t> next;
        if (s.speculative) {
            auto r = s.speculative->step(last, s.context, sampler, route, s.draft_route(), *s.pool);
            if (!r) {
                s.clear_context();
                return r.status();
            }
            next = std::move(r).value();
            if (!s.draft) {  // the target now caches `last` and every returned token but the final one
                s.context.push_back(last);
                s.context.insert(s.context.end(), next.begin(), next.end() - 1);
            }
        } else {
            const ForwardHooks hooks = s.hooks(policy, true);
            auto logits =
                s.target->forward(std::span<const int32_t>(&last, 1), Transformer::Logits::Last, route, *s.pool, &hooks);
            if (!logits) {
                s.clear_context();
                return logits.status();
            }
            s.context.push_back(last);
            if (hooks.head_mask != nullptr) ++stats.head_pruned_steps;
            stats.ffn_blocks_skipped += s.target->last_ffn_skips();
            if (hooks.ffn_matmul != nullptr) ++stats.sparse_ffn_steps;
            if (const int32_t exit_layer = s.target->last_exit_layer(); exit_layer >= 0) {
                ++stats.early_exits;
                stats.early_exit_layers_skipped += s.target->config().n_layers - 1 - exit_layer;
            }
            sampler.probabilities(*logits, probs);
            next.push_back(sampler.greedy() ? Sampler::argmax(probs) : sampler.sample(probs));
        }

        if (stats.ttft_ms == 0.0) stats.ttft_ms = ms_since(t_prefill);
        for (const int32_t token : next) {
            if (s.tokenizer->is_end_of_generation(token)) {
                stop = true;
                break;
            }
            // Duty cycling: hold this token until its slot at the target
            // rate; the SoC idles in between instead of racing ahead.
            idle += s.power->pace_token();
            ++stats.generated_tokens;
            s.tokens_generated.fetch_add(1, std::memory_order_relaxed);
            pending_utf8 += s.tokenizer->piece(token);
            const size_t ready = complete_utf8_prefix(pending_utf8);
            if (ready > 0) {
                const bool keep_going = on_token ? on_token(std::string_view(pending_utf8).substr(0, ready), token) : true;
                pending_utf8.erase(0, ready);
                if (!keep_going) {
                    stop = true;
                    break;
                }
            }
            if (stats.generated_tokens >= params.max_tokens || s.cancel.load(std::memory_order_relaxed)) {
                stats.cancelled = s.cancel.load(std::memory_order_relaxed);
                stop = true;
                break;
            }
        }
        last = next.back();
    }
    if (!pending_utf8.empty() && on_token) on_token(pending_utf8, last);

    stats.decode_ms = ms_since(t_decode);
    stats.tokens_per_second = stats.decode_ms > 0.0 ? stats.generated_tokens * 1000.0 / stats.decode_ms : 0.0;
    stats.paced_idle_ms = static_cast<double>(idle.count()) / 1000.0;
    if (s.weight_loader) {
        const TripleBufferLoader::Stats after = s.weight_loader->stats();
        stats.weight_stalls = static_cast<int32_t>(after.stalls - loader_before.stalls);
        stats.weight_wait_ms = after.consumer_wait_ms - loader_before.consumer_wait_ms;
    }
    stats.kv_cache_bytes = s.target->kv_cache().bytes_in_use();
    if (const ExpertStore* store = s.target->expert_store()) {
        const ExpertStore::Stats after = store->stats();
        stats.expert_hits = static_cast<int32_t>(after.hits - experts_before.hits);
        stats.expert_late = static_cast<int32_t>(after.late - experts_before.late);
        stats.expert_misses = static_cast<int32_t>(after.misses - experts_before.misses);
        stats.expert_bytes_read = after.bytes_read - experts_before.bytes_read;
        stats.expert_stall_ms = after.stall_ms - experts_before.stall_ms;
        stats.expert_unused = static_cast<int32_t>(after.unused - experts_before.unused);
        stats.expert_dropped = static_cast<int32_t>(after.dropped - experts_before.dropped);
        const Transformer::ExpertPredictions& predictions = s.target->expert_predictions();
        stats.expert_skipped = static_cast<int32_t>(predictions.skipped - predictions_before.skipped);
        stats.expert_predicted = static_cast<int32_t>(predictions.predicted - predictions_before.predicted);
        stats.expert_predicted_used = static_cast<int32_t>(predictions.used - predictions_before.used);
    }
    const Transformer::PhaseTimes& phases = s.target->phase_times();
    stats.decode_phases = {phases.attention - phases_before.attention,
                           phases.delta_net - phases_before.delta_net,
                           phases.router - phases_before.router,
                           phases.experts - phases_before.experts,
                           phases.shared_expert - phases_before.shared_expert,
                           phases.dense_ffn - phases_before.dense_ffn,
                           phases.lm_head - phases_before.lm_head};
    if (s.speculative) {
        stats.draft_tokens_proposed = s.speculative->proposed();
        stats.draft_tokens_accepted = s.speculative->accepted();
    }
    if (text != nullptr && !s.draft) s.remember_text(prompt, *text);
    return stats;
}

}  // namespace liyab
