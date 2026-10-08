#include "liyab/engine.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>

#include "core/log.h"
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

Result<std::unique_ptr<Transformer>> load_model(const std::string& path, const EngineConfig& config,
                                                int32_t max_batch) {
    LoaderOptions loader_options;
    loader_options.streaming = config.streaming;
    auto file = MmapLoader::open(path, loader_options);
    if (!file) return file.status();
    TransformerOptions options;
    options.context_length = config.context_length;
    options.sliding_window = config.sliding_window;
    options.sink_tokens = config.kv_sink_tokens;
    options.kv_type = config.kv_cache_type;
    options.max_batch = max_batch;
    options.expert_cache_bytes = config.expert_cache_mb > 0 ? config.expert_cache_mb << 20 : config.expert_cache_mb;
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
            const uint64_t available = available_memory_bytes();
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
        // Stage 2 is a pass-through: the kernels consume packed blocks directly.
        auto loader = TripleBufferLoader::create(std::move(ranges), std::move(fetch));
        if (!loader) return loader.status();
        weight_loader = std::move(loader).value();
        // Resident blocks and the other weights stay paged in; streamed blocks
        // are dropped from the page cache and never read through the mapping.
        const size_t resident_bytes = target->mutable_file().keep_resident([&](const TensorView& t) {
            for (int32_t l = 0; l < c.n_layers; ++l) {
                if (items[static_cast<size_t>(l)] < 0) continue;
                const auto [begin, end] = file.layer_range(l);
                if (t.file_offset >= begin && t.file_offset < end) return false;
            }
            return true;
        });
        target->set_layer_source(weight_loader.get(), std::move(items));
        LIYAB_LOG_INFO("block streaming: %d of %d blocks streamed (%s direct reads, 3 x %.0f MiB slots), %.2f GiB resident",
                       streamed, c.n_layers, direct_file->direct() ? "O_DIRECT" : "buffered",
                       static_cast<double>(weight_loader->slot_bytes()) / (1024.0 * 1024.0),
                       static_cast<double>(resident_bytes) / (1024.0 * 1024.0 * 1024.0));
        return Status::ok();
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
    auto target = load_model(config.model_path, config, max_batch);
    if (!target) return target.status();
    impl->target = std::move(target).value();

    LIYAB_RETURN_IF_ERROR(impl->attach_layer_streaming(config.triple_buffer_loading));

    auto tokenizer = Tokenizer::load(impl->target->file());
    if (!tokenizer) return tokenizer.status();
    impl->tokenizer.emplace(std::move(tokenizer).value());

    if (!config.draft_model_path.empty() && config.draft_tokens > 0) {
        auto draft = load_model(config.draft_model_path, config, max_batch);
        if (!draft) return draft.status();
        if (draft.value()->config().n_vocab != impl->target->config().n_vocab) {
            return Status(ErrorCode::InvalidArgument, "draft and target models must share a vocabulary");
        }
        if (impl->target->config().hybrid() || draft.value()->config().hybrid()) {
            return Status(ErrorCode::Unsupported,
                          "speculative decoding rolls back rejected tokens, which recurrent (DeltaNet) models cannot do");
        }
        impl->draft = std::move(draft).value();
        impl->speculative = std::make_unique<SpeculativeDecoder>(*impl->target, *impl->draft, config.draft_tokens);
    }

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

Result<GenerationStats> Engine::generate(std::string_view prompt, const SamplingParams& params,
                                         const TokenCallback& on_token) {
    auto tokens = tokenize(prompt, params.add_bos);
    if (!tokens) return tokens.status();
    return generate_tokens(*tokens, params, on_token);
}

Result<GenerationStats> Engine::generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                                const TokenCallback& on_token) {
    std::unique_lock<std::mutex> lock(impl_->busy, std::try_to_lock);
    if (!lock.owns_lock()) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    if (prompt.empty()) return Status(ErrorCode::InvalidArgument, "empty prompt");
    if (params.max_tokens <= 0) return Status(ErrorCode::InvalidArgument, "max_tokens must be positive");

    Impl& s = *impl_;
    s.cancel.store(false, std::memory_order_relaxed);
    s.target->reset();
    if (s.draft) s.draft->reset();
    if (s.speculative) s.speculative->reset_stats();
    s.power->reset_pacing();
    Sampler sampler(params);
    GenerationStats stats;
    const TripleBufferLoader::Stats loader_before =
        s.weight_loader ? s.weight_loader->stats() : TripleBufferLoader::Stats{};
    stats.prompt_tokens = static_cast<int32_t>(prompt.size());
    const ExpertStore::Stats experts_before =
        s.target->expert_store() != nullptr ? s.target->expert_store()->stats() : ExpertStore::Stats{};

    // Prefill everything but the last prompt token, which seeds the decode
    // loop (the speculative decoder expects it uncached).
    const auto t_prefill = Clock::now();
    size_t first_uncached = 0;
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
    // Persistent prefix cache: attach snapshotted KV pages for the longest
    // known prefix (full attention, no draft model).
    // (Not for hybrid models: the snapshots hold KV pages, not recurrent states.)
    const bool use_dedup =
        s.kv_dedup && !s.draft && s.target->kv_cache().window() == 0 && !s.target->config().hybrid();
    if (use_dedup) {
        const int32_t restored = s.kv_dedup->restore(prompt, static_cast<int32_t>(prompt.size()) - 1,
                                                     s.target->mutable_kv_cache());
        if (restored > 0 && s.target->adopt_cached_prefix(restored).is_ok()) {
            first_uncached = static_cast<size_t>(restored);
            stats.cached_prefix_tokens = restored;
        } else {
            s.target->reset();
        }
    }
#endif
    const PowerPolicy prefill_policy = s.power->policy();
    s.apply_policy(prefill_policy);
    const Route prefill_route = prefill_policy.throttled ? s.throttled_route() : s.normal_route();
    const ForwardHooks prefill_hooks = s.hooks(prefill_policy, false);
    const auto chunk = static_cast<size_t>(s.target->max_batch());
    const std::span<const int32_t> prefix = prompt.first(prompt.size() - 1).subspan(first_uncached);
    for (size_t i = 0; i < prefix.size(); i += chunk) {
        if (s.cancel.load(std::memory_order_relaxed)) {
            stats.cancelled = true;
            return stats;
        }
        const auto part = prefix.subspan(i, std::min(chunk, prefix.size() - i));
        if (auto r = s.target->forward(part, Transformer::Logits::None, prefill_route, *s.pool, &prefill_hooks); !r) {
            return r.status();
        }
        if (s.draft) {
            if (auto r = s.draft->forward(part, Transformer::Logits::None, s.draft_route(), *s.pool); !r) return r.status();
        }
    }
    stats.prefill_ms = ms_since(t_prefill);
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
    if (use_dedup) {
        auto saved = s.kv_dedup->save(prompt, static_cast<int32_t>(prompt.size()) - 1, s.target->kv_cache());
        if (!saved) LIYAB_LOG_WARN("KV snapshot not saved: %s", saved.status().to_string().c_str());
    }
#endif

    const auto t_decode = Clock::now();
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
            auto r = s.speculative->step(last, sampler, route, s.draft_route(), *s.pool);
            if (!r) return r.status();
            next = std::move(r).value();
        } else {
            const ForwardHooks hooks = s.hooks(policy, true);
            auto logits =
                s.target->forward(std::span<const int32_t>(&last, 1), Transformer::Logits::Last, route, *s.pool, &hooks);
            if (!logits) return logits.status();
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
    }
    if (s.speculative) {
        stats.draft_tokens_proposed = s.speculative->proposed();
        stats.draft_tokens_accepted = s.speculative->accepted();
    }
    return stats;
}

}  // namespace liyab
