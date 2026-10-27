#include "liyab/engine.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#include "core/chat_template.h"
#include "core/grammar.h"
#include "core/log.h"
#include "core/quant.h"
#include "core/sampling.h"
#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/direct_io.h"
#include "core/key_scaling.h"
#include "core/kv_reuse.h"
#include "core/kv_store.h"
#include "core/lora.h"
#include "core/os.h"
#include "core/pooling.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"
#include "liyab/speculative_decoder.h"

#include "liyab/triple_buffer_loader.h"

#if defined(LIYAB_ENABLE_EXPERIMENTAL)
#include "liyab/experimental/early_exit.h"
#include "liyab/experimental/egls.h"
#include "liyab/experimental/tdss.h"
#include "liyab/experimental/head_pruner.h"
#endif

namespace liyab {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Key scales kept in EngineConfig::cache_dir, one file per model:
// "keys-<id>.bin" = magic, kKeyScalingVersion, id, count, factors. The id
// covers what the factors depend on: the model file, the rule's version, the
// cache format (the measurement runs through it) and the kernel set.
constexpr char kKeyScalesMagic[4] = {'L', 'Y', 'K', 'S'};
constexpr size_t kKeptKeyScales = 16;  // files in cache_dir at most, least recently used out first

uint64_t key_scales_id(const Transformer& model) {
    const uint64_t parts[] = {model.file_fingerprint(), kKeyScalingVersion,
                              static_cast<uint64_t>(model.kv_cache().dtype())};
    const std::string_view isa = quant::kernel_isa();
    return xxhash64(isa.data(), isa.size(), xxhash64(parts, sizeof parts));
}

std::string key_scales_path(const std::string& dir, uint64_t id) {
    char name[32];
    std::snprintf(name, sizeof name, "keys-%016llx.bin", static_cast<unsigned long long>(id));
    return os::path_utf8(os::utf8_path(dir) / name);
}

// The factors kept at `path` for `id` (empty: the model needs none);
// nothing when no file was kept or it is damaged (they are then measured again).
std::optional<std::vector<float>> read_key_scales(const std::string& path, uint64_t id) {
    std::ifstream in(os::utf8_path(path), std::ios::binary);
    char magic[4] = {};
    uint32_t version = 0;
    uint64_t stored_id = 0, count = 0;
    in.read(magic, sizeof magic);
    in.read(reinterpret_cast<char*>(&version), sizeof version);
    in.read(reinterpret_cast<char*>(&stored_id), sizeof stored_id);
    in.read(reinterpret_cast<char*>(&count), sizeof count);
    if (!in || std::memcmp(magic, kKeyScalesMagic, sizeof magic) != 0 || version != kKeyScalingVersion ||
        stored_id != id || count > (uint64_t{1} << 24)) {
        return std::nullopt;
    }
    std::vector<float> scales(static_cast<size_t>(count));
    in.read(reinterpret_cast<char*>(scales.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!in) return std::nullopt;
    return scales;
}

// Writes the factors (temporary file, then renamed) and keeps the directory
// to its kKeptKeyScales most recently used files. Failures only cost a new
// measurement at the next load: logged, not returned.
void keep_key_scales(const std::string& dir, uint64_t id, std::span<const float> scales) {
    std::error_code ec;
    std::filesystem::create_directories(os::utf8_path(dir), ec);
    const std::string path = key_scales_path(dir, id);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(os::utf8_path(tmp), std::ios::binary | std::ios::trunc);
        const uint32_t version = kKeyScalingVersion;
        const uint64_t count = scales.size();
        out.write(kKeyScalesMagic, sizeof kKeyScalesMagic);
        out.write(reinterpret_cast<const char*>(&version), sizeof version);
        out.write(reinterpret_cast<const char*>(&id), sizeof id);
        out.write(reinterpret_cast<const char*>(&count), sizeof count);
        out.write(reinterpret_cast<const char*>(scales.data()), static_cast<std::streamsize>(scales.size_bytes()));
        if (!out) {
            LIYAB_LOG_WARN("keys: cannot write %s", tmp.c_str());
            std::filesystem::remove(os::utf8_path(tmp), ec);
            return;
        }
    }
    std::filesystem::rename(os::utf8_path(tmp), os::utf8_path(path), ec);
    if (ec) {
        LIYAB_LOG_WARN("keys: cannot keep the key scales in %s: %s", dir.c_str(), ec.message().c_str());
        return;
    }
    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> kept;
    std::filesystem::directory_iterator it(os::utf8_path(dir), ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string name = os::path_utf8(it->path().filename());
        if (name.starts_with("keys-") && name.ends_with(".bin")) {
            std::error_code ignored;
            kept.emplace_back(std::filesystem::last_write_time(it->path(), ignored), it->path());
        }
    }
    if (kept.size() <= kKeptKeyScales) return;
    std::sort(kept.begin(), kept.end());  // oldest first
    for (size_t i = 0; i + kKeptKeyScales < kept.size(); ++i) std::filesystem::remove(kept[i].second, ec);
}

// Latent attention's form ratio (LatentFormChooser) kept in
// EngineConfig::cache_dir, one file per model: "latent-<id>.bin" = magic,
// kLatentRatioVersion, id, ratio. The id covers what the measurement depends
// on: the model file, the KV cache format and the kernel set (the device is
// the one the directory is on). Measured again when missing or damaged.
// Without a cache_dir a ratio measured in this process is kept in memory, so
// every engine of the process takes the same forms.
constexpr char kLatentRatioMagic[4] = {'L', 'Y', 'L', 'F'};
constexpr uint32_t kLatentRatioVersion = 1;
constexpr size_t kKeptLatentRatios = 16;  // files in cache_dir at most, least recently used out first

uint64_t latent_ratio_id(const Transformer& model) {
    const uint64_t parts[] = {model.file_fingerprint(), kLatentRatioVersion,
                              static_cast<uint64_t>(model.kv_cache().dtype())};
    const std::string_view isa = quant::kernel_isa();
    return xxhash64(isa.data(), isa.size(), xxhash64(parts, sizeof parts));
}

std::string latent_ratio_path(const std::string& dir, uint64_t id) {
    char name[40];
    std::snprintf(name, sizeof name, "latent-%016llx.bin", static_cast<unsigned long long>(id));
    return os::path_utf8(os::utf8_path(dir) / name);
}

std::optional<float> read_latent_ratio(const std::string& path, uint64_t id) {
    std::ifstream in(os::utf8_path(path), std::ios::binary);
    char magic[4] = {};
    uint32_t version = 0;
    uint64_t stored_id = 0;
    float ratio = 0.0f;
    in.read(magic, sizeof magic);
    in.read(reinterpret_cast<char*>(&version), sizeof version);
    in.read(reinterpret_cast<char*>(&stored_id), sizeof stored_id);
    in.read(reinterpret_cast<char*>(&ratio), sizeof ratio);
    if (!in || std::memcmp(magic, kLatentRatioMagic, sizeof magic) != 0 || version != kLatentRatioVersion ||
        stored_id != id || !(ratio >= LatentFormChooser::kMinRatio && ratio <= LatentFormChooser::kMaxRatio)) {
        return std::nullopt;
    }
    return ratio;
}

// Writes the ratio (temporary file, then renamed) and keeps the directory to
// its kKeptLatentRatios most recently used ratio files. Failures only cost a
// new measurement at the next load: logged, not returned.
void keep_latent_ratio(const std::string& dir, uint64_t id, float ratio) {
    std::error_code ec;
    std::filesystem::create_directories(os::utf8_path(dir), ec);
    const std::string path = latent_ratio_path(dir, id);
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(os::utf8_path(tmp), std::ios::binary | std::ios::trunc);
        out.write(kLatentRatioMagic, sizeof kLatentRatioMagic);
        out.write(reinterpret_cast<const char*>(&kLatentRatioVersion), sizeof kLatentRatioVersion);
        out.write(reinterpret_cast<const char*>(&id), sizeof id);
        out.write(reinterpret_cast<const char*>(&ratio), sizeof ratio);
        if (!out) {
            LIYAB_LOG_WARN("latent attention: cannot write %s", tmp.c_str());
            std::filesystem::remove(os::utf8_path(tmp), ec);
            return;
        }
    }
    std::filesystem::rename(os::utf8_path(tmp), os::utf8_path(path), ec);
    if (ec) {
        LIYAB_LOG_WARN("latent attention: cannot keep the form ratio in %s: %s", dir.c_str(), ec.message().c_str());
        return;
    }
    std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> kept;
    std::filesystem::directory_iterator it(os::utf8_path(dir), ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string name = os::path_utf8(it->path().filename());
        if (name.starts_with("latent-") && name.ends_with(".bin")) {
            std::error_code ignored;
            kept.emplace_back(std::filesystem::last_write_time(it->path(), ignored), it->path());
        }
    }
    if (kept.size() <= kKeptLatentRatios) return;
    std::sort(kept.begin(), kept.end());  // oldest first
    for (size_t i = 0; i + kKeptLatentRatios < kept.size(); ++i) std::filesystem::remove(kept[i].second, ec);
}

// Ratios measured in this process for models without a cache_dir, by id.
std::mutex latent_ratios_mutex;
std::map<uint64_t, float>& latent_ratios() {
    static std::map<uint64_t, float> ratios;
    return ratios;
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
    // The prefix cache comes out of the memory budget: set aside when deciding
    // whether the model fits, and on a MoE whose experts stream, taken only
    // from what the expert cache can spare (TransformerOptions::reserve_bytes).
    const int64_t budget_mb = std::max<int64_t>(config.memory_budget_mb, 0);
    const int64_t reserve_mb = std::min(std::max<int64_t>(config.prefix_cache_mb, 0), budget_mb / 2);
    loader_options.memory_budget_bytes = static_cast<uint64_t>(budget_mb - reserve_mb) << 20;
    auto file = MmapLoader::open(path, loader_options);
    if (!file) return file.status();
    TransformerOptions options;
    options.context_length = config.context_length;
    // Segment reuse needs every block's rows of a segment: with it on,
    // windowed attention blocks keep every position (their per-block layout
    // keeps only their window otherwise).
    options.keep_windowed_rows = config.segment_reuse;
    options.sliding_window = config.sliding_window;
    options.sink_tokens = config.kv_sink_tokens;
    // A pooling model holds one input at a time: its own KV format (EngineConfig::pooling_kv_cache_type).
    options.kv_type = declares_pooling(**file) ? config.pooling_kv_cache_type : config.kv_cache_type;
    options.max_batch = max_batch;
    options.expert_cache_bytes = config.expert_cache_mb > 0 ? config.expert_cache_mb << 20 : config.expert_cache_mb;
    options.memory_budget_bytes = loader_options.memory_budget_bytes;
    options.reserve_bytes = static_cast<size_t>(reserve_mb) << 20;
    options.requant_bits = config.requant_bits;
    options.expert_mass = config.moe_expert_mass;
    options.max_experts = config.moe_max_experts;
    options.skip_slow = config.moe_skip_slow;
    options.repack_cpu = repack_cpu && !config.triple_buffer_loading;
    options.latent_form = config.latent_form == LatentAttentionForm::Absorbed   ? LatentForm::Absorbed
                          : config.latent_form == LatentAttentionForm::Expanded ? LatentForm::Expanded
                                                                                : LatentForm::Auto;
    // EngineConfig::max_sequences plays no part here: the plan is the one of
    // a single conversation, and each sequence opened later borrows its
    // memory from the expert cache (Engine::open_sequence).
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
    static constexpr size_t kExact = SIZE_MAX;
    struct AdapterEntry;

    // ---- Sequences (Engine::open_sequence) ----------------------------------
    // One conversation the engine keeps: the default one (id 0, the Engine's
    // own calls) or one opened with open_sequence(). The transformer holds
    // one context live (`bound`'s); every other sequence's is parked in its
    // `parked` Context (bind() swaps them in O(1)).
    struct Seq {
        int32_t id = 0;
        std::atomic<int32_t> priority{static_cast<int32_t>(SequencePriority::Foreground)};
        // Its transformer state while another sequence's is live; an empty
        // Context while its own is.
        std::unique_ptr<Transformer::Context> parked;
        // What it borrowed from the expert cache for its recurrent states,
        // snapshots and checkpoints (open_sequence(); empty for the default
        // sequence, planned at load, and for models without streamed
        // experts), repaid when it closes.
        ExpertStore::Loan loan;
        // Tokens whose KV / recurrent state its context holds, in order
        // (n_past() == context.size()); context lookup drafts from it. Not
        // used with a draft model, whose cache moves in step with the target's
        // during speculation.
        std::vector<int32_t> context;
        // The text the first `context_text_tokens` tokens of `context` encode (0:
        // not known): a prompt given as text, followed by the pieces of the reply
        // tokens generated after it. Those tokens need not be the text's own
        // encoding (a model may write a word in pieces the tokenizer would
        // merge), so a prompt is matched against this text, not re-encoded
        // (encode_continuation). The text outlives work that was cancelled or cut
        // short, which leaves its tokens in place, and follows rewinds
        // (cut_text): losing it made the next turn re-encode the conversation,
        // diverge at the first such reply and, on a hybrid model, start over.
        std::string context_text;
        size_t context_text_tokens = 0;
        // Where the last prompt generate() processed ends in `context` (or the
        // end of what a restore or load_state() brought back): a prompt that
        // leaves the context before it belongs to another conversation, or
        // branches this one (an edited message), and the prefix cache sets the
        // context aside; one that leaves it after edits a draft typed ahead.
        size_t last_prompt_end = 0;
        // Segment reuse: the first position of `context` whose rows were
        // reused away from their prefix (kExact: none): a request that did not
        // opt in never continues them, and the prefix cache never keeps them,
        // so its restores stay exact. The request being served: its segments
        // as given (`request_marks`, byte offsets, or token indices with
        // `request_marks_tokens`), as token ranges (`request_segments`), and
        // whether it opted in (`request_reuse`).
        size_t exact_end = kExact;
        std::span<const PromptSegment> request_marks;
        bool request_marks_tokens = false;
        std::vector<TokenSegment> request_segments;
        bool request_reuse = false;
        // Its LoRA adapter: `active` at `active_scale` (nullptr: none), and
        // what its context was computed with (context_key, 0: the base
        // model; the store's seed follows it whenever the sequence is bound).
        AdapterEntry* active = nullptr;
        float active_scale = 1.0f;
        uint64_t context_key = 0;
        std::atomic<bool> cancel{false};
        std::atomic<bool> cancel_prefill{false};
        std::mutex call;  // one call at a time on a sequence
        // Scheduler state, under Impl::sched: where its call is, and the
        // OS thread running it (performance hints).
        enum class Phase { Idle, Away, Working, Waiting } phase = Phase::Idle;
        int32_t thread_id = 0;
        [[nodiscard]] bool foreground() const noexcept {
            return priority.load(std::memory_order_relaxed) == static_cast<int32_t>(SequencePriority::Foreground);
        }
    };
    // Sequences by id order; seqs[0] is the default one. Elements keep their
    // address (unique_ptr); the vector changes under `sched` and `model`.
    std::vector<std::unique_ptr<Seq>> seqs;
    Seq* main = nullptr;   // seqs[0], set at creation (read without a lock)
    Seq* bound = nullptr;  // whose context is the transformer's live one
    int32_t next_sequence_id = 1;

    // Segment reuse (EngineConfig::segment_reuse, core/kv_reuse.h). `reuse`
    // holds the calibrated settings and `reuse_info` whether it is on (set at
    // creation, or when the background calibration ends: under `model`, and
    // under reuse_info_mutex for segment_reuse()).
    ReuseSettings reuse;
    SegmentReuseInfo reuse_info;
    mutable std::mutex reuse_info_mutex;
    // The calibration, run in the background (calibrate_in_background): the
    // job, the prompts' style (the chat template's), the context it runs on
    // (swapped with the conversation's for each stage), its thread, and how
    // calls stop it: each one counts itself in `calls_in_flight` and sets
    // `calibration_stop`, which the calibration checks between blocks.
    std::unique_ptr<ReuseCalibrator> calibrator;
    ReusePromptStyle reuse_style;
    std::unique_ptr<KvCache> calibration_kv;
    int32_t calibration_n_past = 0;
    std::thread calibration_thread;
    std::mutex calibration_mutex;  // calibration_closing, last_call
    std::condition_variable calibration_wake;
    bool calibration_closing = false;
    Clock::time_point last_call = Clock::now();  // when the last call ended (or the engine was made)
    std::atomic<int32_t> calls_in_flight{0};
    std::atomic<bool> calibration_stop{false};
    // Sets a sequence's request segments for the scope of one call.
    struct Request {
        Impl& s;
        Seq& q;
        Request(Impl& impl, Seq& seq, std::span<const PromptSegment> marks, bool tokens) : s(impl), q(seq) {
            q.request_marks = marks;
            q.request_marks_tokens = tokens;
            q.request_segments.clear();
            // Not while an adapter is selected: the share was calibrated on the base model.
            q.request_reuse = !marks.empty() && s.reuse_ready() && s.store != nullptr && q.context_key == 0;
        }
        ~Request() {
            q.request_marks = {};
            q.request_segments.clear();
            q.request_reuse = false;
        }
        Request(const Request&) = delete;
        Request& operator=(const Request&) = delete;
    };
    [[nodiscard]] bool reuse_ready() const {
        std::lock_guard<std::mutex> lock(reuse_info_mutex);
        return reuse_info.status.is_ok();
    }

    // Forgets sequence q's context. The model must be held with q bound.
    void clear_context(Seq& q) {
        target->reset();
        if (draft) draft->reset();
        q.context.clear();
        q.context_text.clear();
        q.context_text_tokens = 0;
        q.last_prompt_end = 0;
        q.exact_end = kExact;
    }

    // Moves `end` back over the text that tokens[i] stands for at the end of
    // text[0, end): a prompt's tokens as encode() matched them (a special
    // token's own text, such as a chat template's "<|im_end|>"), a reply's as
    // their pieces (a special one generated decodes to ""). False when the
    // text does not end with the token's.
    bool step_back(std::span<const int32_t> tokens, std::string_view text, size_t& end, size_t i) const {
        const int32_t id = tokens[i];
        const bool implicit_bos = i == 1 && tokens[0] == tokenizer->bos() &&
                                  !text.starts_with(tokenizer->token_text(tokens[0]));
        const int32_t previous = i == 0 || implicit_bos ? -1 : tokens[i - 1];
        for (const std::string_view t : {tokenizer->prompt_text(id, previous), std::string_view(tokenizer->piece(id))}) {
            if (t.size() <= end && text.substr(end - t.size(), t.size()) == t) {
                end -= t.size();
                return true;
            }
        }
        return false;
    }

    // Before q's `context` is cut to its first `n` tokens: the text of those
    // tokens, by stepping back over the others from the end of the text (not
    // known when it does not end with them).
    void cut_text(Seq& q, size_t n) {
        if (n >= q.context_text_tokens) return;
        size_t end = q.context_text.size();
        bool known = n > 0;
        for (size_t i = q.context_text_tokens; known && i-- > n;) known = step_back(q.context, q.context_text, end, i);
        if (known) {
            q.context_text.resize(end);
            q.context_text_tokens = n;
        } else {
            q.context_text.clear();
            q.context_text_tokens = 0;
        }
    }

    // The text of q's first n context tokens and how many of them it covers,
    // as cut_text(n) would leave them (empty and 0: not known).
    std::pair<std::string_view, size_t> text_of(const Seq& q, size_t n) const {
        if (n >= q.context_text_tokens) return {q.context_text, q.context_text_tokens};
        size_t end = q.context_text.size();
        for (size_t i = q.context_text_tokens; i-- > n;) {
            if (!step_back(q.context, q.context_text, end, i)) return {{}, 0};
        }
        return n > 0 ? std::pair<std::string_view, size_t>(std::string_view(q.context_text).substr(0, end), n)
                     : std::pair<std::string_view, size_t>({}, 0);
    }

    // Segment reuse: a request that did not opt in never continues rows
    // reused away from their prefix: the context goes back to the first of
    // them, and what follows is processed again (any position restores on
    // the attention-only models segment reuse runs on). Model held, q bound.
    void drop_approximate(Seq& q) {
        const size_t keep = q.exact_end;
        LIYAB_LOG_INFO("segment reuse: the context's reused rows (from token %zu of %zu) are processed again for a "
                       "request that does not reuse segments",
                       keep, q.context.size());
        if (keep == 0 || !target->truncate(static_cast<int32_t>(keep)).is_ok()) {
            clear_context(q);
            return;
        }
        cut_text(q, keep);
        q.context.resize(keep);
        q.last_prompt_end = std::min(q.last_prompt_end, keep);
        q.exact_end = kExact;
    }

    // Tokens for `text` on sequence q. When it continues a text the engine
    // knows (q's context's, or a context set aside in the prefix cache, which
    // running the prompt then restores), that text's own tokens followed by
    // the encoding of the rest: re-encoding an earlier reply need not give
    // back the tokens that were generated, and on a hybrid model any
    // difference would recompute from a state snapshot before it.
    Result<std::vector<int32_t>> encode_continuation(Seq& q, std::string_view text, bool add_bos) {
        const bool bos = add_bos && tokenizer->add_bos_default();
        KnownText k = known_text(q, text);
        if (!k.tokens.empty()) {
            auto rest = tokenizer->encode_continuation(text.substr(k.bytes), k.tokens.back());
            if (!rest) return rest.status();
            k.tokens.insert(k.tokens.end(), rest->begin(), rest->end());
            return std::move(k.tokens);
        }
        return tokenizer->encode(text, bos);
    }

    // Segment reuse: tokens for a request's text with marked segments, as
    // encode_continuation() finds the text it continues; the rest is split
    // at the segments' boundaries (encode_segmented), whose token ranges go
    // to q.request_segments. A segment inside the known text needs no reuse
    // (the context or the prefix cache restores it exactly).
    Result<std::vector<int32_t>> encode_segments(Seq& q, std::string_view text, bool add_bos) {
        KnownText k = known_text(q, text);
        std::vector<int32_t> tokens = std::move(k.tokens);
        q.request_segments.clear();
        LIYAB_RETURN_IF_ERROR(encode_segmented(*tokenizer, text, !tokens.empty() ? k.bytes : 0, q.request_marks,
                                               add_bos && tokenizer->add_bos_default(), tokens, q.request_segments));
        return tokens;
    }

    // The tokens a text a request sends starts with, when it continues a
    // text the engine knows (see encode_continuation): a copy of those
    // tokens (spelling its first `bytes` bytes), or none. The prefix cache's
    // records are read with the model held (other sequences change them).
    struct KnownText {
        std::vector<int32_t> tokens;
        size_t bytes = 0;
    };
    KnownText known_text(Seq& q, std::string_view text) {
        const std::vector<int32_t>* known = nullptr;
        size_t known_tokens = 0, known_bytes = 0;
        // The rest of the prompt is encoded from a character's first byte:
        // when the known text ends inside a character (a reply's byte-level
        // pieces may split one between two tokens), the tokens step back to
        // the last one ending on a character boundary. Refusing the match
        // instead re-encoded the whole conversation and, on a hybrid model,
        // processed it again.
        auto mid_character = [&](size_t at) { return (static_cast<unsigned char>(text[at]) & 0xC0) == 0x80; };
        auto to_boundary = [&](std::span<const int32_t> tokens, std::string_view known_text, size_t& n, size_t& end) {
            while (n > 0 && end < text.size() && mid_character(end)) {
                if (!step_back(tokens, known_text, end, n - 1)) {
                    n = 0;
                    return;
                }
                --n;
            }
        };
        auto consider = [&](const std::vector<int32_t>& tokens, const std::string& known_text, size_t n) {
            if (n == 0 || text.size() <= known_text.size() || !text.starts_with(known_text)) return;
            size_t end = known_text.size();
            to_boundary(tokens, known_text, n, end);
            if (n > 0 && end >= known_bytes) {
                known = &tokens;
                known_tokens = n;
                known_bytes = end;
            }
        };
        KnownText out;
        if (draft) return out;
        std::unique_ptr<Hold> held;
        if (store) {
            held = std::make_unique<Hold>(*this, q, /*bind=*/false);
            for (const KvStore::Record* r : store->records()) consider(r->tokens, r->text, r->text_tokens);
        }
        consider(q.context, q.context_text, q.context_text_tokens);  // preferred on a tie: nothing to restore
        // A prompt that leaves this context's text partway (a draft edited,
        // a message prepared ahead then typed over): the context's tokens
        // up to the last one whose text the prompt still starts with,
        // found by stepping back from the end of the text as cut_text()
        // does. Re-encoding the whole prompt instead could diverge in an
        // earlier reply and, on a hybrid model, go back past the draft's
        // snapshots.
        size_t same = 0;
        const size_t most = std::min(text.size(), q.context_text.size());
        while (same < most && text[same] == q.context_text[same]) ++same;
        if (q.context_text_tokens > 0 && same < q.context_text.size() && same > known_bytes) {
            size_t n = q.context_text_tokens, end = q.context_text.size();
            while (n > 0 && end > same) {
                if (!step_back(q.context, q.context_text, end, n - 1)) {
                    n = 0;  // the tokens do not spell the text: unknown
                    break;
                }
                --n;
            }
            to_boundary(q.context, q.context_text, n, end);
            if (n > 0 && end > known_bytes) {
                known = &q.context;
                known_tokens = n;
                known_bytes = end;
            }
        }
        if (known != nullptr) {
            out.tokens.assign(known->begin(), known->begin() + static_cast<std::ptrdiff_t>(known_tokens));
            out.bytes = known_bytes;
        }
        return out;
    }

    // After a run over `prompt` (the encoding of `text`): records the text
    // q's context now holds, if the context still starts with the whole prompt.
    void remember_text(Seq& q, std::span<const int32_t> prompt, std::string_view text) {
        if (q.context.size() < prompt.size() || !std::equal(prompt.begin(), prompt.end(), q.context.begin())) {
            q.context_text.clear();
            q.context_text_tokens = 0;
            return;
        }
        q.context_text.assign(text);
        for (size_t i = prompt.size(); i < q.context.size(); ++i) q.context_text += tokenizer->piece(q.context[i]);
        q.context_text_tokens = q.context.size();
    }

    // Prefix cache (EngineConfig::prefix_cache_mb, prefix_cache_dir,
    // prefix_cache_disk_mb): contexts set aside when a prompt from another
    // conversation replaced them (the chat, the assistant sheet, a background
    // agent, other apps through an API) or an embedding call took the
    // context, and the context left when the engine is destroyed (storage
    // tier only), kept with the text they encode in a content-addressed store
    // of KV chunks (core/kv_store.h). A prompt that continues one restores it
    // instead of processing it again, up to where they diverge; a prefix
    // several conversations share (system prompt, tools) is stored once and
    // restored for any of them. A context is set aside when a prompt replaces
    // it (see reuse_prefix: another conversation, or a branch of this one),
    // not for a draft edited after the last reply, which keeps using the
    // context's own snapshots. RAM holds what prefix_cache_mb allows, the
    // rest goes to files in prefix_cache_dir, where later engines and
    // processes find it: a phone's RAM is better spent on weights, and its
    // storage writes and reads back a ~100 MB context in a fraction of a
    // second, where processing it again takes minutes.
    std::unique_ptr<KvStore> store;
    static constexpr size_t kMinPrefixEntry = 128;  // shorter contexts are cheaper to process again
    // Engine::prefix_cache_stats(): updated after each use of the store,
    // behind a lock of its own, so a monitor may read them while a
    // generation runs.
    mutable std::mutex prefix_stats_mutex;
    PrefixCacheStats prefix_stats;

    [[nodiscard]] bool prefix_cache_on() const { return store != nullptr; }

    // ---- LoRA adapters (Engine::load_adapter / set_adapter, core/lora.h) ----
    // The loaded adapters, each mapped while resident: at most
    // config.adapter_memory_mb of them, the least recently selected
    // released first and mapped again from its file when selected. Each
    // sequence selects its own (Seq::active, at Seq::active_scale, nullptr:
    // none), bound to its transformer context; Seq::context_key names what
    // its context was computed with (0: the base model), and the store's
    // seed follows the bound sequence's (bind(), seed_for()), so a context is
    // never continued, set aside or restored under another adapter or
    // scale. Changed with the model held, and under adapters_mutex where
    // adapters() reads.
    struct AdapterEntry {
        std::string name;
        std::string path;
        std::unique_ptr<LoraAdapter> adapter;  // nullptr: released under the cap
        size_t bytes = 0;
        int32_t rank = 0;
        int32_t matrices = 0;
        float alpha = 0.0f;
        uint64_t last_used = 0;
    };
    std::vector<std::unique_ptr<AdapterEntry>> adapters;
    uint64_t adapter_clock = 0;
    KvKey base_seed;  // the store's seed without an adapter (store_seed())
    mutable std::mutex adapters_mutex;

    AdapterEntry* find_adapter(std::string_view name) {
        for (auto& e : adapters) {
            if (e->name == name) return e.get();
        }
        return nullptr;
    }
    // Whether some sequence has `e` selected.
    [[nodiscard]] bool selected(const AdapterEntry* e) const {
        return std::any_of(seqs.begin(), seqs.end(), [e](const std::unique_ptr<Seq>& q) { return q->active == e; });
    }
    // What contexts computed with `adapter` at `scale` are keyed by: its
    // file's hash and the scale; 0 for none.
    static uint64_t adapter_key(const LoraAdapter* adapter, float scale) {
        if (adapter == nullptr) return 0;
        uint32_t bits = 0;
        std::memcpy(&bits, &scale, sizeof bits);
        const uint64_t parts[] = {adapter->id(), bits};
        const uint64_t key = xxhash64(parts, sizeof parts);
        return key != 0 ? key : 1;
    }
    [[nodiscard]] KvKey seed_for(uint64_t key) const {
        if (key == 0) return base_seed;
        const uint64_t parts[] = {base_seed.hi, base_seed.lo, key};
        return kv_seed_key(parts);
    }
    // Releases the least recently selected adapters but `keep` and those a
    // sequence has selected until `bytes` more fit under the cap. Model held.
    void fit_adapters(size_t bytes, const AdapterEntry* keep) {
        const size_t cap = static_cast<size_t>(std::max<int64_t>(config.adapter_memory_mb, 0)) << 20;
        for (;;) {
            size_t used = 0;
            AdapterEntry* oldest = nullptr;
            for (auto& e : adapters) {
                if (!e->adapter) continue;
                used += e->bytes;
                if (e.get() != keep && !selected(e.get()) && (oldest == nullptr || e->last_used < oldest->last_used)) {
                    oldest = e.get();
                }
            }
            if (used + bytes <= cap || oldest == nullptr) return;
            LIYAB_LOG_INFO("adapters: %s released (%.1f MiB) to stay within %lld MiB", oldest->name.c_str(),
                           static_cast<double>(oldest->bytes) / (1024.0 * 1024.0),
                           static_cast<long long>(config.adapter_memory_mb));
            std::lock_guard<std::mutex> lock(adapters_mutex);
            oldest->adapter.reset();
        }
    }
    // Checks an adapter against the model: its architecture, and every
    // target bound once to the live context (then the bound sequence's own
    // adapter is bound again). Model held.
    Status check_adapter(const LoraAdapter& adapter) {
        const auto arch = target->file().get_string("general.architecture");
        if (!adapter.architecture().empty() && arch && *arch != adapter.architecture()) {
            return Status(ErrorCode::InvalidArgument, "the adapter is for " + adapter.architecture() +
                                                          " models; this model is " + std::string(*arch));
        }
        const Status fits = target->set_adapter(&adapter, 1.0f);
        const AdapterEntry* own = bound != nullptr ? bound->active : nullptr;
        const Status back = target->set_adapter(own != nullptr ? own->adapter.get() : nullptr,
                                                bound != nullptr ? bound->active_scale : 1.0f, pool.get());
        LIYAB_RETURN_IF_ERROR(fits);
        return back;
    }
    // Makes `e` (nullptr: none) the adapter q's passes apply at `scale`. A
    // context computed otherwise is set aside under its own adapter (the
    // store's seed is still its) and cleared; then the store's seed follows.
    // Model held, q bound.
    Status select_adapter(Seq& q, AdapterEntry* e, float scale) {
        if (e != nullptr && !e->adapter) {  // released under the cap: mapped again
            auto loaded = LoraAdapter::load(e->path);
            Status fits = loaded ? check_adapter(**loaded) : loaded.status();
            if (fits.is_ok()) {
                fit_adapters((*loaded)->bytes(), e);
                std::lock_guard<std::mutex> lock(adapters_mutex);
                e->bytes = (*loaded)->bytes();
                e->adapter = std::move(loaded).value();
            } else {
                LIYAB_LOG_WARN("adapters: %s could not be mapped again: %s", e->name.c_str(), fits.to_string().c_str());
                (void)select_adapter(q, nullptr, 1.0f);
                return fits;
            }
        }
        const LoraAdapter* adapter = e != nullptr ? e->adapter.get() : nullptr;
        const uint64_t key = adapter_key(adapter, scale);
        if (key != q.context_key) {
            stash_context(q);  // under the adapter it was computed with (no-op without a store)
            clear_context(q);
        }
        if (Status s = target->set_adapter(adapter, scale, pool.get()); !s.is_ok()) {
            (void)target->set_adapter(nullptr);
            clear_context(q);
            if (store) store->set_seed(base_seed);
            q.context_key = 0;
            std::lock_guard<std::mutex> lock(adapters_mutex);
            q.active = nullptr;
            return s;
        }
        if (store) store->set_seed(seed_for(key));
        if (key != q.context_key && prefix_cache_on()) note_store();
        q.context_key = key;
        std::lock_guard<std::mutex> lock(adapters_mutex);
        q.active = e;
        q.active_scale = scale;
        if (e != nullptr) e->last_used = ++adapter_clock;
        return Status::ok();
    }

    ~Impl() {
        stop_calibration();
        // The contexts left go to storage, so a later engine or process
        // continues them instead of processing them again. Best effort: a
        // destructor cannot report a failure.
        if (!store || !store->has_storage() || !target || draft) return;
        try {
            std::lock_guard<std::mutex> lock(model);
            for (auto& q : seqs) {
                bind(*q);
                stash_context(*q, /*storage_only=*/true);
            }
        } catch (...) {
            LIYAB_LOG_WARN("prefix cache: the context could not be set aside");
        }
    }

    // What the store's bytes depend on beyond the model file and the cache
    // (Transformer::context_fingerprint): the kernels and backends that
    // computed them and every option that changes hidden states, so chunks
    // are only restored where recomputing them would give the same bits.
    // Latent attention's prefill form changes how the latents of later blocks
    // round, so stored chunks belong to the form and the device's measured
    // ratio they were computed with (0 for every other model, whose keys this
    // leaves as they were).
    [[nodiscard]] uint64_t latent_seed() const {
        if (target->config().mla.kv_rank == 0) return 0;
        const float ratio = target->latent_chooser().ratio();
        uint32_t bits = 0;
        std::memcpy(&bits, &ratio, sizeof bits);
        return (static_cast<uint64_t>(static_cast<uint32_t>(target->latent_form()) + 1) << 32) | bits;
    }

    [[nodiscard]] KvKey store_seed(bool repacked, int32_t chunk_pages) const {
        auto hash = [](std::string_view text) { return xxhash64(text.data(), text.size()); };
        auto kind = [&](const Backend* b) {  // "cpu", "metal", "vulkan": not the thread count it describes
            const std::string d = b->description();
            return hash(std::string_view(d).substr(0, d.find(' ')));
        };
        auto bits = [](float f) {
            uint32_t u = 0;
            std::memcpy(&u, &f, sizeof u);
            return static_cast<uint64_t>(u);
        };
        constexpr uint64_t kStoreSeedVersion = 1;  // chunk_pages below: chunks of another size are other keys
        const Route route = normal_route();
        const ExperimentalConfig& x = config.experimental;
        const uint64_t parts[] = {kStoreSeedVersion,
                                  target->file_fingerprint(),
                                  target->context_fingerprint(),
                                  hash(quant::kernel_isa()),
                                  kind(route.attention),
                                  kind(route.ffn),
                                  repacked ? 1u : 0u,
                                  static_cast<uint64_t>(target->memory_plan().requant_bits),
                                  bits(config.moe_expert_mass),
                                  static_cast<uint64_t>(config.moe_max_experts),
                                  bits(config.moe_skip_slow),
                                  (x.early_exit ? 1u : 0u) | (x.egls ? 2u : 0u) | (x.head_pruning ? 4u : 0u) |
                                      (x.tdss ? 8u : 0u) | (x.tdss_force ? 16u : 0u),
                                  bits(x.early_exit_threshold),
                                  bits(x.egls_threshold),
                                  bits(x.head_keep_ratio),
                                  static_cast<uint64_t>(chunk_pages),
                                  latent_seed()};
        return kv_seed_key(parts);
    }

    void open_store(const EngineConfig& cfg, bool repacked) {
        // With a budget, a MoE whose experts stream gives the prefix cache only
        // what its expert cache can spare (see load_model).
        size_t ram = cfg.prefix_cache_mb > 0 ? static_cast<size_t>(cfg.prefix_cache_mb) << 20 : 0;
        if (ram > 0 && cfg.memory_budget_mb > 0 && target->expert_store() != nullptr) {
            const size_t granted = target->memory_plan().reserved_bytes;
            if (granted < ram) {
                LIYAB_LOG_INFO("prefix cache: %zu of %lld MiB (more would leave the expert cache under %.0f%% of the experts)",
                               granted >> 20, static_cast<long long>(cfg.prefix_cache_mb),
                               100.0 * Transformer::kTightCache);
            }
            ram = granted;
        }
        const bool disk = cfg.prefix_cache_disk_mb > 0 && !cfg.prefix_cache_dir.empty();
        if (ram == 0 && !disk) return;
        const KvCache& kv = target->kv_cache();
        if (kv.window() > 0) {
            LIYAB_LOG_INFO("prefix cache: off with a sliding window (the positions it releases are gone)");
            return;
        }
        if (draft) {
            LIYAB_LOG_INFO("prefix cache: off with a draft model");
            return;
        }
        // Chunks of about kChunkBytes, in whole pages: each is a file on
        // storage, and creating and renaming a file costs ~120 us on a Mac
        // (APFS), about writing 1 MB, so pages of a few hundred KB (Qwen3.5,
        // the 35B: 0.4-0.7 MB) would make a long context hundreds of small
        // files. At most kMaxChunkPages, so a prefix several conversations
        // share is still kept to within 256 tokens (the rest of its last
        // chunk is each conversation's own).
        constexpr size_t kChunkBytes = size_t{2} << 20;
        constexpr int32_t kMaxChunkPages = 4;
        // Windowed attention blocks keep only their window's rows: chunks
        // hold the other blocks' rows, and the windowed rows go to the store
        // page by page as window pages (KvStore::Config::window; a model whose
        // every attention block is windowed has 0-byte chunks of tokens only).
        const int32_t window = kv.window_of_windowed();
        if (window < 0 || (window > 0 && (target->recurrent_state_bytes() > 0 || kv.window_page_bytes() % sizeof(float) != 0))) {
            LIYAB_LOG_INFO("prefix cache: off with windowed attention blocks of several windows or next to recurrent "
                           "blocks (their window pages are not kept apart)");
            return;
        }
        const auto chunk_pages = static_cast<int32_t>(
            std::clamp<size_t>(kChunkBytes / std::max<size_t>(kv.page_bytes(), 1), 1, kMaxChunkPages));
        KvStore::Config sc;
        sc.seed = store_seed(repacked, chunk_pages);
        base_seed = sc.seed;  // an adapter's contexts are keyed under seed_for() (select_adapter)
        sc.chunk_tokens = kv.config().page_tokens * chunk_pages;
        sc.chunk_bytes = kv.page_bytes() * static_cast<size_t>(chunk_pages);
        sc.state_floats = (window > 0 ? kv.window_page_bytes() : target->recurrent_state_bytes()) / sizeof(float);
        sc.window = window;
        sc.ram_bytes = ram;
        if (disk) {
            sc.dir = cfg.prefix_cache_dir;
            sc.disk_bytes = static_cast<size_t>(cfg.prefix_cache_disk_mb) << 20;
        }
        // Chunks go to storage compressed where that pays (core/kv_codec.h),
        // encoded and decoded on the engine's pool: the store is used under
        // the engine's lock, never while a forward pass runs.
        sc.layout = kv_row_layout(kv.config());
        sc.page_tokens = kv.config().page_tokens;
        sc.lossy = cfg.prefix_cache_lossy;
        sc.pool = pool.get();
        sc.calibration_dir = cfg.cache_dir;  // what reads and decoding cost here, measured once
        store = KvStore::open(sc);
        if (!store) return;
        const KvStore::Stats st = store->stats();
        LIYAB_LOG_INFO("prefix cache: %zu MiB of RAM, %zu MiB of storage%s%s; %d chunks and %d contexts of this model "
                       "kept from earlier runs",
                       st.ram_cap >> 20, st.disk_cap >> 20, store->has_storage() ? " in " : "",
                       store->has_storage() ? cfg.prefix_cache_dir.c_str() : "", st.chunks_disk, st.records);
        note_store();
    }

    // Copies the store's state into prefix_stats.
    void note_store() {
        if (!store) return;
        const KvStore::Stats st = store->stats();
        std::lock_guard<std::mutex> lock(prefix_stats_mutex);
        prefix_stats.ram_bytes = st.ram_bytes;
        prefix_stats.ram_cap_bytes = st.ram_cap;
        prefix_stats.storage_bytes = st.disk_bytes;
        prefix_stats.storage_cap_bytes = st.disk_cap;
        prefix_stats.chunks = st.chunks_ram + st.chunks_disk;
        prefix_stats.state_images = st.states;
        prefix_stats.contexts = st.records;
        prefix_stats.chunk_tokens = store->chunk_tokens();
        prefix_stats.segments = st.segments;
    }

    // Sets the context aside in the store (see above); `storage_only` when
    // the engine is being destroyed.
    void stash_context(Seq& q, bool storage_only = false) {
        // Rows reused away from their prefix (segment reuse) are never kept:
        // a context holding them is set aside up to the first one, so what
        // the store restores is always exact.
        const size_t exact = std::min(q.context.size(), q.exact_end);
        if (!store || exact < kMinPrefixEntry || q.context.size() != static_cast<size_t>(target->n_past())) return;
        // The whole context, the last reply included: that conversation's next
        // prompt re-sends it, and on a hybrid model a reply that encodes
        // differently goes back to the pinned snapshot kept with it (the last
        // prompt's end). Setting it aside at that snapshot instead made every
        // return process the last reply again (123 tokens, 8 s on a 35B).
        const auto watch = Clock::now();
        const KvCache& kv = target->kv_cache();
        KvStore::ContextSource source;
        source.tokens = std::span<const int32_t>(q.context).first(exact);
        std::tie(source.text, source.text_tokens) = text_of(q, exact);
        // Storage is written straight from the cache's pages and the model's
        // states: a chunk's image is its pages' images one after the other,
        // a whole page's being the page itself; only a partial last page is
        // packed (into `packed`, alive while put() runs).
        const int32_t page_tokens = kv.config().page_tokens;
        const int32_t chunk_pages = store->chunk_tokens() / page_tokens;
        std::deque<std::vector<uint8_t>> packed;
        source.rows = [&](int32_t chunk, int32_t n) {
            std::vector<std::span<const uint8_t>> parts;
            for (int32_t k = 0; k * page_tokens < n; ++k) {
                const int32_t page = chunk * chunk_pages + k;
                const int32_t rows = std::min(page_tokens, n - k * page_tokens);
                const uint8_t* data = kv.page_data(page);
                if (data != nullptr && rows == page_tokens) {
                    parts.emplace_back(data, kv.page_bytes());
                } else {
                    packed.emplace_back(kv.rows_bytes(rows));
                    if (!kv.read_rows(page, rows, packed.back().data()).is_ok()) return std::vector<std::span<const uint8_t>>();
                    parts.emplace_back(packed.back());
                }
            }
            return parts;
        };
        // (Segment reuse runs on attention-only models only: a hybrid
        // model's context is always exact, so its states are its end's.)
        std::deque<Transformer::StateBuffer> window_images;
        if (kv.has_windowed_layers()) {
            // Windowed blocks: every window page the cache still holds (the
            // window before the end, the snapshots' windows), keyed at its
            // page's end, or at the context's end or a snapshot inside a page
            // (the slots after it zero). The restore points the model keeps
            // (Transformer::kept_states(): the oldest and the pinned snapshot,
            // the end) are marked pinned, so a restored context gets them
            // back as its snapshots.
            const auto n = static_cast<int32_t>(exact);
            std::vector<KvStore::StateMark> at;
            for (int32_t pg = 0; pg * page_tokens < n; ++pg) {
                if (kv.window_page_mapped(pg)) at.push_back({std::min((pg + 1) * page_tokens, n), false});
            }
            for (const Transformer::KeptState& k : target->kept_states()) {
                if (k.pos > 0 && k.pos <= n && kv.window_page_mapped((k.pos - 1) / page_tokens)) at.push_back({k.pos, true});
            }
            std::sort(at.begin(), at.end(), [](const KvStore::StateMark& a, const KvStore::StateMark& b) {
                return a.pos != b.pos ? a.pos < b.pos : a.pinned > b.pinned;  // a restore point first
            });
            at.erase(std::unique(at.begin(), at.end(),
                                 [](const KvStore::StateMark& a, const KvStore::StateMark& b) { return a.pos == b.pos; }),
                     at.end());
            source.states = std::move(at);
            source.state_parts = [&](int32_t pos) {
                const int32_t pg = (pos - 1) / page_tokens;
                window_images.emplace_back(kv.window_page_bytes() / sizeof(float));
                auto* bytes = reinterpret_cast<uint8_t*>(window_images.back().data());
                if (!kv.read_window_rows(pg, pos - pg * page_tokens, bytes).is_ok()) return std::vector<std::span<const float>>();
                return std::vector<std::span<const float>>{std::span<const float>(window_images.back())};
            };
        } else {
            for (const Transformer::KeptState& k : target->kept_states()) source.states.push_back({k.pos, k.pinned});
            source.state_parts = [this](int32_t pos) { return target->kept_state_parts(pos); };
        }
        const KvStore::PutResult put = store->put(source, storage_only);
        const double ms = ms_since(watch);
        const double mib = 1.0 / (1024.0 * 1024.0);
        if (put.whole) {
            LIYAB_LOG_INFO("prefix cache: set aside %zu tokens in %.0f ms: %d of %d chunks new, %d state image(s) new, "
                           "%.1f MiB to RAM, %.1f MiB to storage",
                           exact, ms, put.chunks_written, put.chunks, put.states_written,
                           static_cast<double>(put.bytes_ram) * mib, static_cast<double>(put.bytes_disk) * mib);
        } else {
            LIYAB_LOG_INFO("prefix cache: a context of %zu tokens does not fit the cache whole (%d of %d chunks new)",
                           exact, put.chunks_written, put.chunks);
        }
        {
            std::lock_guard<std::mutex> lock(prefix_stats_mutex);
            ++prefix_stats.set_asides;
            prefix_stats.chunks_written += static_cast<uint64_t>(put.chunks_written);
            prefix_stats.chunks_reused += static_cast<uint64_t>(put.chunks_reused);
            prefix_stats.set_aside_ms += ms;
        }
        note_store();
    }

    // Writes the KV rows of `plan` from `rows_from` on (the live context
    // holds the rows below for the same tokens) and its recurrent states,
    // then makes it the context.
    Status restore_plan(const KvStore::Plan& plan, size_t rows_from) {
        KvCache& kv = target->mutable_kv_cache();
        const int32_t positions = plan.positions;
        // Windowed attention blocks: only the pages of the window before the
        // end (and before each snapshot), never the whole context's; their
        // rows come from the plan's window pages.
        std::vector<int32_t> pins;
        for (const KvStore::RecordState& rs : plan.snapshots) pins.push_back(rs.pos);
        LIYAB_RETURN_IF_ERROR(kv.reserve_context(positions, pins));
        const int32_t c = store->chunk_tokens();
        const int32_t page_tokens = kv.config().page_tokens;
        const int32_t chunk_pages = c / page_tokens;
        // A chunk's pages that it restores whole are read straight into the
        // cache; the others (partial, or partly kept in place) into scratch,
        // then copied to the rows wanted. The whole chunk is read either way:
        // its hash covers all of it.
        std::vector<std::vector<uint8_t>> scratch;
        struct Pending {
            int32_t page, rows, from, to;
            size_t slot;
        };
        std::vector<Pending> pending;
        std::vector<std::span<uint8_t>> parts;
        store->read_ahead(plan.chunks);  // compressed chunks: the next file is read while one decodes
        for (size_t i = 0; i < plan.chunks.size(); ++i) {
            const auto start = static_cast<int32_t>(i) * c;
            const int32_t n = plan.chunk_sizes[i];
            const int32_t from = std::max(static_cast<int32_t>(rows_from) - start, 0);
            const int32_t to = std::min(positions - start, n);
            if (from >= to) continue;
            parts.clear();
            pending.clear();
            size_t used = 0;
            for (int32_t k = 0; k * page_tokens < n; ++k) {
                const int32_t page = static_cast<int32_t>(i) * chunk_pages + k;
                const int32_t first = k * page_tokens;
                const int32_t rows = std::min(page_tokens, n - first);
                uint8_t* data = kv.mutable_page_data(page);
                if (data != nullptr && rows == page_tokens && from <= first && first + rows <= to) {
                    parts.emplace_back(data, kv.page_bytes());
                    continue;
                }
                if (scratch.size() <= used) scratch.emplace_back();
                scratch[used].resize(kv.rows_bytes(rows));
                parts.emplace_back(scratch[used]);
                const int32_t lo = std::clamp(from - first, 0, rows), hi = std::clamp(to - first, 0, rows);
                pending.push_back({page, rows, lo, hi, used});
                ++used;
            }
            LIYAB_RETURN_IF_ERROR(store->read_chunk(plan.chunks[i], parts));
            for (const Pending& p : pending) {
                if (p.from < p.to) LIYAB_RETURN_IF_ERROR(kv.write_rows(p.page, scratch[p.slot].data(), p.rows, p.from, p.to));
            }
        }
        // Windowed blocks: the window pages of the windows before the end and
        // before each snapshot, each up to its own position (the snapshots
        // are restore points of the windowed rows only: no recurrent states).
        std::vector<Transformer::OwnedSnapshot> window_snapshots;
        if (!plan.window_pages.empty()) {
            for (const KvStore::RecordState& rs : plan.snapshots) window_snapshots.push_back({rs.pos, rs.pinned, {}});
            Transformer::StateBuffer image(kv.window_page_bytes() / sizeof(float));
            for (const KvStore::RecordState& rs : plan.window_pages) {
                const std::span<float> into(image);
                LIYAB_RETURN_IF_ERROR(store->read_state(rs.key, {&into, 1}));
                const int32_t q = (rs.pos - 1) / page_tokens;
                LIYAB_RETURN_IF_ERROR(
                    kv.write_window_rows(q, reinterpret_cast<const uint8_t*>(image.data()), 0, rs.pos - q * page_tokens));
            }
        }
        // Recurrent states straight into the model and into the buffers of
        // the snapshots being replaced: no intermediate copy, no allocation
        // or zero-fill of 62.8 MiB images on a 35B.
        std::vector<Transformer::OwnedSnapshot> snapshots;
        if (target->recurrent_state_bytes() > 0) {
            target->begin_restore();
            const std::vector<std::span<float>> live = target->recurrent_parts();
            LIYAB_RETURN_IF_ERROR(store->read_state(plan.state, live));
            for (const KvStore::RecordState& rs : plan.snapshots) {
                Transformer::OwnedSnapshot snap{rs.pos, rs.pinned, target->take_snapshot_buffer()};
                const std::span<float> into(snap.data);
                LIYAB_RETURN_IF_ERROR(store->read_state(rs.key, {&into, 1}));
                snapshots.push_back(std::move(snap));
            }
        } else {
            snapshots = std::move(window_snapshots);
        }
        return target->adopt_context(positions, std::move(snapshots), plan.state_pinned);
    }

    // Restores the longest prefix of `tokens` (at most `limit` of them) the
    // store holds when it is longer than what the live context keeps of its
    // `have` tokens in common with them, cut back to where they diverge (a
    // reply sent back as text may encode differently from the tokens
    // generated; on hybrid models, to a stored state image); the tokens kept,
    // or 0.
    size_t restore_cached(Seq& q, std::span<const int32_t> tokens, size_t limit, size_t have) {
        const KvStore::Plan plan = store->plan(tokens, limit);
        const auto positions = static_cast<size_t>(plan.positions);
        const size_t live = q.context.empty() ? 0 : static_cast<size_t>(target->restorable_prefix(static_cast<int32_t>(have)));
        if (positions <= live || positions < kMinPrefixEntry) return 0;
        const auto watch = Clock::now();
        const size_t rows_from = q.context.empty() ? 0 : std::min(have, positions);
        if (Status st = restore_plan(plan, rows_from); !st.is_ok()) {
            LIYAB_LOG_WARN("prefix cache: %s", st.to_string().c_str());
            clear_context(q);
            note_store();
            return 0;
        }
        if (plan.record != nullptr) {
            q.context = plan.record->tokens;
            q.context_text = plan.record->text;
            q.context_text_tokens = plan.record->text_tokens;
            cut_text(q, positions);
            q.context.resize(positions);
        } else {
            q.context.assign(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(positions));
            q.context_text.clear();
            q.context_text_tokens = 0;
        }
        q.last_prompt_end = positions;
        // The rows kept in place may be reused ones (a request that opted in
        // to segment reuse); the restored ones are exact.
        if (q.exact_end >= rows_from) q.exact_end = kExact;
        store->used(plan);
        const double ms = ms_since(watch);
        LIYAB_LOG_INFO("prefix cache: restored %zu tokens (%zu kept in place) from %zu chunk(s) in %.0f ms", positions,
                       rows_from, plan.chunks.size(), ms);
        {
            std::lock_guard<std::mutex> lock(prefix_stats_mutex);
            ++prefix_stats.restores;
            prefix_stats.restored_tokens += positions;
            prefix_stats.restore_ms += ms;
        }
        note_store();
        return positions;
    }

    // `approximate_ok`: the request opted in to segment reuse, so it may
    // continue rows reused away from their prefix (else they go first).
    size_t reuse_prefix(Seq& q, std::span<const int32_t> tokens, size_t limit, bool approximate_ok = false) {
        if (draft || q.context.size() != static_cast<size_t>(target->n_past())) {
            clear_context(q);
            return 0;
        }
        if (!approximate_ok && q.exact_end < q.context.size()) drop_approximate(q);
        // An empty context (reset_context(), or embed() / rerank(), which use
        // the context and leave it empty): a prompt that continues a context
        // the prefix cache set aside gets it back.
        if (q.context.empty()) return prefix_cache_on() ? restore_cached(q, tokens, limit, 0) : 0;
        size_t common = 0;
        const size_t n = std::min({q.context.size(), tokens.size(), limit});
        while (common < n && q.context[common] == tokens[common]) ++common;
        if (common == q.context.size()) return common;  // pure continuation: nothing to drop
        // Attention-only models rewind to any position; hybrid ones to the
        // nearest recurrent-state checkpoint or snapshot below it.
        const auto keep = static_cast<size_t>(target->restorable_prefix(static_cast<int32_t>(common)));
        if (prefix_cache_on()) {
            // Set aside when the prompt replaces the context: it shares less
            // than half of it, or leaves it before its last prompt's end and
            // rewinding would process at least kMinPrefixEntry tokens again
            // (another chat after a long shared system prompt; on a hybrid
            // model, back to its nearest snapshot, possibly the start). A
            // draft edited after the last reply is not: it would write state
            // images at every keystroke.
            const bool replaced = common < q.context.size() / 2 ||
                                  (common < q.last_prompt_end && q.context.size() - keep >= kMinPrefixEntry);
            if (replaced) stash_context(q);
            // The store may hold more of this prompt than the context keeps
            // (the conversation it continues, set aside earlier).
            if (const size_t restored = restore_cached(q, tokens, limit, common); restored > 0) return restored;
            if (q.context.empty()) return 0;  // a cached context failed to restore
        }
        // Diverging from the context costs a recomputation from `keep`: say
        // where, so a slow turn can be traced to its cause.
        LIYAB_LOG_INFO("context: the prompt leaves the context at token %zu of %zu; %zu kept", common, q.context.size(),
                       keep);
        if (keep > 0 && target->truncate(static_cast<int32_t>(keep)).is_ok()) {
            cut_text(q, keep);
            q.context.resize(keep);
            q.last_prompt_end = std::min(q.last_prompt_end, keep);
            if (keep <= q.exact_end) q.exact_end = kExact;  // cut before any reused row: exact again
            return keep;
        }
        clear_context(q);
        return 0;
    }
    // Segment reuse for the request being served: the stored segments of
    // `tokens` after its first `exact` positions (the context holds them)
    // and before `limit`, reused (core/kv_reuse.h). Returns where processing
    // resumes: past the last reused segment, or `exact` when nothing was
    // reused or reuse failed (the context then still holds the exact prefix).
    size_t reuse_request_segments(Seq& q, std::span<const int32_t> tokens, size_t exact, size_t limit, const Route& route,
                                  int32_t chunk, const std::atomic<bool>* stop, GenerationStats* stats) {
        if (!q.request_reuse || q.request_segments.empty() || exact >= limit) return exact;
        const auto t0 = Clock::now();
        auto r = reuse_segments(*target, *store, tokens, q.request_segments, static_cast<int32_t>(exact),
                                static_cast<int32_t>(limit), reuse, route, *pool, chunk, stop);
        if (!r) {
            if (r.status().code() != ErrorCode::Cancelled) {
                LIYAB_LOG_WARN("segment reuse: %s; the prompt is processed", r.status().to_string().c_str());
            }
            if (static_cast<size_t>(target->n_past()) != exact) clear_context(q);
            note_store();
            return q.context.size();
        }
        if (r->end <= static_cast<int32_t>(exact)) return exact;
        q.context.assign(tokens.begin(), tokens.begin() + r->end);
        if (r->first_reused >= 0) q.exact_end = std::min(q.exact_end, static_cast<size_t>(r->first_reused));
        const double ms = ms_since(t0);
        LIYAB_LOG_INFO("segment reuse: %d tokens of %d segment(s) reused (%d of them processed again), %d processed "
                       "between them, in %.0f ms (rows %.0f ms)",
                       r->reused, r->segments, r->recomputed, r->computed, ms, r->rows_ms);
        if (stats != nullptr) {
            stats->segment_reused_tokens += r->reused;
            stats->segment_recomputed_tokens += r->recomputed;
            stats->segment_ms += ms;
        }
        {
            std::lock_guard<std::mutex> lock(prefix_stats_mutex);
            prefix_stats.segment_reuses += static_cast<uint64_t>(r->segments);
            prefix_stats.segment_reused_tokens += static_cast<uint64_t>(r->reused);
            prefix_stats.segment_recomputed_tokens += static_cast<uint64_t>(r->recomputed);
        }
        note_store();
        return static_cast<size_t>(r->end);
    }

    // Keeps the request's segments the context now holds and the store has
    // no copy of (core/kv_reuse.h, keep_segments).
    void keep_request_segments(Seq& q) {
        if (!q.request_reuse || q.request_segments.empty() || !store) return;
        // The context starts with the prompt the segments index (else
        // nothing of them is kept: the context was replaced or cut).
        const int32_t kept = keep_segments(*target, *store, q.context, q.request_segments, reuse.min_tokens);
        if (kept > 0) {
            std::lock_guard<std::mutex> lock(prefix_stats_mutex);
            prefix_stats.segments_kept += static_cast<uint64_t>(kept);
        }
        note_store();
    }

    // Whether segment reuse can run (EngineConfig::segment_reuse), and the
    // share of reused tokens processed again: measured once per model and
    // settings (ReuseCalibrator, on the CPU like the key scales, so it
    // depends on the model and the kernels only) and kept in cache_dir. The
    // measurement does not hold up loading: it runs in the background while
    // the engine is idle (calibrate_in_background), and until it ends
    // segment reuse is not on, so marked prompts are processed as usual.
    void prepare_segment_reuse() {
        auto off = [&](Status why) { set_reuse_off(std::move(why)); };
        if (!config.segment_reuse) return off(Status(ErrorCode::Unsupported, "off in EngineConfig::segment_reuse"));
        if (draft) return off(Status(ErrorCode::Unsupported, "not with a draft model (no prefix cache)"));
        if (Status s = target->reuse_support(); !s.is_ok()) return off(s);
        if (!store) {
            return off(Status(ErrorCode::Unsupported,
                              "segments are kept in the prefix cache's store: set prefix_cache_mb or "
                              "prefix_cache_disk_mb (none with a sliding window)"));
        }
        if (!(config.segment_reuse_max_kl > 0.0f)) {
            return off(Status(ErrorCode::InvalidArgument, "segment_reuse_max_kl must be positive"));
        }
        const uint64_t id = reuse_calibration_id();
        const std::string path = config.cache_dir.empty() ? std::string() : reuse_calibration_path(config.cache_dir, id);
        bool met = false;
        if (!path.empty() && read_reuse_calibration(path, id, met)) {
            reuse_info.from_cache = true;
            std::error_code ignored;
            std::filesystem::last_write_time(os::utf8_path(path), std::filesystem::file_time_type::clock::now(), ignored);
            return publish_reuse(met);
        }
        // Prompts shaped like real ones: through the model's chat template
        // (plain text without one), thinking off where it has the switch.
        reuse_style.chat = chat_template ? &*chat_template : nullptr;
        if (chat_template) reuse_style.special = chat_special;
        const auto source = target->file().get_string("tokenizer.chat_template");
        reuse_style.thinking_off = source && source->find("enable_thinking") != std::string::npos;
        auto job = ReuseCalibrator::create(*target, *tokenizer, reuse_style, reuse, config.segment_reuse_max_kl);
        if (!job) return off(job.status());
        calibrator = std::move(job).value();
        std::lock_guard<std::mutex> lock(reuse_info_mutex);
        reuse_info.calibrating = true;
        reuse_info.progress = 0.0f;
        reuse_info.status = Status(ErrorCode::Busy,
                                   "calibrating in the background while the engine is idle; until then marked "
                                   "prompts are processed as usual");
        LIYAB_LOG_INFO("segment reuse: calibrating in the background (%s prompts)",
                       reuse_style.chat != nullptr ? "chat-template" : "plain-text");
    }

    void set_reuse_off(Status why) {
        {
            std::lock_guard<std::mutex> lock(reuse_info_mutex);
            reuse_info.calibrating = false;
            reuse_info.status = std::move(why);
        }
        if (config.segment_reuse) LIYAB_LOG_INFO("segment reuse: off: %s", reuse_info.status.message().c_str());
    }

    // On with the calibrated share, or off when no share met the target.
    void publish_reuse(bool met) {
        if (!met) {
            char why[240];
            std::snprintf(why, sizeof why,
                          "processing %.0f%% of reused tokens again still left KL %.4f (bound %.4f), above the "
                          "target %.4f",
                          100.0 * reuse_info.fraction, reuse_info.kl, reuse_info.bound, config.segment_reuse_max_kl);
            return set_reuse_off(Status(ErrorCode::Unsupported, why));
        }
        {
            std::lock_guard<std::mutex> lock(reuse_info_mutex);
            reuse.fraction = reuse_info.fraction;
            reuse_info.calibrating = false;
            reuse_info.progress = 1.0f;
            reuse_info.status = Status::ok();
        }
        LIYAB_LOG_INFO("segment reuse: on, %.0f%% of reused tokens processed again (KL %.4f, bound %.4f, top-1 "
                       "%.1f%%)%s",
                       100.0 * reuse.fraction, reuse_info.kl, reuse_info.bound, 100.0 * reuse_info.top1,
                       reuse_info.from_cache ? ", as measured before" : "");
    }

    // ---- The calibration in the background -----------------------------------
    // A thread runs it one stage at a time (a few passes over one prompt's
    // documents) once no call has used the engine for kCalibrationIdle, on a
    // context of its own swapped in for the stage, so the conversation's is
    // left as it was. A call arriving meanwhile stops the stage between two
    // blocks and waits for that (Call); the stage runs again at the next
    // idle time. Started at the end of Engine::create(), stopped by ~Impl.
    static constexpr auto kCalibrationIdle = std::chrono::seconds(2);
    void start_calibration() {
        if (!calibrator) return;
        {
            std::lock_guard<std::mutex> lock(calibration_mutex);
            last_call = Clock::now();
        }
        calibration_thread = std::thread([this] { calibrate_in_background(); });
    }
    void stop_calibration() {
        {
            std::lock_guard<std::mutex> lock(calibration_mutex);
            calibration_closing = true;
        }
        calibration_stop.store(true);
        calibration_wake.notify_all();
        if (calibration_thread.joinable()) calibration_thread.join();
    }
    void calibrate_in_background() {
        for (;;) {
            {
                std::unique_lock<std::mutex> wait(calibration_mutex);
                for (;;) {
                    if (calibration_closing) return;
                    const auto now = Clock::now();
                    const auto ready = last_call + kCalibrationIdle;
                    if (now >= ready && calls_in_flight.load() == 0) break;
                    // A call ending notifies; the timeout covers the idle time.
                    calibration_wake.wait_until(wait, std::max(ready, now + std::chrono::milliseconds(100)));
                }
            }
            std::unique_lock<std::mutex> lock(model, std::try_to_lock);
            if (!lock.owns_lock()) {
                std::unique_lock<std::mutex> wait(calibration_mutex);
                calibration_wake.wait_for(wait, std::chrono::milliseconds(100));
                continue;
            }
            // A call that arrived before this store set the flag earlier and
            // counts in calls_in_flight; one arriving after sets it again.
            calibration_stop.store(false);
            bool over = false;
            if (calls_in_flight.load() == 0) over = calibration_stage();
            lock.unlock();
            if (over) return;
        }
    }
    // One stage, the model held: true when the calibration is over (segment
    // reuse then on or off). It runs on the live context's cache, swapped
    // out for its own and back (whichever sequence is bound).
    bool calibration_stage() {
        if (!calibration_kv) {
            auto kv = target->make_kv_cache(calibrator->positions());
            if (!kv) return end_calibration(kv.status());
            calibration_kv = std::move(kv).value();
            calibration_n_past = 0;
        }
        if (Status s = target->swap_context(calibration_kv, calibration_n_past); !s.is_ok()) return end_calibration(s);
        Result<bool> r = false;
        {
            const PowerManager::Work thermal(*power);  // thermal sampling while this runs
            // The base model's measurement: no adapter, whichever is selected.
            const LoraAdapter* adapter = target->adapter();
            if (adapter != nullptr) (void)target->set_adapter(nullptr);
            r = calibrator->step(*target, draft_route(), *pool, target->max_batch(), &calibration_stop);
            if (adapter != nullptr) (void)target->set_adapter(adapter, bound->active_scale, pool.get());  // bound before
        }
        // The conversation's context back (the same caches: cannot fail).
        (void)target->swap_context(calibration_kv, calibration_n_past);
        {
            std::lock_guard<std::mutex> lock(reuse_info_mutex);
            reuse_info.progress = calibrator->progress();
        }
        if (!r) return r.status().code() == ErrorCode::Cancelled ? false : end_calibration(r.status());
        if (!*r) return false;
        const ReuseCalibration& c = calibrator->result();
        {
            std::lock_guard<std::mutex> lock(reuse_info_mutex);
            reuse_info.fraction = c.fraction;
            reuse_info.kl = c.kl;
            reuse_info.bound = c.bound;
            reuse_info.top1 = c.top1;
        }
        std::string tried;
        for (size_t i = 0; i < c.kls.size(); ++i) {
            char one[64];
            std::snprintf(one, sizeof one, "%s%.0f%%: %.4f (%.4f)", i > 0 ? ", " : "", 100.0 * kReuseFractions[i],
                          c.kls[i], c.bounds[i]);
            tried += one;
        }
        LIYAB_LOG_INFO("segment reuse: calibrated in %.0f ms of background work: %.0f%% of reused tokens processed "
                       "again, KL %.4f (bound %.4f), top-1 %.1f%% (target KL %.4f; KL (bound) by share: %s)",
                       c.ms, 100.0 * c.fraction, c.kl, c.bound, 100.0 * c.top1, config.segment_reuse_max_kl,
                       tried.c_str());
        if (!config.cache_dir.empty()) {
            const uint64_t id = reuse_calibration_id();
            keep_reuse_calibration(config.cache_dir, reuse_calibration_path(config.cache_dir, id), id, c.met);
        }
        publish_reuse(c.met);
        calibrator.reset();
        calibration_kv.reset();
        return true;
    }
    bool end_calibration(Status why) {
        set_reuse_off(std::move(why));
        calibrator.reset();
        calibration_kv.reset();
        return true;
    }

    // ---- Scheduler: sequences decoded together --------------------------------
    // `model` serializes everything that uses the transformer, the prefix
    // cache's store and the thread pool: a forward pass, an exclusive
    // operation of one sequence (Hold: a prefix-cache restore or set-aside,
    // a speculative step, a saved state, embeddings), the background
    // calibration. A sequence whose call is the only one running runs exactly
    // as the engine always ran a single conversation (alone()); while others
    // run, each forward pass it needs is a request (submit()) that one of
    // the waiting threads, the leader, runs together with the other
    // sequences' requests in one forward_batch() pass, each sequence keeping
    // its own sampling, callbacks and stopping on its own thread.
    std::mutex model;
    std::mutex sched;  // what follows, and Seq::phase / thread_id
    std::condition_variable sched_cv;
    std::mutex pace_mutex;  // PowerManager pacing and hint threads (not thread-safe)
    int32_t in_call = 0;     // sequences in a call
    int32_t waiting_fg = 0;  // foreground sequences waiting for the model (Hold)
    bool leading = false;    // a thread runs the pending requests
    bool preemptible = false;  // the pass running may be stopped for foreground work
    // The sequences of the stoppable pass running whose request is a
    // prefill() call: their cancel_prefill() stops it too.
    std::vector<const Seq*> running;
    std::atomic<bool> preempt{false};  // that pass's cancel flag
    // What a pass is for: a prompt's chunk (generate), a prefill() chunk, a
    // forced continuation, a decoding step. The last two are small (a few
    // tokens): they share passes freely and are paced.
    enum class PassKind { Prompt, Prefill, Forced, Decode };
    struct PassRequest {
        Seq* seq = nullptr;
        std::span<const int32_t> tokens;
        Transformer::Logits logits = Transformer::Logits::None;
        PassKind kind = PassKind::Prompt;
        // Filled by the pass:
        bool done = false;
        Status status;
        std::vector<float> rows;  // its logits, copied out of the pass (the next one reuses the transformer's)
        bool throttled = false, head_pruned = false, sparse_ffn = false;
        int32_t ffn_skips = 0, exit_layer = -1;
        double paced_ms = 0.0;
        int32_t sequences = 0;  // in its pass
        [[nodiscard]] bool small() const noexcept { return kind == PassKind::Forced || kind == PassKind::Decode; }
    };
    std::vector<PassRequest*> pending;
    // Passes timed by the scheduler: every pass (how long to wait for a
    // sequence about to submit), and passes of small requests by how many
    // sequences they served (whether a background step may join the user's).
    static constexpr int32_t kMaxShared = 16;
    double pass_ms = 0.0;
    std::array<double, kMaxShared + 2> small_pass_ms{};
    SchedulerStats sched_stats;
    // A background step joins a pass of foreground ones only while passes of
    // one more sequence measured at most this many times as long: on a model
    // in RAM a row more costs little, while a mixture of experts streamed
    // from storage reads every chosen expert of the batch, and the user's
    // step would wait for the reads of the background one's.
    static constexpr double kRideAlong = 1.25;
    static constexpr int32_t kReprobe = 64;
    int32_t ride_refused = 0;  // background steps kept out since the last re-measure (ride_along())

    // A call's hold on one sequence: false (Busy) while another call runs on
    // it. Calls on different sequences run at once. The background
    // calibration is not a call: one arriving while it runs stops it between
    // two blocks and waits for that, a fraction of a second, when it needs
    // the model. `wait`: wait for another call on the sequence too
    // (reset_context()).
    class Call {
    public:
        Call(Impl& s, Seq& q, bool wait = false) : s_(s), q_(q), lock_(q.call, std::defer_lock) {
            s.calls_in_flight.fetch_add(1);
            s.calibration_stop.store(true);
            if (wait) {
                lock_.lock();
            } else {
                (void)lock_.try_lock();
            }
            if (!lock_.owns_lock()) {
                s.calls_in_flight.fetch_sub(1);
                return;
            }
            s.enter(q);
        }
        ~Call() {
            if (!lock_.owns_lock()) return;
            s_.leave(q_);
            lock_.unlock();
            {
                std::lock_guard<std::mutex> lock(s_.calibration_mutex);
                s_.last_call = Clock::now();
            }
            s_.calls_in_flight.fetch_sub(1);
            s_.calibration_wake.notify_all();
        }
        Call(const Call&) = delete;
        Call& operator=(const Call&) = delete;
        explicit operator bool() const noexcept { return lock_.owns_lock(); }

    private:
        Impl& s_;
        Seq& q_;
        std::unique_lock<std::mutex> lock_;
    };

    // The model for sequence q, its context live (unless `bind` is false) and
    // the store keyed for it. A foreground sequence goes before passes that
    // have not started yet, and stops a background prefill pass running.
    class Hold {
    public:
        Hold(Impl& s, Seq& q, bool bind = true) : s_(s), q_(q) {
            {
                std::lock_guard<std::mutex> g(s.sched);
                before_ = q.phase;
                if (q.phase != Seq::Phase::Idle) q.phase = Seq::Phase::Away;
                foreground_ = q.foreground();
                if (foreground_) {
                    ++s.waiting_fg;
                    if (s.preemptible) s.preempt.store(true);
                }
            }
            lock_ = std::unique_lock<std::mutex>(s.model);
            if (foreground_) {
                std::lock_guard<std::mutex> g(s.sched);
                --s.waiting_fg;
                s.sched_cv.notify_all();
            }
            if (bind) s.bind(q);
        }
        ~Hold() {
            lock_.unlock();
            std::lock_guard<std::mutex> g(s_.sched);
            if (q_.phase == Seq::Phase::Away) q_.phase = before_;  // Working again: about to submit
            s_.sched_cv.notify_all();
        }
        Hold(const Hold&) = delete;
        Hold& operator=(const Hold&) = delete;

    private:
        Impl& s_;
        Seq& q_;
        Seq::Phase before_ = Seq::Phase::Idle;
        bool foreground_ = false;
        std::unique_lock<std::mutex> lock_;
    };

    // While it lives (the model held), the pass q runs alone may be stopped
    // for foreground work as a shared background pass may: `preempt` is its
    // cancel flag, set by a foreground Hold or submit() and, for a prefill()
    // call's pass (`prefill_call`), by q's cancel_prefill(). `on`: q is a
    // background sequence.
    class Stoppable {
    public:
        Stoppable(Impl& s, Seq& q, bool on, bool prefill_call) : s_(s), on_(on) {
            if (on_) s.begin_stoppable(prefill_call ? std::vector<const Seq*>{&q} : std::vector<const Seq*>{});
        }
        ~Stoppable() {
            if (on_) s_.end_stoppable();
        }
        Stoppable(const Stoppable&) = delete;
        Stoppable& operator=(const Stoppable&) = delete;

    private:
        Impl& s_;
        bool on_;
    };
    // A stoppable pass starts / ends, the model held; `prefill_calls`: its
    // sequences whose request is a prefill() call. One whose stop was
    // already asked (foreground work waits, such a call was cancelled) is
    // stopped at once. Only prefill() calls heed cancel_prefill(): the flag
    // stays set after one, and a generation's pass stopped by it would run
    // again and be stopped again.
    void begin_stoppable(std::vector<const Seq*> prefill_calls) {
        std::lock_guard<std::mutex> g(sched);
        preemptible = true;
        running = std::move(prefill_calls);
        const bool asked = waiting_fg > 0 || std::any_of(pending.begin(), pending.end(), [](const PassRequest* r) {
                               return r->seq->foreground();
                           }) || std::any_of(running.begin(), running.end(), [](const Seq* q) {
                               return q->cancel_prefill.load(std::memory_order_relaxed);
                           });
        preempt.store(asked);
    }
    void count_stop() {
        std::lock_guard<std::mutex> g(sched);
        ++sched_stats.preempted;
        LIYAB_LOG_INFO("scheduler: a background prompt pass stopped for foreground work (%llu so far); it runs again",
                       static_cast<unsigned long long>(sched_stats.preempted));
    }
    void end_stoppable() {
        std::lock_guard<std::mutex> g(sched);
        preemptible = false;
        running.clear();
    }
    // Tokens per prompt pass while other sequences decode: what their next
    // token waits for, so at most max_batch (the passes a streamed MoE or
    // latent attention would make longer read more per token, see
    // prefill_chunk()). Latent attention keeps its own passes: each pass's
    // form follows from its size, and a sequence's passes must take the
    // forms they take alone.
    [[nodiscard]] int32_t shared_chunk() const {
        if (target->config().mla.kv_rank > 0) return prefill_chunk();
        return std::min(prefill_chunk(), target->max_batch());
    }

    // Generation and prefill on a sequence, its call held (defined below).
    Result<GenerationStats> generate(Seq& q, std::span<const int32_t> prompt, const SamplingParams& params,
                                     const TokenCallback& on_token, const std::string_view* text,
                                     const ForceCallback* force = nullptr, const GenerateOptions* options = nullptr);
    Result<GenerationStats> generate_text(Seq& q, std::string_view prompt, const SamplingParams& params,
                                          const TokenCallback& on_token, const ForceCallback* force,
                                          const GenerateOptions* options);
    Result<GenerationStats> generate_tokens_on(Seq& q, std::span<const int32_t> prompt, const SamplingParams& params,
                                               const TokenCallback& on_token, const GenerateOptions& options);
    Status prefill_on(Seq& q, std::string_view text, std::span<const PromptSegment> segments, bool add_bos);

    // Makes q's context the transformer's live one (O(1): the contexts are
    // swapped, not copied) and keys the store for its adapter. Model held.
    void bind(Seq& q) {
        if (bound != &q) {
            target->swap_context(*bound->parked);  // bound's state back into its Context, an empty one live
            target->swap_context(*q.parked);       // q's live, its Context empty
            bound = &q;
        }
        if (store) store->set_seed(seed_for(q.context_key));
    }

    Seq* find_sequence(int32_t id) {
        std::lock_guard<std::mutex> g(sched);
        for (auto& q : seqs) {
            if (q->id == id) return q.get();
        }
        return nullptr;
    }

    // A call starts / ends on q (Call).
    void enter(Seq& q) {
        std::vector<int32_t> threads = pool->worker_thread_ids();
        {
            std::lock_guard<std::mutex> g(sched);
            ++in_call;
            q.phase = Seq::Phase::Away;
            q.thread_id = ThreadPool::current_thread_id();
            for (const auto& other : seqs) {
                if (other->thread_id != 0) threads.push_back(other->thread_id);
            }
        }
        // The threads that decode, for the power manager's performance hints:
        // the pool's workers and every call's thread (a Dart isolate may move
        // between OS threads from one call to the next). Set outside `sched`:
        // pacing sleeps holding pace_mutex. Not changed when a call ends, so
        // one conversation's calls keep the hint session open, as before.
        if (power) {
            std::lock_guard<std::mutex> p(pace_mutex);
            power->set_hint_threads(std::move(threads));
        }
    }
    void leave(Seq& q) {
        std::lock_guard<std::mutex> g(sched);
        --in_call;
        q.phase = Seq::Phase::Idle;
        q.thread_id = 0;
        sched_cv.notify_all();
    }
    // Whether q's call is the only one: then it runs as a single sequence
    // always did (exact same passes, speculative drafts, pacing per token).
    bool alone() {
        std::lock_guard<std::mutex> g(sched);
        return in_call <= 1 && pending.empty();
    }
    // Whether requests may share a pass: not with the experimental lossy
    // hooks, which act on single passes (early exit, FFN skipping) or would
    // apply to every sequence in it; each sequence then has passes of its
    // own, so its output is what it gives alone.
    [[nodiscard]] bool share_passes() const noexcept {
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
        return !early_exit && !egls && !tdss && !head_pruner;
#else
        return true;
#endif
    }
    // How long a leader waits for the sequences about to submit (they
    // sample and call back between passes, well under a millisecond): half
    // a pass, so a late one costs the others little, and joins the next.
    [[nodiscard]] Clock::duration straggler_wait() const {
        const double ms = std::clamp(0.5 * pass_ms, 0.5, 50.0);
        return std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms));
    }
    [[nodiscard]] bool any_working() const {
        return std::any_of(seqs.begin(), seqs.end(),
                           [](const std::unique_ptr<Seq>& q) { return q->phase == Seq::Phase::Working; });
    }
    // Whether a background step may join a pass of `k` sequences' steps.
    // A refusal rests on an estimate only passes of k + 1 refresh, so one
    // step in kReprobe that would be refused joins anyway and re-measures
    // (an expert cache warming up, the host's load changing): that costs the
    // user's steps at most 1/kReprobe of the difference.
    [[nodiscard]] bool ride_along(size_t k) {
        if (k + 1 >= small_pass_ms.size()) return false;
        const double now = small_pass_ms[k], more = small_pass_ms[k + 1];
        if (now == 0.0 || more == 0.0 || more <= kRideAlong * now) return true;
        if (++ride_refused < kReprobe) return false;
        ride_refused = 0;
        return true;
    }

    // Engine::Sequence's destructor: q's call cancelled and waited for, its
    // context set aside in the prefix cache (when there is one) and freed.
    void close_sequence(int32_t id) {
        Seq* q = find_sequence(id);
        if (q == nullptr || q == main) return;
        q->cancel.store(true, std::memory_order_relaxed);
        cancel_prefill(*q);
        std::unique_lock<std::mutex> call(q->call);  // its call, if any, ends (cancelled)
        std::unique_ptr<Seq> gone;
        {
            const std::lock_guard<std::mutex> lock(model);
            bind(*q);
            if (!draft) stash_context(*q);
            bind(*main);
            const std::lock_guard<std::mutex> g(sched);
            for (auto it = seqs.begin(); it != seqs.end(); ++it) {
                if (it->get() != q) continue;
                gone = std::move(*it);
                seqs.erase(it);
                break;
            }
            LIYAB_LOG_INFO("sequences: %d closed (%zu open)", id, seqs.size());
        }
        call.unlock();
        const ExpertStore::Loan loan = gone != nullptr ? std::move(gone->loan) : ExpertStore::Loan{};
        gone.reset();  // its KV pages and recurrent states
        // Their memory back to the expert cache, once freed (never both held
        // at once). Without the model lock: other sequences decode meanwhile.
        if (ExpertStore* store = target->expert_store(); store != nullptr && loan.bytes > 0) {
            store->repay(loan);
            LIYAB_LOG_INFO("sequences: %.1f MiB back to the expert cache (%.2f GiB)",
                           static_cast<double>(loan.bytes) / (1024.0 * 1024.0),
                           static_cast<double>(store->capacity_bytes()) / (1024.0 * 1024.0 * 1024.0));
        }
    }

    // Engine::set_adapter / Sequence::set_adapter, the call held on q.
    Status set_sequence_adapter(Seq& q, std::string_view name, float scale) {
        const Hold hold(*this, q);
        if (name.empty()) return select_adapter(q, nullptr, 1.0f);
        AdapterEntry* e = find_adapter(name);
        if (e == nullptr) return Status(ErrorCode::InvalidArgument, "no adapter named " + std::string(name));
        if (!std::isfinite(scale)) return Status(ErrorCode::InvalidArgument, "the adapter scale must be finite");
        return select_adapter(q, e, scale);
    }

    // Engine::cancel_prefill / Sequence::cancel_prefill: q's prefill() stops
    // between passes, or now when its pass may be stopped.
    void cancel_prefill(Seq& q) noexcept {
        q.cancel_prefill.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> g(sched);
        if (preemptible && std::find(running.begin(), running.end(), &q) != running.end()) preempt.store(true);
    }

    // Runs q's request with the other sequences' (see above) and returns its
    // status once its pass ran. Its logits are in r.rows.
    Status submit(Seq& q, PassRequest& r) {
        std::unique_lock<std::mutex> lock(sched);
        r.seq = &q;
        r.done = false;
        q.phase = Seq::Phase::Waiting;
        pending.push_back(&r);
        if (q.foreground() && preemptible) preempt.store(true);
        sched_cv.notify_all();
        while (!r.done) {
            if (!leading) {
                leading = true;
                lead(lock, r);
                leading = false;
                sched_cv.notify_all();
            } else {
                sched_cv.wait(lock);
            }
        }
        return r.status;
    }

    // The leader's loop, `sched` held: passes until its own request ran.
    void lead(std::unique_lock<std::mutex>& lock, const PassRequest& mine) {
        while (!mine.done) {
            // A foreground sequence's exclusive work first (it waits for the model).
            sched_cv.wait(lock, [&] { return waiting_fg == 0; });
            // The sequences about to submit join this pass, for a while.
            const auto gather = Clock::now();
            sched_cv.wait_until(lock, gather + straggler_wait(), [&] { return !any_working(); });
            sched_stats.gather_ms += ms_since(gather);
            std::vector<PassRequest*> chosen = choose();
            if (chosen.empty()) {
                sched_cv.wait_for(lock, std::chrono::milliseconds(1));
                continue;
            }
            for (PassRequest* r : chosen) std::erase(pending, r);
            // Background work alone in the pass, a prompt among it: it may be
            // stopped when foreground work comes (run_pass()).
            const bool can_stop = std::all_of(chosen.begin(), chosen.end(), [](const PassRequest* r) {
                                      return !r->seq->foreground();
                                  }) && std::any_of(chosen.begin(), chosen.end(), [](const PassRequest* r) {
                                      return !r->small();
                                  });
            lock.unlock();
            const auto t0 = Clock::now();
            bool stopped = false;
            try {
                stopped = run_pass(chosen, can_stop);
            } catch (const std::exception& e) {  // e.g. out of memory: each sequence sees it as its pass's failure
                for (PassRequest* r : chosen) r->status = Status(ErrorCode::Internal, std::string("pass failed: ") + e.what());
            }
            const double ms = ms_since(t0);
            lock.lock();
            sched_stats.pass_ms += ms;
            if (!stopped) {
                pass_ms = pass_ms == 0.0 ? ms : 0.8 * pass_ms + 0.2 * ms;
                const bool small = std::all_of(chosen.begin(), chosen.end(), [](const PassRequest* r) { return r->small(); });
                if (small && chosen.size() < small_pass_ms.size()) {
                    double& e = small_pass_ms[chosen.size()];
                    e = e == 0.0 ? ms : 0.8 * e + 0.2 * ms;
                }
                ++sched_stats.passes;
                if (chosen.size() > 1) {
                    ++sched_stats.shared_passes;
                    for (const PassRequest* r : chosen) sched_stats.shared_rows += r->tokens.size();
                }
            } else {
                ++sched_stats.preempted;
                LIYAB_LOG_INFO("scheduler: a background pass of %zu sequence(s) stopped for foreground work (%llu so "
                               "far); it runs again",
                               chosen.size(), static_cast<unsigned long long>(sched_stats.preempted));
            }
            for (PassRequest* r : chosen) {
                if (stopped && r->status.code() == ErrorCode::Cancelled) {
                    // Stopped for foreground work (its context is back where
                    // the pass started): it runs again later, unless its own
                    // prefill() was cancelled.
                    const bool own = r->kind == PassKind::Prefill && r->seq->cancel_prefill.load();
                    if (!own) {
                        r->status = Status::ok();
                        pending.push_back(r);
                        continue;
                    }
                }
                r->done = true;
                r->seq->phase = Seq::Phase::Working;
            }
            sched_cv.notify_all();
        }
    }

    // The requests of the next pass, `sched` held: every foreground one, and
    // background steps when riding along is cheap (see kRideAlong); else
    // every background one. One at a time when passes are not shared.
    std::vector<PassRequest*> choose() {
        std::vector<PassRequest*> fg, bg;
        for (PassRequest* r : pending) (r->seq->foreground() ? fg : bg).push_back(r);
        if (!share_passes()) return {fg.empty() ? bg.front() : fg.front()};
        if (fg.empty()) return bg;
        for (PassRequest* r : bg) {
            if (r->small() && ride_along(fg.size())) fg.push_back(r);
        }
        return fg;
    }

    // One pass over `chosen` (the model taken here): a single request as its
    // sequence alone would run it (its hooks, its checkpointing), several in
    // one forward_batch(). Returns true when a preemptible pass was stopped
    // (the contexts are then back where it started).
    bool run_pass(const std::vector<PassRequest*>& chosen, bool can_stop) {
        std::unique_lock<std::mutex> lock(model);
        struct Stop {
            Impl& s;
            bool on;
            ~Stop() {
                if (on) s.end_stoppable();
            }
        } stop{*this, can_stop};
        if (can_stop) {
            // A hybrid context returns to a snapshot when its pass is
            // stopped (abandon_pass()): one where the pass starts.
            if (target->config().hybrid()) {
                for (const PassRequest* r : chosen) {
                    bind(*r->seq);
                    target->snapshot_state();
                }
            }
            std::vector<const Seq*> prefill_calls;
            for (const PassRequest* r : chosen) {
                if (r->kind == PassKind::Prefill) prefill_calls.push_back(r->seq);
            }
            begin_stoppable(std::move(prefill_calls));
        }
        const PowerPolicy policy = power->policy();
        apply_policy(policy);
        const Route route = policy.throttled ? throttled_route() : normal_route();
        double slept = 0.0;
        if (std::any_of(chosen.begin(), chosen.end(), [](const PassRequest* r) { return r->small(); })) {
            // Duty cycling per pass: each sequence in it gets a token per slot.
            std::lock_guard<std::mutex> p(pace_mutex);
            slept = static_cast<double>(power->pace_token().count()) / 1000.0;
        }
        Status status;
        if (chosen.size() == 1) {
            PassRequest& r = *chosen.front();
            bind(*r.seq);
            ForwardHooks hooks = r.kind == PassKind::Prompt || r.kind == PassKind::Prefill || !speculative
                                     ? this->hooks(policy, r.kind == PassKind::Decode)
                                     : ForwardHooks{};
            if (r.kind == PassKind::Prefill) hooks.cancel = &r.seq->cancel_prefill;
            if (can_stop) hooks.cancel = &preempt;  // cancel_prefill() sets it too (Engine::cancel_prefill)
            if (r.small() && speculative) target->set_checkpointing(false);  // never rolled back
            auto out = target->forward(r.tokens, r.logits, route, *pool, &hooks);
            r.sequences = 1;
            r.throttled = policy.throttled;
            r.head_pruned = hooks.head_mask != nullptr;
            r.sparse_ffn = hooks.ffn_matmul != nullptr;
            r.ffn_skips = target->last_ffn_skips();
            r.exit_layer = target->last_exit_layer();
            r.paced_ms = slept;
            if (out) {
                r.rows.assign(out->begin(), out->end());
                r.status = Status::ok();
            } else {
                status = out.status();
                if (status.code() == ErrorCode::Cancelled && !target->abandon_pass().is_ok()) target->reset();
                r.status = status;
            }
            return status.code() == ErrorCode::Cancelled && can_stop;
        }
        std::vector<Transformer::BatchItem> items;
        items.reserve(chosen.size());
        for (const PassRequest* r : chosen) {
            Transformer::BatchItem item;
            item.context = r->seq == bound ? nullptr : r->seq->parked.get();
            item.tokens = r->tokens;
            item.logits = r->logits;
            item.checkpoint = !r->small();  // steps are never rolled back here (no drafts in shared passes)
            items.push_back(item);
        }
        ForwardHooks hooks;
        if (can_stop) hooks.cancel = &preempt;
        std::vector<std::span<const float>> out;
        status = target->forward_batch(items, route, *pool, out, &hooks);
        for (size_t i = 0; i < chosen.size(); ++i) {
            PassRequest& r = *chosen[i];
            r.sequences = static_cast<int32_t>(chosen.size());
            r.throttled = policy.throttled;
            r.paced_ms = slept;
            r.status = status;
            if (status.is_ok()) r.rows.assign(out[i].begin(), out[i].end());
        }
        return status.code() == ErrorCode::Cancelled && can_stop;
    }

    // What a calibration depends on: the model, the cache and its key
    // scales (context_fingerprint), the kernels, the procedure's version,
    // the target and the selection settings.
    [[nodiscard]] uint64_t reuse_calibration_id() const {
        uint32_t target_bits = 0;
        std::memcpy(&target_bits, &config.segment_reuse_max_kl, sizeof target_bits);
        const uint64_t parts[] = {target->file_fingerprint(),
                                  target->context_fingerprint(),
                                  kReuseCalibrationVersion,
                                  target_bits,
                                  static_cast<uint64_t>(reuse.criterion),
                                  static_cast<uint64_t>(reuse.select_block),
                                  static_cast<uint64_t>(reuse.first_tokens),
                                  static_cast<uint64_t>(reuse.min_tokens),
                                  reuse.relative ? 1u : 0u};
        const std::string_view isa = quant::kernel_isa();
        return xxhash64(isa.data(), isa.size(), xxhash64(parts, sizeof parts));
    }
    static std::string reuse_calibration_path(const std::string& dir, uint64_t id) {
        char name[32];
        std::snprintf(name, sizeof name, "reuse-%016llx.bin", static_cast<unsigned long long>(id));
        return os::path_utf8(os::utf8_path(dir) / name);
    }
    // A kept calibration: magic, version, id, met, fraction, KL, top-1, bound.
    static constexpr char kReuseMagic[4] = {'L', 'Y', 'R', 'C'};
    bool read_reuse_calibration(const std::string& path, uint64_t id, bool& met) {
        std::ifstream in(os::utf8_path(path), std::ios::binary);
        char magic[4] = {};
        uint32_t version = 0;
        uint64_t stored = 0;
        uint8_t ok = 0;
        float fraction = 0.0f;
        double kl = 0.0, top1 = 0.0, bound = 0.0;
        in.read(magic, sizeof magic);
        in.read(reinterpret_cast<char*>(&version), sizeof version);
        in.read(reinterpret_cast<char*>(&stored), sizeof stored);
        in.read(reinterpret_cast<char*>(&ok), sizeof ok);
        in.read(reinterpret_cast<char*>(&fraction), sizeof fraction);
        in.read(reinterpret_cast<char*>(&kl), sizeof kl);
        in.read(reinterpret_cast<char*>(&top1), sizeof top1);
        in.read(reinterpret_cast<char*>(&bound), sizeof bound);
        if (!in || std::memcmp(magic, kReuseMagic, sizeof magic) != 0 || version != kReuseCalibrationVersion ||
            stored != id || !(fraction >= 0.0f && fraction <= 1.0f)) {
            return false;
        }
        met = ok != 0;
        reuse_info.fraction = fraction;
        reuse_info.kl = kl;
        reuse_info.top1 = top1;
        reuse_info.bound = bound;
        return true;
    }
    // Written through a temporary file; the directory keeps the
    // kKeptReuseCalibrations most recently used (a failure only costs a
    // calibration at the next load: logged, not returned).
    static constexpr size_t kKeptReuseCalibrations = 16;
    void keep_reuse_calibration(const std::string& dir, const std::string& path, uint64_t id, bool met) const {
        std::error_code ec;
        std::filesystem::create_directories(os::utf8_path(dir), ec);
        const std::string tmp = path + ".tmp";
        {
            std::ofstream out(os::utf8_path(tmp), std::ios::binary | std::ios::trunc);
            const uint32_t version = kReuseCalibrationVersion;
            const uint8_t ok = met ? 1 : 0;
            out.write(kReuseMagic, sizeof kReuseMagic);
            out.write(reinterpret_cast<const char*>(&version), sizeof version);
            out.write(reinterpret_cast<const char*>(&id), sizeof id);
            out.write(reinterpret_cast<const char*>(&ok), sizeof ok);
            out.write(reinterpret_cast<const char*>(&reuse_info.fraction), sizeof reuse_info.fraction);
            out.write(reinterpret_cast<const char*>(&reuse_info.kl), sizeof reuse_info.kl);
            out.write(reinterpret_cast<const char*>(&reuse_info.top1), sizeof reuse_info.top1);
            out.write(reinterpret_cast<const char*>(&reuse_info.bound), sizeof reuse_info.bound);
            if (!out) {
                LIYAB_LOG_WARN("segment reuse: cannot write %s", tmp.c_str());
                std::filesystem::remove(os::utf8_path(tmp), ec);
                return;
            }
        }
        if (!os::replace_file(tmp, path)) {
            LIYAB_LOG_WARN("segment reuse: cannot keep the calibration in %s", dir.c_str());
            os::remove_file(tmp);
            return;
        }
        std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> kept;
        std::filesystem::directory_iterator it(os::utf8_path(dir), ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            const std::string name = os::path_utf8(it->path().filename());
            if (name.starts_with("reuse-") && name.ends_with(".bin")) {
                std::error_code ignored;
                kept.emplace_back(std::filesystem::last_write_time(it->path(), ignored), it->path());
            }
        }
        if (kept.size() <= kKeptReuseCalibrations) return;
        std::sort(kept.begin(), kept.end());  // oldest first
        for (size_t i = 0; i + kKeptReuseCalibrations < kept.size(); ++i) std::filesystem::remove(kept[i].second, ec);
    }

    std::unique_ptr<TripleBufferLoader> weight_loader;
    std::unique_ptr<SpeculativeDecoder> speculative;
    std::optional<Tokenizer> tokenizer;
    // The model's chat template (Engine::render_chat) and the special-token
    // variables it renders with; empty when the model has none or it does
    // not parse (chat_template_error says why).
    std::optional<ChatTemplate> chat_template;
    Status chat_template_error;
    jinja::Value chat_special;
    // Constrained decoding (GenerateOptions): the vocabulary as a trie, built
    // on first use, and the state tables of the grammars used most recently
    // (most recent first), so a grammar used again keeps its transitions and
    // masks. At most 16 tables and 64 MiB together; the newest always stays.
    // Shared: a generation keeps the trie and its tables alive while it uses
    // them (trim_memory() drops the caches' references only), and every use
    // of either, which fills a table's caches as it goes, holds
    // grammar_mutex (sequences may use one grammar at once).
    std::shared_ptr<const grammar::VocabTrie> vocab_trie;
    std::list<std::shared_ptr<grammar::StateTable>> grammar_tables;
    std::mutex grammar_mutex;
    // grammar_mutex held.
    std::shared_ptr<const grammar::VocabTrie> trie() {
        if (!vocab_trie) {
            const auto t0 = Clock::now();
            vocab_trie = grammar::VocabTrie::build(*tokenizer);
            LIYAB_LOG_INFO("grammar: vocabulary trie of %zu nodes built in %.1f ms", vocab_trie->nodes.size(),
                           ms_since(t0));
        }
        return vocab_trie;
    }
    // grammar_mutex held.
    std::shared_ptr<grammar::StateTable> grammar_table(const Grammar& g) {
        for (auto it = grammar_tables.begin(); it != grammar_tables.end(); ++it) {
            const auto& p = (*it)->program();
            if (p == g.program() || p->same(*g.program())) {
                grammar_tables.splice(grammar_tables.begin(), grammar_tables, it);
                return grammar_tables.front();
            }
        }
        grammar_tables.push_front(std::make_shared<grammar::StateTable>(g.program()));
        size_t total = 0;
        for (const auto& t : grammar_tables) total += t->memory_bytes();
        constexpr size_t kMaxTables = 16, kMaxBytes = size_t{64} << 20;
        while (grammar_tables.size() > 1 && (grammar_tables.size() > kMaxTables || total > kMaxBytes)) {
            total -= grammar_tables.back()->memory_bytes();
            grammar_tables.pop_back();
        }
        return grammar_tables.front();
    }
    void load_chat_template() {
        const auto source = target->file().get_string("tokenizer.chat_template");
        if (!source) {
            chat_template_error = Status(ErrorCode::Unsupported, "the model has no chat template");
            return;
        }
        // The special-token variables transformers passes (bos_token...).
        chat_special = jinja::Value::object();
        auto special = [&](const char* name, std::optional<int32_t> id) {
            if (id && *id >= 0 && *id < tokenizer->vocab_size()) {
                chat_special.fields().set(name, jinja::Value(std::string(tokenizer->token_text(*id))));
            }
        };
        auto id_of = [&](std::string_view key) -> std::optional<int32_t> {
            const auto v = target->file().get_int(key);
            return v ? std::optional<int32_t>(static_cast<int32_t>(*v)) : std::nullopt;
        };
        special("bos_token", id_of("tokenizer.ggml.bos_token_id"));
        special("eos_token", id_of("tokenizer.ggml.eos_token_id"));
        special("unk_token", id_of("tokenizer.ggml.unknown_token_id"));
        special("pad_token", id_of("tokenizer.ggml.padding_token_id"));
        auto parsed = ChatTemplate::parse(*source, tokenizer->chat_sort_tools());
        if (!parsed) {
            chat_template_error = parsed.status();
            LIYAB_LOG_WARN("%s", parsed.status().message().c_str());
            return;
        }
        chat_template.emplace(std::move(parsed).value());
        // Whether the template lays out turns as a prefix of the next ones
        // (ChatFormat): front ends that keep a context between turns need it.
        auto format = chat_template->format("", chat_special);
        if (format && format->prefix_stable) {
            LIYAB_LOG_INFO("chat template: prefix-stable across turns");
        } else {
            LIYAB_LOG_INFO("chat template: not prefix-stable across turns (%s)",
                           format ? format->reason.c_str() : format.status().message().c_str());
        }
    }
    std::unique_ptr<PowerManager> power;
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
    std::unique_ptr<experimental::EarlyExit> early_exit;
    std::optional<experimental::HeadPruner> head_pruner;
    std::unique_ptr<experimental::Egls> egls;
    std::unique_ptr<experimental::Tdss> tdss;
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
    // Whether a decoding step runs hooks a batched pass does not (early exit,
    // EGLS: both experimental and lossy). Then a forced pass cannot also
    // produce the next token's logits without changing them, and the next
    // token keeps its own decoding step.
    [[nodiscard]] bool decode_only_hooks() const noexcept {
#if defined(LIYAB_ENABLE_EXPERIMENTAL)
        return early_exit != nullptr || egls != nullptr;
#else
        return false;
#endif
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

    // Key scales of a quantized KV cache (core/key_scaling.h): the ones kept
    // in config.cache_dir for this model, else measured on the CPU, so they
    // depend on the model and the kernel set only, whatever the backends,
    // and kept there.
    Status prepare_key_scales(Transformer& model) {
        if (!quant::is_block_quantized(model.kv_cache().dtype())) return Status::ok();
        const uint64_t id = key_scales_id(model);
        if (!config.cache_dir.empty()) {
            const std::string path = key_scales_path(config.cache_dir, id);
            const auto kept = read_key_scales(path, id);
            if (kept && model.set_key_scales(*kept).is_ok()) {
                std::error_code ignored;
                std::filesystem::last_write_time(os::utf8_path(path), std::filesystem::file_time_type::clock::now(),
                                                 ignored);
                LIYAB_LOG_INFO("keys: %s (%s)", kept->empty() ? "no scaling needed" : "per-channel scales",
                               path.c_str());
                return Status::ok();
            }
        }
        auto tokens = tokenizer->encode(key_calibration_text(), tokenizer->add_bos_default());
        if (!tokens) {
            // The tokenizer cannot encode text (an unknown pre-tokenizer): the
            // model still runs on token ids, and the fixed text is only what
            // the scales are measured on. Load it with its keys stored as
            // computed, as every model without large key channels is, rather
            // than refuse it; models with such channels lose accuracy in a
            // quantized cache, which the warning says. Nothing is kept in
            // cache_dir, so a later build that encodes the text measures them.
            key_scaling_warning = "unscaled: the key calibration text cannot be encoded (" +
                                  tokens.status().message() +
                                  "); a quantized KV cache may lose accuracy on models with large key channels, "
                                  "an F16 KV cache does not";
            LIYAB_LOG_WARN("keys: %s", key_scaling_warning.c_str());
            return Status::ok();
        }
        const auto start = Clock::now();
        LIYAB_RETURN_IF_ERROR(model.calibrate_keys(*tokens, draft_route(), *pool));
        LIYAB_LOG_INFO("keys: calibrated in %.0f ms", ms_since(start));
        if (!config.cache_dir.empty()) keep_key_scales(config.cache_dir, id, model.key_scales());
        return Status::ok();
    }

    // Latent attention with LatentForm::Auto: the form ratio kept in
    // config.cache_dir for this model (or measured earlier in this process
    // without one), else measured once on the CPU now and kept. From then on
    // each pass's form follows from its shape alone (LatentFormChooser).
    Status prepare_latent_form(Transformer& model) {
        if (model.config().mla.kv_rank == 0 || model.latent_form() != LatentForm::Auto) return Status::ok();
        const uint64_t id = latent_ratio_id(model);
        if (!config.cache_dir.empty()) {
            const std::string path = latent_ratio_path(config.cache_dir, id);
            if (const auto kept = read_latent_ratio(path, id); kept && model.set_latent_ratio(*kept).is_ok()) {
                std::error_code ignored;
                std::filesystem::last_write_time(os::utf8_path(path), std::filesystem::file_time_type::clock::now(),
                                                 ignored);
                LIYAB_LOG_INFO("latent attention: form ratio %.3f (%s)", static_cast<double>(*kept), path.c_str());
                return Status::ok();
            }
        } else {
            std::lock_guard<std::mutex> lock(latent_ratios_mutex);
            if (const auto it = latent_ratios().find(id); it != latent_ratios().end()) {
                return model.set_latent_ratio(it->second);
            }
        }
        const auto start = Clock::now();
        auto ratio = model.calibrate_latent_form(draft_route(), *pool);
        if (!ratio) return ratio.status();
        LIYAB_LOG_INFO("latent attention: form ratio measured in %.0f ms", ms_since(start));
        if (!config.cache_dir.empty()) {
            keep_latent_ratio(config.cache_dir, id, *ratio);
        } else {
            std::lock_guard<std::mutex> lock(latent_ratios_mutex);
            latent_ratios().emplace(id, *ratio);
        }
        return Status::ok();
    }

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
        if (largest == 0) return Status::ok(); // a stack of identity blocks has no block weights to stream
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
            const size_t average = std::max<size_t>(block_total / static_cast<size_t>(c.n_layers), 1);
            const auto resident = static_cast<int32_t>(std::min<size_t>(budget / average, static_cast<size_t>(c.n_layers)));
            streamed = c.n_layers - resident;
            if (streamed == 0) return Status::ok();
        }
        std::vector<int32_t> items(static_cast<size_t>(c.n_layers), -1);
        std::vector<TripleBufferLoader::Item> ranges;
        for (int32_t l = 0; l < c.n_layers; ++l) {
            // Evenly spaced: block l streams when floor((l+1)s/n) > floor(ls/n).
            if ((int64_t{l} + 1) * streamed / c.n_layers > int64_t{l} * streamed / c.n_layers) {
                const auto [begin, end] = file.layer_range(l);
                if (end == begin) continue; // absent NAS blocks need no I/O item
                items[static_cast<size_t>(l)] = static_cast<int32_t>(ranges.size());
                ranges.push_back({begin, end - begin});
            }
        }
        if (ranges.empty()) return Status::ok();
        streamed = static_cast<int32_t>(ranges.size()); // empty NAS blocks do not count as streamed weights
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
    // Latent attention (DeepSeek-V2/V3 structures) prefills in passes of at
    // least kLatentPrefillChunk tokens: its expanded form expands every
    // position a pass sees once per pass, which a pass of 64 tokens deep in a
    // context does not repay (the chooser keeps such passes absorbed). On
    // Youtu-LLM-2B (Q8_0, Mac CPU) an 8,192-token prompt took 142 s in passes
    // of 64 (absorbed) and 68 s in passes of 512 (expanded); 1,024 gained
    // nothing more. The form of each pass is still chosen by measured cost.
    static constexpr int32_t kLatentPrefillChunk = 512;
    [[nodiscard]] int32_t prefill_chunk() const {
        if (target->expert_store() != nullptr) return std::max(target->max_batch(), kStreamedPrefillChunk);
        if (target->config().mla.kv_rank > 0) return std::max(target->max_batch(), kLatentPrefillChunk);
        return target->max_batch();
    }

    void apply_policy(const PowerPolicy& policy) {
        const auto threads = static_cast<int32_t>(std::lround(pool->max_threads() * policy.thread_fraction));
        pool->set_active_threads(std::max(1, threads));
    }

    // Pooling models (Engine::embed / rerank; core/pooling.h): what the GGUF
    // declares, or why it could not be read (reported by the calls).
    PoolingModel pooling;
    Status pooling_error;
    // Why a quantized KV cache stores keys without the scales the model may
    // need (prepare_key_scales); empty when they were measured or kept.
    std::string key_scaling_warning;

    // One pooling input: BOS when the model uses one, the text, EOS when the
    // GGUF asks for it. Longer than the context: cut to fit (`truncate`,
    // keeping BOS and EOS) or InvalidArgument naming input `index` of `what`.
    Result<std::vector<int32_t>> encode_pooled(std::string_view text, bool truncate, size_t index,
                                               const char* what) const {
        auto tokens = tokenizer->encode(text, tokenizer->add_bos_default());
        if (!tokens) return tokens.status();
        std::vector<int32_t>& t = tokens.value();
        const auto limit = static_cast<size_t>(target->context_length());
        const size_t eos = pooling.add_eos ? 1 : 0;
        if (t.size() + eos > limit) {
            if (!truncate) {
                return Status(ErrorCode::InvalidArgument, std::string(what) + " " + std::to_string(index) + " has " +
                                                              std::to_string(t.size() + eos) +
                                                              " tokens; the context holds " + std::to_string(limit));
            }
            t.resize(limit - eos);  // a leading BOS stays: the cut is at the end of the text
        }
        if (pooling.add_eos) t.push_back(pooling.eos);
        if (t.empty()) {
            return Status(ErrorCode::InvalidArgument, std::string(what) + " " + std::to_string(index) + " is empty");
        }
        return tokens;
    }

    // Runs pooled sequences (run_pooled) the way prefill() runs a prompt
    // (power policy, route, chunks; cancel() stops it), with the context to
    // itself: a conversation in progress is set aside first, as a prompt that
    // replaces it would set it aside (prefix cache), and the context is left
    // empty. Lossy experimental hooks are not applied: vectors and scores are
    // the model's own.
    // On the default sequence, holding the model for the whole call.
    Result<PooledRun> run_pooling(std::span<const std::vector<int32_t>> sequences, const PoolTarget& pool_target) {
        Seq& q = *main;
        q.cancel.store(false, std::memory_order_relaxed);
        const Hold hold(*this, q);
        if (!draft && prefix_cache_on() && q.context.size() == static_cast<size_t>(target->n_past())) stash_context(q);
        clear_context(q);
        const PowerManager::Work thermal(*power);
        const PowerPolicy policy = power->policy();
        apply_policy(policy);
        const Route route = policy.throttled ? throttled_route() : normal_route();
        ForwardHooks forward_hooks;
        forward_hooks.cancel = &q.cancel;
        auto run = run_pooled(*target, sequences, pool_target, route, *this->pool, prefill_chunk(), &forward_hooks);
        clear_context(q);
        if (!run && run.status().code() == ErrorCode::Cancelled) {
            return Status(ErrorCode::Cancelled, "the embedding or reranking call was cancelled");
        }
        return run;
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
        config.experimental.tdss) {
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
    {
        // The default sequence: its context is the transformer's live one.
        auto q = std::make_unique<Impl::Seq>();
        q->id = 0;
        q->parked = std::make_unique<Transformer::Context>();
        impl->bound = q.get();
        impl->main = q.get();
        impl->seqs.push_back(std::move(q));
    }

    LIYAB_RETURN_IF_ERROR(impl->attach_layer_streaming(config.triple_buffer_loading));
    LIYAB_RETURN_IF_ERROR(impl->share_resident_weights());

    auto tokenizer = Tokenizer::load(impl->target->file());
    if (!tokenizer) return tokenizer.status();
    impl->tokenizer.emplace(std::move(tokenizer).value());
    impl->load_chat_template();
    if (auto pooling = read_pooling_model(impl->target->file(), impl->target->config())) {
        impl->pooling = std::move(pooling).value();
        const PoolingModel& p = impl->pooling;
        if (p.declared != Pooling::None || p.head != nullptr) {
            static constexpr const char* kNames[] = {"none", "mean", "cls", "last token", "rank"};
            LIYAB_LOG_INFO("pooling: %s%s%s%s", kNames[static_cast<int>(p.declared)],
                           p.add_eos ? ", inputs end with EOS" : "",
                           p.head != nullptr ? ", ranking head" : "",
                           p.rerank_template.empty() ? "" : ", rerank prompt template");
        }
        if (!p.head_error.is_ok()) LIYAB_LOG_WARN("pooling: %s", p.head_error.message().c_str());
    } else {
        // Not fatal: the model may still generate; embed() and rerank() report it.
        impl->pooling_error = pooling.status();
        LIYAB_LOG_WARN("pooling: %s", pooling.status().message().c_str());
    }
    LIYAB_RETURN_IF_ERROR(impl->prepare_key_scales(*impl->target));
    LIYAB_RETURN_IF_ERROR(impl->prepare_latent_form(*impl->target));

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
        LIYAB_RETURN_IF_ERROR(impl->prepare_key_scales(*impl->draft));  // same vocabulary: the target's tokenizer
        LIYAB_RETURN_IF_ERROR(impl->prepare_latent_form(*impl->draft));
        impl->speculative = std::make_unique<SpeculativeDecoder>(*impl->target, *impl->draft, config.draft_tokens);
    } else if (config.lookup_drafts && config.draft_tokens > 0) {
        impl->target->set_rollback_window(config.draft_tokens + 1);
        impl->speculative = std::make_unique<SpeculativeDecoder>(
            *impl->target, config.draft_tokens,
            config.suffix_drafts ? SpeculativeDecoder::TextDrafts::Suffix : SpeculativeDecoder::TextDrafts::Lookup);
    }
    if (impl->speculative) {
        impl->speculative->set_adaptive(config.adaptive_drafts);
        impl->speculative->set_tree(config.draft_tree);
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
    impl->open_store(config, repack_cpu);
    impl->prepare_segment_reuse();
    impl->power->poll_once();
    if (config.thermal_polling) impl->power->start();
    const Route route = impl->normal_route();
    LIYAB_LOG_INFO("engine ready in %.0f ms: attention on %s, FFN on %s, %d threads%s", ms_since(t_create),
                   route.attention->description().c_str(), route.ffn->description().c_str(),
                   impl->pool->max_threads(), impl->speculative ? ", speculative decoding" : "");
    impl->start_calibration();

    return std::unique_ptr<Engine>(new Engine(std::move(impl)));
}

void Engine::cancel() noexcept { impl_->main->cancel.store(true, std::memory_order_relaxed); }

void Engine::cancel_prefill() noexcept { impl_->cancel_prefill(*impl_->main); }

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
        std::ofstream out(os::utf8_path(tmp), std::ios::binary | std::ios::trunc);
        out.write(kExpertProfileMagic, sizeof kExpertProfileMagic);
        out.write(reinterpret_cast<const char*>(header), sizeof header);
        out.write(reinterpret_cast<const char*>(keys.data()), static_cast<std::streamsize>(keys.size() * sizeof(int32_t)));
        if (!out) return Status(ErrorCode::IoError, "cannot write " + tmp);
    }
    if (!os::replace_file(tmp, path)) return Status(ErrorCode::IoError, "cannot rename " + tmp);
    return Status::ok();
}

Status Engine::load_expert_profile(const std::string& path) {
    ExpertStore* store = impl_->target->expert_store();
    if (store == nullptr) return Status::ok();
    std::ifstream in(os::utf8_path(path), std::ios::binary);
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
    // The cache as it is now: the plan's, minus what open sequences borrowed.
    const ExpertStore* store = impl_->target->expert_store();
    return {p.resident_bytes, p.expert_bytes, store != nullptr ? store->capacity_bytes() : p.expert_cache_bytes,
            p.recommended_bytes, p.requant_bits};
}

size_t Engine::trim_memory() {
    const std::lock_guard<std::mutex> model(impl_->model);  // not during another sequence's pass
    size_t released = 0;
    if (ExpertStore* store = impl_->target->expert_store()) released += store->trim();
    if (impl_->draft) {
        if (ExpertStore* store = impl_->draft->expert_store()) released += store->trim();
    }
    if (released > 0) LIYAB_LOG_INFO("trimmed %.2f GiB of cached experts", released / (1024.0 * 1024.0 * 1024.0));
    // Constrained decoding's caches are rebuilt on the next constrained
    // generation (the trie in tens of ms): not worth keeping while idle.
    {
        std::lock_guard<std::mutex> lock(impl_->grammar_mutex);
        impl_->grammar_tables.clear();
        impl_->vocab_trie.reset();
    }
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
    const std::lock_guard<std::mutex> model(impl_->model);  // the live context changes while sequences run
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
                  static_cast<double>(KvCache::bytes_per_token(kv.config())) /
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
    if (!impl_->key_scaling_warning.empty()) out += "Keys:       " + impl_->key_scaling_warning + "\n";
    if (c.hybrid()) {
        // Blocks by token mixer, in the order the kinds first appear (FFN-only blocks have none).
        auto name = [](MixerKind m) -> std::string {
            switch (m) {
                case MixerKind::Attention: return "attention";
                case MixerKind::DeltaNet: return "DeltaNet";
                case MixerKind::Mamba: return "Mamba-1";
                case MixerKind::Mamba2: return "Mamba-2";
                case MixerKind::ShortConv: return "short convolution";
                case MixerKind::Kda: return "Kimi Delta Attention";
                case MixerKind::Linear: return "linear";
                case MixerKind::None: return "FFN-only";
                default: return "other";
            }
        };
        std::vector<std::pair<std::string, int32_t>> kinds;
        for (const BlockPlan& b : c.blocks) {
            const std::string k = name(b.mixer);
            auto it = std::find_if(kinds.begin(), kinds.end(), [&](const auto& e) { return e.first == k; });
            if (it == kinds.end()) {
                kinds.emplace_back(k, 1);
            } else {
                ++it->second;
            }
        }
        std::string blocks;
        for (const auto& [k, count] : kinds) blocks += (blocks.empty() ? "" : " + ") + std::to_string(count) + " " + k;
        char mix[200];
        std::snprintf(mix, sizeof mix, "Mixers:     %s blocks, recurrent state %.1f MiB\n", blocks.c_str(),
                      static_cast<double>(impl_->target->recurrent_state_bytes()) / (1024.0 * 1024.0));
        out += mix;
    }
    if (kv.has_windowed_layers()) {  // per-block KV: windowed blocks keep their window only
        int32_t every = 0;
        for (const int32_t w : kv.config().layer_windows) every += w == 0 ? 1 : 0;
        const int32_t ctx = impl_->target->context_length();
        char blocks[256];
        std::snprintf(blocks, sizeof blocks,
                      "KV blocks:  %d of %d keep every position (%.1f KiB/token), the others a window of %d; "
                      "a full context %.1f MiB\n",
                      every, kv.config().n_layers, static_cast<double>(kv.rows_bytes(1)) / 1024.0, c.sliding_window,
                      static_cast<double>(KvCache::kept_bytes(kv.config(), ctx, impl_->target->max_batch())) /
                          (1024.0 * 1024.0));
        out += blocks;
    }
    {
        std::lock_guard<std::mutex> lock(impl_->adapters_mutex);
        for (const auto& e : impl_->adapters) {
            char line[256];
            std::snprintf(line, sizeof line, "Adapter:    %s, LoRA rank %d, %d matrices, %.1f MiB%s%s\n", e->name.c_str(),
                          e->rank, e->matrices, static_cast<double>(e->bytes) / (1024.0 * 1024.0),
                          e->adapter ? "" : " (released)", e.get() == impl_->main->active ? ", selected" : "");
            out += line;
        }
    }
    if (impl_->draft) {
        out += "Draft:      " + impl_->draft->config().arch + ", " + std::to_string(impl_->draft->config().n_layers) +
               " layers, k=" + std::to_string(impl_->config.draft_tokens) + " (cpu)\n";
    }
    if (const PoolingModel& p = impl_->pooling; p.declared != Pooling::None) {  // an embedding model or a reranker
        static constexpr const char* kKinds[] = {"", "embeddings, mean of the tokens", "first token (CLS)",
                                                 "embeddings, last token", "reranking"};
        out += "Pooling:    " + std::string(kKinds[static_cast<int>(p.declared)]);
        out += p.declared == Pooling::Rank
                   ? (p.head != nullptr ? ", ranking head of " + std::to_string(p.head->rows()) + " labels"
                                        : ", no usable ranking head")
                   : ", " + std::to_string(c.n_embd) + " values per vector";
        out += p.add_eos ? ", inputs end with EOS\n" : "\n";
    }
    return out;
}

Engine::PrefixCacheStats Engine::prefix_cache_stats() const {
    std::lock_guard<std::mutex> lock(impl_->prefix_stats_mutex);
    return impl_->prefix_stats;
}

Engine::SegmentReuseInfo Engine::segment_reuse() const {
    std::lock_guard<std::mutex> lock(impl_->reuse_info_mutex);
    return impl_->reuse_info;
}

Status Engine::load_adapter(std::string_view name, const std::string& path) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    if (s.config.adapter_memory_mb <= 0) {
        return Status(ErrorCode::Unsupported, "adapters are off (EngineConfig::adapter_memory_mb is 0)");
    }
    if (name.empty()) return Status(ErrorCode::InvalidArgument, "an adapter needs a name");
    const Impl::Hold hold(s, *s.main);
    if (s.find_adapter(name) != nullptr) {
        return Status(ErrorCode::InvalidArgument, "an adapter named " + std::string(name) + " is loaded already");
    }
    auto loaded = LoraAdapter::load(path);
    if (!loaded) return loaded.status();
    LIYAB_RETURN_IF_ERROR(s.check_adapter(**loaded));
    const size_t bytes = (*loaded)->bytes();
    const size_t cap = static_cast<size_t>(s.config.adapter_memory_mb) << 20;
    if (bytes > cap) {
        return Status(ErrorCode::OutOfMemory, "the adapter's " + std::to_string(bytes >> 20) +
                                                  " MiB exceed EngineConfig::adapter_memory_mb");
    }
    s.fit_adapters(bytes, nullptr);
    auto entry = std::make_unique<Impl::AdapterEntry>();
    entry->name = std::string(name);
    entry->path = path;
    entry->bytes = bytes;
    entry->rank = (*loaded)->max_rank();
    entry->matrices = static_cast<int32_t>((*loaded)->targets().size());
    entry->alpha = (*loaded)->alpha();
    entry->last_used = ++s.adapter_clock;
    entry->adapter = std::move(loaded).value();
    LIYAB_LOG_INFO("adapters: %s loaded from %s: LoRA rank %d, alpha %g, %d matrices, %.1f MiB", entry->name.c_str(),
                   path.c_str(), entry->rank, static_cast<double>(entry->alpha), entry->matrices,
                   static_cast<double>(bytes) / (1024.0 * 1024.0));
    std::lock_guard<std::mutex> lock(s.adapters_mutex);
    s.adapters.push_back(std::move(entry));
    return Status::ok();
}

Status Engine::unload_adapter(std::string_view name) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    Impl::AdapterEntry* e = nullptr;
    {
        const std::lock_guard<std::mutex> model(s.model);
        e = s.find_adapter(name);
    }
    if (e == nullptr) return Status(ErrorCode::InvalidArgument, "no adapter named " + std::string(name));
    // Every sequence that selected it goes back to the base model; one in a
    // call keeps using it, so the adapter stays (Busy).
    std::vector<Impl::Seq*> users;
    std::vector<std::unique_lock<std::mutex>> calls;
    {
        std::lock_guard<std::mutex> g(s.sched);
        for (auto& q : s.seqs) {
            if (q->active != e) continue;
            if (q.get() != s.main) {
                std::unique_lock<std::mutex> other(q->call, std::try_to_lock);
                if (!other.owns_lock()) {
                    return Status(ErrorCode::Busy, "a generation on another sequence applies this adapter");
                }
                calls.push_back(std::move(other));
            }
            users.push_back(q.get());
        }
    }
    for (Impl::Seq* q : users) {
        const Impl::Hold hold(s, *q);
        LIYAB_RETURN_IF_ERROR(s.select_adapter(*q, nullptr, 1.0f));
    }
    const Impl::Hold hold(s, *s.main);
    std::lock_guard<std::mutex> lock(s.adapters_mutex);
    std::erase_if(s.adapters, [e](const std::unique_ptr<Impl::AdapterEntry>& x) { return x.get() == e; });
    return Status::ok();
}

Status Engine::set_adapter(std::string_view name, float scale) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return s.set_sequence_adapter(*s.main, name, scale);
}

std::vector<AdapterInfo> Engine::adapters() const {
    std::lock_guard<std::mutex> lock(impl_->adapters_mutex);
    std::vector<AdapterInfo> out;
    const Impl::Seq& q = *impl_->main;
    for (const auto& e : impl_->adapters) {
        AdapterInfo info;
        info.name = e->name;
        info.path = e->path;
        info.alpha = e->alpha;
        info.rank = e->rank;
        info.matrices = e->matrices;
        info.bytes = e->bytes;
        info.resident = e->adapter != nullptr;
        info.active = e.get() == q.active;
        info.scale = info.active ? q.active_scale : 0.0f;
        out.push_back(std::move(info));
    }
    return out;
}

Result<std::unique_ptr<Engine::Sequence>> Engine::open_sequence(SequencePriority priority) {
    Impl& s = *impl_;
    if (s.draft) {
        return Status(ErrorCode::Unsupported, "sequences are not available with a draft model (its cache follows one)");
    }
    const auto most = static_cast<size_t>(std::max(s.config.max_sequences, 1));
    if (most == 1) return Status(ErrorCode::Unsupported, "EngineConfig::max_sequences is 1: the engine keeps one sequence");
    const std::lock_guard<std::mutex> model(s.model);
    {
        const std::lock_guard<std::mutex> g(s.sched);
        if (s.seqs.size() >= most) return Status(ErrorCode::Busy, "every sequence is open (EngineConfig::max_sequences)");
    }
    // The memory its context will hold beyond KV pages (which it maps as it
    // uses them) comes out of the expert cache now, before it is allocated,
    // and goes back when it closes: an engine with no sequence open keeps the
    // plan of a single conversation, whatever max_sequences allows. Under the
    // model lock no pass runs, so the store waits only for reads in flight.
    // A cache that cannot spare it refuses (OutOfMemory) rather than
    // overcommit; nothing is ever requantized here (that is a load-time
    // decision).
    ExpertStore::Loan loan;
    ExpertStore* store = s.target->expert_store();
    if (const size_t bytes = s.target->context_state_bytes(); store != nullptr && bytes > 0) {
        auto lent = store->lend(bytes);
        if (!lent) return lent.status();
        loan = std::move(lent).value();
        LIYAB_LOG_INFO("sequences: %.1f MiB of the expert cache for a sequence's states (%.2f GiB left)",
                       static_cast<double>(loan.bytes) / (1024.0 * 1024.0),
                       static_cast<double>(store->capacity_bytes()) / (1024.0 * 1024.0 * 1024.0));
    }
    auto context = s.target->make_context();
    if (!context) {
        if (store != nullptr) store->repay(loan);
        return context.status();
    }
    auto q = std::make_unique<Impl::Seq>();
    q->priority.store(static_cast<int32_t>(priority), std::memory_order_relaxed);
    q->loan = std::move(loan);
    q->parked = std::move(context).value();
    const std::lock_guard<std::mutex> g(s.sched);
    q->id = s.next_sequence_id++;
    const int32_t id = q->id;
    s.seqs.push_back(std::move(q));
    LIYAB_LOG_INFO("sequences: %d opened (%zu of %zu), %s", id, s.seqs.size(), most,
                   priority == SequencePriority::Foreground ? "foreground" : "background");
    return std::unique_ptr<Sequence>(new Sequence(*this, id));
}

Engine::SchedulerStats Engine::scheduler_stats() const {
    const std::lock_guard<std::mutex> g(impl_->sched);
    SchedulerStats out = impl_->sched_stats;
    out.open_sequences = static_cast<int32_t>(impl_->seqs.size());
    out.active_sequences = impl_->in_call;
    return out;
}

Engine::Sequence::~Sequence() { engine_.impl_->close_sequence(id_); }

Result<GenerationStats> Engine::Sequence::generate(std::string_view prompt, const SamplingParams& params,
                                                   const TokenCallback& on_token, const GenerateOptions& options) {
    if (options.grammar && options.continuation) {
        return Status(ErrorCode::InvalidArgument, "GenerateOptions: set a grammar or a continuation callback, not both");
    }
    Impl& s = *engine_.impl_;
    Impl::Seq* q = s.find_sequence(id_);
    if (q == nullptr) return Status(ErrorCode::InvalidArgument, "the sequence is closed");
    const Impl::Call call(s, *q);
    if (!call) return Status(ErrorCode::Busy, "a call is already running on this sequence");
    return s.generate_text(*q, prompt, params, on_token, nullptr, &options);
}

Result<GenerationStats> Engine::Sequence::generate(std::string_view prompt, const SamplingParams& params,
                                                   const TokenCallback& on_token, const ForceCallback& force) {
    Impl& s = *engine_.impl_;
    Impl::Seq* q = s.find_sequence(id_);
    if (q == nullptr) return Status(ErrorCode::InvalidArgument, "the sequence is closed");
    const Impl::Call call(s, *q);
    if (!call) return Status(ErrorCode::Busy, "a call is already running on this sequence");
    return s.generate_text(*q, prompt, params, on_token, force ? &force : nullptr, nullptr);
}

Result<GenerationStats> Engine::Sequence::generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                                          const TokenCallback& on_token) {
    Impl& s = *engine_.impl_;
    Impl::Seq* q = s.find_sequence(id_);
    if (q == nullptr) return Status(ErrorCode::InvalidArgument, "the sequence is closed");
    const Impl::Call call(s, *q);
    if (!call) return Status(ErrorCode::Busy, "a call is already running on this sequence");
    return s.generate(*q, prompt, params, on_token, nullptr);
}

Status Engine::Sequence::prefill(std::string_view text, bool add_bos) {
    Impl& s = *engine_.impl_;
    Impl::Seq* q = s.find_sequence(id_);
    if (q == nullptr) return Status(ErrorCode::InvalidArgument, "the sequence is closed");
    const Impl::Call call(s, *q);
    if (!call) return Status(ErrorCode::Busy, "a call is already running on this sequence");
    return s.prefill_on(*q, text, {}, add_bos);
}

void Engine::Sequence::reset_context() {
    Impl& s = *engine_.impl_;
    Impl::Seq* q = s.find_sequence(id_);
    if (q == nullptr) return;
    const Impl::Call call(s, *q, /*wait=*/true);
    const Impl::Hold hold(s, *q);
    s.clear_context(*q);
}

Status Engine::Sequence::set_adapter(std::string_view name, float scale) {
    Impl& s = *engine_.impl_;
    Impl::Seq* q = s.find_sequence(id_);
    if (q == nullptr) return Status(ErrorCode::InvalidArgument, "the sequence is closed");
    const Impl::Call call(s, *q);
    if (!call) return Status(ErrorCode::Busy, "a call is already running on this sequence");
    return s.set_sequence_adapter(*q, name, scale);
}

void Engine::Sequence::set_priority(SequencePriority priority) noexcept {
    if (Impl::Seq* q = engine_.impl_->find_sequence(id_)) {
        q->priority.store(static_cast<int32_t>(priority), std::memory_order_relaxed);
    }
}

SequencePriority Engine::Sequence::priority() const noexcept {
    const Impl::Seq* q = engine_.impl_->find_sequence(id_);
    return q != nullptr ? static_cast<SequencePriority>(q->priority.load(std::memory_order_relaxed))
                        : SequencePriority::Foreground;
}

void Engine::Sequence::cancel() noexcept {
    if (Impl::Seq* q = engine_.impl_->find_sequence(id_)) q->cancel.store(true, std::memory_order_relaxed);
}

void Engine::Sequence::cancel_prefill() noexcept {
    if (Impl::Seq* q = engine_.impl_->find_sequence(id_)) engine_.impl_->cancel_prefill(*q);
}

Status Engine::prefill(std::string_view text, bool add_bos) { return prefill(text, {}, add_bos); }

Status Engine::prefill(std::string_view text, std::span<const PromptSegment> segments, bool add_bos) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return s.prefill_on(*s.main, text, segments, add_bos);
}

namespace {
constexpr uint32_t kContextMagic = 0x5458434C;  // "LCXT": the engine's part after the model's state
}

namespace {

// After the model's state: magic, token count, tokens, then the text they
// encode (its length, UINT64_MAX when not known) and how many of the tokens it
// covers. Files written before that count was saved end after the text, which
// then covers every token. A context holding rows reused away from their
// prefix (segment reuse) then has kReusedMagic and the first such position
// (`exact`), so a later request that does not reuse segments never continues
// them; other contexts end as before, so their files are unchanged.
constexpr uint32_t kReusedMagic = 0x5355524C;  // "LRUS"
// A context computed with a LoRA adapter then has kAdapterMagic and its key
// (Impl::adapter_key: the adapter file's hash and the scale), last, so
// load_state() never continues it under another adapter or without one.
constexpr uint32_t kAdapterMagic = 0x5044414C;  // "LADP"
void write_context_tail(std::ostream& out, const std::vector<int32_t>& tokens, const std::string& text,
                        size_t text_tokens, size_t exact = SIZE_MAX, uint64_t adapter = 0) {
    const auto n = static_cast<uint64_t>(tokens.size());
    const bool known = text_tokens > 0;
    const uint64_t length = known ? text.size() : UINT64_MAX;
    const auto covered = static_cast<uint64_t>(text_tokens);
    out.write(reinterpret_cast<const char*>(&kContextMagic), sizeof kContextMagic);
    out.write(reinterpret_cast<const char*>(&n), sizeof n);
    out.write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(n * sizeof(int32_t)));
    out.write(reinterpret_cast<const char*>(&length), sizeof length);
    if (known) {
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
        out.write(reinterpret_cast<const char*>(&covered), sizeof covered);
    }
    if (exact < tokens.size()) {
        const auto first = static_cast<uint64_t>(exact);
        out.write(reinterpret_cast<const char*>(&kReusedMagic), sizeof kReusedMagic);
        out.write(reinterpret_cast<const char*>(&first), sizeof first);
    }
    if (adapter != 0) {
        out.write(reinterpret_cast<const char*>(&kAdapterMagic), sizeof kAdapterMagic);
        out.write(reinterpret_cast<const char*>(&adapter), sizeof adapter);
    }
}

// Writes `write`'s output to `path` through a temporary file, so `path` is
// either the old file or the new one.
Status write_replacing(const std::string& path, const std::function<Status(std::ostream&)>& write) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(os::utf8_path(tmp), std::ios::binary | std::ios::trunc);
        if (!out) return Status(ErrorCode::IoError, "cannot write " + tmp);
        if (Status s = write(out); !s.is_ok()) {
            out.close();
            os::remove_file(tmp);
            return s;
        }
        if (!out.flush()) return Status(ErrorCode::IoError, "cannot write " + tmp);
    }
    if (!os::replace_file(tmp, path)) {
        os::remove_file(tmp);
        return Status(ErrorCode::IoError, "cannot replace " + path);
    }
    return Status::ok();
}

}  // namespace

Status Engine::save_state(const std::string& path) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    if (s.draft) return Status(ErrorCode::Unsupported, "save_state() is not available with a draft model");
    Impl::Seq& q = *s.main;
    const Impl::Hold hold(s, q);
    return write_replacing(path, [&s, &q](std::ostream& out) {
        LIYAB_RETURN_IF_ERROR(s.target->write_state(out));
        write_context_tail(out, q.context, q.context_text, q.context_text_tokens, q.exact_end, q.context_key);
        return Status::ok();
    });
}

Status Engine::save_state(const std::string& path, std::string_view text) {
    {
        Impl& s = *impl_;
        const Impl::Call call(s, *s.main);
        if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
        if (s.draft) return Status(ErrorCode::Unsupported, "save_state() is not available with a draft model");
        const bool live = s.main->context_text_tokens > 0 && text.starts_with(s.main->context_text);
        if (!live) {
            const Impl::Hold hold(s, *s.main);
            // The longest context set aside for this text, written as it is
            // kept: nothing is restored or processed.
            const KvStore::Record* best = s.store ? s.store->record_for_text(text) : nullptr;
            if (best == nullptr) return Status(ErrorCode::InvalidArgument, "no context continues this text");
            // A copy: reading a damaged chunk removes it, and its record with it.
            const KvStore::Record record = *best;
            return write_replacing(path, [&s, &record](std::ostream& out) {
                // Its KV pages from the store's chunks, its recurrent states
                // and snapshots from the store's images, then its tokens and text.
                const std::vector<KvKey> keys = s.store->chunk_path(record.tokens);
                if (keys.empty()) return Status(ErrorCode::IoError, "the context set aside is no longer complete");
                const KvCache& kv = s.target->kv_cache();
                const int32_t c = s.store->chunk_tokens();
                const int32_t page_tokens = kv.config().page_tokens;
                const auto n = static_cast<int32_t>(record.tokens.size());
                // Pages in order: each chunk's image read once, then cut into
                // its pages' images (all whole but a context's last).
                std::vector<uint8_t> image, page(kv.page_bytes());
                int32_t image_chunk = -1;
                Status read = Status::ok();
                Transformer::ContextImage context;
                context.n = n;
                context.page = [&](int32_t p) -> const uint8_t* {
                    const int32_t chunk = p * page_tokens / c;
                    if (chunk != image_chunk) {
                        image.resize(kv.rows_bytes(std::min(c, n - chunk * c)));
                        const std::span<uint8_t> into(image);
                        read = s.store->read_chunk(keys[static_cast<size_t>(chunk)], {&into, 1});
                        if (!read.is_ok()) return nullptr;
                        image_chunk = chunk;
                    }
                    const int32_t first = p * page_tokens - chunk * c;  // positions of the chunk before this page
                    const int32_t rows = std::min(page_tokens, n - p * page_tokens);
                    kv.image_to_page(image.data() + kv.rows_bytes(first), rows, page.data());
                    return page.data();
                };
                // Windowed blocks: their window before the end from the
                // record's window pages (the one at each page's end, the
                // last one at n), each read once.
                Transformer::StateBuffer window_image;
                int32_t window_at = -1;
                if (kv.has_windowed_layers()) {
                    context.window_page = [&](int32_t p) -> const uint8_t* {
                        const int32_t pos = std::min((p + 1) * page_tokens, n);
                        if (pos == window_at) return reinterpret_cast<const uint8_t*>(window_image.data());
                        const auto rs = std::find_if(record.states.begin(), record.states.end(),
                                                     [pos](const KvStore::RecordState& r) { return r.pos == pos; });
                        if (rs == record.states.end()) return nullptr;
                        window_image.resize(kv.window_page_bytes() / sizeof(float));
                        const std::span<float> into(window_image);
                        read = s.store->read_state(rs->key, {&into, 1});
                        if (!read.is_ok()) return nullptr;
                        window_at = pos;
                        return reinterpret_cast<const uint8_t*>(window_image.data());
                    };
                }
                const size_t floats = s.target->recurrent_state_bytes() / sizeof(float);
                Transformer::StateBuffer state;  // images are not zero-filled before they are read
                std::vector<Transformer::StateBuffer> snapshots;
                if (floats > 0) {
                    const auto end = std::find_if(record.states.begin(), record.states.end(),
                                                  [n](const KvStore::RecordState& rs) { return rs.pos == n; });
                    if (end == record.states.end()) {
                        return Status(ErrorCode::Unsupported, "the recurrent states of the context set aside were not kept");
                    }
                    state.resize(floats);
                    const std::span<float> into(state);
                    LIYAB_RETURN_IF_ERROR(s.store->read_state(end->key, {&into, 1}));
                    snapshots.reserve(record.states.size());
                    for (const KvStore::RecordState& rs : record.states) {
                        if (rs.pos >= n) continue;
                        snapshots.emplace_back(floats);
                        const std::span<float> snap(snapshots.back());
                        LIYAB_RETURN_IF_ERROR(s.store->read_state(rs.key, {&snap, 1}));
                        context.snapshots.push_back({rs.pos, rs.pinned, snapshots.back()});
                    }
                    context.state = state;
                }
                const Status written = s.target->write_state(out, context);
                LIYAB_RETURN_IF_ERROR(read);
                LIYAB_RETURN_IF_ERROR(written);
                write_context_tail(out, record.tokens, record.text, record.text_tokens, SIZE_MAX, s.main->context_key);
                return out ? Status::ok() : Status(ErrorCode::IoError, "cannot write the context state");
            });
        }
    }
    return save_state(path);
}

Result<int32_t> Engine::load_state(const std::string& path) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    if (s.draft) return Status(ErrorCode::Unsupported, "load_state() is not available with a draft model");
    Impl::Seq& q = *s.main;
    const Impl::Hold hold(s, q);
    s.clear_context(q);
    std::ifstream in(os::utf8_path(path), std::ios::binary);
    if (!in) return Status(ErrorCode::IoError, "cannot read " + path);
    LIYAB_RETURN_IF_ERROR(s.target->read_state(in));
    uint32_t magic = 0;
    uint64_t n = 0, text = 0;
    const auto damaged = [&s, &q] {
        s.clear_context(q);
        return Status(ErrorCode::IoError, "the saved context is damaged");
    };
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof magic) || magic != kContextMagic ||
        !in.read(reinterpret_cast<char*>(&n), sizeof n) || n != static_cast<uint64_t>(s.target->n_past())) {
        return damaged();
    }
    q.context.resize(n);
    if (!in.read(reinterpret_cast<char*>(q.context.data()), static_cast<std::streamsize>(n * sizeof(int32_t))) ||
        !in.read(reinterpret_cast<char*>(&text), sizeof text)) {
        return damaged();
    }
    if (text != UINT64_MAX) {
        if (text > (uint64_t{1} << 30)) return damaged();
        q.context_text.resize(text);
        if (!in.read(q.context_text.data(), static_cast<std::streamsize>(text))) return damaged();
        uint64_t covered = 0;
        if (!in.read(reinterpret_cast<char*>(&covered), sizeof covered)) {
            covered = n;  // written before the count was saved: the text covers every token
        } else if (covered == 0 || covered > n) {
            return damaged();
        }
        q.context_text_tokens = static_cast<size_t>(covered);
    }
    // Where rows reused away from their prefix start (segment reuse), when
    // the context held some.
    uint32_t tag = 0;
    uint64_t exact = 0, adapter = 0;
    bool tagged = static_cast<bool>(in.read(reinterpret_cast<char*>(&tag), sizeof tag));
    if (tagged && tag == kReusedMagic) {
        if (!in.read(reinterpret_cast<char*>(&exact), sizeof exact) || exact >= n) return damaged();
        q.exact_end = static_cast<size_t>(exact);
        tagged = static_cast<bool>(in.read(reinterpret_cast<char*>(&tag), sizeof tag));
    }
    // The adapter it was computed with (none when untagged) must be the selected one.
    if (tagged && tag == kAdapterMagic && !in.read(reinterpret_cast<char*>(&adapter), sizeof adapter)) return damaged();
    if (adapter != q.context_key) {
        s.clear_context(q);
        return Status(ErrorCode::InvalidArgument,
                      adapter == 0 ? "the saved context was computed without an adapter: select none first"
                                   : "the saved context was computed with another adapter (or scale) than the selected one");
    }
    q.last_prompt_end = static_cast<size_t>(n);
    return static_cast<int32_t>(n);
}

void Engine::reset_context() {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main, /*wait=*/true);
    const Impl::Hold hold(s, *s.main);
    s.clear_context(*s.main);
}

Result<GenerationStats> Engine::generate(std::string_view prompt, const SamplingParams& params,
                                         const TokenCallback& on_token) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return s.generate_text(*s.main, prompt, params, on_token, nullptr, nullptr);
}

Result<GenerationStats> Engine::generate(std::string_view prompt, const SamplingParams& params,
                                         const TokenCallback& on_token, const ForceCallback& force) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return s.generate_text(*s.main, prompt, params, on_token, force ? &force : nullptr, nullptr);
}

Result<GenerationStats> Engine::generate(std::string_view prompt, const SamplingParams& params,
                                         const TokenCallback& on_token, const GenerateOptions& options) {
    if (options.grammar && options.continuation) {
        return Status(ErrorCode::InvalidArgument, "GenerateOptions: set a grammar or a continuation callback, not both");
    }
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return s.generate_text(*s.main, prompt, params, on_token, nullptr, &options);
}

Result<GenerationStats> Engine::generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                                const TokenCallback& on_token) {
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return s.generate(*s.main, prompt, params, on_token, nullptr);
}

Result<GenerationStats> Engine::generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                                const TokenCallback& on_token, const GenerateOptions& options) {
    if (options.grammar && options.continuation) {
        return Status(ErrorCode::InvalidArgument, "GenerateOptions: set a grammar or a continuation callback, not both");
    }
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    return s.generate_tokens_on(*s.main, prompt, params, on_token, options);
}

// generate() from text on sequence q, its call held: the prompt encoded as
// it continues q's context or one the prefix cache keeps (with marked
// segments when the request opts in to segment reuse).
Result<GenerationStats> Engine::Impl::generate_text(Seq& q, std::string_view prompt, const SamplingParams& params,
                                                    const TokenCallback& on_token, const ForceCallback* force,
                                                    const GenerateOptions* options) {
    const std::span<const PromptSegment> marks =
        options != nullptr ? std::span<const PromptSegment>(options->segments) : std::span<const PromptSegment>();
    LIYAB_RETURN_IF_ERROR(check_segments(marks, prompt.size(), prompt));
    const Request request(*this, q, marks, /*tokens=*/false);
    auto tokens = q.request_reuse ? encode_segments(q, prompt, params.add_bos)
                                  : encode_continuation(q, prompt, params.add_bos);
    if (!tokens) return tokens.status();
    const bool structured = options != nullptr && (options->grammar.has_value() || static_cast<bool>(options->continuation));
    return generate(q, *tokens, params, on_token, &prompt, force, structured ? options : nullptr);
}

// generate_tokens() with options on sequence q, its call held.
Result<GenerationStats> Engine::Impl::generate_tokens_on(Seq& q, std::span<const int32_t> prompt,
                                                         const SamplingParams& params, const TokenCallback& on_token,
                                                         const GenerateOptions& options) {
    LIYAB_RETURN_IF_ERROR(check_segments(options.segments, prompt.size()));
    const Request request(*this, q, options.segments, /*tokens=*/true);
    if (q.request_reuse) {
        for (const PromptSegment& m : options.segments) {
            q.request_segments.push_back({static_cast<int32_t>(m.begin), static_cast<int32_t>(m.end)});
        }
    }
    const bool structured = options.grammar.has_value() || static_cast<bool>(options.continuation);
    return generate(q, prompt, params, on_token, nullptr, nullptr, structured ? &options : nullptr);
}

// A generation on sequence q, its call held. Alone (no call on another
// sequence), each pass runs as a single conversation's always did, under a
// Hold of the model: the same passes, speculative steps, pacing per token.
// With other sequences in a call, each pass is a request the scheduler runs
// together with theirs (Impl::submit), one token per step, no drafts; the
// sampling, the grammar, the callbacks and the bookkeeping stay here, on
// the caller's thread. A run may switch between the two at any pass: either
// way a pass gives each of its tokens the same bits, so the output does not
// depend on when it switched.
Result<GenerationStats> Engine::Impl::generate(Seq& q, std::span<const int32_t> prompt, const SamplingParams& params,
                                               const TokenCallback& on_token, const std::string_view* text,
                                               const ForceCallback* force, const GenerateOptions* options) {
    if (prompt.empty()) return Status(ErrorCode::InvalidArgument, "empty prompt");
    if (params.max_tokens <= 0) return Status(ErrorCode::InvalidArgument, "max_tokens must be positive");

    const auto t_start = Clock::now();
    Impl& s = *this;
    q.cancel.store(false, std::memory_order_relaxed);
    const PowerManager::Work thermal(*s.power);  // thermal sampling while this runs
    // Forgets q's context after a failed pass (its states may be partly advanced).
    auto fail = [&](Status status) {
        const Hold hold(s, q);
        s.clear_context(q);
        return status;
    };
    // The last prompt token seeds the decode loop, so it is never reused.
    size_t reused = 0;
    {
        const Hold hold(s, q);
        reused = s.reuse_prefix(q, prompt, prompt.size() - 1, q.request_reuse);
    }
    if (s.alone()) {  // a new run starts its pacing (another sequence's run keeps it going otherwise)
        const std::lock_guard<std::mutex> pace(s.pace_mutex);
        s.power->reset_pacing();
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
    stats.context_ms = std::chrono::duration<double, std::milli>(t_prefill - t_start).count();
    size_t first_uncached = reused;
    stats.cached_prefix_tokens = static_cast<int32_t>(reused);
    const PowerPolicy prefill_policy = s.power->policy();
    s.apply_policy(prefill_policy);
    const Route prefill_route = prefill_policy.throttled ? s.throttled_route() : s.normal_route();
    const ForwardHooks prefill_hooks = s.hooks(prefill_policy, false);
    {
        // Segment reuse (a request that marked segments): stored segments
        // after the exact prefix, processing resumes after the last one reused.
        const Hold hold(s, q);
        first_uncached = s.reuse_request_segments(q, prompt, first_uncached, prompt.size() - 1, prefill_route,
                                                  s.prefill_chunk(), &q.cancel, &stats);
    }
    const std::span<const int32_t> prefix = prompt.first(prompt.size() - 1).subspan(first_uncached);
    const bool hybrid = s.target->config().hybrid();
    for (size_t i = 0; i < prefix.size();) {
        if (q.cancel.load(std::memory_order_relaxed)) {
            stats.cancelled = true;
            return stats;
        }
        const bool alone = s.alone();
        const size_t chunk = static_cast<size_t>(alone ? s.prefill_chunk() : s.shared_chunk());
        const auto part = prefix.subspan(i, std::min(chunk, prefix.size() - i));
        if (alone) {
            const Hold hold(s, q);
            // A background sequence's pass may be stopped for foreground
            // work (between blocks): a hybrid one returns to a snapshot.
            const bool stoppable = !q.foreground();
            if (stoppable && hybrid) s.target->snapshot_state();
            const Stoppable stop(s, q, stoppable, /*prefill_call=*/false);
            ForwardHooks hooks = prefill_hooks;
            if (stoppable) hooks.cancel = &s.preempt;
            if (auto r = s.target->forward(part, Transformer::Logits::None, prefill_route, *s.pool, &hooks); !r) {
                if (r.status().code() == ErrorCode::Cancelled && stoppable && s.target->abandon_pass().is_ok()) {
                    s.count_stop();
                    continue;  // stopped: the same tokens again, now with the others
                }
                s.clear_context(q);
                return r.status();
            }
            if (s.draft) {
                if (auto r = s.draft->forward(part, Transformer::Logits::None, s.draft_route(), *s.pool); !r) {
                    s.clear_context(q);
                    return r.status();
                }
            } else {
                q.context.insert(q.context.end(), part.begin(), part.end());
            }
        } else {
            PassRequest r;  // a stoppable pass takes the snapshot it may return to (run_pass())
            r.tokens = part;
            r.logits = Transformer::Logits::None;
            r.kind = PassKind::Prompt;
            if (Status st = s.submit(q, r); !st.is_ok()) return fail(st);
            q.context.insert(q.context.end(), part.begin(), part.end());
        }
        i += part.size();
    }
    stats.prefill_ms = ms_since(t_prefill);
    // The next chat turn re-sends this prompt: keep its recurrent state.
    if (!s.draft) {
        const Hold hold(s, q);
        s.target->snapshot_state(/*pin=*/true);
        q.last_prompt_end = q.context.size();
    }

    const auto t_decode = Clock::now();
    const Transformer::PhaseTimes phases_before = s.target->phase_times();
    std::string pending_utf8;
    std::string generated;  // the output so far, kept for `force` / `continuation`
    std::vector<float> probs;
    int32_t last = prompt.back();
    // The next token when a forced pass already sampled it from its last row:
    // the next step emits it instead of running a pass of its own.
    std::optional<int32_t> presampled;
    // Whether the tokens about to be emitted came from a shared pass, which
    // the scheduler paced already (once per pass), and for how long.
    bool paced_by_pass = false;
    double pass_paced_ms = 0.0;
    int32_t shared_passes = 0;  // passes this run shared with other sequences (logged)
    bool stop = false;
    std::chrono::microseconds idle{0};

    // Constrained decoding (GenerateOptions). `span` follows the output from
    // where its grammar started (`whole`: the start of the output); its state
    // set is the table's state id. Tables and the trie are the engine's
    // shared caches: this run keeps the ones it uses alive (trim_memory()
    // may drop the caches meanwhile), and uses them under grammar_mutex.
    const ContinuationCallback* continuation =
        options != nullptr && options->continuation ? &options->continuation : nullptr;
    struct Span {
        grammar::StateTable* table;
        int32_t state;
    };
    std::optional<Span> span;
    std::vector<std::shared_ptr<grammar::StateTable>> tables_held;
    std::shared_ptr<const grammar::VocabTrie> trie_held;
    auto trie = [&]() -> const grammar::VocabTrie& {  // grammar_mutex held
        if (!trie_held) trie_held = s.trie();
        return *trie_held;
    };
    const bool whole = options != nullptr && options->grammar.has_value();
    bool span_started = false;  // set by structure() when it opened a span
    bool output_done = false;   // the whole output's grammar matched completely
    // Token healing (run_forced): the active span's last forced text, as where
    // it starts in what structure() returned and the span's state before it.
    struct Heal {
        size_t offset;
        int32_t state;
    };
    std::optional<Heal> heal;
    std::vector<float> masked;
    auto start_span = [&](const Grammar& g) {  // grammar_mutex held
        span.reset();
        std::shared_ptr<grammar::StateTable> table = s.grammar_table(g);
        if (std::find(tables_held.begin(), tables_held.end(), table) == tables_held.end()) tables_held.push_back(table);
        span = Span{table.get(), table->initial()};
        span_started = true;
    };
    if (whole) {
        const std::lock_guard<std::mutex> g(s.grammar_mutex);
        start_span(*options->grammar);
    }
    // What the output must continue with here: the active span's forced text
    // and, each time a span completes, the callback's next continuation. The
    // guard bounds a callback that keeps opening spans that complete at once.
    auto structure = [&](std::string_view so_far) -> std::string {
        const auto t0 = Clock::now();
        std::string out;
        span_started = false;
        heal.reset();
        for (int guard = 0; guard < 64; ++guard) {
            if (span) {
                const std::lock_guard<std::mutex> g(s.grammar_mutex);
                const grammar::StateTable& t = *span->table;
                if (t.accepting(span->state) && !t.extendable(span->state)) {
                    span.reset();
                    heal.reset();
                    if (whole) {
                        output_done = true;
                        break;
                    }
                    continue;
                }
                std::string f = span->table->forced_text(span->state);
                if (f.empty()) break;
                heal = Heal{out.size(), span->state};
                span->state = span->table->advance(span->state, f);
                out += f;
                continue;
            }
            if (continuation == nullptr) break;
            std::string text_so_far(so_far);
            text_so_far += out;
            Continuation c = (*continuation)(text_so_far);
            out += c.text;
            if (!c.grammar) break;
            const std::lock_guard<std::mutex> g(s.grammar_mutex);
            start_span(*c.grammar);
        }
        stats.grammar_ms += ms_since(t0);
        return out;
    };
    // Samples the next token from `logits`, masked by the active span.
    auto sample_next_token = [&](std::span<const float> logits) -> Result<int32_t> {
        if (!span) {
            sampler.probabilities(logits, probs);
            return sampler.greedy() ? Sampler::argmax(probs) : sampler.sample(probs);
        }
        const auto t0 = Clock::now();
        masked.assign(logits.begin(), logits.end());
        // Greedy and top-k keep only the best allowed tokens, found among the
        // most likely ones; the state's whole mask is built (and cached) only
        // when they are not there or every allowed token counts (top_k 0).
        const size_t k = sampler.greedy() ? 1 : params.top_k > 0 ? static_cast<size_t>(params.top_k) : 0;
        size_t allowed = k;
        {
            const std::lock_guard<std::mutex> g(s.grammar_mutex);
            if (k == 0 || !span->table->keep_top_allowed(span->state, *s.tokenizer, trie(), masked, k)) {
                allowed = span->table->apply_mask(span->state, trie(), masked);
            }
        }
        stats.grammar_ms += ms_since(t0);
        if (allowed == 0) {
            return Status(ErrorCode::InvalidArgument,
                          "grammar: no token of the vocabulary can continue the output here");
        }
        ++stats.constrained_tokens;
        sampler.probabilities(masked, probs);
        return sampler.greedy() ? Sampler::argmax(probs) : sampler.sample(probs);
    };

    // Forced continuation (structured output): the text the caller
    // requires next (a tool call's fixed syntax, a name it completes) goes
    // through the model in one batched pass with the token just sampled,
    // at a fraction of decoding it token by token, and is emitted as
    // output. The pass's last row is the next token's logits: it is
    // sampled here, so forced text costs no decoding step of its own
    // (one per tool call segment, a whole pass over the weights on a
    // large MoE). The CPU backend computes a row the same bits in any
    // batch, so this is the token the step would have sampled. Decoding hooks a batched pass does not run
    // (experimental early exit, EGLS) would change those logits: with
    // them the last forced token seeds the next step instead, as before.
    // With GenerateOptions the forced text is encoded as a continuation of
    // the token before it, so its tokens spell exactly the text emitted (no
    // SentencePiece space prefix), which is what a grammar follows; a
    // ForceCallback's text keeps the encoding it always had (the same
    // tokens on byte-level BPE vocabularies).
    //
    // Token healing: forced text ends where the grammar offers a choice, and
    // its last token can be the start of a longer token the model writes
    // there (a forced ` {` where Qwen writes ` {"`, a forced `<` where it
    // writes `</`). Forcing it would make the model go on from a token
    // boundary it never saw in training, which skews the choice (measured:
    // optional arguments dropped after ` {`, a parameter added after `<`).
    // When a longer token the span allows starts with that last token, it
    // is not forced: the span goes back to the state before it and the next
    // token, sampled under the mask, is the model's (that token or a longer
    // one). The pass stays the same size or shrinks, so this costs no step.
    auto run_forced = [&](std::string forced, const PowerPolicy& policy, const Route& route) -> Status {
        auto forced_tokens =
            options != nullptr ? s.tokenizer->encode_continuation(forced, last) : s.tokenizer->encode(forced, false);
        if (!forced_tokens) return forced_tokens.status();
        if (span && heal && !forced_tokens->empty()) {
            const std::string& tail = s.tokenizer->piece(forced_tokens->back());
            // Only a token within the span's own forced text, whose bytes the
            // span's states cover (text a callback forced before it has none).
            if (!tail.empty() && heal->offset <= forced.size() && tail.size() <= forced.size() - heal->offset) {
                const size_t keep = forced.size() - tail.size();
                const std::lock_guard<std::mutex> g(s.grammar_mutex);
                const int32_t before = span->table->advance(
                    heal->state, std::string_view(forced).substr(heal->offset, keep - heal->offset));
                if (before >= 0 && span->table->extends(before, tail, trie())) {
                    forced_tokens->pop_back();
                    forced.resize(keep);
                    span->state = before;
                }
            }
        }
        heal.reset();
        if (forced_tokens->empty()) return Status::ok();
        const bool sample_next = (s.speculative != nullptr || !s.decode_only_hooks()) && !output_done;
        std::vector<int32_t> batch{last};
        batch.insert(batch.end(), forced_tokens->begin(), forced_tokens->end() - (sample_next ? 0 : 1));
        const auto t_forced = Clock::now();
        if (s.alone()) {
            const Hold hold(s, q);
            // Speculative steps run without hooks; the experimental ones
            // apply to plain decoding only.
            const ForwardHooks hooks = s.speculative ? ForwardHooks{} : s.hooks(policy, false);
            // Forced tokens are never rolled back: no recurrent-state
            // checkpoints (the next speculative step turns them back on).
            if (s.speculative) s.target->set_checkpointing(false);
            auto logits = s.target->forward(batch, sample_next ? Transformer::Logits::Last : Transformer::Logits::None,
                                            route, *s.pool, &hooks);
            if (!logits) {
                s.clear_context(q);
                return logits.status();
            }
            ++stats.decode_passes;
            if (sample_next) {
                if (policy.throttled) ++stats.thermal_reroutes;
                if (hooks.head_mask != nullptr) ++stats.head_pruned_steps;
                if (hooks.ffn_matmul != nullptr) ++stats.sparse_ffn_steps;
                auto token = sample_next_token(*logits);
                if (!token) {
                    s.clear_context(q);
                    return token.status();
                }
                presampled = *token;
            }
            if (s.draft) {  // the draft model reads the same tokens, so both caches stay aligned
                if (auto r = s.draft->forward(batch, Transformer::Logits::None, s.draft_route(), *s.pool); !r) {
                    s.clear_context(q);
                    return r.status();
                }
            } else {
                q.context.insert(q.context.end(), batch.begin(), batch.end());
            }
            paced_by_pass = false;
        } else {
            PassRequest r;
            r.tokens = batch;
            r.logits = sample_next ? Transformer::Logits::Last : Transformer::Logits::None;
            r.kind = PassKind::Forced;
            if (Status st = s.submit(q, r); !st.is_ok()) return fail(st);
            ++stats.decode_passes;
            if (r.sequences > 1) ++shared_passes;
            q.context.insert(q.context.end(), batch.begin(), batch.end());
            if (sample_next) {
                if (r.throttled) ++stats.thermal_reroutes;
                if (r.head_pruned) ++stats.head_pruned_steps;
                if (r.sparse_ffn) ++stats.sparse_ffn_steps;
                auto token = sample_next_token(r.rows);
                if (!token) return fail(token.status());
                presampled = *token;
            }
            paced_by_pass = true;
            pass_paced_ms = r.paced_ms;
        }
        stats.forced_ms += ms_since(t_forced);
        last = forced_tokens->back();
        stats.forced_tokens += static_cast<int32_t>(forced_tokens->size());
        // Whole characters are emitted now. Forced text that ends inside
        // one (a C callback's buffer cut it) keeps the rest pending until
        // the next token completes it, as a sampled piece would: pieces
        // stay whole UTF-8.
        const size_t ready = complete_utf8_prefix(forced);
        pending_utf8.assign(forced, ready);
        generated.append(forced, 0, ready);
        if (ready > 0 && on_token && !on_token(std::string_view(forced).substr(0, ready), last)) stop = true;
        return Status::ok();
    };

    // The output may have to start with forced text (a grammar's fixed
    // opening, a callback's text) or a constrained span.
    if (whole || continuation != nullptr) {
        const std::string forced = structure("");
        if (!forced.empty()) {
            const PowerPolicy policy = s.power->policy();
            s.apply_policy(policy);
            LIYAB_RETURN_IF_ERROR(run_forced(forced, policy, policy.throttled ? s.throttled_route() : s.normal_route()));
        }
        if (output_done) stop = true;
    }

    while (!stop) {
        if (q.cancel.load(std::memory_order_relaxed)) {
            stats.cancelled = true;
            break;
        }
        const PowerPolicy policy = s.power->policy();
        s.apply_policy(policy);
        const Route route = policy.throttled ? s.throttled_route() : s.normal_route();

        std::vector<int32_t> next;
        if (presampled) {
            next.push_back(*presampled);
            presampled.reset();
        } else if (s.alone()) {
            paced_by_pass = false;
            const Hold hold(s, q);
            if (s.speculative && !span) {
                if (policy.throttled) ++stats.thermal_reroutes;
                const int32_t proposed = s.speculative->proposed(), accepted = s.speculative->accepted();
                auto r = s.speculative->step(last, q.context, sampler, route, s.draft_route(), *s.pool);
                stats.draft_tokens_proposed += s.speculative->proposed() - proposed;
                stats.draft_tokens_accepted += s.speculative->accepted() - accepted;
                if (!r) {
                    s.clear_context(q);
                    return r.status();
                }
                ++stats.decode_passes;
                next = std::move(r).value();
                if (!s.draft) {  // the target now caches `last` and every returned token but the final one
                    q.context.push_back(last);
                    q.context.insert(q.context.end(), next.begin(), next.end() - 1);
                }
            } else {
                // A plain step; also with speculative decoding while a grammar
                // constrains the output (one token at a time, no hooks, no
                // rollback checkpoints, the draft model fed the same token).
                if (policy.throttled) ++stats.thermal_reroutes;
                const ForwardHooks hooks = s.speculative ? ForwardHooks{} : s.hooks(policy, true);
                if (s.speculative) s.target->set_checkpointing(false);
                auto logits = s.target->forward(std::span<const int32_t>(&last, 1), Transformer::Logits::Last, route,
                                                *s.pool, &hooks);
                if (!logits) {
                    s.clear_context(q);
                    return logits.status();
                }
                ++stats.decode_passes;
                if (s.draft) {
                    if (auto r = s.draft->forward(std::span<const int32_t>(&last, 1), Transformer::Logits::None,
                                                  s.draft_route(), *s.pool);
                        !r) {
                        s.clear_context(q);
                        return r.status();
                    }
                } else {
                    q.context.push_back(last);
                }
                if (hooks.head_mask != nullptr) ++stats.head_pruned_steps;
                stats.ffn_blocks_skipped += s.target->last_ffn_skips();
                if (hooks.ffn_matmul != nullptr) ++stats.sparse_ffn_steps;
                if (const int32_t exit_layer = s.target->last_exit_layer(); exit_layer >= 0) {
                    ++stats.early_exits;
                    stats.early_exit_layers_skipped += s.target->config().n_layers - 1 - exit_layer;
                }
                auto token = sample_next_token(*logits);
                if (!token) {
                    s.clear_context(q);
                    return token.status();
                }
                next.push_back(*token);
            }
        } else {
            // With other sequences: a plain step in a pass shared with theirs.
            PassRequest r;
            r.tokens = std::span<const int32_t>(&last, 1);
            r.logits = Transformer::Logits::Last;
            r.kind = PassKind::Decode;
            if (Status st = s.submit(q, r); !st.is_ok()) return fail(st);
            ++stats.decode_passes;
            if (r.sequences > 1) ++shared_passes;
            q.context.push_back(last);
            if (r.throttled) ++stats.thermal_reroutes;
            if (r.head_pruned) ++stats.head_pruned_steps;
            if (r.sparse_ffn) ++stats.sparse_ffn_steps;
            stats.ffn_blocks_skipped += r.ffn_skips;
            if (r.exit_layer >= 0) {
                ++stats.early_exits;
                stats.early_exit_layers_skipped += s.target->config().n_layers - 1 - r.exit_layer;
            }
            auto token = sample_next_token(r.rows);
            if (!token) return fail(token.status());
            next.push_back(*token);
            paced_by_pass = true;
            pass_paced_ms = r.paced_ms;
        }

        if (stats.ttft_ms == 0.0) {
            stats.ttft_ms = ms_since(t_prefill);
            if (const ExpertStore* store = s.target->expert_store()) {
                stats.ttft_stall_ms = store->stats().stall_ms - experts_before.stall_ms;
            }
        }
        // Tokens of `next` the output keeps: all of them, unless the output
        // stops at an earlier one (end of generation, max_tokens, the
        // callback, cancel) or the caller forces text after it (a
        // speculative step returns several).
        size_t kept = next.size();
        std::string forced;
        for (size_t i = 0; i < next.size(); ++i) {
            const int32_t token = next[i];
            if (s.tokenizer->is_end_of_generation(token)) {
                stop = true;
                kept = i + 1;
                break;
            }
            // Duty cycling: hold this token until its slot at the target
            // rate; the SoC idles in between instead of racing ahead. A
            // token of a shared pass waited for its pass's slot already.
            if (paced_by_pass) {
                idle += std::chrono::microseconds(static_cast<int64_t>(pass_paced_ms * 1000.0));
                paced_by_pass = false;
            } else {
                const std::lock_guard<std::mutex> pace(s.pace_mutex);
                idle += s.power->pace_token();
            }
            ++stats.generated_tokens;
            s.tokens_generated.fetch_add(1, std::memory_order_relaxed);
            pending_utf8 += s.tokenizer->piece(token);
            if (span) {  // the mask allowed only tokens the grammar accepts here
                const auto t0 = Clock::now();
                const std::lock_guard<std::mutex> g(s.grammar_mutex);
                span->state = span->table->advance(span->state, s.tokenizer->piece(token));
                if (span->state < 0) return fail(Status(ErrorCode::Internal, "grammar: a sampled token left the grammar"));
                span->state = span->table->compact(span->state);
                stats.grammar_ms += ms_since(t0);
            }
            const size_t ready = complete_utf8_prefix(pending_utf8);
            if (ready > 0) {
                const bool keep_going = on_token ? on_token(std::string_view(pending_utf8).substr(0, ready), token) : true;
                if (force != nullptr || continuation != nullptr) generated.append(pending_utf8, 0, ready);
                pending_utf8.erase(0, ready);
                if (!keep_going) {
                    stop = true;
                    kept = i + 1;
                    break;
                }
            }
            if (stats.generated_tokens >= params.max_tokens || q.cancel.load(std::memory_order_relaxed)) {
                stats.cancelled = q.cancel.load(std::memory_order_relaxed);
                stop = true;
                kept = i + 1;
                break;
            }
            // Structured output: asked after every token that ends a whole
            // character, so the model never writes past a point the caller
            // fixes, whatever number of tokens a step returned.
            if (force != nullptr && pending_utf8.empty()) {
                forced = (*force)(generated);
                if (!forced.empty()) {
                    kept = i + 1;
                    break;
                }
            }
            // Constrained decoding: the span's forced text, its end, the
            // callback's next continuation; past a new span or forced text,
            // a speculative step's later tokens are not the output's.
            if ((span || continuation != nullptr) && pending_utf8.empty()) {
                forced = structure(generated);
                if (!forced.empty() || span_started || output_done) {
                    kept = i + 1;
                    break;
                }
            }
        }
        if (kept < next.size()) {
            // A speculative step went past that point: drop its tokens after
            // it. The caches hold `last` and every returned token but the
            // final one, all from the step's own verification pass, so a
            // hybrid model returns from that pass's checkpoints. At a stop,
            // the context then ends as plain decoding leaves it (the last
            // token sampled, not processed): tokens past an end of generation
            // stayed in it, and their text in the text it remembers, so the
            // next turn's prompt matched an end token twice and diverged.
            const auto dropped = static_cast<int32_t>(next.size() - kept);
            const Hold hold(s, q);
            const int32_t keep = s.target->n_past() - dropped;
            Status rolled_back = s.target->truncate(keep);
            if (rolled_back.is_ok() && s.draft) rolled_back = s.draft->truncate(keep);
            if (!rolled_back.is_ok()) {
                s.clear_context(q);
                return rolled_back;
            }
            if (!s.draft) q.context.resize(q.context.size() - static_cast<size_t>(dropped));
            next.resize(kept);
        }
        last = next.back();

        if (!stop && !forced.empty()) LIYAB_RETURN_IF_ERROR(run_forced(forced, policy, route));
        if (output_done) stop = true;
    }
    if (!pending_utf8.empty() && on_token) on_token(pending_utf8, last);
    {
        const std::lock_guard<std::mutex> g(s.sched);
        q.phase = Seq::Phase::Away;  // no more passes: a leader need not wait for this sequence
    }

    stats.decode_ms = ms_since(t_decode);
    stats.tokens_per_second = stats.decode_ms > 0.0 ? stats.generated_tokens * 1000.0 / stats.decode_ms : 0.0;
    stats.paced_idle_ms = static_cast<double>(idle.count()) / 1000.0;
    if (s.weight_loader) {
        const TripleBufferLoader::Stats after = s.weight_loader->stats();
        stats.weight_stalls = static_cast<int32_t>(after.stalls - loader_before.stalls);
        stats.weight_wait_ms = after.consumer_wait_ms - loader_before.consumer_wait_ms;
    }
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
    {
        const Hold hold(s, q);
        stats.kv_cache_bytes = s.target->kv_cache().bytes_in_use();
        s.keep_request_segments(q);
    }
    if (text != nullptr && !s.draft) s.remember_text(q, prompt, *text);
    if (shared_passes > 0) {
        LIYAB_LOG_INFO("sequence %d: %d tokens at %.1f tok/s, %d of %d decoding passes shared with other sequences", q.id,
                       stats.generated_tokens, stats.tokens_per_second, shared_passes, stats.decode_passes);
    }
    return stats;
}

// Engine::prefill on sequence q, its call held. Alone, each pass runs as a
// single conversation's always did; with other sequences, as requests the
// scheduler runs with theirs (passes of at most max_batch tokens but with
// latent attention, see shared_chunk()).
Status Engine::Impl::prefill_on(Seq& q, std::string_view text, std::span<const PromptSegment> segments,
                                bool add_bos) {
    Impl& s = *this;
    if (s.draft) return Status(ErrorCode::Unsupported, "prefill() is not available with a draft model");
    LIYAB_RETURN_IF_ERROR(check_segments(segments, text.size(), text));
    const Request request(s, q, segments, /*tokens=*/false);
    auto tokens = q.request_reuse ? s.encode_segments(q, text, add_bos) : s.encode_continuation(q, text, add_bos);
    if (!tokens) return tokens.status();
    q.cancel_prefill.store(false, std::memory_order_relaxed);
    const PowerManager::Work thermal(*s.power);  // thermal sampling while this runs
    const std::vector<int32_t>& t = tokens.value();
    const PowerPolicy policy = s.power->policy();
    s.apply_policy(policy);
    const Route route = policy.throttled ? s.throttled_route() : s.normal_route();
    ForwardHooks hooks = s.hooks(policy, false);
    // Interruptible within a pass (cancel_prefill() is checked between
    // blocks): a pass of 1024 tokens of a streamed MoE runs for tens of
    // seconds. The work done so far stays: a hybrid model's state is
    // snapshotted before each pass, and an abandoned pass returns to it, so
    // the next prompt continuing this text resumes there.
    hooks.cancel = &q.cancel_prefill;
    size_t reused = 0;
    {
        const Hold hold(s, q);
        reused = s.reuse_prefix(q, t, t.size(), q.request_reuse);
        // Segment reuse (marked segments): stored ones after the exact prefix;
        // cancel_prefill() stops it between blocks, keeping that prefix.
        reused = s.reuse_request_segments(q, t, reused, t.size(), route, s.prefill_chunk(), &q.cancel_prefill, nullptr);
    }
    if (q.cancel_prefill.load(std::memory_order_relaxed)) return Status(ErrorCode::Cancelled, "prefill cancelled");
    const bool hybrid = s.target->config().hybrid();
    for (size_t i = reused; i < t.size();) {
        const bool alone = s.alone();
        const size_t chunk = static_cast<size_t>(alone ? s.prefill_chunk() : s.shared_chunk());
        const std::span<const int32_t> part(t.data() + i, std::min(chunk, t.size() - i));
        if (alone) {
            const Hold hold(s, q);
            if (hybrid) s.target->snapshot_state();
            // A background sequence's pass may be stopped for foreground work
            // too (Stoppable: cancel_prefill() sets the same flag then).
            const bool stoppable = !q.foreground();
            const Stoppable stop(s, q, stoppable, /*prefill_call=*/true);
            ForwardHooks pass_hooks = hooks;
            if (stoppable) pass_hooks.cancel = &s.preempt;
            if (auto r = s.target->forward(part, Transformer::Logits::None, route, *s.pool, &pass_hooks); !r) {
                if (r.status().code() == ErrorCode::Cancelled && s.target->abandon_pass().is_ok()) {
                    if (q.cancel_prefill.load(std::memory_order_relaxed)) {
                        return Status(ErrorCode::Cancelled, "prefill cancelled");
                    }
                    s.count_stop();
                    continue;  // stopped for foreground work: the same tokens again, now with the others
                }
                s.clear_context(q);
                return r.status();
            }
        } else {
            if (q.cancel_prefill.load(std::memory_order_relaxed)) return Status(ErrorCode::Cancelled, "prefill cancelled");
            if (hybrid) {  // the state an abandoned pass returns to
                const Hold hold(s, q);
                s.target->snapshot_state();
            }
            PassRequest r;
            r.tokens = part;
            r.logits = Transformer::Logits::None;
            r.kind = PassKind::Prefill;
            if (Status st = s.submit(q, r); !st.is_ok()) {
                if (st.code() == ErrorCode::Cancelled) return Status(ErrorCode::Cancelled, "prefill cancelled");
                const Hold hold(s, q);
                s.clear_context(q);
                return st;
            }
        }
        q.context.insert(q.context.end(), part.begin(), part.end());
        i += part.size();
    }
    {
        const Hold hold(s, q);
        s.target->snapshot_state();  // the next prompt starts with this text
        s.keep_request_segments(q);
    }
    s.remember_text(q, t, text);
    return Status::ok();
}

bool Engine::has_chat_template() const noexcept { return impl_->chat_template.has_value(); }

Result<std::string> Engine::render_chat(std::string_view request_json) const {
    if (!impl_->chat_template) return impl_->chat_template_error;
    return impl_->chat_template->render_chat(request_json, impl_->chat_special);
}

Pooling Engine::pooling() const noexcept { return impl_->pooling.declared; }

int32_t Engine::embedding_size() const noexcept { return impl_->target->config().n_embd; }

Result<std::vector<std::vector<float>>> Engine::embed(std::span<const std::string_view> texts,
                                                      const EmbedOptions& options, PoolingStats* stats) {
    const auto t_start = Clock::now();
    if (stats != nullptr) *stats = PoolingStats{};
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    if (!s.pooling_error.is_ok()) return s.pooling_error;
    Pooling pooling = options.pooling;
    if (pooling == Pooling::Auto) {
        pooling = s.pooling.declared;
        switch (pooling) {
            case Pooling::None:
                return Status(ErrorCode::Unsupported,
                              "the model declares no pooling (pooling_type): it is not an embedding model; ask for "
                              "Mean or Last pooling explicitly to embed with it anyway");
            case Pooling::Rank:
                return Status(ErrorCode::Unsupported, "the model is a reranker (pooling_type rank): use rerank()");
            case Pooling::Cls:
                return Status(ErrorCode::Unsupported,
                              "the model declares first-token (CLS) pooling, which needs a bidirectional encoder: in "
                              "a causal decoder the first token's state does not depend on the text");
            default:
                break;
        }
    } else if (pooling != Pooling::Mean && pooling != Pooling::Last) {
        return Status(ErrorCode::InvalidArgument, "EmbedOptions::pooling must be Auto, Mean or Last");
    }
    const int32_t size = s.target->config().n_embd;
    if (options.dimensions < 0 || options.dimensions > size) {
        return Status(ErrorCode::InvalidArgument,
                      "EmbedOptions::dimensions must be between 0 and " + std::to_string(size));
    }
    std::vector<std::vector<int32_t>> sequences;
    sequences.reserve(texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        auto tokens = s.encode_pooled(texts[i], options.truncate, i, "input");
        if (!tokens) return tokens.status();
        sequences.push_back(std::move(tokens).value());
    }
    PoolTarget target;
    target.pooling = pooling;
    auto run = s.run_pooling(sequences, target);
    if (!run) return run.status();
    auto vectors = finish_embeddings(run->values, static_cast<size_t>(size), options.dimensions, options.normalize);
    const double ms = ms_since(t_start);
    LIYAB_LOG_INFO("embed: %zu inputs, %lld tokens (%lld shared) in %.0f ms", texts.size(),
                   static_cast<long long>(run->tokens), static_cast<long long>(run->reused), ms);
    if (stats != nullptr) {
        *stats = PoolingStats{static_cast<int32_t>(run->tokens), static_cast<int32_t>(run->reused), ms};
    }
    return vectors;
}

Result<std::vector<float>> Engine::rerank(std::string_view query, std::span<const std::string_view> documents,
                                          const RerankOptions& options, PoolingStats* stats) {
    const auto t_start = Clock::now();
    if (stats != nullptr) *stats = PoolingStats{};
    Impl& s = *impl_;
    const Impl::Call call(s, *s.main);
    if (!call) return Status(ErrorCode::Busy, "a generation is already running on this engine");
    if (!s.pooling_error.is_ok()) return s.pooling_error;
    const std::string& tmpl = options.prompt_template.empty() ? s.pooling.rerank_template : options.prompt_template;
    if (tmpl.empty()) {
        return Status(ErrorCode::InvalidArgument,
                      "no rerank prompt: the model has none (tokenizer.chat_template.rerank) and "
                      "RerankOptions::prompt_template is empty");
    }
    PoolTarget target;
    target.pooling = Pooling::Last;
    for (const std::string& label : options.labels) {
        auto ids = s.tokenizer->encode(label, false);
        if (!ids) return ids.status();
        if (ids->size() != 1) {
            return Status(ErrorCode::InvalidArgument, "the rerank label '" + label + "' is " +
                                                          std::to_string(ids->size()) + " tokens, not one");
        }
        target.logit_ids.push_back(ids->front());
    }
    if (target.logit_ids.empty() && s.pooling.head == nullptr) {
        if (!s.pooling.head_error.is_ok()) return s.pooling.head_error;
        return Status(ErrorCode::Unsupported,
                      "the model has no ranking head (cls.output.weight, pooling_type rank); pass "
                      "RerankOptions::labels to score with its LM head");
    }
    std::vector<std::vector<int32_t>> sequences;
    sequences.reserve(documents.size());
    for (size_t i = 0; i < documents.size(); ++i) {
        auto prompt = format_rerank_prompt(tmpl, query, documents[i]);
        if (!prompt) return prompt.status();
        auto tokens = s.encode_pooled(*prompt, false, i, "the pair of document");
        if (!tokens) return tokens.status();
        sequences.push_back(std::move(tokens).value());
    }
    auto run = s.run_pooling(sequences, target);
    if (!run) return run.status();
    Result<std::vector<float>> scores =
        target.logit_ids.empty() ? head_scores(s.pooling, *s.cpu, run->values, s.target->config().n_embd)
                                 : Result<std::vector<float>>(label_scores(run->values, target.logit_ids.size()));
    if (!scores) return scores.status();
    const double ms = ms_since(t_start);
    LIYAB_LOG_INFO("rerank: %zu documents, %lld tokens (%lld shared) in %.0f ms", documents.size(),
                   static_cast<long long>(run->tokens), static_cast<long long>(run->reused), ms);
    if (stats != nullptr) {
        *stats = PoolingStats{static_cast<int32_t>(run->tokens), static_cast<int32_t>(run->reused), ms};
    }
    return scores;
}

Result<ChatFormat> Engine::chat_format(std::string_view probe_json) const {
    if (!impl_->chat_template) return impl_->chat_template_error;
    return impl_->chat_template->format(probe_json, impl_->chat_special);
}

}  // namespace liyab
