// Liyab — inference engine: ties the mmap loader, device detection, power
// manager, backends and (optionally) speculative decoding together.
#ifndef LIYAB_ENGINE_H
#define LIYAB_ENGINE_H

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "liyab/device_detect.h"
#include "liyab/power_manager.h"
#include "liyab/types.h"

namespace liyab {

// Experimental features: require a build with LIYAB_ENABLE_EXPERIMENTAL=ON,
// otherwise Engine::create() fails with ErrorCode::Unsupported. Both apply to
// non-speculative decoding only. See include/liyab/experimental/.
struct ExperimentalConfig {
    bool early_exit = false;              // skip remaining blocks on confident tokens (lossy)
    float early_exit_threshold = 0.98f;   // max softmax probability needed to exit
    int32_t early_exit_min_layer = -1;    // -1: n_layers / 2
    int32_t early_exit_interval = 4;      // probe every N blocks
    bool head_pruning = false;            // mask low-importance heads when hot / LowPower (lossy)
    float head_keep_ratio = 0.75f;        // heads kept per block while pruning
    bool egls = false;                    // entropy-guided FFN skipping (lossy)
    float egls_threshold = 0.002f;        // skip a block's FFN when its entropy delta is below this
    bool tdss = false;                    // 2:4 sparse FFN weights while thermally throttled (lossy)
    bool tdss_force = false;              // use the sparse FFN even when cool (benchmarks)
    std::string kv_dedup_dir;             // non-empty: persistent prefix KV cache in this directory
};

struct EngineConfig {
    std::string model_path;           // GGUF file (F32/F16/Q8_0/Q4_0/Q4_1 tensors)
    std::string draft_model_path;     // optional: enables speculative decoding
    int32_t n_threads = 0;            // 0: performance cores, minus one when there is no efficiency cluster
    int32_t context_length = 0;       // 0: automatic: what 256 MiB of KV cache holds (4096..32768), at most the training context
    int32_t sliding_window = 0;       // 0: full attention; else attend to the last N tokens
    int32_t kv_sink_tokens = 8;       // with a sliding window: first N tokens stay visible (attention sinks)
    // Paged KV cache format. Q8_0 is the quality default; Q4_0 / Q4_1 (INT4,
    // -72% / -69% vs F16) trade some accuracy for memory on long contexts.
    KvCacheType kv_cache_type = KvCacheType::Q8_0;
    // Stream transformer blocks through the triple-buffered loader (3 block
    // buffers: fetch | prepare | execute) instead of reading the mmap in place.
    // For models larger than RAM only: weight memory is bounded at 3 blocks,
    // but every token re-reads every block from storage, so decode speed is
    // capped by flash bandwidth / model size. With LIYAB_ENABLE_EXPERIMENTAL
    // stage 1 uses O_DIRECT / io_uring.
    bool triple_buffer_loading = false;
    // Mixture-of-experts models: RAM cache for routed experts streamed from
    // storage with direct I/O, in MiB. -1: automatic (only when the model does
    // not fit in RAM; uses the free RAM left after the other weights), 0:
    // never stream experts, > 0: always, with this cache size.
    int64_t expert_cache_mb = -1;
    // Total memory the engine may keep resident (weights, expert cache,
    // streaming slots), in MiB; 0: limited by free RAM only. Streaming
    // decisions use min(free RAM, budget). Set it on platforms with per-app
    // caps the OS counters do not show (HyperOS stops apps above 6 GiB PSS).
    int64_t memory_budget_mb = 0;
    // With expert streaming (MoE): convert the resident Q8_0 matrices
    // (attention / DeltaNet projections, shared experts) to Q4_K (4) or Q5_K
    // (5) at load. Lossy, opt-in: fewer bytes read per token, and the memory
    // saved goes to the expert cache. Converted on every core at load;
    // ignored when experts are not streamed. 0: off. -1: automatic, only when
    // the expert cache would otherwise hold under 10% of the experts (Q5_K if
    // that frees enough memory, else Q4_K); see Engine::memory_plan().
    int32_t requant_bits = 0;
    // MoE: per token, run only the fewest top-ranked experts of the top-k
    // whose router probabilities cover this fraction of the top-k's total
    // (e.g. 0.9), renormalized over the kept ones. Lossy, opt-in: fewer
    // expert reads and less compute per token. >= 1: all top-k (default).
    float moe_expert_mass = 1.0f;
    // MoE: at most this many experts per token (the top ones, renormalized);
    // 0: the model's own top-k. Lossy when below it.
    int32_t moe_max_experts = 0;
    // MoE with streamed experts: a chosen expert that is not in RAM yet is
    // skipped instead of waited for when, for every token using it, its
    // router weight is below this share of the token's experts (the others
    // renormalized). Lossy, opt-in: cuts stalls on low-impact experts. 0: off.
    float moe_skip_slow = 0.0f;
    // Prefix cache, MiB of RAM at most (0: off): contexts another
    // conversation's prompt replaced are kept (KV pages, recurrent states and
    // the text they encode, LRU) and restored when a prompt continues one,
    // instead of processing it again. Taken out of memory_budget_mb; on a MoE
    // whose experts stream, only as far as the expert cache can spare: it
    // never takes the cache under 10% of the experts, so it never causes a
    // requantization (requant_bits -1). On Qwen3.6-35B-A3B a 2000-token
    // conversation costs ~85 MiB.
    int64_t prefix_cache_mb = 0;
    // Prefix cache on storage: contexts set aside that RAM does not hold
    // (prefix_cache_mb, possibly 0) are written to files in this directory
    // (created if missing), up to prefix_cache_disk_mb (0: off), least
    // recently used out first, and read back when a prompt continues one.
    // No RAM spent: worth it wherever RAM is tight and storage is fast (UFS 4
    // writes or reads a ~100 MB context in a fraction of a second; processing
    // it again takes minutes on a large MoE). The engine owns the ctx-*.state
    // files there: it removes its own when destroyed and, when created, those
    // that processes which ended left behind.
    std::string prefix_cache_dir;
    int64_t prefix_cache_disk_mb = 0;
    // Most drafts per speculative step (k). 3 by default: a verification pass
    // then holds 4 tokens, exactly one tile of the repacked CPU kernels.
    int32_t draft_tokens = 3;
    // Without a draft model: draft up to draft_tokens tokens by looking the
    // last tokens up in the conversation (SpeculativeDecoder::lookup). Same
    // output distribution, no extra weights, any model; pays off when the text
    // repeats its context and on models that verify k tokens for about the
    // cost of one (dense models in RAM).
    bool lookup_drafts = false;
    // Learn on the device how many drafts per step pay off (0..draft_tokens;
    // see SpeculativeDecoder). false: always draft_tokens, which keeps sampled
    // output reproducible for a fixed seed.
    bool adaptive_drafts = true;
    std::optional<BackendKind> backend;  // force one backend; default: device ranking
    std::optional<bool> streaming;       // force mmap streaming mode; default: by RAM
    PowerConfig power;                   // profile, thermal thresholds, pacing
    bool thermal_polling = true;         // background thermal sampling thread
    ExperimentalConfig experimental;
};

// GGUF `general.architecture` values this build can run, comma-separated
// (e.g. "llama,mistral,qwen2,..."), so front ends can filter models before
// downloading them.
LIYAB_API std::string supported_architectures();

// Receives each decoded piece (complete UTF-8 only) and the token that
// completed it. Return false to stop generation.
using TokenCallback = std::function<bool(std::string_view piece, int32_t token)>;

// Structured output: called after each decoded piece with the output so far;
// returns text the output must continue with (empty: none). The engine runs
// it through the model in one batched pass, as cheap as a short prefill, and
// emits it like generated text; decoding resumes after it. Used for the fixed
// parts of a tool call and to complete a name once only one is possible.
// Called on the generating thread, only at whole-UTF-8 boundaries; forced
// tokens count in GenerationStats::forced_tokens, not generated_tokens.
using ForceCallback = std::function<std::string(std::string_view generated)>;

class LIYAB_API Engine {
public:
    static Result<std::unique_ptr<Engine>> create(const EngineConfig& config);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Generates a continuation of `prompt` (already chat-formatted by the
    // caller). The context is reused across calls: when `prompt` continues the
    // text the engine already processed (previous prompt + its generated
    // reply, or a prefill()), that text keeps the tokens it was processed as
    // (re-encoding a reply need not reproduce the tokens that were generated)
    // and only the rest is processed, which makes chat turns start fast.
    // Otherwise the longest common token prefix is kept: any length on
    // attention-only models; on hybrid (DeltaNet) models, whose recurrent
    // states cannot rewind, up to the nearest state snapshot, taken at the end
    // of every prompt and prefill (Transformer::snapshot_state), so an edited
    // last reply or a dropped turn recomputes only what follows that point.
    // Output is the same as from a fresh context, except that a reused reply
    // keeps the tokenization it was generated with. Not reentrant: a
    // concurrent call returns ErrorCode::Busy.
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token);
    // Same, with forced continuations (see ForceCallback; not with a draft
    // model or lookup speculation, where it is ignored).
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token, const ForceCallback& force);
    // Same, from token ids (for vocabularies whose text encoder Liyab lacks);
    // reuse follows token prefixes only.
    Result<GenerationStats> generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                            const TokenCallback& on_token);

    // Processes `text` into the context without generating, so a later
    // generate() whose prompt starts with it skips that work (e.g. the system
    // prompt, prepared right after loading). Follows the same reuse rules as
    // generate(). Not reentrant (Busy); cancel() stops it between chunks.
    Status prefill(std::string_view text, bool add_bos = true);
    // Forgets the context (the next call starts from scratch).
    void reset_context();
    // Saves the context (e.g. a prefilled system prompt) to `path`: KV pages,
    // recurrent states, and the tokens and text they encode. Written to a
    // temporary file then renamed, so `path` is either the old file or the new
    // one. Unsupported with a draft model or a sliding window. Busy while a
    // generation runs.
    Status save_state(const std::string& path);
    // Replaces the context with one save_state() wrote for this model file
    // loaded with the same cache settings, so the next prompt that continues
    // its text starts at once instead of being processed again (on a 35B MoE a
    // 900-token system prompt takes about a minute). Returns the tokens
    // restored. InvalidArgument for another model or settings, IoError for a
    // damaged file; the context is empty after a failure.
    Result<int32_t> load_state(const std::string& path);

    // Gives memory back to the OS while the engine is idle (e.g. the app is in
    // the background): empties the MoE expert cache. The model stays loaded,
    // context included; the next generation reads its experts from storage
    // again, so its first tokens are slower until the cache refills. Returns
    // the bytes released (0 for models without streamed experts). Call it
    // between generations, from the thread that generates.
    size_t trim_memory();

    // How a MoE model with streamed experts uses memory; all zero otherwise.
    struct MemoryPlan {
        uint64_t resident_bytes = 0;      // weights kept in RAM (after any requantization)
        uint64_t expert_bytes = 0;        // routed experts read from storage
        uint64_t expert_cache_bytes = 0;  // RAM caching experts
        uint64_t recommended_bytes = 0;   // memory budget at which most expert uses would come from RAM
        int32_t requant_bits = 0;         // resident Q8_0 matrices converted at load (4 / 5); 0: none
    };
    [[nodiscard]] MemoryPlan memory_plan() const;
    // The context length in positions (EngineConfig::context_length, resolved when 0).
    [[nodiscard]] int32_t context_length() const noexcept;

    // The expert cache's hot list (MoE with streamed experts; no-ops
    // otherwise): the cached experts by use, recorded by trim_memory().
    // warm_memory() queues them behind any demand read, so the cache refills
    // in one background pass of large reads once the model is used again
    // instead of one wait per expert; returns how many were queued.
    // save_expert_profile() writes the list (a few KB) for a later process,
    // load_expert_profile() takes it back (InvalidArgument for another model).
    size_t warm_memory();
    Status save_expert_profile(const std::string& path) const;
    Status load_expert_profile(const std::string& path);

    // Thread-safe: stops an in-flight generate() at the next token boundary.
    void cancel() noexcept;
    // Thread-safe: stops an in-flight prefill() between two blocks of the
    // model (within a fraction of a second, also inside a long pass); it
    // returns Cancelled and keeps the context processed so far, so the next
    // prompt that continues the text resumes there. Never affects generate(),
    // so a scheduler can preempt background prefills without a race with the
    // foreground request that follows.
    void cancel_prefill() noexcept;

    // `add_bos` prepends BOS only for models that use one (tokenizer.ggml.add_bos_token).
    [[nodiscard]] Result<std::vector<int32_t>> tokenize(std::string_view text, bool add_bos) const;
    [[nodiscard]] std::string token_to_piece(int32_t token) const;

    [[nodiscard]] const DeviceInfo& device() const noexcept;
    [[nodiscard]] PowerManager& power() noexcept;
    // Model, backends and memory summary for logs / UIs.
    [[nodiscard]] std::string describe() const;

    // Live counters for monitoring UIs; cumulative since creation, so callers
    // sample them periodically and divide the deltas by the elapsed time.
    // Thread-safe: may be read while a generation runs on another thread.
    struct Counters {
        double accelerator_busy_ms = 0.0;  // time the GPU spent on this engine's work (0: none / not tracked)
        uint64_t storage_bytes_read = 0;   // weights streamed from storage (expert cache + block streaming)
        uint64_t tokens_generated = 0;     // decoded tokens, all generations
    };
    [[nodiscard]] Counters counters() const;
    // A scalar GGUF metadata value of the loaded model as text (strings as-is,
    // numbers in decimal, booleans as "true"/"false"), e.g. the publisher's
    // recommended "general.sampling.temp". nullopt when absent or an array.
    [[nodiscard]] std::optional<std::string> model_metadata(std::string_view key) const;

private:
    struct Impl;
    explicit Engine(std::unique_ptr<Impl> impl);
    // generate_tokens() with the busy lock held; `text`, when given, is the
    // text `prompt` encodes, remembered for the next call's reuse.
    Result<GenerationStats> generate_locked(std::span<const int32_t> prompt, const SamplingParams& params,
                                            const TokenCallback& on_token, const std::string_view* text,
                                            const ForceCallback* force = nullptr);
    std::unique_ptr<Impl> impl_;
};

}  // namespace liyab

#endif  // LIYAB_ENGINE_H
