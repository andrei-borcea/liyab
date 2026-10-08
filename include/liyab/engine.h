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
    int32_t context_length = 0;       // 0: min(model training context, 4096)
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
    // ignored when experts are not streamed. 0: off.
    int32_t requant_bits = 0;
    int32_t draft_tokens = 4;         // k for speculative decoding
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

class LIYAB_API Engine {
public:
    static Result<std::unique_ptr<Engine>> create(const EngineConfig& config);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    // Generates a continuation of `prompt` (already chat-formatted by the
    // caller). The context is reused across calls: when `prompt` continues the
    // tokens the engine already processed (previous prompt + its generated
    // reply, or a prefill()), only the new tokens are processed, which makes
    // chat turns start fast; recurrent (DeltaNet) states are reused the same
    // way. Attention-only models also keep the longest common prefix; anything
    // else starts from an empty context. Output is the same as from a fresh
    // context. Not reentrant: a concurrent call returns ErrorCode::Busy.
    Result<GenerationStats> generate(std::string_view prompt, const SamplingParams& params,
                                     const TokenCallback& on_token);
    // Same, from token ids (for vocabularies whose text encoder Liyab lacks).
    Result<GenerationStats> generate_tokens(std::span<const int32_t> prompt, const SamplingParams& params,
                                            const TokenCallback& on_token);

    // Processes `text` into the context without generating, so a later
    // generate() whose prompt starts with it skips that work (e.g. the system
    // prompt, prepared right after loading). Follows the same reuse rules as
    // generate(). Not reentrant (Busy); cancel() stops it between chunks.
    Status prefill(std::string_view text, bool add_bos = true);
    // Forgets the context (the next call starts from scratch).
    void reset_context();

    // Thread-safe: stops an in-flight generate() at the next token boundary.
    void cancel() noexcept;

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
    std::unique_ptr<Impl> impl_;
};

}  // namespace liyab

#endif  // LIYAB_ENGINE_H
