// Liyab — decoder-only transformer over memory-mapped GGUF weights (internal).
//
// Supports the Llama family layout (llama, mistral, qwen2, qwen3): RMSNorm,
// GQA attention with RoPE (normal or NeoX, optional QKV bias and per-head
// Q/K norm), SwiGLU FFN. Tensor precision is per tensor, so mixed-precision
// files (e.g. Q8_0 attention + Q4_0 FFN) run without conversion.
#ifndef LIYAB_CORE_TRANSFORMER_H
#define LIYAB_CORE_TRANSFORMER_H

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "liyab/backend.h"
#include "liyab/experimental/forward_hooks.h"
#include "liyab/kv_cache.h"
#include "liyab/triple_buffer_loader.h"
#include "liyab/mmap_loader.h"

namespace liyab {

class ThreadPool;

struct ModelConfig {
    std::string arch;
    int32_t n_layers = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim = 0;
    int32_t rope_dim = 0;
    int32_t n_vocab = 0;
    int32_t n_ctx_train = 0;
    float rms_eps = 1e-5f;
    float rope_base = 10000.0f;
    float rope_scale = 1.0f;  // linear position scaling
    bool rope_neox = false;
};

// Which backend runs which matmuls for one forward pass. `cpu` is the
// fallback when a backend reports Unsupported for a tensor.
struct Route {
    Backend* attention = nullptr;
    Backend* ffn = nullptr;
    Backend* cpu = nullptr;
};

struct TransformerOptions {
    int32_t context_length = 0;   // 0: min(model training context, 4096)
    int32_t sliding_window = 0;   // 0: full attention
    int32_t sink_tokens = 8;      // attention-sink anchors kept with a sliding window
    KvCacheType kv_type = KvCacheType::Q8_0;
    int32_t max_batch = 64;       // largest rollback the caller performs (speculative k + 1)
};

class Transformer {
public:
    static Result<std::unique_ptr<Transformer>> load(std::unique_ptr<MmapLoader> file, const TransformerOptions& options);

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] const MmapLoader& file() const noexcept { return *file_; }
    [[nodiscard]] const KvCache& kv_cache() const noexcept { return *kv_; }
    [[nodiscard]] int32_t context_length() const noexcept { return context_length_; }
    [[nodiscard]] int32_t n_past() const noexcept { return n_past_; }

    enum class Logits { None, Last, All };
    // Appends `tokens` at positions [n_past, n_past + n) and returns logits
    // (n_vocab floats per row): none (prefill), the last token's, or every
    // token's (speculative verification). Valid until the next forward().
    // `hooks` (optional) enables early exit (single-token Logits::Last steps
    // only) and attention-head masking.
    Result<std::span<const float>> forward(std::span<const int32_t> tokens, Logits logits, const Route& route,
                                           ThreadPool& pool, const ForwardHooks* hooks = nullptr);
    // Block after which the last forward() exited early, or -1.
    [[nodiscard]] int32_t last_exit_layer() const noexcept { return last_exit_layer_; }
    // FFN blocks skipped by ForwardHooks::ffn_skip during the last forward().
    [[nodiscard]] int32_t last_ffn_skips() const noexcept { return last_ffn_skips_; }
    // Attention output projection of block `layer` (used by the head pruner).
    [[nodiscard]] const TensorView& attn_output(int32_t layer) const { return *layers_[static_cast<size_t>(layer)].wo; }
    // FFN projections {gate, up, down} of block `layer` (experimental sparsification).
    [[nodiscard]] std::array<const TensorView*, 3> ffn_weights(int32_t layer) const {
        const Layer& L = layers_[static_cast<size_t>(layer)];
        return {L.w_gate, L.w_up, L.w_down};
    }
    [[nodiscard]] int32_t max_batch() const noexcept { return max_batch_; }

    // Discards cached positions >= n (speculative-decoding rollback).
    Status truncate(int32_t n);
    // Declares positions [0, n) cached after their KV pages were attached
    // externally (KvCache::attach_external_pages). Requires an empty context.
    Status adopt_cached_prefix(int32_t n);
    [[nodiscard]] KvCache& mutable_kv_cache() noexcept { return *kv_; }
    void reset() noexcept;

    // Streams transformer blocks through `source` (item i = block i, laid out
    // as MmapLoader::layer_range(i)) instead of reading the mapping in place.
    // nullptr restores in-place reads. `source` must outlive its use here.
    void set_layer_source(TripleBufferLoader* source) noexcept { layer_source_ = source; }

private:
    struct Layer {
        const TensorView* wq = nullptr;
        const TensorView* wk = nullptr;
        const TensorView* wv = nullptr;
        const TensorView* wo = nullptr;
        const TensorView* w_gate = nullptr;
        const TensorView* w_up = nullptr;
        const TensorView* w_down = nullptr;
        std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, bq, bk, bv;
    };

    // Weight views of one block, rebased into a triple-buffer slot when a
    // layer source is attached. Releases the slot on destruction.
    class BlockWeights;

    Transformer() = default;
    Status bind_weights();
    void rope(float* vec, int32_t n_heads, int32_t pos) const;
    void attention(int32_t layer, int32_t n, ThreadPool& pool, const uint8_t* head_mask);
    Status matmul(const Route& route, Backend* backend, const TensorView& w, const float* x, float* y, int32_t n);
    // Grouped variant (shared input); falls back to the CPU as a whole group.
    Status matmul_group(const Route& route, Backend* backend, std::span<const TensorView* const> ws, const float* x,
                        std::span<float* const> ys, int32_t n);
    // Final norm + LM head over `rows` hidden rows starting at `first_row` of x_.
    Status compute_logits(const Route& route, size_t first_row, int32_t rows);
    // Early exit: writes K/V for blocks [from, n_layers) of the single token
    // at `pos` from the current hidden state, so later tokens stay coherent.
    Status propagate_kv(const Route& route, int32_t from, int32_t pos);

    std::unique_ptr<MmapLoader> file_;
    ModelConfig config_;
    int32_t context_length_ = 0;
    int32_t max_batch_ = 0;
    int32_t n_past_ = 0;
    int32_t last_exit_layer_ = -1;
    int32_t last_ffn_skips_ = 0;
    std::unique_ptr<KvCache> kv_;
    TripleBufferLoader* layer_source_ = nullptr;

    const TensorView* token_embd_ = nullptr;
    const TensorView* output_ = nullptr;
    std::vector<float> output_norm_;
    std::vector<Layer> layers_;
    std::vector<float> inv_freq_;  // per rotary pair, includes rope_freqs factors

    // Scratch, sized for the current batch.
    std::vector<float> x_, xb_, q_, k_, v_, att_, hb_, hb2_, logits_;
    std::vector<float> x_block_in_;  // residual entering the block (FFN-skip hook only)
};

}  // namespace liyab

#endif  // LIYAB_CORE_TRANSFORMER_H
