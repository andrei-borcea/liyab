// Liyab — decoder-only transformer over memory-mapped GGUF weights (internal).
//
// The block structure is resolved per layer from the file, not hard-coded per
// model: each block has a token mixer — softmax attention (GQA, RoPE normal or
// NeoX, optional QKV bias, per-head Q/K norm, optional sigmoid output gate) or a
// Gated DeltaNet linear-attention recurrence — followed by a SwiGLU FFN. Which
// mixer a block uses, and which optional tensors it has, is decided by the
// tensors present, so hybrid models (Qwen3.5: three DeltaNet blocks per
// attention block) and plain transformers (Llama, Mistral, Qwen2/3) share one
// forward pass. The FFN is dense SwiGLU or a mixture of experts (router +
// top-k routed experts + optional gated shared expert), again per block. Architectures only contribute a small traits row (RoPE style).
// Tensor precision is per tensor, so mixed-precision files run unconverted.
#ifndef LIYAB_CORE_TRANSFORMER_H
#define LIYAB_CORE_TRANSFORMER_H

#include <algorithm>
#include <array>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/expert_store.h"
#include "liyab/backend.h"
#include "liyab/experimental/forward_hooks.h"
#include "liyab/kv_cache.h"
#include "liyab/triple_buffer_loader.h"
#include "liyab/mmap_loader.h"

namespace liyab {

class ThreadPool;

// How a mixture-of-experts router turns logits into expert probabilities.
enum class MoeGating : uint8_t { Softmax, Sigmoid };

// Token mixer of one block.
enum class MixerKind : uint8_t {
    Attention,  // softmax attention over the KV cache
    DeltaNet,   // Gated DeltaNet: causal conv + delta-rule recurrent state, no KV cache
};

struct ModelConfig {
    std::string arch;
    int32_t n_layers = 0;       // executed blocks (excludes multi-token-prediction heads)
    int32_t n_attn_layers = 0;  // blocks with MixerKind::Attention (they own the KV cache)
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
    std::vector<MixerKind> mixers;  // per block

    // Gated DeltaNet geometry (zero when every block is attention).
    int32_t ssm_conv_kernel = 0;  // causal depthwise conv width
    int32_t ssm_head_dim = 0;     // key and value head size (ssm.state_size)
    int32_t ssm_k_heads = 0;      // query/key heads (ssm.group_count)
    int32_t ssm_v_heads = 0;      // value heads = recurrent states per block (ssm.time_step_rank)

    // Mixture of experts (zero when every FFN is dense).
    int32_t n_expert = 0;        // routed experts per MoE block
    int32_t n_expert_used = 0;   // experts selected per token (top-k)
    int32_t n_ff_expert = 0;     // hidden size of one routed expert
    int32_t n_ff_shared = 0;     // hidden size of the shared expert (0: none)
    MoeGating moe_gating = MoeGating::Softmax;
    bool moe_norm_weights = true;  // renormalize the top-k weights to sum to 1
    float moe_weights_scale = 1.0f;

    // True when some block keeps a recurrent state: such models cannot roll
    // back to an arbitrary earlier position (only to 0 or the current one).
    [[nodiscard]] bool hybrid() const noexcept { return ssm_v_heads > 0; }
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
    // MoE models: RAM for routed experts streamed from storage (ExpertStore).
    // -1: automatic (stream when the model does not fit in RAM, budget = free
    // RAM minus the resident weights and a safety margin), 0: never (experts
    // are read through the mapping), > 0: always, with this budget.
    int64_t expert_cache_bytes = -1;
};

class Transformer {
public:
    static Result<std::unique_ptr<Transformer>> load(std::unique_ptr<MmapLoader> file, const TransformerOptions& options);

    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] const MmapLoader& file() const noexcept { return *file_; }
    [[nodiscard]] MmapLoader& mutable_file() noexcept { return *file_; }
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
    // Attention output projection of block `layer` (used by the head pruner);
    // nullptr for blocks whose mixer is not attention.
    [[nodiscard]] const TensorView* attn_output(int32_t layer) const {
        return layers_[static_cast<size_t>(layer)].w[kO];
    }
    // FFN projections {gate, up, down} of block `layer` (experimental sparsification).
    [[nodiscard]] std::array<const TensorView*, 3> ffn_weights(int32_t layer) const {
        const Layer& L = layers_[static_cast<size_t>(layer)];
        return {L.w[kGate], L.w[kUp], L.w[kDown]};
    }
    // Routed-expert streaming (MoE models larger than RAM); nullptr when the
    // experts are read in place from the mapping.
    [[nodiscard]] const ExpertStore* expert_store() const noexcept { return expert_store_.get(); }
    // Bytes held by the recurrent (DeltaNet) states; 0 for plain transformers.
    [[nodiscard]] size_t recurrent_state_bytes() const noexcept;
    [[nodiscard]] int32_t max_batch() const noexcept { return max_batch_; }

    // Discards cached positions >= n (speculative-decoding rollback). Hybrid
    // models only accept n == 0 or n == n_past(): a recurrent state cannot be
    // rewound (Unsupported otherwise).
    Status truncate(int32_t n);
    // Declares positions [0, n) cached after their KV pages were attached
    // externally (KvCache::attach_external_pages). Requires an empty context.
    // Unsupported for hybrid models (KV pages do not carry recurrent states).
    Status adopt_cached_prefix(int32_t n);
    [[nodiscard]] KvCache& mutable_kv_cache() noexcept { return *kv_; }
    void reset() noexcept;

    // Streams transformer blocks through `source` instead of reading the
    // mapping in place. Block l is item `items[l]` of the source (laid out as
    // MmapLoader::layer_range(l)), or read in place when items[l] < 0; an
    // empty `items` streams every block (item i = block i). nullptr restores
    // in-place reads. `source` must outlive its use here.
    void set_layer_source(TripleBufferLoader* source, std::vector<int32_t> items = {}) {
        layer_source_ = source;
        layer_items_ = std::move(items);
    }

private:
    // Matrices of one block, by role; unused roles stay nullptr.
    enum WeightRole : size_t {
        kQ, kK, kV, kO,              // attention (kQ yields [q | gate] per head when gated)
        kQkv, kZ, kAlpha, kBeta, kSsmOut,  // DeltaNet: conv input, output gate, decay, write strength, output
        kGate, kUp, kDown,           // SwiGLU FFN
        kRouter, kGateExps, kUpExps, kDownExps,  // MoE: router [n_embd x n_expert], experts stacked on dim 2
        kGateShared, kUpShared, kDownShared,     // MoE: shared expert
        kWeightRoles
    };
    struct Layer {
        MixerKind mixer = MixerKind::Attention;
        int32_t kv_slot = -1;     // layer index inside the KV cache (attention)
        int32_t state_slot = -1;  // index into states_ (DeltaNet)
        bool attn_gate = false;   // attention output multiplied by sigmoid(gate)
        std::array<const TensorView*, kWeightRoles> w{};
        std::vector<float> attn_norm, ffn_norm, q_norm, k_norm, bq, bk, bv;
        std::vector<float> conv1d;   // [channel][kernel] taps
        std::vector<float> ssm_a;    // per value head: -exp(A_log)
        std::vector<float> ssm_dt;   // per value head: decay bias
        std::vector<float> ssm_norm; // gated RMSNorm weight, per head dim
        bool moe = false;                 // FFN is a mixture of experts
        std::vector<float> shared_gate;   // optional: sigmoid(shared_gate . x) scales the shared expert
        std::vector<float> expert_bias;   // optional: added to probabilities for selection only
    };
    // Recurrent memory of one DeltaNet block.
    struct RecurrentState {
        std::vector<float> conv;  // last (kernel - 1) inputs per channel, oldest first: [channel][kernel - 1]
        std::vector<float> ssm;   // per value head a [head_dim (value) x head_dim (key)] matrix
    };

    // Weight views of one block, rebased into a triple-buffer slot when a
    // layer source is attached. Releases the slot on destruction.
    class BlockWeights;

    Transformer() = default;
    Status bind_weights();
    void rope(float* vec, int32_t n_heads, int32_t pos) const;
    // Mixers: read the normalized input from xb_, leave the block output (before
    // the residual add) in xb_.
    Status attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool,
                           const uint8_t* head_mask);
    Status delta_net_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool);
    // FFNs: read the normalized input from xb_, leave the output in xb_.
    Status dense_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, FfnMatmulHook* hook);
    Status moe_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route);
    // Expert prefetch: applies block `layer`'s router to the current hidden
    // states (x_) and queues the top-k experts it picks in the ExpertStore.
    void predict_experts(int32_t layer, int32_t n, const Route& route);
    Status attach_expert_store(int64_t budget_option);
    void attention(int32_t kv_slot, int32_t n, ThreadPool& pool, const uint8_t* head_mask);
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
    std::vector<int32_t> layer_items_;

    const TensorView* token_embd_ = nullptr;
    const TensorView* output_ = nullptr;
    std::vector<float> output_norm_;
    std::vector<Layer> layers_;
    std::vector<RecurrentState> states_;
    std::unique_ptr<ExpertStore> expert_store_;
    int64_t decode_steps_ = 0;  // expert cache aging clock
    std::vector<float> inv_freq_;  // per rotary pair, includes rope_freqs factors

    // Scratch, sized for the current batch.
    std::vector<float> x_, xb_, q_, k_, v_, att_, hb_, hb2_, logits_;
    std::vector<float> qg_, gate_;                     // gated attention: raw [q | gate] rows, gates
    std::vector<float> mix_, z_, alpha_, beta_, dn_;  // DeltaNet projections and output
    std::vector<float> router_, moe_out_, ein_, eout_, eh_, eh2_;  // MoE scratch
    std::vector<float> pred_in_, pred_logits_;                       // expert prediction scratch
    std::vector<float> x_block_in_;  // residual entering the block (FFN-skip hook only)
};

}  // namespace liyab

#endif  // LIYAB_CORE_TRANSFORMER_H
