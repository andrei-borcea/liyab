// Liyab — what a model is, read from its GGUF file (internal).
//
// A model is described in two layers, both resolved once at load, so the
// forward pass never looks anything up by name:
//
//  * ArchSpec: the facts about an architecture that its tensors cannot tell
//    (how RoPE pairs a head's dimensions, which tensors hold the norms before
//    the FFN and after the mixer and the FFN, the FFN's activation, where Q/K
//    norms sit relative to RoPE, whether embeddings are scaled, which blocks
//    attend through a sliding window, a default the metadata may omit). One
//    row per architecture in architecture.cpp: a model whose blocks are built
//    from parts the engine already runs needs only its row.
//  * ModelConfig and one BlockPlan per block: the hyperparameters from the
//    metadata, and each block's structure from the tensors present (which
//    token mixer, fused or separate projections, a gated attention output,
//    a windowed attention, a dense or mixture-of-experts FFN, a shared
//    expert). Optional per-element tensors (QKV biases, Q/K norms, the norms
//    after the mixer and the FFN, the shared expert's gate, the router's
//    selection bias, RoPE frequency factors) are bound where present and
//    need no plan entry.
//
// describe_model() is a pure function of the file: it reads metadata and
// tensor names, never weights, so a model can be described (and refused with
// a precise reason) before anything is loaded.
#ifndef LIYAB_CORE_ARCHITECTURE_H
#define LIYAB_CORE_ARCHITECTURE_H

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "liyab/mmap_loader.h"
#include "liyab/types.h"

namespace liyab {

// How rotary position embedding pairs the dimensions of a head it rotates.
enum class RopeLayout : uint8_t {
    Adjacent,  // (2i, 2i + 1): Llama, Mistral (GGUF stores their Q/K rows permuted for this)
    Halves,    // (i, i + rope_dim / 2): GPT-NeoX style, the Qwen families
};

// Where a block's per-head Q/K norms (attn_q_norm / attn_k_norm) are applied.
enum class QkNormOrder : uint8_t {
    BeforeRope,  // normalize, then rotate: Qwen3, Gemma 3
    AfterRope,   // rotate, then normalize: Hunyuan
};

// Activation of a gated FFN's gate projection: down(act(gate) * up).
enum class FfnActivation : uint8_t {
    SiLU,      // SwiGLU: Llama, Qwen, Phi, Hunyuan
    GeluTanh,  // GeGLU, with the tanh approximation of GELU: Gemma
};

// The facts about one architecture that its tensors cannot tell. The fields
// after moe_norm_weights default to a plain pre-norm transformer.
struct ArchSpec {
    std::string_view name;      // general.architecture, also the prefix of its metadata keys
    RopeLayout rope;
    // Suffix of the norm applied before each block's FFN (blk.N.<ffn_norm>.weight).
    // Families name it differently, and the same name can mean another norm in
    // another family (Gemma's post_attention_norm follows the attention output).
    std::string_view ffn_norm;
    // MoE: whether the top-k router weights are renormalized to sum to 1 when
    // the file has no <arch>.expert_weights_norm.
    bool moe_norm_weights;
    QkNormOrder qk_norm = QkNormOrder::BeforeRope;
    FfnActivation ffn_activation = FfnActivation::SiLU;
    // Suffixes of the norms applied to the mixer's and the FFN's outputs
    // before their residual adds ("sandwich" norms: Gemma's
    // post_attention_norm and post_ffw_norm). Bound where a block has them;
    // empty: the architecture has none.
    std::string_view post_mixer_norm = {};
    std::string_view post_ffn_norm = {};
    // Token embeddings are multiplied by sqrt(n_embd) before the first block.
    bool scale_embeddings = false;
    // Which attention blocks see only the last <arch>.attention.sliding_window
    // positions when the file sets a window: 0 none (the key is ignored), 1
    // every block, N > 1 all but every N-th (blocks N-1, 2N-1, ... attend
    // globally). <arch>.attention.sliding_window_pattern, when present,
    // overrides it (a period, or one bool per block: true = windowed).
    int32_t swa_period = 0;
    // > 0: RoPE base of the windowed blocks when the file has no
    // <arch>.rope.freq_base_swa. They then rotate with a table of their own:
    // that base, no position scaling, no frequency factors (Gemma 3: 10000 in
    // windowed blocks, 1M, linearly scaled on the larger models, in global
    // ones). 0: windowed blocks rotate like the others.
    float rope_base_swa = 0.0f;
};

// Every architecture Liyab runs, in registry order.
std::span<const ArchSpec> architectures() noexcept;
// The row for `name`, or nullptr when the architecture is not supported.
const ArchSpec* find_architecture(std::string_view name) noexcept;

// GGUF name of a block tensor: "blk.<block>.<suffix>".
std::string block_tensor(int32_t block, std::string_view suffix);

// How a mixture-of-experts router turns logits into expert probabilities.
enum class MoeGating : uint8_t { Softmax, Sigmoid };

// Token mixer of one block.
enum class MixerKind : uint8_t {
    Attention,  // softmax attention over the KV cache
    DeltaNet,   // Gated DeltaNet: causal conv + delta-rule recurrent state, no KV cache
};

// Feed-forward network of one block.
enum class FfnKind : uint8_t {
    Dense,  // gated: down(act(gate) * up), act = ArchSpec::ffn_activation
    Moe,    // router + top-k routed gated experts
};

// The structure of one block, from the tensors present.
struct BlockPlan {
    MixerKind mixer = MixerKind::Attention;
    // Attention: attn_q projects [q | gate] per head (twice as tall) and the
    // attention output is multiplied by sigmoid(gate).
    bool attn_gate = false;
    // Attention: sees only the last ModelConfig::sliding_window positions
    // (ArchSpec::swa_period), and rotates with ModelConfig::rope_base_swa
    // when that is set.
    bool windowed = false;
    // Attention: one attn_qkv matrix projects [q | k | v] (Phi-3), instead of
    // attn_q, attn_k and attn_v.
    bool fused_qkv = false;
    FfnKind ffn = FfnKind::Dense;
    // Dense: ffn_up projects [gate | up] (twice as tall) and there is no
    // ffn_gate (Phi-3).
    bool fused_gate_up = false;
    // Moe: an always-active shared expert next to the routed ones.
    bool shared_expert = false;
};

struct ModelConfig {
    const ArchSpec* spec = nullptr;  // registry row (static lifetime)
    std::string arch;                // spec->name
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
    // LongRoPE (Phi-3): rope_factors_short.weight and rope_factors_long.weight
    // divide the frequencies, the long ones when the context is longer than
    // rope_ctx_orig (the Transformer picks once, from its context length).
    int32_t rope_ctx_orig = 0;
    // Multiplies the rotated dimensions of queries and keys
    // (<arch>.rope.scaling.attn_factor: LongRoPE's attention factor; 1 when absent).
    float rope_magnitude = 1.0f;
    // Blocks with BlockPlan::windowed see this many positions, the query's
    // own included (0: no windowed block); rope_base_swa > 0 gives them a
    // RoPE table of their own (ArchSpec::rope_base_swa).
    int32_t sliding_window = 0;
    float rope_base_swa = 0.0f;
    // Multiplies the token embeddings: sqrt(n_embd) with ArchSpec::scale_embeddings, else 1.
    float embd_scale = 1.0f;
    std::vector<BlockPlan> blocks;  // per executed block

    // Gated DeltaNet geometry (zero when every block is attention).
    int32_t ssm_conv_kernel = 0;  // causal depthwise conv width
    int32_t ssm_head_dim = 0;     // key and value head size (ssm.state_size)
    int32_t ssm_k_heads = 0;      // query/key heads (ssm.group_count)
    int32_t ssm_v_heads = 0;      // value heads = recurrent states per block (ssm.time_step_rank)

    // Mixture of experts (zero when every FFN is dense).
    int32_t n_expert = 0;        // routed experts per MoE block
    int32_t n_expert_used = 0;   // experts selected per token (top-k)
    int32_t n_ff_expert = 0;     // hidden size of one routed expert
    int32_t n_ff_shared = 0;     // hidden size of the shared expert (0: from its tensors)
    MoeGating moe_gating = MoeGating::Softmax;
    bool moe_norm_weights = true;  // renormalize the top-k weights to sum to 1
    float moe_weights_scale = 1.0f;

    // True when some block keeps a recurrent state: such models cannot roll
    // back to an arbitrary earlier position (only to 0 or the current one).
    [[nodiscard]] bool hybrid() const noexcept { return ssm_v_heads > 0; }
};

// Reads the model's hyperparameters and plans its blocks. Errors:
// InvalidModel when metadata is missing or inconsistent (with the key or
// tensor at fault), Unsupported for an architecture or variant the engine does
// not run (the message lists the supported architectures).
Result<ModelConfig> describe_model(const MmapLoader& file);

}  // namespace liyab

#endif  // LIYAB_CORE_ARCHITECTURE_H
