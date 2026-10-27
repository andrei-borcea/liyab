// liyab-compat — load-time model adaptation and execution planning (internal).
//
// This layer owns format/architecture facts and planning. The core executes
// the resulting plan; it does not maintain a parallel model registry.
// Planning performs no downloads and never rewrites the source GGUF.
//
// A model is described in two layers, both resolved once at load, so the
// forward pass never looks anything up by name:
//
//  * ArchSpec: the facts about an architecture that its tensors cannot tell
//    (how RoPE pairs a head's dimensions, which tensors hold the norms before
//    the FFN and after the mixer and the FFN, the FFN's activation, where Q/K
//    norms sit relative to RoPE, whether embeddings are scaled, which blocks
//    attend through a sliding window, which tensor holds the final norm, how
//    DeltaNet value heads share key heads, a default the metadata may omit). One
//    row per architecture in architecture.cpp: a model whose blocks are built
//    from parts the engine already runs needs only its row.
//  * ModelConfig and one BlockPlan per block: the hyperparameters from the
//    metadata, and each block's structure from the tensors present (which
//    token mixer: attention, Gated DeltaNet, LFM2's short convolution or
//    Kimi Delta Attention; fused or separate projections, a gated attention
//    output,
//    a windowed attention, a dense or mixture-of-experts FFN, a shared
//    expert). Optional per-element tensors (QKV biases, the attention output
//    bias, attention sinks, Q/K norms, the norms after the mixer and the FFN,
//    the shared expert's gate, the router's bias and selection bias, the
//    experts' biases, RoPE frequency factors) are bound where present and
//    need no plan entry.
//
// describe_model() is a pure function of the file: it reads metadata and
// tensor names, never weights, so a model can be described (and refused with
// a precise reason) before anything is loaded.
#ifndef LIYAB_COMPAT_ARCHITECTURE_H
#define LIYAB_COMPAT_ARCHITECTURE_H

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "liyab/mmap_loader.h"
#include "liyab/types.h"
#include "core/compression.h"

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

// Activation of a gated FFN's gate projection, down(act(gate) * up), or of
// an ungated FFN's up projection, down(act(up)) (FfnKind::Ungated).
enum class FfnActivation : uint8_t {
    SiLU,         // SwiGLU: Llama, Qwen, Phi, Hunyuan
    GeluTanh,     // GeGLU, with the tanh approximation of GELU: Gemma; GPT-2, BLOOM, Phi-2 (ungated)
    Gelu,         // exact GELU, x Phi(x) with erf: GPT-NeoX, Falcon, MPT (ungated)
    Relu,         // max(x, 0)
    ReluSquared,  // max(x, 0)^2: Nemotron, AFM (ungated)
    SwigluOai,  // clamped gate * sigmoid(1.702 gate) * (clamp(up, -7, 7) + 1)
    SiLUInputClamp, // SiLU(min(gate, limit)) * clamp(up, -limit, limit), DeepSeek-V4
    XIelu,        // xIELU with per-block scalars (ModelConfig::xielu): Apertus (ungated)
};

// How every norm of a model normalizes its input: the norms before each
// block's mixer and FFN, the final norm, per-head Q/K norms and the
// embedding norm. Each multiplies by its weight and adds its bias where the
// tensor is present.
enum class NormKind : uint8_t {
    Rms,    // x / sqrt(mean(x^2) + eps) * w
    Layer,  // (x - mean(x)) / sqrt(var(x) + eps) * w + b; w and b optional (OLMo: neither)
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
    // the file has no <arch>.expert_weights_norm. With softmax gating, true
    // is also a softmax over the selected top-k logits alone (gpt-oss): the
    // softmax over every expert renormalized over the top k is the same
    // function. An architecture whose models are all dense says false.
    bool moe_norm_weights;
    QkNormOrder qk_norm = QkNormOrder::BeforeRope;
    FfnActivation ffn_activation = FfnActivation::SiLU;
    // Suffixes of the norms applied to the mixer's and the FFN's outputs
    // before their residual adds ("sandwich" norms: Gemma's
    // post_attention_norm and post_ffw_norm). Bound where a block has them;
    // empty: the architecture has none. Both <suffix>.weight and a bare
    // <suffix> vector are accepted (the published converter writes both).
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
    // The norms' formula. LayerNorm architectures read their epsilon from
    // <arch>.attention.layer_norm_epsilon. Each LayerNorm independently
    // binds its optional weight and bias (OLMo's non-parametric norms have
    // neither); RMSNorm requires a weight.
    NormKind norm = NormKind::Rms;
    // Every block is parallel (BlockPlan::parallel) when the file has no
    // <arch>.use_parallel_residual: the FFN reads the block input normed
    // (by the block's ffn_norm when the file has it, else the mixer's own
    // normed input) and the residual adds the mixer's and the FFN's outputs.
    bool parallel_residual = false;
    // Queries and keys are rotated (RoPE). false: no block rotates
    // (ModelConfig::rope_dim 0); positions come from learned absolute
    // embeddings (position_embd.weight, added to the token embeddings) or
    // ALiBi (alibi_bias).
    bool rotary = true;
    // > 0: ALiBi with this maximum bias when the file has no
    // <arch>.attention.max_alibi_bias (the file's value wins; 0 there: none).
    float alibi_bias = 0.0f;
    // The reference configuration's architecture-wide window when the
    // published converter omits attention.sliding_window (StarCoder2).
    // Explicit metadata, including 0 for full attention, always wins.
    int32_t sliding_window_default = 0;
    // SentencePiece/Metaspace: prepend a dummy space after a control token,
    // as well as at the start of text. false: prepend_scheme "first"
    // (Nemotron); GGUF does not record this distinction. Unused for BPE.
    bool spm_prefix_after_control = true;
    // SentencePiece's BOS policy when tokenizer.ggml.add_bos_token is absent.
    // Explicit metadata wins. Unused for BPE, whose absent-key default is false.
    bool spm_add_bos_default = true;
    // Absent add_space_prefix metadata: SentencePiece's dummy-space policy.
    bool spm_add_space_prefix_default = true;
    // SHA-256 of a registered SentencePiece character map when GGUF omits
    // precompiled_charsmap. Empty: identity. Explicit map metadata wins.
    std::string_view spm_charsmap = "";
    // Exact registered user-defined symbols when a converter loses their
    // token types along with the omitted character map; empty, or explicit
    // character-map metadata: the GGUF's types fully describe them.
    std::string_view spm_user_defined = "";
    // Norm before the mixer. A parallel block with this tensor absent uses
    // ffn_norm as its shared input norm; two present tensors keep their own
    // roles (Falcon stores ln_attn as attn_norm_2 and ln_mlp as attn_norm).
    std::string_view attn_norm = "attn_norm";
    // Blocks without positional rotation (NoPE: their queries and keys are
    // not rotated), for files that do not list them in
    // <arch>.attention.rope_pattern (one bool per block, true = rotated, which
    // overrides both rules). nope_period N > 0: blocks N-1, 2N-1, ... (SmolLM3:
    // every 4th). nope_global: when the file sets an attention window, the
    // blocks attending globally (EXAONE 4: the windowed ones carry the
    // positions, the global ones see every position unrotated).
    int32_t nope_period = 0;
    bool nope_global = false;
    // Windowed blocks rotate with rope.freq_base but unscaled: no YaRN, linear
    // scaling or frequency factors, which only the global blocks take (OLMo 3,
    // whose converter writes the global blocks' YaRN).
    bool swa_rope_unscaled = false;
    // <arch>.logit_scale divides the output logits (Granite's logits_scaling,
    // MiniCPM's n_embd / dim_model_base) instead of multiplying them.
    bool logit_scale_divides = false;
    // <arch>.rope.scaling.yarn_log_multiplier holds the reference's
    // mscale_all_dim, as the converter writes it for every architecture but
    // DeepSeek's (0.1 x mscale_all_dim, scaling q·k): YaRN's attention factor
    // is then mscale(1) / mscale(mscale_all_dim), mscale(m) = 0.1 m ln(factor)
    // + 1, on the rotation's magnitude (Ministral 3).
    bool yarn_mscale_all_dim = false;
    // YaRN correction-range rounding when the GGUF omits yarn_truncate.
    // gpt-oss uses fractional endpoints; explicit file metadata overrides it.
    bool yarn_truncate = true;
    // Mamba-1 blocks whose file sets <arch>.ssm.dt_b_c_rms (Falcon-Mamba):
    // the epsilon of the weightless RMS norms of dt, B and C, which the
    // converter does not write (the reference's mixer_rms_eps; Falcon-Mamba's
    // configuration keeps its default). Norms of dt, B and C with weights of
    // their own (ssm_dt_norm, ssm_b_norm, ssm_c_norm) use the file's
    // layer_norm_rms_epsilon instead.
    float ssm_dt_b_c_eps = 1e-6f;
    // Mamba-2's gated RMS norm: true spans each B/C group separately;
    // false spans the complete inner stream (transformers Mamba-2/Bamba).
    // The converter reshapes both forms identically, so this is a reference
    // implementation fact rather than something the tensor shape tells us.
    bool ssm_norm_grouped = true;
    // Name of the norm applied to the last block's output before the LM head
    // (<output_norm>.weight). LFM2's reference calls its final norm
    // embedding_norm, which the converter stores as token_embd_norm: there
    // it is the final norm, not a norm of the embeddings.
    std::string_view output_norm = "output_norm";
    // Gated DeltaNet: value head h reads key head h / (value heads / key
    // heads): consecutive value heads share a key head, the reference's
    // order, which the converter keeps (Qwen3-Next). false: key head
    // h % key heads, the tiled order the Qwen3.5 converter reorders the value
    // heads into.
    bool ssm_grouped_heads = false;
    // false: attention is bidirectional (EmbeddingGemma): a query sees every
    // position of its pass, later ones included, and a windowed block the
    // positions at most half its window away on either side.
    // <arch>.attention.causal, when present, overrides it.
    bool causal = true;
    // q·k is not scaled when the file states no <arch>.attention.scale: the
    // weights of the Q norm carry the scale (Gemma 3n, Gemma 4).
    bool unscaled_scores = false;
    // Values are RMS-normalized per head, without a weight, before they are
    // cached (Gemma 3n, Gemma 4: a norm the tensors cannot show).
    bool value_norm = false;
    // Pairing of DSA index queries/keys (independent of the main MLA RoPE).
    RopeLayout indexer_rope = RopeLayout::Halves;
    // Query/KV latent RMSNorm epsilon; 0: the file's block norm epsilon.
    float latent_norm_eps = 0.0f;
    // Shared K=V attention: unweighted query-head RMSNorm, trailing partial
    // RoPE (unit magnitude, including YaRN) and its inverse on the output
    // before grouped low-rank projection.
    bool shared_kv = false;

};

// Every architecture Liyab runs, in registry order.
std::span<const ArchSpec> architectures() noexcept;
// The row for `name`, or nullptr when the architecture is not supported.
const ArchSpec* find_architecture(std::string_view name) noexcept;

// GGUF name of a block tensor: "blk.<block>.<suffix>".
std::string block_tensor(int32_t block, std::string_view suffix);
// Resolves a norm vector's name: prefer <suffix>.weight, else the bare
// suffix when present. A missing vector returns the conventional .weight
// name, so required bindings retain their precise missing-tensor error.
std::string block_norm_tensor(const MmapLoader& file, int32_t block, std::string_view suffix);

// How a mixture-of-experts router turns logits into expert probabilities.
enum class MoeGating : uint8_t { Softmax, Sigmoid, SqrtSoftplus };

// Token mixer of one block.
enum class MixerKind : uint8_t {
    Attention,  // softmax attention over the KV cache
    DeltaNet,   // Gated DeltaNet: causal conv + delta-rule recurrent state, no KV cache
    None,       // no mixer branch (zero query and KV heads in per-block metadata)
    Linear,     // one normalized residual projection, no attention or KV rows
    Mamba,      // Mamba-1 selective scan: causal conv + a diagonal state per channel, no KV cache
    Mamba2,     // Mamba-2 (SSD): causal conv + a scalar-decay state per head, no KV cache
    // LFM2's gated short convolution: in_proj to [B | C | x], a causal
    // depthwise convolution of B * x over the last ModelConfig::
    // shortconv_l_cache positions, C * conv, out_proj. Its state is the
    // convolution's inputs at the last l_cache - 1 positions.
    ShortConv,
    // Kimi Delta Attention (Kimi-Linear): the gated delta rule with a decay
    // per key channel (not one per head), causal convolutions of q, k and v,
    // and the output gated through a per-head RMSNorm. Its state is the
    // three convolutions' last inputs and one key x value matrix per head.
    Kda,
};

// Whether a block with this mixer keeps a recurrent state (every mixer but
// softmax attention and none).
[[nodiscard]] constexpr bool keeps_state(MixerKind m) noexcept {
    return m == MixerKind::DeltaNet || m == MixerKind::Mamba || m == MixerKind::Mamba2 || m == MixerKind::ShortConv || m == MixerKind::Kda;
}

// Feed-forward network of one block.
enum class FfnKind : uint8_t {
    Dense,    // gated: down(act(gate) * up), act = ArchSpec::ffn_activation
    Moe,      // router + top-k routed gated experts
    Ungated,  // down(act(up x + b)) + b: ffn_up and ffn_down only (GPT-2, Falcon, Nemotron)
    None,     // no FFN branch (zero per-block feed_forward_length)
};

// The structure of one block, from the tensors present.
struct BlockPlan {
    // A dense FFN and routed experts in parallel, with independent input
    // and output norms, before the block's outer post-FFN norm.
    bool parallel_ffn = false;
    bool dense_fused_gate_up = false;
    // Learned ffn_gate_inp.scale: an independent unweighted residual RMS
    // norm, channel scale and hidden_size^-0.5 before the router projection.
    bool router_input_rms = false;
    bool hash_routing = false;  // expert IDs come from original token IDs, not learned top-k

    MixerKind mixer = MixerKind::Attention;
    // Attention: multi-head latent attention (attn_kv_a_mqa present, see
    // ModelConfig::mla). Keys and values come from one compressed latent per
    // position, which is what the KV cache stores.
    bool latent = false;
    // Older MLA files keep per-head [key | value] expansion rows in one
    // attn_kv_b matrix. Binding normalizes it to the split execution layout.
    bool latent_combined = false;
    // Latent attention restricted by a lightning indexer (indexer.attn_k).
    bool indexed = false;
    // No own projections: reuse the preceding full indexer's selection.
    bool index_reuse = false;
    // Attention: attn_q projects [q | gate] per head (twice as tall), or
    // attn_gate projects the gate separately. The attention output is
    // multiplied by sigmoid(gate).
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
    // ffn_gate (Phi-3). Moe: ffn_gate_up_exps holds each expert's [gate | up]
    // instead of ffn_gate_exps and ffn_up_exps.
    bool fused_gate_up = false;
    // Moe: an always-active shared expert next to the routed ones.
    bool shared_expert = false;
    CompressionPlan compression;  // stride 0: no long-range window compressor
    // Attention: the block computes queries only and attends over the keys
    // and values that block kv_source (an earlier attention block of the same
    // kind, windowed or not) cached in the same pass: KV sharing, the last
    // <arch>.attention.shared_kv_layers blocks of Gemma 3n / Gemma 4. -1:
    // the block keeps keys and values of its own.
    int32_t kv_source = -1;
    // Attention: no attn_v; each head's values are its keys' projection,
    // taken before the key norm and RoPE (Gemma 4's attention_k_eq_v).
    bool values_from_keys = false;
    // Laurel (laurel_l present, Gemma 3n): a low-rank branch of the
    // normalized block input, x_n + norm(laurel_r · laurel_l · x_n), joins
    // the mixer's residual add: x = (x + mixer + laurel) / sqrt(2).
    bool laurel = false;
    // Dense FFN: the gate's Gaussian top-k (Gemma 3n's activation sparsity,
    // <arch>.activation_sparsity_scale finite for the block): before its
    // activation, each token's gate projection g becomes
    // relu(g - (mean(g) + gate_sparsity * std(g))), so only the values above
    // that cutoff (the top 5% of a normal distribution at 1.645) remain.
    bool sparse_gate = false;
    float gate_sparsity = 0.0f;
    // The mixer and the FFN both read the block input (ArchSpec::
    // parallel_residual, <arch>.use_parallel_residual): the FFN through the
    // block's own FFN norm when bound (GPT-NeoX's post_attention_layernorm,
    // Falcon-40B's ln_mlp), else through the mixer's normed input (Phi-2,
    // Falcon-7B, Command-R); the residual adds both outputs.
    bool parallel = false;
    // Optional per-block post-SiLU gate / symmetric up clamps from GGUF.
    // Zero disables the clamp. Shared applies to dense FFNs as well.
    float expert_swiglu_clamp = 0.0f;
    float shared_swiglu_clamp = 0.0f;

    // Attention: queries and keys are rotated by RoPE; false: the block has no
    // positional encoding of its own (NoPE, ArchSpec::nope_period /
    // nope_global or the file's attention.rope_pattern).
    bool rope = true;
    // A norm before the mixer (attn_norm) and before the FFN (the
    // architecture's ffn_norm), from the tensors present. false: the mixer or
    // FFN reads the residual stream as it is, and the block must then have
    // the norm after it (post-only norms: OLMo 2, EXAONE 4).
    bool norm_before_mixer = true;
    bool norm_before_ffn = true;
    // Attention: attn_q_norm / attn_k_norm are one RMSNorm over the whole
    // projection (n_head x head_dim, n_head_kv x head_dim: OLMo 2) instead of
    // one per head (head_dim, shared by every head: Qwen3).
    bool qk_norm_whole = false;
    // This block's geometry, from scalar or per-block metadata. ModelConfig's
    // dimensions are the maxima used for scratch buffers and the fixed KV
    // image layout; an attention block uses only its first n_head_kv heads.
    int32_t n_head = 0;
    int32_t n_head_kv = 0;
    int32_t n_ff = 0;
    // Logical projection dimensions. ModelConfig keeps the maxima for scratch
    // and cache images; padded cache channels never enter this block's math.
    int32_t head_dim = 0;
    int32_t value_dim = 0;
    int32_t rope_dim = 0;
    float attn_scale = 0.0f;
};

// xIELU's scalars of one block (Apertus), as the reference computes them
// from its parameters: x > 0: alpha_p x^2 + beta x; else
// (expm1(min(x, eps)) - x) alpha_n + beta x, with alpha_p = softplus(p) and
// alpha_n = beta + softplus(n) of the file's xielu.alpha_p / xielu.alpha_n.
struct XIeluScalars {
    float alpha_p = 0.0f;
    float alpha_n = 0.0f;
    float beta = 0.0f;
    float eps = 0.0f;

};

// YaRN context extension (<arch>.rope.scaling.type "yarn"), as the GGUF
// states it. Each rotary pair's frequency is blended between its original
// value and that value divided by `factor`: pairs that turn more than
// beta_fast times over the original context keep their frequency
// (extrapolation), pairs that turn fewer than beta_slow times are divided
// by the factor (interpolation), and the pairs between follow a linear
// ramp. Transformer::load builds the table; the attention factor that comes
// with it is folded into ModelConfig::rope_magnitude and attn_scale.
struct YarnRope {
    float factor = 0.0f;       // rope.scaling.factor; > 0: YaRN is on
    int32_t ctx_orig = 0;      // rope.scaling.original_context_length (default: context_length)
    float beta_fast = 32.0f;   // rope.scaling.yarn_beta_fast
    float beta_slow = 1.0f;    // rope.scaling.yarn_beta_slow
    // rope.scaling.yarn_ext_factor: the share of the ramp that
    // extrapolates (0: every pair interpolated).
    float ext_factor = 1.0f;
    // rope.scaling.yarn_truncate: the ramp's ends are rounded outwards to
    // whole pairs (the reference's default).
    bool truncate = true;
};

// Multi-head latent attention (DeepSeek-V2/V3, `deepseek2`), in its absorbed
// form. Each position's keys and values come from one latent c (kv_rank
// wide, RMS-normalized) and one RoPE key k_pe (rope_dim, shared by every
// head): head h's key is [W_kb,h c | k_pe] and its value W_vb,h c. Since
// q_h . (W_kb,h c) = (W_kb,h^T q_h) . c, each head's query is projected onto
// the latent once (attn_k_b, stored transposed per head) and attention runs
// as one KV head of [k_pe | c] (ModelConfig::head_dim = rope_dim + kv_rank)
// whose values are its latent channels; the weighted latent is then
// expanded per head (attn_v_b) before attn_output. The KV cache keeps
// rope_dim + kv_rank values per position and block instead of
// n_head * (key + value) widths: 576 instead of 5,120 on DeepSeek-V2-Lite.
struct LatentAttention {
    int32_t kv_rank = 0;   // <arch>.attention.kv_lora_rank: latent width; 0: no latent attention
    int32_t q_rank = 0;    // <arch>.attention.q_lora_rank: queries through attn_q_a, a norm and attn_q_b; 0: attn_q
    int32_t nope_dim = 0;  // per-head query/key channels without RoPE: key_length_mla - rope_dim
    int32_t v_dim = 0;     // per-head value width once expanded: <arch>.attention.value_length_mla
};

// Mamba (selective state space) geometry, from the ssm.* keys as the
// converter writes them for Mamba-1 (`mamba`) and Mamba-2 (`mamba2`). Each
// block's scanned stream x has d_inner channels; every channel keeps a state
// of d_state values. Mamba-1 gives each (channel, state) pair its own decay
// exp(dt A) and projects dt through a rank-dt_rank bottleneck; Mamba-2 splits
// the channels into n_heads heads of d_inner / n_heads, each with one scalar
// decay per token, and n_groups groups of heads share their B and C.
struct MambaGeometry {
    int32_t d_inner = 0;      // ssm.inner_size: channels of x (and of the gate z); 0: no Mamba block
    int32_t d_state = 0;      // ssm.state_size: state values per channel (B and C width)
    int32_t conv_kernel = 0;  // ssm.conv_kernel: taps of the causal depthwise convolution
    int32_t dt_rank = 0;      // Mamba-1: ssm.time_step_rank, the rank of dt's projection
    int32_t n_heads = 0;      // Mamba-2: ssm.time_step_rank, heads of d_inner / n_heads channels
    int32_t n_groups = 0;     // Mamba-2: ssm.group_count, groups of heads sharing B and C
    // Mamba-1: ssm.dt_b_c_rms, weightless RMS norms of dt, B and C before the
    // scan (Falcon-Mamba; epsilon ArchSpec::ssm_dt_b_c_eps).
    bool dt_b_c_rms = false;
};

struct ModelConfig {
    // Metadata beyond the legacy first-MiB fingerprint affects execution or
    // tokenizer identity; hash the complete header for states/calibration.
    bool full_header_identity = false;

    const ArchSpec* spec = nullptr;  // registry row (static lifetime)
    std::string arch;                // spec->name
    int32_t n_layers = 0;       // executed blocks (excludes multi-token-prediction heads)
    int32_t n_attn_layers = 0;  // blocks with MixerKind::Attention
    // Attention blocks keeping keys and values of their own: the KV cache's
    // layers (n_attn_layers less the blocks sharing another's, kv_source).
    int32_t n_kv_layers = 0;
    int32_t n_embd = 0;
    int32_t n_ff = 0;       // maximum block FFN width (individual widths in blocks)
    int32_t n_head = 0;     // maximum query heads, at least one for scratch sizing
    int32_t n_head_kv = 0;  // maximum KV heads, at least one for the fixed cache layout
    int32_t head_dim = 0;    // maximum key/query stride; logical widths live in BlockPlan
    int32_t value_dim = 0;   // maximum value width; mla.kv_rank with latent attention
    int32_t rope_dim = 0;  // rotated channels per head; 0: no rotation (ArchSpec::rotary false)
    int32_t n_vocab = 0;
    int32_t n_ctx_train = 0;
    // Epsilon of every norm: attention.layer_norm_rms_epsilon, or
    // attention.layer_norm_epsilon with ArchSpec::norm Layer.
    float rms_eps = 1e-5f;
    float rope_base = 10000.0f;
    float rope_scale = 1.0f;  // linear position scaling
    // LongRoPE (Phi-3): rope_factors_short.weight and rope_factors_long.weight
    // divide the frequencies, the long ones when the context is longer than
    // rope_ctx_orig (the Transformer picks once, from its context length).
    int32_t rope_ctx_orig = 0;
    // Multiplies the rotated dimensions of queries and keys
    // (<arch>.rope.scaling.attn_factor: LongRoPE's attention factor; 1 when
    // absent), times YaRN's attention factor with YaRN:
    // rope.scaling.yarn_attn_factor when the file has it, else
    // 1 + 0.1 ln(factor); 1 when the file has a yarn_log_multiplier, whose
    // factor goes into attn_scale instead (DeepSeek's convention).
    float rope_magnitude = 1.0f;
    YarnRope yarn;  // factor 0: no YaRN
    // Multiplies q·k before the softmax: <arch>.attention.scale when the
    // file has it, else 1/sqrt(head_dim); with YaRN and a
    // rope.scaling.yarn_log_multiplier m, times (1 + m ln(factor))^2.
    float attn_scale = 0.0f;
    // Logit soft-capping, cap * tanh(x / cap) (0: none): of the scaled
    // attention scores before the softmax (<arch>.attn_logit_softcapping)
    // and of the output logits (<arch>.final_logit_softcapping). Gemma 2.
    float attn_softcap = 0.0f;
    float final_softcap = 0.0f;
    LatentAttention mla;  // kv_rank 0: plain attention
    // DSA geometry from attention.indexer.* metadata; zero without an indexer.
    int32_t index_heads = 0;
    int32_t index_dim = 0;
    int32_t index_topk = 0;
    // Shared K=V attention's low-rank query and grouped low-rank output.
    int32_t shared_q_rank = 0;
    int32_t output_groups = 0;
    int32_t output_rank = 0;
    int32_t compressed_layers = 0;
    float compress_rope_base = 0.0f;
    // Manifold-constrained residual streams (tensors hc_*_fn/base/scale).
    int32_t hc_count = 0;
    int32_t hc_iterations = 0;
    float hc_eps = 0.0f;
    // Blocks with BlockPlan::windowed see this many positions, the query's
    // own included (0: no windowed block); rope_base_swa > 0 gives them a
    // RoPE table of their own (ArchSpec::rope_base_swa).
    int32_t sliding_window = 0;
    float rope_base_swa = 0.0f;
    // Multiplies the token embeddings: <arch>.embedding_scale when the file
    // has it (μP: Granite's embedding_multiplier, MiniCPM's scale_emb), else
    // sqrt(n_embd) with ArchSpec::scale_embeddings, else 1.
    float embd_scale = 1.0f;
    // Multiplies the mixer's and the FFN's outputs before each residual add
    // (<arch>.residual_scale: Granite's residual_multiplier, MiniCPM's
    // scale_depth / sqrt(n_layers)); 1: none.
    float residual_scale = 1.0f;
    // Multiplies the output logits (after the output projection, before any
    // soft cap): <arch>.logit_scale, or its inverse with
    // ArchSpec::logit_scale_divides; 1: none.
    float logit_scale = 1.0f;
    // Attention temperature by position (Ministral 3's llama_4_scaling): each
    // query is multiplied by 1 + attn_temp_scale ln(1 + floor(pos /
    // attn_temp_length)) after RoPE (<arch>.attention.temperature_scale, the
    // length <arch>.attention.temperature_length or YaRN's original context).
    // 0: none; the factor is exactly 1 below attn_temp_length.
    float attn_temp_scale = 0.0f;
    int32_t attn_temp_length = 0;
    std::vector<BlockPlan> blocks;  // per executed block

    // Gated DeltaNet geometry (zero when every block is attention).
    int32_t ssm_conv_kernel = 0;  // causal depthwise conv width (DeltaNet and Kda blocks)
    int32_t ssm_head_dim = 0;     // key and value head size (ssm.state_size)
    int32_t ssm_k_heads = 0;      // query/key heads (ssm.group_count)
    int32_t ssm_v_heads = 0;      // value heads = recurrent states per block (ssm.time_step_rank)
    MambaGeometry mamba;  // d_inner 0: no Mamba block
    // LFM2 short convolution (0 without ShortConv blocks): taps of its causal
    // depthwise convolution (<arch>.shortconv.l_cache).
    int32_t shortconv_l_cache = 0;
    // Kimi Delta Attention (0 without Kda blocks): the key and value head
    // size (<arch>.kda.head_dim) and the heads (a Kda block's attn_q rows over
    // it). Its convolutions are ssm_conv_kernel wide.
    int32_t kda_head_dim = 0;
    int32_t kda_heads = 0;
    // Blocks whose mixer keeps a recurrent state (DeltaNet, ShortConv, Kda).
    int32_t n_recurrent_layers = 0;

    // Mixture of experts (zero when every FFN is dense).
    int32_t n_expert = 0;        // routed experts per MoE block
    int32_t n_expert_used = 0;   // experts selected per token (top-k)
    // Hidden size of one routed expert: <arch>.expert_feed_forward_length,
    // else the experts' tensors (the llama converter writes Mixtral without
    // the key).
    int32_t n_ff_expert = 0;
    int32_t n_ff_shared = 0;     // hidden size of the shared expert (0: from its tensors)
    MoeGating moe_gating = MoeGating::Softmax;
    bool moe_norm_additive = false;  // divide by sum + 1e-20 instead of flooring sum
    bool moe_norm_weights = true;  // renormalize the top-k weights to sum to 1
    float moe_weights_scale = 1.0f;
    // Group-limited expert selection (DeepSeek-V2/V3, Ling 2.0): the experts
    // form n_expert_groups equal groups of consecutive experts, each group is
    // scored, and the top-k is taken only among the experts of the
    // n_expert_groups_used best groups (both 1: plain top-k). A group's score
    // is the sum of its two best selection scores (probability plus the
    // block's selection bias) when the block has a selection bias
    // (exp_probs_b: DeepSeek-V3's noaux_tc, Ling 2.0), else its best
    // probability (DeepSeek-V2's group_limited_greedy). Ties go to the lower
    // group, as they go to the lower expert in the top-k.
    int32_t n_expert_groups = 1;       // <arch>.expert_group_count
    int32_t n_expert_groups_used = 1;  // <arch>.expert_group_used_count

    // AltUp (Gemma 3n): the residual stream is altup_streams streams
    // (<arch>.altup.num_inputs; 0: one stream, no AltUp). The first is the
    // scaled embedding, the others its projections through altup_proj
    // rescaled to its magnitude. Each block predicts every stream from all
    // of them (coefficients from a router over the active stream), runs on
    // the active one's prediction (<arch>.altup.active_idx), and corrects
    // every stream by what the block changed in it; after the last block the
    // streams other than the first go back through altup_unembd_proj and
    // their mean is the final hidden state.
    int32_t altup_streams = 0;
    int32_t altup_active = 0;
    // Per-layer embeddings (Gemma 3n, Gemma 4; 0: none): every token gets
    // ple_dim values per block (<arch>.embedding_length_per_layer_input),
    // its row of per_layer_token_embd (scaled by sqrt(ple_dim)) plus the
    // normalized projection of its embedding through per_layer_model_proj,
    // times 1/sqrt(2). After its FFN, each block gates them with its own
    // output (inp_gate, the FFN's activation), projects them back (proj,
    // post_norm) and adds the result to the residual stream (with AltUp: to
    // every stream but the first).
    int32_t ple_dim = 0;
    // false: bidirectional attention (ArchSpec::causal, or the file's
    // <arch>.attention.causal). Such a model reads a text in one pass.
    bool causal = true;

    // > 0: queries, keys and values are clamped to [-clamp_kqv, clamp_kqv]
    // after their projections and biases (<arch>.attention.clamp_kqv: MPT,
    // DBRX, OLMo's clip_qkv).
    float clamp_kqv = 0.0f;
    // > 0: ALiBi's maximum bias (ArchSpec::alibi_bias, <arch>.attention.
    // max_alibi_bias): head h adds slope_h x (key position - query position)
    // to its scores, slopes 2^(-bias (h + 1) / n) for the n = 2^floor(log2
    // heads) first heads and 2^(-bias (2 (h - n) + 1) / 2n) for the others.
    float alibi_bias = 0.0f;
    // > 0: rows of position_embd.weight, the learned absolute position
    // embeddings added to the token embeddings (GPT-2, GPTBigCode): the
    // longest context the model can address.
    int32_t n_positions = 0;
    // FfnActivation::XIelu: one entry per executed block.
    std::vector<XIeluScalars> xielu;

    // True when some block keeps a recurrent state (keeps_state()): such
    // models cannot roll back to an arbitrary earlier position (only to 0,
    // the current one, a rollback checkpoint or a state snapshot).
    [[nodiscard]] bool hybrid() const noexcept {
        for (const BlockPlan& b : blocks) {
            if (keeps_state(b.mixer) || b.compression.stride > 0) return true;
        }
        return false;
    }
};

// Reads the model's hyperparameters and plans its blocks. Errors:
// InvalidModel when metadata is missing or inconsistent (with the key or
// tensor at fault), Unsupported for an architecture or variant the engine does
// not run (the message lists the supported architectures).
Result<ModelConfig> describe_model(const MmapLoader& file);

}  // namespace liyab

#endif  // LIYAB_COMPAT_ARCHITECTURE_H
