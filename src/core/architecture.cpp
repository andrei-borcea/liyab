#include "core/architecture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>

#include "liyab/engine.h"

namespace liyab {

namespace {

// The registry. Adding an architecture whose blocks the engine already runs
// is one row here, plus a case in tests/test_golden.cpp and a conformance run
// against its reference implementation (tools/conform_reference.py).
constexpr ArchSpec kArchitectures[] = {
    // Llama, and Mixtral, which the converter writes as llama with experts:
    // its router renormalizes the top-k weights (the file does not say so).
    {"llama", RopeLayout::Adjacent, "ffn_norm", true},
    {"mistral", RopeLayout::Adjacent, "ffn_norm", false},
    {"qwen2", RopeLayout::Halves, "ffn_norm", false},
    {"qwen3", RopeLayout::Halves, "ffn_norm", false},
    {"qwen35", RopeLayout::Halves, "post_attention_norm", false},
    {"qwen3moe", RopeLayout::Halves, "ffn_norm", true},
    {"qwen35moe", RopeLayout::Halves, "post_attention_norm", true},
    // Hunyuan dense (Hunyuan-MT, Hunyuan 0.5B-7B): Q/K norms after RoPE. The
    // converter folds the dynamic-NTK "alpha" scaling into rope.freq_base.
    {"hunyuan-dense", RopeLayout::Halves, "ffn_norm", false, QkNormOrder::AfterRope},
    // ERNIE 4.5 dense (0.3B): a pre-norm block with SwiGLU; the reference
    // rotates adjacent pairs and the converter keeps the rows as they are.
    // Row proposed and gated by tools/describe_model.py
    // (docs/model-descriptions.md).
    {"ernie4_5", RopeLayout::Adjacent, "ffn_norm", false},
    // Gemma 2: embeddings scaled by sqrt(d), GeGLU, norms after the attention
    // and the FFN; every other block (the even ones) attends through the
    // sliding window, all with the one RoPE base. The soft caps of the
    // attention scores and the logits come from the file. As for Gemma 3, the
    // converter stores RMSNorm weights as 1 + w.
    {.name = "gemma2",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm",
     .scale_embeddings = true,
     .swa_period = 2},
    // Gemma 3: embeddings scaled by sqrt(d), GeGLU, norms after the attention
    // and the FFN, Q/K norms; every sixth block attends globally with RoPE
    // base rope.freq_base, the others through the sliding window with base
    // 10000. The converter stores RMSNorm weights as 1 + w, so they are plain
    // RMSNorms here.
    {.name = "gemma3",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm",
     .scale_embeddings = true,
     .swa_period = 6,
     .rope_base_swa = 10000.0f},
    // DeepSeek-V2 / V3 and the models converted alike (Youtu-LLM):
    // multi-head latent attention (planned from attn_kv_a_mqa,
    // geometry from attention.*_lora_rank and *_mla), RoPE on adjacent pairs
    // of the shared k_pe, a dense FFN in the leading blocks and a MoE with
    // shared experts in the others (both from the tensors); the top-k
    // weights are renormalized only when the file says so.
    {"deepseek2", RopeLayout::Adjacent, "ffn_norm", false},
    // Phi-3 / Phi-3.5 / Phi-4-mini: fused attn_qkv and [gate | up] ffn_up
    // (planned from the tensors), LongRoPE factors; every block uses the
    // sliding window when the file sets one (Phi-3-mini-4k: 2047).
    {.name = "phi3", .rope = RopeLayout::Halves, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .swa_period = 1},
    // The classic families: LayerNorm (weight and bias where present),
    // biases on every projection where present, ungated FFNs, parallel
    // blocks, learned absolute positions or ALiBi. Architectures without
    // rotary embeddings keep the RopeLayout enum's first value (unused).
    // GPT-2: learned positions, fused QKV, GELU (tanh), tied output.
    {.name = "gpt2",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .norm = NormKind::Layer,
     .rotary = false},
    // GPT-NeoX (Pythia): parallel attention and FFN with a norm each
    // (use_parallel_residual), partial RoPE, exact GELU.
    {.name = "gptneox",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::Gelu,
     .norm = NormKind::Layer,
     .parallel_residual = true},
    // Falcon: parallel blocks, one shared norm (7B), or ln_attn stored as
    // attn_norm_2 and ln_mlp as attn_norm (40B); exact GELU.
    {.name = "falcon",
     .rope = RopeLayout::Halves,
     .ffn_norm = "attn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::Gelu,
     .norm = NormKind::Layer,
     .parallel_residual = true,
     .attn_norm = "attn_norm_2"},
    // BLOOM: ALiBi (maximum bias 8), a LayerNorm on the embeddings, GELU (tanh).
    {.name = "bloom",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .norm = NormKind::Layer,
     .rotary = false,
     .alibi_bias = 8.0f},
    // MPT: ALiBi (the file's alibi_bias_max), LayerNorms without biases,
    // optional Q/K/V clamping, exact GELU.
    {.name = "mpt",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::Gelu,
     .norm = NormKind::Layer,
     .rotary = false,
     .alibi_bias = 8.0f},
    // StarCoder (GPTBigCode): GPT-2's block with multi-query attention.
    {.name = "starcoder",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .norm = NormKind::Layer,
     .rotary = false},
    // StarCoder2: LayerNorm, biased projections, RoPE, GELU (tanh).
    {.name = "starcoder2",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .swa_period = 1,
     .norm = NormKind::Layer,
     .sliding_window_default = 4096},
    // Phi-2: parallel blocks with one shared norm, partial RoPE, a biased LM head.
    {.name = "phi2",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .norm = NormKind::Layer,
     .parallel_residual = true},
    // StableLM (2): LayerNorm, partial RoPE, SwiGLU; parallel blocks and
    // per-head Q/K LayerNorms where the file has them (12B).
    {.name = "stablelm", .rope = RopeLayout::Halves, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .norm = NormKind::Layer},
    // Orion: a Llama block with LayerNorms.
    {.name = "orion", .rope = RopeLayout::Halves, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .norm = NormKind::Layer,
     .spm_prefix_after_control = false,
     .spm_add_bos_default = false,
     .spm_add_space_prefix_default = false,
     .spm_charsmap = "51d3684747b35ef99cd4802f37cc1ee0c6bec4d060814f774efb1066d3c167f2",
     .spm_user_defined = "e98a1435e08f8182b13816335751bcea918df2a67331f73288af701d0905c7f4"},
    // OLMo (1): non-parametric LayerNorms (no norm tensors), optional clip_qkv.
    {.name = "olmo", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .norm = NormKind::Layer},
    // Command-R: parallel blocks with one shared LayerNorm (no bias),
    // adjacent RoPE pairs, logits multiplied by the file's logit_scale;
    // per-head Q/K LayerNorms where the file has them (Command-R+).
    {.name = "command-r",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .norm = NormKind::Layer,
     .parallel_residual = true},
    // Nemotron: LayerNorm1P (the converter stores 1 + w), partial RoPE, ReLU^2.
    {.name = "nemotron",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::ReluSquared,
     .norm = NormKind::Layer,
     .spm_prefix_after_control = false,
     .spm_add_bos_default = false},
    // Apertus: Q/K RMSNorms, xIELU with per-block scalars.
    {.name = "apertus",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::XIelu},
    // AFM (Arcee): a Llama block with an ungated ReLU^2 FFN.
    {.name = "arcee",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::ReluSquared},
    // DBRX: LayerNorms without biases (the FFN's is attn_output_norm), fused
    // QKV clamped, a softmax top-k MoE renormalized.
    {.name = "dbrx", .rope = RopeLayout::Halves, .ffn_norm = "attn_output_norm", .moe_norm_weights = true, .norm = NormKind::Layer},
    // SmolLM3: Llama's block (the converter interleaves the Q/K rows), every
    // fourth block without RoPE. The converter writes no per-block list
    // (attention.rope_pattern), so the period is the row's.
    {.name = "smollm3", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .nope_period = 4},
    // OLMo 2 and OLMo 3: post-only norms (after the attention and after the
    // FFN, none before: planned from the tensors) and RMSNorms of Q and K
    // over the whole projections (from their size), before RoPE. OLMo 3
    // attends through a window in the blocks its file lists (all but every
    // fourth by default); those rotate with the base unscaled, the YaRN the
    // converter writes being the global blocks'.
    {.name = "olmo2",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm",
     .swa_period = 4,
     .swa_rope_unscaled = true},
    // EXAONE 4: post-only norms as OLMo 2's, per-head Q/K norms before RoPE.
    // With a window (EXAONE 4.0 32B: three windowed blocks, then a global
    // one), only the windowed blocks rotate: the global ones are NoPE.
    {.name = "exaone4",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm",
     .swa_period = 4,
     .nope_global = true},
    // EXAONE 3 / 3.5: a pre-norm block with SwiGLU, rotating halves (the
    // converter keeps the rows), Llama 3 frequency factors in rope_freqs.
    {"exaone", RopeLayout::Halves, "ffn_norm", false},
    // Granite 3.x dense: Llama's block with μP scalars from the file
    // (embedding_scale, residual_scale, attention.scale) and logits divided
    // by its logit_scale (logits_scaling).
    {.name = "granite", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .logit_scale_divides = true},
    // MiniCPM (1, 2, 4 dense): Llama's block with μP scalars from the file:
    // scale_emb, scale_depth / sqrt(n_layers) and logits divided by n_embd /
    // dim_model_base; LongRoPE factors when the model has them.
    {.name = "minicpm", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .logit_scale_divides = true},
    // InternLM 2 / 2.5: Llama's block once the converter has split and
    // interleaved the fused wqkv. (InternLM 3 converts as llama.)
    {"internlm2", RopeLayout::Adjacent, "ffn_norm", false},
    // Seed-OSS: QKV biases, halves; the norm before the FFN is stored as
    // post_attention_norm (as Qwen3.5's).
    {"seed_oss", RopeLayout::Halves, "post_attention_norm", false},
    // ChatGLM 2 / 3 and GLM-4 (2024, GlmForCausalLM): fused attn_qkv with a
    // bias, fused [gate | up], RoPE on adjacent pairs of the first half of
    // each head (rope.dimension_count).
    {"chatglm", RopeLayout::Adjacent, "ffn_norm", false},
    // GLM-4-0414: ChatGLM's rotation (adjacent pairs, half of each head), QKV
    // biases, fused [gate | up], and norms after the attention and the FFN
    // besides the ones before them.
    {.name = "glm4",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm"},
    // Gemma (1): embeddings scaled by sqrt(d), GeGLU; the converter stores
    // the RMSNorm weights as 1 + w.
    {.name = "gemma",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .scale_embeddings = true},
    // Ministral 3 (mistral3): Llama's block with YaRN whose
    // yarn_log_multiplier is the reference's mscale_all_dim, and the
    // queries' temperature by position (attention.temperature_scale).
    {.name = "mistral3", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = false, .yarn_mscale_all_dim = true},
    // Deci / Nemotron-NAS: block geometry comes from scalar or per-block
    // head_count, head_count_kv and feed_forward_length, and the tensors.
    {"deci", RopeLayout::Adjacent, "ffn_norm", false},
    // PLaMo 3: fused QKV and gate/up, folded RMSNorm offsets, Q/K norms,
    // sandwich norms and windowed blocks with an unscaled local RoPE base.
    {.name = "plamo3", .rope = RopeLayout::Halves, .ffn_norm = "ffn_norm", .moe_norm_weights = false,
     .post_mixer_norm = "post_attention_norm", .post_ffn_norm = "post_ffw_norm", .swa_period = 8,
     .rope_base_swa = 10000.0f},
    // gpt-oss: biases on Q/K/V and the attention output, a learned sink
    // logit per head (attn_sinks), every other block (the even ones)
    // windowed, YaRN from the file; the norm before the FFN is the GGUF's
    // post_attention_norm. The router has a bias and takes a softmax over the
    // top-k logits (renormalized softmax: the same function); the experts
    // have biases and the clamped SwiGLU.
    {.name = "gpt-oss",
     .rope = RopeLayout::Halves,
     .ffn_norm = "post_attention_norm",
     .moe_norm_weights = true,
     .ffn_activation = FfnActivation::SwigluOai,
     .swa_period = 2,
     .yarn_truncate = false},
    {"qwen2moe", RopeLayout::Halves, "ffn_norm", false},
    {"glm4moe", RopeLayout::Halves, "post_attention_norm", true},
    {"dots1", RopeLayout::Halves, "ffn_norm", true},
    {"ernie4_5-moe", RopeLayout::Adjacent, "ffn_norm", true},
    {"hy_v3", RopeLayout::Halves, "ffn_norm", true},
    {"deepseek", RopeLayout::Adjacent, "ffn_norm", false},
    {"bailingmoe", RopeLayout::Adjacent, "ffn_norm", true},
    // Mamba-1 and Falcon-Mamba: every block is a Mamba-1 mixer alone
    // (planned from ssm_x and ssm_d) after attn_norm, with no FFN and no
    // attention (the RoPE pairing is unused). Falcon-Mamba's weightless
    // norms of dt, B and C are switched on by the file (ssm.dt_b_c_rms),
    // with ssm_dt_b_c_eps (the default, its reference's mixer_rms_eps).
    // Rows proposed and gated by tools/describe_model.py.
    {"mamba", RopeLayout::Halves, "", false},
    // Mamba-2 and Mamba-Codestral: every block is a Mamba-2 mixer alone
    // (planned from ssm_d without ssm_x), heads and groups from the file.
    {.name = "mamba2", .rope = RopeLayout::Halves, .ffn_norm = "", .moe_norm_weights = false,
     .ssm_norm_grouped = false},
    // LFM2 (LiquidAI): gated short-convolution blocks (shortconv.*, planned
    // from the tensors) and attention blocks with per-head Q/K norms before
    // RoPE, which pairs halves; SwiGLU FFNs. head_count_kv is per block (0 in
    // the convolution blocks). The final norm, which the reference names
    // embedding_norm, is the converter's token_embd_norm. Row proposed and
    // gated by tools/describe_model.py (docs/model-descriptions.md).
    {.name = "lfm2",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .output_norm = "token_embd_norm"},
    // LFM2-MoE (LFM2-8B-A1B): LFM2's blocks, dense FFNs in the leading blocks
    // and sigmoid-routed experts in the others (the gating and the selection
    // bias exp_probs_b from the file); the top-k weights are renormalized
    // (the reference's norm_topk_prob, which the converter does not write).
    {.name = "lfm2moe",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = true,
     .output_norm = "token_embd_norm"},
    // Qwen3-Next: Qwen3.5-MoE's parts (Gated DeltaNet blocks, gated attention
    // with Q/K norms and partial RoPE, experts with a gated shared expert) in
    // Qwen3-Next's file layout: DeltaNet value heads in the reference's
    // grouped order, and beta and alpha in one ssm_ba projection (planned
    // from the tensors). The converter stores its (1 + w) norms as 1 + w.
    {.name = "qwen3next",
     .rope = RopeLayout::Halves,
     .ffn_norm = "post_attention_norm",
     .moe_norm_weights = true,
     .ssm_grouped_heads = true},
    // Kimi-Linear: Kimi Delta Attention blocks (ssm_conv1d_q/k/v, planned from
    // the tensors), latent attention blocks without positional rotation
    // (nope_period = 1), a dense
    // leading FFN, then sigmoid-routed experts with a selection bias, the
    // file's weight scale and an ungated shared expert; the top-k weights
    // are renormalized (moe_renormalize). The RoPE pairing is DeepSeek's
    // convention for latent attention; no block rotates with it.
    {.name = "kimi-linear",
     .rope = RopeLayout::Adjacent,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = true,
     .nope_period = 1},
    // Gemma 3n text models: Gemma 3's block (scaled embeddings, GeGLU,
    // sandwich norms, Q/K norms, windowed blocks with RoPE base 10000) with
    // unscaled q·k and values normalized per head, inside AltUp's residual
    // streams, with Laurel, per-layer embeddings, the Gaussian top-k of the
    // first blocks' FFN gates and the last blocks sharing earlier blocks'
    // keys and values: each planned from its tensors and metadata. Every
    // fifth block attends globally unless the file lists the windowed ones.
    // The converter stores the norm weights as they are (no 1 + w).
    {.name = "gemma3n",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm",
     .scale_embeddings = true,
     .swa_period = 5,
     .rope_base_swa = 10000.0f,
     .unscaled_scores = true,
     .value_norm = true},
    // Gemma 4 text models: Gemma 3n's attention (unscaled q·k, value norm,
    // KV sharing) and per-layer embeddings without AltUp or Laurel, a
    // per-block output scale (layer_output_scale), global blocks whose
    // values may be their keys, and partial "proportional" RoPE in the global
    // blocks (rope_freqs keeps a quarter of the pairs turning). Every sixth
    // block attends globally unless the file lists the windowed ones.
    {.name = "gemma4",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm",
     .scale_embeddings = true,
     .swa_period = 6,
     .rope_base_swa = 10000.0f,
     .unscaled_scores = true,
     .value_norm = true},
    // EmbeddingGemma: Gemma 3's blocks (norm weights stored as 1 + w) with
    // bidirectional attention, the windowed blocks seeing half their window
    // on either side; mean pooling and the dense projections come from the
    // file (core/pooling.h).
    {.name = "gemma-embedding",
     .rope = RopeLayout::Halves,
     .ffn_norm = "ffn_norm",
     .moe_norm_weights = false,
     .ffn_activation = FfnActivation::GeluTanh,
     .post_mixer_norm = "post_attention_norm",
     .post_ffn_norm = "post_ffw_norm",
     .scale_embeddings = true,
     .swa_period = 6,
     .rope_base_swa = 10000.0f,
     .causal = false},
    {"deepseek32", RopeLayout::Adjacent, "ffn_norm", true},
    {.name = "glm-dsa", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = true,
     .indexer_rope = RopeLayout::Adjacent, .latent_norm_eps = 1e-06f},
    {"bailingmoe2", RopeLayout::Halves, "ffn_norm", true},
    {.name = "deepseek4", .rope = RopeLayout::Adjacent, .ffn_norm = "ffn_norm", .moe_norm_weights = true,
     .swa_period = 1, .shared_kv = true},
};

}  // namespace

std::span<const ArchSpec> architectures() noexcept { return kArchitectures; }

const ArchSpec* find_architecture(std::string_view name) noexcept {
    for (const ArchSpec& a : kArchitectures) {
        if (a.name == name) return &a;
    }
    return nullptr;
}

std::string supported_architectures() {
    std::string out;
    for (const ArchSpec& a : kArchitectures) out += (out.empty() ? "" : ",") + std::string(a.name);
    return out;
}

namespace {

// Element i of an integer array value (any integer element type), or nullopt.
std::optional<int64_t> array_int(const GgufValue& v, uint64_t i) {
    if (v.type != GgufValue::Type::Array || i >= v.array_size || v.array_data == nullptr) return std::nullopt;
    auto read = [&](auto zero) -> int64_t {
        decltype(zero) x;
        std::memcpy(&x, v.array_data + i * sizeof x, sizeof x);
        return static_cast<int64_t>(x);
    };
    switch (v.array_type) {
        case GgufValue::Type::U8: return read(uint8_t{});
        case GgufValue::Type::I8: return read(int8_t{});
        case GgufValue::Type::U16: return read(uint16_t{});
        case GgufValue::Type::I16: return read(int16_t{});
        case GgufValue::Type::U32: return read(uint32_t{});
        case GgufValue::Type::I32: return read(int32_t{});
        case GgufValue::Type::U64: return read(uint64_t{});
        case GgufValue::Type::I64: return read(int64_t{});
        default: return std::nullopt;
    }
}

}  // namespace

std::string block_tensor(int32_t block, std::string_view suffix) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "blk.%d.%.*s", block, static_cast<int>(suffix.size()), suffix.data());
    return buf;
}

std::string block_norm_tensor(const MmapLoader& file, int32_t block, std::string_view suffix) {
    const std::string weighted = block_tensor(block, std::string(suffix) + ".weight");
    const std::string bare = block_tensor(block, suffix);
    return file.tensor(weighted) != nullptr || file.tensor(bare) == nullptr ? weighted : bare;
}

namespace {

// The parts Gemma 3n and Gemma 4 add to Gemma 3's blocks, from the metadata
// and the tensors (BlockPlan / ModelConfig say what each does), and the
// attention's direction. Called once the blocks are planned; sets
// n_kv_layers. `key` prefixes a metadata key with the architecture.
template <typename Key>
Status describe_gemma_parts(const MmapLoader& f, ModelConfig& c, const Key& key) {
    c.causal = c.spec->causal;
    if (const GgufValue* v = f.metadata(key("attention.causal"))) {
        const auto causal = v->as_bool();
        if (!causal) return Status(ErrorCode::InvalidModel, "invalid " + key("attention.causal"));
        c.causal = *causal;
    }

    // KV sharing: the last shared_kv_layers blocks attend over the keys and
    // values of the last block of their kind (windowed or global) before
    // them that keeps its own.
    if (const auto* v = f.metadata(key("attention.shared_kv_layers")); v && !v->as_int()) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("attention.shared_kv_layers"));
    }
    const int64_t shared = f.get_int(key("attention.shared_kv_layers")).value_or(0);
    if (shared < 0 || shared >= c.n_layers) return Status(ErrorCode::InvalidModel, "invalid " + key("attention.shared_kv_layers"));
    const auto first_shared = static_cast<int32_t>(c.n_layers - shared);
    for (int32_t l = first_shared; l < c.n_layers; ++l) {
        BlockPlan& b = c.blocks[static_cast<size_t>(l)];
        if (b.mixer != MixerKind::Attention || b.latent) {
            return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) + " shares keys and values (" +
                                                       key("attention.shared_kv_layers") + ") but is not attention");
        }
        for (int32_t s = first_shared - 1; s >= 0 && b.kv_source < 0; --s) {
            const BlockPlan& src = c.blocks[static_cast<size_t>(s)];
            if (src.mixer == MixerKind::Attention && !src.latent && src.windowed == b.windowed) b.kv_source = s;
        }
        if (b.kv_source < 0) {
            return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) + " shares keys and values with no "
                                                       "earlier block of its kind");
        }
        const BlockPlan& source = c.blocks[static_cast<size_t>(b.kv_source)];
        if (source.n_head != b.n_head || source.n_head_kv != b.n_head_kv || source.rope != b.rope) {
            return Status(ErrorCode::Unsupported, "block " + std::to_string(l) +
                              " shares keys and values with incompatible head geometry or rotation");
        }
        b.values_from_keys = false;  // it computes none
    }
    for (const BlockPlan& b : c.blocks) c.n_kv_layers += b.mixer == MixerKind::Attention && b.kv_source < 0 ? 1 : 0;

    // AltUp: several residual streams, with the projections that make and
    // merge them.
    if (const TensorView* proj = f.tensor("altup_proj.weight")) {
        const int64_t streams = f.get_int(key("altup.num_inputs")).value_or(0);
        const int64_t active = f.get_int(key("altup.active_idx")).value_or(0);
        const TensorView* unembd = f.tensor("altup_unembd_proj.weight");
        if (streams < 2 || streams > 16 || active < 0 || active >= streams || unembd == nullptr) {
            return Status(ErrorCode::InvalidModel, "AltUp tensors without valid " + key("altup.num_inputs") + " / " +
                                                       key("altup.active_idx") + " or altup_unembd_proj");
        }
        for (const TensorView* t : {proj, unembd}) {
            if (t->ne[0] != c.n_embd || t->ne[1] != c.n_embd || t->ne[2] != streams - 1 || t->ne[3] != 1) {
                return Status(ErrorCode::InvalidModel, std::string(t->name) + " is not [n_embd, n_embd, streams - 1]");
            }
        }
        c.altup_streams = static_cast<int32_t>(streams);
        c.altup_active = static_cast<int32_t>(active);
    } else if (f.tensor("altup_unembd_proj.weight") != nullptr || f.get_int(key("altup.num_inputs")).value_or(0) > 1) {
        return Status(ErrorCode::InvalidModel, "AltUp requires altup_proj.weight");
    }

    // Per-layer embeddings.
    if (const auto* v = f.metadata(key("embedding_length_per_layer_input")); v && !v->as_int()) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("embedding_length_per_layer_input"));
    }
    const int64_t ple = f.get_int(key("embedding_length_per_layer_input")).value_or(0);
    if (ple < 0 || ple > (int64_t{1} << 16)) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("embedding_length_per_layer_input"));
    }
    if (ple > 0) {
        if (f.tensor("per_layer_token_embd.weight") == nullptr) {
            return Status(ErrorCode::InvalidModel, "per-layer embeddings require per_layer_token_embd.weight");
        }
        c.ple_dim = static_cast<int32_t>(ple);
    } else if (f.tensor("per_layer_token_embd.weight") != nullptr) {
        return Status(ErrorCode::InvalidModel, "per-layer token table without " + key("embedding_length_per_layer_input"));
    }

    // The Gaussian top-k of the FFN gates: a standard-deviation multiplier per
    // block, -inf (no sparsity) for the others.
    if (const GgufValue* v = f.metadata(key("activation_sparsity_scale"))) {
        if (v->type != GgufValue::Type::Array || v->array_type != GgufValue::Type::F32 ||
            v->array_size < static_cast<uint64_t>(c.n_layers)) {
            return Status(ErrorCode::InvalidModel, "invalid " + key("activation_sparsity_scale"));
        }
        for (int32_t l = 0; l < c.n_layers; ++l) {
            float k = 0.0f;
            std::memcpy(&k, v->array_data + 4 * static_cast<size_t>(l), 4);
            BlockPlan& b = c.blocks[static_cast<size_t>(l)];
            if (std::isnan(k) || k == INFINITY) return Status(ErrorCode::InvalidModel, "invalid " + key("activation_sparsity_scale"));
            if (k == -INFINITY) continue;
            if (b.ffn != FfnKind::Dense || b.fused_gate_up) {
                return Status(ErrorCode::Unsupported, "activation sparsity outside a dense FFN with its own gate");
            }
            b.sparse_gate = true;
            b.gate_sparsity = k;
        }
    }
    return Status::ok();
}

}  // namespace

Result<ModelConfig> describe_model(const MmapLoader& f) {
    ModelConfig c;
    const auto arch = f.get_string("general.architecture");
    if (!arch) return Status(ErrorCode::InvalidModel, "not a model file: missing general.architecture");
    c.arch = std::string(*arch);
    c.spec = find_architecture(c.arch);
    if (c.spec == nullptr) {
        return Status(ErrorCode::Unsupported,
                      "architecture '" + c.arch + "' is not supported yet (supported: " + supported_architectures() + ")");
    }

    auto key = [&](const char* suffix) { return c.arch + "." + suffix; };
    auto require_int = [&](const char* suffix, int32_t& out) -> Status {
        const auto v = f.get_int(key(suffix));
        if (!v || *v <= 0 || *v > (int64_t{1} << 24)) {
            return Status(ErrorCode::InvalidModel, "missing or invalid " + key(suffix));
        }
        out = static_cast<int32_t>(*v);
        return Status::ok();
    };
    LIYAB_RETURN_IF_ERROR(require_int("block_count", c.n_layers));
    // Multi-token-prediction heads are stored as trailing blocks; the main pass skips them.
    c.n_layers -= static_cast<int32_t>(f.get_int(key("nextn_predict_layers")).value_or(0));
    if (c.n_layers <= 0) return Status(ErrorCode::InvalidModel, "invalid nextn_predict_layers");
    LIYAB_RETURN_IF_ERROR(require_int("embedding_length", c.n_embd));
    // Each block's token mixer, from its tensors: a causal convolution makes
    // it recurrent (with the D skip a Mamba mixer, Mamba-1 when dt, B and C
    // come from ssm_x, else Gated DeltaNet), attention projections make it
    // attention, and a block with neither has no mixer (its FFN alone).
    auto mixer_of = [&](int32_t l) {
        // Specific recurrent layouts precede the shared partial-SSM markers.
        if (f.tensor(block_tensor(l, "shortconv.conv.weight")) != nullptr ||
            f.tensor(block_tensor(l, "shortconv.in_proj.weight")) != nullptr ||
            f.tensor(block_tensor(l, "shortconv.out_proj.weight")) != nullptr) return MixerKind::ShortConv;
        if (f.tensor(block_tensor(l, "ssm_conv1d_q.weight")) != nullptr ||
            f.tensor(block_tensor(l, "ssm_conv1d_k.weight")) != nullptr ||
            f.tensor(block_tensor(l, "ssm_conv1d_v.weight")) != nullptr) return MixerKind::Kda;
        if (f.tensor(block_tensor(l, "ssm_conv1d.weight")) != nullptr ||
            f.tensor(block_tensor(l, "ssm_in.weight")) != nullptr ||
            f.tensor(block_tensor(l, "ssm_x.weight")) != nullptr ||
            f.tensor(block_tensor(l, "ssm_out.weight")) != nullptr ||
            f.tensor(block_tensor(l, "ssm_a")) != nullptr || f.tensor(block_tensor(l, "ssm_d")) != nullptr) {
            if (f.tensor(block_tensor(l, "ssm_d")) == nullptr &&
                f.tensor(block_tensor(l, "ssm_in.weight")) == nullptr &&
                f.tensor(block_tensor(l, "ssm_x.weight")) == nullptr) return MixerKind::DeltaNet;
            return f.tensor(block_tensor(l, "ssm_x.weight")) != nullptr ? MixerKind::Mamba : MixerKind::Mamba2;
        }
        for (const char* t : {"attn_q.weight", "attn_qkv.weight", "attn_kv_a_mqa.weight",
                              "attn_k.weight", "attn_v.weight", "attn_output.weight", "attn_kv.weight",
                              "attn_q_a.weight", "attn_q_b.weight", "attn_output_a.weight", "attn_output_b.weight"}) {
            if (f.tensor(block_tensor(l, t)) != nullptr) return MixerKind::Attention;
        }
        return MixerKind::None;
    };
    bool attends = false;
    for (int32_t l = 0; l < c.n_layers && !attends; ++l) attends = mixer_of(l) == MixerKind::Attention;
    // Mixture of experts: dense FFN size is optional when every FFN is MoE.
    c.n_expert = static_cast<int32_t>(f.get_int(key("expert_count")).value_or(0));
    if (c.n_expert > 0) {
        if (c.spec->shared_kv) {
            return Status(ErrorCode::Unsupported,
                          "shared K=V MoE needs hash/sqrtsoftplus routing, clipped SwiGLU and learned sinks (MoE integration)");
        }
        LIYAB_RETURN_IF_ERROR(require_int("expert_used_count", c.n_expert_used));
        // Absent (Mixtral, written by the llama converter): from the experts' tensors, below.
        if (f.get_int(key("expert_feed_forward_length"))) {
            LIYAB_RETURN_IF_ERROR(require_int("expert_feed_forward_length", c.n_ff_expert));
        }
        c.n_ff_shared = static_cast<int32_t>(f.get_int(key("expert_shared_feed_forward_length")).value_or(0));
        const int64_t gating = f.get_int(key("expert_gating_func")).value_or(1);
        if (gating != 1 && gating != 2) {
            return Status(ErrorCode::Unsupported, "MoE gating function " + std::to_string(gating) + " is not supported");
        }
        c.moe_gating = gating == 2 ? MoeGating::Sigmoid : MoeGating::Softmax;
        const GgufValue* norm = f.metadata(key("expert_weights_norm"));
        c.moe_norm_weights = norm != nullptr && norm->as_bool() ? *norm->as_bool() : c.spec->moe_norm_weights;
        c.moe_weights_scale = static_cast<float>(f.get_float(key("expert_weights_scale")).value_or(1.0));
        if (c.moe_weights_scale == 0.0f) c.moe_weights_scale = 1.0f;
        if (c.n_expert_used > c.n_expert) return Status(ErrorCode::InvalidModel, "expert_used_count > expert_count");
        // Group-limited selection (ModelConfig::n_expert_groups): the top-k
        // among the experts of the best groups only. Every group used means
        // plain top-k.
        const int64_t groups = f.get_int(key("expert_group_count")).value_or(1);
        const int64_t used = f.get_int(key("expert_group_used_count")).value_or(groups);
        if (groups < 1 || groups > c.n_expert || c.n_expert % groups != 0 || used < 1 || used > groups ||
            used * (c.n_expert / groups) < c.n_expert_used) {
            return Status(ErrorCode::InvalidModel, "invalid " + key("expert_group_count") + " / " +
                                                       key("expert_group_used_count") + " for " +
                                                       std::to_string(c.n_expert) + " experts, top-" +
                                                       std::to_string(c.n_expert_used));
        }
        if (used < groups) {
            c.n_expert_groups = static_cast<int32_t>(groups);
            c.n_expert_groups_used = static_cast<int32_t>(used);
        }
    }
    auto block_values = [&](const char* suffix, const std::vector<int32_t>& fallback,
                            std::vector<int32_t>& out) -> Status {
        const GgufValue* value = f.metadata(key(suffix));
        out = fallback;
        if (value == nullptr) return Status::ok();
        if (value->type == GgufValue::Type::Array) {
            if (value->array_size < static_cast<uint64_t>(c.n_layers) ||
                (value->array_type != GgufValue::Type::U32 && value->array_type != GgufValue::Type::I32)) {
                return Status(ErrorCode::InvalidModel, "expected one 32-bit integer per block in " + key(suffix));
            }
            for (size_t l = 0; l < out.size(); ++l) {
                const auto v = array_int(*value, l);
                if (!v || *v < 0 || *v > (int64_t{1} << 24)) {
                    return Status(ErrorCode::InvalidModel, "invalid " + key(suffix));
                }
                out[l] = static_cast<int32_t>(*v);
            }
        } else {
            const auto v = value->as_int();
            if (!v || *v < 0 || *v > (int64_t{1} << 24)) {
                return Status(ErrorCode::InvalidModel, "invalid " + key(suffix));
            }
            std::fill(out.begin(), out.end(), static_cast<int32_t>(*v));
        }
        return Status::ok();
    };
    const std::vector<int32_t> zero(static_cast<size_t>(c.n_layers), 0);
    std::vector<int32_t> block_heads, block_kv_heads, block_ff;
    LIYAB_RETURN_IF_ERROR(block_values("attention.head_count", zero, block_heads));
    LIYAB_RETURN_IF_ERROR(block_values("attention.head_count_kv", block_heads, block_kv_heads));
    LIYAB_RETURN_IF_ERROR(block_values("feed_forward_length", zero, block_ff));
    c.n_ff = *std::max_element(block_ff.begin(), block_ff.end());
    c.n_head = std::max(*std::max_element(block_heads.begin(), block_heads.end()), 1);
    c.n_head_kv = std::max(*std::max_element(block_kv_heads.begin(), block_kv_heads.end()), 1);
    if (attends && f.metadata(key("attention.head_count")) == nullptr) {
        return Status(ErrorCode::InvalidModel, "missing attention.head_count");
    }
    c.head_dim = attends ? static_cast<int32_t>(f.get_int(key("attention.key_length")).value_or(c.n_embd / c.n_head)) : 0;
    c.value_dim = c.head_dim;
    c.rope_dim = static_cast<int32_t>(f.get_int(key("rope.dimension_count")).value_or(c.head_dim));
    const auto value_length = f.get_int(key("attention.value_length"));
    if (c.spec->shared_kv) {
        LIYAB_RETURN_IF_ERROR(require_int("attention.q_lora_rank", c.shared_q_rank));
        LIYAB_RETURN_IF_ERROR(require_int("attention.output_group_count", c.output_groups));
        LIYAB_RETURN_IF_ERROR(require_int("attention.output_lora_rank", c.output_rank));
        if (c.n_head_kv != 1 || c.output_groups > c.n_head || c.n_head % c.output_groups != 0 ||
            value_length.value_or(c.head_dim) != c.head_dim ||
            f.get_int(key("attention.kv_lora_rank")).value_or(0) != 0) {
            return Status(ErrorCode::InvalidModel, "shared K=V needs one KV head and whole output head groups");
        }
        c.compress_rope_base = static_cast<float>(f.get_float(key("attention.compress_rope_freq_base")).value_or(0.0));
        if (!(c.compress_rope_base > 0.0f) || !std::isfinite(c.compress_rope_base)) {
            return Status(ErrorCode::InvalidModel, "invalid " + key("attention.compress_rope_freq_base"));
        }
    }
    if (const auto kv_rank = f.get_int(key("attention.kv_lora_rank")); kv_rank && *kv_rank > 0) {
        // Multi-head latent attention. The converter states the absorbed
        // geometry (one KV head of [k_pe | latent], values the latent) and
        // the per-head one apart (*_mla). Older files describe the expanded
        // head instead and keep attn_kv_b combined. Normalize both metadata
        // layouts here; the block plan still records the tensor's layout.
        LatentAttention& m = c.mla;
        const auto key_mla = f.get_int(key("attention.key_length_mla"));
        const auto value_mla = f.get_int(key("attention.value_length_mla"));
        if (key_mla.has_value() != value_mla.has_value()) {
            return Status(ErrorCode::InvalidModel, "latent attention needs both key_length_mla and value_length_mla");
        }
        const int64_t trained_key = key_mla.value_or(c.head_dim);
        const int64_t trained_value = value_mla.value_or(value_length.value_or(0));
        const int64_t query_rank = f.get_int(key("attention.q_lora_rank")).value_or(0);
        if (*kv_rank > (int64_t{1} << 20) || c.rope_dim <= 0 || int64_t{c.rope_dim} + *kv_rank > INT32_MAX ||
            trained_key <= c.rope_dim || trained_key > INT32_MAX || trained_value <= 0 || trained_value > INT32_MAX ||
            query_rank < 0 || query_rank > INT32_MAX) {
            return Status(ErrorCode::InvalidModel, "inconsistent latent attention dimensions (" + key("attention.*") + ")");
        }
        m.kv_rank = static_cast<int32_t>(*kv_rank);
        m.q_rank = static_cast<int32_t>(query_rank);
        m.nope_dim = static_cast<int32_t>(trained_key - c.rope_dim);
        m.v_dim = static_cast<int32_t>(trained_value);
        c.value_dim = m.kv_rank;
        if (key_mla && c.n_head_kv == c.n_head) c.n_head_kv = 1;
        if (!key_mla) {
            if (c.n_head_kv != c.n_head) {
                return Status(ErrorCode::InvalidModel, "combined latent attention must describe one expanded KV head per query head");
            }
            c.n_head_kv = 1;
            c.head_dim = m.kv_rank + c.rope_dim;
            c.value_dim = m.kv_rank;
        }
        if (c.n_head_kv != 1 || c.head_dim != m.kv_rank + c.rope_dim || c.value_dim != m.kv_rank ||
            m.nope_dim <= 0 || m.v_dim <= 0 || m.q_rank < 0 || *kv_rank > (int64_t{1} << 20)) {
            return Status(ErrorCode::InvalidModel, "inconsistent latent attention dimensions (" + key("attention.*") + ")");
        }
    } else if (value_length && *value_length != c.head_dim) {
        return Status(ErrorCode::Unsupported, "key_length != value_length is not supported");
    }
    // Windowed blocks with heads or a rotation of other sizes than the
    // global ones (Gemma 4: 256-wide windowed heads, 512-wide global ones)
    // need per-block geometry.
    for (const char* swa : {"attention.key_length_swa", "attention.value_length_swa", "rope.dimension_count_swa"}) {
        const auto v = f.get_int(key(swa));
        const int32_t global = std::string_view(swa).starts_with("rope") ? static_cast<int32_t>(
                                                                                 f.get_int(key("rope.dimension_count")).value_or(c.head_dim))
                                                                         : c.head_dim;
        if (v && *v != global) {
            return Status(ErrorCode::Unsupported, "windowed blocks of other head or rotation sizes than the global "
                                                  "ones (" + key(swa) + " " + std::to_string(*v) + ", global " +
                                                      std::to_string(global) + ") are not supported yet");
        }
    }
    if (f.get_int(key("attention.indexer.head_count")).value_or(0) != 0) {
        LIYAB_RETURN_IF_ERROR(require_int("attention.indexer.head_count", c.index_heads));
        LIYAB_RETURN_IF_ERROR(require_int("attention.indexer.key_length", c.index_dim));
        LIYAB_RETURN_IF_ERROR(require_int("attention.indexer.top_k", c.index_topk));
        if ((c.mla.q_rank <= 0 && c.shared_q_rank <= 0) || c.index_dim < c.rope_dim) {
            return Status(ErrorCode::InvalidModel, "DSA needs low-rank queries and index keys at least rope.dimension_count wide");
        }
        if (f.get_int(key("attention.indexer.kpool")).value_or(1) > 1) {
            return Status(ErrorCode::Unsupported, "pooled DSA index keys are not supported");
        }
    }
    c.n_ctx_train = static_cast<int32_t>(f.get_int(key("context_length")).value_or(0));
    c.rms_eps = static_cast<float>(f.get_float(key("attention.layer_norm_rms_epsilon")).value_or(1e-5));
    if (c.spec->norm == NormKind::Layer) {  // LayerNorm architectures state their epsilon under its own key
        c.rms_eps = static_cast<float>(f.get_float(key("attention.layer_norm_epsilon")).value_or(c.rms_eps));
    }
    if (!(c.rms_eps >= 0.0f) || !std::isfinite(c.rms_eps)) {
        return Status(ErrorCode::InvalidModel, "invalid norm epsilon (" + key("attention.layer_norm_*epsilon") + ")");
    }
    // Architectures without rotary embeddings rotate nothing, whatever the
    // file says of RoPE: their positions come from learned embeddings or ALiBi.
    if (!c.spec->rotary) c.rope_dim = 0;
    if (f.tensor("output_hc_fn.weight") != nullptr) {
        LIYAB_RETURN_IF_ERROR(require_int("hyper_connection.count", c.hc_count));
        LIYAB_RETURN_IF_ERROR(require_int("hyper_connection.sinkhorn_iterations", c.hc_iterations));
        c.hc_eps = static_cast<float>(f.get_float(key("hyper_connection.epsilon")).value_or(0.0));
        if (c.hc_count > 64 || c.hc_iterations > 1024 || !(c.hc_eps > 0.0f) || !std::isfinite(c.hc_eps)) {
            return Status(ErrorCode::InvalidModel, "invalid hyper-connection geometry or epsilon");
        }
    }
    c.rope_base = static_cast<float>(f.get_float(key("rope.freq_base")).value_or(10000.0));
    // LongRoPE: frequency factors for short and long contexts (Phi-3), and the
    // attention factor that comes with them.
    const TensorView* long_factors = f.tensor("rope_factors_long.weight");
    const TensorView* short_factors = f.tensor("rope_factors_short.weight");
    const bool longrope = long_factors != nullptr || short_factors != nullptr;
    if (longrope) {
        if (long_factors == nullptr || short_factors == nullptr || f.tensor("rope_freqs.weight") != nullptr) {
            return Status(ErrorCode::InvalidModel,
                          "LongRoPE needs both rope_factors_short and rope_factors_long, and no rope_freqs");
        }
        LIYAB_RETURN_IF_ERROR(require_int("rope.scaling.original_context_length", c.rope_ctx_orig));
    }
    c.rope_magnitude = static_cast<float>(f.get_float(key("rope.scaling.attn_factor")).value_or(1.0));
    if (!(c.rope_magnitude > 0.0f)) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("rope.scaling.attn_factor"));
    }
    double qk_mscale = 1.0;  // YaRN's factor on q·k (yarn_log_multiplier), squared into attn_scale below
    if (const auto scaling = f.get_string(key("rope.scaling.type")); scaling && *scaling != "none") {
        if (*scaling == "longrope" || *scaling == "su") {
            if (!longrope) return Status(ErrorCode::InvalidModel, "LongRoPE scaling without rope_factors_long/short");
        } else if (*scaling == "yarn") {
            YarnRope& y = c.yarn;
            y.factor = static_cast<float>(f.get_float(key("rope.scaling.factor")).value_or(0.0));
            y.ctx_orig = static_cast<int32_t>(
                f.get_int(key("rope.scaling.original_context_length")).value_or(c.n_ctx_train));
            y.beta_fast = static_cast<float>(f.get_float(key("rope.scaling.yarn_beta_fast")).value_or(32.0));
            y.beta_slow = static_cast<float>(f.get_float(key("rope.scaling.yarn_beta_slow")).value_or(1.0));
            y.ext_factor = static_cast<float>(f.get_float(key("rope.scaling.yarn_ext_factor")).value_or(1.0));
            const GgufValue* truncate = f.metadata(key("rope.scaling.yarn_truncate"));
            if (truncate != nullptr && !truncate->as_bool()) {
                return Status(ErrorCode::InvalidModel, "invalid " + key("rope.scaling.yarn_truncate"));
            }
            y.truncate = truncate == nullptr ? c.spec->yarn_truncate : *truncate->as_bool();
            if (!(y.factor > 0.0f) || !std::isfinite(y.factor) || y.ctx_orig <= 0 || !(y.beta_fast > 0.0f) ||
                !(y.beta_slow > 0.0f) || !(y.ext_factor >= 0.0f) || !std::isfinite(y.ext_factor)) {
                return Status(ErrorCode::InvalidModel, "invalid YaRN parameters (" + key("rope.scaling.*") + ")");
            }
            // The attention factor that comes with the frequencies: the
            // file's own, or 0.1 ln(factor) + 1. A yarn_log_multiplier m
            // (DeepSeek, which stores 0.1 x its mscale_all_dim) scales q·k by
            // (1 + m ln(factor))^2 instead and leaves the rotation's
            // magnitude at 1: the reference's two mscales are equal there.
            const double ln_factor = y.factor > 1.0f ? std::log(static_cast<double>(y.factor)) : 0.0;
            const auto log_mul = f.get_float(key("rope.scaling.yarn_log_multiplier"));
            const auto attn_factor = f.get_float(key("rope.scaling.yarn_attn_factor"));
            // With ArchSpec::yarn_mscale_all_dim the multiplier is the
            // reference's mscale_all_dim m instead: the rotation's magnitude
            // is (0.1 ln(factor) + 1) / (0.1 m ln(factor) + 1) and q·k keeps
            // its scale (Ministral 3: m = 1, a magnitude of 1).
            double magnitude = attn_factor ? *attn_factor : (log_mul || c.spec->shared_kv) ? 1.0 : 0.1 * ln_factor + 1.0;
            if (log_mul && c.spec->yarn_mscale_all_dim) {
                if (!attn_factor) magnitude = (0.1 * ln_factor + 1.0) / (0.1 * *log_mul * ln_factor + 1.0);
            } else if (log_mul) {
                qk_mscale = 1.0 + *log_mul * ln_factor;
            }
            if (!(magnitude > 0.0) || !(qk_mscale > 0.0)) {
                return Status(ErrorCode::InvalidModel, "invalid YaRN attention factor (" + key("rope.scaling.*") + ")");
            }
            c.rope_magnitude = static_cast<float>(c.rope_magnitude * magnitude);
        } else if (*scaling != "linear") {
            return Status(ErrorCode::Unsupported, "rope scaling '" + std::string(*scaling) + "' is not supported");
        } else {
            c.rope_scale = static_cast<float>(f.get_float(key("rope.scaling.factor")).value_or(1.0));
            if (c.rope_scale <= 0.0f) return Status(ErrorCode::InvalidModel, "invalid rope scaling factor");
        }
    }
    // As the reference scales them: sqrt(d) rounded once to float.
    if (c.spec->scale_embeddings) c.embd_scale = static_cast<float>(std::sqrt(static_cast<double>(c.n_embd)));
    // μP scalars, wherever the file states them: the embeddings' multiplier,
    // the residual branches' multiplier and the logits' scale (a divisor in
    // the architectures whose converter writes one, ArchSpec::logit_scale_divides).
    const auto finite_positive = [](double v) {
        const float stored = static_cast<float>(v);
        return stored > 0.0f && std::isfinite(stored);
    };
    if (const auto scale = f.get_float(key("embedding_scale"))) {
        if (!finite_positive(*scale)) return Status(ErrorCode::InvalidModel, "invalid " + key("embedding_scale"));
        c.embd_scale = static_cast<float>(*scale);
    }
    if (const auto scale = f.get_float(key("residual_scale"))) {
        if (!finite_positive(*scale)) return Status(ErrorCode::InvalidModel, "invalid " + key("residual_scale"));
        c.residual_scale = static_cast<float>(*scale);
    }
    if (const auto scale = f.get_float(key("logit_scale"))) {
        if (!finite_positive(*scale)) return Status(ErrorCode::InvalidModel, "invalid " + key("logit_scale"));
        const double multiplier = c.spec->logit_scale_divides ? 1.0 / *scale : *scale;
        if (!finite_positive(multiplier)) return Status(ErrorCode::InvalidModel, "invalid " + key("logit_scale"));
        c.logit_scale = static_cast<float>(multiplier);
    }
    // Attention temperature by position (llama_4_scaling): its length is the
    // file's, else YaRN's original context.
    if (const auto beta = f.get_float(key("attention.temperature_scale")); beta && *beta != 0.0) {
        c.attn_temp_scale = static_cast<float>(*beta);
        const int64_t length =
            f.get_int(key("attention.temperature_length")).value_or(c.yarn.factor > 0.0f ? c.yarn.ctx_orig : 0);
        if (!std::isfinite(c.attn_temp_scale) || c.attn_temp_scale < 0.0f || length <= 0 || length > (int64_t{1} << 24)) {
            return Status(ErrorCode::InvalidModel, "invalid " + key("attention.temperature_scale") + " / " +
                                                       key("attention.temperature_length"));
        }
        c.attn_temp_length = static_cast<int32_t>(length);
    }

    // Sliding-window attention blocks: the architecture's period, or the
    // file's own pattern (a period, or one flag per block).
    std::vector<bool> windowed(static_cast<size_t>(c.n_layers), false);
    const int64_t window = c.spec->swa_period > 0
                               ? f.get_int(key("attention.sliding_window")).value_or(c.spec->sliding_window_default) : 0;
    if (window < 0 || window > (int64_t{1} << 30)) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("attention.sliding_window"));
    }
    if (window > 0) {
        c.sliding_window = static_cast<int32_t>(window);
        c.rope_base_swa = static_cast<float>(f.get_float(key("rope.freq_base_swa")).value_or(c.spec->rope_base_swa));
        if (c.rope_base_swa == 0.0f && c.spec->swa_rope_unscaled) c.rope_base_swa = c.rope_base;
        int64_t period = c.spec->swa_period;
        const GgufValue* pattern = f.metadata(key("attention.sliding_window_pattern"));
        if (pattern != nullptr && pattern->type == GgufValue::Type::Array) {
            if (pattern->array_type != GgufValue::Type::Bool || pattern->array_size < static_cast<uint64_t>(c.n_layers)) {
                return Status(ErrorCode::InvalidModel, "invalid " + key("attention.sliding_window_pattern"));
            }
            for (size_t l = 0; l < windowed.size(); ++l) windowed[l] = pattern->array_data[l] != 0;
            period = 0;
        } else if (pattern != nullptr) {
            period = pattern->as_int().value_or(0);
            if (period <= 0) return Status(ErrorCode::InvalidModel, "invalid " + key("attention.sliding_window_pattern"));
        }
        for (int64_t l = 0; period > 0 && l < c.n_layers; ++l) {
            windowed[static_cast<size_t>(l)] = period == 1 || (l + 1) % period != 0;
        }
    }

    // Blocks without positional rotation (NoPE): the file's own list (true =
    // rotated), else the architecture's rules: a period, and the global
    // blocks of a model that has windowed ones.
    std::vector<bool> rotated(static_cast<size_t>(c.n_layers), true);
    if (const GgufValue* rp = f.metadata(key("attention.rope_pattern"))) {
        if (rp->type != GgufValue::Type::Array || rp->array_type != GgufValue::Type::Bool ||
            rp->array_size < static_cast<uint64_t>(c.n_layers)) {
            return Status(ErrorCode::InvalidModel, "invalid " + key("attention.rope_pattern"));
        }
        for (size_t l = 0; l < rotated.size(); ++l) rotated[l] = rp->array_data[l] != 0;
    } else {
        for (int32_t l = 0; l < c.n_layers; ++l) {
            const auto i = static_cast<size_t>(l);
            if (c.spec->nope_period > 0 && (l + 1) % c.spec->nope_period == 0) rotated[i] = false;
            if (c.spec->nope_global && c.sliding_window > 0 && !windowed[i]) rotated[i] = false;
        }
    }
    if (attends && (c.head_dim <= 0 || (c.rope_dim <= 0 && c.spec->rotary) || c.rope_dim > c.head_dim || c.rope_dim % 2 != 0)) {
        return Status(ErrorCode::InvalidModel, "invalid head / rope dimensions");
    }
    // The q·k scale: the file's, else 1/sqrt(head_dim) (computed as the
    // attention always computed it, so existing models keep their bits).
    // Latent attention scores the per-head [q_nope | q_pe] . [k_nope | k_pe]
    // the model was trained with, whatever width the absorbed form attends
    // over: 1/sqrt(nope_dim + rope_dim).
    if (const auto scale = f.get_float(key("attention.scale"))) {
        c.attn_scale = static_cast<float>(*scale);
        if (!(c.attn_scale > 0.0f) || !std::isfinite(c.attn_scale)) {
            return Status(ErrorCode::InvalidModel, "invalid " + key("attention.scale"));
        }
    } else if (c.spec->unscaled_scores) {
        c.attn_scale = 1.0f;  // the Q norm's weights carry the scale
    } else if (attends) {
        const int32_t scored = c.mla.kv_rank > 0 ? c.mla.nope_dim + c.rope_dim : c.head_dim;
        c.attn_scale = 1.0f / std::sqrt(static_cast<float>(scored));
    }
    if (qk_mscale != 1.0) c.attn_scale = static_cast<float>(c.attn_scale * qk_mscale * qk_mscale);
    // Soft caps, wherever the file states them (0 or absent: none).
    c.attn_softcap = static_cast<float>(f.get_float(key("attn_logit_softcapping")).value_or(0.0));
    c.final_softcap = static_cast<float>(f.get_float(key("final_logit_softcapping")).value_or(0.0));
    if (!(c.attn_softcap >= 0.0f) || !(c.final_softcap >= 0.0f) || !std::isfinite(c.attn_softcap) ||
        !std::isfinite(c.final_softcap)) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("attn_logit_softcapping / final_logit_softcapping"));
    }
    const TensorView* embd = f.tensor("token_embd.weight");
    if (embd == nullptr || embd->cols() != c.n_embd) {
        return Status(ErrorCode::InvalidModel, "missing or malformed token_embd.weight");
    }
    c.n_vocab = static_cast<int32_t>(embd->rows());
    // Command-R: the logits are multiplied by the file's scale.
    // MPT, DBRX, OLMo: queries, keys and values clamped after their projections.
    c.clamp_kqv = static_cast<float>(f.get_float(key("attention.clamp_kqv")).value_or(0.0));
    // ALiBi, for architectures whose row has it: the file's maximum bias, else the row's.
    if (c.spec->alibi_bias > 0.0f) {
        c.alibi_bias = static_cast<float>(f.get_float(key("attention.max_alibi_bias")).value_or(c.spec->alibi_bias));
    }
    if (!(c.logit_scale > 0.0f) || !std::isfinite(c.logit_scale) || !(c.clamp_kqv >= 0.0f) ||
        !std::isfinite(c.clamp_kqv) || !(c.alibi_bias >= 0.0f) || !std::isfinite(c.alibi_bias)) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("logit_scale / attention.clamp_kqv / max_alibi_bias"));
    }
    // Learned absolute positions (GPT-2, GPTBigCode): one embedding row per
    // position, added to the token embeddings.
    if (const TensorView* pos = f.tensor("position_embd.weight")) {
        if (pos->cols() != c.n_embd || pos->rows() <= 0 || pos->rows() > (int64_t{1} << 24)) {
            return Status(ErrorCode::InvalidModel, "malformed position_embd.weight");
        }
        if (f.tensor("token_embd_norm.weight") != nullptr || f.tensor("token_embd_norm.bias") != nullptr) {
            // The embedding norm would have to follow the position embeddings,
            // which the engine adds after it (no supported model has both).
            return Status(ErrorCode::Unsupported, "an embedding norm with learned position embeddings is not supported");
        }
        c.n_positions = static_cast<int32_t>(pos->rows());
    }
    if (c.mla.kv_rank > 0 && (c.alibi_bias > 0.0f || c.rope_dim == 0)) {
        return Status(ErrorCode::Unsupported, "latent attention without its RoPE key is not supported");
    }
    // xIELU (Apertus): four scalars per block, as global arrays of the
    // reference's parameters; the effective ones are computed here once.
    if (c.spec->ffn_activation == FfnActivation::XIelu) {
        const char* names[] = {"xielu.alpha_p", "xielu.alpha_n", "xielu.beta", "xielu.eps"};
        std::vector<double> values[4];
        for (int i = 0; i < 4; ++i) {
            const GgufValue* v = f.metadata(names[i]);
            if (v == nullptr || v->type != GgufValue::Type::Array || v->array_type != GgufValue::Type::F32 ||
                v->array_size < static_cast<uint64_t>(c.n_layers)) {
                return Status(ErrorCode::InvalidModel, std::string("missing or invalid ") + names[i] + " (one float per block)");
            }
            for (int32_t l = 0; l < c.n_layers; ++l) {
                float x = 0.0f;
                std::memcpy(&x, v->array_data + static_cast<size_t>(l) * sizeof(float), sizeof x);
                values[i].push_back(x);
            }
        }
        auto softplus = [](double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); };
        for (int32_t l = 0; l < c.n_layers; ++l) {
            const auto ul = static_cast<size_t>(l);
            XIeluScalars x;
            x.alpha_p = static_cast<float>(softplus(values[0][ul]));
            x.beta = static_cast<float>(values[2][ul]);
            x.alpha_n = static_cast<float>(values[2][ul] + softplus(values[1][ul]));
            x.eps = static_cast<float>(values[3][ul]);
            if (!std::isfinite(x.alpha_p) || !std::isfinite(x.alpha_n) || !std::isfinite(x.beta) || !std::isfinite(x.eps)) {
                return Status(ErrorCode::InvalidModel, "invalid xIELU scalars");
            }
            c.xielu.push_back(x);
        }
    }

    // Block plans, from the tensors present: a block with a DeltaNet
    // convolution, LFM2's short convolution or Kimi Delta Attention's
    // convolutions is recurrent, every other block is attention; a router
    // makes the FFN a mixture of experts.
    // Parallel blocks: the file's flag (GPT-NeoX, StableLM), else the architecture's.
    const GgufValue* parallel_key = f.metadata(key("use_parallel_residual"));
    const bool parallel = parallel_key != nullptr && parallel_key->as_bool() ? *parallel_key->as_bool()
                                                                             : c.spec->parallel_residual;
    c.blocks.resize(static_cast<size_t>(c.n_layers));
    auto block_clamp = [&](const char* suffix, int32_t l, float& out) -> Status {
        const GgufValue* v = f.metadata(key(suffix));
        if (v == nullptr) return Status::ok();
        if (v->type != GgufValue::Type::Array || v->array_type != GgufValue::Type::F32 ||
            v->array_size < static_cast<uint64_t>(c.n_layers)) {
            return Status(ErrorCode::InvalidModel, "invalid " + key(suffix));
        }
        std::memcpy(&out, v->array_data + static_cast<size_t>(l) * sizeof(float), sizeof(float));
        if (!std::isfinite(out) || out < 0.0f) return Status(ErrorCode::InvalidModel, "invalid " + key(suffix));
        if (out > 0.0f && c.spec->ffn_activation != FfnActivation::SiLU) {
            return Status(ErrorCode::Unsupported, "post-activation SwiGLU clamps require SiLU");
        }
        return Status::ok();
    };
    for (int32_t l = 0; l < c.n_layers; ++l) {
        BlockPlan& b = c.blocks[static_cast<size_t>(l)];
        b.n_head = block_heads[static_cast<size_t>(l)];
        b.n_head_kv = c.mla.kv_rank > 0 ? 1 : block_kv_heads[static_cast<size_t>(l)];
        b.n_ff = block_ff[static_cast<size_t>(l)];
        const int64_t q_dim = int64_t{b.n_head} * c.head_dim;
        LIYAB_RETURN_IF_ERROR(block_clamp("swiglu_clamp_exp", l, b.expert_swiglu_clamp));
        LIYAB_RETURN_IF_ERROR(block_clamp("swiglu_clamp_shexp", l, b.shared_swiglu_clamp));
        b.mixer = mixer_of(l);
        if ((b.mixer == MixerKind::Mamba || b.mixer == MixerKind::Mamba2) &&
            f.tensor(block_tensor(l, "attn_output.weight")) != nullptr) {
            return Status(ErrorCode::Unsupported, "block " + std::to_string(l) +
                              ": parallel attention and Mamba mixers in one block are not supported");
        }
        if (keeps_state(b.mixer)) {
            // Recurrent geometry comes from ssm.*, not attention head arrays.
        } else if (b.mixer == MixerKind::None) {
            // Tensor-planned FFN-only blocks have no mixer even when other
            // blocks supply a uniform positive attention geometry.
        } else if (b.n_head_kv == 0) {
            b.mixer = b.n_head == 0 ? MixerKind::None : MixerKind::Linear;
            if (f.tensor(block_tensor(l, "attn_q.weight")) != nullptr ||
                f.tensor(block_tensor(l, "attn_qkv.weight")) != nullptr ||
                f.tensor(block_tensor(l, "attn_k.weight")) != nullptr ||
                f.tensor(block_tensor(l, "attn_v.weight")) != nullptr ||
                f.tensor(block_tensor(l, "attn_q_norm.weight")) != nullptr ||
                f.tensor(block_tensor(l, "attn_k_norm.weight")) != nullptr ||
                (b.mixer == MixerKind::None && f.tensor(block_tensor(l, "attn_output.weight")) != nullptr)) {
                return Status(ErrorCode::InvalidModel, "attention tensors with zero KV heads in block " + std::to_string(l));
            }
        } else {
            if (b.n_head <= 0 || b.n_head % b.n_head_kv != 0) {
                return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) +
                                                           ": head_count must be a positive multiple of head_count_kv");
            }
            ++c.n_attn_layers;
            b.latent = f.tensor(block_tensor(l, "attn_kv_a_mqa.weight")) != nullptr;
            if (b.latent != (c.mla.kv_rank > 0)) {
                // The KV cache is laid out for one kind of attention.
                return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) + ": latent attention tensors and " +
                                                           key("attention.kv_lora_rank") + " disagree");
            }
            if (b.latent && (b.n_head != c.n_head || b.n_head_kv != 1)) {
                return Status(ErrorCode::Unsupported, "block " + std::to_string(l) +
                                                          ": per-block latent attention head counts are not supported");
            }
            b.latent_combined = f.tensor(block_tensor(l, "attn_kv_b.weight")) != nullptr;
            const bool split_key = f.tensor(block_tensor(l, "attn_k_b.weight")) != nullptr;
            const bool split_value = f.tensor(block_tensor(l, "attn_v_b.weight")) != nullptr;
            if (b.latent && (b.latent_combined ? (split_key || split_value) : (!split_key || !split_value))) {
                return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) +
                    ": latent attention needs either combined attn_kv_b or both split expansion tensors");
            }
            b.indexed = f.tensor(block_tensor(l, "indexer.attn_k.weight")) != nullptr;
            if (b.indexed && (!b.latent || c.index_heads == 0)) {
                return Status(ErrorCode::InvalidModel, "indexer tensors need latent attention and attention.indexer metadata");
            }
            if (c.index_heads > 0 && !c.spec->shared_kv) {
                const GgufValue* types = f.metadata(key("attention.indexer.types"));
                if (types != nullptr) {
                    if (types->type != GgufValue::Type::Array || types->array_type != GgufValue::Type::Bool ||
                        types->array_size < static_cast<uint64_t>(c.n_layers)) {
                        return Status(ErrorCode::InvalidModel, "invalid " + key("attention.indexer.types"));
                    }
                    b.index_reuse = types->array_data[l] == 0;
                }
                if (!b.latent || (b.index_reuse == b.indexed) || (b.index_reuse && l == 0)) {
                    return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) +
                        ": indexer tensors disagree with full/shared indexer metadata");
                }
            }
            const TensorView* wq = f.tensor(block_tensor(l, "attn_q.weight"));
            b.attn_gate = !b.latent && wq != nullptr && wq->rows() == 2 * q_dim;
            if (!b.latent && f.tensor(block_tensor(l, "attn_gate.weight")) != nullptr) {
                if (b.attn_gate) return Status(ErrorCode::InvalidModel, "both fused and separate attention gates in block " + std::to_string(l));
                b.attn_gate = true;
            }
            b.fused_qkv = wq == nullptr && f.tensor(block_tensor(l, "attn_qkv.weight")) != nullptr;
            b.windowed = windowed[static_cast<size_t>(l)];
            b.values_from_keys = !b.latent && !b.fused_qkv && f.tensor(block_tensor(l, "attn_k.weight")) != nullptr &&
                                 f.tensor(block_tensor(l, "attn_v.weight")) == nullptr;

            b.rope = rotated[static_cast<size_t>(l)];
            // Q/K norms over the whole projection (one weight per channel of
            // every head) rather than one head's weights shared by all heads.
            const TensorView* qn = f.tensor(block_tensor(l, "attn_q_norm.weight"));
            const TensorView* kn = f.tensor(block_tensor(l, "attn_k_norm.weight"));
            if (qn != nullptr || kn != nullptr) {
                const int64_t kv_dim = int64_t{b.n_head_kv} * c.head_dim;
                const bool q_whole = qn != nullptr && qn->n_dims == 1 && qn->elements() == q_dim && q_dim != c.head_dim;
                const bool k_whole = kn != nullptr && kn->n_dims == 1 && kn->elements() == kv_dim && kv_dim != c.head_dim;
                b.qk_norm_whole = q_whole || k_whole;
                auto valid_norm = [&](const TensorView* t, int64_t heads) {
                    if (t == nullptr) return true;
                    const int64_t width = b.qk_norm_whole ? heads * c.head_dim : c.head_dim;
                    return t->elements() == width || (!b.qk_norm_whole &&
                        c.spec->norm == NormKind::Layer && t->n_dims == 2 &&
                        t->ne[0] == c.head_dim && t->ne[1] == heads);
                };
                if (!valid_norm(qn, b.n_head) || !valid_norm(kn, b.n_head_kv)) {
                    return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) +
                                                               ": attn_q_norm / attn_k_norm span neither one head "
                                                               "nor the whole projection");
                }
                if (b.qk_norm_whole && c.spec->qk_norm == QkNormOrder::AfterRope) {
                    return Status(ErrorCode::Unsupported, "Q/K norms over the whole projection after RoPE");
                }
            }
            if (c.spec->shared_kv) {
                if (!b.windowed) {
                    return Status(ErrorCode::InvalidModel, "shared K=V attention needs a sliding window in every block");
                }
                if (f.tensor(block_tensor(l, "attn_sinks.weight")) != nullptr || f.tensor(block_tensor(l, "attn_sinks")) != nullptr) {
                    return Status(ErrorCode::Unsupported, "shared K=V attention needs learned attention sinks (MoE integration)");
                }
                const GgufValue* ratios = f.metadata(key("attention.compress_ratios"));
                if (ratios == nullptr || ratios->type != GgufValue::Type::Array ||
                    (ratios->array_type != GgufValue::Type::I32 && ratios->array_type != GgufValue::Type::U32) ||
                    ratios->array_size < static_cast<uint64_t>(c.n_layers)) {
                    return Status(ErrorCode::InvalidModel, "invalid " + key("attention.compress_ratios"));
                }
                uint32_t stride;
                std::memcpy(&stride, ratios->array_data + static_cast<size_t>(l) * sizeof(stride), sizeof(stride));
                const TensorView* projection = f.tensor(block_tensor(l, "attn_compressor_kv.weight"));
                if (stride > (1u << 20) || (stride > 0) != (projection != nullptr)) {
                    return Status(ErrorCode::InvalidModel, "compressor tensors disagree with attention.compress_ratios");
                }
                if (projection != nullptr) {
                    if (projection->cols() != c.n_embd ||
                        (projection->rows() != c.head_dim && projection->rows() != 2 * int64_t{c.head_dim})) {
                        return Status(ErrorCode::InvalidModel, "invalid compressor projected width");
                    }
                    b.compression = {c.head_dim, static_cast<int32_t>(stride), projection->rows() != c.head_dim};
                    b.indexed = f.tensor(block_tensor(l, "indexer_compressor_kv.weight")) != nullptr;
                    if (b.indexed != b.compression.overlap || (b.indexed && c.index_heads <= 0)) {
                        return Status(ErrorCode::InvalidModel, "overlapping compression needs its own indexer");
                    }
                    ++c.compressed_layers;
                }
            }
        }
        if (f.tensor(block_tensor(l, "ffn_gate_inp.weight")) == nullptr) {
            for (const char* name : {"ffn_up_exps.weight", "ffn_gate_exps.weight", "ffn_down_exps.weight",
                                     "ffn_gate_up_exps.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}) {
                if (f.tensor(block_tensor(l, name)) != nullptr) {
                    return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) +
                                      ": expert projections require ffn_gate_inp.weight");
                }
            }
        }
        b.parallel = parallel;
        b.laurel = f.tensor(block_tensor(l, "laurel_l.weight")) != nullptr;
        if (const TensorView* up = f.tensor(block_tensor(l, "ffn_up.weight"))) {  // dense FFN: one or two matrices
            const bool gate = f.tensor(block_tensor(l, "ffn_gate.weight")) != nullptr;
            b.fused_gate_up = !gate && up->rows() == 2 * int64_t{b.n_ff};
            // No gate and up as tall as the FFN: down(act(up)).
            if (!gate && up->rows() == int64_t{b.n_ff}) b.ffn = FfnKind::Ungated;
        } else if (f.tensor(block_tensor(l, "ffn_gate_inp.weight")) == nullptr &&
                   f.tensor(block_tensor(l, "ffn_gate.weight")) == nullptr &&
                   f.tensor(block_tensor(l, "ffn_down.weight")) == nullptr) {
            b.ffn = FfnKind::None;
            const std::string prefix = block_tensor(l, "ffn_");
            for (const TensorView& tensor : f.tensors()) {
                if (tensor.name.starts_with(prefix)) {
                    return Status(ErrorCode::InvalidModel, "tensor '" + std::string(tensor.name) + "' with zero FFN width");
                }
            }
        }
        if (f.tensor(block_tensor(l, "ffn_gate_inp.weight")) != nullptr) {
            if (c.n_expert <= 0) return Status(ErrorCode::InvalidModel, "MoE tensors without expert_count");
            b.ffn = FfnKind::Moe;
            // Each expert's gate and up projections as one [gate | up]
            // matrix (a converter option, Granite MoE), instead of two tensors.
            b.fused_gate_up = f.tensor(block_tensor(l, "ffn_gate_exps.weight")) == nullptr &&
                              f.tensor(block_tensor(l, "ffn_gate_up_exps.weight")) != nullptr;
            b.shared_expert = f.tensor(block_tensor(l, "ffn_up_shexp.weight")) != nullptr;
            // Without expert_feed_forward_length, the hidden size is the
            // down projections' input width (bind_weights() checks the rest).
            const TensorView* down = f.tensor(block_tensor(l, "ffn_down_exps.weight"));
            if (c.n_ff_expert == 0 && down != nullptr && down->ne[0] > 0 && down->ne[0] <= (int64_t{1} << 24)) {
                c.n_ff_expert = static_cast<int32_t>(down->ne[0]);
            }
        }
        if (b.ffn != FfnKind::None && b.ffn != FfnKind::Moe && b.n_ff <= 0) {
            return Status(ErrorCode::InvalidModel, "missing or invalid " + key("feed_forward_length"));
        }
        if (b.mixer == MixerKind::None && b.ffn == FfnKind::None &&
            (b.n_head != 0 || b.n_head_kv != 0 || b.n_ff != 0)) {
            return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) + " has no projections");
        }
        if (c.spec->ffn_activation == FfnActivation::XIelu && b.ffn != FfnKind::Ungated) {
            return Status(ErrorCode::Unsupported, "gated xIELU FFN in block " + std::to_string(l) + " is not implemented");
        }
        // Post-only norms: a block without the norm before its mixer (or FFN)
        // feeds it the residual stream as it is, and normalizes its output
        // instead. A block with neither norm is a damaged file, not a variant.
        auto has = [&](std::string_view suffix) {
            return !suffix.empty() && f.tensor(block_norm_tensor(f, l, suffix)) != nullptr;
        };
        // A parameterless LayerNorm still normalizes. In a parallel block
        // the FFN can share the mixer's norm without a second tensor.
        const bool parameterless = c.spec->norm == NormKind::Layer;
        b.norm_before_mixer = b.mixer != MixerKind::None &&
            (parameterless || has(c.spec->attn_norm) || (b.parallel && has(c.spec->ffn_norm)));
        b.norm_before_ffn = b.ffn != FfnKind::None &&
            (parameterless || has(c.spec->ffn_norm) || b.parallel ||
             (b.mixer == MixerKind::None && has(c.spec->attn_norm)));
        if ((b.mixer == MixerKind::None && (b.norm_before_mixer || has(c.spec->post_mixer_norm))) ||
            (b.ffn == FfnKind::None && (b.norm_before_ffn || has(c.spec->post_ffn_norm)))) {
            return Status(ErrorCode::InvalidModel, "normalization tensor for an absent branch in block " + std::to_string(l));
        }
        if (b.mixer != MixerKind::None && !b.norm_before_mixer && !has(c.spec->post_mixer_norm)) {
            return Status(ErrorCode::InvalidModel, "missing tensor '" + block_tensor(l, "attn_norm.weight") + "'");
        }
        if (b.ffn != FfnKind::None && !b.norm_before_ffn && !has(c.spec->post_ffn_norm)) {
            return Status(ErrorCode::InvalidModel,
                          "missing tensor '" + block_tensor(l, std::string(c.spec->ffn_norm) + ".weight") + "'");
        }
        // Attention sinks (a learned logit per query head joining the
        // softmax) are applied by the softmax attention; latent attention's
        // expanded form has no place for them.
        if (b.latent && f.tensor(block_tensor(l, "attn_sinks.weight")) != nullptr) {
            return Status(ErrorCode::Unsupported, "block " + std::to_string(l) +
                                                      ": attention sinks with latent attention are not supported");
        }
    }
    LIYAB_RETURN_IF_ERROR(describe_gemma_parts(f, c, key));
    c.n_recurrent_layers = static_cast<int32_t>(std::count_if(c.blocks.begin(), c.blocks.end(),
        [](const BlockPlan& b) { return keeps_state(b.mixer) || b.compression.stride > 0; }));
    auto has_mixer = [&](MixerKind kind) {
        return std::any_of(c.blocks.begin(), c.blocks.end(), [kind](const BlockPlan& b) { return b.mixer == kind; });
    };
    if (has_mixer(MixerKind::ShortConv)) {
        LIYAB_RETURN_IF_ERROR(require_int("shortconv.l_cache", c.shortconv_l_cache));
    }
    if (has_mixer(MixerKind::Kda)) {
        LIYAB_RETURN_IF_ERROR(require_int("ssm.conv_kernel", c.ssm_conv_kernel));
        LIYAB_RETURN_IF_ERROR(require_int("kda.head_dim", c.kda_head_dim));
        // Heads from the first Kda block's query projection (the same in every
        // block: bind_weights() checks each against it).
        for (int32_t l = 0; l < c.n_layers && c.kda_heads == 0; ++l) {
            if (c.blocks[static_cast<size_t>(l)].mixer != MixerKind::Kda) continue;
            const TensorView* wq = f.tensor(block_tensor(l, "attn_q.weight"));
            if (wq == nullptr || wq->rows() % c.kda_head_dim != 0 || wq->rows() == 0) {
                return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) +
                                                           ": Kimi Delta Attention without a whole number of heads "
                                                           "in attn_q (" + key("kda.head_dim") + ")");
            }
            c.kda_heads = static_cast<int32_t>(wq->rows() / c.kda_head_dim);
        }
        if (c.ssm_conv_kernel < 2) return Status(ErrorCode::InvalidModel, "invalid " + key("ssm.conv_kernel"));
    }
    if (c.n_expert > 0 && c.n_ff_expert <= 0 &&
        std::any_of(c.blocks.begin(), c.blocks.end(), [](const BlockPlan& b) { return b.ffn == FfnKind::Moe; })) {
        return Status(ErrorCode::InvalidModel, "missing or invalid " + key("expert_feed_forward_length"));
    }
    bool delta_net = false, mamba1 = false, mamba2 = false;
    for (const BlockPlan& b : c.blocks) {
        delta_net = delta_net || b.mixer == MixerKind::DeltaNet;
        mamba1 = mamba1 || b.mixer == MixerKind::Mamba;
        mamba2 = mamba2 || b.mixer == MixerKind::Mamba2;
    }
    if (int{delta_net} + int{mamba1} + int{mamba2} > 1) {
        // The ssm.* keys describe one kind of recurrent mixer.
        return Status(ErrorCode::Unsupported, "a model mixing Gated DeltaNet, Mamba-1 and Mamba-2 blocks is not supported");
    }
    if (mamba1 || mamba2) {
        // Mamba: in_proj gives [x | z] (Mamba-1) or [z | x B C | dt]
        // (Mamba-2); the convolution runs over x (Mamba-1) or [x | B | C].
        MambaGeometry& m = c.mamba;
        LIYAB_RETURN_IF_ERROR(require_int("ssm.conv_kernel", m.conv_kernel));
        LIYAB_RETURN_IF_ERROR(require_int("ssm.inner_size", m.d_inner));
        LIYAB_RETURN_IF_ERROR(require_int("ssm.state_size", m.d_state));
        if (mamba1) {
            LIYAB_RETURN_IF_ERROR(require_int("ssm.time_step_rank", m.dt_rank));
            const GgufValue* norms = f.metadata(key("ssm.dt_b_c_rms"));
            if (norms != nullptr && !norms->as_bool()) {
                return Status(ErrorCode::InvalidModel, "invalid " + key("ssm.dt_b_c_rms"));
            }
            m.dt_b_c_rms = norms != nullptr && *norms->as_bool();
        } else {
            LIYAB_RETURN_IF_ERROR(require_int("ssm.time_step_rank", m.n_heads));
            LIYAB_RETURN_IF_ERROR(require_int("ssm.group_count", m.n_groups));
            if (m.d_inner % m.n_heads != 0 || m.n_heads % m.n_groups != 0) {
                return Status(ErrorCode::InvalidModel, "inconsistent Mamba-2 (ssm.*) dimensions: heads must divide "
                                                       "inner_size, and groups the heads");
            }
        }
    }
    if (delta_net) {
        LIYAB_RETURN_IF_ERROR(require_int("ssm.conv_kernel", c.ssm_conv_kernel));
        LIYAB_RETURN_IF_ERROR(require_int("ssm.state_size", c.ssm_head_dim));
        LIYAB_RETURN_IF_ERROR(require_int("ssm.group_count", c.ssm_k_heads));
        LIYAB_RETURN_IF_ERROR(require_int("ssm.time_step_rank", c.ssm_v_heads));
        int32_t inner = 0;
        LIYAB_RETURN_IF_ERROR(require_int("ssm.inner_size", inner));
        if (inner != c.ssm_v_heads * c.ssm_head_dim || c.ssm_v_heads % c.ssm_k_heads != 0 || c.ssm_conv_kernel < 2) {
            return Status(ErrorCode::InvalidModel, "inconsistent DeltaNet (ssm.*) dimensions");
        }
    }
    return c;
}

}  // namespace liyab
