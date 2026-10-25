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
    {"llama", RopeLayout::Adjacent, "ffn_norm", false},
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
};

// The values of an integer array of the metadata; nullopt for another type.
std::optional<std::vector<int64_t>> int_array(const GgufValue& v) {
    if (v.type != GgufValue::Type::Array) return std::nullopt;
    std::vector<int64_t> out(static_cast<size_t>(v.array_size));
    for (size_t i = 0; i < out.size(); ++i) {
        const uint8_t* at = v.array_data;
        switch (v.array_type) {
            case GgufValue::Type::I32: { int32_t x; std::memcpy(&x, at + 4 * i, 4); out[i] = x; break; }
            case GgufValue::Type::U32: { uint32_t x; std::memcpy(&x, at + 4 * i, 4); out[i] = x; break; }
            case GgufValue::Type::I64: { int64_t x; std::memcpy(&x, at + 8 * i, 8); out[i] = x; break; }
            case GgufValue::Type::U64: { uint64_t x; std::memcpy(&x, at + 8 * i, 8); out[i] = static_cast<int64_t>(x); break; }
            default: return std::nullopt;
        }
    }
    return out;
}

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

std::string block_tensor(int32_t block, std::string_view suffix) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "blk.%d.%.*s", block, static_cast<int>(suffix.size()), suffix.data());
    return buf;
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
    // Mixture of experts: dense FFN size is optional when every FFN is MoE.
    c.n_expert = static_cast<int32_t>(f.get_int(key("expert_count")).value_or(0));
    if (c.n_expert > 0) {
        LIYAB_RETURN_IF_ERROR(require_int("expert_used_count", c.n_expert_used));
        LIYAB_RETURN_IF_ERROR(require_int("expert_feed_forward_length", c.n_ff_expert));
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
        // Group-limited routing (DeepSeek-V3: top-k only among the best
        // groups of experts) is not implemented: plain top-k would choose
        // other experts.
        const int64_t groups = f.get_int(key("expert_group_count")).value_or(1);
        if (groups > 1 && f.get_int(key("expert_group_used_count")).value_or(groups) < groups) {
            return Status(ErrorCode::Unsupported, "group-limited expert routing (" + key("expert_group_count") +
                                                      " " + std::to_string(groups) + ") is not supported yet");
        }
    }
    c.n_ff = static_cast<int32_t>(f.get_int(key("feed_forward_length")).value_or(0));
    if (const GgufValue* ff = f.metadata(key("feed_forward_length")); ff != nullptr && ff->type == GgufValue::Type::Array) {
        // One width per block (Gemma 3n's converter writes the list even when
        // every block has the same width): one value for all is what the
        // blocks run; differing widths are refused.
        auto widths = int_array(*ff);
        if (!widths || widths->size() != static_cast<size_t>(c.n_layers) || widths->front() <= 0 ||
            widths->front() > (int64_t{1} << 24)) {
            return Status(ErrorCode::InvalidModel, "invalid " + key("feed_forward_length"));
        }
        if (std::any_of(widths->begin(), widths->end(), [&](int64_t w) { return w != widths->front(); })) {
            return Status(ErrorCode::Unsupported, "blocks of different FFN widths (" + key("feed_forward_length") +
                                                      ") are not supported yet");
        }
        c.n_ff = static_cast<int32_t>(widths->front());
    }
    if (c.n_ff <= 0 && c.n_expert == 0) return Status(ErrorCode::InvalidModel, "missing or invalid " + key("feed_forward_length"));
    LIYAB_RETURN_IF_ERROR(require_int("attention.head_count", c.n_head));
    c.n_head_kv = static_cast<int32_t>(f.get_int(key("attention.head_count_kv")).value_or(c.n_head));
    c.head_dim = static_cast<int32_t>(f.get_int(key("attention.key_length")).value_or(c.n_embd / c.n_head));
    c.value_dim = c.head_dim;
    c.rope_dim = static_cast<int32_t>(f.get_int(key("rope.dimension_count")).value_or(c.head_dim));
    const auto value_length = f.get_int(key("attention.value_length"));
    if (const auto kv_rank = f.get_int(key("attention.kv_lora_rank")); kv_rank && *kv_rank > 0) {
        // Multi-head latent attention. The converter states the absorbed
        // geometry (one KV head of [k_pe | latent], values the latent) and
        // the per-head one apart (*_mla): files from before it split
        // attn_kv_b into attn_k_b / attn_v_b have neither.
        LatentAttention& m = c.mla;
        const auto key_mla = f.get_int(key("attention.key_length_mla"));
        const auto value_mla = f.get_int(key("attention.value_length_mla"));
        if (!key_mla || !value_mla) {
            return Status(ErrorCode::Unsupported,
                          "this " + c.arch + " file predates split attn_k_b / attn_v_b tensors (no " +
                              key("attention.key_length_mla") + "): convert the model again with a current converter");
        }
        m.kv_rank = static_cast<int32_t>(*kv_rank);
        m.q_rank = static_cast<int32_t>(f.get_int(key("attention.q_lora_rank")).value_or(0));
        m.nope_dim = static_cast<int32_t>(*key_mla - c.rope_dim);
        m.v_dim = static_cast<int32_t>(*value_mla);
        c.value_dim = static_cast<int32_t>(value_length.value_or(0));
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
    c.n_ctx_train = static_cast<int32_t>(f.get_int(key("context_length")).value_or(0));
    c.rms_eps = static_cast<float>(f.get_float(key("attention.layer_norm_rms_epsilon")).value_or(1e-5));
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
            y.truncate = truncate == nullptr || *truncate->as_bool();
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
            const double magnitude = attn_factor ? *attn_factor : log_mul ? 1.0 : 0.1 * ln_factor + 1.0;
            if (log_mul) qk_mscale = 1.0 + *log_mul * ln_factor;
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

    // Sliding-window attention blocks: the architecture's period, or the
    // file's own pattern (a period, or one flag per block).
    std::vector<bool> windowed(static_cast<size_t>(c.n_layers), false);
    const int64_t window = c.spec->swa_period > 0 ? f.get_int(key("attention.sliding_window")).value_or(0) : 0;
    if (window < 0 || window > (int64_t{1} << 30)) {
        return Status(ErrorCode::InvalidModel, "invalid " + key("attention.sliding_window"));
    }
    if (window > 0) {
        c.sliding_window = static_cast<int32_t>(window);
        c.rope_base_swa = static_cast<float>(f.get_float(key("rope.freq_base_swa")).value_or(c.spec->rope_base_swa));
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

    if (c.n_head_kv <= 0 || c.n_head % c.n_head_kv != 0) {
        return Status(ErrorCode::InvalidModel, "head_count must be a multiple of head_count_kv");
    }
    if (c.head_dim <= 0 || c.rope_dim <= 0 || c.rope_dim > c.head_dim || c.rope_dim % 2 != 0) {
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
    } else {
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

    // Block plans, from the tensors present: a block with a DeltaNet
    // convolution is recurrent, every other block is attention; a router
    // makes the FFN a mixture of experts.
    const int64_t q_dim = int64_t{c.n_head} * c.head_dim;
    c.blocks.resize(static_cast<size_t>(c.n_layers));
    for (int32_t l = 0; l < c.n_layers; ++l) {
        BlockPlan& b = c.blocks[static_cast<size_t>(l)];
        if (f.tensor(block_tensor(l, "ssm_conv1d.weight")) != nullptr) {
            b.mixer = MixerKind::DeltaNet;
        } else {
            ++c.n_attn_layers;
            b.latent = f.tensor(block_tensor(l, "attn_kv_a_mqa.weight")) != nullptr;
            if (b.latent != (c.mla.kv_rank > 0)) {
                // The KV cache is laid out for one kind of attention.
                return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) + ": latent attention tensors and " +
                                                           key("attention.kv_lora_rank") + " disagree");
            }
            const TensorView* wq = f.tensor(block_tensor(l, "attn_q.weight"));
            b.attn_gate = !b.latent && wq != nullptr && wq->rows() == 2 * q_dim;
            b.fused_qkv = wq == nullptr && f.tensor(block_tensor(l, "attn_qkv.weight")) != nullptr;
            b.windowed = windowed[static_cast<size_t>(l)];
            b.values_from_keys = !b.latent && !b.fused_qkv && f.tensor(block_tensor(l, "attn_k.weight")) != nullptr &&
                                 f.tensor(block_tensor(l, "attn_v.weight")) == nullptr;
        }
        b.laurel = f.tensor(block_tensor(l, "laurel_l.weight")) != nullptr;
        if (const TensorView* up = f.tensor(block_tensor(l, "ffn_up.weight"))) {  // dense FFN: one or two matrices
            b.fused_gate_up =
                f.tensor(block_tensor(l, "ffn_gate.weight")) == nullptr && up->rows() == 2 * int64_t{c.n_ff};
        }
        if (f.tensor(block_tensor(l, "ffn_gate_inp.weight")) != nullptr) {
            if (c.n_expert <= 0) return Status(ErrorCode::InvalidModel, "MoE tensors without expert_count");
            b.ffn = FfnKind::Moe;
            // Each expert's gate and up projections as one [gate | up]
            // matrix (a converter option, Granite MoE), instead of two tensors.
            b.fused_gate_up = f.tensor(block_tensor(l, "ffn_gate_exps.weight")) == nullptr &&
                              f.tensor(block_tensor(l, "ffn_gate_up_exps.weight")) != nullptr;
            b.shared_expert = f.tensor(block_tensor(l, "ffn_up_shexp.weight")) != nullptr;
        }
    }
    LIYAB_RETURN_IF_ERROR(describe_gemma_parts(f, c, key));
    if (c.n_attn_layers < c.n_layers) {
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
