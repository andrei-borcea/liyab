#include "core/transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>

#include "core/log.h"
#include "core/quant.h"
#include "core/thread_pool.h"

namespace liyab {

namespace {

void rmsnorm(const float* x, const float* w, float* out, int32_t n, float eps) {
    double ss = 0.0;
    for (int32_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const auto scale = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
    for (int32_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

std::string layer_name(int32_t layer, const char* suffix) {
    char buf[96];
    std::snprintf(buf, sizeof buf, "blk.%d.%s", layer, suffix);
    return buf;
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }  // ggml's threshold

// x / sqrt(sum(x^2) + eps): the L2 normalization of DeltaNet queries and keys.
void l2norm(float* x, int32_t n, float eps) {
    double ss = 0.0;
    for (int32_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const auto scale = static_cast<float>(1.0 / std::sqrt(ss + eps));
    for (int32_t i = 0; i < n; ++i) x[i] *= scale;
}

// Per-architecture facts that the tensors cannot tell. Everything else
// (block mixers, gates, biases, norms) is discovered from the file.
struct ArchTraits {
    std::string_view name;
    bool rope_neox;         // rotate pairs (i, i + rope_dim/2) instead of (2i, 2i + 1)
    bool moe_norm_weights;  // MoE top-k weights renormalized when the file does not say
};
constexpr ArchTraits kArchs[] = {
    {"llama", false, false},  {"mistral", false, false}, {"qwen2", true, false},     {"qwen3", true, false},
    {"qwen35", true, false},  {"qwen3moe", true, true},  {"qwen35moe", true, true},
};

// Adds the wall time of its scope to `total_ms` (Transformer::PhaseTimes).
class PhaseTimer {
public:
    explicit PhaseTimer(double& total_ms) : total_ms_(total_ms), start_(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() { stop(); }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
    // Ends the measurement early (later calls and the destructor add nothing).
    void stop() {
        if (stopped_) return;
        stopped_ = true;
        total_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    double& total_ms_;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};

// Expert `e` of a stacked expert tensor [cols, rows, n_expert] as a 2-D view.
TensorView expert_slice(const TensorView& t, int64_t e) {
    TensorView v = t;
    v.n_dims = 2;
    v.ne = {t.ne[0], t.ne[1], 1, 1};
    v.nbytes = t.row_bytes() * static_cast<size_t>(t.ne[1]);
    v.data = t.data + static_cast<size_t>(e) * v.nbytes;
    v.file_offset = t.file_offset + static_cast<uint64_t>(e) * v.nbytes;
    return v;
}


}  // namespace

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
std::string supported_architectures() {
    std::string out;
    for (const ArchTraits& a : kArchs) out += (out.empty() ? "" : ",") + std::string(a.name);
    return out;
}

Result<std::unique_ptr<Transformer>> Transformer::load(std::unique_ptr<MmapLoader> file,
                                                       const TransformerOptions& options) {
    if (!file) return Status(ErrorCode::InvalidArgument, "null model file");
    std::unique_ptr<Transformer> model(new Transformer());
    model->file_ = std::move(file);
    const MmapLoader& f = *model->file_;
    ModelConfig& c = model->config_;

    const auto arch = f.get_string("general.architecture");
    if (!arch) return Status(ErrorCode::InvalidModel, "not a model file: missing general.architecture");
    c.arch = std::string(*arch);
    const ArchTraits* traits = nullptr;
    for (const ArchTraits& a : kArchs) {
        if (a.name == c.arch) traits = &a;
    }
    if (traits == nullptr) {
        return Status(ErrorCode::Unsupported,
                      "architecture '" + c.arch + "' is not supported yet (supported: " + supported_architectures() + ")");
    }
    c.rope_neox = traits->rope_neox;

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
    model->expert_mass_ = options.expert_mass;
    model->max_experts_ = options.max_experts;
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
        c.moe_norm_weights = norm != nullptr && norm->as_bool() ? *norm->as_bool() : traits->moe_norm_weights;
        c.moe_weights_scale = static_cast<float>(f.get_float(key("expert_weights_scale")).value_or(1.0));
        if (c.moe_weights_scale == 0.0f) c.moe_weights_scale = 1.0f;
        if (c.n_expert_used > c.n_expert) return Status(ErrorCode::InvalidModel, "expert_used_count > expert_count");
    }
    c.n_ff = static_cast<int32_t>(f.get_int(key("feed_forward_length")).value_or(0));
    if (c.n_ff <= 0 && c.n_expert == 0) return Status(ErrorCode::InvalidModel, "missing or invalid " + key("feed_forward_length"));
    LIYAB_RETURN_IF_ERROR(require_int("attention.head_count", c.n_head));
    c.n_head_kv = static_cast<int32_t>(f.get_int(key("attention.head_count_kv")).value_or(c.n_head));
    c.head_dim = static_cast<int32_t>(f.get_int(key("attention.key_length")).value_or(c.n_embd / c.n_head));
    if (const auto vlen = f.get_int(key("attention.value_length")); vlen && *vlen != c.head_dim) {
        return Status(ErrorCode::Unsupported, "key_length != value_length is not supported");
    }
    c.rope_dim = static_cast<int32_t>(f.get_int(key("rope.dimension_count")).value_or(c.head_dim));
    c.n_ctx_train = static_cast<int32_t>(f.get_int(key("context_length")).value_or(0));
    c.rms_eps = static_cast<float>(f.get_float(key("attention.layer_norm_rms_epsilon")).value_or(1e-5));
    c.rope_base = static_cast<float>(f.get_float(key("rope.freq_base")).value_or(10000.0));
    if (const auto scaling = f.get_string(key("rope.scaling.type")); scaling && *scaling != "none") {
        if (*scaling != "linear") {
            return Status(ErrorCode::Unsupported, "rope scaling '" + std::string(*scaling) + "' is not supported");
        }
        c.rope_scale = static_cast<float>(f.get_float(key("rope.scaling.factor")).value_or(1.0));
        if (c.rope_scale <= 0.0f) return Status(ErrorCode::InvalidModel, "invalid rope scaling factor");
    }

    if (c.n_head_kv <= 0 || c.n_head % c.n_head_kv != 0) {
        return Status(ErrorCode::InvalidModel, "head_count must be a multiple of head_count_kv");
    }
    if (c.head_dim <= 0 || c.rope_dim <= 0 || c.rope_dim > c.head_dim || c.rope_dim % 2 != 0) {
        return Status(ErrorCode::InvalidModel, "invalid head / rope dimensions");
    }

    // Block mixers, from the tensors present: a block with a DeltaNet
    // convolution is recurrent, every other block is attention.
    c.mixers.resize(static_cast<size_t>(c.n_layers));
    for (int32_t l = 0; l < c.n_layers; ++l) {
        const bool recurrent = f.tensor(layer_name(l, "ssm_conv1d.weight")) != nullptr;
        c.mixers[static_cast<size_t>(l)] = recurrent ? MixerKind::DeltaNet : MixerKind::Attention;
        if (!recurrent) ++c.n_attn_layers;
    }
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

    LIYAB_RETURN_IF_ERROR(model->bind_weights());
    const Layer& first = model->layers_[0];
    const TensorView* ffn_weight = first.moe ? first.w[kUpExps] : first.w[kUp];
    LIYAB_LOG_INFO("bound %d blocks of %s weights (%s, head_dim %d, %d attention + %d DeltaNet blocks)", c.n_layers,
                   std::string(dtype_traits(ffn_weight->type).name).c_str(), c.arch.c_str(), c.head_dim,
                   c.n_attn_layers, c.n_layers - c.n_attn_layers);
    if (c.n_expert > 0) {
        LIYAB_LOG_INFO("MoE: %d experts, top-%d, expert ff %d, shared ff %d, %s gating%s", c.n_expert, c.n_expert_used,
                       c.n_ff_expert, c.n_ff_shared, c.moe_gating == MoeGating::Softmax ? "softmax" : "sigmoid",
                       c.moe_norm_weights ? ", renormalized" : "");
    }

    // Context and KV cache.
    model->context_length_ = options.context_length > 0
                                 ? options.context_length
                                 : (c.n_ctx_train > 0 ? std::min(c.n_ctx_train, 4096) : 4096);
    if (c.n_ctx_train > 0 && model->context_length_ > c.n_ctx_train) {
        LIYAB_LOG_WARN("context %d exceeds the model's training context %d", model->context_length_, c.n_ctx_train);
    }
    model->max_batch_ = std::max(1, options.max_batch);
    KvCacheConfig kv;
    // Only attention blocks have keys and values (at least one slot keeps the
    // cache well-formed for purely recurrent stacks).
    kv.n_layers = std::max(c.n_attn_layers, 1);
    kv.n_head_kv = c.n_head_kv;
    kv.head_dim = c.head_dim;
    kv.type = options.kv_type;
    kv.max_positions = model->context_length_;
    kv.window = options.sliding_window > 0 && options.sliding_window < model->context_length_ ? options.sliding_window : 0;
    kv.sink_tokens = std::clamp(options.sink_tokens, 0, kv.page_tokens);
    if (kv.type != KvCacheType::F16 && c.head_dim % quant::kBlock != 0) {
        LIYAB_LOG_WARN("head_dim %d is not a multiple of %d; using an F16 KV cache", c.head_dim, quant::kBlock);
        kv.type = KvCacheType::F16;
    }
    auto cache = KvCache::create(kv);
    if (!cache) return cache.status();
    model->kv_ = std::move(cache).value();

    // RoPE inverse frequencies, with optional per-pair factors (Llama 3.x).
    std::vector<float> factors(static_cast<size_t>(c.rope_dim / 2), 1.0f);
    if (const TensorView* rf = f.tensor("rope_freqs.weight")) {
        if (rf->elements() != c.rope_dim / 2 || rf->type != DType::F32) {
            return Status(ErrorCode::InvalidModel, "rope_freqs.weight has an unexpected shape");
        }
        quant::dequantize_row(DType::F32, rf->data, factors.data(), rf->elements());
    }
    model->inv_freq_.resize(factors.size());
    for (size_t i = 0; i < factors.size(); ++i) {
        const double freq = std::pow(static_cast<double>(c.rope_base), -2.0 * static_cast<double>(i) / c.rope_dim);
        model->inv_freq_[i] = static_cast<float>(freq / factors[i]);
    }

    if (c.n_expert > 0) {
        LIYAB_RETURN_IF_ERROR(
            model->attach_expert_store(options.expert_cache_bytes, options.memory_budget_bytes, options.requant_bits));
    }
    model->file_->configure_layers(c.n_layers);
    LIYAB_LOG_INFO("%s: %d layers, d=%d, ff=%d, heads=%d/%d x %d, vocab=%d, ctx=%d%s, kv=%s, %.1f KiB/token",
                   c.arch.c_str(), c.n_layers, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim, c.n_vocab,
                   model->context_length_, kv.window ? " (sliding)" : "",
                   std::string(dtype_traits(model->kv_->dtype()).name).c_str(),
                   static_cast<double>(KvCache::bytes_per_token(kv.n_layers, c.n_head_kv, c.head_dim, kv.type)) / 1024.0);
    if (c.hybrid()) {
        LIYAB_LOG_INFO("DeltaNet: %d recurrent blocks, %d/%d heads x %d, conv %d, state %.1f MiB", c.n_layers - c.n_attn_layers,
                       c.ssm_k_heads, c.ssm_v_heads, c.ssm_head_dim, c.ssm_conv_kernel,
                       static_cast<double>(model->recurrent_state_bytes()) / (1024.0 * 1024.0));
    }
    return model;
}

Status Transformer::bind_weights() {
    const MmapLoader& f = *file_;
    const ModelConfig& c = config_;
    const int64_t q_dim = int64_t{c.n_head} * c.head_dim;
    const int64_t kv_dim = int64_t{c.n_head_kv} * c.head_dim;

    auto matrix = [&](const std::string& name, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
        if (t->cols() != cols || t->rows() != rows) {
            return Status(ErrorCode::InvalidModel, "tensor '" + name + "' has shape [" + std::to_string(t->cols()) +
                                                       ", " + std::to_string(t->rows()) + "], expected [" +
                                                       std::to_string(cols) + ", " + std::to_string(rows) + "]");
        }
        out = t;
        return Status::ok();
    };
    auto vec = [&](const std::string& name, int64_t n, bool required, std::vector<float>& out) -> Status {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) {
            return required ? Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'") : Status::ok();
        }
        if (t->elements() != n) return Status(ErrorCode::InvalidModel, "tensor '" + name + "' has the wrong size");
        out.resize(static_cast<size_t>(n));
        quant::dequantize_row(t->type, t->data, out.data(), n);
        return Status::ok();
    };

    token_embd_ = f.tensor("token_embd.weight");
    if (token_embd_ == nullptr || token_embd_->cols() != c.n_embd) {
        return Status(ErrorCode::InvalidModel, "missing or malformed token_embd.weight");
    }
    config_.n_vocab = static_cast<int32_t>(token_embd_->rows());
    if (f.tensor("output.weight") != nullptr) {
        LIYAB_RETURN_IF_ERROR(matrix("output.weight", c.n_embd, config_.n_vocab, output_));
    } else {
        output_ = token_embd_;  // tied embeddings
    }
    LIYAB_RETURN_IF_ERROR(vec("output_norm.weight", c.n_embd, true, output_norm_));

    const int64_t dn_key = int64_t{c.ssm_k_heads} * c.ssm_head_dim;
    const int64_t dn_value = int64_t{c.ssm_v_heads} * c.ssm_head_dim;
    const int64_t conv_dim = 2 * dn_key + dn_value;

    layers_.resize(static_cast<size_t>(c.n_layers));
    int32_t kv_slots = 0;
    int32_t state_slots = 0;
    for (int32_t l = 0; l < c.n_layers; ++l) {
        Layer& L = layers_[static_cast<size_t>(l)];
        L.mixer = c.mixers[static_cast<size_t>(l)];
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_norm.weight"), c.n_embd, true, L.attn_norm));
        if (L.mixer == MixerKind::Attention) {
            L.kv_slot = kv_slots++;
            // A query projection twice as tall carries a per-head output gate: [q | gate].
            const TensorView* wq = f.tensor(layer_name(l, "attn_q.weight"));
            L.attn_gate = wq != nullptr && wq->rows() == 2 * q_dim;
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_q.weight"), c.n_embd, L.attn_gate ? 2 * q_dim : q_dim, L.w[kQ]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_k.weight"), c.n_embd, kv_dim, L.w[kK]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_v.weight"), c.n_embd, kv_dim, L.w[kV]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_output.weight"), q_dim, c.n_embd, L.w[kO]));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_q.bias"), q_dim, false, L.bq));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_k.bias"), kv_dim, false, L.bk));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_v.bias"), kv_dim, false, L.bv));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_q_norm.weight"), c.head_dim, false, L.q_norm));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_k_norm.weight"), c.head_dim, false, L.k_norm));
        } else {
            L.state_slot = state_slots++;
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_qkv.weight"), c.n_embd, conv_dim, L.w[kQkv]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_gate.weight"), c.n_embd, dn_value, L.w[kZ]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ssm_alpha.weight"), c.n_embd, c.ssm_v_heads, L.w[kAlpha]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ssm_beta.weight"), c.n_embd, c.ssm_v_heads, L.w[kBeta]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ssm_out.weight"), dn_value, c.n_embd, L.w[kSsmOut]));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "ssm_conv1d.weight"), conv_dim * c.ssm_conv_kernel, true, L.conv1d));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "ssm_a"), c.ssm_v_heads, true, L.ssm_a));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "ssm_dt.bias"), c.ssm_v_heads, true, L.ssm_dt));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "ssm_norm.weight"), c.ssm_head_dim, true, L.ssm_norm));
        }
        // The pre-FFN norm is named after the attention it follows in some families.
        const std::string ffn_norm = f.tensor(layer_name(l, "ffn_norm.weight")) != nullptr
                                         ? layer_name(l, "ffn_norm.weight")
                                         : layer_name(l, "post_attention_norm.weight");
        LIYAB_RETURN_IF_ERROR(vec(ffn_norm, c.n_embd, true, L.ffn_norm));
        // A router tensor makes the FFN a mixture of experts.
        L.moe = f.tensor(layer_name(l, "ffn_gate_inp.weight")) != nullptr;
        if (!L.moe) {
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_gate.weight"), c.n_embd, c.n_ff, L.w[kGate]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_up.weight"), c.n_embd, c.n_ff, L.w[kUp]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_down.weight"), c.n_ff, c.n_embd, L.w[kDown]));
            continue;
        }
        if (c.n_expert <= 0) return Status(ErrorCode::InvalidModel, "MoE tensors without expert_count");
        if (f.tensor(layer_name(l, "ffn_gate_up_exps.weight")) != nullptr) {
            return Status(ErrorCode::Unsupported, "fused ffn_gate_up_exps tensors are not supported yet");
        }
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_gate_inp.weight"), c.n_embd, c.n_expert, L.w[kRouter]));
        auto experts = [&](const char* suffix, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
            const std::string name = layer_name(l, suffix);
            const TensorView* t = f.tensor(name);
            if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
            if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != c.n_expert || t->ne[3] != 1) {
                return Status(ErrorCode::InvalidModel, "tensor '" + name + "' does not have the expert shape");
            }
            out = t;
            return Status::ok();
        };
        LIYAB_RETURN_IF_ERROR(experts("ffn_gate_exps.weight", c.n_embd, c.n_ff_expert, L.w[kGateExps]));
        LIYAB_RETURN_IF_ERROR(experts("ffn_up_exps.weight", c.n_embd, c.n_ff_expert, L.w[kUpExps]));
        LIYAB_RETURN_IF_ERROR(experts("ffn_down_exps.weight", c.n_ff_expert, c.n_embd, L.w[kDownExps]));
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "exp_probs_b.bias"), c.n_expert, false, L.expert_bias));
        if (f.tensor(layer_name(l, "ffn_up_shexp.weight")) != nullptr) {
            const int64_t sh = c.n_ff_shared > 0 ? c.n_ff_shared : f.tensor(layer_name(l, "ffn_up_shexp.weight"))->rows();
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_gate_shexp.weight"), c.n_embd, sh, L.w[kGateShared]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_up_shexp.weight"), c.n_embd, sh, L.w[kUpShared]));
            LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_down_shexp.weight"), sh, c.n_embd, L.w[kDownShared]));
            LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "ffn_gate_inp_shexp.weight"), c.n_embd, false, L.shared_gate));
        }
    }

    states_.resize(static_cast<size_t>(state_slots));
    for (RecurrentState& st : states_) {
        st.conv.assign(static_cast<size_t>(conv_dim) * static_cast<size_t>(c.ssm_conv_kernel - 1), 0.0f);
        st.ssm.assign(static_cast<size_t>(c.ssm_v_heads) * static_cast<size_t>(c.ssm_head_dim) *
                          static_cast<size_t>(c.ssm_head_dim),
                      0.0f);
    }
    return Status::ok();
}

size_t Transformer::recurrent_state_bytes() const noexcept {
    size_t bytes = 0;
    for (const RecurrentState& st : states_) bytes += (st.conv.size() + st.ssm.size()) * sizeof(float);
    return bytes;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
class Transformer::BlockWeights {
public:
    BlockWeights(Transformer& model, int32_t layer) : model_(model) {
        const Layer& L = model.layers_[static_cast<size_t>(layer)];
        for (size_t i = 0; i < kWeightRoles; ++i) {
            if (L.w[i] != nullptr) views_[i] = *L.w[i];
        }
        item_ = model.layer_source_ == nullptr ? -1
                : model.layer_items_.empty()  ? layer
                                              : model.layer_items_[static_cast<size_t>(layer)];
        if (item_ < 0) {
            model.file_->begin_layer(layer);  // mmap path: advance the prefetch window (if any)
            return;
        }
        auto base = model.layer_source_->acquire(item_);
        if (!base) {
            status_ = base.status();
            return;
        }
        acquired_ = true;
        // Every block tensor lies inside the block's byte range, so its slot
        // address is the slot base plus its offset within that range.
        const size_t range_begin = model.file_->layer_range(layer).first;
        for (size_t i = 0; i < kWeightRoles; ++i) {
            if (L.w[i] != nullptr) views_[i].data = base.value() + (views_[i].file_offset - range_begin);
        }
    }
    ~BlockWeights() {
        if (acquired_) model_.layer_source_->release(item_);
    }
    BlockWeights(const BlockWeights&) = delete;
    BlockWeights& operator=(const BlockWeights&) = delete;

    [[nodiscard]] const Status& status() const noexcept { return status_; }
    // True when the views point into a reused streaming slot rather than the
    // mapping. Backends that keep per-tensor device copies (keyed by address)
    // must not see such views: the next block reuses the same address.
    [[nodiscard]] bool streamed() const noexcept { return item_ >= 0 && !model_.layer_shared_; }
    // The block's matrix with role `role` (only roles the block has).
    const TensorView& operator[](WeightRole role) const { return views_[role]; }

private:
    Transformer& model_;
    int32_t item_ = -1;  // layer source item, -1: read in place
    bool acquired_ = false;
    Status status_;
    TensorView views_[kWeightRoles];
};

void Transformer::reset() noexcept {
    kv_->clear();
    for (RecurrentState& st : states_) {
        std::fill(st.conv.begin(), st.conv.end(), 0.0f);
        std::fill(st.ssm.begin(), st.ssm.end(), 0.0f);
    }
    n_past_ = 0;
}
void Transformer::rope(float* vec, int32_t n_heads, int32_t pos) const {
    const int32_t half = config_.rope_dim / 2;
    const float p = static_cast<float>(pos) / config_.rope_scale;
    for (int32_t h = 0; h < n_heads; ++h) {
        float* x = vec + static_cast<size_t>(h) * static_cast<size_t>(config_.head_dim);
        for (int32_t i = 0; i < half; ++i) {
            const float theta = p * inv_freq_[static_cast<size_t>(i)];
            const float cs = std::cos(theta);
            const float sn = std::sin(theta);
            const int32_t a = config_.rope_neox ? i : 2 * i;
            const int32_t b = config_.rope_neox ? i + half : 2 * i + 1;
            const float x0 = x[a];
            const float x1 = x[b];
            x[a] = x0 * cs - x1 * sn;
            x[b] = x0 * sn + x1 * cs;
        }
    }
}

void Transformer::attention(int32_t kv_slot, int32_t n, ThreadPool& pool, const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const int32_t hd = c.head_dim;
    const int32_t group = c.n_head / c.n_head_kv;
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(hd);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const DType kv_type = kv_->dtype();
    const bool quantized_kv = quant::is_block_quantized(kv_type);

    pool.parallel_for(int64_t{n} * c.n_head, [&](int64_t begin, int64_t end) {
        thread_local std::vector<float> scores;
        thread_local std::vector<quant::BlockQ8_0> q8;
        for (int64_t task = begin; task < end; ++task) {
            const auto t = static_cast<int32_t>(task / c.n_head);
            const auto h = static_cast<int32_t>(task % c.n_head);
            float* out = att_.data() + static_cast<size_t>(t) * q_dim + static_cast<size_t>(h) * hd;
            if (head_mask != nullptr && head_mask[h] == 0) {
                std::fill(out, out + hd, 0.0f);  // pruned head: contributes nothing
                continue;
            }
            const int32_t g = h / group;
            const int32_t pos = n_past_ + t;
            // Visible positions: attention sinks [0, sink_end) + [first, pos].
            const KvCache::Visible vis = kv_->visible(pos);
            const int32_t n_visible = vis.sink_end + (pos - vis.first + 1);
            auto position = [&](int32_t i) { return i < vis.sink_end ? i : vis.first + (i - vis.sink_end); };
            const float* q = q_.data() + static_cast<size_t>(t) * q_dim + static_cast<size_t>(h) * hd;

            // Quantized caches are scored with integer dot products against a
            // Q8_0 copy of the query (same kernels as the weight matmuls).
            if (quantized_kv) {
                q8.resize(static_cast<size_t>(hd / quant::kBlock));
                quant::quantize_row_q8_0(q, q8.data(), hd);
            }
            scores.resize(static_cast<size_t>(n_visible));
            float max_score = -INFINITY;
            for (int32_t i = 0; i < n_visible; ++i) {
                const uint8_t* k = kv_->k_row(kv_slot, position(i), g);
                const float s = (quantized_kv ? quant::dot_quantized(kv_type, k, q8.data(), nullptr, hd)
                                              : quant::dot_f16_f32(reinterpret_cast<const uint16_t*>(k), q, hd)) *
                                scale;
                scores[static_cast<size_t>(i)] = s;
                max_score = std::max(max_score, s);
            }
            float sum = 0.0f;
            for (float& s : scores) {
                s = std::exp(s - max_score);
                sum += s;
            }
            std::fill(out, out + hd, 0.0f);
            const float inv_sum = 1.0f / sum;
            for (int32_t i = 0; i < n_visible; ++i) {
                quant::axpy_row(kv_type, kv_->v_row(kv_slot, position(i), g), scores[static_cast<size_t>(i)] * inv_sum,
                                out, hd);
            }
        }
    });
}

Status Transformer::matmul(const Route& route, Backend* backend, const TensorView& w, const float* x, float* y,
                           int32_t n) {
    Status s = backend->matmul(w, x, y, n);
    if (s.code() == ErrorCode::Unsupported && backend != route.cpu) s = route.cpu->matmul(w, x, y, n);
    return s;
}

Status Transformer::matmul_group(const Route& route, Backend* backend, std::span<const TensorView* const> ws,
                                 const float* x, std::span<float* const> ys, int32_t n) {
    Status s = backend->matmul_group(ws, x, ys, n);
    if (s.code() == ErrorCode::Unsupported && backend != route.cpu) s = route.cpu->matmul_group(ws, x, ys, n);
    return s;
}

Status Transformer::compute_logits(const Route& route, size_t first_row, int32_t rows) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const PhaseTimer timer(phases_.lm_head);
    file_->begin_layer(c.n_layers);  // output head slot of the prefetch window
    for (int32_t r = 0; r < rows; ++r) {
        rmsnorm(x_.data() + (first_row + static_cast<size_t>(r)) * d, output_norm_.data(),
                xb_.data() + static_cast<size_t>(r) * d, c.n_embd, c.rms_eps);
    }
    logits_.resize(static_cast<size_t>(rows) * static_cast<size_t>(c.n_vocab));
    return matmul(route, route.ffn, *output_, xb_.data(), logits_.data(), rows);
}

Status Transformer::propagate_kv(const Route& route, int32_t from, int32_t pos) {
    const ModelConfig& c = config_;
    // Only reached for attention-only stacks: early exit is disabled for hybrid models.
    for (int32_t l = from; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        Backend* backend = w.streamed() ? route.cpu : route.attention;  // see BlockWeights::streamed()
        rmsnorm(x_.data(), L.attn_norm.data(), xb_.data(), c.n_embd, c.rms_eps);
        LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kK], xb_.data(), k_.data(), 1));
        LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kV], xb_.data(), v_.data(), 1));
        for (size_t i = 0; i < L.bk.size(); ++i) k_[i] += L.bk[i];
        for (size_t i = 0; i < L.bv.size(); ++i) v_[i] += L.bv[i];
        if (!L.k_norm.empty()) {
            for (int32_t h = 0; h < c.n_head_kv; ++h) {
                float* kh = k_.data() + static_cast<size_t>(h) * c.head_dim;
                rmsnorm(kh, L.k_norm.data(), kh, c.head_dim, c.rms_eps);
            }
        }
        rope(k_.data(), c.n_head_kv, pos);
        kv_->store(L.kv_slot, pos, k_.data(), v_.data());
    }
    return Status::ok();
}

Status Transformer::attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                    ThreadPool& pool, const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(c.head_dim);
    const size_t kv_dim = static_cast<size_t>(c.n_head_kv) * static_cast<size_t>(c.head_dim);
    const size_t hd = static_cast<size_t>(c.head_dim);

    // Gated blocks project [q_h | gate_h] per head into qg_; split it afterwards.
    if (L.attn_gate) qg_.resize(un * 2 * q_dim);
    {
        const TensorView* qkv[] = {&w[kQ], &w[kK], &w[kV]};
        float* outs[] = {L.attn_gate ? qg_.data() : q_.data(), k_.data(), v_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, qkv, xb_.data(), outs, n));
    }
    if (L.attn_gate) {
        gate_.resize(un * q_dim);
        for (size_t t = 0; t < un; ++t) {
            for (size_t h = 0; h < static_cast<size_t>(c.n_head); ++h) {
                const float* src = qg_.data() + (t * static_cast<size_t>(c.n_head) + h) * 2 * hd;
                std::copy(src, src + hd, q_.data() + t * q_dim + h * hd);
                std::copy(src + hd, src + 2 * hd, gate_.data() + t * q_dim + h * hd);
            }
        }
    }
    for (size_t t = 0; t < un; ++t) {
        float* q = q_.data() + t * q_dim;
        float* k = k_.data() + t * kv_dim;
        float* v = v_.data() + t * kv_dim;
        for (size_t i = 0; i < L.bq.size(); ++i) q[i] += L.bq[i];
        for (size_t i = 0; i < L.bk.size(); ++i) k[i] += L.bk[i];
        for (size_t i = 0; i < L.bv.size(); ++i) v[i] += L.bv[i];
        if (!L.q_norm.empty()) {
            for (int32_t h = 0; h < c.n_head; ++h) {
                float* qh = q + static_cast<size_t>(h) * hd;
                rmsnorm(qh, L.q_norm.data(), qh, c.head_dim, c.rms_eps);
            }
        }
        if (!L.k_norm.empty()) {
            for (int32_t h = 0; h < c.n_head_kv; ++h) {
                float* kh = k + static_cast<size_t>(h) * hd;
                rmsnorm(kh, L.k_norm.data(), kh, c.head_dim, c.rms_eps);
            }
        }
        const int32_t pos = n_past_ + static_cast<int32_t>(t);
        rope(q, c.n_head, pos);
        rope(k, c.n_head_kv, pos);
        kv_->store(L.kv_slot, pos, k, v);
    }
    attention(L.kv_slot, n, pool, head_mask);
    if (L.attn_gate) {
        for (size_t i = 0; i < un * q_dim; ++i) att_[i] *= sigmoid(gate_[i]);
    }
    return matmul(route, route.attention, w[kO], att_.data(), xb_.data(), n);
}

// Gated DeltaNet (Qwen3.5 / Qwen3-Next linear attention), token by token:
//   [q|k|v] = SiLU(causal_conv1d(W_qkv x)),  q, k L2-normalized, q scaled by 1/sqrt(d)
//   decay = exp(softplus(W_a x + dt_bias) * A),  b = sigmoid(W_b x)
//   S = decay * S;  S += b * (v - S k) k^T;  o = S q
//   out = W_out (RMSNorm(o) * SiLU(W_z x))      (per value head)
// Key/query heads are shared by value heads modulo their count (ggml's layout).
Status Transformer::delta_net_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                    ThreadPool& pool) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    RecurrentState& st = states_[static_cast<size_t>(L.state_slot)];
    const auto un = static_cast<size_t>(n);
    const auto hd = static_cast<size_t>(c.ssm_head_dim);
    const auto hk = static_cast<size_t>(c.ssm_k_heads);
    const auto hv = static_cast<size_t>(c.ssm_v_heads);
    const size_t key_dim = hk * hd;
    const size_t value_dim = hv * hd;
    const size_t conv_dim = 2 * key_dim + value_dim;
    const auto taps = static_cast<size_t>(c.ssm_conv_kernel);
    const size_t history = taps - 1;

    mix_.resize(un * conv_dim);
    z_.resize(un * value_dim);
    alpha_.resize(un * hv);
    beta_.resize(un * hv);
    dn_.resize(un * value_dim);
    {
        const TensorView* proj[] = {&w[kQkv], &w[kZ], &w[kAlpha], &w[kBeta]};
        float* outs[] = {mix_.data(), z_.data(), alpha_.data(), beta_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, proj, xb_.data(), outs, n));
    }

    const float q_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    for (size_t t = 0; t < un; ++t) {
        float* m = mix_.data() + t * conv_dim;
        // Causal depthwise convolution over [history, current], then SiLU; the
        // history shifts by one input.
        for (size_t ch = 0; ch < conv_dim; ++ch) {
            float* hist = st.conv.data() + ch * history;
            const float* tap = L.conv1d.data() + ch * taps;
            float acc = m[ch] * tap[history];
            for (size_t j = 0; j < history; ++j) acc += hist[j] * tap[j];
            for (size_t j = 0; j + 1 < history; ++j) hist[j] = hist[j + 1];
            hist[history - 1] = m[ch];
            m[ch] = silu(acc);
        }
        for (size_t h = 0; h < 2 * hk; ++h) l2norm(m + h * hd, static_cast<int32_t>(hd), c.rms_eps);

        const float* a = alpha_.data() + t * hv;
        const float* b = beta_.data() + t * hv;
        float* out = dn_.data() + t * value_dim;
        pool.parallel_for(static_cast<int64_t>(hv), [&](int64_t begin, int64_t end) {
            for (auto h = static_cast<size_t>(begin); h < static_cast<size_t>(end); ++h) {
                const float* q = m + (h % hk) * hd;
                const float* k = m + key_dim + (h % hk) * hd;
                const float* v = m + 2 * key_dim + h * hd;
                const float decay = std::exp(softplus(a[h] + L.ssm_dt[h]) * L.ssm_a[h]);
                const float strength = sigmoid(b[h]);
                float* S = st.ssm.data() + h * hd * hd;
                float* o = out + h * hd;
                // Row j of S maps keys to value component j, so each row is
                // decayed, corrected and read independently in two passes.
                for (size_t j = 0; j < hd; ++j) {
                    float* row = S + j * hd;
                    float sk = 0.0f;
                    for (size_t i = 0; i < hd; ++i) {
                        row[i] *= decay;
                        sk += row[i] * k[i];
                    }
                    const float delta = (v[j] - sk) * strength;
                    float oq = 0.0f;
                    for (size_t i = 0; i < hd; ++i) {
                        row[i] += k[i] * delta;
                        oq += row[i] * q[i];
                    }
                    o[j] = oq * q_scale;
                }
                // Gated RMSNorm of the head output.
                const float* z = z_.data() + t * value_dim + h * hd;
                rmsnorm(o, L.ssm_norm.data(), o, static_cast<int32_t>(hd), c.rms_eps);
                for (size_t j = 0; j < hd; ++j) o[j] *= silu(z[j]);
            }
        });
    }
    return matmul(route, route.attention, w[kSsmOut], dn_.data(), xb_.data(), n);
}

// Expert streaming: the routed experts stay on storage and are read on
// demand into a fixed RAM cache; every other tensor stays resident in the
// mapping (paged in now, never swept by the streaming window).
Status Transformer::attach_expert_store(int64_t budget_option, uint64_t memory_budget, int32_t requant_bits) {
    if (budget_option == 0) return Status::ok();
    if (requant_bits != 0 && requant_bits != 4 && requant_bits != 5) {
        return Status(ErrorCode::InvalidArgument, "requant_bits must be 0, 4 or 5");
    }
    const ModelConfig& c = config_;
    auto is_expert = [](const TensorView& t) {
        const std::string_view n = t.name;
        return n.size() > 13 && n.substr(n.size() - 12) == "_exps.weight";
    };
    size_t expert_bytes = 0;
    for (const TensorView& t : file_->tensors()) {
        if (is_expert(t)) expert_bytes += t.nbytes;
    }
    const uint64_t available = usable_memory_bytes(memory_budget);
    // Automatic mode streams only when the model does not fit.
    if (budget_option < 0 &&
        (available == 0 || static_cast<double>(file_->file_size()) <= 0.8 * static_cast<double>(available))) {
        return Status::ok();
    }

    // Optional lossy conversion of the resident Q8_0 matrices (the UD quants
    // keep attention and DeltaNet projections in Q8_0, ~2/3 of the bytes read
    // per token): fewer bytes per token, and the freed memory goes to the
    // expert cache sized below. The token embedding is skipped: decode reads
    // one row of it per token.
    size_t saved = 0;
    if (requant_bits > 0) {
        auto requantized = file_->requantize(
            [&](const TensorView& t) {
                return !is_expert(t) && t.type == DType::Q8_0 && t.n_dims == 2 && t.rows() > 1 &&
                       t.cols() % quant::kSuperBlock == 0 && t.name != "token_embd.weight";
            },
            requant_bits == 4 ? DType::Q4_K : DType::Q5_K);
        if (!requantized) return requantized.status();
        saved = requantized.value();
        LIYAB_LOG_INFO("resident Q8_0 matrices requantized to %s: %.2f GiB saved", requant_bits == 4 ? "Q4_K" : "Q5_K",
                       static_cast<double>(saved) / (1024.0 * 1024.0 * 1024.0));
    }
    const size_t resident_bytes = file_->file_size() - expert_bytes - saved;
    size_t budget = 0;
    if (budget_option > 0) {
        budget = static_cast<size_t>(budget_option);
    } else {
        // Keep room for the KV cache, activations and the rest of the app:
        // 1 GiB of free RAM, or 512 MiB inside an explicit budget (the
        // embedder already left its own headroom below the platform's cap).
        const size_t margin = memory_budget > 0 ? size_t{512} << 20 : size_t{1} << 30;
        budget = available > resident_bytes + margin ? static_cast<size_t>(available) - resident_bytes - margin : 0;
    }
    std::vector<std::array<const TensorView*, 3>> experts(static_cast<size_t>(c.n_layers));
    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        if (L.moe) experts[static_cast<size_t>(l)] = {L.w[kGateExps], L.w[kUpExps], L.w[kDownExps]};
    }
    auto store = ExpertStore::create(*file_, std::move(experts), c.n_expert, budget, 4);
    if (!store) return store.status();
    expert_store_ = std::move(store).value();
    file_->keep_resident([&](const TensorView& t) { return !is_expert(t) && !file_->converted(t); });
    LIYAB_LOG_INFO("expert streaming: %.2f GiB resident weights, %.2f GiB of experts on storage",
                   static_cast<double>(resident_bytes) / (1024.0 * 1024.0 * 1024.0),
                   static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0));
    return Status::ok();
}

void Transformer::predict_experts(int32_t layer, int32_t n, const Route& route) {
    if (expert_store_ == nullptr || layer >= config_.n_layers) return;
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    if (!L.moe) return;
    const PhaseTimer timer(phases_.router);
    const auto d = static_cast<size_t>(c.n_embd);
    const auto E = static_cast<size_t>(c.n_expert);
    const auto un = static_cast<size_t>(n);
    pred_in_.resize(un * d);
    pred_logits_.resize(un * E);
    for (size_t t = 0; t < un; ++t) {
        rmsnorm(x_.data() + t * d, L.ffn_norm.data(), pred_in_.data() + t * d, c.n_embd, c.rms_eps);
    }
    if (!route.cpu->matmul(*L.w[kRouter], pred_in_.data(), pred_logits_.data(), n).is_ok()) return;
    std::vector<int32_t> order(E);
    std::vector<int32_t> picks;
    const bool cut = expert_mass_ < 1.0f || (max_experts_ > 0 && max_experts_ < c.n_expert_used);
    std::vector<float> probs(cut ? E : 0);
    for (size_t t = 0; t < un; ++t) {
        const float* logits = pred_logits_.data() + t * E;
        auto score = [&](int32_t e) {  // softmax and sigmoid are monotonic; the selection bias is added on top
            const float p = c.moe_gating == MoeGating::Softmax ? logits[e] : sigmoid(logits[e]);
            return p + (L.expert_bias.empty() ? 0.0f : L.expert_bias[static_cast<size_t>(e)]);
        };
        for (size_t e = 0; e < E; ++e) order[e] = static_cast<int32_t>(e);
        const auto k = static_cast<std::ptrdiff_t>(c.n_expert_used);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int32_t a, int32_t b) { return score(a) > score(b); });
        size_t kept = static_cast<size_t>(k);
        if (cut) {  // guess only the experts the router would keep (unnormalized: ratios suffice)
            float sum = 0.0f;
            const float top = logits[order[0]];
            for (std::ptrdiff_t i = 0; i < k; ++i) {
                const auto e = static_cast<size_t>(order[static_cast<size_t>(i)]);
                sum += probs[e] = c.moe_gating == MoeGating::Softmax ? std::exp(logits[e] - top) : sigmoid(logits[e]);
            }
            kept = experts_to_run(probs, order, sum);
        }
        picks.insert(picks.end(), order.begin(), order.begin() + static_cast<std::ptrdiff_t>(kept));
    }
    std::sort(picks.begin(), picks.end());
    picks.erase(std::unique(picks.begin(), picks.end()), picks.end());
    expert_store_->prefetch(layer, picks, true);
    predicted_ = std::move(picks);
    predicted_layer_ = layer;
}

Status Transformer::dense_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                              FfnMatmulHook* hook) {
    const ModelConfig& c = config_;
    const size_t rows = static_cast<size_t>(n) * static_cast<size_t>(c.n_ff);
    hb_.resize(rows);
    hb2_.resize(rows);
    // FFN projections: an experimental override first, else the routed backend.
    auto project = [&](FfnProjection p, const TensorView& m, const float* x, float* y) -> Status {
        if (hook != nullptr && hook->ffn_matmul(layer, p, x, y, n)) return Status::ok();
        return matmul(route, route.ffn, m, x, y, n);
    };
    if (hook != nullptr) {
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Gate, w[kGate], xb_.data(), hb_.data()));
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Up, w[kUp], xb_.data(), hb2_.data()));
    } else {
        const TensorView* gate_up[] = {&w[kGate], &w[kUp]};
        float* outs[] = {hb_.data(), hb2_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.ffn, gate_up, xb_.data(), outs, n));
    }
    for (size_t i = 0; i < rows; ++i) hb_[i] = silu(hb_[i]) * hb2_[i];
    return project(FfnProjection::Down, w[kDown], hb_.data(), xb_.data());
}

size_t Transformer::experts_to_run(std::span<const float> probs, std::span<const int32_t> order, float top_sum) const {
    const auto K = static_cast<size_t>(config_.n_expert_used);
    const size_t cap = max_experts_ > 0 ? std::min(K, static_cast<size_t>(max_experts_)) : K;
    if (expert_mass_ >= 1.0f) return cap;
    float covered = 0.0f;
    for (size_t k = 0; k < cap; ++k) {
        covered += probs[static_cast<size_t>(order[k])];
        if (covered >= expert_mass_ * top_sum) return k + 1;
    }
    return cap;
}

// Mixture of experts, as llama.cpp's build_moe_ffn: router probabilities
// (softmax or sigmoid), top-k selection (optionally on probability + bias),
// optional renormalization and scale; each selected expert is a SwiGLU FFN.
// Tokens are grouped by expert so a batch reads every expert once. A shared
// expert, optionally scaled by sigmoid(shared_gate . x), is added on top.
Status Transformer::moe_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto d = static_cast<size_t>(c.n_embd);
    const auto E = static_cast<size_t>(c.n_expert);
    const auto K = static_cast<size_t>(c.n_expert_used);
    const auto ff = static_cast<size_t>(c.n_ff_expert);

    PhaseTimer router_timer(phases_.router);
    router_.resize(un * E);
    LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kRouter], xb_.data(), router_.data(), n));

    // Routing: per expert, the tokens that selected it, with the gate weight
    // and the expert's rank in that token's top-k.
    struct Use {
        int32_t token;
        int32_t rank;
        float weight;
    };
    std::vector<std::vector<Use>> assigned(E);
    std::vector<float> probs(E);
    std::vector<int32_t> order(E);
    experts_kept_.assign(un, static_cast<int32_t>(K));
    for (size_t t = 0; t < un; ++t) {
        const float* logits = router_.data() + t * E;
        if (c.moe_gating == MoeGating::Softmax) {
            const float mx = *std::max_element(logits, logits + E);
            float sum = 0.0f;
            for (size_t e = 0; e < E; ++e) sum += probs[e] = std::exp(logits[e] - mx);
            for (float& p : probs) p /= sum;
        } else {
            for (size_t e = 0; e < E; ++e) probs[e] = sigmoid(logits[e]);
        }
        auto selection = [&](int32_t e) {
            return probs[static_cast<size_t>(e)] + (L.expert_bias.empty() ? 0.0f : L.expert_bias[static_cast<size_t>(e)]);
        };
        for (size_t e = 0; e < E; ++e) order[e] = static_cast<int32_t>(e);
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(K), order.end(),
                          [&](int32_t a, int32_t b) { return selection(a) > selection(b) || (selection(a) == selection(b) && a < b); });
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) sum += probs[static_cast<size_t>(order[k])];
        const size_t kept = experts_to_run(probs, order, sum);
        experts_kept_[t] = static_cast<int32_t>(kept);
        if (kept < K) {
            sum = 0.0f;
            for (size_t k = 0; k < kept; ++k) sum += probs[static_cast<size_t>(order[k])];
        }
        const float norm = c.moe_norm_weights ? 1.0f / std::max(sum, 6.103515625e-5f) : 1.0f;
        for (size_t k = 0; k < kept; ++k) {
            const auto e = static_cast<size_t>(order[k]);
            assigned[e].push_back(
                {static_cast<int32_t>(t), static_cast<int32_t>(k), probs[e] * norm * c.moe_weights_scale});
        }
    }

    // With expert streaming, queue every chosen expert now (missed
    // predictions jump the queue) and compute cached ones first, so reads
    // overlap the work on the others.
    std::vector<int32_t> chosen;
    for (size_t e = 0; e < E; ++e) {
        if (!assigned[e].empty()) chosen.push_back(static_cast<int32_t>(e));
    }
    if (expert_store_ != nullptr) {
        if (predicted_layer_ == layer) {  // both lists are sorted
            predictions_.predicted += predicted_.size();
            for (size_t i = 0, j = 0; i < predicted_.size() && j < chosen.size();) {
                if (predicted_[i] == chosen[j]) ++predictions_.used;
                if (predicted_[i] <= chosen[j]) ++i; else ++j;
            }
            predicted_layer_ = -1;
        }
        expert_store_->prefetch(layer, chosen, false);
        std::stable_partition(chosen.begin(), chosen.end(),
                              [&](int32_t e) { return expert_store_->ready(layer, e); });
    }
    router_timer.stop();
    // The next block's guess goes behind this block's own reads in the queue.
    if (expert_store_ != nullptr) predict_experts(layer + 1, n, route);

    PhaseTimer experts_timer(phases_.experts);
    moe_out_.assign(un * d, 0.0f);
    ranked_out_.resize(un * K * d);
    // Experts run in waves: the next expert (waiting for it if needed) plus
    // every following one already in RAM, as one batched pass for gate/up and
    // one for down, so the CPU pays two thread fork/joins per wave instead of
    // two per expert. Expert slices are transient views: they run on the CPU,
    // which reads them in place.
    constexpr size_t kMaxWave = 32;  // bounds the cache slots one wave pins
    std::vector<int32_t> wave;
    std::vector<std::array<TensorView, 3>> views;
    std::vector<size_t> first_row;  // per wave expert: its first row in the wave's scratch
    std::vector<Backend::MatmulItem> items;
    auto release_wave = [&] {
        if (expert_store_ != nullptr) {
            for (const int32_t ex : wave) expert_store_->release(layer, ex);
        }
    };
    for (size_t next = 0; next < chosen.size();) {
        wave.clear();
        views.clear();
        first_row.clear();
        size_t rows_total = 0;
        do {
            const int32_t ex = chosen[next++];
            if (expert_store_ != nullptr) {
                auto loaded = expert_store_->acquire(layer, ex);
                if (!loaded) {
                    release_wave();
                    return loaded.status();
                }
                views.push_back(*loaded);
            } else {
                views.push_back({expert_slice(w[kGateExps], ex), expert_slice(w[kUpExps], ex),
                                 expert_slice(w[kDownExps], ex)});
            }
            wave.push_back(ex);
            first_row.push_back(rows_total);
            rows_total += assigned[static_cast<size_t>(ex)].size();
        } while (next < chosen.size() && wave.size() < kMaxWave &&
                 (expert_store_ == nullptr || expert_store_->ready(layer, chosen[next])));

        // Inputs: an expert every token chose reads xb_ itself (always the
        // case in decode), so all of them share one quantized copy.
        ein_.resize(rows_total * d);
        eh_.resize(rows_total * ff);
        eh2_.resize(rows_total * ff);
        eout_.resize(rows_total * d);
        items.clear();
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto& users = assigned[static_cast<size_t>(wave[i])];
            const auto rows = static_cast<int32_t>(users.size());
            const float* in = xb_.data();
            if (users.size() != un) {
                float* gathered = ein_.data() + first_row[i] * d;
                for (size_t u = 0; u < users.size(); ++u) {
                    std::copy_n(xb_.data() + static_cast<size_t>(users[u].token) * d, d, gathered + u * d);
                }
                in = gathered;
            }
            items.push_back({&views[i][0], in, eh_.data() + first_row[i] * ff, rows});
            items.push_back({&views[i][1], in, eh2_.data() + first_row[i] * ff, rows});
        }
        if (Status st = route.cpu->matmul_batch(items); !st.is_ok()) {
            release_wave();
            return st;
        }
        for (size_t i = 0; i < rows_total * ff; ++i) eh_[i] = silu(eh_[i]) * eh2_[i];
        items.clear();
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto rows = static_cast<int32_t>(assigned[static_cast<size_t>(wave[i])].size());
            items.push_back({&views[i][2], eh_.data() + first_row[i] * ff, eout_.data() + first_row[i] * d, rows});
        }
        const Status down_status = route.cpu->matmul_batch(items);
        release_wave();  // the outputs are in eout_; the weights are no longer read
        LIYAB_RETURN_IF_ERROR(down_status);
        for (size_t i = 0; i < first_row.size(); ++i) {
            const auto& users = assigned[static_cast<size_t>(wave[i])];
            for (size_t u = 0; u < users.size(); ++u) {
                float* dst = ranked_out_.data() + (static_cast<size_t>(users[u].token) * K + users[u].rank) * d;
                const float* src = eout_.data() + (first_row[i] + u) * d;
                const float weight = users[u].weight;
                for (size_t j = 0; j < d; ++j) dst[j] = weight * src[j];
            }
        }
    }
    // Sum each token's experts in router order. Experts run in whatever order
    // their weights arrive in RAM; adding their outputs in that order made
    // results depend on I/O timing (float addition is not associative, and
    // the int8-quantized activations downstream turn last-bit differences into
    // whole rounding steps: run-to-run perplexity varied by ~10%).
    for (size_t t = 0; t < un; ++t) {
        float* dst = moe_out_.data() + t * d;
        for (size_t k = 0; k < static_cast<size_t>(experts_kept_[t]); ++k) {
            const float* src = ranked_out_.data() + (t * K + k) * d;
            for (size_t j = 0; j < d; ++j) dst[j] += src[j];
        }
    }

    experts_timer.stop();

    if (L.w[kUpShared] != nullptr) {
        const PhaseTimer timer(phases_.shared_expert);
        const auto sh = static_cast<size_t>(w[kUpShared].rows());
        hb_.resize(un * sh);
        hb2_.resize(un * sh);
        const TensorView* gate_up[] = {&w[kGateShared], &w[kUpShared]};
        float* outs[] = {hb_.data(), hb2_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.ffn, gate_up, xb_.data(), outs, n));
        for (size_t i = 0; i < un * sh; ++i) hb_[i] = silu(hb_[i]) * hb2_[i];
        eout_.resize(un * d);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kDownShared], hb_.data(), eout_.data(), n));
        for (size_t t = 0; t < un; ++t) {
            float scale = 1.0f;
            if (!L.shared_gate.empty()) {
                float dot = 0.0f;
                for (size_t j = 0; j < d; ++j) dot += L.shared_gate[j] * xb_[t * d + j];
                scale = sigmoid(dot);
            }
            for (size_t j = 0; j < d; ++j) moe_out_[t * d + j] += scale * eout_[t * d + j];
        }
    }
    std::copy(moe_out_.begin(), moe_out_.end(), xb_.begin());
    return Status::ok();
}

Result<std::span<const float>> Transformer::forward(std::span<const int32_t> tokens, Logits logits,
                                                    const Route& route, ThreadPool& pool, const ForwardHooks* hooks) {
    const ModelConfig& c = config_;
    const auto n = static_cast<int32_t>(tokens.size());
    last_exit_layer_ = -1;
    predicted_layer_ = -1;  // an early exit can leave a guess for a block that never ran
    if (n == 0) return Status(ErrorCode::InvalidArgument, "forward() needs at least one token");
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    for (const int32_t id : tokens) {
        if (id < 0 || id >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
    }
    LIYAB_RETURN_IF_ERROR(kv_->reserve(n_past_ + n));  // maps KV pages; ContextFull past the limit
    // Early exit would leave the skipped blocks' recurrent states behind: attention-only stacks.
    EarlyExitHook* early_exit =
        hooks != nullptr && n == 1 && logits == Logits::Last && !c.hybrid() ? hooks->early_exit : nullptr;
    const HeadMaskHook* head_mask = hooks != nullptr ? hooks->head_mask : nullptr;
    FfnSkipHook* ffn_skip = hooks != nullptr && n == 1 ? hooks->ffn_skip : nullptr;
    FfnMatmulHook* ffn_hook = hooks != nullptr ? hooks->ffn_matmul : nullptr;
    last_ffn_skips_ = 0;

    const size_t d = static_cast<size_t>(c.n_embd);
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(c.head_dim);
    const size_t kv_dim = static_cast<size_t>(c.n_head_kv) * static_cast<size_t>(c.head_dim);
    const auto un = static_cast<size_t>(n);
    x_.resize(un * d);
    xb_.resize(un * std::max(d, q_dim));
    q_.resize(un * q_dim);
    k_.resize(un * kv_dim);
    v_.resize(un * kv_dim);
    att_.resize(un * q_dim);
    for (size_t t = 0; t < un; ++t) {
        quant::dequantize_row(token_embd_->type, token_embd_->row(tokens[t]), x_.data() + t * d, c.n_embd);
    }

    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        const Route cpu_route{route.cpu, route.cpu, route.cpu};
        const Route& block_route = w.streamed() ? cpu_route : route;

        // --- expert prefetch for the first block (the others are predicted
        // one block ahead, see predict_experts) ---
        if (expert_store_ != nullptr && l == 0) predict_experts(0, n, route);

        // --- token mixer (attention or DeltaNet) ---
        if (ffn_skip != nullptr) x_block_in_.assign(x_.begin(), x_.end());
        for (size_t t = 0; t < un; ++t) {
            rmsnorm(x_.data() + t * d, L.attn_norm.data(), xb_.data() + t * d, c.n_embd, c.rms_eps);
        }
        if (L.mixer == MixerKind::DeltaNet) {
            const PhaseTimer timer(phases_.delta_net);
            LIYAB_RETURN_IF_ERROR(delta_net_mixer(l, w, n, block_route, pool));
        } else {
            const PhaseTimer timer(phases_.attention);
            LIYAB_RETURN_IF_ERROR(
                attention_mixer(l, w, n, block_route, pool, head_mask != nullptr ? head_mask->mask(l) : nullptr));
        }
        for (size_t i = 0; i < un * d; ++i) x_[i] += xb_[i];
        // A MoE block predicts the next one once its own experts are queued (moe_ffn).
        if (expert_store_ != nullptr && !L.moe) predict_experts(l + 1, n, route);

        // --- FFN block: dense SwiGLU or mixture of experts (skippable by an experimental hook) ---
        if (ffn_skip != nullptr && ffn_skip->skip_ffn(l, c.n_layers, x_block_in_, x_)) {
            ++last_ffn_skips_;
            if (expert_store_ != nullptr && L.moe) predict_experts(l + 1, n, route);
        } else {
            for (size_t t = 0; t < un; ++t) {
                rmsnorm(x_.data() + t * d, L.ffn_norm.data(), xb_.data() + t * d, c.n_embd, c.rms_eps);
            }
            if (L.moe) {
                LIYAB_RETURN_IF_ERROR(moe_ffn(l, w, n, block_route));
            } else {
                const PhaseTimer timer(phases_.dense_ffn);
                LIYAB_RETURN_IF_ERROR(dense_ffn(l, w, n, block_route, ffn_hook));
            }
            for (size_t i = 0; i < un * d; ++i) x_[i] += xb_[i];
        }

        // --- early-exit probe (experimental) ---
        if (early_exit != nullptr && early_exit->probe_after(l, c.n_layers)) {
            LIYAB_RETURN_IF_ERROR(compute_logits(route, 0, 1));
            if (early_exit->should_exit(std::span<const float>(logits_.data(), logits_.size()))) {
                LIYAB_RETURN_IF_ERROR(propagate_kv(route, l + 1, n_past_));
                last_exit_layer_ = l;
                n_past_ += 1;
                kv_->release_unreachable(n_past_, max_batch_);
                return std::span<const float>(logits_.data(), logits_.size());
            }
        }
    }

    n_past_ += n;
    kv_->release_unreachable(n_past_, max_batch_);  // recycle pages that left the window
    if (expert_store_ != nullptr && ++decode_steps_ % 16 == 0) expert_store_->age();
    if (logits == Logits::None) return std::span<const float>();

    // --- output head ---
    const int32_t out_rows = logits == Logits::All ? n : 1;
    if (Status s = compute_logits(route, logits == Logits::All ? 0 : un - 1, out_rows); !s.is_ok()) {
        n_past_ -= n;
        return s;
    }
    return std::span<const float>(logits_.data(), logits_.size());
}

Status Transformer::adopt_cached_prefix(int32_t n) {
    if (config_.hybrid()) {
        return Status(ErrorCode::Unsupported, "cached KV prefixes do not include recurrent (DeltaNet) states");
    }
    if (n_past_ != 0 || n < 0 || n > context_length_) {
        return Status(ErrorCode::InvalidArgument, "a cached prefix can only be adopted by an empty context");
    }
    for (int32_t p = 0; p * kv_->config().page_tokens < n; ++p) {
        if (kv_->page_data(p) == nullptr) return Status(ErrorCode::InvalidArgument, "cached prefix pages are not mapped");
    }
    n_past_ = n;
    return Status::ok();
}

Status Transformer::truncate(int32_t n) {
    if (n < 0 || n > n_past_) return Status(ErrorCode::InvalidArgument, "truncate beyond cached positions");
    if (n == n_past_) return Status::ok();
    if (n == 0) {
        reset();
        return Status::ok();
    }
    if (config_.hybrid()) {
        return Status(ErrorCode::Unsupported, "recurrent (DeltaNet) states cannot be rolled back to an earlier position");
    }
    LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
    n_past_ = n;
    return Status::ok();
}

}  // namespace liyab
