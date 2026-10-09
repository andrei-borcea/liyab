#include "core/transformer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <istream>
#include <ostream>

#include "core/log.h"
#include "core/quant.h"
#include "core/thread_pool.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

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

// One token of the gated delta rule for one value head, S [hd x hd] row j
// mapping keys to value component j:
//   S = decay * S;  S += strength * (v - S k) k^T;  o = (S q) * q_scale
// Rows are independent: each is decayed and read against k, corrected, then
// read against q, in two passes over its hd values. On NEON four
// accumulators break the dependency chains of the dot products (a fixed
// order, so results do not depend on the thread count); the scalar loop
// elsewhere keeps one.
void delta_rule_step(float* S, const float* q, const float* k, const float* v, float decay, float strength,
                     float q_scale, float* o, size_t hd) {
    for (size_t j = 0; j < hd; ++j) {
        float* row = S + j * hd;
        size_t i = 0;
        float sk = 0.0f;
#if defined(__ARM_NEON)
        if (hd % 16 == 0) {
            const float32x4_t dv = vdupq_n_f32(decay);
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
            for (; i < hd; i += 16) {
                float32x4_t r0 = vmulq_f32(vld1q_f32(row + i), dv), r1 = vmulq_f32(vld1q_f32(row + i + 4), dv);
                float32x4_t r2 = vmulq_f32(vld1q_f32(row + i + 8), dv), r3 = vmulq_f32(vld1q_f32(row + i + 12), dv);
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(k + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(k + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(k + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(k + i + 12));
            }
            sk = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            const float32x4_t delta = vdupq_n_f32((v[j] - sk) * strength);
            a0 = a1 = a2 = a3 = vdupq_n_f32(0.0f);
            for (i = 0; i < hd; i += 16) {
                const float32x4_t r0 = vfmaq_f32(vld1q_f32(row + i), vld1q_f32(k + i), delta);
                const float32x4_t r1 = vfmaq_f32(vld1q_f32(row + i + 4), vld1q_f32(k + i + 4), delta);
                const float32x4_t r2 = vfmaq_f32(vld1q_f32(row + i + 8), vld1q_f32(k + i + 8), delta);
                const float32x4_t r3 = vfmaq_f32(vld1q_f32(row + i + 12), vld1q_f32(k + i + 12), delta);
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(q + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(q + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(q + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(q + i + 12));
            }
            o[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * q_scale;
            continue;
        }
#endif
        for (i = 0; i < hd; ++i) {
            row[i] *= decay;
            sk += row[i] * k[i];
        }
        const float delta = (v[j] - sk) * strength;
        float oq = 0.0f;
        for (i = 0; i < hd; ++i) {
            row[i] += k[i] * delta;
            oq += row[i] * q[i];
        }
        o[j] = oq * q_scale;
    }
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
    model->skip_slow_ = options.skip_slow;
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
            model->attach_expert_store(options.expert_cache_bytes, options.memory_budget_bytes, options.requant_bits,
                                       options.repack_cpu && quant::repack_kernels_available()));
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
    // CPU-only models: the resident Q4_K / Q5_K / Q6_K / Q8_0 matrices in the
    // layout of the i8mm batched kernels (a lossless rearrangement, same
    // size). With expert streaming that is every matrix but the experts (3-D,
    // read from storage in the file's layout); a model under the layer
    // streaming window keeps the file's layout throughout.
    if (options.repack_cpu && !model->file_->streaming()) {
        struct Repack {
            DType from, to;
            int64_t rows, cols;  // required multiples
        };
        for (const Repack& r : {Repack{DType::Q4_K, DType::Q4_K_R8, 8, quant::kSuperBlock},
                                Repack{DType::Q5_K, DType::Q5_K_R8, 8, quant::kSuperBlock},
                                Repack{DType::Q6_K, DType::Q6_K_R8, 8, quant::kSuperBlock},
                                Repack{DType::Q8_0, DType::Q8_0_R4, 4, quant::kBlock}}) {
            auto repacked = model->file_->requantize(
                [&](const TensorView& t) {
                    return t.type == r.from && t.n_dims == 2 && t.rows() % r.rows == 0 && t.cols() % r.cols == 0 &&
                           t.name != "token_embd.weight" && !model->file_->converted(t);
                },
                r.to);
            if (!repacked) return repacked.status();
        }
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
    std::fill(checkpoint_pos_.begin(), checkpoint_pos_.end(), -1);
    snapshots_.clear();
    n_past_ = 0;
}

void Transformer::snapshot_state(bool pin) {
    if (states_.empty() || n_past_ == 0) return;
    if (!snapshots_.empty() && snapshots_.back().pos == n_past_) {  // already held
        if (pin) {
            for (StateSnapshot& s : snapshots_) s.pinned = false;
            snapshots_.back().pinned = true;
        }
        return;
    }
    if (pin) {
        for (StateSnapshot& s : snapshots_) s.pinned = false;
    }
    StateSnapshot snap;
    if (static_cast<int32_t>(snapshots_.size()) >= kStateSnapshots) {
        // Keep the oldest and the pinned one; recycle the buffer of the
        // newest other (a draft's), else of the one after the oldest.
        size_t victim = 0;
        for (size_t i = snapshots_.size(); i-- > 1;) {
            if (!snapshots_[i].pinned) {
                victim = i;
                break;
            }
        }
        if (victim == 0) victim = 1;
        snap = std::move(snapshots_[victim]);
        snapshots_.erase(snapshots_.begin() + static_cast<std::ptrdiff_t>(victim));
    }
    snap.pinned = pin;
    snap.pos = n_past_;
    snap.data.resize(recurrent_state_bytes() / sizeof(float));
    float* out = snap.data.data();
    for (const RecurrentState& st : states_) {
        out = std::copy(st.conv.begin(), st.conv.end(), out);
        out = std::copy(st.ssm.begin(), st.ssm.end(), out);
    }
    snapshots_.push_back(std::move(snap));
}

namespace {

constexpr uint32_t kStateMagic = 0x5453594C;  // "LYST"
constexpr uint32_t kStateVersion = 1;

template <typename T>
void put(std::ostream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
bool get(std::istream& in, T& v) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof v));
}

}  // namespace

// What must match for a saved state to be valid here: the model (file size
// and shape) and how the KV cache stores rows.
std::array<int64_t, 9> Transformer::state_fingerprint() const noexcept {
    const KvCacheConfig& kv = kv_->config();
    return {static_cast<int64_t>(file_->file_size()), config_.n_layers, config_.n_embd, config_.n_vocab,
            kv.n_layers, kv.n_head_kv, kv.head_dim, static_cast<int64_t>(kv_->dtype()), kv.page_tokens};
}

Status Transformer::write_state(std::ostream& out) const {
    if (kv_->window() > 0) return Status(ErrorCode::Unsupported, "a sliding-window context cannot be saved");
    put(out, kStateMagic);
    put(out, kStateVersion);
    for (const int64_t v : state_fingerprint()) put(out, v);
    put(out, n_past_);
    const int32_t pages = (n_past_ + kv_->config().page_tokens - 1) / kv_->config().page_tokens;
    put(out, static_cast<uint64_t>(kv_->page_bytes()));
    put(out, pages);
    for (int32_t p = 0; p < pages; ++p) {
        const uint8_t* data = kv_->page_data(p);
        if (data == nullptr) return Status(ErrorCode::Internal, "a cached page is not mapped");
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(kv_->page_bytes()));
    }
    put(out, static_cast<int32_t>(states_.size()));
    for (const RecurrentState& st : states_) {
        put(out, static_cast<uint64_t>(st.conv.size()));
        put(out, static_cast<uint64_t>(st.ssm.size()));
        out.write(reinterpret_cast<const char*>(st.conv.data()), static_cast<std::streamsize>(st.conv.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(st.ssm.data()), static_cast<std::streamsize>(st.ssm.size() * sizeof(float)));
    }
    return out ? Status::ok() : Status(ErrorCode::IoError, "cannot write the context state");
}

Status Transformer::read_state(std::istream& in) {
    reset();
    auto fail = [this](ErrorCode code, const char* why) {
        reset();
        return Status(code, why);
    };
    uint32_t magic = 0, version = 0;
    if (!get(in, magic) || !get(in, version) || magic != kStateMagic || version != kStateVersion) {
        return fail(ErrorCode::InvalidArgument, "not a Liyab context state (or another version)");
    }
    for (const int64_t expected : state_fingerprint()) {
        int64_t v = 0;
        if (!get(in, v) || v != expected) {
            return fail(ErrorCode::InvalidArgument, "the state was saved for another model or other cache settings");
        }
    }
    int32_t n = 0, pages = 0;
    uint64_t page_bytes = 0;
    if (!get(in, n) || !get(in, page_bytes) || !get(in, pages) || n < 0 || n > context_length_ ||
        page_bytes != kv_->page_bytes() || pages != (n + kv_->config().page_tokens - 1) / kv_->config().page_tokens) {
        return fail(ErrorCode::InvalidArgument, "the state's cache layout does not match");
    }
    if (Status s = kv_->reserve(n); !s.is_ok()) return fail(s.code(), "cannot map the state's cache pages");
    for (int32_t p = 0; p < pages; ++p) {
        uint8_t* data = kv_->mutable_page_data(p);
        if (data == nullptr || !in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(page_bytes))) {
            return fail(ErrorCode::IoError, "the state's cache pages are truncated");
        }
    }
    int32_t count = 0;
    if (!get(in, count) || count != static_cast<int32_t>(states_.size())) {
        return fail(ErrorCode::InvalidArgument, "the state's recurrent blocks do not match");
    }
    for (RecurrentState& st : states_) {
        uint64_t conv = 0, ssm = 0;
        if (!get(in, conv) || !get(in, ssm) || conv != st.conv.size() || ssm != st.ssm.size() ||
            !in.read(reinterpret_cast<char*>(st.conv.data()), static_cast<std::streamsize>(conv * sizeof(float))) ||
            !in.read(reinterpret_cast<char*>(st.ssm.data()), static_cast<std::streamsize>(ssm * sizeof(float)))) {
            return fail(ErrorCode::IoError, "the state's recurrent states are truncated");
        }
    }
    n_past_ = n;
    snapshot_state(/*pin=*/true);  // a restored context is where the next prompt continues
    return Status::ok();
}

int32_t Transformer::restorable_prefix(int32_t n) const noexcept {
    n = std::clamp(n, 0, n_past_);
    if (!config_.hybrid() || n == n_past_) return n;
    int32_t best = 0;
    for (const int32_t p : checkpoint_pos_) {  // a checkpoint at position p restores n = p + 1
        if (p >= 0 && p + 1 <= n) best = std::max(best, p + 1);
    }
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos <= n) best = std::max(best, snap.pos);
    }
    return best;
}

void Transformer::set_rollback_window(int32_t positions) {
    rollback_window_ = states_.empty() ? 0 : std::max(0, positions);
    checkpoint_pos_.assign(static_cast<size_t>(rollback_window_), -1);
    for (RecurrentState& st : states_) {
        st.checkpoints.assign(static_cast<size_t>(rollback_window_) * (st.conv.size() + st.ssm.size()), 0.0f);
        st.checkpoints.shrink_to_fit();
    }
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

// Attention over the KV cache, for n query tokens. Work is split by (token,
// KV head, span of positions): the query heads sharing a KV head (grouped
// query attention: 8 per KV head on Qwen3.6) are scored together, so each K
// and V row is fetched and decoded once for the whole group, not once per
// query head; and in decoding, where one token has only n_head_kv groups,
// the visible positions are split into spans so every thread works (as in
// flash decoding). Each span keeps a partial softmax (its maximum, sum and
// weighted V); the spans are merged exactly (log-sum-exp). On a 2838-token
// context the per-head loop took 48 ms of a 161 ms decode step.
void Transformer::attention(int32_t kv_slot, int32_t n, ThreadPool& pool, const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const int32_t hd = c.head_dim;
    const int32_t group = c.n_head / c.n_head_kv;
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(hd);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const DType kv_type = kv_->dtype();
    const bool quantized_kv = quant::is_block_quantized(kv_type);
    const auto uhd = static_cast<size_t>(hd);
    const auto ugroup = static_cast<size_t>(group);

    // Spans per (token, KV head): enough tasks for every thread, spans of at
    // least kMinSpan positions (shorter ones cost more to merge than they save).
    constexpr int32_t kMinSpan = 64;
    const int32_t longest = kv_->visible(n_past_ + n - 1).sink_end + (n_past_ + n);
    const int64_t base_tasks = int64_t{n} * c.n_head_kv;
    const int64_t wanted = int64_t{pool.active_threads()} * ThreadPool::kChunksPerThread;
    const auto spans = static_cast<int32_t>(std::clamp<int64_t>((wanted + base_tasks - 1) / base_tasks, 1,
                                                                std::max<int64_t>(1, longest / kMinSpan)));
    // Partial results per (token, query head, span): maximum, sum, weighted V.
    const size_t parts = static_cast<size_t>(n) * static_cast<size_t>(c.n_head) * static_cast<size_t>(spans);
    part_max_.resize(parts);
    part_sum_.resize(parts);
    part_out_.resize(parts * uhd);
    auto part = [&](int32_t t, int32_t h, int32_t sp) {
        return (static_cast<size_t>(t) * static_cast<size_t>(c.n_head) + static_cast<size_t>(h)) *
                   static_cast<size_t>(spans) + static_cast<size_t>(sp);
    };

    pool.parallel_for(base_tasks * spans, [&](int64_t begin, int64_t end) {
        thread_local std::vector<float> scores;           // [group][span length]
        thread_local std::vector<quant::BlockQ8_0> q8;    // [group][hd / 32]
        thread_local std::vector<float> vrow;             // one V row, decoded
        for (int64_t task = begin; task < end; ++task) {
            const auto sp = static_cast<int32_t>(task % spans);
            const auto tg = task / spans;
            const auto t = static_cast<int32_t>(tg / c.n_head_kv);
            const auto g = static_cast<int32_t>(tg % c.n_head_kv);
            const int32_t pos = n_past_ + t;
            // Visible positions: attention sinks [0, sink_end) + [first, pos].
            const KvCache::Visible vis = kv_->visible(pos);
            const int32_t n_visible = vis.sink_end + (pos - vis.first + 1);
            auto position = [&](int32_t i) { return i < vis.sink_end ? i : vis.first + (i - vis.sink_end); };
            const auto i0 = static_cast<int32_t>(int64_t{n_visible} * sp / spans);
            const auto i1 = static_cast<int32_t>(int64_t{n_visible} * (sp + 1) / spans);
            const auto len = static_cast<size_t>(std::max(0, i1 - i0));
            const int32_t h0 = g * group;
            // Quantized caches are scored with integer dot products against a
            // Q8_0 copy of each query (same kernels as the weight matmuls).
            const size_t q8_per_head = uhd / quant::kBlock;
            if (quantized_kv) {
                q8.resize(ugroup * q8_per_head);
                for (int32_t j = 0; j < group; ++j) {
                    const float* q = q_.data() + static_cast<size_t>(t) * q_dim + static_cast<size_t>(h0 + j) * uhd;
                    quant::quantize_row_q8_0(q, q8.data() + static_cast<size_t>(j) * q8_per_head, hd);
                }
            }
            scores.resize(ugroup * std::max<size_t>(len, 1));
            for (size_t i = 0; i < len; ++i) {
                const uint8_t* k = kv_->k_row(kv_slot, position(i0 + static_cast<int32_t>(i)), g);
                for (int32_t j = 0; j < group; ++j) {
                    const float* q = q_.data() + static_cast<size_t>(t) * q_dim + static_cast<size_t>(h0 + j) * uhd;
                    scores[static_cast<size_t>(j) * len + i] =
                        (quantized_kv ? quant::dot_quantized(kv_type, k, q8.data() + static_cast<size_t>(j) * q8_per_head,
                                                             nullptr, hd)
                                      : quant::dot_f16_f32(reinterpret_cast<const uint16_t*>(k), q, hd)) *
                        scale;
                }
            }
            for (int32_t j = 0; j < group; ++j) {
                const int32_t h = h0 + j;
                const size_t pi = part(t, h, sp);
                float* out = part_out_.data() + pi * uhd;
                std::fill(out, out + hd, 0.0f);
                part_max_[pi] = -INFINITY;
                part_sum_[pi] = 0.0f;
                if (len == 0 || (head_mask != nullptr && head_mask[h] == 0)) continue;
                float* sc = scores.data() + static_cast<size_t>(j) * len;
                float mx = -INFINITY;
                for (size_t i = 0; i < len; ++i) mx = std::max(mx, sc[i]);
                float sum = 0.0f;
                for (size_t i = 0; i < len; ++i) sum += sc[i] = std::exp(sc[i] - mx);
                part_max_[pi] = mx;
                part_sum_[pi] = sum;
            }
            // V: each row decoded once into floats, then added to every head
            // of the group (decoding it per head cost 8x the work).
            vrow.resize(uhd);
            for (size_t i = 0; i < len; ++i) {
                const uint8_t* v = kv_->v_row(kv_slot, position(i0 + static_cast<int32_t>(i)), g);
                if (kv_type == DType::F32) {
                    std::copy_n(reinterpret_cast<const float*>(v), uhd, vrow.data());
                } else {
                    quant::dequantize_row(kv_type, v, vrow.data(), hd);
                }
                for (int32_t j = 0; j < group; ++j) {
                    const int32_t h = h0 + j;
                    if (head_mask != nullptr && head_mask[h] == 0) continue;
                    const float w = scores[static_cast<size_t>(j) * len + i];
                    float* out = part_out_.data() + part(t, h, sp) * uhd;
                    for (size_t d = 0; d < uhd; ++d) out[d] += w * vrow[d];
                }
            }
        }
    });

    // Merge the spans of every (token, head): exp(m_s - M) rescales each
    // span's sum and weighted V to the global maximum M.
    for (int32_t t = 0; t < n; ++t) {
        for (int32_t h = 0; h < c.n_head; ++h) {
            float* out = att_.data() + static_cast<size_t>(t) * q_dim + static_cast<size_t>(h) * uhd;
            std::fill(out, out + hd, 0.0f);
            if (head_mask != nullptr && head_mask[h] == 0) continue;  // pruned head: contributes nothing
            float mx = -INFINITY;
            for (int32_t sp = 0; sp < spans; ++sp) mx = std::max(mx, part_max_[part(t, h, sp)]);
            float sum = 0.0f;
            for (int32_t sp = 0; sp < spans; ++sp) {
                const size_t pi = part(t, h, sp);
                if (part_sum_[pi] == 0.0f) continue;
                const float w = std::exp(part_max_[pi] - mx);
                sum += w * part_sum_[pi];
                const float* src = part_out_.data() + pi * uhd;
                for (size_t d = 0; d < uhd; ++d) out[d] += w * src[d];
            }
            const float inv = 1.0f / sum;
            for (size_t d = 0; d < uhd; ++d) out[d] *= inv;
        }
    }
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

// Gated DeltaNet (Qwen3.5 / Qwen3-Next linear attention):
//   [q|k|v] = SiLU(causal_conv1d(W_qkv x)),  q, k L2-normalized, q scaled by 1/sqrt(d)
//   decay = exp(softplus(W_a x + dt_bias) * A),  b = sigmoid(W_b x)
//   S = decay * S;  S += b * (v - S k) k^T;  o = S q
//   out = W_out (RMSNorm(o) * SiLU(W_z x))      (per value head)
// Key/query heads are shared by value heads modulo their count (ggml's layout).
// A batch takes two parallel passes whatever its length: the convolution by
// channel heads through all tokens, then the recurrence by value head through
// all tokens (both are independent across their tasks, sequential in time).
// Token by token, a batch of n cost 2 n thread-pool rounds per block and ran
// the convolution, SiLU and L2 norms on one thread.
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

    // Checkpoints: the state after each of the last rollback_window_ tokens of
    // the batch, for truncate(); slot = position % window, layout [conv | ssm].
    const size_t per_state = st.conv.size() + st.ssm.size();
    const bool checkpoint = rollback_window_ > 0 && checkpointing_;
    auto slot_of = [&](size_t t) -> float* {
        if (!checkpoint || t + static_cast<size_t>(rollback_window_) < un) return nullptr;
        return st.checkpoints.data() +
               static_cast<size_t>((n_past_ + static_cast<int32_t>(t)) % rollback_window_) * per_state;
    };

    // 1. Causal depthwise convolution over [history, current] and SiLU, every
    //    channel through all n tokens (channels are independent, tokens are
    //    not), then the L2 norm of each query / key head. One task per head of
    //    channels: 2 hk query/key heads, then hv value heads.
    const auto conv_tasks = static_cast<int64_t>(2 * hk + hv);
    pool.parallel_for(conv_tasks, [&](int64_t begin, int64_t end) {
        for (auto task = static_cast<size_t>(begin); task < static_cast<size_t>(end); ++task) {
            const size_t ch0 = task * hd;  // query heads, key heads and value heads are hd channels each
            for (size_t ch = ch0; ch < ch0 + hd; ++ch) {
                float* hist = st.conv.data() + ch * history;
                const float* tap = L.conv1d.data() + ch * taps;
                for (size_t t = 0; t < un; ++t) {
                    float* m = mix_.data() + t * conv_dim;
                    float acc = m[ch] * tap[history];
                    for (size_t j = 0; j < history; ++j) acc += hist[j] * tap[j];
                    for (size_t j = 0; j + 1 < history; ++j) hist[j] = hist[j + 1];
                    hist[history - 1] = m[ch];
                    m[ch] = silu(acc);
                    if (float* slot = slot_of(t)) std::copy(hist, hist + history, slot + ch * history);
                }
            }
            if (task < 2 * hk) {
                for (size_t t = 0; t < un; ++t) l2norm(mix_.data() + t * conv_dim + ch0, static_cast<int32_t>(hd), c.rms_eps);
            }
        }
    });

    // 2. The recurrence, one task per value head through all n tokens (heads
    //    are independent), then the head's gated RMSNorm.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    pool.parallel_for(static_cast<int64_t>(hv), [&](int64_t begin, int64_t end) {
        for (auto h = static_cast<size_t>(begin); h < static_cast<size_t>(end); ++h) {
            float* S = st.ssm.data() + h * hd * hd;
            for (size_t t = 0; t < un; ++t) {
                const float* m = mix_.data() + t * conv_dim;
                const float* q = m + (h % hk) * hd;
                const float* k = m + key_dim + (h % hk) * hd;
                const float* v = m + 2 * key_dim + h * hd;
                const float decay = std::exp(softplus(alpha_[t * hv + h] + L.ssm_dt[h]) * L.ssm_a[h]);
                const float strength = sigmoid(beta_[t * hv + h]);
                float* o = dn_.data() + t * value_dim + h * hd;
                delta_rule_step(S, q, k, v, decay, strength, q_scale, o, hd);
                const float* z = z_.data() + t * value_dim + h * hd;
                rmsnorm(o, L.ssm_norm.data(), o, static_cast<int32_t>(hd), c.rms_eps);
                for (size_t j = 0; j < hd; ++j) o[j] *= silu(z[j]);
                if (float* slot = slot_of(t)) std::copy(S, S + hd * hd, slot + st.conv.size() + h * hd * hd);
            }
        }
    });
    return matmul(route, route.attention, w[kSsmOut], dn_.data(), xb_.data(), n);
}

// Expert streaming: the routed experts stay on storage and are read on
// demand into a fixed RAM cache; every other tensor stays resident in the
// mapping (paged in now, never swept by the streaming window).
Status Transformer::attach_expert_store(int64_t budget_option, uint64_t memory_budget, int32_t requant_bits,
                                        bool repack) {
    if (budget_option == 0) return Status::ok();
    if (requant_bits != -1 && requant_bits != 0 && requant_bits != 4 && requant_bits != 5) {
        return Status(ErrorCode::InvalidArgument, "requant_bits must be -1, 0, 4 or 5");
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

    // The expert cache gets what the resident weights leave, after room for
    // the KV cache, activations and the rest of the app: 1 GiB of free RAM,
    // or 512 MiB inside an explicit budget (the embedder already left its own
    // headroom below the platform's cap). Recurrent-state snapshots
    // (snapshot_state) come out of the cache too.
    const size_t margin = (memory_budget > 0 ? size_t{512} << 20 : size_t{1} << 30) +
                          static_cast<size_t>(kStateSnapshots) * recurrent_state_bytes();
    auto cache_budget = [&](size_t resident) -> size_t {
        if (budget_option > 0) return static_cast<size_t>(budget_option);
        return available > resident + margin ? static_cast<size_t>(available) - resident - margin : 0;
    };

    // Optional lossy conversion of the resident Q8_0 matrices (the UD quants
    // keep attention and DeltaNet projections in Q8_0, ~2/3 of the bytes read
    // per token): fewer bytes per token, and the freed memory goes to the
    // expert cache. The token embedding is skipped: decode reads one row of
    // it per token. Automatic (-1): only when the cache would hold less than
    // kTightCache of the experts, where hit rates fall off and nearly every
    // token waits for storage; Q5_K if that frees enough, else Q4_K.
    auto convertible = [&](const TensorView& t) {
        return !is_expert(t) && t.type == DType::Q8_0 && t.n_dims == 2 && t.rows() > 1 &&
               t.cols() % quant::kSuperBlock == 0 && t.name != "token_embd.weight";
    };
    const size_t resident_q8 = file_->file_size() - expert_bytes;
    int32_t bits = requant_bits;
    if (bits < 0) {
        bits = 0;
        const auto wanted = static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes));
        if (cache_budget(resident_q8) < wanted) {
            size_t q8_bytes = 0;
            for (const TensorView& t : file_->tensors()) {
                if (convertible(t)) q8_bytes += t.nbytes;
            }
            // Q8_0: 8.5 bits per weight; Q5_K 5.5, Q4_K 4.5.
            const auto saving = [&](double bpw) { return static_cast<size_t>(static_cast<double>(q8_bytes) * (1.0 - bpw / 8.5)); };
            bits = q8_bytes == 0 ? 0 : (cache_budget(resident_q8 - saving(5.5)) >= wanted ? 5 : 4);
            LIYAB_LOG_INFO("memory is tight for this model (expert cache %.2f GiB, %.0f%% of experts): %s",
                           static_cast<double>(cache_budget(resident_q8)) / (1024.0 * 1024.0 * 1024.0),
                           100.0 * static_cast<double>(cache_budget(resident_q8)) / static_cast<double>(expert_bytes),
                           bits == 0 ? "no Q8_0 matrices to convert" : bits == 5 ? "converting resident Q8_0 to Q5_K"
                                                                      : "converting resident Q8_0 to Q4_K");
        }
    }
    size_t saved = 0;
    if (bits > 0) {
        auto requantized = file_->requantize(convertible, bits == 4 ? DType::Q4_K : DType::Q5_K);
        if (!requantized) return requantized.status();
        saved = requantized.value();
        LIYAB_LOG_INFO("resident Q8_0 matrices requantized to %s: %.2f GiB saved", bits == 4 ? "Q4_K" : "Q5_K",
                       static_cast<double>(saved) / (1024.0 * 1024.0 * 1024.0));
    }
    const size_t resident_bytes = resident_q8 - saved;
    const size_t budget = cache_budget(resident_bytes);
    memory_plan_.resident_bytes = resident_bytes;
    memory_plan_.expert_bytes = expert_bytes;
    memory_plan_.requant_bits = bits;
    // What the cache needs for most experts to come from RAM (~85% hits at
    // 12-15% of the experts cached on Qwen3.6-35B-A3B; the share, not the size,
    // is what matters across models).
    memory_plan_.recommended_bytes =
        resident_q8 + static_cast<size_t>(kComfortableCache * static_cast<double>(expert_bytes)) + margin;
    std::vector<std::array<const TensorView*, 3>> experts(static_cast<size_t>(c.n_layers));
    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        if (L.moe) experts[static_cast<size_t>(l)] = {L.w[kGateExps], L.w[kUpExps], L.w[kDownExps]};
    }
    auto store = ExpertStore::create(*file_, std::move(experts), c.n_expert, budget, 4, repack);
    if (!store) return store.status();
    expert_store_ = std::move(store).value();
    memory_plan_.expert_cache_bytes = expert_store_->capacity_bytes();
    if (memory_plan_.expert_cache_bytes < static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes))) {
        LIYAB_LOG_WARN("expert cache holds %.0f%% of the experts: most tokens will wait for storage; about %.1f GiB of "
                       "memory budget would let this model run well",
                       100.0 * static_cast<double>(memory_plan_.expert_cache_bytes) / static_cast<double>(expert_bytes),
                       static_cast<double>(memory_plan_.recommended_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
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
    std::vector<int32_t> best_rank(E, -1);
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
        for (size_t r = 0; r < kept; ++r) {
            int32_t& best = best_rank[static_cast<size_t>(order[r])];
            if (best < 0 || static_cast<int32_t>(r) < best) best = static_cast<int32_t>(r);
        }
    }
    if (rank_guessed_.size() != static_cast<size_t>(c.n_expert_used)) {
        rank_guessed_.assign(static_cast<size_t>(c.n_expert_used), 0.0);
        rank_used_.assign(static_cast<size_t>(c.n_expert_used), 0.0);
    }
    guesses_.clear();
    std::vector<int32_t> prefetched;
    for (size_t e = 0; e < E; ++e) {
        const int32_t r = best_rank[e];
        if (r < 0) continue;
        guesses_.push_back({static_cast<int32_t>(e), r});
        picks.push_back(static_cast<int32_t>(e));
        // Laplace-smoothed precision of the rank: every rank starts as worth prefetching (2/3).
        const auto ur = static_cast<size_t>(r);
        if ((rank_used_[ur] + 2.0) / (rank_guessed_[ur] + 3.0) > 0.5) prefetched.push_back(static_cast<int32_t>(e));
    }
    expert_store_->prefetch(layer, prefetched, true);
    predicted_ = std::move(picks);  // sorted: built in expert order
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
    if (expert_store_ != nullptr) expert_store_->set_batch(n > 1);  // its reads are repacked for batched kernels
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
    std::vector<float> token_weight(un, 0.0f);  // per token: the weights of its kept experts (for skip_slow_)
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
            token_weight[t] += assigned[e].back().weight;
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
            // Per-rank precision, decayed so it follows the conversation.
            for (const Guess& g : guesses_) {
                const auto r = static_cast<size_t>(g.rank);
                rank_guessed_[r] += 1.0;
                if (std::binary_search(chosen.begin(), chosen.end(), g.expert)) rank_used_[r] += 1.0;
                if (rank_guessed_[r] > 2048.0) {
                    rank_guessed_[r] *= 0.5;
                    rank_used_[r] *= 0.5;
                }
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

    // The shared expert (when the block has one) runs where the routed ones
    // would wait for a read, or after them; its output is added last either
    // way, so the result does not depend on when it ran.
    double shared_in_experts_ms = 0.0;  // shared-expert time inside the experts phase, moved back to its own
    bool shared_done = L.w[kUpShared] == nullptr;
    auto run_shared = [&]() -> Status {
        const auto t0 = std::chrono::steady_clock::now();
        const auto sh = static_cast<size_t>(w[kUpShared].rows());
        hb_.resize(un * sh);
        hb2_.resize(un * sh);
        const TensorView* gate_up[] = {&w[kGateShared], &w[kUpShared]};
        float* outs[] = {hb_.data(), hb2_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.ffn, gate_up, xb_.data(), outs, n));
        for (size_t i = 0; i < un * sh; ++i) hb_[i] = silu(hb_[i]) * hb2_[i];
        shared_out_.resize(un * d);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kDownShared], hb_.data(), shared_out_.data(), n));
        // The gate scale is applied when the output is added (below), in one
        // multiply-add as before: scaling here first rounded differently.
        shared_scale_.assign(un, 1.0f);
        if (!L.shared_gate.empty()) {
            for (size_t t = 0; t < un; ++t) {
                float dot = 0.0f;
                for (size_t j = 0; j < d; ++j) dot += L.shared_gate[j] * xb_[t * d + j];
                shared_scale_[t] = sigmoid(dot);
            }
        }
        shared_done = true;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        phases_.shared_expert += ms;
        shared_in_experts_ms += ms;
        return Status::ok();
    };

    PhaseTimer experts_timer(phases_.experts);
    moe_out_.assign(un * d, 0.0f);
    std::vector<float> skipped_weight(un, 0.0f);  // per token, the weight of experts skipped (skip_slow_)
    std::vector<uint8_t> rank_skipped(un * K, 0);
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
            if (expert_store_ != nullptr && skip_slow_ > 0.0f && !expert_store_->ready(layer, ex)) {
                // Not in RAM yet and light for every token using it: skip it
                // rather than wait (lossy; the token's others are renormalized below).
                const auto& users = assigned[static_cast<size_t>(ex)];
                const bool light = std::all_of(users.begin(), users.end(), [&](const Use& u) {
                    return u.weight < skip_slow_ * token_weight[static_cast<size_t>(u.token)];
                });
                if (light) {
                    for (const Use& u : users) {
                        skipped_weight[static_cast<size_t>(u.token)] += u.weight;
                        rank_skipped[static_cast<size_t>(u.token) * K + static_cast<size_t>(u.rank)] = 1;
                    }
                    ++predictions_.skipped;
                    continue;
                }
            }
            if (expert_store_ != nullptr) {
                // About to wait for a read: do the shared expert's work meanwhile.
                if (!shared_done && wave.empty() && !expert_store_->ready(layer, ex)) {
                    if (Status st = run_shared(); !st.is_ok()) return st;
                }
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

        if (wave.empty()) continue;  // every expert of this wave was skipped
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
            if (rank_skipped[t * K + k]) continue;
            const float* src = ranked_out_.data() + (t * K + k) * d;
            for (size_t j = 0; j < d; ++j) dst[j] += src[j];
        }
        // Skipped experts: the others carry the token's whole weight, as the
        // model's own top-k normalization would have given it.
        if (skipped_weight[t] > 0.0f && c.moe_norm_weights && skipped_weight[t] < token_weight[t]) {
            const float scale = token_weight[t] / (token_weight[t] - skipped_weight[t]);
            for (size_t j = 0; j < d; ++j) dst[j] *= scale;
        }
    }

    experts_timer.stop();
    phases_.experts -= shared_in_experts_ms;

    if (!shared_done) LIYAB_RETURN_IF_ERROR(run_shared());
    if (L.w[kUpShared] != nullptr) {
        for (size_t t = 0; t < un; ++t) {
            const float scale = shared_scale_[t];
            for (size_t j = 0; j < d; ++j) moe_out_[t * d + j] += scale * shared_out_[t * d + j];
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
    // The DeltaNet mixers overwrite these rollback slots; they become valid
    // only once every block has run (a failed pass leaves no half-written checkpoint).
    const int32_t recorded = checkpointing_ ? std::min(n, rollback_window_) : 0;
    for (int32_t i = 0; i < recorded; ++i) {
        checkpoint_pos_[static_cast<size_t>((n_past_ + n - 1 - i) % rollback_window_)] = -1;
    }
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

    for (int32_t i = 0; i < recorded; ++i) {
        const int32_t pos = n_past_ + n - 1 - i;
        checkpoint_pos_[static_cast<size_t>(pos % rollback_window_)] = pos;
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
        const int32_t slot = rollback_window_ > 0 ? (n - 1) % rollback_window_ : -1;
        const bool in_ring = slot >= 0 && checkpoint_pos_[static_cast<size_t>(slot)] == n - 1;
        const auto snap = std::find_if(snapshots_.begin(), snapshots_.end(),
                                       [n](const StateSnapshot& s) { return s.pos == n; });
        if (!in_ring && snap == snapshots_.end()) {
            return Status(ErrorCode::Unsupported,
                          "recurrent (DeltaNet) states can only be rolled back inside the rollback window or to a "
                          "snapshot");
        }
        LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
        const float* saved = in_ring ? nullptr : snap->data.data();
        for (RecurrentState& st : states_) {
            if (in_ring) saved = st.checkpoints.data() + static_cast<size_t>(slot) * (st.conv.size() + st.ssm.size());
            std::copy_n(saved, st.conv.size(), st.conv.begin());
            std::copy_n(saved + st.conv.size(), st.ssm.size(), st.ssm.begin());
            if (!in_ring) saved += st.conv.size() + st.ssm.size();
        }
        // Checkpoints and snapshots of the discarded positions describe a
        // future that no longer exists.
        for (int32_t& p : checkpoint_pos_) {
            if (p >= n) p = -1;
        }
        std::erase_if(snapshots_, [n](const StateSnapshot& s) { return s.pos > n; });
        n_past_ = n;
        return Status::ok();
    }
    LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
    n_past_ = n;
    return Status::ok();
}

}  // namespace liyab
