#include "core/transformer.h"

#include <algorithm>
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

}  // namespace

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
Result<std::unique_ptr<Transformer>> Transformer::load(std::unique_ptr<MmapLoader> file,
                                                       const TransformerOptions& options) {
    if (!file) return Status(ErrorCode::InvalidArgument, "null model file");
    std::unique_ptr<Transformer> model(new Transformer());
    model->file_ = std::move(file);
    const MmapLoader& f = *model->file_;
    ModelConfig& c = model->config_;

    const auto arch = f.get_string("general.architecture");
    if (!arch) return Status(ErrorCode::InvalidModel, "missing general.architecture");
    c.arch = std::string(*arch);
    if (c.arch == "llama" || c.arch == "mistral") c.rope_neox = false;
    else if (c.arch == "qwen2" || c.arch == "qwen3") c.rope_neox = true;
    else return Status(ErrorCode::Unsupported, "architecture '" + c.arch + "' is not supported (llama, mistral, qwen2, qwen3)");

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
    LIYAB_RETURN_IF_ERROR(require_int("embedding_length", c.n_embd));
    LIYAB_RETURN_IF_ERROR(require_int("feed_forward_length", c.n_ff));
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

    LIYAB_RETURN_IF_ERROR(model->bind_weights());
    LIYAB_LOG_INFO("bound %d blocks of %s weights (%s, head_dim %d)", c.n_layers,
                   std::string(dtype_traits(model->layers_[0].w_up->type).name).c_str(), c.arch.c_str(), c.head_dim);

    // Context and KV cache.
    model->context_length_ = options.context_length > 0
                                 ? options.context_length
                                 : (c.n_ctx_train > 0 ? std::min(c.n_ctx_train, 4096) : 4096);
    if (c.n_ctx_train > 0 && model->context_length_ > c.n_ctx_train) {
        LIYAB_LOG_WARN("context %d exceeds the model's training context %d", model->context_length_, c.n_ctx_train);
    }
    model->max_batch_ = std::max(1, options.max_batch);
    KvCacheConfig kv;
    kv.n_layers = c.n_layers;
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

    model->file_->configure_layers(c.n_layers);
    LIYAB_LOG_INFO("%s: %d layers, d=%d, ff=%d, heads=%d/%d x %d, vocab=%d, ctx=%d%s, kv=%s, %.1f KiB/token",
                   c.arch.c_str(), c.n_layers, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim, c.n_vocab,
                   model->context_length_, kv.window ? " (sliding)" : "",
                   std::string(dtype_traits(model->kv_->dtype()).name).c_str(),
                   static_cast<double>(KvCache::bytes_per_token(c.n_layers, c.n_head_kv, c.head_dim, kv.type)) / 1024.0);
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

    layers_.resize(static_cast<size_t>(c.n_layers));
    for (int32_t l = 0; l < c.n_layers; ++l) {
        Layer& L = layers_[static_cast<size_t>(l)];
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_norm.weight"), c.n_embd, true, L.attn_norm));
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_q.weight"), c.n_embd, q_dim, L.wq));
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_k.weight"), c.n_embd, kv_dim, L.wk));
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_v.weight"), c.n_embd, kv_dim, L.wv));
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "attn_output.weight"), q_dim, c.n_embd, L.wo));
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_q.bias"), q_dim, false, L.bq));
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_k.bias"), kv_dim, false, L.bk));
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_v.bias"), kv_dim, false, L.bv));
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_q_norm.weight"), c.head_dim, false, L.q_norm));
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "attn_k_norm.weight"), c.head_dim, false, L.k_norm));
        LIYAB_RETURN_IF_ERROR(vec(layer_name(l, "ffn_norm.weight"), c.n_embd, true, L.ffn_norm));
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_gate.weight"), c.n_embd, c.n_ff, L.w_gate));
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_up.weight"), c.n_embd, c.n_ff, L.w_up));
        LIYAB_RETURN_IF_ERROR(matrix(layer_name(l, "ffn_down.weight"), c.n_ff, c.n_embd, L.w_down));
    }
    return Status::ok();
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
class Transformer::BlockWeights {
public:
    BlockWeights(Transformer& model, int32_t layer) : model_(model), layer_(layer) {
        const Layer& L = model.layers_[static_cast<size_t>(layer)];
        const TensorView* src[] = {L.wq, L.wk, L.wv, L.wo, L.w_gate, L.w_up, L.w_down};
        for (size_t i = 0; i < 7; ++i) views_[i] = *src[i];
        if (model.layer_source_ == nullptr) {
            model.file_->begin_layer(layer);  // mmap path: advance the prefetch window
            return;
        }
        auto base = model.layer_source_->acquire(layer);
        if (!base) {
            status_ = base.status();
            return;
        }
        acquired_ = true;
        // Every block tensor lies inside the block's byte range, so its slot
        // address is the slot base plus its offset within that range.
        const size_t range_begin = model.file_->layer_range(layer).first;
        for (TensorView& v : views_) v.data = base.value() + (v.file_offset - range_begin);
    }
    ~BlockWeights() {
        if (acquired_) model_.layer_source_->release(layer_);
    }
    BlockWeights(const BlockWeights&) = delete;
    BlockWeights& operator=(const BlockWeights&) = delete;

    [[nodiscard]] const Status& status() const noexcept { return status_; }
    const TensorView& wq() const { return views_[0]; }
    const TensorView& wk() const { return views_[1]; }
    const TensorView& wv() const { return views_[2]; }
    const TensorView& wo() const { return views_[3]; }
    const TensorView& gate() const { return views_[4]; }
    const TensorView& up() const { return views_[5]; }
    const TensorView& down() const { return views_[6]; }

private:
    Transformer& model_;
    int32_t layer_;
    bool acquired_ = false;
    Status status_;
    TensorView views_[7];
};

void Transformer::reset() noexcept {
    kv_->clear();
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

void Transformer::attention(int32_t layer, int32_t n, ThreadPool& pool, const uint8_t* head_mask) {
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
                const uint8_t* k = kv_->k_row(layer, position(i), g);
                const float s = (quantized_kv ? quant::dot_quantized(kv_type, k, q8.data(), hd)
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
                quant::axpy_row(kv_type, kv_->v_row(layer, position(i), g), scores[static_cast<size_t>(i)] * inv_sum,
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

Status Transformer::compute_logits(const Route& route, size_t first_row, int32_t rows) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
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
    for (int32_t l = from; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        rmsnorm(x_.data(), L.attn_norm.data(), xb_.data(), c.n_embd, c.rms_eps);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w.wk(), xb_.data(), k_.data(), 1));
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w.wv(), xb_.data(), v_.data(), 1));
        for (size_t i = 0; i < L.bk.size(); ++i) k_[i] += L.bk[i];
        for (size_t i = 0; i < L.bv.size(); ++i) v_[i] += L.bv[i];
        if (!L.k_norm.empty()) {
            for (int32_t h = 0; h < c.n_head_kv; ++h) {
                float* kh = k_.data() + static_cast<size_t>(h) * c.head_dim;
                rmsnorm(kh, L.k_norm.data(), kh, c.head_dim, c.rms_eps);
            }
        }
        rope(k_.data(), c.n_head_kv, pos);
        kv_->store(l, pos, k_.data(), v_.data());
    }
    return Status::ok();
}

Result<std::span<const float>> Transformer::forward(std::span<const int32_t> tokens, Logits logits,
                                                    const Route& route, ThreadPool& pool, const ForwardHooks* hooks) {
    const ModelConfig& c = config_;
    const auto n = static_cast<int32_t>(tokens.size());
    last_exit_layer_ = -1;
    if (n == 0) return Status(ErrorCode::InvalidArgument, "forward() needs at least one token");
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    for (const int32_t id : tokens) {
        if (id < 0 || id >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
    }
    LIYAB_RETURN_IF_ERROR(kv_->reserve(n_past_ + n));  // maps KV pages; ContextFull past the limit
    EarlyExitHook* early_exit =
        hooks != nullptr && n == 1 && logits == Logits::Last ? hooks->early_exit : nullptr;
    const HeadMaskHook* head_mask = hooks != nullptr ? hooks->head_mask : nullptr;
    FfnSkipHook* ffn_skip = hooks != nullptr && n == 1 ? hooks->ffn_skip : nullptr;
    FfnMatmulHook* ffn_hook = hooks != nullptr ? hooks->ffn_matmul : nullptr;
    // FFN projections: an experimental override first, else the routed backend.
    auto ffn_matmul = [&](int32_t layer, FfnProjection p, const TensorView& w, const float* x, float* y) -> Status {
        if (ffn_hook != nullptr && ffn_hook->ffn_matmul(layer, p, x, y, n)) return Status::ok();
        return matmul(route, route.ffn, w, x, y, n);
    };
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
    hb_.resize(un * static_cast<size_t>(c.n_ff));
    hb2_.resize(un * static_cast<size_t>(c.n_ff));

    for (size_t t = 0; t < un; ++t) {
        quant::dequantize_row(token_embd_->type, token_embd_->row(tokens[t]), x_.data() + t * d, c.n_embd);
    }

    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());

        // --- attention block ---
        if (ffn_skip != nullptr) x_block_in_.assign(x_.begin(), x_.end());
        for (size_t t = 0; t < un; ++t) {
            rmsnorm(x_.data() + t * d, L.attn_norm.data(), xb_.data() + t * d, c.n_embd, c.rms_eps);
        }
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w.wq(), xb_.data(), q_.data(), n));
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w.wk(), xb_.data(), k_.data(), n));
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w.wv(), xb_.data(), v_.data(), n));
        for (size_t t = 0; t < un; ++t) {
            float* q = q_.data() + t * q_dim;
            float* k = k_.data() + t * kv_dim;
            float* v = v_.data() + t * kv_dim;
            for (size_t i = 0; i < L.bq.size(); ++i) q[i] += L.bq[i];
            for (size_t i = 0; i < L.bk.size(); ++i) k[i] += L.bk[i];
            for (size_t i = 0; i < L.bv.size(); ++i) v[i] += L.bv[i];
            if (!L.q_norm.empty()) {
                for (int32_t h = 0; h < c.n_head; ++h) {
                    float* qh = q + static_cast<size_t>(h) * c.head_dim;
                    rmsnorm(qh, L.q_norm.data(), qh, c.head_dim, c.rms_eps);
                }
            }
            if (!L.k_norm.empty()) {
                for (int32_t h = 0; h < c.n_head_kv; ++h) {
                    float* kh = k + static_cast<size_t>(h) * c.head_dim;
                    rmsnorm(kh, L.k_norm.data(), kh, c.head_dim, c.rms_eps);
                }
            }
            const int32_t pos = n_past_ + static_cast<int32_t>(t);
            rope(q, c.n_head, pos);
            rope(k, c.n_head_kv, pos);
            kv_->store(l, pos, k, v);
        }
        attention(l, n, pool, head_mask != nullptr ? head_mask->mask(l) : nullptr);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w.wo(), att_.data(), xb_.data(), n));
        for (size_t i = 0; i < un * d; ++i) x_[i] += xb_[i];

        // --- SwiGLU FFN block (skippable by an experimental hook) ---
        if (ffn_skip != nullptr && ffn_skip->skip_ffn(l, c.n_layers, x_block_in_, x_)) {
            ++last_ffn_skips_;
        } else {
            for (size_t t = 0; t < un; ++t) {
                rmsnorm(x_.data() + t * d, L.ffn_norm.data(), xb_.data() + t * d, c.n_embd, c.rms_eps);
            }
            LIYAB_RETURN_IF_ERROR(ffn_matmul(l, FfnProjection::Gate, w.gate(), xb_.data(), hb_.data()));
            LIYAB_RETURN_IF_ERROR(ffn_matmul(l, FfnProjection::Up, w.up(), xb_.data(), hb2_.data()));
            for (size_t i = 0; i < hb_.size(); ++i) {
                const float g = hb_[i];
                hb_[i] = g / (1.0f + std::exp(-g)) * hb2_[i];
            }
            LIYAB_RETURN_IF_ERROR(ffn_matmul(l, FfnProjection::Down, w.down(), hb_.data(), xb_.data()));
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
    LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
    n_past_ = n;
    return Status::ok();
}

}  // namespace liyab
