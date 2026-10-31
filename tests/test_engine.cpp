// End-to-end tests: quantization kernels, tokenizer, the transformer against
// an independent float reference, KV-cache rollback and sliding window,
// speculative decoding invariants, and the Engine through C++ and C APIs.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "core/draft_budget.h"
#include "core/kernel_conformance.h"
#include "core/moe_routing.h"
#include "core/quant.h"
#include "core/quant_blocks.h"
#include "core/sampling.h"
#include "core/suffix_drafter.h"
#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/transformer.h"
#include "liyab/liyab.h"
#include "liyab/device_detect.h"
#include "liyab/liyab_c_api.h"
#include "liyab/speculative_decoder.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

std::string temp_path(const std::string& name) {
    return test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_" + name;
}

// Float32 model so the reference comparison is tight.
test::TinyModelSpec f32_spec() {
    test::TinyModelSpec s;
    s.attn_type = s.ffn_type = s.embd_type = s.output_type = DType::F32;
    return s;
}

// The models the cases share, written once per process and removed at exit
// (a device run must leave nothing in TMPDIR). A deque, so the paths handed
// out stay valid as more models are written.
const std::string& model_path(const std::string& name, const test::TinyModelSpec& spec) {
    struct Written {
        std::deque<std::pair<std::string, std::string>> files;
        ~Written() {
            for (const auto& file : files) std::remove(file.second.c_str());
        }
    };
    static Written written;
    for (const auto& [n, p] : written.files) {
        if (n == name) return p;
    }
    written.files.emplace_back(name, temp_path(name + ".gguf"));
    test::write_tiny_model(written.files.back().second, spec);
    return written.files.back().second;
}

const std::string& mixed_model() { return model_path("mixed", {}); }
const std::string& f32_model() { return model_path("f32", f32_spec()); }
const std::string& ssm_model(const char* arch) {
    test::TinyModelSpec s;
    s.arch = arch;
    return model_path(arch, s);
}

std::unique_ptr<Transformer> load_transformer(const std::string& path, TransformerOptions options = {}, LoaderOptions loader = {}) {
    auto file = MmapLoader::open(path, loader);
    if (!file) return nullptr;
    auto model = Transformer::load(std::move(file).value(), options);
    if (!model) {
        std::printf("  load failed: %s\n", model.status().to_string().c_str());
        return nullptr;
    }
    return std::move(model).value();
}

// ---------------------------------------------------------------------------
// Independent float reference of the llama forward pass (full recompute, no
// KV cache, no activation quantization). Optional sliding window.
// ---------------------------------------------------------------------------
class Reference {
public:
    explicit Reference(const MmapLoader& f) : f_(f) {
        d_ = static_cast<int>(*f.get_int("llama.embedding_length"));
        n_head_ = static_cast<int>(*f.get_int("llama.attention.head_count"));
        n_kv_ = static_cast<int>(*f.get_int("llama.attention.head_count_kv"));
        n_layers_ = static_cast<int>(*f.get_int("llama.block_count"));
        hd_ = d_ / n_head_;
    }

    Reference(const MmapLoader& f, const test::TinyModelSpec& spec)
        : f_(f), d_(spec.n_embd), n_head_(spec.n_head), n_kv_(spec.n_head_kv), n_layers_(spec.n_layers),
          hd_(spec.head_dim > 0 ? spec.head_dim : spec.n_embd / spec.n_head), block_heads_(spec.block_heads), block_kv_(spec.block_kv_heads),
          block_ff_(spec.block_ff), local_(spec.swa_layers), local_head_(spec.swa_head_dim),
          local_rope_(spec.swa_rope_dim), rotary_(spec.rope_dim), local_window_(spec.sliding_window) {}

    std::vector<std::vector<float>> logits(const std::vector<int32_t>& tokens, int window = 0, int sinks = 0, DType cache_type = DType::F32) const {
        const int n = static_cast<int>(tokens.size());
        std::vector<std::vector<float>> x(static_cast<size_t>(n));
        const auto emb = weights("token_embd.weight");
        const int vocab = static_cast<int>(emb.size()) / d_;
        for (int t = 0; t < n; ++t) {
            x[t].assign(emb.begin() + tokens[t] * d_, emb.begin() + (tokens[t] + 1) * d_);
        }
        for (int l = 0; l < n_layers_; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const int heads = block_heads_.empty() ? n_head_ : block_heads_[static_cast<size_t>(l)];
            const int kv_heads = block_kv_.empty() ? n_kv_ : block_kv_[static_cast<size_t>(l)];
            const bool local = !local_.empty() && local_[static_cast<size_t>(l)];
            const int hd = local && local_head_ > 0 ? local_head_ : hd_;
            const int rd = local && local_rope_ > 0 ? local_rope_ : rotary_ > 0 ? rotary_ : hd;
            const int visible_window = local ? local_window_ : window;
            const bool has_ffn = block_ff_.empty() || block_ff_[static_cast<size_t>(l)] > 0;
            if (kv_heads == 0) {
                if (heads > 0) {
                    const auto wo = weights(p + "attn_output.weight");
                    for (auto& row : x) {
                        const auto branch = matvec(wo, norm(row, weights(p + "attn_norm.weight")));
                        for (int i = 0; i < d_; ++i) row[i] += branch[i];
                    }
                }
                if (has_ffn) ffn(x, p);
                continue;
            }
            const auto wq = weights(p + "attn_q.weight"), wk = weights(p + "attn_k.weight");
            const auto wv = weights(p + "attn_v.weight"), wo = weights(p + "attn_output.weight");
            const auto sink_logits = f_.tensor(p + "attn_sinks.weight") ? weights(p + "attn_sinks.weight") : std::vector<float>{};
            const auto output_bias = f_.tensor(p + "attn_output.bias") ? weights(p + "attn_output.bias") : std::vector<float>{};
            std::vector<std::vector<float>> q(n), k(n), v(n);
            for (int t = 0; t < n; ++t) {
                const auto h = norm(x[t], weights(p + "attn_norm.weight"));
                q[t] = matvec(wq, h);
                k[t] = matvec(wk, h);
                v[t] = matvec(wv, h);
                rope(q[t], t, hd, rd);
                rope(k[t], t, hd, rd);
                // Independently round compact heads to the configured cache
                // precision; padded channels cannot alter these blocks.
                auto round = [&](std::vector<float>& row, DType type) {
                    std::vector<uint8_t> encoded(dtype_row_bytes(type, hd));
                    for (size_t at = 0; at < row.size(); at += static_cast<size_t>(hd)) {
                        quant::quantize_row(type, row.data() + at, encoded.data(), hd);
                        quant::dequantize_row(type, encoded.data(), row.data() + at, hd);
                    }
                };
                if (cache_type != DType::F32) {
                    round(k[t], cache_type);
                    round(v[t], cache_type);
                    if (quant::is_block_quantized(cache_type)) round(q[t], DType::Q8_0);
                }
            }
            for (int t = 0; t < n; ++t) {
                std::vector<float> att(static_cast<size_t>(heads * hd), 0.0f);
                for (int h = 0; h < heads; ++h) {
                    const int g = h / (heads / kv_heads);
                    const int first = visible_window > 0 ? std::max(0, t - visible_window + 1) : 0;
                    const int sink_end = std::min(sinks, first);  // attention sinks before the window
                    std::vector<int> visible;
                    for (int j = 0; j < sink_end; ++j) visible.push_back(j);
                    for (int j = first; j <= t; ++j) visible.push_back(j);
                    std::vector<double> s;
                    double mx = -1e30;
                    for (const int j : visible) {
                        double dot = 0;
                        for (int i = 0; i < hd; ++i) dot += q[t][h * hd + i] * k[j][g * hd + i];
                        s.push_back(dot / std::sqrt(static_cast<double>(hd)));
                        mx = std::max(mx, s.back());
                    }
                    if (!sink_logits.empty()) mx = std::max(mx, static_cast<double>(sink_logits[h]));
                    double sum = sink_logits.empty() ? 0.0 : std::exp(sink_logits[h] - mx);
                    for (double& e : s) sum += (e = std::exp(e - mx));
                    for (size_t idx = 0; idx < visible.size(); ++idx) {
                        for (int i = 0; i < hd; ++i) {
                            att[h * hd + i] += static_cast<float>(s[idx] / sum * v[visible[idx]][g * hd + i]);
                        }
                    }
                }
                const auto o = matvec(wo, att);
                for (int i = 0; i < d_; ++i) x[t][i] += o[i] + (output_bias.empty() ? 0.0f : output_bias[i]);
            }
            if (has_ffn) ffn(x, p);
        }
        const auto out_w = f_.tensor("output.weight") ? weights("output.weight") : emb;
        std::vector<std::vector<float>> result;
        for (int t = 0; t < n; ++t) {
            result.push_back(matvec(out_w, norm(x[t], weights("output_norm.weight"))));
            result.back().resize(static_cast<size_t>(vocab));
        }
        return result;
    }

private:
    void ffn(std::vector<std::vector<float>>& x, const std::string& p) const {
        const auto wg = weights(p + "ffn_gate.weight"), wu = weights(p + "ffn_up.weight");
        const auto wd = weights(p + "ffn_down.weight");
        for (size_t t = 0; t < x.size(); ++t) {
            const auto h = norm(x[t], weights(p + "ffn_norm.weight"));
            auto g = matvec(wg, h);
            const auto u = matvec(wu, h);
            float limit = 0.0f;
            const auto layer = static_cast<size_t>(std::stoi(p.substr(4)));
            if (const auto* value = f_.metadata("llama.swiglu_clamp_shexp")) {
                std::memcpy(&limit, value->array_data + layer * sizeof(float), sizeof(float));
            }
            for (size_t i = 0; i < g.size(); ++i) {
                const float activated = g[i] / (1.0f + std::exp(-g[i]));
                g[i] = limit > 0 ? std::min(activated, limit) * std::clamp(u[i], -limit, limit) : activated * u[i];
            }
            const auto down = matvec(wd, g);
            for (int i = 0; i < d_; ++i) x[t][i] += down[i];
        }
    }
    std::vector<float> weights(const std::string& name) const {
        const TensorView* t = f_.tensor(name);
        std::vector<float> out(static_cast<size_t>(t->elements()));
        for (int64_t r = 0; r < t->rows(); ++r) quant::dequantize_row(t->type, t->row(r), out.data() + r * t->cols(), t->cols());
        return out;
    }
    static std::vector<float> matvec(const std::vector<float>& w, const std::vector<float>& x) {
        const size_t cols = x.size(), rows = w.size() / cols;
        std::vector<float> y(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0;
            for (size_t c = 0; c < cols; ++c) acc += static_cast<double>(w[r * cols + c]) * x[c];
            y[r] = static_cast<float>(acc);
        }
        return y;
    }
    static std::vector<float> norm(const std::vector<float>& x, const std::vector<float>& w) {
        double ss = 0;
        for (const float v : x) ss += static_cast<double>(v) * v;
        const double scale = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + 1e-5);
        std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i] * scale * w[i]);
        return y;
    }
    // GGUF llama layout: rotate adjacent pairs (2i, 2i+1) of every head.
    void rope(std::vector<float>& v, int pos, int hd, int rd) const {
        for (size_t h = 0; h < v.size() / static_cast<size_t>(hd); ++h) {
            for (int i = 0; i < rd / 2; ++i) {
                // The model's rotary frequencies and angles are F32. A
                // double-precision angle can cross a later Q4 rounding
                // boundary even when both attention implementations agree.
                const float frequency = static_cast<float>(std::pow(10000.0, -2.0 * i / rd));
                const float theta = static_cast<float>(pos) * frequency;
                float& a = v[h * hd + 2 * i];
                float& b = v[h * hd + 2 * i + 1];
                const float a0 = a, b0 = b;
                a = static_cast<float>(a0 * std::cos(theta) - b0 * std::sin(theta));
                b = static_cast<float>(a0 * std::sin(theta) + b0 * std::cos(theta));
            }
        }
    }

    const MmapLoader& f_;
    int d_, n_head_, n_kv_, n_layers_, hd_;
    std::vector<int32_t> block_heads_, block_kv_, block_ff_;
    std::vector<bool> local_;
    int local_head_ = 0, local_rope_ = 0, rotary_ = 0, local_window_ = 0;
};

// Independent float reference of a deepseek2 model with multi-head latent
// attention in the form it is trained in: every position's latent expanded
// into per-head keys [W_kb,h c | k_pe] and values W_vb,h c, scored with
// 1/sqrt(nope + rope) (full recompute, no cache, no absorption, no
// quantized activations). Dense SwiGLU FFNs; the RoPE key is rotated on
// adjacent pairs with base 10000.
class LatentReference {
public:
    explicit LatentReference(const MmapLoader& f) : f_(f) {
        d_ = static_cast<int>(*f.get_int("deepseek2.embedding_length"));
        heads_ = static_cast<int>(*f.get_int("deepseek2.attention.head_count"));
        layers_ = static_cast<int>(*f.get_int("deepseek2.block_count"));
        rank_ = static_cast<int>(*f.get_int("deepseek2.attention.kv_lora_rank"));
        q_rank_ = static_cast<int>(f.get_int("deepseek2.attention.q_lora_rank").value_or(0));
        rd_ = static_cast<int>(*f.get_int("deepseek2.rope.dimension_count"));
        nope_ = static_cast<int>(f.get_int("deepseek2.attention.key_length_mla").value_or(
            *f.get_int("deepseek2.attention.key_length"))) - rd_;
        vd_ = static_cast<int>(f.get_int("deepseek2.attention.value_length_mla").value_or(
            *f.get_int("deepseek2.attention.value_length")));
    }

    std::vector<std::vector<float>> logits(const std::vector<int32_t>& tokens) const {
        const int n = static_cast<int>(tokens.size());
        const int qk = nope_ + rd_;
        const auto emb = weights("token_embd.weight");
        std::vector<std::vector<float>> x(static_cast<size_t>(n));
        for (int t = 0; t < n; ++t) x[t].assign(emb.begin() + tokens[t] * d_, emb.begin() + (tokens[t] + 1) * d_);
        for (int l = 0; l < layers_; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const auto kv_a = weights(p + "attn_kv_a_mqa.weight");
            std::vector<float> kb, vb;
            if (f_.tensor(p + "attn_kv_b.weight")) {
                const auto combined = weights(p + "attn_kv_b.weight");
                kb.resize(static_cast<size_t>(heads_ * rank_ * nope_));
                vb.resize(static_cast<size_t>(heads_ * vd_ * rank_));
                for (int h = 0; h < heads_; ++h) {
                    for (int j = 0; j < nope_ + vd_; ++j) {
                        for (int r = 0; r < rank_; ++r) {
                            const float v = combined[static_cast<size_t>((h * (nope_ + vd_) + j) * rank_ + r)];
                            if (j < nope_) kb[static_cast<size_t>((h * rank_ + r) * nope_ + j)] = v;
                            else vb[static_cast<size_t>((h * vd_ + j - nope_) * rank_ + r)] = v;
                        }
                    }
                }
            } else {
                kb = weights(p + "attn_k_b.weight");  // [head][rank][nope]
                vb = weights(p + "attn_v_b.weight");  // [head][v][rank]
            }
            std::vector<std::vector<float>> q(n), keys(n), values(n);  // per token: [head][qk], [head][qk], [head][v]
            for (int t = 0; t < n; ++t) {
                const auto h = norm(x[t], weights(p + "attn_norm.weight"));
                q[t] = q_rank_ > 0 ? matvec(weights(p + "attn_q_b.weight"),
                                            norm(matvec(weights(p + "attn_q_a.weight"), h), weights(p + "attn_q_a_norm.weight")))
                                   : matvec(weights(p + "attn_q.weight"), h);
                const auto kv = matvec(kv_a, h);  // [latent | k_pe]
                const auto c = norm(std::vector<float>(kv.begin(), kv.begin() + rank_), weights(p + "attn_kv_a_norm.weight"));
                std::vector<float> k_pe(kv.begin() + rank_, kv.end());
                rope(k_pe, t);
                keys[t].assign(static_cast<size_t>(heads_ * qk), 0.0f);
                values[t].assign(static_cast<size_t>(heads_ * vd_), 0.0f);
                for (int hh = 0; hh < heads_; ++hh) {
                    for (int j = 0; j < nope_; ++j) {
                        double acc = 0;
                        for (int r = 0; r < rank_; ++r) acc += static_cast<double>(kb[(static_cast<size_t>(hh) * rank_ + r) * nope_ + j]) * c[r];
                        keys[t][static_cast<size_t>(hh * qk + j)] = static_cast<float>(acc);
                    }
                    for (int j = 0; j < rd_; ++j) keys[t][static_cast<size_t>(hh * qk + nope_ + j)] = k_pe[j];
                    for (int i = 0; i < vd_; ++i) {
                        double acc = 0;
                        for (int r = 0; r < rank_; ++r) acc += static_cast<double>(vb[(static_cast<size_t>(hh) * vd_ + i) * rank_ + r]) * c[r];
                        values[t][static_cast<size_t>(hh * vd_ + i)] = static_cast<float>(acc);
                    }
                    std::vector<float> q_pe(q[t].begin() + hh * qk + nope_, q[t].begin() + (hh + 1) * qk);
                    rope(q_pe, t);
                    std::copy(q_pe.begin(), q_pe.end(), q[t].begin() + hh * qk + nope_);
                }
            }
            const auto wo = weights(p + "attn_output.weight");
            const auto bo = f_.tensor(p + "attn_output.bias") ? weights(p + "attn_output.bias") : std::vector<float>{};
            for (int t = 0; t < n; ++t) {
                std::vector<float> att(static_cast<size_t>(heads_ * vd_), 0.0f);
                for (int hh = 0; hh < heads_; ++hh) {
                    std::vector<double> s;
                    double mx = -1e30;
                    for (int j = 0; j <= t; ++j) {
                        double dot = 0;
                        for (int i = 0; i < qk; ++i) dot += static_cast<double>(q[t][hh * qk + i]) * keys[j][hh * qk + i];
                        s.push_back(dot / std::sqrt(static_cast<double>(qk)));
                        mx = std::max(mx, s.back());
                    }
                    double sum = 0;
                    for (double& e : s) sum += (e = std::exp(e - mx));
                    for (int j = 0; j <= t; ++j) {
                        for (int i = 0; i < vd_; ++i) att[hh * vd_ + i] += static_cast<float>(s[j] / sum * values[j][hh * vd_ + i]);
                    }
                }
                auto o = matvec(wo, att);
                for (size_t i = 0; i < bo.size(); ++i) o[i] += bo[i];
                for (int i = 0; i < d_; ++i) x[t][i] += o[i];
            }
            const auto wg = weights(p + "ffn_gate.weight"), wu = weights(p + "ffn_up.weight");
            const auto wd = weights(p + "ffn_down.weight");
            for (int t = 0; t < n; ++t) {
                const auto h = norm(x[t], weights(p + "ffn_norm.weight"));
                auto g = matvec(wg, h);
                const auto u = matvec(wu, h);
                float limit = 0.0f;
            const auto layer = static_cast<size_t>(std::stoi(p.substr(4)));
            if (const auto* value = f_.metadata("llama.swiglu_clamp_shexp")) {
                std::memcpy(&limit, value->array_data + layer * sizeof(float), sizeof(float));
            }
            for (size_t i = 0; i < g.size(); ++i) {
                const float activated = g[i] / (1.0f + std::exp(-g[i]));
                g[i] = limit > 0 ? std::min(activated, limit) * std::clamp(u[i], -limit, limit) : activated * u[i];
            }
                const auto down = matvec(wd, g);
                for (int i = 0; i < d_; ++i) x[t][i] += down[i];
            }
        }
        const auto out_w = f_.tensor("output.weight") ? weights("output.weight") : emb;
        std::vector<std::vector<float>> result;
        for (int t = 0; t < n; ++t) result.push_back(matvec(out_w, norm(x[t], weights("output_norm.weight"))));
        return result;
    }

private:
    std::vector<float> weights(const std::string& name) const {
        const TensorView* t = f_.tensor(name);
        std::vector<float> out(static_cast<size_t>(t->elements()));
        for (int64_t r = 0; r < t->rows(); ++r) quant::dequantize_row(t->type, t->row(r), out.data() + r * t->cols(), t->cols());
        return out;
    }
    static std::vector<float> matvec(const std::vector<float>& w, const std::vector<float>& x) {
        const size_t cols = x.size(), rows = w.size() / cols;
        std::vector<float> y(rows);
        for (size_t r = 0; r < rows; ++r) {
            double acc = 0;
            for (size_t c = 0; c < cols; ++c) acc += static_cast<double>(w[r * cols + c]) * x[c];
            y[r] = static_cast<float>(acc);
        }
        return y;
    }
    static std::vector<float> norm(const std::vector<float>& x, const std::vector<float>& w) {
        double ss = 0;
        for (const float v : x) ss += static_cast<double>(v) * v;
        const double scale = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + 1e-5);
        std::vector<float> y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = static_cast<float>(x[i] * scale * w[i]);
        return y;
    }
    // Adjacent pairs (2i, 2i + 1) of the rope_dim-wide part, as DeepSeek's interleaved RoPE.
    void rope(std::vector<float>& v, int pos) const {
        for (int i = 0; i < rd_ / 2; ++i) {
            const double theta = pos * std::pow(10000.0, -2.0 * i / rd_);
            const float a0 = v[2 * i], b0 = v[2 * i + 1];
            v[2 * i] = static_cast<float>(a0 * std::cos(theta) - b0 * std::sin(theta));
            v[2 * i + 1] = static_cast<float>(a0 * std::sin(theta) + b0 * std::cos(theta));
        }
    }

    const MmapLoader& f_;
    int d_, heads_, layers_, rank_, q_rank_, rd_, nope_, vd_;
};

double max_abs_diff(std::span<const float> a, std::span<const float> b) {
    double m = 0;
    for (size_t i = 0; i < b.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

EngineConfig engine_config(const std::string& path) {
    EngineConfig c;
    c.model_path = path;
    c.n_threads = 4;
    c.thermal_polling = false;
    c.power.profile = PowerProfile::Performance;  // unpaced: keep tests fast
    // The reference CPU: automatic routing decides by timing which passes an accelerator rounds, so two engines
    // would not always give the same text, and these tests compare engines. Tests of the accelerators name them.
    c.backend = BackendKind::Cpu;
    return c;
}

SamplingParams greedy(int32_t max_tokens) {
    SamplingParams p;
    p.temperature = 0.0f;
    p.max_tokens = max_tokens;
    return p;
}

std::vector<int32_t> run(Engine& engine, const std::vector<int32_t>& prompt, const SamplingParams& params,
                         GenerationStats* stats_out = nullptr) {
    std::vector<int32_t> out;
    auto stats = engine.generate_tokens(prompt, params, [&](std::string_view, int32_t token) {
        out.push_back(token);
        return true;
    });
    if (!stats) std::printf("  generate failed: %s\n", stats.status().to_string().c_str());
    if (stats && stats_out) *stats_out = stats.value();
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------
TEST_CASE("fp16 conversion round-trips representative values") {
    for (const float v : {0.0f, -0.0f, 1.0f, -2.5f, 65504.0f, 6.1035156e-05f, 5.9604645e-08f, 0.333251953125f}) {
        CHECK(quant::fp16_to_fp32(quant::fp32_to_fp16(v)) == v);
    }
    CHECK(std::isinf(quant::fp16_to_fp32(quant::fp32_to_fp16(1e6f))));
    CHECK(std::isnan(quant::fp16_to_fp32(quant::fp32_to_fp16(NAN))));
}

TEST_CASE("Q4_0 / Q8_0 dot kernels match the float dot product") {
    std::mt19937 rng(3);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (const int64_t n : {32, 64, 96, 4096}) {
        std::vector<float> w(static_cast<size_t>(n)), x(static_cast<size_t>(n));
        for (auto& v : w) v = normal(rng);
        for (auto& v : x) v = normal(rng);
        double exact = 0;
        for (int64_t i = 0; i < n; ++i) exact += static_cast<double>(w[i]) * x[i];

        std::vector<quant::BlockQ8_0> xq(static_cast<size_t>(n / 32)), wq8(xq.size());
        std::vector<quant::BlockQ4_0> wq4(xq.size());
        quant::quantize_row_q8_0(x.data(), xq.data(), n);
        quant::quantize_row_q8_0(w.data(), wq8.data(), n);
        quant::quantize_row_q4_0(w.data(), wq4.data(), n);

        // Kernel == scalar dot of the dequantized operands (exactness of SIMD).
        std::vector<float> wd(static_cast<size_t>(n)), xd(static_cast<size_t>(n));
        quant::dequantize_row(DType::Q8_0, xq.data(), xd.data(), n);
        quant::dequantize_row(DType::Q4_0, wq4.data(), wd.data(), n);
        double dequant_dot = 0;
        for (int64_t i = 0; i < n; ++i) dequant_dot += static_cast<double>(wd[i]) * xd[i];
        CHECK_NEAR(quant::dot_q4_0_q8_0(wq4.data(), xq.data(), n), dequant_dot, 1e-3 * std::sqrt(n));

        // Quantization error bounds versus the exact float dot.
        const double tol = 0.02 * std::sqrt(static_cast<double>(n));
        CHECK_NEAR(quant::dot_q8_0_q8_0(wq8.data(), xq.data(), n), exact, tol);
        CHECK_NEAR(quant::dot_q4_0_q8_0(wq4.data(), xq.data(), n), exact, 6 * tol);

        std::vector<uint16_t> wh(static_cast<size_t>(n));
        quant::quantize_row(DType::F16, w.data(), wh.data(), n);
        CHECK_NEAR(quant::dot_f16_f32(wh.data(), x.data(), n), exact, 1e-2 * std::sqrt(n));
        CHECK_NEAR(quant::dot_f32(w.data(), x.data(), n), exact, 1e-4 * n);
    }
}

TEST_CASE("Q4_K / Q5_K quantization round-trips within the formats' error") {
    // Gaussian weights with a few outliers, and an all-zero super-block.
    std::mt19937 rng(11);
    std::normal_distribution<float> normal(0.0f, 0.02f);
    const int64_t n = 256 * 64;
    std::vector<float> w(static_cast<size_t>(n));
    for (auto& v : w) v = normal(rng);
    for (int64_t i = 0; i < n; i += 997) w[static_cast<size_t>(i)] *= 30.0f;
    std::fill_n(w.begin(), 256, 0.0f);
    double power = 0;
    for (const float v : w) power += static_cast<double>(v) * v;
    auto relative_rmse = [&](DType type, size_t block_bytes) {
        std::vector<uint8_t> q(static_cast<size_t>(n / 256) * block_bytes);
        std::vector<float> back(w.size());
        quant::quantize_row(type, w.data(), q.data(), n);
        quant::dequantize_row(type, q.data(), back.data(), n);
        double err = 0;
        for (size_t i = 0; i < w.size(); ++i) {
            if (i < 256) CHECK(back[i] == 0.0f);
            err += (static_cast<double>(back[i]) - w[i]) * (static_cast<double>(back[i]) - w[i]);
        }
        return std::sqrt(err / power);
    };
    const double e4 = relative_rmse(DType::Q4_K, sizeof(quant::BlockQ4_K));
    const double e5 = relative_rmse(DType::Q5_K, sizeof(quant::BlockQ5_K));
    CHECK(e4 < 0.12);  // measured: 0.087 and 0.043 (Q8_0: 0.009)
    CHECK(e5 < 0.06);
    CHECK(e5 < e4 * 0.7);
}

TEST_CASE("K-quant and Q5 dot kernels equal the dot of their dequantized rows") {
    std::mt19937 rng(17);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int64_t n = 512;  // two 256-value super-blocks
    std::vector<float> x(static_cast<size_t>(n));
    for (auto& v : x) v = normal(rng);
    std::vector<quant::BlockQ8_0> xq(static_cast<size_t>(n / 32));
    std::vector<int32_t> xs(xq.size());
    quant::quantize_row_q8_0(x.data(), xq.data(), n);
    quant::block_sums(xq.data(), xs.data(), n);
    std::vector<float> xd(static_cast<size_t>(n));
    quant::dequantize_row(DType::Q8_0, xq.data(), xd.data(), n);

    for (const DType type : {DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::Q5_0, DType::Q5_1}) {
        // Random payload with sane fp16 scales (random bits could be inf/NaN).
        std::vector<uint8_t> row(dtype_row_bytes(type, n));
        for (auto& b : row) b = static_cast<uint8_t>(rng());
        const size_t block = dtype_row_bytes(type, dtype_traits(type).block_size);
        for (size_t off = 0; off < row.size(); off += block) {
            const uint16_t d = quant::fp32_to_fp16(0.01f + 0.001f * static_cast<float>(rng() % 16));
            const uint16_t m = quant::fp32_to_fp16(0.005f);
            if (type == DType::Q6_K) {
                std::memcpy(&row[off + 208], &d, 2);  // d is the last field
            } else {
                std::memcpy(&row[off], &d, 2);
                if (type != DType::Q5_0) std::memcpy(&row[off + 2], &m, 2);  // dmin / m
            }
        }
        std::vector<float> w(static_cast<size_t>(n));
        quant::dequantize_row(type, row.data(), w.data(), n);
        double expected = 0.0;
        for (int64_t i = 0; i < n; ++i) expected += static_cast<double>(w[i]) * xd[i];
        const float got = quant::dot_quantized(type, row.data(), xq.data(), xs.data(), n);
        double scale = 0.0;
        for (int64_t i = 0; i < n; ++i) scale += std::fabs(static_cast<double>(w[i]) * xd[i]);
        if (std::fabs(got - expected) > 1e-4 * std::max(1.0, scale)) {
            std::printf("  %s: kernel %f vs reference %f\n", std::string(dtype_traits(type).name).c_str(), got, expected);
        }
        CHECK_NEAR(got, expected, 1e-4 * std::max(1.0, scale));
    }
}

using test::QuantVector;
using test::load_quant_vectors;

TEST_CASE("Every GGML format decodes exactly like llama.cpp's reference (tests/data/quant_vectors.bin)") {
    std::vector<QuantVector> vectors;
    if (!load_quant_vectors(vectors)) return;
    std::mt19937 rng(23);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    int checked = 0;
    for (const QuantVector& qv : vectors) {
        const DType type = qv.type;
        const DTypeTraits t = dtype_traits(type);
        const int64_t n = qv.n;
        const uint8_t* blocks = qv.blocks.data();
        const std::vector<float>& expected = qv.expected;

        // 1. Decoding: bit-exact, except IQ1 (reference rounds dl*(g+delta), we compute dl*g + dl*delta).
        std::vector<float> got(static_cast<size_t>(n));
        quant::dequantize_row(type, blocks, got.data(), n);
        const bool iq1 = type == DType::IQ1_S || type == DType::IQ1_M;
        double max_err = 0.0, scale = 1e-6;
        for (int64_t i = 0; i < n; ++i) {
            max_err = std::max(max_err, std::fabs(static_cast<double>(got[i]) - expected[i]));
            scale = std::max(scale, std::fabs(static_cast<double>(expected[i])));
        }
        const bool decode_ok = iq1 ? max_err <= 1e-6 * scale : max_err == 0.0;
        CHECK(decode_ok);

        // 2. Dot kernel against the float dot of the decoded weights (block-aligned length).
        const int64_t m = n / 256 * 256 > 0 ? n / 256 * 256 : n / 32 * 32;
        std::vector<float> x(static_cast<size_t>(m));
        for (auto& v : x) v = normal(rng);
        float kernel = 0.0f;
        double reference = 0.0;
        if (type == DType::BF16) {
            kernel = quant::dot_bf16_f32(reinterpret_cast<const uint16_t*>(blocks), x.data(), m);
            for (int64_t i = 0; i < m; ++i) reference += static_cast<double>(expected[i]) * x[i];
        } else {
            std::vector<quant::BlockQ8_0> xq(static_cast<size_t>(m / 32));
            std::vector<int32_t> xs(xq.size());
            quant::quantize_row_q8_0(x.data(), xq.data(), m);
            quant::block_sums(xq.data(), xs.data(), m);
            std::vector<float> xd(static_cast<size_t>(m));
            quant::dequantize_row(DType::Q8_0, xq.data(), xd.data(), m);
            kernel = quant::dot_quantized(type, blocks, xq.data(), xs.data(), m);
            for (int64_t i = 0; i < m; ++i) reference += static_cast<double>(expected[i]) * xd[i];
        }
        double mag = 1e-3;
        for (int64_t i = 0; i < m; ++i) mag += std::fabs(static_cast<double>(expected[i]) * x[i]);
        const bool dot_ok = std::fabs(kernel - reference) <= 1e-4 * mag;
        CHECK(dot_ok);
        std::printf("  %-8s decode %s (max err %.1e)  dot %s\n", std::string(t.name).c_str(),
                    decode_ok ? "exact" : "MISMATCH", max_err, dot_ok ? "ok" : "MISMATCH");
        ++checked;
    }
    CHECK(checked == static_cast<int>(vectors.size()));
}

// A Q1_0 block is 128 values and the kernels take it against four Q8_0 blocks, so rows of one block and of an odd
// number of blocks matter, as do blocks whose bits are all set or all clear and activations at both ends of the
// int8 range. The reference is the decoded weights against the decoded activations, so only float rounding of the
// epilogue separates it from the kernel.
TEST_CASE("Q1_0 dot equals the decoded weights against the quantized activations, for any row length") {
    std::mt19937 rng(61);
    std::uniform_real_distribution<float> uniform(-3.0f, 3.0f);
    for (const int64_t blocks : {1, 2, 3, 5, 7, 40}) {
        const int64_t n = blocks * 128;
        std::vector<uint8_t> row(static_cast<size_t>(blocks) * 18);
        for (auto& byte : row) byte = static_cast<uint8_t>(rng());
        for (int64_t b = 0; b < blocks; ++b) {
            const uint16_t scale = quant::fp32_to_fp16(0.01f + static_cast<float>(rng() % 1000) * 0.001f);
            std::memcpy(row.data() + b * 18, &scale, 2);
        }
        std::memset(row.data() + 2, 0xFF, 16);                            // first block: every sign +
        if (blocks > 1) std::memset(row.data() + 18 + 2, 0x00, 16);       // second: every sign -
        std::vector<float> x(static_cast<size_t>(n));
        for (float& v : x) v = uniform(rng);
        for (int64_t k = 0; k < n / 32; ++k) {  // each Q8_0 block reaches +127 and -127
            x[static_cast<size_t>(32 * k)] = 40.0f;
            x[static_cast<size_t>(32 * k + 1)] = -40.0f;
        }
        std::vector<quant::BlockQ8_0> xq(static_cast<size_t>(n / 32));
        quant::quantize_row_q8_0(x.data(), xq.data(), n);
        std::vector<float> w(static_cast<size_t>(n)), xd(static_cast<size_t>(n));
        quant::dequantize_row(DType::Q1_0, row.data(), w.data(), n);
        quant::dequantize_row(DType::Q8_0, xq.data(), xd.data(), n);
        double reference = 0.0, magnitude = 1e-6;
        for (int64_t i = 0; i < n; ++i) {
            reference += static_cast<double>(w[static_cast<size_t>(i)]) * xd[static_cast<size_t>(i)];
            magnitude += std::fabs(static_cast<double>(w[static_cast<size_t>(i)]) * xd[static_cast<size_t>(i)]);
        }
        const float got = quant::dot_quantized(DType::Q1_0, row.data(), xq.data(), nullptr, n);
        CHECK(std::fabs(got - reference) <= 1e-5 * magnitude);
        // The generic decode path is the reference of every other format; it must agree here too.
        CHECK(std::fabs(quant::dot_ext_q8_0(DType::Q1_0, row.data(), xq.data(), n) - reference) <= 1e-5 * magnitude);
    }
}

TEST_CASE("Q8_K activation quantization matches llama.cpp's semantics") {
    std::mt19937 rng(41);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int64_t n = 1024;
    std::vector<float> x(static_cast<size_t>(n));
    for (auto& v : x) v = normal(rng);
    x[300] = 9.0f;                         // positive extreme: d < 0 and x[300] -> -127
    for (int j = 512; j < 768; ++j) x[j] = 0.0f;  // all-zero super-block
    std::vector<quant::BlockQ8_K> q(static_cast<size_t>(n / 256));
    quant::quantize_row_q8_K(x.data(), q.data(), n);
    CHECK(q[1].qs[300 - 256] == -127);
    CHECK(q[1].d < 0.0f);
    CHECK(q[2].d == 0.0f);
    for (size_t b = 0; b < q.size(); ++b) {
        for (int j = 0; j < 16; ++j) {
            int sum = 0;
            for (int e = 0; e < 16; ++e) sum += q[b].qs[16 * j + e];
            CHECK(q[b].bsums[j] == sum);
        }
        for (int e = 0; e < 256; ++e) {
            const float xv = x[b * 256 + e];
            // Round-to-nearest: the reconstruction error is at most half a step.
            CHECK(std::fabs(q[b].d * q[b].qs[e] - xv) <= 0.5f * std::fabs(q[b].d) + 1e-6f);
            CHECK(q[b].qs[e] >= -127);
        }
    }
}

TEST_CASE("Multi-row and i8mm tile kernels equal one single-row dot per activation row") {
    std::mt19937 rng(21);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int64_t n = 512;
    std::vector<float> w(static_cast<size_t>(n));
    for (float& v : w) v = normal(rng);
    for (const int32_t rows : {1, 2, 3, 4, 5, 6, 9}) {
        std::vector<std::vector<quant::BlockQ8_K>> xk(static_cast<size_t>(rows));
        std::vector<std::vector<quant::BlockQ8_0>> x0(static_cast<size_t>(rows));
        std::vector<const quant::BlockQ8_K*> pk;
        std::vector<const quant::BlockQ8_0*> p0;
        for (int32_t r = 0; r < rows; ++r) {
            std::vector<float> x(static_cast<size_t>(n));
            for (float& v : x) v = normal(rng);
            xk[static_cast<size_t>(r)].resize(static_cast<size_t>(n / 256));
            x0[static_cast<size_t>(r)].resize(static_cast<size_t>(n / 32));
            quant::quantize_row_q8_K(x.data(), xk[static_cast<size_t>(r)].data(), n);
            quant::quantize_row_q8_0(x.data(), x0[static_cast<size_t>(r)].data(), n);
            pk.push_back(xk[static_cast<size_t>(r)].data());
            p0.push_back(x0[static_cast<size_t>(r)].data());
        }
        std::vector<float> out(static_cast<size_t>(rows));
        for (const DType type : {DType::Q4_K, DType::Q5_K, DType::Q6_K}) {
            const size_t block = type == DType::Q4_K ? sizeof(quant::BlockQ4_K)
                                 : type == DType::Q5_K ? sizeof(quant::BlockQ5_K)
                                                       : sizeof(quant::BlockQ6_K);
            std::vector<uint8_t> q(static_cast<size_t>(n / 256) * block);
            if (type == DType::Q6_K) {  // no Q6_K quantizer: random bytes are valid blocks (fp16 d kept finite)
                for (uint8_t& b : q) b = static_cast<uint8_t>(rng());
                for (size_t i = 0; i < q.size(); i += block) {
                    const uint16_t d = quant::fp32_to_fp16(0.01f);
                    std::memcpy(q.data() + i + block - 2, &d, 2);
                }
            } else {
                quant::quantize_row(type, w.data(), q.data(), n);
            }
            quant::dot_lowbit_q8_K_rows(type, q.data(), pk.data(), rows, n, out.data());
            for (int32_t r = 0; r < rows; ++r) {
                const float single = quant::dot_lowbit_q8_K(type, q.data(), pk[static_cast<size_t>(r)], n);
                CHECK_NEAR(out[static_cast<size_t>(r)], single, 1e-4f * (1.0f + std::fabs(single)));
            }
            // i8mm 2x2 tiles (weight row and a second, shifted one): bit-identical to the single dots.
            if (rows >= 2 && quant::i8mm_kernels_compiled() && detect_cpu().i8mm) {
                std::vector<uint8_t> q2(q.rbegin(), q.rend());  // another valid-looking row
                for (size_t i = 0; i < q2.size(); i += block) std::memcpy(q2.data() + i, q.data() + i, 4);  // keep d / dmin sane
                float tile[4];
                REQUIRE(quant::dot_q8_K_2x2(type, q.data(), q2.data(), pk[0], pk[1], n, tile));
                CHECK(tile[0] == quant::dot_lowbit_q8_K(type, q.data(), pk[0], n));
                CHECK(tile[1] == quant::dot_lowbit_q8_K(type, q.data(), pk[1], n));
                CHECK(tile[2] == quant::dot_lowbit_q8_K(type, q2.data(), pk[0], n));
                CHECK(tile[3] == quant::dot_lowbit_q8_K(type, q2.data(), pk[1], n));
            }
        }
        std::vector<quant::BlockQ8_0> w0(static_cast<size_t>(n / 32));
        quant::quantize_row_q8_0(w.data(), w0.data(), n);
        quant::dot_q8_0_q8_0_rows(w0.data(), p0.data(), rows, n, out.data());
        for (int32_t r = 0; r < rows; ++r) {
            const float single = quant::dot_q8_0_q8_0(w0.data(), p0[static_cast<size_t>(r)], n);
            CHECK_NEAR(out[static_cast<size_t>(r)], single, 1e-4f * (1.0f + std::fabs(single)));
        }
    }
}

// x86-64: the AVX2 kernels are chosen at run time, so both sets live in one
// binary. Activation quantization and decoding must give the scalar code's
// bits (a different rounding there would move every later result); the dots
// may round differently but must agree with the scalar ones closely.
TEST_CASE("Run-time SIMD kernels (x86-64 AVX2): activations and decodes are the scalar bits, dots agree") {
    if (!quant::use_simd_kernels(true) || quant::use_simd_kernels(false)) {
        std::printf("  skipped: the kernel set (%s) is fixed in this build or on this CPU\n", quant::kernel_isa());
        quant::use_simd_kernels(true);
        return;
    }
    struct RestoreSimd {
        ~RestoreSimd() { quant::use_simd_kernels(true); }
    } restore;
    std::mt19937 rng(53);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int64_t n = 1024;
    std::vector<float> x(static_cast<size_t>(n));
    for (float& v : x) v = normal(rng) * (rng() % 64 == 0 ? 8.0f : 1.0f);
    // Block 3: d = 1 and every value a tie (k + 0.5), where std::lround
    // (away from zero) and the vector rounding (to even) differ.
    for (int j = 0; j < 31; ++j) x[static_cast<size_t>(96 + j)] = static_cast<float>(j - 15) - 0.5f;
    x[127] = 127.0f;
    std::fill(x.begin() + 256, x.begin() + 512, 0.0f);  // all-zero blocks and super-block
    x[600] = -1e-40f;                                   // a subnormal
    x[601] = 70000.0f;                                  // beyond fp16: infinity

    struct Outputs {
        std::vector<quant::BlockQ8_0> q8;
        std::vector<quant::BlockQ8_K> q8k;
        std::vector<uint16_t> f16;
        std::vector<float> from_f16, from_bf16, from_q8, from_q4_0, from_q4_1;
    };
    std::vector<quant::BlockQ4_0> q4_0(static_cast<size_t>(n / 32));
    std::vector<quant::BlockQ4_1> q4_1(static_cast<size_t>(n / 32));
    quant::quantize_row_q4_0(x.data(), q4_0.data(), n);
    quant::quantize_row_q4_1(x.data(), q4_1.data(), n);
    std::vector<uint16_t> bf16(static_cast<size_t>(n));
    for (auto& h : bf16) h = static_cast<uint16_t>(rng());
    auto run = [&](bool simd) {
        CHECK(quant::use_simd_kernels(simd) == simd);
        Outputs o;
        o.q8.resize(static_cast<size_t>(n / 32));
        o.q8k.resize(static_cast<size_t>(n / 256));
        o.f16.resize(static_cast<size_t>(n));
        quant::quantize_row_q8_0(x.data(), o.q8.data(), n);
        quant::quantize_row_q8_K(x.data(), o.q8k.data(), n);
        quant::quantize_row(DType::F16, x.data(), o.f16.data(), n);
        for (auto* v : {&o.from_f16, &o.from_bf16, &o.from_q8, &o.from_q4_0, &o.from_q4_1}) v->resize(x.size());
        quant::dequantize_row(DType::F16, o.f16.data(), o.from_f16.data(), n);
        quant::dequantize_row(DType::BF16, bf16.data(), o.from_bf16.data(), n);
        quant::dequantize_row(DType::Q8_0, o.q8.data(), o.from_q8.data(), n);
        quant::dequantize_row(DType::Q4_0, q4_0.data(), o.from_q4_0.data(), n);
        quant::dequantize_row(DType::Q4_1, q4_1.data(), o.from_q4_1.data(), n);
        return o;
    };
    const Outputs scalar = run(false);
    const Outputs simd = run(true);
    auto same = [](const auto& a, const auto& b) {
        return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(a[0])) == 0;
    };
    CHECK(same(scalar.q8, simd.q8));
    CHECK(same(scalar.q8k, simd.q8k));
    CHECK(same(scalar.f16, simd.f16));
    CHECK(same(scalar.from_f16, simd.from_f16));
    CHECK(same(scalar.from_bf16, simd.from_bf16));
    CHECK(same(scalar.from_q8, simd.from_q8));
    CHECK(same(scalar.from_q4_0, simd.from_q4_0));
    CHECK(same(scalar.from_q4_1, simd.from_q4_1));
    CHECK(simd.q8[3].qs[0] == -16 && simd.q8[3].qs[30] == 15);  // -15.5 -> -16, 14.5 -> 15: ties away from zero

    // Extreme Q8_0 bytes: -128 weights (GGUF rows may hold them) against
    // +-127 activations stay exact (no int16 saturation in the pair sums).
    {
        quant::BlockQ8_0 w{}, a{};
        w.d = a.d = quant::fp32_to_fp16(1.0f);
        for (int j = 0; j < 32; ++j) {
            w.qs[j] = -128;
            a.qs[j] = static_cast<int8_t>(j % 2 ? 127 : -127);
        }
        CHECK(quant::dot_q8_0_q8_0(&w, &a, 32) == 0.0f);
        for (int j = 0; j < 32; ++j) a.qs[j] = -127;
        CHECK(quant::dot_q8_0_q8_0(&w, &a, 32) == 32.0f * 128 * 127);
        const quant::BlockQ8_0* rows[2] = {&a, &a};
        float out[2] = {};
        quant::dot_q8_0_q8_0_rows(&w, rows, 2, 32, out);
        CHECK(out[0] == 32.0f * 128 * 127 && out[1] == out[0]);
    }

    // Every format's dot with both sets (the reference vectors as weights).
    std::vector<QuantVector> vectors;
    if (!load_quant_vectors(vectors)) return;
    int compared = 0;
    for (const QuantVector& qv : vectors) {
        const int64_t m = qv.n / 256 * 256 > 0 ? qv.n / 256 * 256 : qv.n / 32 * 32;
        std::vector<quant::BlockQ8_0> xq(static_cast<size_t>(m / 32));
        std::vector<int32_t> xs(xq.size());
        quant::quantize_row_q8_0(simd.from_q8.data(), xq.data(), m);  // any activations; the same for both sets
        quant::block_sums(xq.data(), xs.data(), m);
        double mag = 1e-3;
        for (int64_t i = 0; i < m; ++i) mag += std::fabs(static_cast<double>(qv.expected[i]) * simd.from_q8[i]);
        float got[2], got_k[2] = {0.0f, 0.0f};
        for (const bool use : {false, true}) {
            quant::use_simd_kernels(use);
            got[use] = qv.type == DType::BF16 ? quant::dot_bf16_f32(reinterpret_cast<const uint16_t*>(qv.blocks.data()),
                                                                     simd.from_q8.data(), m)
                                              : quant::dot_quantized(qv.type, qv.blocks.data(), xq.data(), xs.data(), m);
            if (m % 256 == 0) {
                std::vector<quant::BlockQ8_K> xk(static_cast<size_t>(m / 256));
                quant::quantize_row_q8_K(simd.from_q8.data(), xk.data(), m);
                got_k[use] = quant::dot_lowbit_q8_K(qv.type, qv.blocks.data(), xk.data(), m);
            }
        }
        const bool ok = std::fabs(got[0] - got[1]) <= 1e-5 * mag && std::fabs(got_k[0] - got_k[1]) <= 1e-5 * mag;
        if (!ok) {
            std::printf("  %s: scalar %g / %g, simd %g / %g\n", std::string(dtype_traits(qv.type).name).c_str(),
                        got[0], got_k[0], got[1], got_k[1]);
        }
        CHECK(ok);
        ++compared;
    }
    // The float dots (F32 weights, the F16 KV cache).
    for (const int64_t len : {int64_t{7}, int64_t{64}, int64_t{500}}) {  // finite values (x[601] is infinite in F16)
        float got[2], got_h[2];
        for (const bool use : {false, true}) {
            quant::use_simd_kernels(use);
            got[use] = quant::dot_f32(x.data() + 1, simd.from_q8.data(), len);
            got_h[use] = quant::dot_f16_f32(simd.f16.data() + 32, simd.from_q8.data(), len);
        }
        CHECK_NEAR(got[0], got[1], 1e-5 * (1.0 + std::fabs(got[0])) * len);
        CHECK_NEAR(got_h[0], got_h[1], 1e-5 * (1.0 + std::fabs(got_h[0])) * len);
    }
    std::printf("  %d formats compared, kernels %s\n", compared, quant::kernel_isa());
}

TEST_CASE("Repacked K-quants (Q4_K_R8 / Q5_K_R8 / Q6_K_R8): gemm equals gemv exactly, and both match the row kernel") {
    if (!quant::repack_kernels_available() || !detect_cpu().i8mm) {
        std::printf("  skipped: no i8mm on this CPU or build\n");
        return;
    }
    std::mt19937 rng(31);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int64_t rows = 24, cols = 512, nb = cols / 256;
    std::vector<float> w(static_cast<size_t>(rows * cols));
    for (float& v : w) v = normal(rng);
    std::vector<quant::BlockQ4_K> q(static_cast<size_t>(rows * nb));
    for (int64_t r = 0; r < rows; ++r) quant::quantize_row(DType::Q4_K, w.data() + r * cols, q.data() + r * nb, cols);
    std::vector<quant::BlockQ4_Kx8> packed(static_cast<size_t>(rows / 8 * nb));
    quant::repack_q4_K_r8(q.data(), rows, cols, packed.data());
    std::vector<quant::BlockQ8_K> act(static_cast<size_t>(4 * nb));
    for (int t = 0; t < 4; ++t) {
        std::vector<float> x(static_cast<size_t>(cols));
        for (float& v : x) v = normal(rng);
        quant::quantize_row_q8_K(x.data(), act.data() + t * nb, cols);
    }
    const quant::BlockQ8_K* rows4[4] = {act.data(), act.data() + nb, act.data() + 2 * nb, act.data() + 3 * nb};
    std::vector<quant::BlockQ8_Kx4> act4(static_cast<size_t>(nb));
    quant::interleave_q8_K_x4(rows4, cols, act4.data());
    std::vector<float> batch(static_cast<size_t>(4 * rows));
    quant::gemm_q4_K_r8(packed.data(), cols, 0, rows / 8, act4.data(), batch.data(), rows);
    int mismatches = 0;
    for (int t = 0; t < 4; ++t) {
        std::vector<float> single(static_cast<size_t>(rows));
        quant::gemv_q4_K_r8(packed.data(), cols, 0, rows / 8, rows4[t], single.data());
        for (int64_t r = 0; r < rows; ++r) {
            mismatches += single[static_cast<size_t>(r)] != batch[static_cast<size_t>(t * rows + r)];
            const float ref = quant::dot_lowbit_q8_K(DType::Q4_K, q.data() + r * nb, rows4[t], cols);
            CHECK_NEAR(single[static_cast<size_t>(r)], ref, 1e-4f * (1.0f + std::fabs(ref)));
        }
    }
    CHECK(mismatches == 0);
    // A group range writes only its own rows.
    std::vector<float> part(static_cast<size_t>(rows), -1.0f);
    quant::gemv_q4_K_r8(packed.data(), cols, 1, 2, rows4[0], part.data());
    CHECK(part[0] == -1.0f && part[7] == -1.0f && part[16] == -1.0f);
    CHECK(part[8] != -1.0f && part[15] != -1.0f);

    // Q5_K_R8 (5th bits interleaved alongside the nibbles).
    std::vector<quant::BlockQ5_K> q5(static_cast<size_t>(rows * nb));
    for (int64_t r = 0; r < rows; ++r) quant::quantize_row(DType::Q5_K, w.data() + r * cols, q5.data() + r * nb, cols);
    std::vector<quant::BlockQ5_Kx8> packed5(static_cast<size_t>(rows / 8 * nb));
    quant::repack_q5_K_r8(q5.data(), rows, cols, packed5.data());
    quant::gemm_q5_K_r8(packed5.data(), cols, 0, rows / 8, act4.data(), batch.data(), rows);
    for (int t = 0; t < 4; ++t) {
        std::vector<float> single(static_cast<size_t>(rows));
        quant::gemv_q5_K_r8(packed5.data(), cols, 0, rows / 8, rows4[t], single.data());
        for (int64_t r = 0; r < rows; ++r) {
            mismatches += single[static_cast<size_t>(r)] != batch[static_cast<size_t>(t * rows + r)];
            const float ref = quant::dot_lowbit_q8_K(DType::Q5_K, q5.data() + r * nb, rows4[t], cols);
            CHECK_NEAR(single[static_cast<size_t>(r)], ref, 1e-4f * (1.0f + std::fabs(ref)));
        }
    }
    CHECK(mismatches == 0);

    // Q6_K_R8: no Q6_K quantizer, so random blocks (valid for any bytes; d kept finite).
    std::vector<quant::BlockQ6_K> q6(static_cast<size_t>(rows * nb));
    for (quant::BlockQ6_K& blk : q6) {
        auto* bytes = reinterpret_cast<uint8_t*>(&blk);
        for (size_t i = 0; i < sizeof blk; ++i) bytes[i] = static_cast<uint8_t>(rng());
        blk.d = quant::fp32_to_fp16(0.002f + 0.001f * static_cast<float>(rng() % 7));
    }
    std::vector<quant::BlockQ6_Kx8> packed6(static_cast<size_t>(rows / 8 * nb));
    quant::repack_q6_K_r8(q6.data(), rows, cols, packed6.data());
    quant::gemm_q6_K_r8(packed6.data(), cols, 0, rows / 8, act4.data(), batch.data(), rows);
    mismatches = 0;
    for (int t = 0; t < 4; ++t) {
        std::vector<float> single(static_cast<size_t>(rows));
        quant::gemv_q6_K_r8(packed6.data(), cols, 0, rows / 8, rows4[t], single.data());
        for (int64_t r = 0; r < rows; ++r) {
            mismatches += single[static_cast<size_t>(r)] != batch[static_cast<size_t>(t * rows + r)];
            const float ref = quant::dot_lowbit_q8_K(DType::Q6_K, q6.data() + r * nb, rows4[t], cols);
            CHECK_NEAR(single[static_cast<size_t>(r)], ref, 1e-4f * (1.0f + std::fabs(ref)));
        }
    }
    CHECK(mismatches == 0);
}

TEST_CASE("Repacked Q8_0_R4: gemm equals gemv exactly, and both match the row kernel") {
    if (!quant::repack_kernels_available() || !detect_cpu().i8mm) {
        std::printf("  skipped: no i8mm on this CPU or build\n");
        return;
    }
    std::mt19937 rng(37);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int64_t rows = 12, cols = 96, nb = cols / 32;
    std::vector<float> w(static_cast<size_t>(rows * cols));
    for (float& v : w) v = normal(rng);
    std::vector<quant::BlockQ8_0> q(static_cast<size_t>(rows * nb));
    for (int64_t r = 0; r < rows; ++r) quant::quantize_row_q8_0(w.data() + r * cols, q.data() + r * nb, cols);
    std::vector<quant::BlockQ8_0x4> packed(static_cast<size_t>(rows / 4 * nb));
    quant::repack_q8_0_r4(q.data(), rows, cols, packed.data());
    std::vector<quant::BlockQ8_0> act(static_cast<size_t>(4 * nb));
    for (int t = 0; t < 4; ++t) {
        std::vector<float> x(static_cast<size_t>(cols));
        for (float& v : x) v = normal(rng);
        quant::quantize_row_q8_0(x.data(), act.data() + t * nb, cols);
    }
    const quant::BlockQ8_0* rows4[4] = {act.data(), act.data() + nb, act.data() + 2 * nb, act.data() + 3 * nb};
    std::vector<quant::BlockQ8_0x4> act4(static_cast<size_t>(nb));
    quant::interleave_q8_0_x4(rows4, cols, act4.data());
    std::vector<float> batch(static_cast<size_t>(4 * rows));
    quant::gemm_q8_0_r4(packed.data(), cols, 0, rows / 4, act4.data(), batch.data(), rows);
    int mismatches = 0;
    for (int t = 0; t < 4; ++t) {
        std::vector<float> single(static_cast<size_t>(rows));
        quant::gemv_q8_0_r4(packed.data(), cols, 0, rows / 4, rows4[t], single.data());
        for (int64_t r = 0; r < rows; ++r) {
            mismatches += single[static_cast<size_t>(r)] != batch[static_cast<size_t>(t * rows + r)];
            const float ref = quant::dot_q8_0_q8_0(q.data() + r * nb, rows4[t], cols);
            CHECK_NEAR(single[static_cast<size_t>(r)], ref, 1e-4f * (1.0f + std::fabs(ref)));
        }
    }
    CHECK(mismatches == 0);
    std::vector<float> part(static_cast<size_t>(rows), -1.0f);
    quant::gemv_q8_0_r4(packed.data(), cols, 1, 2, rows4[0], part.data());
    CHECK(part[3] == -1.0f && part[8] == -1.0f);
    CHECK(part[4] != -1.0f && part[7] != -1.0f);
}

TEST_CASE("CPU backend on repacked weights: any batch size gives each row's single-row result") {
    if (!quant::repack_kernels_available() || !detect_cpu().i8mm) {
        std::printf("  skipped: no i8mm on this CPU or build\n");
        return;
    }
    std::mt19937 rng(41);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int64_t rows = 52, cols = 256;  // groups of 8 / 4 cut by the threads' row ranges
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    std::vector<float> x(static_cast<size_t>(6 * cols));
    for (float& v : x) v = normal(rng);
    std::vector<float> w(static_cast<size_t>(rows * cols));
    for (float& v : w) v = normal(rng);
    for (const auto& [from, to] : {std::pair{DType::Q4_K, DType::Q4_K_R8}, std::pair{DType::Q5_K, DType::Q5_K_R8},
                                   std::pair{DType::Q8_0, DType::Q8_0_R4}}) {
        const int64_t group = to == DType::Q8_0_R4 ? 4 : 8;
        TensorView t;
        t.type = from;
        t.n_dims = 2;
        t.ne = {cols, rows - rows % group, 1, 1};
        const size_t bytes = t.row_bytes() * static_cast<size_t>(t.rows());
        std::vector<uint8_t> file(bytes), packed(bytes);
        for (int64_t r = 0; r < t.rows(); ++r) quant::quantize_row(from, w.data() + r * cols, file.data() + r * t.row_bytes(), cols);
        if (to == DType::Q4_K_R8) {
            quant::repack_q4_K_r8(reinterpret_cast<const quant::BlockQ4_K*>(file.data()), t.rows(), cols,
                                  reinterpret_cast<quant::BlockQ4_Kx8*>(packed.data()));
        } else if (to == DType::Q5_K_R8) {
            quant::repack_q5_K_r8(reinterpret_cast<const quant::BlockQ5_K*>(file.data()), t.rows(), cols,
                                  reinterpret_cast<quant::BlockQ5_Kx8*>(packed.data()));
        } else {
            quant::repack_q8_0_r4(reinterpret_cast<const quant::BlockQ8_0*>(file.data()), t.rows(), cols,
                                  reinterpret_cast<quant::BlockQ8_0x4*>(packed.data()));
        }
        t.type = to;
        t.data = packed.data();
        t.nbytes = bytes;
        std::vector<float> single(static_cast<size_t>(6 * t.rows()));
        for (int k = 0; k < 6; ++k) {
            REQUIRE(cpu->matmul(t, x.data() + k * cols, single.data() + k * t.rows(), 1).is_ok());
        }
        int mismatches = 0;
        for (int32_t n = 2; n <= 6; ++n) {
            std::vector<float> y(static_cast<size_t>(n * t.rows()), -1.0f);
            REQUIRE(cpu->matmul(t, x.data(), y.data(), n).is_ok());
            for (size_t i = 0; i < y.size(); ++i) mismatches += y[i] != single[i];
        }
        CHECK(mismatches == 0);
    }
}

// Reproducibility rests on this: whichever kernel computes a row (one
// activation row or several, the file's layout or the repacked one a streamed
// expert gets when read for a batched pass), the result is the same bits. The
// same prompt otherwise gave two different greedy replies, since the layout
// follows I/O timing and the batch follows how the prompt was split.
TEST_CASE("Every weight format: a row's result is the same bits whatever the batch or the layout") {
    std::vector<QuantVector> vectors;
    if (!load_quant_vectors(vectors)) return;
    std::mt19937 rng(43);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const bool i8mm = quant::repack_kernels_available() && detect_cpu().i8mm;
    constexpr int64_t rows = 16;  // whole groups of every repacked layout (8 or 4 rows)
    constexpr int32_t batch = 6;  // single-row, 2-4-row and 4 + remainder kernels
    // The reference formats (tests/data/quant_vectors.bin), and F32 / F16 rows written here.
    std::vector<std::pair<DType, std::vector<uint8_t>>> formats;
    for (const QuantVector& qv : vectors) {
        if (std::none_of(formats.begin(), formats.end(), [&](const auto& f) { return f.first == qv.type; })) {
            formats.emplace_back(qv.type, qv.blocks);
        }
    }
    for (const DType type : {DType::F32, DType::F16}) {
        std::vector<float> v(256);
        for (float& e : v) e = normal(rng);
        std::vector<uint8_t> bytes(v.size() * dtype_traits(type).block_bytes);
        quant::quantize_row(type, v.data(), bytes.data(), static_cast<int64_t>(v.size()));
        formats.emplace_back(type, std::move(bytes));
    }
    int checked = 0;
    for (const auto& [type, blocks] : formats) {
        const DTypeTraits traits = dtype_traits(type);
        const size_t block_bytes = traits.block_bytes;
        const auto n_blocks = static_cast<int64_t>(blocks.size() / block_bytes);
        const int64_t cols = n_blocks * traits.block_size;
        // Row r: the blocks rotated by r, so the rows differ.
        std::vector<uint8_t> file(static_cast<size_t>(rows) * blocks.size());
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t b = 0; b < n_blocks; ++b) {
                std::memcpy(file.data() + static_cast<size_t>(r * n_blocks + b) * block_bytes,
                            blocks.data() + static_cast<size_t>((b + r) % n_blocks) * block_bytes, block_bytes);
            }
        }
        TensorView w;
        w.name = "w";
        w.type = type;
        w.n_dims = 2;
        w.ne = {cols, rows, 1, 1};
        w.data = file.data();
        w.nbytes = file.size();
        // Activations with a heavy tail, as real hidden states have.
        std::vector<float> x(static_cast<size_t>(batch * cols));
        for (float& v : x) v = normal(rng) * (rng() % 64 == 0 ? 8.0f : 1.0f);
        std::vector<float> single(static_cast<size_t>(batch * rows));
        for (int32_t t = 0; t < batch; ++t) {
            REQUIRE(cpu->matmul(w, x.data() + t * cols, single.data() + t * rows, 1).is_ok());
        }
        auto same_bits = [&](const TensorView& m, int32_t n) {
            std::vector<float> y(static_cast<size_t>(n * rows), -1.0f);
            if (!cpu->matmul(m, x.data(), y.data(), n).is_ok()) return false;
            return std::memcmp(y.data(), single.data(), y.size() * sizeof(float)) == 0;
        };
        const std::string name(traits.name);
        for (int32_t n = 2; n <= batch; ++n) {
            const bool same = same_bits(w, n);
            if (!same) std::printf("  %s: %d rows at once differ from one at a time\n", name.c_str(), n);
            CHECK(same);
        }
        // The formats with an i8mm layout: the repacked matrix gives the same bits too.
        DType packed_type = type;
        std::vector<uint8_t> packed(file.size());
        if (i8mm && type == DType::Q4_K) {
            quant::repack_q4_K_r8(reinterpret_cast<const quant::BlockQ4_K*>(file.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ4_Kx8*>(packed.data()));
            packed_type = DType::Q4_K_R8;
        } else if (i8mm && type == DType::Q5_K) {
            quant::repack_q5_K_r8(reinterpret_cast<const quant::BlockQ5_K*>(file.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ5_Kx8*>(packed.data()));
            packed_type = DType::Q5_K_R8;
        } else if (i8mm && type == DType::Q6_K) {
            quant::repack_q6_K_r8(reinterpret_cast<const quant::BlockQ6_K*>(file.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ6_Kx8*>(packed.data()));
            packed_type = DType::Q6_K_R8;
        } else if (i8mm && type == DType::Q8_0) {
            quant::repack_q8_0_r4(reinterpret_cast<const quant::BlockQ8_0*>(file.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ8_0x4*>(packed.data()));
            packed_type = DType::Q8_0_R4;
        }
        if (packed_type != type) {
            TensorView r = w;
            r.type = packed_type;
            r.data = packed.data();
            for (int32_t n = 1; n <= batch; ++n) {
                const bool same = same_bits(r, n);
                if (!same) std::printf("  %s: the repacked layout differs (%d rows)\n", name.c_str(), n);
                CHECK(same);
            }
        }
        ++checked;
    }
    CHECK(checked == static_cast<int>(conformance::weight_formats().size()));  // every format the loader reads
}

TEST_CASE("Low-bit Q8_K kernels equal the dot of their dequantized operands (llama.cpp vectors)") {
    std::vector<QuantVector> vectors;
    if (!load_quant_vectors(vectors)) return;
    const DType lowbit[] = {DType::Q2_K,   DType::Q3_K,    DType::Q4_K,   DType::Q5_K,  DType::Q6_K,
                            DType::TQ2_0,  DType::IQ1_S,   DType::IQ1_M,  DType::IQ2_XXS, DType::IQ2_XS,
                            DType::IQ2_S,  DType::IQ3_XXS, DType::IQ3_S,  DType::IQ4_XS};
    std::mt19937 rng(29);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    int checked = 0;
    for (const QuantVector& qv : vectors) {
        if (std::find(std::begin(lowbit), std::end(lowbit), qv.type) == std::end(lowbit)) continue;
        const int64_t n = qv.n;  // 4 super-blocks of llama.cpp-encoded weights
        REQUIRE(n % 256 == 0);
        // Activations with a heavy tail (as in real hidden states) so the
        // per-256 scale differs from the per-32 one.
        std::vector<float> x(static_cast<size_t>(n));
        for (auto& v : x) v = normal(rng) * (rng() % 64 == 0 ? 8.0f : 1.0f);
        std::vector<quant::BlockQ8_K> xk(static_cast<size_t>(n / 256));
        quant::quantize_row_q8_K(x.data(), xk.data(), n);
        std::vector<float> w(static_cast<size_t>(n));
        quant::dequantize_row(qv.type, qv.blocks.data(), w.data(), n);

        // 1. Kernel == float dot of the dequantized weights and the dequantized Q8_K activations.
        double reference = 0.0, mag = 1e-6, exact = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double xd = static_cast<double>(xk[i / 256].d) * xk[i / 256].qs[i % 256];
            reference += static_cast<double>(w[i]) * xd;
            exact += static_cast<double>(w[i]) * x[i];
            mag += std::fabs(static_cast<double>(w[i]) * x[i]);
        }
        const float kernel = quant::dot_lowbit_q8_K(qv.type, qv.blocks.data(), xk.data(), n);
        const bool exact_ok = std::fabs(kernel - reference) <= 1e-5 * mag;
        CHECK(exact_ok);

        // 2. Loosely equal to the Q8_0 path and to the unquantized activations
        //    (only activation-quantization error separates them).
        std::vector<quant::BlockQ8_0> xq(static_cast<size_t>(n / 32));
        quant::quantize_row_q8_0(x.data(), xq.data(), n);
        std::vector<int32_t> xsums(xq.size());
        quant::block_sums(xq.data(), xsums.data(), n);
        const float old_path = quant::is_extended(qv.type)
                                   ? quant::dot_ext_q8_0(qv.type, qv.blocks.data(), xq.data(), n)
                                   : quant::dot_quantized(qv.type, qv.blocks.data(), xq.data(), xsums.data(), n);
        const bool old_ok = std::fabs(kernel - old_path) <= 2e-2 * mag;
        const bool float_ok = std::fabs(kernel - exact) <= 2e-2 * mag;
        CHECK(old_ok);
        CHECK(float_ok);

        // 3. The CPU backend routes the type through the same kernel (2 activation rows).
        TensorView view;
        view.name = "lowbit";
        view.type = qv.type;
        view.n_dims = 2;
        view.ne = {n, 1, 1, 1};
        view.data = qv.blocks.data();
        view.nbytes = qv.blocks.size();
        std::vector<float> x2(x);
        for (int64_t i = 0; i < n; ++i) x2.push_back(-0.5f * x[n - 1 - i]);
        float y[2] = {0.0f, 0.0f};
        REQUIRE(cpu->matmul(view, x2.data(), y, 2).is_ok());
        quant::quantize_row_q8_K(x2.data() + n, xk.data(), n);
        const float second = quant::dot_lowbit_q8_K(qv.type, qv.blocks.data(), xk.data(), n);
        if (quant::uses_q8_K(qv.type)) {
            CHECK(y[0] == kernel);
            CHECK(y[1] == second);
        } else {
            CHECK(y[0] == old_path);
        }
        std::printf("  %-8s kernel-vs-dequant %.1e  vs Q8_0 path %.1e  vs float %.1e  (of %.2f)\n",
                    std::string(dtype_traits(qv.type).name).c_str(), std::fabs(kernel - reference) / mag,
                    std::fabs(kernel - old_path) / mag, std::fabs(kernel - exact) / mag, mag);
        ++checked;
    }
    CHECK(checked == static_cast<int>(std::size(lowbit)));
}

TEST_CASE("CPU backend batched matmul equals per-row dots") {
    const auto path = mixed_model();
    auto loader = MmapLoader::open(path);
    REQUIRE(loader.has_value());
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    for (const char* name : {"blk.0.attn_q.weight", "blk.0.ffn_up.weight", "token_embd.weight"}) {
        const TensorView& w = *loader.value()->tensor(name);
        const int n = 3;
        std::vector<float> x(static_cast<size_t>(n * w.cols())), y(static_cast<size_t>(n * w.rows()));
        std::mt19937 rng(5);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto& v : x) v = normal(rng);
        REQUIRE(cpu->matmul(w, x.data(), y.data(), n).is_ok());
        for (int t = 0; t < n; ++t) {
            std::vector<float> y1(static_cast<size_t>(w.rows()));
            REQUIRE(cpu->matmul(w, x.data() + t * w.cols(), y1.data(), 1).is_ok());
            CHECK(std::equal(y1.begin(), y1.end(), y.begin() + t * w.rows()));
        }
    }
}

TEST_CASE("GPU backends (Metal / Vulkan) agree with the CPU backend") {
  for (auto make : {make_metal_backend, make_vulkan_backend}) {
    auto metal = make();
    if (!metal) {
        std::printf("  skipped: %s\n", metal.status().to_string().c_str());
        continue;
    }
    std::printf("  %s\n", metal.value()->description().c_str());
    auto loader = MmapLoader::open(mixed_model());
    REQUIRE(loader.has_value());
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    for (const char* name : {"blk.0.attn_q.weight", "blk.1.ffn_down.weight", "token_embd.weight", "output.weight"}) {
        const TensorView& w = *loader.value()->tensor(name);
        const int n = 2;
        std::vector<float> x(static_cast<size_t>(n * w.cols())), a(static_cast<size_t>(n * w.rows())), b(a.size());
        std::mt19937 rng(9);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (auto& v : x) v = normal(rng);
        REQUIRE(cpu->matmul(w, x.data(), a.data(), n).is_ok());
        REQUIRE(metal.value()->matmul(w, x.data(), b.data(), n).is_ok());
        double scale = 0;
        for (const float v : a) scale = std::max(scale, std::fabs(static_cast<double>(v)));
        CHECK(max_abs_diff(b, a) <= 0.02 * std::max(1.0, scale));  // CPU quantizes activations
    }
  }
}

TEST_CASE("Classic CPU operations compose with GPU projections through prefill, decode and rollback") {
    for (auto make : {make_metal_backend, make_vulkan_backend}) {
        for (const std::string arch : {"gpt2", "falcon", "bloom", "apertus", "deepseek2"}) {
            auto spec = test::classic_model_spec(arch);
            if (arch == "deepseek2") {
                spec = {};
                spec.arch = arch;
                spec.n_head_kv = 1;
                spec.rope_dim = 32;
                spec.mla_kv_rank = spec.mla_q_rank = 64;
                spec.attn_output_bias = true;
            }
            spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F16;
            TransformerOptions options;
            options.kv_type = KvCacheType::F16;
            options.latent_form = LatentForm::Expanded;  // latent prefill expands; single-token decode absorbs
            const auto path = model_path("classic_gpu_" + arch, spec);
            auto reference = load_transformer(path, options), accelerated = load_transformer(path, options);
            REQUIRE(reference != nullptr && accelerated != nullptr);
            // Destroy the accelerator before the models whose mapped weights
            // it borrows, even when its buffer cache retains registrations.
            auto accelerator = make();
            if (!accelerator) {
                std::printf("  skipped: %s\n", accelerator.status().to_string().c_str());
                break;
            }
            ThreadPool pool(4);
            auto cpu = make_cpu_backend(pool);
            const Route plain{cpu.get(), cpu.get(), cpu.get()};
            const Route mixed{cpu.get(), accelerator.value().get(), accelerator.value().get()};
            auto compare = [&](const std::vector<int32_t>& ids) {
                auto a = reference->forward(ids, Transformer::Logits::All, plain, pool);
                auto b = accelerated->forward(ids, Transformer::Logits::All, mixed, pool);
                if (!a || !b) {
                    std::printf("  %s: CPU %s; accelerator %s\n", arch.c_str(),
                                a ? "ok" : a.status().to_string().c_str(), b ? "ok" : b.status().to_string().c_str());
                    return false;
                }
                if (a->size() != b->size()) return false;
                double scale = 1.0, error = 0.0;
                for (size_t i = 0; i < a->size(); ++i) {
                    if (!std::isfinite((*a)[i]) || !std::isfinite((*b)[i])) return false;
                    scale = std::max(scale, std::fabs(static_cast<double>((*a)[i])));
                    error = std::max(error, std::fabs(static_cast<double>((*a)[i]) - (*b)[i]));
                }
                if (error > 1e-3 * scale) std::printf("  %s: relative GPU error %.6g\n", arch.c_str(), error / scale);
                return error <= 1e-3 * scale;
            };
            REQUIRE(compare({1, 270, 280, 290, 300, 271}));
            REQUIRE(compare({280}));
            REQUIRE(reference->truncate(3).is_ok() && accelerated->truncate(3).is_ok());
            REQUIRE(compare({272, 273}));
        }
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
TEST_CASE("SPM tokenizer merges by score and falls back to bytes") {
    auto loader = MmapLoader::open(mixed_model());
    REQUIRE(loader.has_value());
    auto tok = Tokenizer::load(*loader.value());
    REQUIRE(tok.has_value());
    const Tokenizer& t = tok.value();
    CHECK(t.bos() == 1 && t.eos() == 2);
    CHECK(t.is_end_of_generation(2));

    auto ids = t.encode("hello world", true);
    REQUIRE(ids.has_value());
    const std::vector<int32_t> expected = {1, test::tiny_token("\xE2\x96\x81hello"), test::tiny_token("\xE2\x96\x81world")};
    CHECK(ids.value() == expected);

    // "é" is not in the vocabulary: two byte tokens that decode back to UTF-8.
    auto accented = t.encode("hé", false);
    REQUIRE(accented.has_value());
    std::string decoded;
    for (const int32_t id : accented.value()) decoded += t.piece(id);
    CHECK(decoded == " h\xC3\xA9");
    CHECK(t.piece(1).empty());  // control tokens produce no text

    // Control tokens written in the text map to their ids (chat templates).
    auto templated = t.encode("hello</s>world", false);
    REQUIRE(templated.has_value());
    const std::vector<int32_t> expected_ids = {test::tiny_token("\xE2\x96\x81hello"), 2,
                                               test::tiny_token("\xE2\x96\x81world")};
    CHECK(templated.value() == expected_ids);
}

TEST_CASE("Metaspace first: controls preserve real spaces without adding dummy spaces") {
    auto spec = test::classic_model_spec("nemotron");
    auto explicit_file = MmapLoader::open(model_path("nemotron_explicit_bos", spec));
    REQUIRE(explicit_file.has_value());
    auto explicit_tok = Tokenizer::load(*explicit_file.value());
    REQUIRE(explicit_tok.has_value());
    CHECK(explicit_tok->add_bos_default());  // metadata wins over the absent-key default
    spec.add_bos = -1;
    auto loader = MmapLoader::open(model_path("nemotron_metaspace", spec));
    REQUIRE(loader.has_value());
    auto tok = Tokenizer::load(*loader.value());
    REQUIRE(tok.has_value());
    const Tokenizer& t = *tok;
    CHECK(!t.add_bos_default());
    const int32_t hello = test::tiny_token("▁hello"), world = test::tiny_token("▁world");
    auto initial = t.encode("hello world", true);
    REQUIRE(initial.has_value());
    CHECK(*initial == std::vector<int32_t>({t.bos(), hello, world}));
    auto joined = t.encode("hello</s>world", false);
    auto continuation = t.encode_continuation("world", t.eos());
    REQUIRE(joined.has_value() && continuation.has_value());
    std::vector<int32_t> expected = {hello, t.eos()};
    expected.insert(expected.end(), continuation->begin(), continuation->end());
    CHECK(*joined == expected);
    std::string plain;
    for (const auto id : *continuation) plain += t.piece(id);
    CHECK(plain == "world");
    auto spaced = t.encode("hello</s> world", false);
    REQUIRE(spaced.has_value());
    CHECK(*spaced == std::vector<int32_t>({hello, t.eos(), world}));
    CHECK(t.prompt_text(world, -1) == "world");
    CHECK(t.prompt_text(world, t.eos()) == " world");
    auto explicit_bos = t.encode("<s> hello", true);
    REQUIRE(explicit_bos.has_value());
    CHECK(*explicit_bos == std::vector<int32_t>({t.bos(), hello}));
}

TEST_CASE("SentencePiece byte fallback follows token types and groups unknown runs") {
    for (const int32_t type : {1, 6}) {
        test::GgufWriter w;
        w.add_string("general.architecture", "llama");
        w.add_string("tokenizer.ggml.model", "llama");
        w.add_string_array("tokenizer.ggml.tokens", {"<unk>", "<s>", "</s>", "<0xCD>", "<0x91>", "x"});
        w.add_i32_array("tokenizer.ggml.token_type", {2, 3, 3, type, type, 1});
        w.add_u32("tokenizer.ggml.unknown_token_id", 0);
        w.add_bool("tokenizer.ggml.add_space_prefix", false);
        const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_byte_types.gguf";
        w.write(path);
        auto file = MmapLoader::open(path);
        REQUIRE(file.has_value());
        auto tok = Tokenizer::load(*file.value());
        REQUIRE(tok.has_value());
        auto ids = tok->encode("\u0351\u0351x", false);
        REQUIRE(ids.has_value());
        CHECK(*ids == (type == 6 ? std::vector<int32_t>({3, 4, 3, 4, 5}) : std::vector<int32_t>({0, 5})));
        CHECK(tok->piece(3) == (type == 6 ? std::string_view("\xcd") : std::string_view("<0xCD>")));
        std::remove(path.c_str());
    }
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------
TEST_CASE("Transformer matches the float reference (F32 weights, F16 KV)") {
    TransformerOptions options;
    options.kv_type = KvCacheType::F16;
    auto model = load_transformer(f32_model(), options);
    REQUIRE(model != nullptr);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};

    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(logits.has_value());
    const Reference ref(model->file());
    const auto expected = ref.logits(tokens);
    const auto vocab = static_cast<size_t>(model->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        const auto row = logits->subspan(t * vocab, vocab);
        CHECK(max_abs_diff(row, expected[t]) < 2e-2);
        CHECK(Sampler::argmax(row) == Sampler::argmax(expected[t]));
    }
}

TEST_CASE("Learned attention sinks and output bias match a full softmax reference") {
    auto spec = f32_spec();
    spec.attention_sinks = spec.attention_output_bias = true;
    const auto& path = model_path("learned_sinks_reference", spec);
    auto file = MmapLoader::open(path);
    REQUIRE(file.has_value());
    const Reference ref(*file.value());
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 36; ++i) tokens.push_back(3 + (i * 37) % 250);
    for (const int window : {0, 12}) {
        TransformerOptions options;
        options.kv_type = KvCacheType::F16;
        options.sliding_window = window;
        options.sink_tokens = window ? 2 : 0;
        auto model = load_transformer(path, options);
        REQUIRE(model != nullptr);
        const auto expected = ref.logits(tokens, window, options.sink_tokens);
        auto got = model->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(got.has_value());
        for (size_t t = 0; t < tokens.size(); ++t) {
            CHECK(max_abs_diff(got->subspan(t * expected[t].size(), expected[t].size()), expected[t]) < 0.003f);
        }
    }
}

TEST_CASE("Biased clamped experts and sinks agree through streamed fused projections and rollback") {
    test::TinyModelSpec spec;
    spec.arch = "gpt-oss";
    spec.n_expert = 8;
    spec.n_expert_used = 3;
    spec.identical_experts = false;
    spec.router_bias = spec.expert_projection_bias = true;
    spec.attention_sinks = spec.attention_output_bias = true;
    spec.sliding_window = 12;
    auto fused = spec;
    fused.fused_gate_up = true;
    const auto& split_path = model_path("gpt_oss_split_bias", spec);
    const auto& fused_path = model_path("gpt_oss_fused_bias", fused);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 36; ++i) tokens.push_back(3 + (i * 37) % 250);
    for (const auto type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        TransformerOptions options;
        options.kv_type = type;
        options.expert_cache_bytes = 0;
        options.keep_windowed_rows = true;  // speculative rejection can revisit rows before the window
        auto a = load_transformer(split_path, options);
        options.expert_cache_bytes = 1;
        auto b = load_transformer(fused_path, options);
        REQUIRE(a != nullptr && b != nullptr);
        REQUIRE(b->expert_store() != nullptr);
        auto expected = a->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(expected.has_value());
        const std::vector<float> expected_copy(expected->begin(), expected->end());
        auto got = b->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(got.has_value());
        CHECK(max_abs_diff(expected_copy, *got) == 0.0f);
        a->reset();
        b->reset();
        for (const int32_t token : tokens) {
            auto x = a->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
            auto y = b->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
            REQUIRE(x.has_value() && y.has_value());
            CHECK(max_abs_diff(*x, *y) == 0.0f);
        }
        REQUIRE(b->truncate(24).is_ok());
        auto replay = b->forward(std::span<const int32_t>(tokens).subspan(24), Transformer::Logits::Last, route, pool);
        REQUIRE(replay.has_value());
        auto last = std::span<const float>(expected_copy).last(replay->size());
        CHECK(max_abs_diff(*replay, last) < 0.002f);
    }
}

TEST_CASE("Mixtral expert width and router normalization follow tensors and architecture defaults") {
    test::TinyModelSpec spec;
    spec.n_expert = 4;
    spec.n_expert_used = 2;
    spec.omit_expert_width = true;
    spec.expert_weights_norm = -1;
    auto model = load_transformer(model_path("mixtral_metadata", spec));
    REQUIRE(model != nullptr);
    CHECK(model->config().n_ff_expert == spec.n_ff);
    CHECK(model->config().moe_norm_weights);
}

TEST_CASE("YaRN endpoint rounding follows architecture defaults and explicit file overrides") {
    test::TinyModelSpec spec;
    spec.arch = "gpt-oss";
    spec.n_expert = 4;
    spec.n_expert_used = 2;
    spec.yarn_factor = 4.0f;
    spec.yarn_ctx = 32;
    auto implicit = load_transformer(model_path("gpt_yarn_implicit", spec));
    REQUIRE(implicit != nullptr);
    CHECK(!implicit->config().yarn.truncate);
    spec.yarn_truncate = 0;
    auto fractional = load_transformer(model_path("gpt_yarn_fractional", spec));
    REQUIRE(fractional != nullptr);
    CHECK(!fractional->config().yarn.truncate);
    spec.yarn_truncate = 1;
    auto rounded = load_transformer(model_path("gpt_yarn_rounded", spec));
    REQUIRE(rounded != nullptr);
    CHECK(rounded->config().yarn.truncate);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto a = implicit->forward(tokens, Transformer::Logits::All, route, pool);
    auto b = fractional->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(a.has_value() && b.has_value());
    CHECK(max_abs_diff(*a, *b) == 0.0f);
}

TEST_CASE("Post-SiLU clamps match the independent float reference and change a saturated FFN") {
    auto spec = f32_spec();
    spec.shared_swiglu_clamp = 0.25f;
    TransformerOptions options;
    options.kv_type = KvCacheType::F16;
    auto model = load_transformer(model_path("post_silu_clamp", spec), options);
    REQUIRE(model != nullptr);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto got = model->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(got.has_value());
    const auto expected = Reference(model->file()).logits(tokens);
    const size_t vocab = expected.front().size();
    for (size_t t = 0; t < tokens.size(); ++t) CHECK(max_abs_diff(got->subspan(t * vocab, vocab), expected[t]) < 0.003f);
    auto plain = load_transformer(f32_model(), options);
    REQUIRE(plain != nullptr);
    auto before = plain->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(before.has_value());
    CHECK(max_abs_diff(*before, *got) > 0.01f);
}

TEST_CASE("Biased MXFP4 experts and wider attention agree through block streaming") {
    test::TinyModelSpec spec;
    spec.arch = "gpt-oss";
    spec.head_dim = 64;
    spec.n_expert = 8;
    spec.n_expert_used = 3;
    spec.ffn_type = DType::MXFP4;
    spec.router_bias = spec.expert_projection_bias = true;
    spec.attention_sinks = spec.attention_output_bias = true;
    spec.identical_experts = false;
    const auto& path = model_path("moe_wider_attention_streaming", spec);
    TransformerOptions options;
    options.expert_cache_bytes = 0;
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    LoaderOptions resident, streaming;
    resident.streaming = false;
    streaming.streaming = true;
    auto a = load_transformer(path, options, resident);
    auto b = load_transformer(path, options, streaming);
    REQUIRE(a != nullptr && b != nullptr);
    CHECK(b->file().streaming());
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto x = a->forward(tokens, Transformer::Logits::All, route, pool);
    auto y = b->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(x.has_value() && y.has_value());
    CHECK(max_abs_diff(*x, *y) == 0.0f);
    const int32_t token = 280;
    x = a->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
    y = b->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
    REQUIRE(x.has_value() && y.has_value());
    CHECK(max_abs_diff(*x, *y) == 0.0f);
}

TEST_CASE("MXFP4 prefill kernels preserve decode bits for odd blocks and partial row tiles") {
    std::mt19937 rng(818);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (const int64_t n : {32, 96, 256, 1056}) {
        std::vector<quant::BlockMxfp4> w(static_cast<size_t>(n / 32));
        for (auto& b : w) {
            b.e = static_cast<uint8_t>(124 + rng() % 7);
            for (auto& q : b.qs) q = static_cast<uint8_t>(rng());
        }
        std::vector<quant::BlockQ8_0> x(static_cast<size_t>(9 * n / 32));
        std::vector<float> input(static_cast<size_t>(9 * n));
        for (float& v : input) v = normal(rng);
        quant::quantize_row_q8_0(input.data(), x.data(), 9 * n);
        const quant::BlockQ8_0* rows[9];
        for (size_t r = 0; r < 9; ++r) rows[r] = x.data() + r * static_cast<size_t>(n / 32);
        for (const int count : {1, 2, 3, 4, 5, 7, 8, 9}) {
            float got[9]{};
            quant::dot_mxfp4_q8_0_rows(w.data(), rows, count, n, got);
            for (int r = 0; r < count; ++r) {
                const float expected = quant::dot_mxfp4_q8_0(w.data(), rows[r], n);
                CHECK(std::memcmp(&got[r], &expected, sizeof(float)) == 0);
            }
        }
    }
}

TEST_CASE("Separate attention output gates preserve fused-query gate bits for every KV type") {
    for (const bool head : {false, true}) {
    test::TinyModelSpec spec;
    spec.arch = "qwen3";
    spec.qk_norm = spec.attn_gate = true;
    spec.head_attention_gate = head;
    auto split = spec;
    split.separate_attention_gate = true;
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    for (const auto type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        TransformerOptions options;
        options.kv_type = type;
        auto a = load_transformer(model_path("attention_fused_gate" + std::to_string(head), spec), options);
        auto b = load_transformer(model_path("attention_separate_gate" + std::to_string(head), split), options);
        REQUIRE(a != nullptr && b != nullptr);
        CHECK(a->file().tensor("blk.0.attn_gate.weight") == nullptr);
        REQUIRE(b->file().tensor("blk.0.attn_gate.weight") != nullptr);
        auto x = a->forward(tokens, Transformer::Logits::All, route, pool);
        auto y = b->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(x.has_value() && y.has_value());
        CHECK(max_abs_diff(*x, *y) == 0.0f);
        const int32_t token = 280;
        x = a->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
        y = b->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
        REQUIRE(x.has_value() && y.has_value());
        CHECK(max_abs_diff(*x, *y) == 0.0f);
    }
    }
}

// Grouped-query attention scores a KV head's query group together: every
// group size must give the reference's logits, odd and prime ones included
// (Qwen2.5-0.5B has 7 query heads per KV head).
TEST_CASE("Transformer matches the float reference for every query-group size") {
    struct Heads {
        int32_t n_head, n_head_kv;
    };
    for (const Heads h : {Heads{4, 4}, Heads{4, 2}, Heads{6, 2}, Heads{14, 2}, Heads{5, 1}, Heads{12, 4}}) {
        test::TinyModelSpec spec = f32_spec();
        spec.n_head = h.n_head;
        spec.n_head_kv = h.n_head_kv;
        spec.n_embd = 32 * h.n_head;  // head_dim 32
        TransformerOptions options;
        options.kv_type = KvCacheType::F16;
        const std::string name = "heads_" + std::to_string(h.n_head) + "_" + std::to_string(h.n_head_kv);
        auto model = load_transformer(model_path(name, spec), options);
        REQUIRE(model != nullptr);
        ThreadPool pool(4);
        auto cpu = make_cpu_backend(pool);
        const Route route{cpu.get(), cpu.get(), cpu.get()};
        const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
        auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(logits.has_value());
        const auto expected = Reference(model->file()).logits(tokens);
        const auto vocab = static_cast<size_t>(model->config().n_vocab);
        double worst = 0.0;
        for (size_t t = 0; t < tokens.size(); ++t) {
            worst = std::max(worst, static_cast<double>(max_abs_diff(logits->subspan(t * vocab, vocab), expected[t])));
        }
        if (worst >= 2e-2) std::printf("  %d query heads over %d KV heads: max |diff| %g\n", h.n_head, h.n_head_kv, worst);
        CHECK(worst < 2e-2);
    }
}

// describe_model() reads what a model is from its metadata and tensor names:
// the architecture's row, then one plan per block. It never reads weights.
TEST_CASE("describe_model plans every block from the tensors present, and names what it cannot run") {
    {
        test::TinyModelSpec spec;
        spec.arch = "qwen35moe";
        spec.n_layers = 4;
        spec.delta_net_interval = 2;  // blocks 1 and 3 are attention
        spec.attn_gate = true;
        spec.n_expert = 4;
        spec.n_expert_used = 2;
        spec.n_ff_shared = 96;
        auto file = MmapLoader::open(model_path("describe_hybrid_moe", spec));
        REQUIRE(file.has_value());
        auto described = describe_model(**file);
        REQUIRE(described.has_value());
        const ModelConfig& c = *described;
        CHECK(c.spec == find_architecture("qwen35moe"));
        CHECK(c.spec->rope == RopeLayout::Halves);
        CHECK(c.n_vocab == static_cast<int32_t>(test::tiny_vocab().size()));
        REQUIRE(c.blocks.size() == 4);
        CHECK(c.n_attn_layers == 2);
        for (size_t l = 0; l < c.blocks.size(); ++l) {
            const BlockPlan& b = c.blocks[l];
            const bool attention = l % 2 == 1;
            CHECK(b.mixer == (attention ? MixerKind::Attention : MixerKind::DeltaNet));
            CHECK(b.attn_gate == attention);
            CHECK(b.ffn == FfnKind::Moe);
            CHECK(b.shared_expert);
        }
        CHECK(c.hybrid());
    }
    {
        auto file = MmapLoader::open(f32_model());
        REQUIRE(file.has_value());
        auto described = describe_model(**file);
        REQUIRE(described.has_value());
        CHECK(described->spec->rope == RopeLayout::Adjacent);
        for (const BlockPlan& b : described->blocks) {
            CHECK(b.mixer == MixerKind::Attention);
            CHECK(!b.attn_gate);
            CHECK(b.ffn == FfnKind::Dense);
        }
        CHECK(!described->hybrid());
    }
    {
        test::TinyModelSpec spec;
        spec.arch = "unsupported-test-architecture";
        auto file = MmapLoader::open(model_path("describe_unknown", spec));
        REQUIRE(file.has_value());
        auto described = describe_model(**file);
        REQUIRE(!described.has_value());
        CHECK(described.status().code() == ErrorCode::Unsupported);
        // The message names the architecture and every supported one.
        const std::string message = described.status().to_string();
        CHECK(message.find("'unsupported-test-architecture'") != std::string::npos);
        for (const ArchSpec& a : architectures()) CHECK(message.find(std::string(a.name)) != std::string::npos);
    }
}

TEST_CASE("describe_model rejects invalid modern scales, rotation patterns and norm placement") {
    {
        test::TinyModelSpec s;
        s.arch = "exaone4";
        s.post_only_norms = true;
        s.sliding_window = 8;
        s.swa_layers = {false, false}; // a file may contain only global blocks beside its window setting
        auto file = MmapLoader::open(model_path("modern_all_global_nope", s));
        REQUIRE(file.has_value());
        auto described = describe_model(**file);
        REQUIRE(described.has_value());
        for (const BlockPlan& b : described->blocks) CHECK(!b.rope && !b.windowed);
    }
    for (int variant = 0; variant < 6; ++variant) {
        test::TinyModelSpec s;
        if (variant == 0) s.rope_layers = {true};  // fewer flags than blocks
        if (variant == 1) s.embedding_scale = std::numeric_limits<float>::infinity();
        if (variant == 2) s.residual_scale = -1.0f;
        if (variant == 3) { s.arch = "granite"; s.logit_scale = std::numeric_limits<float>::denorm_min(); }
        if (variant == 4) { s.attn_temp_scale = 0.1f; s.attn_temp_length = -1; }
        if (variant == 5) s.post_only_norms = true; // llama has no post-norm contract
        auto file = MmapLoader::open(model_path("invalid_modern_" + std::to_string(variant), s));
        REQUIRE(file.has_value());
        auto described = describe_model(**file);
        CHECK(!described.has_value());
        if (!described) CHECK(described.status().code() == ErrorCode::InvalidModel);
    }
}

TEST_CASE("describe_model: windowed blocks from the architecture or the file, fused projections, LongRoPE") {
    auto describe = [](const std::string& name, const test::TinyModelSpec& spec) -> ModelConfig {
        auto file = MmapLoader::open(model_path(name, spec));
        auto described = file ? describe_model(**file) : Result<ModelConfig>(file.status());
        CHECK(described.has_value());
        if (!described) {
            std::printf("  %s: %s\n", name.c_str(), described.status().to_string().c_str());
            return {};
        }
        return std::move(described).value();
    };
    using Flags = std::vector<bool>;
    auto windowed = [](const ModelConfig& c) {
        std::vector<bool> out;
        for (const BlockPlan& b : c.blocks) out.push_back(b.windowed);
        return out;
    };
    test::TinyModelSpec gemma;
    gemma.arch = "gemma3";
    gemma.n_layers = 12;
    gemma.sliding_window = 8;
    {
        // The architecture's period: every sixth block attends globally.
        const ModelConfig c = describe("describe_gemma", gemma);
        const Flags every_sixth_global = {true, true, true, true, true, false, true, true, true, true, true, false};
        CHECK(windowed(c) == every_sixth_global);
        CHECK(c.sliding_window == 8);
        CHECK(c.rope_base_swa == 10000.0f);
        CHECK(c.embd_scale == static_cast<float>(std::sqrt(128.0)));
        CHECK(c.spec->ffn_activation == FfnActivation::GeluTanh);
    }
    {
        test::TinyModelSpec s = gemma;  // the file's period
        s.swa_pattern = 3;
        const Flags every_third_global = {true, true, false, true, true, false, true, true, false, true, true, false};
        CHECK(windowed(describe("describe_gemma_period", s)) == every_third_global);
        s.swa_layers = {false, true, false, true, false, true, false, true, false, true, false, true};  // per block
        CHECK(windowed(describe("describe_gemma_flags", s)) == s.swa_layers);
    }
    {
        test::TinyModelSpec s = gemma;  // no window in the file: every block global
        s.sliding_window = 0;
        const ModelConfig c = describe("describe_gemma_global", s);
        CHECK(windowed(c) == std::vector<bool>(12, false));
        CHECK(c.rope_base_swa == 0.0f);
    }
    {
        test::TinyModelSpec s;  // an architecture without windowed blocks ignores the key
        s.sliding_window = 8;
        CHECK(windowed(describe("describe_llama_window", s)) == std::vector<bool>(2, false));
    }
    {
        test::TinyModelSpec s;  // Gemma 2: the even blocks windowed, one RoPE base, the file's soft caps and scale
        s.arch = "gemma2";
        s.n_layers = 4;
        s.sliding_window = 4096;
        s.attn_softcap = 50.0f;
        s.final_softcap = 30.0f;
        const ModelConfig c = describe("describe_gemma2", s);
        CHECK(windowed(c) == (Flags{true, false, true, false}));
        CHECK(c.sliding_window == 4096);
        CHECK(c.rope_base_swa == 0.0f);
        CHECK(c.attn_softcap == 50.0f);
        CHECK(c.final_softcap == 30.0f);
        CHECK(c.attn_scale == 1.0f / std::sqrt(32.0f));  // 1/sqrt(head_dim) when the file states none
        s.attn_scale = 1.0f / 12.0f;
        CHECK(describe("describe_gemma2_scale", s).attn_scale == 1.0f / 12.0f);
    }
    {
        test::TinyModelSpec s;
        s.arch = "phi3";
        s.fused_qkv = true;
        s.fused_gate_up = true;
        s.longrope_ctx = 256;
        s.sliding_window = 32;
        const ModelConfig c = describe("describe_phi3", s);
        for (const BlockPlan& b : c.blocks) CHECK(b.fused_qkv && b.fused_gate_up && b.windowed);
        CHECK(c.rope_ctx_orig == 256);
        CHECK(c.rope_magnitude == 1.19f);
        CHECK(c.rope_base_swa == 0.0f);  // windowed blocks rotate like the others
    }
    {
        test::TinyModelSpec s;  // DeepSeek: latent attention, a dense leading block, then MoE with a shared expert
        s.arch = "deepseek2";
        s.n_layers = 3;
        s.n_head_kv = 1;
        s.rope_dim = 32;
        s.mla_kv_rank = 64;
        s.mla_q_rank = 64;
        s.n_expert = 4;
        s.n_expert_used = 2;
        s.n_ff_shared = 64;
        s.dense_lead = 1;
        s.expert_weights_norm = -1;
        const ModelConfig c = describe("describe_deepseek2", s);
        CHECK(c.spec->rope == RopeLayout::Adjacent);
        CHECK(c.mla.kv_rank == 64 && c.mla.q_rank == 64 && c.mla.nope_dim == 32 && c.mla.v_dim == 32);
        CHECK(c.head_dim == 96 && c.value_dim == 64 && c.n_head_kv == 1);
        CHECK(!c.moe_norm_weights);  // DeepSeek-V2 does not renormalize unless the file says so
        REQUIRE(c.blocks.size() == 3);
        for (size_t l = 0; l < 3; ++l) {
            CHECK(c.blocks[l].latent);
            CHECK(c.blocks[l].ffn == (l == 0 ? FfnKind::Dense : FfnKind::Moe));
            CHECK(c.blocks[l].shared_expert == (l > 0));
        }
        s.mla_legacy = true;  // attn_kv_b era: normalized to the same cached-latent geometry
        auto file = MmapLoader::open(model_path("describe_deepseek2_legacy", s));
        REQUIRE(file.has_value());
        auto legacy = describe_model(**file);
        REQUIRE(legacy.has_value());
        CHECK(legacy->head_dim == c.head_dim && legacy->value_dim == c.value_dim && legacy->n_head_kv == 1);
        CHECK(legacy->mla.nope_dim == c.mla.nope_dim && legacy->mla.v_dim == c.mla.v_dim);
        for (const BlockPlan& b : legacy->blocks) CHECK(b.latent && b.latent_combined);
    }
    {
        test::TinyModelSpec s;  // YaRN with the reference's defaults
        s.arch = "qwen3";
        s.yarn_factor = 4.0f;
        s.yarn_ctx = 32;
        const ModelConfig c = describe("describe_yarn", s);
        CHECK(c.yarn.factor == 4.0f);
        CHECK(c.yarn.ctx_orig == 32);
        CHECK(c.yarn.beta_fast == 32.0f);
        CHECK(c.yarn.beta_slow == 1.0f);
        CHECK(c.yarn.ext_factor == 1.0f);
        CHECK(c.yarn.truncate);
        CHECK(c.rope_magnitude == static_cast<float>(0.1 * std::log(4.0) + 1.0));  // the attention factor
        CHECK(c.attn_scale == 1.0f / std::sqrt(32.0f));
        s.yarn_ctx = 0;  // the original context defaults to the training context
        s.yarn_log_mul = 0.0707f;
        const ModelConfig d = describe("describe_yarn_logmul", s);
        CHECK(d.yarn.ctx_orig == s.context_length);
        CHECK(d.rope_magnitude == 1.0f);  // DeepSeek's convention: the factor scales q·k
        const double m = 1.0 + static_cast<double>(0.0707f) * std::log(4.0);
        CHECK(d.attn_scale == static_cast<float>(1.0f / std::sqrt(32.0f) * m * m));
    }
}

TEST_CASE("describe_model refuses a gated xIELU variant instead of choosing another activation") {
    auto spec = test::classic_model_spec("apertus");
    spec.ungated = false;
    auto file = MmapLoader::open(model_path("gated_xielu", spec));
    REQUIRE(file.has_value());
    auto described = describe_model(**file);
    REQUIRE(!described.has_value());
    CHECK(described.status().code() == ErrorCode::Unsupported);
    CHECK(described.status().message().find("gated xIELU FFN in block 0") != std::string::npos);
}

TEST_CASE("describe_model refuses learned positions with a scale-only or bias-only embedding norm") {
    for (const bool scale : {false, true}) {
        test::GgufWriter w;
        w.add_string("general.architecture", "gpt2");
        w.add_u32("gpt2.block_count", 1);
        w.add_u32("gpt2.embedding_length", 32);
        w.add_u32("gpt2.feed_forward_length", 32);
        w.add_u32("gpt2.attention.head_count", 1);
        w.add_tensor("token_embd.weight", {32, 8}, DType::F32, std::vector<float>(256));
        w.add_tensor("position_embd.weight", {32, 8}, DType::F32, std::vector<float>(256));
        w.add_tensor(scale ? "token_embd_norm.weight" : "token_embd_norm.bias", {32}, DType::F32,
                     std::vector<float>(32, 1.0f));
        const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) +
                                 (scale ? "_position_norm_scale.gguf" : "_position_norm_bias.gguf");
        w.write(path);
        auto file = MmapLoader::open(path);
        REQUIRE(file.has_value());
        auto described = describe_model(**file);
        REQUIRE(!described.has_value());
        CHECK(described.status().code() == ErrorCode::Unsupported);
        CHECK(described.status().message().find("embedding norm with learned position embeddings") != std::string::npos);
        std::remove(path.c_str());
    }
}

// ---------------------------------------------------------------------------
// Key scaling for quantized KV caches (core/key_scaling.h)
// ---------------------------------------------------------------------------
TEST_CASE("Key ranges: outlier channels get large factors, ordinary ones stay near 1, all bounded; even keys get none") {
    KeyScalingShape shape;
    shape.blocks = 1;
    shape.n_head = 2;
    shape.n_head_kv = 1;
    shape.head_dim = 32;
    shape.rope_dim = 32;
    shape.rope = RopeLayout::Halves;
    shape.positions = 1000;
    shape.pair_speed.assign(16, 0.0f);  // no pair turns: each channel keeps its own range
    shape.pair_speed[2] = 1.0f;         // pair 2 = channels (2, 18) turns fully over the context
    KeyRanges ranges(shape);
    std::mt19937 rng(5);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int t = 0; t < 20; ++t) {
        std::vector<float> q(64), k(32);
        for (float& x : q) x = normal(rng);
        for (float& x : k) x = normal(rng);
        k[5] = 100.0f;   // a large constant key channel
        k[2] = 50.0f;    // another, in a pair that turns: channel 18 reaches it too
        k[9] = 1e7f;     // beyond the clamp
        q[7] = q[32 + 7] = 0.0f;  // a query channel never reached: floored, not infinite
        ranges.add(0, q.data(), k.data());
    }
    const std::vector<float> t = ranges.factors();
    REQUIRE(t.size() == 32);
    CHECK(t[5] > 5.0f);
    CHECK(t[2] > 3.0f && std::fabs(t[18] - t[2]) < 0.5f);
    CHECK(t[9] == kMaxKeyScale);
    CHECK(std::isfinite(t[7]) && t[7] <= 4.0f);
    for (const int i : {0, 1, 3, 10, 20, 31}) CHECK(t[i] > 0.4f && t[i] < 2.5f);
    for (const float f : t) CHECK(f >= 1.0f / kMaxKeyScale && f <= kMaxKeyScale);
    CHECK(ranges.outlier_ratio() >= kOutlierRatio);

    // Keys without such channels: nothing to fix, every factor stays 1.
    KeyRanges even(shape);
    for (int t2 = 0; t2 < 20; ++t2) {
        std::vector<float> q(64), k(32);
        for (float& x : q) x = normal(rng);
        for (float& x : k) x = normal(rng);
        even.add(0, q.data(), k.data());
    }
    CHECK(even.outlier_ratio() < kOutlierRatio);
    const std::vector<float> ones = even.factors();
    CHECK(std::all_of(ones.begin(), ones.end(), [](float f) { return f == 1.0f; }));
}

namespace {

// Logits of every position of a fixed sequence (Logits::All) through `model`.
std::vector<float> all_logits(Transformer& model, ThreadPool& pool) {
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens = {1};
    for (int32_t i = 1; i < 40; ++i) tokens.push_back(3 + (i * 53) % (model.config().n_vocab - 3));
    model.reset();
    auto out = model.forward(tokens, Transformer::Logits::All, route, pool);
    return out ? std::vector<float>(out->begin(), out->end()) : std::vector<float>();
}

std::vector<int32_t> calibration_tokens(const Transformer& model) {
    std::vector<int32_t> tokens = {1};
    for (int32_t i = 1; i < 48; ++i) tokens.push_back(3 + (i * 29) % (model.config().n_vocab - 3));
    return tokens;
}

test::TinyModelSpec outlier_spec() {
    test::TinyModelSpec spec;
    spec.arch = "qwen2";
    spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
    spec.key_outlier = 60.0f;
    return spec;
}

}  // namespace

namespace {

test::TinyModelSpec latent_spec(int32_t q_rank, bool combined = false) {
    test::TinyModelSpec spec;
    spec.arch = "deepseek2";
    spec.n_head_kv = 1;
    spec.rope_dim = 32;    // the shared RoPE key
    spec.mla_kv_rank = 64;  // cached rows [k_pe | latent] of 96 values, values from channel 32
    spec.mla_q_rank = q_rank;
    spec.mla_legacy = combined;
    return spec;
}

}  // namespace

TEST_CASE("Per-block latent head counts are refused before the uniform latent path can run") {
    test::TinyModelSpec spec = latent_spec(64);
    spec.block_heads = {2, 4};
    auto file = MmapLoader::open(model_path("latent_varying_heads", spec));
    REQUIRE(file.has_value());
    auto config = describe_model(**file);
    REQUIRE(!config.has_value());
    CHECK(config.status().code() == ErrorCode::Unsupported);
    CHECK(config.status().message().find("per-block latent attention head counts") != std::string::npos);
}

// Latent attention attends with absorbed queries over the cached [k_pe |
// latent] rows; the model was trained with per-head keys and values
// expanded from the latent. Both forms must give the same logits, the cache
// must hold one row per position and block (half a K + V cache of that
// width), and batched passes must give token-by-token passes' bits.
TEST_CASE("Latent attention over cached latents matches an explicit per-head reference") {
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 33, 120, 7, 260};
    for (const int32_t q_rank : {0, 64}) {  // one attn_q, or the low-rank attn_q_a / attn_q_b pair
        test::TinyModelSpec spec = latent_spec(q_rank);
        spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
        TransformerOptions options;
        options.kv_type = KvCacheType::F16;
        options.latent_form = LatentForm::Absorbed;  // the expanded form: the next case
        auto model = load_transformer(model_path("latent_f32_q" + std::to_string(q_rank), spec), options);
        REQUIRE(model != nullptr);
        const ModelConfig& c = model->config();
        CHECK(c.blocks[0].latent);
        CHECK(c.head_dim == 96);
        CHECK(c.value_dim == 64);
        CHECK(c.attn_scale == 1.0f / std::sqrt(64.0f));  // the trained per-head width: nope 32 + rope 32
        const KvCacheConfig& kv = model->kv_cache().config();
        CHECK(kv.values_in_keys && kv.n_head_kv == 1 && kv.value_offset == 32);
        CHECK(KvCache::bytes_per_token(kv) == static_cast<size_t>(c.n_layers) * 96 * 2);  // F16, keys only
        auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(logits.has_value());
        const std::vector<float> batched(logits->begin(), logits->end());
        const auto expected = LatentReference(model->file()).logits(tokens);
        const auto vocab = static_cast<size_t>(c.n_vocab);
        double worst = 0.0;
        for (size_t t = 0; t < tokens.size(); ++t) {
            const auto row = std::span<const float>(batched).subspan(t * vocab, vocab);
            worst = std::max(worst, max_abs_diff(row, expected[t]));
            CHECK(Sampler::argmax(row) == Sampler::argmax(expected[t]));
        }
        std::printf("  latent attention (q_rank %d) vs explicit reference: max |diff| %g\n", q_rank, worst);
        CHECK(worst < 2e-2);
        // Token by token: the same bits as the batched pass.
        model->reset();
        std::vector<float> stepwise;
        for (const int32_t t : tokens) {
            auto r = model->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool);
            REQUIRE(r.has_value());
            stepwise.insert(stepwise.end(), r->begin(), r->end());
        }
        CHECK(stepwise == batched);
    }
}

TEST_CASE("Latent attention applies its output bias in both forms and on every KV type") {
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 33, 120, 7, 260};
    for (const int32_t q_rank : {0, 64}) {
        auto spec = latent_spec(q_rank);
        spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
        const auto plain_path = model_path("latent_bias_plain_q" + std::to_string(q_rank), spec);
        spec.attn_output_bias = true;
        const auto path = model_path("latent_bias_q" + std::to_string(q_rank), spec);
        for (const auto form : {LatentForm::Absorbed, LatentForm::Expanded}) {
            for (const auto kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
                TransformerOptions options;
                options.kv_type = kv;
                options.latent_form = form;
                auto model = load_transformer(path, options), plain = load_transformer(plain_path, options);
                REQUIRE(model && plain);
                auto got = model->forward(tokens, Transformer::Logits::All, route, pool);
                auto without_bias = plain->forward(tokens, Transformer::Logits::All, route, pool);
                REQUIRE(got && without_bias);
                const std::vector<float> saved(got->begin(), got->end());
                CHECK(max_abs_diff(*without_bias, saved) > 1e-3);
                const auto vocab = static_cast<size_t>(model->config().n_vocab);
                if (kv == KvCacheType::F16) {
                    const auto expected = LatentReference(model->file()).logits(tokens);
                    for (size_t t = 0; t < tokens.size(); ++t) {
                        CHECK(max_abs_diff(std::span(saved).subspan(t * vocab, vocab), expected[t]) < 2e-2);
                    }
                }
                REQUIRE(model->truncate(7).is_ok());
                auto replay = model->forward(std::span(tokens).subspan(7), Transformer::Logits::All, route, pool);
                REQUIRE(replay.has_value());
                CHECK(std::memcmp(replay->data(), saved.data() + 7 * vocab, replay->size_bytes()) == 0);
                if (form == LatentForm::Absorbed) {
                    model->reset();
                    for (size_t t = 0; t < tokens.size(); ++t) {
                        auto step = model->forward(std::span(tokens).subspan(t, 1), Transformer::Logits::All, route, pool);
                        REQUIRE(step.has_value());
                        CHECK(std::memcmp(step->data(), saved.data() + t * vocab, step->size_bytes()) == 0);
                    }
                }
            }
        }
    }
}

namespace {

// Logits of every token of `tokens`, prefilled in passes of the given sizes
// (cycled), rows concatenated.
std::vector<float> latent_pass_logits(Transformer& model, const std::vector<int32_t>& tokens,
                                      const std::vector<size_t>& sizes, ThreadPool& pool) {
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    model.reset();
    std::vector<float> out;
    for (size_t at = 0, i = 0; at < tokens.size(); ++i) {
        const size_t n = std::min(sizes[i % sizes.size()], tokens.size() - at);
        auto r = model.forward(std::span<const int32_t>(tokens).subspan(at, n), Transformer::Logits::All, route, pool);
        if (!r) return {};
        out.insert(out.end(), r->begin(), r->end());
        at += n;
    }
    return out;
}

// Rows (n_vocab each) of `a` whose argmax differs from `b`'s.
size_t argmax_differences(const std::vector<float>& a, const std::vector<float>& b, size_t vocab) {
    size_t diff = 0;
    for (size_t r = 0; r * vocab < a.size(); ++r) {
        diff += Sampler::argmax(std::span<const float>(a).subspan(r * vocab, vocab)) !=
                Sampler::argmax(std::span<const float>(b).subspan(r * vocab, vocab));
    }
    return diff;
}

}  // namespace

TEST_CASE("Combined latent expansion normalizes to split weights in every KV format") {
    ThreadPool pool(4);
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 33, 120, 7, 260};
    for (const int32_t q_rank : {0, 64}) {
        auto spec = latent_spec(q_rank);
        spec.attn_type = DType::F32;
        const auto split_path = model_path("latent_split_layout_q" + std::to_string(q_rank), spec);
        spec.mla_legacy = true;
        const auto legacy_path = model_path("latent_combined_layout_q" + std::to_string(q_rank), spec);
        for (const auto kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            for (const auto form : {LatentForm::Absorbed, LatentForm::Expanded}) {
                TransformerOptions options;
                options.kv_type = kv;
                options.latent_form = form;
                auto split = load_transformer(split_path, options);
                auto legacy = load_transformer(legacy_path, options);
                REQUIRE(split && legacy);
                const auto expected = latent_pass_logits(*split, tokens, {tokens.size()}, pool);
                REQUIRE(!expected.empty());
                CHECK(latent_pass_logits(*legacy, tokens, {tokens.size()}, pool) == expected);
                // Single-token passes use absorption even when expanded
                // prefill is pinned. Compare the same execution schedule.
                const auto chunked = latent_pass_logits(*split, tokens, {3, 1, 5}, pool);
                CHECK(latent_pass_logits(*legacy, tokens, {3, 1, 5}, pool) == chunked);
                if (q_rank == 0 && kv == KvCacheType::F16) {
                    std::vector<TripleBufferLoader::Item> items;
                    for (int32_t l = 0; l < legacy->config().n_layers; ++l) {
                        const auto [begin, end] = legacy->file().layer_range(l);
                        REQUIRE(end > begin);
                        items.push_back({begin, end - begin});
                    }
                    const auto& mapped = legacy->file().file();
                    auto source = TripleBufferLoader::create(std::move(items),
                        [&](uint64_t offset, size_t length, uint8_t* dst) {
                            std::memset(dst, 0, length);
                            if (offset < mapped.size()) {
                                std::memcpy(dst, mapped.data() + offset,
                                            std::min(length, mapped.size() - static_cast<size_t>(offset)));
                            }
                            return Status::ok();
                        });
                    REQUIRE(source.has_value());
                    legacy->set_layer_source(source->get());
                    CHECK(latent_pass_logits(*legacy, tokens, {3, 1, 5}, pool) == chunked);
                    legacy->set_layer_source(nullptr);
                }
            }
        }
    }
}

TEST_CASE("Combined quantized latent expansion matches an independent per-head reference") {
    ThreadPool pool(4);
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    for (const auto type : {DType::F32, DType::F16, DType::BF16, DType::Q8_0, DType::Q4_0, DType::Q4_1,
                            DType::Q4_K, DType::Q5_K}) {
        auto spec = latent_spec(0);
        spec.n_embd = spec.n_ff = spec.mla_kv_rank = 256;
        spec.mla_v = 64;
        spec.mla_legacy = true;
        spec.attn_type = type;
        spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
        spec.weight_scale = 0.02f;
        const auto path = model_path("latent_combined_quant_" + std::to_string(static_cast<int>(type)), spec);
        auto reference = MmapLoader::open(path);
        REQUIRE(reference.has_value());
        const auto expected = LatentReference(**reference).logits(tokens);
        for (const auto form : {LatentForm::Absorbed, LatentForm::Expanded}) {
            TransformerOptions options;
            options.kv_type = KvCacheType::F16;
            options.latent_form = form;
            options.repack_cpu = quant::repack_kernels_available();
            auto model = load_transformer(path, options);
            REQUIRE(model);
            CHECK(model->file().tensor("blk.0.attn_kv_b.weight")->type == type);
            const auto actual = latent_pass_logits(*model, tokens, {tokens.size()}, pool);
            REQUIRE(actual.size() == tokens.size() * static_cast<size_t>(model->config().n_vocab));
            double worst = 0.0;
            for (size_t t = 0; t < tokens.size(); ++t) {
                worst = std::max(worst, max_abs_diff(std::span<const float>(actual).subspan(
                    t * expected[t].size(), expected[t].size()), expected[t]));
            }
            std::printf("  combined latent %s, form %d: max |diff| %g\n",
                        std::string(dtype_traits(type).name).c_str(), static_cast<int>(form), worst);
            CHECK(worst < 3e-2);
        }
    }
}

TEST_CASE("Combined latent weight views preserve every GGML format's decoded keys and raw values") {
    std::vector<QuantVector> vectors;
    REQUIRE(load_quant_vectors(vectors));
    for (const DType type : {DType::F32, DType::F16}) {
        QuantVector v;
        v.type = type;
        v.n = 256;
        std::vector<float> values(256);
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(static_cast<float>(i));
        v.blocks.resize(dtype_row_bytes(type, v.n));
        quant::quantize_row(type, values.data(), v.blocks.data(), v.n);
        vectors.push_back(std::move(v));
    }
    LatentAttention geometry;
    geometry.kv_rank = 256;
    geometry.nope_dim = 3;
    geometry.v_dim = 5;
    constexpr int32_t heads = 2;
    const int64_t rows = heads * (geometry.nope_dim + geometry.v_dim);
    size_t checked = 0;
    for (const auto& qv : vectors) {
        const auto traits = dtype_traits(qv.type);
        const size_t row_bytes = dtype_row_bytes(qv.type, geometry.kv_rank);
        std::vector<uint8_t> raw(static_cast<size_t>(rows) * row_bytes);
        std::vector<float> decoded(static_cast<size_t>(rows) * geometry.kv_rank);
        const size_t blocks = qv.blocks.size() / static_cast<size_t>(traits.block_bytes);
        for (size_t row = 0; row < static_cast<size_t>(rows); ++row) {
            for (size_t b = 0; b < static_cast<size_t>(geometry.kv_rank / traits.block_size); ++b) {
                const size_t source_block = (row + b) % blocks;
                std::memcpy(raw.data() + row * row_bytes + b * static_cast<size_t>(traits.block_bytes),
                            qv.blocks.data() + source_block * static_cast<size_t>(traits.block_bytes),
                            static_cast<size_t>(traits.block_bytes));
            }
            // Preserve this engine's decoded bits, including IQ1's existing
            // rounding convention and signed zero, through the transpose.
            quant::dequantize_row(qv.type, raw.data() + row * row_bytes,
                                 decoded.data() + row * geometry.kv_rank, geometry.kv_rank);
        }
        TensorView source;
        source.name = "blk.0.attn_kv_b.weight";
        source.type = qv.type;
        source.n_dims = 2;
        source.ne = {geometry.kv_rank, rows, 1, 1};
        source.data = raw.data();
        source.nbytes = raw.size();
        CombinedLatentWeights split;
        REQUIRE(split.bind(source, geometry, heads).is_ok());
        const auto* key = reinterpret_cast<const float*>(split.key().data);
        for (int32_t h = 0; h < heads; ++h) {
            for (int32_t j = 0; j < geometry.nope_dim; ++j) {
                for (int32_t r = 0; r < geometry.kv_rank; ++r) {
                    const size_t src = static_cast<size_t>((h * (geometry.nope_dim + geometry.v_dim) + j) * geometry.kv_rank + r);
                    const size_t dst = static_cast<size_t>((h * geometry.kv_rank + r) * geometry.nope_dim + j);
                    CHECK(std::memcmp(key + dst, decoded.data() + src, sizeof(float)) == 0);
                }
            }
            CHECK(std::memcmp(split.value().data + static_cast<size_t>(h * geometry.v_dim) * row_bytes,
                              source.row(h * (geometry.nope_dim + geometry.v_dim) + geometry.nope_dim),
                              static_cast<size_t>(geometry.v_dim) * row_bytes) == 0);
        }
        source.ne[0] -= 1;
        CHECK(split.bind(source, geometry, heads).code() == ErrorCode::InvalidModel);
        source.ne[0] += 1;
        source.type = DType::Q8_0_R4;
        CHECK(split.bind(source, geometry, heads).code() == ErrorCode::Unsupported);
        ++checked;
    }
    CHECK(checked == conformance::weight_formats().size());
}

// The expanded form (per-head keys and values expanded from the cached
// latents, for prefills) computes the absorbed form's function, rounded
// differently: both must match the explicit reference, the expanded form must
// give the same bits however a prompt is split into passes and however many
// threads run, and decoding after an expanded prefill continues absorbed over
// the same cache of latents.
TEST_CASE("Latent attention in the expanded form matches the absorbed form and the explicit reference") {
    ThreadPool pool(4);
    ThreadPool one(1);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 33, 120, 7, 260};
    for (const int32_t q_rank : {0, 64}) {
        test::TinyModelSpec spec = latent_spec(q_rank);
        spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
        const std::string& path = model_path("latent_f32_q" + std::to_string(q_rank), spec);
        TransformerOptions options;
        options.kv_type = KvCacheType::F16;
        options.latent_form = LatentForm::Expanded;
        auto expanded = load_transformer(path, options);
        options.latent_form = LatentForm::Absorbed;
        auto absorbed = load_transformer(path, options);
        REQUIRE(expanded != nullptr && absorbed != nullptr);
        const auto vocab = static_cast<size_t>(expanded->config().n_vocab);
        const std::vector<float> batched = latent_pass_logits(*expanded, tokens, {tokens.size()}, pool);
        REQUIRE(batched.size() == tokens.size() * vocab);
        CHECK(expanded->latent_stats().expanded_passes == 1);
        CHECK(expanded->latent_stats().absorbed_passes == 0);
        const auto expected = LatentReference(expanded->file()).logits(tokens);
        double worst = 0.0;
        for (size_t t = 0; t < tokens.size(); ++t) {
            const auto row = std::span<const float>(batched).subspan(t * vocab, vocab);
            worst = std::max(worst, max_abs_diff(row, expected[t]));
            CHECK(Sampler::argmax(row) == Sampler::argmax(expected[t]));
        }
        const std::vector<float> plain = latent_pass_logits(*absorbed, tokens, {tokens.size()}, pool);
        const double forms = max_abs_diff(batched, plain);
        std::printf("  expanded latent attention (q_rank %d): max |diff| %g vs explicit reference, %g vs absorbed\n",
                    q_rank, worst, forms);
        CHECK(worst < 2e-2);
        CHECK(forms < 1e-3);
        // Any split into passes of two tokens or more, any thread count: the same bits.
        for (const std::vector<size_t>& sizes : std::vector<std::vector<size_t>>{{5, 7}, {2, 4, 6}, {3}, {10, 2}}) {
            CHECK(latent_pass_logits(*expanded, tokens, sizes, pool) == batched);
        }
        CHECK(latent_pass_logits(*expanded, tokens, {tokens.size()}, one) == batched);
        // An expanded prefill, then absorbed decoding steps over its latents:
        // the absorbed model's steps within float rounding (they read the same
        // kind of cache, filled from slightly different hidden states).
        expanded->reset();
        absorbed->reset();
        const auto prompt = std::span<const int32_t>(tokens).first(8);
        REQUIRE(expanded->forward(prompt, Transformer::Logits::None, route, pool).has_value());
        REQUIRE(absorbed->forward(prompt, Transformer::Logits::None, route, pool).has_value());
        for (size_t t = 8; t < tokens.size(); ++t) {
            const auto step = std::span<const int32_t>(tokens).subspan(t, 1);
            auto a = expanded->forward(step, Transformer::Logits::Last, route, pool);
            REQUIRE(a.has_value());
            const std::vector<float> got(a->begin(), a->end());
            auto b = absorbed->forward(step, Transformer::Logits::Last, route, pool);
            REQUIRE(b.has_value());
            CHECK(max_abs_diff(*b, got) < 1e-3);
            CHECK(Sampler::argmax(got) == Sampler::argmax(expected[t]));
        }
        CHECK(expanded->latent_stats().absorbed_passes == static_cast<int64_t>(tokens.size() - 8));
    }
}

// Long prompts span several tiles of expanded rows (latent_tile() positions
// each, at absolute bounds): still the same bits for every split into passes,
// with an F16 cache, a Q8_0 cache with per-channel key scales (multiplied back
// before the expansion; calibration itself runs absorbed), and a sliding window
// with attention sinks, where tokens see different positions; and close to the
// absorbed form throughout.
TEST_CASE("Sparse latent attention: dense identity, causal batching and saved contexts in every KV format") {
    ThreadPool pool(2);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t t = 0; t < 33; ++t) tokens.push_back(3 + (37 * t) % 250);
    for (const KvCacheType type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        auto dense = latent_spec(64);
        dense.arch = "glm-dsa";
        auto indexed = dense;
        indexed.index_dim = 64;
        indexed.index_topk = 128;
        indexed.index_full.assign(static_cast<size_t>(indexed.n_layers), false);
        indexed.index_full[0] = true;
        const std::string plain_path = model_path("dsa_dense", dense);
        const std::string all_path = model_path("dsa_all", indexed);
        indexed.index_topk = 4;
        const std::string sparse_path = model_path("dsa_sparse", indexed);
        for (const LatentForm form : {LatentForm::Absorbed, LatentForm::Expanded}) {
            TransformerOptions options;
            options.kv_type = type;
            options.latent_form = form;
            auto plain = load_transformer(plain_path, options);
            auto all = load_transformer(all_path, options);
            auto sparse = load_transformer(sparse_path, options);
            REQUIRE(plain != nullptr && all != nullptr && sparse != nullptr);
            CHECK(sparse->config().blocks[0].indexed);
            CHECK(sparse->kv_cache().config().auxiliary_dim == 64);
            auto a = plain->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(a.has_value());
            const std::vector<float> expected(a->begin(), a->end());
            auto b = all->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(b.has_value());
            CHECK(max_abs_diff(*b, expected) == 0.0f);  // index_topk >= context is dense, bit for bit
            auto c = sparse->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(c.has_value());
            const std::vector<float> batched(c->begin(), c->end());
            CHECK(max_abs_diff(*c, expected) > 0.0f);  // the sparse selection actually runs
            sparse->reset();
            if (form == LatentForm::Absorbed) {
                std::vector<float> decoded;
                for (const int32_t token : tokens) {
                    auto row = sparse->forward({&token, 1}, Transformer::Logits::Last, route, pool);
                    REQUIRE(row.has_value());
                    decoded.insert(decoded.end(), row->begin(), row->end());
                }
                CHECK(max_abs_diff(decoded, batched) == 0.0f);
            } else {
                REQUIRE(sparse->forward(tokens, Transformer::Logits::None, route, pool).has_value());
            }
            std::stringstream image(std::ios::in | std::ios::out | std::ios::binary);
            REQUIRE(sparse->write_state(image).is_ok());
            auto restored = load_transformer(sparse_path, options);
            REQUIRE(restored != nullptr);
            REQUIRE(restored->read_state(image).is_ok());
            for (const int32_t token : {71, 5, 250}) {
                auto left = sparse->forward({&token, 1}, Transformer::Logits::Last, route, pool);
                auto right = restored->forward({&token, 1}, Transformer::Logits::Last, route, pool);
                REQUIRE(left.has_value() && right.has_value());
                CHECK(left->size() == right->size());
                CHECK(std::memcmp(left->data(), right->data(), left->size_bytes()) == 0);
            }
            REQUIRE(sparse->truncate(static_cast<int32_t>(tokens.size())).is_ok());
            REQUIRE(restored->truncate(static_cast<int32_t>(tokens.size())).is_ok());
            const int32_t token = 42;
            auto left = sparse->forward({&token, 1}, Transformer::Logits::Last, route, pool);
            auto right = restored->forward({&token, 1}, Transformer::Logits::Last, route, pool);
            REQUIRE(left.has_value() && right.has_value());
            CHECK(std::memcmp(left->data(), right->data(), left->size_bytes()) == 0);
        }
    }
}

TEST_CASE("Expanded latent attention over many tiles: every split the same bits, close to the absorbed form") {
    test::TinyModelSpec spec = latent_spec(64);
    spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
    spec.k_norm_outlier = 60.0f;  // large latent channels: key scaling has something to do
    const std::string& path = model_path("latent_outlier", spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    struct Setup {
        const char* name;
        KvCacheType kv;
        bool scale_keys;
        int32_t window;
    };
    for (const Setup& setup : {Setup{"f16", KvCacheType::F16, false, 0}, Setup{"q8_0 scaled", KvCacheType::Q8_0, true, 0},
                               Setup{"q8_0 window", KvCacheType::Q8_0, false, 150}}) {
        TransformerOptions options;
        options.kv_type = setup.kv;
        options.context_length = 1024;
        options.max_batch = 640;
        options.sliding_window = setup.window;
        options.sink_tokens = 4;
        options.latent_form = LatentForm::Expanded;
        auto expanded = load_transformer(path, options);
        options.latent_form = LatentForm::Absorbed;
        auto absorbed = load_transformer(path, options);
        REQUIRE(expanded != nullptr && absorbed != nullptr);
        if (setup.scale_keys) {
            REQUIRE(expanded->calibrate_keys(calibration_tokens(*expanded), route, pool).is_ok());
            REQUIRE(absorbed->calibrate_keys(calibration_tokens(*absorbed), route, pool).is_ok());
            REQUIRE(!expanded->key_scales().empty());
            CHECK(std::vector<float>(expanded->key_scales().begin(), expanded->key_scales().end()) ==
                  std::vector<float>(absorbed->key_scales().begin(), absorbed->key_scales().end()));
        }
        const int32_t vocab = expanded->config().n_vocab;
        std::vector<int32_t> tokens = {1};
        for (int32_t i = 1; i < 600; ++i) tokens.push_back(3 + (i * 37 + i / 7) % (vocab - 3));
        const std::vector<float> whole = latent_pass_logits(*expanded, tokens, {600}, pool);
        REQUIRE(whole.size() == tokens.size() * static_cast<size_t>(vocab));
        CHECK(expanded->latent_stats().expanded_passes >= 1);
        CHECK(latent_pass_logits(*expanded, tokens, {64}, pool) == whole);
        CHECK(latent_pass_logits(*expanded, tokens, {257, 100, 13}, pool) == whole);
        // Against the absorbed form over an F16 cache (the same window): a
        // Q8_0 cache's rounding costs the expanded form no more than the
        // absorbed one, whose Q8_0 copy of each absorbed query adds rounding
        // of its own (most visible on large latent channels left unscaled).
        options.kv_type = KvCacheType::F16;
        auto exact = load_transformer(path, options);
        REQUIRE(exact != nullptr);
        const std::vector<float> want = latent_pass_logits(*exact, tokens, {64}, pool);
        const std::vector<float> plain = latent_pass_logits(*absorbed, tokens, {64}, pool);
        const double err_expanded = max_abs_diff(whole, want);
        const double err_absorbed = max_abs_diff(plain, want);
        const size_t flips_expanded = argmax_differences(whole, want, static_cast<size_t>(vocab));
        const size_t flips_absorbed = argmax_differences(plain, want, static_cast<size_t>(vocab));
        std::printf("  over %zu positions (%s), vs absorbed over F16: max |diff| %g expanded, %g absorbed; argmax "
                    "differences %zu, %zu\n",
                    tokens.size(), setup.name, err_expanded, err_absorbed, flips_expanded, flips_absorbed);
        CHECK(err_expanded <= std::max(1e-3, 1.25 * err_absorbed));
        CHECK(flips_expanded <= std::max<size_t>(flips_absorbed, tokens.size() / 100));
    }
}

TEST_CASE("Sparse latent form calibration uses valid index keys and leaves an empty context") {
    auto spec = latent_spec(64);
    spec.arch = "glm-dsa";
    spec.index_dim = 64;
    spec.index_topk = 8;
    auto model = load_transformer(model_path("indexed_calibration", spec));
    REQUIRE(model != nullptr);
    ThreadPool pool(2);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    auto ratio = model->calibrate_latent_form(route, pool);
    REQUIRE(ratio.has_value());
    CHECK(*ratio >= LatentFormChooser::kMinRatio && *ratio <= LatentFormChooser::kMaxRatio);
    CHECK(model->latent_chooser().ratio() == *ratio);
    CHECK(model->n_past() == 0);
    const std::vector<int32_t> tokens = {1, 3, 7, 9};
    auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(logits.has_value());
    for (const float value : *logits) CHECK(std::isfinite(value));
}

// The automatic choice: work counted per form, and a threshold set by one
// calibrated ratio, nothing else.
TEST_CASE("Latent attention's form is a pure function of a pass's shape and the calibrated ratio") {
    ModelConfig c;  // DeepSeek-V2-Lite's attention: 16 heads, a 512-wide latent, keys 128 + 64, values 128
    c.n_head = 16;
    c.rope_dim = 64;
    c.mla.kv_rank = 512;
    c.mla.nope_dim = 128;
    c.mla.v_dim = 128;
    auto prefill = [&](int64_t n, int64_t past) {  // full attention: n tokens after `past` positions
        return LatentFormChooser::work(c, n, n * past + n * (n + 1) / 2, past + n);
    };
    const auto first = prefill(512, 0);
    CHECK(first.absorbed == 16.0 * (512.0 * 512 * 256 + 131328.0 * (64 + 1024)));
    CHECK(first.expanded == 16.0 * (513.0 * 512 * 256 + 131328.0 * (128 + 64 + 128)));
    LatentFormChooser chooser;
    CHECK(chooser.ratio() == 1.0f);
    // Uncalibrated: the counts decide. A long prompt's batches expand, a few
    // tokens deep into a context (speculative verification) stay absorbed.
    CHECK(chooser.expand(first));
    CHECK(chooser.expand(prefill(512, 8000)));
    CHECK(!chooser.expand(prefill(8, 8000)));
    CHECK(!chooser.expand(prefill(2, 0)));
    // Expanded multiply-adds twice as costly: a 512-token batch deep in a
    // context stays absorbed, a long prompt's first batch still expands.
    REQUIRE(chooser.set_ratio(2.0f).is_ok());
    CHECK(!chooser.expand(prefill(512, 8000)));
    CHECK(chooser.expand(prefill(2048, 0)));
    REQUIRE(chooser.set_ratio(0.5f).is_ok());
    CHECK(chooser.expand(prefill(512, 8000)));
    CHECK(!chooser.expand(prefill(8, 8000)));
    // Out of range or not a number: refused, the ratio kept.
    for (const float bad : {0.0f, -1.0f, 100.0f, std::nanf("")}) CHECK(!chooser.set_ratio(bad).is_ok());
    CHECK(chooser.ratio() == 0.5f);
    // Measurements are rounded to 1/8 octave and clamped.
    CHECK(LatentFormChooser::round_ratio(1.0) == 1.0f);
    CHECK(LatentFormChooser::round_ratio(1.04) == 1.0f);
    CHECK(LatentFormChooser::round_ratio(2.0) == 2.0f);
    CHECK(LatentFormChooser::round_ratio(1.09) == static_cast<float>(std::exp2(0.125)));
    CHECK(LatentFormChooser::round_ratio(1e9) == LatentFormChooser::kMaxRatio);
    CHECK(LatentFormChooser::round_ratio(-1.0) == 1.0f);
}

// Automatic: decoding steps absorbed, a prompt's batch expanded when its
// counts say so; calibrated once, after which load on the machine changes
// no pass's form and no bit.
TEST_CASE("Automatic latent form: calibrated once, then the same forms and bits under timing noise") {
    test::TinyModelSpec spec = latent_spec(64);
    spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
    auto model = load_transformer(model_path("latent_f32_q64", spec));
    REQUIRE(model != nullptr);
    CHECK(model->latent_form() == LatentForm::Auto);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens = {1};
    for (int32_t i = 1; i < 300; ++i) tokens.push_back(3 + (i * 37) % (model->config().n_vocab - 3));
    // Uncalibrated (ratio 1): the counts, which favour expanding a first batch.
    REQUIRE(model->forward(std::span<const int32_t>(tokens).first(64), Transformer::Logits::None, route, pool).has_value());
    CHECK(model->latent_stats().expanded_passes == 1);
    const int32_t next = 5;
    REQUIRE(model->forward(std::span<const int32_t>(&next, 1), Transformer::Logits::Last, route, pool).has_value());
    CHECK(model->latent_stats().absorbed_passes == 1);  // decoding
    // The calibration measures once, leaves an empty context and sets the ratio.
    auto ratio = model->calibrate_latent_form(route, pool);
    REQUIRE(ratio.has_value());
    std::printf("  calibrated latent form ratio: %.3f\n", static_cast<double>(*ratio));
    CHECK(*ratio >= LatentFormChooser::kMinRatio);
    CHECK(*ratio <= LatentFormChooser::kMaxRatio);
    CHECK(model->latent_chooser().ratio() == *ratio);
    CHECK(model->n_past() == 0);
    // Passes of every size, then decoding steps: the reference run.
    auto pass_logits = [&]() {
        std::vector<float> out;
        model->reset();
        for (const size_t n : {200, 64, 2, 30}) {
            const size_t at = out.size() / static_cast<size_t>(model->config().n_vocab);
            auto r = model->forward(std::span<const int32_t>(tokens).subspan(at, n), Transformer::Logits::All, route, pool);
            if (!r) return std::vector<float>();
            out.insert(out.end(), r->begin(), r->end());
        }
        return out;
    };
    const auto before = model->latent_stats();
    const std::vector<float> want = pass_logits();
    REQUIRE(!want.empty());
    const int64_t expanded = model->latent_stats().expanded_passes - before.expanded_passes;
    // Timing noise: threads burning every core, random pauses between passes.
    std::atomic<bool> stop{false};
    std::vector<std::thread> noise;
    for (int i = 0; i < 4; ++i) {
        noise.emplace_back([&stop, i] {
            volatile double x = i;
            while (!stop.load(std::memory_order_relaxed)) x = x * 1.0000001 + 1.0;
        });
    }
    std::mt19937 rng(3);
    for (int round = 0; round < 3; ++round) {
        std::this_thread::sleep_for(std::chrono::milliseconds(rng() % 20));
        const auto at = model->latent_stats();
        CHECK(pass_logits() == want);
        CHECK(model->latent_stats().expanded_passes - at.expanded_passes == expanded);
    }
    stop = true;
    for (std::thread& t : noise) t.join();
    CHECK(model->latent_chooser().ratio() == *ratio);
    // The ratio alone moves the threshold: the cheapest expansion takes every
    // pass of two tokens or more, the dearest none of the short ones.
    REQUIRE(model->set_latent_ratio(LatentFormChooser::kMinRatio).is_ok());
    auto at = model->latent_stats();
    REQUIRE(!pass_logits().empty());
    CHECK(model->latent_stats().expanded_passes - at.expanded_passes == 4);
    REQUIRE(model->set_latent_ratio(LatentFormChooser::kMaxRatio).is_ok());
    at = model->latent_stats();
    REQUIRE(!pass_logits().empty());
    CHECK(model->latent_stats().expanded_passes - at.expanded_passes <= 1);  // at most the first, deepest-free pass
    CHECK(!model->set_latent_ratio(0.0f).is_ok());
}

// Through the engine: the ratio is measured once per model and device and
// kept in cache_dir, so two engines (two runs of an app) attend in the same
// forms and give the same bits; a damaged file is measured again.
TEST_CASE("Latent form ratio is kept in cache_dir: two engines give the same bits, damaged files measured again") {
    const std::string dir = temp_path("latent_cache");
    std::filesystem::remove_all(dir);
    EngineConfig config = engine_config(model_path("latent", latent_spec(64)));
    config.backend = BackendKind::Cpu;
    config.cache_dir = dir;
    std::vector<int32_t> prompt = {1};
    for (int32_t i = 1; i < 300; ++i) prompt.push_back(3 + (i * 37) % 250);
    // A seeded, sampled reply: every logit of every step counts.
    auto reply = [&](const EngineConfig& c) {
        auto e = Engine::create(c);
        if (!e) return std::vector<int32_t>();
        SamplingParams p;
        p.temperature = 1.0f;
        p.seed = 7;
        p.max_tokens = 24;
        return run(*e.value(), prompt, p);
    };
    const std::vector<int32_t> measured = reply(config);  // measures the ratio and keeps it
    REQUIRE(!measured.empty());
    std::vector<std::filesystem::path> kept;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().filename().string().starts_with("latent-")) kept.push_back(entry.path());
    }
    REQUIRE(kept.size() == 1);
    REQUIRE(std::filesystem::file_size(kept[0]) == 20);  // magic, version, id, ratio
    auto kept_ratio = [&] {
        std::ifstream in(kept[0], std::ios::binary);
        char header[16];
        float ratio = 0.0f;
        in.read(header, sizeof header);
        in.read(reinterpret_cast<char*>(&ratio), sizeof ratio);
        return ratio;
    };
    const float ratio = kept_ratio();
    // Read back, not measured: the same reply in every engine.
    CHECK(reply(config) == measured);
    CHECK(reply(config) == measured);
    CHECK(kept_ratio() == ratio);
    // What the engines do with it: two models given that ratio produce the
    // same logits for the prompt, bit for bit, in passes of the engine's size.
    ThreadPool pool(4);
    std::vector<std::vector<float>> logits;
    for (int i = 0; i < 2; ++i) {
        auto model = load_transformer(config.model_path);
        REQUIRE(model != nullptr);
        REQUIRE(model->set_latent_ratio(ratio).is_ok());
        logits.push_back(latent_pass_logits(*model, prompt, {512}, pool));
        CHECK(model->latent_stats().expanded_passes + model->latent_stats().absorbed_passes == 1);
    }
    REQUIRE(!logits[0].empty());
    CHECK(logits[0] == logits[1]);
    {
        std::ofstream damaged(kept[0], std::ios::binary | std::ios::trunc);
        damaged << "damaged";
    }
    CHECK(!reply(config).empty());  // measured again, and kept again
    CHECK(std::filesystem::file_size(kept[0]) == 20);
    // Without a cache_dir the ratio measured first in the process is kept in
    // memory: the engines of one process agree too.
    config.cache_dir.clear();
    const std::vector<int32_t> memory = reply(config);
    CHECK(!memory.empty());
    CHECK(reply(config) == memory);
    std::filesystem::remove_all(dir);
}

// Through the engine: describe() reports the latent cache's own size (one
// Q8_0 row of rope_dim + kv_rank values per position and block, no V rows).
TEST_CASE("An engine runs latent attention and reports its cache of latents") {
    // Every form (EngineConfig::latent_form), over a prompt longer than one
    // pass of 64 tokens: the cache is the same whatever the form.
    std::vector<int32_t> prompt = {1};
    for (int32_t i = 1; i < 150; ++i) prompt.push_back(3 + (i * 37) % 250);
    for (const LatentAttentionForm form :
         {LatentAttentionForm::Auto, LatentAttentionForm::Absorbed, LatentAttentionForm::Expanded}) {
        EngineConfig config = engine_config(model_path("latent", latent_spec(64)));
        config.latent_form = form;
        auto engine = Engine::create(config);
        REQUIRE(engine.has_value());
        const std::vector<int32_t> out = run(*engine.value(), prompt, greedy(12));
        CHECK(!out.empty());  // pieces, not tokens: byte tokens of a split character come out together
        const std::string described = engine.value()->describe();
        char expected[64];
        // 2 transformer blocks, each a row of 96 values: 3 Q8_0 blocks of 34 bytes.
        std::snprintf(expected, sizeof expected, "%.1f KiB/token", 2.0 * 3 * 34 / 1024.0);
        CHECK(described.find(expected) != std::string::npos);
        if (described.find(expected) == std::string::npos) std::printf("%s\n", described.c_str());
    }
}

// Key scaling over latent rows: the factors divide the latent channels too,
// which are the values, so attention multiplies its output back. Large
// latent channels (a learned latent norm) are what a Q8_0 cache rounds
// badly in keys and values alike.
TEST_CASE("Key scaling on latent attention brings a Q8_0 cache close to F16, values included") {
    test::TinyModelSpec spec = latent_spec(64);
    spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
    spec.k_norm_outlier = 60.0f;
    const std::string& path = model_path("latent_outlier", spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    TransformerOptions f16;
    f16.kv_type = KvCacheType::F16;
    TransformerOptions q8;
    q8.kv_type = KvCacheType::Q8_0;
    auto reference = load_transformer(path, f16);
    auto plain = load_transformer(path, q8);
    auto scaled = load_transformer(path, q8);
    REQUIRE(reference && plain && scaled);
    REQUIRE(scaled->calibrate_keys(calibration_tokens(*scaled), route, pool).is_ok());
    REQUIRE(scaled->key_scales().size() == static_cast<size_t>(scaled->config().n_layers) * 96);
    const std::vector<float> want = all_logits(*reference, pool);
    const std::vector<float> got_plain = all_logits(*plain, pool);
    const std::vector<float> got_scaled = all_logits(*scaled, pool);
    REQUIRE(!want.empty() && got_plain.size() == want.size() && got_scaled.size() == want.size());
    const double err_plain = max_abs_diff(got_plain, want);
    const double err_scaled = max_abs_diff(got_scaled, want);
    std::printf("  latent rows, Q8_0 KV vs F16: max logit error %.4f plain, %.4f with key scaling\n", err_plain,
                err_scaled);
    CHECK(err_scaled * 4 < err_plain);
}

TEST_CASE("Key scaling brings a quantized KV cache close to F16 on keys with large channels") {
    const std::string& path = model_path("key_outlier", outlier_spec());
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    TransformerOptions f16;
    f16.kv_type = KvCacheType::F16;
    auto reference = load_transformer(path, f16);
    TransformerOptions q8;
    q8.kv_type = KvCacheType::Q8_0;
    auto plain = load_transformer(path, q8);
    auto scaled = load_transformer(path, q8);
    REQUIRE(reference && plain && scaled);
    REQUIRE(scaled->calibrate_keys(calibration_tokens(*scaled), route, pool).is_ok());
    REQUIRE(scaled->key_scales().size() ==
            static_cast<size_t>(scaled->config().n_layers * scaled->config().n_head_kv * scaled->config().head_dim));
    // F16 caches need no factors, nor do keys without large channels.
    REQUIRE(reference->calibrate_keys(calibration_tokens(*reference), route, pool).is_ok());
    CHECK(reference->key_scales().empty());
    TransformerOptions q8_even;
    q8_even.kv_type = KvCacheType::Q8_0;
    auto even = load_transformer(f32_model(), q8_even);
    REQUIRE(even);
    REQUIRE(even->calibrate_keys(calibration_tokens(*even), route, pool).is_ok());
    CHECK(even->key_scales().empty());

    const std::vector<float> want = all_logits(*reference, pool);
    const std::vector<float> got_plain = all_logits(*plain, pool);
    const std::vector<float> got_scaled = all_logits(*scaled, pool);
    REQUIRE(!want.empty() && got_plain.size() == want.size() && got_scaled.size() == want.size());
    const double err_plain = max_abs_diff(got_plain, want);
    const double err_scaled = max_abs_diff(got_scaled, want);
    std::printf("  Q8_0 KV vs F16: max logit error %.4f plain, %.4f with key scaling\n", err_plain, err_scaled);
    CHECK(err_scaled * 4 < err_plain);

    // set_key_scales: the same factors give the same bits; wrong ones are refused.
    auto again = load_transformer(path, q8);
    REQUIRE(again);
    REQUIRE(again->set_key_scales(scaled->key_scales()).is_ok());
    const std::vector<float> got_again = all_logits(*again, pool);
    CHECK(got_again == got_scaled);
    std::vector<float> bad(scaled->key_scales().begin(), scaled->key_scales().end());
    CHECK(again->set_key_scales(std::span<const float>(bad).first(bad.size() - 1)).code() == ErrorCode::InvalidArgument);
    bad[0] = 0.0f;
    CHECK(again->set_key_scales(bad).code() == ErrorCode::InvalidArgument);
    CHECK(reference->set_key_scales(scaled->key_scales()).code() == ErrorCode::InvalidArgument);  // F16 cache
}

TEST_CASE("A saved context is converted to the key scales in use when it was stored with others") {
    const std::string& path = model_path("key_outlier", outlier_spec());
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    TransformerOptions q8;
    q8.kv_type = KvCacheType::Q8_0;
    auto scaled = load_transformer(path, q8);
    auto plain = load_transformer(path, q8);
    REQUIRE(scaled && plain);
    REQUIRE(scaled->calibrate_keys(calibration_tokens(*scaled), route, pool).is_ok());
    std::vector<int32_t> prompt = {1};
    for (int32_t i = 1; i < 30; ++i) prompt.push_back(3 + (i * 31) % (scaled->config().n_vocab - 3));
    const std::vector<int32_t> next = {270, 300};
    // Saved without scales (as before them), loaded with: converted, then
    // close to the scaled model's own continuation.
    REQUIRE(plain->forward(prompt, Transformer::Logits::None, route, pool).has_value());
    std::stringstream without;
    REQUIRE(plain->write_state(without).is_ok());
    REQUIRE(scaled->forward(prompt, Transformer::Logits::None, route, pool).has_value());
    auto own = scaled->forward(next, Transformer::Logits::All, route, pool);
    REQUIRE(own.has_value());
    const std::vector<float> want(own->begin(), own->end());
    REQUIRE(scaled->read_state(without).is_ok());
    CHECK(scaled->n_past() == static_cast<int32_t>(prompt.size()));
    auto converted = scaled->forward(next, Transformer::Logits::All, route, pool);
    REQUIRE(converted.has_value());
    const double err = max_abs_diff(*converted, want);
    std::printf("  converted context vs the scaled model's own: max logit error %.4f\n", err);
    CHECK(err < 0.25);
    // Saved with scales, loaded with the same: bit for bit.
    scaled->reset();
    REQUIRE(scaled->forward(prompt, Transformer::Logits::None, route, pool).has_value());
    std::stringstream with;
    REQUIRE(scaled->write_state(with).is_ok());
    REQUIRE(scaled->read_state(with).is_ok());
    auto same = scaled->forward(next, Transformer::Logits::All, route, pool);
    REQUIRE(same.has_value());
    CHECK(std::vector<float>(same->begin(), same->end()) == want);
}

TEST_CASE("Key scales are measured once and kept in cache_dir: the same replies, damaged files measured again") {
    const std::string dir = temp_path("engine_cache");
    std::filesystem::remove_all(dir);
    EngineConfig config = engine_config(model_path("key_outlier", outlier_spec()));
    config.backend = BackendKind::Cpu;
    auto reply = [](const EngineConfig& c) {
        auto e = Engine::create(c);
        std::string text;
        if (!e) return std::string("<error>");
        auto st = e.value()->generate("hello world", greedy(8), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        return st ? text : std::string("<error>");
    };
    const std::string measured = reply(config);  // no cache_dir: measured at load
    config.cache_dir = dir;
    CHECK(reply(config) == measured);  // measured and kept
    std::vector<std::filesystem::path> kept;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) kept.push_back(entry.path());
    REQUIRE(kept.size() == 1);
    CHECK(kept[0].filename().string().starts_with("keys-"));
    CHECK(reply(config) == measured);  // read back
    {
        std::ofstream damaged(kept[0], std::ios::binary | std::ios::trunc);
        damaged << "not key scales";
    }
    CHECK(reply(config) == measured);  // measured again, and kept again
    CHECK(std::filesystem::file_size(kept[0]) > 64);
    // At most 16 models' files: the least recently used go.
    for (int i = 0; i < 20; ++i) {
        std::ofstream(dir + "/keys-old" + std::to_string(i) + ".bin") << "x";
        std::filesystem::last_write_time(dir + "/keys-old" + std::to_string(i) + ".bin",
                                         std::filesystem::file_time_type::clock::now() - std::chrono::hours(1 + i));
    }
    std::filesystem::remove(kept[0]);
    CHECK(reply(config) == measured);
    size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) files += entry.path().filename() == kept[0].filename() ? 100 : 1;
    CHECK(files == 100 + 15);  // this model's file kept, 15 of the old ones
    // A model whose keys need no scaling keeps that result too (no factors),
    // so its later loads measure nothing either.
    std::filesystem::remove_all(dir);
    EngineConfig even = engine_config(f32_model());
    even.backend = BackendKind::Cpu;
    even.cache_dir = dir;
    const std::string plain = reply(even);
    std::vector<std::filesystem::path> none;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) none.push_back(entry.path());
    REQUIRE(none.size() == 1);
    CHECK(std::filesystem::file_size(none[0]) == 24);  // header only: magic, version, id, count 0
    CHECK(reply(even) == plain);
    std::filesystem::remove_all(dir);
}

// Key scales are measured on a fixed text. A model whose tokenizer cannot
// encode text (an unknown pre-tokenizer) still loads with a quantized KV
// cache: keys stored as computed, the reason in describe(), nothing kept in
// cache_dir, and generation from token ids runs the unscaled model. The same
// weights with a tokenizer that encodes the text keep their scales.
TEST_CASE("A model whose text cannot be encoded loads with a quantized KV cache, keys unscaled, and says so") {
    const std::string dir = temp_path("engine_cache_unscaled");
    std::filesystem::remove_all(dir);
    test::TinyModelSpec spec = outlier_spec();
    spec.bpe_pre = "no-such-pre-tokenizer";
    EngineConfig config = engine_config(model_path("key_outlier_unknown_pre", spec));
    config.backend = BackendKind::Cpu;
    config.cache_dir = dir;
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    const std::string described = engine.value()->describe();
    CHECK(described.find("Keys:       unscaled") != std::string::npos);
    CHECK(described.find("no-such-pre-tokenizer") != std::string::npos);
    CHECK(engine.value()->tokenize("hello", false).status().code() == ErrorCode::Unsupported);
    const std::vector<int32_t> prompt = {1, 270, 300, 5, 290, 77, 310, 280};
    const std::vector<int32_t> out = run(*engine.value(), prompt, greedy(4));
    REQUIRE(!out.empty());
    // The first token is the unscaled model's choice.
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    TransformerOptions q8;
    q8.kv_type = KvCacheType::Q8_0;
    auto plain = load_transformer(model_path("key_outlier_unknown_pre", spec), q8);
    REQUIRE(plain != nullptr);
    CHECK(plain->key_scales().empty());
    auto logits = plain->forward(prompt, Transformer::Logits::Last, Route{cpu.get(), cpu.get(), cpu.get()}, pool);
    REQUIRE(logits.has_value());
    CHECK(out[0] == Sampler::argmax(*logits));
    CHECK((!std::filesystem::exists(dir) || std::filesystem::is_empty(dir)));  // nothing measured, nothing kept
    engine.value().reset();

    // The same weights with their text tokenizer: measured and kept, no warning.
    EngineConfig scaled = engine_config(model_path("key_outlier", outlier_spec()));
    scaled.backend = BackendKind::Cpu;
    scaled.cache_dir = dir;
    auto with_scales = Engine::create(scaled);
    REQUIRE(with_scales.has_value());
    CHECK(with_scales.value()->describe().find("Keys:") == std::string::npos);
    CHECK(!std::filesystem::is_empty(dir));
    std::filesystem::remove_all(dir);
}

TEST_CASE("Group-limited routing scores groups before selecting experts") {
    // Max scoring chooses group 0; the sum of the top two chooses group 1.
    // Negative biased scores stay eligible: masking by multiplication would
    // incorrectly prefer a zero from a group that was excluded.
    const float scores[] = {-0.1f, -0.9f, -0.3f, -0.4f, -0.8f, -0.8f, -0.8f, -0.8f};
    std::vector<uint8_t> allowed;
    select_expert_groups(scores, 8, 4, 1, false, allowed);
    CHECK(allowed == (std::vector<uint8_t>{1, 0, 0, 0}));
    select_expert_groups(scores, 8, 4, 1, true, allowed);
    CHECK(allowed == (std::vector<uint8_t>{0, 1, 0, 0}));
    select_expert_groups(scores, 8, 4, 3, true, allowed);
    CHECK(allowed == (std::vector<uint8_t>{1, 1, 1, 0}));  // tied groups: lower first
    select_expert_groups(scores, 8, 8, 2, true, allowed);
    CHECK(allowed == (std::vector<uint8_t>{1, 0, 1, 0, 0, 0, 0, 0}));
    select_expert_groups(scores, 8, 4, 4, true, allowed);
    CHECK(allowed == (std::vector<uint8_t>{1, 1, 1, 1}));
}

TEST_CASE("Group-limited routing validates metadata and streamed experts keep its logits") {
    test::TinyModelSpec s;
    s.arch = "deepseek2";
    s.n_expert = 8;
    s.n_expert_used = 2;
    s.identical_experts = false;
    s.n_expert_groups = 4;
    s.n_expert_groups_used = 2;
    for (int mode = 0; mode < 2; ++mode) {
        s.sigmoid_gating = mode != 0;
        s.expert_bias = mode != 0;
        const std::string path = model_path("grouped_" + std::to_string(mode), s);
        TransformerOptions opts;
        auto a = load_transformer(path, opts);
        opts.expert_cache_bytes = 1;
        auto b = load_transformer(path, opts);
        REQUIRE(a != nullptr && b != nullptr);
        CHECK(a->config().n_expert_groups == 4);
        CHECK(a->config().n_expert_groups_used == 2);
        ThreadPool pool(2);
        auto cpu = make_cpu_backend(pool);
        const Route route{cpu.get(), cpu.get(), cpu.get()};
        const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
        auto first = a->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(first.has_value());
        const std::vector<float> expected(first->begin(), first->end());
        auto second = b->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(second.has_value());
        CHECK(max_abs_diff(*second, expected) == 0.0f);
        a->reset();
        b->reset();
        for (const int32_t token : tokens) {
            auto la = a->forward({&token, 1}, Transformer::Logits::Last, route, pool);
            auto lb = b->forward({&token, 1}, Transformer::Logits::Last, route, pool);
            REQUIRE(la.has_value() && lb.has_value());
            CHECK(la->size() == lb->size());
            CHECK(std::memcmp(la->data(), lb->data(), la->size_bytes()) == 0);
        }
    }
    const std::pair<int32_t, int32_t> invalid[] = {{3, 1}, {4, 0}, {4, 5}, {16, 1}, {8, 1}};
    for (const auto [groups, used] : invalid) {
        s.n_expert_groups = groups;
        s.n_expert_groups_used = used;
        auto file = MmapLoader::open(model_path("bad_groups_" + std::to_string(groups) + "_" +
                                              std::to_string(used), s));
        REQUIRE(file.has_value());
        auto result = describe_model(**file);
        REQUIRE(!result.has_value());
        CHECK(result.status().code() == ErrorCode::InvalidModel);
        CHECK(result.status().to_string().find("expert_group") != std::string::npos);
    }
}

TEST_CASE("Mixture of experts with identical experts equals its dense twin (routing, top-k, renormalization)") {
    test::TinyModelSpec dense = f32_spec();
    dense.arch = "qwen3";
    test::TinyModelSpec moe = dense;
    moe.arch = "qwen3moe";
    moe.n_expert = 8;
    moe.n_expert_used = 3;
    TransformerOptions options;
    options.kv_type = KvCacheType::F16;
    auto a = load_transformer(model_path("dense_twin", dense), options);
    auto b = load_transformer(model_path("moe_twin", moe), options);
    REQUIRE(a != nullptr && b != nullptr);
    CHECK(b->config().n_expert == 8 && b->config().n_expert_used == 3);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto la = a->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(la.has_value());
    const std::vector<float> dense_logits(la->begin(), la->end());
    auto lb = b->forward(tokens, Transformer::Logits::All, route, pool);  // batched: tokens grouped per expert
    REQUIRE(lb.has_value());
    CHECK(max_abs_diff(*lb, dense_logits) < 1e-3);
    b->reset();
    std::vector<float> stepwise;
    for (const int32_t t : tokens) {  // decode path: one token, top-3 experts each
        auto r = b->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool);
        REQUIRE(r.has_value());
        stepwise.insert(stepwise.end(), r->begin(), r->end());
    }
    CHECK(max_abs_diff(stepwise, dense_logits) < 1e-3);

    // expert_mass < 1 runs a renormalized subset of each token's top-3; with
    // identical experts any subset still equals the dense twin. Unstreamed
    // and streamed (prediction applies the same cut).
    for (const int64_t cache : {int64_t{0}, int64_t{1}}) {
        TransformerOptions fewer = options;
        fewer.expert_mass = 0.4f;
        fewer.expert_cache_bytes = cache;
        auto c = load_transformer(model_path("moe_twin", moe), fewer);
        REQUIRE(c != nullptr);
        auto lc = c->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(lc.has_value());
        CHECK(max_abs_diff(*lc, dense_logits) < 1e-3);
    }
}

TEST_CASE("Fused attn_qkv and [gate | up] matrices give the separate matrices' logits, bit for bit") {
    // The same weights either way (the writer draws them in the same order),
    // so the fused projections and their split must change no bit.
    test::TinyModelSpec separate;
    separate.arch = "phi3";
    test::TinyModelSpec fused = separate;
    fused.fused_qkv = true;
    fused.fused_gate_up = true;
    auto a = load_transformer(model_path("phi3_separate", separate));
    auto b = load_transformer(model_path("phi3_fused", fused));
    REQUIRE(a != nullptr && b != nullptr);
    CHECK(!a->config().blocks[0].fused_qkv && !a->config().blocks[0].fused_gate_up);
    CHECK(b->config().blocks[0].fused_qkv && b->config().blocks[0].fused_gate_up);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto la = a->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(la.has_value());
    const std::vector<float> expected(la->begin(), la->end());
    auto lb = b->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(lb.has_value());
    REQUIRE(lb->size() == expected.size());
    CHECK(std::memcmp(lb->data(), expected.data(), expected.size() * sizeof(float)) == 0);
}

TEST_CASE("MoE expert_mass and max_experts run fewer experts per token") {
    test::TinyModelSpec spec;
    spec.arch = "qwen3moe";
    spec.n_layers = 2;
    spec.n_expert = 16;
    spec.n_expert_used = 4;
    spec.identical_experts = false;
    const std::string& path = model_path("moe_mass", spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 16; ++i) tokens.push_back(3 + (i * 37) % 250);
    auto uses = [&](float mass, int32_t max_experts = 0) {
        TransformerOptions options;
        options.expert_cache_bytes = 1;  // streamed: the store counts expert uses
        options.expert_mass = mass;
        options.max_experts = max_experts;
        auto model = load_transformer(path, options);
        if (model == nullptr || model->expert_store() == nullptr) return uint64_t{0};
        for (const int32_t t : tokens) {
            if (!model->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool)) return uint64_t{0};
        }
        const ExpertStore::Stats st = model->expert_store()->stats();
        return st.hits + st.late + st.misses;
    };
    const uint64_t all = uses(1.0f);
    const uint64_t top1 = uses(1e-6f);  // the first expert alone covers it
    CHECK(all == tokens.size() * 2 * 4);
    CHECK(top1 == tokens.size() * 2);
    const uint64_t half = uses(0.5f);
    CHECK(half >= top1 && half <= all);
    CHECK(uses(1.0f, 2) == tokens.size() * 2 * 2);  // a cap of 2 of the model's 4
    CHECK(uses(1.0f, 9) == all);                    // a cap above the model's top-k changes nothing
}

TEST_CASE("Expert streaming with a small cache (evictions, prefetch, direct I/O) equals in-place expert reads") {
    test::TinyModelSpec spec;  // Q4_0 experts: block-quantized slices at unaligned offsets
    spec.arch = "qwen3moe";
    spec.n_layers = 3;
    spec.n_expert = 32;
    spec.n_expert_used = 4;
    spec.identical_experts = false;
    const std::string& path = model_path("moe_streamed", spec);
    TransformerOptions in_place;
    in_place.expert_cache_bytes = 0;
    TransformerOptions streamed;
    streamed.expert_cache_bytes = 1;  // minimum: a few dozen slots for 96 experts
    auto a = load_transformer(path, in_place);
    auto b = load_transformer(path, streamed);
    REQUIRE(a != nullptr && b != nullptr);
    CHECK(a->expert_store() == nullptr);
    REQUIRE(b->expert_store() != nullptr);
    CHECK(b->expert_store()->entries() < 96);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 40; ++i) tokens.push_back(3 + (i * 37) % 250);
    auto la = a->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(la.has_value());
    const std::vector<float> expected(la->begin(), la->end());
    auto lb = b->forward(tokens, Transformer::Logits::All, route, pool);
    if (!lb) std::printf("  streamed forward failed: %s\n", lb.status().to_string().c_str());
    REQUIRE(lb.has_value());
    CHECK(max_abs_diff(*lb, expected) == 0.0f);
    b->reset();
    std::vector<float> stepwise;
    for (const int32_t t : tokens) {
        auto r = b->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool);
        REQUIRE(r.has_value());
        stepwise.insert(stepwise.end(), r->begin(), r->end());
    }
    a->reset();
    std::vector<float> stepwise_ref;
    for (const int32_t t : tokens) {
        auto r = a->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool);
        REQUIRE(r.has_value());
        stepwise_ref.insert(stepwise_ref.end(), r->begin(), r->end());
    }
    CHECK(max_abs_diff(stepwise, stepwise_ref) == 0.0f);
    const ExpertStore::Stats st = b->expert_store()->stats();
    std::printf("  expert cache: %llu hits, %llu late, %llu misses, %llu loads, %.1f MiB read\n",
                static_cast<unsigned long long>(st.hits), static_cast<unsigned long long>(st.late),
                static_cast<unsigned long long>(st.misses), static_cast<unsigned long long>(st.loads),
                static_cast<double>(st.bytes_read) / (1024.0 * 1024.0));
    CHECK(st.loads > b->expert_store()->entries());  // evictions happened
    CHECK(st.hits + st.late > 0);                     // predictions were used

    // trim() empties the cache but keeps its hot list; warm() queues it again
    // (behind demand reads), and the experts are read again with the output unchanged.
    const size_t hot = b->expert_store()->hot_keys().size();
    CHECK(hot > 0);
    CHECK(b->expert_store()->trim() > 0);
    CHECK(b->expert_store()->trim() == 0);  // nothing left to release
    CHECK(b->expert_store()->hot_keys().size() == hot);
    CHECK(b->expert_store()->warm() == hot);
    b->reset();
    std::vector<float> after_trim;
    for (const int32_t t : tokens) {
        auto r = b->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool);
        REQUIRE(r.has_value());
        after_trim.insert(after_trim.end(), r->begin(), r->end());
    }
    CHECK(max_abs_diff(after_trim, stepwise_ref) == 0.0f);
    CHECK(b->expert_store()->stats().loads > st.loads);
}

// A file may hold each expert's gate and up projections as one [gate | up]
// matrix (ffn_gate_up_exps). Its rows are the split tensors' rows, so the
// logits must not move by a bit, read in place or streamed; streamed, a
// fused expert is one read per matrix: two instead of three.
TEST_CASE("Fused [gate | up] experts give the split experts' logits bit for bit, in place and streamed") {
    test::TinyModelSpec split;  // Q4_0 experts: block-quantized slices at unaligned offsets
    split.arch = "qwen3moe";
    split.n_layers = 3;
    split.n_expert = 32;
    split.n_expert_used = 4;
    split.identical_experts = false;
    split.n_ff_shared = 64;  // a shared expert next to the fused routed ones
    test::TinyModelSpec fused = split;
    fused.fused_gate_up = true;
    TransformerOptions in_place;
    in_place.expert_cache_bytes = 0;
    TransformerOptions streamed;
    streamed.expert_cache_bytes = 1;  // minimum: a few dozen slots for 96 experts, so experts are evicted and read again
    auto a = load_transformer(model_path("moe_split_experts", split), in_place);
    auto b = load_transformer(model_path("moe_fused_experts", fused), in_place);
    auto c = load_transformer(model_path("moe_fused_experts", fused), streamed);
    auto d = load_transformer(model_path("moe_split_experts", split), streamed);
    REQUIRE(a != nullptr && b != nullptr && c != nullptr && d != nullptr);
    for (const BlockPlan& plan : b->config().blocks) CHECK(plan.ffn == FfnKind::Moe && plan.fused_gate_up);
    CHECK(!a->config().blocks[0].fused_gate_up);
    REQUIRE(c->expert_store() != nullptr && d->expert_store() != nullptr);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 40; ++i) tokens.push_back(3 + (i * 37) % 250);
    auto run = [&](Transformer& model, bool stepwise) {  // every row's logits, batched or token by token
        model.reset();
        std::vector<float> out;
        for (size_t at = 0; at < tokens.size(); at += stepwise ? 1 : tokens.size()) {
            const auto batch = std::span<const int32_t>(tokens).subspan(at, stepwise ? 1 : tokens.size());
            auto r = model.forward(batch, stepwise ? Transformer::Logits::Last : Transformer::Logits::All, route, pool);
            CHECK(r.has_value());
            if (!r) return std::vector<float>();
            out.insert(out.end(), r->begin(), r->end());
        }
        return out;
    };
    for (const bool stepwise : {false, true}) {
        const std::vector<float> expected = run(*a, stepwise);
        for (Transformer* model : {b.get(), c.get(), d.get()}) {
            const std::vector<float> got = run(*model, stepwise);
            REQUIRE(got.size() == expected.size());
            CHECK(std::memcmp(got.data(), expected.data(), expected.size() * sizeof(float)) == 0);
        }
    }
    const ExpertStore::Stats fused_reads = c->expert_store()->stats();
    const ExpertStore::Stats split_reads = d->expert_store()->stats();
    std::printf("  fused: %llu experts read in %llu reads; split: %llu in %llu\n",
                static_cast<unsigned long long>(fused_reads.loads), static_cast<unsigned long long>(fused_reads.reads),
                static_cast<unsigned long long>(split_reads.loads), static_cast<unsigned long long>(split_reads.reads));
    CHECK(fused_reads.loads > c->expert_store()->entries());  // evictions happened
    CHECK(fused_reads.reads == 2 * fused_reads.loads);
    CHECK(split_reads.reads == 3 * split_reads.loads);
}

TEST_CASE("Automatic requantization converts the resident Q8_0 matrices only when the expert cache is tight") {
    test::TinyModelSpec spec;  // as below: Q8_0 attention and output, 256 columns
    spec.arch = "qwen3moe";
    spec.n_layers = 2;
    spec.n_embd = 256;
    spec.n_expert = 8;
    spec.n_expert_used = 2;
    spec.identical_experts = false;
    const std::string& path = model_path("moe_requant", spec);
    auto plan_for = [&](int64_t cache_bytes) -> std::pair<Transformer::MemoryPlan, DType> {
        TransformerOptions options;
        options.expert_cache_bytes = cache_bytes;
        options.requant_bits = -1;
        auto model = load_transformer(path, options);
        CHECK(model != nullptr);
        if (model == nullptr || model->expert_store() == nullptr) return {Transformer::MemoryPlan{}, DType::F32};
        return {model->memory_plan(), model->file().tensor("blk.0.attn_q.weight")->type};
    };
    const auto [tight, tight_type] = plan_for(1);  // far below 10% of the experts
    CHECK(tight.requant_bits != 0);
    CHECK(tight_type != DType::Q8_0);
    CHECK(tight.expert_bytes > 0);
    CHECK(tight.recommended_bytes > tight.resident_bytes);
    const auto [roomy, roomy_type] = plan_for(int64_t{1} << 30);  // every expert fits
    CHECK(roomy.requant_bits == 0);
    CHECK(roomy_type == DType::Q8_0);
}

TEST_CASE("skip_slow skips light experts that are not in RAM yet, and stays finite") {
    test::TinyModelSpec spec;
    spec.arch = "qwen3moe";
    spec.n_layers = 3;
    spec.n_expert = 32;
    spec.n_expert_used = 4;
    spec.identical_experts = false;
    const std::string& path = model_path("moe_streamed", spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 24; ++i) tokens.push_back(3 + (i * 37) % 250);
    TransformerOptions options;
    options.expert_cache_bytes = 1;  // a few dozen slots: most experts are not in RAM when chosen
    options.skip_slow = 0.5f;        // below half of the token's weight: every expert but a dominant one
    auto model = load_transformer(path, options);
    REQUIRE(model != nullptr);
    for (const int32_t t : tokens) {
        auto r = model->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool);
        REQUIRE(r.has_value());
        for (const float v : *r) REQUIRE(std::isfinite(v));
    }
    CHECK(model->expert_predictions().skipped > 0);
}

TEST_CASE("Expert streaming with requant_bits converts the resident Q8_0 matrices and stays close") {
    test::TinyModelSpec spec;  // Q8_0 attention and output, 256 columns: eligible for Q4_K / Q5_K
    spec.arch = "qwen3moe";
    spec.n_layers = 2;
    spec.n_embd = 256;
    spec.n_expert = 8;
    spec.n_expert_used = 2;
    spec.identical_experts = false;
    const std::string& path = model_path("moe_requant", spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 24; ++i) tokens.push_back(3 + (i * 37) % 250);
    // Logits of the whole prompt, or empty if loading or the forward pass failed.
    auto run = [&](int32_t bits, DType expected_type) -> std::vector<float> {
        TransformerOptions options;
        options.expert_cache_bytes = 1;
        options.requant_bits = bits;
        auto model = load_transformer(path, options);
        if (model == nullptr || model->expert_store() == nullptr) return {};
        CHECK(model->file().tensor("blk.0.attn_q.weight")->type == expected_type);
        CHECK(model->file().tensor("blk.1.attn_output.weight")->type == expected_type);
        CHECK(model->file().tensor("blk.0.ffn_gate_exps.weight")->type == spec.ffn_type);  // experts untouched
        {  // one converted matrix against its Q8_0 original, within the format's error
            auto original = MmapLoader::open(path);
            if (!original) return {};
            const TensorView& w0 = *original.value()->tensor("blk.0.attn_q.weight");
            const TensorView& w1 = *model->file().tensor("blk.0.attn_q.weight");
            std::vector<float> x(static_cast<size_t>(w0.cols())), y0(static_cast<size_t>(w0.rows())), y1(y0.size());
            for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(0.37f * static_cast<float>(i));
            CHECK(cpu->matmul(w0, x.data(), y0.data(), 1).is_ok());
            CHECK(cpu->matmul(w1, x.data(), y1.data(), 1).is_ok());
            double err = 0, power = 0;
            for (size_t i = 0; i < y0.size(); ++i) {
                err += (static_cast<double>(y1[i]) - y0[i]) * (static_cast<double>(y1[i]) - y0[i]);
                power += static_cast<double>(y0[i]) * y0[i];
            }
            const double bound = bits == 0 ? 0.0 : bits == 4 ? 0.1 : 0.06;  // measured: 0.071 and 0.041
            CHECK(std::sqrt(err / power) <= bound);
        }
        auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
        if (!logits) return {};
        return std::vector<float>(logits->begin(), logits->end());
    };
    const std::vector<float> exact = run(0, DType::Q8_0);
    REQUIRE(!exact.empty());
    auto relative_rms = [&](const std::vector<float>& v) {
        if (v.size() != exact.size()) return 1e9;
        double err = 0, power = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            err += (static_cast<double>(v[i]) - exact[i]) * (static_cast<double>(v[i]) - exact[i]);
            power += static_cast<double>(exact[i]) * exact[i];
        }
        return std::sqrt(err / power);
    };
    const double e4 = relative_rms(run(4, DType::Q4_K));
    const double e5 = relative_rms(run(5, DType::Q5_K));
    // A random two-block model amplifies the weight error (measured 0.45 and
    // 0.22); the real model's quality is measured on its own outputs.
    std::printf("  logits relative RMS error: Q4_K %.4f, Q5_K %.4f\n", e4, e5);
    CHECK(e4 > 0.0);
    CHECK(e4 < 1.0);
    CHECK(e5 < e4);
    TransformerOptions invalid;
    invalid.expert_cache_bytes = 1;
    invalid.requant_bits = 3;
    CHECK(load_transformer(path, invalid) == nullptr);
}

TEST_CASE("requant_bits lowers the feed-forward and query / output matrices of a dense model and stays close") {
    test::TinyModelSpec spec;  // Q8_0 attention and output head, Q5_K feed-forward, 256 columns everywhere
    spec.n_layers = 2;
    spec.n_embd = 256;
    spec.n_ff = 256;
    spec.ffn_type = DType::Q5_K;
    const std::string& path = model_path("dense_lowering", spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 24; ++i) tokens.push_back(3 + (i * 37) % 250);
    auto type_of = [](const Transformer& m, const char* name) { return m.file().tensor(name)->type; };
    // Logits of the whole prompt (empty if loading or the pass failed); checks which matrices were converted.
    auto run = [&](int32_t bits, DType ffn, DType attn) -> std::vector<float> {
        TransformerOptions options;
        options.requant_bits = bits;
        auto model = load_transformer(path, options);
        if (model == nullptr) return {};
        for (const char* name : {"blk.0.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.ffn_down.weight"}) {
            CHECK(type_of(*model, name) == ffn);
        }
        for (const char* name : {"blk.0.attn_q.weight", "blk.1.attn_output.weight"}) CHECK(type_of(*model, name) == attn);
        // Keys, values, the output head and the embeddings cost more quality per byte saved: untouched.
        CHECK(type_of(*model, "blk.0.attn_k.weight") == DType::Q8_0);
        CHECK(type_of(*model, "blk.1.attn_v.weight") == DType::Q8_0);
        CHECK(type_of(*model, "output.weight") == DType::Q8_0);
        CHECK(type_of(*model, "token_embd.weight") == DType::F16);
        CHECK(model->memory_plan().requant_bits == bits);
        auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
        if (!logits) return {};
        return std::vector<float>(logits->begin(), logits->end());
    };
    const std::vector<float> exact = run(0, DType::Q5_K, DType::Q8_0);
    REQUIRE(!exact.empty());
    auto relative_rms = [&](const std::vector<float>& v) {
        if (v.size() != exact.size()) return 1e9;
        double err = 0, power = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            err += (static_cast<double>(v[i]) - exact[i]) * (static_cast<double>(v[i]) - exact[i]);
            power += static_cast<double>(exact[i]) * exact[i];
        }
        return std::sqrt(err / power);
    };
    // 4: Q5_K -> Q4_K and Q8_0 -> Q4_K. 5: Q8_0 -> Q5_K; the Q5_K feed-forward is not wider than its target.
    const double e4 = relative_rms(run(4, DType::Q4_K, DType::Q4_K));
    const double e5 = relative_rms(run(5, DType::Q5_K, DType::Q5_K));
    std::printf("  logits relative RMS error: Q4_K %.4f, Q5_K %.4f\n", e4, e5);
    CHECK(e4 > 0.0);
    CHECK(e4 < 1.0);
    CHECK(e5 > 0.0);
    CHECK(e5 < e4);
    TransformerOptions invalid;
    invalid.requant_bits = 3;
    CHECK(load_transformer(path, invalid) == nullptr);
}

TEST_CASE("Converted weights are kept in the cache directory and mapped by the next load, giving the same bits") {
    namespace fs = std::filesystem;
    // The engine repacks only where the CPU has the i8mm kernels; so does this test (x86: lowered matrices only).
    const bool repack = quant::repack_kernels_available();
    const DType lowered = repack ? DType::Q4_K_R8 : DType::Q4_K;
    test::TinyModelSpec spec;  // Q8_0 attention, Q5_K feed-forward: lowered to Q4_K, then repacked
    spec.n_layers = 2;
    spec.n_embd = 256;
    spec.n_ff = 256;
    spec.ffn_type = DType::Q5_K;
    const std::string& path = model_path("converted_cache", spec);
    const fs::path dir = fs::path(test::temp_dir()) / ("liyab_cwe_" + std::to_string(test::process_id()));
    fs::remove_all(dir);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens;
    for (int32_t i = 0; i < 24; ++i) tokens.push_back(3 + (i * 37) % 250);
    auto options = [&](int32_t bits, uint64_t cap) {
        TransformerOptions o;
        o.requant_bits = bits;
        o.repack_cpu = repack;
        o.converted_cache_dir = dir.string();
        o.converted_cache_bytes = cap;
        return o;
    };
    auto files = [&] {
        size_t n = 0;
        std::error_code ec;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) n += it->path().extension() == ".bin";
        return n;
    };
    auto logits_of = [&](Transformer& model) -> std::vector<float> {
        auto out = model.forward(tokens, Transformer::Logits::All, route, pool);
        return out ? std::vector<float>(out->begin(), out->end()) : std::vector<float>();
    };
    auto from_file = [&](const Transformer& model, const char* name) {
        const TensorView* t = model.file().tensor(name);
        const uint8_t* mapped = model.file().file().data();
        return model.file().converted(*t) && !(t->data >= mapped && t->data < mapped + model.file().file_size());
    };
    // Too small a cap: converted as usual, nothing written.
    {
        auto model = load_transformer(path, options(4, 1000));
        REQUIRE(model != nullptr);
        CHECK(files() == 0);
    }
    std::vector<float> first, second;
    {
        auto model = load_transformer(path, options(4, uint64_t{1} << 30));
        REQUIRE(model != nullptr);
        CHECK(files() == 1);
        CHECK(model->file().tensor("blk.0.ffn_gate.weight")->type == lowered);  // lowered, then repacked
        first = logits_of(*model);
    }
    {
        auto model = load_transformer(path, options(4, uint64_t{1} << 30));
        REQUIRE(model != nullptr);
        CHECK(files() == 1);
        CHECK(model->file().tensor("blk.0.ffn_gate.weight")->type == lowered);
        CHECK(model->file().tensor("blk.0.attn_q.weight")->type == lowered);
        CHECK(from_file(*model, "blk.1.ffn_down.weight"));
        CHECK(model->memory_plan().requant_bits == 4);
        second = logits_of(*model);
    }
    REQUIRE(!first.empty());
    CHECK(first == second);  // the cached bytes are the converted ones
    // Another recipe is another file: requant_bits 5 makes its own.
    {
        auto model = load_transformer(path, options(5, uint64_t{1} << 30));
        REQUIRE(model != nullptr);
        CHECK(files() == 2);
    }
    // The repack alone is a third, under a cap that holds two of them in all: the directory stays within it, and
    // the file kept is the one the next load maps.
    if (repack) {
        uint64_t total = 0;
        for (const auto& entry : fs::directory_iterator(dir)) total += entry.file_size();
        {
            auto model = load_transformer(path, options(0, total));
            REQUIRE(model != nullptr);
            CHECK(files() >= 1);  // how many older ones go depends on the sizes
            uint64_t now = 0;
            for (const auto& entry : fs::directory_iterator(dir)) now += entry.file_size();
            CHECK(now <= total);
        }
        {
            auto model = load_transformer(path, options(0, total));
            REQUIRE(model != nullptr);
            CHECK(model->file().tensor("blk.0.ffn_gate.weight")->type == DType::Q5_K_R8);
            CHECK(from_file(*model, "blk.0.ffn_gate.weight"));
        }
    }
    fs::remove_all(dir);
}

TEST_CASE("Transformer with mixed Q4_0/Q8_0 weights and Q8_0 KV stays close to the reference") {
    auto model = load_transformer(mixed_model());
    REQUIRE(model != nullptr);
    CHECK(model->kv_cache().dtype() == DType::Q8_0);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77};
    auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(logits.has_value());
    const auto expected = Reference(model->file()).logits(tokens);
    const auto vocab = static_cast<size_t>(model->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        double scale = 0;
        for (const float v : expected[t]) scale = std::max(scale, std::fabs(static_cast<double>(v)));
        CHECK(max_abs_diff(logits->subspan(t * vocab, vocab), expected[t]) < 0.1 * std::max(1.0, scale));
    }
}

TEST_CASE("Batched forward equals token-by-token forward, and truncate() rolls back exactly") {
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99};

    auto batched = load_transformer(mixed_model());
    auto stepwise = load_transformer(mixed_model());
    REQUIRE(batched && stepwise);
    auto all = batched->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(all.has_value());
    const std::vector<float> all_copy(all->begin(), all->end());
    const auto vocab = static_cast<size_t>(batched->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        auto one = stepwise->forward(std::span<const int32_t>(&tokens[t], 1), Transformer::Logits::Last, route, pool);
        REQUIRE(one.has_value());
        CHECK(max_abs_diff(one.value(), std::vector<float>(all_copy.begin() + t * vocab, all_copy.begin() + (t + 1) * vocab)) == 0.0);
    }

    // Roll back 4 positions and replay them: identical logits.
    REQUIRE(batched->truncate(6).is_ok());
    CHECK(batched->n_past() == 6);
    auto replay = batched->forward(std::span<const int32_t>(tokens).subspan(6), Transformer::Logits::All, route, pool);
    REQUIRE(replay.has_value());
    CHECK(max_abs_diff(replay.value(), std::vector<float>(all_copy.begin() + 6 * vocab, all_copy.end())) == 0.0);
    CHECK(!batched->truncate(11).is_ok());
}

TEST_CASE("Classic models: batching and rollback preserve every logit on every KV type") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens(70);
    for (size_t i = 0; i < tokens.size(); ++i) tokens[i] = 3 + static_cast<int32_t>((i * 37) % 280);
    for (const std::string arch : {"gpt2", "gptneox", "falcon", "bloom", "mpt", "starcoder", "starcoder2", "phi2",
                                  "stablelm", "orion", "olmo", "command-r", "nemotron", "apertus", "arcee", "dbrx"}) {
        auto spec = test::classic_model_spec(arch);
        if (arch == "arcee") {
            spec.n_ff = 67;  // batch/decode boundaries put different values in SIMD tails
            spec.ffn_type = DType::F32;
        }
        for (const KvCacheType kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            TransformerOptions options;
            options.kv_type = kv;
            const auto& path = model_path("classic_batch_" + arch, spec);
            auto batch = load_transformer(path, options), step = load_transformer(path, options);
            REQUIRE(batch && step);
            auto all = batch->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(all.has_value());
            const std::vector<float> saved(all->begin(), all->end());
            const auto vocab = static_cast<size_t>(batch->config().n_vocab);
            for (size_t i = 0; i < tokens.size(); ++i) {
                auto one = step->forward(std::span(tokens).subspan(i, 1), Transformer::Logits::Last, route, pool);
                REQUIRE(one.has_value());
                CHECK(std::memcmp(one->data(), saved.data() + i * vocab, vocab * sizeof(float)) == 0);
            }
            REQUIRE(batch->truncate(65).is_ok());
            auto replay = batch->forward(std::span(tokens).subspan(65), Transformer::Logits::All, route, pool);
            REQUIRE(replay.has_value());
            CHECK(std::memcmp(replay->data(), saved.data() + 65 * vocab, replay->size_bytes()) == 0);
        }
    }
}

TEST_CASE("Per-block geometry matches an independent float reference") {
    test::TinyModelSpec spec = test::nas_spec();
    spec.tokenizer_padding = (size_t{1} << 20) + 64;
    spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
    TransformerOptions options;
    options.kv_type = KvCacheType::F16;
    auto model = load_transformer(model_path("nas_float_reference", spec), options);
    REQUIRE(model != nullptr);
    CHECK(model->config().n_attn_layers == 3);
    CHECK(model->config().blocks[3].mixer == MixerKind::None && model->config().blocks[3].ffn == FfnKind::None);
    CHECK(model->config().blocks[4].mixer == MixerKind::Linear);
    CHECK(model->config().blocks[2].n_ff == 192);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(logits.has_value());
    const auto expected = Reference(model->file(), spec).logits(tokens);
    const auto vocab = static_cast<size_t>(model->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        const auto row = logits->subspan(t * vocab, vocab);
        CHECK(max_abs_diff(row, expected[t]) < 2e-2);
        CHECK(Sampler::argmax(row) == Sampler::argmax(expected[t]));
    }
    std::vector<float> padding(static_cast<size_t>(model->config().head_dim));
    for (const int32_t slot : {1, 2}) {
        quant::dequantize_row(model->kv_cache().dtype(), model->kv_cache().k_row(slot, 7, 1), padding.data(),
                             model->config().head_dim);
        for (const float v : padding) CHECK(v == 0.0f);
    }
    for (auto make : {make_metal_backend, make_vulkan_backend}) {
        auto gpu = make();
        if (!gpu) continue;
        model->reset();
        const Route gpu_route{gpu->get(), gpu->get(), cpu.get()};
        auto accelerated = model->forward(tokens, Transformer::Logits::All, gpu_route, pool);
        REQUIRE(accelerated.has_value());
        for (size_t t = 0; t < tokens.size(); ++t) {
            const auto row = accelerated->subspan(t * vocab, vocab);
            CHECK(max_abs_diff(row, expected[t]) < 2e-2);
            CHECK(Sampler::argmax(row) == Sampler::argmax(expected[t]));
        }
    }
    // Swapping KV heads between equal-width query blocks preserves the file
    // size and global maxima, but a saved context must still be refused.
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(model->write_state(saved).is_ok());
    std::swap(spec.block_kv_heads[0], spec.block_kv_heads[1]);
    auto other = load_transformer(model_path("nas_other_geometry", spec), options);
    REQUIRE(other != nullptr);
    CHECK(model->file().file_size() == other->file().file_size());
    CHECK(model_fingerprint(model->file().file()) == model_fingerprint(other->file().file()));
    CHECK(model->context_fingerprint() != other->context_fingerprint());
    CHECK(other->read_state(saved).code() == ErrorCode::InvalidArgument);
    CHECK(other->n_past() == 0);
}

TEST_CASE("Absent first blocks and stacks with no FFN load and run") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    for (const bool all : {false, true}) {
        test::TinyModelSpec s = test::nas_spec();
        s.block_heads[0] = s.block_kv_heads[0] = s.block_ff[0] = 0;
        if (all) {
            std::fill(s.block_heads.begin(), s.block_heads.end(), 0);
            std::fill(s.block_kv_heads.begin(), s.block_kv_heads.end(), 0);
            std::fill(s.block_ff.begin(), s.block_ff.end(), 0);
        }
        auto model = load_transformer(model_path(all ? "nas_all_absent" : "nas_leading_absent", s));
        REQUIRE(model != nullptr);
        const std::vector<int32_t> tokens = {1, 270, 300};
        auto logits = model->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(logits.has_value());
        for (const float value : *logits) CHECK(std::isfinite(value));
        CHECK(model->n_past() == 3);
    }
}

TEST_CASE("Saved states and calibration reject residual-scale changes beyond the first MiB") {
    test::TinyModelSpec s;
    s.arch = "granite";
    s.tokenizer_padding = (size_t{1} << 20) + 64;
    s.residual_scale = 0.22f;
    const std::string& first_path = model_path("modern_state_scale_a", s);
    auto first = load_transformer(first_path);
    s.residual_scale = 0.44f;
    const std::string& second_path = model_path("modern_state_scale_b", s);
    auto second = load_transformer(second_path);
    REQUIRE(first && second);
    CHECK(first->file().file_size() == second->file().file_size());
    CHECK(model_fingerprint(first->file().file()) == model_fingerprint(second->file().file()));
    CHECK(first->context_fingerprint() != second->context_fingerprint());
    CHECK(first->file_fingerprint() != second->file_fingerprint());
    for (const std::string arch : {"llama", "qwen35", "gemma2"}) {
        test::TinyModelSpec legacy;
        legacy.arch = arch;
        auto model = load_transformer(model_path("legacy_header_identity_" + arch, legacy));
        REQUIRE(model != nullptr);
        CHECK(model->file_fingerprint() == model_fingerprint(model->file().file()));
    }
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300};
    REQUIRE(first->forward(tokens, Transformer::Logits::Last, route, pool).has_value());
    std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(first->write_state(saved).is_ok());
    CHECK(second->read_state(saved).code() == ErrorCode::InvalidArgument);
    CHECK(second->n_past() == 0);
    const std::string dir = temp_path("modern_scale_keys");
    std::filesystem::remove_all(dir);
    EngineConfig config = engine_config(first_path);
    config.backend = BackendKind::Cpu;
    config.kv_cache_type = KvCacheType::Q8_0;
    config.cache_dir = dir;
    for (const std::string* path : {&first_path, &second_path, &first_path}) {
        config.model_path = *path;
        auto engine = Engine::create(config);
        REQUIRE(engine.has_value());
    }
    size_t files = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().filename().string().starts_with("keys-")) ++files;
    }
    CHECK(files == 2);  // distinct models, then a repeat that reuses its own calibration
    std::filesystem::remove_all(dir);
}

TEST_CASE("Corrected tokenizer modes isolate saved contexts and KV keys beyond the first MiB") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300};
    for (const bool hybrid : {false, true}) {
        for (const std::string mode : {"spm-greedy", "gpt2-nfkc-digits", "llama-bpe-merges"}) {
            const bool greedy = mode == "spm-greedy";
            for (const auto kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
                test::TinyModelSpec spec;
                spec.arch = hybrid ? "qwen35" : "llama";
                spec.tokenizer_padding = (size_t{1} << 20) + 64;
                if (greedy) spec.spm_pre = "spm-greedy";
                else spec.bpe_pre = mode;
                spec.space_prefix = 0;
                TransformerOptions options;
                options.kv_type = kv;
                const std::string label = std::string(hybrid ? "hybrid" : "attn") + mode +
                                          std::to_string(static_cast<int>(kv));
                auto first = load_transformer(model_path("tokenizer_state_a_" + label, spec), options);
                auto same = load_transformer(model_path("tokenizer_state_same_" + label, spec), options);
                spec.space_prefix = 1;
                auto other = load_transformer(model_path("tokenizer_state_b_" + label, spec), options);
                REQUIRE(first && same && other);
                // The standard first-MiB fingerprint cannot see the changed flag.
                CHECK(first->file().file_size() == other->file().file_size());
                CHECK(model_fingerprint(first->file().file()) == model_fingerprint(other->file().file()));
                CHECK(first->context_fingerprint() != other->context_fingerprint());
                REQUIRE(first->forward(tokens, Transformer::Logits::Last, route, pool).has_value());
                std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
                REQUIRE(first->write_state(saved).is_ok());
                REQUIRE(same->read_state(saved).is_ok());
                CHECK(same->n_past() == 3);
                saved.clear(); saved.seekg(0);
                CHECK(other->read_state(saved).code() == ErrorCode::InvalidArgument);
                CHECK(other->n_past() == 0);
            }
        }
    }
}

TEST_CASE("Gated ReLU squared: fused and split biased projections agree across SIMD tails") {
    auto spec = test::classic_model_spec("arcee");
    spec.ungated = false;
    spec.projection_bias = true;
    spec.n_ff = 67;
    spec.ffn_type = DType::F32;
    const auto split_path = model_path("relu2_split_tail", spec);
    spec.fused_gate_up = true;
    const auto fused_path = model_path("relu2_fused_tail", spec);
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    for (const KvCacheType kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        TransformerOptions options;
        options.kv_type = kv;
        auto split = load_transformer(split_path, options), fused = load_transformer(fused_path, options);
        REQUIRE(split && fused);
        for (const std::vector<int32_t> ids : {std::vector<int32_t>{1, 270, 280, 290, 300, 271, 272, 273},
                                              std::vector<int32_t>{274}}) {
            auto a = split->forward(ids, Transformer::Logits::All, route, pool);
            auto b = fused->forward(ids, Transformer::Logits::All, route, pool);
            REQUIRE(a && b && a->size() == b->size());
            CHECK(std::memcmp(a->data(), b->data(), a->size_bytes()) == 0);
        }
    }
}

TEST_CASE("Modern block plans preserve batching and rollback for every KV format") {
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99, 36, 64};
    for (const std::string arch : {"smollm3", "olmo2", "exaone4", "granite", "minicpm", "mistral3", "deci"}) {
        test::TinyModelSpec spec;
        spec.arch = arch;
        spec.n_layers = 4;
        if (arch == "deci") {
            spec = test::nas_spec();
            spec.qk_norm = spec.qk_norm_whole = true;
        } else if (arch == "olmo2" || arch == "exaone4") {
            spec.qk_norm = spec.post_only_norms = true;
            spec.qk_norm_whole = arch == "olmo2";
            if (arch == "exaone4") spec.sliding_window = 8;
        } else if (arch == "granite" || arch == "minicpm") {
            spec.embedding_scale = 12.0f;
            spec.residual_scale = 0.22f;
            spec.logit_scale = 8.0f;
        } else if (arch == "mistral3") {
            spec.attn_temp_length = 4;
            spec.attn_temp_scale = 0.1f;
        }
        const std::string& path = model_path("modern_" + arch, spec);
        for (const auto type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            TransformerOptions options;
            options.kv_type = type;
            auto batched = load_transformer(path, options);
            auto stepwise = load_transformer(path, options);
            REQUIRE(batched && stepwise);
            auto all = batched->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(all.has_value());
            const std::vector<float> expected(all->begin(), all->end());
            const auto vocab = static_cast<size_t>(batched->config().n_vocab);
            for (size_t t = 0; t < tokens.size(); ++t) {
                auto one = stepwise->forward(std::span(tokens).subspan(t, 1), Transformer::Logits::Last, route, pool);
                REQUIRE(one.has_value());
                CHECK(max_abs_diff(*one, std::vector<float>(expected.begin() + t * vocab,
                                                           expected.begin() + (t + 1) * vocab)) == 0.0);
            }
            REQUIRE(batched->truncate(6).is_ok());
            auto replay = batched->forward(std::span(tokens).subspan(6), Transformer::Logits::All, route, pool);
            REQUIRE(replay.has_value());
            CHECK(max_abs_diff(*replay, std::vector<float>(expected.begin() + 6 * vocab, expected.end())) == 0.0);
        }
    }
}

TEST_CASE("Hybrid (DeltaNet) models roll back inside the rollback window, exactly") {
    for (const char* arch : {"qwen35", "mamba", "mamba2"}) {
    test::TinyModelSpec spec;
    spec.arch = arch;
    spec.n_layers = 4;
    if (std::string_view(arch) == "qwen35") spec.delta_net_interval = 2;  // DeltaNet, attention, DeltaNet, attention
    const std::string& path = model_path(std::string("rollback_") + arch, spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99};

    auto model = load_transformer(path);
    auto reference = load_transformer(path);
    REQUIRE(model && reference);
    REQUIRE(model->config().hybrid());
    CHECK(model->recurrent_state_bytes() > 0);
    const auto vocab = static_cast<size_t>(model->config().n_vocab);
    auto all = reference->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(all.has_value());
    const std::vector<float> expected(all->begin(), all->end());
    auto rows = [&](size_t from) { return std::vector<float>(expected.begin() + from * vocab, expected.end()); };

    // Without a window only the trivial rollbacks work.
    REQUIRE(model->forward(tokens, Transformer::Logits::None, route, pool).has_value());
    CHECK(model->truncate(6).code() == ErrorCode::Unsupported);
    CHECK(model->truncate(10).is_ok());

    // One batched pass (speculative verification), then back to any of its last 5 positions.
    model->reset();
    model->set_rollback_window(5);
    REQUIRE(model->forward(tokens, Transformer::Logits::None, route, pool).has_value());
    CHECK(model->truncate(4).code() == ErrorCode::Unsupported);  // position 3 left the window
    REQUIRE(model->truncate(6).is_ok());
    auto replay = model->forward(std::span<const int32_t>(tokens).subspan(6), Transformer::Logits::All, route, pool);
    REQUIRE(replay.has_value());
    CHECK(max_abs_diff(*replay, rows(6)) < 1e-4);

    // Token by token (a draft model), rolled back twice in a row.
    model->reset();
    for (size_t t = 0; t < 8; ++t) {
        REQUIRE(model->forward(std::span<const int32_t>(&tokens[t], 1), Transformer::Logits::None, route, pool));
    }
    REQUIRE(model->truncate(7).is_ok());
    REQUIRE(model->truncate(5).is_ok());
    CHECK(model->truncate(6).code() == ErrorCode::InvalidArgument);  // beyond n_past
    auto again = model->forward(std::span<const int32_t>(tokens).subspan(5), Transformer::Logits::All, route, pool);
    REQUIRE(again.has_value());
    CHECK(max_abs_diff(*again, rows(5)) < 1e-4);
    }
}

// A model with `ffn` FFNs and Q8_0 attention / output: repacked, it computes
// the plain model's logits, and one token at a time exactly the batch's.
static void check_repacked_model(DType ffn, DType packed, const char* name) {
    test::TinyModelSpec spec;  // Q8_0 attention and output
    spec.n_embd = 256;
    spec.n_ff = 512;
    spec.ffn_type = ffn;
    const std::string& path = model_path(name, spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99};
    auto plain = load_transformer(path);
    TransformerOptions options;
    options.repack_cpu = true;
    auto repacked = load_transformer(path, options);
    auto stepwise = load_transformer(path, options);
    REQUIRE(plain && repacked && stepwise);
    CHECK(repacked->file().tensor("blk.0.ffn_gate.weight")->type == packed);
    CHECK(repacked->file().tensor("blk.1.ffn_down.weight")->type == packed);
    CHECK(repacked->file().tensor("blk.0.attn_q.weight")->type == DType::Q8_0_R4);
    CHECK(repacked->file().tensor("output.weight")->type == DType::Q8_0_R4);
    CHECK(plain->file().tensor("blk.0.ffn_gate.weight")->type == ffn);
    auto a = plain->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(a.has_value());
    const std::vector<float> expected(a->begin(), a->end());
    auto b = repacked->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(b.has_value());
    const std::vector<float> batched(b->begin(), b->end());
    CHECK(batched == expected);  // the repacked kernels round like the file-layout ones
    // One token at a time gives exactly the batch's logits (gemv == gemm).
    const auto vocab = static_cast<size_t>(stepwise->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        auto one = stepwise->forward(std::span<const int32_t>(&tokens[t], 1), Transformer::Logits::Last, route, pool);
        REQUIRE(one.has_value());
        const std::vector<float> row(batched.begin() + t * vocab, batched.begin() + (t + 1) * vocab);
        CHECK(max_abs_diff(*one, row) == 0.0);
    }
}

TEST_CASE("Streamed experts repacked on read (CPU i8mm): the same bits as in-place experts, whatever the batch") {
    if (!quant::repack_kernels_available() || !detect_cpu().i8mm) {
        std::printf("  skipped: no i8mm on this CPU or build\n");
        return;
    }
    test::TinyModelSpec spec;  // K-quant experts with repackable shapes
    spec.arch = "qwen3moe";
    spec.n_layers = 2;
    spec.n_embd = 256;
    spec.n_ff = 512;
    spec.n_expert = 8;
    spec.n_expert_used = 2;
    spec.identical_experts = false;
    spec.ffn_type = DType::Q4_K;
    const std::string& path = model_path("moe_repack_experts", spec);
    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99};
    TransformerOptions in_place;
    in_place.expert_cache_bytes = 0;
    TransformerOptions streamed;
    streamed.expert_cache_bytes = 1;
    streamed.repack_cpu = true;
    auto a = load_transformer(path, in_place);
    auto b = load_transformer(path, streamed);
    auto c = load_transformer(path, streamed);
    REQUIRE(a && b && c);
    REQUIRE(b->expert_store() != nullptr);
    auto la = a->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(la.has_value());
    const std::vector<float> expected(la->begin(), la->end());
    auto lb = b->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(lb.has_value());
    const std::vector<float> batched(lb->begin(), lb->end());
    // In place: the file's layout, plain resident matrices. Streamed: experts
    // repacked as the batched pass reads them, resident matrices repacked at
    // load. One token at a time (c), experts read for single-token passes
    // stay in the file's layout. All three give the same bits.
    CHECK(batched == expected);
    const auto vocab = static_cast<size_t>(c->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        auto one = c->forward(std::span<const int32_t>(&tokens[t], 1), Transformer::Logits::Last, route, pool);
        REQUIRE(one.has_value());
        const std::vector<float> row(batched.begin() + t * vocab, batched.begin() + (t + 1) * vocab);
        CHECK(max_abs_diff(*one, row) == 0.0);
    }
}

// A token's attention is cut into spans that follow its own position only. At a long context the logits are then the same bits
// however the prompt was split into passes (typing ahead, chunked prefill,
// speculative verification), whichever schedule ran the spans (a task per
// span when decoding, a task per token in a large batch) and however many
// threads the power manager left active; for every KV cache type, with a
// sliding window and on a hybrid model too.
TEST_CASE("Attention at a long context: the same bits whatever the batch, the threads or the KV cache type") {
    test::TinyModelSpec attention_only;
    attention_only.context_length = 2900;
    test::TinyModelSpec hybrid = attention_only;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    // Tokens with 1 span, several, and the most a token gets.
    const int32_t n = Transformer::kAttentionMinSpan * Transformer::kAttentionMaxSpans + 76;
    std::vector<int32_t> tokens(static_cast<size_t>(n));
    std::mt19937 rng(47);
    for (int32_t& t : tokens) t = 3 + static_cast<int32_t>(rng() % 250);
    const std::vector<int32_t> probes = {Transformer::kAttentionMinSpan - 1, 2 * Transformer::kAttentionMinSpan + 5,
                                         700, n - 1};
    struct Setup {
        const char* model;
        KvCacheType kv;
        int32_t window;
    };
    std::vector<Setup> setups;
    for (const KvCacheType kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        setups.push_back({"long_attention", kv, 0});
        setups.push_back({"long_attention", kv, 300});
    }
    setups.push_back({"long_hybrid", KvCacheType::Q8_0, 0});
    for (const Setup& setup : setups) {
        const bool is_hybrid = std::string_view(setup.model) == "long_hybrid";
        const std::string& path = model_path(setup.model, is_hybrid ? hybrid : attention_only);
        TransformerOptions options;
        options.context_length = 2900;
        options.kv_type = setup.kv;
        options.sliding_window = setup.window;
        options.sink_tokens = 4;
        // The probes' logits, the tokens fed in passes of the `passes` sizes, then one at a time.
        auto run = [&](ThreadPool& pool, const std::vector<int32_t>& passes) {
            std::vector<std::vector<float>> out(probes.size());
            auto model = load_transformer(path, options);
            if (!model) return out;
            auto cpu = make_cpu_backend(pool);
            const Route route{cpu.get(), cpu.get(), cpu.get()};
            const auto vocab = static_cast<size_t>(model->config().n_vocab);
            int32_t at = 0;
            size_t pass = 0;
            while (at < n) {
                const int32_t len = pass < passes.size() ? std::min(passes[pass++], n - at) : 1;
                auto logits = model->forward(std::span<const int32_t>(tokens).subspan(static_cast<size_t>(at),
                                                                                       static_cast<size_t>(len)),
                                             Transformer::Logits::All, route, pool);
                if (!logits) return out;
                for (size_t p = 0; p < probes.size(); ++p) {
                    if (probes[p] >= at && probes[p] < at + len) {
                        const auto row = static_cast<size_t>(probes[p] - at);
                        out[p].assign(logits->begin() + static_cast<std::ptrdiff_t>(row * vocab),
                                      logits->begin() + static_cast<std::ptrdiff_t>((row + 1) * vocab));
                    }
                }
                at += len;
            }
            return out;
        };
        ThreadPool four(4), one(1), shed(4);
        shed.set_active_threads(2);
        const auto whole = run(four, {n});                     // one pass: a task per (token, KV head)
        const auto split = run(four, {1, 37, 255, 256, 1500});  // passes of every size, then decoding
        const auto single = run(one, {300, 1700});             // one thread: other schedules
        const auto shedded = run(shed, {2000});                // half the threads active, then decoding
        for (size_t p = 0; p < probes.size(); ++p) {
            REQUIRE(!whole[p].empty() && !split[p].empty() && !single[p].empty() && !shedded[p].empty());
            const bool same = whole[p] == split[p] && whole[p] == single[p] && whole[p] == shedded[p];
            if (!same) {
                std::printf("  %s, kv %d, window %d: position %d differs\n", setup.model, static_cast<int>(setup.kv),
                            setup.window, probes[p]);
            }
            CHECK(same);
        }
    }
}

// Decode-time attention at a long context with Qwen3.6's attention shape (16
// query heads, 2 KV heads of 256): set LIYAB_BENCH_ATTENTION=1 to time it.
TEST_CASE("Benchmark: attention at a long context (LIYAB_BENCH_ATTENTION=1)") {
    if (std::getenv("LIYAB_BENCH_ATTENTION") == nullptr) return;
    const int positions = std::max(1, std::atoi(std::getenv("LIYAB_BENCH_ATTENTION")) > 1
                                          ? std::atoi(std::getenv("LIYAB_BENCH_ATTENTION"))
                                          : 3000);  // LIYAB_BENCH_ATTENTION=N: N positions
    const int steps = 20;
    test::TinyModelSpec spec;
    spec.arch = "qwen3";
    spec.n_layers = 1;
    spec.n_embd = 4096;
    spec.n_ff = 64;
    spec.n_head = 16;
    spec.n_head_kv = 2;
    spec.context_length = std::max(4096, positions + steps);
    const std::string& path = model_path("attention_bench", spec);
    ThreadPool pool(0);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    TransformerOptions options;
    options.context_length = spec.context_length;
    auto model = load_transformer(path, options);
    REQUIRE(model != nullptr);
    std::vector<int32_t> prompt;
    for (int32_t i = 0; i < positions; ++i) prompt.push_back(3 + (i * 37) % 250);
    for (size_t i = 0; i < prompt.size(); i += 512) {
        const std::span<const int32_t> part(prompt.data() + i, std::min<size_t>(512, prompt.size() - i));
        REQUIRE(model->forward(part, Transformer::Logits::None, route, pool).has_value());
    }
    const double before = model->phase_times().attention;
    for (int i = 0; i < steps; ++i) {
        const int32_t t = 3 + i;
        REQUIRE(model->forward(std::span<const int32_t>(&t, 1), Transformer::Logits::Last, route, pool).has_value());
    }
    std::printf("  attention (projections included) at %d positions: %.2f ms per token per layer\n", positions,
                (model->phase_times().attention - before) / steps);
}

TEST_CASE("Repacked weights (CPU i8mm): same logits, batch-size independent") {
    if (!quant::repack_kernels_available() || !detect_cpu().i8mm) {
        std::printf("  skipped: no i8mm on this CPU or build\n");
        return;
    }
    check_repacked_model(DType::Q4_K, DType::Q4_K_R8, "q4k_ffn");
    check_repacked_model(DType::Q5_K, DType::Q5_K_R8, "q5k_ffn");
}

TEST_CASE("Context limit is enforced") {
    TransformerOptions options;
    options.context_length = 8;
    auto model = load_transformer(mixed_model(), options);
    REQUIRE(model != nullptr);
    ThreadPool pool(2);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens(8, 270);
    REQUIRE(model->forward(tokens, Transformer::Logits::None, route, pool).has_value());
    auto full = model->forward(std::span<const int32_t>(tokens).first(1), Transformer::Logits::Last, route, pool);
    CHECK(!full.has_value());
    CHECK(full.status().code() == ErrorCode::ContextFull);
    const int32_t bad = 100000;
    model->reset();
    CHECK(!model->forward(std::span<const int32_t>(&bad, 1), Transformer::Logits::Last, route, pool).has_value());
}

TEST_CASE("Sliding window with attention sinks matches the reference") {
    for (const int sinks : {0, 4}) {
        TransformerOptions options;
        options.kv_type = KvCacheType::F16;
        options.sliding_window = 8;
        options.sink_tokens = sinks;
        options.max_batch = 4;
        options.context_length = 256;
        auto model = load_transformer(f32_model(), options);
        REQUIRE(model != nullptr);
        CHECK(model->kv_cache().window() == 8);
        CHECK(model->kv_cache().sink_tokens() == sinks);
        ThreadPool pool(4);
        auto cpu = make_cpu_backend(pool);
        const Route route{cpu.get(), cpu.get(), cpu.get()};

        std::vector<int32_t> tokens;
        std::mt19937 rng(11);
        for (int i = 0; i < 30; ++i) tokens.push_back(3 + static_cast<int32_t>(rng() % 300));
        std::vector<float> last_logits;
        for (size_t i = 0; i < tokens.size(); i += 7) {  // paged cache: any batch size works
            const auto part = std::span<const int32_t>(tokens).subspan(i, std::min<size_t>(7, tokens.size() - i));
            auto r = model->forward(part, Transformer::Logits::Last, route, pool);
            REQUIRE(r.has_value());
            last_logits.assign(r->begin(), r->end());
        }
        const auto expected = Reference(model->file()).logits(tokens, 8, sinks);
        CHECK(max_abs_diff(last_logits, expected.back()) < 2e-2);
    }
}

// ---------------------------------------------------------------------------
// Engine and speculative decoding
// ---------------------------------------------------------------------------
TEST_CASE("Engine generates text from a prompt and streams pieces") {
    EngineConfig config = engine_config(mixed_model());
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    std::printf("%s", engine.value()->describe().c_str());

    std::string text;
    int pieces = 0;
    auto stats = engine.value()->generate("hello world", greedy(16), [&](std::string_view piece, int32_t) {
        text += piece;
        ++pieces;
        return true;
    });
    REQUIRE(stats.has_value());
    CHECK(stats->prompt_tokens == 3);
    CHECK(stats->generated_tokens > 0 && stats->generated_tokens <= 16);
    CHECK(pieces > 0);
    CHECK(stats->tokens_per_second > 0);

    // Greedy decoding is deterministic across calls (the context is rewound to the common prefix).
    std::string again;
    REQUIRE(engine.value()->generate("hello world", greedy(16), [&](std::string_view piece, int32_t) {
        again += piece;
        return true;
    }).has_value());
    CHECK(text == again);
}

TEST_CASE("Context reuse: continuing a conversation or a prefill equals a fresh context") {
    EngineConfig config = engine_config(f32_model());
    auto reused = Engine::create(config);
    auto fresh = Engine::create(config);
    REQUIRE(reused.has_value() && fresh.has_value());
    auto run = [](Engine& e, const std::string& prompt, GenerationStats* out = nullptr) {
        std::string text;
        auto st = e.generate(prompt, greedy(12), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (st && out != nullptr) *out = *st;
        return st ? text : std::string("<error>");
    };
    // Turn 1, then turn 2 continuing turn 1's prompt and reply: only the new tokens are processed.
    const std::string turn1 = "hello world";
    const std::string reply = run(*reused.value(), turn1);
    const std::string turn2 = turn1 + reply + " hello";
    GenerationStats st{};
    const std::string continued = run(*reused.value(), turn2, &st);
    CHECK(st.cached_prefix_tokens > 0);
    fresh.value()->reset_context();
    CHECK(continued == run(*fresh.value(), turn2));
    // A prefill followed by a prompt that extends it.
    reused.value()->reset_context();
    REQUIRE(reused.value()->prefill("hello", true).is_ok());
    GenerationStats st2{};
    const std::string after_prefill = run(*reused.value(), "hello world", &st2);
    CHECK(st2.cached_prefix_tokens > 0);
    fresh.value()->reset_context();
    CHECK(after_prefill == run(*fresh.value(), "hello world"));
    // An unrelated prompt drops the old context.
    GenerationStats st3{};
    run(*reused.value(), "world", &st3);
    CHECK(st3.cached_prefix_tokens <= 1);
}

TEST_CASE("Forced continuations are emitted, counted, and decoding continues exactly from them") {
    EngineConfig config = engine_config(f32_model());
    auto forced_engine = Engine::create(config);
    auto fresh = Engine::create(config);
    REQUIRE(forced_engine.has_value() && fresh.has_value());
    Engine& e = *forced_engine.value();

    // Forced once, after the first sampled piece.
    const std::string forced = " world hello";
    std::string text;
    std::vector<int32_t> tokens;
    auto st = e.generate("hello world", greedy(8),
                         [&](std::string_view piece, int32_t token) {
                             text += piece;
                             tokens.push_back(token);
                             return true;
                         },
                         [&](std::string_view generated) {
                             return generated == text && tokens.size() == 1 ? forced : std::string();
                         });
    REQUIRE(st.has_value());
    const auto forced_tokens = e.tokenize(forced, false);
    REQUIRE(forced_tokens.has_value());
    CHECK(st->generated_tokens == 8);  // sampled tokens only
    CHECK(st->forced_tokens == static_cast<int32_t>(forced_tokens->size()));
    // Pieces are whole UTF-8, so a piece may carry several byte tokens: compare text.
    REQUIRE(tokens.size() >= 2);
    CHECK(tokens[1] == forced_tokens->back());
    const std::string first = text.substr(0, text.find(forced));
    REQUIRE(first.size() < text.size());

    // The rest equals greedy decoding of the prompt extended with the first token and the forced text.
    auto prompt = fresh.value()->tokenize("hello world", true);
    REQUIRE(prompt.has_value());
    prompt->push_back(tokens[0]);
    prompt->insert(prompt->end(), forced_tokens->begin(), forced_tokens->end());
    std::string expected;
    REQUIRE(fresh.value()->generate_tokens(*prompt, greedy(7), [&](std::string_view piece, int32_t) {
        expected += piece;
        return true;
    }).has_value());
    CHECK(text.substr(first.size() + forced.size()) == expected);

    // The context holds the forced tokens: continuing the conversation reuses all of it.
    auto st2 = e.generate_tokens(*prompt, greedy(1), nullptr);
    REQUIRE(st2.has_value());
    CHECK(st2->cached_prefix_tokens >= static_cast<int32_t>(prompt->size()) - 1);
}

namespace {

// The engine's rule for emitting pieces: up to the last whole character.
size_t whole_utf8_prefix(std::string_view s) {
    const size_t n = s.size();
    for (size_t back = 1; back <= std::min<size_t>(4, n); ++back) {
        const auto c = static_cast<unsigned char>(s[n - back]);
        if ((c & 0xC0) == 0x80) continue;
        const size_t need = (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return back >= need ? n : n - back;
    }
    return n;
}

// Forced text by call: the force callback's n-th call (1-based) forces
// plan[n]. Calls come after every sampled token that ends a whole character.
using ForcePlan = std::vector<std::pair<int, std::string>>;

std::string planned(const ForcePlan& plan, int call) {
    for (const auto& [n, text] : plan) {
        if (n == call) return text;
    }
    return {};
}

// What generate() with `plan` must produce, built one sampled token at a
// time with plain greedy decoding (a fresh generate_tokens() per token) and
// the forced tokens appended to the prompt: the output text and whether
// generation ended on an end-of-generation token.
struct ForcedReference {
    std::string text;
    std::vector<int32_t> tokens;  // the prompt's, the sampled and the forced ones
    int32_t sampled = 0;
    int32_t forced_tokens = 0;
    bool ended = false;
};

void forced_reference(Engine& e, const std::string& prompt, int32_t max_tokens, const ForcePlan& plan,
                      ForcedReference& ref) {
    auto seq = e.tokenize(prompt, true);
    REQUIRE(seq.has_value());
    e.reset_context();
    std::string generated, pending;
    int call = 0;
    while (ref.sampled < max_tokens) {
        int32_t token = -1;
        auto st = e.generate_tokens(*seq, greedy(1), [&](std::string_view, int32_t t) {
            token = t;
            return true;
        });
        REQUIRE(st.has_value());
        if (st->generated_tokens == 0) {  // end of generation
            ref.ended = true;
            break;
        }
        REQUIRE(token >= 0);  // a token whose piece is empty would not be seen
        ++ref.sampled;
        seq->push_back(token);
        pending += e.token_to_piece(token);
        const size_t ready = whole_utf8_prefix(pending);
        generated += pending.substr(0, ready);
        pending.erase(0, ready);
        if (ref.sampled >= max_tokens || !pending.empty()) continue;
        const std::string forced = planned(plan, ++call);
        if (forced.empty()) continue;
        const auto forced_tokens = e.tokenize(forced, false);
        REQUIRE(forced_tokens.has_value());
        seq->insert(seq->end(), forced_tokens->begin(), forced_tokens->end());
        ref.forced_tokens += static_cast<int32_t>(forced_tokens->size());
        const size_t whole = whole_utf8_prefix(forced);
        generated += forced.substr(0, whole);
        pending = forced.substr(whole);
    }
    ref.text = generated + pending;
    ref.tokens = *seq;
}

}  // namespace

// The forced tokens and the token sampled after them share one pass: the
// last forced token no longer needs a decoding step of its own. Same output
// as appending the forced text to the prompt, on attention-only and hybrid
// models, with plain decoding and with speculative decoding (lookup and a
// draft model, whose steps return several tokens and are cut where the
// output is forced), and with forced text that ends inside a character.
TEST_CASE("Forced continuations share their pass with the next token: one pass less, the same output") {
    test::TinyModelSpec hybrid;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    const std::string prompt = "hello world hello world hello world hello world hello";
    // A plain forced text, one that ends inside "é" (its first byte only: a
    // cut buffer), one with a whole "é".
    const ForcePlan plan = {{1, " world hello"}, {3, " hel\xC3"}, {4, " w\xC3\xA9rld"}, {6, " hello"}};
    constexpr int32_t kMaxTokens = 16;
    for (const std::string& path : {mixed_model(), model_path("hybrid", hybrid), ssm_model("mamba"), ssm_model("mamba2")}) {
        EngineConfig plain = engine_config(path);
        plain.backend = BackendKind::Cpu;
        auto reference_engine = Engine::create(plain);
        REQUIRE(reference_engine.has_value());
        ForcedReference ref;
        forced_reference(*reference_engine.value(), prompt, kMaxTokens, plan, ref);
        REQUIRE(ref.forced_tokens > 0);

        EngineConfig lookup = plain;
        lookup.lookup_drafts = true;
        lookup.suffix_drafts = false;     // context lookup
        lookup.adaptive_drafts = false;  // every step drafts
        EngineConfig drafted = lookup;
        drafted.lookup_drafts = false;
        drafted.draft_model_path = path;  // the target itself: every draft accepted, steps of 4 tokens
        EngineConfig suffix = lookup;
        suffix.suffix_drafts = true;
        for (const EngineConfig* config : {&plain, &lookup, &suffix, &drafted}) {
            const bool speculative = config != &plain;
            std::printf("  %s, %s\n", path.c_str(),
                        config == &plain    ? "plain"
                        : config == &lookup ? "lookup"
                        : config == &suffix ? "suffix"
                                            : "draft model");
            auto engine = Engine::create(*config);
            REQUIRE(engine.has_value());
            std::string text;
            std::vector<std::string> pieces;
            int calls = 0;
            auto st = engine.value()->generate(
                prompt, greedy(kMaxTokens),
                [&](std::string_view piece, int32_t) {
                    text += piece;
                    pieces.emplace_back(piece);
                    return true;
                },
                [&](std::string_view) { return planned(plan, ++calls); });
            REQUIRE(st.has_value());
            CHECK(text == ref.text);
            CHECK(st->generated_tokens == ref.sampled);
            CHECK(st->forced_tokens == ref.forced_tokens);
            // Whole characters only: " hel" is emitted at once, its cut "é"
            // (the byte 0xC3) with the next token's piece.
            const auto hel = std::find(pieces.begin(), pieces.end(), " hel");
            REQUIRE(hel != pieces.end() && hel + 1 != pieces.end());
            CHECK((hel + 1)->starts_with("\xC3"));
            // One pass per sampled token (and one for an end-of-generation
            // token): none for the forced texts. Speculative steps verify
            // several tokens per pass.
            const int32_t sampled_passes = st->generated_tokens + (ref.ended ? 1 : 0);
            if (config == &drafted) {
                CHECK(st->decode_passes < sampled_passes);
                CHECK(st->draft_tokens_accepted > 0);
            } else if (speculative) {
                CHECK(st->decode_passes <= sampled_passes);
                CHECK(st->draft_tokens_proposed > 0);
            } else {
                CHECK(st->decode_passes == sampled_passes);
            }

            // The context holds the forced tokens: continuing it keeps all of
            // it (all but the last token sampled, which is never processed).
            if (config != &drafted) {
                auto st2 = engine.value()->generate_tokens(ref.tokens, greedy(1), nullptr);
                REQUIRE(st2.has_value());
                CHECK(st2->cached_prefix_tokens == static_cast<int32_t>(ref.tokens.size()) - 1);
            }
        }
    }
}

TEST_CASE("The prefix cache brings back a conversation another one replaced, exactly") {
    for (const char* arch : {"llama", "qwen35", "gpt2", "bloom", "gptneox", "command-r", "nemotron", "mamba", "mamba2"}) {
        test::TinyModelSpec spec = std::string(arch) == "llama" || std::string(arch) == "qwen35" || std::string(arch) == "mamba" || std::string(arch) == "mamba2"
                                       ? test::TinyModelSpec{} : test::classic_model_spec(arch);
        spec.arch = arch;
        spec.n_layers = 4;
        if (std::string(arch) == "qwen35") spec.delta_net_interval = 2;
        EngineConfig config = engine_config(model_path(std::string("prefix_") + arch, spec));
        config.backend = BackendKind::Cpu;
        config.prefix_cache_mb = 64;
        auto cached = Engine::create(config);
        config.prefix_cache_mb = 0;
        auto fresh = Engine::create(config);
        REQUIRE(cached.has_value() && fresh.has_value());
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        std::string a, b;
        for (int i = 0; i < 200; ++i) {  // past kMinPrefixEntry (128) tokens
            a += i % 3 ? " hello" : " world";
            b += i % 2 ? " world" : " hello world";
        }
        Engine& e = *cached.value();
        const bool classic = std::string(arch) != "llama" && std::string(arch) != "qwen35";
        std::string reply_a;
        if (classic) {
            // Keep an exact prompt prefix: random classic weights can emit
            // EOS early or text that re-encodes differently at the boundary.
            REQUIRE(e.prefill(a, true).is_ok());
            REQUIRE(e.prefill(b, true).is_ok());
        } else {
            reply_a = run(e, a);
            run(e, b);  // another conversation: A's context goes to the prefix cache
        }
        const std::string next_a = a + reply_a + " hello";
        GenerationStats st{};
        const std::string out = run(e, next_a, &st);
        const auto a_tokens = static_cast<int32_t>(e.tokenize(a, true)->size());
        // A's prompt and its reply as generated, but the reply's last token,
        // which was sampled and never processed: nothing of A done again.
        CHECK(st.cached_prefix_tokens == a_tokens + (classic ? 0 : 5));
        CHECK(out == run(*fresh.value(), next_a));
    }
}

TEST_CASE("A conversation the prefix cache set aside is saved as kept, and restores whole") {
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;
    EngineConfig config = engine_config(model_path("prefix_qwen35", spec));
    config.backend = BackendKind::Cpu;
    config.prefix_cache_dir = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_saved_as_kept";
    std::filesystem::remove_all(config.prefix_cache_dir);
    for (const int ram_mb : {64, 0}) {  // the copy in RAM, then on storage
        config.prefix_cache_mb = ram_mb;
        config.prefix_cache_disk_mb = ram_mb > 0 ? 0 : 64;
        auto engine = Engine::create(config);
        auto loader = Engine::create(config);
        REQUIRE(engine.has_value() && loader.has_value());
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        std::string a, b;
        for (int i = 0; i < 200; ++i) {  // past kMinPrefixEntry (128) tokens
            a += i % 3 ? " hello" : " world";
            b += i % 2 ? " world" : " hello world";
        }
        Engine& e = *engine.value();
        const std::string next_a = a + run(e, a) + " hello";
        run(e, b);  // another conversation: A's context goes to the prefix cache
        const std::string file = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_set_aside.bin";
        CHECK(!e.save_state(file, "unrelated text").is_ok());
        REQUIRE(e.save_state(file, next_a).is_ok());
        REQUIRE(loader.value()->load_state(file).has_value());
        GenerationStats st{};
        run(*loader.value(), next_a, &st);
        // A's prompt and its reply, but the reply's last token (sampled, never processed).
        CHECK(st.cached_prefix_tokens == static_cast<int32_t>(e.tokenize(a, true)->size()) + 5);
        std::remove(file.c_str());
    }
    std::filesystem::remove_all(config.prefix_cache_dir);
}

namespace {

// The prefix cache's store in `dir`: its files (kv-*; not the lock) and every byte there.
size_t store_files(const std::string& dir) {
    size_t n = 0;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        n += it->path().filename().string().starts_with("kv-") ? 1 : 0;
    }
    return n;
}

uint64_t dir_bytes(const std::string& dir) {
    uint64_t bytes = 0;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code size_ec;
        const auto size = it->file_size(size_ec);
        if (!size_ec) bytes += size;
    }
    return bytes;
}

}  // namespace

TEST_CASE("Sparse index keys survive storage prefix reuse and an engine restart in every KV format") {
    for (const bool compressed : {false, true}) {
    auto spec = latent_spec(64);
    spec.arch = "glm-dsa";
    spec.index_dim = 64;
    spec.index_topk = 8;
    spec.index_full = {true, false};
    if (compressed) spec = test::compressed_spec();
    const std::string path = model_path(compressed ? "compressed_prefix" : "indexed_prefix", spec);
    std::string a, b;
    for (int i = 0; i < 200; ++i) {
        a += i % 3 ? " hello" : " world";
        b += i % 2 ? " world" : " hello world";
    }
    auto run = [](Engine& engine, const std::string& prompt, GenerationStats* stats = nullptr) {
        std::string text;
        auto result = engine.generate(prompt, greedy(6), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        CHECK(result.has_value());
        if (result && stats != nullptr) *stats = *result;
        return text;
    };
    for (const KvCacheType type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        EngineConfig config = engine_config(path);
        config.backend = BackendKind::Cpu;
        config.kv_cache_type = type;
        auto reference = Engine::create(config);
        REQUIRE(reference.has_value());
        const std::string expected = run(*reference.value(), a);
        const std::string dir = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) +
                                "_indexed_prefix_" + std::to_string(static_cast<int>(type));
        std::filesystem::remove_all(dir);
        config.prefix_cache_dir = dir;
        config.prefix_cache_disk_mb = 4;
        config.prefix_cache_lossy = true;  // F32 index images must use the exact raw fallback
        {
            auto cached = Engine::create(config);
            REQUIRE(cached.has_value());
            CHECK(run(*cached.value(), a) == expected);
            run(*cached.value(), b);
            GenerationStats restored{};
            CHECK(run(*cached.value(), a, &restored) == expected);
            CHECK(restored.cached_prefix_tokens >= 128);
        }
        {
            auto restarted = Engine::create(config);
            REQUIRE(restarted.has_value());
            GenerationStats restored{};
            CHECK(run(*restarted.value(), a, &restored) == expected);
            CHECK(restored.cached_prefix_tokens >= 128);
        }
        std::filesystem::remove_all(dir);
    }
    }
}

TEST_CASE("The prefix cache keeps contexts on storage within its cap, for later engines too") {
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;
    EngineConfig config = engine_config(model_path("prefix_qwen35", spec));
    config.backend = BackendKind::Cpu;
    const std::string dir = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_prefix_files";
    std::filesystem::remove_all(dir);
    auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
        std::string text;
        auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (st && out != nullptr) *out = *st;
        return st ? text : std::string("<error>");
    };
    std::string a, b;
    for (int i = 0; i < 200; ++i) {  // past kMinPrefixEntry (128) tokens
        a += i % 3 ? " hello" : " world";
        b += i % 2 ? " world" : " hello world";
    }
    auto reference = Engine::create(config);  // never interrupted
    REQUIRE(reference.has_value());
    config.prefix_cache_mb = 0;  // no RAM: storage only
    config.prefix_cache_dir = dir;
    config.prefix_cache_disk_mb = 64;
    const uint64_t cap = uint64_t{64} << 20;
    std::string next_a, last_a;
    GenerationStats held{};
    {
        // A file of a build before the store is removed when the engine opens the directory.
        std::filesystem::create_directories(dir);
        std::ofstream(dir + "/ctx-1-1-1.state") << "stale";
        auto cached = Engine::create(config);
        REQUIRE(cached.has_value());
        CHECK(!std::filesystem::exists(dir + "/ctx-1-1-1.state"));
        CHECK(store_files(dir) == 0);
        Engine& e = *cached.value();
        const std::string reply_a = run(e, a);
        run(e, b);  // A's context goes to storage
        Engine::PrefixCacheStats ps = e.prefix_cache_stats();
        CHECK(ps.ram_bytes == 0);
        CHECK(ps.storage_bytes > 0);
        CHECK(ps.storage_bytes == dir_bytes(dir));
        CHECK(ps.storage_cap_bytes == cap);
        CHECK(ps.contexts == 1);
        CHECK(ps.set_asides == 1);
        next_a = a + reply_a + " hello";
        GenerationStats st{};
        const std::string out = run(e, next_a, &st);
        REQUIRE(reference.value()->prefill(a, true).is_ok());
        CHECK(out == run(*reference.value(), next_a, &held));
        CHECK(st.cached_prefix_tokens >= held.cached_prefix_tokens - 1);
        CHECK(e.prefix_cache_stats().restores == 1);
        last_a = next_a + out + " world";
        run(e, b);  // B back from storage, A to it
        ps = e.prefix_cache_stats();
        CHECK(ps.restores == 2);
        CHECK(ps.storage_bytes <= cap);
        CHECK(ps.storage_bytes == dir_bytes(dir));
    }
    // The files outlive the engine, which set its context (B) aside there
    // too: a later one restores A from them, exactly.
    CHECK(store_files(dir) > 0);
    CHECK(dir_bytes(dir) <= cap);
    {
        auto later = Engine::create(config);
        REQUIRE(later.has_value());
        CHECK(later.value()->prefix_cache_stats().contexts == 2);
        GenerationStats st{};
        const std::string out = run(*later.value(), last_a, &st);
        CHECK(out == run(*reference.value(), last_a, &held));
        CHECK(st.cached_prefix_tokens >= held.cached_prefix_tokens - 1);
        CHECK(later.value()->prefix_cache_stats().restores == 1);
    }
    // Little RAM, many conversations: the least recently used move from RAM
    // to storage and come back exactly from there; neither tier goes above
    // its cap.
    config.prefix_cache_mb = 1;  // a dozen of these contexts
    {
        auto cached = Engine::create(config);
        REQUIRE(cached.has_value());
        std::vector<std::string> conversations, replies;
        for (int c = 0; c < 40; ++c) {
            std::string text = std::to_string(c);
            for (int i = 0; i < 200; ++i) text += (i + c) % 3 ? " hello" : " world";
            conversations.push_back(text);
            replies.push_back(run(*cached.value(), text));
            const Engine::PrefixCacheStats ps = cached.value()->prefix_cache_stats();
            CHECK(ps.ram_bytes <= (uint64_t{1} << 20));
            CHECK(ps.storage_bytes <= cap);
        }
        const Engine::PrefixCacheStats ps = cached.value()->prefix_cache_stats();
        CHECK(ps.ram_bytes > 0);
        CHECK(ps.storage_bytes == dir_bytes(dir));
        CHECK(ps.contexts >= 20);  // 39 set aside, besides A's and B's
        const std::string next = conversations[0] + replies[0] + " hello";
        GenerationStats st{};
        const std::string out = run(*cached.value(), next, &st);
        EngineConfig plain = engine_config(model_path("prefix_qwen35", spec));
        plain.backend = BackendKind::Cpu;  // as `cached`: automatic routing would round some passes on the GPU
        auto fresh = Engine::create(plain);
        REQUIRE(fresh.has_value() && fresh.value()->prefill(conversations[0], true).is_ok());
        CHECK(out == run(*fresh.value(), next, &held));
        CHECK(st.cached_prefix_tokens >= held.cached_prefix_tokens - 1);
    }
    // A lower cap: the least recently used go when the directory is opened.
    config.prefix_cache_mb = 0;
    config.prefix_cache_disk_mb = 1;
    {
        auto small = Engine::create(config);
        REQUIRE(small.has_value());
        CHECK(small.value()->prefix_cache_stats().storage_bytes <= (uint64_t{1} << 20));
        CHECK(dir_bytes(dir) <= (uint64_t{1} << 20));
    }
    CHECK(dir_bytes(dir) <= (uint64_t{1} << 20));
    // No storage tier: the directory is left as it is.
    config.prefix_cache_disk_mb = 0;
    {
        const size_t before = store_files(dir);
        auto cached = Engine::create(config);
        REQUIRE(cached.has_value());
        run(*cached.value(), a);
        run(*cached.value(), b);
        CHECK(store_files(dir) == before);
        CHECK(cached.value()->prefix_cache_stats().set_asides == 0);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("Conversations sharing a system prompt keep it once, and a new one restores it exactly") {
    for (const char* arch : {"llama", "qwen35", "mamba", "mamba2"}) {
        test::TinyModelSpec spec;
        spec.arch = arch;
        spec.n_layers = 4;
        if (std::string(arch) == "qwen35") spec.delta_net_interval = 2;
        spec.context_length = 1024;  // a system prompt of several chunks and a question
        EngineConfig config = engine_config(model_path(std::string("prefix_1k_") + arch, spec));
        config.backend = BackendKind::Cpu;
        // Passes of 1024 tokens (max_batch follows draft_tokens; no drafting
        // without a draft model or lookup): the system prompt is processed
        // in one pass, so its end is a hybrid model's oldest snapshot, the
        // one set-aside contexts keep, as on a MoE whose experts stream
        // (1024-token passes). With 64-token passes the oldest is at 64.
        config.draft_tokens = 1023;
        auto fresh = Engine::create(config);
        config.prefix_cache_mb = 64;
        auto cached = Engine::create(config);
        REQUIRE(cached.has_value() && fresh.has_value());
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        std::string system = "system:";
        for (int i = 0; i < 600; ++i) system += i % 4 ? " hello" : " world";
        // Each question drops more than 128 tokens when the next one comes,
        // so the next one sets the conversation aside; they part after a few
        // words.
        auto question = [](int q) {
            std::string text = q % 2 ? " world world" : " hello hello";
            for (int i = 0; i < 220; ++i) text += (i / (q + 1)) % 2 ? " hello" : " world";
            return text;
        };
        // The system prompt as the app prepares it: processed once, saved,
        // and restored after a load (its end is then the oldest snapshot, the
        // one a hybrid model's set-aside contexts keep).
        const std::string file = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_" + arch + "_system.bin";
        REQUIRE(fresh.value()->prefill(system, true).is_ok());
        REQUIRE(fresh.value()->save_state(file).is_ok());
        Engine& e = *cached.value();
        REQUIRE(e.load_state(file).has_value());
        const auto system_tokens = static_cast<int32_t>(e.tokenize(system, true)->size());
        for (int q = 0; q < 3; ++q) run(e, system + question(q));
        // Two conversations set aside: the second found the system prompt's
        // whole chunks stored already.
        Engine::PrefixCacheStats ps = e.prefix_cache_stats();
        const uint64_t restores = ps.restores;  // attention-only: the third question restored the first's start
        CHECK(ps.set_asides == 2);
        REQUIRE(ps.chunk_tokens > 0);
        CHECK(system_tokens / ps.chunk_tokens >= 2);
        CHECK(ps.chunks_reused >= static_cast<uint64_t>(system_tokens / ps.chunk_tokens));
        CHECK(ps.chunks < static_cast<int32_t>(ps.chunks_written + ps.chunks_reused));
        // A new conversation with nothing live: the system prompt comes from the store.
        e.reset_context();
        const std::string next = system + question(7);
        GenerationStats st{};
        const std::string out = run(e, next, &st);
        REQUIRE(fresh.value()->load_state(file).has_value());
        CHECK(out == run(*fresh.value(), next));
        CHECK(st.cached_prefix_tokens >= system_tokens);
        if (std::string(arch) != "llama") CHECK(st.cached_prefix_tokens == system_tokens);  // its state image
        CHECK(e.prefix_cache_stats().restores == restores + 1);
        std::remove(file.c_str());
    }
}

TEST_CASE("A chat after a long shared system prompt is set aside and comes back; an edited draft is not") {
    for (const char* arch : {"llama", "qwen35", "mamba", "mamba2"}) {
        test::TinyModelSpec spec;
        spec.arch = arch;
        spec.n_layers = 4;
        if (std::string(arch) == "qwen35") spec.delta_net_interval = 2;
        spec.context_length = 1024;  // the chat and a long draft
        EngineConfig config = engine_config(model_path(std::string("prefix_1k_") + arch, spec));
        config.backend = BackendKind::Cpu;
        auto reference = Engine::create(config);  // never interrupted
        config.prefix_cache_mb = 64;
        auto cached = Engine::create(config);
        REQUIRE(cached.has_value() && reference.has_value());
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        // The system prompt is most of each chat: the next chat shares more
        // than half of the context, and drops more than 128 tokens of it.
        std::string system = "system:", a = " a:", b = " b:", draft1, draft2;
        for (int i = 0; i < 300; ++i) system += i % 3 ? " hello" : " world";
        for (int i = 0; i < 150; ++i) {
            a += i % 2 ? " hello" : " world";
            b += i % 4 ? " world" : " hello";
        }
        for (int i = 0; i < 200; ++i) {
            draft1 += i % 5 ? " hello" : " world";
            draft2 += i % 7 ? " world" : " hello";
        }
        Engine& e = *cached.value();
        const std::string a1 = system + a;
        const std::string reply_a = run(e, a1);
        CHECK(reply_a == run(*reference.value(), a1));
        run(e, system + b);  // another chat: A is set aside
        CHECK(e.prefix_cache_stats().set_asides == 1);
        const std::string a2 = a1 + reply_a + " hello";
        GenerationStats st{};
        const std::string out = run(e, a2, &st);
        CHECK(out == run(*reference.value(), a2));
        // A's prompt and its reply, but the reply's last token (sampled, never processed).
        CHECK(st.cached_prefix_tokens == static_cast<int32_t>(e.tokenize(a1, true)->size()) + 5);
        CHECK(e.prefix_cache_stats().restores == 1);
        // A draft typed ahead after the reply, then edited: left in place, not set aside.
        const uint64_t before = e.prefix_cache_stats().set_asides;
        REQUIRE(e.prefill(a2 + out + draft1, true).is_ok());
        REQUIRE(e.prefill(a2 + out + draft2, true).is_ok());
        CHECK(e.prefix_cache_stats().set_asides == before);
        // A short chat C, then A again. Leaving C rewinds it to the system
        // prompt on an attention-only model (40 tokens to process again: C
        // is not set aside), to its start on a hybrid one (no snapshot below:
        // C is set aside). Either way the store gives A back, more than the
        // live context keeps.
        e.reset_context();
        std::string c = " c:";
        for (int i = 0; i < 40; ++i) c += i % 2 ? " world" : " hello";
        run(e, system + c);
        const uint64_t aside = e.prefix_cache_stats().set_asides, restores = e.prefix_cache_stats().restores;
        const std::string a3 = a2 + out + " world";
        const std::string out3 = run(e, a3, &st);
        CHECK(out3 == run(*reference.value(), a3));
        CHECK(st.cached_prefix_tokens >= static_cast<int32_t>(e.tokenize(a1, true)->size()) + 5);
        CHECK(e.prefix_cache_stats().restores == restores + 1);
        CHECK(e.prefix_cache_stats().set_asides == aside + (std::string(arch) != "llama" ? 1 : 0));
    }
}

TEST_CASE("A later engine restores the prefix an earlier one left on storage") {
    for (const char* arch : {"llama", "qwen35", "mamba", "mamba2"}) {
        test::TinyModelSpec spec;
        spec.arch = arch;
        spec.n_layers = 4;
        if (std::string(arch) == "qwen35") spec.delta_net_interval = 2;
        EngineConfig config = engine_config(model_path(std::string("prefix_") + arch, spec));
        config.backend = BackendKind::Cpu;
        config.draft_tokens = 255;  // one pass for the system prompt (see the test above)
        auto fresh = Engine::create(config);
        REQUIRE(fresh.has_value());
        const std::string dir = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_later_" + arch;
        std::filesystem::remove_all(dir);
        config.prefix_cache_dir = dir;
        config.prefix_cache_disk_mb = 64;
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        std::string system = "system:";
        for (int i = 0; i < 200; ++i) system += i % 3 ? " hello" : " world";
        // The system prompt saved once and restored after the load, as the app does.
        const std::string file = dir + "_system.bin";
        REQUIRE(fresh.value()->prefill(system, true).is_ok());
        REQUIRE(fresh.value()->save_state(file).is_ok());
        {
            auto first = Engine::create(config);
            REQUIRE(first.has_value());
            REQUIRE(first.value()->load_state(file).has_value());
            run(*first.value(), system + " world hello world");
        }  // destroyed: its context goes to storage
        auto later = Engine::create(config);
        REQUIRE(later.has_value());
        const std::string prompt = system + " hello hello world";
        GenerationStats st{};
        const std::string out = run(*later.value(), prompt, &st);
        const auto system_tokens = static_cast<int32_t>(later.value()->tokenize(system, true)->size());
        CHECK(st.cached_prefix_tokens >= system_tokens);
        CHECK(out == run(*fresh.value(), prompt));
        std::remove(file.c_str());
        later.value().reset();  // destroyed: the directory is released
        std::filesystem::remove_all(dir);
    }
}

TEST_CASE("The prefix cache removes what it cannot trust on storage, and the output stays exact") {
    test::TinyModelSpec spec;
    spec.n_layers = 4;
    EngineConfig config = engine_config(model_path("prefix_llama", spec));
    config.backend = BackendKind::Cpu;
    auto reference = Engine::create(config);
    REQUIRE(reference.has_value());
    const std::string dir = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_untrusted";
    std::filesystem::remove_all(dir);
    config.prefix_cache_dir = dir;
    config.prefix_cache_disk_mb = 64;
    auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
        std::string text;
        auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (st && out != nullptr) *out = *st;
        return st ? text : std::string("<error>");
    };
    std::string a, b;
    for (int i = 0; i < 300; ++i) {
        a += i % 3 ? " hello" : " world";
        b += i % 2 ? " world" : " hello world";
    }
    std::string next_a;
    {
        auto e = Engine::create(config);
        REQUIRE(e.has_value());
        next_a = a + run(*e.value(), a) + " hello";
        run(*e.value(), b);
    }
    std::vector<std::filesystem::path> chunks;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.path().extension() == ".chunk") chunks.push_back(entry.path());
    }
    REQUIRE(chunks.size() >= 4);
    std::sort(chunks.begin(), chunks.end());
    // A write a crash cut short, a file that is not the store's format, a
    // truncated chunk: removed when the directory is opened.
    const std::string name(32, 'a');
    std::ofstream(dir + "/kv-" + name + ".chunk.tmp") << "partial";
    std::ofstream(dir + "/kv-" + std::string(32, 'b') + ".chunk") << "not a chunk";
    std::filesystem::resize_file(chunks[0], std::filesystem::file_size(chunks[0]) - 1);
    // The others with one payload byte changed: found when read.
    for (size_t i = 1; i < chunks.size(); ++i) {
        std::fstream f(chunks[i], std::ios::in | std::ios::out | std::ios::binary);
        f.seekg(-1, std::ios::end);
        const char last = static_cast<char>(f.get());
        f.seekp(-1, std::ios::end);
        f.put(static_cast<char>(last ^ 0x5A));
    }
    {
        auto e = Engine::create(config);
        REQUIRE(e.has_value());
        CHECK(!std::filesystem::exists(dir + "/kv-" + name + ".chunk.tmp"));
        CHECK(!std::filesystem::exists(dir + "/kv-" + std::string(32, 'b') + ".chunk"));
        CHECK(!std::filesystem::exists(chunks[0]));
        GenerationStats st{};
        const std::string out = run(*e.value(), next_a, &st);
        REQUIRE(reference.value()->prefill(a, true).is_ok());
        CHECK(out == run(*reference.value(), next_a));
        // Whatever was restored passed its hash; a damaged chunk is gone.
        size_t damaged_left = 0;
        for (size_t i = 1; i < chunks.size(); ++i) damaged_left += std::filesystem::exists(chunks[i]) ? 1 : 0;
        CHECK(damaged_left < chunks.size() - 1);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("One engine at a time uses a prefix cache directory") {
    test::TinyModelSpec spec;
    spec.n_layers = 4;
    EngineConfig config = engine_config(model_path("prefix_llama", spec));
    config.backend = BackendKind::Cpu;
    config.prefix_cache_dir = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_one_engine";
    config.prefix_cache_disk_mb = 16;
    std::filesystem::remove_all(config.prefix_cache_dir);
    {
        auto first = Engine::create(config);
        auto second = Engine::create(config);  // the first holds the directory: no storage tier
        REQUIRE(first.has_value() && second.has_value());
        CHECK(first.value()->prefix_cache_stats().storage_cap_bytes == (uint64_t{16} << 20));
        CHECK(second.value()->prefix_cache_stats().storage_cap_bytes == 0);
    }
    auto third = Engine::create(config);  // released with the first
    REQUIRE(third.has_value());
    CHECK(third.value()->prefix_cache_stats().storage_cap_bytes == (uint64_t{16} << 20));
    third.value().reset();  // destroyed: the directory is released
    std::filesystem::remove_all(config.prefix_cache_dir);
}

TEST_CASE("A prefill cancelled mid-pass keeps its work and the output stays exact") {
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;
    EngineConfig config = engine_config(model_path("hybrid", spec));
    config.backend = BackendKind::Cpu;
    auto fresh = Engine::create(config);
    REQUIRE(fresh.has_value());
    std::string text;
    for (int i = 0; i < 60; ++i) text += i % 2 ? " hello" : " world";
    const std::string prompt = text + " hello";
    auto run = [](Engine& e, const std::string& p) {
        std::string out;
        auto st = e.generate(p, greedy(8), [&](std::string_view piece, int32_t) {
            out += piece;
            return true;
        });
        return st ? out : std::string("<error>");
    };
    const std::string expected = run(*fresh.value(), prompt);
    int cancelled = 0;
    for (const int delay_us : {0, 50, 200, 800, 3000}) {
        auto e = Engine::create(config);
        REQUIRE(e.has_value());
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
            e.value()->cancel_prefill();
        });
        const Status st = e.value()->prefill(text, true);
        canceller.join();
        CHECK((st.is_ok() || st.code() == ErrorCode::Cancelled));
        if (st.code() == ErrorCode::Cancelled) ++cancelled;
        CHECK(run(*e.value(), prompt) == expected);  // resumes from what the prefill kept
    }
    std::printf("  %d of 5 prefills were cancelled before they ended\n", cancelled);
}

// A context can hold other tokens than its text's own encoding, as after a
// generated reply (a model may write a word in pieces the tokenizer would
// merge): "hello world hel" processed, then continued with "lo …", leaves
// "▁hel" + "lo" where encoding the text gives "▁hello". The next turn reuses
// such a context only while the engine knows which text its tokens encode;
// otherwise it re-encodes the prompt, diverges at the split word, and a
// hybrid model, which rewinds only to its state snapshots, starts over.
TEST_CASE("After cancelled work a turn still reuses a context whose text re-encodes differently") {
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;
    spec.context_length = 4096;
    EngineConfig config = engine_config(model_path("hybrid_4k", spec));
    config.backend = BackendKind::Cpu;
    const std::string head = "hello world hel";
    std::string conversation = head + "lo";
    for (int i = 0; i < 40; ++i) conversation += i % 3 ? " world" : " hello";
    std::string typed;  // long enough to be cancelled while it is processed
    for (int i = 0; i < 1500; ++i) typed += i % 2 ? " hello" : " world";
    const std::string next = conversation + typed + " hello";
    auto prepare = [&](Engine& e) { return e.prefill(head, true).is_ok() && e.prefill(conversation, true).is_ok(); };
    auto run = [](Engine& e, const std::string& prompt, GenerationStats* out) {
        std::string text;
        auto st = e.generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (st && out != nullptr) *out = *st;
        return st ? text : std::string("<error>");
    };

    // Never interrupted: the turn continues the tokens the context holds.
    auto reference = Engine::create(config);
    REQUIRE(reference.has_value() && prepare(*reference.value()));
    GenerationStats st{};
    const std::string expected = run(*reference.value(), next, &st);
    const int32_t held = st.cached_prefix_tokens;
    const auto own = reference.value()->tokenize(conversation, true);
    REQUIRE(own.has_value());
    REQUIRE(held == static_cast<int32_t>(own->size()) + 1);  // "▁hel" + "lo" instead of "▁hello"

    const std::string file = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_split_state.bin";
    // 0: a prefill typed ahead, cancelled for the user's message; 1: a
    // generation stopped while its prompt is processed; 2: as 0, then the
    // context saved and restored by a new engine (a later process).
    for (int kind = 0; kind < 3; ++kind) {
        // The work, true when it was cancelled before it ended.
        auto work = [&](Engine& e) {
            if (kind != 1) return e.prefill(conversation + typed, true).code() == ErrorCode::Cancelled;
            auto r = e.generate(next + typed, greedy(8), nullptr);
            return r.has_value() && r->cancelled && r->generated_tokens == 0;
        };
        // Stopped part way through: how long it takes uninterrupted sets when.
        std::unique_ptr<Engine> e;
        {
            auto created = Engine::create(config);
            REQUIRE(created.has_value() && prepare(*created.value()));
            const auto start = std::chrono::steady_clock::now();
            work(*created.value());
            e = std::move(created).value();
            const auto took = std::chrono::steady_clock::now() - start;
            bool cancelled = false;
            for (const double share : {0.5, 0.25, 0.75, 0.1}) {
                auto fresh_engine = Engine::create(config);
                REQUIRE(fresh_engine.has_value());
                e = std::move(fresh_engine).value();
                REQUIRE(prepare(*e));
                std::thread stopper([&] {
                    std::this_thread::sleep_for(std::chrono::duration_cast<std::chrono::microseconds>(took * share));
                    kind == 1 ? e->cancel() : e->cancel_prefill();
                });
                cancelled = work(*e);
                stopper.join();
                if (cancelled) break;
            }
            REQUIRE(cancelled);
        }
        if (kind == 2) {
            REQUIRE(e->save_state(file).is_ok());
            auto created = Engine::create(config);
            REQUIRE(created.has_value());
            e = std::move(created).value();
            REQUIRE(e->load_state(file).has_value());
        }
        GenerationStats after{};
        CHECK(run(*e, next, &after) == expected);
        CHECK(after.cached_prefix_tokens >= held);
    }
    std::remove(file.c_str());
}

TEST_CASE("The prefix cache brings back a conversation whose text re-encodes differently") {
    for (const char* arch : {"llama", "qwen35", "mamba", "mamba2"}) {
        test::TinyModelSpec spec;
        spec.arch = arch;
        spec.n_layers = 4;
        if (std::string(arch) == "qwen35") spec.delta_net_interval = 2;
        EngineConfig config = engine_config(model_path(std::string("prefix_") + arch, spec));
        config.backend = BackendKind::Cpu;
        config.prefix_cache_mb = 64;
        auto cached = Engine::create(config);
        config.prefix_cache_mb = 0;
        auto reference = Engine::create(config);
        REQUIRE(cached.has_value() && reference.has_value());
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        // Conversation A holds a split word (see the test above); B is another one.
        const std::string head = "hello world hel";
        std::string a = head + "lo", b;
        for (int i = 0; i < 200; ++i) {  // past kMinPrefixEntry (128) tokens
            a += i % 3 ? " hello" : " world";
            b += i % 2 ? " world" : " hello world";
        }
        Engine& e = *cached.value();
        REQUIRE(e.prefill(head, true).is_ok());
        const std::string reply_a = run(e, a);
        run(e, b);  // A's context goes to the prefix cache
        const std::string next_a = a + reply_a + " hello";
        GenerationStats st{};
        const std::string out = run(e, next_a, &st);
        // The reference holds the same tokens for A and was never interrupted.
        // A hybrid model sets A aside at its pinned snapshot, the end of A's
        // prompt but its last token (which decoding processes), and encodes the
        // reply again; an attention-only one keeps the reply's tokens.
        Engine& r = *reference.value();
        REQUIRE(r.prefill(head, true).is_ok());
        if (std::string(arch) == "qwen35") {
            REQUIRE(r.prefill(a, true).is_ok());
        } else {
            CHECK(run(r, a) == reply_a);
        }
        GenerationStats held{};
        CHECK(out == run(r, next_a, &held));
        CHECK(st.cached_prefix_tokens >= held.cached_prefix_tokens - 1);
    }
}

// A reply's byte-level pieces may split a character between two tokens, so
// a context's text can end inside a character. A prompt continuing it must
// still reuse the context (up to the token before), from the live context
// and from the prefix cache alike; it used to be re-encoded whole, which on a
// hybrid model processed the conversation again.
TEST_CASE("A conversation whose last token ends inside a character is reused, live or from the prefix cache") {
    for (const char* arch : {"llama", "qwen35", "mamba", "mamba2"}) {
        test::TinyModelSpec spec;
        spec.arch = arch;
        spec.n_layers = 4;
        if (std::string(arch) == "qwen35") spec.delta_net_interval = 2;
        EngineConfig config = engine_config(model_path(std::string("split_char_") + arch, spec));
        config.backend = BackendKind::Cpu;
        config.prefix_cache_mb = 64;
        auto created = Engine::create(config);
        REQUIRE(created.has_value());
        Engine& e = *created.value();
        std::string a, b;
        for (int i = 0; i < 200; ++i) {  // past kMinPrefixEntry (128) tokens
            a += i % 3 ? " hello" : " world";
            b += i % 2 ? " world" : " hello world";
        }
        // The context holds "hel" + "lo" (prefilled apart, as typing ahead
        // does), so encoding its text again gives other tokens and only the
        // text it remembers finds it. That text ends inside "é": its first byte
        // is the last token the context holds (a byte token; the one token
        // sampled is not processed). A reply's pieces end that way when the
        // model writes a character in two tokens; a prompt is the simplest way.
        const std::string split = "hello world hel";
        const std::string head = split + "lo" + a + "\xC3";
        const auto prompt = e.tokenize(head, true);
        REQUIRE(prompt.has_value());
        const auto held = static_cast<int32_t>(prompt->size()) - 1;  // all but the split character's first byte
        for (const bool set_aside : {false, true}) {
            e.reset_context();
            REQUIRE(e.prefill(split, true).is_ok());
            REQUIRE(e.generate(head, greedy(1), nullptr).has_value());
            // Another conversation replaces it: it comes back from the prefix cache.
            if (set_aside) REQUIRE(e.generate(b, greedy(1), nullptr).has_value());
            // The conversation goes on with "é" whole: the text continues with
            // the character's second byte.
            auto next = e.generate(head + "\xA9 hello", greedy(1), nullptr);
            REQUIRE(next.has_value());
            CHECK(next->cached_prefix_tokens >= held);
        }
    }
}

TEST_CASE("Hybrid models reuse the context up to the last prompt's state snapshot, exactly") {
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;
    EngineConfig config = engine_config(model_path("hybrid", spec));
    config.backend = BackendKind::Cpu;
    auto reused = Engine::create(config);
    auto fresh = Engine::create(config);
    REQUIRE(reused.has_value() && fresh.has_value());
    auto run = [](Engine& e, const std::string& prompt, GenerationStats* out = nullptr) {
        std::string text;
        auto st = e.generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (st && out != nullptr) *out = *st;
        return st ? text : std::string("<error>");
    };
    Engine& e = *reused.value();
    const std::string system = "hello world hello";
    REQUIRE(e.prefill(system, true).is_ok());
    const auto system_tokens = static_cast<int32_t>(e.tokenize(system, true)->size());
    const std::string turn1 = system + " world";
    const std::string reply = run(e, turn1);
    const auto turn1_tokens = static_cast<int32_t>(e.tokenize(turn1, true)->size());

    // The exact reply sent back: everything already processed is kept.
    GenerationStats st{};
    run(e, turn1 + reply + " hello", &st);
    CHECK(st.cached_prefix_tokens >= turn1_tokens);

    // An edited reply: recomputes from the snapshot at the end of turn 2's
    // prompt is impossible (it diverges earlier), so from turn 1's prompt end,
    // which the snapshot ring still holds (oldest = system, newest = last).
    // Output equals a fresh context's.
    const std::string edited = turn1 + "hello world" + " hello";
    GenerationStats st2{};
    const std::string after_edit = run(e, edited, &st2);
    CHECK(st2.cached_prefix_tokens > 0);
    fresh.value()->reset_context();
    CHECK(after_edit == run(*fresh.value(), edited));

    // A dropped turn (the text after the system prompt changes): back to the
    // system prompt's snapshot, never to an empty context.
    const std::string dropped = system + " hello";
    GenerationStats st3{};
    const std::string after_drop = run(e, dropped, &st3);
    CHECK(st3.cached_prefix_tokens == system_tokens);
    fresh.value()->reset_context();
    CHECK(after_drop == run(*fresh.value(), dropped));
}

TEST_CASE("Hybrid models keep a draft's words when its prepared whole message is typed over") {
    // Typing ahead prepares a draft's words, then the whole message as it
    // would be sent: typing on must go back to the words, not to the last
    // prompt, and the output must equal that of the same conversation
    // without drafts (the same tokens: the reply as generated).
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;
    EngineConfig config = engine_config(model_path("hybrid", spec));
    config.backend = BackendKind::Cpu;
    auto run = [](Engine& en, const std::string& prompt, GenerationStats* st = nullptr) {
        std::string text;
        auto r = en.generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (r && st != nullptr) *st = *r;
        return r ? text : std::string("<error>");
    };
    const std::string system = "hello world hello";
    const std::string turn1 = system + " world";
    auto converse = [&](Engine& en) -> std::string {  // the system prompt, a first turn and its reply
        return en.prefill(system, true).is_ok() ? run(en, turn1) : std::string("<error>");
    };
    // Plain text, then turns closed by a special token as a chat template's
    // are ("</s>" here; "<|im_end|>" on ChatML), which decodes to "".
    for (const std::string end : {"", "</s>"}) {
        auto engine = Engine::create(config);
        auto plain = Engine::create(config);
        REQUIRE(engine.has_value() && plain.has_value());
        Engine& e = *engine.value();
        const std::string head = turn1 + converse(e) + end + "hello";  // the history and the next turn's opening
        converse(*plain.value());
        const std::string words = head + " world world";
        REQUIRE(e.prefill(words, true).is_ok());
        REQUIRE(e.prefill(words + " hello" + end + "world hello", true).is_ok());  // the whole message, prepared
        const std::string typed_on = words + " world" + end + "hello";
        GenerationStats st{};
        const std::string out = run(e, typed_on, &st);
        CHECK(st.cached_prefix_tokens == static_cast<int32_t>(e.tokenize(words, true)->size()));
        CHECK(out == run(*plain.value(), typed_on));
    }
}

TEST_CASE("C API log buffer keeps whole lines for polling UIs") {
    liyab_set_log_level(1);
    liyab_log_buffer_enable(1 << 16);
    liyab_engine_config config;
    liyab_engine_config_default(&config);
    const std::string path = mixed_model();
    config.model_path = path.c_str();
    config.backend = LIYAB_BACKEND_CPU;
    liyab_engine* engine = nullptr;
    REQUIRE(liyab_engine_create(&config, &engine) == LIYAB_OK);
    liyab_engine_destroy(engine);
    std::string all;
    char chunk[64];
    int takes = 0;
    while (size_t n = liyab_log_buffer_take(chunk, sizeof chunk)) {
        CHECK(n < sizeof chunk);
        // Whole lines, or one line longer than the buffer, cut at its size.
        CHECK((chunk[n - 1] == '\n' || n == sizeof chunk - 1));
        all.append(chunk, n);
        ++takes;
    }
    CHECK(takes > 1);
    CHECK(all.find("I ") == 0);
    CHECK(all.find("mapped ") != std::string::npos);
    CHECK(liyab_log_buffer_take(chunk, sizeof chunk) == 0);
    liyab_log_buffer_enable(0);
    liyab_set_log_level(2);
}

TEST_CASE("A saved context restores in a new engine, exactly, and only for the same model") {
    test::TinyModelSpec hybrid;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    const std::string dir = test::temp_dir();
    // ... also with latent attention: its cache pages hold key rows only, values inside them.
    for (const std::string& path : {f32_model(), model_path("hybrid", hybrid), model_path("latent", latent_spec(64)),
                                   model_path("saved_gpt2", test::classic_model_spec("gpt2")),
                                   model_path("saved_bloom", test::classic_model_spec("bloom")),
                                   model_path("saved_command-r", test::classic_model_spec("command-r")),
                                   model_path("saved_nemotron", test::classic_model_spec("nemotron")),
                                   model_path("latent_combined_state", latent_spec(64, true)),
                                   model_path("compressed_state", test::compressed_spec())}) {

        EngineConfig config = engine_config(path);
        config.backend = BackendKind::Cpu;
        auto saver = Engine::create(config);
        auto loader = Engine::create(config);
        auto fresh = Engine::create(config);
        REQUIRE(saver.has_value() && loader.has_value() && fresh.has_value());
        const std::string system = "hello world hello";
        REQUIRE(saver.value()->prefill(system, true).is_ok());
        const std::string file = dir + "/liyab_" + std::to_string(test::process_id()) + "_state.bin";
        REQUIRE(saver.value()->save_state(file).is_ok());
        auto restored = loader.value()->load_state(file);
        REQUIRE(restored.has_value());
        CHECK(restored.value() == static_cast<int32_t>(saver.value()->tokenize(system, true)->size()));
        auto run = [](Engine& e, const std::string& prompt, GenerationStats* st) {
            std::string text;
            auto r = e.generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (r && st != nullptr) *st = *r;
            return r ? text : std::string("<error>");
        };
        GenerationStats st{};
        const std::string out = run(*loader.value(), system + " world", &st);
        CHECK(st.cached_prefix_tokens >= restored.value());
        CHECK(out == run(*fresh.value(), system + " world", nullptr));
    }
    // A state from another model is refused, and leaves an empty context.
    EngineConfig other = engine_config(f32_model());
    auto engine = Engine::create(other);
    REQUIRE(engine.has_value());
    auto wrong = engine.value()->load_state(dir + "/liyab_" + std::to_string(test::process_id()) + "_state.bin");  // the hybrid model's
    CHECK(!wrong.has_value());
    CHECK(wrong.status().code() == ErrorCode::InvalidArgument);
    CHECK(!engine.value()->load_state(dir + "/liyab_missing_state.bin").has_value());
    std::remove((dir + "/liyab_" + std::to_string(test::process_id()) + "_state.bin").c_str());
}

TEST_CASE("A saved hybrid context keeps its snapshots: a restored one reuses what the live one would") {
    // After a restart, the next prompt re-sends the last reply as text, which
    // may encode differently from the tokens generated (or be edited): the
    // restored context must rewind where the live one does (the last prompt's
    // snapshot, or the system prompt's for an earlier edit), never to nothing.
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;
    EngineConfig config = engine_config(model_path("hybrid", spec));
    config.backend = BackendKind::Cpu;
    auto run = [](Engine& e, const std::string& prompt, GenerationStats* st = nullptr) {
        std::string text;
        auto r = e.generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (r && st != nullptr) *st = *r;
        return r ? text : std::string("<error>");
    };
    const std::string system = "hello world hello";
    const std::string turn1 = system + " world";
    auto converse = [&](Engine& e) {  // the system prompt, then a first turn and its reply
        REQUIRE(e.prefill(system, true).is_ok());
        run(e, turn1);
    };
    auto saver = Engine::create(config);
    auto fresh = Engine::create(config);
    REQUIRE(saver.has_value() && fresh.has_value());
    converse(*saver.value());
    const auto system_tokens = static_cast<int32_t>(saver.value()->tokenize(system, true)->size());
    const std::string file = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_snapshots.bin";
    REQUIRE(saver.value()->save_state(file).is_ok());

    for (const std::string& prompt : {turn1 + "hello world" + " hello", system + " hello"}) {
        auto live = Engine::create(config);
        auto loader = Engine::create(config);
        REQUIRE(live.has_value() && loader.has_value());
        converse(*live.value());
        GenerationStats live_st{};
        run(*live.value(), prompt, &live_st);
        REQUIRE(loader.value()->load_state(file).has_value());
        GenerationStats st{};
        const std::string out = run(*loader.value(), prompt, &st);
        CHECK(st.cached_prefix_tokens == live_st.cached_prefix_tokens);
        CHECK(st.cached_prefix_tokens >= system_tokens);
        // The rewind to a restored snapshot comes before the prefill: timed apart from the first token.
        CHECK(st.context_ms > 0.0);
        CHECK(st.ttft_ms > 0.0);
        CHECK(st.ttft_stall_ms == 0.0);  // experts read in place: nothing to wait for
        fresh.value()->reset_context();
        CHECK(out == run(*fresh.value(), prompt));
    }
    std::remove(file.c_str());
}

TEST_CASE("Engine routes to CPU when forced and matches the default route") {
    EngineConfig cpu_config = engine_config(mixed_model());
    cpu_config.backend = BackendKind::Cpu;
    auto cpu_engine = Engine::create(cpu_config);
    REQUIRE(cpu_engine.has_value());
    const std::vector<int32_t> prompt = {1, 270, 300, 5};
    const auto a = run(*cpu_engine.value(), prompt, greedy(12));
    CHECK(a.size() == 12 || !a.empty());

    EngineConfig unsupported = engine_config(mixed_model());
    unsupported.backend = BackendKind::Qnn;
    auto qnn = Engine::create(unsupported);
    CHECK(!qnn.has_value());
    CHECK(qnn.status().code() == ErrorCode::Unsupported);
}

TEST_CASE("Triple-buffered block streaming produces the same tokens as in-place mmap reads") {
    const std::vector<int32_t> prompt = {1, 270, 300, 5, 290};
    test::TinyModelSpec dense_k;
    dense_k.n_embd = 256;
    dense_k.n_ff = 768;
    dense_k.n_head = 8;
    dense_k.attn_type = DType::Q5_K;
    dense_k.ffn_type = dense_k.output_type = DType::Q4_K;
    for (const std::string& path : {mixed_model(), model_path("streamed_nas", test::nas_spec()),
                                  model_path("streamed_dense_k", dense_k)}) {
    EngineConfig baseline = engine_config(path);
    baseline.backend = BackendKind::Cpu;
    auto reference = Engine::create(baseline);
    REQUIRE(reference.has_value());
    const auto expected = run(*reference.value(), prompt, greedy(20));
    for (const BackendKind backend : {BackendKind::Cpu, BackendKind::Metal, BackendKind::Vulkan}) {
        EngineConfig config = engine_config(path);
        config.backend = backend;
        auto in_place = Engine::create(config);
        if (!in_place) continue;  // Metal unavailable on this platform
        config.triple_buffer_loading = true;
        auto streamed = Engine::create(config);
        REQUIRE(streamed.has_value());
        CHECK(streamed.value()->describe().find("triple-buffered") != std::string::npos);
        const auto resident_layout = in_place.value()->weight_layout();
        const auto streamed_layout = streamed.value()->weight_layout();
        CHECK(resident_layout.file_bytes == streamed_layout.file_bytes);
        CHECK(resident_layout.streamed_bytes == 0);
        CHECK(resident_layout.streamed_blocks == 0);
        CHECK(streamed_layout.streamed_bytes > 0);
        CHECK(streamed_layout.streamed_blocks > 0);
        CHECK(streamed_layout.total_blocks == resident_layout.total_blocks);
        GenerationStats stats;
        // Shared GPU allocations keep both routes on the same kernels;
        // refilling a slot must not reuse stale weight contents. CPU and GPU
        // activation rounding differ, so compare each backend to itself.
        CHECK(run(*streamed.value(), prompt, greedy(20), &stats) ==
              run(*in_place.value(), prompt, greedy(20)));
        CHECK(!expected.empty());
    }
    }
}

TEST_CASE("Speculative decoding with greedy sampling reproduces plain greedy output") {
    const std::vector<int32_t> prompt = {1, 270, 300, 5, 290};
    for (const BackendKind backend : {BackendKind::Cpu, BackendKind::Metal, BackendKind::Vulkan}) {
        EngineConfig plain = engine_config(mixed_model());
        plain.backend = backend;
        auto base_engine = Engine::create(plain);
        if (!base_engine) {
            std::printf("  skipped %s: %s\n", backend_kind_name(backend), base_engine.status().to_string().c_str());
            continue;
        }
        const auto expected = run(*base_engine.value(), prompt, greedy(24));
        REQUIRE(!expected.empty());

        // Draft == target: every draft token is accepted.
        EngineConfig same = plain;
        same.draft_model_path = mixed_model();
        same.draft_tokens = 4;
        same.adaptive_drafts = false;  // exercise every draft, not the learned count
        auto same_engine = Engine::create(same);
        REQUIRE(same_engine.has_value());
        GenerationStats stats;
        CHECK(run(*same_engine.value(), prompt, greedy(24), &stats) == expected);
        CHECK(stats.draft_tokens_proposed > 0);
        if (backend == BackendKind::Cpu) CHECK(stats.draft_tokens_accepted == stats.draft_tokens_proposed);

        // Unrelated draft model: lower acceptance, identical output.
        test::TinyModelSpec other;
        other.seed = 99;
        other.n_layers = 1;
        EngineConfig different = plain;
        different.draft_model_path = model_path("draft", other);
        different.draft_tokens = 3;
        different.adaptive_drafts = false;  // exercise every draft, not the learned count
        auto diff_engine = Engine::create(different);
        REQUIRE(diff_engine.has_value());
        CHECK(run(*diff_engine.value(), prompt, greedy(24), &stats) == expected);
        CHECK(stats.draft_tokens_accepted < stats.draft_tokens_proposed);
    }
}

TEST_CASE("DraftBudget tries every draft count, then keeps the fastest and still explores") {
    DraftBudget budget(3);
    // First the counts never measured, in order.
    for (int32_t expected = 0; expected <= 3; ++expected) {
        const int32_t arm = budget.choose();
        CHECK(arm == expected);
        // Verification that costs a token's time per draft: plain decoding is fastest.
        budget.record(arm, arm == 0 ? 1 : 1 + arm / 2, 10.0 * (1 + arm));
    }
    // Losing counts are retried with a doubling wait: a few tries in 1000 steps.
    int32_t plain = 0, explored = 0;
    for (int32_t i = 0; i < 1000; ++i) {
        const int32_t arm = budget.choose();
        (arm == 0 ? plain : explored) += 1;
        budget.record(arm, arm == 0 ? 1 : 1 + arm / 2, 10.0 * (1 + arm));
    }
    CHECK(explored > 0);
    CHECK(explored < 30);
    // Cheap verification with good acceptance: drafting wins.
    DraftBudget cheap(3);
    for (int32_t i = 0; i < 64; ++i) {
        const int32_t arm = cheap.choose();
        cheap.record(arm, 1 + arm, 10.0 + arm);
    }
    CHECK(cheap.rate(3) > cheap.rate(0));
    int32_t next = cheap.choose();
    if (next != 3) next = cheap.choose();  // that one was an exploration step
    CHECK(next == 3);
}

TEST_CASE("Context lookup drafts the tokens that followed the longest repeated suffix") {
    std::vector<int32_t> out;
    // ... 7 8 9 | 5 ... 7 8 9: the 3-token suffix occurred once, followed by 5 6.
    SpeculativeDecoder::lookup(std::vector<int32_t>{1, 7, 8, 9, 5, 6, 2, 7, 8, 9}, 4, out);
    CHECK((out == std::vector<int32_t>{5, 6, 2, 7}));
    // The most recent occurrence wins.
    SpeculativeDecoder::lookup(std::vector<int32_t>{3, 4, 10, 3, 4, 11, 12, 3, 4}, 2, out);
    CHECK((out == std::vector<int32_t>{11, 12}));
    // A single repeated token is not enough, and nothing repeats here at all.
    SpeculativeDecoder::lookup(std::vector<int32_t>{1, 4, 2, 3, 4}, 3, out);
    CHECK(out.empty());
    SpeculativeDecoder::lookup(std::vector<int32_t>{1}, 3, out);
    CHECK(out.empty());
}

TEST_CASE("Context-lookup speculative decoding reproduces plain greedy output, also on hybrid models") {
    // A prompt that repeats itself, so lookup finds drafts.
    std::vector<int32_t> prompt = {1};
    for (int r = 0; r < 4; ++r) prompt.insert(prompt.end(), {270, 300, 5, 290, 77, 310});
    test::TinyModelSpec hybrid;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    for (const std::string& path : {mixed_model(), model_path("hybrid", hybrid), ssm_model("mamba"), ssm_model("mamba2")}) {
        EngineConfig plain = engine_config(path);
        plain.backend = BackendKind::Cpu;
        auto base_engine = Engine::create(plain);
        REQUIRE(base_engine.has_value());
        const auto expected = run(*base_engine.value(), prompt, greedy(32));
        REQUIRE(!expected.empty());

        EngineConfig lookup = plain;
        lookup.lookup_drafts = true;
        lookup.suffix_drafts = false;  // context lookup
        lookup.draft_tokens = 3;
        lookup.adaptive_drafts = false;  // exercise every draft, not the learned count
        auto engine = Engine::create(lookup);
        REQUIRE(engine.has_value());
        GenerationStats stats;
        CHECK(run(*engine.value(), prompt, greedy(32), &stats) == expected);
        CHECK(stats.draft_tokens_proposed > 0);
        // A second turn continues the same context (prefix reuse) and still matches.
        auto again = run(*engine.value(), prompt, greedy(32), &stats);
        CHECK(again == expected);
    }
}

TEST_CASE("A draft model padded to another width than the target still speculates, with the target's tokens") {
    std::vector<int32_t> prompt = {1};
    for (int r = 0; r < 4; ++r) prompt.insert(prompt.end(), {270, 300, 5, 290, 77, 310});
    for (const int32_t pad : {5, -5}) {  // the draft wider than the target, then narrower
        test::TinyModelSpec target, draft;
        (pad > 0 ? draft : target).vocab_padding = 5;
        draft.seed = 99;
        draft.n_layers = 1;
        EngineConfig plain = engine_config(model_path(pad > 0 ? "pad_target_a" : "pad_target_b", target));
        plain.backend = BackendKind::Cpu;
        auto base_engine = Engine::create(plain);
        REQUIRE(base_engine.has_value());
        const auto expected = run(*base_engine.value(), prompt, greedy(24));
        REQUIRE(!expected.empty());
        EngineConfig drafted = plain;
        drafted.draft_model_path = model_path(pad > 0 ? "pad_draft_a" : "pad_draft_b", draft);
        drafted.draft_tokens = 3;
        drafted.adaptive_drafts = false;
        auto engine = Engine::create(drafted);
        REQUIRE(engine.has_value());
        GenerationStats stats;
        CHECK(run(*engine.value(), prompt, greedy(24), &stats) == expected);
        CHECK(stats.draft_tokens_proposed > 0);
        SamplingParams sampled;  // the draft's distributions are padded or cut to the target's width
        sampled.temperature = 0.8f;
        sampled.seed = 5;
        sampled.max_tokens = 24;
        CHECK(!run(*engine.value(), prompt, sampled).empty());  // a random model may end early on its stop token
    }
}

TEST_CASE("Adaptive draft counts keep greedy output identical") {
    std::vector<int32_t> prompt = {1};
    for (int r = 0; r < 4; ++r) prompt.insert(prompt.end(), {270, 300, 5, 290, 77, 310});
    EngineConfig plain = engine_config(mixed_model());
    plain.backend = BackendKind::Cpu;
    auto base_engine = Engine::create(plain);
    REQUIRE(base_engine.has_value());
    const auto expected = run(*base_engine.value(), prompt, greedy(48));
    EngineConfig adaptive = plain;
    adaptive.lookup_drafts = true;
    adaptive.suffix_drafts = false;  // context lookup with the learned count
    adaptive.draft_tokens = 4;  // adaptive_drafts is the default
    auto engine = Engine::create(adaptive);
    REQUIRE(engine.has_value());
    CHECK(run(*engine.value(), prompt, greedy(48)) == expected);
}

TEST_CASE("Suffix arrays order every suffix on its first tokens") {
    std::mt19937 rng(7);
    for (const int32_t n : {1, 2, 17, 300, 2000}) {
        std::vector<int32_t> tokens(static_cast<size_t>(n));
        // A small alphabet makes long repeats; large ids check the dense ranking.
        for (int32_t& t : tokens) t = static_cast<int32_t>(rng() % 4) * 100000;
        for (const int32_t depth : {1, 4, SuffixDrafter::kMaxMatch}) {
            std::vector<int32_t> sa;
            SuffixDrafter::build_suffix_array(tokens, depth, sa);
            REQUIRE(sa.size() == tokens.size());
            std::vector<int32_t> sorted = sa;
            std::sort(sorted.begin(), sorted.end());
            bool permutation = true;
            for (int32_t i = 0; i < n; ++i) permutation = permutation && sorted[static_cast<size_t>(i)] == i;
            CHECK(permutation);
            auto prefix = [&](int32_t pos) {
                const int32_t end = std::min(n, pos + depth);
                return std::vector<int32_t>(tokens.begin() + pos, tokens.begin() + end);
            };
            bool ordered = true;
            for (int32_t i = 1; i < n; ++i) {
                ordered = ordered && prefix(sa[static_cast<size_t>(i) - 1]) <= prefix(sa[static_cast<size_t>(i)]);
            }
            CHECK(ordered);
        }
    }
}

TEST_CASE("The suffix drafter proposes the branch most occurrences of the longest match agree on") {
    SuffixDrafter drafter;
    // "5 6" occurs three times before the end, followed by 7 8, 7 9 and 7 8:
    // the branch is 7 8, then on a tie the newest occurrence's 4 5 6.
    drafter.sync(std::vector<int32_t>{1, 5, 6, 7, 8, 2, 5, 6, 7, 9, 3, 5, 6, 7, 8, 4, 5, 6});
    const SuffixDrafter::Draft& d = drafter.propose(10);
    CHECK(d.match == 2);
    CHECK((d.tokens == std::vector<int32_t>{7, 8, 4, 5, 6}));
    REQUIRE(d.survival.size() == d.tokens.size());
    bool decreasing = d.survival[0] > 0.0f && d.survival[0] <= 1.0f;
    for (size_t i = 1; i < d.survival.size(); ++i) decreasing = decreasing && d.survival[i] <= d.survival[i - 1];
    CHECK(decreasing);
    CHECK(drafter.propose(2).tokens.size() == 2);
    // Nothing repeats: no draft.
    drafter.sync(std::vector<int32_t>{1, 2, 3, 4, 5});
    CHECK(drafter.propose(4).tokens.empty());
}

TEST_CASE("The suffix drafter drafts from earlier sequences, counts none twice and stays bounded") {
    std::vector<int32_t> a;  // an earlier conversation
    for (int32_t i = 0; i < 64; ++i) a.push_back(1000 + i);
    std::vector<int32_t> b = {7, 7, 7};
    for (int32_t i = 0; i < 40; ++i) b.push_back(2000 + i);
    SuffixDrafter drafter;
    drafter.sync(a);
    drafter.sync(b);  // a new conversation: `a` is stored
    CHECK(drafter.stored_sequences() == 1);
    CHECK(drafter.stored_tokens() == 64);
    // The new conversation quotes `a`: the continuation comes from the store.
    std::vector<int32_t> quoting = b;
    quoting.insert(quoting.end(), {1010, 1011, 1012});
    drafter.sync(quoting);
    CHECK((drafter.propose(3).tokens == std::vector<int32_t>{1013, 1014, 1015}));
    // `a` taken up again: the live sequence contains it, so it is not matched
    // (the draft rests on the live occurrence and the one `quoting` holds).
    std::vector<int32_t> resumed = a;
    resumed.insert(resumed.end(), {1010, 1011});
    drafter.sync(resumed);  // `quoting` is stored
    CHECK(drafter.stored_sequences() == 2);
    CHECK((drafter.propose(4).tokens == std::vector<int32_t>{1012, 1013, 1014, 1015}));
    // Set aside, the resumed conversation replaces the stored one it continues.
    drafter.sync(std::vector<int32_t>{9, 9, 9});
    CHECK(drafter.stored_sequences() == 2);
    CHECK(drafter.stored_tokens() == static_cast<int64_t>(resumed.size() + quoting.size()));
    // A rollback of a few tokens stores nothing.
    std::vector<int32_t> c(100, 3);
    drafter.sync(c);
    c.resize(95);
    drafter.sync(c);
    CHECK(drafter.stored_sequences() == 2);

    // Not counted twice: with `a` stored and taken up again, "1010 1011"
    // occurred once (in the live sequence), not twice.
    SuffixDrafter single;
    single.sync(a);
    single.sync(b);
    single.sync(resumed);
    const SuffixDrafter::Draft& once = single.propose(1);
    REQUIRE(once.tokens.size() == 1);
    CHECK(once.tokens[0] == 1012);
    CHECK((once.bins[0] / 2) % 3 == 0);  // the support class of a single occurrence

    // Caps: tokens and sequences, least recently used out first.
    SuffixDrafter::Limits limits;
    limits.max_stored_tokens = 1000;
    limits.max_sequences = 4;
    SuffixDrafter small(limits);
    for (int32_t s = 0; s < 20; ++s) {
        std::vector<int32_t> seq;
        for (int32_t i = 0; i < 300; ++i) seq.push_back(s * 1000 + i);
        small.sync(seq);
        CHECK(small.stored_tokens() <= limits.max_stored_tokens);
        CHECK(small.stored_sequences() <= static_cast<size_t>(limits.max_sequences));
        CHECK(small.memory_bytes() <= (1000 * 2 + 600) * sizeof(int32_t));
    }
    // The newest stored sequence (18) is there, the oldest (0) is gone.
    small.sync(std::vector<int32_t>{5, 18000, 18001});
    CHECK(!small.propose(1).tokens.empty());
    small.sync(std::vector<int32_t>{5, 0, 1});
    CHECK(small.propose(1).tokens.empty());
    // A sequence longer than the store keeps its end.
    std::vector<int32_t> huge;
    for (int32_t i = 0; i < 5000; ++i) huge.push_back(500000 + i);
    small.sync(huge);
    small.sync(std::vector<int32_t>{1});
    CHECK(small.stored_tokens() == 1000);
    CHECK(small.stored_sequences() == 1);
    small.sync(std::vector<int32_t>{2, 504990, 504991});
    CHECK((small.propose(2).tokens == std::vector<int32_t>{504992, 504993}));
}

TEST_CASE("Suffix draft acceptance is calibrated by verification outcomes") {
    SuffixDrafter drafter;
    drafter.sync(std::vector<int32_t>{1, 2, 3, 4, 1, 2});
    const SuffixDrafter::Draft first = drafter.propose(2);
    REQUIRE(first.tokens.size() == 2);
    const float before = first.survival[0];
    for (int i = 0; i < 50; ++i) drafter.feedback(first, 2, 0);  // the first draft always rejected
    const float rejected = drafter.propose(2).survival[0];
    CHECK(rejected < before);
    CHECK(rejected < 0.1f);
    for (int i = 0; i < 400; ++i) drafter.feedback(first, 2, 2);  // then always accepted
    CHECK(drafter.propose(2).survival[0] > 0.8f);
}

TEST_CASE("DraftBudget sizes a draft by its acceptance probabilities and the measured step cost") {
    // Choices over `steps` steps for one draft's survival, with a step
    // costing `ms(j)` for j drafts and producing the expected tokens.
    auto histogram = [](DraftBudget& budget, const std::vector<float>& survival, double (*ms)(int32_t), int steps) {
        std::vector<int> counts(survival.size() + 1, 0);
        for (int i = 0; i < steps; ++i) {
            const int32_t j = budget.choose(survival);
            ++counts[static_cast<size_t>(j)];
            budget.record(j, j > 0 ? 1 + static_cast<int32_t>(survival[static_cast<size_t>(j) - 1] * j + 0.5f) : 1,
                          ms(j));
        }
        return counts;
    };
    const std::vector<float> sure = {0.99f, 0.98f, 0.97f, 0.96f, 0.95f, 0.94f, 0.93f, 0.92f};
    const std::vector<float> doubtful = {0.02f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    // Cheap verification (a dense model in RAM): sure drafts at full length,
    // doubtful ones not at all.
    DraftBudget dense(8);
    CHECK(dense.choose(sure) == 0);  // plain decoding is measured first
    dense.record(0, 1, 10.0);
    auto cheap = [](int32_t j) { return 10.0 + 0.5 * j; };
    CHECK(histogram(dense, sure, cheap, 200)[8] >= 180);
    CHECK(histogram(dense, doubtful, cheap, 200)[0] >= 180);
    CHECK(dense.cost(8) > dense.cost(0));
    // A verified token costing 55% of a step (a MoE reading each token's
    // experts): drafts that are 90% sure pay at length 3, 40% sure ones never.
    DraftBudget moe(4);
    auto dear = [](int32_t j) { return 10.0 + 5.5 * j; };
    const std::vector<float> likely = {0.9f, 0.81f, 0.729f, 0.656f};
    const std::vector<float> unlikely = {0.4f, 0.16f, 0.064f, 0.026f};
    CHECK(histogram(moe, likely, dear, 200)[3] >= 170);
    CHECK(histogram(moe, unlikely, dear, 200)[0] >= 180);
}

TEST_CASE("Suffix-draft speculative decoding: greedy and seeded sampling give plain decoding's tokens") {
    // A prompt that repeats itself, so the drafter finds drafts.
    std::vector<int32_t> prompt = {1};
    for (int r = 0; r < 4; ++r) prompt.insert(prompt.end(), {270, 300, 5, 290, 77, 310});
    const std::vector<int32_t> other = {1, 260, 261, 262, 263, 5, 290, 77};
    test::TinyModelSpec hybrid;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    SamplingParams sampled;
    sampled.temperature = 0.9f;
    sampled.seed = 1234;
    sampled.max_tokens = 40;
    for (const std::string& path : {mixed_model(), model_path("hybrid", hybrid), ssm_model("mamba"), ssm_model("mamba2")}) {
        EngineConfig plain = engine_config(path);
        plain.backend = BackendKind::Cpu;
        auto base_engine = Engine::create(plain);
        REQUIRE(base_engine.has_value());
        const auto expected = run(*base_engine.value(), prompt, greedy(40));
        const auto expected_other = run(*base_engine.value(), other, greedy(40));
        const auto expected_sampled = run(*base_engine.value(), prompt, sampled);
        REQUIRE(!expected.empty());
        REQUIRE(!expected_sampled.empty());

        for (const bool adaptive : {false, true}) {
            for (const int32_t k : {3, 6, -6}) {  // -6: 6 drafts, the best branch only (no tree)
                EngineConfig suffix = plain;
                suffix.lookup_drafts = true;  // suffix drafts: the default drafter
                suffix.draft_tokens = std::abs(k);
                suffix.draft_tree = k > 0;
                suffix.adaptive_drafts = adaptive;
                auto engine = Engine::create(suffix);
                REQUIRE(engine.has_value());
                GenerationStats stats;
                CHECK(run(*engine.value(), prompt, greedy(40), &stats) == expected);
                if (!adaptive) {
                    CHECK(stats.draft_tokens_proposed > 0);
                    // Tiny random models reject drafts too, so hybrid models
                    // go back to their recurrent-state checkpoints.
                    CHECK(stats.draft_tokens_accepted < stats.draft_tokens_proposed);
                }
                // Another conversation (the first one is stored), then the
                // first again: drafts from the store, the same tokens.
                CHECK(run(*engine.value(), other, greedy(40)) == expected_other);
                CHECK(run(*engine.value(), prompt, greedy(40), &stats) == expected);
                if (!adaptive) CHECK(stats.draft_tokens_accepted > 0);
                // Sampling: plain decoding's tokens for the seed, whatever was
                // drafted and however many drafts were verified, run after run.
                CHECK(run(*engine.value(), prompt, sampled) == expected_sampled);
                CHECK(run(*engine.value(), prompt, sampled, &stats) == expected_sampled);
                if (!adaptive) CHECK(stats.draft_tokens_proposed > 0);
            }
        }
    }
}

TEST_CASE("A speculative turn that stops partway through a step leaves the context plain decoding leaves") {
    // A step returns several tokens; when the output stops at an earlier one
    // (here max_tokens; an end of generation alike), the context must end
    // there as with plain decoding, the last token sampled but not processed.
    // Tokens past the stop once stayed in it, and the next turn's prompt
    // (after an end token that a stored conversation continued past) then
    // matched it differently than plain decoding and diverged.
    std::vector<int32_t> prompt = {1};
    for (int r = 0; r < 4; ++r) prompt.insert(prompt.end(), {270, 300, 5, 290, 77, 310});
    const std::vector<int32_t> other = {1, 260, 261, 262, 263, 5, 290, 77};
    test::TinyModelSpec hybrid;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    for (const std::string& path : {mixed_model(), model_path("hybrid", hybrid), ssm_model("mamba"), ssm_model("mamba2")}) {
        EngineConfig plain = engine_config(path);
        plain.backend = BackendKind::Cpu;
        EngineConfig suffix = plain;
        suffix.lookup_drafts = true;
        suffix.draft_tokens = 6;
        suffix.adaptive_drafts = false;
        for (int32_t n = 3; n <= 11; ++n) {
            auto reference = Engine::create(plain);
            auto engine = Engine::create(suffix);
            REQUIRE(reference.has_value());
            REQUIRE(engine.has_value());
            // The conversation once in full, then another one (the first is
            // stored), then the first again stopped at n tokens: its drafts
            // come from the stored copy and are all accepted, so steps run
            // past the stop.
            for (Engine* e : {reference.value().get(), engine.value().get()}) {
                run(*e, prompt, greedy(40));
                run(*e, other, greedy(8));
            }
            GenerationStats expected_stats, stats;
            const auto reply = run(*reference.value(), prompt, greedy(n), &expected_stats);
            CHECK(run(*engine.value(), prompt, greedy(n), &stats) == reply);
            CHECK(stats.draft_tokens_accepted > 0);
            // The next turn continues the reply: the same tokens reused, the same output.
            std::vector<int32_t> next = prompt;
            next.insert(next.end(), reply.begin(), reply.end());
            next.insert(next.end(), {260, 261, 262});
            const auto expected = run(*reference.value(), next, greedy(12), &expected_stats);
            CHECK(run(*engine.value(), next, greedy(12), &stats) == expected);
            // A hybrid model may return closer than plain decoding when the
            // prompt leaves the context (to a verification checkpoint instead
            // of the prompt's snapshot), so the reused prefix is compared on
            // the attention-only model.
            if (path == mixed_model()) CHECK(stats.cached_prefix_tokens == expected_stats.cached_prefix_tokens);
        }
    }
}

TEST_CASE("Speculative decoding works on hybrid (DeltaNet) target and draft models") {
    const std::vector<int32_t> prompt = {1, 270, 300, 5, 290};
    test::TinyModelSpec target_spec;
    target_spec.arch = "qwen35";
    target_spec.n_layers = 4;
    target_spec.delta_net_interval = 2;
    EngineConfig plain = engine_config(model_path("hybrid", target_spec));
    plain.backend = BackendKind::Cpu;
    auto base_engine = Engine::create(plain);
    REQUIRE(base_engine.has_value());
    const auto expected = run(*base_engine.value(), prompt, greedy(24));
    REQUIRE(!expected.empty());

    // A smaller hybrid draft rejects tokens, so both models roll back their recurrent states.
    test::TinyModelSpec draft_spec = target_spec;
    draft_spec.n_layers = 2;
    draft_spec.seed = 99;
    EngineConfig spec = plain;
    spec.draft_model_path = model_path("hybrid_draft", draft_spec);
    spec.draft_tokens = 3;
    spec.adaptive_drafts = false;  // exercise every draft, not the learned count
    auto engine = Engine::create(spec);
    if (!engine) std::printf("  %s\n", engine.status().to_string().c_str());
    REQUIRE(engine.has_value());
    GenerationStats stats;
    CHECK(run(*engine.value(), prompt, greedy(24), &stats) == expected);
    CHECK(stats.draft_tokens_proposed > 0);
    CHECK(stats.draft_tokens_accepted < stats.draft_tokens_proposed);
}

TEST_CASE("Speculative sampling at temperature > 0 is reproducible for a fixed seed") {
    EngineConfig config = engine_config(mixed_model());
    config.backend = BackendKind::Cpu;
    config.adaptive_drafts = false;  // a learned draft count would follow timing
    config.draft_model_path = model_path("draft", [] {
        test::TinyModelSpec s;
        s.seed = 99;
        s.n_layers = 1;
        return s;
    }());
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    SamplingParams p;
    p.temperature = 0.9f;
    p.seed = 1234;
    p.max_tokens = 20;
    const std::vector<int32_t> prompt = {1, 270, 300};
    const auto a = run(*engine.value(), prompt, p);
    const auto b = run(*engine.value(), prompt, p);
    CHECK(!a.empty());
    CHECK(a == b);
}

TEST_CASE("Engine cancellation and stop-by-callback") {
    auto engine = Engine::create(engine_config(mixed_model()));
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    int count = 0;
    auto stats = e.generate_tokens(std::vector<int32_t>{1, 270}, greedy(100), [&](std::string_view, int32_t) {
        if (++count == 3) e.cancel();
        return true;
    });
    REQUIRE(stats.has_value());
    CHECK(stats->cancelled);
    // Pieces are delivered only once they form complete UTF-8, so a callback
    // can cover several tokens; generation stops right after the cancel.
    CHECK(stats->generated_tokens >= 3 && stats->generated_tokens < 100);

    count = 0;
    stats = e.generate_tokens(std::vector<int32_t>{1, 270}, greedy(100), [&](std::string_view, int32_t) {
        return ++count < 2;
    });
    REQUIRE(stats.has_value());
    CHECK(!stats->cancelled);
    CHECK(count == 2);
    CHECK(!e.generate_tokens({}, greedy(4), {}).has_value());
}

TEST_CASE("Power pacing caps the decode rate") {
    EngineConfig config = engine_config(mixed_model());
    config.power.profile = PowerProfile::Balanced;
    config.power.target_tps = 40.0;
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    GenerationStats stats;
    run(*engine.value(), {1, 270, 300}, greedy(10), &stats);
    REQUIRE(stats.generated_tokens == 10);
    // 10 tokens span 9 inter-token gaps of 25 ms at 40 tok/s.
    CHECK(stats.decode_ms >= 9 * 25.0 * 0.95);
    CHECK(stats.paced_idle_ms > 0.0);
}

TEST_CASE("C API: create, generate, describe, errors") {
    liyab_engine_config config;
    liyab_engine_config_default(&config);
    CHECK(config.backend == LIYAB_BACKEND_AUTO);
    CHECK(config.kv_cache_type == LIYAB_KV_Q8_0);
    CHECK(config.suffix_drafts == (EngineConfig{}.suffix_drafts ? 1 : 0));
    CHECK(config.draft_tree == (EngineConfig{}.draft_tree ? 1 : 0));

    config.model_path = "/nonexistent/model.gguf";
    liyab_engine* engine = nullptr;
    CHECK(liyab_engine_create(&config, &engine) == LIYAB_ERR_IO);
    CHECK(engine == nullptr);
    CHECK(std::string(liyab_last_error()).find("nonexistent") != std::string::npos);
    CHECK(liyab_engine_create(nullptr, &engine) == LIYAB_ERR_INVALID_ARGUMENT);

    config.model_path = mixed_model().c_str();
    config.power_profile = LIYAB_POWER_PERFORMANCE;
    config.thermal_polling = 0;
    REQUIRE(liyab_engine_create(&config, &engine) == LIYAB_OK);

    liyab_sampling_params params;
    liyab_sampling_params_default(&params);
    params.temperature = 0.0f;
    params.max_tokens = 8;
    std::string text;
    auto callback = [](const char* piece, size_t len, int32_t, void* user) -> int32_t {
        static_cast<std::string*>(user)->append(piece, len);
        return 1;
    };
    liyab_generation_stats stats{};
    CHECK(liyab_engine_generate(engine, "hello world", &params, callback, &text, &stats) == LIYAB_OK);
    CHECK(stats.prompt_tokens == 3);
    CHECK(stats.generated_tokens > 0);
    liyab_engine_counters counters{};
    CHECK(liyab_engine_get_counters(engine, &counters) == LIYAB_OK);
    CHECK(counters.tokens_generated == static_cast<uint64_t>(stats.generated_tokens));
    CHECK(liyab_engine_get_counters(nullptr, &counters) == LIYAB_ERR_INVALID_ARGUMENT);

    const int32_t tokens[] = {1, 270, 300};
    CHECK(liyab_engine_generate_tokens(engine, tokens, 3, &params, nullptr, nullptr, &stats) == LIYAB_OK);
    CHECK(liyab_engine_generate(engine, nullptr, &params, nullptr, nullptr, nullptr) == LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_set_power_profile(engine, LIYAB_POWER_LOW_POWER) == LIYAB_OK);

    char buf[16];
    const size_t needed = liyab_engine_describe(engine, buf, sizeof buf);
    CHECK(needed > sizeof buf);
    CHECK(std::strlen(buf) == sizeof buf - 1);
    CHECK(liyab_describe_device(nullptr, 0) > 0);
    liyab_resource_snapshot resources{};
    CHECK(liyab_get_resource_snapshot(nullptr, nullptr) == LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_get_resource_snapshot(nullptr, &resources) == LIYAB_OK);
    CHECK(resources.cpu_fraction == 1.0f);
    CHECK(resources.slowdown == 1.0f);
    CHECK(liyab_get_resource_snapshot(engine, &resources) == LIYAB_OK);
    CHECK(resources.cpu_fraction > 0.0f && resources.cpu_fraction <= 1.0f);
    CHECK(resources.slowdown >= 1.0f);
    CHECK(std::string(liyab_version()) == LIYAB_VERSION_STRING);
    liyab_engine_destroy(engine);
    liyab_engine_destroy(nullptr);
}

// ---------------------------------------------------------------------------
// Per-block KV: windowed attention blocks keep only their window.
// ---------------------------------------------------------------------------
namespace {

// Models with windowed blocks, a window of 16 so tiny contexts leave it.
test::TinyModelSpec windowed_spec(const std::string& arch) {
    test::TinyModelSpec s;
    s.arch = arch;
    s.sliding_window = 16;
    if (arch == "gemma3") {  // one block in six global (block 5), its own RoPE table for the others
        s.n_layers = 6;
        s.qk_norm = true;
        s.post_norms = true;
        s.tied_output = true;
    } else if (arch == "gemma2") {  // every other block windowed, soft caps
        s.n_layers = 4;
        s.post_norms = true;
        s.attn_softcap = 50.0f;
        s.final_softcap = 30.0f;
    } else if (arch == "phi3") {  // every block windowed: chunks of tokens only
        s.fused_qkv = true;
        s.fused_gate_up = true;
    }
    return s;
}

const std::string& windowed_model(const std::string& arch) { return model_path("windowed_" + arch, windowed_spec(arch)); }

struct WindowedPair {
    std::unique_ptr<Transformer> per_block, every;  // the default layout, and windowed blocks keeping every position
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<Backend> cpu;
    Route route;
};

WindowedPair load_windowed(const std::string& arch, KvCacheType type = KvCacheType::Q8_0) {
    WindowedPair p;
    for (const bool keep : {false, true}) {
        auto file = MmapLoader::open(windowed_model(arch));
        if (!file) {
            CHECK(file.has_value());
            std::abort();  // the cases below cannot run without their models
        }
        TransformerOptions options;
        options.context_length = 512;
        options.max_batch = 8;
        options.kv_type = type;
        options.keep_windowed_rows = keep;
        auto model = Transformer::load(std::move(file).value(), options);
        if (!model) {
            CHECK(model.has_value());
            std::abort();
        }
        (keep ? p.every : p.per_block) = std::move(model).value();
    }
    p.pool = std::make_unique<ThreadPool>(3);
    p.cpu = make_cpu_backend(*p.pool);
    p.route = {p.cpu.get(), p.cpu.get(), p.cpu.get()};
    return p;
}

// Token t of a deterministic stream.
int32_t stream_token(int32_t t, int32_t vocab) { return 3 + (t * 37 + 11) % (vocab - 3); }

// Feeds tokens [from, to) of the stream to both models in passes of `chunk`,
// the last pass's logits compared bit for bit.
bool feed_both(WindowedPair& p, int32_t from, int32_t to, int32_t chunk) {
    const int32_t vocab = p.per_block->config().n_vocab;
    bool same = true;
    for (int32_t at = from; at < to; at += chunk) {
        std::vector<int32_t> tokens;
        for (int32_t t = at; t < std::min(to, at + chunk); ++t) tokens.push_back(stream_token(t, vocab));
        auto a = p.per_block->forward(tokens, Transformer::Logits::Last, p.route, *p.pool);
        if (!a) return false;
        const std::vector<float> la(a->begin(), a->end());
        auto b = p.every->forward(tokens, Transformer::Logits::Last, p.route, *p.pool);
        if (!b) return false;
        same = same && std::memcmp(la.data(), b->data(), la.size() * sizeof(float)) == 0;
    }
    return same;
}

}  // namespace

TEST_CASE("Windowed blocks keep their window: the same logits as keeping every position, in less memory") {
    for (const std::string arch : {"gemma3", "gemma2", "phi3"}) {
        for (const KvCacheType type : {KvCacheType::Q8_0, KvCacheType::F16}) {
            WindowedPair p = load_windowed(arch, type);
            CHECK(p.per_block->kv_cache().has_windowed_layers());
            CHECK(!p.every->kv_cache().has_windowed_layers());
            CHECK(p.per_block->has_position_state());
            // A long prompt in passes, then decoding steps: every logit the same.
            CHECK(feed_both(p, 0, 300, 32));
            CHECK(feed_both(p, 300, 340, 1));
            const size_t kept = p.per_block->kv_cache().bytes_in_use(), all = p.every->kv_cache().bytes_in_use();
            CHECK(kept < all);
            std::printf("  %s %s: %zu KiB of KV at 340 positions instead of %zu KiB\n", arch.c_str(),
                        type == KvCacheType::F16 ? "f16" : "q8_0", kept / 1024, all / 1024);
            // A rollback inside the margin (speculative verification) is exact.
            REQUIRE(p.per_block->truncate(337).is_ok());
            REQUIRE(p.every->truncate(337).is_ok());
            CHECK(feed_both(p, 337, 345, 4));
            // Further back the windowed rows are gone: refused, and the
            // context is left as it was.
            CHECK(!p.per_block->truncate(200).is_ok());
            CHECK(p.per_block->n_past() == 345);
            CHECK(p.per_block->restorable_prefix(200) == 0);
            // A snapshot's window stays: the context returns to it exactly.
            p.per_block->snapshot_state(true);
            p.every->snapshot_state(true);
            CHECK(feed_both(p, 345, 420, 16));
            CHECK(p.per_block->restorable_prefix(400) >= 345);
            REQUIRE(p.per_block->truncate(345).is_ok());
            REQUIRE(p.every->truncate(345).is_ok());
            CHECK(feed_both(p, 345, 380, 8));
        }
    }
}

TEST_CASE("Windowed blocks: a saved context restores exactly, older files convert, other layouts are refused") {
    for (const std::string arch : {"gemma3", "phi3"}) {
        WindowedPair p = load_windowed(arch);
        CHECK(feed_both(p, 0, 200, 32));
        p.per_block->snapshot_state(true);
        p.every->snapshot_state(true);
        CHECK(feed_both(p, 200, 260, 16));
        std::stringstream saved;
        REQUIRE(p.per_block->write_state(saved).is_ok());
        std::stringstream every;
        REQUIRE(p.every->write_state(every).is_ok());
        // A context's state is a fraction of the one keeping every position.
        CHECK(saved.str().size() < every.str().size());

        // Restored into another model of the same layout: it continues exactly,
        // and returns to the saved snapshot (its window came with it).
        WindowedPair q = load_windowed(arch);
        REQUIRE(q.per_block->read_state(saved).is_ok());
        CHECK(q.per_block->n_past() == 260);
        q.every = std::move(p.every);
        CHECK(feed_both(q, 260, 280, 4));
        REQUIRE(q.per_block->truncate(200).is_ok());
        REQUIRE(q.every->truncate(200).is_ok());
        CHECK(feed_both(q, 200, 230, 8));

        // Another layout (the same model keeping every position) is refused,
        // and so is this layout's file there.
        WindowedPair r = load_windowed(arch);
        every.seekg(0);
        CHECK(r.per_block->read_state(every).code() == ErrorCode::InvalidArgument);
        CHECK(r.per_block->n_past() == 0);
        saved.seekg(0);
        CHECK(r.every->read_state(saved).code() == ErrorCode::InvalidArgument);

        // A version 3 file (every block's rows of every position in one
        // table's pages, no layout or window pages): the keeping model's v4
        // file without those fields (its layout, 0, after the fingerprint, and
        // its count of window pages, 0, at the end). It converts: the windowed
        // blocks keep the window before its end.
        const std::string v4 = every.str();
        const size_t header = 4 + 4 + 9 * 8;  // magic, version, fingerprint
        std::string v3 = v4.substr(0, header) + v4.substr(header + 8, v4.size() - header - 8 - 4);
        const uint32_t three = 3;
        std::memcpy(v3.data() + 4, &three, sizeof three);
        std::stringstream old(v3);
        REQUIRE(r.per_block->read_state(old).is_ok());
        CHECK(r.per_block->n_past() == 260);
        WindowedPair fresh = load_windowed(arch);
        CHECK(feed_both(fresh, 0, 260, 64));
        // The converted context continues as one never saved.
        r.every = std::move(fresh.every);
        CHECK(feed_both(r, 260, 290, 5));
    }
}

TEST_CASE("A context saved without a window loads into a cache with the engine's own window") {
    WindowedPair p = load_windowed("gemma3");
    CHECK(feed_both(p, 0, 120, 32));
    std::stringstream saved;
    REQUIRE(p.every->write_state(saved).is_ok());
    auto file = MmapLoader::open(windowed_model("gemma3"));
    REQUIRE(file.has_value());
    TransformerOptions options;
    options.context_length = 512;
    options.max_batch = 8;
    options.keep_windowed_rows = true;
    options.sliding_window = 64;  // the engine's window: every block keeps its last 64 and the sinks
    auto sliding = Transformer::load(std::move(file).value(), options);
    REQUIRE(sliding.has_value());
    REQUIRE(sliding.value()->read_state(saved).is_ok());
    CHECK(sliding.value()->n_past() == 120);
    const std::vector<int32_t> next = {stream_token(120, sliding.value()->config().n_vocab)};
    CHECK(sliding.value()->forward(next, Transformer::Logits::Last, p.route, *p.pool).has_value());
}

TEST_CASE("Windowed models: the prefix cache brings back a replaced conversation and a snapshot, exactly") {
    for (const std::string arch : {"gemma3", "gemma2", "phi3"}) {
        EngineConfig config = engine_config(windowed_model(arch));
        config.backend = BackendKind::Cpu;
        config.prefix_cache_mb = 64;
        auto cached = Engine::create(config);
        config.prefix_cache_mb = 0;
        auto fresh = Engine::create(config);
        REQUIRE(cached.has_value() && fresh.has_value());
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        std::string a, b;
        for (int i = 0; i < 200; ++i) {  // past kMinPrefixEntry (128) tokens, and far past the window
            a += i % 3 ? " hello" : " world";
            b += i % 2 ? " world" : " hello world";
        }
        Engine& e = *cached.value();
        const std::string reply_a = run(e, a);
        run(e, b);  // another conversation: A's context goes to the prefix cache
        const std::string next_a = a + reply_a + " hello";
        GenerationStats st{};
        const std::string out = run(e, next_a, &st);
        const auto a_tokens = static_cast<int32_t>(e.tokenize(a, true)->size());
        // Its window pages came back with it: nothing of A done again.
        CHECK(st.cached_prefix_tokens == a_tokens + 5);
        CHECK(out == run(*fresh.value(), next_a));
        // An edited reply: back to A's prompt end, a snapshot whose window
        // pages were kept (from the store when the context moved on).
        run(e, b);
        const std::string edited = a + " world world hello";
        GenerationStats st2{};
        const std::string after_edit = run(e, edited, &st2);
        CHECK(st2.cached_prefix_tokens >= a_tokens - 1);
        CHECK(after_edit == run(*fresh.value(), edited));
        const Engine::PrefixCacheStats ps = e.prefix_cache_stats();
        CHECK(ps.restores >= 2);
    }
}

TEST_CASE("Windowed models: a context set aside is saved as kept, and a later engine restores it from storage") {
    for (const std::string arch : {"gemma3", "phi3"}) {
        EngineConfig config = engine_config(windowed_model(arch));
        config.backend = BackendKind::Cpu;
        auto fresh = Engine::create(config);
        REQUIRE(fresh.has_value());
        const std::string dir = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_windowed_" + arch;
        std::filesystem::remove_all(dir);
        config.prefix_cache_dir = dir;
        config.prefix_cache_disk_mb = 64;
        auto run = [](Engine& e, const std::string& p, GenerationStats* out = nullptr) {
            std::string text;
            auto st = e.generate(p, greedy(6), [&](std::string_view piece, int32_t) {
                text += piece;
                return true;
            });
            if (st && out != nullptr) *out = *st;
            return st ? text : std::string("<error>");
        };
        std::string a, b;
        for (int i = 0; i < 200; ++i) {
            a += i % 3 ? " hello" : " world";
            b += i % 2 ? " world" : " hello world";
        }
        const std::string file = dir + "_set_aside.bin";
        std::string next_a;
        {
            auto first = Engine::create(config);
            REQUIRE(first.has_value());
            Engine& e = *first.value();
            next_a = a + run(e, a) + " hello";
            run(e, b);  // A goes to the store (on storage)
            // Saved as it is kept: its rows, and its window from the window pages.
            REQUIRE(e.save_state(file, next_a).is_ok());
            auto loader = Engine::create(engine_config(windowed_model(arch)));
            REQUIRE(loader.has_value());
            REQUIRE(loader.value()->load_state(file).has_value());
            GenerationStats st{};
            const std::string out = run(*loader.value(), next_a, &st);
            CHECK(st.cached_prefix_tokens == static_cast<int32_t>(e.tokenize(a, true)->size()) + 5);
            CHECK(out == run(*fresh.value(), next_a));
        }  // destroyed: B goes to storage too
        auto later = Engine::create(config);
        REQUIRE(later.has_value());
        GenerationStats st{};
        const std::string out = run(*later.value(), next_a, &st);
        CHECK(st.cached_prefix_tokens >= static_cast<int32_t>(later.value()->tokenize(a, true)->size()));
        CHECK(out == run(*fresh.value(), next_a));
        std::remove(file.c_str());
        later.value().reset();
        std::filesystem::remove_all(dir);
    }
}

TEST_CASE("Windowed models reuse the live context up to a snapshot whose window is kept, exactly") {
    EngineConfig config = engine_config(windowed_model("gemma3"));
    config.backend = BackendKind::Cpu;
    auto reused = Engine::create(config);
    auto fresh = Engine::create(config);
    REQUIRE(reused.has_value() && fresh.has_value());
    auto run = [](Engine& e, const std::string& prompt, GenerationStats* out = nullptr) {
        std::string text;
        auto st = e.generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
            text += piece;
            return true;
        });
        if (st && out != nullptr) *out = *st;
        return st ? text : std::string("<error>");
    };
    std::string system = "system:";
    for (int i = 0; i < 120; ++i) system += i % 3 ? " hello" : " world";
    Engine& e = *reused.value();
    REQUIRE(e.prefill(system, true).is_ok());
    const auto system_tokens = static_cast<int32_t>(e.tokenize(system, true)->size());
    std::string turn1 = system;
    for (int i = 0; i < 100; ++i) turn1 += i % 2 ? " world" : " hello";
    const std::string reply = run(e, turn1);
    // A dropped turn: back to the system prompt's snapshot (its window pinned
    // while the context went on past it), never to an empty context.
    const std::string dropped = system + " hello";
    GenerationStats st{};
    const std::string after_drop = run(e, dropped, &st);
    CHECK(st.cached_prefix_tokens == system_tokens);
    CHECK(after_drop == run(*fresh.value(), dropped));
    CHECK(e.describe().find("KV blocks:") != std::string::npos);
}

TEST_MAIN()

// ---------------------------------------------------------------------------
// Tree verification: alternative branches of drafts in one pass.
// ---------------------------------------------------------------------------
TEST_CASE("Tree passes give every branch a linear pass's bits, and keep_path leaves that pass's context") {
    struct Case {
        std::string name;
        test::TinyModelSpec spec;
    };
    std::vector<Case> cases = {{"mixed", {}}, {"windowed_gemma3", windowed_spec("gemma3")},
                               {"windowed_gemma2", windowed_spec("gemma2")}, {"latent", latent_spec(64)},
                               {"latent_combined", latent_spec(64, true)}, {"compressed", test::compressed_spec()}};
    for (const std::string arch : {"gpt2", "gptneox", "bloom", "olmo", "stablelm", "command-r", "apertus"}) {
        cases.push_back({"tree_classic_" + arch, test::classic_model_spec(arch)});
    }
    {
        test::TinyModelSpec s;
        s.arch = "qwen35";
        s.n_layers = 4;
        s.delta_net_interval = 2;
        cases.push_back({"hybrid", s});
    }
    {
        test::TinyModelSpec s;
        s.arch = "qwen3moe";
        s.n_expert = 4;
        s.n_expert_used = 2;
        s.identical_experts = false;
        cases.push_back({"tree_moe", s});
    }
    cases.push_back({"tree_nas", test::nas_spec()});
    cases.push_back({"tree_hash_v4", test::deepseek4_spec()});
    {
        test::TinyModelSpec s;
        s.arch = "olmo2";
        s.post_only_norms = s.qk_norm = s.qk_norm_whole = true;
        cases.push_back({"tree_olmo2", s});
        s = {};
        s.arch = "smollm3";
        s.n_layers = 4;
        cases.push_back({"tree_nope", s});
    }
    for (const char* arch : {"mamba", "mamba2"}) {
        test::TinyModelSpec s;
        s.arch = arch;
        cases.push_back({arch, s});
    }
    for (const auto& spec : test::recurrent_model_specs()) cases.push_back({"tree_" + spec.arch, spec});
    {
        auto s = latent_spec(64);
        s.arch = "glm-dsa";
        s.index_dim = 64;
        s.index_topk = 4;
        cases.push_back({"tree_dsa", s});
    }
    // Depths 0 1 2 3 1 2 2 3 2: the first branch, then others off the root and off token 1.
    const std::vector<int32_t> parents = {-1, 0, 1, 2, 0, 4, 1, 6, 4};
    const std::vector<std::vector<int32_t>> paths = {{0, 4, 8}, {0, 1, 2, 3}, {0}, {0, 1, 6, 7}, {0, 4, 5}};
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    constexpr int32_t kPrefix = 40;  // past the windowed blocks' window of 16
    for (const Case& c : cases) {
        for (const KvCacheType type : {KvCacheType::Q8_0, KvCacheType::F16, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            std::printf("  tree pass: %s %s\n", c.name.c_str(), type == KvCacheType::F16 ? "f16" : "q8_0");
            TransformerOptions options;
            options.context_length = 256;
            options.max_batch = 16;
            options.kv_type = type;
            auto tree = load_transformer(model_path(c.name, c.spec), options);
            auto linear = load_transformer(model_path(c.name, c.spec), options);
            REQUIRE(tree != nullptr);
            REQUIRE(linear != nullptr);
            const auto vocab = static_cast<size_t>(tree->config().n_vocab);
            std::vector<int32_t> prefix, tokens;
            for (int32_t t = 0; t < kPrefix; ++t) prefix.push_back(stream_token(t, static_cast<int32_t>(vocab)));
            for (int32_t i = 0; i < 9; ++i) tokens.push_back(stream_token(100 + 7 * i, static_cast<int32_t>(vocab)));
            for (Transformer* m : {tree.get(), linear.get()}) {
                m->set_rollback_window(16);
                REQUIRE(m->forward(prefix, Transformer::Logits::None, route, pool).has_value());
                m->snapshot_state();  // every branch below starts from here
            }
            for (const auto& path : paths) {
                auto rows = tree->forward_tree(tokens, parents, route, pool);
                REQUIRE(rows.has_value());
                REQUIRE(rows->size() == tokens.size() * vocab);
                const std::vector<float> got(rows->begin(), rows->end());
                CHECK(tree->tree_pending());
                CHECK(!tree->forward(std::span<const int32_t>(tokens.data(), 1), Transformer::Logits::Last, route, pool)
                           .has_value());
                CHECK(!tree->truncate(kPrefix + 1).is_ok());
                // Each token's row: what a linear pass over its branch gives it.
                bool same = true;
                for (size_t t = 0; t < tokens.size(); ++t) {
                    std::vector<int32_t> branch;
                    for (auto a = static_cast<int32_t>(t); a >= 0; a = parents[static_cast<size_t>(a)]) {
                        branch.insert(branch.begin(), tokens[static_cast<size_t>(a)]);
                    }
                    auto ref = linear->forward(branch, Transformer::Logits::All, route, pool);
                    REQUIRE(ref.has_value());
                    same = same && std::memcmp(ref->data() + (branch.size() - 1) * vocab, got.data() + t * vocab,
                                               vocab * sizeof(float)) == 0;
                    REQUIRE(linear->truncate(kPrefix).is_ok());
                }
                CHECK(same);
                // Kept: exactly the context of the branch decoded linearly.
                CHECK(!tree->keep_path(std::vector<int32_t>{0, 2}).is_ok());  // not a branch
                REQUIRE(tree->keep_path(path).is_ok());
                CHECK(!tree->tree_pending());
                CHECK(tree->n_past() == kPrefix + static_cast<int32_t>(path.size()));
                std::vector<int32_t> branch;
                for (const int32_t i : path) branch.push_back(tokens[static_cast<size_t>(i)]);
                REQUIRE(linear->forward(branch, Transformer::Logits::None, route, pool).has_value());
                auto step_both = [&](int32_t from, int32_t count) {
                    bool equal = true;
                    for (int32_t t = from; t < from + count; ++t) {
                        const int32_t token = stream_token(300 + t, static_cast<int32_t>(vocab));
                        auto a = tree->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route, pool);
                        if (!a) return false;
                        const std::vector<float> la(a->begin(), a->end());
                        auto b = linear->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, route,
                                                 pool);
                        if (!b) return false;
                        equal = equal && std::memcmp(la.data(), b->data(), la.size() * sizeof(float)) == 0;
                    }
                    return equal;
                };
                CHECK(step_both(0, 5));
                // A rollback into the kept branch (hybrid: its checkpoints).
                const int32_t back = kPrefix + static_cast<int32_t>(path.size()) - 1;
                REQUIRE(tree->truncate(back).is_ok());
                REQUIRE(linear->truncate(back).is_ok());
                CHECK(step_both(5, 3));
                REQUIRE(tree->truncate(kPrefix).is_ok());
                REQUIRE(linear->truncate(kPrefix).is_ok());
            }
            // Parents that do not form a tree are refused, and nothing is pending.
            const std::vector<int32_t> bad = {-1, 0, 2};
            CHECK(!tree->forward_tree(std::span<const int32_t>(tokens.data(), 3), bad, route, pool).has_value());
            CHECK(!tree->tree_pending());
        }
    }
}

TEST_CASE("The suffix drafter's tree holds the alternative continuations, likeliest first") {
    SuffixDrafter drafter;
    // "5 6" occurs three times before the end, followed by 7 8 2, 7 9 3 and
    // 7 8 4: the tree is 7, then 8 (two occurrences) and 9 (one), 8's
    // children 4 (the newest) and 2.
    drafter.sync(std::vector<int32_t>{1, 5, 6, 7, 8, 2, 5, 6, 7, 9, 3, 5, 6, 7, 8, 4, 5, 6});
    const SuffixDrafter::Draft branch = drafter.propose(10);
    CHECK((branch.parents == std::vector<int32_t>{-1, 0, 1, 2, 3}));
    const SuffixDrafter::Draft& tree = drafter.propose_tree(16);
    CHECK(tree.match == 2);
    REQUIRE(tree.tokens.size() >= 5);
    REQUIRE(tree.parents.size() == tree.tokens.size());
    REQUIRE(tree.survival.size() == tree.tokens.size());
    CHECK(tree.tokens[0] == 7);
    CHECK(tree.parents[0] == -1);
    CHECK(tree.tokens[1] == 8);  // the likelier child first
    CHECK(tree.parents[1] == 0);
    CHECK(tree.survival[0] == branch.survival[0]);
    CHECK(tree.survival[1] <= branch.survival[1]);  // its siblings share the parent's acceptance
    bool ordered = true, siblings_differ = true;
    std::vector<size_t> nine;
    for (size_t i = 0; i < tree.tokens.size(); ++i) {
        ordered = ordered && tree.parents[i] < static_cast<int32_t>(i) && tree.survival[i] > 0.0f &&
                  (i == 0 || tree.survival[i] <= tree.survival[i - 1]);
        for (size_t j = 0; j < i; ++j) {
            siblings_differ = siblings_differ && !(tree.parents[j] == tree.parents[i] && tree.tokens[j] == tree.tokens[i]);
        }
        if (tree.tokens[i] == 9 && tree.parents[i] == 0) nine.push_back(i);
    }
    CHECK(ordered);
    CHECK(siblings_differ);
    REQUIRE(nine.size() == 1);
    // Up to max_tokens, a parent before its children: any first j form a tree.
    CHECK(drafter.propose_tree(1).tokens == std::vector<int32_t>{7});
    // Siblings' probabilities sum to at most their parent's.
    float under_seven = 0.0f;
    const SuffixDrafter::Draft& again = drafter.propose_tree(16);
    for (size_t i = 0; i < again.tokens.size(); ++i) {
        if (again.parents[i] == 0) under_seven += again.survival[i];
    }
    CHECK(under_seven <= again.survival[0] * 1.0001f);

    // Calibration from tree outcomes: 9 always accepted after 7, 8 never.
    const SuffixDrafter::Draft outcome = drafter.propose_tree(3);  // 7, 8, 9
    REQUIRE((outcome.tokens == std::vector<int32_t>{7, 8, 9}));
    for (int i = 0; i < 100; ++i) drafter.feedback(outcome, 3, std::vector<int32_t>{0, 2});
    const SuffixDrafter::Draft& learned = drafter.propose_tree(16);
    REQUIRE(learned.tokens.size() >= 3);
    CHECK(learned.tokens[1] == 9);  // now the likelier child
    CHECK(learned.parents[1] == 0);
    CHECK(learned.survival[1] > 0.5f * learned.survival[0]);
    for (size_t i = 2; i < learned.tokens.size(); ++i) {
        if (learned.tokens[i] == 8 && learned.parents[i] == 0) CHECK(learned.survival[i] < 0.2f * learned.survival[0]);
    }
}

TEST_CASE("Suffix drafts verified as a tree give plain decoding's tokens, and keep other branches than the likeliest") {
    struct Case {
        std::string name;
        test::TinyModelSpec spec;
    };
    std::vector<Case> cases = {{"mixed", {}}, {"windowed_gemma3", windowed_spec("gemma3")}, {"latent", latent_spec(64)},
                              {"latent_combined", latent_spec(64, true)}, {"compressed", test::compressed_spec()}};
    {
        test::TinyModelSpec s;
        s.arch = "qwen35";
        s.n_layers = 4;
        s.delta_net_interval = 2;
        cases.push_back({"hybrid", s});
    }
    {
        test::TinyModelSpec s;
        s.arch = "qwen3moe";
        s.n_expert = 4;
        s.n_expert_used = 2;
        s.identical_experts = false;
        cases.push_back({"tree_moe", s});
    }
    cases.push_back({"tree_nas", test::nas_spec()});
    cases.push_back({"tree_hash_v4", test::deepseek4_spec()});
    {
        test::TinyModelSpec s;
        s.arch = "olmo2";
        s.post_only_norms = s.qk_norm = s.qk_norm_whole = true;
        cases.push_back({"tree_olmo2", s});
        s = {};
        s.arch = "smollm3";
        s.n_layers = 4;
        cases.push_back({"tree_nope", s});
    }
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    constexpr int32_t kTokens = 48;
    for (const Case& c : cases) {
        for (const float temperature : {0.0f, 0.9f}) {
            TransformerOptions options;
            options.context_length = 512;
            options.max_batch = 16;
            auto plain = load_transformer(model_path(c.name, c.spec), options);
            auto target = load_transformer(model_path(c.name, c.spec), options);
            REQUIRE(plain != nullptr);
            REQUIRE(target != nullptr);
            const int32_t vocab = plain->config().n_vocab;
            // "x y" followed twice by w and once by alt, then once more at
            // the end, each time after another token (so "x y" is the longest
            // match): the drafter's likeliest branch is w, its sibling alt.
            // alt is the model's own greedy token after the end (a fixed
            // point: alt is part of the text it is predicted from), so greedy
            // verification keeps the sibling.
            const int32_t x = stream_token(1, vocab), y = stream_token(2, vocab), z = stream_token(3, vocab);
            auto text = [&](int32_t w, int32_t alt) {
                std::vector<int32_t> t = {1};
                for (int32_t i = 0; i < 6; ++i) t.push_back(stream_token(20 + i, vocab));
                t.insert(t.end(), {stream_token(40, vocab), x, y, w, z, stream_token(41, vocab), x, y, w, z,
                                   stream_token(42, vocab), x, y, alt, z, stream_token(43, vocab), x, y});
                return t;
            };
            std::vector<int32_t> prompt;
            for (int32_t w = 60; w < 120 && prompt.empty(); ++w) {
                int32_t alt = stream_token(80, vocab);
                for (int32_t round = 0; round < 6; ++round) {
                    plain->reset();
                    const std::vector<int32_t> t = text(stream_token(w, vocab), alt);
                    auto logits = plain->forward(t, Transformer::Logits::Last, route, pool);
                    REQUIRE(logits.has_value());
                    const int32_t g = Sampler::argmax(*logits);
                    if (g == stream_token(w, vocab)) break;
                    if (g == alt) {
                        prompt = t;
                        break;
                    }
                    alt = g;
                }
            }
            REQUIRE(!prompt.empty());  // found for every model below
            plain->reset();
            SamplingParams params;
            params.temperature = temperature;
            params.seed = 77;
            // Plain decoding, one token per pass.
            std::vector<int32_t> expected;
            {
                Sampler sampler(params);
                REQUIRE(plain->forward(std::span<const int32_t>(prompt.data(), prompt.size() - 1),
                                       Transformer::Logits::None, route, pool)
                            .has_value());
                int32_t last = prompt.back();
                std::vector<float> p;
                while (static_cast<int32_t>(expected.size()) < kTokens) {
                    auto logits = plain->forward(std::span<const int32_t>(&last, 1), Transformer::Logits::Last, route, pool);
                    REQUIRE(logits.has_value());
                    sampler.probabilities(*logits, p);
                    last = sampler.greedy() ? Sampler::argmax(p) : sampler.sample(p);
                    expected.push_back(last);
                }
            }
            // Speculative: every step verifies the drafter's tree of 8 drafts.
            target->set_rollback_window(9);
            SpeculativeDecoder decoder(*target, 8, SpeculativeDecoder::TextDrafts::Suffix);
            decoder.set_adaptive(false);
            Sampler sampler(params);
            std::vector<int32_t> context(prompt.begin(), prompt.end() - 1), out;
            REQUIRE(target->forward(context, Transformer::Logits::None, route, pool).has_value());
            int32_t last = prompt.back();
            while (static_cast<int32_t>(out.size()) < kTokens) {
                auto next = decoder.step(last, context, sampler, route, route, pool);
                REQUIRE(next.has_value());
                context.push_back(last);
                context.insert(context.end(), next->begin(), next->end() - 1);
                last = next->back();
                out.insert(out.end(), next->begin(), next->end());
                REQUIRE(target->n_past() == static_cast<int32_t>(context.size()));
            }
            out.resize(kTokens);
            if (out != expected) std::printf("  %s, temperature %.1f: not plain decoding's tokens\n", c.name.c_str(), temperature);
            CHECK(out == expected);
            CHECK(decoder.tree_steps() > 0);
            if (temperature == 0.0f) CHECK(decoder.side_branches() > 0);  // w drafted first, alt kept
            std::printf("  %s, temperature %.1f: %d drafts, %d accepted, %d tree steps, %d kept another branch\n",
                        c.name.c_str(), temperature, decoder.proposed(), decoder.accepted(), decoder.tree_steps(),
                        decoder.side_branches());
        }
    }
}

TEST_CASE("New recurrent mixers preserve checkpoints, snapshots and saved state for every KV type") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99};
    for (const auto& spec : test::recurrent_model_specs()) {
        const auto& path = model_path("state_" + spec.arch, spec);
        for (auto kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            TransformerOptions options;
            options.kv_type = kv;
            options.expert_cache_bytes = spec.n_expert > 0 ? 1 : 0;
            auto model = load_transformer(path, options);
            auto reference = load_transformer(path, options);
            REQUIRE(model && reference);
            const auto floats = model->recurrent_state_bytes() / sizeof(float);
            size_t parts = 0;
            for (auto part : model->recurrent_parts()) parts += part.size();
            CHECK(parts == floats);
            CHECK(model->context_state_bytes() >= model->recurrent_state_bytes());
            REQUIRE(model->forward(std::span(tokens).first(5), Transformer::Logits::None, route, pool));
            model->snapshot_state(true);
            std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
            REQUIRE(model->write_state(saved).is_ok());
            REQUIRE(reference->read_state(saved).is_ok());
            model->set_rollback_window(8);
            auto expected = reference->forward(std::span(tokens).subspan(5), Transformer::Logits::All, route, pool);
            REQUIRE(expected);
            const std::vector<float> rows(expected->begin(), expected->end());
            auto actual = model->forward(std::span(tokens).subspan(5), Transformer::Logits::All, route, pool);
            REQUIRE(actual);
            CHECK(max_abs_diff(*actual, rows) == 0.0);
            REQUIRE(model->truncate(7).is_ok());
            auto replay = model->forward(std::span(tokens).subspan(7), Transformer::Logits::All, route, pool);
            REQUIRE(replay);
            CHECK(max_abs_diff(*replay, std::vector<float>(rows.begin() + 2 * model->config().n_vocab, rows.end())) == 0.0);
            REQUIRE(model->truncate(5).is_ok());
            auto restored = model->forward(std::span(tokens).subspan(5), Transformer::Logits::All, route, pool);
            REQUIRE(restored);
            CHECK(max_abs_diff(*restored, rows) == 0.0);
        }
    }
}

TEST_CASE("New recurrent models reuse bounded prefix images across engines and verify suffix drafts") {
    for (const auto& spec : test::recurrent_model_specs()) {
        const auto& path = model_path("prefix_recurrent_" + spec.arch, spec);
        EngineConfig config = engine_config(path);
        config.backend = BackendKind::Cpu;
        config.expert_cache_mb = spec.n_expert > 0 ? 1 : 0;
        const std::string dir = temp_path("prefix_" + spec.arch);
        std::filesystem::remove_all(dir);
        config.prefix_cache_mb = 0;
        config.prefix_cache_disk_mb = 8;
        config.prefix_cache_dir = dir;
        auto generate = [](Engine& engine, const std::string& prompt) {
            std::string reply;
            auto status = engine.generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
                reply += piece;
                return true;
            });
            return status ? reply : std::string("<error>");
        };
        auto fresh_config = config;
        fresh_config.prefix_cache_disk_mb = 0;
        fresh_config.prefix_cache_dir.clear();
        auto reference = Engine::create(fresh_config);
        REQUIRE(reference);
        std::string a, b;
        for (int i = 0; i < 70; ++i) {
            a += "hello world hello ";
            b += "world world hello ";
        }
        const auto prompt_ids = (*reference)->tokenize(a, true);
        REQUIRE(prompt_ids && prompt_ids->size() >= 128);  // the store ignores shorter contexts
        const std::string expected = generate(**reference, a);
        REQUIRE(expected != "<error>");
        {
            auto cached = Engine::create(config);
            REQUIRE(cached);
            CHECK(generate(**cached, a) == expected);
            REQUIRE(generate(**cached, b) != "<error>");
            CHECK(generate(**cached, a) == expected);
            CHECK((*cached)->prefix_cache_stats().restores > 0);
            CHECK((*cached)->prefix_cache_stats().storage_bytes <= (uint64_t{8} << 20));
        }
        {
            auto later = Engine::create(config);
            REQUIRE(later);
            CHECK(generate(**later, a) == expected);
            CHECK((*later)->prefix_cache_stats().restores > 0);
        }
        std::filesystem::remove_all(dir);
        fresh_config.lookup_drafts = true;
        fresh_config.draft_tokens = 4;
        fresh_config.adaptive_drafts = false;
        auto drafted = Engine::create(fresh_config);
        REQUIRE(drafted);
        CHECK(generate(**drafted, a) == expected);
    }
}

TEST_CASE("New recurrent mixers route projections to available GPU backends with CPU state updates") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route reference_route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77};
    for (auto make : {make_metal_backend, make_vulkan_backend}) {
        auto gpu = make();
        if (!gpu) continue;
        const Route route{gpu->get(), gpu->get(), cpu.get()};
        for (auto spec : test::recurrent_model_specs()) {
            spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
            const auto& path = model_path("gpu_recurrent_" + spec.arch, spec);
            TransformerOptions options;
            options.kv_type = KvCacheType::F16;
            auto reference = load_transformer(path, options);
            auto model = load_transformer(path, options);
            REQUIRE(reference && model);
            auto expected = reference->forward(tokens, Transformer::Logits::All, reference_route, pool);
            REQUIRE(expected);
            const std::vector<float> rows(expected->begin(), expected->end());
            auto actual = model->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(actual);
            CHECK(max_abs_diff(*actual, rows) < 1e-3);
        }
    }
}


TEST_CASE("Compatibility plans heterogeneous local widths without padded-channel math") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    for (const int local_width : {32, 64, 24}) {
        test::TinyModelSpec spec;
        spec.n_layers = 4;
        spec.head_dim = 64;
        spec.rope_dim = 16;
        spec.swa_head_dim = local_width;
        spec.swa_rope_dim = local_width;
        spec.swa_layers = {true, false, true, false};
        spec.sliding_window = 3;
        spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
        for (const auto kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            TransformerOptions options;
            options.kv_type = kv;
            auto model = load_transformer(model_path("compat_geometry_" + std::to_string(local_width), spec), options);
            REQUIRE(model != nullptr);
            CHECK(model->config().blocks[0].head_dim == local_width);
            CHECK(model->config().blocks[1].head_dim == 64);
            CHECK(model->config().blocks[0].rope_dim == local_width);
            CHECK(model->config().blocks[1].rope_dim == 16);
            const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
            auto result = model->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(result.has_value());
            const std::vector<float> batched(result->begin(), result->end());
            const auto expected = Reference(model->file(), spec).logits(tokens, 0, 0, model->kv_cache().dtype());
            const size_t vocab = static_cast<size_t>(model->config().n_vocab);
            for (size_t t = 0; t < tokens.size(); ++t) {
                const auto row = std::span<const float>(batched).subspan(t * vocab, vocab);
                // Q4 rounding can turn tiny differences in independent F32
                // projections into different bins, then compound them through
                // four blocks. Isolated Q4 mixers have their own strict oracle
                // below; this stack checks their cache/state composition.
                if (kv == KvCacheType::F16 || kv == KvCacheType::Q8_0 || local_width == 24) {
                    CHECK(max_abs_diff(row, expected[t]) < 2e-2);
                }
            }
            // Streaming uses compact projections but the same padded images.
            model->reset();
            for (size_t t = 0; t < tokens.size(); ++t) {
                auto row = model->forward(std::span(tokens).subspan(t, 1), Transformer::Logits::Last, route, pool);
                REQUIRE(row.has_value());
                CHECK(max_abs_diff(*row, std::span<const float>(batched).subspan(t * vocab, vocab)) < 2e-4);
            }
            REQUIRE(model->truncate(7).is_ok());
            auto replay = model->forward(std::span(tokens).last(1), Transformer::Logits::Last, route, pool);
            REQUIRE(replay.has_value());
            CHECK(max_abs_diff(*replay, std::span<const float>(batched).last(vocab)) < 2e-4);
            std::stringstream saved(std::ios::in | std::ios::out | std::ios::binary);
            REQUIRE(model->write_state(saved).is_ok());
            model->reset();
            REQUIRE(model->read_state(saved).is_ok());
            CHECK(model->n_past() == static_cast<int32_t>(tokens.size()));
            std::vector<float> padding(64);
            quant::dequantize_row(model->kv_cache().dtype(), model->kv_cache().k_row(0, 7, 1), padding.data(), 64);
            for (size_t i = static_cast<size_t>(local_width); i < padding.size(); ++i) CHECK(padding[i] == 0.0f);
            if (local_width == 24) CHECK(model->kv_cache().dtype() == DType::F16);
        }
    }
}


TEST_CASE("Key calibration excludes inactive channels and uses compact heterogeneous heads") {
    KeyScalingShape compact;
    compact.blocks = 1;
    compact.n_head = 4;
    compact.n_head_kv = 2;
    compact.head_dim = 32;
    compact.rope_dim = 16;
    compact.positions = 120;
    compact.pair_speed.assign(8, 0.2f);
    KeyRanges reference(compact);
    auto padded = compact;
    padded.head_dim = 64;
    padded.rope_dim = 32;
    padded.pair_speed.resize(16, 0.0f);
    padded.head_dims = {32};
    padded.rope_dims = {16};
    KeyRanges actual(padded);
    std::mt19937 random(19);
    std::normal_distribution<float> normal;
    for (int token = 0; token < 20; ++token) {
        std::vector<float> q(128), k(64);
        for (float& v : q) v = normal(random);
        for (float& v : k) v = normal(random);
        k[19] = 100.0f;
        k[32 + 22] = 80.0f;
        reference.add(0, q.data(), k.data());
        actual.add(0, q.data(), k.data());
    }
    CHECK(actual.outlier_ratio() == reference.outlier_ratio());
    CHECK(actual.outlier_ratio() > kOutlierRatio);
    const auto wanted = reference.factors(), got = actual.factors();
    for (size_t h = 0; h < 2; ++h) {
        for (size_t i = 0; i < 32; ++i) CHECK(got[h * 64 + i] == wanted[h * 32 + i]);
        for (size_t i = 32; i < 64; ++i) CHECK(got[h * 64 + i] == 1.0f);
    }
}


TEST_CASE("Parallel dense/expert plans preserve quantized streaming and rollback with shared heterogeneous KV") {
    test::TinyModelSpec s;
    s.arch = "gemma4";
    s.n_layers = 4;
    s.head_dim = 64;
    s.swa_head_dim = 32;
    s.swa_rope_dim = 32;
    s.rope_dim = 64;
    s.sliding_window = 4;
    s.swa_layers = {true, false, true, false};
    s.shared_kv_layers = 2;
    s.values_from_keys = true;
    s.qk_norm = s.post_norms = true;
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.parallel_experts = true;
    s.identical_experts = false;
    s.fused_gate_up = true;  // fused experts, separate dense projections
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 17, 99};
    const auto& path = model_path("compat_parallel_experts", s);
    for (const auto type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        TransformerOptions options;
        options.kv_type = type;
        auto resident = load_transformer(path, options);
        options.expert_cache_bytes = 1;
        auto streamed = load_transformer(path, options);
        REQUIRE(resident && streamed);
        CHECK(resident->config().blocks[0].parallel_ffn);
        CHECK(resident->config().blocks[0].router_input_rms);
        CHECK(resident->config().blocks[0].fused_gate_up);
        CHECK(!resident->config().blocks[0].dense_fused_gate_up);
        auto all = resident->forward(tokens, Transformer::Logits::All, route, pool);
        REQUIRE(all.has_value());
        const std::vector<float> expected(all->begin(), all->end());
        const size_t vocab = static_cast<size_t>(resident->config().n_vocab);
        for (size_t t = 0; t < tokens.size(); ++t) {
            auto row = streamed->forward(std::span(tokens).subspan(t, 1), Transformer::Logits::Last, route, pool);
            REQUIRE(row.has_value());
            CHECK(max_abs_diff(*row, std::span<const float>(expected).subspan(t * vocab, vocab)) < 2e-3);
        }
        REQUIRE(streamed->truncate(10).is_ok());
        auto replay = streamed->forward(std::span(tokens).last(1), Transformer::Logits::Last, route, pool);
        REQUIRE(replay.has_value());
        CHECK(max_abs_diff(*replay, std::span<const float>(expected).last(vocab)) < 2e-3);
    }
}


TEST_CASE("Compatibility validates integer expert routes before numerical execution") {
    std::vector<int32_t> ids = {0, 2, 3, 1, 2, 2};  // duplicate routes retain their separate contributions
    TensorView table;
    table.type = DType::I32;
    table.n_dims = 2;
    table.ne = {2, 3, 1, 1};
    table.data = reinterpret_cast<const uint8_t*>(ids.data());
    table.nbytes = ids.size() * sizeof(int32_t);
    auto routed = adapt_expert_routes(table, 3, 2, 4);
    REQUIRE(routed.has_value());
    CHECK(*routed == ids);
    ids[2] = 4;
    CHECK(adapt_expert_routes(table, 3, 2, 4).status().code() == ErrorCode::InvalidModel);
    ids[2] = -1;
    CHECK(adapt_expert_routes(table, 3, 2, 4).status().code() == ErrorCode::InvalidModel);
    ids[2] = 3;
    --table.nbytes;
    CHECK(adapt_expert_routes(table, 3, 2, 4).status().code() == ErrorCode::InvalidModel);
    ++table.nbytes;
    table.ne[0] = 3;
    CHECK(adapt_expert_routes(table, 3, 2, 4).status().code() == ErrorCode::InvalidModel);
    table.ne[0] = 2;
    table.type = DType::F32;
    CHECK(adapt_expert_routes(table, 3, 2, 4).status().code() == ErrorCode::Unsupported);
}

TEST_CASE("Hash routing composes with compressed attention, sinks, mHC and streamed experts") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    for (const bool compressed : {false, true}) {
        test::TinyModelSpec s = compressed ? test::compressed_spec() : test::TinyModelSpec{};
        s.n_expert = 4;
        s.n_expert_used = 2;
        s.identical_experts = false;
        if (compressed) {
            s.attn_type = s.output_type = DType::Q8_0;
            s.ffn_type = DType::Q4_0;
            s.embd_type = DType::F16;
        }
        s.hash_layers = compressed ? 1 : 2;
        s.sqrtsoftplus_gating = true;
        s.attention_sinks = true;
        s.n_ff_shared = 128;
        s.expert_bias = true;
        s.expert_swiglu_clamp = s.shared_swiglu_clamp = 0.3f;
        // The clamped activation is a profile fact; ordinary SiLU fixtures
        // continue to exercise the established post-activation clamp.
        const auto& path = model_path(compressed ? "compat_v4_hash" : "compat_gqa_hash", s);
        for (const auto kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            TransformerOptions options;
            options.kv_type = kv;
            options.keep_windowed_rows = true;
            auto resident = load_transformer(path, options);
            options.expert_cache_bytes = 1;
            auto streamed = load_transformer(path, options);
            REQUIRE(resident && streamed);
            resident->set_rollback_window(16);
            streamed->set_rollback_window(16);
            CHECK(resident->config().blocks[0].hash_routing);
            CHECK(resident->config().moe_gating == MoeGating::SqrtSoftplus);
            const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280, 12, 17, 99, 110};
            auto batch = resident->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(batch.has_value());
            const std::vector<float> expected(batch->begin(), batch->end());
            const size_t vocab = static_cast<size_t>(resident->config().n_vocab);
            for (size_t t = 0; t < tokens.size(); ++t) {
                auto row = streamed->forward(std::span(tokens).subspan(t, 1), Transformer::Logits::Last, route, pool);
                REQUIRE(row.has_value());
                CHECK(max_abs_diff(*row, std::span<const float>(expected).subspan(t * vocab, vocab)) < 2e-3);
            }
            REQUIRE(streamed->truncate(8).is_ok());
            auto replay = streamed->forward(std::span(tokens).subspan(8), Transformer::Logits::Last, route, pool);
            REQUIRE(replay.has_value());
            CHECK(max_abs_diff(*replay, std::span<const float>(expected).last(vocab)) < 2e-3);
            std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
            REQUIRE(streamed->write_state(state).is_ok());
            streamed->reset();
            REQUIRE(streamed->read_state(state).is_ok());
            CHECK(streamed->n_past() == static_cast<int32_t>(tokens.size()));
        }
    }
}


TEST_CASE("Isolated local and global mixers match an independent Q4 reference") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    for (const int width : {24, 32, 64}) for (const bool local : {false, true}) {
        test::TinyModelSpec spec;
        spec.n_layers = 1;
        spec.head_dim = 64;
        spec.rope_dim = 16;
        spec.swa_head_dim = width;
        spec.swa_rope_dim = width;
        spec.swa_layers = {local};
        spec.sliding_window = 3;
        spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
        for (const auto type : {KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            TransformerOptions options;
            options.kv_type = type;
            auto model = load_transformer(model_path("compat_q4_oracle_" + std::to_string(width) +
                                                     (local ? "_local" : "_global"), spec), options);
            REQUIRE(model != nullptr);
            const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
            auto result = model->forward(tokens, Transformer::Logits::All, route, pool);
            REQUIRE(result.has_value());
            const auto expected = Reference(model->file(), spec).logits(tokens, 0, 0, model->kv_cache().dtype());
            const size_t vocab = static_cast<size_t>(model->config().n_vocab);
            for (size_t t = 0; t < tokens.size(); ++t) {
                CHECK(max_abs_diff(result->subspan(t * vocab, vocab), expected[t]) < 2e-2);
            }
        }
    }
}


TEST_CASE("Equal-sized hash-table edits beyond the first MiB invalidate saved contexts") {
    test::TinyModelSpec spec;
    spec.n_expert = 4;
    spec.n_expert_used = 2;
    spec.hash_layers = 1;
    spec.tokenizer_padding = (size_t{1} << 20) + 64;
    const auto& path = model_path("compat_hash_identity", spec);
    auto first = load_transformer(path);
    REQUIRE(first != nullptr);
    const std::string other_path = temp_path("compat_hash_identity_other.gguf");
    std::filesystem::copy_file(path, other_path, std::filesystem::copy_options::overwrite_existing);
    const TensorView* table = first->file().tensor("blk.0.ffn_gate_tid2eid.weight");
    REQUIRE(table != nullptr);
    {
        std::fstream output(other_path, std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(output.is_open());
        output.seekp(static_cast<std::streamoff>(table->file_offset));
        const int32_t changed = 3;
        output.write(reinterpret_cast<const char*>(&changed), sizeof(changed));
        REQUIRE(output.good());
    }
    auto other = load_transformer(other_path);
    REQUIRE(other != nullptr);
    CHECK(first->file().file_size() == other->file().file_size());
    CHECK(model_fingerprint(first->file().file()) == model_fingerprint(other->file().file()));
    CHECK(first->file_fingerprint() != other->file_fingerprint());
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300};
    REQUIRE(first->forward(tokens, Transformer::Logits::Last, route, pool).has_value());
    std::stringstream state(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(first->write_state(state).is_ok());
    CHECK(other->read_state(state).code() == ErrorCode::InvalidArgument);
    CHECK(other->n_past() == 0);
    other.reset();
    std::filesystem::remove(other_path);
}


TEST_CASE("Parallel MoE accepts per-expert scales folded into down projections") {
    test::TinyModelSpec spec;
    spec.arch = "gemma4";
    spec.n_expert = 4;
    spec.n_expert_used = 2;
    spec.parallel_experts = spec.post_norms = spec.qk_norm = true;
    spec.identical_experts = false;
    spec.attn_type = spec.ffn_type = spec.embd_type = spec.output_type = DType::F32;
    const auto& original_path = model_path("compat_separate_expert_scale", spec);
    TransformerOptions options;
    options.kv_type = KvCacheType::F16;
    auto original = load_transformer(original_path, options);
    REQUIRE(original != nullptr);
    std::ifstream input(original_path, std::ios::binary);
    std::string bytes((std::istreambuf_iterator<char>(input)), {});
    for (int32_t l = 0; l < spec.n_layers; ++l) {
        const std::string name = block_tensor(l, "ffn_down_exps.scale");
        const TensorView* scale = original->file().tensor(name);
        const TensorView* down = original->file().tensor(block_tensor(l, "ffn_down_exps.weight"));
        REQUIRE(scale && down && down->type == DType::F32);
        const size_t count = static_cast<size_t>(down->ne[0] * down->ne[1]);
        for (size_t e = 0; e < static_cast<size_t>(spec.n_expert); ++e) {
            float gain;
            std::memcpy(&gain, scale->data + e * sizeof(float), sizeof(float));
            for (size_t i = 0; i < count; ++i) {
                const size_t at = static_cast<size_t>(down->file_offset) + (e * count + i) * sizeof(float);
                float value;
                std::memcpy(&value, bytes.data() + at, sizeof(value));
                value *= gain;
                std::memcpy(bytes.data() + at, &value, sizeof(value));
            }
        }
        // Keep the file size fixed while removing the separate scale from
        // its recognized role. The renamed auxiliary tensor is unused.
        const size_t at = bytes.find(name);
        REQUIRE(at != std::string::npos);
        bytes.replace(at, name.size(), block_tensor(l, "ffn_down_exps.baked"));
    }
    const std::string folded_path = temp_path("compat_folded_expert_scale.gguf");
    {
        std::ofstream output(folded_path, std::ios::binary);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        REQUIRE(output.good());
    }
    auto folded = load_transformer(folded_path, options);
    REQUIRE(folded != nullptr);
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290, 77, 310, 280};
    auto a = original->forward(tokens, Transformer::Logits::All, route, pool);
    auto b = folded->forward(tokens, Transformer::Logits::All, route, pool);
    REQUIRE(a.has_value() && b.has_value());
    CHECK(max_abs_diff(*a, *b) < 2e-3);
    folded.reset();
    // Without the input scale the file no longer specifies the independent
    // router normalization. Reject it instead of routing the expert input.
    for (int32_t l = 0; l < spec.n_layers; ++l) {
        const std::string name = block_tensor(l, "ffn_gate_inp.scale");
        const size_t at = bytes.find(name);
        REQUIRE(at != std::string::npos);
        bytes.replace(at, name.size(), block_tensor(l, "ffn_gate_inp.baked"));
    }
    {
        std::ofstream output(folded_path, std::ios::binary);
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        REQUIRE(output.good());
    }
    {
        auto file = MmapLoader::open(folded_path);
        REQUIRE(file.has_value());
        auto config = describe_model(**file);
        REQUIRE(!config.has_value());
        CHECK(config.status().code() == ErrorCode::Unsupported);
    }
    std::filesystem::remove(folded_path);
}
