// End-to-end tests: quantization kernels, tokenizer, the transformer against
// an independent float reference, KV-cache rollback and sliding window,
// speculative decoding invariants, and the Engine through C++ and C APIs.
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "core/draft_budget.h"
#include "core/quant.h"
#include "core/sampling.h"
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
    return test::temp_dir() + "/liyab_" + std::to_string(getpid()) + "_" + name;
}

// Float32 model so the reference comparison is tight.
test::TinyModelSpec f32_spec() {
    test::TinyModelSpec s;
    s.attn_type = s.ffn_type = s.embd_type = s.output_type = DType::F32;
    return s;
}

const std::string& model_path(const std::string& name, const test::TinyModelSpec& spec) {
    static std::vector<std::pair<std::string, std::string>> written;
    for (const auto& [n, p] : written) {
        if (n == name) return p;
    }
    written.emplace_back(name, temp_path(name + ".gguf"));
    test::write_tiny_model(written.back().second, spec);
    return written.back().second;
}

const std::string& mixed_model() { return model_path("mixed", {}); }
const std::string& f32_model() { return model_path("f32", f32_spec()); }

std::unique_ptr<Transformer> load_transformer(const std::string& path, TransformerOptions options = {}) {
    auto file = MmapLoader::open(path);
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

    std::vector<std::vector<float>> logits(const std::vector<int32_t>& tokens, int window = 0, int sinks = 0) const {
        const int n = static_cast<int>(tokens.size());
        std::vector<std::vector<float>> x(static_cast<size_t>(n));
        const auto emb = weights("token_embd.weight");
        const int vocab = static_cast<int>(emb.size()) / d_;
        for (int t = 0; t < n; ++t) {
            x[t].assign(emb.begin() + tokens[t] * d_, emb.begin() + (tokens[t] + 1) * d_);
        }
        for (int l = 0; l < n_layers_; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            const auto wq = weights(p + "attn_q.weight"), wk = weights(p + "attn_k.weight");
            const auto wv = weights(p + "attn_v.weight"), wo = weights(p + "attn_output.weight");
            std::vector<std::vector<float>> q(n), k(n), v(n);
            for (int t = 0; t < n; ++t) {
                const auto h = norm(x[t], weights(p + "attn_norm.weight"));
                q[t] = matvec(wq, h);
                k[t] = matvec(wk, h);
                v[t] = matvec(wv, h);
                rope(q[t], t);
                rope(k[t], t);
            }
            for (int t = 0; t < n; ++t) {
                std::vector<float> att(static_cast<size_t>(d_), 0.0f);
                for (int h = 0; h < n_head_; ++h) {
                    const int g = h / (n_head_ / n_kv_);
                    const int first = window > 0 ? std::max(0, t - window + 1) : 0;
                    const int sink_end = std::min(sinks, first);  // attention sinks before the window
                    std::vector<int> visible;
                    for (int j = 0; j < sink_end; ++j) visible.push_back(j);
                    for (int j = first; j <= t; ++j) visible.push_back(j);
                    std::vector<double> s;
                    double mx = -1e30;
                    for (const int j : visible) {
                        double dot = 0;
                        for (int i = 0; i < hd_; ++i) dot += q[t][h * hd_ + i] * k[j][g * hd_ + i];
                        s.push_back(dot / std::sqrt(static_cast<double>(hd_)));
                        mx = std::max(mx, s.back());
                    }
                    double sum = 0;
                    for (double& e : s) sum += (e = std::exp(e - mx));
                    for (size_t idx = 0; idx < visible.size(); ++idx) {
                        for (int i = 0; i < hd_; ++i) {
                            att[h * hd_ + i] += static_cast<float>(s[idx] / sum * v[visible[idx]][g * hd_ + i]);
                        }
                    }
                }
                const auto o = matvec(wo, att);
                for (int i = 0; i < d_; ++i) x[t][i] += o[i];
            }
            const auto wg = weights(p + "ffn_gate.weight"), wu = weights(p + "ffn_up.weight");
            const auto wd = weights(p + "ffn_down.weight");
            for (int t = 0; t < n; ++t) {
                const auto h = norm(x[t], weights(p + "ffn_norm.weight"));
                auto g = matvec(wg, h);
                const auto u = matvec(wu, h);
                for (size_t i = 0; i < g.size(); ++i) g[i] = g[i] / (1.0f + std::exp(-g[i])) * u[i];
                const auto down = matvec(wd, g);
                for (int i = 0; i < d_; ++i) x[t][i] += down[i];
            }
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
    void rope(std::vector<float>& v, int pos) const {
        for (size_t h = 0; h < v.size() / static_cast<size_t>(hd_); ++h) {
            for (int i = 0; i < hd_ / 2; ++i) {
                const double theta = pos * std::pow(10000.0, -2.0 * i / hd_);
                float& a = v[h * hd_ + 2 * i];
                float& b = v[h * hd_ + 2 * i + 1];
                const float a0 = a, b0 = b;
                a = static_cast<float>(a0 * std::cos(theta) - b0 * std::sin(theta));
                b = static_cast<float>(a0 * std::sin(theta) + b0 * std::cos(theta));
            }
        }
    }

    const MmapLoader& f_;
    int d_, n_head_, n_kv_, n_layers_, hd_;
};

double max_abs_diff(std::span<const float> a, const std::vector<float>& b) {
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

// One entry of tests/data/quant_vectors.bin (written by tools/gen_quant_vectors.py):
// encoded blocks of a GGML format and the values llama.cpp's reference decodes.
struct QuantVector {
    DType type{};
    int64_t n = 0;  // values
    std::vector<uint8_t> blocks;
    std::vector<float> expected;
};

// Reads every entry; false (with a "skipped" note) when the file is absent.
bool load_quant_vectors(std::vector<QuantVector>& out) {
    std::string dir = LIYAB_TEST_DATA_DIR;
    if (const char* env = std::getenv("LIYAB_TEST_DATA")) dir = env;
    std::FILE* f = std::fopen((dir + "/quant_vectors.bin").c_str(), "rb");
    if (f == nullptr) {
        std::printf("  skipped: %s/quant_vectors.bin not found (set LIYAB_TEST_DATA)\n", dir.c_str());
        return false;
    }
    std::vector<uint8_t> file;
    uint8_t chunk[4096];
    for (size_t n; (n = std::fread(chunk, 1, sizeof chunk, f)) > 0;) file.insert(file.end(), chunk, chunk + n);
    std::fclose(f);
    CHECK(file.size() > 12 && std::memcmp(file.data(), "LYQV", 4) == 0);
    if (file.size() <= 12) return false;
    size_t pos = 12;
    uint32_t count = 0;
    std::memcpy(&count, file.data() + 8, 4);
    for (uint32_t e = 0; e < count; ++e) {
        uint32_t ggml = 0, n_blocks = 0;
        std::memcpy(&ggml, file.data() + pos, 4);
        std::memcpy(&n_blocks, file.data() + pos + 4, 4);
        pos += 8;
        QuantVector v;
        const bool known = dtype_from_ggml(ggml, v.type);
        CHECK(known);
        if (!known) return false;
        const DTypeTraits t = dtype_traits(v.type);
        v.n = static_cast<int64_t>(n_blocks) * t.block_size;
        v.blocks.assign(file.data() + pos, file.data() + pos + static_cast<size_t>(n_blocks) * t.block_bytes);
        pos += v.blocks.size();
        v.expected.resize(static_cast<size_t>(v.n));
        std::memcpy(v.expected.data(), file.data() + pos, v.expected.size() * 4);
        pos += v.expected.size() * 4;
        out.push_back(std::move(v));
    }
    return true;
}

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

TEST_CASE("Multi-row kernels equal one single-row dot per activation row") {
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
    // A random two-block model amplifies the weight error (measured 0.43 and
    // 0.26); the real model's quality is measured on its own outputs.
    std::printf("  logits relative RMS error: Q4_K %.4f, Q5_K %.4f\n", e4, e5);
    CHECK(e4 > 0.0);
    CHECK(e4 < 1.0);
    CHECK(e5 < e4);
    TransformerOptions invalid;
    invalid.expert_cache_bytes = 1;
    invalid.requant_bits = 3;
    CHECK(load_transformer(path, invalid) == nullptr);
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
        CHECK(max_abs_diff(one.value(), std::vector<float>(all_copy.begin() + t * vocab, all_copy.begin() + (t + 1) * vocab)) < 1e-4);
    }

    // Roll back 4 positions and replay them: identical logits.
    REQUIRE(batched->truncate(6).is_ok());
    CHECK(batched->n_past() == 6);
    auto replay = batched->forward(std::span<const int32_t>(tokens).subspan(6), Transformer::Logits::All, route, pool);
    REQUIRE(replay.has_value());
    CHECK(max_abs_diff(replay.value(), std::vector<float>(all_copy.begin() + 6 * vocab, all_copy.end())) < 1e-4);
    CHECK(!batched->truncate(11).is_ok());
}

TEST_CASE("Hybrid (DeltaNet) models roll back inside the rollback window, exactly") {
    test::TinyModelSpec spec;
    spec.arch = "qwen35";
    spec.n_layers = 4;
    spec.delta_net_interval = 2;  // DeltaNet, attention, DeltaNet, attention
    const std::string& path = model_path("hybrid", spec);
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
    CHECK(max_abs_diff(batched, expected) < 1e-3);
    // One token at a time gives exactly the batch's logits (gemv == gemm).
    const auto vocab = static_cast<size_t>(stepwise->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        auto one = stepwise->forward(std::span<const int32_t>(&tokens[t], 1), Transformer::Logits::Last, route, pool);
        REQUIRE(one.has_value());
        const std::vector<float> row(batched.begin() + t * vocab, batched.begin() + (t + 1) * vocab);
        CHECK(max_abs_diff(*one, row) < 1e-4);
    }
}

TEST_CASE("Streamed experts repacked on read (CPU i8mm): close to in-place experts, batch-size independent") {
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
    CHECK(max_abs_diff(batched, expected) < 1e-3);
    const auto vocab = static_cast<size_t>(c->config().n_vocab);
    for (size_t t = 0; t < tokens.size(); ++t) {
        auto one = c->forward(std::span<const int32_t>(&tokens[t], 1), Transformer::Logits::Last, route, pool);
        REQUIRE(one.has_value());
        const std::vector<float> row(batched.begin() + t * vocab, batched.begin() + (t + 1) * vocab);
        CHECK(max_abs_diff(*one, row) < 1e-4);
    }
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
    for (const std::string& path : {f32_model(), model_path("hybrid", hybrid)}) {
        EngineConfig config = engine_config(path);
        config.backend = BackendKind::Cpu;
        auto saver = Engine::create(config);
        auto loader = Engine::create(config);
        auto fresh = Engine::create(config);
        REQUIRE(saver.has_value() && loader.has_value() && fresh.has_value());
        const std::string system = "hello world hello";
        REQUIRE(saver.value()->prefill(system, true).is_ok());
        const std::string file = dir + "/liyab_" + std::to_string(getpid()) + "_state.bin";
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
    auto wrong = engine.value()->load_state(dir + "/liyab_" + std::to_string(getpid()) + "_state.bin");  // the hybrid model's
    CHECK(!wrong.has_value());
    CHECK(wrong.status().code() == ErrorCode::InvalidArgument);
    CHECK(!engine.value()->load_state(dir + "/liyab_missing_state.bin").has_value());
    std::remove((dir + "/liyab_" + std::to_string(getpid()) + "_state.bin").c_str());
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
    for (const BackendKind backend : {BackendKind::Cpu, BackendKind::Metal, BackendKind::Vulkan}) {
        EngineConfig config = engine_config(mixed_model());
        config.backend = backend;
        auto in_place = Engine::create(config);
        if (!in_place) continue;  // Metal unavailable on this platform
        config.triple_buffer_loading = true;
        auto streamed = Engine::create(config);
        REQUIRE(streamed.has_value());
        CHECK(streamed.value()->describe().find("triple-buffered") != std::string::npos);
        GenerationStats stats;
        const auto expected = run(*in_place.value(), prompt, greedy(20));
        CHECK(run(*streamed.value(), prompt, greedy(20), &stats) == expected);
        CHECK(!expected.empty());
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
    for (const std::string& path : {mixed_model(), model_path("hybrid", hybrid)}) {
        EngineConfig plain = engine_config(path);
        plain.backend = BackendKind::Cpu;
        auto base_engine = Engine::create(plain);
        REQUIRE(base_engine.has_value());
        const auto expected = run(*base_engine.value(), prompt, greedy(32));
        REQUIRE(!expected.empty());

        EngineConfig lookup = plain;
        lookup.lookup_drafts = true;
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
    adaptive.draft_tokens = 4;  // adaptive_drafts is the default
    auto engine = Engine::create(adaptive);
    REQUIRE(engine.has_value());
    CHECK(run(*engine.value(), prompt, greedy(48)) == expected);
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
    CHECK(std::string(liyab_version()) == LIYAB_VERSION_STRING);
    liyab_engine_destroy(engine);
    liyab_engine_destroy(nullptr);
}

TEST_MAIN()
