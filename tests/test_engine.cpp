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

#include "core/quant.h"
#include "core/sampling.h"
#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/transformer.h"
#include "liyab/liyab.h"
#include "liyab/liyab_c_api.h"
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

    // Greedy decoding is deterministic across calls (fresh context each time).
    std::string again;
    REQUIRE(engine.value()->generate("hello world", greedy(16), [&](std::string_view piece, int32_t) {
        again += piece;
        return true;
    }).has_value());
    CHECK(text == again);
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
        auto diff_engine = Engine::create(different);
        REQUIRE(diff_engine.has_value());
        CHECK(run(*diff_engine.value(), prompt, greedy(24), &stats) == expected);
        CHECK(stats.draft_tokens_accepted < stats.draft_tokens_proposed);
    }
}

TEST_CASE("Speculative sampling at temperature > 0 is reproducible for a fixed seed") {
    EngineConfig config = engine_config(mixed_model());
    config.backend = BackendKind::Cpu;
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
