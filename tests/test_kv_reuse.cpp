// Tests for KV reused away from its prefix (segment reuse, core/kv_reuse.h):
// rows copied at any position, keys moved to other positions for every RoPE
// variant the engine runs, blocks run at scattered positions, and the
// engine's segment reuse built on them.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/kv_cache.h"
#include "liyab/mmap_loader.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

// A tiny model written to a temporary file and loaded on the CPU backend.
struct Loaded {
    std::string path;
    std::unique_ptr<Transformer> model;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<Backend> cpu;
    Route route;
    ~Loaded() {
        model.reset();
        if (!path.empty()) std::remove(path.c_str());
    }
};

std::unique_ptr<Loaded> load(const char* name, const test::TinyModelSpec& spec, TransformerOptions options = {},
                             bool scale_keys = false) {
    auto out = std::make_unique<Loaded>();
    out->path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_reuse_" + name + ".gguf";
    test::write_tiny_model(out->path, spec);
    auto file = MmapLoader::open(out->path);
    if (!file) return nullptr;
    if (options.context_length == 0) options.context_length = 512;
    options.max_batch = 32;
    auto model = Transformer::load(std::move(file).value(), options);
    if (!model) {
        std::printf("  %s: %s\n", name, model.status().to_string().c_str());
        return nullptr;
    }
    out->model = std::move(model).value();
    out->pool = std::make_unique<ThreadPool>(3);
    out->cpu = make_cpu_backend(*out->pool);
    out->route = {out->cpu.get(), out->cpu.get(), out->cpu.get()};
    if (scale_keys) {
        const int32_t vocab = out->model->config().n_vocab;
        std::vector<int32_t> calibration = {1};
        for (int32_t i = 1; i < 48; ++i) calibration.push_back(3 + (i * 29) % (vocab - 3));
        if (!out->model->calibrate_keys(calibration, out->route, *out->pool).is_ok() ||
            out->model->key_scales().empty()) {
            std::printf("  %s: key calibration failed\n", name);
            return nullptr;
        }
    }
    return out;
}

std::vector<int32_t> tokens_of(const Transformer& model, size_t n, int32_t salt) {
    const int32_t vocab = model.config().n_vocab;
    std::vector<int32_t> t;
    for (size_t i = 0; i < n; ++i) t.push_back(3 + static_cast<int32_t>((i * 37 + static_cast<size_t>(salt) * 101) % static_cast<size_t>(vocab - 3)));
    return t;
}

std::vector<float> dequantized(const KvCache& kv, const uint8_t* row) {
    std::vector<float> out(static_cast<size_t>(kv.config().head_dim));
    quant::dequantize_row(kv.dtype(), row, out.data(), kv.config().head_dim);
    return out;
}

// Largest difference between two rows, in units of the coarsest rounding step
// either row may carry (a Q8_0 block's scale, or F16's relative precision).
double steps_apart(const KvCache& kv, const std::vector<float>& a, const std::vector<float>& b) {
    const bool quantized = quant::is_block_quantized(kv.dtype());
    double worst = 0.0;
    for (size_t b0 = 0; b0 < a.size(); b0 += quantized ? quant::kBlock : a.size()) {
        const size_t b1 = quantized ? std::min(a.size(), b0 + quant::kBlock) : a.size();
        double amax = 1e-12;
        for (size_t i = b0; i < b1; ++i) amax = std::max({amax, std::fabs(static_cast<double>(a[i])), std::fabs(static_cast<double>(b[i]))});
        const double step = quantized ? amax / (kv.dtype() == DType::Q8_0 ? 127.0 : 7.0) : amax / 1024.0;
        for (size_t i = b0; i < b1; ++i) worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]) / step);
    }
    return worst;
}

struct MoveCase {
    const char* name;
    test::TinyModelSpec spec;
    TransformerOptions options;
    bool scale_keys = false;
};

std::vector<MoveCase> move_cases() {
    std::vector<MoveCase> out;
    auto add = [&](const char* name, test::TinyModelSpec spec, TransformerOptions options = {}, bool scale = false) {
        out.push_back({name, std::move(spec), options, scale});
    };
    TransformerOptions f16;
    f16.kv_type = KvCacheType::F16;
    TransformerOptions q4;
    q4.kv_type = KvCacheType::Q4_1;
    test::TinyModelSpec s;
    add("llama", s);  // adjacent pairs
    add("llama-f16", s, f16);
    add("llama-q4_1", s, q4);
    s.rope_freqs = true;  // per-pair frequency factors, linear position scaling
    s.rope_scale = 2.0f;
    add("llama-ropefreqs-linear", s);
    s = {};
    s.arch = "qwen2";  // halves
    s.key_outlier = 60.0f;
    add("qwen2-keyscaled", s, {}, true);
    s.yarn_factor = 8.0f;
    s.yarn_ctx = 48;
    add("qwen2-yarn-keyscaled", s, {}, true);
    s = {};
    s.arch = "qwen3";
    s.qk_norm = true;
    s.rope_dim = 16;  // partial rotation
    add("qwen3-partial", s);
    s.rope_dim = 0;
    s.yarn_factor = 4.0f;  // YaRN's magnitude multiplies the rotated channels
    s.yarn_ctx = 32;
    add("qwen3-yarn", s);
    s = {};
    s.arch = "hunyuan-dense";  // Q/K norms after RoPE, with key scales
    s.qk_norm = true;
    s.k_norm_outlier = 60.0f;
    add("hunyuan-keyscaled", s, {}, true);
    s = {};
    s.arch = "gemma3";  // block 0 windowed: the windowed blocks' own RoPE table
    s.n_layers = 6;
    s.sliding_window = 8;
    s.rope_base = 1e6f;
    s.qk_norm = true;
    s.post_norms = true;
    s.tied_output = true;
    add("gemma3-windowed-table", s);
    s.swa_layers = {false, true, true, true, true, true};  // block 0 global: the global table
    add("gemma3-global-table", s);
    s = {};
    s.arch = "phi3";  // LongRoPE (long factors, attention factor), partial rotation
    s.fused_qkv = true;
    s.fused_gate_up = true;
    s.longrope_ctx = 32;
    s.rope_dim = 16;
    add("phi3-longrope-partial", s);
    s = {};
    s.arch = "deepseek2";  // latent attention: k_pe turns, the latent (the values) does not
    s.n_layers = 3;
    s.n_head_kv = 1;
    s.rope_dim = 32;
    s.mla_kv_rank = 64;
    s.k_norm_outlier = 60.0f;
    add("deepseek2-mla-keyscaled", s, {}, true);
    return out;
}

}  // namespace

TEST_CASE("Rows at any position: out of one cache and into another at other positions, bit for bit") {
    for (const bool values_in_keys : {false, true}) {
        KvCacheConfig c;
        c.n_layers = 3;
        c.n_head_kv = 2;
        c.head_dim = 64;
        c.type = KvCacheType::Q8_0;
        c.max_positions = 512;
        c.page_tokens = 16;
        if (values_in_keys) {
            c.values_in_keys = true;
            c.value_offset = 32;
        }
        auto a = KvCache::create(c);
        auto b = KvCache::create(c);
        REQUIRE(a.has_value() && b.has_value());
        KvCache& src = *a.value();
        KvCache& dst = *b.value();
        REQUIRE(src.reserve(100).is_ok());
        REQUIRE(dst.reserve(200).is_ok());
        for (int32_t layer = 0; layer < c.n_layers; ++layer) {
            for (int32_t pos = 0; pos < 100; ++pos) {
                std::vector<float> k(128), v(128);
                for (size_t i = 0; i < k.size(); ++i) {
                    k[i] = std::sin(0.3f * static_cast<float>(pos) + 0.07f * static_cast<float>(i + 13 * layer));
                    v[i] = std::cos(0.2f * static_cast<float>(pos) + 0.05f * static_cast<float>(i + 7 * layer));
                }
                src.store(layer, pos, k.data(), v.data());
            }
        }
        // Positions [9, 70) span four pages; into slots [3, 64) of a 70-slot
        // image, then to positions [131, 192) of the other cache.
        std::vector<uint8_t> image(src.rows_bytes(70), 0);
        REQUIRE(src.read_rows_at(9, 61, image.data(), 70, 3).is_ok());
        REQUIRE(dst.write_rows_at(131, image.data(), 70, 3, 61).is_ok());
        bool same = true;
        const size_t row = src.head_row_bytes();
        for (int32_t layer = 0; layer < c.n_layers; ++layer) {
            for (int32_t i = 0; i < 61; ++i) {
                for (int32_t h = 0; h < c.n_head_kv; ++h) {
                    same = same && std::memcmp(src.k_row(layer, 9 + i, h), dst.k_row(layer, 131 + i, h), row) == 0;
                    same = same && std::memcmp(src.k_row(layer, 9 + i, h),
                                               image.data() + src.image_key_offset(70, layer, 3 + i, h), row) == 0;
                    if (!values_in_keys) {
                        same = same && std::memcmp(src.v_row(layer, 9 + i, h), dst.v_row(layer, 131 + i, h), row) == 0;
                    }
                    same = same && std::memcmp(src.v_row(layer, 9 + i, h),
                                               image.data() + src.image_value_offset(70, layer, 3 + i, h),
                                               values_in_keys ? row - src.image_value_offset(70, 0, 0, 0) : row) == 0;
                }
            }
        }
        CHECK(same);
        CHECK(!src.read_rows_at(100, 20, image.data(), 70, 0).is_ok());  // into a page never mapped
        CHECK(!src.read_rows_at(0, 10, image.data(), 70, 65).is_ok());   // past the image
        CHECK(!dst.write_rows_at(250, image.data(), 70, 0, 4).is_ok());  // not mapped
    }
}

TEST_CASE("Moved keys equal the keys computed at the new positions, for every RoPE variant; values keep their bits") {
    // Block 0's keys depend on the token and its position only (its input is
    // the token's embedding), so the rows a context computes for tokens at
    // positions d + i are the truth for those computed at i and moved by d.
    for (const MoveCase& mc : move_cases()) {
        auto at_start = load(mc.name, mc.spec, mc.options, mc.scale_keys);
        REQUIRE(at_start != nullptr);
        Transformer& m = *at_start->model;
        const KvCache& kv = m.kv_cache();
        const int32_t n = 40;
        const std::vector<int32_t> doc = tokens_of(m, n, 1);
        for (const int32_t d : {37, 203}) {
            m.reset();
            REQUIRE(m.forward(doc, Transformer::Logits::None, at_start->route, *at_start->pool).has_value());
            std::vector<uint8_t> image(kv.rows_bytes(n));
            REQUIRE(kv.read_rows_at(0, n, image.data(), n, 0).is_ok());
            const std::vector<uint8_t> before = image;
            REQUIRE(m.move_keys(image.data(), n, 0, n, 0, d, at_start->pool.get()).is_ok());
            // The truth: the same tokens after d others.
            std::vector<int32_t> shifted = tokens_of(m, static_cast<size_t>(d), 2);
            shifted.insert(shifted.end(), doc.begin(), doc.end());
            m.reset();
            for (size_t i = 0; i < shifted.size(); i += 32) {
                const auto part = std::span<const int32_t>(shifted).subspan(i, std::min<size_t>(32, shifted.size() - i));
                REQUIRE(m.forward(part, Transformer::Logits::None, at_start->route, *at_start->pool).has_value());
            }
            double worst = 0.0;
            bool values_kept = true;
            const size_t row = kv.head_row_bytes();
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t g = 0; g < kv.config().n_head_kv; ++g) {
                    const uint8_t* moved = image.data() + kv.image_key_offset(n, 0, i, g);
                    worst = std::max(worst, steps_apart(kv, dequantized(kv, moved), dequantized(kv, kv.k_row(0, d + i, g))));
                    // Values (and with latent attention the latent, the
                    // unrotated part of the key row) keep their bits.
                    for (int32_t layer = 0; layer < kv.config().n_layers; ++layer) {
                        const size_t v = kv.image_value_offset(n, layer, i, g);
                        const size_t len = kv.config().values_in_keys ? row - (v - kv.image_key_offset(n, layer, i, g)) : row;
                        values_kept = values_kept && std::memcmp(image.data() + v, before.data() + v, len) == 0;
                    }
                }
            }
            if (worst > 2.5) std::printf("  %s, moved by %d: %.2f rounding steps apart\n", mc.name, d, worst);
            CHECK(worst <= 2.5);
            CHECK(values_kept);
            // Moved back: close to where it started; a move by 0 keeps every bit.
            std::vector<uint8_t> back = image;
            REQUIRE(m.move_keys(back.data(), n, 0, n, d, 0).is_ok());
            double back_worst = 0.0;
            for (int32_t i = 0; i < n; ++i) {
                for (int32_t g = 0; g < kv.config().n_head_kv; ++g) {
                    const size_t at = kv.image_key_offset(n, 0, i, g);
                    back_worst = std::max(back_worst, steps_apart(kv, dequantized(kv, back.data() + at), dequantized(kv, before.data() + at)));
                }
            }
            CHECK(back_worst <= 2.5);
            std::vector<uint8_t> still = image;
            REQUIRE(m.move_keys(still.data(), n, 0, n, 5, 5).is_ok());
            CHECK(still == image);
        }
    }
    // Refused where moving cannot be done: a hybrid model, a sliding-window cache.
    test::TinyModelSpec hybrid;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    hybrid.qk_norm = true;
    hybrid.attn_gate = true;
    auto h = load("hybrid", hybrid);
    REQUIRE(h != nullptr);
    CHECK(h->model->reuse_support().code() == ErrorCode::Unsupported);
    std::vector<uint8_t> image(h->model->kv_cache().rows_bytes(4));
    CHECK(h->model->move_keys(image.data(), 4, 0, 4, 0, 9).code() == ErrorCode::Unsupported);
    TransformerOptions window;
    window.sliding_window = 12;
    auto w = load("window", {}, window);
    REQUIRE(w != nullptr);
    CHECK(w->model->reuse_support().code() == ErrorCode::Unsupported);
}

TEST_CASE("Blocks run at scattered positions give the rows and logits a whole pass gives, bit for bit") {
    struct Arch {
        const char* name;
        test::TinyModelSpec spec;
    };
    std::vector<Arch> archs;
    archs.push_back({"llama", {}});
    test::TinyModelSpec s;
    s.arch = "qwen3";
    s.qk_norm = true;
    archs.push_back({"qwen3", s});
    s = {};
    s.arch = "gemma3";
    s.n_layers = 6;
    s.sliding_window = 8;
    s.qk_norm = true;
    s.post_norms = true;
    s.tied_output = true;
    archs.push_back({"gemma3", s});
    s = {};
    s.arch = "deepseek2";
    s.n_layers = 3;
    s.n_head_kv = 1;
    s.rope_dim = 32;
    s.mla_kv_rank = 64;
    archs.push_back({"deepseek2", s});
    s = {};
    s.arch = "qwen3moe";
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.identical_experts = false;
    archs.push_back({"qwen3moe", s});
    for (const Arch& a : archs) {
        auto l = load(a.name, a.spec);
        REQUIRE(l != nullptr);
        Transformer& m = *l->model;
        const KvCache& kv = m.kv_cache();
        const std::vector<int32_t> t = tokens_of(m, 61, 3);
        const int32_t k = 13, n = 60;
        const size_t d = static_cast<size_t>(m.config().n_embd);
        // The reference: a plain context of 60 tokens, then the 61st's logits.
        REQUIRE(m.forward(std::span(t).first(32), Transformer::Logits::None, l->route, *l->pool).has_value());
        REQUIRE(m.forward(std::span(t).subspan(32, 28), Transformer::Logits::None, l->route, *l->pool).has_value());
        std::vector<uint8_t> rows(kv.rows_bytes(n));
        REQUIRE(kv.read_rows_at(0, n, rows.data(), n, 0).is_ok());
        auto ref = m.forward(std::span(t).subspan(60, 1), Transformer::Logits::Last, l->route, *l->pool);
        REQUIRE(ref.has_value());
        const std::vector<float> want(ref->begin(), ref->end());

        // Positions [13, 60) run block by block (passes of 5) after a context of 13.
        m.reset();
        REQUIRE(m.forward(std::span(t).first(k), Transformer::Logits::None, l->route, *l->pool).has_value());
        REQUIRE(m.mutable_kv_cache().reserve(n).is_ok());
        REQUIRE(m.adopt_context(n, {}, {}, false).is_ok());
        std::vector<int32_t> positions;
        for (int32_t p = k; p < n; ++p) positions.push_back(p);
        std::vector<float> x(positions.size() * d);
        REQUIRE(m.embed(std::span(t).subspan(k, positions.size()), x).is_ok());
        for (int32_t layer = 0; layer < m.config().n_layers; ++layer) {
            REQUIRE(m.run_block_at(layer, positions, x, 5, l->route, *l->pool).is_ok());
        }
        std::vector<uint8_t> got(kv.rows_bytes(n));
        REQUIRE(kv.read_rows_at(0, n, got.data(), n, 0).is_ok());
        CHECK(got == rows);
        auto step = m.forward(std::span(t).subspan(60, 1), Transformer::Logits::Last, l->route, *l->pool);
        REQUIRE(step.has_value());
        CHECK(std::equal(want.begin(), want.end(), step->begin(), step->end()));

        // Scattered positions recomputed in a context whose other rows are
        // right: their rows come out as they were.
        m.reset();
        REQUIRE(m.forward(std::span(t).first(n), Transformer::Logits::None, l->route, *l->pool).has_value());
        positions = {14, 17, 18, 30, 44, 58};
        std::vector<int32_t> picked;
        for (const int32_t p : positions) picked.push_back(t[static_cast<size_t>(p)]);
        // Their rows spoiled first, so only the recompute can put them back.
        std::vector<uint8_t> zeros(kv.rows_bytes(1), 0);
        for (const int32_t p : positions) REQUIRE(m.mutable_kv_cache().write_rows_at(p, zeros.data(), 1, 0, 1).is_ok());
        x.assign(positions.size() * d, 0.0f);
        REQUIRE(m.embed(picked, x).is_ok());
        for (int32_t layer = 0; layer < m.config().n_layers; ++layer) {
            REQUIRE(m.run_block_at(layer, positions, x, 4, l->route, *l->pool).is_ok());
        }
        REQUIRE(kv.read_rows_at(0, n, got.data(), n, 0).is_ok());
        CHECK(got == rows);
        // Positions out of order or past the context are refused.
        CHECK(!m.run_block_at(0, std::vector<int32_t>{5, 4}, std::span<float>(x).first(2 * d), 4, l->route, *l->pool).is_ok());
        CHECK(!m.run_block_at(0, std::vector<int32_t>{n}, std::span<float>(x).first(d), 4, l->route, *l->pool).is_ok());
    }
}

TEST_MAIN()
