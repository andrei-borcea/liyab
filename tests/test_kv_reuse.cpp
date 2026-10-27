// Tests for KV reused away from its prefix (segment reuse, core/kv_reuse.h):
// rows copied at any position, keys moved to other positions for every RoPE
// variant the engine runs, blocks run at scattered positions, and the
// engine's segment reuse built on them.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <filesystem>
#include <fstream>
#include <random>

#include "core/kv_reuse.h"
#include "core/kv_store.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/tokenizer.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/engine.h"
#include "liyab/kv_cache.h"
#include "liyab/liyab_c_api.h"
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
    // Segment reuse moves every block's rows: windowed blocks keep every
    // position, as the engine has them with segment_reuse on.
    options.keep_windowed_rows = true;
    // Latent attention pinned to the absorbed form, the one run_block_at()
    // uses: LatentForm::Auto picks a form per pass shape, so a reference
    // prefill's bits would depend on how its passes were split.
    if (options.latent_form == LatentForm::Auto) options.latent_form = LatentForm::Absorbed;
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
    s.rope_layers = {false, true};  // the first block's keys must keep their bits at every position
    add("nope-f16", s, f16);
    add("nope-q8", s);
    add("nope-q4_1", s, q4);
    s = {};
    s.arch = "olmo2";
    s.post_only_norms = s.qk_norm = s.qk_norm_whole = true;
    add("olmo2-whole-qk", s, f16);
    s = test::nas_spec();
    s.qk_norm = s.qk_norm_whole = true;
    add("deci-block-geometry", s, f16);
    s = {};
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
    add("bloom-alibi", test::classic_model_spec("bloom"));
    add("mpt-alibi", test::classic_model_spec("mpt"));
    add("command-r-headnorms", test::classic_model_spec("command-r"));
    s = {};
    s.arch = "gpt-oss";
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.router_bias = s.expert_projection_bias = true;
    s.attention_sinks = s.attention_output_bias = true;
    add("gpt-oss-learned-sinks", s);
    add("gpt-oss-learned-sinks-f16", s, f16);
    s.qk_norm = true;
    s.k_norm_outlier = 60.0f;
    add("gpt-oss-learned-sinks-q4_1", s, q4, true);
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
    // Windowed blocks keeping only their window (the per-block default):
    // a segment's rows of those blocks would be gone.
    test::TinyModelSpec windowed;
    windowed.arch = "gemma3";
    windowed.n_layers = 6;
    windowed.sliding_window = 8;
    windowed.qk_norm = true;
    windowed.post_norms = true;
    windowed.tied_output = true;
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_reuse_perblock.gguf";
    test::write_tiny_model(path, windowed);
    auto file = MmapLoader::open(path);
    REQUIRE(file.has_value());
    TransformerOptions per_block;
    per_block.context_length = 512;
    auto pb = Transformer::load(std::move(file).value(), per_block);
    REQUIRE(pb.has_value());
    CHECK(pb.value()->kv_cache().has_windowed_layers());
    CHECK(pb.value()->reuse_support().code() == ErrorCode::Unsupported);
    pb.value().reset();
    std::remove(path.c_str());
}

TEST_CASE("Blocks run at scattered positions give the rows and logits a whole pass gives, bit for bit") {
    struct Arch {
        const char* name;
        test::TinyModelSpec spec;
    };
    std::vector<Arch> archs;
    archs.push_back({"llama", {}});
    test::TinyModelSpec modern;
    modern.arch = "smollm3";
    modern.n_layers = 4;
    archs.push_back({"smollm3-nope", modern});
    modern = {};
    modern.arch = "olmo2";
    modern.post_only_norms = modern.qk_norm = modern.qk_norm_whole = true;
    archs.push_back({"olmo2-post-only-whole-qk", modern});
    modern = test::nas_spec();
    archs.push_back({"deci-block-geometry", modern});
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
    s.arch = "gpt-oss";
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.attention_sinks = s.attention_output_bias = true;
    s.router_bias = s.expert_projection_bias = true;
    archs.push_back({"gpt-oss-learned-sinks", s});
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
        REQUIRE(m.adopt_context(n, {}, false).is_ok());
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

namespace {

// A store for a model's chunks of `pages` KV pages each: in RAM, or only on
// storage in `dir`.
std::unique_ptr<KvStore> store_for(const Transformer& m, const std::string& dir = {}, int32_t pages = 1) {
    KvStore::Config sc;
    const uint64_t parts[] = {m.context_fingerprint()};
    sc.seed = kv_seed_key(parts);
    sc.chunk_tokens = m.kv_cache().config().page_tokens * pages;
    sc.chunk_bytes = m.kv_cache().page_bytes() * static_cast<size_t>(pages);
    sc.ram_bytes = dir.empty() ? size_t{64} << 20 : 0;
    sc.dir = dir;
    sc.disk_bytes = dir.empty() ? 0 : size_t{64} << 20;
    return KvStore::open(sc);
}

bool prefill(Loaded& l, std::span<const int32_t> t) {
    for (size_t i = 0; i < t.size(); i += 32) {
        if (!l.model->forward(t.subspan(i, std::min<size_t>(32, t.size() - i)), Transformer::Logits::None, l.route,
                              *l.pool)) {
            return false;
        }
    }
    return true;
}

std::vector<float> last_logits(Loaded& l, int32_t token) {
    auto r = l.model->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, l.route, *l.pool);
    return r ? std::vector<float>(r->begin(), r->end()) : std::vector<float>();
}

std::vector<int32_t> join(std::initializer_list<std::vector<int32_t>> parts) {
    std::vector<int32_t> out;
    for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());
    return out;
}

}  // namespace

TEST_CASE("Reused segments: recomputing every token gives a full prefill's bits; a copy where it was computed needs none") {
    struct Arch {
        const char* name;
        test::TinyModelSpec spec;
        bool scale_keys = false;
    };
    std::vector<Arch> archs;
    archs.push_back({"llama", {}});
    archs.push_back({"deci-block-geometry", test::nas_spec()});
    {
        test::TinyModelSpec s;
        s.arch = "smollm3";
        s.n_layers = 4;
        archs.push_back({"smollm3-nope", s});
        s = {};
        s.arch = "olmo2";
        s.post_only_norms = s.qk_norm = s.qk_norm_whole = true;
        archs.push_back({"olmo2-post-only", s});
    }
    test::TinyModelSpec s;
    s.arch = "qwen3";
    s.qk_norm = true;
    s.k_norm_outlier = 60.0f;
    archs.push_back({"qwen3-keyscaled", s, true});
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
    for (const Arch& a : archs) {
      // Chunks of one page, two and four (the most a store's chunk holds): a
      // segment's pages are packed and read back page by page. Its documents
      // span 6 and 10 pages of 16 tokens, start inside a page, and end inside
      // a chunk.
      for (const int32_t pages : {1, 2, 4}) {
        auto l = load(a.name, a.spec, {}, a.scale_keys);
        REQUIRE(l != nullptr);
        Transformer& m = *l->model;
        auto store = store_for(m, {}, pages);
        REQUIRE(store != nullptr);
        const std::vector<int32_t> x = tokens_of(m, 20, 11), doc_a = tokens_of(m, 90, 12), doc_b = tokens_of(m, 150, 13);
        const std::vector<int32_t> mid = tokens_of(m, 10, 14), mid2 = tokens_of(m, 10, 15), q = tokens_of(m, 12, 16);
        // P = X A M B Q stores A and B...
        const std::vector<int32_t> p = join({x, doc_a, mid, doc_b, q});
        const std::vector<TokenSegment> p_segments = {{20, 110}, {120, 270}};
        m.reset();
        REQUIRE(prefill(*l, std::span(p).first(p.size() - 1)));
        const std::vector<float> p_logits = last_logits(*l, p.back());
        CHECK(keep_segments(m, *store, p, p_segments, 16) == 2);
        CHECK(keep_segments(m, *store, p, p_segments, 16) == 0);  // stored already
        // ...reused in P2 = X B M' A Q, in another order after another middle.
        const std::vector<int32_t> p2 = join({x, doc_b, mid2, doc_a, q});
        const std::vector<TokenSegment> p2_segments = {{20, 170}, {180, 270}};
        const auto n2 = static_cast<int32_t>(p2.size());
        m.reset();
        REQUIRE(prefill(*l, std::span(p2).first(p2.size() - 1)));
        std::vector<uint8_t> want_rows(m.kv_cache().rows_bytes(n2 - 1));
        REQUIRE(m.kv_cache().read_rows_at(0, n2 - 1, want_rows.data(), n2 - 1, 0).is_ok());
        const std::vector<float> want = last_logits(*l, p2.back());
        for (const ReuseCriterion criterion : {ReuseCriterion::Deviation, ReuseCriterion::First, ReuseCriterion::Random,
                                               ReuseCriterion::Attention, ReuseCriterion::Impact}) {
            ReuseSettings all;
            all.fraction = 1.0f;
            all.criterion = criterion;
            m.reset();
            REQUIRE(prefill(*l, x));
            auto r = reuse_segments(m, *store, p2, p2_segments, 20, n2 - 1, all, l->route, *l->pool, 7);
            REQUIRE(r.has_value());
            // The question-guided criteria run the tail (Q) with the span.
            const bool guided = criterion == ReuseCriterion::Attention || criterion == ReuseCriterion::Impact;
            CHECK(r->end == (guided ? n2 - 1 : 270));
            CHECK(r->reused == 240);
            CHECK(r->recomputed == 240);
            CHECK(r->computed == (guided ? 21 : 10));
            CHECK(r->segments == 2);
            CHECK(r->first_reused == 20);
            REQUIRE(prefill(*l, std::span(p2).subspan(static_cast<size_t>(r->end), p2.size() - 1 - static_cast<size_t>(r->end))));
            std::vector<uint8_t> rows(want_rows.size());
            REQUIRE(m.kv_cache().read_rows_at(0, n2 - 1, rows.data(), n2 - 1, 0).is_ok());
            CHECK(rows == want_rows);
            CHECK(last_logits(*l, p2.back()) == want);
        }
        // Reused where they were computed, after the same tokens: their rows
        // are the ones a full prefill writes, without recomputing any.
        for (const ReuseCriterion criterion : {ReuseCriterion::Deviation, ReuseCriterion::First}) {
            ReuseSettings none;
            none.fraction = 0.0f;
            none.criterion = criterion;
            m.reset();
            REQUIRE(prefill(*l, x));
            auto r = reuse_segments(m, *store, p, p_segments, 20, static_cast<int32_t>(p.size()) - 1, none, l->route,
                                    *l->pool, 64);
            REQUIRE(r.has_value());
            CHECK(r->recomputed == 0);
            CHECK(r->reused == 240);
            REQUIRE(prefill(*l, std::span(p).subspan(270, p.size() - 271)));
            CHECK(last_logits(*l, p.back()) == p_logits);
        }
        // A share of them elsewhere: lossy, but the context is whole and sane.
        ReuseSettings some;
        some.fraction = 0.2f;
        m.reset();
        REQUIRE(prefill(*l, x));
        auto r = reuse_segments(m, *store, p2, p2_segments, 20, n2 - 1, some, l->route, *l->pool, 64);
        REQUIRE(r.has_value());
        CHECK(r->recomputed == 48);
        CHECK(m.n_past() == 270);
        REQUIRE(prefill(*l, std::span(p2).subspan(270, p2.size() - 271)));
        const std::vector<float> lossy = last_logits(*l, p2.back());
        CHECK(!lossy.empty() && std::all_of(lossy.begin(), lossy.end(), [](float v) { return std::isfinite(v); }));
        // A prefix that already covers them: nothing done.
        m.reset();
        REQUIRE(prefill(*l, std::span(p2).first(270)));
        r = reuse_segments(m, *store, p2, p2_segments, 270, n2 - 1, some, l->route, *l->pool, 64);
        REQUIRE(r.has_value());
        CHECK(r->end == 270);
        CHECK(r->reused == 0);
      }
    }
}

TEST_CASE("Reused segments whose files are damaged are processed instead, and the output stays a full prefill's") {
    auto l = load("damaged", {});
    REQUIRE(l != nullptr);
    Transformer& m = *l->model;
    const std::string dir = test::temp_dir() + "/liyab_reuse_" + std::to_string(test::process_id()) + "_damaged";
    std::filesystem::remove_all(dir);
    const std::vector<int32_t> x = tokens_of(m, 20, 1), a = tokens_of(m, 40, 2), b = tokens_of(m, 30, 3);
    {
        auto store = store_for(m, dir);
        REQUIRE(store != nullptr && store->has_storage());
        const std::vector<int32_t> p = join({x, a, b, x});
        m.reset();
        REQUIRE(prefill(*l, p));
        CHECK(keep_segments(m, *store, p, std::vector<TokenSegment>{{20, 60}, {60, 90}}, 16) == 2);
    }
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() == ".chunk") {
            // The payload changed under an intact header: found when read.
            std::fstream f(e.path(), std::ios::in | std::ios::out | std::ios::binary);
            f.seekp(-4, std::ios::end);
            f.write("\x5a\x5a\x5a\x5a", 4);
        }
    }
    auto store = store_for(m, dir);
    REQUIRE(store != nullptr);
    const std::vector<int32_t> p2 = join({x, b, a, x});
    const std::vector<TokenSegment> segs = {{20, 50}, {50, 90}};
    REQUIRE(store->segments(a).size() == 1);
    m.reset();
    REQUIRE(prefill(*l, std::span(p2).first(p2.size() - 1)));
    const std::vector<float> want = last_logits(*l, p2.back());
    m.reset();
    REQUIRE(prefill(*l, x));
    ReuseSettings settings;
    auto r = reuse_segments(m, *store, p2, segs, 20, static_cast<int32_t>(p2.size()) - 1, settings, l->route,
                            *l->pool, 64);
    REQUIRE(r.has_value());
    CHECK(r->segments == 0);
    CHECK(r->end == 20);
    CHECK(m.n_past() == 20);
    REQUIRE(prefill(*l, std::span(p2).subspan(20, p2.size() - 21)));
    CHECK(last_logits(*l, p2.back()) == want);
    CHECK(store->segments(a).empty());  // removed with its damaged file
    store.reset();
    std::filesystem::remove_all(dir);
}

TEST_CASE("Segments are encoded on their own: the same tokens wherever they appear, BOS outside them") {
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_reuse_tok.gguf";
    test::write_tiny_model(path, {});
    auto file = MmapLoader::open(path);
    REQUIRE(file.has_value());
    auto tokenizer = Tokenizer::load(*file.value());
    REQUIRE(tokenizer.has_value());
    const std::string doc = "hello world hello hello world";
    auto encode = [&](const std::string& before, const std::string& after, std::vector<TokenSegment>& out) {
        const std::string text = before + doc + after;
        const std::vector<PromptSegment> marks = {{before.size(), before.size() + doc.size()}};
        std::vector<int32_t> tokens;
        out.clear();
        const Status s = encode_segmented(*tokenizer, text, 0, marks, true, tokens, out);
        return s.is_ok() ? tokens : std::vector<int32_t>();
    };
    std::vector<TokenSegment> a, b, c;
    const std::vector<int32_t> ta = encode("world: ", " hello", a);
    const std::vector<int32_t> tb = encode("hello hello world, then ", ".", b);
    const std::vector<int32_t> tc = encode("", " world", c);
    REQUIRE(a.size() == 1 && b.size() == 1 && c.size() == 1);
    const std::vector<int32_t> da(ta.begin() + a[0].begin, ta.begin() + a[0].end);
    const std::vector<int32_t> db(tb.begin() + b[0].begin, tb.begin() + b[0].end);
    const std::vector<int32_t> dc(tc.begin() + c[0].begin, tc.begin() + c[0].end);
    CHECK(!da.empty());
    CHECK(da == db);
    // BOS before a segment that starts the text, not in it. After it (a
    // special token) a SentencePiece vocabulary starts the segment with the
    // space it adds at the start of a text, as encoding the whole text does.
    CHECK(tc.front() == tokenizer->bos());
    CHECK(c[0].begin == 1);
    CHECK(dc == tokenizer->encode_continuation(doc, tokenizer->bos()).value());
    CHECK(da == tokenizer->encode_continuation(doc, ta[static_cast<size_t>(a[0].begin) - 1]).value());
    // Known text (the context's) is kept; a segment inside it is not marked again.
    std::vector<int32_t> tokens = {tokenizer->bos()};
    std::vector<TokenSegment> out;
    const std::string text = "hello " + doc;
    CHECK(encode_segmented(*tokenizer, text, 6, std::vector<PromptSegment>{{0, 3}, {6, text.size()}}, true, tokens, out)
              .is_ok());
    CHECK(out.size() == 1);
    // Ranges that overlap, go backwards, leave the text, cut a character or are empty.
    const std::string accented = "caf\xc3\xa9 hello world";
    const std::vector<std::vector<PromptSegment>> bad = {{{0, 5}, {3, 8}}, {{6, 9}, {0, 3}}, {{2, 40}}, {{4, 9}}, {{5, 5}}};
    for (const auto& marks : bad) {
        tokens.clear();
        CHECK(encode_segmented(*tokenizer, accented, 0, marks, true, tokens, out).code() == ErrorCode::InvalidArgument);
    }
    std::remove(path.c_str());
}

namespace {

EngineConfig reuse_config(const std::string& path) {
    EngineConfig c;
    c.model_path = path;
    c.n_threads = 3;
    c.thermal_polling = false;
    c.power.profile = PowerProfile::Performance;
    c.backend = BackendKind::Cpu;
    c.context_length = 8192;  // the calibration's text is about 3,000 byte tokens of the tiny vocabulary
    c.prefix_cache_mb = 64;
    c.segment_reuse = true;
    c.segment_reuse_max_kl = 1000.0f;  // a random model: the first share on the grid meets it
    // Kept for the test process: the first engine calibrates in the
    // background, the next ones read the result.
    c.cache_dir = test::temp_dir() + "/liyab_reuse_" + std::to_string(test::process_id()) + "_calibrations";
    return c;
}

SamplingParams greedy(int32_t max_tokens) {
    SamplingParams p;
    p.temperature = 0.0f;
    p.max_tokens = max_tokens;
    return p;
}

// A RAG-shaped prompt of the tiny vocabulary and its documents' byte ranges.
struct Marked {
    std::string text;
    std::vector<PromptSegment> segments;
};
Marked rag(const std::vector<std::string>& docs, const std::string& question) {
    Marked m;
    m.text = "system: hello world answer from the documents:";
    for (const std::string& d : docs) {
        m.text += "\n";
        m.segments.push_back({m.text.size(), m.text.size() + d.size()});
        m.text += d;
    }
    m.text += "\nquestion: " + question;
    return m;
}

std::string doc_text(int seed, int words) {
    static const char* kWords[] = {"hello", "world", "low", "held", "word", "or", "he", "we", "do", "owl"};
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::string out;
    for (int i = 0; i < words; ++i) out += std::string(i ? " " : "") + kWords[rng() % 10];
    return out;
}

std::string run(Engine& e, const std::string& prompt, const std::vector<PromptSegment>& segments = {},
                GenerationStats* stats = nullptr, int32_t tokens = 6) {
    std::string out;
    GenerateOptions options;
    options.segments = segments;
    auto st = e.generate(
        prompt, greedy(tokens),
        [&](std::string_view piece, int32_t) {
            out += piece;
            return true;
        },
        options);
    if (!st) return "<" + st.status().to_string() + ">";
    if (stats != nullptr) *stats = *st;
    return out;
}

// The background calibration's end (the engine idle meanwhile), within
// minutes; the info as it is then.
Engine::SegmentReuseInfo wait_calibrated(Engine& e) {
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    Engine::SegmentReuseInfo info = e.segment_reuse();
    while (info.calibrating && std::chrono::steady_clock::now() < give_up) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        info = e.segment_reuse();
    }
    return info;
}

const std::string& reuse_model() {
    static const std::string path = [] {
        const std::string p = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_reuse_engine.gguf";
        test::TinyModelSpec spec;
        spec.n_layers = 4;
        test::write_tiny_model(p, spec);
        return p;
    }();
    return path;
}

}  // namespace

TEST_CASE("Engine: segments are kept, reused elsewhere, and a request that marks none gets the exact output") {
    auto engine = Engine::create(reuse_config(reuse_model()));
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    const Engine::SegmentReuseInfo info = wait_calibrated(e);
    CHECK(info.status.is_ok());
    CHECK(!info.calibrating);
    CHECK(info.progress == 1.0f);
    CHECK(info.fraction == kReuseFractions[0]);
    CHECK(info.bound >= info.kl);
    EngineConfig plain = reuse_config(reuse_model());
    plain.prefix_cache_mb = 0;
    plain.segment_reuse = false;
    auto fresh_engine = Engine::create(plain);
    REQUIRE(fresh_engine.has_value());
    Engine& fresh = *fresh_engine.value();
    auto exact = [&](const std::string& prompt) {
        fresh.reset_context();
        return run(fresh, prompt);
    };
    const std::string a = doc_text(1, 30), b = doc_text(2, 40), c = doc_text(3, 25);
    // First seen: nothing to reuse; both kept.
    const Marked r1 = rag({a, b}, "which one");
    GenerationStats st{};
    run(e, r1.text, r1.segments, &st);
    CHECK(st.segment_reused_tokens == 0);
    Engine::PrefixCacheStats pc = e.prefix_cache_stats();
    CHECK(pc.segments_kept == 2);
    CHECK(pc.segments == 2);
    // In another order, after another document: both reused, part of them again.
    const Marked r2 = rag({c, b, a}, "and now");
    const std::string reused_reply = run(e, r2.text, r2.segments, &st);
    CHECK(st.segment_reused_tokens > 100);
    CHECK(st.segment_recomputed_tokens > 0);
    CHECK(st.segment_recomputed_tokens < st.segment_reused_tokens / 4);
    pc = e.prefix_cache_stats();
    CHECK(pc.segment_reuses == 2);
    CHECK(pc.segments_kept == 3);  // the new document is kept too
    CHECK(pc.segment_reused_tokens == static_cast<uint64_t>(st.segment_reused_tokens));
    CHECK(reused_reply.find('<') == std::string::npos);
    // The same prompt unmarked: the reused rows are processed again, so the
    // reply is a fresh context's, and so are the next unmarked turns.
    CHECK(run(e, r2.text) == exact(r2.text));
    CHECK(run(e, r2.text + " hello") == exact(r2.text + " hello"));
    // A prompt the context holds whole needs no segment: exact reuse wins.
    run(e, r2.text, r2.segments, &st);
    CHECK(st.segment_reused_tokens == 0);
    // Marked in another order: reused; an unmarked request diverging after
    // it keeps none of the reused rows.
    const Marked r3 = rag({a, c, b}, "again");
    run(e, r3.text, r3.segments, &st);
    CHECK(st.segment_reused_tokens > 0);
    const std::string next = r3.text + " world hello";
    CHECK(run(e, next, {}, &st) == exact(next));
    // Kept: the tokens before the first reused row (A's), no more.
    const size_t before_a = e.tokenize(r3.text.substr(0, r3.segments[0].begin), true)->size() + 2;
    CHECK(st.cached_prefix_tokens > 0);
    CHECK(static_cast<size_t>(st.cached_prefix_tokens) <= before_a);
}

TEST_CASE("Engine: a segment longer than two chunks of several pages is kept and reused whole") {
    auto engine = Engine::create(reuse_config(reuse_model()));
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    REQUIRE(wait_calibrated(e).status.is_ok());
    EngineConfig plain = reuse_config(reuse_model());
    plain.prefix_cache_mb = 0;
    plain.segment_reuse = false;
    auto fresh_engine = Engine::create(plain);
    REQUIRE(fresh_engine.has_value());
    // The store's chunks hold several pages (up to 256 tokens); the long
    // document needs more than two of them and starts inside a page.
    const int32_t chunk = e.prefix_cache_stats().chunk_tokens;
    REQUIRE(chunk > 0);
    const std::string long_doc = doc_text(10, 3 * chunk), short_doc = doc_text(11, 30);
    const auto long_tokens = e.tokenize(long_doc, false);
    REQUIRE(long_tokens.has_value());
    REQUIRE(long_tokens->size() > static_cast<size_t>(2 * chunk));
    const Marked first = rag({long_doc, short_doc}, "first");
    GenerationStats st{};
    run(e, first.text, first.segments, &st);
    CHECK(e.prefix_cache_stats().segments_kept == 2);
    // After the short one, in another position: every chunk of the long one
    // is read back, its pages turned to their new positions.
    const Marked second = rag({short_doc, long_doc}, "second");
    const std::string reply = run(e, second.text, second.segments, &st);
    CHECK(reply.find('<') == std::string::npos);
    CHECK(st.segment_reused_tokens > 2 * chunk);
    CHECK(st.segment_recomputed_tokens > 0);
    // Unmarked, the same prompt gets a fresh context's reply.
    fresh_engine.value()->reset_context();
    CHECK(run(e, second.text) == run(*fresh_engine.value(), second.text));
}

TEST_CASE("Engine: the prefix cache and saved contexts keep only exact rows of a context that reused segments") {
    const std::string dir = test::temp_dir() + "/liyab_reuse_" + std::to_string(test::process_id()) + "_state";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    auto engine = Engine::create(reuse_config(reuse_model()));
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    REQUIRE(wait_calibrated(e).status.is_ok());
    EngineConfig plain = reuse_config(reuse_model());
    plain.prefix_cache_mb = 0;
    plain.segment_reuse = false;
    auto fresh_engine = Engine::create(plain);
    REQUIRE(fresh_engine.has_value());
    auto exact = [&](const std::string& prompt) {
        fresh_engine.value()->reset_context();
        return run(*fresh_engine.value(), prompt);
    };
    const std::string a = doc_text(4, 60), b = doc_text(5, 70), c = doc_text(6, 50);
    const Marked first = rag({a, b}, "first");
    run(e, first.text, first.segments);
    const Marked mixed = rag({c, b, a}, "mixed");
    GenerationStats st{};
    run(e, mixed.text, mixed.segments, &st);
    CHECK(st.segment_reused_tokens > 0);
    // Saved with its reused rows: restored elsewhere, an unmarked request
    // processes them again.
    const std::string state = dir + "/reused.state";
    REQUIRE(e.save_state(state).is_ok());
    auto other = Engine::create(reuse_config(reuse_model()));
    REQUIRE(other.has_value());
    REQUIRE(other.value()->load_state(state).has_value());
    // (The reply is left out: re-encoding a reply need not give back the
    // tokens generated, which only a context that keeps them reuses.)
    const std::string next = mixed.text + " hello world";
    CHECK(run(*other.value(), next, {}, &st) == exact(next));
    // C was processed (not stored yet), B reused first: rows up to B's are kept.
    // (Counted on the whole text's encoding: the segments' own may differ by
    // a token at a boundary.)
    const size_t before_b = e.tokenize(mixed.text.substr(0, mixed.segments[1].begin), true)->size() + 2;
    CHECK(static_cast<size_t>(st.cached_prefix_tokens) <= before_b);
    CHECK(static_cast<size_t>(st.cached_prefix_tokens) > before_b - 12);
    // Another conversation replaces it (an unmarked one): the prefix cache
    // keeps its exact part only, and coming back stays exact.
    const std::string unrelated = "world world hello: " + doc_text(9, 90);
    run(e, unrelated);
    CHECK(run(e, next, {}, &st) == exact(next));
    CHECK(static_cast<size_t>(st.cached_prefix_tokens) <= before_b);
    std::filesystem::remove_all(dir);
}

TEST_CASE("Engine: segment reuse off, refused with a reason, or never meeting its target leaves marked prompts exact") {
    const std::string dir = test::temp_dir() + "/liyab_reuse_" + std::to_string(test::process_id()) + "_refused";
    std::filesystem::remove_all(dir);
    EngineConfig plain = reuse_config(reuse_model());
    plain.prefix_cache_mb = 0;
    plain.segment_reuse = false;
    auto fresh_engine = Engine::create(plain);
    REQUIRE(fresh_engine.has_value());
    const Marked m = rag({doc_text(7, 40), doc_text(8, 40)}, "why");
    const std::string want = run(*fresh_engine.value(), m.text);
    test::TinyModelSpec hybrid;
    hybrid.arch = "qwen35";
    hybrid.n_layers = 4;
    hybrid.delta_net_interval = 2;
    const std::string hybrid_path =
        test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_reuse_hybrid.gguf";
    test::write_tiny_model(hybrid_path, hybrid);
    struct Refusal {
        EngineConfig config;
        const char* why;
    };
    std::vector<Refusal> refusals;
    refusals.push_back({plain, "off"});
    EngineConfig no_cache = reuse_config(reuse_model());
    no_cache.prefix_cache_mb = 0;
    refusals.push_back({no_cache, "prefix cache"});
    EngineConfig strict = reuse_config(reuse_model());
    strict.segment_reuse_max_kl = 1e-9f;
    strict.cache_dir = dir;
    refusals.push_back({strict, "above the target"});
    refusals.push_back({strict, "above the target"});  // the same result, read from cache_dir
    refusals.push_back({reuse_config(hybrid_path), "hybrid"});
    for (size_t i = 0; i < refusals.size(); ++i) {
        auto engine = Engine::create(refusals[i].config);
        REQUIRE(engine.has_value());
        // The strict target, measured for the first time: calibrating in the
        // background, the marked prompt processed as usual meanwhile.
        if (i == 2) {
            const Engine::SegmentReuseInfo pending = engine.value()->segment_reuse();
            CHECK(pending.calibrating);
            CHECK(pending.status.code() == ErrorCode::Busy);
            GenerationStats st{};
            CHECK(run(*engine.value(), m.text, m.segments, &st) == want);
            CHECK(st.segment_reused_tokens == 0);
        }
        const Engine::SegmentReuseInfo info = wait_calibrated(*engine.value());
        CHECK(!info.calibrating);
        CHECK(info.status.code() == ErrorCode::Unsupported);
        CHECK(info.status.message().find(refusals[i].why) != std::string::npos);
        if (i == 3) CHECK(info.from_cache);
        GenerationStats st{};
        const std::string got = run(*engine.value(), m.text, m.segments, &st);
        if (refusals[i].config.model_path == reuse_model()) CHECK(got == want);
        CHECK(st.segment_reused_tokens == 0);
        // Malformed segments are refused whether or not they would be used.
        GenerateOptions bad;
        bad.segments = {{5, 3}};
        CHECK(engine.value()->generate(m.text, greedy(2), {}, bad).status().code() == ErrorCode::InvalidArgument);
    }
    std::remove(hybrid_path.c_str());
    std::filesystem::remove_all(dir);
}

TEST_CASE("C API: generation and prefill with marked segments, the calibration, the statistics") {
    liyab_engine_config c;
    liyab_engine_config_default(&c);
    CHECK(c.segment_reuse == 0);
    CHECK(c.segment_reuse_max_kl > 0.0f);
    c.model_path = reuse_model().c_str();
    c.backend = LIYAB_BACKEND_CPU;
    c.n_threads = 3;
    c.thermal_polling = 0;
    c.power_profile = LIYAB_POWER_PERFORMANCE;
    c.context_length = 8192;
    c.prefix_cache_mb = 64;
    c.segment_reuse = 1;
    c.segment_reuse_max_kl = 1000.0f;
    liyab_engine* engine = nullptr;
    REQUIRE(liyab_engine_create(&c, &engine) == LIYAB_OK);
    liyab_segment_reuse_info info{};
    CHECK(liyab_engine_segment_reuse(engine, &info) == LIYAB_ERR_BUSY);  // calibrating in the background
    CHECK(info.calibrating == 1);
    const auto give_up = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (liyab_engine_segment_reuse(engine, &info) == LIYAB_ERR_BUSY && std::chrono::steady_clock::now() < give_up) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    CHECK(liyab_engine_segment_reuse(engine, &info) == LIYAB_OK);
    CHECK(info.calibrating == 0);
    CHECK(info.progress == 1.0f);
    CHECK(info.fraction == kReuseFractions[0]);
    CHECK(info.bound >= info.kl);
    liyab_sampling_params params;
    liyab_sampling_params_default(&params);
    params.temperature = 0.0f;
    params.max_tokens = 4;
    auto to_c = [](const Marked& m) {
        std::vector<liyab_prompt_segment> out;
        for (const PromptSegment& s : m.segments) out.push_back({s.begin, s.end});
        return out;
    };
    const std::string a = doc_text(11, 40), b = doc_text(12, 40);
    const Marked r1 = rag({a, b}, "one"), r2 = rag({b, a}, "two");
    std::vector<liyab_prompt_segment> s1 = to_c(r1), s2 = to_c(r2);
    CHECK(liyab_engine_prefill_segmented(engine, r1.text.c_str(), s1.data(), s1.size(), 1) == LIYAB_OK);
    liyab_generation_stats stats{};
    CHECK(liyab_engine_generate_segmented(engine, r2.text.c_str(), s2.data(), s2.size(), &params, nullptr, nullptr,
                                          &stats) == LIYAB_OK);
    CHECK(stats.segment_reused_tokens > 0);
    CHECK(stats.segment_recomputed_tokens > 0);
    liyab_prefix_cache_stats pc{};
    REQUIRE(liyab_engine_prefix_cache_stats(engine, &pc) == LIYAB_OK);
    CHECK(pc.segments == 2);
    CHECK(pc.segment_reuses == 2);
    std::vector<liyab_prompt_segment> overlapping = {{0, 10}, {5, 20}};
    CHECK(liyab_engine_generate_segmented(engine, r2.text.c_str(), overlapping.data(), overlapping.size(), &params,
                                          nullptr, nullptr, nullptr) == LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_generate_segmented(engine, r2.text.c_str(), nullptr, 1, &params, nullptr, nullptr, nullptr) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_generate_segmented(engine, r2.text.c_str(), nullptr, 0, &params, nullptr, nullptr, nullptr) ==
          LIYAB_OK);
    liyab_engine_destroy(engine);
    // Off: the reason, the call still answering.
    c.segment_reuse = 0;
    REQUIRE(liyab_engine_create(&c, &engine) == LIYAB_OK);
    CHECK(liyab_engine_segment_reuse(engine, nullptr) == LIYAB_ERR_UNSUPPORTED);
    CHECK(std::string(liyab_last_error()).find("off") != std::string::npos);
    CHECK(liyab_engine_generate_segmented(engine, r2.text.c_str(), s2.data(), s2.size(), &params, nullptr, nullptr,
                                          &stats) == LIYAB_OK);
    CHECK(stats.segment_reused_tokens == 0);
    liyab_engine_destroy(engine);
    CHECK(liyab_engine_segment_reuse(nullptr, nullptr) == LIYAB_ERR_INVALID_ARGUMENT);
}

TEST_CASE("Calibration finds the smallest share whose bound, and every larger one's, is within the target") {
    TransformerOptions options;
    options.context_length = 8192;  // the corpus's prompts in the tiny vocabulary's byte tokens
    test::TinyModelSpec spec;
    spec.n_layers = 4;  // blocks after the selection's, which only the selected tokens run
    auto l = load("calibrate", spec, options);
    REQUIRE(l != nullptr);
    auto tokenizer = Tokenizer::load(l->model->file());
    REQUIRE(tokenizer.has_value());
    const ReusePromptStyle plain_text;  // the tiny model has no chat template
    // A stage at a time, stopped once and resumed: the same result.
    auto job = ReuseCalibrator::create(*l->model, *tokenizer, plain_text, ReuseSettings{}, 1000.0);
    REQUIRE(job.has_value());
    ReuseCalibrator& c = **job;
    CHECK(c.positions() > 0);
    CHECK(c.progress() == 0.0f);
    std::atomic<bool> stop{true};
    auto stopped = c.step(*l->model, l->route, *l->pool, 32, &stop);
    CHECK(stopped.status().code() == ErrorCode::Cancelled);
    CHECK(c.progress() == 0.0f);
    int steps = 0;
    for (;;) {
        auto r = c.step(*l->model, l->route, *l->pool, 32);
        REQUIRE(r.has_value());
        ++steps;
        if (*r) break;
    }
    CHECK(c.done());
    CHECK(c.progress() == 1.0f);
    CHECK(steps > 10);
    const ReuseCalibration& loose = c.result();
    CHECK(loose.met);
    CHECK(loose.fraction == kReuseFractions[0]);
    CHECK(loose.kl > 0.0);
    CHECK(loose.bound >= loose.kl);
    CHECK(loose.top1 > 0.0 && loose.top1 <= 1.0);
    REQUIRE(loose.kls.size() == std::size(kReuseFractions));
    REQUIRE(loose.bounds.size() == std::size(kReuseFractions));
    // None within: refused, at the largest share.
    const ReuseCalibration strict = c.result_for(1e-12);
    CHECK(!strict.met);
    CHECK(strict.fraction == kReuseFractions[std::size(kReuseFractions) - 1]);
    // A target between the bounds: the smallest share whose bound, and
    // every larger share's, is within it.
    const double target = *std::max_element(loose.bounds.begin() + 2, loose.bounds.end());
    const ReuseCalibration middle = c.result_for(target);
    size_t want = std::size(kReuseFractions);
    for (size_t k = std::size(kReuseFractions); k-- > 0 && loose.bounds[k] <= target;) want = k;
    CHECK(middle.met);
    CHECK(middle.fraction == kReuseFractions[want]);
    CHECK(want <= 2);
    // calibrate_reuse(): every stage at once, the context left empty; with
    // a question-guided criterion, each question gets a pass of its own.
    ReuseSettings guided;
    guided.criterion = ReuseCriterion::Impact;
    auto g = calibrate_reuse(*l->model, *tokenizer, plain_text, guided, 1000.0, l->route, *l->pool, 32);
    REQUIRE(g.has_value());
    CHECK(g->met);
    CHECK(g->kl > 0.0);
    CHECK(g->kls.size() == std::size(kReuseFractions));
    CHECK(l->model->n_past() == 0);
}

TEST_CASE("Engine: the calibration runs in the background, gives way to calls, and leaves the context as it was") {
    const std::string dir = test::temp_dir() + "/liyab_reuse_" + std::to_string(test::process_id()) + "_background";
    std::filesystem::remove_all(dir);
    EngineConfig plain = reuse_config(reuse_model());
    plain.prefix_cache_mb = 0;
    plain.segment_reuse = false;
    auto fresh_engine = Engine::create(plain);
    REQUIRE(fresh_engine.has_value());
    auto exact = [&](const std::string& prompt) {
        fresh_engine.value()->reset_context();
        return run(*fresh_engine.value(), prompt);
    };
    const Marked m = rag({doc_text(21, 40), doc_text(22, 40)}, "meanwhile");
    EngineConfig config = reuse_config(reuse_model());
    config.cache_dir = dir;
    {
        auto engine = Engine::create(config);  // returns at once: the calibration has not run
        REQUIRE(engine.has_value());
        Engine& e = *engine.value();
        Engine::SegmentReuseInfo info = e.segment_reuse();
        CHECK(info.calibrating);
        CHECK(info.status.code() == ErrorCode::Busy);
        // Meanwhile a marked prompt is processed as usual.
        GenerationStats st{};
        CHECK(run(e, m.text, m.segments, &st) == exact(m.text));
        CHECK(st.segment_reused_tokens == 0);
        // A context in progress, then idle time: the calibration runs on a
        // context of its own, and a call stops it (no Busy) and continues
        // the conversation's context, which it found as it was left.
        REQUIRE(e.prefill(m.text, true).is_ok());
        const uint64_t restores = e.prefix_cache_stats().restores;
        std::this_thread::sleep_for(std::chrono::milliseconds(2600));
        const std::string next = m.text + " hello";
        const auto t0 = std::chrono::steady_clock::now();
        CHECK(run(e, next, {}, &st) == exact(next));
        const double waited = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(st.cached_prefix_tokens >= static_cast<int32_t>(e.tokenize(m.text, true)->size()) - 1);
        CHECK(e.prefix_cache_stats().restores == restores);  // the live context, not a copy set aside
        CHECK(waited < 30.0);
        info = wait_calibrated(e);
        CHECK(info.status.is_ok());
        CHECK(!info.from_cache);
        CHECK(info.progress == 1.0f);
    }
    // Kept in cache_dir: the next engine has it at once.
    {
        auto engine = Engine::create(config);
        REQUIRE(engine.has_value());
        const Engine::SegmentReuseInfo info = engine.value()->segment_reuse();
        CHECK(!info.calibrating);
        CHECK(info.from_cache);
        CHECK(info.status.is_ok());
        CHECK(info.fraction == kReuseFractions[0]);
    }
    // An engine closed while its calibration runs stops it.
    {
        EngineConfig closing = reuse_config(reuse_model());
        closing.cache_dir.clear();  // nothing kept: it calibrates
        auto engine = Engine::create(closing);
        REQUIRE(engine.has_value());
        std::this_thread::sleep_for(std::chrono::milliseconds(2600));
        CHECK(engine.value()->segment_reuse().calibrating);
        const auto t0 = std::chrono::steady_clock::now();
        engine.value().reset();
        CHECK(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 30.0);
    }
    std::filesystem::remove_all(dir);
    std::filesystem::remove_all(reuse_config(reuse_model()).cache_dir);
}

TEST_MAIN()
