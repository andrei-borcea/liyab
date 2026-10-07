// Tests for the paged, quantized KV cache: memory consumption, page reuse,
// attention sinks, rollback rules and INT4 / INT8 quantization accuracy.
#include <cmath>
#include <cstdio>
#include <random>
#include <unistd.h>
#include <vector>

#include "core/quant.h"
#include "liyab/engine.h"
#include "liyab/kv_cache.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

KvCacheConfig small_config(KvCacheType type, int32_t window = 0, int32_t sinks = 8) {
    KvCacheConfig c;
    c.n_layers = 4;
    c.n_head_kv = 2;
    c.head_dim = 64;
    c.type = type;
    c.max_positions = 1024;
    c.window = window;
    c.sink_tokens = sinks;
    c.page_tokens = 16;
    return c;
}

// K/V row for a position: a deterministic pattern so rows can be checked later.
std::vector<float> row_for(int32_t pos, size_t n) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = std::sin(0.37f * static_cast<float>(pos) + 0.11f * static_cast<float>(i));
    return v;
}

std::vector<float> read_k(const KvCache& cache, int32_t layer, int32_t pos, int32_t head) {
    std::vector<float> out(static_cast<size_t>(cache.config().head_dim));
    quant::dequantize_row(cache.dtype(), cache.k_row(layer, pos, head), out.data(), cache.config().head_dim);
    return out;
}

double rel_rms_error(const std::vector<float>& ref, const std::vector<float>& got) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        num += (ref[i] - got[i]) * (ref[i] - got[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

}  // namespace

TEST_CASE("INT4 KV formats cut memory per token versus F16") {
    // A 32-layer, 8 KV-head, head_dim 128 model (Llama-3-8B / Qwen-2.5-7B class).
    const auto f16 = KvCache::bytes_per_token(32, 8, 128, KvCacheType::F16);
    const auto q8 = KvCache::bytes_per_token(32, 8, 128, KvCacheType::Q8_0);
    const auto q4 = KvCache::bytes_per_token(32, 8, 128, KvCacheType::Q4_0);
    const auto q41 = KvCache::bytes_per_token(32, 8, 128, KvCacheType::Q4_1);
    CHECK(f16 == 131072);  // 128 KiB per token
    CHECK_NEAR(1.0 - static_cast<double>(q8) / f16, 0.46875, 1e-9);
    CHECK_NEAR(1.0 - static_cast<double>(q4) / f16, 0.71875, 1e-9);   // 4.5 bits/value
    CHECK_NEAR(1.0 - static_cast<double>(q41) / f16, 0.6875, 1e-9);   // 5 bits/value
    std::printf("  per token: f16 %zu KiB, q8_0 %zu KiB (-%.1f%%), q4_0 %zu KiB (-%.1f%%), q4_1 %zu KiB (-%.1f%%)\n",
                f16 / 1024, q8 / 1024, 100.0 * (1.0 - static_cast<double>(q8) / f16), q4 / 1024,
                100.0 * (1.0 - static_cast<double>(q4) / f16), q41 / 1024,
                100.0 * (1.0 - static_cast<double>(q41) / f16));
    std::printf("  8K-token context: f16 %.2f GiB -> q4_0 %.2f GiB\n", 8192.0 * f16 / (1 << 30), 8192.0 * q4 / (1 << 30));
}

TEST_CASE("Pages are allocated on demand, not for the whole context") {
    auto cache = KvCache::create(small_config(KvCacheType::Q4_0));
    REQUIRE(cache.has_value());
    KvCache& kv = *cache.value();
    CHECK(kv.pages_allocated() == 0);
    CHECK(kv.bytes_allocated() == 0);
    REQUIRE(kv.reserve(1).is_ok());
    CHECK(kv.pages_in_use() == 1);
    REQUIRE(kv.reserve(100).is_ok());
    CHECK(kv.pages_in_use() == 7);  // ceil(100 / 16)
    CHECK(kv.bytes_in_use() == 7 * kv.page_bytes());
    CHECK(kv.page_bytes() == 2 * 4 * 16 * 2 * 36u);  // K+V, layers, tokens, heads, 2 x 18-byte blocks

    CHECK(kv.reserve(1025).code() == ErrorCode::ContextFull);
    REQUIRE(kv.truncate(20).is_ok());  // pages starting at >= 32 are returned
    CHECK(kv.pages_in_use() == 2);
    REQUIRE(kv.reserve(100).is_ok());
    CHECK(kv.pages_allocated() == 7);  // freed pages were reused, nothing new allocated
    kv.clear();
    CHECK(kv.pages_in_use() == 0);
}

TEST_CASE("Sliding window keeps sink tokens and recycles pages for unbounded streams") {
    auto cache = KvCache::create(small_config(KvCacheType::Q8_0, /*window=*/32, /*sinks=*/8));
    REQUIRE(cache.has_value());
    KvCache& kv = *cache.value();
    const size_t kv_dim = 2 * 64;
    int32_t peak_pages = 0;
    for (int32_t pos = 0; pos < 5000; ++pos) {  // far beyond max_positions: windows are unbounded
        REQUIRE(kv.reserve(pos + 1).is_ok());
        const auto k = row_for(pos, kv_dim);
        for (int32_t l = 0; l < 4; ++l) kv.store(l, pos, k.data(), k.data());
        kv.release_unreachable(pos + 1, /*keep_back=*/4);
        peak_pages = std::max(peak_pages, kv.pages_in_use());
    }
    // Sink page + window (32) + rollback margin (4) spread over 16-token pages.
    CHECK(peak_pages <= 5);
    CHECK(kv.pages_allocated() <= 5);  // the pool never grows past the peak
    std::printf("  5000 positions, window 32 + 8 sinks: %d pages (%zu KiB) instead of %d\n", peak_pages,
                kv.bytes_allocated() / 1024, (5000 + 15) / 16);

    const KvCache::Visible v = kv.visible(4999);
    CHECK(v.sink_end == 8);
    CHECK(v.first == 4968);
    for (const int32_t pos : {0, 7, 4968, 4999}) {  // sinks and window survived
        const auto expected = row_for(pos, kv_dim);
        const auto got = read_k(kv, 3, pos, 1);
        CHECK(rel_rms_error(std::vector<float>(expected.begin() + 64, expected.end()), got) < 0.01);
    }

    // Rollback within the margin is fine; deeper would need released pages.
    CHECK(kv.truncate(4997).is_ok());
    CHECK(!kv.truncate(4000).is_ok());
}

TEST_CASE("Quantized KV rows stay accurate; asymmetric INT4 wins on shifted data") {
    std::mt19937 rng(21);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const int32_t hd = 64;
    // Keys commonly carry large per-channel offsets: model that with a bias.
    std::vector<float> centered(2 * hd), shifted(2 * hd);
    double err[4][2] = {};
    const KvCacheType types[] = {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1};
    for (int trial = 0; trial < 200; ++trial) {
        for (int i = 0; i < 2 * hd; ++i) {
            centered[i] = normal(rng);
            shifted[i] = 2.5f + 0.5f * normal(rng);
        }
        for (int t = 0; t < 4; ++t) {
            auto cache = KvCache::create(small_config(types[t]));
            REQUIRE(cache.has_value());
            KvCache& kv = *cache.value();
            REQUIRE(kv.reserve(2).is_ok());
            kv.store(0, 0, centered.data(), centered.data());
            kv.store(0, 1, shifted.data(), shifted.data());
            err[t][0] += rel_rms_error(std::vector<float>(centered.begin(), centered.begin() + hd), read_k(kv, 0, 0, 0));
            err[t][1] += rel_rms_error(std::vector<float>(shifted.begin(), shifted.begin() + hd), read_k(kv, 0, 1, 0));
        }
    }
    for (auto& e : err) {
        e[0] /= 200;
        e[1] /= 200;
    }
    std::printf("  relative RMS error (centered / shifted): f16 %.4f/%.4f  q8_0 %.4f/%.4f  q4_0 %.4f/%.4f  q4_1 %.4f/%.4f\n",
                err[0][0], err[0][1], err[1][0], err[1][1], err[2][0], err[2][1], err[3][0], err[3][1]);
    CHECK(err[0][0] < 1e-3);
    CHECK(err[1][0] < 0.01);
    CHECK(err[2][0] < 0.12);
    CHECK(err[3][0] < 0.12);
    CHECK(err[3][1] < err[2][1]);  // asymmetric INT4 tracks the offset
}

TEST_CASE("Engine with an INT4 sliding-window KV cache uses less KV memory") {
    const std::string path = test::temp_dir() + "/liyab_kv_" + std::to_string(getpid()) + ".gguf";
    test::write_tiny_model(path);
    auto run = [&](KvCacheType type) {
        EngineConfig config;
        config.model_path = path;
        config.backend = BackendKind::Cpu;
        config.thermal_polling = false;
        config.power.profile = PowerProfile::Performance;
        config.kv_cache_type = type;
        config.sliding_window = 64;
        auto engine = Engine::create(config);
        GenerationStats stats;
        if (!engine) return stats;
        SamplingParams p;
        p.temperature = 0.0f;
        p.max_tokens = 100;
        auto r = engine.value()->generate_tokens(std::vector<int32_t>{1, 270, 300}, p, {});
        if (r) stats = r.value();
        return stats;
    };
    const GenerationStats q8 = run(KvCacheType::Q8_0);
    const GenerationStats q41 = run(KvCacheType::Q4_1);
    CHECK(q8.generated_tokens > 0);
    CHECK(q41.generated_tokens > 0);
    CHECK(q41.kv_cache_bytes > 0 && q41.kv_cache_bytes < q8.kv_cache_bytes);
    std::remove(path.c_str());
}

TEST_MAIN()
