// Tests for the paged, quantized KV cache: memory consumption, page reuse,
// attention sinks, rollback rules and INT4 / INT8 quantization accuracy; its
// pages as chunk images, and the engine's KV store of such chunks.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "core/kv_codec.h"
#include "core/kv_store.h"
#include "core/thread_pool.h"
#include "liyab/mmap_loader.h"
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

TEST_CASE("Pages hold zeros when mapped, new or recycled: rows past the end never carry old bytes") {
    auto cache = KvCache::create(small_config(KvCacheType::F16));
    REQUIRE(cache.has_value());
    KvCache& kv = *cache.value();
    auto all_zero = [&](int32_t page) {
        const uint8_t* p = kv.page_data(page);
        if (p == nullptr) return false;
        for (size_t i = 0; i < kv.page_bytes(); ++i) {
            if (p[i] != 0) return false;
        }
        return true;
    };
    REQUIRE(kv.reserve(40).is_ok());  // pages 0-2, new
    for (int32_t page = 0; page < 3; ++page) CHECK(all_zero(page));
    for (int32_t page = 0; page < 3; ++page) std::memset(kv.mutable_page_data(page), 0xA5, kv.page_bytes());
    REQUIRE(kv.truncate(10).is_ok());  // pages 1-2 go back to the free list with their bytes
    REQUIRE(kv.reserve(40).is_ok());   // and come back recycled
    CHECK(kv.pages_allocated() == 3);
    CHECK(!all_zero(0));               // a page kept in use keeps its rows
    CHECK(all_zero(1));
    CHECK(all_zero(2));
    kv.clear();
    REQUIRE(kv.reserve(1).is_ok());
    CHECK(all_zero(0));
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

// Multi-head latent attention keeps one row per position: [RoPE part |
// latent], the latent being the value. The value rows are those channels of
// the key rows, bit for bit, in half the memory, for every cache type.
TEST_CASE("Values read from the key rows (latent attention): the same bits in half the memory") {
    for (const KvCacheType type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        KvCacheConfig separate = small_config(type);
        separate.n_head_kv = 1;
        separate.head_dim = 96;  // 32 RoPE channels, then a 64-wide latent
        KvCacheConfig shared = separate;
        shared.values_in_keys = true;
        shared.value_offset = 32;
        CHECK(KvCache::bytes_per_token(shared) * 2 == KvCache::bytes_per_token(separate));
        auto a = KvCache::create(separate);
        auto b = KvCache::create(shared);
        REQUIRE(a.has_value() && b.has_value());
        CHECK(b.value()->page_bytes() * 2 == a.value()->page_bytes());
        REQUIRE(a.value()->reserve(40).is_ok());
        REQUIRE(b.value()->reserve(40).is_ok());
        const auto latent = static_cast<size_t>(separate.head_dim - shared.value_offset);
        for (int32_t layer = 0; layer < separate.n_layers; ++layer) {
            for (int32_t pos = 0; pos < 40; ++pos) {
                const std::vector<float> k = row_for(pos + 100 * layer, 96);
                std::vector<float> v(96, 0.0f);  // the latent, at the start of a separate value row
                std::copy(k.begin() + 32, k.end(), v.begin());
                a.value()->store(layer, pos, k.data(), v.data());
                b.value()->store(layer, pos, k.data(), nullptr);
            }
        }
        const size_t value_bytes = dtype_row_bytes(b.value()->dtype(), static_cast<int64_t>(latent));
        bool same = true;
        for (int32_t layer = 0; layer < separate.n_layers; ++layer) {
            for (int32_t pos = 0; pos < 40; ++pos) {
                const size_t key_bytes = dtype_row_bytes(a.value()->dtype(), 96);
                same = same && std::memcmp(a.value()->k_row(layer, pos, 0), b.value()->k_row(layer, pos, 0), key_bytes) == 0;
                same = same && std::memcmp(a.value()->v_row(layer, pos, 0), b.value()->v_row(layer, pos, 0), value_bytes) == 0;
            }
        }
        CHECK(same);
    }
    KvCacheConfig unaligned = small_config(KvCacheType::Q8_0);
    unaligned.values_in_keys = true;
    unaligned.value_offset = 16;  // splits a Q8_0 block
    CHECK(!KvCache::create(unaligned).has_value());
}

TEST_CASE("Engine with an INT4 sliding-window KV cache uses less KV memory") {
    const std::string path = test::temp_dir() + "/liyab_kv_" + std::to_string(test::process_id()) + ".gguf";
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

TEST_CASE("Chunk images of KV pages: out and back in bit for bit, whole and partial, values in keys too") {
    for (const bool values_in_keys : {false, true}) {
        KvCacheConfig c = small_config(KvCacheType::Q8_0);
        if (values_in_keys) {
            c.values_in_keys = true;
            c.value_offset = 32;
        }
        auto a = KvCache::create(c);
        auto b = KvCache::create(c);
        REQUIRE(a.has_value() && b.has_value());
        KvCache& src = *a.value();
        KvCache& dst = *b.value();
        const int32_t n = 2 * c.page_tokens + 5;  // two whole pages and a partial one
        REQUIRE(src.reserve(n).is_ok());
        for (int32_t layer = 0; layer < c.n_layers; ++layer) {
            for (int32_t pos = 0; pos < n; ++pos) {
                // Every KV head of the position at once, as the model stores them.
                const std::vector<float> k = row_for(pos * 7 + layer * 131, 64 * static_cast<size_t>(c.n_head_kv));
                const std::vector<float> v = row_for(pos * 5 + layer * 17 + 1000, k.size());
                src.store(layer, pos, k.data(), v.data());
            }
        }
        CHECK(src.rows_bytes(c.page_tokens) == src.page_bytes());
        CHECK(src.rows_bytes(5) * static_cast<size_t>(c.page_tokens) == src.page_bytes() * 5);
        // Out as chunk images, in again into another cache: every row the same bits.
        REQUIRE(dst.reserve(n).is_ok());
        for (int32_t p = 0; p * c.page_tokens < n; ++p) {
            const int32_t rows = std::min(c.page_tokens, n - p * c.page_tokens);
            std::vector<uint8_t> image(src.rows_bytes(rows));
            REQUIRE(src.read_rows(p, rows, image.data()).is_ok());
            // The partial page in two parts, to cover a write from inside an image.
            if (rows < c.page_tokens) {
                REQUIRE(dst.write_rows(p, image.data(), rows, 0, 2).is_ok());
                REQUIRE(dst.write_rows(p, image.data(), rows, 2, rows).is_ok());
            } else {
                REQUIRE(dst.write_rows(p, image.data(), rows, 0, rows).is_ok());
            }
            // And expanded into a page's layout: the rows in place, the rest zero.
            std::vector<uint8_t> page(src.page_bytes(), 0xEE);
            src.image_to_page(image.data(), rows, page.data());
            bool same_rows = true;
            const size_t row = dtype_row_bytes(src.dtype(), 64);
            for (int32_t layer = 0; layer < c.n_layers; ++layer) {
                for (int32_t slot = 0; slot < rows; ++slot) {
                    const int32_t pos = p * c.page_tokens + slot;
                    for (int32_t h = 0; h < c.n_head_kv; ++h) {
                        const size_t at = (static_cast<size_t>(layer * c.page_tokens + slot) * c.n_head_kv + h) * row;
                        same_rows = same_rows && std::memcmp(page.data() + at, src.k_row(layer, pos, h), row) == 0;
                    }
                }
            }
            CHECK(same_rows);
            if (rows < c.page_tokens) CHECK(page.back() == 0);
        }
        bool same = true;
        const size_t row = dtype_row_bytes(src.dtype(), 64);
        for (int32_t layer = 0; layer < c.n_layers; ++layer) {
            for (int32_t pos = 0; pos < n; ++pos) {
                for (int32_t h = 0; h < c.n_head_kv; ++h) {
                    same = same && std::memcmp(src.k_row(layer, pos, h), dst.k_row(layer, pos, h), row) == 0;
                    if (!values_in_keys) same = same && std::memcmp(src.v_row(layer, pos, h), dst.v_row(layer, pos, h), row) == 0;
                }
            }
        }
        CHECK(same);
        std::vector<uint8_t> image(src.rows_bytes(4));
        CHECK(!src.read_rows(9, 4, image.data()).is_ok());           // not mapped
        CHECK(!src.read_rows(0, c.page_tokens + 1, image.data()).is_ok());
        CHECK(!dst.write_rows(0, image.data(), 4, 3, 5).is_ok());    // past the image
    }
}

// ---------------------------------------------------------------------------
// The KV store (core/kv_store.h), on synthetic chunks: a position's rows are
// a function of its token, a state image one of the tokens before it, as the
// store assumes (identical prefixes, identical bytes).
// ---------------------------------------------------------------------------
namespace {

constexpr int32_t kChunk = 16;
constexpr size_t kPosBytes = 32;
constexpr size_t kStateFloats = 8;

KvStore::Config store_config(size_t ram, const std::string& dir = {}, size_t disk = 0, bool hybrid = false,
                             uint64_t seed = 1) {
    KvStore::Config c;
    const uint64_t parts[] = {seed};
    c.seed = kv_seed_key(parts);
    c.chunk_tokens = kChunk;
    c.chunk_bytes = kChunk * kPosBytes;
    c.state_floats = hybrid ? kStateFloats : 0;
    c.ram_bytes = ram;
    c.dir = dir;
    c.disk_bytes = disk;
    return c;
}

uint8_t row_byte(int32_t token, int32_t pos, size_t b) {
    return static_cast<uint8_t>(token * 31 + pos * 7 + static_cast<int32_t>(b));
}

std::vector<float> state_image(std::span<const int32_t> tokens, int32_t pos) {
    std::vector<float> s(kStateFloats);
    double acc = 0;
    for (int32_t i = 0; i < pos; ++i) acc = acc * 0.5 + tokens[static_cast<size_t>(i)];
    for (size_t i = 0; i < s.size(); ++i) s[i] = static_cast<float>(acc + static_cast<double>(i));
    return s;
}

// A context as the engine sets it aside, with states at `marks` (hybrid stores).
KvStore::PutResult put(KvStore& store, const std::vector<int32_t>& tokens, std::vector<KvStore::StateMark> marks = {},
                       bool storage_only = false) {
    KvStore::ContextSource src;
    src.tokens = tokens;
    src.text = "";
    // The rows as parts, as a cache's pages are: a whole chunk in two (two
    // pages), a partial one in one. The store writes storage straight from
    // them and copies them once for RAM.
    std::vector<std::vector<uint8_t>> images;
    src.rows = [&](int32_t chunk, int32_t n) {
        images.emplace_back(static_cast<size_t>(n) * kPosBytes);
        uint8_t* out = images.back().data();
        for (int32_t s = 0; s < n; ++s) {
            const int32_t pos = chunk * kChunk + s;
            for (size_t b = 0; b < kPosBytes; ++b) out[static_cast<size_t>(s) * kPosBytes + b] = row_byte(tokens[static_cast<size_t>(pos)], pos, b);
        }
        const std::span<const uint8_t> all(images.back());
        if (n < kChunk) return std::vector<std::span<const uint8_t>>{all};
        return std::vector<std::span<const uint8_t>>{all.first(all.size() / 2), all.subspan(all.size() / 2)};
    };
    src.states = std::move(marks);
    // An image in two parts, as a model's per-block states are.
    std::vector<std::vector<float>> states;
    src.state_parts = [&](int32_t pos) {
        states.push_back(state_image(tokens, pos));
        const std::span<const float> all(states.back());
        return std::vector<std::span<const float>>{all.first(3), all.subspan(3)};
    };
    return store.put(src, storage_only);
}

// A state image read in two parts, as into a model's per-block states.
Status read_state(KvStore& store, const KvKey& key, std::vector<float>& out) {
    const std::span<float> all(out);
    const std::span<float> parts[] = {all.first(5), all.subspan(5)};
    return store.read_state(key, parts);
}

// Whether the store gives back the right bytes for the first `n` positions of `tokens` along `plan`.
bool rows_match(KvStore& store, const KvStore::Plan& plan, const std::vector<int32_t>& tokens) {
    for (size_t i = 0; i < plan.chunks.size(); ++i) {
        const int32_t n = plan.chunk_sizes[i];
        std::vector<uint8_t> image(static_cast<size_t>(n) * kPosBytes);
        const std::span<uint8_t> into(image);
        if (!store.read_chunk(plan.chunks[i], {&into, 1}).is_ok()) return false;
        for (int32_t s = 0; s < n; ++s) {
            const int32_t pos = static_cast<int32_t>(i) * kChunk + s;
            if (pos >= plan.positions) break;
            for (size_t b = 0; b < kPosBytes; ++b) {
                if (image[static_cast<size_t>(s) * kPosBytes + b] != row_byte(tokens[static_cast<size_t>(pos)], pos, b)) return false;
            }
        }
    }
    return true;
}

std::vector<int32_t> random_tokens(std::mt19937& rng, size_t n) {
    std::vector<int32_t> t(n);
    for (int32_t& x : t) x = 1 + static_cast<int32_t>(rng() % 5000);
    return t;
}

std::vector<int32_t> join(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

std::string store_dir(const std::string& name) {
    const std::string dir = test::temp_dir() + "/liyab_kvs_" + std::to_string(test::process_id()) + "_" + name;
    std::filesystem::remove_all(dir);
    return dir;
}

}  // namespace

TEST_CASE("The KV store's payload hash, taken in pieces, is xxhash64 of the whole") {
    std::mt19937 rng(9);
    std::vector<uint8_t> data(3000);
    for (uint8_t& b : data) b = static_cast<uint8_t>(rng());
    bool same = true;
    for (size_t length : {size_t{0}, size_t{1}, size_t{7}, size_t{31}, size_t{32}, size_t{33}, size_t{100}, size_t{3000}}) {
        for (int trial = 0; trial < 20; ++trial) {
            KvStreamHash hash;
            for (size_t at = 0; at < length;) {
                const size_t n = std::min(length - at, static_cast<size_t>(rng() % 70));
                hash.update(data.data() + at, n);
                at += n;
            }
            same = same && hash.digest() == xxhash64(data.data(), length);
        }
    }
    CHECK(same);
}

TEST_CASE("The KV store keeps a shared prefix once and restores a prompt up to where it diverges") {
    std::mt19937 rng(7);
    auto store = KvStore::open(store_config(size_t{1} << 20));
    REQUIRE(store != nullptr);
    const std::vector<int32_t> x = random_tokens(rng, 100);  // six whole chunks and 4 tokens
    KvStore::PutResult r = put(*store, x);
    CHECK(r.whole);
    CHECK(r.chunks == 7);
    CHECK(r.chunks_written == 7);
    KvStore::Plan p = store->plan(x, x.size() - 1);
    CHECK(p.positions == 99);
    CHECK(p.chunks.size() == 7);
    CHECK(p.chunk_sizes.back() == 4);
    CHECK(rows_match(*store, p, x));
    // Another context sharing 40 tokens: its first two chunks are stored already.
    const std::vector<int32_t> y = join(std::vector<int32_t>(x.begin(), x.begin() + 40), random_tokens(rng, 50));
    r = put(*store, y);
    CHECK(r.chunks == 6);
    CHECK(r.chunks_written == 4);
    CHECK(store->stats().chunks_ram == 11);
    // A prompt that leaves X after 50 tokens: three whole chunks, then X's
    // fourth (which holds the most of it) cut where they part.
    std::vector<int32_t> z(x.begin(), x.begin() + 50);
    z.push_back(9999);
    z.push_back(9998);
    p = store->plan(z, z.size());
    CHECK(p.positions == 50);
    CHECK(p.chunks.size() == 4);
    CHECK(rows_match(*store, p, z));
    // Limits are kept, and an unknown prompt gets nothing.
    CHECK(store->plan(x, 37).positions == 37);
    CHECK(store->plan(random_tokens(rng, 80), 80).positions == 0);
    // Records: the texts the engine matches; setting X aside again with more
    // tokens replaces its older record.
    CHECK(store->records().size() == 2);
    put(*store, join(x, random_tokens(rng, 20)));
    CHECK(store->records().size() == 2);
    // The same chunk tokens elsewhere: found by their content key.
    const std::span<const int32_t> first(x.data(), kChunk);
    const KvKey content = kv_content_key(store_config(0).seed, first);
    CHECK(store->chunks_with_content(content).size() == 1);
    put(*store, join(random_tokens(rng, kChunk), std::vector<int32_t>(x.begin(), x.begin() + 3 * kChunk)));
    const auto found = store->chunks_with_content(content);
    REQUIRE(found.size() == 2);
    CHECK(found[0].start != found[1].start);
}

TEST_CASE("The KV store restores a hybrid model up to the deepest state its prompt reaches") {
    std::mt19937 rng(11);
    auto store = KvStore::open(store_config(size_t{1} << 20, {}, 0, /*hybrid=*/true));
    REQUIRE(store != nullptr);
    const std::vector<int32_t> x = random_tokens(rng, 100);
    // The oldest snapshot, the last prompt's end (pinned), the end.
    KvStore::PutResult r = put(*store, x, {{20, false}, {70, true}, {100, false}});
    CHECK(r.whole);
    CHECK(r.states_written == 3);
    std::vector<int32_t> prompt = join(std::vector<int32_t>(x.begin(), x.begin() + 80), {7, 7, 7});
    KvStore::Plan p = store->plan(prompt, prompt.size());
    CHECK(p.positions == 70);
    CHECK(p.state_pinned);
    REQUIRE(p.snapshots.size() == 1);
    CHECK(p.snapshots[0].pos == 20);
    CHECK(rows_match(*store, p, prompt));
    std::vector<float> state(kStateFloats);
    REQUIRE(read_state(*store, p.state, state).is_ok());
    CHECK(state == state_image(x, 70));
    REQUIRE(read_state(*store, p.snapshots[0].key, state).is_ok());
    CHECK(state == state_image(x, 20));
    prompt = join(std::vector<int32_t>(x.begin(), x.begin() + 50), {7, 7, 7});
    p = store->plan(prompt, prompt.size());
    CHECK(p.positions == 20);
    CHECK(p.snapshots.empty());
    CHECK(store->plan(x, 99).positions == 70);  // the end needs every token: the limit leaves the last one out
    CHECK(store->plan(x, 100).positions == 100);
    // Another conversation from the same start shares the image at 20 (inside
    // a chunk each holds): it stays while either chunk does.
    const std::vector<int32_t> y = join(std::vector<int32_t>(x.begin(), x.begin() + 20), random_tokens(rng, 40));
    r = put(*store, y, {{20, false}, {50, true}, {60, false}});
    CHECK(r.states_written == 2);
    CHECK(store->stats().states == 5);
    prompt = join(std::vector<int32_t>(x.begin(), x.begin() + 20), {3, 3, 3, 3});
    CHECK(store->plan(prompt, prompt.size()).positions == 20);
}

TEST_CASE("The KV store restores windowed blocks where a prompt's own prefix leads to the whole window's pages") {
    std::mt19937 rng(17);
    KvStore::Config config = store_config(size_t{1} << 20, {}, 0, /*hybrid=*/true);
    config.window = 20;      // a window of 20 positions
    config.page_tokens = 8;  // over pages of 8 (two per chunk)
    auto store = KvStore::open(config);
    REQUIRE(store != nullptr);
    const std::vector<int32_t> x = random_tokens(rng, 100);
    // The window pages a cache still held: the live window (pages 9-12, the
    // end marked as a restore point) and a snapshot's at 37 (pages 2-4, each
    // at its end, and at the snapshot).
    KvStore::PutResult r = put(*store, x,
                               {{24, false}, {32, false}, {37, true}, {40, false}, {80, false}, {88, false}, {96, false},
                                {100, true}});
    CHECK(r.whole);
    KvStore::Plan p = store->plan(x, 100);
    CHECK(p.positions == 100);
    REQUIRE(p.snapshots.size() == 1);  // the snapshot, with its window
    CHECK(p.snapshots[0].pos == 37);
    CHECK(p.snapshots[0].pinned);
    CHECK(!p.state_pinned);
    std::vector<int32_t> pages;
    for (const KvStore::RecordState& rs : p.window_pages) pages.push_back(rs.pos);
    // 81..99 and 18..36, page by page, each page whole (37's last page by its end).
    CHECK(pages == std::vector<int32_t>({24, 32, 40, 88, 96, 100}));
    CHECK(rows_match(*store, p, x));
    std::vector<float> image(kStateFloats);
    REQUIRE(read_state(*store, p.window_pages[1].key, image).is_ok());
    CHECK(image == state_image(x, 32));
    // A prompt leaving x at 60: back to 40, the deepest position whose whole
    // window is stored (the end of the snapshot's last page), the snapshot
    // below it.
    std::vector<int32_t> prompt = join(std::vector<int32_t>(x.begin(), x.begin() + 60), {7, 7, 7});
    p = store->plan(prompt, prompt.size());
    CHECK(p.positions == 40);
    REQUIRE(p.snapshots.size() == 1);
    CHECK(p.snapshots[0].pos == 37);
    // Leaving it at 39: the snapshot itself.
    prompt = join(std::vector<int32_t>(x.begin(), x.begin() + 39), {7, 7, 7});
    p = store->plan(prompt, prompt.size());
    CHECK(p.positions == 37);
    CHECK(p.snapshots.empty());
    CHECK(p.state_pinned);
    // Leaving it at 30: page 24's window would need the pages ending at 8
    // and 16, which no context kept: nothing to restore.
    prompt = join(std::vector<int32_t>(x.begin(), x.begin() + 30), {7, 7, 7});
    CHECK(store->plan(prompt, prompt.size()).positions == 0);
    // A context of window pages that lost one (evicted, or never kept):
    // restored only above the hole.
    const std::vector<int32_t> y = random_tokens(rng, 60);
    put(*store, y, {{8, false}, {16, false}, {24, false}, {40, false}, {48, false}, {56, false}, {60, true}});
    CHECK(store->plan(y, 60).positions == 60);  // the window of 60: pages 5-7 (41..59)
    prompt = join(std::vector<int32_t>(y.begin(), y.begin() + 30), {7, 7});
    CHECK(store->plan(prompt, prompt.size()).positions == 24);  // 32 is missing: 24, whose window is 5..23
}

TEST_CASE("The KV store stays within its caps, least recently used out first, a chunk never before what continues it") {
    std::mt19937 rng(3);
    const std::vector<int32_t> system = random_tokens(rng, 2 * kChunk);
    const size_t context_bytes = 7 * kChunk * kPosBytes;  // a 100-token context's chunks
    const size_t cap = 3 * context_bytes;
    auto store = KvStore::open(store_config(cap));
    REQUIRE(store != nullptr);
    std::vector<std::vector<int32_t>> contexts;
    bool within = true, chains = true;
    for (int i = 0; i < 12; ++i) {
        contexts.push_back(join(system, random_tokens(rng, 68)));
        put(*store, contexts.back());
        within = within && store->stats().ram_bytes <= cap;
        for (const KvStore::Record* rec : store->records()) chains = chains && !store->chunk_path(rec->tokens).empty();
    }
    CHECK(within);
    CHECK(chains);
    // The newest come back whole; the oldest only as far as the shared system prompt.
    CHECK(store->plan(contexts.back(), 100).positions == 100);
    CHECK(store->plan(contexts.front(), 100).positions == static_cast<int32_t>(system.size()));
    // A context used again is the most recent: it outlives newer ones.
    KvStore::Plan p = store->plan(contexts[9], 100);
    REQUIRE(p.positions == 100);
    store->used(p);
    for (int i = 0; i < 2; ++i) put(*store, join(system, random_tokens(rng, 68)));
    CHECK(store->plan(contexts[9], 100).positions == 100);
    CHECK(store->plan(contexts[10], 100).positions < 100);
    // With storage behind RAM: what RAM cannot hold moves there, and comes back the same.
    const std::string dir = store_dir("tiers");
    {
        auto tiered = KvStore::open(store_config(context_bytes, dir, 4 * context_bytes));
        REQUIRE(tiered != nullptr);
        CHECK(tiered->has_storage());
        for (int i = 0; i < 12; ++i) {
            put(*tiered, contexts[static_cast<size_t>(i)]);
            const KvStore::Stats st = tiered->stats();
            within = within && st.ram_bytes <= context_bytes && st.disk_bytes <= 4 * context_bytes;
        }
        CHECK(within);
        const KvStore::Stats st = tiered->stats();
        CHECK(st.chunks_disk > 0);
        CHECK(st.chunks_ram > 0);
        p = tiered->plan(contexts[9], 100);
        CHECK(p.positions == 100);
        CHECK(rows_match(*tiered, p, contexts[9]));
        // Too large for both tiers: its first chunks are kept, no record.
        const std::vector<int32_t> huge = random_tokens(rng, 100 * kChunk);
        const KvStore::PutResult r = put(*tiered, huge);
        CHECK(!r.whole);
        CHECK(tiered->stats().disk_bytes <= 4 * context_bytes);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("The KV store keeps segments as contexts no prompt reaches, found by their tokens wherever they were computed") {
    std::mt19937 rng(13);
    const std::string dir = store_dir("segments");
    const std::vector<int32_t> doc = random_tokens(rng, 40);  // 42 positions with its markers: three chunks
    const std::vector<int32_t> longer = join(doc, random_tokens(rng, 9));
    {
        auto store = KvStore::open(store_config(0, dir, size_t{1} << 20));  // storage only: files a later store finds
        REQUIRE(store != nullptr);
        const std::vector<int32_t> at100 = kv_segment_tokens(doc, 100);
        CHECK(kv_is_segment(at100));
        CHECK(!kv_is_segment(doc));
        CHECK(put(*store, at100).whole);
        std::vector<KvStore::SegmentCopy> copies = store->segments(doc);
        REQUIRE(copies.size() == 1);
        CHECK(copies[0].base == 100);
        REQUIRE(copies[0].chunks.size() == 3);
        CHECK(copies[0].chunk_sizes.back() == 42 - 2 * kChunk);
        KvStore::Plan as_read;
        as_read.positions = static_cast<int32_t>(at100.size());
        as_read.chunks = copies[0].chunks;
        as_read.chunk_sizes = copies[0].chunk_sizes;
        CHECK(rows_match(*store, as_read, at100));
        // Never a prefix: a prompt of the same tokens restores nothing, and
        // segments are neither contexts nor texts a prompt is matched against.
        CHECK(store->plan(doc, doc.size()).positions == 0);
        CHECK(store->records().empty());
        CHECK(store->stats().segments == 1);
        CHECK(store->stats().records == 0);
        // The same tokens computed elsewhere: a second copy, the newest first;
        // using the older brings it to the front.
        put(*store, kv_segment_tokens(doc, 7));
        copies = store->segments(doc);
        REQUIRE(copies.size() == 2);
        CHECK(copies[0].base == 7);
        CHECK(copies[1].base == 100);
        KvStore::Plan older;
        older.chunks = copies[1].chunks;
        store->used(older);
        CHECK(store->segments(doc)[0].base == 100);
        // A segment that starts another stays its own (the end marker), and
        // other or shorter tokens find nothing.
        put(*store, kv_segment_tokens(longer, 100));
        CHECK(store->segments(doc).size() == 2);
        CHECK(store->segments(longer).size() == 1);
        CHECK(store->segments(std::span(doc).first(39)).empty());
        // Contexts and segments side by side.
        const std::vector<int32_t> context = join(doc, random_tokens(rng, 30));
        put(*store, context);
        CHECK(store->records().size() == 1);
        CHECK(store->plan(context, context.size()).positions == static_cast<int32_t>(context.size()));
        CHECK(store->stats().segments == 3);
    }
    {
        // On storage they outlive the store, as contexts do.
        auto store = KvStore::open(store_config(0, dir, size_t{1} << 20));
        REQUIRE(store != nullptr);
        CHECK(store->stats().removed_files == 0);
        const std::vector<KvStore::SegmentCopy> copies = store->segments(doc);
        REQUIRE(copies.size() == 2);
        KvStore::Plan as_read;
        as_read.positions = static_cast<int32_t>(doc.size() + 2);
        as_read.chunks = copies[0].chunks;
        as_read.chunk_sizes = copies[0].chunk_sizes;
        CHECK(rows_match(*store, as_read, kv_segment_tokens(doc, copies[0].base)));
        CHECK(store->segments(longer).size() == 1);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("The KV store's files outlive it, in LRU order, and what it cannot trust is removed") {
    std::mt19937 rng(5);
    const std::string dir = store_dir("files");
    const std::vector<int32_t> a = random_tokens(rng, 100), b = random_tokens(rng, 100), c = random_tokens(rng, 100);
    {
        auto store = KvStore::open(store_config(0, dir, size_t{1} << 20, /*hybrid=*/true));
        REQUIRE(store != nullptr);
        put(*store, a, {{30, false}, {90, true}, {100, false}});
        put(*store, b, {{100, true}});
        put(*store, c, {{100, true}});
        store->used(store->plan(a, 100));  // A again: the most recent
        // One store per directory.
        auto second = KvStore::open(store_config(size_t{1} << 20, dir, size_t{1} << 20, true));
        REQUIRE(second != nullptr);
        CHECK(!second->has_storage());
    }
    {
        auto store = KvStore::open(store_config(0, dir, size_t{1} << 20, true));
        REQUIRE(store != nullptr);
        CHECK(store->records().size() == 3);
        const KvStore::Plan p = store->plan(a, 100);
        CHECK(p.positions == 100);
        CHECK(p.snapshots.size() == 2);
        CHECK(rows_match(*store, p, a));
        std::vector<float> state(kStateFloats);
        REQUIRE(read_state(*store, p.state, state).is_ok());
        CHECK(state == state_image(a, 100));
    }
    // A lower cap: B goes first (A was used after it), then C; A stays.
    const size_t one = [&] {
        size_t bytes = 0;
        for (const auto& e : std::filesystem::directory_iterator(dir)) bytes += e.file_size();
        return bytes / 3 + 2048;
    }();
    {
        auto store = KvStore::open(store_config(0, dir, one, true));
        REQUIRE(store != nullptr);
        CHECK(store->stats().disk_bytes <= one);
        CHECK(store->plan(a, 100).positions == 100);
        CHECK(store->plan(b, 100).positions == 0);
    }
    // Another seed's files are kept and counted, never returned.
    {
        auto other = KvStore::open(store_config(0, dir, size_t{1} << 20, true, /*seed=*/2));
        REQUIRE(other != nullptr);
        CHECK(other->records().empty());
        CHECK(other->plan(a, 100).positions == 0);
        CHECK(other->stats().foreign_files > 0);
        put(*other, b, {{100, true}});
    }
    // Untrusted files: an interrupted write, another format, a truncated
    // chunk (its chain goes with it), an older build's prefix-cache file.
    std::vector<std::filesystem::path> chunks;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() == ".chunk") chunks.push_back(e.path());
    }
    REQUIRE(!chunks.empty());
    std::ofstream(dir + "/kv-" + std::string(32, '1') + ".state.tmp") << "cut short";
    std::ofstream(dir + "/kv-" + std::string(32, '2') + ".ctx") << "not a record";
    std::ofstream(dir + "/ctx-42-1-1.state") << "older build";
    std::ofstream(dir + "/notes.txt") << "not the store's";
    for (const auto& path : chunks) std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    {
        auto store = KvStore::open(store_config(0, dir, size_t{1} << 20, true));
        REQUIRE(store != nullptr);
        CHECK(store->stats().removed_files > 3);
        CHECK(store->plan(a, 100).positions == 0);
        CHECK(store->records().empty());
    }
    size_t left = 0;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        const std::string name = e.path().filename().string();
        left += name.starts_with("kv-") || name.starts_with("ctx-") ? 1 : 0;
    }
    CHECK(left == 0);  // every chunk was damaged: nothing of the store is left
    CHECK(std::filesystem::exists(dir + "/notes.txt"));
    std::filesystem::remove_all(dir);
}

// ---------------------------------------------------------------------------
// Encoded chunks on storage (core/kv_codec.h): rows shaped like real keys and
// values (channels of their own magnitude, a few large and nearly constant).
// ---------------------------------------------------------------------------
namespace {

// The value of channel `ch` of a position's row: a function of its token, so
// identical prefixes give identical rows, as the store assumes.
float kv_value(int32_t token, int32_t pos, size_t row, size_t ch) {
    const float scale = ch % 29 == 3 ? 40.0f : 0.25f + static_cast<float>(ch % 7) * 0.3f;
    const float bias = ch % 29 == 3 ? -60.0f : 0.0f;
    return bias + scale * std::sin(0.37f * static_cast<float>(token) + 0.11f * static_cast<float>(pos) +
                                   1.7f * static_cast<float>(row) + static_cast<float>(ch));
}

DType kv_dtype(KvCacheType type) {
    switch (type) {
        case KvCacheType::F16: return DType::F16;
        case KvCacheType::Q8_0: return DType::Q8_0;
        case KvCacheType::Q4_0: return DType::Q4_0;
        case KvCacheType::Q4_1: return DType::Q4_1;
    }
    return DType::F16;
}

// An image of positions [first, first + n) of `tokens` laid out as `l` says.
std::vector<uint8_t> kv_image(const KvRowLayout& l, std::span<const int32_t> tokens, int32_t first, int32_t n) {
    const size_t row_bytes = l.row_bytes();
    std::vector<uint8_t> image(l.image_bytes(n));
    std::vector<float> values(static_cast<size_t>(l.head_dim));
    size_t at = 0;
    for (int32_t run = 0; run < l.planes * l.layers; ++run) {
        for (int32_t s = 0; s < n; ++s) {
            for (int32_t h = 0; h < l.heads; ++h, at += row_bytes) {
                const int32_t pos = first + s;
                for (size_t c = 0; c < values.size(); ++c) {
                    values[c] = kv_value(tokens[static_cast<size_t>(pos)], pos, static_cast<size_t>(run * l.heads + h), c);
                }
                quant::quantize_row(kv_dtype(l.type), values.data(), image.data() + at, l.head_dim);
            }
        }
    }
    return image;
}

KvRowLayout codec_layout(KvCacheType type, int32_t planes) {
    KvRowLayout l;
    l.type = type;
    l.planes = planes;
    l.layers = 3;
    l.heads = 2;
    l.head_dim = 64;
    return l;
}

// The values of an image.
std::vector<float> kv_values(const KvRowLayout& l, std::span<const uint8_t> image, int32_t n) {
    const size_t rows = static_cast<size_t>(l.planes * l.layers * n * l.heads);
    std::vector<float> v(rows * static_cast<size_t>(l.head_dim));
    for (size_t r = 0; r < rows; ++r) {
        quant::dequantize_row(kv_dtype(l.type), image.data() + r * l.row_bytes(), v.data() + r * static_cast<size_t>(l.head_dim),
                              l.head_dim);
    }
    return v;
}

// The largest error of a lossy round trip (a -> b) in units of what Lossy4
// allows: one 4-bit step of the range it quantizes over (a key channel's
// over the positions, a value row's 32-channel block) plus the cache type's
// own rounding.
double lossy_error(const KvRowLayout& l, const std::vector<float>& a, const std::vector<float>& b, int32_t n) {
    const auto dim = static_cast<size_t>(l.head_dim);
    const size_t width = static_cast<size_t>(l.heads) * dim;  // values of one position in a run
    const auto slots = static_cast<size_t>(n);
    double worst = 0;
    auto check = [&](size_t i, double range) {
        // Q8_0 rows are quantized again: a step of their block's largest value.
        double q8 = 0;
        if (l.type == KvCacheType::Q8_0) {
            for (size_t j = i / 32 * 32; j < i / 32 * 32 + 32; ++j) q8 = std::max(q8, std::fabs(static_cast<double>(a[j])));
            q8 /= 127.0;
        }
        const double bound = range / 15.0 + q8 + 0.01 * std::fabs(a[i]) + 1e-3;
        worst = std::max(worst, std::fabs(a[i] - b[i]) / bound);
    };
    for (int32_t run = 0; run < l.planes * l.layers; ++run) {
        const size_t base = static_cast<size_t>(run) * slots * width;
        if (l.planes == 1 || run < l.layers) {
            for (size_t c = 0; c < width; ++c) {
                float lo = a[base + c], hi = lo;
                for (size_t s = 0; s < slots; ++s) {
                    lo = std::min(lo, a[base + s * width + c]);
                    hi = std::max(hi, a[base + s * width + c]);
                }
                for (size_t s = 0; s < slots; ++s) check(base + s * width + c, hi - lo);
            }
        } else {
            for (size_t blk = base; blk < base + slots * width; blk += 32) {
                const auto [lo, hi] = std::minmax_element(a.begin() + static_cast<std::ptrdiff_t>(blk),
                                                          a.begin() + static_cast<std::ptrdiff_t>(blk + 32));
                for (size_t i = blk; i < blk + 32; ++i) check(i, *hi - *lo);
            }
        }
    }
    return worst;
}

// A store chunk of two pages (kPage positions each), as the engine's chunks
// of whole pages are: its rows are the page images one after the other.
constexpr int32_t kPage = kChunk / 2;

std::vector<uint8_t> chunk_image(const KvRowLayout& l, std::span<const int32_t> tokens, int32_t first, int32_t n) {
    std::vector<uint8_t> all;
    for (int32_t at = 0; at < n; at += kPage) {
        const std::vector<uint8_t> page = kv_image(l, tokens, first + at, std::min(kPage, n - at));
        all.insert(all.end(), page.begin(), page.end());
    }
    return all;
}

// The rows of a context as the store asks for them: one part per page, kept in `keep`.
std::function<std::vector<std::span<const uint8_t>>(int32_t, int32_t)> page_rows(
    const KvRowLayout& l, const std::vector<int32_t>& tokens, std::vector<std::vector<uint8_t>>& keep) {
    return [&l, &tokens, &keep](int32_t chunk, int32_t n) {
        std::vector<std::span<const uint8_t>> parts;
        for (int32_t at = 0; at < n; at += kPage) {
            keep.push_back(kv_image(l, tokens, chunk * kChunk + at, std::min(kPage, n - at)));
        }
        for (size_t k = keep.size() - static_cast<size_t>((n + kPage - 1) / kPage); k < keep.size(); ++k) parts.emplace_back(keep[k]);
        return parts;
    };
}

// Reads chunk `key` of n positions into one buffer through two parts cut
// off the page boundary (so decoding scatters), as a restore's pages and
// scratch are.
Status read_split(KvStore& store, const KvKey& key, std::vector<uint8_t>& image) {
    const size_t cut = image.size() / 3;
    const std::span<uint8_t> parts[] = {std::span<uint8_t>(image).first(cut), std::span<uint8_t>(image).subspan(cut)};
    return store.read_chunk(key, parts);
}

// Lossy restore of a chunk within bounds, page by page.
bool chunk_close(const KvRowLayout& l, const std::vector<uint8_t>& want, const std::vector<uint8_t>& got, int32_t n) {
    size_t at = 0;
    for (int32_t first = 0; first < n; first += kPage) {
        const int32_t m = std::min(kPage, n - first);
        const size_t bytes = l.image_bytes(m);
        const std::span<const uint8_t> a(want.data() + at, bytes), b(got.data() + at, bytes);
        if (lossy_error(l, kv_values(l, a, m), kv_values(l, b, m), m) > 1.0) return false;
        at += bytes;
    }
    return true;
}

}  // namespace

TEST_CASE("KV chunk encodings: lossless gives the bytes back for every cache type, lossy stays within its steps") {
    std::mt19937 rng(11);
    const std::vector<int32_t> tokens = random_tokens(rng, 64);
    ThreadPool pool(4);
    for (const KvCacheType type : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
        for (const int32_t planes : {2, 1}) {
            for (const int32_t n : {64, 37, 1}) {
                const KvRowLayout l = codec_layout(type, planes);
                const std::vector<uint8_t> image = kv_image(l, tokens, 0, n);
                for (ThreadPool* p : {static_cast<ThreadPool*>(nullptr), &pool}) {
                    std::vector<uint8_t> enc;
                    if (!kv_encode(KvEncoding::Lossless, l, n, image, enc, p)) {
                        // Only 4-bit quants, or a single position, may not shrink.
                        CHECK((type == KvCacheType::Q4_0 || type == KvCacheType::Q4_1 || n == 1));
                        continue;
                    }
                    CHECK(enc.size() < image.size());
                    std::vector<uint8_t> back(image.size());
                    REQUIRE(kv_decode(KvEncoding::Lossless, l, n, enc, back, p));
                    CHECK(back == image);
                    // Malformed payloads fail cleanly: cut short, too long, another size.
                    CHECK(!kv_decode(KvEncoding::Lossless, l, n, std::span<const uint8_t>(enc).first(enc.size() - 1), back, p));
                    std::vector<uint8_t> longer = enc;
                    longer.push_back(0);
                    CHECK(!kv_decode(KvEncoding::Lossless, l, n, longer, back, p));
                    if (n > 1) {
                        CHECK(!kv_decode(KvEncoding::Lossless, l, n - 1, enc, std::span<uint8_t>(back).first(l.image_bytes(n - 1)), p));
                    }
                    std::vector<uint8_t> garbage = enc;
                    for (size_t i = garbage.size() / 2; i < garbage.size(); ++i) garbage[i] = static_cast<uint8_t>(rng());
                    (void)kv_decode(KvEncoding::Lossless, l, n, garbage, back, p);  // false or wrong bytes, within bounds
                }
                // Lossy: F16 and Q8_0 caches only.
                std::vector<uint8_t> enc;
                const bool lossy = kv_encode(KvEncoding::Lossy4, l, n, image, enc, &pool);
                // One position does not pay for the keys' ranges.
                CHECK(lossy == ((type == KvCacheType::F16 || type == KvCacheType::Q8_0) && n > 1));
                if (!lossy) continue;
                std::vector<uint8_t> back(image.size());
                REQUIRE(kv_decode(KvEncoding::Lossy4, l, n, enc, back, nullptr));
                CHECK(!kv_decode(KvEncoding::Lossy4, l, n, std::span<const uint8_t>(enc).first(enc.size() - 1), back, nullptr));
                const std::vector<float> a = kv_values(l, image, n), b = kv_values(l, back, n);
                CHECK(lossy_error(l, a, b, n) <= 1.0);
                if (n == 64) {
                    const double ratio = static_cast<double>(image.size()) / static_cast<double>(enc.size());
                    CHECK(ratio > (type == KvCacheType::F16 ? 3.0 : 1.6));
                }
            }
        }
    }
}

TEST_CASE("The KV store keeps chunks compressed on storage, counts their stored size and reads older raw files") {
    const KvRowLayout l = codec_layout(KvCacheType::F16, 2);
    auto config = [&](const std::string& dir, bool layout, bool lossy, ThreadPool* pool) {
        KvStore::Config c;
        const uint64_t parts[] = {77};
        c.seed = kv_seed_key(parts);
        c.chunk_tokens = kChunk;
        c.page_tokens = kPage;
        c.chunk_bytes = l.image_bytes(kChunk);
        c.dir = dir;
        c.disk_bytes = size_t{64} << 20;
        if (layout) c.layout = l;
        c.lossy = lossy;
        c.pool = pool;
        c.costs = KvCodecCosts{1.0, 0.0};  // decoding free: every stream that shrinks is coded
        return c;
    };
    auto put_rows = [&](KvStore& store, const std::vector<int32_t>& tokens) {
        KvStore::ContextSource src;
        src.tokens = tokens;
        std::vector<std::vector<uint8_t>> keep;
        src.rows = page_rows(l, tokens, keep);
        return store.put(src, /*storage_only=*/true);
    };
    // Whether `tokens` are restored: bit for bit, or (lossy) close.
    auto restored = [&](KvStore& store, const std::vector<int32_t>& tokens, bool exact) {
        const KvStore::Plan plan = store.plan(tokens, tokens.size());
        if (plan.positions != static_cast<int32_t>(tokens.size())) return false;
        for (size_t i = 0; i < plan.chunks.size(); ++i) {
            const int32_t n = plan.chunk_sizes[i];
            std::vector<uint8_t> image(l.image_bytes(n));
            if (!read_split(store, plan.chunks[i], image).is_ok()) return false;
            const std::vector<uint8_t> want = chunk_image(l, tokens, static_cast<int32_t>(i) * kChunk, n);
            if (exact && image != want) return false;
            if (!chunk_close(l, want, image, n)) return false;
        }
        return true;
    };
    auto dir_bytes = [](const std::string& dir) {
        size_t bytes = 0;
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            const std::string name = e.path().filename().string();
            if (name.starts_with("kv-")) bytes += e.file_size();
        }
        return bytes;
    };
    std::mt19937 rng(5);
    const std::vector<int32_t> a = random_tokens(rng, 100), b = random_tokens(rng, 70);
    ThreadPool pool(4);
    const std::string dir = store_dir("encoded");
    {
        // An earlier build's raw chunks (no layout)...
        auto raw = KvStore::open(config(dir, false, false, nullptr));
        REQUIRE(raw != nullptr);
        CHECK(put_rows(*raw, a).whole);
        CHECK(raw->stats().chunk_file_bytes > raw->stats().chunk_raw_bytes);  // rows, headers and tokens
    }
    {
        // ... stay readable; new chunks are compressed, the cap counting their files.
        auto store = KvStore::open(config(dir, true, false, &pool));
        REQUIRE(store != nullptr);
        CHECK(restored(*store, a, true));
        const KvStore::Stats before = store->stats();
        CHECK(put_rows(*store, b).whole);
        const KvStore::Stats st = store->stats();
        CHECK(st.chunk_file_bytes - before.chunk_file_bytes < st.chunk_raw_bytes - before.chunk_raw_bytes);
        CHECK(st.disk_bytes == dir_bytes(dir));
        CHECK(restored(*store, b, true));
    }
    {
        // A later engine, decoding on one thread.
        auto store = KvStore::open(config(dir, true, false, nullptr));
        REQUIRE(store != nullptr);
        CHECK(store->stats().removed_files == 0);
        CHECK(restored(*store, a, true));
        CHECK(restored(*store, b, true));
        CHECK(store->stats().disk_bytes == dir_bytes(dir));
    }
    {
        // A lossy store keys its chunks apart: it finds none of the lossless
        // ones, and its own come back close, several times smaller.
        auto lossy = KvStore::open(config(dir, true, true, &pool));
        REQUIRE(lossy != nullptr);
        CHECK(lossy->plan(a, a.size()).positions == 0);
        CHECK(lossy->stats().foreign_files > 0);
        CHECK(put_rows(*lossy, a).whole);
        CHECK(restored(*lossy, a, false));
        const KvStore::Stats st = lossy->stats();
        CHECK(st.chunk_raw_bytes > 2 * st.chunk_file_bytes);
    }
    {
        // ... and a lossless one never gets the lossy chunks.
        auto store = KvStore::open(config(dir, true, false, nullptr));
        REQUIRE(store != nullptr);
        CHECK(store->stats().removed_files == 0);
        CHECK(restored(*store, a, true));
    }
    {
        // A damaged compressed payload is found when read: removed, the read fails.
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            if (e.path().extension() != ".chunk") continue;
            std::fstream f(e.path(), std::ios::in | std::ios::out | std::ios::binary);
            f.seekg(-3, std::ios::end);
            const int c = f.get();
            f.seekp(-3, std::ios::end);
            f.put(static_cast<char>(c ^ 0x5a));
        }
        auto store = KvStore::open(config(dir, true, false, nullptr));
        REQUIRE(store != nullptr);
        const KvStore::Plan plan = store->plan(b, b.size());
        REQUIRE(!plan.chunks.empty());
        std::vector<uint8_t> image(l.image_bytes(plan.chunk_sizes[0]));
        CHECK(read_split(*store, plan.chunks[0], image).code() == ErrorCode::IoError);
        CHECK(store->plan(b, b.size()).positions == 0);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("The KV store codes chunks only where it pays, measures that once, and reads ahead in batches") {
    const KvRowLayout l = codec_layout(KvCacheType::Q8_0, 2);
    ThreadPool pool(4);
    auto config = [&](const std::string& dir, std::optional<KvCodecCosts> costs, const std::string& calibration = {}) {
        KvStore::Config c;
        const uint64_t parts[] = {78};
        c.seed = kv_seed_key(parts);
        c.chunk_tokens = kChunk;
        c.page_tokens = kPage;
        c.chunk_bytes = l.image_bytes(kChunk);
        c.dir = dir;
        c.disk_bytes = size_t{64} << 20;
        c.layout = l;
        c.pool = &pool;
        c.costs = costs;
        c.calibration_dir = calibration;
        return c;
    };
    auto put_rows = [&](KvStore& store, const std::vector<int32_t>& tokens) {
        KvStore::ContextSource src;
        src.tokens = tokens;
        std::vector<std::vector<uint8_t>> keep;
        src.rows = page_rows(l, tokens, keep);
        return store.put(src, /*storage_only=*/true);
    };
    // A restore's reads: read_ahead() over the plan, then read_chunk() in
    // order (`skip`: one chunk left out, as a restore skips rows it keeps).
    auto batch_restored = [&](KvStore& store, const std::vector<int32_t>& tokens, size_t skip = SIZE_MAX) {
        const KvStore::Plan plan = store.plan(tokens, tokens.size());
        if (plan.positions != static_cast<int32_t>(tokens.size())) return false;
        store.read_ahead(plan.chunks);
        for (size_t i = 0; i < plan.chunks.size(); ++i) {
            if (i == skip) continue;
            std::vector<uint8_t> image(l.image_bytes(plan.chunk_sizes[i]));
            if (!read_split(store, plan.chunks[i], image).is_ok()) return false;
            if (image != chunk_image(l, tokens, static_cast<int32_t>(i) * kChunk, plan.chunk_sizes[i])) return false;
        }
        return true;
    };
    std::mt19937 rng(9);
    const std::vector<int32_t> a = random_tokens(rng, 150);
    {
        // Decoding slower than the reads it saves: nothing is coded, the files are raw.
        const std::string dir = store_dir("pays_not");
        auto store = KvStore::open(config(dir, KvCodecCosts{0.1, 50.0}));
        REQUIRE(store != nullptr);
        CHECK(put_rows(*store, a).whole);
        const KvStore::Stats st = store->stats();
        CHECK(st.chunk_file_bytes > st.chunk_raw_bytes);
        CHECK(batch_restored(*store, a));
        store.reset();
        std::filesystem::remove_all(dir);
    }
    {
        // Reads slow, decoding cheap: coded, smaller, and the batch read gives the same rows.
        const std::string dir = store_dir("pays");
        auto store = KvStore::open(config(dir, KvCodecCosts{10.0, 0.1}));
        REQUIRE(store != nullptr);
        CHECK(put_rows(*store, a).whole);
        const KvStore::Stats st = store->stats();
        CHECK(st.chunk_file_bytes < st.chunk_raw_bytes);
        CHECK(batch_restored(*store, a));
        CHECK(batch_restored(*store, a, 0));
        CHECK(batch_restored(*store, a, 3));
        // Reads out of the announced order forget it and still give the rows.
        const KvStore::Plan plan = store->plan(a, a.size());
        store->read_ahead(plan.chunks);
        for (size_t i = plan.chunks.size(); i-- > 0;) {
            std::vector<uint8_t> image(l.image_bytes(plan.chunk_sizes[i]));
            CHECK(read_split(*store, plan.chunks[i], image).is_ok());
            CHECK(image == chunk_image(l, a, static_cast<int32_t>(i) * kChunk, plan.chunk_sizes[i]));
        }
        // A damaged file in the middle: removed, with what follows it; the batch fails.
        std::filesystem::path victim;
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            if (e.path().extension() == ".chunk") victim = e.path();
        }
        {
            std::fstream f(victim, std::ios::in | std::ios::out | std::ios::binary);
            f.seekg(-2, std::ios::end);
            const int ch = f.get();
            f.seekp(-2, std::ios::end);
            f.put(static_cast<char>(ch ^ 0x33));
        }
        CHECK(!batch_restored(*store, a));
        CHECK(store->plan(a, a.size()).positions < static_cast<int32_t>(a.size()));
        store.reset();
        std::filesystem::remove_all(dir);
    }
    {
        // Measured costs are kept: a later store finds the same and makes the same choice.
        const std::string dir = store_dir("measured");
        const std::string calibration = store_dir("measured_costs");
        KvCodecCosts first;
        {
            auto store = KvStore::open(config(dir, std::nullopt, calibration));
            REQUIRE(store != nullptr);
            first = store->codec_costs();
            CHECK(first.decode_ns_per_symbol > 0);
            CHECK(put_rows(*store, a).whole);
            CHECK(batch_restored(*store, a));
        }
        size_t kept = 0;
        for (const auto& e : std::filesystem::directory_iterator(calibration)) {
            kept += e.path().filename().string().starts_with("kv-costs-") ? 1 : 0;
        }
        CHECK(kept == 1);
        for (const auto& e : std::filesystem::directory_iterator(dir)) {
            CHECK(e.path().filename().string() != "kv-calibration.tmp");  // the probe is gone
        }
        {
            auto store = KvStore::open(config(dir, std::nullopt, calibration));
            REQUIRE(store != nullptr);
            CHECK(store->codec_costs().read_ns_per_byte == first.read_ns_per_byte);
            CHECK(store->codec_costs().decode_ns_per_symbol == first.decode_ns_per_symbol);
            CHECK(batch_restored(*store, a));
        }
        std::filesystem::remove_all(dir);
        std::filesystem::remove_all(calibration);
    }
}

// ---------------------------------------------------------------------------
// Per-block layout: windowed layers keep only their window.
// ---------------------------------------------------------------------------
namespace {

// Six layers: 1, 2, 4 and 5 see the last 40 positions, 0 and 3 every one.
KvCacheConfig per_block_config(KvCacheType type = KvCacheType::Q8_0) {
    KvCacheConfig c = small_config(type);
    c.n_layers = 6;
    c.layer_windows = {0, 40, 40, 0, 40, 40};
    return c;
}

// Every layer's rows of positions [from, to), each a function of (layer, position).
void store_rows(KvCache& kv, int32_t from, int32_t to) {
    const KvCacheConfig& c = kv.config();
    const auto width = static_cast<size_t>(c.head_dim) * static_cast<size_t>(c.n_head_kv);
    for (int32_t pos = from; pos < to; ++pos) {
        for (int32_t l = 0; l < c.n_layers; ++l) {
            const std::vector<float> k = row_for(pos * 7 + l * 131, width);
            const std::vector<float> v = row_for(pos * 5 + l * 17 + 1000, width);
            kv.store(l, pos, k.data(), v.data());
        }
    }
}

bool same_row(const KvCache& a, const KvCache& b, int32_t layer, int32_t pos) {
    const size_t row = a.head_row_bytes();
    for (int32_t h = 0; h < a.config().n_head_kv; ++h) {
        if (std::memcmp(a.k_row(layer, pos, h), b.k_row(layer, pos, h), row) != 0) return false;
        if (!a.config().values_in_keys && std::memcmp(a.v_row(layer, pos, h), b.v_row(layer, pos, h), row) != 0) {
            return false;
        }
    }
    return true;
}

}  // namespace

TEST_CASE("Per-block KV: one group keeps the single-table layout, bytes and memory") {
    KvCacheConfig single = small_config(KvCacheType::Q8_0);
    KvCacheConfig zeros = single;
    zeros.layer_windows.assign(static_cast<size_t>(single.n_layers), 0);
    auto a = KvCache::create(single);
    auto b = KvCache::create(zeros);
    REQUIRE(a.has_value() && b.has_value());
    CHECK(!b.value()->has_windowed_layers());
    CHECK(b.value()->layout_fingerprint() == 0);
    CHECK(b.value()->unit_bytes() == a.value()->page_bytes());
    CHECK(b.value()->page_bytes() == a.value()->page_bytes());
    CHECK(b.value()->legacy_page_bytes() == a.value()->page_bytes());
    CHECK(b.value()->window_page_bytes() == 0 && b.value()->window_pages_of(40, {}).empty());
    REQUIRE(a.value()->reserve(40).is_ok());
    REQUIRE(b.value()->reserve(40).is_ok());
    store_rows(*a.value(), 0, 40);
    store_rows(*b.value(), 0, 40);
    for (int32_t p = 0; p < 3; ++p) {
        REQUIRE(b.value()->page_data(p) != nullptr);
        CHECK(std::memcmp(a.value()->page_data(p), b.value()->page_data(p), a.value()->page_bytes()) == 0);
    }
    CHECK(a.value()->bytes_in_use() == b.value()->bytes_in_use());
    CHECK(KvCache::kept_bytes(zeros, 1000, 64) == KvCache::kept_bytes(single, 1000, 64));
    CHECK(KvCache::kept_bytes(single, 1000, 64) == 63 * a.value()->page_bytes());  // ceil(1000 / 16) pages
}

TEST_CASE("Per-block KV: windowed layers keep their window, the others every position, from one pool") {
    const KvCacheConfig c = per_block_config();
    auto made = KvCache::create(c);
    REQUIRE(made.has_value());
    KvCache& kv = *made.value();
    CHECK(kv.has_windowed_layers());
    CHECK(kv.layout_fingerprint() != 0);
    CHECK(kv.window_of_windowed() == 40);
    CHECK(kv.keeps_every_position(0) && kv.keeps_every_position(3) && !kv.keeps_every_position(1));
    // Two groups (2 and 4 layers): pool pages of 2 layers each.
    CHECK(kv.unit_bytes() * 3 == kv.legacy_page_bytes());
    CHECK(kv.page_bytes() == kv.unit_bytes());  // the two layers keeping every position
    CHECK(kv.window_page_bytes() == 2 * kv.unit_bytes());
    CHECK(kv.window_page_bytes() == 2u * 4u * 16u * 2u * kv.head_row_bytes());  // K+V, 4 layers, 16 slots, 2 heads
    int32_t peak = 0;
    const int32_t keep_back = 4;
    for (int32_t pos = 0; pos < 1000; ++pos) {
        REQUIRE(kv.reserve(pos + 1).is_ok());
        store_rows(kv, pos, pos + 1);
        kv.release_unreachable(pos + 1, keep_back);
        peak = std::max(peak, kv.pages_in_use());
    }
    // Every page of the 2 layers keeping every position (63 pool pages), and
    // the window plus the rollback margin of the 4 windowed ones: at most
    // 4 logical pages of 2 pool pages each.
    CHECK(peak <= 63 + 8);
    CHECK(kv.pages_allocated() == peak);  // freed pages were reused by both groups
    KvCacheConfig flat = c;
    flat.layer_windows.clear();
    CHECK(kv.bytes_in_use() <= KvCache::kept_bytes(c, 1000, keep_back));
    CHECK(KvCache::kept_bytes(c, 1000, keep_back) * 2 < KvCache::kept_bytes(flat, 1000, keep_back));
    std::printf("  1000 positions, 4 of 6 layers windowed (40): %d pool pages (%zu KiB) instead of %d (%zu KiB)\n", peak,
                kv.bytes_allocated() / 1024, 63 * 3, 63 * 3 * kv.unit_bytes() / 1024);
    // Rows kept where queries may still look, with their bits.
    auto all = KvCache::create(flat);
    REQUIRE(all.has_value());
    REQUIRE(all.value()->reserve(1000).is_ok());
    store_rows(*all.value(), 0, 1000);
    bool same = true;
    for (int32_t pos = 0; pos < 1000; ++pos) {
        same = same && same_row(kv, *all.value(), 0, pos) && same_row(kv, *all.value(), 3, pos);
    }
    for (int32_t pos = 1000 - 40 - keep_back; pos < 1000; ++pos) {
        for (const int32_t l : {1, 2, 4, 5}) same = same && same_row(kv, *all.value(), l, pos);
    }
    CHECK(same);
    // Rollback within the margin; beyond it the window's rows are gone.
    // Kept: the pages from the one holding 957 (the window of 996, the
    // oldest query after a rollback of 4), i.e. positions from 944 on.
    CHECK(kv.restorable(1000) == 1000);
    CHECK(kv.restorable(990) == 990);
    CHECK(kv.restorable(983) == 983);  // its window starts at 944
    CHECK(kv.restorable(982) == 0);
    CHECK(kv.restorable(900) == 0);
    CHECK(kv.truncate(997).is_ok());
    CHECK(!kv.truncate(900).is_ok());
}

TEST_CASE("Per-block KV: a pinned position keeps its window, so the context returns to it") {
    const KvCacheConfig c = per_block_config();
    auto made = KvCache::create(c);
    auto flat_made = [&] {
        KvCacheConfig flat = c;
        flat.layer_windows.clear();
        return KvCache::create(flat);
    }();
    REQUIRE(made.has_value() && flat_made.has_value());
    KvCache& kv = *made.value();
    KvCache& all = *flat_made.value();
    REQUIRE(all.reserve(600).is_ok());
    store_rows(all, 0, 600);
    for (int32_t pos = 0; pos < 600; ++pos) {
        REQUIRE(kv.reserve(pos + 1).is_ok());
        store_rows(kv, pos, pos + 1);
        if (pos + 1 == 200) {
            const int32_t at[] = {200};
            kv.pin(at);
        }
        kv.release_unreachable(pos + 1, 4);
    }
    // The pin keeps whole pages: those of [161, 200), i.e. positions 160-207.
    CHECK(kv.restorable(250) == 208);
    CHECK(kv.restorable(199) == 199);
    CHECK(kv.restorable(198) == 0);
    const int32_t pin_at[] = {200};
    // Pages of 16: the window of 600 (561-599) and the pin's (161-199).
    CHECK(kv.window_pages_of(600, pin_at) == std::vector<int32_t>({10, 11, 12, 35, 36, 37}));
    std::vector<uint8_t> page(kv.window_page_bytes());
    CHECK(kv.read_window_rows(12, 16, page.data()).is_ok());  // positions 192-207 of the pinned page 12
    CHECK(!kv.read_window_rows(13, 16, page.data()).is_ok());  // released
    // Unpinned, the pages it held go once no query sees them.
    const int32_t before = kv.pages_in_use();
    kv.pin({});
    CHECK(kv.pages_in_use() < before);
    CHECK(kv.restorable(250) == 0);
    // Pinned again (from a fresh run), then back to it: the rows after it are
    // written again as the window moves on, the same bits as a cache keeping
    // every position.
    kv.clear();
    for (int32_t pos = 0; pos < 600; ++pos) {
        REQUIRE(kv.reserve(pos + 1).is_ok());
        store_rows(kv, pos, pos + 1);
        if (pos + 1 == 200) {
            const int32_t at[] = {200};
            kv.pin(at);
        }
        kv.release_unreachable(pos + 1, 4);
    }
    REQUIRE(kv.truncate(200).is_ok());
    CHECK(kv.pins().size() == 1);
    for (int32_t pos = 200; pos < 300; ++pos) {
        REQUIRE(kv.reserve(pos + 1).is_ok());
        store_rows(kv, pos, pos + 1);
        kv.release_unreachable(pos + 1, 4);
    }
    bool same = true;
    for (int32_t pos = 300 - 44; pos < 300; ++pos) {
        for (int32_t l = 0; l < c.n_layers; ++l) same = same && same_row(kv, all, l, pos);
    }
    CHECK(same);
    REQUIRE(kv.truncate(0).is_ok());
    CHECK(kv.pins().empty());
}

TEST_CASE("Per-block KV: rows images, window pages and older pages, bit for bit") {
    for (const bool values_in_keys : {false, true}) {
        KvCacheConfig c = per_block_config();
        if (values_in_keys) {
            c.values_in_keys = true;
            c.value_offset = 32;
        }
        KvCacheConfig flat = c;
        flat.layer_windows.clear();
        auto src_made = KvCache::create(flat);  // keeps every position: the truth
        auto dst_made = KvCache::create(c);
        REQUIRE(src_made.has_value() && dst_made.has_value());
        KvCache& src = *src_made.value();
        KvCache& dst = *dst_made.value();
        const int32_t n = 150;  // 9 pages and a partial one
        REQUIRE(src.reserve(n).is_ok());
        store_rows(src, 0, n);
        // Older saved contexts: the single-table pages convert, the windowed
        // layers keeping the window before n.
        REQUIRE(dst.reserve_context(n, {}).is_ok());
        for (int32_t p = 0; p * c.page_tokens < n; ++p) REQUIRE(dst.write_legacy_page(p, src.page_data(p)).is_ok());
        bool same = true;
        for (int32_t pos = 0; pos < n; ++pos) same = same && same_row(src, dst, 0, pos) && same_row(src, dst, 3, pos);
        for (int32_t pos = n - 39; pos < n; ++pos) {
            for (const int32_t l : {1, 2, 4, 5}) same = same && same_row(src, dst, l, pos);
        }
        CHECK(same);
        CHECK(dst.restorable(n) == n);
        CHECK(dst.restorable(100) == 0);  // the window before 100 starts in a page never mapped
        // Rows images: only the layers keeping every position, laid out
        // [plane][layer][slot][head][row] whatever the pool pages.
        CHECK(dst.page_bytes() * 3 == src.page_bytes());
        std::vector<uint8_t> image(dst.rows_bytes(70));
        REQUIRE(dst.read_rows_at(9, 61, image.data(), 70, 3).is_ok());
        for (int32_t i = 0; i < 61; ++i) {
            for (const int32_t l : {0, 3}) {
                for (int32_t h = 0; h < c.n_head_kv; ++h) {
                    same = same && std::memcmp(image.data() + dst.image_key_offset(70, l, 3 + i, h), src.k_row(l, 9 + i, h),
                                               dst.head_row_bytes()) == 0;
                }
            }
        }
        CHECK(same);
        // Window pages, as saved contexts and the KV store keep them: a page's
        // image holds zeros past the position it ends at; the pages a context
        // returning to n keeps, written whole into another cache, give the
        // rows of every position in them.
        const std::vector<int32_t> kept = dst.window_pages_of(n, {});
        CHECK(kept == std::vector<int32_t>({6, 7, 8, 9}));  // 111..149 over pages of 16
        std::vector<std::vector<uint8_t>> pages;
        for (int32_t q = 0; q * c.page_tokens < n; ++q) {
            pages.emplace_back(dst.window_page_bytes(), 0xEE);
            const int32_t rows = std::min(c.page_tokens, n - q * c.page_tokens);
            if (dst.window_page_mapped(q)) {
                REQUIRE(dst.read_window_rows(q, rows, pages.back().data()).is_ok());
                if (rows < c.page_tokens) CHECK(pages.back().back() == 0);
            } else {
                CHECK(!dst.read_window_rows(q, rows, pages.back().data()).is_ok());
            }
        }
        auto restored = KvCache::create(c);
        REQUIRE(restored.has_value());
        REQUIRE(restored.value()->reserve_context(n, {}).is_ok());
        for (const int32_t q : kept) {
            REQUIRE(restored.value()->window_page_mapped(q));
            REQUIRE(restored.value()->write_window_rows(q, pages[static_cast<size_t>(q)].data(), 0,
                                                        std::min(c.page_tokens, n - q * c.page_tokens)).is_ok());
        }
        CHECK(!restored.value()->window_page_mapped(5));
        for (int32_t pos = kept.front() * c.page_tokens; pos < n; ++pos) {
            for (const int32_t l : {1, 2, 4, 5}) same = same && same_row(src, *restored.value(), l, pos);
        }
        CHECK(same);
        // Pinned positions keep their windows too.
        const int32_t pins[] = {40};
        CHECK(dst.window_pages_of(n, pins) == std::vector<int32_t>({0, 1, 2, 6, 7, 8, 9}));
        CHECK(!dst.reserve_context(n, std::vector<int32_t>{n + 1}).is_ok());
    }
}

TEST_CASE("Per-block KV: key scaling converts the rows each layer keeps") {
    const KvCacheConfig c = per_block_config();
    auto made = KvCache::create(c);
    REQUIRE(made.has_value());
    KvCache& kv = *made.value();
    for (int32_t pos = 0; pos < 200; ++pos) {
        REQUIRE(kv.reserve(pos + 1).is_ok());
        store_rows(kv, pos, pos + 1);
        kv.release_unreachable(pos + 1, 4);
    }
    std::vector<float> factors(static_cast<size_t>(c.n_layers) * c.n_head_kv * c.head_dim, 2.0f);
    std::vector<float> before = read_k(kv, 1, 190, 1);
    REQUIRE(kv.scale_keys(200, factors).is_ok());  // windowed rows that were released are skipped
    std::vector<float> after = read_k(kv, 1, 190, 1);
    for (size_t i = 0; i < before.size(); ++i) CHECK_NEAR(after[i], 2.0f * before[i], 0.05f);
    CHECK(!kv.scale_keys(200, std::vector<float>(3, 1.0f)).is_ok());
}

TEST_CASE("Per-block KV: the windows' share is bounded in the memory a context is sized with") {
    // A Gemma 3 4B-like layout: 34 blocks, one in six global, 4 KV heads of
    // 256, a window of 1024; a Gemma 2 2B-like one: 26 blocks, every other one
    // windowed (4096).
    struct Layout {
        const char* name;
        int32_t layers, period, window;
    };
    for (const Layout& l : {Layout{"gemma3-4b", 34, 6, 1024}, Layout{"gemma2-2b", 26, 2, 4096}}) {
        KvCacheConfig c;
        c.n_layers = l.layers;
        c.n_head_kv = 4;
        c.head_dim = 256;
        c.type = KvCacheType::Q8_0;
        c.max_positions = 32768;
        c.page_tokens = 64;
        for (int32_t i = 0; i < l.layers; ++i) c.layer_windows.push_back((i + 1) % l.period == 0 ? 0 : l.window);
        KvCacheConfig flat = c;
        flat.layer_windows.clear();
        const size_t budget = size_t{256} << 20;
        auto longest = [&](const KvCacheConfig& cfg) {
            int32_t lo = 0, hi = 1 << 20;
            while (lo < hi) {
                const int32_t mid = lo + (hi - lo + 1) / 2;
                if (KvCache::kept_bytes(cfg, mid, 64) <= budget) {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            return lo;
        };
        const int32_t before = longest(flat), after = longest(c);
        CHECK(after > before);
        std::printf("  %s: %.1f KiB/token kept for every position (was %.1f); %d -> %d positions in 256 MiB\n", l.name,
                    static_cast<double>(KvCache::create(c).value()->rows_bytes(1)) / 1024.0,
                    static_cast<double>(KvCache::bytes_per_token(flat)) / 1024.0, before, after);
    }
}
