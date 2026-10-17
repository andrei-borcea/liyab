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

#include "core/kv_store.h"
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
    src.rows = [&tokens](int32_t chunk, int32_t n, uint8_t* out) {
        for (int32_t s = 0; s < n; ++s) {
            const int32_t pos = chunk * kChunk + s;
            for (size_t b = 0; b < kPosBytes; ++b) out[static_cast<size_t>(s) * kPosBytes + b] = row_byte(tokens[static_cast<size_t>(pos)], pos, b);
        }
        return Status::ok();
    };
    src.states = std::move(marks);
    src.state = [&tokens](int32_t pos, float* out) {
        const std::vector<float> s = state_image(tokens, pos);
        std::copy(s.begin(), s.end(), out);
        return Status::ok();
    };
    return store.put(src, storage_only);
}

// Whether the store gives back the right bytes for the first `n` positions of `tokens` along `plan`.
bool rows_match(KvStore& store, const KvStore::Plan& plan, const std::vector<int32_t>& tokens) {
    for (size_t i = 0; i < plan.chunks.size(); ++i) {
        const int32_t n = plan.chunk_sizes[i];
        std::vector<uint8_t> image(static_cast<size_t>(n) * kPosBytes);
        if (!store.read_chunk(plan.chunks[i], image).is_ok()) return false;
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
    REQUIRE(store->read_state(p.state, state).is_ok());
    CHECK(state == state_image(x, 70));
    REQUIRE(store->read_state(p.snapshots[0].key, state).is_ok());
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
        REQUIRE(store->read_state(p.state, state).is_ok());
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
