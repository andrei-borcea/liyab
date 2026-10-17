// Backend micro-benchmarks: per-matmul latency on CPU and GPU backends for
// the shapes of a real model, plus the fixed dispatch cost of each backend.
//
//   LIYAB_BENCH_MODEL=tinyllama-q4_0.gguf ./test_backends
//
// Without LIYAB_BENCH_MODEL the shapes come from a small synthetic model.

#include <atomic>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "core/portable_math.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

struct Named {
    std::string name;
    std::unique_ptr<Backend> backend;
};

std::vector<Named> backends(ThreadPool& pool) {
    std::vector<Named> out;
    out.push_back({"cpu", make_cpu_backend(pool)});
    if (auto m = make_metal_backend()) out.push_back({"metal", std::move(m).value()});
    if (auto v = make_vulkan_backend()) out.push_back({"vulkan", std::move(v).value()});
    return out;
}

// Median-free mean latency of `iters` calls. The first call uploads weights
// on GPUs; then the backend is kept busy for ~0.5 s so mobile DVFS governors
// raise the clocks before measuring (otherwise numbers vary by 10x).
double time_matmul(Backend& b, const TensorView& w, int32_t n, int iters) {
    std::vector<float> x(static_cast<size_t>(w.cols() * n), 0.5f), y(static_cast<size_t>(w.rows() * n));
    if (!b.matmul(w, x.data(), y.data(), n).is_ok()) return -1.0;
    for (const auto warm = Clock::now(); ms_since(warm) < 500.0;) (void)b.matmul(w, x.data(), y.data(), n);
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) (void)b.matmul(w, x.data(), y.data(), n);
    return ms_since(t0) / iters;
}

}  // namespace

TEST_CASE("[bench] per-matmul latency by backend on the model's shapes") {
    std::string path;
    if (const char* env = std::getenv("LIYAB_BENCH_MODEL")) {
        path = env;
    } else {
        path = test::temp_dir() + "/liyab_bb_" + std::to_string(test::process_id()) + ".gguf";
        test::TinyModelSpec spec;
        spec.n_embd = 512;
        spec.n_ff = 1408;
        spec.n_head = 8;
        test::write_tiny_model(path, spec);
    }
    auto file = MmapLoader::open(path);
    REQUIRE(file.has_value());
    ThreadPool pool(4);
    auto list = backends(pool);

    // One tensor per distinct role in block 0, plus the output head.
    const char* roles[] = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_output.weight",
                           "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "output.weight"};
    std::printf("  %-26s %-14s", "tensor", "shape");
    for (const auto& b : list) std::printf(" %10s", b.name.c_str());
    std::printf("   (ms per call, n=1)\n");
    std::map<std::string, double> per_block;  // backend -> ms for one block's 7 matmuls (approx.)
    for (const char* role : roles) {
        const TensorView* w = file.value()->tensor(role);
        if (w == nullptr) continue;
        char shape[32];
        std::snprintf(shape, sizeof shape, "%lldx%lld %s", static_cast<long long>(w->rows()),
                      static_cast<long long>(w->cols()), std::string(dtype_traits(w->type).name).c_str());
        std::printf("  %-26s %-14s", role, shape);
        for (auto& b : list) {
            const double ms = time_matmul(*b.backend, *w, 1, 100);
            std::printf(" %10.3f", ms);
            const std::string r = role;
            const int count = r.find("ffn_gate") != std::string::npos ? 2  // gate + up
                              : r.find("attn_k") != std::string::npos ? 2  // k + v
                              : r.find("output.weight") == 0 ? 0 : 1;
            per_block[b.name] += ms * count;
        }
        std::printf("\n");
    }
    // Grouped submissions as the transformer issues them.
    const TensorView* q = file.value()->tensor("blk.0.attn_q.weight");
    const TensorView* k = file.value()->tensor("blk.0.attn_k.weight");
    const TensorView* v = file.value()->tensor("blk.0.attn_v.weight");
    const TensorView* g = file.value()->tensor("blk.0.ffn_gate.weight");
    const TensorView* u = file.value()->tensor("blk.0.ffn_up.weight");
    if (q && k && v && g && u) {
        auto time_group = [&](Backend& b, std::vector<const TensorView*> ws) {
            std::vector<float> x(static_cast<size_t>(ws[0]->cols()), 0.5f);
            std::vector<std::vector<float>> out;
            std::vector<float*> ys;
            for (const TensorView* w : ws) out.emplace_back(static_cast<size_t>(w->rows()));
            for (auto& o : out) ys.push_back(o.data());
            for (const auto warm = Clock::now(); ms_since(warm) < 300.0;) (void)b.matmul_group(ws, x.data(), ys, 1);
            const auto t0 = Clock::now();
            for (int i = 0; i < 100; ++i) (void)b.matmul_group(ws, x.data(), ys, 1);
            return ms_since(t0) / 100;
        };
        std::printf("  %-26s %-14s", "grouped q+k+v", "1 submit");
        for (auto& b : list) std::printf(" %10.3f", time_group(*b.backend, {q, k, v}));
        std::printf("\n  %-26s %-14s", "grouped gate+up", "1 submit");
        for (auto& b : list) std::printf(" %10.3f", time_group(*b.backend, {g, u}));
        std::printf("\n");
    }
    // Fixed cost: a 32x32 F32 matmul is pure dispatch / synchronization.
    std::vector<float> tiny(32 * 32, 0.01f);
    TensorView t;
    t.name = "tiny";
    t.type = DType::F32;
    t.n_dims = 2;
    t.ne = {32, 32, 1, 1};
    t.data = reinterpret_cast<const uint8_t*>(tiny.data());
    t.nbytes = tiny.size() * sizeof(float);
    std::printf("  %-26s %-14s", "dispatch overhead", "32x32 f32");
    for (auto& b : list) std::printf(" %10.3f", time_matmul(*b.backend, t, 1, 200));
    std::printf("\n  one block's 7 matmuls     %-14s", "");
    for (auto& b : list) std::printf(" %10.3f", per_block[b.name]);
    std::printf("\n");
    if (std::getenv("LIYAB_BENCH_MODEL") == nullptr) std::remove(path.c_str());
}

TEST_CASE("CPU matmul_batch over mixed formats and shared inputs equals one matmul per item") {
    // Weights in several formats, quantized from the same random floats.
    const int64_t cols = 256;
    const DType types[] = {DType::F32, DType::F16, DType::Q8_0, DType::Q4_0, DType::Q4_1};
    std::vector<std::vector<uint8_t>> storage;
    std::vector<TensorView> ws;
    uint32_t seed = 1;
    auto rnd = [&] {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8) / static_cast<float>(1u << 24) - 0.5f;
    };
    for (size_t i = 0; i < 10; ++i) {
        const DType type = types[i % (sizeof types / sizeof types[0])];
        const int64_t rows = 3 + static_cast<int64_t>(i) * 7;  // uneven, so chunks straddle matrices
        TensorView w;
        w.type = type;
        w.n_dims = 2;
        w.ne = {cols, rows, 1, 1};
        storage.emplace_back(w.row_bytes() * static_cast<size_t>(rows));
        std::vector<float> row(static_cast<size_t>(cols));
        for (int64_t r = 0; r < rows; ++r) {
            for (float& v : row) v = rnd();
            quant::quantize_row(type, row.data(), storage.back().data() + static_cast<size_t>(r) * w.row_bytes(), cols);
        }
        w.data = storage.back().data();
        w.nbytes = storage.back().size();
        ws.push_back(w);
    }
    // Two inputs of two rows each; even items share the first, odd the second.
    std::vector<float> x0(static_cast<size_t>(2 * cols)), x1(static_cast<size_t>(2 * cols));
    for (float& v : x0) v = rnd();
    for (float& v : x1) v = rnd();

    ThreadPool pool(4);
    auto cpu = make_cpu_backend(pool);
    std::vector<std::vector<float>> batched(ws.size()), single(ws.size());
    std::vector<Backend::MatmulItem> items;
    for (size_t i = 0; i < ws.size(); ++i) {
        batched[i].resize(static_cast<size_t>(2 * ws[i].rows()));
        single[i].resize(batched[i].size());
        items.push_back({&ws[i], i % 2 == 0 ? x0.data() : x1.data(), batched[i].data(), 2});
    }
    REQUIRE(cpu->matmul_batch(items).is_ok());
    ThreadPool one(1);
    auto serial = make_cpu_backend(one);
    for (size_t i = 0; i < ws.size(); ++i) {
        REQUIRE(serial->matmul(ws[i], items[i].x, single[i].data(), 2).is_ok());
        for (size_t j = 0; j < single[i].size(); ++j) CHECK(batched[i][j] == single[i][j]);
    }
}

TEST_CASE("ThreadPool::parallel_for runs every index exactly once") {
    // Chunks are claimed dynamically: back-to-back jobs of every size (fewer
    // indices than threads, uneven splits, many chunks), with the active
    // thread count changing in between, must neither skip nor repeat work.
    ThreadPool pool(6);
    std::vector<std::atomic<int32_t>> hits(5000);
    for (int32_t round = 0; round < 2000; ++round) {
        if (round % 7 == 0) pool.set_active_threads(1 + round % 6);
        const int64_t n = 1 + (round * 37) % static_cast<int32_t>(hits.size());
        for (int64_t i = 0; i < n; ++i) hits[static_cast<size_t>(i)].store(0, std::memory_order_relaxed);
        pool.parallel_for(n, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) hits[static_cast<size_t>(i)].fetch_add(1, std::memory_order_relaxed);
        });
        int32_t wrong = 0;
        for (int64_t i = 0; i < n; ++i) wrong += hits[static_cast<size_t>(i)].load(std::memory_order_relaxed) != 1;
        REQUIRE(wrong == 0);
    }
}

TEST_CASE("portable_tanh: within 1.32 ulp of tanh, odd, monotone, saturating, and the same bits everywhere") {
    // Arguments: a uniform sweep over [-12, 12], every power-of-two decade
    // down to 1e-30, and a dense run around the branch point 0.625.
    std::vector<float> xs;
    for (int i = -1200000; i <= 1200000; ++i) xs.push_back(static_cast<float>(i) * 1e-5f);
    for (float v = 1e-30f; v < 12.0f; v *= 1.0001f) {
        xs.push_back(v);
        xs.push_back(-v);
    }
    float near = 0.625f;
    for (int i = 0; i < 4000; ++i) near = std::nextafter(near, 0.0f);
    for (int i = 0; i < 8000; ++i, near = std::nextafter(near, 1.0f)) xs.push_back(near);
    double worst = 0.0;
    bool odd = true;
    std::vector<float> ys(xs.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        const float y = portable_tanh(xs[i]);
        ys[i] = y;
        const double ref = std::tanh(static_cast<double>(xs[i]));
        const float mag = static_cast<float>(std::fabs(ref));
        const double ulp = static_cast<double>(std::nextafter(mag, 2.0f) - mag);
        if (ref != 0.0) worst = std::max(worst, std::fabs(static_cast<double>(y) - ref) / ulp);
        odd = odd && portable_tanh(-xs[i]) == -y;
    }
    std::printf("  portable_tanh: worst error %.4f ulp over %zu arguments\n", worst, xs.size());
    CHECK(worst <= 1.32);
    CHECK(odd);
    std::vector<float> sorted = xs;
    std::sort(sorted.begin(), sorted.end());
    bool monotone = true;
    for (size_t i = 1; i < sorted.size(); ++i) monotone = monotone && portable_tanh(sorted[i - 1]) <= portable_tanh(sorted[i]);
    CHECK(monotone);
    CHECK(portable_tanh(10.0f) == 1.0f);
    CHECK(portable_tanh(-1e30f) == -1.0f);
    CHECK(portable_tanh(INFINITY) == 1.0f);
    CHECK(portable_tanh(-INFINITY) == -1.0f);
    CHECK(std::isnan(portable_tanh(NAN)));
    CHECK(portable_tanh(0.0f) == 0.0f);
    CHECK(std::signbit(portable_tanh(-0.0f)));
    // The point of the function: these bits are the same on every platform
    // and compiler (pinned on the Mac; the phone and x86-64 must give them).
    const uint64_t h = xxhash64(ys.data(), ys.size() * sizeof(float));
    std::printf("  portable_tanh: results hash %016llx\n", static_cast<unsigned long long>(h));
    CHECK(h == 0x1f63ba73ed842e5bull);
}

TEST_CASE("Final logit soft-cap: same bits for 1 and N threads, and its cost per token") {
    // Gemma 2's vocabulary (256000) plus odd sizes around the inline/parallel
    // cut-over, so uneven chunks are covered too.
    ThreadPool one(1);
    ThreadPool many(8);
    const float cap = 30.0f;
    for (const size_t n : {size_t{7}, size_t{16383}, size_t{16384}, size_t{100003}, size_t{256000}}) {
        std::vector<float> src(n);
        uint32_t seed = 12345;
        for (float& v : src) {
            seed = seed * 1664525u + 1013904223u;
            v = (static_cast<float>(seed >> 8) / 16777216.0f - 0.5f) * 200.0f;  // well past the cap both ways
        }
        std::vector<float> want = src;
        for (float& v : want) v = cap * portable_tanh(v / cap);  // the single-thread loop
        for (ThreadPool* pool : {&one, &many}) {
            for (const int32_t active : {1, 3, 8}) {
                pool->set_active_threads(active);
                std::vector<float> got = src;
                apply_final_softcap(got, cap, *pool);
                REQUIRE(std::memcmp(got.data(), want.data(), n * sizeof(float)) == 0);
            }
        }
    }
    // Cost per token on a 256k vocabulary (best of 50), before (one thread)
    // and after (the pool).
    std::vector<float> buf(256000, 3.0f);
    auto best = [&](ThreadPool& pool) {
        double b = 1e9;
        for (int i = 0; i < 50; ++i) {
            const auto t = Clock::now();
            apply_final_softcap(buf, cap, pool);
            b = std::min(b, ms_since(t));
        }
        return b;
    };
    many.set_active_threads(8);
    const double t1 = best(one);
    const double tn = best(many);
    std::printf("  final soft-cap, 256000 logits: %.3f ms on 1 thread, %.3f ms on %d threads\n", t1, tn,
                many.max_threads());
}

TEST_MAIN()
