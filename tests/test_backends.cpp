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
#include "core/route_picker.h"
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

// Runs the picker's whole schedule for a class of passes of `tokens` tokens with the given times per token.
static RoutePicker::Arm run_picker(RoutePicker& picker, size_t tokens, double host_ms, double accelerator_ms,
                                   int* passes = nullptr) {
    int count = 0;
    while (picker.measuring(tokens) && count < 20) {
        const RoutePicker::Arm arm = picker.next(tokens);
        picker.report(tokens, arm, arm == RoutePicker::Arm::Host ? host_ms : accelerator_ms);
        ++count;
    }
    if (passes != nullptr) *passes = count;
    return picker.next(tokens);
}

TEST_CASE("The route picker measures each class of pass once and keeps the faster route") {
    using Arm = RoutePicker::Arm;
    RoutePicker picker;
    int passes = 0;
    CHECK(picker.next(1) == Arm::Host);  // the schedule of steps starts on the CPU
    CHECK(run_picker(picker, 1, 10.0, 100.0, &passes) == Arm::Host);  // decoding: the CPU is faster
    CHECK(passes == 5);
    CHECK(picker.next(64) == Arm::Accelerator);  // a block is a class of its own, not yet decided; it starts on the GPU
    CHECK(picker.measuring(64));
    CHECK(run_picker(picker, 64, 20.0, 5.0, &passes) == Arm::Accelerator);  // prompts: the accelerator is faster
    CHECK(passes == 3);  // a block's CPU pass is costly: measured once
    CHECK(picker.next(1) == Arm::Host);  // the first decision stands
    CHECK(picker.next(8) == Arm::Host);  // a speculative verification pass is a step
    CHECK(!picker.measuring(1));

    RoutePicker close;  // within the margin the CPU, the numerical reference, keeps the work
    CHECK(run_picker(close, 1, 10.0, 9.0) == Arm::Host);
    RoutePicker clear;
    CHECK(run_picker(clear, 1, 10.0, 7.9) == Arm::Accelerator);

    RoutePicker warm;  // the accelerator's first pass wraps the weights and is not counted
    for (int i = 0; i < 5; ++i) {
        const Arm arm = warm.next(1);
        const bool first_accelerator = arm == Arm::Accelerator && i == 1;
        warm.report(1, arm, arm == Arm::Host ? 10.0 : (first_accelerator ? 1000.0 : 2.0));
    }
    CHECK(warm.next(1) == Arm::Accelerator);

    RoutePicker short_blocks;  // 17 tokens is the shortest block measured; a pass in between is a step
    CHECK(short_blocks.next(RoutePicker::kMinBlockTokens) == Arm::Accelerator);

    // While a block class is measured the CPU takes a short slice of a chunk; the accelerator, and the CPU once the
    // class is decided or for a step or a short block, take all of it.
    RoutePicker probes;
    CHECK(probes.limit(256, Arm::Host) == RoutePicker::kHostProbeTokens);
    CHECK(probes.limit(256, Arm::Accelerator) == 256);
    CHECK(probes.limit(RoutePicker::kHostProbeTokens, Arm::Host) == RoutePicker::kHostProbeTokens);
    CHECK(probes.limit(1, Arm::Host) == 1);
    CHECK(run_picker(probes, 256, 1.0, 5.0) == Arm::Host);  // decided: the CPU keeps the whole chunk
    CHECK(probes.limit(256, Arm::Host) == 256);
}

TEST_CASE("Metal K-quants match dequantized float reference in refillable shared regions") {
    auto made = make_metal_backend();
    if (!made) {
        std::printf("  skipped: %s\n", made.status().to_string().c_str());
        return;
    }
    auto& metal = *made.value();
    uint32_t seed = 97;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return seed; };
    for (DType type : {DType::Q4_K, DType::Q5_K, DType::Q6_K}) {
        REQUIRE(metal.supports(type));
        for (int64_t cols : {256, 768}) {
            const int64_t rows = 7;
            TensorView w;
            w.type = type;
            w.n_dims = 2;
            w.ne = {cols, rows, 1, 1};
            w.nbytes = w.row_bytes() * static_cast<size_t>(rows);
            uint8_t* region = metal.allocate_shared(w.nbytes + 64);
            REQUIRE(region != nullptr);
            w.data = region + 32;  // GGUF alignment, not page aligned
            for (int refill = 0; refill < 2; ++refill) {
                for (int64_t row = 0; row < rows; ++row) {
                    uint8_t* dst = region + 32 + row * w.row_bytes();
                    if (type == DType::Q6_K) {
                        for (int64_t b = 0; b < cols / 256; ++b) {
                            quant::BlockQ6_K block{};
                            for (auto& v : block.ql) v = static_cast<uint8_t>(rnd() >> 16);
                            for (auto& v : block.qh) v = static_cast<uint8_t>(rnd() >> 16);
                            for (auto& v : block.scales) v = static_cast<int8_t>(static_cast<int>(rnd() % 255) - 127);
                            block.d = quant::fp32_to_fp16(0.0007f);
                            std::memcpy(dst + b * sizeof block, &block, sizeof block);
                        }
                    } else {
                        std::vector<float> values(static_cast<size_t>(cols));
                        for (int64_t j = 0; j < cols; ++j) values[static_cast<size_t>(j)] =
                            (static_cast<float>(rnd() >> 8) / 16777216.0f - 0.4f) * (1 + j / 32);
                        quant::quantize_row(type, values.data(), dst, cols);
                    }
                }
                std::vector<float> decoded(static_cast<size_t>(cols * rows));
                for (int64_t row = 0; row < rows; ++row) quant::dequantize_row(type,
                    w.data + row * w.row_bytes(), decoded.data() + row * cols, cols);
                for (int n : {1, 3}) {
                    std::vector<float> x(static_cast<size_t>(n * cols)), result(static_cast<size_t>(n * rows));
                    for (float& v : x) v = static_cast<float>(rnd() >> 8) / 16777216.0f - 0.5f;
                    REQUIRE(metal.matmul(w, x.data(), result.data(), n).is_ok());
                    std::vector<float> grouped_a(result.size()), grouped_b(result.size());
                    const TensorView* together[] = {&w, &w};
                    float* outputs[] = {grouped_a.data(), grouped_b.data()};
                    REQUIRE(metal.matmul_group(together, x.data(), outputs, n).is_ok());
                    CHECK(grouped_a == result);
                    CHECK(grouped_b == result);
                    for (int t = 0; t < n; ++t) for (int64_t row = 0; row < rows; ++row) {
                        double reference = 0;
                        for (int64_t j = 0; j < cols; ++j) reference +=
                            static_cast<double>(decoded[static_cast<size_t>(row * cols + j)]) * x[static_cast<size_t>(t * cols + j)];
                        CHECK(std::fabs(result[static_cast<size_t>(t * rows + row)] - reference) < 2e-4 * (1 + std::fabs(reference)));
                    }
                }
            }
            metal.free_shared(region);
        }
    }
    CHECK(metal.busy_ms() > 0);
}

TEST_CASE("Metal fused SiLU FFN matches separate K-quant projections and supports in-place output") {
    auto made = make_metal_backend();
    if (!made) return;
    auto& metal = *made.value();
    constexpr int64_t width = 256, hidden = 256, output = 128;
    uint32_t seed = 703;
    auto rnd = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8) / 16777216.0f - 0.5f;
    };
    auto make_weight = [&](int64_t cols, int64_t rows, uint8_t* storage) {
        TensorView w;
        w.type = DType::Q4_K;
        w.n_dims = 2;
        w.ne = {cols, rows, 1, 1};
        w.nbytes = w.row_bytes() * static_cast<size_t>(rows);
        w.data = storage;
        std::vector<float> values(static_cast<size_t>(cols));
        for (int64_t row = 0; row < rows; ++row) {
            for (float& v : values) v = rnd();
            quant::quantize_row(w.type, values.data(), storage + row * w.row_bytes(), cols);
        }
        return w;
    };
    TensorView shape;
    shape.type = DType::Q4_K;
    shape.n_dims = 2;
    shape.ne = {width, hidden, 1, 1};
    const size_t hidden_weight_bytes = shape.row_bytes() * static_cast<size_t>(hidden);
    const size_t output_weight_bytes = shape.row_bytes() * static_cast<size_t>(output);
    uint8_t* gate_storage = metal.allocate_shared(hidden_weight_bytes);
    uint8_t* up_storage = metal.allocate_shared(hidden_weight_bytes);
    uint8_t* down_storage = metal.allocate_shared(output_weight_bytes);
    REQUIRE(gate_storage != nullptr && up_storage != nullptr && down_storage != nullptr);
    TensorView gate = make_weight(width, hidden, gate_storage);
    TensorView up = make_weight(width, hidden, up_storage);
    TensorView down = make_weight(hidden, output, down_storage);
    for (int32_t n : {1, 3}) {
        std::vector<float> x(static_cast<size_t>(n * width));
        for (float& v : x) v = rnd();
        std::vector<float> g(static_cast<size_t>(n * hidden)), u(g.size()), reference(static_cast<size_t>(n * output));
        const TensorView* pair[] = {&gate, &up};
        float* products[] = {g.data(), u.data()};
        REQUIRE(metal.matmul_group(pair, x.data(), products, n).is_ok());
        for (size_t i = 0; i < g.size(); ++i) g[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
        REQUIRE(metal.matmul(down, g.data(), reference.data(), n).is_ok());

        std::vector<float> fused(reference.size());
        REQUIRE(metal.gated_ffn_silu(gate, up, down, x.data(), fused.data(), n).is_ok());
        std::vector<float> in_place = x;
        REQUIRE(metal.gated_ffn_silu(gate, up, down, in_place.data(), in_place.data(), n).is_ok());
        for (size_t i = 0; i < reference.size(); ++i) {
            CHECK(std::fabs(fused[i] - reference[i]) < 2e-4f * (1.0f + std::fabs(reference[i])));
            CHECK(fused[i] == in_place[i]);
        }
    }
    metal.free_shared(gate_storage);
    metal.free_shared(up_storage);
    metal.free_shared(down_storage);
}

// Causal attention of a prompt block through the backend, against the same sum in double from the cache's own rows.
TEST_CASE("Metal prompt-block attention matches a double-precision reference over F16 and Q8_0 caches") {
    auto made = make_metal_backend();
    if (!made) {
        std::printf("  skipped: %s\n", made.status().to_string().c_str());
        return;
    }
    auto& metal = *made.value();
    uint32_t seed = 4242;
    auto uniform = [&]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8) / static_cast<float>(1 << 24) * 2.0f - 1.0f;
    };
    struct Shape {
        int32_t head_dim, n_head, n_head_kv, n_past, n;
        float softcap;
    };
    const Shape shapes[] = {{128, 6, 2, 37, 53, 0.0f},  {64, 4, 4, 0, 16, 0.0f},   {96, 6, 3, 100, 33, 0.0f},
                            {256, 4, 2, 21, 40, 0.0f},  {128, 5, 1, 300, 64, 50.0f}};
    for (const KvCacheType type : {KvCacheType::F16, KvCacheType::Q8_0}) {
        for (const Shape& sh : shapes) {
            KvCacheConfig config;
            config.n_layers = 2;
            config.n_head_kv = sh.n_head_kv;
            config.head_dim = sh.head_dim;
            config.type = type;
            config.max_positions = 1024;
            config.page_tokens = 16;
            auto cache = KvCache::create(config);
            REQUIRE(cache.has_value());
            KvCache& kv = *cache.value();
            const int32_t total = sh.n_past + sh.n;
            REQUIRE(kv.reserve(total).is_ok());
            const size_t kv_dim = static_cast<size_t>(sh.n_head_kv) * static_cast<size_t>(sh.head_dim);
            std::vector<float> k(kv_dim), v(kv_dim);
            for (int32_t pos = 0; pos < total; ++pos) {
                for (float& x : k) x = uniform() * 2.0f;
                for (float& x : v) x = uniform();
                kv.store(1, pos, k.data(), v.data());
            }
            const size_t q_dim = static_cast<size_t>(sh.n_head) * static_cast<size_t>(sh.head_dim);
            std::vector<float> q(static_cast<size_t>(sh.n) * q_dim), out(q.size(), -1.0f);
            for (float& x : q) x = uniform();
            const KvCache::RowView rows = kv.row_view(1);
            Backend::AttentionJob job;
            job.rows = &rows;
            job.kv_type = kv.dtype();
            job.head_dim = sh.head_dim;
            job.n_head = sh.n_head;
            job.n_head_kv = sh.n_head_kv;
            job.n_past = sh.n_past;
            job.n = sh.n;
            job.scale = 1.0f / std::sqrt(static_cast<float>(sh.head_dim));
            job.softcap = sh.softcap;
            job.q = q.data();
            job.out = out.data();
            REQUIRE(metal.attention(job).is_ok());

            const int32_t group = sh.n_head / sh.n_head_kv;
            double worst = 0.0;
            std::vector<float> krow(static_cast<size_t>(sh.head_dim)), vrow(krow.size());
            for (int32_t t = 0; t < sh.n; ++t) {
                const int32_t last = sh.n_past + t;
                for (int32_t h = 0; h < sh.n_head; ++h) {
                    const int32_t g = h / group;
                    const float* qh = q.data() + static_cast<size_t>(t) * q_dim + static_cast<size_t>(h) * static_cast<size_t>(sh.head_dim);
                    std::vector<double> score(static_cast<size_t>(last) + 1);
                    double top = -1e300;
                    for (int32_t p = 0; p <= last; ++p) {
                        quant::dequantize_row(kv.dtype(), rows.k(p, g), krow.data(), sh.head_dim);
                        double dot = 0.0;
                        for (int32_t i = 0; i < sh.head_dim; ++i) dot += static_cast<double>(qh[i]) * krow[static_cast<size_t>(i)];
                        dot *= job.scale;
                        if (sh.softcap > 0.0f) dot = sh.softcap * std::tanh(dot / sh.softcap);
                        score[static_cast<size_t>(p)] = dot;
                        top = std::max(top, dot);
                    }
                    double sum = 0.0;
                    std::vector<double> acc(static_cast<size_t>(sh.head_dim), 0.0);
                    for (int32_t p = 0; p <= last; ++p) {
                        const double w = std::exp(score[static_cast<size_t>(p)] - top);
                        sum += w;
                        quant::dequantize_row(kv.dtype(), rows.v(p, g), vrow.data(), sh.head_dim);
                        for (int32_t i = 0; i < sh.head_dim; ++i) acc[static_cast<size_t>(i)] += w * vrow[static_cast<size_t>(i)];
                    }
                    const float* got = out.data() + static_cast<size_t>(t) * q_dim + static_cast<size_t>(h) * static_cast<size_t>(sh.head_dim);
                    for (int32_t i = 0; i < sh.head_dim; ++i) {
                        worst = std::max(worst, std::fabs(acc[static_cast<size_t>(i)] / sum - got[i]));
                    }
                }
            }
            // The values are on the half grid for F16 and within 2^-11 of it for Q8_0; scores and sums are float.
            std::printf("  %s head_dim %d, %d heads / %d kv, %d + %d rows: max error %.2e\n",
                        type == KvCacheType::F16 ? "f16" : "q8_0", sh.head_dim, sh.n_head, sh.n_head_kv, sh.n_past, sh.n, worst);
            CHECK(worst < 2e-3);
        }
    }
    // Shorter blocks and widths the kernels do not cover stay on the CPU.
    KvCacheConfig config;
    config.n_layers = 1;
    config.n_head_kv = 1;
    config.head_dim = 128;
    config.type = KvCacheType::F16;
    auto cache = KvCache::create(config);
    REQUIRE(cache.has_value());
    REQUIRE(cache.value()->reserve(20).is_ok());
    const KvCache::RowView rows = cache.value()->row_view(0);
    std::vector<float> q(16 * 128), out(q.size());
    Backend::AttentionJob job;
    job.rows = &rows;
    job.kv_type = DType::F16;
    job.head_dim = 128;
    job.n_head = 1;
    job.n_head_kv = 1;
    job.n_past = 4;
    job.n = 15;
    job.q = q.data();
    job.out = out.data();
    CHECK(metal.attention(job).code() == ErrorCode::Unsupported);
    job.n = 16;
    job.head_dim = 80;
    CHECK(metal.attention(job).code() == ErrorCode::Unsupported);
}

TEST_MAIN()
