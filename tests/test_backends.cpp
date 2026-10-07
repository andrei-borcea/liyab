// Backend micro-benchmarks: per-matmul latency on CPU and GPU backends for
// the shapes of a real model, plus the fixed dispatch cost of each backend.
//
//   LIYAB_BENCH_MODEL=tinyllama-q4_0.gguf ./test_backends
//
// Without LIYAB_BENCH_MODEL the shapes come from a small synthetic model.
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "core/thread_pool.h"
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
        path = test::temp_dir() + "/liyab_bb_" + std::to_string(getpid()) + ".gguf";
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

TEST_MAIN()
