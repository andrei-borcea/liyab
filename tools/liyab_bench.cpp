// liyab-bench — CPU matvec throughput for the decode-time shapes of a MoE model.
//
// Decode is bound by how fast weights stream from RAM, so each kernel is
// reported in GB/s of weight bytes next to the device's read ceiling (a
// parallel sum over a large buffer). Shapes follow Qwen3.6-35B-A3B (hidden
// 2048, expert FFN 512, DeltaNet inner 4096, vocabulary 248k); weights are
// random bytes, which costs the kernels the same as real ones. Every case
// cycles through enough copies of its matrix to exceed the CPU caches, as a
// real forward pass does.
//
//   liyab-bench [--threads 1,2,4,8] [--seconds 1.0]
//
// A last table times one matmul over 1, 2, 4 and 5 activation rows (the
// shape of speculative verification and prefill) on every thread, in the
// file's layouts and, on i8mm CPUs, in the repacked ones the CPU backend
// loads them into (Q4_K_R8, Q5_K_R8, Q6_K_R8, Q8_0_R4).
//
// Uses internal headers: built with the tests (LIYAB_BUILD_TESTS).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/device_detect.h"
#include "liyab/backend.h"

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kWorkingSet = size_t{512} << 20;  // larger than any CPU cache level

struct Case {
    const char* name;
    liyab::DType type;
    int64_t cols;
    int64_t rows;
};

// Runs `step` (which processes `bytes_per_step` bytes) for about `seconds`;
// returns {GB/s, microseconds per step}.
std::pair<double, double> measure(const std::function<void(size_t)>& step, size_t bytes_per_step, double seconds) {
    step(0);  // warm-up: thread wake-up, page faults
    size_t i = 0;
    const auto t0 = Clock::now();
    double elapsed = 0.0;
    while (elapsed < seconds) {
        step(++i);
        elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    return {static_cast<double>(bytes_per_step) * static_cast<double>(i) / elapsed / 1e9, elapsed / i * 1e6};
}

std::vector<int32_t> parse_threads(const char* s) {
    std::vector<int32_t> out;
    for (const char* p = s; *p != '\0';) {
        out.push_back(std::atoi(p));
        while (*p != '\0' && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int32_t> threads;
    double seconds = 1.0;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) {
            threads = parse_threads(argv[++i]);
        } else if (arg == "--seconds" && i + 1 < argc) {
            seconds = std::atof(argv[++i]);
        } else {
            std::fprintf(stderr, "usage: liyab-bench [--threads 1,2,4,8] [--seconds 1.0]\n");
            return arg == "--help" || arg == "-h" ? 0 : 1;
        }
    }
    const int32_t big = liyab::performance_core_count();
    liyab::ThreadPool pool(std::max(big, threads.empty() ? 0 : *std::max_element(threads.begin(), threads.end())));
    if (threads.empty()) {
        for (int32_t t = 1; t < pool.max_threads(); t *= 2) threads.push_back(t);
        threads.push_back(pool.max_threads());
    }
    auto cpu = liyab::make_cpu_backend(pool);

    std::mt19937_64 rng(42);
    std::vector<uint8_t> buffer(kWorkingSet);
    for (size_t i = 0; i < buffer.size(); i += 8) {
        const uint64_t r = rng();
        std::memcpy(buffer.data() + i, &r, std::min<size_t>(8, buffer.size() - i));
    }
    std::vector<float> x(size_t{1} << 20);
    for (float& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
    std::vector<float> y(size_t{1} << 20);

    std::printf("%-28s", "case");
    for (const int32_t t : threads) std::printf(" | %2d thr GB/s (us)", t);
    std::printf("\n");

    // Read ceiling: every thread sums its share of the working set.
    std::printf("%-28s", "read ceiling (sum)");
    for (const int32_t t : threads) {
        pool.set_active_threads(t);
        std::atomic<uint64_t> sink{0};  // keeps the sums observable
        const size_t step_bytes = size_t{64} << 20;
        const auto [gbs, us] = measure(
            [&](size_t i) {
                const size_t base = (i * step_bytes) % kWorkingSet;
                pool.parallel_for(static_cast<int64_t>(step_bytes / 64), [&](int64_t b, int64_t e) {
                    const auto* p = reinterpret_cast<const uint64_t*>(buffer.data() + base);
                    uint64_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
                    for (int64_t k = b * 8; k < e * 8; k += 4) {
                        acc0 += p[k];
                        acc1 += p[k + 1];
                        acc2 += p[k + 2];
                        acc3 += p[k + 3];
                    }
                    sink.fetch_add(acc0 + acc1 + acc2 + acc3, std::memory_order_relaxed);
                });
            },
            step_bytes, seconds);
        std::printf(" | %8.1f (%6.0f)", gbs, us);
    }
    std::printf("\n");

    const Case cases[] = {
        {"Q8_0 attn_qkv 2048x8192", liyab::DType::Q8_0, 2048, 8192},
        // Candidate formats for requantizing the Q8_0 projections (fewer bytes per token):
        {"Q4_0 attn_qkv 2048x8192", liyab::DType::Q4_0, 2048, 8192},
        {"Q4_1 attn_qkv 2048x8192", liyab::DType::Q4_1, 2048, 8192},
        {"Q5_0 attn_qkv 2048x8192", liyab::DType::Q5_0, 2048, 8192},
        {"Q4_K attn_qkv 2048x8192", liyab::DType::Q4_K, 2048, 8192},
        {"Q5_K attn_qkv 2048x8192", liyab::DType::Q5_K, 2048, 8192},
        {"Q6_K attn_qkv 2048x8192", liyab::DType::Q6_K, 2048, 8192},
        {"Q8_0 ssm_out 4096x2048", liyab::DType::Q8_0, 4096, 2048},
        {"Q8_0 shared down 512x2048", liyab::DType::Q8_0, 512, 2048},
        {"F32 router 2048x256", liyab::DType::F32, 2048, 256},
        {"Q6_K lm_head 2048x248320", liyab::DType::Q6_K, 2048, 248320},
        {"Q4_K expert 2048x512", liyab::DType::Q4_K, 2048, 512},
        {"Q5_K expert 2048x512", liyab::DType::Q5_K, 2048, 512},
        {"Q6_K expert 2048x512", liyab::DType::Q6_K, 2048, 512},
        {"Q4_K expert down 512x2048", liyab::DType::Q4_K, 512, 2048},
        {"Q6_K expert down 512x2048", liyab::DType::Q6_K, 512, 2048},
    };
    for (const Case& c : cases) {
        liyab::TensorView w;
        w.type = c.type;
        w.n_dims = 2;
        w.ne = {c.cols, c.rows, 1, 1};
        const size_t bytes = w.row_bytes() * static_cast<size_t>(c.rows);
        w.nbytes = bytes;
        // Copies of the matrix laid out back to back in the working set.
        const size_t copies = std::max<size_t>(1, kWorkingSet / bytes);
        std::vector<uint8_t> own;
        if (bytes > kWorkingSet) {  // the LM head: its own buffer, read once per step
            own.resize(bytes);
            std::memcpy(own.data(), buffer.data(), std::min(bytes, buffer.size()));
        }
        std::printf("%-28s", c.name);
        for (const int32_t t : threads) {
            pool.set_active_threads(t);
            const auto [gbs, us] = measure(
                [&](size_t i) {
                    liyab::TensorView v = w;
                    v.data = own.empty() ? buffer.data() + (i % copies) * bytes : own.data();
                    (void)cpu->matmul(v, x.data(), y.data(), 1);
                },
                bytes, seconds);
            std::printf(" | %8.1f (%6.0f)", gbs, us);
        }
        std::printf("\n");
    }

    // Batched rows (speculative verification, prefill): one pass over the
    // weights for n activation rows. Cost per row should fall as n grows.
    const int32_t all = pool.max_threads();
    pool.set_active_threads(all);
    std::printf("\n%-28s | %d threads, us per pass for n activation rows (us per row)\n", "batched rows", all);
    std::vector<Case> batched = {
        {"Q8_0 2048x8192", liyab::DType::Q8_0, 2048, 8192},
        {"Q4_K 2048x8192", liyab::DType::Q4_K, 2048, 8192},
        {"Q5_K 2048x8192", liyab::DType::Q5_K, 2048, 8192},
        {"Q6_K 2048x8192", liyab::DType::Q6_K, 2048, 8192},
    };
    if (liyab::quant::repack_kernels_available() && liyab::detect_cpu().i8mm) {
        batched.insert(batched.end(), {
                                          {"Q8_0_R4 2048x8192", liyab::DType::Q8_0_R4, 2048, 8192},
                                          {"Q4_K_R8 2048x8192", liyab::DType::Q4_K_R8, 2048, 8192},
                                          {"Q5_K_R8 2048x8192", liyab::DType::Q5_K_R8, 2048, 8192},
                                          {"Q6_K_R8 2048x8192", liyab::DType::Q6_K_R8, 2048, 8192},
                                      });
    }
    for (const Case& c : batched) {
        liyab::TensorView w;
        w.type = c.type;
        w.n_dims = 2;
        w.ne = {c.cols, c.rows, 1, 1};
        const size_t bytes = w.row_bytes() * static_cast<size_t>(c.rows);
        w.nbytes = bytes;
        const size_t copies = std::max<size_t>(1, kWorkingSet / bytes);
        std::printf("%-28s", c.name);
        for (const int32_t n : {1, 2, 4, 5}) {
            const auto [gbs, us] = measure(
                [&](size_t i) {
                    liyab::TensorView v = w;
                    v.data = buffer.data() + (i % copies) * bytes;
                    (void)cpu->matmul(v, x.data(), y.data(), n);
                },
                bytes, seconds);
            (void)gbs;
            std::printf(" | n=%d %6.0f (%4.0f)", n, us, us / n);
        }
        std::printf("\n");
    }
    return 0;
}
