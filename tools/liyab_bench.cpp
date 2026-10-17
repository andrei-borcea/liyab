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
// The kernel set in use is printed first; on x86-64, LIYAB_KERNELS=scalar runs
// the scalar kernels in place of the AVX2 ones, for a comparison.
//
// A last table times one matmul over 1, 2, 4 and 5 activation rows (the
// shape of speculative verification and prefill) on every thread, in the
// file's layouts and, on i8mm CPUs, in the repacked ones the CPU backend
// loads them into (Q4_K_R8, Q5_K_R8, Q6_K_R8, Q8_0_R4).
//
// --prefill times a model's prefill instead: prompts of each --tokens length
// (synthetic token ids: attention and dense blocks cost what real text
// costs; a MoE routes them like any text) in passes of --batch tokens on the
// CPU, the best of --repeat runs, in tokens per second. --latent-form sets
// how a latent-attention model attends (auto, absorbed or expanded; auto is
// calibrated first, as the engine does at load) and the line reports how
// many passes took each form.
//
//   liyab-bench --prefill model.gguf [--tokens 512,2048,8192] [--batch 512]
//               [--latent-form auto|absorbed|expanded] [--kv q8_0|f16] [--threads N] [--repeat 3]
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
#include "core/transformer.h"
#include "liyab/device_detect.h"
#include "liyab/backend.h"
#include "liyab/mmap_loader.h"

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

struct PrefillOptions {
    std::string model;
    std::vector<int32_t> lengths = {512, 2048, 8192};
    int32_t batch = 512;
    int32_t threads = 0;  // 0: performance cores
    int32_t repeat = 3;
    liyab::LatentForm form = liyab::LatentForm::Auto;
    liyab::KvCacheType kv = liyab::KvCacheType::Q8_0;
};

// Prefill speed of a model: each length from an empty context, in passes of
// `batch` tokens, no logits; the best of `repeat` runs.
int prefill_bench(const PrefillOptions& o) {
    const int32_t longest = *std::max_element(o.lengths.begin(), o.lengths.end());
    auto file = liyab::MmapLoader::open(o.model);
    if (!file) {
        std::fprintf(stderr, "%s\n", file.status().to_string().c_str());
        return 1;
    }
    liyab::TransformerOptions options;
    options.context_length = longest + o.batch;
    options.max_batch = o.batch;
    options.kv_type = o.kv;
    options.latent_form = o.form;
    auto loaded = liyab::Transformer::load(std::move(file).value(), options);
    if (!loaded) {
        std::fprintf(stderr, "%s\n", loaded.status().to_string().c_str());
        return 1;
    }
    auto model = std::move(loaded).value();
    liyab::ThreadPool pool(o.threads > 0 ? o.threads : liyab::performance_core_count());
    auto cpu = liyab::make_cpu_backend(pool);
    const liyab::Route route{cpu.get(), cpu.get(), cpu.get()};
    if (model->config().mla.kv_rank > 0 && o.form == liyab::LatentForm::Auto) {
        const auto t0 = Clock::now();
        auto ratio = model->calibrate_latent_form(route, pool);  // as the engine does at load
        if (!ratio) {
            std::fprintf(stderr, "%s\n", ratio.status().to_string().c_str());
            return 1;
        }
        std::printf("latent form ratio (expanded / absorbed cost per multiply-add): %.3f, measured in %.0f ms\n",
                    static_cast<double>(*ratio), std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    const int32_t vocab = model->config().n_vocab;
    std::vector<int32_t> tokens(static_cast<size_t>(longest));
    std::mt19937 rng(7);
    for (int32_t& t : tokens) t = 3 + static_cast<int32_t>(rng() % static_cast<uint32_t>(vocab - 3));
    std::printf("%s: %d threads, passes of %d tokens, kv %s\n", o.model.c_str(), pool.max_threads(), o.batch,
                o.kv == liyab::KvCacheType::F16 ? "f16" : "q8_0");
    for (const int32_t length : o.lengths) {
        double best = 0.0, attention_ms = 0.0;
        int64_t absorbed = 0, expanded = 0;
        for (int32_t r = 0; r < std::max(1, o.repeat); ++r) {
            model->reset();
            const auto stats = model->latent_stats();
            const double attention0 = model->phase_times().attention;
            const auto t0 = Clock::now();
            for (int32_t at = 0; at < length; at += o.batch) {
                const auto n = static_cast<size_t>(std::min(o.batch, length - at));
                auto out = model->forward(std::span<const int32_t>(tokens).subspan(static_cast<size_t>(at), n),
                                          liyab::Transformer::Logits::None, route, pool);
                if (!out) {
                    std::fprintf(stderr, "%s\n", out.status().to_string().c_str());
                    return 1;
                }
            }
            const double s = std::chrono::duration<double>(Clock::now() - t0).count();
            if (length / s > best) {
                best = length / s;
                attention_ms = model->phase_times().attention - attention0;
                absorbed = model->latent_stats().absorbed_passes - stats.absorbed_passes;
                expanded = model->latent_stats().expanded_passes - stats.expanded_passes;
            }
        }
        std::printf("  %6d tokens: %8.1f tok/s (attention %.0f ms)", length, best, attention_ms);
        if (model->config().mla.kv_rank > 0) {
            std::printf(" | latent passes: %lld absorbed, %lld expanded", static_cast<long long>(absorbed),
                        static_cast<long long>(expanded));
        }
        std::printf("\n");
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<int32_t> threads;
    double seconds = 1.0;
    PrefillOptions prefill;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        const bool has_value = i + 1 < argc;
        if (arg == "--threads" && has_value) {
            threads = parse_threads(argv[++i]);
            prefill.threads = threads.empty() ? 0 : threads.back();
        } else if (arg == "--seconds" && has_value) {
            seconds = std::atof(argv[++i]);
        } else if (arg == "--prefill" && has_value) {
            prefill.model = argv[++i];
        } else if (arg == "--tokens" && has_value) {
            prefill.lengths = parse_threads(argv[++i]);
        } else if (arg == "--batch" && has_value) {
            prefill.batch = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--repeat" && has_value) {
            prefill.repeat = std::atoi(argv[++i]);
        } else if (arg == "--kv" && has_value) {
            const std::string v = argv[++i];
            prefill.kv = v == "f16" ? liyab::KvCacheType::F16 : liyab::KvCacheType::Q8_0;
        } else if (arg == "--latent-form" && has_value) {
            const std::string v = argv[++i];
            prefill.form = v == "absorbed"   ? liyab::LatentForm::Absorbed
                           : v == "expanded" ? liyab::LatentForm::Expanded
                                             : liyab::LatentForm::Auto;
        } else {
            std::fprintf(stderr,
                         "usage: liyab-bench [--threads 1,2,4,8] [--seconds 1.0]\n"
                         "       liyab-bench --prefill model.gguf [--tokens 512,2048,8192] [--batch 512]\n"
                         "                   [--latent-form auto|absorbed|expanded] [--kv q8_0|f16] [--threads N]\n"
                         "                   [--repeat 3]\n");
            return arg == "--help" || arg == "-h" ? 0 : 1;
        }
    }
    if (!prefill.model.empty()) {
        if (prefill.lengths.empty()) return 1;
        return prefill_bench(prefill);
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

    std::printf("kernels: %s\n", liyab::quant::kernel_isa());
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
        // Candidate formats for a smaller tier of streamed experts (docs/research/cold-experts.md):
        {"Q3_K expert 2048x512", liyab::DType::Q3_K, 2048, 512},
        {"IQ3_XXS expert 2048x512", liyab::DType::IQ3_XXS, 2048, 512},
        {"Q2_K expert 2048x512", liyab::DType::Q2_K, 2048, 512},
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
