// Independent scalar equations and end-to-end state invariants for SSMs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <vector>

#include "core/ssm_scan.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

TEST_CASE("SSM SIMD kernels agree with the scalar equations, including vector tails") {
    for (const size_t ns : {size_t{1}, size_t{7}, size_t{16}, size_t{37}, size_t{128}}) {
        constexpr size_t tokens = 11, width = 3;
        std::vector<float> a(ns), h(ns, 0.1f), ref = h, dt(tokens * width), x(dt.size()), bc(2 * ns * tokens), y(x.size());
        for (size_t j = 0; j < ns; ++j) a[j] = -std::exp(float(j) / 20);
        for (size_t t = 0; t < tokens; ++t) {
            dt[t * width] = 0.01f + 0.1f * float(t);
            x[t * width] = std::sin(float(t));
            for (size_t j = 0; j < 2 * ns; ++j) bc[t * 2 * ns + j] = std::cos(float(j + t)) * 0.2f;
        }
        ssm::mamba_scan(h.data(), a.data(), ns, dt.data(), x.data(), width, bc.data(), 2 * ns, y.data(), tokens);
        for (size_t t = 0; t < tokens; ++t) {
            double sum = 0;
            for (size_t j = 0; j < ns; ++j) {
                ref[j] = std::exp(dt[t * width] * a[j]) * ref[j] + dt[t * width] * x[t * width] * bc[t * 2 * ns + j];
                sum += ref[j] * bc[t * 2 * ns + ns + j];
            }
            CHECK_NEAR(y[t * width], sum, 2e-6);
        }
        for (size_t j = 0; j < ns; ++j) CHECK_NEAR(h[j], ref[j], 2e-6);
        std::vector<float> state(3 * ns, 0.1f), expected = state, input = {0.2f, -0.3f, 0.7f}, out(3);
        ssm::mamba2_step(state.data(), 3, ns, 0.95f, 0.2f, input.data(), bc.data(), bc.data() + ns, out.data());
        for (size_t p = 0; p < 3; ++p) {
            double sum = 0;
            for (size_t j = 0; j < ns; ++j) {
                expected[p * ns + j] = 0.95f * expected[p * ns + j] + 0.2f * input[p] * bc[j];
                sum += expected[p * ns + j] * bc[ns + j];
                CHECK_NEAR(state[p * ns + j], expected[p * ns + j], 1e-7);
            }
            CHECK_NEAR(out[p], sum, 2e-6);
        }
    }
    for (size_t k : {size_t{1}, size_t{4}, size_t{7}}) {
        std::vector<float> seq(19 + k - 1), taps(k), got(19);
        for (size_t i = 0; i < seq.size(); ++i) seq[i] = std::sin(float(i));
        for (size_t i = 0; i < k; ++i) taps[i] = 0.1f * float(i + 1);
        ssm::causal_conv(seq.data(), taps.data(), 0.3f, k, 19, got.data());
        for (size_t t = 0; t < 19; ++t) {
            float sum = 0.3f;
            for (size_t j = 0; j < k; ++j) sum += taps[j] * seq[t + j];
            CHECK_NEAR(got[t], sum, 1e-6);
        }
    }
}

TEST_CASE("Mamba prefill, decode and rollback retain the same states for every KV type and projection format") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_ssm.gguf";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    for (const char* arch : {"mamba", "mamba2"}) {
        for (const DType type : {DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q4_1,
                                DType::Q5_0, DType::Q5_1, DType::Q8_0, DType::Q2_K, DType::Q3_K,
                                DType::Q4_K, DType::Q5_K, DType::Q6_K, DType::IQ4_NL, DType::IQ4_XS,
                                DType::IQ2_XXS, DType::IQ2_XS, DType::IQ2_S, DType::IQ3_XXS,
                                DType::IQ3_S, DType::IQ1_S, DType::IQ1_M, DType::TQ1_0, DType::TQ2_0,
                                DType::MXFP4, DType::NVFP4}) {
            test::TinyModelSpec s;
            s.arch = arch;
            s.n_embd = 256;
            s.attn_type = type;
            if (type == DType::F32) s.mamba_kernel = 1;  // no convolution history, including rollback
            s.n_layers = 1;
            test::write_tiny_model(path, s);
            for (const auto kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
                auto load = [&](bool repack = false) {
                    TransformerOptions o;
                    o.context_length = 64;
                    o.max_batch = 32;
                    o.kv_type = kv;
                    o.repack_cpu = repack && quant::repack_kernels_available();
                    return Transformer::load(std::move(MmapLoader::open(path)).value(), o);
                };
                auto batched = load(), stepped = load(true);
                REQUIRE(batched && stepped);
                CHECK((*batched)->config().hybrid());
                CHECK((*batched)->config().n_attn_layers == 0);
                CHECK((*batched)->recurrent_state_bytes() > 0);
                (*batched)->set_rollback_window(8);
                (*stepped)->set_rollback_window(8);
                std::vector<int32_t> ids = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99};
                auto all = (*batched)->forward(ids, Transformer::Logits::All, route, pool);
                REQUIRE(all);
                const std::vector<float> expected(all->begin(), all->end());
                const size_t vocab = static_cast<size_t>((*batched)->config().n_vocab);
                for (size_t t = 0; t < ids.size(); ++t) {
                    auto one = (*stepped)->forward(std::span<const int32_t>(&ids[t], 1), Transformer::Logits::Last, route, pool);
                    REQUIRE(one);
                    for (size_t v = 0; v < vocab; ++v) CHECK_NEAR((*one)[v], expected[t * vocab + v], 2e-4);
                }
                REQUIRE((*batched)->truncate(6).is_ok());
                auto replay = (*batched)->forward(std::span(ids).subspan(6), Transformer::Logits::All, route, pool);
                REQUIRE(replay);
                for (size_t i = 0; i < replay->size(); ++i) CHECK_NEAR((*replay)[i], expected[6 * vocab + i], 2e-4);
            }
        }
    }
}

TEST_CASE("Mamba GPU projections retain CPU recurrent-scan semantics") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route reference{cpu.get(), cpu.get(), cpu.get()};
    const std::vector<int32_t> ids = {1, 270, 300, 5, 290, 77};
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_ssm_gpu.gguf";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    for (auto make : {make_metal_backend, make_vulkan_backend}) {
        for (const char* arch : {"mamba", "mamba2"}) {
            test::TinyModelSpec s;
            s.arch = arch;
            // Keep the whole graph float: small projection roundoff must
            // not become a discontinuous Q8 activation rounding difference
            // in the output head when checking the scan's GPU routing.
            s.attn_type = s.ffn_type = s.embd_type = s.output_type = DType::F32;
            test::write_tiny_model(path, s);
            auto load = [&]() { return Transformer::load(std::move(MmapLoader::open(path)).value(), {}); };
            auto ref = load(), model = load();
            REQUIRE(ref && model);
            // A GPU backend caches wrappers around model mappings; destroy
            // it before either mapping, and do not reuse it for another file.
            auto gpu = make();
            if (!gpu) {
                std::printf("  skipped: %s\n", gpu.status().to_string().c_str());
                break;
            }
            std::printf("  %s: %s\n", arch, gpu.value()->description().c_str());
            const Route route{gpu->get(), gpu->get(), cpu.get()};
            auto a = (*ref)->forward(ids, Transformer::Logits::All, reference, pool);
            REQUIRE(a);
            const std::vector<float> expected(a->begin(), a->end());
            auto b = (*model)->forward(ids, Transformer::Logits::All, route, pool);
            REQUIRE(b);
            for (size_t i = 0; i < expected.size(); ++i) CHECK_NEAR((*b)[i], expected[i], 1e-3);
            for (const int32_t token : ids) {
                const std::span<const int32_t> step(&token, 1);
                auto c = (*ref)->forward(step, Transformer::Logits::Last, reference, pool);
                REQUIRE(c);
                const std::vector<float> row(c->begin(), c->end());
                auto d = (*model)->forward(step, Transformer::Logits::Last, route, pool);
                REQUIRE(d);
                for (size_t i = 0; i < row.size(); ++i) CHECK_NEAR((*d)[i], row[i], 1e-3);
            }
        }
    }
}

TEST_CASE("Incomplete projection sets are rejected instead of silently dropping a block part") {
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_ssm_incomplete.gguf";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    for (const std::string suffix : {"attn_q.weight", "ffn_up.weight", "ffn_gate_inp.weight", "ssm_conv1d.weight"}) {
        test::TinyModelSpec spec;
        if (suffix == "ffn_gate_inp.weight") {
            spec.arch = "qwen3moe";
            spec.n_expert = 4;
            spec.n_expert_used = 2;
        }
        if (suffix == "ssm_conv1d.weight") spec.arch = "mamba";
        test::write_tiny_model(path, spec);
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        const std::string bytes((std::istreambuf_iterator<char>(file)), {});
        const std::string name = "blk.0." + suffix;
        std::string renamed = name;
        renamed[6] = 'x';  // preserve the GGUF header length and every tensor's offset
        const auto at = bytes.find(name);
        REQUIRE(at != std::string::npos);
        file.clear();
        file.seekp(static_cast<std::streamoff>(at));
        file.write(renamed.data(), static_cast<std::streamsize>(renamed.size()));
        file.close();
        auto mapped = MmapLoader::open(path);
        REQUIRE(mapped);
        auto model = Transformer::load(std::move(mapped).value(), {});
        REQUIRE(!model);
        CHECK(model.status().code() == ErrorCode::InvalidModel);
        CHECK(model.status().to_string().find(suffix) != std::string::npos);
    }
}

TEST_CASE("Parallel attention and Mamba are refused with the missing topology named") {
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_ssm_parallel.gguf";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    for (const char* arch : {"mamba", "mamba2"}) {
        test::TinyModelSpec spec;
        spec.arch = arch;
        spec.mamba_parallel = true;
        test::write_tiny_model(path, spec);
        auto file = MmapLoader::open(path);
        REQUIRE(file);
        auto description = describe_model(**file);
        REQUIRE(!description);
        CHECK(description.status().code() == ErrorCode::Unsupported);
        CHECK(description.status().to_string().find("parallel attention and Mamba") != std::string::npos);
    }
}

TEST_MAIN()
