// Independent scalar equations and end-to-end state invariants for SSMs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <vector>

#include "core/kernel_conformance.h"
#include "core/ssm_scan.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

double energy(std::span<const float> values) {
    double sum = 0;
    for (const float x : values) {
        if (!std::isfinite(x)) return -1;
        sum += static_cast<double>(x) * x;
    }
    return sum;
}

// Observe the real projection calls, or ablate exactly one projection after
// computing it. This checks that valid payloads actually contribute to the
// recurrent graph, rather than merely loading successfully.
class ProjectionProbe final : public Backend {
public:
    explicit ProjectionProbe(Backend& cpu) : cpu_(cpu) {}
    BackendKind kind() const noexcept override { return BackendKind::Cpu; }
    std::string description() const override { return "Recurrent projection probe"; }
    bool supports(DType type) const noexcept override { return cpu_.supports(type); }
    Status matmul(const TensorView& w, const float* x, float* y, int32_t n) override {
        LIYAB_RETURN_IF_ERROR(cpu_.matmul(w, x, y, n));
        if (w.name.starts_with("blk.0.")) {
            const auto values = std::span(y, static_cast<size_t>(n * w.rows()));
            outputs[std::string(w.name)] += energy(values);
            if (w.name == ablate) std::fill(values.begin(), values.end(), 0.0f);
        }
        return Status::ok();
    }
    std::map<std::string, double> outputs;
    std::string ablate;

private:
    Backend& cpu_;
};

std::vector<float> state_image(Transformer& model) {
    std::vector<float> image;
    for (const auto part : model.recurrent_parts()) image.insert(image.end(), part.begin(), part.end());
    return image;
}

double max_difference(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size()) return -1;
    double difference = 0;
    for (size_t i = 0; i < a.size(); ++i) difference = std::max(difference, std::fabs(double{a[i]} - b[i]));
    return difference;
}

}  // namespace

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

TEST_CASE("Mamba random writable projections retain prefill, decode and rollback for every KV type") {
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_ssm.gguf";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    for (const char* arch : {"mamba", "mamba2"}) {
        // The writer implements these eight encoders. Reader-only formats
        // use real encoded blocks in the separate all-format test below.
        for (const DType type : {DType::F32, DType::F16, DType::BF16, DType::Q4_0, DType::Q4_1,
                                DType::Q8_0, DType::Q4_K, DType::Q5_K}) {
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

TEST_CASE("Nonzero encoded recurrent projections and states contribute across every format and KV type") {
    std::vector<test::QuantVector> vectors;
    REQUIRE(test::load_quant_vectors(vectors, "ssm_quant_vectors.bin"));
    REQUIRE(vectors.size() + 2 == conformance::weight_formats().size());  // all but F32 / F16, written below
    // F32/F16 use the original random writer; all other formats use the
    // gguf-py fixture blocks with uniformly RMS-conditioned scale fields,
    // including reader-only formats. Integer codes are unchanged.
    vectors.push_back({DType::F32, 0, {}, {}});
    vectors.push_back({DType::F16, 0, {}, {}});
    std::set<DType> formats;
    for (const auto& v : vectors) REQUIRE(formats.insert(v.type).second);
    REQUIRE(formats.size() == conformance::weight_formats().size());
    ThreadPool pool(3);
    auto cpu = make_cpu_backend(pool);
    const Route route{cpu.get(), cpu.get(), cpu.get()};
    const std::string path = test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_ssm_encoded.gguf";
    struct Cleanup { std::string path; ~Cleanup() { std::remove(path.c_str()); } } cleanup{path};
    const std::vector<int32_t> ids = {1, 270, 300, 5, 290, 77, 310, 280, 12, 99};
    double minimum_projection_delta = std::numeric_limits<double>::infinity();
    double minimum_carried_delta = std::numeric_limits<double>::infinity();
    double minimum_scan_energy = std::numeric_limits<double>::infinity();
    double maximum_logit_error = 0, maximum_state_error = 0;
    size_t checked = 0;
    for (const char* arch : {"mamba", "mamba2", "lfm2", "kimi-linear", "qwen3next"}) {
        for (const auto& qv : vectors) {
            std::printf("  nonzero %s %s\n", arch, std::string(dtype_traits(qv.type).name).c_str());
            test::TinyModelSpec s;
            s.arch = arch;
            s.n_embd = 256;
            s.n_layers = 1;
            s.attn_type = qv.type;
            s.ssm_block_pattern = qv.blocks;
            if (s.arch == "lfm2") s.short_conv_interval = 2;
            if (s.arch == "kimi-linear") s.kda_interval = 2;
            if (s.arch == "qwen3next") {
                s.delta_net_interval = 2;
                s.fused_ssm_ba = true;
            }
            s.ssm_head_dim = s.kda_head_dim = 256;
            s.ssm_key_heads = s.ssm_value_heads = 1;
            // A float output head keeps projection roundoff separate from
            // discontinuous activation rounding when comparing full graphs.
            s.embd_type = s.output_type = DType::F32;
            const std::vector<std::string> roles = s.arch == "mamba"
                ? std::vector<std::string>{"ssm_in", "ssm_x", "ssm_dt", "ssm_out"}
                : s.arch == "mamba2" ? std::vector<std::string>{"ssm_in", "ssm_out"}
                : s.arch == "lfm2" ? std::vector<std::string>{"shortconv.in_proj", "shortconv.out_proj"}
                : s.arch == "kimi-linear" ? std::vector<std::string>{"attn_q", "attn_k", "attn_v", "ssm_f_a", "ssm_g_a", "attn_output"}
                : std::vector<std::string>{"attn_qkv", "attn_gate", "ssm_out"};
            test::write_tiny_model(path, s);
            auto file = MmapLoader::open(path);
            REQUIRE(file);
            for (const auto& role : roles) {
                const auto* w = (*file)->tensor("blk.0." + role + ".weight");
                REQUIRE(w);
                REQUIRE(std::any_of(w->data, w->data + w->nbytes, [](uint8_t b) { return b != 0; }));
                std::vector<float> row(static_cast<size_t>(w->cols()));
                double weights = 0;
                for (int64_t r = 0; r < w->rows(); ++r) {
                    quant::dequantize_row(w->type, w->row(r), row.data(), w->cols());
                    REQUIRE(energy(row) > 0);
                    weights += energy(row);
                    if (!qv.blocks.empty()) {
                        for (size_t c = 0; c < row.size(); ++c) {
                            const float expected = qv.expected[(static_cast<size_t>(r) * row.size() + c) % qv.expected.size()];
                            CHECK_NEAR(row[c], expected, 1e-6 * std::max(1.0f, std::fabs(expected)));
                        }
                    }
                }
                REQUIRE(weights > 0);
            }
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
                (*batched)->set_rollback_window(8);
                (*stepped)->set_rollback_window(8);
                ProjectionProbe probe(*cpu);
                const Route observed{&probe, &probe, &probe};
                auto all = (*batched)->forward(ids, Transformer::Logits::All, observed, pool);
                REQUIRE(all);
                const std::vector<float> expected(all->begin(), all->end());
                REQUIRE(energy(expected) > 0);
                const auto state = state_image(**batched);
                const auto parts = (*batched)->recurrent_parts();
                REQUIRE(parts.size() == (s.arch == "lfm2" ? 1 : 2));
                REQUIRE(energy(parts[0]) > 0);
                REQUIRE(energy(parts.back()) > 0);
                minimum_scan_energy = std::min(minimum_scan_energy, energy(parts.back()));
                for (const auto& role : roles) REQUIRE(probe.outputs["blk.0." + role + ".weight"] > 0);
                const size_t vocab = static_cast<size_t>((*batched)->config().n_vocab);
                for (size_t t = 0; t < ids.size(); ++t) {
                    auto one = (*stepped)->forward(std::span(ids).subspan(t, 1), Transformer::Logits::Last, route, pool);
                    REQUIRE(one);
                    maximum_logit_error = std::max(maximum_logit_error,
                        max_difference(*one, std::span(expected).subspan(t * vocab, vocab)));
                    for (size_t v = 0; v < vocab; ++v) CHECK_NEAR((*one)[v], expected[t * vocab + v], 2e-4);
                }
                const auto decoded = state_image(**stepped);
                REQUIRE(decoded.size() == state.size());
                maximum_state_error = std::max(maximum_state_error, max_difference(decoded, state));
                for (size_t i = 0; i < state.size(); ++i) CHECK_NEAR(decoded[i], state[i], 2e-4);
                REQUIRE((*batched)->truncate(6).is_ok());
                auto replay = (*batched)->forward(std::span(ids).subspan(6), Transformer::Logits::All, route, pool);
                REQUIRE(replay);
                maximum_logit_error = std::max(maximum_logit_error, max_difference(*replay, std::span(expected).subspan(6 * vocab)));
                for (size_t i = 0; i < replay->size(); ++i) CHECK_NEAR((*replay)[i], expected[6 * vocab + i], 2e-4);
                const auto rolled = state_image(**batched);
                REQUIRE(rolled.size() == state.size());
                maximum_state_error = std::max(maximum_state_error, max_difference(rolled, state));
                for (size_t i = 0; i < state.size(); ++i) CHECK_NEAR(rolled[i], state[i], 2e-4);
                // Ablating each projection must change the actual logits;
                // nonzero stored bytes alone cannot satisfy this proof.
                for (const auto& role : roles) {
                    auto ablated = load();
                    REQUIRE(ablated);
                    ProjectionProbe zero(*cpu);
                    zero.ablate = "blk.0." + role + ".weight";
                    const Route removed{&zero, &zero, &zero};
                    auto logits = (*ablated)->forward(ids, Transformer::Logits::All, removed, pool);
                    REQUIRE(logits);
                    REQUIRE(energy(*logits) > 0);
                    const double difference = max_difference(*logits, expected);
                    minimum_projection_delta = std::min(minimum_projection_delta, difference);
                    CHECK(difference > 1e-6);
                }
                // Remove the carried scan state, or the sole convolution history
                // of ShortConv: it must contribute to the following token.
                auto forget = load();
                REQUIRE(forget);
                REQUIRE((*forget)->forward(std::span(ids).first(ids.size() - 1), Transformer::Logits::Last, route, pool));
                auto scan = (*forget)->recurrent_parts();
                REQUIRE(!scan.empty() && energy(scan.back()) > 0);
                std::fill(scan.back().begin(), scan.back().end(), 0.0f);
                auto last = (*forget)->forward(std::span(ids).last(1), Transformer::Logits::Last, route, pool);
                REQUIRE(last);
                const double difference = max_difference(*last, std::span(expected).last(vocab));
                minimum_carried_delta = std::min(minimum_carried_delta, difference);
                CHECK(difference > 1e-6);
                ++checked;
            }
        }
    }
    CHECK(checked == 5 * 4 * formats.size());  // five mixers, four KV cache types
    std::printf("  %zu nonzero mixer/format/KV cases; minimum projection ablation %.9g, carried-state ablation %.9g, "
                "scan energy %.9g; maximum logit error %.9g, state error %.9g\n", checked,
                minimum_projection_delta, minimum_carried_delta, minimum_scan_energy,
                maximum_logit_error, maximum_state_error);
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
