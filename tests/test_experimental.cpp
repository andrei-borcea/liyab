// Experimental modules: correctness tests plus a benchmark harness that
// prints the measured effect on decode throughput and I/O latency.
//
//   ./test_experimental                 # tests + benchmarks (small sizes)
//   LIYAB_BENCH=0 ./test_experimental   # tests only
//   LIYAB_BENCH_MB=512 ./test_experimental   # larger I/O benchmark file
//   LIYAB_BENCH_MODEL=tinyllama-q4_0.gguf ./test_experimental   # real-model benchmarks
//
// Benchmarks never fail the run: their numbers depend on the device, the
// filesystem and the (random-weight) benchmark model, so read them as
// relative costs, not as quality claims.
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <optional>
#include <vector>

#include "core/sampling.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "liyab/engine.h"
#include "core/tokenizer.h"
#include "liyab/experimental/early_exit.h"
#include "liyab/experimental/egls.h"
#include "liyab/experimental/head_pruner.h"
#include "liyab/experimental/io_uring_loader.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;
using namespace liyab::experimental;
using Clock = std::chrono::steady_clock;

namespace {

std::string temp_path(const std::string& name) {
    return test::temp_dir() + "/liyab_x" + std::to_string(getpid()) + "_" + name;
}

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

bool benchmarks_enabled() {
    const char* v = std::getenv("LIYAB_BENCH");
    return v == nullptr || std::strcmp(v, "0") != 0;
}

const std::string& tiny_model() {
    static const std::string path = [] {
        std::string p = temp_path("tiny.gguf");
        test::write_tiny_model(p);
        return p;
    }();
    return path;
}

std::unique_ptr<Transformer> load(const std::string& path, int32_t ctx = 0) {
    auto file = MmapLoader::open(path);
    if (!file) return nullptr;
    TransformerOptions options;
    options.context_length = ctx;
    auto model = Transformer::load(std::move(file).value(), options);
    return model ? std::move(model).value() : nullptr;
}

// Pass-through mask hook used to check that "all heads on" equals no mask.
struct AllHeads final : HeadMaskHook {
    std::vector<uint8_t> ones;
    explicit AllHeads(int32_t n) : ones(static_cast<size_t>(n), 1) {}
    const uint8_t* mask(int32_t) const override { return ones.data(); }
};

struct Rig {
    ThreadPool pool{4};
    std::unique_ptr<Backend> cpu = make_cpu_backend(pool);
    Route route{cpu.get(), cpu.get(), cpu.get()};
};

}  // namespace

// ---------------------------------------------------------------------------
// Early exit
// ---------------------------------------------------------------------------
TEST_CASE("EarlyExit confidence and entropy") {
    const std::vector<float> uniform(100, 0.5f);
    CHECK_NEAR(EarlyExit::confidence(uniform), 0.01, 1e-6);
    CHECK_NEAR(EarlyExit::entropy(uniform), std::log(100.0), 1e-4);
    std::vector<float> peaked(100, 0.0f);
    peaked[7] = 30.0f;
    CHECK(EarlyExit::confidence(peaked) > 0.999f);
    CHECK(EarlyExit::entropy(peaked) < 1e-3f);
}

TEST_CASE("EarlyExit probe schedule respects min_layer and interval") {
    EarlyExit e({0.98f, 2, 3});
    std::vector<int32_t> probed;
    for (int32_t l = 0; l < 12; ++l) {
        if (e.probe_after(l, 12)) probed.push_back(l);
    }
    CHECK(probed == (std::vector<int32_t>{2, 5, 8}));  // never after the last block (11)
    EarlyExit defaults;
    CHECK(!defaults.probe_after(5, 12) && defaults.probe_after(6, 12));  // min_layer = n/2
}

TEST_CASE("Early exit stops after the probed block and keeps the KV cache usable") {
    Rig rig;
    auto model = load(tiny_model());
    auto reference = load(tiny_model());
    REQUIRE(model && reference);
    const std::vector<int32_t> prompt = {1, 270, 300};
    REQUIRE(model->forward(prompt, Transformer::Logits::None, rig.route, rig.pool).has_value());
    REQUIRE(reference->forward(prompt, Transformer::Logits::None, rig.route, rig.pool).has_value());

    // Threshold above 1: never exits, output identical to no hooks.
    EarlyExit never({2.0f, 0, 1});
    ForwardHooks hooks{&never, nullptr};
    const int32_t tok = 290;
    auto a = model->forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool, &hooks);
    auto b = reference->forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool);
    REQUIRE(a.has_value() && b.has_value());
    CHECK(std::equal(a->begin(), a->end(), b->begin()));
    CHECK(model->last_exit_layer() == -1);
    CHECK(never.probes() == 1);  // tiny model: 2 blocks, probe after block 0 only

    // Threshold below 0: always exits at the first probe (block 0).
    EarlyExit always({-1.0f, 0, 1});
    hooks.early_exit = &always;
    const int32_t tok2 = 77;
    auto c = model->forward(std::span<const int32_t>(&tok2, 1), Transformer::Logits::Last, rig.route, rig.pool, &hooks);
    REQUIRE(c.has_value());
    CHECK(model->last_exit_layer() == 0);
    CHECK(always.exits() == 1);
    CHECK(model->n_past() == 5);
    for (const float v : c.value()) CHECK(std::isfinite(v));

    // Later tokens attend to the propagated K/V of the skipped block.
    auto d = model->forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool);
    REQUIRE(d.has_value());
    for (const float v : d.value()) CHECK(std::isfinite(v));

    // Batched calls (prefill / verification) never exit.
    const std::vector<int32_t> batch = {5, 6};
    REQUIRE(model->forward(batch, Transformer::Logits::Last, rig.route, rig.pool, &hooks).has_value());
    CHECK(model->last_exit_layer() == -1);
}

// ---------------------------------------------------------------------------
// Head pruning
// ---------------------------------------------------------------------------
TEST_CASE("HeadPruner ranks heads by Wo norm and activates with the power policy") {
    auto model = load(tiny_model());
    REQUIRE(model != nullptr);
    const ModelConfig& c = model->config();
    std::vector<const TensorView*> wo;
    for (int32_t l = 0; l < c.n_layers; ++l) wo.push_back(&model->attn_output(l));

    auto full = HeadPruner::create(wo, c.n_head, c.head_dim, {1.0f});
    REQUIRE(full.has_value());
    full->set_active(true);
    CHECK(full->mask(0) == nullptr);  // keep everything: no mask needed

    auto half = HeadPruner::create(wo, c.n_head, c.head_dim, {0.5f});
    REQUIRE(half.has_value());
    CHECK(half->kept_heads() == c.n_head / 2);
    CHECK(half->mask(0) == nullptr);  // inactive by default

    PowerPolicy cool;
    CHECK(!half->update(cool, PowerProfile::Balanced));
    PowerPolicy hot;
    hot.throttled = true;  // e.g. skin >= 40 °C
    CHECK(half->update(hot, PowerProfile::Balanced));
    CHECK(half->update(cool, PowerProfile::LowPower));

    for (int32_t l = 0; l < c.n_layers; ++l) {
        const uint8_t* m = half->mask(l);
        REQUIRE(m != nullptr);
        CHECK(std::count(m, m + c.n_head, 1) == c.n_head / 2);
        float min_kept = 1e30f;
        float max_dropped = -1.0f;
        for (int32_t h = 0; h < c.n_head; ++h) {
            if (m[h]) min_kept = std::min(min_kept, half->importance(l, h));
            else max_dropped = std::max(max_dropped, half->importance(l, h));
        }
        CHECK(min_kept >= max_dropped);
    }
    CHECK(!HeadPruner::create(wo, c.n_head, c.head_dim, {0.0f}).has_value());
}

TEST_CASE("Head masks change attention output; an all-ones mask does not") {
    Rig rig;
    auto base = load(tiny_model());
    auto ones = load(tiny_model());
    auto pruned = load(tiny_model());
    REQUIRE(base && ones && pruned);
    const ModelConfig& c = base->config();
    std::vector<const TensorView*> wo;
    for (int32_t l = 0; l < c.n_layers; ++l) wo.push_back(&pruned->attn_output(l));
    auto pruner = HeadPruner::create(wo, c.n_head, c.head_dim, {0.5f});
    REQUIRE(pruner.has_value());
    pruner->set_active(true);
    AllHeads all(c.n_head);

    const std::vector<int32_t> tokens = {1, 270, 300, 5, 290};
    ForwardHooks all_hooks{nullptr, &all};
    ForwardHooks prune_hooks{nullptr, &pruner.value()};
    auto a = base->forward(tokens, Transformer::Logits::Last, rig.route, rig.pool);
    const std::vector<float> la(a->begin(), a->end());
    auto b = ones->forward(tokens, Transformer::Logits::Last, rig.route, rig.pool, &all_hooks);
    const std::vector<float> lb(b->begin(), b->end());
    auto p = pruned->forward(tokens, Transformer::Logits::Last, rig.route, rig.pool, &prune_hooks);
    REQUIRE(p.has_value());
    CHECK(la == lb);
    double diff = 0;
    for (size_t i = 0; i < la.size(); ++i) diff = std::max(diff, std::fabs(static_cast<double>(la[i]) - (*p)[i]));
    CHECK(diff > 1e-4);
}

// ---------------------------------------------------------------------------
// Direct I/O
// ---------------------------------------------------------------------------
TEST_CASE("DirectReader streams exact bytes for unaligned ranges with every method") {
    const std::string path = temp_path("direct.bin");
    std::vector<uint8_t> bytes(3 * 1024 * 1024 + 123);
    std::mt19937 rng(17);
    for (auto& b : bytes) b = static_cast<uint8_t>(rng());
    std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(bytes.data()),
                                                static_cast<std::streamsize>(bytes.size()));

    for (const bool uring : {true, false}) {
        for (const bool bypass : {true, false}) {
            DirectReaderOptions options;
            options.use_io_uring = uring;
            options.bypass_page_cache = bypass;
            options.chunk_bytes = 256 * 1024;
            options.queue_depth = 4;
            auto reader = DirectReader::open(path, options);
            REQUIRE(reader.has_value());
            DirectReader& r = *reader.value();
            std::printf("  io_uring=%d bypass=%d -> %s%s%s\n", uring, bypass, io_method_name(r.method()),
                        r.fallback_reason().empty() ? "" : " (", r.fallback_reason().empty() ? "" : (r.fallback_reason() + ")").c_str());
            CHECK(r.file_size() == bytes.size());

            for (const auto [off, len] : std::vector<std::pair<uint64_t, uint64_t>>{
                     {0, bytes.size()}, {4095, 1}, {12345, 1000000}, {bytes.size() - 77, 500}, {bytes.size() + 10, 5}}) {
                std::vector<uint8_t> out(static_cast<size_t>(std::min<uint64_t>(len, bytes.size() - std::min<uint64_t>(off, bytes.size()))), 0);
                size_t received = 0;
                const Status s = r.stream(off, len, [&](const ReadChunk& chunk) {
                    std::memcpy(out.data() + (chunk.offset - off), chunk.data, chunk.length);
                    received += chunk.length;
                    return true;
                });
                CHECK(s.is_ok());
                CHECK(received == out.size());
                if (!out.empty()) CHECK(std::memcmp(out.data(), bytes.data() + off, out.size()) == 0);
            }

            size_t chunks = 0;  // early stop
            CHECK(r.stream(0, bytes.size(), [&](const ReadChunk&) { return ++chunks < 2; }).is_ok());
            CHECK(chunks == 2);
        }
    }
    CHECK(!DirectReader::open(temp_path("missing.bin")).has_value());
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Engine integration
// ---------------------------------------------------------------------------
TEST_CASE("Engine reports early exits and pruned steps") {
    EngineConfig config;
    config.model_path = tiny_model();
    config.backend = BackendKind::Cpu;
    config.thermal_polling = false;
    config.power.profile = PowerProfile::LowPower;  // activates head pruning
    config.power.target_tps = 10000.0;              // effectively unpaced
    config.experimental.early_exit = true;
    config.experimental.early_exit_threshold = -1.0f;  // force exits to exercise the path
    config.experimental.early_exit_min_layer = 0;
    config.experimental.head_pruning = true;
    config.experimental.head_keep_ratio = 0.5f;
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    SamplingParams p;
    p.temperature = 0.0f;
    p.max_tokens = 8;
    auto stats = engine.value()->generate_tokens(std::vector<int32_t>{1, 270, 300}, p, {});
    REQUIRE(stats.has_value());
    CHECK(stats->early_exits > 0);
    CHECK(stats->early_exit_layers_skipped == stats->early_exits);  // 2 blocks: exits after block 0
    CHECK(stats->head_pruned_steps > 0);
}

// ---------------------------------------------------------------------------
// EGLS
// ---------------------------------------------------------------------------
TEST_CASE("EGLS energy entropy is normalized and scale-free") {
    std::vector<float> flat(64, 0.3f);
    CHECK_NEAR(Egls::energy_entropy(flat), 1.0, 1e-5);
    std::vector<float> spike(64, 0.0f);
    spike[3] = 5.0f;
    CHECK_NEAR(Egls::energy_entropy(spike), 0.0, 1e-6);
    std::vector<float> scaled = flat;
    for (float& v : scaled) v *= 1000.0f;
    CHECK_NEAR(Egls::energy_entropy(scaled), Egls::energy_entropy(flat), 1e-5);
    CHECK(Egls::energy_entropy(std::vector<float>(8, 0.0f)) == 0.0f);
}

TEST_CASE("EGLS protects edge blocks and skips below the threshold") {
    Egls egls({0.05f, 1, 1});
    std::vector<float> a(32, 1.0f), b(32, 1.0f);
    b[0] = 1.2f;  // tiny entropy change
    CHECK(!egls.skip_ffn(0, 4, a, b));  // protected first block
    CHECK(!egls.skip_ffn(3, 4, a, b));  // protected last block
    CHECK(egls.skip_ffn(1, 4, a, b));
    std::vector<float> c(32, 0.0f);
    c[0] = 10.0f;  // large entropy change
    CHECK(!egls.skip_ffn(2, 4, a, c));
    CHECK(egls.decisions() == 2);
    CHECK(egls.skips() == 1);
}

TEST_CASE("EGLS hook skips FFN blocks in decode only and never when the threshold is 0") {
    Rig rig;
    test::TinyModelSpec spec;
    spec.n_layers = 6;
    const std::string path = temp_path("egls.gguf");
    test::write_tiny_model(path, spec);
    auto a = load(path);
    auto b = load(path);
    REQUIRE(a && b);
    Egls never({0.0f, 1, 1});
    Egls always({2.0f, 1, 1});  // dH is in [0, 1]: always below 2
    ForwardHooks never_hooks;
    never_hooks.ffn_skip = &never;
    ForwardHooks always_hooks;
    always_hooks.ffn_skip = &always;

    const std::vector<int32_t> prompt = {1, 270, 300};
    REQUIRE(a->forward(prompt, Transformer::Logits::None, rig.route, rig.pool, &always_hooks).has_value());
    CHECK(a->last_ffn_skips() == 0);  // batched prefill is never skipped
    REQUIRE(b->forward(prompt, Transformer::Logits::None, rig.route, rig.pool).has_value());

    const int32_t tok = 290;
    auto ra = a->forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool, &never_hooks);
    const std::vector<float> la(ra->begin(), ra->end());
    auto rb = b->forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool);
    CHECK(la == std::vector<float>(rb->begin(), rb->end()));
    CHECK(a->last_ffn_skips() == 0);

    auto rc = a->forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool, &always_hooks);
    REQUIRE(rc.has_value());
    CHECK(a->last_ffn_skips() == 4);  // 6 blocks minus the protected first and last
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// Real-model benchmark helpers (LIYAB_BENCH_MODEL=<SentencePiece GGUF>)
// ---------------------------------------------------------------------------
namespace {

struct RealModel {
    std::string path;
    std::vector<int32_t> prompt;
};

std::optional<RealModel> real_model() {
    const char* path = std::getenv("LIYAB_BENCH_MODEL");
    if (path == nullptr || !benchmarks_enabled()) return std::nullopt;
    auto file = MmapLoader::open(path);
    if (!file) return std::nullopt;
    auto tok = Tokenizer::load(*file.value());
    if (!tok) return std::nullopt;
    auto ids = tok->encode("<|user|>\nExplain why the sky is blue to a ten year old.</s>\n<|assistant|>\n", true);
    if (!ids) return std::nullopt;
    return RealModel{path, ids.value()};
}

struct GreedyRun {
    std::vector<int32_t> tokens;
    double tps = 0.0;
};

GreedyRun greedy_run(Transformer& model, Rig& rig, const std::vector<int32_t>& prompt, int32_t steps,
                     const ForwardHooks* hooks) {
    model.reset();
    GreedyRun run;
    if (!model.forward(std::span<const int32_t>(prompt).first(prompt.size() - 1), Transformer::Logits::None,
                       rig.route, rig.pool)) {
        return run;
    }
    int32_t tok = prompt.back();
    const auto t0 = Clock::now();
    for (int32_t i = 0; i < steps; ++i) {
        auto logits = model.forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool, hooks);
        if (!logits) break;
        tok = Sampler::argmax(logits.value());
        run.tokens.push_back(tok);
    }
    run.tps = static_cast<double>(run.tokens.size()) * 1000.0 / ms_since(t0);
    return run;
}

// Tokens identical to the reference before the first divergence.
size_t common_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
    return i;
}

// Records every dH without skipping, for threshold calibration.
struct DeltaRecorder final : FfnSkipHook {
    std::vector<float> deltas;
    bool skip_ffn(int32_t layer, int32_t n_layers, std::span<const float> before, std::span<const float> after) override {
        if (layer >= 2 && layer < n_layers - 2) {
            deltas.push_back(std::fabs(Egls::energy_entropy(after) - Egls::energy_entropy(before)));
        }
        return false;
    }
};

}  // namespace

TEST_CASE("[bench] EGLS on a real model: threshold sensitivity, speed, agreement") {
    const auto rm = real_model();
    if (!rm) {
        std::printf("  skipped: set LIYAB_BENCH_MODEL to a SentencePiece GGUF (e.g. TinyLlama Q4_0)\n");
        return;
    }
    Rig rig;
    auto model = load(rm->path, 1024);
    REQUIRE(model != nullptr);
    const int32_t steps = 64;
    const GreedyRun base = greedy_run(*model, rig, rm->prompt, steps, nullptr);

    DeltaRecorder recorder;
    ForwardHooks rec_hooks;
    rec_hooks.ffn_skip = &recorder;
    greedy_run(*model, rig, rm->prompt, steps, &rec_hooks);
    std::vector<float> sorted = recorder.deltas;
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) { return sorted[static_cast<size_t>(p * static_cast<double>(sorted.size() - 1))]; };
    std::printf("  dH distribution over %zu decisions: p10 %.5f  p25 %.5f  p50 %.5f  p75 %.5f  p90 %.5f\n",
                sorted.size(), pct(0.10), pct(0.25), pct(0.50), pct(0.75), pct(0.90));
    std::printf("  %-22s %8s %10s %14s %12s\n", "threshold", "tok/s", "FFN skip", "same prefix", "same tokens");
    std::printf("  %-22s %8.1f %9.0f%% %11zu/%d %11.0f%%\n", "off (baseline)", base.tps, 0.0, base.tokens.size(), steps, 100.0);
    for (const double p : {0.10, 0.25, 0.50, 0.75}) {
        Egls egls({pct(p), 2, 2});
        ForwardHooks hooks;
        hooks.ffn_skip = &egls;
        const GreedyRun r = greedy_run(*model, rig, rm->prompt, steps, &hooks);
        size_t same = 0;
        for (size_t i = 0; i < std::min(r.tokens.size(), base.tokens.size()); ++i) same += r.tokens[i] == base.tokens[i];
        char label[64];
        std::snprintf(label, sizeof label, "p%.0f = %.5f", p * 100, pct(p));
        std::printf("  %-22s %8.1f %9.0f%% %11zu/%d %11.0f%%\n", label, r.tps,
                    100.0 * static_cast<double>(egls.skips()) / std::max<int64_t>(1, egls.decisions()),
                    common_prefix(r.tokens, base.tokens), steps, 100.0 * static_cast<double>(same) / steps);
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------
namespace {

struct DecodeResult {
    double tps = 0;
    int64_t exits = 0;
};

DecodeResult bench_decode(Transformer& model, Rig& rig, int32_t prefill, int32_t steps, const ForwardHooks* hooks) {
    model.reset();
    std::vector<int32_t> prompt(static_cast<size_t>(prefill));
    std::mt19937 rng(3);
    for (auto& t : prompt) t = 3 + static_cast<int32_t>(rng() % 300);
    for (size_t i = 0; i < prompt.size(); i += 64) {
        const auto part = std::span<const int32_t>(prompt).subspan(i, std::min<size_t>(64, prompt.size() - i));
        if (!model.forward(part, Transformer::Logits::None, rig.route, rig.pool, hooks)) return {};
    }
    int32_t tok = 270;
    DecodeResult r;
    const auto t0 = Clock::now();
    for (int32_t s = 0; s < steps; ++s) {
        auto logits = model.forward(std::span<const int32_t>(&tok, 1), Transformer::Logits::Last, rig.route, rig.pool, hooks);
        if (!logits) break;
        tok = Sampler::argmax(logits.value());
        r.exits += model.last_exit_layer() >= 0 ? 1 : 0;
    }
    r.tps = steps * 1000.0 / ms_since(t0);
    return r;
}

}  // namespace

TEST_CASE("[bench] early exit and head pruning: decode throughput") {
    if (!benchmarks_enabled()) return;
    test::TinyModelSpec spec;  // larger random-weight model for timing
    spec.n_layers = 12;
    spec.n_embd = 512;
    spec.n_ff = 1408;
    spec.n_head = 8;
    spec.n_head_kv = 2;
    spec.context_length = 2048;
    const std::string path = temp_path("bench.gguf");
    test::write_tiny_model(path, spec);
    auto model = load(path, 2048);
    REQUIRE(model != nullptr);
    Rig rig;
    const int32_t steps = 48;

    const DecodeResult base = bench_decode(*model, rig, 32, steps, nullptr);
    EarlyExit realistic({0.98f, -1, 2});
    ForwardHooks h1{&realistic, nullptr};
    const DecodeResult ee = bench_decode(*model, rig, 32, steps, &h1);
    EarlyExit forced({-1.0f, -1, 2});
    ForwardHooks h2{&forced, nullptr};
    const DecodeResult ee_forced = bench_decode(*model, rig, 32, steps, &h2);
    std::printf("  early exit (12 blocks, d=512, ctx 32):\n"
                "    baseline           %7.1f tok/s\n"
                "    threshold 0.98     %7.1f tok/s  exits %lld/%d (random weights are rarely confident)\n"
                "    forced at block 6  %7.1f tok/s  exits %lld/%d (upper bound of the saving)\n",
                base.tps, ee.tps, static_cast<long long>(ee.exits), steps, ee_forced.tps,
                static_cast<long long>(ee_forced.exits), steps);

    std::vector<const TensorView*> wo;
    for (int32_t l = 0; l < spec.n_layers; ++l) wo.push_back(&model->attn_output(l));
    auto pruner = HeadPruner::create(wo, spec.n_head, spec.n_embd / spec.n_head, {0.5f});
    REQUIRE(pruner.has_value());
    pruner->set_active(true);
    ForwardHooks h3{nullptr, &pruner.value()};
    for (const int32_t ctx : {64, 1536}) {
        const DecodeResult full = bench_decode(*model, rig, ctx, steps, nullptr);
        const DecodeResult pruned = bench_decode(*model, rig, ctx, steps, &h3);
        std::printf("  head pruning 50%% at context %4d: %7.1f -> %7.1f tok/s (%+.0f%%)\n", ctx, full.tps, pruned.tps,
                    100.0 * (pruned.tps / full.tps - 1.0));
    }
    std::remove(path.c_str());
}

TEST_CASE("[bench] weight streaming I/O: mmap vs pread vs direct I/O") {
    if (!benchmarks_enabled()) return;
    const char* mb_env = std::getenv("LIYAB_BENCH_MB");
    const size_t mb = mb_env ? static_cast<size_t>(std::max(1, std::atoi(mb_env))) : 64;
    const std::string path = temp_path("io.bin");
    {
        std::vector<uint8_t> block(1 << 20);
        std::mt19937 rng(5);
        for (auto& b : block) b = static_cast<uint8_t>(rng());
        std::ofstream out(path, std::ios::binary);
        for (size_t i = 0; i < mb; ++i) out.write(reinterpret_cast<const char*>(block.data()), static_cast<std::streamsize>(block.size()));
    }
    const double bytes = static_cast<double>(mb) * 1024 * 1024;
    auto drop_cache = [&] {
#if defined(__linux__)
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd >= 0) {
            posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);  // evicts this file's clean pages (no root needed)
            ::close(fd);
        }
#endif
    };
    auto report = [&](const char* name, double ms, size_t chunks) {
        std::printf("    %-34s %8.1f MB/s  %7.3f ms/chunk\n", name, bytes / (1024 * 1024) / (ms / 1000.0),
                    chunks ? ms / static_cast<double>(chunks) : 0.0);
    };
    std::printf("  streaming %zu MiB in 1 MiB chunks%s:\n", mb,
#if defined(__linux__)
                " (page cache dropped before each run)"
#else
                " (macOS: mmap/pread runs may hit a warm page cache)"
#endif
    );

    {  // mmap + touch, the default engine path
        drop_cache();
        const int fd = ::open(path.c_str(), O_RDONLY);
        void* map = mmap(nullptr, static_cast<size_t>(bytes), PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        REQUIRE(map != MAP_FAILED);
        const auto t0 = Clock::now();
        volatile uint8_t sink = 0;
        const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
        for (size_t off = 0; off < static_cast<size_t>(bytes); off += page) sink ^= static_cast<const uint8_t*>(map)[off];
        report("mmap + page touch", ms_since(t0), mb);
        munmap(map, static_cast<size_t>(bytes));
    }
    for (const auto& [name, uring, bypass, qd] : std::vector<std::tuple<const char*, bool, bool, uint32_t>>{
             {"pread (page cache)", false, false, 1},
             {"direct, queue depth 1", true, true, 1},
             {"direct, queue depth 8", true, true, 8}}) {
        drop_cache();
        DirectReaderOptions options;
        options.use_io_uring = uring;
        options.bypass_page_cache = bypass;
        options.queue_depth = qd;
        auto reader = DirectReader::open(path, options);
        REQUIRE(reader.has_value());
        size_t chunks = 0;
        uint64_t checksum = 0;
        const auto t0 = Clock::now();
        CHECK(reader.value()->stream(0, static_cast<uint64_t>(bytes), [&](const ReadChunk& c) {
            checksum += c.data[0];
            ++chunks;
            return true;
        }).is_ok());
        const std::string label = std::string(name) + " [" + io_method_name(reader.value()->method()) + "]";
        report(label.c_str(), ms_since(t0), chunks);
    }
    std::remove(path.c_str());
}

TEST_MAIN()
