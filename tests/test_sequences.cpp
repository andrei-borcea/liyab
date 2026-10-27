// Tests for several sequences decoded together: Transformer::forward_batch()
// gives every sequence the bits forward() gives it alone, on every block kind
// and cache layout (attention with and without biases, Q/K norms, windowed
// blocks, fused projections, DeltaNet hybrids with rollback checkpoints,
// mixtures of experts read in place or streamed, latent attention in both
// forms, LoRA adapters per sequence, quantized and F16 caches), whatever the
// other sequences in the pass do; then the engine's sequences (Engine::
// open_sequence): concurrent generations give each one its output alone, with
// priorities, cancellation, adapters and the prefix cache per sequence.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "core/lora.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "core/transformer.h"
#include "liyab/backend.h"
#include "liyab/engine.h"
#include "liyab/liyab_c_api.h"
#include "liyab/mmap_loader.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

std::string temp_path(const std::string& name) {
    return test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_seq_" + name;
}

// Models written once per process, removed at exit.
const std::string& model_path(const std::string& name, const test::TinyModelSpec& spec) {
    struct Written {
        std::deque<std::pair<std::string, std::string>> files;
        ~Written() {
            for (const auto& file : files) std::remove(file.second.c_str());
        }
    };
    static Written written;
    for (const auto& [n, p] : written.files) {
        if (n == name) return p;
    }
    written.files.emplace_back(name, temp_path(name + ".gguf"));
    test::write_tiny_model(written.files.back().second, spec);
    return written.files.back().second;
}

std::unique_ptr<Transformer> load_transformer(const std::string& path, const TransformerOptions& options) {
    auto file = MmapLoader::open(path);
    if (!file) return nullptr;
    auto model = Transformer::load(std::move(file).value(), options);
    if (!model) {
        std::printf("  load failed: %s\n", model.status().to_string().c_str());
        return nullptr;
    }
    return std::move(model).value();
}

int32_t argmax(std::span<const float> row) {
    return static_cast<int32_t>(std::max_element(row.begin(), row.end()) - row.begin());
}

bool same_bits(std::span<const float> a, std::span<const float> b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size_bytes()) == 0;
}

// Token t of sequence `seed`'s deterministic stream.
int32_t stream_token(int32_t seed, int32_t t, int32_t vocab) { return 3 + (t * 37 + seed * 101 + 11) % (vocab - 3); }

// One sequence of a schedule: the passes of its prompt (`chunks` tokens
// each), then `steps` greedy decoding steps, starting at pass `start` of the
// schedule.
struct Plan {
    int32_t seed = 0;
    std::vector<int32_t> chunks;
    int32_t steps = 0;
    int32_t start = 0;
    const LoraAdapter* adapter = nullptr;
    float scale = 1.0f;
};

// The logits rows a plan asks for (its prompt's last pass, then each step),
// and the tokens it decodes.
struct Trace {
    std::vector<std::vector<float>> rows;
    std::vector<int32_t> decoded;
};

struct Rig {
    std::unique_ptr<ThreadPool> pool = std::make_unique<ThreadPool>(4);
    std::unique_ptr<Backend> cpu = make_cpu_backend(*pool);
    Route route{cpu.get(), cpu.get(), cpu.get()};
};

// The plan alone, on a model of its own: forward() per pass.
bool run_alone(const std::string& path, const TransformerOptions& options, const Plan& plan, int32_t rollback,
               Rig& rig, Trace& trace) {
    auto model = load_transformer(path, options);
    if (!model) return false;
    model->set_rollback_window(rollback);
    if (plan.adapter != nullptr && !model->set_adapter(plan.adapter, plan.scale, rig.pool.get()).is_ok()) return false;
    const int32_t vocab = model->config().n_vocab;
    int32_t t = 0;
    for (size_t c = 0; c < plan.chunks.size(); ++c) {
        std::vector<int32_t> tokens;
        for (int32_t i = 0; i < plan.chunks[c]; ++i) tokens.push_back(stream_token(plan.seed, t++, vocab));
        const bool last = c + 1 == plan.chunks.size();
        auto r = model->forward(tokens, last ? Transformer::Logits::Last : Transformer::Logits::None, rig.route,
                                *rig.pool);
        if (!r) return false;
        if (last) trace.rows.emplace_back(r->begin(), r->end());
    }
    for (int32_t s = 0; s < plan.steps; ++s) {
        const int32_t token = argmax(trace.rows.back());
        trace.decoded.push_back(token);
        model->set_checkpointing(false);  // a plain decoding step, as the engine runs it
        auto r = model->forward(std::span<const int32_t>(&token, 1), Transformer::Logits::Last, rig.route, *rig.pool);
        if (!r) return false;
        trace.rows.emplace_back(r->begin(), r->end());
    }
    return true;
}

// Every plan on one model: plan 0 in the live context, the others in
// contexts of their own, one forward_batch() per pass of the schedule over
// every plan with a pass to run then. Returns false when a row differs from
// the plan's alone.
bool run_batched(const std::string& path, const TransformerOptions& options, const std::vector<Plan>& plans,
                 int32_t rollback, Rig& rig, const std::vector<Trace>& alone, int32_t* max_items = nullptr) {
    auto model = load_transformer(path, options);
    if (!model) return false;
    model->set_rollback_window(rollback);
    const int32_t vocab = model->config().n_vocab;
    std::vector<std::unique_ptr<Transformer::Context>> contexts(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        if (i > 0) {
            auto made = model->make_context();
            if (!made) return false;
            contexts[i] = std::move(made).value();
        }
        if (plans[i].adapter == nullptr) continue;
        if (i > 0) model->swap_context(*contexts[i]);
        const Status s = model->set_adapter(plans[i].adapter, plans[i].scale, rig.pool.get());
        if (i > 0) model->swap_context(*contexts[i]);
        if (!s.is_ok()) return false;
    }
    struct Cursor {
        size_t chunk = 0;
        int32_t token = 0;  // next prompt token
        int32_t step = 0;
        std::vector<int32_t> tokens;
        Trace trace;
    };
    std::vector<Cursor> cursors(plans.size());
    bool ok = true;
    std::vector<std::span<const float>> out;
    for (int32_t pass = 0;; ++pass) {
        std::vector<Transformer::BatchItem> items;
        std::vector<size_t> owners;
        for (size_t i = 0; i < plans.size(); ++i) {
            const Plan& p = plans[i];
            Cursor& cur = cursors[i];
            if (pass < p.start) continue;
            cur.tokens.clear();
            Transformer::BatchItem item;
            item.context = i == 0 ? nullptr : contexts[i].get();
            if (cur.chunk < p.chunks.size()) {
                for (int32_t k = 0; k < p.chunks[cur.chunk]; ++k) cur.tokens.push_back(stream_token(p.seed, cur.token++, vocab));
                item.logits = cur.chunk + 1 == p.chunks.size() ? Transformer::Logits::Last : Transformer::Logits::None;
                item.checkpoint = true;  // as forward() checkpoints a prompt by default
                ++cur.chunk;
            } else if (cur.step < p.steps) {
                cur.tokens.push_back(argmax(cur.trace.rows.back()));
                cur.trace.decoded.push_back(cur.tokens.back());
                item.logits = Transformer::Logits::Last;
                ++cur.step;
            } else {
                continue;
            }
            item.tokens = cur.tokens;
            items.push_back(item);
            owners.push_back(i);
        }
        if (items.empty()) break;
        if (max_items != nullptr) *max_items = std::max(*max_items, static_cast<int32_t>(items.size()));
        if (Status s = model->forward_batch(items, rig.route, *rig.pool, out); !s.is_ok()) {
            std::printf("  forward_batch failed: %s\n", s.to_string().c_str());
            return false;
        }
        for (size_t k = 0; k < items.size(); ++k) {
            if (items[k].logits == Transformer::Logits::Last) cursors[owners[k]].trace.rows.emplace_back(out[k].begin(), out[k].end());
        }
    }
    for (size_t i = 0; i < plans.size(); ++i) {
        const Trace& a = alone[i];
        const Trace& b = cursors[i].trace;
        if (a.rows.size() != b.rows.size() || a.decoded != b.decoded) {
            std::printf("  sequence %zu: %zu rows / %zu tokens alone, %zu / %zu together\n", i, a.rows.size(),
                        a.decoded.size(), b.rows.size(), b.decoded.size());
            ok = false;
            continue;
        }
        for (size_t r = 0; r < a.rows.size(); ++r) {
            if (!same_bits(a.rows[r], b.rows[r])) {
                std::printf("  sequence %zu, row %zu: other bits than alone\n", i, r);
                ok = false;
                break;
            }
        }
    }
    // Each context ends where its plan alone leaves it.
    for (size_t i = 0; i < plans.size(); ++i) {
        int32_t expected = 0;
        for (const int32_t c : plans[i].chunks) expected += c;
        expected += plans[i].steps;
        const int32_t held = i == 0 ? model->n_past() : Transformer::positions_of(*contexts[i]);
        if (held != expected) {
            std::printf("  sequence %zu holds %d positions, expected %d\n", i, held, expected);
            ok = false;
        }
    }
    return ok;
}

// The plans alone, then together; true when every row matches.
bool check_schedule(const std::string& name, const std::string& path, const TransformerOptions& options,
                    const std::vector<Plan>& plans, int32_t rollback = 0) {
    Rig rig;
    std::vector<Trace> alone(plans.size());
    for (size_t i = 0; i < plans.size(); ++i) {
        if (!run_alone(path, options, plans[i], rollback, rig, alone[i])) {
            std::printf("  %s: sequence %zu alone failed\n", name.c_str(), i);
            return false;
        }
    }
    int32_t widest = 0;
    const bool ok = run_batched(path, options, plans, rollback, rig, alone, &widest);
    if (!ok) std::printf("  %s: sequences together differ from alone\n", name.c_str());
    if (widest < 2) {
        std::printf("  %s: no pass held several sequences\n", name.c_str());
        return false;
    }
    return ok;
}

// Three sequences: one decodes while another processes its prompt in
// passes, a third joins later; prompts of different lengths.
std::vector<Plan> mixed_plans() {
    return {Plan{1, {9}, 12, 0}, Plan{2, {5, 7, 3}, 9, 1}, Plan{3, {20}, 6, 4}};
}

}  // namespace

TEST_CASE("Sequences together: dense attention models give each sequence its bits alone") {
    struct Spec {
        const char* name;
        test::TinyModelSpec spec;
        KvCacheType kv;
    };
    std::vector<Spec> specs;
    specs.push_back({"llama-q8", {}, KvCacheType::Q8_0});
    {
        test::TinyModelSpec s;
        s.arch = "qwen2";  // QKV bias, NeoX RoPE
        specs.push_back({"qwen2-f16", s, KvCacheType::F16});
    }
    {
        test::TinyModelSpec s;
        s.arch = "qwen3";
        s.qk_norm = true;
        s.attn_gate = false;
        specs.push_back({"qwen3-q4", s, KvCacheType::Q4_0});
    }
    {
        test::TinyModelSpec s;
        s.arch = "qwen3";
        s.qk_norm = true;
        s.k_norm_outlier = 40.0f;  // key scales in a quantized cache
        specs.push_back({"qwen3-outlier-q8", s, KvCacheType::Q8_0});
    }
    {
        test::TinyModelSpec s;
        s.arch = "llama";
        s.attn_gate = true;
        s.rope_dim = 16;  // partial RoPE
        specs.push_back({"gated-partial-rope", s, KvCacheType::Q8_0});
    }
    for (const Spec& sp : specs) {
        TransformerOptions options;
        options.kv_type = sp.kv;
        const std::string& path = model_path(std::string("dense_") + sp.name, sp.spec);
        CHECK(check_schedule(sp.name, path, options, mixed_plans()));
    }
}

TEST_CASE("Classic models: sequences keep positions, parallel inputs, norms and logits separate") {
    for (const std::string arch : {"gpt2", "gptneox", "bloom", "olmo", "stablelm", "command-r", "apertus", "dbrx"}) {
        for (const KvCacheType kv : {KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0, KvCacheType::Q4_1}) {
            const auto s = test::classic_model_spec(arch);
            TransformerOptions options;
            options.kv_type = kv;
            CHECK(check_schedule("classic_" + arch, model_path("classic_" + arch, s), options, mixed_plans()));
        }
    }
}

TEST_CASE("Sequences together: windowed blocks and fused projections keep each sequence's bits") {
    for (const std::string arch : {"gemma3", "gemma2", "phi3"}) {
        test::TinyModelSpec s;
        s.arch = arch;
        s.sliding_window = 16;
        if (arch == "gemma3") {
            s.n_layers = 6;
            s.qk_norm = true;
            s.post_norms = true;
            s.tied_output = true;
        } else if (arch == "gemma2") {
            s.n_layers = 4;
            s.post_norms = true;
            s.attn_softcap = 50.0f;
            s.final_softcap = 30.0f;
        } else {
            s.fused_qkv = true;
            s.fused_gate_up = true;
        }
        TransformerOptions options;
        options.context_length = 512;
        options.max_batch = 8;
        const std::string& path = model_path("windowed_" + arch, s);
        // Prompts longer than the window, decoding past it.
        const std::vector<Plan> plans = {Plan{1, {8, 8, 8}, 20, 0}, Plan{2, {30}, 14, 2}, Plan{3, {3}, 25, 1}};
        CHECK(check_schedule("windowed " + arch, path, options, plans));
    }
}

TEST_CASE("Sequences together: hybrid (DeltaNet) models continue each sequence's states, checkpoints included") {
    test::TinyModelSpec s;
    s.arch = "qwen35";
    s.n_layers = 4;
    s.delta_net_interval = 2;
    const std::string& path = model_path("hybrid", s);
    TransformerOptions options;
    CHECK(check_schedule("hybrid", path, options, mixed_plans()));
    CHECK(check_schedule("hybrid, rollback window", path, options, mixed_plans(), 4));

    // A parked context rolls back inside its own window after a shared pass.
    Rig rig;
    auto model = load_transformer(path, options);
    REQUIRE(model != nullptr);
    model->set_rollback_window(4);
    auto other = model->make_context();
    REQUIRE(other.has_value());
    const int32_t vocab = model->config().n_vocab;
    std::vector<int32_t> a, b;
    for (int32_t t = 0; t < 10; ++t) a.push_back(stream_token(1, t, vocab));
    for (int32_t t = 0; t < 7; ++t) b.push_back(stream_token(2, t, vocab));
    std::vector<std::span<const float>> out;
    Transformer::BatchItem items[2];
    items[0].tokens = a;
    items[0].logits = Transformer::Logits::None;
    items[0].checkpoint = true;
    items[1].context = other->get();
    items[1].tokens = b;
    items[1].logits = Transformer::Logits::All;
    items[1].checkpoint = true;
    REQUIRE(model->forward_batch(items, rig.route, *rig.pool, out).is_ok());
    const auto rows = static_cast<size_t>(vocab);
    const std::vector<float> all(out[1].begin(), out[1].end());
    model->swap_context(**other);
    REQUIRE(model->truncate(4).is_ok());  // position 3 is in the window of 4 behind 7
    auto again = model->forward(std::span<const int32_t>(b).subspan(4), Transformer::Logits::All, rig.route, *rig.pool);
    REQUIRE(again.has_value());
    CHECK(same_bits(*again, std::span<const float>(all).subspan(4 * rows)));
    model->swap_context(**other);
    CHECK(model->n_past() == 10);
}

TEST_CASE("Sequences together: a mixture of experts reads the union of the experts, each sequence its bits") {
    test::TinyModelSpec s;
    s.arch = "qwen3moe";
    s.n_layers = 3;
    s.n_expert = 16;
    s.n_expert_used = 4;
    s.identical_experts = false;
    s.n_ff_shared = 128;
    s.shared_gate = true;
    const std::string& path = model_path("moe", s);
    TransformerOptions in_place;
    in_place.expert_cache_bytes = 0;
    CHECK(check_schedule("moe in place", path, in_place, mixed_plans()));
    TransformerOptions streamed;
    streamed.expert_cache_bytes = 1;  // a cache of a few slots: reads, evictions and waits every pass
    CHECK(check_schedule("moe streamed", path, streamed, mixed_plans()));
    test::TinyModelSpec fused = s;
    fused.fused_gate_up = true;
    CHECK(check_schedule("moe fused gate/up streamed", model_path("moe_fused", fused), streamed, mixed_plans()));
}

TEST_CASE("Sequences together: latent attention takes each sequence's own form, absorbed or expanded") {
    for (const int32_t q_rank : {0, 64}) {
        test::TinyModelSpec s;
        s.arch = "deepseek2";
        s.n_head_kv = 1;
        s.rope_dim = 32;
        s.mla_kv_rank = 64;
        s.mla_q_rank = q_rank;
        const std::string& path = model_path("latent_q" + std::to_string(q_rank), s);
        for (const LatentForm form : {LatentForm::Absorbed, LatentForm::Expanded, LatentForm::Auto}) {
            TransformerOptions options;
            options.latent_form = form;
            const char* names[] = {"auto", "absorbed", "expanded"};
            CHECK(check_schedule(std::string("latent ") + names[static_cast<int>(form)], path, options, mixed_plans()));
        }
    }
}

namespace {

// One adapted matrix's A [rank x cols] and B [rows x rank] (per expert).
struct Pair {
    std::string base;
    int64_t cols = 0, rows = 0;
    int32_t rank = 0;
    int32_t experts = 0;
    std::vector<float> a, b;
};

// A LoRA adapter for `bases` of the model at `model_path`, F32, random.
std::string write_adapter(const std::string& name, const std::string& model_path, const std::vector<std::string>& bases,
                          int32_t rank, uint32_t seed, const std::string& arch = "llama") {
    auto file = MmapLoader::open(model_path);
    if (!file) return {};
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.1f, 0.1f);
    test::GgufWriter w;
    w.add_string("general.architecture", arch);
    w.add_string("general.type", "adapter");
    w.add_string("adapter.type", "lora");
    w.add_f32("adapter.lora.alpha", 2.0f * static_cast<float>(rank));
    for (const std::string& base : bases) {
        const TensorView* t = (*file)->tensor(base);
        if (t == nullptr) return {};
        const int32_t experts = t->n_dims >= 3 ? static_cast<int32_t>(t->ne[2]) : 0;
        const auto e = static_cast<size_t>(std::max(experts, 1));
        std::vector<float> a(e * static_cast<size_t>(rank * t->ne[0])), b(e * static_cast<size_t>(t->ne[1] * rank));
        for (float& x : a) x = u(rng);
        for (float& x : b) x = u(rng);
        std::vector<int64_t> ne_a = {t->ne[0], rank}, ne_b = {rank, t->ne[1]};
        if (experts > 0) {
            ne_a.push_back(experts);
            ne_b.push_back(experts);
        }
        w.add_tensor(base + ".lora_a", ne_a, DType::F32, a);
        w.add_tensor(base + ".lora_b", ne_b, DType::F32, b);
    }
    const std::string path = temp_path(name + ".gguf");
    w.write(path);
    return path;
}

}  // namespace

TEST_CASE("Sequences together: each sequence applies its own LoRA adapter, or none") {
    // Dense: attention, FFN and the LM head adapted.
    {
        const std::string& path = model_path("lora_dense", {});
        const std::vector<std::string> bases = {"blk.0.attn_q.weight", "blk.1.attn_v.weight", "blk.0.ffn_up.weight",
                                                "blk.1.ffn_down.weight", "output.weight"};
        const std::string pa = write_adapter("dense_a", path, bases, 8, 7);
        const std::string pb = write_adapter("dense_b", path, {"blk.1.attn_q.weight", "blk.0.ffn_gate.weight"}, 4, 9);
        auto a = LoraAdapter::load(pa);
        auto b = LoraAdapter::load(pb);
        REQUIRE(a.has_value() && b.has_value());
        std::vector<Plan> plans = mixed_plans();
        plans[0].adapter = a->get();
        plans[2].adapter = b->get();
        plans[2].scale = 0.5f;
        CHECK(check_schedule("lora dense", path, {}, plans));
        std::remove(pa.c_str());
        std::remove(pb.c_str());
    }
    // Mixture of experts: per-expert stacks adapted for one sequence, the shared expert for another.
    {
        test::TinyModelSpec s;
        s.arch = "qwen3moe";
        s.n_layers = 2;
        s.n_expert = 8;
        s.n_expert_used = 2;
        s.identical_experts = false;
        s.n_ff_shared = 128;
        const std::string& path = model_path("lora_moe", s);
        const std::string pa = write_adapter("moe_a", path, {"blk.0.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight"},
                                             4, 3, "qwen3moe");
        const std::string pb = write_adapter("moe_b", path, {"blk.1.ffn_gate_shexp.weight", "blk.0.attn_k.weight"}, 4,
                                             5, "qwen3moe");
        auto a = LoraAdapter::load(pa);
        auto b = LoraAdapter::load(pb);
        REQUIRE(a.has_value() && b.has_value());
        std::vector<Plan> plans = mixed_plans();
        plans[1].adapter = a->get();
        plans[2].adapter = b->get();
        TransformerOptions streamed;
        streamed.expert_cache_bytes = 1;
        CHECK(check_schedule("lora moe", path, {}, plans));
        CHECK(check_schedule("lora moe streamed", path, streamed, plans));
        std::remove(pa.c_str());
        std::remove(pb.c_str());
    }
}

TEST_CASE("forward_batch: a cancelled pass returns every context to where it started") {
    for (const bool hybrid : {false, true}) {
        test::TinyModelSpec s;
        if (hybrid) {
            s.arch = "qwen35";
            s.n_layers = 4;
            s.delta_net_interval = 2;
        }
        const std::string& path = model_path(hybrid ? "hybrid" : "dense_llama-q8", s);
        Rig rig;
        auto model = load_transformer(path, {});
        REQUIRE(model != nullptr);
        auto other = model->make_context();
        REQUIRE(other.has_value());
        const int32_t vocab = model->config().n_vocab;
        std::vector<int32_t> a, b, a2, b2;
        for (int32_t t = 0; t < 6; ++t) a.push_back(stream_token(1, t, vocab));
        for (int32_t t = 0; t < 9; ++t) b.push_back(stream_token(2, t, vocab));
        for (int32_t t = 6; t < 11; ++t) a2.push_back(stream_token(1, t, vocab));
        for (int32_t t = 9; t < 12; ++t) b2.push_back(stream_token(2, t, vocab));
        std::vector<std::span<const float>> out;
        Transformer::BatchItem items[2];
        items[0].tokens = a;
        items[1].context = other->get();
        items[1].tokens = b;
        REQUIRE(model->forward_batch(items, rig.route, *rig.pool, out).is_ok());
        // Snapshots at the start of the cancellable pass (hybrid models restore from them).
        model->snapshot_state();
        model->swap_context(**other);
        model->snapshot_state();
        model->swap_context(**other);
        std::atomic<bool> cancel{true};
        ForwardHooks hooks;
        hooks.cancel = &cancel;
        items[0].tokens = a2;
        items[1].tokens = b2;
        CHECK(model->forward_batch(items, rig.route, *rig.pool, out, &hooks).code() == ErrorCode::Cancelled);
        CHECK(model->n_past() == 6);
        CHECK(Transformer::positions_of(**other) == 9);
        REQUIRE(model->forward_batch(items, rig.route, *rig.pool, out).is_ok());
        const std::vector<float> ra(out[0].begin(), out[0].end()), rb(out[1].begin(), out[1].end());
        // The same as never cancelled.
        auto fresh = load_transformer(path, {});
        REQUIRE(fresh != nullptr);
        std::vector<int32_t> whole_a = a, whole_b = b;
        whole_a.insert(whole_a.end(), a2.begin(), a2.end());
        whole_b.insert(whole_b.end(), b2.begin(), b2.end());
        auto fa = fresh->forward(whole_a, Transformer::Logits::Last, rig.route, *rig.pool);
        REQUIRE(fa.has_value());
        CHECK(same_bits(*fa, ra));
        fresh->reset();
        auto fb = fresh->forward(whole_b, Transformer::Logits::Last, rig.route, *rig.pool);
        REQUIRE(fb.has_value());
        CHECK(same_bits(*fb, rb));
    }
}

TEST_CASE("forward_batch refuses what it cannot run together") {
    Rig rig;
    auto model = load_transformer(model_path("dense_llama-q8", {}), {});
    REQUIRE(model != nullptr);
    auto other = model->make_context();
    REQUIRE(other.has_value());
    const std::vector<int32_t> tokens = {1, 270, 300};
    std::vector<std::span<const float>> out;
    Transformer::BatchItem two[2];
    two[0].tokens = tokens;
    two[1].tokens = tokens;  // the live context twice
    CHECK(model->forward_batch(two, rig.route, *rig.pool, out).code() == ErrorCode::InvalidArgument);
    two[1].context = other->get();
    two[1].tokens = {};
    CHECK(model->forward_batch(two, rig.route, *rig.pool, out).code() == ErrorCode::InvalidArgument);
    Transformer::Context empty;
    two[1].context = &empty;
    two[1].tokens = tokens;
    CHECK(model->forward_batch(two, rig.route, *rig.pool, out).code() == ErrorCode::InvalidArgument);
    two[1].context = other->get();
    ForwardHooks hooks;
    struct Never : EarlyExitHook {
        bool probe_after(int32_t, int32_t) const override { return false; }
        bool should_exit(std::span<const float>) override { return false; }
    } never;
    hooks.early_exit = &never;
    CHECK(model->forward_batch(two, rig.route, *rig.pool, out, &hooks).code() == ErrorCode::InvalidArgument);
    CHECK(model->forward_batch(two, rig.route, *rig.pool, out).is_ok());
    CHECK(model->n_past() == 3);
    CHECK(Transformer::positions_of(**other) == 3);
    CHECK(out.size() == 2 && out[0].size() == static_cast<size_t>(model->config().n_vocab));
}

// ---------------------------------------------------------------------------
// The engine's sequences
// ---------------------------------------------------------------------------
namespace {

EngineConfig engine_config(const std::string& path, int32_t sequences) {
    EngineConfig c;
    c.model_path = path;
    c.n_threads = 4;
    c.thermal_polling = false;
    c.power.profile = PowerProfile::Performance;  // unpaced: keep tests fast
    c.backend = BackendKind::Cpu;                 // the same bits in any batch: the CPU backend's contract
    c.max_sequences = sequences;
    return c;
}

SamplingParams greedy(int32_t max_tokens) {
    SamplingParams p;
    p.temperature = 0.0f;
    p.max_tokens = max_tokens;
    return p;
}

SamplingParams sampled(int32_t max_tokens, uint64_t seed) {
    SamplingParams p;
    p.temperature = 0.9f;
    p.top_k = 20;
    p.top_p = 0.95f;
    p.seed = seed;
    p.max_tokens = max_tokens;
    return p;
}

// One generation of a test: on `seq` (nullptr: the engine's default sequence).
struct Job {
    Engine::Sequence* seq = nullptr;
    std::string prompt;
    SamplingParams params;
    std::string out;
    std::vector<int32_t> tokens;
    Status status;
    GenerationStats stats;
    std::function<bool(int32_t)> stop_after;  // optional: called with the tokens so far; true stops
};

void run_job(Engine& engine, Job& job) {
    auto on_token = [&](std::string_view piece, int32_t token) {
        job.out += piece;
        job.tokens.push_back(token);
        return !job.stop_after || !job.stop_after(static_cast<int32_t>(job.tokens.size()));
    };
    auto r = job.seq != nullptr ? job.seq->generate(job.prompt, job.params, on_token)
                                : engine.generate(job.prompt, job.params, on_token);
    job.status = r.status();
    if (r) job.stats = *r;
}

// Every job at once, each on a thread of its own, released together.
void run_together(Engine& engine, std::vector<Job>& jobs) {
    std::atomic<int32_t> ready{0};
    std::vector<std::thread> threads;
    for (Job& job : jobs) {
        threads.emplace_back([&] {
            ready.fetch_add(1);
            while (ready.load() < static_cast<int32_t>(jobs.size())) std::this_thread::yield();
            run_job(engine, job);
        });
    }
    for (std::thread& t : threads) t.join();
}

// Each job alone, on an engine of one sequence with a fresh context.
std::vector<std::string> run_alone(const EngineConfig& config, std::vector<Job> jobs) {
    EngineConfig one = config;
    one.max_sequences = 1;
    auto engine = Engine::create(one);
    std::vector<std::string> out;
    if (!engine) return out;
    for (Job& job : jobs) {
        engine.value()->reset_context();
        job.seq = nullptr;
        job.out.clear();
        job.tokens.clear();
        run_job(*engine.value(), job);
        out.push_back(job.status.is_ok() ? job.out : "<" + job.status.to_string() + ">");
    }
    return out;
}

std::vector<Job> four_jobs() {
    std::vector<Job> jobs(4);
    jobs[0].prompt = "hello world";
    jobs[0].params = greedy(24);
    jobs[1].prompt = "world hello world hello, how is the world";
    jobs[1].params = greedy(20);
    jobs[2].prompt = "hello";
    jobs[2].params = sampled(28, 1234);
    jobs[3].prompt = "a quick brown fox jumps over the lazy dog, hello world";
    jobs[3].params = greedy(16);
    return jobs;
}

// The jobs together on `config`'s engine (job 0 on the default sequence,
// the others on sequences of their own) against each alone.
bool check_together(const std::string& name, const EngineConfig& config, std::vector<Job> jobs,
                    Engine::SchedulerStats* stats_out = nullptr) {
    const std::vector<std::string> alone = run_alone(config, jobs);
    if (alone.size() != jobs.size()) {
        std::printf("  %s: the reference engine failed\n", name.c_str());
        return false;
    }
    auto engine = Engine::create(config);
    if (!engine) {
        std::printf("  %s: %s\n", name.c_str(), engine.status().to_string().c_str());
        return false;
    }
    std::vector<std::unique_ptr<Engine::Sequence>> sequences;
    for (size_t i = 1; i < jobs.size(); ++i) {
        auto seq = engine.value()->open_sequence();
        if (!seq) {
            std::printf("  %s: open_sequence: %s\n", name.c_str(), seq.status().to_string().c_str());
            return false;
        }
        jobs[i].seq = seq->get();
        sequences.push_back(std::move(seq).value());
    }
    run_together(*engine.value(), jobs);
    bool ok = true;
    for (size_t i = 0; i < jobs.size(); ++i) {
        if (!jobs[i].status.is_ok()) {
            std::printf("  %s: job %zu: %s\n", name.c_str(), i, jobs[i].status.to_string().c_str());
            ok = false;
        } else if (jobs[i].out != alone[i]) {
            std::printf("  %s: job %zu together \"%s\", alone \"%s\"\n", name.c_str(), i, jobs[i].out.c_str(),
                        alone[i].c_str());
            ok = false;
        }
    }
    const Engine::SchedulerStats st = engine.value()->scheduler_stats();
    std::printf("  %s: %llu passes, %llu shared (%llu rows), %llu stopped\n", name.c_str(),
                static_cast<unsigned long long>(st.passes), static_cast<unsigned long long>(st.shared_passes),
                static_cast<unsigned long long>(st.shared_rows), static_cast<unsigned long long>(st.preempted));
    if (stats_out != nullptr) *stats_out = st;
    return ok;
}

test::TinyModelSpec hybrid_spec() {
    test::TinyModelSpec s;
    s.arch = "qwen35";
    s.n_layers = 4;
    s.delta_net_interval = 2;
    return s;
}

test::TinyModelSpec moe_spec() {
    test::TinyModelSpec s;
    s.arch = "qwen3moe";
    s.n_layers = 3;
    s.n_expert = 16;
    s.n_expert_used = 4;
    s.identical_experts = false;
    return s;
}

test::TinyModelSpec latent_spec() {
    test::TinyModelSpec s;
    s.arch = "deepseek2";
    s.n_head_kv = 1;
    s.rope_dim = 32;
    s.mla_kv_rank = 64;
    return s;
}

}  // namespace

TEST_CASE("Engine sequences generate together, each one's output what it gives alone") {
    struct Model {
        const char* name;
        test::TinyModelSpec spec;
        int64_t expert_cache_mb = -1;
    };
    test::TinyModelSpec windowed;
    windowed.arch = "gemma2";
    windowed.n_layers = 4;
    windowed.sliding_window = 16;
    windowed.post_norms = true;
    for (const Model& m : {Model{"llama", {}}, Model{"hybrid", hybrid_spec()}, Model{"moe streamed", moe_spec(), 1},
                           Model{"latent", latent_spec()}, Model{"gemma2 windowed", windowed}}) {
        EngineConfig config = engine_config(model_path(std::string("engine_") + m.name, m.spec), 4);
        config.expert_cache_mb = m.expert_cache_mb;
        Engine::SchedulerStats st;
        CHECK(check_together(m.name, config, four_jobs(), &st));
        CHECK(st.shared_passes > 0);
    }
}

TEST_CASE("Engine sequences: speculative drafts run alone, shared steps keep every output") {
    for (const bool hybrid : {false, true}) {
        EngineConfig config =
            engine_config(hybrid ? model_path("engine_hybrid", hybrid_spec()) : model_path("engine_llama", {}), 3);
        config.lookup_drafts = true;  // suffix drafts, tree verification
        std::vector<Job> jobs = four_jobs();
        jobs.resize(3);
        CHECK(check_together(hybrid ? "drafts hybrid" : "drafts", config, jobs));
    }
}

TEST_CASE("Engine sequences: one is cancelled, the others go on as alone") {
    const EngineConfig config = engine_config(model_path("engine_llama", {}), 3);
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    auto a = engine.value()->open_sequence();
    auto b = engine.value()->open_sequence();
    REQUIRE(a.has_value() && b.has_value());
    std::vector<Job> jobs = four_jobs();
    jobs.resize(3);
    jobs[0].params.max_tokens = 64;
    jobs[1].seq = a->get();
    jobs[2].seq = b->get();
    jobs[2].params = greedy(64);
    Engine::Sequence* cancelled = b->get();
    jobs[2].stop_after = [cancelled](int32_t n) {
        if (n == 5) cancelled->cancel();
        return false;
    };
    const std::vector<std::string> alone = run_alone(config, jobs);
    run_together(*engine.value(), jobs);
    REQUIRE(jobs[2].status.is_ok());
    CHECK(jobs[2].stats.cancelled);
    CHECK(jobs[2].stats.generated_tokens <= 6);
    CHECK(jobs[0].out == alone[0]);
    CHECK(jobs[1].out == alone[1]);
    CHECK(!jobs[0].stats.cancelled && !jobs[1].stats.cancelled);
    // The cancelled sequence continues its conversation as a fresh one would.
    Job again;
    again.seq = b->get();
    again.prompt = "hello world";
    again.params = greedy(12);
    run_job(*engine.value(), again);
    std::vector<Job> ref(1);
    ref[0].prompt = again.prompt;
    ref[0].params = again.params;
    CHECK(again.out == run_alone(config, ref)[0]);
}

TEST_CASE("Engine sequences: limits, Busy, and engines that refuse them") {
    {
        auto engine = Engine::create(engine_config(model_path("engine_llama", {}), 1));
        REQUIRE(engine.has_value());
        CHECK(engine.value()->open_sequence().status().code() == ErrorCode::Unsupported);
    }
    auto engine = Engine::create(engine_config(model_path("engine_llama", {}), 2));
    REQUIRE(engine.has_value());
    auto a = engine.value()->open_sequence(SequencePriority::Background);
    REQUIRE(a.has_value());
    CHECK(a.value()->priority() == SequencePriority::Background);
    CHECK(a.value()->id() > 0);
    CHECK(engine.value()->open_sequence().status().code() == ErrorCode::Busy);
    CHECK(engine.value()->scheduler_stats().open_sequences == 2);
    // A second call on a sequence while one runs: Busy (a callback calls it).
    Status inner;
    Engine::Sequence* seq = a->get();
    auto r = seq->generate("hello", greedy(4), [&](std::string_view, int32_t) {
        inner = seq->prefill("world").code() == ErrorCode::Busy ? Status::ok() : Status(ErrorCode::Internal, "not busy");
        return true;
    });
    REQUIRE(r.has_value());
    CHECK(inner.is_ok());
    a.value().reset();  // closed: its slot can be taken again
    CHECK(engine.value()->scheduler_stats().open_sequences == 1);
    auto b = engine.value()->open_sequence();
    CHECK(b.has_value());
    // With a draft model: no sequences (its cache follows one).
    EngineConfig drafted = engine_config(model_path("engine_llama", {}), 2);
    drafted.draft_model_path = drafted.model_path;
    auto with_draft = Engine::create(drafted);
    REQUIRE(with_draft.has_value());
    CHECK(with_draft.value()->open_sequence().status().code() == ErrorCode::Unsupported);
}

TEST_CASE("Engine sequences: a background sequence yields, and both outputs stay their own") {
    test::TinyModelSpec bigger;
    bigger.n_layers = 8;
    bigger.n_embd = 512;
    bigger.n_head = 8;
    bigger.n_head_kv = 4;
    bigger.n_ff = 2048;
    bigger.context_length = 8192;
    for (const bool hybrid : {false, true}) {
        test::TinyModelSpec spec = bigger;
        if (hybrid) {
            spec.arch = "qwen35";
            spec.delta_net_interval = 2;
        }
        const EngineConfig config =
            engine_config(model_path(hybrid ? "engine_bigger_hybrid" : "engine_bigger", spec), 2);
        // A background sequence processes a long prompt; the user's chat
        // starts meanwhile. The background pass running is stopped between
        // two blocks (its work so far kept), the user's tokens come first,
        // and each output is its own all the same.
        std::string long_prompt;
        for (int32_t i = 0; i < 1500; ++i) long_prompt += i % 3 == 0 ? "hello world " : "world hello ";
        std::vector<Job> jobs(2);
        jobs[0].prompt = "hello world";
        jobs[0].params = greedy(24);
        jobs[1].prompt = long_prompt;
        jobs[1].params = greedy(8);
        const std::vector<std::string> alone = run_alone(config, jobs);
        auto engine = Engine::create(config);
        REQUIRE(engine.has_value());
        auto background = engine.value()->open_sequence(SequencePriority::Background);
        REQUIRE(background.has_value());
        jobs[1].seq = background->get();
        using Clock = std::chrono::steady_clock;
        Clock::time_point user_first{}, background_done{};
        jobs[0].stop_after = [&](int32_t n) {
            if (n == 1) user_first = Clock::now();
            return false;
        };
        std::thread bg([&] {
            run_job(*engine.value(), jobs[1]);
            background_done = Clock::now();
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(60));
        run_job(*engine.value(), jobs[0]);
        bg.join();
        REQUIRE(jobs[0].status.is_ok() && jobs[1].status.is_ok());
        CHECK(jobs[0].out == alone[0]);
        CHECK(jobs[1].out == alone[1]);
        CHECK(user_first < background_done);  // the user's first token before the background prompt ended
        const Engine::SchedulerStats st = engine.value()->scheduler_stats();
        std::printf("  %s: background prompt %d tokens in %.0f ms, %llu passes, %llu shared, %llu stopped for the user's\n",
                    hybrid ? "hybrid" : "dense", jobs[1].stats.prompt_tokens, jobs[1].stats.prefill_ms,
                    static_cast<unsigned long long>(st.passes), static_cast<unsigned long long>(st.shared_passes),
                    static_cast<unsigned long long>(st.preempted));
        // (A stop happens when the user's call arrives during a background
        // pass; arriving between two, it simply goes first: timing decides,
        // so the count is printed, not checked.)
    }
}

TEST_CASE("Engine sequences: a shared pass of background prompts stops for the user's, every output its own") {
    test::TinyModelSpec spec;
    spec.n_layers = 8;
    spec.n_embd = 512;
    spec.n_head = 8;
    spec.n_head_kv = 4;
    spec.n_ff = 2048;
    spec.context_length = 8192;
    for (const bool hybrid : {false, true}) {
        test::TinyModelSpec s = spec;
        if (hybrid) {
            s.arch = "qwen35";
            s.delta_net_interval = 2;
        }
        const EngineConfig config = engine_config(model_path(hybrid ? "engine_bigger_hybrid" : "engine_bigger", s), 3);
        std::string a, b;
        for (int32_t i = 0; i < 1200; ++i) {
            a += i % 3 == 0 ? "hello world " : "world hello ";
            b += i % 4 == 0 ? "world world " : "hello hello ";
        }
        std::vector<Job> jobs(3);
        jobs[0].prompt = "hello world";
        jobs[0].params = greedy(20);
        jobs[1].prompt = a;
        jobs[1].params = greedy(6);
        jobs[2].prompt = b;
        jobs[2].params = greedy(6);
        const std::vector<std::string> alone = run_alone(config, jobs);
        auto engine = Engine::create(config);
        REQUIRE(engine.has_value());
        auto s1 = engine.value()->open_sequence(SequencePriority::Background);
        auto s2 = engine.value()->open_sequence(SequencePriority::Background);
        REQUIRE(s1.has_value() && s2.has_value());
        jobs[1].seq = s1->get();
        jobs[2].seq = s2->get();
        // The two background prompts share passes; the user's chat arrives
        // meanwhile and goes first: a background pass running then is stopped
        // (when the chat arrives between two passes, nothing needs stopping).
        using Clock = std::chrono::steady_clock;
        Clock::time_point user_first{}, background_done[2]{};
        jobs[0].stop_after = [&](int32_t n) {
            if (n == 1) user_first = Clock::now();
            return false;
        };
        std::atomic<int32_t> ready{0};
        std::vector<std::thread> background;
        for (size_t i = 1; i < 3; ++i) {
            background.emplace_back([&, i] {
                ready.fetch_add(1);
                while (ready.load() < 2) std::this_thread::yield();
                run_job(*engine.value(), jobs[i]);
                background_done[i - 1] = Clock::now();
            });
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        run_job(*engine.value(), jobs[0]);
        for (std::thread& t : background) t.join();
        for (size_t i = 0; i < jobs.size(); ++i) {
            REQUIRE(jobs[i].status.is_ok());
            CHECK(jobs[i].out == alone[i]);
        }
        CHECK(user_first < background_done[0]);
        CHECK(user_first < background_done[1]);
        const Engine::SchedulerStats st = engine.value()->scheduler_stats();
        std::printf("  %s: %llu passes, %llu shared, %llu stopped for the user's\n", hybrid ? "hybrid" : "dense",
                    static_cast<unsigned long long>(st.passes), static_cast<unsigned long long>(st.shared_passes),
                    static_cast<unsigned long long>(st.preempted));
        CHECK(st.shared_passes > 0);

        // One of two background prefills sharing passes cancelled: it stops
        // (its pass, when running, at the next block), the other goes on, and
        // both conversations continue exactly.
        s1.value()->reset_context();
        s2.value()->reset_context();
        Status cancelled, finished;
        std::thread p1([&] { cancelled = s1.value()->prefill(a); });
        std::thread p2([&] { finished = s2.value()->prefill(b); });
        std::this_thread::sleep_for(std::chrono::milliseconds(40));
        s1.value()->cancel_prefill();
        p1.join();
        p2.join();
        CHECK(finished.is_ok());
        CHECK(cancelled.is_ok() || cancelled.code() == ErrorCode::Cancelled);  // it may have ended first
        for (size_t i = 1; i < 3; ++i) {
            jobs[i].out.clear();
            run_job(*engine.value(), jobs[i]);
            CHECK(jobs[i].status.is_ok());
            CHECK(jobs[i].out == alone[i]);
        }
    }
}

TEST_CASE("Engine sequences: each applies its own LoRA adapter") {
    const std::string& path = model_path("engine_llama", {});
    const std::string pa = write_adapter("engine_a", path, {"blk.0.attn_q.weight", "blk.1.ffn_down.weight", "output.weight"},
                                         8, 21);
    const std::string pb = write_adapter("engine_b", path, {"blk.1.attn_v.weight", "blk.0.ffn_up.weight"}, 4, 22);
    REQUIRE(!pa.empty() && !pb.empty());
    EngineConfig config = engine_config(path, 3);
    // Alone: each job with its adapter on a one-sequence engine.
    std::vector<Job> jobs = four_jobs();
    jobs.resize(3);
    const char* chosen[] = {"", "a", "b"};
    std::vector<std::string> alone;
    {
        EngineConfig one = config;
        one.max_sequences = 1;
        auto engine = Engine::create(one);
        REQUIRE(engine.has_value());
        REQUIRE(engine.value()->load_adapter("a", pa).is_ok());
        REQUIRE(engine.value()->load_adapter("b", pb).is_ok());
        for (size_t i = 0; i < jobs.size(); ++i) {
            REQUIRE(engine.value()->set_adapter(chosen[i]).is_ok());
            engine.value()->reset_context();
            Job job = jobs[i];
            run_job(*engine.value(), job);
            alone.push_back(job.out);
        }
        CHECK(alone[1] != alone[0] || alone[2] != alone[0]);  // the adapters change something
    }
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    REQUIRE(engine.value()->load_adapter("a", pa).is_ok());
    REQUIRE(engine.value()->load_adapter("b", pb).is_ok());
    auto s1 = engine.value()->open_sequence();
    auto s2 = engine.value()->open_sequence();
    REQUIRE(s1.has_value() && s2.has_value());
    REQUIRE(s1.value()->set_adapter("a").is_ok());
    REQUIRE(s2.value()->set_adapter("b").is_ok());
    CHECK(s1.value()->set_adapter("missing").code() == ErrorCode::InvalidArgument);
    jobs[1].seq = s1->get();
    jobs[2].seq = s2->get();
    run_together(*engine.value(), jobs);
    for (size_t i = 0; i < jobs.size(); ++i) {
        CHECK(jobs[i].status.is_ok());
        CHECK(jobs[i].out == alone[i]);
    }
    // The default sequence still applies none; unloading an adapter another
    // sequence selected takes it back to the base model.
    for (const AdapterInfo& info : engine.value()->adapters()) CHECK(!info.active);
    REQUIRE(engine.value()->unload_adapter("a").is_ok());
    Job base;
    base.seq = s1->get();
    base.prompt = jobs[1].prompt;
    base.params = jobs[1].params;
    run_job(*engine.value(), base);
    std::vector<Job> ref(1);
    ref[0].prompt = base.prompt;
    ref[0].params = base.params;
    CHECK(base.out == run_alone(config, ref)[0]);
    std::remove(pa.c_str());
    std::remove(pb.c_str());
}

TEST_CASE("Engine sequences share the prefix cache: a context one set aside, another restores") {
    EngineConfig config = engine_config(model_path("engine_llama", {}), 2);
    config.prefix_cache_mb = 64;
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    std::string prompt;
    for (int32_t i = 0; i < 60; ++i) prompt += "hello world ";
    std::string reply;
    {
        auto seq = engine.value()->open_sequence();
        REQUIRE(seq.has_value());
        auto r = seq.value()->generate(prompt, greedy(8), [&](std::string_view piece, int32_t) {
            reply += piece;
            return true;
        });
        REQUIRE(r.has_value());
    }  // closed: its context set aside
    CHECK(engine.value()->prefix_cache_stats().set_asides >= 1);
    const std::string turn = prompt + reply + " hello";
    Job job;
    job.prompt = turn;
    job.params = greedy(10);
    run_job(*engine.value(), job);
    REQUIRE(job.status.is_ok());
    CHECK(job.stats.cached_prefix_tokens >= 128);
    CHECK(engine.value()->prefix_cache_stats().restores >= 1);
    std::vector<Job> ref(1);
    ref[0].prompt = turn;
    ref[0].params = job.params;
    EngineConfig plain = config;
    plain.prefix_cache_mb = 0;
    CHECK(job.out == run_alone(plain, ref)[0]);
}

TEST_CASE("Engine sequences under one grammar at once: each output valid and its own") {
    const EngineConfig config = engine_config(model_path("engine_llama", {}), 3);
    auto grammar = Grammar::parse(R"(root ::= "hello" (" world" | " hello") (" world" | " hello") "!")");
    REQUIRE(grammar.has_value());
    std::vector<std::string> alone;
    {
        EngineConfig one = config;
        one.max_sequences = 1;
        auto engine = Engine::create(one);
        REQUIRE(engine.has_value());
        for (const char* prompt : {"hello", "world hello", "hello world hello"}) {
            engine.value()->reset_context();
            std::string out;
            GenerateOptions options;
            options.grammar = *grammar;
            auto r = engine.value()->generate(prompt, greedy(24), [&](std::string_view p, int32_t) {
                out += p;
                return true;
            }, options);
            REQUIRE(r.has_value());
            alone.push_back(out);
        }
    }
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    auto s1 = engine.value()->open_sequence();
    auto s2 = engine.value()->open_sequence();
    REQUIRE(s1.has_value() && s2.has_value());
    Engine::Sequence* seqs[] = {nullptr, s1->get(), s2->get()};
    const char* prompts[] = {"hello", "world hello", "hello world hello"};
    std::string outs[3];
    std::atomic<int32_t> ready{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (ready.load() < 3) std::this_thread::yield();
            GenerateOptions options;
            options.grammar = *grammar;
            auto cb = [&](std::string_view p, int32_t) {
                outs[i] += p;
                return true;
            };
            auto r = seqs[i] != nullptr ? seqs[i]->generate(prompts[i], greedy(24), cb, options)
                                        : engine.value()->generate(prompts[i], greedy(24), cb, options);
            CHECK(r.has_value());
        });
    }
    for (std::thread& t : threads) t.join();
    for (int i = 0; i < 3; ++i) {
        CHECK(outs[i] == alone[static_cast<size_t>(i)]);
        CHECK(grammar->matches(outs[i]));
    }
}

// ---------------------------------------------------------------------------
// Sequence memory: borrowed from the expert cache while a sequence is open
// ---------------------------------------------------------------------------
namespace {

// A hybrid (DeltaNet) mixture of experts, as Qwen3.6-35B-A3B: recurrent
// states per sequence and streamed experts.
test::TinyModelSpec hybrid_moe_spec() {
    test::TinyModelSpec s;
    s.arch = "qwen35moe";
    s.n_layers = 4;
    s.delta_net_interval = 2;
    s.n_expert = 64;
    s.n_expert_used = 4;
    s.identical_experts = false;
    return s;
}

// The expert cache in use now, from the engine's public plan.
uint64_t cache_bytes(const Engine& engine) { return engine.memory_plan().expert_cache_bytes; }

// One acquired expert holds the file's bytes (its slot was not released or
// reused under its user).
bool expert_intact(const Transformer& model, ExpertStore& store, int32_t layer, int32_t expert) {
    auto views = store.acquire(layer, expert);
    if (!views) return false;
    bool ok = true;
    int32_t compared = 0;
    const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
    for (size_t m = 0; m < 3; ++m) {
        const TensorView& v = (*views)[m];
        if (v.data == nullptr) continue;
        const TensorView* t = model.file().tensor("blk." + std::to_string(layer) + "." + names[m] + ".weight");
        if (t == nullptr || t->data == nullptr) return false;
        const uint8_t* want = t->data + static_cast<size_t>(expert) * v.nbytes;
        if (std::memcmp(v.data, want, v.nbytes) != 0) ok = false;
        ++compared;
    }
    store.release(layer, expert);
    return ok && compared > 0;
}

}  // namespace

TEST_CASE("Sequence memory: with no sequence open, max_sequences changes nothing (plan, cache, requant, output)") {
    struct Model {
        const char* name;
        test::TinyModelSpec spec;
        int64_t expert_cache_mb;
    };
    test::TinyModelSpec moe_q8 = hybrid_moe_spec();
    moe_q8.n_embd = 256;  // Q8_0 matrices of 256 columns: requant_bits -1 converts them under a tight cache
    for (const Model& m : {Model{"dense", {}, -1}, Model{"hybrid", hybrid_spec(), -1},
                           Model{"hybrid moe streamed", hybrid_moe_spec(), 8},
                           Model{"hybrid moe streamed tight", moe_q8, 1}}) {
        const std::string path = model_path(std::string("mem_") + m.name, m.spec);
        std::string out[2];
        Engine::MemoryPlan plans[2];
        int i = 0;
        for (const int32_t sequences : {1, 4}) {
            EngineConfig config = engine_config(path, sequences);
            config.expert_cache_mb = m.expert_cache_mb;
            config.requant_bits = -1;  // automatic: the decision a reservation at load used to flip
            config.lookup_drafts = true;
            auto engine = Engine::create(config);
            REQUIRE(engine.has_value());
            plans[i] = engine.value()->memory_plan();
            Job job;
            job.prompt = "hello world, a quick brown fox";
            job.params = greedy(24);
            run_job(*engine.value(), job);
            CHECK(job.status.is_ok());
            out[i] = job.out;
            // Still the same after a sequence came and went.
            if (sequences > 1 && m.expert_cache_mb != 1) {
                auto seq = engine.value()->open_sequence();
                REQUIRE(seq.has_value());
                seq.value().reset();
                const Engine::MemoryPlan after = engine.value()->memory_plan();
                CHECK(after.expert_cache_bytes == plans[i].expert_cache_bytes);
                CHECK(after.requant_bits == plans[i].requant_bits);
            }
            ++i;
        }
        std::printf("  %s: cache %.2f MiB, requant %d\n", m.name,
                    static_cast<double>(plans[0].expert_cache_bytes) / (1024.0 * 1024.0), plans[0].requant_bits);
        CHECK(plans[0].resident_bytes == plans[1].resident_bytes);
        CHECK(plans[0].expert_bytes == plans[1].expert_bytes);
        CHECK(plans[0].expert_cache_bytes == plans[1].expert_cache_bytes);
        CHECK(plans[0].recommended_bytes == plans[1].recommended_bytes);
        CHECK(plans[0].requant_bits == plans[1].requant_bits);
        CHECK(!out[0].empty());
        CHECK(out[0] == out[1]);
    }
}

TEST_CASE("Sequence memory: an open sequence borrows its states from the expert cache, closing gives them back") {
    EngineConfig config = engine_config(model_path("mem_hybrid moe streamed", hybrid_moe_spec()), 4);
    config.expert_cache_mb = 8;
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    const uint64_t full = cache_bytes(e);
    REQUIRE(full > 0);
    std::vector<std::unique_ptr<Engine::Sequence>> open;
    std::vector<uint64_t> after;
    for (int32_t i = 0; i < 3; ++i) {
        auto seq = e.open_sequence();
        REQUIRE(seq.has_value());
        open.push_back(std::move(seq).value());
        after.push_back(cache_bytes(e));
    }
    // Each one takes the same share (whole slots: at least its states).
    CHECK(after[0] < full);
    const uint64_t each = full - after[0];
    CHECK(after[1] == full - 2 * each);
    CHECK(after[2] == full - 3 * each);
    // They work, and each output stays its own.
    std::vector<Job> jobs = four_jobs();
    for (size_t i = 1; i < jobs.size(); ++i) jobs[i].seq = open[i - 1].get();
    const std::vector<std::string> alone = run_alone(config, jobs);
    run_together(e, jobs);
    for (size_t i = 0; i < jobs.size(); ++i) {
        CHECK(jobs[i].status.is_ok());
        CHECK(jobs[i].out == alone[i]);
    }
    open[1].reset();  // closed out of order: its share comes back all the same
    CHECK(cache_bytes(e) == full - 2 * each);
    open.clear();
    CHECK(cache_bytes(e) == full);
    // Plain transformers (no recurrent state) and in-place experts borrow nothing.
    auto dense = Engine::create(engine_config(model_path("engine_llama", {}), 2));
    REQUIRE(dense.has_value());
    CHECK(cache_bytes(*dense.value()) == 0);
    auto plain = dense.value()->open_sequence();
    CHECK(plain.has_value());
    EngineConfig moe = engine_config(model_path("engine_moe streamed", moe_spec()), 2);
    moe.expert_cache_mb = 1;
    auto plain_moe = Engine::create(moe);
    REQUIRE(plain_moe.has_value());
    const uint64_t moe_full = cache_bytes(*plain_moe.value());
    CHECK(moe_full > 0);
    auto seq = plain_moe.value()->open_sequence();
    CHECK(seq.has_value());
    CHECK(cache_bytes(*plain_moe.value()) == moe_full);
}

TEST_CASE("Sequence memory: a cache at its minimum refuses the sequence (OutOfMemory), nothing changes") {
    EngineConfig config = engine_config(model_path("mem_hybrid moe streamed", hybrid_moe_spec()), 3);
    config.expert_cache_mb = 1;  // every class at its minimum of slots
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    const uint64_t full = cache_bytes(e);
    auto seq = e.open_sequence();
    REQUIRE(!seq.has_value());
    CHECK(seq.status().code() == ErrorCode::OutOfMemory);
    std::printf("  refused: %s\n", seq.status().to_string().c_str());
    CHECK(cache_bytes(e) == full);
    CHECK(e.scheduler_stats().open_sequences == 1);
    Job job;
    job.prompt = "hello world";
    job.params = greedy(12);
    run_job(e, job);
    CHECK(job.status.is_ok());
    // The C ABI says the same.
    liyab_engine_config c;
    liyab_engine_config_default(&c);
    c.model_path = config.model_path.c_str();
    c.backend = LIYAB_BACKEND_CPU;
    c.power_profile = LIYAB_POWER_PERFORMANCE;
    c.thermal_polling = 0;
    c.n_threads = 4;
    c.max_sequences = 2;
    c.expert_cache_mb = 1;
    liyab_engine* engine_c = nullptr;
    REQUIRE(liyab_engine_create(&c, &engine_c) == LIYAB_OK);
    liyab_sequence* s = nullptr;
    CHECK(liyab_sequence_open(engine_c, LIYAB_SEQUENCE_FOREGROUND, &s) == LIYAB_ERR_OUT_OF_MEMORY);
    CHECK(s == nullptr);
    liyab_engine_destroy(engine_c);
}

TEST_CASE("Sequence memory: ExpertStore lends and repays slots while other threads acquire experts") {
    TransformerOptions options;
    options.expert_cache_bytes = int64_t{8} << 20;
    auto model = load_transformer(model_path("mem_hybrid moe streamed", hybrid_moe_spec()), options);
    REQUIRE(model != nullptr);
    ExpertStore* store = model->expert_store();
    REQUIRE(store != nullptr);
    const size_t full = store->capacity_bytes();
    const size_t slots = store->entries();
    const size_t states = model->context_state_bytes();
    REQUIRE(states > 0);
    // A loan covers what was asked in whole slots: at most one slot more
    // (this model has one size class).
    auto loan = store->lend(states);
    REQUIRE(loan.has_value());
    CHECK(loan->bytes >= states);
    CHECK(store->capacity_bytes() == full - loan->bytes);
    CHECK(store->lent_bytes() == loan->bytes);
    const size_t slot = full / slots;
    CHECK(loan->bytes < states + slot);
    store->repay(*loan);
    CHECK(store->capacity_bytes() == full);
    CHECK(store->entries() == slots);
    CHECK(store->lent_bytes() == 0);
    // More than the cache can spare: refused, nothing taken.
    auto too_much = store->lend(full);
    CHECK(!too_much.has_value());
    CHECK(too_much.status().code() == ErrorCode::OutOfMemory);
    CHECK(store->capacity_bytes() == full);
    CHECK(store->lend(0).has_value());

    // Readers acquire experts (pinned while they check the bytes) while a
    // thread lends and repays: a slot is never taken away from under a pin,
    // and every expert keeps the file's bytes.
    const int32_t n_expert = model->config().n_expert;
    std::atomic<bool> stop{false};
    std::atomic<int32_t> bad{0};
    std::atomic<int64_t> checked{0};
    std::vector<std::thread> readers;
    for (int32_t r = 0; r < 3; ++r) {
        readers.emplace_back([&, r] {
            std::mt19937 rng(static_cast<uint32_t>(r + 1));
            while (!stop.load()) {
                const auto layer = static_cast<int32_t>(rng() % 4);
                const auto expert = static_cast<int32_t>(rng() % static_cast<uint32_t>(n_expert));
                if (!expert_intact(*model, *store, layer, expert)) bad.fetch_add(1);
                checked.fetch_add(1);
            }
        });
    }
    std::vector<ExpertStore::Loan> loans;
    for (int32_t round = 0; round < 60 || checked.load() < 3000; ++round) {
        if (round % 3 != 2) {
            auto l = store->lend(states * static_cast<size_t>(1 + round % 4));
            if (l) loans.push_back(std::move(l).value());
        } else if (!loans.empty()) {
            store->repay(loans.front());
            loans.erase(loans.begin());
        }
    }
    for (const ExpertStore::Loan& l : loans) store->repay(l);
    stop.store(true);
    for (std::thread& t : readers) t.join();
    std::printf("  %lld experts checked while lending\n", static_cast<long long>(checked.load()));
    CHECK(bad.load() == 0);
    CHECK(checked.load() > 0);
    CHECK(store->capacity_bytes() == full);
    CHECK(store->lent_bytes() == 0);
}

TEST_CASE("Sequence memory: sequences open and close while another decodes, its output unchanged") {
    EngineConfig config = engine_config(model_path("mem_hybrid moe streamed", hybrid_moe_spec()), 3);
    config.expert_cache_mb = 8;
    auto engine = Engine::create(config);
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    const uint64_t full = cache_bytes(e);
    std::vector<Job> jobs(1);
    jobs[0].prompt = "a quick brown fox jumps over the lazy dog, hello world";
    jobs[0].params = greedy(400);
    const std::vector<std::string> alone = run_alone(config, jobs);
    std::atomic<bool> done{false};
    // The default sequence decodes the same reply several times over (a
    // fresh context each), while sequences open and close.
    std::vector<std::string> replies;
    std::thread decoder([&] {
        for (int32_t run = 0; run < 8; ++run) {
            e.reset_context();
            jobs[0].out.clear();
            jobs[0].tokens.clear();
            run_job(e, jobs[0]);
            replies.push_back(jobs[0].status.is_ok() ? jobs[0].out : jobs[0].status.to_string());
        }
        done.store(true);
    });
    int32_t opened = 0;
    int32_t with_work = 0;
    double open_ms = 0.0, close_ms = 0.0;
    auto since = [](std::chrono::steady_clock::time_point t0) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    };
    const auto start = std::chrono::steady_clock::now();
    while (!done.load()) {
        auto t0 = std::chrono::steady_clock::now();
        // Some do a little work of their own before closing: foreground ones
        // (a background sequence's passes wait while the default one decodes).
        const bool work = opened % 3 == 2;
        auto seq = e.open_sequence(work || opened % 2 == 0 ? SequencePriority::Foreground : SequencePriority::Background);
        open_ms += since(t0);
        REQUIRE(seq.has_value());
        ++opened;
        if (work) {
            Job other;
            other.seq = seq->get();
            other.prompt = "hello";
            other.params = greedy(3);
            run_job(e, other);
            CHECK(other.status.is_ok());
            ++with_work;
        }
        t0 = std::chrono::steady_clock::now();
        seq.value().reset();
        close_ms += since(t0);
    }
    decoder.join();
    std::printf("  %d sequences opened and closed during 8 generations of %zu tokens in %.0f ms (%d generated; "
                "%.1f ms opening, %.1f ms closing)\n",
                opened, jobs[0].tokens.size(), since(start), with_work, open_ms, close_ms);
    CHECK(opened > 8);
    REQUIRE(replies.size() == 8);
    for (const std::string& reply : replies) CHECK(reply == alone[0]);
    CHECK(cache_bytes(e) == full);
}

namespace {

int32_t collect(const char* piece, size_t len, int32_t, void* user) {
    static_cast<std::string*>(user)->append(piece, len);
    return 1;
}

}  // namespace

TEST_CASE("C API: sequences open, generate together as alone, cancel and close") {
    const std::string& path = model_path("engine_llama", {});
    liyab_engine_config config;
    liyab_engine_config_default(&config);
    CHECK(config.max_sequences == 1);
    config.model_path = path.c_str();
    config.backend = LIYAB_BACKEND_CPU;
    config.power_profile = LIYAB_POWER_PERFORMANCE;
    config.thermal_polling = 0;
    config.n_threads = 4;
    liyab_sampling_params params;
    liyab_sampling_params_default(&params);
    params.temperature = 0.0f;
    params.max_tokens = 20;
    const char* prompts[] = {"hello world", "world hello world", "hello"};
    // Alone: one sequence, a fresh context each.
    std::string alone[3];
    {
        liyab_engine* engine = nullptr;
        REQUIRE(liyab_engine_create(&config, &engine) == LIYAB_OK);
        liyab_sequence* seq = nullptr;
        CHECK(liyab_sequence_open(engine, LIYAB_SEQUENCE_FOREGROUND, &seq) == LIYAB_ERR_UNSUPPORTED);
        CHECK(seq == nullptr);
        for (int i = 0; i < 3; ++i) {
            REQUIRE(liyab_engine_reset_context(engine) == LIYAB_OK);
            REQUIRE(liyab_engine_generate(engine, prompts[i], &params, collect, &alone[i], nullptr) == LIYAB_OK);
        }
        liyab_engine_destroy(engine);
    }
    config.max_sequences = 3;
    liyab_engine* engine = nullptr;
    REQUIRE(liyab_engine_create(&config, &engine) == LIYAB_OK);
    liyab_sequence* a = nullptr;
    liyab_sequence* b = nullptr;
    REQUIRE(liyab_sequence_open(engine, LIYAB_SEQUENCE_FOREGROUND, &a) == LIYAB_OK);
    REQUIRE(liyab_sequence_open(engine, LIYAB_SEQUENCE_BACKGROUND, &b) == LIYAB_OK);
    liyab_sequence* c = nullptr;
    CHECK(liyab_sequence_open(engine, LIYAB_SEQUENCE_FOREGROUND, &c) == LIYAB_ERR_BUSY);
    CHECK(liyab_sequence_id(a) > 0 && liyab_sequence_id(b) > 0 && liyab_sequence_id(a) != liyab_sequence_id(b));
    CHECK(liyab_sequence_set_priority(b, LIYAB_SEQUENCE_FOREGROUND) == LIYAB_OK);
    liyab_grammar* grammar = nullptr;
    REQUIRE(liyab_grammar_parse(R"(root ::= "x")", &grammar) == LIYAB_OK);
    CHECK(liyab_sequence_generate(a, "hello", &params, collect, nullptr, grammar,
                                  [](const char*, size_t, char*, size_t, const liyab_grammar**, void*) -> size_t { return 0; },
                                  nullptr, nullptr, nullptr) == LIYAB_ERR_INVALID_ARGUMENT);
    std::string out[3];
    liyab_generation_stats stats[3] = {};
    std::atomic<int32_t> ready{0};
    std::vector<std::thread> threads;
    liyab_sequence* seqs[3] = {nullptr, a, b};
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([&, i] {
            ready.fetch_add(1);
            while (ready.load() < 3) std::this_thread::yield();
            const liyab_status st =
                seqs[i] == nullptr
                    ? liyab_engine_generate(engine, prompts[i], &params, collect, &out[i], &stats[i])
                    : liyab_sequence_generate(seqs[i], prompts[i], &params, collect, nullptr, nullptr, nullptr, &out[i],
                                              &stats[i], nullptr);
            CHECK(st == LIYAB_OK);
        });
    }
    for (std::thread& t : threads) t.join();
    for (int i = 0; i < 3; ++i) CHECK(out[i] == alone[i]);
    liyab_scheduler_stats sched;
    REQUIRE(liyab_engine_scheduler_stats(engine, &sched) == LIYAB_OK);
    CHECK(sched.open_sequences == 3 && sched.active_sequences == 0);
    CHECK(sched.shared_passes > 0);
    CHECK(sched.pass_ms > 0.0 && sched.gather_ms >= 0.0);
    CHECK(liyab_sequence_prefill(a, "hello world", 1) == LIYAB_OK);
    CHECK(liyab_sequence_reset_context(a) == LIYAB_OK);
    CHECK(liyab_sequence_set_adapter(a, "missing", 1.0f) == LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_sequence_set_adapter(a, nullptr, 1.0f) == LIYAB_OK);
    liyab_sequence_cancel(a);
    liyab_sequence_cancel_prefill(a);
    liyab_sequence_close(a);
    liyab_sequence_close(b);
    liyab_sequence_close(nullptr);
    REQUIRE(liyab_engine_scheduler_stats(engine, &sched) == LIYAB_OK);
    CHECK(sched.open_sequences == 1);
    liyab_grammar_destroy(grammar);
    liyab_engine_destroy(engine);
}

TEST_MAIN()
