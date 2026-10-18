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

TEST_MAIN()
