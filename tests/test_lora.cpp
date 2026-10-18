// Tests for LoRA adapters (core/lora.h): an adapter applied in the forward
// pass gives the logits of the base model with the adapter merged into its
// weights (dense blocks, the LM head, hybrid DeltaNet blocks, per-expert
// stacks of a mixture of experts, fused gate/up experts), a pass without
// one is the base model's bit for bit, and malformed or unsupported
// adapters are refused; then the engine's adapters: loading, switching
// per generation, the memory cap, and contexts kept apart by adapter (the
// live context, the prefix cache, saved states).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <random>
#include <span>
#include <string>
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
    return test::temp_dir() + "/liyab_" + std::to_string(test::process_id()) + "_lora_" + name;
}

// One adapted matrix: A [rank x cols] and B [rows x rank] per expert, row-major.
struct Pair {
    std::string base;
    int64_t cols = 0, rows = 0;
    int32_t rank = 0;
    int32_t experts = 0;  // 0: a matrix
    std::vector<float> a, b;
};

// Values as a tensor of `type` stores them (what the engine reads back).
std::vector<float> rounded(std::vector<float> v, DType type, int64_t row) {
    if (type == DType::F32) return v;
    if (type == DType::BF16) {  // as test::GgufWriter writes it: rounded to nearest even
        for (float& x : v) {
            uint32_t bits = 0;
            std::memcpy(&bits, &x, sizeof bits);
            bits = ((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16) << 16;
            std::memcpy(&x, &bits, sizeof bits);
        }
        return v;
    }
    std::vector<uint8_t> bytes(dtype_row_bytes(type, row));
    for (size_t r = 0; r < v.size() / static_cast<size_t>(row); ++r) {
        quant::quantize_row(type, v.data() + r * row, bytes.data(), row);
        quant::dequantize_row(type, bytes.data(), v.data() + r * row, row);
    }
    return v;
}

// Random pairs for `bases` of the model at `model_path`, stored as `type`.
std::vector<Pair> random_pairs(const std::string& model_path, const std::vector<std::string>& bases, int32_t rank,
                               DType type, uint32_t seed) {
    auto file = MmapLoader::open(model_path);
    std::vector<Pair> out;
    if (!file) return out;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.1f, 0.1f);
    for (const std::string& base : bases) {
        const TensorView* w = (*file)->tensor(base);
        if (w == nullptr) {
            std::printf("  no tensor %s\n", base.c_str());
            return {};
        }
        Pair p;
        p.base = base;
        p.cols = w->ne[0];
        p.rows = w->ne[1];
        p.rank = rank;
        p.experts = w->n_dims >= 3 ? static_cast<int32_t>(w->ne[2]) : 0;
        const auto e = static_cast<size_t>(std::max(p.experts, 1));
        p.a.resize(e * static_cast<size_t>(rank * p.cols));
        p.b.resize(e * static_cast<size_t>(p.rows * rank));
        for (float& x : p.a) x = u(rng);
        for (float& x : p.b) x = u(rng);
        p.a = rounded(std::move(p.a), type, p.cols);
        p.b = rounded(std::move(p.b), type, rank);
        out.push_back(std::move(p));
    }
    return out;
}

// Writes an adapter GGUF as the converters do; `extra` adds metadata.
void write_adapter(const std::string& path, const std::vector<Pair>& pairs, float alpha, DType type,
                   const std::function<void(test::GgufWriter&)>& extra = {}) {
    test::GgufWriter w;
    w.add_string("general.architecture", "llama");
    w.add_string("general.type", "adapter");
    w.add_string("adapter.type", "lora");
    w.add_f32("adapter.lora.alpha", alpha);
    if (extra) extra(w);
    for (const Pair& p : pairs) {
        std::vector<int64_t> ne_a = {p.cols, p.rank}, ne_b = {p.rank, p.rows};
        if (p.experts > 0) {
            ne_a.push_back(p.experts);
            ne_b.push_back(p.experts);
        }
        w.add_tensor(p.base + ".lora_a", ne_a, type, p.a);
        w.add_tensor(p.base + ".lora_b", ne_b, type, p.b);
    }
    w.write(path);
}

// A copy of the F32 model at `base_path` with each pair merged into its
// weights: W += s · B·A (per expert for a stack).
bool write_merged(const std::string& base_path, const std::string& out_path, const std::vector<Pair>& pairs,
                  float s) {
    std::filesystem::copy_file(base_path, out_path, std::filesystem::copy_options::overwrite_existing);
    auto file = MmapLoader::open(base_path);
    if (!file) return false;
    std::fstream out(out_path, std::ios::in | std::ios::out | std::ios::binary);
    for (const Pair& p : pairs) {
        const TensorView* w = (*file)->tensor(p.base);
        if (w == nullptr || w->type != DType::F32) return false;
        std::vector<float> merged(static_cast<size_t>(w->elements()));
        std::memcpy(merged.data(), w->data, merged.size() * sizeof(float));
        const auto experts = static_cast<size_t>(std::max(p.experts, 1));
        const auto cols = static_cast<size_t>(p.cols), rows = static_cast<size_t>(p.rows);
        const auto r = static_cast<size_t>(p.rank);
        for (size_t e = 0; e < experts; ++e) {
            const float* a = p.a.data() + e * r * cols;
            const float* b = p.b.data() + e * rows * r;
            float* m = merged.data() + e * rows * cols;
            for (size_t o = 0; o < rows; ++o) {
                for (size_t c = 0; c < cols; ++c) {
                    double sum = 0.0;
                    for (size_t k = 0; k < r; ++k) sum += static_cast<double>(b[o * r + k]) * a[k * cols + c];
                    m[o * cols + c] += static_cast<float>(s * sum);
                }
            }
        }
        out.seekp(static_cast<std::streamoff>(w->file_offset));
        out.write(reinterpret_cast<const char*>(merged.data()), static_cast<std::streamsize>(merged.size() * sizeof(float)));
    }
    return static_cast<bool>(out);
}

struct Model {
    std::unique_ptr<Transformer> model;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<Backend> cpu;
    Route route;
};

std::unique_ptr<Model> load_model(const std::string& path) {
    auto file = MmapLoader::open(path);
    if (!file) return nullptr;
    TransformerOptions options;
    options.context_length = 256;
    options.kv_type = KvCacheType::F16;
    options.max_batch = 32;
    options.latent_form = LatentForm::Absorbed;
    auto t = Transformer::load(std::move(file).value(), options);
    if (!t) {
        std::printf("  %s\n", t.status().to_string().c_str());
        return nullptr;
    }
    auto m = std::make_unique<Model>();
    m->model = std::move(t).value();
    m->pool = std::make_unique<ThreadPool>(3);
    m->cpu = make_cpu_backend(*m->pool);
    m->route = {m->cpu.get(), m->cpu.get(), m->cpu.get()};
    return m;
}

const std::vector<int32_t> kTokens = {1, 7, 42, 99, 5, 260, 17, 3, 88, 64, 12, 200};

// Every token's logits for kTokens on a fresh context: a prefill of all but
// the last 3 tokens, then 3 single-token steps (decoding).
std::vector<float> logits_of(Model& m) {
    m.model->reset();
    std::vector<float> out;
    const std::span<const int32_t> all(kTokens);
    auto first = m.model->forward(all.first(kTokens.size() - 3), Transformer::Logits::All, m.route, *m.pool);
    if (!first) return {};
    out.assign(first->begin(), first->end());
    for (size_t i = kTokens.size() - 3; i < kTokens.size(); ++i) {
        auto step = m.model->forward(all.subspan(i, 1), Transformer::Logits::Last, m.route, *m.pool);
        if (!step) return {};
        out.insert(out.end(), step->begin(), step->end());
    }
    return out;
}

double max_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 1e30;
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i) worst = std::max(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
    return worst;
}

double max_abs(const std::vector<float>& a) {
    double worst = 0.0;
    for (const float v : a) worst = std::max(worst, std::fabs(static_cast<double>(v)));
    return worst;
}

test::TinyModelSpec f32_spec(const char* arch) {
    test::TinyModelSpec spec;
    spec.arch = arch;
    spec.attn_type = DType::F32;
    spec.ffn_type = DType::F32;
    spec.embd_type = DType::F32;
    spec.output_type = DType::F32;
    return spec;
}

// Base + adapter against the merged model, for `bases` of a model of `spec`.
void check_merged(const char* name, const test::TinyModelSpec& spec, const std::vector<std::string>& bases,
                  DType type = DType::F32, int32_t rank = 8) {
    const std::string base_path = temp_path(std::string(name) + "_base.gguf");
    const std::string merged_path = temp_path(std::string(name) + "_merged.gguf");
    const std::string adapter_path = temp_path(std::string(name) + "_adapter.gguf");
    test::write_tiny_model(base_path, spec);
    const std::vector<Pair> pairs = random_pairs(base_path, bases, rank, type, 7);
    REQUIRE(pairs.size() == bases.size());
    constexpr float kAlpha = 16.0f, kUserScale = 0.75f;  // s = 0.75 * 16 / rank
    write_adapter(adapter_path, pairs, kAlpha, type);
    REQUIRE(write_merged(base_path, merged_path, pairs, kUserScale * kAlpha / static_cast<float>(rank)));

    auto base = load_model(base_path);
    auto merged = load_model(merged_path);
    REQUIRE(base != nullptr && merged != nullptr);
    auto adapter = LoraAdapter::load(adapter_path);
    REQUIRE(adapter.has_value());
    CHECK((*adapter)->targets().size() == bases.size());
    CHECK((*adapter)->max_rank() == rank);
    CHECK_NEAR((*adapter)->alpha(), kAlpha, 0.0);

    const std::vector<float> plain = logits_of(*base);
    const std::vector<float> reference = logits_of(*merged);
    REQUIRE(!plain.empty() && !reference.empty());
    REQUIRE(base->model->set_adapter(adapter->get(), kUserScale).is_ok());
    CHECK(base->model->adapter() == adapter->get());
    const std::vector<float> adapted = logits_of(*base);
    const double scale = max_abs(reference);
    const double moved = max_diff(plain, reference);
    const double error = max_diff(adapted, reference);
    std::printf("  %s: the adapter moves logits by %.4f (of %.3f); base + adapter vs merged: %.2e\n", name, moved,
                scale, error);
    CHECK(moved > 0.02 * scale);       // the adapter matters
    CHECK(error < 2e-3 * scale);       // and is the merged model's
    // Off again: the base model's bits; on again: the same bits as before,
    // the updates spread over the pool's threads or not.
    REQUIRE(base->model->set_adapter(nullptr).is_ok());
    CHECK(logits_of(*base) == plain);
    REQUIRE(base->model->set_adapter(adapter->get(), kUserScale).is_ok());
    CHECK(logits_of(*base) == adapted);
    REQUIRE(base->model->set_adapter(adapter->get(), kUserScale, base->pool.get()).is_ok());
    CHECK(logits_of(*base) == adapted);
    ThreadPool one(1);
    auto single = make_cpu_backend(one);
    const Route single_route{single.get(), single.get(), single.get()};
    base->model->reset();
    REQUIRE(base->model->set_adapter(adapter->get(), kUserScale, &one).is_ok());
    {
        std::vector<float> serial;
        const std::span<const int32_t> all(kTokens);
        auto first = base->model->forward(all.first(kTokens.size() - 3), Transformer::Logits::All, single_route, one);
        REQUIRE(first.has_value());
        serial.assign(first->begin(), first->end());
        for (size_t i = kTokens.size() - 3; i < kTokens.size(); ++i) {
            auto step = base->model->forward(all.subspan(i, 1), Transformer::Logits::Last, single_route, one);
            REQUIRE(step.has_value());
            serial.insert(serial.end(), step->begin(), step->end());
        }
        CHECK(serial == adapted);  // one thread: the same bits
    }
    CHECK(base->model->set_adapter(nullptr).is_ok());
    base.reset();
    merged.reset();
    std::remove(base_path.c_str());
    std::remove(merged_path.c_str());
    std::remove(adapter_path.c_str());
}

std::vector<std::string> per_block(int32_t layers, const std::vector<std::string>& names) {
    std::vector<std::string> out;
    for (int32_t l = 0; l < layers; ++l) {
        for (const std::string& n : names) out.push_back("blk." + std::to_string(l) + "." + n + ".weight");
    }
    return out;
}

}  // namespace

TEST_CASE("lora: dense blocks and the LM head give the merged model's logits") {
    std::vector<std::string> bases =
        per_block(2, {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"});
    bases.push_back("output.weight");
    check_merged("dense", f32_spec("llama"), bases);
}

TEST_CASE("lora: F16, BF16 and Q8_0 adapter tensors") {
    const std::vector<std::string> bases = per_block(2, {"attn_q", "attn_v", "ffn_down"});
    check_merged("f16", f32_spec("llama"), bases, DType::F16);
    check_merged("bf16", f32_spec("llama"), bases, DType::BF16);
    check_merged("q8_0", f32_spec("llama"), per_block(2, {"attn_q", "ffn_up"}), DType::Q8_0, 32);
}

TEST_CASE("lora: some matrices only, with QKV bias") {
    check_merged("qwen2", f32_spec("qwen2"), per_block(2, {"attn_k", "ffn_up"}));
}

TEST_CASE("lora: per-expert adapters of a mixture of experts") {
    test::TinyModelSpec spec = f32_spec("qwen3moe");
    spec.n_expert = 4;
    spec.n_expert_used = 2;
    spec.identical_experts = false;
    spec.n_ff_shared = 64;
    spec.qk_norm = true;
    check_merged("moe", spec,
                 per_block(2, {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_shexp", "ffn_down_shexp",
                               "ffn_gate_inp", "attn_q"}));
    spec.fused_gate_up = true;
    check_merged("moe_fused", spec, per_block(2, {"ffn_gate_up_exps", "ffn_down_exps"}));
}

TEST_CASE("lora: hybrid DeltaNet blocks") {
    test::TinyModelSpec spec = f32_spec("qwen35");
    spec.delta_net_interval = 2;
    spec.qk_norm = true;
    spec.attn_gate = true;
    check_merged("hybrid", spec, per_block(2, {"ffn_up"}));
    // The DeltaNet block's projections (block 0), the attention block's (block 1).
    check_merged("hybrid_mixers", spec,
                 {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.1.attn_q.weight",
                  "blk.1.attn_output.weight"});
}

TEST_CASE("lora: the update kernel, on any number of threads, against a double-precision sum") {
    // One adapted matrix large enough for the update to be spread over threads.
    constexpr int64_t kCols = 1024, kRows = 4096 + 77;  // a last column chunk of 77
    constexpr int32_t kRank = 64, kTokens = 5;
    std::mt19937 rng(21);
    std::uniform_real_distribution<float> u(-0.1f, 0.1f);
    for (const DType type : {DType::F32, DType::F16, DType::BF16}) {
        Pair p;
        p.base = "blk.0.ffn_up.weight";
        p.cols = kCols;
        p.rows = kRows;
        p.rank = kRank;
        p.a.resize(static_cast<size_t>(kRank * kCols));
        p.b.resize(static_cast<size_t>(kRows * kRank));
        for (float& x : p.a) x = u(rng);
        for (float& x : p.b) x = u(rng);
        p.a = rounded(std::move(p.a), type, kCols);
        p.b = rounded(std::move(p.b), type, kRank);
        const std::string path = temp_path("kernel.gguf");
        write_adapter(path, {p}, 128.0f, type);
        auto adapter = LoraAdapter::load(path);
        REQUIRE(adapter.has_value());
        const LoraTarget* t = (*adapter)->find(p.base);
        REQUIRE(t != nullptr && t->rows == kRows && t->rank == kRank);
        std::vector<float> x(static_cast<size_t>(kTokens * kCols)), y0(static_cast<size_t>(kTokens * kRows));
        for (float& v : x) v = u(rng);
        for (float& v : y0) v = u(rng);
        ThreadPool pool(4);
        auto cpu = make_cpu_backend(pool);
        auto run = [&](ThreadPool* threads) {
            std::vector<float> y = y0;
            LoraRunner runner;
            const LoraRunner::Item item{t, 0, 2.0f, x.data(), y.data(), kTokens};
            if (!runner.apply(*cpu, {&item, 1}, threads).is_ok()) return std::vector<float>();
            return y;
        };
        const std::vector<float> serial = run(nullptr), parallel = run(&pool);
        REQUIRE(!serial.empty());
        CHECK(serial == parallel);
        double worst = 0.0;
        for (int32_t tok = 0; tok < kTokens; ++tok) {
            std::vector<double> ax(kRank, 0.0);
            for (int32_t r = 0; r < kRank; ++r) {
                for (int64_t c = 0; c < kCols; ++c) ax[r] += static_cast<double>(p.a[r * kCols + c]) * x[tok * kCols + c];
            }
            for (int64_t o = 0; o < kRows; ++o) {
                double d = 0.0;
                for (int32_t r = 0; r < kRank; ++r) d += static_cast<double>(p.b[o * kRank + r]) * ax[r];
                const double want = y0[tok * kRows + o] + 2.0 * d;
                worst = std::max(worst, std::fabs(want - serial[tok * kRows + o]));
            }
        }
        std::printf("  %s: largest error %.2e\n", std::string(dtype_traits(type).name).c_str(), worst);
        CHECK(worst < 1e-4);
        std::remove(path.c_str());
    }
}

TEST_CASE("lora: malformed and unsupported adapters are refused") {
    const std::string base_path = temp_path("refused_base.gguf");
    test::write_tiny_model(base_path, f32_spec("llama"));
    auto m = load_model(base_path);
    REQUIRE(m != nullptr);
    const std::string path = temp_path("refused.gguf");
    std::vector<Pair> pairs = random_pairs(base_path, {"blk.0.attn_q.weight"}, 4, DType::F32, 3);
    REQUIRE(pairs.size() == 1);

    // A tensor the model does not have; one of another shape.
    Pair missing = pairs[0];
    missing.base = "blk.7.attn_q.weight";
    write_adapter(path, {missing}, 8.0f, DType::F32);
    auto a = LoraAdapter::load(path);
    REQUIRE(a.has_value());
    CHECK(m->model->set_adapter(a->get()).code() == ErrorCode::InvalidArgument);
    CHECK(m->model->adapter() == nullptr);
    Pair wrong = pairs[0];
    wrong.base = "blk.0.ffn_down.weight";  // 256 columns, not 128
    write_adapter(path, {wrong}, 8.0f, DType::F32);
    a = LoraAdapter::load(path);
    REQUIRE(a.has_value());
    CHECK(m->model->set_adapter(a->get()).code() == ErrorCode::InvalidArgument);

    // The token embeddings; activated LoRA; another adapter type.
    Pair embd = pairs[0];
    embd.base = "token_embd.weight";
    write_adapter(path, {embd}, 8.0f, DType::F32);
    CHECK(LoraAdapter::load(path).status().code() == ErrorCode::Unsupported);
    write_adapter(path, pairs, 8.0f, DType::F32,
                  [](test::GgufWriter& w) { w.add_i32_array("adapter.alora.invocation_tokens", {5, 6}); });
    CHECK(LoraAdapter::load(path).status().code() == ErrorCode::Unsupported);

    // Half a pair; a stray tensor; mismatched ranks; a model file.
    {
        test::GgufWriter w;
        w.add_string("general.type", "adapter");
        w.add_string("adapter.type", "lora");
        w.add_tensor("blk.0.attn_q.weight.lora_a", {128, 4}, DType::F32, pairs[0].a);
        w.write(path);
        CHECK(LoraAdapter::load(path).status().code() == ErrorCode::InvalidModel);
    }
    {
        test::GgufWriter w;
        w.add_string("general.type", "adapter");
        w.add_string("adapter.type", "lora");
        w.add_tensor("blk.0.attn_q.weight", {128, 4}, DType::F32, pairs[0].a);
        w.write(path);
        CHECK(LoraAdapter::load(path).status().code() == ErrorCode::InvalidModel);
    }
    {
        test::GgufWriter w;
        w.add_string("general.type", "adapter");
        w.add_string("adapter.type", "lora");
        w.add_tensor("blk.0.attn_q.weight.lora_a", {128, 4}, DType::F32, pairs[0].a);
        w.add_tensor("blk.0.attn_q.weight.lora_b", {2, 64}, DType::F32, std::vector<float>(128, 0.0f));
        w.write(path);
        CHECK(LoraAdapter::load(path).status().code() == ErrorCode::InvalidModel);
    }
    CHECK(LoraAdapter::load(base_path).status().code() == ErrorCode::InvalidModel);
    m.reset();
    std::remove(path.c_str());
    std::remove(base_path.c_str());

    // Latent attention's per-head matrices are not multiplied as a whole.
    test::TinyModelSpec mla = f32_spec("deepseek2");
    mla.n_head_kv = 1;
    mla.rope_dim = 32;
    mla.mla_kv_rank = 64;
    const std::string mla_path = temp_path("refused_mla.gguf");
    test::write_tiny_model(mla_path, mla);
    auto lm = load_model(mla_path);
    REQUIRE(lm != nullptr);
    const std::vector<Pair> kb = random_pairs(mla_path, {"blk.0.attn_k_b.weight"}, 4, DType::F32, 5);
    REQUIRE(kb.size() == 1);
    write_adapter(path, kb, 8.0f, DType::F32);
    a = LoraAdapter::load(path);
    REQUIRE(a.has_value());
    CHECK(lm->model->set_adapter(a->get()).code() == ErrorCode::Unsupported);
    // Its other matrices are.
    const std::vector<Pair> kv = random_pairs(mla_path, {"blk.0.attn_kv_a_mqa.weight", "blk.0.attn_output.weight"}, 4,
                                              DType::F32, 5);
    write_adapter(path, kv, 8.0f, DType::F32);
    a = LoraAdapter::load(path);
    REQUIRE(a.has_value());
    CHECK(lm->model->set_adapter(a->get()).is_ok());
    CHECK(lm->model->set_adapter(nullptr).is_ok());
    lm.reset();
    std::remove(path.c_str());
    std::remove(mla_path.c_str());
}

TEST_CASE("lora: latent attention's whole matrices give the merged model's logits") {
    test::TinyModelSpec mla = f32_spec("deepseek2");
    mla.n_head_kv = 1;
    mla.rope_dim = 32;
    mla.mla_kv_rank = 64;
    mla.mla_q_rank = 64;
    check_merged("mla", mla, per_block(2, {"attn_q_a", "attn_q_b", "attn_kv_a_mqa", "attn_output"}));
}

// ---- The engine's adapters --------------------------------------------------

namespace {

// A base model, two adapters (A on attention and FFN, B on the FFN only) and
// the base model with A merged at scale 1, written once.
struct EngineFiles {
    std::string base, merged_a, adapter_a, adapter_b;
    explicit EngineFiles(int32_t rank = 32) {
        base = temp_path("engine_base.gguf");
        merged_a = temp_path("engine_merged_a.gguf");
        adapter_a = temp_path("engine_a.gguf");
        adapter_b = temp_path("engine_b.gguf");
        test::write_tiny_model(base, f32_spec("llama"));
        const std::vector<Pair> a =
            random_pairs(base, per_block(2, {"attn_q", "attn_v", "attn_output", "ffn_up", "ffn_down"}), rank, DType::F32, 11);
        write_adapter(adapter_a, a, 2.0f * static_cast<float>(rank), DType::F32);  // s = 2
        write_merged(base, merged_a, a, 2.0f);
        write_adapter(adapter_b, random_pairs(base, per_block(2, {"ffn_gate", "ffn_up"}), 32, DType::F16, 12), 32.0f,
                      DType::F16);
    }
    ~EngineFiles() {
        for (const std::string& p : {base, merged_a, adapter_a, adapter_b}) std::remove(p.c_str());
    }
};

EngineConfig adapter_engine_config(const std::string& path) {
    EngineConfig c;
    c.model_path = path;
    c.n_threads = 3;
    c.thermal_polling = false;
    c.power.profile = PowerProfile::Performance;
    c.backend = BackendKind::Cpu;
    c.kv_cache_type = KvCacheType::F16;
    return c;
}

SamplingParams greedy(int32_t max_tokens) {
    SamplingParams p;
    p.temperature = 0.0f;
    p.max_tokens = max_tokens;
    return p;
}

std::vector<int32_t> prompt_of(size_t n, int32_t salt) {
    std::vector<int32_t> t = {1};
    for (size_t i = 1; i < n; ++i) t.push_back(3 + static_cast<int32_t>((i * 31 + static_cast<size_t>(salt) * 7) % 250));
    return t;
}

std::vector<int32_t> run(Engine& e, const std::vector<int32_t>& prompt, int32_t max_tokens = 12,
                         GenerationStats* stats = nullptr) {
    std::vector<int32_t> out;
    auto st = e.generate_tokens(prompt, greedy(max_tokens), [&](std::string_view, int32_t t) {
        out.push_back(t);
        return true;
    });
    if (!st) std::printf("  generate failed: %s\n", st.status().to_string().c_str());
    if (st && stats != nullptr) *stats = st.value();
    return out;
}

}  // namespace

TEST_CASE("engine adapters: selected per generation, the merged model's output, the base model's without") {
    const EngineFiles f;
    auto merged = Engine::create(adapter_engine_config(f.merged_a));
    auto engine = Engine::create(adapter_engine_config(f.base));
    REQUIRE(merged.has_value() && engine.has_value());
    Engine& e = **engine;
    const std::vector<int32_t> prompt = prompt_of(40, 1);
    const std::vector<int32_t> base_out = run(e, prompt);
    const std::vector<int32_t> merged_out = run(**merged, prompt);
    std::printf("  replies of %zu and %zu tokens\n", base_out.size(), merged_out.size());
    REQUIRE(!base_out.empty() && !merged_out.empty());
    CHECK(base_out != merged_out);  // the adapter changes the reply

    REQUIRE(e.load_adapter("a", f.adapter_a).is_ok());
    REQUIRE(e.load_adapter("b", f.adapter_b).is_ok());
    std::vector<AdapterInfo> info = e.adapters();
    REQUIRE(info.size() == 2);
    CHECK(info[0].name == "a" && info[0].rank == 32 && info[0].matrices == 10 && info[0].resident && !info[0].active);
    CHECK_NEAR(info[0].alpha, 64.0, 0.0);
    CHECK(info[1].name == "b" && info[1].matrices == 4);
    CHECK(e.describe().find("Adapter:    a, LoRA rank 32") != std::string::npos);
    // Loading selects nothing: still the base model, its context kept.
    GenerationStats st;
    CHECK(run(e, prompt, 12, &st) == base_out);
    CHECK(st.cached_prefix_tokens > 0);

    // Selected: the merged model's reply, the base context not continued.
    REQUIRE(e.set_adapter("a").is_ok());
    info = e.adapters();
    CHECK(info[0].active && info[0].scale == 1.0f && !info[1].active);
    CHECK(run(e, prompt, 12, &st) == merged_out);
    CHECK(st.cached_prefix_tokens == 0);
    // The same adapter again keeps its context; another scale does not.
    REQUIRE(e.set_adapter("a").is_ok());
    CHECK(run(e, prompt, 12, &st) == merged_out);
    CHECK(st.cached_prefix_tokens > 0);
    REQUIRE(e.set_adapter("a", 0.0f).is_ok());  // a zero scale: the base model's output, its own context
    CHECK(run(e, prompt, 12, &st) == base_out);
    CHECK(st.cached_prefix_tokens == 0);
    REQUIRE(e.set_adapter("b", 1.0f).is_ok());
    const std::vector<int32_t> b_out = run(e, prompt);
    CHECK(b_out != base_out && b_out != merged_out);
    // None again: the base model's reply.
    REQUIRE(e.set_adapter("").is_ok());
    CHECK(run(e, prompt) == base_out);
    // Unloading the selected adapter goes back to the base model.
    REQUIRE(e.set_adapter("b").is_ok());
    REQUIRE(e.unload_adapter("b").is_ok());
    CHECK(e.adapters().size() == 1);
    CHECK(run(e, prompt) == base_out);

    // Refused: unknown names, a taken name, a bad scale, a non-adapter, an
    // adapter that does not fit the model.
    CHECK(e.set_adapter("nope").code() == ErrorCode::InvalidArgument);
    CHECK(e.set_adapter("a", std::nanf("")).code() == ErrorCode::InvalidArgument);
    CHECK(e.unload_adapter("nope").code() == ErrorCode::InvalidArgument);
    CHECK(e.load_adapter("a", f.adapter_b).code() == ErrorCode::InvalidArgument);
    CHECK(e.load_adapter("", f.adapter_b).code() == ErrorCode::InvalidArgument);
    CHECK(e.load_adapter("m", f.base).code() == ErrorCode::InvalidModel);
    const std::string moe_path = temp_path("engine_moe.gguf");
    test::TinyModelSpec moe = f32_spec("qwen3moe");
    moe.n_expert = 4;
    moe.n_expert_used = 2;
    test::write_tiny_model(moe_path, moe);
    const std::string moe_adapter = temp_path("engine_moe_adapter.gguf");
    write_adapter(moe_adapter, random_pairs(moe_path, {"blk.0.ffn_up_exps.weight"}, 8, DType::F32, 3), 8.0f, DType::F32);
    CHECK(e.load_adapter("moe", moe_adapter).code() == ErrorCode::InvalidArgument);
    std::remove(moe_path.c_str());
    std::remove(moe_adapter.c_str());
    // Adapters off.
    EngineConfig off = adapter_engine_config(f.base);
    off.adapter_memory_mb = 0;
    auto without = Engine::create(off);
    REQUIRE(without.has_value());
    CHECK((*without)->load_adapter("a", f.adapter_a).code() == ErrorCode::Unsupported);
}

TEST_CASE("engine adapters: speculative decoding verifies with the adapter") {
    const EngineFiles f;
    auto merged = Engine::create(adapter_engine_config(f.merged_a));
    EngineConfig spec = adapter_engine_config(f.base);
    spec.lookup_drafts = true;
    spec.draft_tokens = 3;
    spec.adaptive_drafts = false;
    auto engine = Engine::create(spec);
    REQUIRE(merged.has_value() && engine.has_value());
    REQUIRE((*engine)->load_adapter("a", f.adapter_a).is_ok());
    REQUIRE((*engine)->set_adapter("a").is_ok());
    // A prompt that repeats itself, so drafts are proposed.
    std::vector<int32_t> prompt = prompt_of(24, 3);
    const std::vector<int32_t> again(prompt.begin() + 1, prompt.end());
    prompt.insert(prompt.end(), again.begin(), again.end());
    GenerationStats st;
    const std::vector<int32_t> expected = run(**merged, prompt, 24);
    const std::vector<int32_t> out = run(**engine, prompt, 24, &st);
    CHECK(out == expected);
    std::printf("  lookup drafts proposed %d, accepted %d\n", st.draft_tokens_proposed, st.draft_tokens_accepted);
    // A draft model (the base model itself) drafts without the adapter.
    EngineConfig with_draft = adapter_engine_config(f.base);
    with_draft.draft_model_path = f.base;
    with_draft.draft_tokens = 3;
    with_draft.adaptive_drafts = false;
    auto drafted = Engine::create(with_draft);
    REQUIRE(drafted.has_value());
    REQUIRE((*drafted)->load_adapter("a", f.adapter_a).is_ok());
    REQUIRE((*drafted)->set_adapter("a").is_ok());
    CHECK(run(**drafted, prompt, 24, &st) == expected);
    CHECK(st.draft_tokens_proposed > 0);
    std::printf("  draft model: drafts proposed %d, accepted %d\n", st.draft_tokens_proposed, st.draft_tokens_accepted);
}

TEST_CASE("engine adapters: the prefix cache and saved states keep contexts apart by adapter") {
    const EngineFiles f;
    const std::string dir = temp_path("engine_store");
    std::filesystem::remove_all(dir);
    EngineConfig c = adapter_engine_config(f.base);
    c.prefix_cache_mb = 0;  // storage only: every context set aside outlives the engine
    c.prefix_cache_dir = dir;
    c.prefix_cache_disk_mb = 64;
    const std::vector<int32_t> prompt = prompt_of(200, 2), other = prompt_of(200, 5);
    std::vector<int32_t> base_out, a_out;
    {
        auto engine = Engine::create(c);
        REQUIRE(engine.has_value());
        Engine& e = **engine;
        REQUIRE(e.load_adapter("a", f.adapter_a).is_ok());
        base_out = run(e, prompt);
        // Selecting A sets the base context aside; A's prompt finds nothing of
        // it (another seed): processed in full.
        REQUIRE(e.set_adapter("a").is_ok());
        const uint64_t restores = e.prefix_cache_stats().restores;
        GenerationStats st;
        a_out = run(e, prompt, 12, &st);
        CHECK(a_out != base_out);
        CHECK(st.cached_prefix_tokens == 0);
        CHECK(e.prefix_cache_stats().restores == restores);
        // Another conversation under A sets A's aside; returning restores it.
        run(e, other);
        CHECK(run(e, prompt, 12, &st) == a_out);
        CHECK(st.cached_prefix_tokens >= 128);
        CHECK(e.prefix_cache_stats().restores == restores + 1);
        // Back to the base model: its own context restored, its reply.
        REQUIRE(e.set_adapter("").is_ok());
        CHECK(run(e, prompt, 12, &st) == base_out);
        CHECK(st.cached_prefix_tokens >= 128);

        // A saved state belongs to its adapter.
        const std::string state = temp_path("engine_state.bin");
        REQUIRE(e.save_state(state).is_ok());  // the base context
        REQUIRE(e.set_adapter("a").is_ok());
        CHECK(e.load_state(state).status().code() == ErrorCode::InvalidArgument);
        REQUIRE(e.set_adapter("").is_ok());
        CHECK(e.load_state(state).has_value());
        REQUIRE(e.set_adapter("a").is_ok());
        run(e, prompt);
        REQUIRE(e.save_state(state).is_ok());  // A's context
        REQUIRE(e.set_adapter("").is_ok());
        CHECK(e.load_state(state).status().code() == ErrorCode::InvalidArgument);
        REQUIRE(e.set_adapter("a", 0.5f).is_ok());
        CHECK(e.load_state(state).status().code() == ErrorCode::InvalidArgument);
        REQUIRE(e.set_adapter("a").is_ok());
        auto restored = e.load_state(state);
        CHECK(restored.has_value());
        CHECK(run(e, prompt, 12, &st) == a_out);
        CHECK(st.cached_prefix_tokens >= 128);
        std::remove(state.c_str());
    }  // destroyed with A selected: A's context goes to storage under A
    {
        // A later engine finds each adapter's contexts on storage under it.
        auto engine = Engine::create(c);
        REQUIRE(engine.has_value());
        Engine& e = **engine;
        GenerationStats st;
        CHECK(run(e, prompt, 12, &st) == base_out);
        CHECK(st.cached_prefix_tokens >= 128);
        REQUIRE(e.load_adapter("a", f.adapter_a).is_ok());
        REQUIRE(e.set_adapter("a").is_ok());
        CHECK(run(e, prompt, 12, &st) == a_out);
        CHECK(st.cached_prefix_tokens >= 128);
    }
    std::filesystem::remove_all(dir);
}

TEST_CASE("engine adapters: the memory cap releases the least recently selected, mapped again when selected") {
    const EngineFiles f(56);  // A: 0.63 MiB, B: 0.09 MiB
    EngineConfig c = adapter_engine_config(f.base);
    c.adapter_memory_mb = 1;  // two copies of A do not fit
    auto engine = Engine::create(c);
    REQUIRE(engine.has_value());
    Engine& e = **engine;
    auto merged = Engine::create(adapter_engine_config(f.merged_a));
    REQUIRE(merged.has_value());
    const std::vector<int32_t> prompt = prompt_of(30, 4);
    REQUIRE(e.load_adapter("a", f.adapter_a).is_ok());
    REQUIRE(e.set_adapter("a").is_ok());
    REQUIRE(e.load_adapter("b", f.adapter_b).is_ok());  // A is selected: kept, over the cap with B
    std::vector<AdapterInfo> info = e.adapters();
    REQUIRE(info.size() == 2);
    CHECK(info[0].resident && info[1].resident);
    std::printf("  adapter sizes: %llu and %llu bytes\n", static_cast<unsigned long long>(info[0].bytes),
                static_cast<unsigned long long>(info[1].bytes));
    REQUIRE(e.set_adapter("b").is_ok());
    REQUIRE(e.set_adapter("").is_ok());
    REQUIRE(e.load_adapter("a2", f.adapter_a).is_ok());  // A is the least recently selected: released
    info = e.adapters();
    CHECK(!info[0].resident && info[1].resident && info[2].resident);
    REQUIRE(e.set_adapter("a").is_ok());  // mapped again (B released)
    info = e.adapters();
    CHECK(info[0].resident && info[0].active);
    CHECK(run(e, prompt) == run(**merged, prompt));
    // Too large for the cap on its own.
    const std::string big = temp_path("engine_big.gguf");
    write_adapter(big, random_pairs(f.base, per_block(2, {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate",
                                                          "ffn_up", "ffn_down"}),
                                    128, DType::F32, 9),
                  1.0f, DType::F32);
    CHECK(e.load_adapter("big", big).code() == ErrorCode::OutOfMemory);
    std::remove(big.c_str());
}

TEST_CASE("engine adapters through the C ABI") {
    const EngineFiles f;
    liyab_engine_config c;
    liyab_engine_config_default(&c);
    CHECK(c.adapter_memory_mb == 256);
    c.model_path = f.base.c_str();
    c.n_threads = 2;
    c.thermal_polling = 0;
    c.backend = LIYAB_BACKEND_CPU;
    liyab_engine* engine = nullptr;
    REQUIRE(liyab_engine_create(&c, &engine) == LIYAB_OK);
    CHECK(liyab_engine_adapter_count(engine) == 0);
    CHECK(liyab_engine_load_adapter(engine, "a", f.adapter_a.c_str()) == LIYAB_OK);
    CHECK(liyab_engine_load_adapter(engine, std::string(128, 'x').c_str(), f.adapter_b.c_str()) ==
          LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_set_adapter(engine, "a", 0.5f) == LIYAB_OK);
    CHECK(liyab_engine_adapter_count(engine) == 1);
    liyab_adapter_info info{};
    REQUIRE(liyab_engine_adapter_info(engine, 0, &info) == LIYAB_OK);
    CHECK(std::string(info.name) == "a" && info.rank == 32 && info.matrices == 10 && info.active == 1 &&
          info.resident == 1 && info.scale == 0.5f && info.bytes > 0);
    CHECK(liyab_engine_adapter_info(engine, 1, &info) == LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_set_adapter(engine, "missing", 1.0f) == LIYAB_ERR_INVALID_ARGUMENT);
    CHECK(liyab_engine_set_adapter(engine, nullptr, 1.0f) == LIYAB_OK);
    CHECK(liyab_engine_unload_adapter(engine, "a") == LIYAB_OK);
    CHECK(liyab_engine_adapter_count(engine) == 0);
    liyab_engine_destroy(engine);
}

TEST_MAIN()
