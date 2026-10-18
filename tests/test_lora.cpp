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
    // Off again: the base model's bits; on again: the same bits as before.
    REQUIRE(base->model->set_adapter(nullptr).is_ok());
    CHECK(logits_of(*base) == plain);
    REQUIRE(base->model->set_adapter(adapter->get(), kUserScale).is_ok());
    CHECK(logits_of(*base) == adapted);
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

TEST_MAIN()
