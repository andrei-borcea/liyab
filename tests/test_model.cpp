#include "test_model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>

#include "core/quant.h"
#include "core/quant_tables.h"

namespace liyab::test {

namespace {

constexpr uint32_t kAlignment = 32;

template <typename T>
void put(std::vector<uint8_t>& out, T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof v);
}

void put_string(std::vector<uint8_t>& out, const std::string& s) {
    put<uint64_t>(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

// GGUF value type ids.
enum : uint32_t { kU32 = 4, kI32 = 5, kF32 = 6, kBool = 7, kString = 8, kArray = 9 };

std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    for (size_t i = 0; i < s.size();) {
        const auto c = static_cast<unsigned char>(s[i]);
        const size_t len = c < 0x80 ? 1 : (c >> 5) == 0x6 ? 2 : (c >> 4) == 0xE ? 3 : 4;
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

}  // namespace

void GgufWriter::add_u32(const std::string& key, uint32_t v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kU32);
    put(kv_, v);
    ++n_kv_;
}

void GgufWriter::add_f32(const std::string& key, float v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kF32);
    put(kv_, v);
    ++n_kv_;
}

void GgufWriter::add_bool(const std::string& key, bool v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kBool);
    put<uint8_t>(kv_, v ? 1 : 0);
    ++n_kv_;
}

void GgufWriter::add_string(const std::string& key, const std::string& v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kString);
    put_string(kv_, v);
    ++n_kv_;
}

void GgufWriter::add_string_array(const std::string& key, const std::vector<std::string>& v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kArray);
    put<uint32_t>(kv_, kString);
    put<uint64_t>(kv_, v.size());
    for (const auto& s : v) put_string(kv_, s);
    ++n_kv_;
}

void GgufWriter::add_f32_array(const std::string& key, const std::vector<float>& v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kArray);
    put<uint32_t>(kv_, kF32);
    put<uint64_t>(kv_, v.size());
    for (const float x : v) put(kv_, x);
    ++n_kv_;
}

void GgufWriter::add_i32_array(const std::string& key, const std::vector<int32_t>& v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kArray);
    put<uint32_t>(kv_, kI32);
    put<uint64_t>(kv_, v.size());
    for (const int32_t x : v) put(kv_, x);
    ++n_kv_;
}

void GgufWriter::add_bool_array(const std::string& key, const std::vector<bool>& v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kArray);
    put<uint32_t>(kv_, kBool);
    put<uint64_t>(kv_, v.size());
    for (const bool x : v) put<uint8_t>(kv_, x ? 1 : 0);
    ++n_kv_;
}

void GgufWriter::add_tensor(const std::string& name, std::vector<int64_t> ne, DType type,
                            const std::vector<float>& values) {
    Tensor t{name, std::move(ne), type, {}};
    const int64_t row = t.ne[0];
    const auto rows = static_cast<int64_t>(values.size()) / row;
    const size_t row_bytes = dtype_row_bytes(type, row);
    t.data.resize(row_bytes * static_cast<size_t>(rows));
    if (type == DType::MXFP4) {
        // Test-only nearest-grid encoding. The production engine reads this
        // format; it does not expose an MXFP4 quantizer.
        for (size_t b = 0; b < values.size() / 32; ++b) {
            const float* x = values.data() + b * 32;
            float amax = 0.0f;
            for (size_t j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
            const int exponent = amax > 0 ? std::clamp(static_cast<int>(std::ceil(std::log2(amax / 6.0f))), -127, 127) : 0;
            uint8_t* dst = t.data.data() + b * 17;
            dst[0] = static_cast<uint8_t>(exponent + 127);
            const float scale = std::ldexp(0.5f, exponent);
            auto index = [&](float value) {
                uint8_t best = 0;
                float error = INFINITY;
                for (uint8_t k = 0; k < 16; ++k) {
                    const float e = std::fabs(value - scale * quant::kMxfp4Values[k]);
                    if (e < error) { error = e; best = k; }
                }
                return best;
            };
            for (size_t j = 0; j < 16; ++j) dst[j + 1] = static_cast<uint8_t>(index(x[j]) | (index(x[j + 16]) << 4));
        }
        tensors_.push_back(std::move(t));
        return;
    }
    if (type == DType::BF16) {  // quant::quantize_row has no BF16: the upper half, rounded to nearest even
        for (size_t i = 0; i < values.size(); ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, &values[i], sizeof bits);
            const auto half = static_cast<uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
            std::memcpy(t.data.data() + 2 * i, &half, sizeof half);
        }
        tensors_.push_back(std::move(t));
        return;
    }
    for (int64_t r = 0; r < rows; ++r) {
        quant::quantize_row(type, values.data() + r * row, t.data.data() + static_cast<size_t>(r) * row_bytes, row);
    }
    tensors_.push_back(std::move(t));
}

void GgufWriter::write(const std::string& path) const {
    std::vector<uint8_t> out;
    put<uint32_t>(out, 0x46554747);  // "GGUF"
    put<uint32_t>(out, 3);
    put<uint64_t>(out, tensors_.size());
    put<uint64_t>(out, n_kv_);
    out.insert(out.end(), kv_.begin(), kv_.end());

    uint64_t offset = 0;
    std::vector<uint64_t> offsets;
    for (const Tensor& t : tensors_) {
        put_string(out, t.name);
        put<uint32_t>(out, static_cast<uint32_t>(t.ne.size()));
        for (const int64_t d : t.ne) put<uint64_t>(out, static_cast<uint64_t>(d));
        put<uint32_t>(out, static_cast<uint32_t>(t.type));
        put<uint64_t>(out, offset);
        offsets.push_back(offset);
        offset = (offset + t.data.size() + kAlignment - 1) / kAlignment * kAlignment;
    }
    out.resize((out.size() + kAlignment - 1) / kAlignment * kAlignment, 0);
    const size_t data_start = out.size();
    for (size_t i = 0; i < tensors_.size(); ++i) {
        out.resize(data_start + offsets[i], 0);
        out.insert(out.end(), tensors_[i].data.begin(), tensors_[i].data.end());
    }

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr || std::fwrite(out.data(), 1, out.size(), f) != out.size()) {
        std::fprintf(stderr, "cannot write %s\n", path.c_str());
        std::abort();
    }
    std::fclose(f);
}

std::vector<std::string> tiny_vocab() {
    std::vector<std::string> vocab = {"<unk>", "<s>", "</s>"};
    for (int b = 0; b < 256; ++b) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "<0x%02X>", b);
        vocab.emplace_back(buf);
    }
    std::set<std::string> seen;
    auto add = [&](const std::string& s) {
        if (seen.insert(s).second) vocab.push_back(s);
    };
    add("\xE2\x96\x81");
    for (char c = 'a'; c <= 'z'; ++c) add(std::string(1, c));
    for (const std::string word : {"\xE2\x96\x81hello", "\xE2\x96\x81world"}) {
        const auto chars = utf8_chars(word);
        for (size_t i = 0; i < chars.size(); ++i) {
            std::string s;
            for (size_t j = i; j < chars.size(); ++j) {
                s += chars[j];
                add(s);
            }
        }
    }
    return vocab;
}

int32_t tiny_token(const std::string& text) {
    static const std::vector<std::string> vocab = tiny_vocab();
    const auto it = std::find(vocab.begin(), vocab.end(), text);
    return it == vocab.end() ? -1 : static_cast<int32_t>(it - vocab.begin());
}

void write_tiny_model(const std::string& path, const TinyModelSpec& s) {
    const std::vector<std::string> vocab = tiny_vocab();
    const auto n_vocab = static_cast<int64_t>(vocab.size());
    std::vector<float> scores;
    std::vector<int32_t> types;
    for (size_t i = 0; i < vocab.size(); ++i) {
        if (i == 0) types.push_back(2);         // unknown
        else if (i < 3) types.push_back(3);     // control
        else if (i < 259) types.push_back(6);   // byte
        else types.push_back(1);                // normal
        scores.push_back(i < 259 ? 0.0f : static_cast<float>(utf8_chars(vocab[i]).size()));
    }

    GgufWriter w;
    const std::string& a = s.arch;
    w.add_string("general.architecture", a);
    w.add_string("general.name", "liyab-tiny-test");
    w.add_u32(a + ".block_count", static_cast<uint32_t>(s.n_layers));
    w.add_u32(a + ".embedding_length", static_cast<uint32_t>(s.n_embd));
    w.add_u32(a + ".feed_forward_length", static_cast<uint32_t>(s.n_ff));
    w.add_u32(a + ".attention.head_count", static_cast<uint32_t>(s.n_head));
    w.add_u32(a + ".attention.head_count_kv", static_cast<uint32_t>(s.n_head_kv));
    if (s.head_dim > 0) w.add_u32(a + ".attention.key_length", static_cast<uint32_t>(s.head_dim));
    w.add_u32(a + ".context_length", static_cast<uint32_t>(s.context_length));
    if (s.n_expert > 0) {
        w.add_u32(a + ".expert_count", static_cast<uint32_t>(s.n_expert));
        w.add_u32(a + ".expert_used_count", static_cast<uint32_t>(s.n_expert_used));
        if (!s.omit_expert_width) w.add_u32(a + ".expert_feed_forward_length", static_cast<uint32_t>(s.n_ff));
        if (s.expert_weights_norm >= 0) w.add_bool(a + ".expert_weights_norm", s.expert_weights_norm != 0);
        if (s.n_ff_shared > 0) w.add_u32(a + ".expert_shared_feed_forward_length", static_cast<uint32_t>(s.n_ff_shared));
        if (s.sigmoid_gating) w.add_u32(a + ".expert_gating_func", 2);
    }
    if (s.dense_lead > 0) w.add_u32(a + ".leading_dense_block_count", static_cast<uint32_t>(s.dense_lead));
    if (s.expert_swiglu_clamp > 0) w.add_f32_array(a + ".swiglu_clamp_exp", std::vector<float>(static_cast<size_t>(s.n_layers), s.expert_swiglu_clamp));
    if (s.shared_swiglu_clamp > 0) w.add_f32_array(a + ".swiglu_clamp_shexp", std::vector<float>(static_cast<size_t>(s.n_layers), s.shared_swiglu_clamp));
    if (s.mla_kv_rank > 0) {  // as the converter writes deepseek2 since it splits attn_k_b / attn_v_b
        w.add_u32(a + ".attention.kv_lora_rank", static_cast<uint32_t>(s.mla_kv_rank));
        if (s.mla_q_rank > 0) w.add_u32(a + ".attention.q_lora_rank", static_cast<uint32_t>(s.mla_q_rank));
        w.add_u32(a + ".attention.key_length", static_cast<uint32_t>(s.mla_kv_rank + s.rope_dim));
        w.add_u32(a + ".attention.value_length", static_cast<uint32_t>(s.mla_kv_rank));
        if (!s.mla_legacy) {
            w.add_u32(a + ".attention.key_length_mla", static_cast<uint32_t>(s.mla_nope + s.rope_dim));
            w.add_u32(a + ".attention.value_length_mla", static_cast<uint32_t>(s.mla_v));
        }
    }
    const int64_t ssm_hd = 16, ssm_hk = 2, ssm_hv = 4, ssm_kernel = 4;
    if (s.delta_net_interval > 0) {
        w.add_u32(a + ".ssm.conv_kernel", static_cast<uint32_t>(ssm_kernel));
        w.add_u32(a + ".ssm.state_size", static_cast<uint32_t>(ssm_hd));
        w.add_u32(a + ".ssm.group_count", static_cast<uint32_t>(ssm_hk));
        w.add_u32(a + ".ssm.time_step_rank", static_cast<uint32_t>(ssm_hv));
        w.add_u32(a + ".ssm.inner_size", static_cast<uint32_t>(ssm_hv * ssm_hd));
    }
    w.add_f32(a + ".attention.layer_norm_rms_epsilon", 1e-5f);
    w.add_f32(a + ".rope.freq_base", s.rope_base);
    if (s.sliding_window > 0) w.add_u32(a + ".attention.sliding_window", static_cast<uint32_t>(s.sliding_window));
    if (!s.swa_layers.empty()) {
        w.add_bool_array(a + ".attention.sliding_window_pattern", s.swa_layers);
    } else if (s.swa_pattern > 0) {
        w.add_u32(a + ".attention.sliding_window_pattern", static_cast<uint32_t>(s.swa_pattern));
    }
    if (s.rope_dim > 0) w.add_u32(a + ".rope.dimension_count", static_cast<uint32_t>(s.rope_dim));
    if (s.rope_scale > 0.0f) {
        w.add_string(a + ".rope.scaling.type", "linear");
        w.add_f32(a + ".rope.scaling.factor", s.rope_scale);
    }
    if (s.yarn_factor > 0.0f) {  // the keys the GGUF converter writes for a YaRN config
        w.add_string(a + ".rope.scaling.type", "yarn");
        w.add_f32(a + ".rope.scaling.factor", s.yarn_factor);
        if (s.yarn_ctx > 0) w.add_u32(a + ".rope.scaling.original_context_length", static_cast<uint32_t>(s.yarn_ctx));
        if (s.yarn_beta_fast > 0.0f) w.add_f32(a + ".rope.scaling.yarn_beta_fast", s.yarn_beta_fast);
        if (s.yarn_beta_slow > 0.0f) w.add_f32(a + ".rope.scaling.yarn_beta_slow", s.yarn_beta_slow);
        if (s.yarn_log_mul != 0.0f) w.add_f32(a + ".rope.scaling.yarn_log_multiplier", s.yarn_log_mul);
        if (s.yarn_truncate >= 0) w.add_bool(a + ".rope.scaling.yarn_truncate", s.yarn_truncate != 0);
    }
    if (s.attn_softcap > 0.0f) w.add_f32(a + ".attn_logit_softcapping", s.attn_softcap);
    if (s.final_softcap > 0.0f) w.add_f32(a + ".final_logit_softcapping", s.final_softcap);
    if (s.attn_scale > 0.0f) w.add_f32(a + ".attention.scale", s.attn_scale);
    if (s.longrope_ctx > 0) {  // as the Phi-3 converter writes it: no scaling type, the factors as tensors
        w.add_u32(a + ".rope.scaling.original_context_length", static_cast<uint32_t>(s.longrope_ctx));
        w.add_f32(a + ".rope.scaling.attn_factor", 1.19f);
    }
    if (s.bpe_pre.empty()) {
        w.add_string("tokenizer.ggml.model", "llama");
    } else {
        w.add_string("tokenizer.ggml.model", "gpt2");
        w.add_string("tokenizer.ggml.pre", s.bpe_pre);
        w.add_string_array("tokenizer.ggml.merges", {"h e"});
    }
    w.add_string_array("tokenizer.ggml.tokens", vocab);
    w.add_f32_array("tokenizer.ggml.scores", scores);
    w.add_i32_array("tokenizer.ggml.token_type", types);
    w.add_u32("tokenizer.ggml.unknown_token_id", 0);
    w.add_u32("tokenizer.ggml.bos_token_id", 1);
    w.add_u32("tokenizer.ggml.eos_token_id", 2);
    w.add_bool("tokenizer.ggml.add_bos_token", true);
    if (s.pooling_type >= 0) w.add_u32(a + ".pooling_type", static_cast<uint32_t>(s.pooling_type));
    if (s.add_eos >= 0) w.add_bool("tokenizer.ggml.add_eos_token", s.add_eos != 0);
    if (!s.cls_labels.empty()) w.add_string_array(a + ".classifier.output_labels", s.cls_labels);
    if (!s.rerank_template.empty()) w.add_string("tokenizer.chat_template.rerank", s.rerank_template);

    std::mt19937_64 rng(s.seed);
    std::mt19937_64 router_rng(s.seed ^ 0x5eed);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    auto random = [&](int64_t n, float scale) {
        std::vector<float> v(static_cast<size_t>(n));
        for (float& x : v) x = normal(rng) * scale;
        return v;
    };
    auto norm_weights = [&](int64_t n) {
        std::vector<float> v = random(n, 0.05f);
        for (float& x : v) x += 1.0f;
        return v;
    };

    const int64_t d = s.n_embd;
    const int64_t hd = s.head_dim > 0 ? s.head_dim : d / s.n_head;
    const int64_t q_dim = hd * s.n_head;
    const int64_t kv = hd * s.n_head_kv;
    const std::vector<float> embd = random(d * n_vocab, 0.5f);
    w.add_tensor("token_embd.weight", {d, n_vocab}, s.embd_type, embd);
    w.add_tensor("output_norm.weight", {d}, DType::F32, norm_weights(d));
    std::vector<float> output;  // the LM head's values (the embeddings when tied)
    if (!s.tied_output) {
        output = random(d * n_vocab, s.weight_scale);
        w.add_tensor("output.weight", {d, n_vocab}, s.output_type, output);
    }
    if (s.cls_rows > 0) {  // ranking head: copied rows of the LM head, or its own draws
        const std::vector<float>& head = s.tied_output ? embd : output;
        // A generator and a distribution of their own: a shared distribution
        // caches values, which would shift the draws of the weights after it.
        std::mt19937_64 cls_rng(s.seed ^ 0xc15);
        std::normal_distribution<float> cls_normal(0.0f, 1.0f);
        std::vector<float> rows;
        for (int32_t r = 0; r < s.cls_rows; ++r) {
            if (!s.cls_rows_from.empty()) {
                const auto at = head.begin() + static_cast<std::ptrdiff_t>(s.cls_rows_from[static_cast<size_t>(r)] * d);
                rows.insert(rows.end(), at, at + d);
            } else {
                for (int64_t j = 0; j < d; ++j) rows.push_back(cls_normal(cls_rng) * s.weight_scale * 4.0f);
            }
        }
        w.add_tensor("cls.output.weight", {d, s.cls_rows}, s.tied_output ? s.embd_type : s.output_type, rows);
        if (s.cls_bias) {
            std::vector<float> bias;
            for (int32_t r = 0; r < s.cls_rows; ++r) bias.push_back(cls_normal(cls_rng));
            w.add_tensor("cls.output.bias", {s.cls_rows}, DType::F32, bias);
        }
        if (s.cls_dense) {
            std::vector<float> dense;
            for (int64_t j = 0; j < d * d; ++j) dense.push_back(cls_normal(cls_rng) * s.weight_scale);
            w.add_tensor("cls.weight", {d, d}, DType::F32, dense);
        }
    }
    if (s.rope_freqs) {
        const int64_t pairs = (s.rope_dim > 0 ? s.rope_dim : hd) / 2;
        std::vector<float> factors(static_cast<size_t>(pairs));
        for (size_t i = 0; i < factors.size(); ++i) factors[i] = 1.0f + static_cast<float>(i % 4);  // 1..4
        w.add_tensor("rope_freqs.weight", {pairs}, DType::F32, factors);
    }
    if (s.longrope_ctx > 0) {
        const int64_t pairs = (s.rope_dim > 0 ? s.rope_dim : hd) / 2;
        std::vector<float> shorter(static_cast<size_t>(pairs)), longer(static_cast<size_t>(pairs));
        for (size_t i = 0; i < shorter.size(); ++i) {
            shorter[i] = 1.0f + 0.25f * static_cast<float>(i % 3);  // 1..1.5
            longer[i] = 1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(pairs);  // 1..3, the slow pairs most
        }
        w.add_tensor("rope_factors_short.weight", {pairs}, DType::F32, shorter);
        w.add_tensor("rope_factors_long.weight", {pairs}, DType::F32, longer);
    }
    for (int32_t l = 0; l < s.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        w.add_tensor(p + "attn_norm.weight", {d}, DType::F32, norm_weights(d));
        if (s.delta_net_interval > 0 && (l + 1) % s.delta_net_interval != 0) {
            const int64_t key_dim = ssm_hk * ssm_hd, value_dim = ssm_hv * ssm_hd;
            const int64_t conv_dim = 2 * key_dim + value_dim;
            w.add_tensor(p + "attn_qkv.weight", {d, conv_dim}, s.attn_type, random(d * conv_dim, s.weight_scale));
            w.add_tensor(p + "attn_gate.weight", {d, value_dim}, s.attn_type, random(d * value_dim, s.weight_scale));
            w.add_tensor(p + "ssm_alpha.weight", {d, ssm_hv}, DType::F32, random(d * ssm_hv, s.weight_scale));
            w.add_tensor(p + "ssm_beta.weight", {d, ssm_hv}, DType::F32, random(d * ssm_hv, s.weight_scale));
            w.add_tensor(p + "ssm_out.weight", {value_dim, d}, s.attn_type, random(value_dim * d, s.weight_scale));
            w.add_tensor(p + "ssm_conv1d.weight", {ssm_kernel, conv_dim}, DType::F32,
                         random(ssm_kernel * conv_dim, 0.3f));
            std::vector<float> a_neg = random(ssm_hv, 0.5f);
            for (float& x : a_neg) x = -std::exp(x);  // A = -exp(A_log)
            w.add_tensor(p + "ssm_a", {ssm_hv}, DType::F32, a_neg);
            w.add_tensor(p + "ssm_dt.bias", {ssm_hv}, DType::F32, random(ssm_hv, 0.5f));
            w.add_tensor(p + "ssm_norm.weight", {ssm_hd}, DType::F32, norm_weights(ssm_hd));
        } else if (s.mla_kv_rank > 0) {  // multi-head latent attention, per-head key and value expansions split
            const int64_t rank = s.mla_kv_rank, rd = s.rope_dim, heads = s.n_head;
            const int64_t q_rows = heads * (s.mla_nope + rd);  // per head [q_nope | q_pe]
            if (s.mla_q_rank > 0) {
                const int64_t qr = s.mla_q_rank;
                w.add_tensor(p + "attn_q_a.weight", {d, qr}, s.attn_type, random(d * qr, s.weight_scale));
                w.add_tensor(p + "attn_q_a_norm.weight", {qr}, DType::F32, norm_weights(qr));
                w.add_tensor(p + "attn_q_b.weight", {qr, q_rows}, s.attn_type, random(qr * q_rows, s.weight_scale));
            } else {
                w.add_tensor(p + "attn_q.weight", {d, q_rows}, s.attn_type, random(d * q_rows, s.weight_scale));
            }
            w.add_tensor(p + "attn_kv_a_mqa.weight", {d, rank + rd}, s.attn_type,
                         random(d * (rank + rd), s.weight_scale));
            std::vector<float> kv_norm = norm_weights(rank);
            if (s.k_norm_outlier > 0.0f) {  // large latent channels: keys and, with them, values
                kv_norm[static_cast<size_t>(rank / 2 - 1)] = s.k_norm_outlier;
                kv_norm[static_cast<size_t>(rank - 3)] = s.k_norm_outlier;
            }
            w.add_tensor(p + "attn_kv_a_norm.weight", {rank}, DType::F32, kv_norm);
            w.add_tensor(p + "attn_k_b.weight", {s.mla_nope, rank, heads}, s.attn_type,
                         random(s.mla_nope * rank * heads, s.weight_scale));
            w.add_tensor(p + "attn_v_b.weight", {rank, s.mla_v, heads}, s.attn_type,
                         random(rank * s.mla_v * heads, s.weight_scale));
            w.add_tensor(p + "attn_output.weight", {heads * s.mla_v, d}, s.attn_type,
                         random(heads * s.mla_v * d, s.weight_scale));
        } else {
            const int64_t q_rows = s.attn_gate ? 2 * q_dim : q_dim;  // [q_h | gate_h] per head when gated
            if (s.fused_qkv) {  // the same draws as separate matrices, stacked [q | k | v]
                std::vector<float> qkv = random(d * q_rows, s.weight_scale);
                for (int part = 0; part < 2; ++part) {  // k, then v
                    const std::vector<float> rows = random(d * kv, s.weight_scale);
                    qkv.insert(qkv.end(), rows.begin(), rows.end());
                }
                w.add_tensor(p + "attn_qkv.weight", {d, q_rows + 2 * kv}, s.attn_type, qkv);
            } else {
                auto query = random(d * q_rows, s.weight_scale);
                if (s.head_attention_gate) {
                    for (int32_t h = 0; h < s.n_head; ++h) {
                        const auto at = static_cast<size_t>((h * 2 * hd + hd) * d);
                        for (int64_t j = 1; j < hd; ++j) std::copy_n(query.data() + at, d, query.data() + at + static_cast<size_t>(j * d));
                    }
                }
                if (s.separate_attention_gate) {
                    std::vector<float> q, gate;
                    for (int32_t h = 0; h < s.n_head; ++h) {
                        const auto at = static_cast<std::ptrdiff_t>(h * 2 * hd * d);
                        const auto count = static_cast<std::ptrdiff_t>(hd * d);
                        q.insert(q.end(), query.begin() + at, query.begin() + at + count);
                        if (s.head_attention_gate) {
                            gate.insert(gate.end(), query.begin() + at + count, query.begin() + at + count + d);
                        } else {
                            gate.insert(gate.end(), query.begin() + at + count, query.begin() + at + 2 * count);
                        }
                    }
                    w.add_tensor(p + "attn_q.weight", {d, q_dim}, s.attn_type, q);
                    w.add_tensor(p + "attn_gate.weight", {d, s.head_attention_gate ? s.n_head : q_dim}, s.attn_type, gate);
                } else {
                    w.add_tensor(p + "attn_q.weight", {d, q_rows}, s.attn_type, query);
                }
                w.add_tensor(p + "attn_k.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
                w.add_tensor(p + "attn_v.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
            }
            w.add_tensor(p + "attn_output.weight", {q_dim, d}, s.attn_type, random(d * q_dim, s.weight_scale));
            if (s.qk_norm) {
                w.add_tensor(p + "attn_q_norm.weight", {hd}, DType::F32, norm_weights(hd));
                std::vector<float> k_norm = norm_weights(hd);
                if (s.k_norm_outlier > 0.0f) {
                    k_norm[static_cast<size_t>(hd / 2 - 1)] = s.k_norm_outlier;
                    k_norm[static_cast<size_t>(hd - 3)] = s.k_norm_outlier;
                }
                w.add_tensor(p + "attn_k_norm.weight", {hd}, DType::F32, k_norm);
            }
        }
        if (a == "qwen2") {
            w.add_tensor(p + "attn_q.bias", {q_dim}, DType::F32, random(q_dim, 0.1f));
            std::vector<float> k_bias = random(kv, 0.1f);
            // A few large, constant key channels, as Qwen2.5's K bias has
            // (the slowest-turning RoPE pair of each KV head).
            for (int64_t h = 0; h < kv / hd && s.key_outlier != 0.0f; ++h) {
                k_bias[static_cast<size_t>(h * hd + hd / 2 - 1)] = s.key_outlier;
                k_bias[static_cast<size_t>(h * hd + hd - 1)] = -s.key_outlier;
            }
            w.add_tensor(p + "attn_k.bias", {kv}, DType::F32, k_bias);
            w.add_tensor(p + "attn_v.bias", {kv}, DType::F32, random(kv, 0.1f));
        }
        // Qwen3.5 names the pre-FFN norm after the attention it follows (as its GGUFs do).
        const bool post_attention = a == "qwen35" || a == "qwen35moe" || a == "gpt-oss" || a == "glm4moe";
        w.add_tensor(p + (post_attention ? "post_attention_norm.weight" : "ffn_norm.weight"), {d}, DType::F32,
                     norm_weights(d));
        if (s.post_norms) {  // Gemma: post_attention_norm is the norm after the attention, not before the FFN
            w.add_tensor(p + "post_attention_norm.weight", {d}, DType::F32, norm_weights(d));
            w.add_tensor(p + "post_ffw_norm.weight", {d}, DType::F32, norm_weights(d));
        }
        const std::vector<float> gate = random(d * s.n_ff, s.weight_scale);
        const std::vector<float> up = random(d * s.n_ff, s.weight_scale);
        const std::vector<float> down = random(s.n_ff * d, s.weight_scale);
        if (s.n_expert == 0 && s.fused_gate_up) {
            std::vector<float> gate_up = gate;
            gate_up.insert(gate_up.end(), up.begin(), up.end());
            w.add_tensor(p + "ffn_up.weight", {d, 2 * s.n_ff}, s.ffn_type, gate_up);
            w.add_tensor(p + "ffn_down.weight", {s.n_ff, d}, s.ffn_type, down);
            continue;
        }
        if (s.n_expert == 0 || l < s.dense_lead) {
            w.add_tensor(p + "ffn_gate.weight", {d, s.n_ff}, s.ffn_type, gate);
            w.add_tensor(p + "ffn_up.weight", {d, s.n_ff}, s.ffn_type, up);
            w.add_tensor(p + "ffn_down.weight", {s.n_ff, d}, s.ffn_type, down);
            continue;
        }
        auto stacked = [&](const std::vector<float>& one) {
            std::vector<float> all;
            for (int32_t e = 0; e < s.n_expert; ++e) {
                if (s.identical_experts || e == 0) {
                    all.insert(all.end(), one.begin(), one.end());
                    continue;
                }
                for (size_t i = 0; i < one.size(); ++i) all.push_back(normal(router_rng) * s.weight_scale);
            }
            return all;
        };
        const std::vector<float> gate_exps = stacked(gate);
        const std::vector<float> up_exps = stacked(up);
        if (s.fused_gate_up) {  // each expert's [gate | up] rows: the same draws as the split tensors
            std::vector<float> gate_up;
            const auto per_expert = static_cast<std::ptrdiff_t>(d * s.n_ff);
            for (std::ptrdiff_t e = 0; e < s.n_expert; ++e) {
                gate_up.insert(gate_up.end(), gate_exps.begin() + e * per_expert, gate_exps.begin() + (e + 1) * per_expert);
                gate_up.insert(gate_up.end(), up_exps.begin() + e * per_expert, up_exps.begin() + (e + 1) * per_expert);
            }
            w.add_tensor(p + "ffn_gate_up_exps.weight", {d, 2 * s.n_ff, s.n_expert}, s.ffn_type, gate_up);
        } else {
            w.add_tensor(p + "ffn_gate_exps.weight", {d, s.n_ff, s.n_expert}, s.ffn_type, gate_exps);
            w.add_tensor(p + "ffn_up_exps.weight", {d, s.n_ff, s.n_expert}, s.ffn_type, up_exps);
        }
        w.add_tensor(p + "ffn_down_exps.weight", {s.n_ff, d, s.n_expert}, s.ffn_type, stacked(down));
        // The router uses its own generator so every other weight matches the dense twin.
        std::vector<float> router(static_cast<size_t>(d * s.n_expert));
        for (float& x : router) x = normal(router_rng) * 0.5f;
        w.add_tensor(p + "ffn_gate_inp.weight", {d, s.n_expert}, DType::F32, router);
        if (s.expert_bias) w.add_tensor(p + (s.bare_expert_bias ? "exp_probs_b" : "exp_probs_b.bias"), {s.n_expert}, DType::F32, random(s.n_expert, 0.2f));
        if (s.n_ff_shared > 0) {
            const int64_t sh = s.n_ff_shared;
            w.add_tensor(p + "ffn_gate_shexp.weight", {d, sh}, s.ffn_type, random(d * sh, s.weight_scale));
            w.add_tensor(p + "ffn_up_shexp.weight", {d, sh}, s.ffn_type, random(d * sh, s.weight_scale));
            w.add_tensor(p + "ffn_down_shexp.weight", {sh, d}, s.ffn_type, random(sh * d, s.weight_scale));
            if (s.shared_gate) w.add_tensor(p + "ffn_gate_inp_shexp.weight", {d}, DType::F32, random(d, s.weight_scale));
        }
    }
    // Keep optional tensors independent of the original random stream.
    std::mt19937_64 bias_rng(s.seed ^ 0xb1a5);
    auto bias = [&](int64_t n) {
        std::vector<float> v(static_cast<size_t>(n));
        for (float& x : v) x = normal(bias_rng) * 0.2f;
        return v;
    };
    for (int32_t l = 0; l < s.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        if (s.attention_output_bias) w.add_tensor(p + "attn_output.bias", {d}, DType::F32, bias(d));
        if (s.attention_sinks) w.add_tensor(p + "attn_sinks.weight", {s.n_head}, DType::F32, bias(s.n_head));
        if (s.n_expert == 0 || l < s.dense_lead) continue;
        if (s.router_bias) w.add_tensor(p + "ffn_gate_inp.bias", {s.n_expert}, DType::F32, bias(s.n_expert));
        if (s.expert_projection_bias) {
            w.add_tensor(p + "ffn_gate_exps.bias", {s.n_ff, s.n_expert}, DType::F32, bias(s.n_ff * s.n_expert));
            w.add_tensor(p + "ffn_up_exps.bias", {s.n_ff, s.n_expert}, DType::F32, bias(s.n_ff * s.n_expert));
            w.add_tensor(p + "ffn_down_exps.bias", {d, s.n_expert}, DType::F32, bias(d * s.n_expert));
        }
    }
    w.write(path);
}

}  // namespace liyab::test
