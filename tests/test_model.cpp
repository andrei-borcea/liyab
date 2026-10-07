#include "test_model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>

#include "core/quant.h"

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

void GgufWriter::add_tensor(const std::string& name, std::vector<int64_t> ne, DType type,
                            const std::vector<float>& values) {
    Tensor t{name, std::move(ne), type, {}};
    const int64_t row = t.ne[0];
    const auto rows = static_cast<int64_t>(values.size()) / row;
    const size_t row_bytes = dtype_row_bytes(type, row);
    t.data.resize(row_bytes * static_cast<size_t>(rows));
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
    w.add_u32(a + ".context_length", static_cast<uint32_t>(s.context_length));
    w.add_f32(a + ".attention.layer_norm_rms_epsilon", 1e-5f);
    w.add_f32(a + ".rope.freq_base", 10000.0f);
    w.add_string("tokenizer.ggml.model", "llama");
    w.add_string_array("tokenizer.ggml.tokens", vocab);
    w.add_f32_array("tokenizer.ggml.scores", scores);
    w.add_i32_array("tokenizer.ggml.token_type", types);
    w.add_u32("tokenizer.ggml.unknown_token_id", 0);
    w.add_u32("tokenizer.ggml.bos_token_id", 1);
    w.add_u32("tokenizer.ggml.eos_token_id", 2);
    w.add_bool("tokenizer.ggml.add_bos_token", true);

    std::mt19937_64 rng(s.seed);
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
    const int64_t hd = d / s.n_head;
    const int64_t kv = hd * s.n_head_kv;
    w.add_tensor("token_embd.weight", {d, n_vocab}, s.embd_type, random(d * n_vocab, 0.5f));
    w.add_tensor("output_norm.weight", {d}, DType::F32, norm_weights(d));
    if (!s.tied_output) w.add_tensor("output.weight", {d, n_vocab}, s.output_type, random(d * n_vocab, s.weight_scale));
    for (int32_t l = 0; l < s.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        w.add_tensor(p + "attn_norm.weight", {d}, DType::F32, norm_weights(d));
        w.add_tensor(p + "attn_q.weight", {d, d}, s.attn_type, random(d * d, s.weight_scale));
        w.add_tensor(p + "attn_k.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
        w.add_tensor(p + "attn_v.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
        w.add_tensor(p + "attn_output.weight", {d, d}, s.attn_type, random(d * d, s.weight_scale));
        if (a == "qwen2") {
            w.add_tensor(p + "attn_q.bias", {d}, DType::F32, random(d, 0.1f));
            w.add_tensor(p + "attn_k.bias", {kv}, DType::F32, random(kv, 0.1f));
            w.add_tensor(p + "attn_v.bias", {kv}, DType::F32, random(kv, 0.1f));
        }
        w.add_tensor(p + "ffn_norm.weight", {d}, DType::F32, norm_weights(d));
        w.add_tensor(p + "ffn_gate.weight", {d, s.n_ff}, s.ffn_type, random(d * s.n_ff, s.weight_scale));
        w.add_tensor(p + "ffn_up.weight", {d, s.n_ff}, s.ffn_type, random(d * s.n_ff, s.weight_scale));
        w.add_tensor(p + "ffn_down.weight", {s.n_ff, d}, s.ffn_type, random(s.n_ff * d, s.weight_scale));
    }
    w.write(path);
}

}  // namespace liyab::test
