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
#include "test_util.h"

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

void GgufWriter::add_u8_array(const std::string& key, const std::vector<uint8_t>& v) {
    put_string(kv_, key);
    put<uint32_t>(kv_, kArray);
    put<uint32_t>(kv_, 0);  // GGUF U8
    put<uint64_t>(kv_, v.size());
    kv_.insert(kv_.end(), v.begin(), v.end());
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

void GgufWriter::add_i32_tensor(const std::string& name, std::vector<int64_t> ne, const std::vector<int32_t>& values) {
    Tensor tensor{name, std::move(ne), DType::I32, {}};
    for (int32_t value : values) put(tensor.data, value);
    tensors_.push_back(std::move(tensor));
}

void GgufWriter::add_tensor(const std::string& name, std::vector<int64_t> ne, DType type,
                            const std::vector<float>& values, std::span<const uint8_t> block_pattern) {
    Tensor t{name, std::move(ne), type, {}};
    const int64_t row = t.ne[0];
    const auto rows = static_cast<int64_t>(values.size()) / row;
    const size_t row_bytes = dtype_row_bytes(type, row);
    t.data.resize(row_bytes * static_cast<size_t>(rows));
    if (!block_pattern.empty()) {
        tensors_.push_back(std::move(t));
        if (!set_tensor_block_pattern(name, block_pattern)) {
            std::fprintf(stderr, "invalid encoded block pattern for %s (%s)\n", name.c_str(),
                         std::string(dtype_traits(type).name).c_str());
            std::abort();
        }
        return;
    }
    if (!block_pattern_.empty() && type == block_pattern_type_ && t.ne.size() > 1) {
        const auto traits = dtype_traits(type);
        if (row % traits.block_size != 0) std::abort();
        const size_t blocks = block_pattern_.size() / static_cast<size_t>(traits.block_bytes);
        const size_t per_row = static_cast<size_t>(row / traits.block_size);
        // A short cyclic pattern repeats at head/group boundaries and could
        // hide an incorrect grouped-output slice. Vary blocks independently
        // across rows and projections, deterministically from the tensor name.
        std::seed_seq seed(name.begin(), name.end());
        std::mt19937_64 pattern_rng(seed);
        for (size_t r = 0; r < static_cast<size_t>(rows); ++r) {
            for (size_t b = 0; b < per_row; ++b) {
                std::memcpy(t.data.data() + (r * per_row + b) * static_cast<size_t>(traits.block_bytes),
                            block_pattern_.data() + pattern_rng() % blocks * static_cast<size_t>(traits.block_bytes),
                            static_cast<size_t>(traits.block_bytes));
            }
        }
        tensors_.push_back(std::move(t));
        return;
    }
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
    // Keep this list aligned with the runtime encoders, plus BF16 below.
    // Unsupported types must never silently produce all-zero test weights.
    switch (type) {
        case DType::F32: case DType::F16: case DType::BF16:
        case DType::Q8_0: case DType::Q4_0: case DType::Q4_1:
        case DType::Q4_K: case DType::Q5_K:
            break;
        default:
            std::fprintf(stderr, "no test encoder for %s (%s); supply explicit encoded blocks\n", name.c_str(),
                         std::string(dtype_traits(type).name).c_str());
            std::abort();
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

void GgufWriter::set_tensor_block_pattern(DType type, const std::vector<uint8_t>& blocks) {
    const auto traits = dtype_traits(type);
    if (blocks.empty() || blocks.size() % static_cast<size_t>(traits.block_bytes) != 0) std::abort();
    block_pattern_type_ = type;
    block_pattern_ = blocks;
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

bool GgufWriter::set_tensor_block_pattern(const std::string& name, std::span<const uint8_t> pattern) {
    const auto it = std::find_if(tensors_.begin(), tensors_.end(), [&](const Tensor& t) { return t.name == name; });
    if (it == tensors_.end()) return false;
    const auto bytes = static_cast<size_t>(dtype_traits(it->type).block_bytes);
    if (pattern.empty() || pattern.size() % bytes != 0 || it->data.size() % bytes != 0) return false;
    for (size_t at = 0; at < it->data.size(); at += bytes) {
        std::memcpy(it->data.data() + at, pattern.data() + at % pattern.size(), bytes);
    }
    return true;
}

bool load_quant_vectors(std::vector<QuantVector>& out, const std::string& filename) {
    std::string dir = LIYAB_TEST_DATA_DIR;
    if (const char* env = std::getenv("LIYAB_TEST_DATA")) dir = env;
    const std::string path = dir + "/" + filename;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        std::printf("  quant vectors unavailable: %s (set LIYAB_TEST_DATA)\n", path.c_str());
        return false;
    }
    std::vector<uint8_t> file;
    uint8_t chunk[4096];
    for (size_t n; (n = std::fread(chunk, 1, sizeof chunk, f)) > 0;) file.insert(file.end(), chunk, chunk + n);
    const bool read_ok = std::ferror(f) == 0;
    std::fclose(f);
    size_t pos = 0;
    auto read = [&](void* dst, size_t size) {
        if (size > file.size() - pos) return false;
        std::memcpy(dst, file.data() + pos, size);
        pos += size;
        return true;
    };
    char magic[4];
    uint32_t version = 0, count = 0;
    std::vector<QuantVector> vectors;
    bool valid = read_ok && read(magic, 4) && std::memcmp(magic, "LYQV", 4) == 0 && read(&version, 4) &&
                 version == 1 && read(&count, 4) && count <= (file.size() - pos) / 8;
    for (uint32_t i = 0; valid && i < count; ++i) {
        uint32_t type = 0, blocks = 0;
        QuantVector v;
        valid = read(&type, 4) && read(&blocks, 4) && dtype_from_ggml(type, v.type) && blocks > 0;
        if (!valid) break;
        const auto traits = dtype_traits(v.type);
        const uint64_t bytes = uint64_t{blocks} * traits.block_bytes;
        const uint64_t values = uint64_t{blocks} * traits.block_size;
        valid = bytes + values * sizeof(float) <= file.size() - pos;
        if (!valid) break;
        v.n = static_cast<int64_t>(values);
        v.blocks.resize(static_cast<size_t>(bytes));
        v.expected.resize(static_cast<size_t>(values));
        valid = read(v.blocks.data(), v.blocks.size()) && read(v.expected.data(), v.expected.size() * sizeof(float)) &&
                std::all_of(v.expected.begin(), v.expected.end(), [](float x) { return std::isfinite(x); });
        vectors.push_back(std::move(v));
    }
    if (!valid || pos != file.size()) {
        std::printf("  malformed quant vectors: %s\n", path.c_str());
        return false;
    }
    out = std::move(vectors);
    return true;
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

TinyModelSpec nas_spec() {
    TinyModelSpec s;
    s.arch = "deci";
    s.n_layers = 5;
    s.block_heads = {4, 4, 2, 0, 4};
    s.block_kv_heads = {2, 1, 1, 0, 0};
    s.block_ff = {256, 128, 192, 0, 256};
    return s;
}

std::vector<TinyModelSpec> recurrent_model_specs() {
    std::vector<TinyModelSpec> out;
    TinyModelSpec s;
    s.arch = "lfm2";
    s.n_layers = 3;
    s.short_conv_interval = 3;
    s.shortconv_bias = true;
    s.qk_norm = true;
    out.push_back(s);
    s.arch = "lfm2moe";
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.identical_experts = false;
    s.expert_bias = true;
    s.sigmoid_gating = true;
    s.dense_lead = 1;
    out.push_back(s);
    s = {};
    s.arch = "qwen3next";
    s.n_layers = 4;
    s.delta_net_interval = 2;
    s.fused_ssm_ba = true;
    s.qk_norm = true;
    s.attn_gate = true;
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.identical_experts = false;
    s.n_ff_shared = 96;
    s.shared_gate = true;
    out.push_back(s);
    s = {};
    s.arch = "kimi-linear";
    s.n_layers = 4;
    s.kda_interval = 4;
    s.n_head_kv = 1;
    s.rope_dim = 32;
    s.mla_kv_rank = 64;
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.identical_experts = false;
    s.sigmoid_gating = true;
    s.expert_bias = true;
    s.dense_lead = 1;
    s.n_ff_shared = 64;
    out.push_back(s);
    return out;
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
    if (!s.attention_block_pattern.empty()) w.set_tensor_block_pattern(s.attn_type, s.attention_block_pattern);
    const std::string& a = s.arch;
    const int32_t mamba = s.mamba > 0 ? s.mamba : a == "mamba" ? 1 : a == "mamba2" ? 2 : 0;
    const int64_t mamba_rank = std::max(32, dtype_traits(s.attn_type).block_size);
    w.add_string("general.architecture", a);
    if (s.tokenizer_padding > 0) w.add_string("liyab.test.padding", std::string(s.tokenizer_padding, 'x'));
    w.add_string("general.name", "liyab-tiny-test");
    w.add_u32(a + ".block_count", static_cast<uint32_t>(s.n_layers));
    w.add_u32(a + ".embedding_length", static_cast<uint32_t>(s.n_embd));
    if (s.block_ff.empty()) w.add_u32(a + ".feed_forward_length", static_cast<uint32_t>(s.n_ff));
    else w.add_i32_array(a + ".feed_forward_length", s.block_ff);
    if (s.block_heads.empty()) w.add_u32(a + ".attention.head_count", static_cast<uint32_t>(s.n_head));
    else w.add_i32_array(a + ".attention.head_count", s.block_heads);
    if (s.short_conv_interval > 0 || s.kda_interval > 0) {
        const int32_t interval = s.short_conv_interval > 0 ? s.short_conv_interval : s.kda_interval;
        std::vector<int32_t> heads(static_cast<size_t>(s.n_layers));
        for (int32_t l = 0; l < s.n_layers; ++l) heads[static_cast<size_t>(l)] = (l + 1) % interval == 0 ? s.n_head_kv : 0;
        w.add_i32_array(a + ".attention.head_count_kv", heads);
    } else if (s.block_kv_heads.empty()) w.add_u32(a + ".attention.head_count_kv", static_cast<uint32_t>(s.mla_legacy ? s.n_head : s.n_head_kv));
    else w.add_i32_array(a + ".attention.head_count_kv", s.block_kv_heads);
    if (!s.block_heads.empty() && s.mla_kv_rank == 0) {
        w.add_u32(a + ".attention.key_length", static_cast<uint32_t>(s.n_embd / s.n_head));
    }
    if (s.head_dim > 0 && s.block_heads.empty()) w.add_u32(a + ".attention.key_length", static_cast<uint32_t>(s.head_dim));
    w.add_u32(a + ".context_length", static_cast<uint32_t>(s.context_length));
    if (s.n_expert > 0) {
        if (s.hash_layers > 0) w.add_u32(a + ".hash_layer_count", static_cast<uint32_t>(s.hash_layers));
        if (s.sqrtsoftplus_gating) w.add_u32(a + ".expert_gating_func", 4);
        w.add_u32(a + ".expert_count", static_cast<uint32_t>(s.n_expert));
        w.add_u32(a + ".expert_used_count", static_cast<uint32_t>(s.n_expert_used));
        if (!s.omit_expert_width) w.add_u32(a + ".expert_feed_forward_length", static_cast<uint32_t>(s.n_ff));
        if (s.expert_weights_norm >= 0) w.add_bool(a + ".expert_weights_norm", s.expert_weights_norm != 0);
        if (s.n_ff_shared > 0) w.add_u32(a + ".expert_shared_feed_forward_length", static_cast<uint32_t>(s.n_ff_shared));
        if (s.sigmoid_gating) w.add_u32(a + ".expert_gating_func", 2);
        if (s.n_expert_groups > 0) {
            w.add_u32(a + ".expert_group_count", static_cast<uint32_t>(s.n_expert_groups));
            w.add_u32(a + ".expert_group_used_count", static_cast<uint32_t>(s.n_expert_groups_used));
        }
    }
    if (s.dense_lead > 0) w.add_u32(a + ".leading_dense_block_count", static_cast<uint32_t>(s.dense_lead));
    if (s.expert_swiglu_clamp > 0) w.add_f32_array(a + ".swiglu_clamp_exp", std::vector<float>(static_cast<size_t>(s.n_layers), s.expert_swiglu_clamp));
    if (s.shared_swiglu_clamp > 0) w.add_f32_array(a + ".swiglu_clamp_shexp", std::vector<float>(static_cast<size_t>(s.n_layers), s.shared_swiglu_clamp));
    if (s.mla_kv_rank > 0) {  // as the converter writes deepseek2 since it splits attn_k_b / attn_v_b
        w.add_u32(a + ".attention.kv_lora_rank", static_cast<uint32_t>(s.mla_kv_rank));
        if (s.mla_q_rank > 0) w.add_u32(a + ".attention.q_lora_rank", static_cast<uint32_t>(s.mla_q_rank));
        w.add_u32(a + ".attention.key_length", static_cast<uint32_t>((s.mla_legacy ? s.mla_nope : s.mla_kv_rank) + s.rope_dim));
        w.add_u32(a + ".attention.value_length", static_cast<uint32_t>(s.mla_legacy ? s.mla_v : s.mla_kv_rank));
        if (!s.mla_legacy) {
            w.add_u32(a + ".attention.key_length_mla", static_cast<uint32_t>(s.mla_nope + s.rope_dim));
            w.add_u32(a + ".attention.value_length_mla", static_cast<uint32_t>(s.mla_v));
        }
    }
    if (a == "deepseek4") {
        w.add_u32(a + ".attention.key_length", static_cast<uint32_t>(s.shared_head_dim > 0 ? s.shared_head_dim : s.n_embd / s.n_head));
        w.add_u32(a + ".attention.q_lora_rank", static_cast<uint32_t>(s.mla_q_rank));
        w.add_u32(a + ".attention.output_group_count", static_cast<uint32_t>(s.shared_output_groups));
        w.add_u32(a + ".attention.output_lora_rank", static_cast<uint32_t>(s.shared_output_rank));
        w.add_f32(a + ".attention.compress_rope_freq_base", s.compress_rope_base);
        w.add_i32_array(a + ".attention.compress_ratios", s.compression_strides);
    }
    if (s.hc_count > 0) {
        w.add_u32(a + ".hyper_connection.count", static_cast<uint32_t>(s.hc_count));
        w.add_u32(a + ".hyper_connection.sinkhorn_iterations", 20);
        w.add_f32(a + ".hyper_connection.epsilon", 1e-6f);
    }
    const int64_t ssm_hd = s.ssm_head_dim, ssm_hk = s.ssm_key_heads, ssm_hv = s.ssm_value_heads, ssm_kernel = 4;
    if (mamba > 0) {
        w.add_u32(a + ".ssm.conv_kernel", static_cast<uint32_t>(s.mamba_kernel));
        w.add_u32(a + ".ssm.inner_size", static_cast<uint32_t>(2 * s.n_embd));
        w.add_u32(a + ".ssm.state_size", 16);
        w.add_u32(a + ".ssm.time_step_rank", mamba == 1 ? static_cast<uint32_t>(mamba_rank) : 4);
        if (mamba == 2) w.add_u32(a + ".ssm.group_count", static_cast<uint32_t>(s.mamba_groups));
        if (s.mamba_norms) w.add_bool(a + ".ssm.dt_b_c_rms", true);
    }
    if (s.delta_net_interval > 0) {
        w.add_u32(a + ".ssm.conv_kernel", static_cast<uint32_t>(ssm_kernel));
        w.add_u32(a + ".ssm.state_size", static_cast<uint32_t>(ssm_hd));
        w.add_u32(a + ".ssm.group_count", static_cast<uint32_t>(ssm_hk));
        w.add_u32(a + ".ssm.time_step_rank", static_cast<uint32_t>(ssm_hv));
        w.add_u32(a + ".ssm.inner_size", static_cast<uint32_t>(ssm_hv * ssm_hd));
    }
    if (s.short_conv_interval > 0) w.add_u32(a + ".shortconv.l_cache", static_cast<uint32_t>(s.shortconv_taps));
    if (s.kda_interval > 0) {
        w.add_u32(a + ".ssm.conv_kernel", static_cast<uint32_t>(ssm_kernel));
        w.add_u32(a + ".kda.head_dim", static_cast<uint32_t>(s.kda_head_dim));
    }
    w.add_f32(a + ".attention.layer_norm_rms_epsilon", s.rms_eps);
    if (s.norm_bias || s.no_norm_weights) w.add_f32(a + ".attention.layer_norm_epsilon", s.rms_eps);
    if (s.clamp_kqv > 0.0f) w.add_f32(a + ".attention.clamp_kqv", s.clamp_kqv);
    if (s.parallel_residual >= 0) w.add_bool(a + ".use_parallel_residual", s.parallel_residual != 0);
    if (a == "apertus") {
        for (const auto& [key, value] : std::vector<std::pair<const char*, float>>{
                 {"alpha_p", -0.4f}, {"alpha_n", 0.6f}, {"beta", 0.5f}, {"eps", -1e-6f}}) {
            w.add_f32_array(std::string("xielu.") + key, std::vector<float>(static_cast<size_t>(s.n_layers), value));
        }
    }
    w.add_f32(a + ".rope.freq_base", s.rope_base);
    if (!s.rope_layers.empty()) w.add_bool_array(a + ".attention.rope_pattern", s.rope_layers);
    if (s.embedding_scale != 0.0f) w.add_f32(a + ".embedding_scale", s.embedding_scale);
    if (s.residual_scale != 0.0f) w.add_f32(a + ".residual_scale", s.residual_scale);
    if (s.logit_scale != 0.0f) w.add_f32(a + ".logit_scale", s.logit_scale);
    if (s.attn_temp_scale != 0.0f) w.add_f32(a + ".attention.temperature_scale", s.attn_temp_scale);
    if (s.attn_temp_length != 0) w.add_u32(a + ".attention.temperature_length", static_cast<uint32_t>(s.attn_temp_length));
    if (s.swa_head_dim > 0) {
        w.add_u32(a + ".attention.key_length_swa", static_cast<uint32_t>(s.swa_head_dim));
        w.add_u32(a + ".attention.value_length_swa", static_cast<uint32_t>(s.swa_head_dim));
    }
    if (s.swa_rope_dim > 0) w.add_u32(a + ".rope.dimension_count_swa", static_cast<uint32_t>(s.swa_rope_dim));
    if (s.sliding_window > 0) w.add_u32(a + ".attention.sliding_window", static_cast<uint32_t>(s.sliding_window));
    if (!s.swa_layers.empty()) {
        w.add_bool_array(a + ".attention.sliding_window_pattern", s.swa_layers);
    } else if (s.swa_pattern > 0) {
        w.add_u32(a + ".attention.sliding_window_pattern", static_cast<uint32_t>(s.swa_pattern));
    }
    if (s.index_dim > 0) {
        w.add_u32(a + ".attention.indexer.head_count", static_cast<uint32_t>(s.index_heads));
        w.add_u32(a + ".attention.indexer.key_length", static_cast<uint32_t>(s.index_dim));
        w.add_u32(a + ".attention.indexer.top_k", static_cast<uint32_t>(s.index_topk));
        if (!s.index_full.empty()) w.add_bool_array(a + ".attention.indexer.types", s.index_full);
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
        if (!s.spm_pre.empty()) w.add_string("tokenizer.ggml.pre", s.spm_pre);
    } else {
        w.add_string("tokenizer.ggml.model", "gpt2");
        w.add_string("tokenizer.ggml.pre", s.bpe_pre);
        w.add_string_array("tokenizer.ggml.merges", {"h e"});
    }
    if (s.space_prefix >= 0) w.add_bool("tokenizer.ggml.add_space_prefix", s.space_prefix != 0);
    w.add_string_array("tokenizer.ggml.tokens", vocab);
    w.add_f32_array("tokenizer.ggml.scores", scores);
    w.add_i32_array("tokenizer.ggml.token_type", types);
    w.add_u32("tokenizer.ggml.unknown_token_id", 0);
    w.add_u32("tokenizer.ggml.bos_token_id", 1);
    w.add_u32("tokenizer.ggml.eos_token_id", 2);
    if (s.add_bos >= 0) w.add_bool("tokenizer.ggml.add_bos_token", s.add_bos != 0);
    if (s.identity_normalizer) {
        w.add_u8_array("tokenizer.ggml.precompiled_charsmap", {});
        w.add_bool("tokenizer.ggml.add_space_prefix", true);
    }
    if (s.pooling_type >= 0) w.add_u32(a + ".pooling_type", static_cast<uint32_t>(s.pooling_type));
    if (s.add_eos >= 0) w.add_bool("tokenizer.ggml.add_eos_token", s.add_eos != 0);
    if (!s.cls_labels.empty()) w.add_string_array(a + ".classifier.output_labels", s.cls_labels);
    if (!s.rerank_template.empty()) w.add_string("tokenizer.chat_template.rerank", s.rerank_template);
    if (s.altup_streams > 1) {
        w.add_u32(a + ".altup.num_inputs", static_cast<uint32_t>(s.altup_streams));
        w.add_u32(a + ".altup.active_idx", 0);
    }
    if (s.ple_dim > 0) w.add_u32(a + ".embedding_length_per_layer_input", static_cast<uint32_t>(s.ple_dim));
    if (s.shared_kv_layers > 0) w.add_u32(a + ".attention.shared_kv_layers", static_cast<uint32_t>(s.shared_kv_layers));
    if (!s.sparsity_scale.empty()) w.add_f32_array(a + ".activation_sparsity_scale", s.sparsity_scale);
    if (s.causal >= 0) w.add_bool(a + ".attention.causal", s.causal != 0);
    for (size_t i = 0; i < s.dense_widths.size(); ++i) {  // as the converter states them
        const std::string dense = i == 0 ? "dense_2" : "dense_3";
        w.add_u32(a + "." + dense + "_feat_in", static_cast<uint32_t>(i == 0 ? s.n_embd : s.dense_widths[i - 1]));
        w.add_u32(a + "." + dense + "_feat_out", static_cast<uint32_t>(s.dense_widths[i]));
    }

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
    auto attention_norm_weights = [&](int64_t n) {
        auto v = norm_weights(n);
        for (float& x : v) x *= s.attention_norm_scale;
        return v;
    };

    const int64_t d = s.n_embd;
    const int64_t hd = s.head_dim > 0 ? s.head_dim : a == "deepseek4" && s.shared_head_dim > 0 ? s.shared_head_dim : d / s.n_head;
    const int64_t q_dim = hd * s.n_head;
    const int64_t kv = hd * s.n_head_kv;
    std::mt19937_64 classic_rng(s.seed ^ 0xc1a551c);
    std::normal_distribution<float> classic_normal(0.0f, 0.1f);
    auto extra = [&](int64_t n, float offset = 0.0f) {
        std::vector<float> values(static_cast<size_t>(n));
        for (float& v : values) v = offset + classic_normal(classic_rng);
        return values;
    };
    auto bias = [&](const std::string& name, int64_t n, bool enabled) {
        if (enabled) w.add_tensor(name, {n}, DType::F32, extra(n));
    };
    const std::vector<float> embd = random(d * n_vocab, 0.5f);
    w.add_tensor("token_embd.weight", {d, n_vocab}, s.embd_type, embd);
    if (!s.no_norm_weights) w.add_tensor((a == "lfm2" || a == "lfm2moe") ? "token_embd_norm.weight" : "output_norm.weight", {d}, DType::F32, norm_weights(d));
    bias("output_norm.bias", d, s.norm_bias);
    bias("output.bias", n_vocab, s.projection_bias);
    if (s.embedding_norm) {
        w.add_tensor("token_embd_norm.weight", {d}, DType::F32, extra(d, 1.0f));
        bias("token_embd_norm.bias", d, s.norm_bias);
    }
    if (s.learned_positions) {
        w.add_tensor("position_embd.weight", {d, s.context_length}, s.embd_type, extra(d * s.context_length));
    }
    if (s.hc_count > 0) {
        const int64_t h = s.hc_count;
        w.add_tensor("output_hc_fn.weight", {h * d, h}, DType::F32, random(h * d * h, 0.01f));
        w.add_tensor("output_hc_base.weight", {h}, DType::F32, random(h, 0.1f));
        w.add_tensor("output_hc_scale.weight", {1}, DType::F32, {0.1f});
    }
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
        const bool windowed = !s.swa_layers.empty() && s.swa_layers[static_cast<size_t>(l)];
        const int64_t hd = windowed && s.swa_head_dim > 0 ? s.swa_head_dim :
            (s.head_dim > 0 ? s.head_dim : a == "deepseek4" && s.shared_head_dim > 0 ? s.shared_head_dim : d / s.n_head);
        const int64_t heads = s.block_heads.empty() ? s.n_head : s.block_heads[static_cast<size_t>(l)];
        const int64_t kv_heads = s.block_kv_heads.empty() ? s.n_head_kv : s.block_kv_heads[static_cast<size_t>(l)];
        const int64_t kv = kv_heads * hd;
        const int64_t ff = s.block_ff.empty() ? s.n_ff : s.block_ff[static_cast<size_t>(l)];
        const std::string attn_norm = a == "falcon" && !s.shared_parallel_norm ? "attn_norm_2" : "attn_norm";
        if (!s.post_only_norms && (mamba > 0 || heads > 0 || kv_heads > 0) && !s.no_norm_weights) w.add_tensor(p + attn_norm + ".weight", {d}, DType::F32, attention_norm_weights(d));
        bias(p + attn_norm + ".bias", d, s.norm_bias && !s.post_only_norms && (heads > 0 || kv_heads > 0));
        if (s.hc_count > 0) {
            const int64_t h = s.hc_count, width = h * (2 + h);
            for (const char* part : {"hc_attn", "hc_ffn"}) {
                const std::string prefix = p + part;
                w.add_tensor(prefix + "_fn.weight", {h * d, width}, DType::F32, random(h * d * width, 0.01f));
                w.add_tensor(prefix + "_base.weight", {width}, DType::F32, random(width, 0.2f));
                w.add_tensor(prefix + "_scale.weight", {3}, DType::F32, {0.1f, 0.2f, 0.3f});
            }
        }
        if (mamba > 0 && (!s.mamba_topology || l % 3 == 0)) {
            const bool v2 = mamba == 2;
            const int64_t di = 2 * d, ns = 16, heads = 4, rank = mamba_rank;
            const int64_t channels = v2 ? di + 2 * s.mamba_groups * ns : di;
            const int64_t input = v2 ? di + channels + heads : 2 * di;
            w.add_tensor(p + "ssm_in.weight", {d, input}, s.attn_type, random(d * input, s.weight_scale), s.ssm_block_pattern);
            w.add_tensor(p + "ssm_conv1d.weight", {s.mamba_kernel, channels}, DType::F32,
                         random(s.mamba_kernel * channels, 0.3f));
            w.add_tensor(p + "ssm_conv1d.bias", {channels}, DType::F32, random(channels, 0.1f));
            const int64_t decay = v2 ? heads : di * ns, skip = v2 ? heads : di;
            std::vector<float> a_neg = random(decay, 0.5f);
            for (float& x : a_neg) x = -std::exp(x);
            w.add_tensor(p + "ssm_a", {v2 ? 1 : ns, v2 ? heads : di}, DType::F32, a_neg);
            w.add_tensor(p + "ssm_d", {skip}, DType::F32, random(skip, 0.5f));
            w.add_tensor(p + "ssm_dt.bias", {skip}, DType::F32, random(skip, 0.5f));
            if (v2) {
                w.add_tensor(p + "ssm_norm.weight", {di / s.mamba_groups, s.mamba_groups}, DType::F32,
                             norm_weights(di));
            } else {
                w.add_tensor(p + "ssm_x.weight", {di, rank + 2 * ns}, s.attn_type,
                             random(di * (rank + 2 * ns), s.weight_scale), s.ssm_block_pattern);
                w.add_tensor(p + "ssm_dt.weight", {rank, di}, s.attn_type, random(rank * di, s.weight_scale), s.ssm_block_pattern);
                if (s.mamba_norms || s.mamba_weighted_norms) {
                    // Current transformers saves these dummy weights although
                    // Falcon-Mamba's mixer norms do not multiply by them.
                    const int64_t dt_norm = s.mamba_norms ? di : rank;
                    w.add_tensor(p + "ssm_dt_norm.weight", {dt_norm}, DType::F32, norm_weights(dt_norm));
                    w.add_tensor(p + "ssm_b_norm.weight", {ns}, DType::F32, norm_weights(ns));
                    w.add_tensor(p + "ssm_c_norm.weight", {ns}, DType::F32, norm_weights(ns));
                }
            }
            w.add_tensor(p + "ssm_out.weight", {di, d}, s.attn_type, random(di * d, s.weight_scale), s.ssm_block_pattern);
            if (s.mamba_parallel) {
                w.add_tensor(p + "attn_q.weight", {d, d}, s.attn_type, random(d * d, s.weight_scale));
                w.add_tensor(p + "attn_k.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
                w.add_tensor(p + "attn_v.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
                w.add_tensor(p + "attn_output.weight", {d, d}, s.attn_type, random(d * d, s.weight_scale));
            }
            if (s.mamba_bias) {
                w.add_tensor(p + "ssm_in.bias", {input}, DType::F32, random(input, 0.1f));
                w.add_tensor(p + "ssm_out.bias", {d}, DType::F32, random(d, 0.1f));
            }
            continue;  // a mixer-only block
        } else if (s.mamba_topology && l % 3 == 2) {
            // No mixer tensors: the block below is an FFN alone.
        } else if (s.short_conv_interval > 0 && (l + 1) % s.short_conv_interval != 0) {
            w.add_tensor(p + "shortconv.in_proj.weight", {d, 3 * d}, s.attn_type, random(3 * d * d, s.weight_scale), s.ssm_block_pattern);
            w.add_tensor(p + "shortconv.out_proj.weight", {d, d}, s.attn_type, random(d * d, s.weight_scale), s.ssm_block_pattern);
            w.add_tensor(p + "shortconv.conv.weight", {s.shortconv_taps, d}, DType::F32,
                         random(s.shortconv_taps * d, 0.3f));
            if (s.shortconv_bias) {
                w.add_tensor(p + "shortconv.in_proj.bias", {3 * d}, DType::F32, random(3 * d, 0.1f));
                w.add_tensor(p + "shortconv.out_proj.bias", {d}, DType::F32, random(d, 0.1f));
                w.add_tensor(p + "shortconv.conv.bias", {d}, DType::F32, random(d, 0.1f));
            }
        } else if (s.kda_interval > 0 && (l + 1) % s.kda_interval != 0) {
            const int64_t channels = ssm_hv * s.kda_head_dim;
            for (const char* part : {"q", "k", "v"}) {
                w.add_tensor(p + "attn_" + part + ".weight", {d, channels}, s.attn_type, random(d * channels, s.weight_scale), s.ssm_block_pattern);
                w.add_tensor(p + "ssm_conv1d_" + part + ".weight", {ssm_kernel, channels}, DType::F32,
                             random(ssm_kernel * channels, 0.3f));
            }
            w.add_tensor(p + "ssm_beta.weight", {d, ssm_hv}, DType::F32, random(d * ssm_hv, s.weight_scale));
            for (const char* part : {"f", "g"}) {
                w.add_tensor(p + "ssm_" + part + "_a.weight", {d, s.kda_head_dim}, s.attn_type, random(d * s.kda_head_dim, s.weight_scale), s.ssm_block_pattern);
                w.add_tensor(p + "ssm_" + part + "_b.weight", {s.kda_head_dim, channels}, DType::F32, random(s.kda_head_dim * channels, s.weight_scale));
            }
            auto a_neg = random(ssm_hv, 0.5f);
            for (float& x : a_neg) x = -std::exp(x);
            w.add_tensor(p + "ssm_a", {ssm_hv}, DType::F32, a_neg);
            w.add_tensor(p + "ssm_dt.bias", {channels}, DType::F32, random(channels, 0.5f));
            w.add_tensor(p + "ssm_norm.weight", {s.kda_head_dim}, DType::F32, norm_weights(s.kda_head_dim));
            w.add_tensor(p + "attn_output.weight", {channels, d}, s.attn_type, random(channels * d, s.weight_scale), s.ssm_block_pattern);
        } else if (s.delta_net_interval > 0 && (l + 1) % s.delta_net_interval != 0) {
            const int64_t key_dim = ssm_hk * ssm_hd, value_dim = ssm_hv * ssm_hd;
            const int64_t conv_dim = 2 * key_dim + value_dim;
            w.add_tensor(p + "attn_qkv.weight", {d, conv_dim}, s.attn_type, random(d * conv_dim, s.weight_scale), s.ssm_block_pattern);
            w.add_tensor(p + "attn_gate.weight", {d, value_dim}, s.attn_type, random(d * value_dim, s.weight_scale), s.ssm_block_pattern);
            if (s.fused_ssm_ba) {
                w.add_tensor(p + "ssm_ba.weight", {d, 2 * ssm_hv}, DType::F32, random(d * 2 * ssm_hv, s.weight_scale));
            } else {
            w.add_tensor(p + "ssm_alpha.weight", {d, ssm_hv}, DType::F32, random(d * ssm_hv, s.weight_scale));
            w.add_tensor(p + "ssm_beta.weight", {d, ssm_hv}, DType::F32, random(d * ssm_hv, s.weight_scale));
            }
            w.add_tensor(p + "ssm_out.weight", {value_dim, d}, s.attn_type, random(value_dim * d, s.weight_scale), s.ssm_block_pattern);
            w.add_tensor(p + "ssm_conv1d.weight", {ssm_kernel, conv_dim}, DType::F32,
                         random(ssm_kernel * conv_dim, 0.3f));
            std::vector<float> a_neg = random(ssm_hv, 0.5f);
            for (float& x : a_neg) x = -std::exp(x);  // A = -exp(A_log)
            w.add_tensor(p + "ssm_a", {ssm_hv}, DType::F32, a_neg);
            w.add_tensor(p + "ssm_dt.bias", {ssm_hv}, DType::F32, random(ssm_hv, 0.5f));
            w.add_tensor(p + "ssm_norm.weight", {ssm_hd}, DType::F32, norm_weights(ssm_hd));
        } else if (a == "deepseek4") {
            const int64_t qr = s.mla_q_rank, groups = s.shared_output_groups, rank = s.shared_output_rank;
            w.add_tensor(p + "attn_q_a.weight", {d, qr}, s.attn_type, random(d * qr, s.weight_scale));
            w.add_tensor(p + "attn_q_a_norm.weight", {qr}, DType::F32, attention_norm_weights(qr));
            w.add_tensor(p + "attn_q_b.weight", {qr, s.n_head * hd}, s.attn_type, random(qr * s.n_head * hd, s.weight_scale));
            w.add_tensor(p + "attn_kv.weight", {d, hd}, s.attn_type, random(d * hd, s.weight_scale));
            w.add_tensor(p + "attn_kv_a_norm.weight", {hd}, DType::F32, attention_norm_weights(hd));
            w.add_tensor(p + "attn_output_a.weight", {s.n_head * hd / groups, groups * rank}, s.attn_type,
                         random(s.n_head * hd * rank, s.weight_scale));
            w.add_tensor(p + "attn_output_b.weight", {groups * rank, d}, s.attn_type, random(groups * rank * d, s.weight_scale));
            const int64_t stride = s.compression_strides[static_cast<size_t>(l)];
            if (stride > 0) {
                const bool overlap = s.compression_overlap[static_cast<size_t>(l)];
                const int64_t width = hd * (overlap ? 2 : 1);
                w.add_tensor(p + "attn_compressor_kv.weight", {d, width}, s.attn_type, random(d * width, s.weight_scale));
                w.add_tensor(p + "attn_compressor_gate.weight", {d, width}, s.attn_type, random(d * width, s.weight_scale));
                w.add_tensor(p + "attn_compressor_ape.weight", {width, stride}, DType::F32, random(width * stride, 0.1f));
                w.add_tensor(p + "attn_compressor_norm.weight", {hd}, DType::F32, attention_norm_weights(hd));
                if (overlap) {
                    const int64_t id = s.index_dim;
                    w.add_tensor(p + "indexer_compressor_kv.weight", {d, 2 * id}, s.attn_type, random(d * 2 * id, s.weight_scale));
                    w.add_tensor(p + "indexer_compressor_gate.weight", {d, 2 * id}, s.attn_type, random(d * 2 * id, s.weight_scale));
                    w.add_tensor(p + "indexer_compressor_ape.weight", {2 * id, stride}, DType::F32, random(2 * id * stride, 0.1f));
                    w.add_tensor(p + "indexer_compressor_norm.weight", {id}, DType::F32, attention_norm_weights(id));
                    w.add_tensor(p + "indexer.attn_q_b.weight", {qr, s.index_heads * id}, s.attn_type,
                                 random(qr * s.index_heads * id, s.weight_scale));
                    w.add_tensor(p + "indexer.proj.weight", {d, s.index_heads}, s.attn_type, random(d * s.index_heads, s.weight_scale));
                }
            }
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
            const auto kb = random(s.mla_nope * rank * heads, s.weight_scale);
            const auto vb = random(rank * s.mla_v * heads, s.weight_scale);
            if (s.mla_legacy) {
                std::vector<float> combined(static_cast<size_t>(heads * (s.mla_nope + s.mla_v) * rank));
                for (int64_t h = 0; h < heads; ++h) {
                    for (int64_t i = 0; i < s.mla_nope; ++i) {
                        for (int64_t r = 0; r < rank; ++r) {
                            combined[static_cast<size_t>((h * (s.mla_nope + s.mla_v) + i) * rank + r)] =
                                kb[static_cast<size_t>((h * rank + r) * s.mla_nope + i)];
                        }
                    }
                    std::copy_n(vb.data() + h * s.mla_v * rank, s.mla_v * rank,
                                combined.data() + (h * (s.mla_nope + s.mla_v) + s.mla_nope) * rank);
                }
                w.add_tensor(p + "attn_kv_b.weight", {rank, heads * (s.mla_nope + s.mla_v)}, s.attn_type, combined);
            } else {
                w.add_tensor(p + "attn_k_b.weight", {s.mla_nope, rank, heads}, s.attn_type, kb);
                w.add_tensor(p + "attn_v_b.weight", {rank, s.mla_v, heads}, s.attn_type, vb);
            }
            w.add_tensor(p + "attn_output.weight", {heads * s.mla_v, d}, s.attn_type,
                         random(heads * s.mla_v * d, s.weight_scale));
            bias(p + "attn_output.bias", d, s.projection_bias || s.attn_output_bias);
        } else if (kv_heads > 0) {
            const int64_t q_dim = heads * hd;
            const int64_t q_rows = s.attn_gate ? 2 * q_dim : q_dim;  // [q_h | gate_h] per head when gated
            if (s.fused_qkv) {  // the same draws as separate matrices, stacked [q | k | v]
                std::vector<float> qkv = random(d * q_rows, s.weight_scale);
                for (int part = 0; part < 2; ++part) {  // k, then v
                    const std::vector<float> rows = random(d * kv, s.weight_scale);
                    qkv.insert(qkv.end(), rows.begin(), rows.end());
                }
                w.add_tensor(p + "attn_qkv.weight", {d, q_rows + 2 * kv}, s.attn_type, qkv);
            } else {
                // KV sharing: the last shared_kv_layers blocks have queries only;
                // values_from_keys: the global blocks have no attn_v.
                const bool shared = l >= s.n_layers - s.shared_kv_layers;
                const bool global = !s.swa_layers.empty() && !s.swa_layers[static_cast<size_t>(l)];
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
                if (!shared) w.add_tensor(p + "attn_k.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
                if (!shared && !(s.values_from_keys && global)) {
                    w.add_tensor(p + "attn_v.weight", {d, kv}, s.attn_type, random(d * kv, s.weight_scale));
                }
            }
            w.add_tensor(p + "attn_output.weight", {q_dim, d}, s.attn_type, random(q_dim * d, s.weight_scale));
            if (s.projection_bias) {
                const auto qb = extra(q_rows), kb = extra(kv), vb = extra(kv);
                if (s.fused_qkv) {
                    auto all = qb;
                    all.insert(all.end(), kb.begin(), kb.end());
                    all.insert(all.end(), vb.begin(), vb.end());
                    w.add_tensor(p + "attn_qkv.bias", {q_rows + 2 * kv}, DType::F32, all);
                } else {
                    w.add_tensor(p + "attn_q.bias", {q_rows}, DType::F32, qb);
                    w.add_tensor(p + "attn_k.bias", {kv}, DType::F32, kb);
                    w.add_tensor(p + "attn_v.bias", {kv}, DType::F32, vb);
                }
                bias(p + "attn_output.bias", d, true);
            }
            if (s.attn_output_bias && !s.projection_bias) bias(p + "attn_output.bias", d, true);
            if (s.qk_norm) {
                const int64_t qn = s.qk_norm_whole ? q_dim : hd, kn = s.qk_norm_whole ? kv : hd;
                w.add_tensor(p + "attn_q_norm.weight", {qn}, DType::F32, norm_weights(qn));
                std::vector<float> k_norm = norm_weights(kn);
                if (s.k_norm_outlier > 0.0f) {
                    k_norm[static_cast<size_t>(hd / 2 - 1)] = s.k_norm_outlier;
                    k_norm[static_cast<size_t>(hd - 3)] = s.k_norm_outlier;
                }
                if (l < s.n_layers - s.shared_kv_layers) w.add_tensor(p + "attn_k_norm.weight", {kn}, DType::F32, k_norm);
            }
            if (s.per_head_qk_norm) {
                w.add_tensor(p + "attn_q_norm.weight", {hd, s.n_head}, DType::F32, extra(d, 1.0f));
                w.add_tensor(p + "attn_k_norm.weight", {hd, s.n_head_kv}, DType::F32, extra(kv, 1.0f));
                if (s.norm_bias) {
                    w.add_tensor(p + "attn_q_norm.bias", {hd, s.n_head}, DType::F32, extra(d));
                    w.add_tensor(p + "attn_k_norm.bias", {hd, s.n_head_kv}, DType::F32, extra(kv));
                }
            }
        } else if (heads > 0) {
            w.add_tensor(p + "attn_output.weight", {d, d}, s.attn_type, random(d * d, s.weight_scale));
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
        const bool post_attention = a == "qwen35" || a == "qwen35moe" || a == "seed_oss" || a == "gpt-oss" || a == "glm4moe" || a == "qwen3next";
        const std::string ffn_norm = a == "falcon" ? "attn_norm" : a == "dbrx" ? "attn_output_norm" :
                                     post_attention ? "post_attention_norm" : "ffn_norm";
        if (!s.shared_parallel_norm && !s.post_only_norms && (ff > 0 || s.n_expert > 0)) {
            if (!s.no_norm_weights) w.add_tensor(p + ffn_norm + ".weight", {d}, DType::F32, norm_weights(d));
            bias(p + ffn_norm + ".bias", d, s.norm_bias);
        }
        if (s.post_norms || s.post_only_norms) {  // Gemma / OLMo: norms after both residual branches
            if (heads > 0 || kv_heads > 0) {
                w.add_tensor(p + "post_attention_norm" + (s.bare_post_norms ? "" : ".weight"), {d}, DType::F32, norm_weights(d));
            }
            if (ff > 0 || s.n_expert > 0) w.add_tensor(p + "post_ffw_norm" + (s.bare_post_norms ? "" : ".weight"), {d}, DType::F32, norm_weights(d));
        }
        if (ff == 0 && s.n_expert == 0) continue;
        const std::vector<float> gate = random(d * ff, s.weight_scale);
        const std::vector<float> up = random(d * ff, s.weight_scale);
        const std::vector<float> down = random(ff * d, s.weight_scale);
        if (s.projection_bias) {
            const auto gb = extra(ff), ub = extra(ff);
            if (!s.ungated && s.fused_gate_up) {
                auto all = gb;
                all.insert(all.end(), ub.begin(), ub.end());
                w.add_tensor(p + "ffn_up.bias", {2 * ff}, DType::F32, all);
            } else {
                if (!s.ungated) w.add_tensor(p + "ffn_gate.bias", {ff}, DType::F32, gb);
                w.add_tensor(p + "ffn_up.bias", {ff}, DType::F32, ub);
            }
            bias(p + "ffn_down.bias", d, true);
        }
        if (s.ungated) {
            w.add_tensor(p + "ffn_up.weight", {d, ff}, s.ffn_type, up);
            w.add_tensor(p + "ffn_down.weight", {ff, d}, s.ffn_type, down);
            continue;
        }
        if (s.parallel_experts && s.n_expert > 0) {
            w.add_tensor(p + "ffn_gate.weight", {d, ff}, s.ffn_type, gate);
            w.add_tensor(p + "ffn_up.weight", {d, ff}, s.ffn_type, up);
            w.add_tensor(p + "ffn_down.weight", {ff, d}, s.ffn_type, down);
            for (const char* name : {"pre_ffw_norm_2.weight", "post_ffw_norm_1.weight", "post_ffw_norm_2.weight"}) {
                w.add_tensor(p + name, {d}, DType::F32, norm_weights(d));
            }
            w.add_tensor(p + "ffn_gate_inp.scale", {d}, DType::F32, norm_weights(d));
            w.add_tensor(p + "ffn_down_exps.scale", {s.n_expert}, DType::F32, norm_weights(s.n_expert));
        }
        if (s.n_expert == 0 && s.fused_gate_up) {
            std::vector<float> gate_up = gate;
            gate_up.insert(gate_up.end(), up.begin(), up.end());
            w.add_tensor(p + "ffn_up.weight", {d, 2 * ff}, s.ffn_type, gate_up);
            w.add_tensor(p + "ffn_down.weight", {ff, d}, s.ffn_type, down);
            continue;
        }
        if (s.n_expert == 0 || l < s.dense_lead) {
            w.add_tensor(p + "ffn_gate.weight", {d, ff}, s.ffn_type, gate);
            w.add_tensor(p + "ffn_up.weight", {d, ff}, s.ffn_type, up);
            w.add_tensor(p + "ffn_down.weight", {ff, d}, s.ffn_type, down);
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
            const auto per_expert = static_cast<std::ptrdiff_t>(d * ff);
            for (std::ptrdiff_t e = 0; e < s.n_expert; ++e) {
                gate_up.insert(gate_up.end(), gate_exps.begin() + e * per_expert, gate_exps.begin() + (e + 1) * per_expert);
                gate_up.insert(gate_up.end(), up_exps.begin() + e * per_expert, up_exps.begin() + (e + 1) * per_expert);
            }
            w.add_tensor(p + "ffn_gate_up_exps.weight", {d, 2 * ff, s.n_expert}, s.ffn_type, gate_up);
        } else {
            w.add_tensor(p + "ffn_gate_exps.weight", {d, ff, s.n_expert}, s.ffn_type, gate_exps);
            w.add_tensor(p + "ffn_up_exps.weight", {d, ff, s.n_expert}, s.ffn_type, up_exps);
        }
        w.add_tensor(p + "ffn_down_exps.weight", {ff, d, s.n_expert}, s.ffn_type, stacked(down));
        // The router uses its own generator so every other weight matches the dense twin.
        std::vector<float> router(static_cast<size_t>(d * s.n_expert));
        for (float& x : router) x = normal(router_rng) * 0.5f;
        w.add_tensor(p + "ffn_gate_inp.weight", {d, s.n_expert}, DType::F32, router);
        if (l < s.hash_layers) {
            std::vector<int32_t> routes(static_cast<size_t>(n_vocab) * static_cast<size_t>(s.n_expert_used));
            for (int32_t token = 0; token < n_vocab; ++token) {
                for (int32_t rank = 0; rank < s.n_expert_used; ++rank) {
                    routes[static_cast<size_t>(token * s.n_expert_used + rank)] = (3 * token + rank) % s.n_expert;
                }
            }
            w.add_i32_tensor(p + "ffn_gate_tid2eid.weight", {s.n_expert_used, n_vocab}, routes);
        }
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
    auto moe_bias = [&](int64_t n) {
        std::vector<float> v(static_cast<size_t>(n));
        for (float& x : v) x = normal(bias_rng) * 0.2f;
        return v;
    };
    for (int32_t l = 0; l < s.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        if (s.attention_output_bias) w.add_tensor(p + "attn_output.bias", {d}, DType::F32, moe_bias(d));
        if (s.attention_sinks) w.add_tensor(p + "attn_sinks.weight", {s.n_head}, DType::F32, moe_bias(s.n_head));
        if (s.n_expert == 0 || l < s.dense_lead) continue;
        if (s.router_bias) w.add_tensor(p + "ffn_gate_inp.bias", {s.n_expert}, DType::F32, moe_bias(s.n_expert));
        if (s.expert_projection_bias) {
            w.add_tensor(p + "ffn_gate_exps.bias", {s.n_ff, s.n_expert}, DType::F32, moe_bias(s.n_ff * s.n_expert));
            w.add_tensor(p + "ffn_up_exps.bias", {s.n_ff, s.n_expert}, DType::F32, moe_bias(s.n_ff * s.n_expert));
            w.add_tensor(p + "ffn_down_exps.bias", {d, s.n_expert}, DType::F32, moe_bias(d * s.n_expert));
        }
    }
    // Gemma 3n / 4 parts and dense pooling projections: a generator of
    // their own, so the weights above stay those of the spec without them.
    std::mt19937_64 extra_rng(s.seed ^ 0x3e3e);
    std::normal_distribution<float> extra_normal(0.0f, 1.0f);
    auto gemma_extra = [&](int64_t count, float scale, float offset = 0.0f) {
        std::vector<float> v(static_cast<size_t>(count));
        for (float& x : v) x = offset + extra_normal(extra_rng) * scale;
        return v;
    };
    if (s.altup_streams > 1) {
        const int64_t k = s.altup_streams;
        w.add_tensor("altup_proj.weight", {d, d, k - 1}, DType::F32, gemma_extra(d * d * (k - 1), 0.09f));
        w.add_tensor("altup_unembd_proj.weight", {d, d, k - 1}, DType::F32, gemma_extra(d * d * (k - 1), 0.09f));
    }
    if (s.ple_dim > 0) {
        const int64_t width = int64_t{s.n_layers} * s.ple_dim;
        auto table = gemma_extra(width * n_vocab, 0.5f);  // preserve the independent random stream
        if (!s.ple_values.empty()) {
            if (s.ple_values.size() != table.size()) std::abort();
            table = s.ple_values;
        }
        w.add_tensor("per_layer_token_embd.weight", {width, n_vocab}, s.ple_type.value_or(s.embd_type), table,
                     s.ple_block_pattern);
        w.add_tensor("per_layer_model_proj.weight", {d, width}, s.attn_type, gemma_extra(d * width, s.weight_scale));
        w.add_tensor("per_layer_proj_norm.weight", {s.ple_dim}, DType::F32, gemma_extra(s.ple_dim, 0.05f, 1.0f));
    }
    for (int32_t l = 0; l < s.n_layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        if (s.altup_streams > 1) {
            const int64_t k = s.altup_streams;
            w.add_tensor(p + "altup_router.weight", {d, k}, DType::F32, gemma_extra(d * k, 1.0f));
            w.add_tensor(p + "altup_router_norm.weight", {d}, DType::F32, gemma_extra(d, 0.05f, 1.0f));
            w.add_tensor(p + "altup_predict_coef.weight", {k, k * k}, DType::F32, gemma_extra(k * k * k, 0.3f));
            w.add_tensor(p + "altup_correct_coef.weight", {k, k}, DType::F32, gemma_extra(k * k, 0.3f));
            if (s.altup_correct_scale) w.add_tensor(p + "altup_correct_scale.weight", {d}, DType::F32, gemma_extra(d, 0.2f, 1.0f));
        }
        if (s.laurel_rank > 0) {
            const int64_t r = s.laurel_rank;
            w.add_tensor(p + "laurel_l.weight", {d, r}, s.attn_type, gemma_extra(d * r, s.weight_scale));
            w.add_tensor(p + "laurel_r.weight", {r, d}, s.attn_type, gemma_extra(r * d, s.weight_scale * 2.0f));
            w.add_tensor(p + "laurel_post_norm.weight", {d}, DType::F32, gemma_extra(d, 0.05f, 1.0f));
        }
        if (s.ple_dim > 0) {
            const int64_t pd = s.ple_dim;
            w.add_tensor(p + "inp_gate.weight", {d, pd}, s.ffn_type, gemma_extra(d * pd, s.weight_scale));
            w.add_tensor(p + "proj.weight", {pd, d}, s.ffn_type, gemma_extra(pd * d, s.weight_scale));
            w.add_tensor(p + "post_norm.weight", {d}, DType::F32, gemma_extra(d, 0.05f, 1.0f));
        }
        if (s.layer_output_scale > 0.0f) {
            w.add_tensor(p + "layer_output_scale.weight", {1}, DType::F32,
                         {s.layer_output_scale * (1.0f + 0.25f * static_cast<float>(l % 3))});
        }
    }
    int64_t in = d;
    for (size_t i = 0; i < s.dense_widths.size(); ++i) {
        const int64_t out = s.dense_widths[i];
        w.add_tensor(i == 0 ? "dense_2.weight" : "dense_3.weight", {in, out}, DType::F32,
                     gemma_extra(in * out, 1.0f / std::sqrt(static_cast<float>(in))));
        in = out;
    }
    if (s.index_dim > 0 && a != "deepseek4") {
        std::mt19937_64 index_rng(s.seed ^ 0xd5a);
        std::normal_distribution<float> index_normal(0.0f, 0.1f);
        auto weights = [&](int64_t count, bool norm = false) {
            std::vector<float> out(static_cast<size_t>(count));
            for (float& value : out) value = index_normal(index_rng) + (norm ? 1.0f : 0.0f);
            return out;
        };
        for (int32_t l = 0; l < s.n_layers; ++l) {
            if (!s.index_full.empty() && !s.index_full[static_cast<size_t>(l)]) continue;
            const std::string p = "blk." + std::to_string(l) + ".indexer.";
            const int64_t dim = s.index_dim, heads = s.index_heads;
            w.add_tensor(p + "attn_q_b.weight", {s.mla_q_rank, dim * heads}, s.attn_type,
                         weights(s.mla_q_rank * dim * heads));
            w.add_tensor(p + "attn_k.weight", {d, dim}, s.attn_type, weights(d * dim));
            w.add_tensor(p + "proj.weight", {d, heads}, s.attn_type, weights(d * heads));
            w.add_tensor(p + "k_norm.weight", {dim}, DType::F32, weights(dim, true));
            w.add_tensor(p + "k_norm.bias", {dim}, DType::F32, weights(dim));
        }
    }
    w.write(path);
}

TinyModelSpec classic_model_spec(const std::string& arch) {
    TinyModelSpec s;
    s.arch = arch;
    s.identity_normalizer = arch == "orion";
    s.norm_bias = arch != "olmo" && arch != "command-r" && arch != "apertus" && arch != "arcee" && arch != "dbrx";
    s.projection_bias = s.norm_bias;
    s.ungated = arch == "gpt2" || arch == "gptneox" || arch == "falcon" || arch == "bloom" || arch == "mpt" ||
                arch == "starcoder" || arch == "starcoder2" || arch == "phi2" || arch == "nemotron" ||
                arch == "apertus" || arch == "arcee";
    s.fused_qkv = arch == "gpt2" || arch == "gptneox" || arch == "bloom" || arch == "falcon" || arch == "mpt" ||
                  arch == "starcoder" || arch == "dbrx";
    s.learned_positions = arch == "gpt2" || arch == "starcoder";
    s.embedding_norm = arch == "bloom";
    s.shared_parallel_norm = arch == "phi2" || arch == "command-r";
    s.no_norm_weights = arch == "olmo";
    s.per_head_qk_norm = arch == "stablelm" || arch == "command-r";
    s.qk_norm = arch == "apertus";
    if (arch == "gptneox" || arch == "phi2" || arch == "stablelm" || arch == "nemotron") s.rope_dim = 16;
    if (arch == "mpt" || arch == "olmo" || arch == "dbrx") s.clamp_kqv = 0.5f;
    if (arch == "command-r") s.logit_scale = 0.5f;
    if (arch == "starcoder2") s.sliding_window = 16;
    if (arch == "dbrx") {
        s.n_expert = 4;
        s.n_expert_used = 2;
        s.identical_experts = false;
    }
    return s;
}

TinyModelSpec compressed_spec() {
    TinyModelSpec s;
    s.arch = "deepseek4";
    s.n_layers = 3;
    s.n_head_kv = 1;
    s.mla_q_rank = 32;
    s.rope_dim = 16;
    s.index_dim = 32;
    s.index_topk = 2;
    s.sliding_window = 7;
    s.compression_strides = {0, 4, 8};
    s.compression_overlap = {false, true, false};
    s.hc_count = 4;
    s.weight_scale = 0.02f;
    s.attn_type = s.ffn_type = s.embd_type = s.output_type = DType::F32;
    return s;
}

TinyModelSpec deepseek4_spec() {
    TinyModelSpec s = compressed_spec();
    s.n_expert = 4;
    s.n_expert_used = 2;
    s.identical_experts = false;
    s.hash_layers = 1;
    s.sqrtsoftplus_gating = true;
    s.attention_sinks = true;
    s.n_ff_shared = 128;
    s.expert_bias = true;
    s.expert_swiglu_clamp = s.shared_swiglu_clamp = 0.3f;
    s.attn_type = s.output_type = DType::Q8_0;
    s.ffn_type = DType::Q4_0;
    s.embd_type = DType::F16;
    return s;
}

}  // namespace liyab::test
