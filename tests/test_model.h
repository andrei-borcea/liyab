// Liyab — synthetic GGUF models for tests. Writes small, deterministic
// llama/qwen2-layout models with a SentencePiece vocabulary so the full
// loader → tokenizer → transformer → engine path runs without downloads.
#ifndef LIYAB_TESTS_TEST_MODEL_H
#define LIYAB_TESTS_TEST_MODEL_H

#include <cstdint>
#include <string>
#include <vector>

#include "liyab/types.h"

namespace liyab::test {

// Low-level GGUF v3 writer.
class GgufWriter {
public:
    void add_u32(const std::string& key, uint32_t v);
    void add_f32(const std::string& key, float v);
    void add_bool(const std::string& key, bool v);
    void add_string(const std::string& key, const std::string& v);
    void add_string_array(const std::string& key, const std::vector<std::string>& v);
    void add_f32_array(const std::string& key, const std::vector<float>& v);
    void add_i32_array(const std::string& key, const std::vector<int32_t>& v);
    // `ne` in GGUF order (ne[0] = row length). Values are quantized to `type`.
    void add_tensor(const std::string& name, std::vector<int64_t> ne, DType type, const std::vector<float>& values);
    // Writes the file; aborts the test binary on I/O failure.
    void write(const std::string& path) const;

private:
    struct Tensor {
        std::string name;
        std::vector<int64_t> ne;
        DType type;
        std::vector<uint8_t> data;
    };
    std::vector<uint8_t> kv_;
    uint64_t n_kv_ = 0;
    std::vector<Tensor> tensors_;
};

struct TinyModelSpec {
    std::string arch = "llama";  // "llama", "qwen2" (adds QKV bias, NeoX RoPE), "qwen3" or "qwen3moe"
    int32_t n_layers = 2;
    int32_t n_embd = 128;
    int32_t n_ff = 256;
    int32_t n_head = 4;
    int32_t n_head_kv = 2;
    int32_t context_length = 512;
    // Mixed precision by default: Q8_0 attention, Q4_0 FFN, F16 embeddings.
    DType attn_type = DType::Q8_0;
    DType ffn_type = DType::Q4_0;
    DType embd_type = DType::F16;
    DType output_type = DType::Q8_0;
    bool tied_output = false;
    uint64_t seed = 1;
    float weight_scale = 0.08f;
    // n_expert > 0 writes mixture-of-experts FFNs: a random router and
    // n_expert copies of the dense FFN weights the same seed would produce,
    // so the model must compute exactly what its dense twin computes.
    int32_t n_expert = 0;
    int32_t n_expert_used = 0;
    // false: every expert gets its own random weights (no dense twin).
    bool identical_experts = true;
    // > 0 (arch "qwen35"): a hybrid stack in which every delta_net_interval-th
    // block is attention and the others are Gated DeltaNet (2 key heads, 4
    // value heads of 16, convolution kernel 4).
    int32_t delta_net_interval = 0;
};

// Vocabulary used by every tiny model: <unk>, <s>, </s>, 256 byte tokens,
// lowercase letters, and every substring of "▁hello" / "▁world".
std::vector<std::string> tiny_vocab();
int32_t tiny_token(const std::string& text);  // id of a vocab entry, or -1

void write_tiny_model(const std::string& path, const TinyModelSpec& spec = {});

}  // namespace liyab::test

#endif  // LIYAB_TESTS_TEST_MODEL_H
