// Liyab — synthetic GGUF models for tests. Writes small, deterministic
// llama/qwen2-layout models with a SentencePiece vocabulary so the full
// loader → tokenizer → transformer → engine path runs without downloads.
#ifndef LIYAB_TESTS_TEST_MODEL_H
#define LIYAB_TESTS_TEST_MODEL_H

#include <cstdint>
#include <span>
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
    void add_u8_array(const std::string& key, const std::vector<uint8_t>& v);
    void add_bool_array(const std::string& key, const std::vector<bool>& v);
    // `ne` in GGUF order (ne[0] = row length). Values are quantized to `type`,
    // or replaced by repeated whole encoded blocks when explicitly supplied.
    // Aborts for reader-only formats without a payload, or malformed patterns:
    // quantize_row deliberately has no encoder for those weight-only types.
    void add_tensor(const std::string& name, std::vector<int64_t> ne, DType type, const std::vector<float>& values,
                    std::span<const uint8_t> block_pattern = {});
    // Replaces an existing tensor with repeated whole encoded blocks. Returns
    // false for a missing tensor or an empty/incomplete block pattern. Opt-in:
    // the ordinary writer and every existing golden fixture stay unchanged.
    bool set_tensor_block_pattern(const std::string& name, std::span<const uint8_t> pattern);
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
    // Any registered architecture: "llama", "qwen2" (adds QKV bias, NeoX RoPE), "qwen3", "qwen3moe",
    // "qwen35"/"qwen35moe" (post_attention_norm before the FFN), "hunyuan-dense" (Q/K norms after RoPE),
    // "gemma2" / "gemma3" (scaled embeddings, GeGLU, windowed blocks; with post_norms)...
    std::string arch = "llama";
    int32_t n_layers = 2;
    int32_t n_embd = 128;
    int32_t n_ff = 256;
    int32_t n_head = 4;
    int32_t n_head_kv = 2;
    // Optional NAS metadata, one entry per block. Zero KV/query heads omit
    // the mixer, positive query heads with zero KV heads write a linear
    // projection; zero FFN width omits the FFN. Head size stays n_embd/n_head.
    std::vector<int32_t> block_heads, block_kv_heads, block_ff;
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

    // Optional features, each one a separate path of the forward pass. Off by
    // default, and drawn from the generator only when on, so every model
    // written before these existed keeps its exact weights.
    bool qk_norm = false;        // per-head RMSNorm of Q and K (attn_q_norm / attn_k_norm)
    bool qk_norm_whole = false;  // one Q/K RMSNorm over all heads (requires qk_norm)
    bool post_only_norms = false; // norms after mixer/FFN, with no norms before either (OLMo 2)
    std::vector<bool> rope_layers; // attention.rope_pattern: true rotates, false leaves Q/K unchanged
    float embedding_scale = 0.0f; // > 0: explicit embedding/residual/logit metadata (no weight changes)
    float residual_scale = 0.0f;
    float logit_scale = 0.0f;
    float attn_temp_scale = 0.0f; // attention.temperature_scale / _length (Ministral 3)
    int32_t attn_temp_length = 0;
    bool attn_gate = false;      // attention output gated per head: attn_q is [q | gate], twice as tall
    int32_t rope_dim = 0;        // rotated dimensions per head (partial RoPE); 0: the whole head
    bool rope_freqs = false;     // per-pair RoPE frequency factors (rope_freqs.weight, Llama 3.x)
    float rope_scale = 0.0f;     // > 0: linear RoPE position scaling with this factor
    int32_t n_ff_shared = 0;     // MoE: shared expert of this hidden size (0: none)
    bool shared_gate = false;    // MoE: sigmoid gate of the shared expert (ffn_gate_inp_shexp)
    bool expert_bias = false;    // MoE: selection bias added to the router probabilities (exp_probs_b)
    bool sigmoid_gating = false; // MoE: sigmoid instead of softmax router (expert_gating_func = 2)
    // MoE: expert_weights_norm metadata: 1 true, 0 false, -1 absent (the
    // architecture's default applies).
    int32_t expert_weights_norm = 1;
    // qwen2: > 0 sets two key-bias channels per KV head to +-key_outlier (the
    // large constant key channels of Qwen2.5 that quantized KV caches round
    // badly); the other weights are unchanged.
    float key_outlier = 0.0f;
    // qk_norm: > 0 sets two channels of every attn_k_norm weight to
    // k_norm_outlier (large key channels made by the norm itself, as Qwen3
    // and Hunyuan have); the other weights are unchanged.
    float k_norm_outlier = 0.0f;
    float rope_base = 10000.0f;  // rope.freq_base (of the global blocks when some are windowed)
    int32_t sliding_window = 0;  // > 0: attention.sliding_window (the architecture says which blocks use it)
    bool post_norms = false;     // post_attention_norm and post_ffw_norm after the mixer and the FFN (Gemma)
    bool bare_post_norms = false; // omit their .weight suffix, as the PLaMo3 converter does
    bool fused_qkv = false;      // one attn_qkv [q | k | v] instead of attn_q, attn_k, attn_v (Phi-3)
    // ffn_up holds [gate | up], no ffn_gate (Phi-3); with n_expert > 0, ffn_gate_up_exps holds each
    // expert's [gate | up] (the split tensors' weights, as a converter fuses them)
    bool fused_gate_up = false;
    // > 0: LongRoPE (Phi-3) with this original context: rope_factors_short /
    // rope_factors_long and an attention factor of 1.19.
    int32_t longrope_ctx = 0;
    // attention.sliding_window_pattern: > 0 writes it as a period; a
    // non-empty swa_layers writes one flag per block (true: windowed) instead.
    int32_t swa_pattern = 0;
    std::vector<bool> swa_layers;
    // > 0: YaRN RoPE scaling (rope.scaling.type "yarn") with this factor;
    // yarn_ctx > 0 writes rope.scaling.original_context_length, the betas
    // are written when > 0 and yarn_log_mul (DeepSeek's
    // rope.scaling.yarn_log_multiplier) when != 0. Metadata only: the
    // weights are those of the same spec without YaRN.
    float yarn_factor = 0.0f;
    int32_t yarn_ctx = 0;
    float yarn_beta_fast = 0.0f;
    float yarn_beta_slow = 0.0f;
    float yarn_log_mul = 0.0f;
    // > 0: attn_logit_softcapping / final_logit_softcapping (Gemma 2) and
    // attention.scale, the q·k scale the file states. Metadata only.
    float attn_softcap = 0.0f;
    float final_softcap = 0.0f;
    float attn_scale = 0.0f;
    // > 0 (arch "deepseek2", n_head_kv 1): multi-head latent attention with
    // a latent this wide, as the converter writes it: attn_kv_a_mqa [d ->
    // mla_kv_rank + rope_dim], attn_kv_a_norm, per head attn_k_b [mla_nope ->
    // mla_kv_rank] and attn_v_b [mla_kv_rank -> mla_v], attn_output [n_head *
    // mla_v -> d], and attn_q [d -> n_head * (mla_nope + rope_dim)] or, with
    // mla_q_rank > 0, attn_q_a, attn_q_a_norm and attn_q_b. rope_dim must be
    // set (the shared RoPE key's width). Every width must be whole blocks
    // of the tensor types (multiples of 32 for Q8_0 / Q4_0). k_norm_outlier
    // then sets two channels of every attn_kv_a_norm (large latent channels,
    // for key scaling).
    int32_t mla_kv_rank = 0;
    int32_t mla_q_rank = 0;
    int32_t mla_nope = 32;
    int32_t mla_v = 32;
    bool mla_legacy = false;  // no *_mla keys, as in files converted before attn_kv_b was split
    // MoE models: this many leading blocks keep a dense FFN (DeepSeek's
    // leading_dense_block_count; the plan follows the tensors).
    int32_t dense_lead = 0;
    // SSM synthetic blocks, selected by tensors rather than the architecture
    // name: 1 = Mamba-1, 2 = Mamba-2. Optional attention/FFN-only blocks
    // exercise mixed topology; norm and biases exercise Falcon-Mamba.
    int32_t mamba = 0;
    int32_t mamba_groups = 1;
    int32_t mamba_kernel = 4;
    bool mamba_norms = false;
    bool mamba_weighted_norms = false;
    bool mamba_bias = false;
    bool mamba_topology = false;
    bool mamba_parallel = false;  // invalid variant: attention and SSM in one block, for rejection tests
    // Explicit encoded blocks for SSM projections only. Empty keeps the
    // original random encoder path and every existing golden unchanged.
    std::vector<uint8_t> ssm_block_pattern;

    // Classic blocks. Optional tensors use an independent generator, so
    // enabling biases never changes the projection weights being compared.
    bool ungated = false;             // ffn_up -> activation -> ffn_down
    bool norm_bias = false;           // LayerNorm offsets, including output_norm
    bool no_norm_weights = false;     // non-parametric LayerNorm (OLMo)
    bool projection_bias = false;     // Q/K/V/O, FFN and output biases
    bool attn_output_bias = false;    // attention output only, also for latent attention
    bool shared_parallel_norm = false; // omit ffn_norm (one shared norm)
    bool embedding_norm = false;      // token_embd_norm, as in BLOOM
    bool learned_positions = false;   // context_length absolute position rows
    bool per_head_qk_norm = false;    // distinct LayerNorm weight for each head
    float clamp_kqv = 0.0f;
    int32_t parallel_residual = -1;    // -1: row default, 0/1: metadata override

    // Pooling models (core/pooling.h), off by default like the features above:
    // only metadata and tensors of their own, drawn from a generator of their
    // own, so the other weights stay those of the same spec without them.
    int32_t pooling_type = -1;            // >= 0: <arch>.pooling_type
    int32_t add_eos = -1;                 // >= 0: tokenizer.ggml.add_eos_token (nonzero: true)
    int32_t add_bos = 1;                  // -1: omit metadata; 0/1: explicit tokenizer BOS policy
    bool identity_normalizer = false;    // explicit identity map and dummy prefix for the synthetic vocabulary
    std::vector<std::string> cls_labels;  // non-empty: <arch>.classifier.output_labels
    // > 0: a ranking head cls.output.weight of this many rows (in output_type):
    // the output head's rows of the tokens in cls_rows_from when given (its
    // size must be cls_rows), else random.
    int32_t cls_rows = 0;
    std::vector<int32_t> cls_rows_from;
    bool cls_bias = false;          // cls.output.bias
    bool cls_dense = false;         // cls.weight: a two-layer head (refused by the engine)
    std::string rerank_template;    // non-empty: tokenizer.chat_template.rerank

    // Non-empty: the vocabulary is declared byte-level BPE (tokenizer.ggml.model
    // "gpt2", one merge) with this tokenizer.ggml.pre, e.g. a name the engine
    // does not know. Metadata only: the weights are those of the same spec.
    std::string bpe_pre;
    // Optional SentencePiece selection and prefix flag, metadata only.
    std::string spm_pre;
    int32_t space_prefix = -1;
    // Optional ignored metadata before the tokenizer, to test large headers.
    size_t tokenizer_padding = 0;
    // Optional MoE projection biases and learned attention sink logits. A separate
    // generator supplies these tensors so existing synthetic cases keep their bits.
    bool router_bias = false;
    bool expert_projection_bias = false;
    bool attention_output_bias = false;
    bool attention_sinks = false;
    bool omit_expert_width = false;  // Mixtral: infer the expert width from down projections
    bool separate_attention_gate = false;  // attn_gate instead of a fused [query | gate], with attn_gate true
    bool head_attention_gate = false;  // one scalar gate per head, with separate_attention_gate
    bool bare_expert_bias = false;     // selection correction named exp_probs_b without .bias
    float expert_swiglu_clamp = 0.0f;
    float shared_swiglu_clamp = 0.0f;
    int32_t head_dim = 0;  // explicit attention head width, independent of the residual width
    int32_t yarn_truncate = -1;  // -1: absent, 0/1: explicit YaRN endpoint rounding
};

// A representative classic block layout, shared by the cache, sequence and
// rollback tests. The architecture must be one of the classic registry rows.
TinyModelSpec classic_model_spec(const std::string& arch);

// Vocabulary used by every tiny model: <unk>, <s>, </s>, 256 byte tokens,
// lowercase letters, and every substring of "▁hello" / "▁world".
std::vector<std::string> tiny_vocab();
int32_t tiny_token(const std::string& text);  // id of a vocab entry, or -1

// NAS geometry covering two GQA ratios, differing query and FFN widths,
// an absent block, and a linear mixer. A single spec keeps every path test
// on the same geometry; optional norms/scales can be added by the caller.
TinyModelSpec nas_spec();

void write_tiny_model(const std::string& path, const TinyModelSpec& spec = {});

// Encoded GGML blocks and independently decoded values from gguf-py. Reads
// tests/data/quant_vectors.bin (LIYAB_TEST_DATA overrides the directory).
struct QuantVector {
    DType type{};
    int64_t n = 0;
    std::vector<uint8_t> blocks;
    std::vector<float> expected;
};
// False for absent or malformed data; reports the reason. No partial result.
bool load_quant_vectors(std::vector<QuantVector>& out, const std::string& filename = "quant_vectors.bin");

}  // namespace liyab::test

#endif  // LIYAB_TESTS_TEST_MODEL_H
