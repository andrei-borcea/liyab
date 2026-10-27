// Liyab — per-channel key scaling for quantized KV caches (internal).
//
// Why: a quantized cache (Q8_0 by default) stores each key row in blocks of
// 32 values sharing one scale, set by the block's largest value. Some model
// families carry a few key channels far larger than the rest and nearly constant
// across tokens (Qwen2.5: about -130 where the others are about 1; Qwen3:
// up to ~300, from the learned K norm), so the small values sharing their
// block are rounded with a step a hundred times too coarse. Measured against
// an F16 cache, that cost Qwen2.5-0.5B a mean KL of 0.070 (top-1 86.9%) and
// Qwen3-0.6B 0.051 (88.7%); almost all of it from the keys, almost none from
// the values. Queries are rounded to Q8_0 too (for the integer dot
// products), with the same problem on a smaller scale.
//
// How: attention only needs q·k, and q·k = (q∘t)·(k/t) for any positive
// per-channel t. Keys are stored divided by t and queries multiplied by it
// (after RoPE), so each channel gets the same range on both sides, the large
// key channels shrink before quantization, and the products stay what they
// were. t is set once per model, from the keys and queries a short fixed
// text produces (key_calibration_text()), and only for models whose keys
// have such channels (kOutlierRatio); the others keep their keys as computed:
//  * t = sqrt(key range / query range) per (attention block, KV head,
//    channel), the query range being the largest of the heads sharing that
//    KV head;
//  * a range is the largest value the channel takes at any position
//    (kKeyRangePositions), not only at the text's few positions: RoPE turns
//    each pair of channels by a known angle per position, so from a token's
//    unrotated pair the largest value either channel reaches is exact;
//  * a range is floored at a quarter of the head's median, so a channel the
//    text barely reached cannot get an extreme factor, and t is clamped to
//    [1/kMaxKeyScale, kMaxKeyScale].
// The factors depend only on the model's weights and the kernels, so a
// prompt gets the same reply in every process; they are kept with saved
// contexts, which are converted when loaded with other factors.
#ifndef LIYAB_CORE_KEY_SCALING_H
#define LIYAB_CORE_KEY_SCALING_H

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "core/architecture.h"

namespace liyab {

// The fixed text the factors are measured on: a mix of prose, code, numbers
// and scripts, about 45 tokens. Changing it (or the rule above) changes
// every model's factors: bump kKeyScalingVersion so cached factors are
// measured again.
std::string_view key_calibration_text() noexcept;
inline constexpr uint32_t kKeyScalingVersion = 1;
inline constexpr float kMaxKeyScale = 64.0f;
// Scaled only when some head's largest key channel reaches this many times
// its median channel. Measured (largest per model): Qwen3-0.6B 107, Qwen3-4B
// 87, Qwen2.5-0.5B 52, against Qwen3.5-0.8B 10.5, Qwen3.5-4B 7.4, Llama
// 3.2-1B 6.3. Models below it gain nothing measurable from the scaling (their
// KL against F16 moved within noise, sometimes up), so they keep their keys
// as computed, bit for bit; above it, every head is scaled (scaling only the
// outlier heads left Qwen3-4B at KL 0.0088 instead of 0.0063).
inline constexpr float kOutlierRatio = 12.0f;
// Tokens of the text processed before deciding whether to go on: models with
// no outlier stop there (the measurement costs a 27B streamed from storage
// 25 s on the whole text).
inline constexpr size_t kKeyProbeTokens = 12;
// Positions the ranges cover: the model's training context, at most the
// engine's largest automatic context. Longer contexts (set explicitly) still
// work; keys beyond it may use a little more of their blocks' range. Measured
// on Qwen3-0.6B, 512 positions instead would gain little at the start of a
// context (KL 0.0005 vs 0.0018) and give the far positions no margin.
inline constexpr int32_t kKeyRangePositions = 32768;

// Geometry of the attention blocks whose keys are scaled.
struct KeyScalingShape {
    int32_t blocks = 0;     // attention blocks (KV cache slots)
    int32_t n_head = 0;     // query heads
    int32_t n_head_kv = 0;  // KV heads (n_head / n_head_kv queries share one)
    int32_t head_dim = 0;
    int32_t rope_dim = 0;   // rotated channels per head (the rest are not paired)
    RopeLayout rope = RopeLayout::Halves;
    // Radians each rotated pair turns per position: rope_dim / 2 entries per
    // block, block after block (blocks with a RoPE table of their own, such
    // as Gemma's windowed ones, turn at their own speeds).
    std::vector<float> pair_speed;
    // Per block: what RoPE multiplies the rotated channels by (LongRoPE's
    // attention factor); empty: 1 everywhere.
    std::vector<float> rope_magnitude;
    int32_t positions = 0;  // context length: the positions a key can take
    // Optional actual head counts in KV-slot order. Empty: the uniform
    // counts above. Maxima still determine the factor/image strides; unused
    // KV heads retain unit factors and are never sampled (Deci / NAS).
    std::vector<int32_t> query_heads;
    std::vector<int32_t> kv_heads;
    // Upper bound on the query's position-dependent temperature across the
    // covered context; 1 without it. The rotary maxima times this gain cover
    // every query position even when its temperature changes in steps.
    float query_scale = 1.0f;
    // Factors per scaled model: blocks * n_head_kv * head_dim.
    [[nodiscard]] size_t factors() const noexcept {
        return static_cast<size_t>(blocks) * static_cast<size_t>(n_head_kv) * static_cast<size_t>(head_dim);
    }
};

// The largest value each key and query channel takes at any position, and
// the factors they give.
class KeyRanges {
public:
    explicit KeyRanges(KeyScalingShape shape);
    // One token of attention block `block`, before RoPE: q [n_head * head_dim]
    // and k [n_head_kv * head_dim]; either may be nullptr to leave that
    // side's ranges unchanged (shared-KV blocks contribute queries only).
    // `q_gain` / `k_gain` (head_dim each,
    // shared by every head; nullptr: none) multiply each channel after the
    // rotation: the weights of Q/K norms applied after RoPE, for which the
    // caller passes q and k already multiplied by their heads' RMS factors.
    void add(int32_t block, const float* q, const float* k, const float* q_gain = nullptr,
             const float* k_gain = nullptr);
    // The largest key channel's range over its head's median channel, the
    // largest over all heads: how far the model's keys are from even.
    [[nodiscard]] float outlier_ratio() const;
    // t per (block, KV head, channel), in that order; all 1 when
    // outlier_ratio() is under kOutlierRatio.
    [[nodiscard]] std::vector<float> factors() const;

private:
    // Folds `heads` unrotated head vectors of one block, whose pairs turn at
    // `speed` (rope_dim / 2 entries) and are multiplied by `magnitude`, into
    // their per-channel maxima, each channel multiplied by |gain| after the
    // rotation (nullptr: 1).
    void update(const float* rows, int32_t heads, float* max, const float* speed, double magnitude,
                const float* gain, double row_scale = 1.0) const;
    KeyScalingShape shape_;
    std::vector<float> key_max_;    // [block][kv head][channel]
    std::vector<float> query_max_;  // [block][query head][channel]
};

}  // namespace liyab

#endif  // LIYAB_CORE_KEY_SCALING_H
