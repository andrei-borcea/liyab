// Liyab (experimental) — Entropy-Guided Layer Skipping (EGLS).
//
// For each decode step and block, EGLS measures how much the attention
// sub-layer reshaped the residual stream, as the change in the normalized
// "energy entropy" of the hidden state:
//
//     p_i = x_i^2 / sum_j x_j^2,   H(x) = -sum_i p_i log p_i / log(n)   in [0, 1]
//     dH  = | H(x_after_attention) - H(x_block_input) |
//
// A small dH marks a "low-complexity" token at that depth; if dH < threshold
// the block's FFN (gate/up/down: ~2/3 of a block's weight bytes) is skipped
// and the residual passes through unchanged (identity shortcut). The first
// and last blocks are protected because skipping them hurts most.
//
// Scope and honesty: a *learned* low-rank substitute for the skipped FFN
// would need offline calibration data (the SwiGLU FFN is nonlinear, so it has
// no exact low-rank form derivable from the weights at load time); this
// module uses the identity shortcut. Output changes whenever a block is
// skipped — tests/test_experimental.cpp reports speed against agreement with
// the full model across thresholds so the trade-off is measured, not assumed.
#ifndef LIYAB_EXPERIMENTAL_EGLS_H
#define LIYAB_EXPERIMENTAL_EGLS_H

#include <cstdint>
#include <span>

#include "liyab/experimental/forward_hooks.h"
#include "liyab/types.h"

namespace liyab::experimental {

struct EglsConfig {
    float threshold = 0.002f;    // skip the FFN when dH < threshold
    int32_t protect_first = 2;   // never skip the first N blocks
    int32_t protect_last = 2;    // never skip the last N blocks
};

class LIYAB_API Egls final : public FfnSkipHook {
public:
    explicit Egls(EglsConfig config = {}) : config_(config) {}

    bool skip_ffn(int32_t layer, int32_t n_layers, std::span<const float> before,
                  std::span<const float> after) override;

    // Normalized energy entropy of `x` in [0, 1] (0 = one dominant channel,
    // 1 = energy spread evenly). 0 for an all-zero vector.
    static float energy_entropy(std::span<const float> x) noexcept;

    [[nodiscard]] const EglsConfig& config() const noexcept { return config_; }
    void set_threshold(float threshold) noexcept { config_.threshold = threshold; }
    [[nodiscard]] int64_t decisions() const noexcept { return decisions_; }
    [[nodiscard]] int64_t skips() const noexcept { return skips_; }
    [[nodiscard]] double mean_delta() const noexcept { return decisions_ ? delta_sum_ / decisions_ : 0.0; }
    void reset_stats() noexcept { decisions_ = skips_ = 0; delta_sum_ = 0.0; }

private:
    EglsConfig config_;
    int64_t decisions_ = 0;
    int64_t skips_ = 0;
    double delta_sum_ = 0.0;
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_EGLS_H
