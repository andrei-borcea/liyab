// Gated window compression for shared-key/value attention. Each channel has
// its own softmax over source positions. Only running maxima, denominators
// and weighted numerators survive a step; source projections are discarded.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace liyab {

struct CompressionPlan {
    int32_t dim = 0;
    int32_t stride = 0;
    // Two projected series: previous window's first half plus this window's
    // second half. The first window has no previous contribution.
    bool overlap = false;

    [[nodiscard]] size_t state_floats() const noexcept {
        return static_cast<size_t>(dim) * (overlap ? 9 : 3);
    }
    // dim/stride are positive and pos is nonnegative. Steps must follow a
    // sequence's absolute positions. state has exactly
    // state_floats() floats, initially zero, and scratch has at least 2*dim.
    // kv/gate/bias have dim (2*dim with overlap) channels; bias is the current
    // position's row of the learned [stride, projected_dim] position bias.
    // Writes dim unnormalized output channels only when a window closes;
    // returns whether it did. State contains no position counter: pos fixes
    // the phase and is restored with the context. Independent states can run
    // concurrently; all spans must be disjoint except scratch across calls.
    bool step(std::span<float> state, const float* kv, const float* gate, const float* bias,
              int32_t pos, float* out, std::span<float> scratch) const noexcept;
};

}  // namespace liyab
