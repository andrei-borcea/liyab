// Liyab (experimental) — thermal-aware dynamic attention-head pruning.
//
// Each query head gets a static importance score: the Frobenius norm of its
// slice of the attention output projection Wo (the columns that read that
// head's output). Heads whose output is projected with small weights move
// the residual stream least. While the PowerManager reports thermal
// throttling (skin >= 40 °C by default, or a severe OS thermal status) or the
// profile is LowPower, the least important heads of every block are masked:
// their QK·softmax·V work, the dominant attention cost at long context
// (it reads the whole KV cache), is skipped.
//
// Scope: K/V are still computed and cached for every KV head (other query
// heads of the GQA group may need them, and masks change over time), and the
// Q / O projections still run at full width. Pruning changes the output; the
// keep ratio trades quality for heat.
#ifndef LIYAB_EXPERIMENTAL_HEAD_PRUNER_H
#define LIYAB_EXPERIMENTAL_HEAD_PRUNER_H

#include <cstdint>
#include <span>
#include <vector>

#include "liyab/experimental/forward_hooks.h"
#include "liyab/power_manager.h"
#include "liyab/types.h"

namespace liyab::experimental {

struct HeadPrunerConfig {
    float keep_ratio = 0.75f;  // fraction of query heads kept per block while pruning (at least one)
};

class LIYAB_API HeadPruner final : public HeadMaskHook {
public:
    // `attn_output` holds every block's Wo (cols = n_head * head_dim).
    static Result<HeadPruner> create(std::span<const TensorView* const> attn_output, int32_t n_head,
                                     int32_t head_dim, HeadPrunerConfig config = {});

    // Switches pruning on while `policy.throttled` or `profile == LowPower`.
    // Returns whether pruning is active.
    bool update(const PowerPolicy& policy, PowerProfile profile);
    void set_active(bool active) noexcept { active_ = active; }
    [[nodiscard]] bool active() const noexcept { return active_; }

    [[nodiscard]] const uint8_t* mask(int32_t layer) const override;
    [[nodiscard]] float importance(int32_t layer, int32_t head) const;
    [[nodiscard]] int32_t kept_heads() const noexcept { return kept_; }
    [[nodiscard]] int32_t n_head() const noexcept { return n_head_; }

private:
    HeadPruner() = default;

    int32_t n_head_ = 0;
    int32_t kept_ = 0;
    bool active_ = false;
    std::vector<float> importance_;  // [layer][head]
    std::vector<uint8_t> masks_;     // [layer][head], precomputed
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_HEAD_PRUNER_H
