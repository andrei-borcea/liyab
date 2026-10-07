// Liyab — extension points of the transformer forward pass.
//
// The core engine calls these optional hooks; the implementations live in
// the experimental modules (LIYAB_ENABLE_EXPERIMENTAL=ON). Keeping only the
// interfaces here means the core never depends on experimental code.
#ifndef LIYAB_EXPERIMENTAL_FORWARD_HOOKS_H
#define LIYAB_EXPERIMENTAL_FORWARD_HOOKS_H

#include <cstdint>
#include <span>

namespace liyab {

// Decides whether a single-token decode step may stop before the last block.
class EarlyExitHook {
public:
    virtual ~EarlyExitHook() = default;
    // Whether to run the exit classifier after block `layer` (0-based). Each
    // probe costs one LM-head matmul, so implementations probe sparingly.
    [[nodiscard]] virtual bool probe_after(int32_t layer, int32_t n_layers) const = 0;
    // `logits`: full-vocabulary logits computed from the intermediate hidden
    // state with the model's final norm and output head. True = exit now.
    virtual bool should_exit(std::span<const float> logits) = 0;
};

// Selects which attention heads a block evaluates.
class HeadMaskHook {
public:
    virtual ~HeadMaskHook() = default;
    // n_head bytes for block `layer` (nonzero = evaluate the head), or
    // nullptr to evaluate every head. Masked heads contribute zeros.
    [[nodiscard]] virtual const uint8_t* mask(int32_t layer) const = 0;
};

// Decides whether a block's FFN can be skipped for the current token.
class FfnSkipHook {
public:
    virtual ~FfnSkipHook() = default;
    // Called for single-token decode steps after the attention residual
    // update. `before` / `after` are the residual stream (n_embd floats)
    // entering block `layer` and after its attention. True = skip the FFN.
    virtual bool skip_ffn(int32_t layer, int32_t n_layers, std::span<const float> before,
                          std::span<const float> after) = 0;
};

struct ForwardHooks {
    EarlyExitHook* early_exit = nullptr;
    const HeadMaskHook* head_mask = nullptr;
    FfnSkipHook* ffn_skip = nullptr;
};

}  // namespace liyab

#endif  // LIYAB_EXPERIMENTAL_FORWARD_HOOKS_H
