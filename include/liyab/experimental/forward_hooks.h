// Liyab — extension points of the transformer forward pass.
//
// The core engine calls these optional hooks; the implementations live in
// the experimental modules (LIYAB_ENABLE_EXPERIMENTAL=ON). Keeping only the
// interfaces here means the core never depends on experimental code.
#ifndef LIYAB_EXPERIMENTAL_FORWARD_HOOKS_H
#define LIYAB_EXPERIMENTAL_FORWARD_HOOKS_H

#include <atomic>
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

enum class FfnProjection : int32_t { Gate = 0, Up = 1, Down = 2 };

// Replaces an FFN matmul with an alternative implementation (e.g. sparse
// weights). Same contract as Backend::matmul; return false to run the
// regular backend instead.
class FfnMatmulHook {
public:
    virtual ~FfnMatmulHook() = default;
    virtual bool ffn_matmul(int32_t layer, FfnProjection projection, const float* x, float* y, int32_t n) = 0;
};

// Watches the streamed-expert predictor against the router (measurement only: it never changes the output).
// Called by Transformer for single-token decode steps of a mixture-of-experts model whose experts stream from
// storage (the prediction exists only then), in block order within a token.
class ExpertPredictionObserver {
public:
    virtual ~ExpertPredictionObserver() = default;
    // When attached: the model's block count, experts per block and experts the router picks per token.
    virtual void begin(int32_t n_layers, int32_t n_experts, int32_t experts_per_token) = 0;
    // The router logits of block `layer` (n_experts floats), computed before that block runs from the residual
    // stream of the block before it: what the predictor sees when it chooses experts to prefetch.
    virtual void predicted(int32_t layer, std::span<const float> router_logits) = 0;
    // The experts block `layer` really ran for the token, sorted. Follows predicted() for the same block.
    virtual void chosen(int32_t layer, std::span<const int32_t> experts) = 0;
};

struct ForwardHooks {
    // Set: forward() checks it between blocks and returns Cancelled when it
    // is true; the pass is then abandoned (Transformer::abandon_pass).
    const std::atomic<bool>* cancel = nullptr;
    EarlyExitHook* early_exit = nullptr;
    const HeadMaskHook* head_mask = nullptr;
    FfnSkipHook* ffn_skip = nullptr;
    FfnMatmulHook* ffn_matmul = nullptr;
};

}  // namespace liyab

#endif  // LIYAB_EXPERIMENTAL_FORWARD_HOOKS_H
