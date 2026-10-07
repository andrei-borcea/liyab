// Liyab (experimental) — confidence-based early exit for decoding.
//
// Between transformer blocks, the intermediate hidden state is projected
// through the model's final norm + LM head (a "logit lens" classifier). When
// the most likely token's probability exceeds the threshold (0.98 by
// default), the remaining blocks are skipped and that token is returned.
//
// Trade-offs, measured by tests/test_experimental.cpp:
//  * Output changes: models not trained for early exit (most checkpoints;
//    LayerSkip-style training fixes this) can pick a different token than the
//    full model. Keep it off where exact output matters.
//  * Each probe is one LM-head matmul (n_vocab x n_embd, often the largest
//    single matrix), so probing every block can cost more than it saves.
//    `probe_interval` and `min_layer` bound that overhead.
//  * KV cache coherence: later tokens attend to every block, so on exit the
//    engine still writes K/V for the skipped blocks from the exited hidden
//    state ("state propagation", CALM 2022). Only Wk/Wv of skipped blocks
//    are read; Wq, Wo and the FFN — the bulk of the bytes — are skipped.
#ifndef LIYAB_EXPERIMENTAL_EARLY_EXIT_H
#define LIYAB_EXPERIMENTAL_EARLY_EXIT_H

#include <cstdint>
#include <span>

#include "liyab/experimental/forward_hooks.h"
#include "liyab/types.h"

namespace liyab::experimental {

struct EarlyExitConfig {
    float confidence_threshold = 0.98f;  // exit when max softmax probability > threshold
    int32_t min_layer = -1;              // first block allowed to exit; -1 = n_layers / 2
    int32_t probe_interval = 4;          // probe every N blocks from min_layer
};

class LIYAB_API EarlyExit final : public EarlyExitHook {
public:
    explicit EarlyExit(EarlyExitConfig config = {}) : config_(config) {}

    [[nodiscard]] bool probe_after(int32_t layer, int32_t n_layers) const override;
    bool should_exit(std::span<const float> logits) override;

    // Maximum softmax probability of `logits`, in [1/n, 1].
    static float confidence(std::span<const float> logits) noexcept;
    // Shannon entropy of softmax(logits) in nats (0 = certain).
    static float entropy(std::span<const float> logits) noexcept;

    [[nodiscard]] const EarlyExitConfig& config() const noexcept { return config_; }
    [[nodiscard]] int64_t probes() const noexcept { return probes_; }
    [[nodiscard]] int64_t exits() const noexcept { return exits_; }
    void reset_stats() noexcept { probes_ = exits_ = 0; }

private:
    EarlyExitConfig config_;
    int64_t probes_ = 0;
    int64_t exits_ = 0;
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_EARLY_EXIT_H
