// Liyab (experimental) — prerouter study: how well can the experts of a block be guessed?
//
// Streaming a mixture-of-experts model from storage prefetches the experts of block l + 1 while block l runs. The
// engine guesses them by running block l + 1's router on the residual stream it has at that moment ("A router").
// This module measures, on decode steps, the precision@k of that guess against the experts the router really
// picked, next to two cheaper-to-learn signals that could correct it:
//
//   * a co-occurrence prior learned online: how often expert b of block l follows expert a of block l - 1 for the
//     tokens seen so far (candidate score = router logit + beta * log P(b | the previous block's experts));
//   * repetition: experts block l ran for the previous token (score = router logit + gamma * [ran last token]).
//
// Seven candidates combine the two (see candidates()). The module only counts: it never changes which experts are
// prefetched or run, so the output is the same with it on. Every `report_interval` tokens it logs the running
// precision of each candidate at INFO level. It needs a model whose experts stream from storage (that is when the
// engine predicts at all) and sees single-token decode steps only.
//
// Not thread-safe: the engine calls it from the thread that runs the forward pass.
#ifndef LIYAB_EXPERIMENTAL_PREROUTER_TRACE_H
#define LIYAB_EXPERIMENTAL_PREROUTER_TRACE_H

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "liyab/experimental/forward_hooks.h"
#include "liyab/types.h"

namespace liyab::experimental {

class LIYAB_API PrerouterTrace final : public ExpertPredictionObserver {
public:
    static constexpr int kCandidates = 7;
    struct Candidate {
        const char* name;
        float prior_weight;   // beta: weight of the co-occurrence prior's log probability
        float repeat_weight;  // gamma: bonus for an expert the block ran for the previous token
    };
    [[nodiscard]] static const std::array<Candidate, kCandidates>& candidates() noexcept;

    // Logs the precision of every candidate each `report_interval` decode tokens (0: never; read report()).
    explicit PrerouterTrace(int32_t report_interval = 64) : report_interval_(report_interval) {}

    void begin(int32_t n_layers, int32_t n_experts, int32_t experts_per_token) override;
    void predicted(int32_t layer, std::span<const float> router_logits) override;
    void chosen(int32_t layer, std::span<const int32_t> experts) override;

    // Decode tokens seen (a token ends when the block index stops increasing).
    [[nodiscard]] int64_t tokens() const noexcept { return tokens_; }
    // Expert guesses scored so far (experts_per_token per block and token).
    [[nodiscard]] int64_t guesses() const noexcept { return guesses_; }
    // Share of guesses of candidate `index` that the router ran, in [0, 1]; 0 before any guess.
    [[nodiscard]] double precision(int index) const noexcept;
    // "prerouter: 64 tokens, precision@8: A router 79.1% | B b=.5 80.2% | ..." for the guesses so far.
    [[nodiscard]] std::string report() const;

private:
    void end_token();

    int32_t report_interval_;
    int32_t n_layers_ = 0, n_experts_ = 0, k_ = 0;
    int32_t last_layer_ = -1;
    int64_t tokens_ = 0, guesses_ = 0;
    std::array<int64_t, kCandidates> hits_{};
    std::vector<std::vector<float>> logits_;      // [layer] router logits predicted for it, until its choice arrives
    std::vector<std::vector<float>> pairs_;       // [layer] n_experts x n_experts: chosen at layer -> at layer + 1
    std::vector<std::vector<float>> times_;       // [layer] n_experts: times each expert was chosen
    std::vector<std::vector<int32_t>> previous_;  // [layer] experts run for the previous token
    std::vector<std::vector<int32_t>> current_;   // [layer] experts run for this token
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_PREROUTER_TRACE_H
