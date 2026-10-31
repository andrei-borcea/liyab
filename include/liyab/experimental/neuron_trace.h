// Liyab (experimental) — FFN neuron study: how much of a dense FFN does a decode token really use, and how much of
// it can be guessed from earlier tokens?
//
// A dense FFN computes down(act(gate x) * up x). The neurons whose product act(gate) * up is small hardly change
// the output, so a sparse FFN could skip their rows of gate / up and columns of down; which neurons those are is
// only known after gate and up are computed, unless it can be predicted. This module measures, on decode steps,
// three things about the vector h = act(gate x) * up x that the down projection reads (one per FFN neuron):
//
//   * Density. The active set S_p of a token is the fewest neurons whose squares add up to the share p of the
//     squared norm of h (p = 90% and 99%): what a lossy skip would have to keep for that share of the energy.
//   * Persistence. The Jaccard index |S_prev ∩ S| / |S_prev ∪ S| between the active sets of consecutive tokens of a
//     block, and the share of this token's energy that the previous token's set carries (recall): a set reused as
//     the next token's guess is worth what its recall is.
//   * Hot neurons. Neurons ranked by how often they were active during the first `train_tokens` tokens; the share
//     of later tokens' energy that the top 10%, 25% and 50% of the ranking carry, and with the previous token's
//     90% set added to the top 25% (with the size of that union). A ranking that carries most of the energy in a
//     small prefix means the neurons could be permuted offline (gate / up rows, down columns together: exact) so
//     that the hot ones are one contiguous range.
//
// The module only counts: it never changes the output. Every `report_interval` decode tokens it logs report() at
// INFO level. It sees single-token steps of dense FFNs (the dense FFN of any block; not experts) in block order
// within a token; a token ends when the block index stops increasing. Costs a sort of n_ff values per block and
// token, so decoding runs slower while it is on.
//
// Not thread-safe: the engine calls it from the thread that runs the forward pass.
#ifndef LIYAB_EXPERIMENTAL_NEURON_TRACE_H
#define LIYAB_EXPERIMENTAL_NEURON_TRACE_H

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "liyab/experimental/forward_hooks.h"
#include "liyab/types.h"

namespace liyab::experimental {

class LIYAB_API NeuronTrace final : public FfnActivationObserver {
public:
    static constexpr std::array<double, 2> kCoverage = {0.90, 0.99};
    static constexpr std::array<double, 3> kHotShare = {0.10, 0.25, 0.50};

    // Means over the observed (block, token) pairs.
    struct Summary {
        int64_t tokens = 0;        // decode tokens seen
        int64_t observations = 0;  // (block, token) pairs
        std::array<double, 2> density{};      // |S_p| / n_ff, per coverage
        std::array<double, 2> jaccard{};      // consecutive tokens' sets (pairs with a previous token only)
        std::array<double, 2> recall{};       // energy of this token carried by the previous token's set
        int64_t persistence_pairs = 0;
        std::array<double, 3> hot_recall{};   // energy carried by the top share of the training ranking
        double union_recall = 0.0;            // top 25% plus the previous token's 90% set
        double union_density = 0.0;           // ... and its size / n_ff
        int64_t hot_observations = 0;         // pairs after training
    };

    // Ranks neurons on the first `train_tokens` tokens; logs report() each `report_interval` tokens (0: never).
    explicit NeuronTrace(int32_t report_interval = 64, int32_t train_tokens = 64)
        : report_interval_(report_interval), train_tokens_(train_tokens) {}

    void begin(int32_t n_layers) override;
    void activations(int32_t layer, std::span<const float> hidden) override;

    [[nodiscard]] Summary summary() const;
    [[nodiscard]] std::string report() const;

private:
    struct Block {
        std::array<std::vector<uint8_t>, 2> previous;  // the previous token's sets per coverage (empty before the first)
        std::vector<uint32_t> frequency;  // tokens each neuron was in the 90% set, during training
        std::vector<uint32_t> hot_rank;   // neuron indices, hottest first (after training)
    };
    void end_token();

    int32_t report_interval_;
    int32_t train_tokens_;
    int32_t n_layers_ = 0;
    int32_t last_layer_ = -1;
    int64_t tokens_ = 0;
    std::vector<Block> blocks_;
    Summary sum_;  // running sums; summary() divides
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_NEURON_TRACE_H
