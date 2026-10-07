// Liyab — token sampling (internal). Produces full-vocabulary distributions
// so speculative decoding can compare target and draft probabilities.
#ifndef LIYAB_CORE_SAMPLING_H
#define LIYAB_CORE_SAMPLING_H

#include <random>
#include <span>
#include <vector>

#include "liyab/types.h"

namespace liyab {

class Sampler {
public:
    explicit Sampler(const SamplingParams& params);

    [[nodiscard]] bool greedy() const noexcept { return params_.temperature <= 0.0f; }

    // Temperature → top-k → top-p, renormalized; zero outside the kept set.
    // Greedy decoding yields a one-hot distribution at the argmax.
    void probabilities(std::span<const float> logits, std::vector<float>& probs) const;
    // Draws from a (not necessarily normalized) non-negative distribution.
    int32_t sample(std::span<const float> probs);
    float uniform();

    static int32_t argmax(std::span<const float> values) noexcept;

private:
    SamplingParams params_;
    std::mt19937_64 rng_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_SAMPLING_H
