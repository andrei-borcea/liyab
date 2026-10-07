#include "core/sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace liyab {

Sampler::Sampler(const SamplingParams& params)
    : params_(params), rng_(params.seed != 0 ? params.seed : std::random_device{}()) {}

int32_t Sampler::argmax(std::span<const float> values) noexcept {
    return static_cast<int32_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

void Sampler::probabilities(std::span<const float> logits, std::vector<float>& probs) const {
    const size_t n = logits.size();
    probs.assign(n, 0.0f);
    if (n == 0) return;
    if (greedy()) {
        probs[static_cast<size_t>(argmax(logits))] = 1.0f;
        return;
    }

    std::vector<int32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    auto by_logit = [&](int32_t a, int32_t b) { return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)]; };
    if (params_.top_k > 0 && static_cast<size_t>(params_.top_k) < n) {
        std::nth_element(ids.begin(), ids.begin() + params_.top_k, ids.end(), by_logit);
        ids.resize(static_cast<size_t>(params_.top_k));
    }
    std::sort(ids.begin(), ids.end(), by_logit);

    const float inv_t = 1.0f / params_.temperature;
    const float max_logit = logits[static_cast<size_t>(ids.front())];
    double sum = 0.0;
    for (const int32_t id : ids) {
        const auto p = static_cast<float>(std::exp((logits[static_cast<size_t>(id)] - max_logit) * inv_t));
        probs[static_cast<size_t>(id)] = p;
        sum += p;
    }
    // Top-p over the sorted candidates, then renormalize what is kept.
    size_t keep = ids.size();
    if (params_.top_p < 1.0f) {
        double cumulative = 0.0;
        for (size_t i = 0; i < ids.size(); ++i) {
            cumulative += probs[static_cast<size_t>(ids[i])] / sum;
            if (cumulative >= params_.top_p) {
                keep = i + 1;
                break;
            }
        }
    }
    double kept = 0.0;
    for (size_t i = 0; i < ids.size(); ++i) {
        float& p = probs[static_cast<size_t>(ids[i])];
        if (i >= keep) p = 0.0f;
        kept += p;
    }
    for (const int32_t id : ids) probs[static_cast<size_t>(id)] = static_cast<float>(probs[static_cast<size_t>(id)] / kept);
}

float Sampler::uniform() { return std::uniform_real_distribution<float>(0.0f, 1.0f)(rng_); }

int32_t Sampler::sample(std::span<const float> probs) {
    double total = 0.0;
    for (const float p : probs) total += p;
    if (total <= 0.0) return argmax(probs);
    double r = std::uniform_real_distribution<double>(0.0, total)(rng_);
    for (size_t i = 0; i < probs.size(); ++i) {
        r -= probs[i];
        if (r < 0.0) return static_cast<int32_t>(i);
    }
    // Floating-point slack: last token with non-zero mass.
    for (size_t i = probs.size(); i-- > 0;) {
        if (probs[i] > 0.0f) return static_cast<int32_t>(i);
    }
    return 0;
}

}  // namespace liyab
