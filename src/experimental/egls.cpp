#include "liyab/experimental/egls.h"

#include <cmath>

namespace liyab::experimental {

float Egls::energy_entropy(std::span<const float> x) noexcept {
    if (x.size() < 2) return 0.0f;
    double energy = 0.0;
    for (const float v : x) energy += static_cast<double>(v) * v;
    if (energy <= 0.0) return 0.0f;
    // H = -sum p log p with p_i = x_i^2 / E  ==  log E - (1/E) sum x_i^2 log x_i^2
    double weighted = 0.0;
    for (const float v : x) {
        const double e = static_cast<double>(v) * v;
        if (e > 0.0) weighted += e * std::log(e);
    }
    const double h = std::log(energy) - weighted / energy;
    return static_cast<float>(h / std::log(static_cast<double>(x.size())));
}

bool Egls::skip_ffn(int32_t layer, int32_t n_layers, std::span<const float> before, std::span<const float> after) {
    if (layer < config_.protect_first || layer >= n_layers - config_.protect_last) return false;
    const double delta = std::fabs(static_cast<double>(energy_entropy(after)) - energy_entropy(before));
    ++decisions_;
    delta_sum_ += delta;
    const bool skip = delta < config_.threshold;
    skips_ += skip ? 1 : 0;
    return skip;
}

}  // namespace liyab::experimental
