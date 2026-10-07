#include "liyab/experimental/early_exit.h"

#include <algorithm>
#include <cmath>

namespace liyab::experimental {

bool EarlyExit::probe_after(int32_t layer, int32_t n_layers) const {
    if (layer + 1 >= n_layers) return false;  // the last block exits anyway
    const int32_t first = config_.min_layer >= 0 ? config_.min_layer : n_layers / 2;
    const int32_t interval = std::max(1, config_.probe_interval);
    return layer >= first && (layer - first) % interval == 0;
}

bool EarlyExit::should_exit(std::span<const float> logits) {
    ++probes_;
    const bool exit = confidence(logits) > config_.confidence_threshold;
    exits_ += exit ? 1 : 0;
    return exit;
}

float EarlyExit::confidence(std::span<const float> logits) noexcept {
    if (logits.empty()) return 0.0f;
    const float max = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float v : logits) sum += std::exp(static_cast<double>(v - max));
    return static_cast<float>(1.0 / sum);  // exp(max - max) / sum
}

float EarlyExit::entropy(std::span<const float> logits) noexcept {
    if (logits.empty()) return 0.0f;
    const float max = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    double weighted = 0.0;  // sum of e^(v-max) * (v-max)
    for (const float v : logits) {
        const double z = static_cast<double>(v - max);
        const double e = std::exp(z);
        sum += e;
        weighted += e * z;
    }
    // H = log(sum) - E[z]
    return static_cast<float>(std::log(sum) - weighted / sum);
}

}  // namespace liyab::experimental
