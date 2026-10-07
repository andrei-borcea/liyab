#include "liyab/experimental/head_pruner.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "core/quant.h"

namespace liyab::experimental {

Result<HeadPruner> HeadPruner::create(std::span<const TensorView* const> attn_output, int32_t n_head,
                                      int32_t head_dim, HeadPrunerConfig config) {
    if (n_head <= 0 || head_dim <= 0 || attn_output.empty()) {
        return Status(ErrorCode::InvalidArgument, "head pruner needs heads and at least one block");
    }
    if (!(config.keep_ratio > 0.0f && config.keep_ratio <= 1.0f)) {
        return Status(ErrorCode::InvalidArgument, "keep_ratio must be in (0, 1]");
    }
    HeadPruner pruner;
    pruner.n_head_ = n_head;
    pruner.kept_ = std::clamp(static_cast<int32_t>(std::ceil(config.keep_ratio * static_cast<float>(n_head))), 1, n_head);
    const auto layers = attn_output.size();
    const auto heads = static_cast<size_t>(n_head);
    pruner.importance_.assign(layers * heads, 0.0f);
    pruner.masks_.assign(layers * heads, 0);

    std::vector<float> row;
    for (size_t l = 0; l < layers; ++l) {
        const TensorView* wo = attn_output[l];
        if (wo == nullptr || wo->cols() != int64_t{n_head} * head_dim) {
            return Status(ErrorCode::InvalidArgument, "attn_output weight does not match n_head * head_dim");
        }
        // Wo is [rows = n_embd][cols = n_head * head_dim]; head h owns columns
        // [h * head_dim, (h + 1) * head_dim) of every row.
        std::vector<double> sq(heads, 0.0);
        row.resize(static_cast<size_t>(wo->cols()));
        for (int64_t r = 0; r < wo->rows(); ++r) {
            quant::dequantize_row(wo->type, wo->row(r), row.data(), wo->cols());
            for (size_t h = 0; h < heads; ++h) {
                for (int32_t i = 0; i < head_dim; ++i) {
                    const double v = row[h * static_cast<size_t>(head_dim) + static_cast<size_t>(i)];
                    sq[h] += v * v;
                }
            }
        }
        float* imp = pruner.importance_.data() + l * heads;
        for (size_t h = 0; h < heads; ++h) imp[h] = static_cast<float>(std::sqrt(sq[h]));

        // Keep the `kept_` most important heads; ties keep the lower index.
        std::vector<int32_t> order(heads);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return imp[a] > imp[b]; });
        for (int32_t k = 0; k < pruner.kept_; ++k) pruner.masks_[l * heads + static_cast<size_t>(order[k])] = 1;
    }
    return pruner;
}

bool HeadPruner::update(const PowerPolicy& policy, PowerProfile profile) {
    active_ = policy.throttled || profile == PowerProfile::LowPower;
    return active_;
}

const uint8_t* HeadPruner::mask(int32_t layer) const {
    if (!active_ || kept_ == n_head_) return nullptr;
    const size_t offset = static_cast<size_t>(layer) * static_cast<size_t>(n_head_);
    return offset < masks_.size() ? masks_.data() + offset : nullptr;
}

float HeadPruner::importance(int32_t layer, int32_t head) const {
    const size_t i = static_cast<size_t>(layer) * static_cast<size_t>(n_head_) + static_cast<size_t>(head);
    return i < importance_.size() ? importance_[i] : 0.0f;
}

}  // namespace liyab::experimental
