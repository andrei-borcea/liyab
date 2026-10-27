#include "core/key_scaling.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace liyab {

namespace {

// Largest |cos ψ| for ψ in [start, start + sweep], sweep >= 0.
double max_abs_cos(double start, double sweep) {
    if (sweep >= std::numbers::pi) return 1.0;
    const double next = std::ceil(start / std::numbers::pi) * std::numbers::pi;  // first multiple of π >= start
    if (next <= start + sweep) return 1.0;
    return std::max(std::fabs(std::cos(start)), std::fabs(std::cos(start + sweep)));
}

}  // namespace

std::string_view key_calibration_text() noexcept {
    // About 45 tokens: as good as twice as many on Qwen2.5, Qwen3 and Llama
    // 3.2 (KL against F16 within noise), at half the cost on a MoE whose
    // experts stream (each token reads its own experts from storage).
    return "The harbour opened at 6:40; il treno parte alle 8:15. def f(xs): return sum(xs) * 1.5  # $3,210 東京 München";
}

KeyRanges::KeyRanges(KeyScalingShape shape)
    : shape_(std::move(shape)),
      key_max_(shape_.factors(), 0.0f),
      query_max_(static_cast<size_t>(shape_.blocks) * static_cast<size_t>(shape_.n_head) *
                     static_cast<size_t>(shape_.head_dim),
                 0.0f) {}

void KeyRanges::update(const float* rows, int32_t heads, float* max, const float* speed, double magnitude,
                       const float* gain, int32_t head_dim, int32_t rope_dim, double row_scale) const {
    if (rows == nullptr) return;  // shared-KV blocks contribute queries only
    const auto hd = static_cast<size_t>(head_dim);
    const int32_t half = rope_dim / 2;
    const double last = std::max(shape_.positions - 1, 0);
    // |gain| of channel i (1 without gains: r * 1.0 keeps r's bits).
    auto g = [&](int32_t i) { return gain == nullptr ? 1.0 : std::fabs(static_cast<double>(gain[i])); };
    for (size_t h = 0; h < static_cast<size_t>(heads); ++h) {
        const float* x = rows + h * hd;
        float* m = max + h * static_cast<size_t>(shape_.head_dim);
        // Rotated pairs: channel a holds r cos(φ + θp), channel b r sin(φ + θp)
        // (Transformer::rope), for p over the whole context.
        for (int32_t i = 0; i < half; ++i) {
            const int32_t a = shape_.rope == RopeLayout::Halves ? i : 2 * i;
            const int32_t b = shape_.rope == RopeLayout::Halves ? i + half : 2 * i + 1;
            const double r = std::hypot(static_cast<double>(x[a]), static_cast<double>(x[b])) * magnitude * row_scale;
            const double phi = std::atan2(static_cast<double>(x[b]), static_cast<double>(x[a]));
            const double sweep = static_cast<double>(speed[i]) * last;
            m[a] = std::max(m[a], static_cast<float>(g(a) * r * max_abs_cos(phi, sweep)));
            m[b] = std::max(m[b], static_cast<float>(g(b) * r * max_abs_cos(phi - std::numbers::pi / 2, sweep)));
        }
        for (size_t i = static_cast<size_t>(rope_dim); i < hd; ++i) {
            const float value = gain == nullptr ? std::fabs(x[i]) : std::fabs(x[i] * gain[i]);
            m[i] = std::max(m[i], static_cast<float>(value * row_scale));
        }
    }
}

void KeyRanges::add(int32_t block, const float* q, const float* k, const float* q_gain, const float* k_gain) {
    const auto hd = static_cast<size_t>(shape_.head_dim);
    const auto b = static_cast<size_t>(block);
    const float* speed = shape_.rope_dim == 0 ? nullptr :
                            shape_.pair_speed.data() + b * static_cast<size_t>(shape_.rope_dim / 2);
    const double magnitude = shape_.rope_magnitude.empty() ? 1.0 : shape_.rope_magnitude[b];
    const int32_t kv_heads = shape_.kv_heads.empty() ? shape_.n_head_kv : shape_.kv_heads[b];
    const int32_t query_heads = shape_.query_heads.empty() ? shape_.n_head : shape_.query_heads[b];
    const int32_t head_dim = shape_.head_dims.empty() ? shape_.head_dim : shape_.head_dims[b];
    const int32_t rope_dim = shape_.rope_dims.empty() ? shape_.rope_dim : shape_.rope_dims[b];
    update(k, kv_heads, key_max_.data() + b * static_cast<size_t>(shape_.n_head_kv) * hd, speed, magnitude,
           k_gain, head_dim, rope_dim);
    update(q, query_heads, query_max_.data() + b * static_cast<size_t>(shape_.n_head) * hd, speed, magnitude,
           q_gain, head_dim, rope_dim, shape_.query_scale);
}

float KeyRanges::outlier_ratio() const {
    const auto hd = static_cast<size_t>(shape_.head_dim);
    std::vector<float> sorted;
    float ratio = 0.0f;
    for (size_t b = 0; b < static_cast<size_t>(shape_.blocks); ++b) {
        const auto width = static_cast<size_t>(shape_.head_dims.empty() ? shape_.head_dim : shape_.head_dims[b]);
        const int32_t heads = shape_.kv_heads.empty() ? shape_.n_head_kv : shape_.kv_heads[b];
        sorted.resize(width);
        for (size_t h = 0; h < static_cast<size_t>(heads); ++h) {
            const float* km = key_max_.data() + (b * static_cast<size_t>(shape_.n_head_kv) + h) * hd;
            std::copy_n(km, width, sorted.begin());
            std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(width / 2), sorted.end());
            if (sorted[width / 2] > 0.0f) ratio = std::max(ratio, *std::max_element(km, km + width) / sorted[width / 2]);
        }
    }
    return ratio;
}

std::vector<float> KeyRanges::factors() const {
    const auto hd = static_cast<size_t>(shape_.head_dim);
    std::vector<float> out(shape_.factors(), 1.0f);
    if (outlier_ratio() < kOutlierRatio) return out;  // nothing for the scaling to fix
    std::vector<float> keys, queries, sorted;
    // Floors a head's ranges at a quarter of its median channel.
    auto floor_ranges = [&](std::vector<float>& range) {
        sorted = range;
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<std::ptrdiff_t>(range.size() / 2), sorted.end());
        const float floor = std::max(sorted[range.size() / 2] / 4.0f, 1e-20f);
        for (float& r : range) r = std::max(r, floor);
    };
    for (size_t b = 0; b < static_cast<size_t>(shape_.blocks); ++b) {
        const auto width = static_cast<size_t>(shape_.head_dims.empty() ? shape_.head_dim : shape_.head_dims[b]);
        keys.resize(width);
        queries.resize(width);
        const int32_t kv_heads = shape_.kv_heads.empty() ? shape_.n_head_kv : shape_.kv_heads[b];
        const int32_t query_heads = shape_.query_heads.empty() ? shape_.n_head : shape_.query_heads[b];
        const auto group = static_cast<size_t>(query_heads / kv_heads);
        for (size_t g = 0; g < static_cast<size_t>(kv_heads); ++g) {
            const float* km = key_max_.data() + (b * static_cast<size_t>(shape_.n_head_kv) + g) * hd;
            std::copy(km, km + width, keys.begin());
            std::fill(queries.begin(), queries.end(), 0.0f);
            for (size_t j = 0; j < group; ++j) {  // the query heads sharing KV head g
                const float* qm = query_max_.data() + (b * static_cast<size_t>(shape_.n_head) + g * group + j) * hd;
                for (size_t i = 0; i < width; ++i) queries[i] = std::max(queries[i], qm[i]);
            }
            floor_ranges(keys);
            floor_ranges(queries);
            float* t = out.data() + (b * static_cast<size_t>(shape_.n_head_kv) + g) * hd;
            for (size_t i = 0; i < width; ++i) {
                t[i] = std::clamp(std::sqrt(keys[i] / queries[i]), 1.0f / kMaxKeyScale, kMaxKeyScale);
            }
        }
    }
    return out;
}

}  // namespace liyab
