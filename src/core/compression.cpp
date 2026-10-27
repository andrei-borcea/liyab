#include "core/compression.h"

#include <algorithm>
#include <cmath>

#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(LIYAB_USE_AVX2)
#include <immintrin.h>
#include "core/quant_x86.h"
#endif

namespace liyab {
namespace {

// Channelwise arithmetic after scalar exp: there is no reduction across
// lanes. Keep multiplication and addition separate on every backend.
#if defined(LIYAB_USE_AVX2)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma,f16c")))
#endif
size_t accumulate_avx(float* z, float* u, const float* a, const float* b, const float* kv, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i), bv = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(z + i, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(z + i), av), bv));
        _mm256_storeu_ps(u + i, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(u + i), av),
                                            _mm256_mul_ps(_mm256_loadu_ps(kv + i), bv)));
    }
    return i;
}
#endif

void accumulate(float* s, const float* kv, const float* gate, const float* bias, size_t n, bool first,
                float* a, float* b) noexcept {
    float* z = s + n;
    float* u = z + n;
    for (size_t i = 0; i < n; ++i) {
        const float g = gate[i] + bias[i];
        const float m = first ? g : std::max(s[i], g);
        a[i] = first ? 0.0f : std::exp(s[i] - m);
        b[i] = std::exp(g - m);
        s[i] = m;
    }
    size_t i = 0;
#if defined(LIYAB_USE_AVX2)
    if (quant::x86::avx2_enabled()) i = accumulate_avx(z, u, a, b, kv, n);
#elif defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
    for (; i + 4 <= n; i += 4) {
        const float32x4_t av = vld1q_f32(a + i), bv = vld1q_f32(b + i);
        vst1q_f32(z + i, vaddq_f32(vmulq_f32(vld1q_f32(z + i), av), bv));
        vst1q_f32(u + i, vaddq_f32(vmulq_f32(vld1q_f32(u + i), av), vmulq_f32(vld1q_f32(kv + i), bv)));
    }
#endif
    for (; i < n; ++i) {
        z[i] = z[i] * a[i] + b[i];
        u[i] = u[i] * a[i] + kv[i] * b[i];
    }
}

}  // namespace

bool CompressionPlan::step(std::span<float> state, const float* kv, const float* gate, const float* bias,
                            int32_t pos, float* out, std::span<float> scratch) const noexcept {
    const auto n = static_cast<size_t>(dim);
    const bool first = pos % stride == 0;
    float* s = state.data();
    // Clear inactive running sums on a new window. In particular, multiplying
    // a prior NaN/inf by zero must not leak it into the next window.
    if (first) std::fill_n(s, (overlap ? 6 : 3) * n, 0.0f);
    accumulate(s, kv, gate, bias, n, first, scratch.data(), scratch.data() + n);
    if (overlap) {
        accumulate(s + 3 * n, kv + n, gate + n, bias + n, n, first, scratch.data(), scratch.data() + n);
    }
    if ((pos + 1) % stride != 0) return false;
    if (!overlap) {
        for (size_t i = 0; i < n; ++i) out[i] = s[2 * n + i] / s[n + i];
    } else {
        const float* current = s + 3 * n;
        const float* previous = s + 6 * n;
        for (size_t i = 0; i < n; ++i) {
            const float m = pos < stride ? current[i] : std::max(previous[i], current[i]);
            const float a = pos < stride ? 0.0f : std::exp(previous[i] - m);
            const float b = std::exp(current[i] - m);
            out[i] = (previous[2 * n + i] * a + current[2 * n + i] * b) /
                     (previous[n + i] * a + current[n + i] * b);
        }
        std::copy_n(s, 3 * n, s + 6 * n);
    }
    return true;
}

}  // namespace liyab
