#include "core/ssm_scan.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#elif defined(LIYAB_USE_AVX2)
#include <immintrin.h>

#include "core/quant_x86.h"
// Per-function target attributes, as in quant_x86.cpp (see its header for why
// not a file-wide -mavx2): only the functions marked use AVX2 and FMA, and
// they run only when quant::x86::avx2_enabled() says so.
#if defined(__GNUC__) || defined(__clang__)
#define LIYAB_SSM_AVX2 __attribute__((target("avx2,fma")))
#else
#define LIYAB_SSM_AVX2
#endif
#endif

namespace liyab::ssm {

namespace {

// e^x = 2^n e^r with n = round(x log2 e) and r = x - n ln 2 (ln 2 split in a
// part exact in float and a correction, Cody-Waite), e^r ~ 1 + r + r^2 P(r)
// on |r| <= ln 2 / 2 with Cephes' expf polynomial. Arguments are clamped to
// [kMin, kMax], where 2^n stays a normal float: below, e^kMin (1.2e-38)
// stands in for values that only ever scale a state towards 0.
constexpr float kMin = -87.33654f;
constexpr float kMax = 88.0f;
constexpr float kLog2e = 1.44269504088896341f;
constexpr float kLn2Hi = 0.693359375f;
constexpr float kLn2Lo = -2.12194440e-4f;
constexpr float kP0 = 1.9875691500e-4f;
constexpr float kP1 = 1.3981999507e-3f;
constexpr float kP2 = 8.3334519073e-3f;
constexpr float kP3 = 4.1665795894e-2f;
constexpr float kP4 = 1.6666665459e-1f;
constexpr float kP5 = 5.0000001201e-1f;

[[maybe_unused]] float exp_scalar(float x) noexcept {
    x = std::min(std::max(x, kMin), kMax);
    const float n = std::nearbyint(x * kLog2e);
    float r = std::fma(-n, kLn2Hi, x);
    r = std::fma(-n, kLn2Lo, r);
    float p = kP0;
    p = std::fma(p, r, kP1);
    p = std::fma(p, r, kP2);
    p = std::fma(p, r, kP3);
    p = std::fma(p, r, kP4);
    p = std::fma(p, r, kP5);
    const float e = std::fma(p, r * r, r) + 1.0f;
    return e * std::bit_cast<float>(static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23);
}

[[maybe_unused]] float silu_scalar(float x) noexcept { return x / (1.0f + exp_scalar(-x)); }

#if defined(__ARM_NEON)

float32x4_t exp_neon(float32x4_t x) noexcept {
    x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(kMin)), vdupq_n_f32(kMax));
    const float32x4_t n = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(kLog2e)));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(kLn2Hi));
    r = vfmsq_f32(r, n, vdupq_n_f32(kLn2Lo));
    float32x4_t p = vdupq_n_f32(kP0);
    p = vfmaq_f32(vdupq_n_f32(kP1), p, r);
    p = vfmaq_f32(vdupq_n_f32(kP2), p, r);
    p = vfmaq_f32(vdupq_n_f32(kP3), p, r);
    p = vfmaq_f32(vdupq_n_f32(kP4), p, r);
    p = vfmaq_f32(vdupq_n_f32(kP5), p, r);
    const float32x4_t e = vaddq_f32(vfmaq_f32(r, p, vmulq_f32(r, r)), vdupq_n_f32(1.0f));
    const int32x4_t bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(e, vreinterpretq_f32_s32(bits));
}

#elif defined(LIYAB_USE_AVX2)

LIYAB_SSM_AVX2 inline __m256 exp_avx2(__m256 x) noexcept {
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(kMin)), _mm256_set1_ps(kMax));
    const __m256 n = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 r = _mm256_fnmadd_ps(n, _mm256_set1_ps(kLn2Hi), x);
    r = _mm256_fnmadd_ps(n, _mm256_set1_ps(kLn2Lo), r);
    __m256 p = _mm256_set1_ps(kP0);
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP1));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP2));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP3));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP4));
    p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP5));
    const __m256 e = _mm256_add_ps(_mm256_fmadd_ps(p, _mm256_mul_ps(r, r), r), _mm256_set1_ps(1.0f));
    const __m256i bits = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(e, _mm256_castsi256_ps(bits));
}

LIYAB_SSM_AVX2 inline float hsum_avx2(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

LIYAB_SSM_AVX2 void silu_avx2(const float* x, float* out, size_t n) noexcept {
    size_t i = 0;
    const __m256 one = _mm256_set1_ps(1.0f);
    for (; i + 8 <= n; i += 8) {
        const __m256 v = _mm256_loadu_ps(x + i);
        const __m256 e = exp_avx2(_mm256_sub_ps(_mm256_setzero_ps(), v));
        _mm256_storeu_ps(out + i, _mm256_div_ps(v, _mm256_add_ps(one, e)));
    }
    for (; i < n; ++i) out[i] = silu_scalar(x[i]);
}

LIYAB_SSM_AVX2 void causal_conv_avx2(const float* seq, const float* taps, float bias, size_t k, size_t n,
                                     float* out) noexcept {
    size_t t = 0;
    for (; t + 8 <= n; t += 8) {
        __m256 acc = _mm256_set1_ps(bias);
        for (size_t j = 0; j < k; ++j) acc = _mm256_fmadd_ps(_mm256_set1_ps(taps[j]), _mm256_loadu_ps(seq + t + j), acc);
        _mm256_storeu_ps(out + t, acc);
    }
    for (; t < n; ++t) {
        float acc = bias;
        for (size_t j = 0; j < k; ++j) acc = std::fma(taps[j], seq[t + j], acc);
        out[t] = acc;
    }
}

LIYAB_SSM_AVX2 void mamba_scan_avx2(float* h, const float* a, size_t n_state, const float* dt, const float* x,
                                    size_t stride, const float* bc, size_t bc_stride, float* y,
                                    size_t tokens) noexcept {
    const size_t vec = n_state / 8 * 8;
    for (size_t t = 0; t < tokens; ++t) {
        const float d = dt[t * stride];
        const float u = d * x[t * stride];
        const float* b = bc + t * bc_stride;
        const float* c = b + n_state;
        const __m256 dv = _mm256_set1_ps(d), uv = _mm256_set1_ps(u);
        __m256 acc = _mm256_setzero_ps();
        for (size_t j = 0; j < vec; j += 8) {
            const __m256 decay = exp_avx2(_mm256_mul_ps(dv, _mm256_loadu_ps(a + j)));
            const __m256 s = _mm256_fmadd_ps(uv, _mm256_loadu_ps(b + j), _mm256_mul_ps(decay, _mm256_loadu_ps(h + j)));
            _mm256_storeu_ps(h + j, s);
            acc = _mm256_fmadd_ps(s, _mm256_loadu_ps(c + j), acc);
        }
        float sum = hsum_avx2(acc);
        for (size_t j = vec; j < n_state; ++j) {
            h[j] = exp_scalar(d * a[j]) * h[j] + u * b[j];
            sum += h[j] * c[j];
        }
        y[t * stride] = sum;
    }
}

LIYAB_SSM_AVX2 void mamba2_step_avx2(float* h, size_t head_dim, size_t n_state, float decay, float dt, const float* x,
                                     const float* b, const float* c, float* y) noexcept {
    const __m256 dv = _mm256_set1_ps(decay);
    const size_t wide = n_state / 32 * 32;
    const size_t vec = n_state / 8 * 8;
    for (size_t p = 0; p < head_dim; ++p) {
        float* row = h + p * n_state;
        const float u = dt * x[p];
        const __m256 uv = _mm256_set1_ps(u);
        __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
        size_t j = 0;
        for (; j < wide; j += 32) {
            const __m256 s0 = _mm256_fmadd_ps(uv, _mm256_loadu_ps(b + j), _mm256_mul_ps(dv, _mm256_loadu_ps(row + j)));
            const __m256 s1 =
                _mm256_fmadd_ps(uv, _mm256_loadu_ps(b + j + 8), _mm256_mul_ps(dv, _mm256_loadu_ps(row + j + 8)));
            const __m256 s2 =
                _mm256_fmadd_ps(uv, _mm256_loadu_ps(b + j + 16), _mm256_mul_ps(dv, _mm256_loadu_ps(row + j + 16)));
            const __m256 s3 =
                _mm256_fmadd_ps(uv, _mm256_loadu_ps(b + j + 24), _mm256_mul_ps(dv, _mm256_loadu_ps(row + j + 24)));
            _mm256_storeu_ps(row + j, s0);
            _mm256_storeu_ps(row + j + 8, s1);
            _mm256_storeu_ps(row + j + 16, s2);
            _mm256_storeu_ps(row + j + 24, s3);
            a0 = _mm256_fmadd_ps(s0, _mm256_loadu_ps(c + j), a0);
            a1 = _mm256_fmadd_ps(s1, _mm256_loadu_ps(c + j + 8), a1);
            a2 = _mm256_fmadd_ps(s2, _mm256_loadu_ps(c + j + 16), a2);
            a3 = _mm256_fmadd_ps(s3, _mm256_loadu_ps(c + j + 24), a3);
        }
        for (; j < vec; j += 8) {
            const __m256 s = _mm256_fmadd_ps(uv, _mm256_loadu_ps(b + j), _mm256_mul_ps(dv, _mm256_loadu_ps(row + j)));
            _mm256_storeu_ps(row + j, s);
            a0 = _mm256_fmadd_ps(s, _mm256_loadu_ps(c + j), a0);
        }
        float sum = hsum_avx2(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        for (; j < n_state; ++j) {
            row[j] = decay * row[j] + u * b[j];
            sum += row[j] * c[j];
        }
        y[p] = sum;
    }
}

#endif

}  // namespace

void silu(const float* x, float* out, size_t n) noexcept {
    size_t i = 0;
#if defined(__ARM_NEON)
    const float32x4_t one = vdupq_n_f32(1.0f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(out + i, vdivq_f32(v, vaddq_f32(one, exp_neon(vnegq_f32(v)))));
    }
#elif defined(LIYAB_USE_AVX2)
    if (quant::x86::avx2_enabled()) {
        silu_avx2(x, out, n);
        return;
    }
#endif
    for (; i < n; ++i) out[i] = silu_scalar(x[i]);
}

void causal_conv(const float* seq, const float* taps, float bias, size_t k, size_t n, float* out) noexcept {
    size_t t = 0;
#if defined(__ARM_NEON)
    for (; t + 4 <= n; t += 4) {
        float32x4_t acc = vdupq_n_f32(bias);
        for (size_t j = 0; j < k; ++j) acc = vfmaq_n_f32(acc, vld1q_f32(seq + t + j), taps[j]);
        vst1q_f32(out + t, acc);
    }
#elif defined(LIYAB_USE_AVX2)
    if (quant::x86::avx2_enabled()) {
        causal_conv_avx2(seq, taps, bias, k, n, out);
        return;
    }
#endif
    for (; t < n; ++t) {
        float acc = bias;
        for (size_t j = 0; j < k; ++j) acc = std::fma(taps[j], seq[t + j], acc);
        out[t] = acc;
    }
}

void mamba_scan(float* h, const float* a, size_t n_state, const float* dt, const float* x, size_t stride,
                const float* bc, size_t bc_stride, float* y, size_t tokens) noexcept {
#if defined(__ARM_NEON)
    const size_t vec = n_state / 4 * 4;
    for (size_t t = 0; t < tokens; ++t) {
        const float d = dt[t * stride];
        const float u = d * x[t * stride];
        const float* b = bc + t * bc_stride;
        const float* c = b + n_state;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (size_t j = 0; j < vec; j += 4) {
            const float32x4_t decay = exp_neon(vmulq_n_f32(vld1q_f32(a + j), d));
            const float32x4_t s = vfmaq_n_f32(vmulq_f32(decay, vld1q_f32(h + j)), vld1q_f32(b + j), u);
            vst1q_f32(h + j, s);
            acc = vfmaq_f32(acc, s, vld1q_f32(c + j));
        }
        float sum = vaddvq_f32(acc);
        for (size_t j = vec; j < n_state; ++j) {
            h[j] = exp_scalar(d * a[j]) * h[j] + u * b[j];
            sum += h[j] * c[j];
        }
        y[t * stride] = sum;
    }
    return;
#elif defined(LIYAB_USE_AVX2)
    if (quant::x86::avx2_enabled()) {
        mamba_scan_avx2(h, a, n_state, dt, x, stride, bc, bc_stride, y, tokens);
        return;
    }
#endif
    for (size_t t = 0; t < tokens; ++t) {
        const float d = dt[t * stride];
        const float u = d * x[t * stride];
        const float* b = bc + t * bc_stride;
        const float* c = b + n_state;
        float sum = 0.0f;
        for (size_t j = 0; j < n_state; ++j) {
            h[j] = exp_scalar(d * a[j]) * h[j] + u * b[j];
            sum += h[j] * c[j];
        }
        y[t * stride] = sum;
    }
}

void mamba2_step(float* h, size_t head_dim, size_t n_state, float decay, float dt, const float* x, const float* b,
                 const float* c, float* y) noexcept {
#if defined(__ARM_NEON)
    // Four accumulators break the dependency chain of the dot product with
    // C; their order is fixed, so the sum does not depend on anything else.
    const size_t wide = n_state / 16 * 16;
    const size_t vec = n_state / 4 * 4;
    const float32x4_t dv = vdupq_n_f32(decay);
    for (size_t p = 0; p < head_dim; ++p) {
        float* row = h + p * n_state;
        const float u = dt * x[p];
        float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
        size_t j = 0;
        for (; j < wide; j += 16) {
            const float32x4_t s0 = vfmaq_n_f32(vmulq_f32(dv, vld1q_f32(row + j)), vld1q_f32(b + j), u);
            const float32x4_t s1 = vfmaq_n_f32(vmulq_f32(dv, vld1q_f32(row + j + 4)), vld1q_f32(b + j + 4), u);
            const float32x4_t s2 = vfmaq_n_f32(vmulq_f32(dv, vld1q_f32(row + j + 8)), vld1q_f32(b + j + 8), u);
            const float32x4_t s3 = vfmaq_n_f32(vmulq_f32(dv, vld1q_f32(row + j + 12)), vld1q_f32(b + j + 12), u);
            vst1q_f32(row + j, s0);
            vst1q_f32(row + j + 4, s1);
            vst1q_f32(row + j + 8, s2);
            vst1q_f32(row + j + 12, s3);
            a0 = vfmaq_f32(a0, s0, vld1q_f32(c + j));
            a1 = vfmaq_f32(a1, s1, vld1q_f32(c + j + 4));
            a2 = vfmaq_f32(a2, s2, vld1q_f32(c + j + 8));
            a3 = vfmaq_f32(a3, s3, vld1q_f32(c + j + 12));
        }
        for (; j < vec; j += 4) {
            const float32x4_t s = vfmaq_n_f32(vmulq_f32(dv, vld1q_f32(row + j)), vld1q_f32(b + j), u);
            vst1q_f32(row + j, s);
            a0 = vfmaq_f32(a0, s, vld1q_f32(c + j));
        }
        float sum = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
        for (; j < n_state; ++j) {
            row[j] = decay * row[j] + u * b[j];
            sum += row[j] * c[j];
        }
        y[p] = sum;
    }
    return;
#elif defined(LIYAB_USE_AVX2)
    if (quant::x86::avx2_enabled()) {
        mamba2_step_avx2(h, head_dim, n_state, decay, dt, x, b, c, y);
        return;
    }
#endif
    for (size_t p = 0; p < head_dim; ++p) {
        float* row = h + p * n_state;
        const float u = dt * x[p];
        float sum = 0.0f;
        for (size_t j = 0; j < n_state; ++j) {
            row[j] = decay * row[j] + u * b[j];
            sum += row[j] * c[j];
        }
        y[p] = sum;
    }
}

}  // namespace liyab::ssm
