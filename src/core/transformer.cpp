#include "core/transformer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <istream>
#include <limits>
#include <numbers>
#include <ostream>
#include <random>

#include "core/log.h"
#include "core/moe_routing.h"
#include "core/portable_math.h"
#include "core/quant.h"
#include "core/ssm_scan.h"
#include "core/thread_pool.h"

#if defined(LIYAB_USE_AVX2)
#include <immintrin.h>
#include "core/quant_x86.h"
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(LIYAB_USE_AVX2)
#include <immintrin.h>

#include "core/quant_x86.h"
#endif

namespace liyab {

namespace {

// The factor RMSNorm multiplies x by before its weight: 1 / sqrt(mean(x^2) + eps).
float rms_scale(const float* x, int32_t n, float eps) {
    double ss = 0.0;
    for (int32_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    return static_cast<float>(1.0 / std::sqrt(ss / n + eps));
}

void rmsnorm(const float* x, const float* w, float* out, int32_t n, float eps) {
    const float scale = rms_scale(x, n, eps);
    for (int32_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

#if defined(LIYAB_USE_AVX2)
// Keep the baseline binary runnable without AVX2, as in quant_x86.cpp.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma,f16c")))
#endif
int32_t layernorm_affine_avx2(const float* x, const float* w, const float* b, float* out, int32_t n,
                              float mean, float scale) {
    const __m256 m = _mm256_set1_ps(mean), s = _mm256_set1_ps(scale);
    int32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x + i), m), s);
        if (w != nullptr) v = _mm256_mul_ps(v, _mm256_loadu_ps(w + i));
        if (b != nullptr) v = _mm256_add_ps(v, _mm256_loadu_ps(b + i));
        _mm256_storeu_ps(out + i, v);
    }
    return i;
}
#endif

// LayerNorm: (x - mean) / sqrt(var + eps), times w and plus b where given.
// Mean and variance are accumulated in double, as rms_scale() accumulates,
// so the result does not depend on how a compiler orders a float sum.
void layernorm(const float* x, const float* w, const float* b, float* out, int32_t n, float eps) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; ++i) sum += x[i];
    const double mean = sum / n;
    double ss = 0.0;
    for (int32_t j = 0; j < n; ++j) {
        const double dev = x[j] - mean;
        ss += dev * dev;
    }
    const auto m = static_cast<float>(mean);
    const auto scale = static_cast<float>(1.0 / std::sqrt(ss / n + eps));
    int32_t i = 0;
#if defined(LIYAB_USE_AVX2)
    if (quant::x86::avx2_enabled()) i = layernorm_affine_avx2(x, w, b, out, n, m, scale);
#elif defined(__ARM_NEON)
    const float32x4_t mv = vdupq_n_f32(m), sv = vdupq_n_f32(scale);
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vmulq_f32(vsubq_f32(vld1q_f32(x + i), mv), sv);
        if (w != nullptr) v = vmulq_f32(v, vld1q_f32(w + i));
        if (b != nullptr) v = vaddq_f32(v, vld1q_f32(b + i));
        vst1q_f32(out + i, v);
    }
#endif
    for (; i < n; ++i) {
        float v = (x[i] - m) * scale;
        if (w != nullptr) v *= w[i];
        if (b != nullptr) v += b[i];
        out[i] = v;
    }
}

// An optional per-element vector as a pointer (nullptr when absent).
const float* opt(const std::vector<float>& v) { return v.empty() ? nullptr : v.data(); }

// Every projection adds its optional bias to each output row before the
// activation, output scaling or residual add that consumes the projection.
void add_bias(const std::vector<float>& bias, float* y, size_t n) {
    if (bias.empty()) return;
    for (size_t t = 0; t < n; ++t) {
        float* row = y + t * bias.size();
        for (size_t i = 0; i < bias.size(); ++i) row[i] += bias[i];
    }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// GELU with the tanh approximation (PyTorch's gelu(approximate="tanh")):
// 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))).
float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    // The C library's tanh (not portable_tanh): Apple's and Android's agree on
    // every golden model's GELU, and switching would change pinned hashes.
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}



// Exact GELU, x Phi(x) = 0.5 x (1 + erf(x / sqrt 2)). Evaluated in double
// and rounded once: the C libraries' erff may disagree in the last bit
// between platforms (the Mac and the phone share golden hashes), their
// double erf rounds to the same float but for one argument in hundreds of
// millions.
float gelu_erf(float x) {
    const double xd = x;
    return static_cast<float>(0.5 * xd * (1.0 + std::erf(xd * 0.70710678118654752440)));
}

// xIELU (Apertus), with its block's scalars; expm1 in double for the same
// reason as gelu_erf().
float xielu(float x, const XIeluScalars& p) {
    if (x > 0.0f) return p.alpha_p * x * x + p.beta * x;
    const auto em1 = static_cast<float>(std::expm1(static_cast<double>(std::min(x, p.eps))));
    return (em1 - x) * p.alpha_n + p.beta * x;
}

#if defined(LIYAB_USE_AVX2)
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2,fma,f16c")))
#endif
size_t relu_avx2(const float* x, const float* up, float* out, size_t n, bool squared) {
    const __m256 zero = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256 input = _mm256_loadu_ps(x + i);
        // Comparison/select preserves std::max(x, 0)'s signed zero and NaN.
        __m256 value = _mm256_blendv_ps(input, zero, _mm256_cmp_ps(input, zero, _CMP_LT_OQ));
        if (squared) value = _mm256_mul_ps(value, value);
        if (up != nullptr) value = _mm256_mul_ps(value, _mm256_loadu_ps(up + i));
        _mm256_storeu_ps(out + i, value);
    }
    return i;
}
#endif

// ReLU / ReLU², optionally times an up projection. out may equal x.
void relu_activation(const float* x, const float* up, float* out, size_t n, bool squared) {
    size_t i = 0;
#if defined(LIYAB_USE_AVX2)
    if (quant::x86::avx2_enabled()) i = relu_avx2(x, up, out, n, squared);
#elif defined(__ARM_NEON)
    const float32x4_t zero = vdupq_n_f32(0.0f);
    for (; i + 4 <= n; i += 4) {
        const float32x4_t input = vld1q_f32(x + i);
        float32x4_t value = vbslq_f32(vcltq_f32(input, zero), zero, input);
        if (squared) value = vmulq_f32(value, value);
        if (up != nullptr) value = vmulq_f32(value, vld1q_f32(up + i));
        vst1q_f32(out + i, value);
    }
#endif
    for (; i < n; ++i) {
        float value = std::max(x[i], 0.0f);
        if (squared) value *= value;
        out[i] = up != nullptr ? value * up[i] : value;
    }
}

// An ungated FFN's activation, in place over n values: act(x).
void ungated_activation(FfnActivation act, const XIeluScalars* scalars, float* x, size_t n) {
    switch (act) {
        case FfnActivation::SiLUInputClamp:
        case FfnActivation::SiLU:
            for (size_t i = 0; i < n; ++i) x[i] = silu(x[i]);
            return;
        case FfnActivation::Gelu:
            for (size_t i = 0; i < n; ++i) x[i] = gelu_erf(x[i]);
            return;
        case FfnActivation::Relu:
        case FfnActivation::ReluSquared:
            relu_activation(x, nullptr, x, n, act == FfnActivation::ReluSquared);
            return;
        case FfnActivation::XIelu:
            for (size_t i = 0; i < n; ++i) x[i] = xielu(x[i], *scalars);
            return;
        default:  // GeluTanh
            break;
    }
    for (size_t i = 0; i < n; ++i) x[i] = gelu_tanh(x[i]);
}
// The gated FFN's elementwise step: out[i] = act(gate[i]) * up[i] (out may be gate).
void gated_activation(FfnActivation act, const float* gate, const float* up, float* out, size_t n, float limit = 0.0f) {
    if (limit > 0.0f && act == FfnActivation::SiLUInputClamp) {
        for (size_t i = 0; i < n; ++i) out[i] = silu(std::min(gate[i], limit)) * std::clamp(up[i], -limit, limit);
        return;
    }
    if (limit > 0.0f) {  // validated SiLU: clamp the activated gate, unlike SwigluOai's input clamp
        for (size_t i = 0; i < n; ++i) out[i] = std::min(silu(gate[i]), limit) * std::clamp(up[i], -limit, limit);
        return;
    }
    if (act == FfnActivation::GeluTanh) {
        for (size_t i = 0; i < n; ++i) out[i] = gelu_tanh(gate[i]) * up[i];
        return;
    }
    if (act == FfnActivation::Relu || act == FfnActivation::ReluSquared) {
        relu_activation(gate, up, out, n, act == FfnActivation::ReluSquared);
        return;
    }
    if (act == FfnActivation::Gelu) {
        for (size_t i = 0; i < n; ++i) {
            float g = gate[i];
            ungated_activation(act, nullptr, &g, 1);
            out[i] = g * up[i];
        }
        return;
    }
    if (act == FfnActivation::SwigluOai) {
        // As the reference computes it: glu = g * sigmoid(1.702 g), then (u + 1) * glu.
        constexpr float kAlpha = 1.702f, kLimit = 7.0f;
        for (size_t i = 0; i < n; ++i) {
            const float g = std::min(gate[i], kLimit);
            const float u = std::clamp(up[i], -kLimit, kLimit);
            const float glu = g * (1.0f / (1.0f + std::exp(-(g * kAlpha))));
            out[i] = (u + 1.0f) * glu;
        }
        return;
    }
    for (size_t i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }  // ggml's threshold

// Non-softmax router scores; softplus uses the upstream threshold to avoid
// overflow, and sqrt keeps these positive without a normalization over E.
float positive_router_score(MoeGating gating, float x) {
    if (gating == MoeGating::SqrtSoftplus) return std::sqrt(softplus(x));
    return sigmoid(x);
}

// x / sqrt(sum(x^2) + eps): the L2 normalization of DeltaNet queries and keys.
void l2norm(float* x, int32_t n, float eps) {
    double ss = 0.0;
    for (int32_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const auto scale = static_cast<float>(1.0 / std::sqrt(ss + eps));
    for (int32_t i = 0; i < n; ++i) x[i] *= scale;
}

// One token of the gated delta rule for one value head, S [hd x hd] row j
// mapping keys to value component j:
//   S = decay * S;  S += strength * (v - S k) k^T;  o = (S q) * q_scale
// Rows are independent: each is decayed and read against k, corrected, then
// read against q, in two passes over its hd values. On NEON four
// accumulators break the dependency chains of the dot products (a fixed
// order, so results do not depend on the thread count); the scalar loop
// elsewhere keeps one.
void delta_rule_step(float* S, const float* q, const float* k, const float* v, float decay, float strength,
                     float q_scale, float* o, size_t hd) {
    for (size_t j = 0; j < hd; ++j) {
        float* row = S + j * hd;
        size_t i = 0;
        float sk = 0.0f;
#if defined(__ARM_NEON)
        if (hd % 16 == 0) {
            const float32x4_t dv = vdupq_n_f32(decay);
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
            for (; i < hd; i += 16) {
                float32x4_t r0 = vmulq_f32(vld1q_f32(row + i), dv), r1 = vmulq_f32(vld1q_f32(row + i + 4), dv);
                float32x4_t r2 = vmulq_f32(vld1q_f32(row + i + 8), dv), r3 = vmulq_f32(vld1q_f32(row + i + 12), dv);
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(k + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(k + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(k + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(k + i + 12));
            }
            sk = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            const float32x4_t delta = vdupq_n_f32((v[j] - sk) * strength);
            a0 = a1 = a2 = a3 = vdupq_n_f32(0.0f);
            for (i = 0; i < hd; i += 16) {
                const float32x4_t r0 = vfmaq_f32(vld1q_f32(row + i), vld1q_f32(k + i), delta);
                const float32x4_t r1 = vfmaq_f32(vld1q_f32(row + i + 4), vld1q_f32(k + i + 4), delta);
                const float32x4_t r2 = vfmaq_f32(vld1q_f32(row + i + 8), vld1q_f32(k + i + 8), delta);
                const float32x4_t r3 = vfmaq_f32(vld1q_f32(row + i + 12), vld1q_f32(k + i + 12), delta);
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(q + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(q + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(q + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(q + i + 12));
            }
            o[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * q_scale;
            continue;
        }
#endif
        for (i = 0; i < hd; ++i) {
            row[i] *= decay;
            sk += row[i] * k[i];
        }
        const float delta = (v[j] - sk) * strength;
        float oq = 0.0f;
        for (i = 0; i < hd; ++i) {
            row[i] += k[i] * delta;
            oq += row[i] * q[i];
        }
        o[j] = oq * q_scale;
    }
}

// Kimi Delta Attention's epsilon of the L2 norms of its queries and keys: the
// reference kernels' own constant (FLA's l2norm), not the model's RMSNorm
// epsilon (1e-5 on Kimi-Linear).
constexpr float kKdaL2Eps = 1e-6f;

// One token of Kimi Delta Attention's gated delta rule for one head, S
// [hd x hd] row j mapping keys to value component j:
//   S = S diag(decay);  S += strength * (v - S k) k^T;  o = (S q) * q_scale
// delta_rule_step() with a decay per key channel instead of one per head.
// Rows are independent: each is decayed and read against k, corrected, then
// read against q, in two passes over its hd values. The SIMD paths keep a
// fixed set of accumulators per row (results do not depend on the thread
// count); each kernel set (NEON, AVX2, scalar) rounds its own way.
void kda_rule_step_scalar(float* S, const float* q, const float* k, const float* v, const float* decay,
                          float strength, float q_scale, float* o, size_t hd, size_t row_count) {
    for (size_t j = 0; j < row_count; ++j) {
        float* row = S + j * hd;
        float sk = 0.0f;
        for (size_t i = 0; i < hd; ++i) {
            row[i] *= decay[i];
            sk += row[i] * k[i];
        }
        const float delta = (v[j] - sk) * strength;
        float oq = 0.0f;
        for (size_t i = 0; i < hd; ++i) {
            row[i] += k[i] * delta;
            oq += row[i] * q[i];
        }
        o[j] = oq * q_scale;
    }
}

#if defined(LIYAB_USE_AVX2)
#if defined(__GNUC__) || defined(__clang__)
#define LIYAB_KDA_AVX2 __attribute__((target("avx2,fma")))
#else
#define LIYAB_KDA_AVX2
#endif
// Horizontal sum of 8 lanes in a fixed order.
LIYAB_KDA_AVX2 inline float kda_hsum(__m256 v) {
    const __m128 lo = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 pairs = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    return _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
}
// AVX2 + FMA, two accumulators of 8 lanes per dot (hd a multiple of 16).
LIYAB_KDA_AVX2 void kda_rule_step_avx2(float* S, const float* q, const float* k, const float* v,
                                       const float* decay, float strength, float q_scale, float* o, size_t hd, size_t row_count) {
    for (size_t j = 0; j < row_count; ++j) {
        float* row = S + j * hd;
        __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (size_t i = 0; i < hd; i += 16) {
            const __m256 r0 = _mm256_mul_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(decay + i));
            const __m256 r1 = _mm256_mul_ps(_mm256_loadu_ps(row + i + 8), _mm256_loadu_ps(decay + i + 8));
            _mm256_storeu_ps(row + i, r0);
            _mm256_storeu_ps(row + i + 8, r1);
            a0 = _mm256_fmadd_ps(r0, _mm256_loadu_ps(k + i), a0);
            a1 = _mm256_fmadd_ps(r1, _mm256_loadu_ps(k + i + 8), a1);
        }
        const float sk = kda_hsum(_mm256_add_ps(a0, a1));
        const __m256 delta = _mm256_set1_ps((v[j] - sk) * strength);
        a0 = _mm256_setzero_ps();
        a1 = _mm256_setzero_ps();
        for (size_t i = 0; i < hd; i += 16) {
            const __m256 r0 = _mm256_fmadd_ps(_mm256_loadu_ps(k + i), delta, _mm256_loadu_ps(row + i));
            const __m256 r1 = _mm256_fmadd_ps(_mm256_loadu_ps(k + i + 8), delta, _mm256_loadu_ps(row + i + 8));
            _mm256_storeu_ps(row + i, r0);
            _mm256_storeu_ps(row + i + 8, r1);
            a0 = _mm256_fmadd_ps(r0, _mm256_loadu_ps(q + i), a0);
            a1 = _mm256_fmadd_ps(r1, _mm256_loadu_ps(q + i + 8), a1);
        }
        o[j] = kda_hsum(_mm256_add_ps(a0, a1)) * q_scale;
    }
}
#endif

void kda_rule_step(float* S, const float* q, const float* k, const float* v, const float* decay, float strength,
                   float q_scale, float* o, size_t hd, size_t row_count) {
#if defined(__ARM_NEON)
    if (hd % 16 == 0) {
        for (size_t j = 0; j < row_count; ++j) {
            float* row = S + j * hd;
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
            for (size_t i = 0; i < hd; i += 16) {
                const float32x4_t r0 = vmulq_f32(vld1q_f32(row + i), vld1q_f32(decay + i));
                const float32x4_t r1 = vmulq_f32(vld1q_f32(row + i + 4), vld1q_f32(decay + i + 4));
                const float32x4_t r2 = vmulq_f32(vld1q_f32(row + i + 8), vld1q_f32(decay + i + 8));
                const float32x4_t r3 = vmulq_f32(vld1q_f32(row + i + 12), vld1q_f32(decay + i + 12));
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(k + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(k + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(k + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(k + i + 12));
            }
            const float sk = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            const float32x4_t delta = vdupq_n_f32((v[j] - sk) * strength);
            a0 = a1 = a2 = a3 = vdupq_n_f32(0.0f);
            for (size_t i = 0; i < hd; i += 16) {
                const float32x4_t r0 = vfmaq_f32(vld1q_f32(row + i), vld1q_f32(k + i), delta);
                const float32x4_t r1 = vfmaq_f32(vld1q_f32(row + i + 4), vld1q_f32(k + i + 4), delta);
                const float32x4_t r2 = vfmaq_f32(vld1q_f32(row + i + 8), vld1q_f32(k + i + 8), delta);
                const float32x4_t r3 = vfmaq_f32(vld1q_f32(row + i + 12), vld1q_f32(k + i + 12), delta);
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(q + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(q + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(q + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(q + i + 12));
            }
            o[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * q_scale;
        }
        return;
    }
#elif defined(LIYAB_USE_AVX2)
    if (hd % 16 == 0 && quant::x86::avx2_enabled()) {
        kda_rule_step_avx2(S, q, k, v, decay, strength, q_scale, o, hd, row_count);
        return;
    }
#endif
    kda_rule_step_scalar(S, q, k, v, decay, strength, q_scale, o, hd, row_count);
}

// Adds the wall time of its scope to `total_ms` (Transformer::PhaseTimes).
class PhaseTimer {
public:
    explicit PhaseTimer(double& total_ms) : total_ms_(total_ms), start_(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() { stop(); }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
    // Ends the measurement early (later calls and the destructor add nothing).
    void stop() {
        if (stopped_) return;
        stopped_ = true;
        total_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    double& total_ms_;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};

// Slice `e` of a stacked tensor [cols, rows, n] as a 2-D view: an expert of
// a MoE block, or a head of latent attention's per-head matrices.
TensorView expert_slice(const TensorView& t, int64_t e) {
    TensorView v = t;
    v.n_dims = 2;
    v.ne = {t.ne[0], t.ne[1], 1, 1};
    v.nbytes = t.row_bytes() * static_cast<size_t>(t.ne[1]);
    v.data = t.data + static_cast<size_t>(e) * v.nbytes;
    v.file_offset = t.file_offset + static_cast<uint64_t>(e) * v.nbytes;
    return v;
}

// Positions a query at `pos` of a block with `window` (0: none) sees through
// the cache: [sink_begin, sink_end) ∪ [first, pos].
struct Seen {
    int32_t sink_begin, sink_end, first;
};
Seen seen_by(const KvCache& kv, int32_t pos, int32_t window) {
    const KvCache::Visible vis = kv.visible(pos);
    Seen s{0, vis.sink_end, vis.first};
    if (window > 0) {
        const int32_t lo = pos - window + 1;
        s.first = std::max(s.first, lo);
        s.sink_begin = std::clamp(lo, 0, s.sink_end);
    }
    return s;
}

// Latent attention's expanded form, the GEMM that expands cached latents
// into per-head keys and values: columns are taken kExpandColumns at a time
// from weights packed per column tile ([tile][kv_rank][kExpandColumns]), so
// one tile's strip (kv_rank x 16 floats, 32 KiB on DeepSeek-V2-Lite) stays in
// L1 while every latent row passes through it, and kExpandRows rows share
// each strip load. Each output is one float sum over the latent channels in
// ascending order, from zero, whatever the rows around it: the vector lanes
// run separate columns, never parts of one sum, so the bits of a position's
// keys and values do not depend on how many positions a pass expands.
constexpr int32_t kExpandColumns = 16;
constexpr int32_t kExpandRows = 4;

// Rows [0, m) of `a` (m x k, row-major) times the strip `wp` (k x
// kExpandColumns): row j's column c lands at out[j * out_stride + offset[c]]
// for the `columns` (<= kExpandColumns) columns that exist.
void expand_strip(const float* a, int64_t m, int64_t k, const float* wp, float* out, size_t out_stride,
                  const size_t* offset, int32_t columns) {
    for (int64_t j0 = 0; j0 < m; j0 += kExpandRows) {
        // A last block of fewer rows repeats the last one (computed, not stored).
        const float* rows[kExpandRows];
        for (int64_t i = 0; i < kExpandRows; ++i) rows[i] = a + std::min(j0 + i, m - 1) * k;
        float acc[kExpandRows][kExpandColumns] = {};
        for (int64_t r = 0; r < k; ++r) {
            const float* w = wp + r * kExpandColumns;
            for (int32_t i = 0; i < kExpandRows; ++i) {
                const float x = rows[i][r];
                for (int32_t c = 0; c < kExpandColumns; ++c) acc[i][c] += x * w[c];
            }
        }
        const int64_t stored = std::min<int64_t>(kExpandRows, m - j0);
        for (int64_t i = 0; i < stored; ++i) {
            float* dst = out + static_cast<size_t>(j0 + i) * out_stride;
            for (int32_t c = 0; c < columns; ++c) dst[offset[c]] = acc[i][c];
        }
    }
}

// The expanded form's attention kernels. Each score is one dot product over
// kScoreLanes interleaved partial sums reduced in a fixed tree, and each
// weighted value one multiply-add chain over the positions in ascending
// order, whatever the queries and rows computed alongside: blocks of
// kScoreQueries tokens and kScoreKeys rows share every load, and a token's
// bits do not depend on which tokens share its block.
constexpr int32_t kScoreQueries = 4;
constexpr int32_t kScoreKeys = 2;
constexpr int32_t kScoreLanes = 8;
constexpr int32_t kValueColumns = 16;
// A task attends kAttendTokens tokens of one head (fewer when the pass has
// too few tasks for the threads), kAttendRows rows of keys and values at a time.
constexpr size_t kAttendTokens = 32;
constexpr size_t kAttendRows = 64;

// out[t][r] = q[t] . k[r] over n channels, for kScoreQueries queries and kScoreKeys rows.
void score_block(const float* const* q, const float* const* k, size_t n, float (*out)[kScoreKeys]) {
#if defined(__ARM_NEON)
    // The same sums as the portable loop below, lanes 0-3 and 4-7 in two
    // vectors (compilers leave the portable loop scalar: 64 accumulators).
    if (n % kScoreLanes == 0) {
        float32x4_t lo[kScoreQueries][kScoreKeys], hi[kScoreQueries][kScoreKeys];
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) lo[t][r] = hi[t][r] = vdupq_n_f32(0.0f);
        }
        for (size_t i = 0; i < n; i += kScoreLanes) {
            float32x4_t k_lo[kScoreKeys], k_hi[kScoreKeys];
            for (int32_t r = 0; r < kScoreKeys; ++r) {
                k_lo[r] = vld1q_f32(k[r] + i);
                k_hi[r] = vld1q_f32(k[r] + i + 4);
            }
            for (int32_t t = 0; t < kScoreQueries; ++t) {
                const float32x4_t q_lo = vld1q_f32(q[t] + i);
                const float32x4_t q_hi = vld1q_f32(q[t] + i + 4);
                for (int32_t r = 0; r < kScoreKeys; ++r) {
                    lo[t][r] = vfmaq_f32(lo[t][r], q_lo, k_lo[r]);
                    hi[t][r] = vfmaq_f32(hi[t][r], q_hi, k_hi[r]);
                }
            }
        }
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) {
                const float32x4_t s = vaddq_f32(lo[t][r], hi[t][r]);  // a[l] + a[l + 4]
                out[t][r] = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1)) + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
            }
        }
        return;
    }
#endif
    float acc[kScoreQueries][kScoreKeys][kScoreLanes] = {};
    size_t i = 0;
    for (; i + kScoreLanes <= n; i += kScoreLanes) {
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) {
                for (int32_t l = 0; l < kScoreLanes; ++l) acc[t][r][l] += q[t][i + l] * k[r][i + l];
            }
        }
    }
    for (int32_t l = 0; i < n; ++i, ++l) {
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) acc[t][r][l] += q[t][i] * k[r][i];
        }
    }
    for (int32_t t = 0; t < kScoreQueries; ++t) {
        for (int32_t r = 0; r < kScoreKeys; ++r) {
            const float* a = acc[t][r];
            out[t][r] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
        }
    }
}

// o[t][d] += sum over rows i in [a, b) of p[t][i] * v[i * stride + d], d in
// [0, width), for T tokens: kValueColumns columns of every token held in
// registers while the rows go by.
template <int32_t T>
void weigh_values(float* const* o, const float* const* p, const float* v, size_t stride, size_t a, size_t b,
                  size_t width) {
    if (a >= b) return;
#if defined(__ARM_NEON)
    // The same multiply-adds as the portable loop below, in vectors (which
    // compilers do not reliably make of it).
    if (width % kValueColumns == 0) {
        for (size_t d0 = 0; d0 < width; d0 += kValueColumns) {
            float32x4_t acc[T][4];
            for (int32_t t = 0; t < T; ++t) {
                for (int32_t c = 0; c < 4; ++c) acc[t][c] = vld1q_f32(o[t] + d0 + 4 * c);
            }
            for (size_t i = a; i < b; ++i) {
                const float* row = v + i * stride + d0;
                const float32x4_t r0 = vld1q_f32(row), r1 = vld1q_f32(row + 4);
                const float32x4_t r2 = vld1q_f32(row + 8), r3 = vld1q_f32(row + 12);
                for (int32_t t = 0; t < T; ++t) {
                    const float32x4_t pi = vdupq_n_f32(p[t][i]);
                    acc[t][0] = vfmaq_f32(acc[t][0], pi, r0);
                    acc[t][1] = vfmaq_f32(acc[t][1], pi, r1);
                    acc[t][2] = vfmaq_f32(acc[t][2], pi, r2);
                    acc[t][3] = vfmaq_f32(acc[t][3], pi, r3);
                }
            }
            for (int32_t t = 0; t < T; ++t) {
                for (int32_t c = 0; c < 4; ++c) vst1q_f32(o[t] + d0 + 4 * c, acc[t][c]);
            }
        }
        return;
    }
#endif
    for (size_t d0 = 0; d0 < width; d0 += kValueColumns) {
        const size_t w = std::min<size_t>(kValueColumns, width - d0);
        float acc[T][kValueColumns] = {};
        for (int32_t t = 0; t < T; ++t) std::copy(o[t] + d0, o[t] + d0 + w, acc[t]);
        if (w == static_cast<size_t>(kValueColumns)) {
            for (size_t i = a; i < b; ++i) {
                const float* row = v + i * stride + d0;
                for (int32_t t = 0; t < T; ++t) {
                    const float pi = p[t][i];
                    for (int32_t c = 0; c < kValueColumns; ++c) acc[t][c] += pi * row[c];
                }
            }
        } else {
            for (size_t i = a; i < b; ++i) {
                const float* row = v + i * stride + d0;
                for (int32_t t = 0; t < T; ++t) {
                    const float pi = p[t][i];
                    for (size_t c = 0; c < w; ++c) acc[t][c] += pi * row[c];
                }
            }
        }
        for (int32_t t = 0; t < T; ++t) std::copy(acc[t], acc[t] + w, o[t] + d0);
    }
}


}  // namespace

LatentFormChooser::Work LatentFormChooser::work(const ModelConfig& c, int64_t n, int64_t visible, int64_t positions) {
    const LatentAttention& m = c.mla;
    const auto heads = static_cast<double>(c.n_head);
    const double project = static_cast<double>(m.kv_rank) * (m.nope_dim + m.v_dim);  // per head: k_b and v_b
    Work w;
    const int64_t selected = c.index_topk > 0 ? std::min(visible, n * c.index_topk) : visible;
    w.absorbed = heads * (static_cast<double>(n) * project +
                          static_cast<double>(selected) * (c.rope_dim + 2.0 * m.kv_rank));
    w.expanded = heads * (static_cast<double>(positions + 1) * project +
                          static_cast<double>(visible) * (m.nope_dim + c.rope_dim + m.v_dim));
    return w;
}

Status LatentFormChooser::set_ratio(float ratio) noexcept {
    if (!(ratio >= kMinRatio && ratio <= kMaxRatio)) {
        return Status(ErrorCode::InvalidArgument, "latent form ratio out of range");
    }
    ratio_ = ratio;
    return Status::ok();
}

float LatentFormChooser::round_ratio(double measured) noexcept {
    if (!(measured > 0.0) || !std::isfinite(measured)) return 1.0f;
    const double steps = std::round(std::log2(measured) * 8.0);
    return std::clamp(static_cast<float>(std::exp2(steps / 8.0)), kMinRatio, kMaxRatio);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
Result<std::unique_ptr<Transformer>> Transformer::load(std::unique_ptr<MmapLoader> file,
                                                       const TransformerOptions& options) {
    if (!file) return Status(ErrorCode::InvalidArgument, "null model file");
    std::unique_ptr<Transformer> model(new Transformer());
    model->file_ = std::move(file);
    const MmapLoader& f = *model->file_;

    auto described = describe_model(f);
    if (!described) return described.status();
    model->config_ = std::move(described).value();
    const ModelConfig& c = model->config_;
    model->variant_fingerprint_ = compatibility_fingerprint(f, c);
    model->expert_mass_ = options.expert_mass;
    model->max_experts_ = options.max_experts;
    model->skip_slow_ = options.skip_slow;
    model->latent_form_ = options.latent_form;

    LIYAB_RETURN_IF_ERROR(model->bind_weights());
    const Layer& first = model->layers_[0];
    // The precision of block 0's largest matrix: its FFN's, else its mixer's input projection.
    const TensorView* typical = nullptr;
    for (const WeightRole role : {kUpExps, kUp, kSsmIn, kQkv, kQ, kQa}) {
        if (typical == nullptr) typical = first.w[role];
    }
    LIYAB_LOG_INFO("bound %d blocks of %s weights (%s, head_dim %d, %d attention + %d recurrent blocks)", c.n_layers,
                   typical != nullptr ? std::string(dtype_traits(typical->type).name).c_str() : "?", c.arch.c_str(),
                   c.head_dim, c.n_attn_layers,
                   static_cast<int>(std::count_if(c.blocks.begin(), c.blocks.end(),
                                                  [](const BlockPlan& b) { return keeps_state(b.mixer); })));
    if (c.n_expert > 0) {
        LIYAB_LOG_INFO("MoE: %d experts, top-%d, expert ff %d, shared ff %d, %s gating%s", c.n_expert, c.n_expert_used,
                       c.n_ff_expert, c.n_ff_shared, c.moe_gating == MoeGating::Softmax ? "softmax" : c.moe_gating == MoeGating::Sigmoid ? "sigmoid" : "sqrtsoftplus",
                       c.moe_norm_weights ? ", renormalized" : "");
    }

    // Context and KV cache. Automatic: what kKvBudgetBytes of KV holds
    // (4096..32768 positions, at most the training context). The KV cache is
    // paged, so memory follows the positions actually used; a hybrid model
    // with few attention blocks gets a long context (Qwen3.6-35B-A3B: 10.6 KiB
    // per position, ~24k), where a flat 4096 forced chats with tools to drop
    // old turns, and each drop meant processing the whole history again.
    KvCacheConfig kv;
    // Only attention blocks have keys and values (at least one slot keeps the
    // cache well-formed for purely recurrent stacks).
    kv.n_layers = std::max(c.n_kv_layers, 1);
    // Without attention (Mamba), the one slot is an unused placeholder with
    // one quantization-block-wide head, valid for every KV storage type.
    kv.n_head_kv = c.n_attn_layers > 0 ? c.n_head_kv : 1;
    kv.head_dim = c.n_attn_layers > 0 ? c.head_dim : quant::kBlock;
    kv.type = options.kv_type;
    // Latent attention: one row [k_pe | latent] per position and block, its
    // latent channels serving as the values (ModelConfig::mla).
    kv.values_in_keys = c.mla.kv_rank > 0 || c.spec->shared_kv;
    kv.value_offset = c.mla.kv_rank > 0 ? c.rope_dim : 0;
    // A completed compressed entry belongs to its closing source token. It
    // follows that token through page images, branches and prefix reuse;
    // unfinished windows live only in bounded recurrent accumulators. Keep
    // precise F32 compressed/index keys independent of the main KV format.
    kv.auxiliary_dim = c.spec->shared_kv ? (c.compressed_layers > 0 ? c.head_dim + c.index_dim : 0) : c.index_dim;
    model->max_batch_ = std::max(1, options.max_batch);
    // Per-block layout: a windowed block (Gemma 2 / 3, Phi-3) keeps only the
    // rows its window still shows, so it costs a bounded amount whatever the
    // context, and a position costs only what the blocks seeing every
    // position keep for it (KvCache::kept_bytes()). Without windowed blocks
    // the automatic length is the plain per-token formula, as before.
    std::vector<int32_t> windows(static_cast<size_t>(kv.n_layers), 0);
    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = model->layers_[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        // Bidirectional blocks see later positions too: their cache keeps every row.
        if (L.kv_slot >= 0 && plan.mixer == MixerKind::Attention && plan.windowed && c.sliding_window > 0 &&
            !options.keep_windowed_rows && c.causal && !c.spec->shared_kv) {
            windows[static_cast<size_t>(L.kv_slot)] = c.sliding_window;
        }
    }
    const size_t kv_per_token = KvCache::bytes_per_token(kv);
    auto automatic = static_cast<int32_t>(std::clamp<size_t>(kKvBudgetBytes / std::max<size_t>(kv_per_token, 1),
                                                             4096, 32768));
    if (std::any_of(windows.begin(), windows.end(), [](int32_t w) { return w > 0; })) {
        // The windowed blocks' share is set aside once (their window plus the
        // rollback margin, whatever the context); the rest of the budget goes
        // to the blocks keeping every position, at their own cost per position.
        KvCacheConfig windowed_only = kv;
        int32_t every = 0;
        for (const int32_t w : windows) {
            if (w > 0) {
                windowed_only.layer_windows.push_back(w);
            } else {
                ++every;
            }
        }
        windowed_only.n_layers = static_cast<int32_t>(windowed_only.layer_windows.size());
        const size_t fixed =
            KvCache::kept_bytes(windowed_only, std::numeric_limits<int32_t>::max() / 2, model->max_batch_);
        KvCacheConfig every_only = kv;
        every_only.n_layers = every;
        const size_t per_position = every > 0 ? KvCache::bytes_per_token(every_only) : 0;
        const size_t room = kKvBudgetBytes > fixed ? kKvBudgetBytes - fixed : 0;
        automatic = per_position == 0 ? 32768
                                      : static_cast<int32_t>(std::clamp<size_t>(room / per_position, 4096, 32768));
    }
    model->context_length_ = options.context_length > 0
                                 ? options.context_length
                                 : (c.n_ctx_train > 0 ? std::min(c.n_ctx_train, automatic) : automatic);
    if (c.n_ctx_train > 0 && model->context_length_ > c.n_ctx_train) {
        LIYAB_LOG_WARN("context %d exceeds the model's training context %d", model->context_length_, c.n_ctx_train);
    }
    // Learned absolute positions: the model addresses only as many positions as it has embeddings for.
    if (c.n_positions > 0 && model->context_length_ > c.n_positions) {
        LIYAB_LOG_WARN("context %d limited to the model's %d learned positions", model->context_length_, c.n_positions);
        model->context_length_ = c.n_positions;
    }
    // ALiBi slopes per query head: a geometric series over the largest power
    // of two of heads, the remaining heads interleaved between its terms (the
    // reference's construction, for any head count).
    if (c.alibi_bias > 0.0f) {
        int32_t pow2 = 1;
        while (pow2 * 2 <= c.n_head) pow2 *= 2;
        const double m0 = std::exp2(-static_cast<double>(c.alibi_bias) / pow2);
        const double m1 = std::exp2(-static_cast<double>(c.alibi_bias) / 2.0 / pow2);
        for (int32_t h = 0; h < c.n_head; ++h) {
            const double slope = h < pow2 ? std::pow(m0, h + 1) : std::pow(m1, 2 * (h - pow2) + 1);
            model->alibi_slopes_.push_back(static_cast<float>(slope));
        }
    }
    // A window holding the whole context keeps every position anyway: such
    // blocks stay with the others (a windowed group would only add bookkeeping).
    for (int32_t& w : windows) {
        if (w >= model->context_length_ || c.spec->shared_kv) w = 0;
    }
    if (std::any_of(windows.begin(), windows.end(), [](int32_t w) { return w > 0; })) kv.layer_windows = windows;
    kv.max_positions = model->context_length_;
    kv.window = options.sliding_window > 0 && options.sliding_window < model->context_length_ ? options.sliding_window : 0;
    if (!c.causal && kv.window > 0) {
        return Status(ErrorCode::Unsupported, "bidirectional attention needs every input row; a rolling cache cannot retain them");
    }
    kv.sink_tokens = std::clamp(options.sink_tokens, 0, kv.page_tokens);
    if (c.spec->shared_kv && kv.window > 0) {
        return Status(ErrorCode::Unsupported, "shared K=V attention requires the model's own sliding window and compressed entries");
    }
    if (kv.type != KvCacheType::F16 && (c.head_dim % quant::kBlock != 0 || kv.value_offset % quant::kBlock != 0 ||
        std::any_of(c.blocks.begin(), c.blocks.end(), [](const BlockPlan& b) {
            return b.mixer == MixerKind::Attention && b.head_dim % quant::kBlock != 0;
        }))) {
        LIYAB_LOG_WARN("head_dim %d (values from channel %d) is not in whole blocks of %d; using an F16 KV cache",
                       c.head_dim, kv.value_offset, quant::kBlock);
        kv.type = KvCacheType::F16;
    }
    auto cache = KvCache::create(kv);
    if (!cache) return cache.status();
    model->kv_ = std::move(cache).value();

    auto rotary = make_rotary_plan(f, c, model->context_length_);
    if (!rotary) return rotary.status();
    model->rope_tables_ = std::move(rotary->tables);
    for (int32_t l = 0; l < c.n_layers; ++l) {
        Layer& layer = model->layers_[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        layer.rope_table = rotary->block_tables[static_cast<size_t>(l)];
        if (plan.mixer == MixerKind::Attention && plan.windowed) layer.window = c.sliding_window;
    }

    if (c.n_expert > 0) {
        LIYAB_RETURN_IF_ERROR(
            model->attach_expert_store(options.expert_cache_bytes, options.memory_budget_bytes, options.reserve_bytes,
                                       options.requant_bits, options.repack_cpu && quant::repack_kernels_available()));
    }
    model->file_->configure_layers(c.n_layers);
    LIYAB_LOG_INFO("%s: %d layers, d=%d, ff=%d, heads=%d/%d x %d, vocab=%d, ctx=%d%s, kv=%s, %.1f KiB/token",
                   c.arch.c_str(), c.n_layers, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim, c.n_vocab,
                   model->context_length_, kv.window ? " (sliding)" : "",
                   std::string(dtype_traits(model->kv_->dtype()).name).c_str(),
                   static_cast<double>(KvCache::bytes_per_token(kv)) / 1024.0);
    if (model->kv_->has_windowed_layers()) {
        const KvCacheConfig& made = model->kv_->config();
        int32_t every = 0;
        for (const int32_t w : made.layer_windows) every += w == 0 ? 1 : 0;
        const int32_t ctx = model->context_length_;
        LIYAB_LOG_INFO("KV per block: %d of %d attention blocks keep every position (%.1f KiB/token), the others their "
                       "window of %d; a full context holds %.1f MiB instead of %.1f MiB",
                       every, made.n_layers, static_cast<double>(model->kv_->rows_bytes(1)) / 1024.0, c.sliding_window,
                       static_cast<double>(KvCache::kept_bytes(made, ctx, model->max_batch_)) / (1024.0 * 1024.0),
                       static_cast<double>(KvCache::bytes_per_token(made)) * ctx / (1024.0 * 1024.0));
    }
    if (c.mla.kv_rank > 0) {
        LIYAB_LOG_INFO("latent attention: a %d-wide latent + %d RoPE channels cached per position, %d heads of %d + %d "
                       "-> %d%s",
                       c.mla.kv_rank, c.rope_dim, c.n_head, c.mla.nope_dim, c.rope_dim, c.mla.v_dim,
                       c.mla.q_rank > 0 ? ", low-rank queries" : "");
    }
    if (c.ssm_v_heads > 0) {
        LIYAB_LOG_INFO("DeltaNet: %d/%d heads x %d, conv %d", c.ssm_k_heads, c.ssm_v_heads, c.ssm_head_dim,
                       c.ssm_conv_kernel);
    }
    if (c.shortconv_l_cache > 0) LIYAB_LOG_INFO("short convolution: %d taps", c.shortconv_l_cache);
    if (c.kda_heads > 0) {
        LIYAB_LOG_INFO("Kimi Delta Attention: %d heads x %d, conv %d", c.kda_heads, c.kda_head_dim, c.ssm_conv_kernel);
    }
    if (c.hybrid()) {
        LIYAB_LOG_INFO("%d recurrent blocks, state %.1f MiB", c.n_recurrent_layers,
                       static_cast<double>(model->recurrent_state_bytes()) / (1024.0 * 1024.0));
    }
    if (c.mamba.d_inner > 0) {
        const MambaGeometry& m = c.mamba;
        LIYAB_LOG_INFO("Mamba-%d: %zu recurrent blocks, d_inner %d, state %d per channel%s, conv %d, state %.1f MiB",
                       m.n_heads > 0 ? 2 : 1, model->states_.size(), m.d_inner, m.d_state,
                       m.n_heads > 0 ? (", " + std::to_string(m.n_heads) + " heads, " + std::to_string(m.n_groups) +
                                        " groups").c_str()
                                     : "",
                       m.conv_kernel, static_cast<double>(model->recurrent_state_bytes()) / (1024.0 * 1024.0));
    }
    // CPU-only models: the resident Q4_K / Q5_K / Q6_K / Q8_0 matrices in the
    // layout of the i8mm batched kernels (a lossless rearrangement, same
    // size). With expert streaming that is every matrix but the experts (3-D,
    // read from storage in the file's layout); a model under the layer
    // streaming window keeps the file's layout throughout.
    if (options.repack_cpu && !model->file_->streaming()) {
        struct Repack {
            DType from, to;
            int64_t rows, cols;  // required multiples
        };
        const Repack repacks[] = {Repack{DType::Q4_K, DType::Q4_K_R8, 8, quant::kSuperBlock},
                                  Repack{DType::Q5_K, DType::Q5_K_R8, 8, quant::kSuperBlock},
                                  Repack{DType::Q6_K, DType::Q6_K_R8, 8, quant::kSuperBlock},
                                  Repack{DType::Q8_0, DType::Q8_0_R4, 4, quant::kBlock}};
        auto selected = [&](const TensorView& t, const Repack& r) {
            return t.type == r.from && t.n_dims == 2 && t.rows() % r.rows == 0 && t.cols() % r.cols == 0 &&
                   t.name != "token_embd.weight" && !t.name.ends_with(".attn_kv_b.weight") &&
                   !t.name.ends_with(".attn_output_a.weight") && !model->file_->converted(t);
        };
        // The copies take memory of their own next to the mapped file. When they do not fit, the file's layout
        // (the same bits, about 14% slower per row) keeps every block resident, which beats streaming blocks
        // from storage: a 27B Q4_K_M decodes at 12.7 tok/s that way against 1.6 with repacked copies and a
        // streamed tail on a 48 GiB Mac.
        size_t copies = 0;
        for (const TensorView& t : model->file_->tensors()) {
            for (const Repack& r : repacks) {
                if (selected(t, r)) copies += t.nbytes;
            }
        }
        const uint64_t available = usable_memory_bytes(options.memory_budget_bytes);
        constexpr uint64_t kHeadroom = uint64_t{4} << 30;  // left for the system and other applications
        if (available != 0 && model->file_->file_size() + copies + kHeadroom > available) {
            LIYAB_LOG_INFO("repacked weights skipped: %.2f GiB of copies next to %.2f GiB of weights exceed the %.2f GiB "
                           "available",
                           static_cast<double>(copies) / (1024.0 * 1024.0 * 1024.0),
                           static_cast<double>(model->file_->file_size()) / (1024.0 * 1024.0 * 1024.0),
                           static_cast<double>(available) / (1024.0 * 1024.0 * 1024.0));
            return model;
        }
        for (const Repack& r : repacks) {
            auto repacked = model->file_->requantize([&](const TensorView& t) { return selected(t, r); }, r.to);
            if (!repacked) return repacked.status();
        }
    }
    return model;
}

Status Transformer::bind_weights() {
    const MmapLoader& f = *file_;
    const ModelConfig& c = config_;

    auto matrix = [&](const std::string& name, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
        if (t->cols() != cols || t->rows() != rows) {
            return Status(ErrorCode::InvalidModel, "tensor '" + name + "' has shape [" + std::to_string(t->cols()) +
                                                       ", " + std::to_string(t->rows()) + "], expected [" +
                                                       std::to_string(cols) + ", " + std::to_string(rows) + "]");
        }
        out = t;
        return Status::ok();
    };
    auto vec = [&](const std::string& name, int64_t n, bool required, std::vector<float>& out) -> Status {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) {
            return required ? Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'") : Status::ok();
        }
        if (t->elements() != n) return Status(ErrorCode::InvalidModel, "tensor '" + name + "' has the wrong size");
        out.resize(static_cast<size_t>(n));
        quant::dequantize_row(t->type, t->data, out.data(), n);
        return Status::ok();
    };
    auto position_bias = [&](const std::string& name, int64_t width, int64_t stride, std::vector<float>& out) -> Status {
        const TensorView* tensor = nullptr;
        LIYAB_RETURN_IF_ERROR(matrix(name, width, stride, tensor));
        if (tensor->n_dims != 2) return Status(ErrorCode::InvalidModel, "position bias must have two dimensions: " + name);
        return vec(name, width * stride, true, out);
    };

    token_embd_ = f.tensor("token_embd.weight");  // shape checked by describe_model()
    // Per-layer embeddings: the token table stays in the mapping (a token
    // reads its row), the projection is an ordinary matrix.
    if (c.ple_dim > 0) {
        const int64_t width = int64_t{c.n_layers} * c.ple_dim;
        ple_embd_ = f.tensor("per_layer_token_embd.weight");
        if (ple_embd_->cols() != width || ple_embd_->rows() < 1 || ple_embd_->ne[2] != 1 || ple_embd_->ne[3] != 1) {
            return Status(ErrorCode::InvalidModel, "per_layer_token_embd.weight is not [n_layers * ple_dim, vocab]");
        }
        LIYAB_RETURN_IF_ERROR(matrix("per_layer_model_proj.weight", c.n_embd, width, ple_model_proj_));
        LIYAB_RETURN_IF_ERROR(vec("per_layer_proj_norm.weight", c.ple_dim, true, ple_proj_norm_));
    }
    // AltUp: one 2-D view per projection of the stacked [n_embd, n_embd, K - 1] tensors.
    if (c.altup_streams > 1) {
        for (const auto& [name, views] : {std::pair{"altup_proj.weight", &altup_proj_},
                                          std::pair{"altup_unembd_proj.weight", &altup_unembd_}}) {
            const TensorView* t = f.tensor(name);  // shape checked by describe_model()
            for (int64_t i = 0; i + 1 < c.altup_streams; ++i) views->push_back(expert_slice(*t, i));
        }
    }
    if (f.tensor("output.weight") != nullptr) {
        LIYAB_RETURN_IF_ERROR(matrix("output.weight", c.n_embd, c.n_vocab, output_));
    } else {
        output_ = token_embd_;  // tied embeddings
    }
    // RMSNorm requires weights. Each LayerNorm independently allows an
    // optional scale and offset, including OLMo's non-parametric norms.
    const bool norm_weights = c.spec->norm == NormKind::Rms;
    LIYAB_RETURN_IF_ERROR(vec(std::string(c.spec->output_norm) + ".weight", c.n_embd, norm_weights, output_norm_));
    LIYAB_RETURN_IF_ERROR(vec(std::string(c.spec->output_norm) + ".bias", c.n_embd, false, output_norm_b_));
    LIYAB_RETURN_IF_ERROR(vec("output.bias", c.n_vocab, false, output_bias_));
    if (c.spec->output_norm != "token_embd_norm") {
        LIYAB_RETURN_IF_ERROR(vec("token_embd_norm.weight", c.n_embd, false, embd_norm_));
        LIYAB_RETURN_IF_ERROR(vec("token_embd_norm.bias", c.n_embd, false, embd_norm_b_));
    }
    if (c.spec->norm == NormKind::Rms && !embd_norm_b_.empty()) {
        return Status(ErrorCode::InvalidModel, "an RMS embedding norm with a bias");
    }
    if (c.n_positions > 0) position_embd_ = f.tensor("position_embd.weight");  // shape checked by describe_model()
    // Per-head Q/K norms: one weight shared by every head, or one row per
    // head (a [head_dim, heads] tensor: StableLM 2 12B, Command-R+).
    auto qk_vec = [&](const std::string& name, int64_t heads, int32_t head_dim, std::vector<float>& out) -> Status {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) return Status::ok();
        const bool per_head = t->n_dims == 2 && t->ne[0] == head_dim && t->ne[1] == heads && heads > 1;
        if (t->elements() != head_dim && !per_head) {
            return Status(ErrorCode::Unsupported, "tensor '" + name + "' is not a per-head norm of " +
                                                      std::to_string(head_dim) + " (norms over all heads together "
                                                      "are not supported)");
        }
        return vec(name, t->elements(), true, out);
    };
    if (c.hc_count > 0) {
        LIYAB_RETURN_IF_ERROR(matrix("output_hc_fn.weight", int64_t{c.hc_count} * c.n_embd, c.hc_count, hc_head_));
        LIYAB_RETURN_IF_ERROR(vec("output_hc_base.weight", c.hc_count, true, hc_head_base_));
        LIYAB_RETURN_IF_ERROR(vec("output_hc_scale.weight", 1, true, hc_head_scale_));
    }

    const int64_t dn_key = int64_t{c.ssm_k_heads} * c.ssm_head_dim;
    const int64_t dn_value = int64_t{c.ssm_v_heads} * c.ssm_head_dim;
    const int64_t conv_dim = 2 * dn_key + dn_value;

    layers_.resize(static_cast<size_t>(c.n_layers));
    int32_t kv_slots = 0;
    // Each recurrent block's state sizes, in state-slot order: [conv | ssm] floats.
    std::vector<std::pair<size_t, size_t>> state_sizes;
    auto add_state = [&](Layer& L, size_t conv, size_t ssm) {
        L.state_slot = static_cast<int32_t>(state_sizes.size());
        state_sizes.emplace_back(conv, ssm);
    };
    for (int32_t l = 0; l < c.n_layers; ++l) {
        Layer& L = layers_[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        auto has_norm = [&](std::string_view name) {
            return f.tensor(block_tensor(l, std::string(name) + ".weight")) != nullptr ||
                   f.tensor(block_tensor(l, std::string(name) + ".bias")) != nullptr;
        };
        const std::string_view attn_norm = plan.parallel && !has_norm(c.spec->attn_norm) && has_norm(c.spec->ffn_norm)
                                               ? c.spec->ffn_norm : c.spec->attn_norm;
        LIYAB_RETURN_IF_ERROR(vec(block_norm_tensor(f, l, attn_norm), c.n_embd, norm_weights && plan.norm_before_mixer, L.attn_norm));
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, std::string(attn_norm) + ".bias"), c.n_embd, false, L.attn_norm_b));
        if (plan.mixer == MixerKind::Attention) {
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_output.bias"), c.n_embd, false, L.bo));
        }
        const int64_t q_dim = int64_t{plan.n_head} * plan.head_dim;
        const int64_t kv_dim = int64_t{plan.n_head_kv} * plan.head_dim;
        if (c.hc_count > 0) {
            const int64_t width = int64_t{c.hc_count} * (2 + c.hc_count);
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "hc_attn_fn.weight"), int64_t{c.hc_count} * c.n_embd, width, L.w[kHcAttn]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "hc_ffn_fn.weight"), int64_t{c.hc_count} * c.n_embd, width, L.w[kHcFfn]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "hc_attn_base.weight"), width, true, L.hc_attn_base));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "hc_ffn_base.weight"), width, true, L.hc_ffn_base));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "hc_attn_scale.weight"), 3, true, L.hc_attn_scale));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "hc_ffn_scale.weight"), 3, true, L.hc_ffn_scale));
        }
        if (plan.mixer == MixerKind::Attention && c.spec->shared_kv) {
            L.kv_slot = kv_slots++;
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q_a.weight"), c.n_embd, c.shared_q_rank, L.w[kQa]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_q_a_norm.weight"), c.shared_q_rank, true, L.q_a_norm));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q_b.weight"), c.shared_q_rank,
                                         int64_t{c.n_head} * c.head_dim, L.w[kQb]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_kv.weight"), c.n_embd, c.head_dim, L.w[kSharedKv]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_kv_a_norm.weight"), c.head_dim, true, L.kv_a_norm));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_output_a.weight"), int64_t{c.n_head} * c.head_dim / c.output_groups,
                                         int64_t{c.output_groups} * c.output_rank, L.w[kOutputA]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_output_b.weight"), int64_t{c.output_groups} * c.output_rank,
                                         c.n_embd, L.w[kOutputB]));
            LIYAB_RETURN_IF_ERROR(vec(block_norm_tensor(f, l, "attn_sinks"), plan.n_head, false, L.sinks));
            if (plan.compression.stride > 0) {
                const size_t index_state = plan.indexed
                    ? CompressionPlan{c.index_dim, plan.compression.stride, true}.state_floats() : 0;
                add_state(L, plan.compression.state_floats() + index_state, 0);
                const int64_t projected = int64_t{c.head_dim} * (plan.compression.overlap ? 2 : 1);
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_compressor_kv.weight"), c.n_embd, projected, L.w[kCompressKv]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_compressor_gate.weight"), c.n_embd, projected, L.w[kCompressGate]));
                LIYAB_RETURN_IF_ERROR(position_bias(block_tensor(l, "attn_compressor_ape.weight"), projected, plan.compression.stride, L.compress_bias));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_compressor_norm.weight"), c.head_dim, true, L.compress_norm));
                if (plan.indexed) {
                    LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "indexer_compressor_kv.weight"), c.n_embd, 2 * int64_t{c.index_dim}, L.w[kIndexCompressKv]));
                    LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "indexer_compressor_gate.weight"), c.n_embd, 2 * int64_t{c.index_dim}, L.w[kIndexCompressGate]));
                    LIYAB_RETURN_IF_ERROR(position_bias(block_tensor(l, "indexer_compressor_ape.weight"), 2 * int64_t{c.index_dim}, plan.compression.stride, L.index_compress_bias));
                    LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "indexer_compressor_norm.weight"), c.index_dim, true, L.index_compress_norm));
                    LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "indexer.attn_q_b.weight"), c.shared_q_rank,
                                                 int64_t{c.index_heads} * c.index_dim, L.w[kIndexQ]));
                    LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "indexer.proj.weight"), c.n_embd, c.index_heads, L.w[kIndexWeights]));
                }
            }
        } else if (plan.mixer == MixerKind::Attention && plan.latent) {
            L.kv_slot = kv_slots++;
            const LatentAttention& m = c.mla;
            const int64_t q_rows = int64_t{c.n_head} * (m.nope_dim + c.rope_dim);  // per head [q_nope | q_pe]
            if (m.q_rank > 0) {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q_a.weight"), c.n_embd, m.q_rank, L.w[kQa]));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_q_a_norm.weight"), m.q_rank, true, L.q_a_norm));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q_b.weight"), m.q_rank, q_rows, L.w[kQb]));
            } else {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q.weight"), c.n_embd, q_rows, L.w[kQ]));
            }
            LIYAB_RETURN_IF_ERROR(
                matrix(block_tensor(l, "attn_kv_a_mqa.weight"), c.n_embd, m.kv_rank + c.rope_dim, L.w[kKvA]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_kv_a_norm.weight"), m.kv_rank, true, L.kv_a_norm));
            // Per head: attn_k_b maps a head's q_nope onto the latent (the
            // key's projection, transposed by the converter), attn_v_b the
            // weighted latent onto the head's value.
            auto per_head = [&](const char* suffix, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
                const std::string name = block_tensor(l, suffix);
                const TensorView* t = f.tensor(name);
                if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
                if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != c.n_head || t->ne[3] != 1) {
                    return Status(ErrorCode::InvalidModel, "tensor '" + name + "' does not have the per-head shape");
                }
                out = t;
                return Status::ok();
            };
            if (plan.indexed) {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "indexer.attn_q_b.weight"), m.q_rank,
                                             int64_t{c.index_heads} * c.index_dim, L.w[kIndexQ]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "indexer.attn_k.weight"), c.n_embd, c.index_dim,
                                             L.w[kIndexK]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "indexer.proj.weight"), c.n_embd, c.index_heads,
                                             L.w[kIndexWeights]));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "indexer.k_norm.weight"), c.index_dim, true, L.index_norm));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "indexer.k_norm.bias"), c.index_dim, true, L.index_bias));
            }
            if (plan.latent_combined) {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_kv_b.weight"), m.kv_rank,
                                             int64_t{c.n_head} * (int64_t{m.nope_dim} + m.v_dim), L.w[kKvB]));
            } else {
                LIYAB_RETURN_IF_ERROR(per_head("attn_k_b.weight", m.nope_dim, m.kv_rank, L.w[kKb]));
                LIYAB_RETURN_IF_ERROR(per_head("attn_v_b.weight", m.kv_rank, m.v_dim, L.w[kVb]));
            }
            LIYAB_RETURN_IF_ERROR(
                matrix(block_tensor(l, "attn_output.weight"), int64_t{c.n_head} * m.v_dim, c.n_embd, L.w[kO]));
        } else if (plan.mixer == MixerKind::Attention) {
            // A block sharing another's keys and values reads that block's
            // cache layer (KV sharing: it has no rows of its own).
            const bool own_kv = plan.kv_source < 0;
            L.kv_slot = own_kv ? kv_slots++ : layers_[static_cast<size_t>(plan.kv_source)].kv_slot;
            if (f.tensor(block_tensor(l, "attn_gate.weight")) != nullptr) {
                const int64_t gates = f.tensor(block_tensor(l, "attn_gate.weight"))->rows();
                if (gates != q_dim && gates != plan.n_head) return Status(ErrorCode::InvalidModel, "invalid attention gate width");
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_gate.weight"), c.n_embd, gates, L.w[kZ]));
            }
            if (plan.fused_qkv) {
                LIYAB_RETURN_IF_ERROR(
                    matrix(block_tensor(l, "attn_qkv.weight"), c.n_embd, q_dim + 2 * kv_dim, L.w[kQkv]));
                // A fused bias, if any, splits like the rows.
                std::vector<float> bias;
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_qkv.bias"), q_dim + 2 * kv_dim, false, bias));
                if (!bias.empty()) {
                    L.bq.assign(bias.begin(), bias.begin() + q_dim);
                    L.bk.assign(bias.begin() + q_dim, bias.begin() + q_dim + kv_dim);
                    L.bv.assign(bias.begin() + q_dim + kv_dim, bias.end());
                }
            } else {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q.weight"), c.n_embd,
                                             plan.attn_gate && L.w[kZ] == nullptr ? 2 * q_dim : q_dim, L.w[kQ]));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_q.bias"), q_dim, false, L.bq));
                if (own_kv) {  // a sharing block's attn_k / attn_v, if the file has them, are never used
                    LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_k.weight"), c.n_embd, kv_dim, L.w[kK]));
                    LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_k.bias"), kv_dim, false, L.bk));
                    if (!plan.values_from_keys) {
                        LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_v.weight"), c.n_embd, kv_dim, L.w[kV]));
                        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_v.bias"), kv_dim, false, L.bv));
                    } else if (f.tensor(block_tensor(l, "attn_v.bias")) != nullptr) {
                        return Status(ErrorCode::InvalidModel, "value bias without a value projection");
                    }
                }
            }
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_output.weight"), q_dim, c.n_embd, L.w[kO]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_sinks.weight"), plan.n_head, false, L.sinks));
            if (plan.qk_norm_whole) {
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_q_norm.weight"), q_dim, false, L.q_norm));
                if (own_kv) LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_k_norm.weight"), kv_dim, false, L.k_norm));
            } else {
            LIYAB_RETURN_IF_ERROR(qk_vec(block_tensor(l, "attn_q_norm.weight"), plan.n_head, plan.head_dim, L.q_norm));
            if (own_kv) LIYAB_RETURN_IF_ERROR(qk_vec(block_tensor(l, "attn_k_norm.weight"), plan.n_head_kv, plan.head_dim, L.k_norm));
            if (c.spec->norm == NormKind::Layer) {
                LIYAB_RETURN_IF_ERROR(qk_vec(block_tensor(l, "attn_q_norm.bias"), plan.n_head, plan.head_dim, L.q_norm_b));
                if (own_kv) LIYAB_RETURN_IF_ERROR(qk_vec(block_tensor(l, "attn_k_norm.bias"), plan.n_head_kv, plan.head_dim, L.k_norm_b));
            }
            }
        } else if (plan.mixer == MixerKind::Mamba || plan.mixer == MixerKind::Mamba2) {
            // in_proj gives [x | z] (Mamba-1) or [z | x B C | dt] (Mamba-2); the
            // convolution runs over x, or over [x | B | C].
            const MambaGeometry& m = c.mamba;
            const bool v2 = plan.mixer == MixerKind::Mamba2;
            const int64_t di = m.d_inner, ns = m.d_state;
            const int64_t channels = v2 ? di + 2 * int64_t{m.n_groups} * ns : di;
            const int64_t in_rows = v2 ? di + channels + m.n_heads : 2 * di;
            const int64_t per_head = v2 ? m.n_heads : di;  // A, D and the dt bias
            add_state(L, static_cast<size_t>(channels * (m.conv_kernel - 1)), static_cast<size_t>(di * ns));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_in.weight"), c.n_embd, in_rows, L.w[kSsmIn]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_in.bias"), in_rows, false, L.ssm_in_bias));
            LIYAB_RETURN_IF_ERROR(
                vec(block_tensor(l, "ssm_conv1d.weight"), channels * m.conv_kernel, true, L.conv1d));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_conv1d.bias"), channels, false, L.conv_bias));
            if (L.conv_bias.empty()) L.conv_bias.assign(static_cast<size_t>(channels), 0.0f);
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_a"), v2 ? per_head : di * ns, true, L.ssm_a));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_d"), per_head, true, L.ssm_d));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_dt.bias"), per_head, true, L.ssm_dt));
            if (v2) {
                // The gated norm's weight, one group of d_inner / n_groups channels after another.
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_norm.weight"), di, true, L.ssm_norm));
            } else {
                const int64_t r = m.dt_rank;
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_x.weight"), di, r + 2 * ns, L.w[kSsmX]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_dt.weight"), r, di, L.w[kSsmDt]));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_dt_norm.weight"), m.dt_b_c_rms ? di : r, false, L.dt_norm));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_b_norm.weight"), ns, false, L.b_norm));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_c_norm.weight"), ns, false, L.c_norm));
                if (L.dt_norm.empty() != L.b_norm.empty() || L.dt_norm.empty() != L.c_norm.empty()) {
                    return Status(ErrorCode::InvalidModel, "block " + std::to_string(l) +
                                                               ": ssm_dt_norm, ssm_b_norm and ssm_c_norm go together");
                }
            }
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_out.weight"), di, c.n_embd, L.w[kSsmOut]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_out.bias"), c.n_embd, false, L.ssm_out_bias));
        } else if (plan.mixer == MixerKind::ShortConv) {
            // in_proj [B | C | x], each n_embd wide; the convolution's taps per channel.
            const int64_t d = c.n_embd;
            const int64_t taps = c.shortconv_l_cache;
            add_state(L, static_cast<size_t>(d * (taps - 1)), 0);
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "shortconv.in_proj.weight"), d, 3 * d, L.w[kConvIn]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "shortconv.out_proj.weight"), d, d, L.w[kConvOut]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "shortconv.conv.weight"), d * taps, true, L.conv1d));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "shortconv.conv.bias"), d, false, L.conv_bias));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "shortconv.in_proj.bias"), 3 * d, false, L.in_bias));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "shortconv.out_proj.bias"), d, false, L.out_bias));
        } else if (plan.mixer == MixerKind::Kda) {
            const int64_t hd = c.kda_head_dim;
            const int64_t qkv = int64_t{c.kda_heads} * hd;
            const int64_t taps = c.ssm_conv_kernel;
            add_state(L, static_cast<size_t>(3 * qkv * (taps - 1)), static_cast<size_t>(qkv * hd));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q.weight"), c.n_embd, qkv, L.w[kQ]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_k.weight"), c.n_embd, qkv, L.w[kK]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_v.weight"), c.n_embd, qkv, L.w[kV]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_beta.weight"), c.n_embd, c.kda_heads, L.w[kBeta]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_f_a.weight"), c.n_embd, hd, L.w[kFa]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_f_b.weight"), hd, qkv, L.w[kFb]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_g_a.weight"), c.n_embd, hd, L.w[kGa]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_g_b.weight"), hd, qkv, L.w[kGb]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_output.weight"), qkv, c.n_embd, L.w[kSsmOut]));
            // The three convolutions' taps, one after another: q, k, then v channels.
            std::vector<float> taps_q, taps_k, taps_v;
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_conv1d_q.weight"), qkv * taps, true, taps_q));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_conv1d_k.weight"), qkv * taps, true, taps_k));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_conv1d_v.weight"), qkv * taps, true, taps_v));
            L.conv1d = std::move(taps_q);
            L.conv1d.insert(L.conv1d.end(), taps_k.begin(), taps_k.end());
            L.conv1d.insert(L.conv1d.end(), taps_v.begin(), taps_v.end());
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_a"), c.kda_heads, true, L.ssm_a));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_dt.bias"), qkv, true, L.ssm_dt));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_norm.weight"), hd, true, L.ssm_norm));
        } else if (plan.mixer == MixerKind::DeltaNet) {
            add_state(L, static_cast<size_t>(conv_dim) * static_cast<size_t>(c.ssm_conv_kernel - 1),
                      static_cast<size_t>(c.ssm_v_heads) * static_cast<size_t>(c.ssm_head_dim) * static_cast<size_t>(c.ssm_head_dim));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_qkv.weight"), c.n_embd, conv_dim, L.w[kQkv]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_gate.weight"), c.n_embd, dn_value, L.w[kZ]));
            if (f.tensor(block_tensor(l, "ssm_ba.weight")) != nullptr) {
                // Qwen3-Next: beta and alpha in one projection, [beta | alpha] of each key head's value heads.
                LIYAB_RETURN_IF_ERROR(
                    matrix(block_tensor(l, "ssm_ba.weight"), c.n_embd, 2 * int64_t{c.ssm_v_heads}, L.w[kBa]));
            } else {
                LIYAB_RETURN_IF_ERROR(
                    matrix(block_tensor(l, "ssm_alpha.weight"), c.n_embd, c.ssm_v_heads, L.w[kAlpha]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_beta.weight"), c.n_embd, c.ssm_v_heads, L.w[kBeta]));
            }
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_out.weight"), dn_value, c.n_embd, L.w[kSsmOut]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_conv1d.weight"), conv_dim * c.ssm_conv_kernel, true, L.conv1d));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_a"), c.ssm_v_heads, true, L.ssm_a));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_dt.bias"), c.ssm_v_heads, true, L.ssm_dt));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_norm.weight"), c.ssm_head_dim, true, L.ssm_norm));
        } else if (plan.mixer == MixerKind::Linear) {
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_output.weight"), c.n_embd, c.n_embd, L.w[kO]));
        }
        if (plan.mixer == MixerKind::None && plan.ffn != FfnKind::None &&
            f.tensor(block_norm_tensor(f, l, c.spec->ffn_norm)) == nullptr) {
            L.ffn_norm = L.attn_norm;
            L.ffn_norm_b = L.attn_norm_b;
        }
        // A parallel block may have no FFN norm of its own: its FFN reads the mixer's normed input.
        if (!plan.parallel || attn_norm != c.spec->ffn_norm) {
            LIYAB_RETURN_IF_ERROR(vec(block_norm_tensor(f, l, c.spec->ffn_norm), c.n_embd,
                                      norm_weights && plan.norm_before_ffn && !plan.parallel && L.ffn_norm.empty(), L.ffn_norm));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, std::string(c.spec->ffn_norm) + ".bias"), c.n_embd, false, L.ffn_norm_b));
        }
        if (!c.spec->post_mixer_norm.empty()) {
            LIYAB_RETURN_IF_ERROR(vec(block_norm_tensor(f, l, c.spec->post_mixer_norm), c.n_embd,
                                      false, L.post_mixer_norm));
        }
        if (!c.spec->post_ffn_norm.empty()) {
            LIYAB_RETURN_IF_ERROR(vec(block_norm_tensor(f, l, c.spec->post_ffn_norm), c.n_embd,
                                      false, L.post_ffn_norm));
        }
        // Gemma 3n / 4 parts (ModelConfig, BlockPlan): Laurel, AltUp's
        // router and coefficients, per-layer embeddings, the output scale.
        if (plan.laurel) {
            const TensorView* left = f.tensor(block_tensor(l, "laurel_l.weight"));
            const int64_t rank = left->rows();
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "laurel_l.weight"), c.n_embd, rank, L.w[kLaurelL]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "laurel_r.weight"), rank, c.n_embd, L.w[kLaurelR]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "laurel_post_norm.weight"), c.n_embd, true, L.laurel_norm));
        }
        if (c.altup_streams > 1) {
            const int64_t k = c.altup_streams;
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "altup_router.weight"), k * c.n_embd, true, L.altup_router));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "altup_router_norm.weight"), c.n_embd, true, L.altup_router_norm));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "altup_predict_coef.weight"), k * k * k, true, L.altup_predict));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "altup_correct_coef.weight"), k * k, true, L.altup_correct));
            LIYAB_RETURN_IF_ERROR(
                vec(block_tensor(l, "altup_correct_scale.weight"), c.n_embd, false, L.altup_correct_scale));
        }
        if (c.ple_dim > 0) {
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "inp_gate.weight"), c.n_embd, c.ple_dim, L.w[kPleGate]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "proj.weight"), c.ple_dim, c.n_embd, L.w[kPleProj]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "post_norm.weight"), c.n_embd, true, L.ple_norm));
        }
        {
            std::vector<float> scale;
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "layer_output_scale.weight"), 1, false, scale));
            if (!scale.empty()) {
                if (c.altup_streams > 1) {
                    return Status(ErrorCode::Unsupported, "a block output scale together with AltUp streams");
                }
                L.output_scale = scale[0];
                if (!std::isfinite(L.output_scale)) {
                    return Status(ErrorCode::InvalidModel, "non-finite block output scale");
                }
            }
        }
        if (plan.ffn == FfnKind::Ungated) {  // down(act(up)), each with its bias where present
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_up.weight"), c.n_embd, plan.n_ff, L.w[kUp]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_down.weight"), plan.n_ff, c.n_embd, L.w[kDown]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_up.bias"), plan.n_ff, false, L.b_up));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_down.bias"), c.n_embd, false, L.b_down));
            continue;
        }
        if (plan.parallel_ffn) {
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "pre_ffw_norm_2.weight"), c.n_embd, true, L.expert_norm));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "post_ffw_norm_1.weight"), c.n_embd, true, L.dense_post_norm));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "post_ffw_norm_2.weight"), c.n_embd, true, L.expert_post_norm));
        }
        if (plan.ffn == FfnKind::Dense || plan.parallel_ffn) {
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_down.bias"), c.n_embd, false, L.b_down));
            if (!(plan.parallel_ffn ? plan.dense_fused_gate_up : plan.fused_gate_up)) {
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_gate.bias"), plan.n_ff, false, L.b_gate));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_up.bias"), plan.n_ff, false, L.b_up));
            }
            if (plan.parallel_ffn ? plan.dense_fused_gate_up : plan.fused_gate_up) {  // ffn_up is [gate | up]; no ffn_gate
                LIYAB_RETURN_IF_ERROR(
                    matrix(block_tensor(l, "ffn_up.weight"), c.n_embd, 2 * int64_t{plan.n_ff}, L.w[kUp]));
                std::vector<float> bias;
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_up.bias"), 2 * int64_t{plan.n_ff}, false, bias));
                if (!bias.empty()) {
                    L.b_gate.assign(bias.begin(), bias.begin() + plan.n_ff);
                    L.b_up.assign(bias.begin() + plan.n_ff, bias.end());
                }
            } else {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_gate.weight"), c.n_embd, plan.n_ff, L.w[kGate]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_up.weight"), c.n_embd, plan.n_ff, L.w[kUp]));
            }
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_down.weight"), plan.n_ff, c.n_embd, L.w[kDown]));
            if (!plan.parallel_ffn) continue;
        }
        if (plan.ffn == FfnKind::None) continue;
        LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_gate_inp.weight"), c.n_embd, c.n_expert, L.w[kRouter]));
        if (plan.hash_routing) {
            auto routes = adapt_expert_routes(*f.tensor(block_tensor(l, "ffn_gate_tid2eid.weight")),
                                               c.n_vocab, c.n_expert_used, c.n_expert);
            if (!routes) return routes.status();
            L.hash_experts = std::move(routes).value();
        }
        if (plan.router_input_rms) {
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_gate_inp.scale"), c.n_embd, true, L.router_scale));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_down_exps.scale"), c.n_expert, false, L.expert_scale));
        }
        auto experts = [&](const char* suffix, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
            const std::string name = block_tensor(l, suffix);
            const TensorView* t = f.tensor(name);
            if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
            if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != c.n_expert || t->ne[3] != 1) {
                return Status(ErrorCode::InvalidModel, "tensor '" + name + "' does not have the expert shape");
            }
            out = t;
            return Status::ok();
        };
        if (plan.fused_gate_up) {  // each expert's [gate | up] in one tensor; no gate of its own
            LIYAB_RETURN_IF_ERROR(
                experts("ffn_gate_up_exps.weight", c.n_embd, 2 * int64_t{c.n_ff_expert}, L.w[kUpExps]));
        } else {
            LIYAB_RETURN_IF_ERROR(experts("ffn_gate_exps.weight", c.n_embd, c.n_ff_expert, L.w[kGateExps]));
            LIYAB_RETURN_IF_ERROR(experts("ffn_up_exps.weight", c.n_embd, c.n_ff_expert, L.w[kUpExps]));
        }
        LIYAB_RETURN_IF_ERROR(experts("ffn_down_exps.weight", c.n_ff_expert, c.n_embd, L.w[kDownExps]));
        const bool named_bias = f.tensor(block_tensor(l, "exp_probs_b.bias")) != nullptr;
        const bool bare_bias = f.tensor(block_tensor(l, "exp_probs_b")) != nullptr;
        if (named_bias && bare_bias) return Status(ErrorCode::InvalidModel, "duplicate expert selection bias in block " + std::to_string(l));
        LIYAB_RETURN_IF_ERROR(vec(named_bias ? block_tensor(l, "exp_probs_b.bias") : block_tensor(l, "exp_probs_b"),
                                  c.n_expert, false, L.expert_bias));
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_gate_inp.bias"), c.n_expert, false, L.router_bias));
        // Per-expert biases ([rows, n_expert] in the file: expert e's at e * rows).
        const int64_t E = c.n_expert;
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_gate_exps.bias"), E * c.n_ff_expert, false, L.gate_exps_bias));
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_up_exps.bias"), E * c.n_ff_expert, false, L.up_exps_bias));
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_down_exps.bias"), E * c.n_embd, false, L.down_exps_bias));
        if (plan.shared_expert) {
            const int64_t sh = c.n_ff_shared > 0 ? c.n_ff_shared : f.tensor(block_tensor(l, "ffn_up_shexp.weight"))->rows();
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_gate_shexp.weight"), c.n_embd, sh, L.w[kGateShared]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_up_shexp.weight"), c.n_embd, sh, L.w[kUpShared]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_down_shexp.weight"), sh, c.n_embd, L.w[kDownShared]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_gate_inp_shexp.weight"), c.n_embd, false, L.shared_gate));
        }
    }

    states_.resize(state_sizes.size());
    for (size_t i = 0; i < states_.size(); ++i) {
        states_[i].conv.assign(state_sizes[i].first, 0.0f);
        states_[i].ssm.assign(state_sizes[i].second, 0.0f);
    }
    return Status::ok();
}

size_t Transformer::context_state_bytes() const noexcept {
    return static_cast<size_t>(1 + kStateSnapshots + std::max(rollback_window_, 0)) * recurrent_state_bytes();
}

size_t Transformer::recurrent_state_bytes() const noexcept {
    size_t bytes = 0;
    for (const RecurrentState& st : states_) bytes += (st.conv.size() + st.ssm.size()) * sizeof(float);
    return bytes;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

class Transformer::BlockWeights {
public:
    BlockWeights(Transformer& model, int32_t layer) : model_(model) {
        const Layer& L = model.layers_[static_cast<size_t>(layer)];
        for (size_t i = 0; i < kWeightRoles; ++i) {
            if (L.w[i] != nullptr) views_[i] = *L.w[i];
        }
        item_ = model.layer_source_ == nullptr ? -1
                : model.layer_items_.empty()  ? layer
                                              : model.layer_items_[static_cast<size_t>(layer)];
        if (item_ < 0) {
            model.file_->begin_layer(layer);  // mmap path: advance the prefetch window (if any)
        } else {
            auto base = model.layer_source_->acquire(item_);
            if (!base) {
                status_ = base.status();
                return;
            }
            acquired_ = true;
            // Every block tensor lies inside the block's byte range, so its slot
            // address is the slot base plus its offset within that range.
            const size_t range_begin = model.file_->layer_range(layer).first;
            for (size_t i = 0; i < kWeightRoles; ++i) {
                if (L.w[i] != nullptr) views_[i].data = base.value() + (views_[i].file_offset - range_begin);
            }
        }
        if (L.w[kKvB] != nullptr) {
            // The old tensor stores [head][nope + value][rank]. Absorption
            // needs its key part transposed. Decode that part losslessly to
            // F32, preserving the original quantized value rows byte for
            // byte. Only this block's temporary copies exist, including under
            // layer streaming; no on-flash cache or model-sized copy is made.
            status_ = legacy_.bind(views_[kKvB], model.config_.mla, model.config_.n_head);
            if (!status_.is_ok()) return;
            views_[kKb] = legacy_.key();
            views_[kVb] = legacy_.value();
        }
    }
    ~BlockWeights() {
        if (acquired_) model_.layer_source_->release(item_);
    }
    BlockWeights(const BlockWeights&) = delete;
    BlockWeights& operator=(const BlockWeights&) = delete;

    [[nodiscard]] const Status& status() const noexcept { return status_; }
    // True when the views point into a reused streaming slot rather than the
    // mapping. Backends that keep per-tensor device copies (keyed by address)
    // must not see such views: the next block reuses the same address.
    [[nodiscard]] bool streamed() const noexcept { return item_ >= 0 && !model_.layer_shared_; }
    // The block's matrix with role `role` (only roles the block has).
    const TensorView& operator[](WeightRole role) const { return views_[role]; }

private:
    Transformer& model_;
    int32_t item_ = -1;  // layer source item, -1: read in place
    bool acquired_ = false;
    Status status_;
    TensorView views_[kWeightRoles];
    CombinedLatentWeights legacy_;
};

void Transformer::reset() noexcept {
    kv_->clear();
    for (RecurrentState& st : states_) {
        std::fill(st.conv.begin(), st.conv.end(), 0.0f);
        std::fill(st.ssm.begin(), st.ssm.end(), 0.0f);
    }
    std::fill(checkpoint_pos_.begin(), checkpoint_pos_.end(), -1);
    snapshots_.clear();
    spare_buffers_.clear();
    n_past_ = 0;
    tree_base_ = -1;
    tree_parent_.clear();
}

bool Transformer::has_position_state() const noexcept { return !states_.empty() || kv_->has_windowed_layers(); }

void Transformer::pin_snapshots() {
    if (!kv_->has_windowed_layers()) return;
    std::vector<int32_t> positions;
    positions.reserve(snapshots_.size());
    for (const StateSnapshot& snap : snapshots_) positions.push_back(snap.pos);
    kv_->pin(positions);
}

void Transformer::snapshot_state(bool pin) {
    if (!has_position_state() || n_past_ == 0) return;
    if (!snapshots_.empty() && snapshots_.back().pos == n_past_) {  // already held
        if (pin) {
            for (StateSnapshot& s : snapshots_) s.pinned = false;
            snapshots_.back().pinned = true;
        }
        return;
    }
    if (pin) {
        for (StateSnapshot& s : snapshots_) s.pinned = false;
    }
    StateSnapshot snap;
    if (static_cast<int32_t>(snapshots_.size()) >= kStateSnapshots) {
        // Keep the oldest and the pinned one; recycle the buffer of the
        // oldest other, so the newest ones stay (a draft's words and the
        // whole draft after them).
        size_t victim = 1;
        while (victim + 1 < snapshots_.size() && snapshots_[victim].pinned) ++victim;
        snap = std::move(snapshots_[victim]);
        snapshots_.erase(snapshots_.begin() + static_cast<std::ptrdiff_t>(victim));
    } else if (!spare_buffers_.empty()) {
        snap.data = std::move(spare_buffers_.back());
        spare_buffers_.pop_back();
    }
    snap.pinned = pin;
    snap.pos = n_past_;
    // The recurrent states are copied; the windowed blocks' window before
    // this position stays in the cache's pages, which the pin keeps (no copy:
    // it is usually the live window).
    snap.data.resize(recurrent_state_bytes() / sizeof(float));
    float* out = snap.data.data();
    for (const RecurrentState& st : states_) {
        if (!st.conv.empty()) out = std::copy(st.conv.begin(), st.conv.end(), out);
        if (!st.ssm.empty()) out = std::copy(st.ssm.begin(), st.ssm.end(), out);
    }
    snapshots_.push_back(std::move(snap));
    pin_snapshots();
}

namespace {

constexpr uint32_t kStateMagic = 0x5453594C;  // "LYST"
// 2: the recurrent-state snapshots follow the states (1 had none).
// 3: the key scales the pages were stored with follow the fingerprint.
// 4: per-block KV layout: the cache's layout fingerprint follows the
//    fingerprint, pages are rows images (the blocks keeping every position),
//    and the windowed blocks' pages (KvCache::window_pages_of() the end and
//    the snapshots kept: a count, then each page's index and window page
//    image) end the file. Older files hold every block's rows of every
//    position in one table's pages: they load converted
//    (KvCache::write_legacy_page()).
constexpr uint32_t kStateVersion = 4;

template <typename T>
void put(std::ostream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
bool get(std::istream& in, T& v) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof v));
}

}  // namespace

// What must match for a saved state to be valid here: the model (file size
// and shape) and how the KV cache stores rows. A LongRoPE model whose context
// picked the long factors rotates its keys otherwise than with the short
// ones, so that choice marks the first field. New per-block and scalar
// metadata also adds the file fingerprint; corrected tokenizer modes hash
// the complete header, including vocabulary and flags. Older layouts keep their state
// compatibility and bytes unchanged.
std::array<int64_t, 9> Transformer::state_fingerprint() const noexcept {
    const KvCacheConfig& kv = kv_->config();
    const bool long_rope = config_.rope_ctx_orig > 0 && context_length_ > config_.rope_ctx_orig;
    const int64_t file = static_cast<int64_t>(file_->file_size()) ^ (long_rope ? int64_t{1} << 62 : 0) ^
                         static_cast<int64_t>(variant_fingerprint_);
    return {file, config_.n_layers, config_.n_embd, config_.n_vocab, kv.n_layers, kv.n_head_kv, kv.head_dim + kv.auxiliary_dim,
            static_cast<int64_t>(kv_->dtype()), kv.page_tokens};
}

uint64_t Transformer::context_fingerprint() const noexcept {
    const std::array<int64_t, 9> fp = state_fingerprint();
    const uint64_t page = kv_->page_bytes();
    uint64_t seed = xxhash64(fp.data(), sizeof fp, xxhash64(&page, sizeof page));
    // Windowed blocks change what chunks hold and add their own images: the
    // layout is part of the key (0, and so no change, without them).
    if (const uint64_t layout = kv_->layout_fingerprint(); layout != 0) seed = xxhash64(&layout, sizeof layout, seed);
    return xxhash64(key_scale_.data(), key_scale_.size() * sizeof(float), seed);
}

std::vector<Transformer::KeptState> Transformer::kept_states() const {
    std::vector<KeptState> kept;
    if (!has_position_state() || n_past_ == 0) return kept;
    bool pinned_below = false;
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos < n_past_ && (&snap == &snapshots_.front() || snap.pinned)) {
            kept.push_back({snap.pos, snap.pinned});
            pinned_below = pinned_below || snap.pinned;
        }
    }
    kept.push_back({n_past_, !pinned_below});  // as read_state() pins the end
    return kept;
}

std::vector<std::span<const float>> Transformer::kept_state_parts(int32_t pos) const {
    std::vector<std::span<const float>> parts;
    if (states_.empty()) return parts;
    if (pos == n_past_) {
        for (const RecurrentState& st : states_) {
            if (!st.conv.empty()) parts.emplace_back(st.conv);
            if (!st.ssm.empty()) parts.emplace_back(st.ssm);
        }
        return parts;
    }
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos == pos && !snap.data.empty()) parts.emplace_back(snap.data);
    }
    return parts;
}

std::vector<int32_t> Transformer::snapshot_positions() const {
    std::vector<int32_t> out;
    for (const StateSnapshot& snap : snapshots_) out.push_back(snap.pos);
    return out;
}

void Transformer::begin_restore() {
    for (StateSnapshot& snap : snapshots_) spare_buffers_.push_back(std::move(snap.data));
    snapshots_.clear();
}

std::vector<std::span<float>> Transformer::recurrent_parts() {
    std::vector<std::span<float>> parts;
    for (RecurrentState& st : states_) {
        if (!st.conv.empty()) parts.emplace_back(st.conv);
        if (!st.ssm.empty()) parts.emplace_back(st.ssm);
    }
    return parts;
}

Transformer::StateBuffer Transformer::take_snapshot_buffer() {
    StateBuffer buffer;
    if (!spare_buffers_.empty()) {
        buffer = std::move(spare_buffers_.back());
        spare_buffers_.pop_back();
    }
    buffer.resize(recurrent_state_bytes() / sizeof(float));  // not zero-filled (StateBuffer)
    return buffer;
}

Status Transformer::write_state(std::ostream& out) const {
    if (kv_->window() > 0) return Status(ErrorCode::Unsupported, "a sliding-window context cannot be saved");
    // The snapshots below the end (one at the end is the states above), so a
    // restored context rewinds where the live one would: the next prompt
    // re-sends the last reply as text, which may encode differently, and
    // without the last prompt's snapshot a hybrid model would keep nothing.
    // The oldest (the system prompt's end) and the pinned one are enough; a
    // draft's is not worth its size (62.8 MiB each on a 35B).
    std::vector<SnapshotImage> kept;
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos < n_past_ && (&snap == &snapshots_.front() || snap.pinned)) {
            kept.push_back({snap.pos, snap.pinned, snap.data});
        }
    }
    // A page spread over several pool pages is copied out; otherwise written in place.
    std::vector<uint8_t> scratch(kv_->page_bytes());
    auto page = [this, &scratch](int32_t p) -> const uint8_t* {
        if (const uint8_t* data = kv_->page_data(p)) return data;
        if (!kv_->read_rows(p, kv_->config().page_tokens, scratch.data()).is_ok()) return nullptr;
        return scratch.data();
    };
    return write_image(out, n_past_, page, nullptr, {}, kept, true);
}

Status Transformer::write_state(std::ostream& out, const ContextImage& image) const {
    if (kv_->window() > 0) return Status(ErrorCode::Unsupported, "a sliding-window context cannot be saved");
    const size_t floats = recurrent_state_bytes() / sizeof(float);
    const bool windowed = kv_->has_windowed_layers();
    if (image.n < 0 || image.n > context_length_ || !image.page || image.state.size() != (states_.empty() ? 0 : floats) ||
        static_cast<int32_t>(image.snapshots.size()) >= kStateSnapshots ||
        (states_.empty() && !image.snapshots.empty()) || (windowed && (!image.window_page || !image.snapshots.empty()))) {
        return Status(ErrorCode::InvalidArgument, "the context image does not fit this model");
    }
    int32_t below = 0;
    for (const SnapshotImage& snap : image.snapshots) {
        if (snap.pos <= below || snap.pos >= image.n || snap.data.size() != floats) {
            return Status(ErrorCode::InvalidArgument, "the context image's snapshots do not fit this model");
        }
        below = snap.pos;
    }
    return write_image(out, image.n, image.page, windowed ? &image.window_page : nullptr, image.state, image.snapshots,
                       false);
}

// The body of both write_state()s: `state` holds the recurrent states at n as
// one image, or `live` takes them from states_ (no copy of the live ones).
// The windowed blocks' pages come from the live cache, or from `window_page`
// (the windowed rows of a logical page, KvCache::read_window_rows()'s image).
Status Transformer::write_image(std::ostream& out, int32_t n, const std::function<const uint8_t*(int32_t)>& page,
                                const std::function<const uint8_t*(int32_t)>* window_page,
                                std::span<const float> state, std::span<const SnapshotImage> snapshots,
                                bool live) const {
    put(out, kStateMagic);
    put(out, kStateVersion);
    for (const int64_t v : state_fingerprint()) put(out, v);
    put(out, kv_->layout_fingerprint());
    put(out, static_cast<uint64_t>(key_scale_.size()));
    out.write(reinterpret_cast<const char*>(key_scale_.data()),
              static_cast<std::streamsize>(key_scale_.size() * sizeof(float)));
    put(out, n);
    const int32_t pages = (n + kv_->config().page_tokens - 1) / kv_->config().page_tokens;
    put(out, static_cast<uint64_t>(kv_->page_bytes()));
    put(out, pages);
    for (int32_t p = 0; p < pages && kv_->page_bytes() > 0; ++p) {
        const uint8_t* data = page(p);
        if (data == nullptr) return Status(ErrorCode::Internal, "a cached page is not mapped");
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(kv_->page_bytes()));
    }
    put(out, static_cast<int32_t>(states_.size()));
    const float* from = state.data();
    for (const RecurrentState& st : states_) {
        put(out, static_cast<uint64_t>(st.conv.size()));
        put(out, static_cast<uint64_t>(st.ssm.size()));
        // ShortConv has no matrix state; a one-tap convolution has no
        // history either. Keep their zero lengths in the format without
        // passing null buffers to streams or doing pointer arithmetic on them.
        if (!st.conv.empty()) {
            const float* conv = live ? st.conv.data() : from;
            out.write(reinterpret_cast<const char*>(conv), static_cast<std::streamsize>(st.conv.size() * sizeof(float)));
        }
        if (!st.ssm.empty()) {
            const float* ssm = live ? st.ssm.data() : from + st.conv.size();
            out.write(reinterpret_cast<const char*>(ssm), static_cast<std::streamsize>(st.ssm.size() * sizeof(float)));
        }
        if (!live && (!st.conv.empty() || !st.ssm.empty())) from += st.conv.size() + st.ssm.size();
    }
    put(out, static_cast<int32_t>(snapshots.size()));
    for (const SnapshotImage& snap : snapshots) {
        put(out, snap.pos);
        put(out, static_cast<uint8_t>(snap.pinned));
        put(out, static_cast<uint64_t>(snap.data.size()));
        if (!snap.data.empty()) {
            out.write(reinterpret_cast<const char*>(snap.data.data()),
                      static_cast<std::streamsize>(snap.data.size() * sizeof(float)));
        }
    }
    // Windowed blocks: every page of the window before the end and before
    // each snapshot kept, whole (out of the live pages, or the image's window
    // pages), so a loaded context returns where the saved one would.
    std::vector<int32_t> pins;
    for (const SnapshotImage& snap : snapshots) pins.push_back(snap.pos);
    const std::vector<int32_t> window = kv_->window_pages_of(n, pins);
    put(out, static_cast<int32_t>(window.size()));
    std::vector<uint8_t> scratch(live ? kv_->window_page_bytes() : 0);
    const int32_t t = kv_->config().page_tokens;
    for (const int32_t q : window) {
        const uint8_t* image = nullptr;
        if (live) {
            LIYAB_RETURN_IF_ERROR(kv_->read_window_rows(q, std::min(t, n - q * t), scratch.data()));
            image = scratch.data();
        } else if (window_page == nullptr || (image = (*window_page)(q)) == nullptr) {
            return Status(ErrorCode::InvalidArgument, "a window page of the context image is missing");
        }
        put(out, q);
        out.write(reinterpret_cast<const char*>(image), static_cast<std::streamsize>(kv_->window_page_bytes()));
    }
    return out ? Status::ok() : Status(ErrorCode::IoError, "cannot write the context state");
}

Status Transformer::read_state(std::istream& in) {
    reset();
    auto fail = [this](ErrorCode code, const char* why) {
        reset();
        return Status(code, why);
    };
    uint32_t magic = 0, version = 0;
    if (!get(in, magic) || !get(in, version) || magic != kStateMagic || version < 1 || version > kStateVersion) {
        return fail(ErrorCode::InvalidArgument, "not a Liyab context state (or another version)");
    }
    for (const int64_t expected : state_fingerprint()) {
        int64_t v = 0;
        if (!get(in, v) || v != expected) {
            return fail(ErrorCode::InvalidArgument, "the state was saved for another model or other cache settings");
        }
    }
    // Before version 4 every block kept every position in one table's pages:
    // such a file converts to any layout. A version 4 file holds what its
    // own layout keeps, so it must be this one.
    const bool legacy = version < 4;
    if (!legacy) {
        uint64_t layout = 0;
        if (!get(in, layout) || layout != kv_->layout_fingerprint()) {
            return fail(ErrorCode::InvalidArgument,
                        "the state was saved with other windowed attention blocks (another context length or "
                        "segment reuse setting)");
        }
    }
    // Key scales the pages were stored with (none before version 3).
    std::vector<float> stored;
    if (version >= 3) {
        uint64_t count = 0;
        if (!get(in, count) || (count != 0 && count != key_scaling_shape().factors())) {
            return fail(ErrorCode::InvalidArgument, "the state's key scales do not match");
        }
        stored.resize(static_cast<size_t>(count));
        if (!in.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(count * sizeof(float)))) {
            return fail(ErrorCode::IoError, "the state's key scales are truncated");
        }
    }
    int32_t n = 0, pages = 0;
    uint64_t page_bytes = 0;
    const int32_t t = kv_->config().page_tokens;
    if (!get(in, n) || !get(in, page_bytes) || !get(in, pages) || n < 0 || n > context_length_ ||
        page_bytes != (legacy ? kv_->legacy_page_bytes() : kv_->page_bytes()) || pages != (n + t - 1) / t) {
        return fail(ErrorCode::InvalidArgument, "the state's cache layout does not match");
    }
    // Every page of the blocks keeping every position, the windowed blocks'
    // pages of the window before n (a snapshot's are mapped when it is read).
    if (Status s = kv_->reserve_context(n, {}); !s.is_ok()) return fail(s.code(), "cannot map the state's cache pages");
    const bool convert = legacy && kv_->has_windowed_layers();
    std::vector<uint8_t> buffer;
    for (int32_t p = 0; p < pages && page_bytes > 0; ++p) {
        uint8_t* data = convert ? nullptr : kv_->mutable_page_data(p);
        if (data == nullptr) {
            buffer.resize(static_cast<size_t>(page_bytes));
            data = buffer.data();
        }
        if (!in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(page_bytes))) {
            return fail(ErrorCode::IoError, "the state's cache pages are truncated");
        }
        if (data != buffer.data()) continue;  // read in place
        const Status written = convert ? kv_->write_legacy_page(p, data) : kv_->write_rows(p, data, t, 0, t);
        if (!written.is_ok()) return fail(ErrorCode::InvalidArgument, "the state's cache pages do not fit");
    }
    int32_t count = 0;
    if (!get(in, count) || count != static_cast<int32_t>(states_.size())) {
        return fail(ErrorCode::InvalidArgument, "the state's recurrent blocks do not match");
    }
    for (RecurrentState& st : states_) {
        uint64_t conv = 0, ssm = 0;
        if (!get(in, conv) || !get(in, ssm) || conv != st.conv.size() || ssm != st.ssm.size() ||
            (conv > 0 && !in.read(reinterpret_cast<char*>(st.conv.data()), static_cast<std::streamsize>(conv * sizeof(float)))) ||
            (ssm > 0 && !in.read(reinterpret_cast<char*>(st.ssm.data()), static_cast<std::streamsize>(ssm * sizeof(float))))) {
            return fail(ErrorCode::IoError, "the state's recurrent states are truncated");
        }
    }
    int32_t snapshots = 0;
    if (version >= 2 && (!get(in, snapshots) || snapshots < 0 || snapshots >= kStateSnapshots)) {
        return fail(ErrorCode::InvalidArgument, "the state's snapshots do not match");
    }
    const size_t floats = recurrent_state_bytes() / sizeof(float);
    int32_t dropped = 0;
    for (int32_t i = 0; i < snapshots; ++i) {
        StateSnapshot snap;
        uint8_t pinned = 0;
        uint64_t size = 0;
        if (!get(in, snap.pos) || !get(in, pinned) || !get(in, size) || size != floats || snap.pos <= 0 ||
            snap.pos >= n || (!snapshots_.empty() && snap.pos <= snapshots_.back().pos)) {
            return fail(ErrorCode::InvalidArgument, "the state's snapshots do not match");
        }
        snap.pinned = pinned != 0;
        snap.data.resize(floats);
        if (size > 0 && !in.read(reinterpret_cast<char*>(snap.data.data()), static_cast<std::streamsize>(size * sizeof(float)))) {
            return fail(ErrorCode::IoError, "the state's snapshots are truncated");
        }
        if (convert) {
            // An older file kept no window of its own for a snapshot below
            // the end: its rows were in pages released here. The context
            // loads without it (it rewinds less far).
            ++dropped;
            continue;
        }
        snapshots_.push_back(std::move(snap));
    }
    // Windowed blocks (version 4): the pages of the window before n and
    // before each snapshot, whole, exactly those (older files had those rows
    // in their pages).
    if (!legacy) {
        std::vector<int32_t> pins;
        for (const StateSnapshot& snap : snapshots_) pins.push_back(snap.pos);
        if (Status s = kv_->reserve_context(n, pins); !s.is_ok()) return fail(s.code(), "cannot map the state's pages");
        const std::vector<int32_t> expected = kv_->window_pages_of(n, pins);
        int32_t count_pages = 0;
        if (!get(in, count_pages) || count_pages != static_cast<int32_t>(expected.size())) {
            return fail(ErrorCode::InvalidArgument, "the state's windowed blocks do not match");
        }
        buffer.resize(kv_->window_page_bytes());
        for (const int32_t q : expected) {
            int32_t at = 0;
            if (!get(in, at) || at != q) return fail(ErrorCode::InvalidArgument, "the state's windowed blocks do not match");
            if (!in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                return fail(ErrorCode::IoError, "the state's window pages are truncated");
            }
            if (!kv_->write_window_rows(q, buffer.data(), 0, std::min(t, n - q * t)).is_ok()) {
                return fail(ErrorCode::InvalidArgument, "the state's window pages do not fit");
            }
        }
    }
    // Keys stored with other scales (an older file: none, i.e. 1) are brought
    // to the ones in use: k_now = k_stored * t_stored / t_now, requantized. The
    // values and the rest of the context are unchanged.
    if (stored != key_scale_) {
        std::vector<float> ratio(key_scaling_shape().factors());
        for (size_t i = 0; i < ratio.size(); ++i) {
            ratio[i] = (stored.empty() ? 1.0f : stored[i]) / (key_scale_.empty() ? 1.0f : key_scale_[i]);
        }
        if (Status s = kv_->scale_keys(n, ratio); !s.is_ok()) return fail(s.code(), "cannot convert the state's keys");
        LIYAB_LOG_INFO("state: keys of %d positions converted to this model's key scales", n);
    }
    if (convert) {
        LIYAB_LOG_INFO("state: a version %u file converted to per-block KV (windowed blocks keep their window)%s",
                       version, dropped > 0 ? "; its snapshots below the end were dropped" : "");
    }
    n_past_ = n;
    // One at the end too, so a draft typed ahead can be dropped back to it. It
    // is where the next prompt continues: pinned, unless the last prompt's end
    // came back pinned (a state saved after a reply).
    snapshot_state(/*pin=*/std::none_of(snapshots_.begin(), snapshots_.end(),
                                         [](const StateSnapshot& snap) { return snap.pinned; }));
    return Status::ok();
}

int32_t Transformer::restorable_prefix(int32_t n) const noexcept {
    n = std::clamp(n, 0, n_past_);
    if (n == n_past_) return n;
    // Windowed blocks: only where their window's rows are still kept (the
    // rollback margin, the snapshots' pins).
    const bool windowed = kv_->has_windowed_layers();
    if (!config_.hybrid()) return windowed ? kv_->restorable(n) : n;
    auto kept = [&](int32_t m) { return !windowed || kv_->restorable(m) == m; };
    int32_t best = 0;
    for (const int32_t p : checkpoint_pos_) {  // a checkpoint at position p restores n = p + 1
        if (p >= 0 && p + 1 <= n && p + 1 > best && kept(p + 1)) best = p + 1;
    }
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos <= n && snap.pos > best && kept(snap.pos)) best = snap.pos;
    }
    return best;
}

void Transformer::set_rollback_window(int32_t positions) {
    rollback_window_ = states_.empty() ? 0 : std::max(0, positions);
    checkpoint_pos_.assign(static_cast<size_t>(rollback_window_), -1);
    for (RecurrentState& st : states_) {
        st.checkpoints.assign(static_cast<size_t>(rollback_window_) * (st.conv.size() + st.ssm.size()), 0.0f);
        st.checkpoints.shrink_to_fit();
    }
}
void Transformer::rope(float* vec, int32_t n_heads, int32_t pos, const RopeTable& table) const {
    rope(vec, n_heads, static_cast<size_t>(table.head_dim), pos, table);
}

void Transformer::rope(float* vec, int32_t n_heads, size_t stride, int32_t pos, const RopeTable& table) const {
    rope(vec, n_heads, stride, pos, table, config_.spec->rope);
}

void Transformer::rope(float* vec, int32_t n_heads, size_t stride, int32_t pos, const RopeTable& table,
                       RopeLayout layout) const {
    const int32_t half = static_cast<int32_t>(table.inv_freq.size());
    const bool halves = layout == RopeLayout::Halves;
    const float p = static_cast<float>(pos) / table.scale;
    for (int32_t h = 0; h < n_heads; ++h) {
        float* x = vec + static_cast<size_t>(h) * stride;
        for (int32_t i = 0; i < half; ++i) {
            const float theta = p * table.inv_freq[static_cast<size_t>(i)];
            const float cs = std::cos(theta) * table.magnitude;  // x 1 keeps the bits
            const float sn = std::sin(theta) * table.magnitude;
            const int32_t a = halves ? i : 2 * i;
            const int32_t b = halves ? i + half : 2 * i + 1;
            const float x0 = x[a];
            const float x1 = x[b];
            x[a] = x0 * cs - x1 * sn;
            x[b] = x0 * sn + x1 * cs;
        }
    }
}

// Attention over the KV cache, for n query tokens. The query heads sharing a
// KV head (grouped query attention: 8 per KV head on Qwen3.6) are scored
// together, so each K and V row is fetched and decoded once for the whole
// group, not once per query head. A token's visible positions are cut into
// equal spans (kAttentionMinSpan, kAttentionMaxSpans); each span keeps a
// partial softmax (its maximum, sum and weighted V) and the spans are merged
// in order (log-sum-exp). The cut follows the token's own position only,
// never the batch, the threads or the load: a token's attention rounds the
// same however its prompt was split (typing ahead, chunked prefill,
// speculative verification) and however many cores the power manager left
// active. Before, the spans were sized for the active threads and the batch,
// and the same prompt could give two different greedy replies on a cool and a
// warm phone. Spans of a fixed 256 positions were deterministic too, but left
// a 1000-position decoding step 8 tasks for 8 threads: 33% slower there; the
// span bounds (transformer.h) keep the tasks even among the threads.
//
// How the spans are scheduled changes no number. Decoding, where one token
// has only n_head_kv groups, scores every span as a task of its own so every
// thread works (as in flash decoding), then merges. A batch with enough
// (token, KV head) pairs scores each pair's spans on one thread and merges
// them right away, which needs no partial results for the whole batch (a
// 1024-token pass at a 20k context would need over a GB of them). On a
// 2838-token context the per-head loop this replaced took 48 ms of a 161 ms
// decode step.
//
// A windowed block (`window` > 0) sees, of what the cache keeps visible, only
// the positions [pos - window + 1, pos]: the sinks drop out once the window
// has passed them. Its spans follow its own visible count, so they too depend
// only on the position. Its rows live in the cache's windowed group, which
// keeps only those positions (KvCache's per-block layout), so a windowed block
// saves memory as well as compute (with TransformerOptions::keep_windowed_rows,
// compute only).
void Transformer::attention(const Layer& L, int32_t n, ThreadPool& pool, const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(&L - layers_.data())];
    const int32_t kv_slot = L.kv_slot, window = L.window;
    const float* sinks = opt(L.sinks);
    // Each segment's rows of the block, their group and offsets resolved
    // once for the pass (a pass of several sequences reads several caches).
    row_views_.clear();
    bool all_rows = true;  // every segment attends here (not latent attention's expanded ones)
    for (const Segment& s : segs_) {
        row_views_.push_back(s.kv->row_view(kv_slot));
        all_rows = all_rows && !s.expanded;
    }
    // The rows attending: every row of the batch, or the absorbed segments'
    // rows of a latent pass that also has expanded ones (attn_rows_). Tasks
    // and partial results are indexed by a row's place among them.
    int32_t m = n;
    if (!all_rows) {
        attn_rows_.clear();
        for (const Segment& s : segs_) {
            if (s.expanded) continue;
            for (int32_t t = s.first; t < s.first + s.n; ++t) attn_rows_.push_back(t);
        }
        m = static_cast<int32_t>(attn_rows_.size());
        if (m == 0) return;
    }
    const int32_t* row_list = all_rows ? nullptr : attn_rows_.data();
    auto row_at = [row_list](int64_t i) { return row_list == nullptr ? static_cast<int32_t>(i) : row_list[i]; };
    const int32_t hd = plan.head_dim;
    const int32_t group = plan.n_head / plan.n_head_kv;
    // Values may be narrower than keys (latent attention: the latent part of
    // each key row); every value-sized buffer below follows value_dim.
    const int32_t vd = plan.value_dim;
    const auto uvd = static_cast<size_t>(vd);
    const size_t out_dim = static_cast<size_t>(plan.n_head) * uvd;  // att_ row: every head's output
    const float scale = plan.attn_scale;  // 1/sqrt(head_dim) unless the file states another (ModelConfig)
    const float softcap = c.attn_softcap;
    const DType kv_type = segs_.front().kv->dtype();  // every context's cache has the same format
    const bool quantized_kv = quant::is_block_quantized(kv_type);
    const auto uhd = static_cast<size_t>(hd);
    const auto ugroup = static_cast<size_t>(group);
    const auto n_head = static_cast<size_t>(plan.n_head);
    const size_t q8_per_head = uhd / quant::kBlock;

    // Quantized caches are scored with integer dot products against a Q8_0
    // copy of each query (same kernels as the weight matmuls), made once per
    // (token, head) rather than in every span.
    if (quantized_kv) {
        const auto rows = static_cast<int64_t>(m) * plan.n_head;
        q8_queries_.resize(static_cast<size_t>(n) * n_head * q8_per_head);
        auto quantize = [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r) {
                const size_t row =
                    static_cast<size_t>(row_at(r / plan.n_head)) * n_head + static_cast<size_t>(r % plan.n_head);
                quant::quantize_row_q8_0(q_.data() + row * uhd, q8_queries_.data() + row * q8_per_head, hd);
            }
        };
        if (rows >= 256) {
            pool.parallel_for(rows, quantize);
        } else {
            quantize(0, rows);
        }
    }
    auto spans_for = [](int32_t n_visible) {
        return std::clamp(n_visible / kAttentionMinSpan, 1, kAttentionMaxSpans);
    };
    // Positions a query of segment s at `pos` sees: [sink_begin, sink_end) ∪
    // [first, last_of(s, pos)]. Causal attention ends at the query; a
    // bidirectional model's (ModelConfig::causal false) at the last position
    // of the segment's pass, and its windowed blocks see the positions at
    // most window / 2 away on either side (EmbeddingGemma: 256 of a 512
    // window, as the reference's |q - k| < 512 / 2 + 1).
    const bool causal = c.causal;
    auto seen = [&](const Segment& s, int32_t pos) {
        return seen_by(*s.kv, pos, causal || window <= 0 ? window : window / 2 + 1);
    };
    auto last_of = [&](const Segment& s, int32_t pos) {
        if (causal) return pos;
        const int32_t end = s.n_past + s.n - 1;
        return window > 0 ? std::min(end, pos + window / 2) : end;
    };
    const bool indexed = !index_positions_.empty();
    auto spans_of = [&](int32_t t) {
        if (indexed) return spans_for(static_cast<int32_t>(index_positions_[static_cast<size_t>(t)].size()));
        const Segment& s = segs_[segment_of(t)];
        const int32_t pos = pos_of(s, t);
        const Seen v = seen(s, pos);
        return spans_for(v.sink_end - v.sink_begin + (last_of(s, pos) - v.first + 1));
    };

    // Span sp of token t for the query heads of KV head g. For head j of the
    // group: the span's maximum score mx[j * stride], the sum of exp(score -
    // maximum) sm[j * stride] and the V rows weighted by them at out + j *
    // stride * vd (stride: span slots per head).
    auto score_span = [&](int32_t t, int32_t g, int32_t sp, float* mx, float* sm, float* out, size_t stride) {
        thread_local std::vector<float> scores;  // [group][span length]
        thread_local std::vector<float> vrow;    // one V row, decoded (vd values)
        thread_local std::vector<float> dots;    // one K row against the group's queries
        thread_local std::vector<const quant::BlockQ8_0*> queries;  // the group's Q8_0 queries
        const size_t si = segment_of(t);
        const Segment& s = segs_[si];
        const KvCache::RowView& rows = row_views_[si];
        const int32_t pos = pos_of(s, t);
        // Visible positions: attention sinks [sink_begin, sink_end) + [first, pos].
        const Seen vis = seen(s, pos);
        const int32_t n_sinks = vis.sink_end - vis.sink_begin;
        const int32_t n_visible = indexed ? static_cast<int32_t>(index_positions_[static_cast<size_t>(t)].size())
                                          : n_sinks + (last_of(s, pos) - vis.first + 1);
        // A tree pass reads the rows of positions n_past + d (the batch's
        // own) from the slot of the token's ancestor at depth d: its branch.
        const int32_t* ancestors =
            tree_depth_.empty() ? nullptr : tree_ancestors_.data() + static_cast<size_t>(t - s.first) * tree_stride_;
        // Visible position i, and the cache slot holding its rows.
        auto logical = [&](int32_t i) { return i < n_sinks ? vis.sink_begin + i : vis.first + (i - n_sinks); };
        auto position = [&](int32_t i) {
            if (indexed) return index_positions_[static_cast<size_t>(t)][static_cast<size_t>(i)];
            const int32_t p = logical(i);
            return ancestors == nullptr || p < s.n_past ? p : s.n_past + ancestors[p - s.n_past];
        };
        const int32_t spans = spans_for(n_visible);
        const auto i0 = static_cast<int32_t>(int64_t{n_visible} * sp / spans);
        const auto i1 = static_cast<int32_t>(int64_t{n_visible} * (sp + 1) / spans);
        const auto len = static_cast<size_t>(std::max(0, i1 - i0));
        const int32_t h0 = g * group;
        const size_t first_row = static_cast<size_t>(t) * n_head + static_cast<size_t>(h0);  // the group's queries
        scores.resize(ugroup * std::max<size_t>(len, 1));
        if (kv_type == DType::Q8_0) {
            // A Q8_0 cache (the default): the group's queries against each K
            // row in multi-row passes, the same bits as one dot per head
            // (dot_q8_0_q8_0_rows) for about half the instructions: the row
            // is loaded once, and 4 heads share every vector multiply-add.
            queries.resize(ugroup);
            dots.resize(ugroup);
            for (size_t j = 0; j < ugroup; ++j) queries[j] = q8_queries_.data() + (first_row + j) * q8_per_head;
            for (size_t i = 0; i < len; ++i) {
                const auto* k = reinterpret_cast<const quant::BlockQ8_0*>(
                    rows.k(position(i0 + static_cast<int32_t>(i)), g));
                quant::dot_q8_0_q8_0_rows(k, queries.data(), group, hd, dots.data());
                for (size_t j = 0; j < ugroup; ++j) scores[j * len + i] = dots[j] * scale;
            }
        } else {
            for (size_t i = 0; i < len; ++i) {
                const uint8_t* k = rows.k(position(i0 + static_cast<int32_t>(i)), g);
                for (int32_t j = 0; j < group; ++j) {
                    const size_t row = first_row + static_cast<size_t>(j);
                    scores[static_cast<size_t>(j) * len + i] =
                        (quantized_kv
                             ? quant::dot_quantized(kv_type, k, q8_queries_.data() + row * q8_per_head, nullptr, hd)
                             : quant::dot_f16_f32(reinterpret_cast<const uint16_t*>(k), q_.data() + row * uhd, hd)) *
                        scale;
                }
            }
        }
        if (!alibi_slopes_.empty()) {
            // ALiBi: each head's slope times the key's distance behind the
            // query, from logical positions (a tree pass's ancestors, the
            // sinks, any cache slot), so cached keys never depend on where they sit.
            for (size_t i = 0; i < len; ++i) {
                const auto dist = static_cast<float>(logical(i0 + static_cast<int32_t>(i)) - pos);
                for (size_t j = 0; j < ugroup; ++j) scores[j * len + i] += alibi_slopes_[static_cast<size_t>(h0) + j] * dist;
            }
        }
        if (softcap > 0.0f) {  // Gemma 2: the scaled scores squashed into (-cap, cap) before the softmax
            for (size_t i = 0; i < ugroup * len; ++i) scores[i] = softcap * portable_tanh(scores[i] / softcap);
        }
        for (int32_t j = 0; j < group; ++j) {
            const size_t at = static_cast<size_t>(j) * stride;
            float* o = out + at * uvd;
            std::fill(o, o + vd, 0.0f);
            mx[at] = -INFINITY;
            sm[at] = 0.0f;
            if (len == 0 || (head_mask != nullptr && head_mask[h0 + j] == 0)) continue;
            float* sc = scores.data() + static_cast<size_t>(j) * len;
            float m = -INFINITY;
            for (size_t i = 0; i < len; ++i) m = std::max(m, sc[i]);
            float sum = 0.0f;
            for (size_t i = 0; i < len; ++i) sum += sc[i] = std::exp(sc[i] - m);
            mx[at] = m;
            sm[at] = sum;
        }
        // V: each row decoded once into floats, then added to every head of
        // the group (decoding it per head cost 8x the work).
        vrow.resize(uvd);
        for (size_t i = 0; i < len; ++i) {
            const uint8_t* v = rows.v(position(i0 + static_cast<int32_t>(i)), g);
            if (kv_type == DType::F32) {
                std::copy_n(reinterpret_cast<const float*>(v), uvd, vrow.data());
            } else {
                quant::dequantize_row(kv_type, v, vrow.data(), vd);
            }
            for (int32_t j = 0; j < group; ++j) {
                if (head_mask != nullptr && head_mask[h0 + j] == 0) continue;
                const float w = scores[static_cast<size_t>(j) * len + i];
                float* o = out + static_cast<size_t>(j) * stride * uvd;
                for (size_t d = 0; d < uvd; ++d) o[d] += w * vrow[d];
            }
        }
    };
    // Token t's spans of head h (mx[sp], sm[sp], out + sp * vd), merged in
    // order into att_: exp(m_s - M) rescales each span's sum and weighted V
    // to the overall maximum M. A sink logit s_h (gpt-oss) is one more score
    // of the softmax, with no value: it joins the maximum and adds exp(s_h -
    // M) to the denominator only. It is the same for every position, so it
    // is added once here rather than per span: the spans, and therefore how a
    // token's attention rounds, stay those of a model without sinks.
    auto merge = [&](int32_t t, int32_t h, int32_t spans, const float* mx, const float* sm, const float* out) {
        float* dst = att_.data() + static_cast<size_t>(t) * out_dim + static_cast<size_t>(h) * uvd;
        std::fill(dst, dst + vd, 0.0f);
        if (head_mask != nullptr && head_mask[h] == 0) return;  // pruned head: contributes nothing
        float m = -INFINITY;
        for (int32_t sp = 0; sp < spans; ++sp) m = std::max(m, mx[sp]);
        if (sinks != nullptr) m = std::max(m, sinks[h]);
        float sum = 0.0f;
        for (int32_t sp = 0; sp < spans; ++sp) {
            if (sm[sp] == 0.0f) continue;
            const float w = std::exp(mx[sp] - m);
            sum += w * sm[sp];
            const float* src = out + static_cast<size_t>(sp) * uvd;
            for (size_t d = 0; d < uvd; ++d) dst[d] += w * src[d];
        }
        if (sinks != nullptr) sum += std::exp(sinks[h] - m);
        const float inv = 1.0f / sum;
        for (size_t d = 0; d < uvd; ++d) dst[d] *= inv;
    };

    // A segment's last token sees the most of its positions; in a tree
    // pass, the deepest.
    int32_t max_spans = 1;
    for (const Segment& s : segs_) {
        if (!s.expanded) max_spans = std::max(max_spans, spans_of(s.first + s.n - 1));
    }
    for (int32_t t = 0; !tree_depth_.empty() && t + 1 < n; ++t) max_spans = std::max(max_spans, spans_of(t));
    for (int32_t t = 0; !causal && t < m; ++t) max_spans = std::max(max_spans, spans_of(row_at(t)));  // any row may see the most
    const int64_t pairs = int64_t{m} * plan.n_head_kv;
    if (max_spans == 1 || pairs >= int64_t{pool.max_threads()} * ThreadPool::kChunksPerThread) {
        pool.parallel_for(pairs, [&](int64_t begin, int64_t end) {
            thread_local std::vector<float> mx, sm, out;  // [head of the group][span]
            for (int64_t task = begin; task < end; ++task) {
                const int32_t t = row_at(task / plan.n_head_kv);
                const auto g = static_cast<int32_t>(task % plan.n_head_kv);
                const int32_t spans = spans_of(t);
                const auto uspans = static_cast<size_t>(spans);
                mx.resize(ugroup * uspans);
                sm.resize(ugroup * uspans);
                out.resize(ugroup * uspans * uvd);
                for (int32_t sp = 0; sp < spans; ++sp) {
                    const auto usp = static_cast<size_t>(sp);
                    score_span(t, g, sp, mx.data() + usp, sm.data() + usp, out.data() + usp * uvd, uspans);
                }
                for (int32_t j = 0; j < group; ++j) {
                    const size_t at = static_cast<size_t>(j) * uspans;
                    merge(t, g * group + j, spans, mx.data() + at, sm.data() + at, out.data() + at * uvd);
                }
            }
        });
        return;
    }
    // Partial results per (attending row, query head, span slot).
    const auto slots = static_cast<size_t>(max_spans);
    part_max_.resize(static_cast<size_t>(m) * n_head * slots);
    part_sum_.resize(part_max_.size());
    part_out_.resize(part_max_.size() * uvd);
    auto base = [&](int64_t i, int32_t h) { return (static_cast<size_t>(i) * n_head + static_cast<size_t>(h)) * slots; };
    pool.parallel_for(pairs * max_spans, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const auto sp = static_cast<int32_t>(task % max_spans);
            const int64_t i = task / max_spans / plan.n_head_kv;
            const int32_t t = row_at(i);
            const auto g = static_cast<int32_t>(task / max_spans % plan.n_head_kv);
            if (sp >= spans_of(t)) continue;  // an earlier token of the batch sees fewer positions
            const size_t at = base(i, g * group) + static_cast<size_t>(sp);
            score_span(t, g, sp, part_max_.data() + at, part_sum_.data() + at, part_out_.data() + at * uvd, slots);
        }
    });
    // Merging many spans is worth a second fork/join (at 3000 positions on
    // Qwen3.6's shape, merging 84 spans on one thread cost 5% of the step).
    constexpr int32_t kParallelMergeSpans = 32;
    auto merge_range = [&](int64_t begin, int64_t end) {
        for (int64_t th = begin; th < end; ++th) {
            const int64_t i = th / plan.n_head;
            const int32_t t = row_at(i);
            const auto h = static_cast<int32_t>(th % plan.n_head);
            const size_t at = base(i, h);
            merge(t, h, spans_of(t), part_max_.data() + at, part_sum_.data() + at, part_out_.data() + at * uvd);
        }
    };
    if (max_spans >= kParallelMergeSpans) {
        pool.parallel_for(int64_t{m} * plan.n_head, merge_range);
    } else {
        merge_range(0, int64_t{m} * plan.n_head);
    }
}

Status Transformer::matmul(const Route& route, Backend* backend, const TensorView& w, const float* x, float* y,
                           int32_t n) {
    if (adapters_in_pass_) {
        const TensorView* ws[] = {&w};
        float* ys[] = {y};
        return matmul_group(route, backend, ws, x, ys, n);
    }
    Status s = backend->matmul(w, x, y, n);
    if (s.code() == ErrorCode::Unsupported && backend != route.cpu) s = route.cpu->matmul(w, x, y, n);
    return s;
}

Status Transformer::matmul_group(const Route& route, Backend* backend, std::span<const TensorView* const> ws,
                                 const float* x, std::span<float* const> ys, int32_t n) {
    if (adapters_in_pass_ && collect_adapter_items(ws, x, ys, n)) {
        // The A·x matmuls go with the base ones on the CPU backend (one
        // parallel pass); an accelerator's base products land in host memory
        // like the CPU's, and its A·x run on the CPU after them.
        adapter_batch_.clear();
        Status s = Status(ErrorCode::Unsupported, "");
        if (backend != route.cpu) {
            s = backend->matmul_group(ws, x, ys, n);
            if (s.is_ok()) {
                adapter_runner_.add_a_matmuls(adapter_items_, adapter_batch_);
                s = route.cpu->matmul_batch(adapter_batch_);
            }
        }
        if (s.code() == ErrorCode::Unsupported) {
            adapter_batch_.clear();
            for (size_t i = 0; i < ws.size(); ++i) adapter_batch_.push_back({ws[i], x, ys[i], n});
            adapter_runner_.add_a_matmuls(adapter_items_, adapter_batch_);
            s = route.cpu->matmul_batch(adapter_batch_);
        }
        if (s.is_ok()) adapter_runner_.add_updates(adapter_items_, adapter_pool_);
        return s;
    }
    Status s = backend->matmul_group(ws, x, ys, n);
    if (s.code() == ErrorCode::Unsupported && backend != route.cpu) s = route.cpu->matmul_group(ws, x, ys, n);
    return s;
}

bool Transformer::collect_adapter_items(std::span<const TensorView* const> ws, const float* x,
                                        std::span<float* const> ys, int32_t n) {
    adapter_items_.clear();
    // Rows [first, first + count) of x and of every y, with `adapter`.
    auto add = [&](const AdapterMap& adapter, int32_t first, int32_t count) {
        for (size_t i = 0; i < ws.size(); ++i) {
            const auto it = adapter.find(ws[i]->name);
            if (it == adapter.end() || it->second.target->experts > 0) continue;
            const LoraTarget* t = it->second.target;
            adapter_items_.push_back({t, 0, it->second.scale, x + static_cast<size_t>(first) * static_cast<size_t>(t->a.ne[0]),
                                      ys[i] + static_cast<size_t>(first) * static_cast<size_t>(t->rows), count});
        }
    };
    if (segs_.size() == 1) {  // one sequence: its adapter over every row of this matmul
        if (segs_.front().adapter != nullptr) add(*segs_.front().adapter, 0, n);
    } else if (logits_rows_) {  // forward_batch()'s logits: the rows each segment has among them
        for (const AdapterRows& r : adapter_rows_) add(*r.adapter, r.first, r.n);
    } else {
        for (const Segment& s : segs_) {
            if (s.adapter != nullptr) add(*s.adapter, s.first, s.n);
        }
    }
    return !adapter_items_.empty();
}

Status Transformer::set_adapter(const LoraAdapter* adapter, float scale, ThreadPool* pool) {
    adapter_ = nullptr;
    adapter_targets_.clear();
    adapter_pool_ = pool;
    if (adapter == nullptr) return Status::ok();
    if (!std::isfinite(scale)) return Status(ErrorCode::InvalidArgument, "adapter scale must be finite");
    // The matrices the forward pass multiplies as a whole (an adapter's
    // update is added to such a product), by name: every block role but
    // latent attention's separate/combined per-head expansions and shared
    // K=V's grouped first output projection. Those slices do not have one
    // common input row. The LM head is separate unless tied to embeddings.
    std::unordered_map<std::string_view, const TensorView*> applied;
    for (const Layer& L : layers_) {
        for (size_t role = 0; role < kWeightRoles; ++role) {
            if (role == kKb || role == kVb || role == kKvB || role == kOutputA || L.w[role] == nullptr) continue;
            applied.emplace(L.w[role]->name, L.w[role]);
        }
    }
    if (output_ != nullptr && output_ != token_embd_) applied.emplace(output_->name, output_);
    if (hc_head_ != nullptr) applied.emplace(hc_head_->name, hc_head_);
    std::unordered_map<std::string_view, BoundAdapter> bound;
    for (const LoraTarget& t : adapter->targets()) {
        const auto it = applied.find(t.base);
        if (it == applied.end()) {
            if (file_->tensor(t.base) == nullptr) {
                return Status(ErrorCode::InvalidArgument,
                              "the adapter changes " + t.base + ", which the model does not have");
            }
            return Status(ErrorCode::Unsupported,
                          "the adapter changes " + t.base + ", which the forward pass does not multiply as a whole");
        }
        const TensorView& w = *it->second;
        const int32_t experts = w.n_dims >= 3 ? static_cast<int32_t>(w.ne[2]) : 0;
        if (t.experts != experts || t.a.ne[0] != w.ne[0] || t.rows != w.ne[1] || w.ne[3] != 1) {
            return Status(ErrorCode::InvalidArgument, "the adapter's " + t.base + " does not fit the model's shape");
        }
        bound.emplace(t.base, BoundAdapter{&t, adapter->scale_of(t, scale)});
    }
    adapter_targets_ = std::move(bound);
    adapter_ = adapter;
    return Status::ok();
}

void apply_final_softcap(std::span<float> logits, float cap, ThreadPool& pool) {
    // Elementwise, so chunking cannot change a bit (portable_tanh: the same
    // bits on every platform, which test_golden pins). tanh costs ~5 ns per
    // element: one thread over Gemma 2's 256k vocabulary took 1-2 ms per token.
    // Below this size the fork/join (a few us) is not worth it.
    constexpr size_t kParallelMin = 16384;
    float* p = logits.data();
    auto run = [p, cap](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) p[i] = cap * portable_tanh(p[i] / cap);
    };
    if (logits.size() < kParallelMin || pool.max_threads() == 1) {
        run(0, static_cast<int64_t>(logits.size()));
    } else {
        pool.parallel_for(static_cast<int64_t>(logits.size()), run);
    }
}

void Transformer::finish_logits(int32_t rows) {
    const ModelConfig& c = config_;
    const auto vocab = static_cast<size_t>(c.n_vocab);
    add_bias(output_bias_, logits_.data(), static_cast<size_t>(rows));
    if (c.logit_scale != 1.0f) {
        for (size_t i = 0; i < static_cast<size_t>(rows) * vocab; ++i) logits_[i] *= c.logit_scale;
    }
}

Status Transformer::compute_logits(const Route& route, size_t first_row, int32_t rows, ThreadPool& pool) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const PhaseTimer timer(phases_.lm_head);
    file_->begin_layer(c.n_layers);  // output head slot of the prefetch window
    for (int32_t r = 0; r < rows; ++r) {
        norm_row(x_.data() + (first_row + static_cast<size_t>(r)) * d, opt(output_norm_), opt(output_norm_b_),
                 xb_.data() + static_cast<size_t>(r) * d, c.n_embd);
    }
    logits_.resize(static_cast<size_t>(rows) * static_cast<size_t>(c.n_vocab));
    LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, *output_, xb_.data(), logits_.data(), rows));
    finish_logits(rows);
    if (c.final_softcap > 0.0f) {  // Gemma 2: logits squashed into (-cap, cap)
        apply_final_softcap(std::span<float>(logits_.data(), logits_.size()), c.final_softcap, pool);
    }
    return Status::ok();
}

Status Transformer::final_hidden(std::span<const int32_t> rows, std::span<float> out) const {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    if (out.size() < rows.size() * d) return Status(ErrorCode::InvalidArgument, "final_hidden: output buffer too small");
    // x_ holds the last pass's residual stream, one row per token (forward() sizes it to the pass).
    const size_t pass_rows = x_.size() / d;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] < 0 || static_cast<size_t>(rows[i]) >= pass_rows) {
            return Status(ErrorCode::InvalidArgument, "final_hidden: row outside the last pass");
        }
        // The same norm compute_logits() applies before the LM head.
        norm_row(x_.data() + static_cast<size_t>(rows[i]) * d, opt(output_norm_), opt(output_norm_b_),
                 out.data() + i * d, c.n_embd);
    }
    return Status::ok();
}

Status Transformer::project_fused_qkv(const Layer& L, const Route& route, Backend* backend, const BlockWeights& w,
                                      int32_t n) {
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(&L - layers_.data())];
    const size_t q_dim = static_cast<size_t>(plan.n_head) * static_cast<size_t>(plan.head_dim);
    const size_t kv_dim = static_cast<size_t>(plan.n_head_kv) * static_cast<size_t>(plan.head_dim);
    const size_t row = q_dim + 2 * kv_dim;
    qkv_.resize(static_cast<size_t>(n) * row);
    LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kQkv], xb_.data(), qkv_.data(), n));
    for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
        const float* src = qkv_.data() + t * row;
        std::copy(src, src + q_dim, q_.data() + t * q_dim);
        std::copy(src + q_dim, src + q_dim + kv_dim, k_.data() + t * kv_dim);
        std::copy(src + q_dim + kv_dim, src + row, v_.data() + t * kv_dim);
    }
    return Status::ok();
}

Status Transformer::propagate_kv(const Route& route, int32_t from, int32_t pos) {
    const ModelConfig& c = config_;
    // Only reached for attention-only stacks: early exit is disabled for hybrid models.
    for (int32_t l = from; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        if (L.kv_slot < 0 || plan.kv_source >= 0) continue;
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        Backend* backend = w.streamed() ? route.cpu : route.attention;  // see BlockWeights::streamed()
        if (plan.norm_before_mixer) norm_row(x_.data(), opt(L.attn_norm), opt(L.attn_norm_b), xb_.data(), c.n_embd);
        else std::copy_n(x_.data(), c.n_embd, xb_.data());
        if (c.blocks[static_cast<size_t>(l)].latent) {
            // The key row only: its latent channels are the values.
            LIYAB_RETURN_IF_ERROR(latent_project(l, w, 1, route, backend, false));
        } else if (c.blocks[static_cast<size_t>(l)].fused_qkv) {
            LIYAB_RETURN_IF_ERROR(project_fused_qkv(L, route, backend, w, 1));
        } else {
            LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kK], xb_.data(), k_.data(), 1));
            if (!plan.values_from_keys) {
                LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kV], xb_.data(), v_.data(), 1));
            }
        }
        for (size_t i = 0; i < L.bk.size(); ++i) k_[i] += L.bk[i];
        if (plan.values_from_keys) {
            std::copy_n(k_.data(), static_cast<size_t>(plan.n_head_kv) * static_cast<size_t>(plan.head_dim), v_.data());
        }
        for (size_t i = 0; i < L.bv.size(); ++i) v_[i] += L.bv[i];
        if (!c.blocks[static_cast<size_t>(l)].latent) {
            const size_t kv_dim = static_cast<size_t>(plan.n_head_kv) * static_cast<size_t>(plan.head_dim);
            clamp_kqv(k_.data(), kv_dim);
            clamp_kqv(v_.data(), kv_dim);
        }
        if (c.spec->value_norm) {
            const auto hd = static_cast<size_t>(plan.head_dim);
            for (size_t h = 0; h < static_cast<size_t>(plan.n_head_kv); ++h) {
                const float scale = rms_scale(v_.data() + h * hd, plan.head_dim, c.rms_eps);
                for (size_t i = 0; i < hd; ++i) v_[h * hd + i] *= scale;
            }
        }
        const bool norm_after_rope = c.spec->qk_norm == QkNormOrder::AfterRope;
        if (!norm_after_rope) qk_norm(L, nullptr, k_.data());
        if (c.blocks[static_cast<size_t>(l)].rope) {
            rope(k_.data(), c.blocks[static_cast<size_t>(l)].n_head_kv, pos, rope_tables_[static_cast<size_t>(L.rope_table)]);
        }
        if (norm_after_rope) qk_norm(L, nullptr, k_.data());
        scale_keys(L, nullptr, k_.data());
        store_kv(L, *kv_, pos, k_.data(), v_.data());
    }
    return Status::ok();
}

KeyScalingShape Transformer::key_scaling_shape() const {
    const ModelConfig& c = config_;
    // The horizon depends on the model only, never on the configured context,
    // so every process gets the same factors (and the same replies).
    const int32_t positions = c.n_ctx_train > 0 ? std::min(c.n_ctx_train, kKeyRangePositions) : kKeyRangePositions;
    KeyScalingShape shape{c.n_kv_layers, c.n_head, c.n_head_kv, c.head_dim, c.rope_dim,
                          c.spec->rope, {}, {}, positions, {}, {}, 1.0f, {}, {}};
    for (const BlockPlan& plan : c.blocks) shape.rope_dim = std::max(shape.rope_dim, plan.rope_dim);
    if (c.attn_temp_scale != 0.0f) {
        shape.query_scale = static_cast<float>(
            1.0 + c.attn_temp_scale * std::log1p(std::floor(static_cast<double>(positions - 1) / c.attn_temp_length)));
    }
    // Transformer::rope: pair i of an attention block turns by
    // inv_freq[i] / scale of the block's table per position.
    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& L = layers_[l];
        // Attention blocks with keys of their own, in KV slot order.
        if (L.kv_slot < 0 || c.blocks[l].kv_source >= 0) continue;
        const RopeTable& t = rope_tables_[static_cast<size_t>(L.rope_table)];
        const bool rotated = c.blocks[l].rope;  // a NoPE block's pairs never turn
        for (size_t i = 0; i < static_cast<size_t>(shape.rope_dim / 2); ++i) {
            shape.pair_speed.push_back(rotated && i < t.inv_freq.size() ? t.inv_freq[i] / t.scale : 0.0f);
        }
        shape.head_dims.push_back(c.blocks[l].head_dim);
        shape.rope_dims.push_back(c.blocks[l].rope_dim);
        shape.rope_magnitude.push_back(rotated ? t.magnitude : 1.0f);
        shape.query_heads.push_back(c.blocks[l].n_head);
        shape.kv_heads.push_back(c.blocks[l].n_head_kv);
    }
    return shape;
}

void Transformer::qk_norm(const Layer& L, float* q, float* k) const {
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(&L - layers_.data())];
    const auto hd = static_cast<size_t>(plan.head_dim);
    // Head h's weight (and bias): row h of a per-head norm, else the one shared row.
    auto row = [hd](const std::vector<float>& v, size_t h) {
        return v.empty() ? nullptr : v.data() + (v.size() > hd ? h * hd : 0);
    };
    if (q != nullptr && (!L.q_norm.empty() || !L.q_norm_b.empty())) {
        if (plan.qk_norm_whole) {
            norm_row(q, opt(L.q_norm), opt(L.q_norm_b), q, plan.n_head * plan.head_dim);
        } else for (size_t h = 0; h < static_cast<size_t>(plan.n_head); ++h) {
            norm_row(q + h * hd, row(L.q_norm, h), row(L.q_norm_b, h), q + h * hd, plan.head_dim);
        }
    }
    if (k != nullptr && (!L.k_norm.empty() || !L.k_norm_b.empty())) {
        if (plan.qk_norm_whole) {
            norm_row(k, opt(L.k_norm), opt(L.k_norm_b), k, plan.n_head_kv * plan.head_dim);
        } else for (size_t h = 0; h < static_cast<size_t>(plan.n_head_kv); ++h) {
            norm_row(k + h * hd, row(L.k_norm, h), row(L.k_norm_b, h), k + h * hd, plan.head_dim);
        }
    }
}

void Transformer::norm_row(const float* x, const float* w, const float* b, float* out, int32_t n) const {
    if (config_.spec->norm == NormKind::Layer) {
        layernorm(x, w, b, out, n, config_.rms_eps);
        return;
    }
    rmsnorm(x, w, out, n, config_.rms_eps);  // RMSNorm: the bits it always had
}

Status Transformer::add_positions(int32_t n) {
    if (position_embd_ == nullptr) return Status::ok();
    const ModelConfig& c = config_;
    const auto d = static_cast<size_t>(c.n_embd);
    thread_local std::vector<float> row;
    row.resize(d);
    for (int32_t t = 0; t < n; ++t) {
        const int32_t pos = pos_of(segs_[segment_of(t)], t);
        if (pos < 0 || pos >= c.n_positions) {
            return Status(ErrorCode::ContextFull, "position " + std::to_string(pos) + " is past the model's " +
                                                      std::to_string(c.n_positions) + " learned positions");
        }
        quant::dequantize_row(position_embd_->type, position_embd_->row(pos), row.data(), c.n_embd);
        float* x = x_.data() + static_cast<size_t>(t) * d;
        for (size_t i = 0; i < d; ++i) x[i] += row[i];
    }
    return Status::ok();
}

void Transformer::clamp_kqv(float* x, size_t n) const {
    const float cap = config_.clamp_kqv;
    if (cap <= 0.0f) return;
    for (size_t i = 0; i < n; ++i) x[i] = std::clamp(x[i], -cap, cap);
}

void Transformer::add_key_ranges(const Layer& L, const float* q, const float* k) {
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(&L - layers_.data())];
    const bool after = c.spec->qk_norm == QkNormOrder::AfterRope;
    if (!after || (L.q_norm.empty() && L.k_norm.empty())) {
        key_ranges_->add(L.kv_slot, q, k);
        return;
    }
    // Norms after RoPE: a head's stored vector is rope(x) * s * w, s the RMS
    // factor, the same for every rotation (a rotation keeps the norm). So the
    // ranges see x * s, unrotated, and the norm weight w as a gain applied
    // after the rotation (KeyRanges::add).
    const auto hd = static_cast<size_t>(plan.head_dim);
    // RoPE's magnitude (LongRoPE) scales the rotated channels before the norm.
    const float magnitude = c.blocks[static_cast<size_t>(&L - layers_.data())].rope
                                ? rope_tables_[static_cast<size_t>(L.rope_table)].magnitude
                                : 1.0f;
    std::vector<float> rotated(hd);
    auto scaled = [&](const float* x, int32_t heads, bool normed, std::vector<float>& out) {
        if (x == nullptr) return;
        out.assign(x, x + static_cast<size_t>(heads) * hd);
        if (!normed) return;
        for (size_t h = 0; h < static_cast<size_t>(heads); ++h) {
            float* xh = out.data() + h * hd;
            std::copy(xh, xh + hd, rotated.begin());
            for (size_t i = 0; i < static_cast<size_t>(plan.rope_dim); ++i) rotated[i] *= magnitude;
            const float s = rms_scale(rotated.data(), plan.head_dim, c.rms_eps);
            for (size_t i = 0; i < hd; ++i) xh[i] *= s;
        }
    };
    std::vector<float> qs, ks;
    scaled(q, plan.n_head, !L.q_norm.empty(), qs);
    scaled(k, plan.n_head_kv, !L.k_norm.empty(), ks);
    key_ranges_->add(L.kv_slot, q == nullptr ? nullptr : qs.data(), k == nullptr ? nullptr : ks.data(), L.q_norm.empty() ? nullptr : L.q_norm.data(),
                     L.k_norm.empty() ? nullptr : L.k_norm.data());
}

void Transformer::scale_keys(const Layer& L, float* q, float* k) const {
    if (key_scale_.empty()) return;
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(&L - layers_.data())];
    const auto hd = static_cast<size_t>(plan.head_dim);
    const auto stored = static_cast<size_t>(c.head_dim);
    const auto group = static_cast<size_t>(plan.n_head / plan.n_head_kv);
    const float* t = key_scale_.data() + static_cast<size_t>(L.kv_slot) * static_cast<size_t>(c.n_head_kv) * stored;
    for (size_t g = 0; g < static_cast<size_t>(plan.n_head_kv); ++g) {
        const float* tg = t + g * stored;
        if (k != nullptr) {
            float* kg = k + g * hd;
            for (size_t i = 0; i < hd; ++i) kg[i] /= tg[i];
        }
        if (q == nullptr) continue;
        for (size_t j = 0; j < group; ++j) {
            float* qh = q + (g * group + j) * hd;
            for (size_t i = 0; i < hd; ++i) qh[i] *= tg[i];
        }
    }
}

void Transformer::store_kv(const Layer& L, KvCache& cache, int32_t pos, const float* k, const float* v) {
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(&L - layers_.data())];
    if (plan.n_head_kv == c.n_head_kv && plan.head_dim == c.head_dim && plan.value_dim == c.value_dim) {
        cache.store(L.kv_slot, pos, k, v);
        return;
    }
    const size_t stride = static_cast<size_t>(c.head_dim);
    padded_k_.assign(static_cast<size_t>(c.n_head_kv) * stride, 0.0f);
    padded_v_.assign(static_cast<size_t>(c.n_head_kv) * stride, 0.0f);
    for (size_t h = 0; h < static_cast<size_t>(plan.n_head_kv); ++h) {
        std::copy_n(k + h * static_cast<size_t>(plan.head_dim), plan.head_dim, padded_k_.data() + h * stride);
        std::copy_n(v + h * static_cast<size_t>(plan.value_dim), plan.value_dim, padded_v_.data() + h * stride);
    }
    cache.store(L.kv_slot, pos, padded_k_.data(), padded_v_.data());
}

Status Transformer::calibrate_keys(std::span<const int32_t> tokens, const Route& route, ThreadPool& pool) {
    key_scale_.clear();
    // Shared K=V must preserve value channels; the ordinary key-only scaling
    // transformation is not applicable to that cache geometry.
    if (config_.spec->shared_kv) return Status::ok();
    if (!quant::is_block_quantized(kv_->dtype()) || config_.n_kv_layers == 0) return Status::ok();
    if (tokens.empty()) return Status(ErrorCode::InvalidArgument, "key calibration needs tokens");
    reset();
    key_ranges_ = std::make_unique<KeyRanges>(key_scaling_shape());
    // A probe first: the outlier channels are nearly constant, so a few
    // tokens show whether the model has them, and fewer tokens only raise the
    // ratio (lower medians). Most models stop there; the rest of the text is
    // processed only to measure their factors, and decides again.
    Status status;
    float ratio = 0.0f;
    size_t measured = 0;
    while (measured < tokens.size() && status.is_ok()) {
        const size_t limit = measured == 0 && config_.causal ? std::min(tokens.size(), kKeyProbeTokens) : tokens.size();
        const auto n = config_.causal ? std::min(limit - measured, static_cast<size_t>(max_batch_)) : tokens.size();
        status = forward(tokens.subspan(measured, n), Logits::None, route, pool).status();
        measured += n;
        if (!status.is_ok() || measured < limit) continue;
        ratio = key_ranges_->outlier_ratio();
        if (ratio < kOutlierRatio) break;
    }
    if (status.is_ok() && ratio >= kOutlierRatio) key_scale_ = key_ranges_->factors();
    key_ranges_.reset();
    reset();
    if (!status.is_ok()) return status;
    if (ratio < kOutlierRatio) {
        LIYAB_LOG_INFO("keys: largest channel %.1fx its head's median over %zu tokens: no scaling needed", ratio,
                       measured);
        return Status::ok();
    }
    const auto largest = std::max_element(key_scale_.begin(), key_scale_.end());
    const auto per_block = static_cast<size_t>(config_.n_head_kv) * static_cast<size_t>(config_.head_dim);
    LIYAB_LOG_INFO("keys: largest channel %.1fx its head's median over %zu tokens: per-channel scales for the %s "
                   "cache (largest %.1f, attention block %zu)",
                   ratio, measured, std::string(dtype_traits(kv_->dtype()).name).c_str(),
                   static_cast<double>(*largest), static_cast<size_t>(largest - key_scale_.begin()) / per_block);
    return Status::ok();
}

Status Transformer::set_key_scales(std::span<const float> scales) {
    if (n_past_ != 0) return Status(ErrorCode::InvalidArgument, "key scales change only with an empty context");
    if (scales.empty()) {
        key_scale_.clear();
        return Status::ok();
    }
    if (config_.spec->shared_kv) return Status(ErrorCode::Unsupported, "key-only scaling cannot preserve shared K=V values");
    if (!quant::is_block_quantized(kv_->dtype())) {
        return Status(ErrorCode::InvalidArgument, "key scales apply to quantized KV caches only");
    }
    if (scales.size() != key_scaling_shape().factors()) {
        return Status(ErrorCode::InvalidArgument, "key scales of another model");
    }
    for (const float t : scales) {
        if (!(t >= 1.0f / kMaxKeyScale && t <= kMaxKeyScale)) {
            return Status(ErrorCode::InvalidArgument, "key scale out of range");
        }
    }
    key_scale_.assign(scales.begin(), scales.end());
    return Status::ok();
}

// One synthetic pass through the first latent block in each form, on its
// real weights: cached rows of random latents (a fixed seed), random
// normalized inputs, timed over the parts that differ between the forms
// (latent_ms_, as in forward()). The ratio of the two forms' time per
// counted multiply-add is what LatentFormChooser needs; one block suffices
// because every latent block has the same shape. The best of kRepeats runs
// per form, alternated, keeps a passing burst of load out of it.
Result<float> Transformer::calibrate_latent_form(const Route& route, ThreadPool& pool) {
    const ModelConfig& c = config_;
    if (c.mla.kv_rank == 0) return 1.0f;
    int32_t layer = 0;
    while (!c.blocks[static_cast<size_t>(layer)].latent) ++layer;
    constexpr int32_t kContext = 1024, kBatch = 256, kRepeats = 3;
    const int32_t context = std::min(kContext, context_length_);
    const int32_t n = std::max(2, std::min(kBatch, context / 2));
    reset();
    LIYAB_RETURN_IF_ERROR(kv_->reserve(context));
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto kd = static_cast<size_t>(c.head_dim);
    std::mt19937 rng(20251017);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> row(kd);
    std::vector<float> index_key(static_cast<size_t>(c.index_dim));
    for (int32_t pos = 0; pos < context - n; ++pos) {
        for (float& v : row) v = normal(rng);
        kv_->store(L.kv_slot, pos, row.data(), nullptr);
        if (c.index_heads > 0) {
            for (float& v : index_key) v = normal(rng);
            kv_->store_auxiliary(L.kv_slot, pos, index_key.data());
        }
    }
    n_past_ = context - n;
    const auto un = static_cast<size_t>(n);
    const size_t d = static_cast<size_t>(c.n_embd);
    const size_t q_dim = static_cast<size_t>(c.n_head) * kd;
    xb_.resize(un * std::max(d, q_dim));
    q_.resize(un * q_dim);
    k_.resize(un * kd);
    att_.resize(un * std::max(q_dim, static_cast<size_t>(c.n_head) * static_cast<size_t>(c.mla.v_dim)));
    std::vector<float> input(un * d);
    for (float& v : input) v = normal(rng);
    for (size_t t = 0; t < un; ++t) {
        norm_row(input.data() + t * d, opt(L.attn_norm), opt(L.attn_norm_b), input.data() + t * d, c.n_embd);
    }
    const int64_t pairs = latent_visible(live_segment(0, n, Logits::None), L.window);
    int64_t positions = 0;
    for (const auto& [a, b] : mla_union_) positions += b - a;
    const LatentFormChooser::Work work = LatentFormChooser::work(c, n, pairs, positions);

    Status status;
    double best[2] = {INFINITY, INFINITY};  // [absorbed, expanded], ms
    {
        const BlockWeights w(*this, layer);
        status = w.status();
        for (int32_t r = 0; r < kRepeats && status.is_ok(); ++r) {
            for (const bool expanded : {false, true}) {
                std::copy(input.begin(), input.end(), xb_.begin());
                latent_expanded_ = expanded;
                begin_live_pass(n, Logits::None);
                latent_ms_ = 0.0;
                status = latent_attention_mixer(layer, w, n, route, pool, nullptr);
                if (!status.is_ok()) break;
                best[expanded ? 1 : 0] = std::min(best[expanded ? 1 : 0], latent_ms_);
            }
        }
    }
    latent_expanded_ = false;
    reset();
    if (!status.is_ok()) return status;
    const double measured = (best[1] / work.expanded) / (best[0] / work.absorbed);
    const float ratio = LatentFormChooser::round_ratio(measured);
    LIYAB_RETURN_IF_ERROR(latent_chooser_.set_ratio(ratio));
    LIYAB_LOG_INFO("latent attention: an expanded multiply-add costs %.2fx an absorbed one (%.1f / %.1f ms on a "
                   "%d-token pass after %d positions); ratio %.3f",
                   measured, best[1], best[0], n, context - n, static_cast<double>(ratio));
    return ratio;
}

Status Transformer::attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                    ThreadPool& pool, const uint8_t* head_mask) {
    index_positions_.clear();
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const BlockPlan& plan = c.blocks[static_cast<size_t>(layer)];
    const bool gated = plan.attn_gate;
    const bool fused_gate = gated && L.w[kZ] == nullptr;
    const bool head_gate = L.w[kZ] != nullptr && L.w[kZ]->rows() == plan.n_head;
    const bool own_kv = plan.kv_source < 0;
    const auto un = static_cast<size_t>(n);
    const size_t q_dim = static_cast<size_t>(plan.n_head) * static_cast<size_t>(plan.head_dim);
    const size_t kv_dim = static_cast<size_t>(plan.n_head_kv) * static_cast<size_t>(plan.head_dim);
    const size_t hd = static_cast<size_t>(plan.head_dim);
    const bool norm_after_rope = c.spec->qk_norm == QkNormOrder::AfterRope;
    const RopeTable& rope_table = rope_tables_[static_cast<size_t>(L.rope_table)];
    const bool rotated = c.blocks[static_cast<size_t>(layer)].rope;  // false: a NoPE block

    // Gated blocks project [q_h | gate_h] per head into qg_; split it afterwards.
    if (fused_gate) qg_.resize(un * 2 * q_dim);
    if (gated) gate_.resize(un * (head_gate ? static_cast<size_t>(plan.n_head) : q_dim));
    if (L.w[kZ] != nullptr) {
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kZ], xb_.data(), gate_.data(), n));
    }
    if (c.blocks[static_cast<size_t>(layer)].fused_qkv) {
        // One [q | k | v] projection per token, split into q_, k_ and v_.
        LIYAB_RETURN_IF_ERROR(project_fused_qkv(L, route, route.attention, w, n));
    } else if (!own_kv) {
        // KV sharing: queries only, over the rows block kv_source cached.
        LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kQ], xb_.data(), fused_gate ? qg_.data() : q_.data(), n));
    } else if (plan.values_from_keys) {
        const TensorView* qk[] = {&w[kQ], &w[kK]};
        float* outs[] = {fused_gate ? qg_.data() : q_.data(), k_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, qk, xb_.data(), outs, n));
    } else {
        const TensorView* qkv[] = {&w[kQ], &w[kK], &w[kV]};
        float* outs[] = {fused_gate ? qg_.data() : q_.data(), k_.data(), v_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, qkv, xb_.data(), outs, n));
    }
    if (fused_gate) {
        for (size_t t = 0; t < un; ++t) {
            for (size_t h = 0; h < static_cast<size_t>(plan.n_head); ++h) {
                const float* src = qg_.data() + (t * static_cast<size_t>(plan.n_head) + h) * 2 * hd;
                std::copy(src, src + hd, q_.data() + t * q_dim + h * hd);
                std::copy(src + hd, src + 2 * hd, gate_.data() + t * q_dim + h * hd);
            }
        }
    }
    for (const Segment& s : segs_) {
        for (auto t = static_cast<size_t>(s.first); t < static_cast<size_t>(s.first + s.n); ++t) {
            float* q = q_.data() + t * q_dim;
            float* k = k_.data() + t * kv_dim;
            float* v = v_.data() + t * kv_dim;
            const int32_t pos = pos_of(s, static_cast<int32_t>(t));
            for (size_t i = 0; i < L.bq.size(); ++i) q[i] += L.bq[i];
            clamp_kqv(q, q_dim);
            if (!own_kv) {  // queries only (k_norm is empty: the keys are the source block's)
                if (!norm_after_rope) qk_norm(L, q, nullptr);
                if (key_ranges_ != nullptr) add_key_ranges(L, q, nullptr);
                if (rotated) rope(q, plan.n_head, pos, rope_table);
                if (norm_after_rope) qk_norm(L, q, nullptr);
                scale_keys(L, q, nullptr);
                continue;
            }
            for (size_t i = 0; i < L.bk.size(); ++i) k[i] += L.bk[i];
            if (plan.values_from_keys) std::copy_n(k, kv_dim, v);  // after projection bias, before norm and RoPE
            for (size_t i = 0; i < L.bv.size(); ++i) v[i] += L.bv[i];
            clamp_kqv(k, kv_dim);
            clamp_kqv(v, kv_dim);
            if (c.spec->value_norm) {
                for (size_t h = 0; h < static_cast<size_t>(plan.n_head_kv); ++h) {
                    float* vh = v + h * hd;
                    const float scale = rms_scale(vh, plan.head_dim, c.rms_eps);
                    for (size_t i = 0; i < hd; ++i) vh[i] *= scale;
                }
            }
            if (!norm_after_rope) qk_norm(L, q, k);
            if (key_ranges_ != nullptr) add_key_ranges(L, q, k);  // unrotated: ranges cover every position
            if (rotated) {
                rope(q, plan.n_head, pos, rope_table);
                rope(k, plan.n_head_kv, pos, rope_table);
            }
            if (norm_after_rope) qk_norm(L, q, k);
            if (c.attn_temp_scale != 0.0f && pos >= c.attn_temp_length) {
                // Attention temperature (llama_4_scaling): 1 below the length, so those rows keep their bits.
                const auto temp = static_cast<float>(
                    1.0 + c.attn_temp_scale * std::log1p(std::floor(static_cast<double>(pos) / c.attn_temp_length)));
                for (size_t i = 0; i < q_dim; ++i) q[i] *= temp;
            }
            scale_keys(L, q, k);
            store_kv(L, *s.kv, row_of(s, static_cast<int32_t>(t)), k, v);
        }
    }
    attention(L, n, pool, head_mask);
    if (head_gate) {
        for (size_t t = 0; t < un; ++t) {
            for (size_t h = 0; h < static_cast<size_t>(plan.n_head); ++h) {
                const float scale = sigmoid(gate_[t * static_cast<size_t>(plan.n_head) + h]);
                for (size_t j = 0; j < hd; ++j) att_[t * q_dim + h * hd + j] *= scale;
            }
        }
    } else if (gated) {
        for (size_t i = 0; i < un * q_dim; ++i) att_[i] *= sigmoid(gate_[i]);
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kO], att_.data(), xb_.data(), n));
    add_bias(L.bo, xb_.data(), un);
    return Status::ok();
}

Status Transformer::per_head_matmul(const Route& route, const TensorView& stacked, const float* in, float* out,
                                    int32_t n) {
    const auto heads = static_cast<size_t>(stacked.ne[2]);
    const auto cols = static_cast<size_t>(stacked.ne[0]);
    const auto rows = static_cast<size_t>(stacked.ne[1]);
    const auto un = static_cast<size_t>(n);
    mla_head_views_.resize(heads);
    mla_items_.clear();
    for (size_t h = 0; h < heads; ++h) {
        mla_head_views_[h] = expert_slice(stacked, static_cast<int64_t>(h));
        mla_items_.push_back({&mla_head_views_[h], in + h * un * cols, out + h * un * rows, n});
    }
    // Transient views into a 3-D tensor, as the experts' are: the CPU reads
    // them in place, one parallel pass for every head (a matmul per head
    // would pay a thread fork/join each).
    return route.cpu->matmul_batch(mla_items_);
}

Status Transformer::latent_project(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                   Backend* backend, bool queries) {
    const ModelConfig& c = config_;
    const LatentAttention& m = c.mla;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const size_t qk = static_cast<size_t>(m.nope_dim) + rd;  // a head's query as projected: [q_nope | q_pe]
    const auto kd = static_cast<size_t>(c.head_dim);          // rd + rank: a cached key row
    const float norm_eps = c.spec->latent_norm_eps > 0.0f ? c.spec->latent_norm_eps : c.rms_eps;
    mla_kv_.resize(un * (rank + rd));
    if (!queries) {
        LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kKvA], xb_.data(), mla_kv_.data(), n));
    } else {
        const bool low_rank = m.q_rank > 0;  // q = W_qb RMSNorm(W_qa x), else q = W_q x
        mla_q_.resize(un * heads * qk);
        if (low_rank) mla_qa_.resize(un * static_cast<size_t>(m.q_rank));
        const TensorView* proj[] = {low_rank ? &w[kQa] : &w[kQ], &w[kKvA]};
        float* outs[] = {low_rank ? mla_qa_.data() : mla_q_.data(), mla_kv_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, backend, proj, xb_.data(), outs, n));
        if (low_rank) {
            for (size_t t = 0; t < un; ++t) {
                float* row = mla_qa_.data() + t * static_cast<size_t>(m.q_rank);
                rmsnorm(row, L.q_a_norm.data(), row, m.q_rank, norm_eps);
            }
            LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kQb], mla_qa_.data(), mla_q_.data(), n));
        }
    }
    // The key row: [k_pe | RMSNorm(latent)] (the projection gives [latent |
    // k_pe]). RoPE's channels come first, as in every other head this
    // engine rotates, so rope() and the key ranges treat it like any head.
    for (size_t t = 0; t < un; ++t) {
        const float* kv = mla_kv_.data() + t * (rank + rd);
        float* k = k_.data() + t * kd;
        std::copy(kv + rank, kv + rank + rd, k);
        rmsnorm(kv, L.kv_a_norm.data(), k + rd, m.kv_rank, norm_eps);
    }
    return Status::ok();
}

Status Transformer::latent_absorb(const BlockWeights& w, int32_t n, const Route& route) {
    const ModelConfig& c = config_;
    const LatentAttention& m = c.mla;
    const auto un = static_cast<size_t>(n);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const auto nope = static_cast<size_t>(m.nope_dim);
    const size_t qk = nope + rd;                      // a head's query as projected: [q_nope | q_pe]
    const auto kd = static_cast<size_t>(c.head_dim);  // rd + rank: an absorbed query, a cached key
    // Each head's query absorbs its key projection: W_kb,h^T q_nope,h is what
    // the latent is scored against. The q_nope of every token are gathered
    // per head so each head is one matmul over the batch.
    mla_heads_in_.resize(heads * un * nope);
    for (size_t t = 0; t < un; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            const float* src = mla_q_.data() + (t * heads + h) * qk;
            std::copy(src, src + nope, mla_heads_in_.data() + (h * un + t) * nope);
        }
    }
    mla_heads_out_.resize(heads * un * rank);
    LIYAB_RETURN_IF_ERROR(per_head_matmul(route, w[kKb], mla_heads_in_.data(), mla_heads_out_.data(), n));
    for (size_t t = 0; t < un; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            float* q = q_.data() + (t * heads + h) * kd;
            const float* pe = mla_q_.data() + (t * heads + h) * qk + nope;
            std::copy(pe, pe + rd, q);
            const float* absorbed = mla_heads_out_.data() + (h * un + t) * rank;
            std::copy(absorbed, absorbed + rank, q + rd);
        }
    }
    return Status::ok();
}

int64_t Transformer::latent_visible(const Segment& seg, int32_t window) {
    mla_union_.clear();
    int64_t pairs = 0;
    for (int32_t t = 0; t < seg.n; ++t) {
        const int32_t pos = seg.n_past + t;
        const Seen s = seen_by(*seg.kv, pos, window);
        pairs += (s.sink_end - s.sink_begin) + (pos - s.first + 1);
        if (s.sink_end > s.sink_begin) mla_union_.emplace_back(s.sink_begin, s.sink_end);
        mla_union_.emplace_back(s.first, pos + 1);
    }
    std::sort(mla_union_.begin(), mla_union_.end());
    size_t kept = 0;
    for (size_t i = 0; i < mla_union_.size(); ++i) {
        if (kept > 0 && mla_union_[i].first <= mla_union_[kept - 1].second) {
            mla_union_[kept - 1].second = std::max(mla_union_[kept - 1].second, mla_union_[i].second);
        } else {
            mla_union_[kept++] = mla_union_[i];
        }
    }
    mla_union_.resize(kept);
    return pairs;
}

int32_t Transformer::latent_tile() const noexcept {
    const ModelConfig& c = config_;
    const auto row = static_cast<int64_t>(c.n_head) * (c.mla.nope_dim + c.rope_dim + c.mla.v_dim) *
                     static_cast<int64_t>(sizeof(float));
    int32_t tile = kLatentTileMax;
    while (tile > kLatentTileMin && tile * row > static_cast<int64_t>(kLatentTileBytes)) tile /= 2;
    return tile;
}

// Latent attention in the expanded form, the form the model was trained in:
// per head h, key [W_kb,h c_j | k_pe,j] and value W_vb,h c_j of every
// position j the pass sees, scored against [q_nope,h | q_pe,h] with the same
// scale. Per (token, position, head) that is nope + rope_dim channels scored
// and v_dim accumulated (192 and 128 on DeepSeek-V2-Lite) where the absorbed
// form takes rope_dim + kv_rank and kv_rank (576 and 512), for the cost of
// expanding every position the pass sees once per pass instead of absorbing
// each of its tokens: long prefills gain, decoding steps lose
// (LatentFormChooser weighs the two). The cache keeps latents only; expanded
// rows exist one tile at a time (latent_tile(), about kLatentTileBytes).
//
// Positions go by tiles at absolute bounds (multiples of latent_tile()). Per
// tile, the rows the pass sees are read from the cache (key scales multiplied
// back: stored rows were divided by them), expanded by one GEMM
// (expand_strip) into [k_nope | k_pe] and v per head, and every (token, head)
// folds the tile into its running softmax (maximum, sum, weighted values),
// tiles in ascending order. A token's result therefore depends only on its
// position, its query and the cached rows: the same bits however a prompt is
// split into passes and however many threads run (within this form; the
// absorbed form rounds differently).
Status Transformer::latent_expanded(int32_t layer, const BlockWeights& w, const Segment& seg, ThreadPool& pool,
                                    const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const LatentAttention& m = c.mla;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    // The segment's tokens, 0..un-1 here; batch row first + i in q_, mla_q_ and att_.
    const auto un = static_cast<size_t>(seg.n);
    const auto first = static_cast<size_t>(seg.first);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const auto nope = static_cast<size_t>(m.nope_dim);
    const auto vd = static_cast<size_t>(m.v_dim);
    const auto kd = static_cast<size_t>(c.head_dim);
    const size_t qk = nope + rd;            // a head's expanded key (and query): [nope | rope]
    const size_t row = heads * (qk + vd);   // an expanded row: every head's key, then every head's value
    const size_t key_cols = heads * nope;   // GEMM columns: W_kb,h's for every head, then W_vb,h's
    const size_t cols = key_cols + heads * vd;
    const auto strip = static_cast<size_t>(kExpandColumns);
    const size_t strips = (cols + strip - 1) / strip;

    // The expansion weights, decoded once per pass and packed per strip:
    // mla_wp_[s][r][i] is column s * kExpandColumns + i's weight on latent
    // channel r (zero past the last column). attn_k_b holds per head the
    // kv_rank rows of nope weights (k_nope = sum over r of c_r row_r),
    // attn_v_b the v_dim rows of kv_rank weights (v_d = row_d . c).
    mla_wp_.assign(strips * rank * strip, 0.0f);
    mla_col_offset_.assign(strips * strip, 0);
    for (size_t col = 0; col < cols; ++col) {
        mla_col_offset_[col] = col < key_cols ? col / nope * qk + col % nope : heads * qk + (col - key_cols);
    }
    const TensorView& kb = w[kKb];
    const TensorView& vb = w[kVb];
    pool.parallel_for(static_cast<int64_t>(heads), [&](int64_t h0, int64_t h1) {
        thread_local std::vector<float> buf;
        buf.resize(std::max(nope, rank));
        for (int64_t h = h0; h < h1; ++h) {
            const auto uh = static_cast<size_t>(h);
            const TensorView kh = expert_slice(kb, h);
            for (size_t r = 0; r < rank; ++r) {
                quant::dequantize_row(kh.type, kh.row(static_cast<int64_t>(r)), buf.data(), kh.cols());
                for (size_t i = 0; i < nope; ++i) {
                    const size_t col = uh * nope + i;
                    mla_wp_[(col / strip * rank + r) * strip + col % strip] = buf[i];
                }
            }
            const TensorView vh = expert_slice(vb, h);
            for (size_t d = 0; d < vd; ++d) {
                quant::dequantize_row(vh.type, vh.row(static_cast<int64_t>(d)), buf.data(), vh.cols());
                const size_t col = key_cols + uh * vd + d;
                float* dst = mla_wp_.data() + col / strip * rank * strip + col % strip;
                for (size_t r = 0; r < rank; ++r) dst[r * strip] = buf[r];
            }
        }
    });

    // Running softmax per (token, head); the weighted values accumulate in
    // att_ in the [token][head][v] order attn_output reads.
    mla_max_.assign(un * heads, -INFINITY);
    mla_sum_.assign(un * heads, 0.0f);
    float* const att = att_.data() + first * heads * vd;  // this segment's rows (others' are left as they are)
    std::fill(att, att + un * heads * vd, 0.0f);
    const float scale = c.attn_scale;
    const float softcap = c.attn_softcap;
    const DType kv_type = seg.kv->dtype();
    const float* key_scales = key_scale_.empty() ? nullptr : key_scale_.data() + static_cast<size_t>(L.kv_slot) * kd;
    const int32_t tile = latent_tile();
    latent_visible(seg, L.window);
    const int32_t lo = mla_union_.front().first;
    const int32_t hi = mla_union_.back().second;
    for (int32_t t0 = lo / tile * tile; t0 < hi; t0 += tile) {
        const int32_t t1 = t0 + tile;
        mla_rows_.clear();  // the tile's positions the pass sees, ascending
        for (const auto& [a, b] : mla_union_) {
            for (int32_t p = std::max(a, t0); p < std::min(b, t1); ++p) mla_rows_.push_back(p);
        }
        if (mla_rows_.empty()) continue;
        const auto rows = static_cast<int64_t>(mla_rows_.size());
        const auto urows = static_cast<size_t>(rows);
        mla_c_.resize(urows * rank);
        mla_ex_.resize(urows * row);

        // The cached rows, decoded: the latent into mla_c_, the RoPE key into
        // every head's key in mla_ex_ (it is shared by the heads).
        auto decode = [&](int64_t j0, int64_t j1) {
            thread_local std::vector<float> key;
            key.resize(kd);
            for (int64_t j = j0; j < j1; ++j) {
                const auto uj = static_cast<size_t>(j);
                const uint8_t* src = seg.kv->k_row(L.kv_slot, mla_rows_[uj], 0);
                quant::dequantize_row(kv_type, src, key.data(), static_cast<int64_t>(kd));
                if (key_scales != nullptr) {
                    for (size_t i = 0; i < kd; ++i) key[i] *= key_scales[i];
                }
                std::copy(key.begin() + static_cast<std::ptrdiff_t>(rd), key.end(), mla_c_.data() + uj * rank);
                float* ex = mla_ex_.data() + uj * row;
                for (size_t h = 0; h < heads; ++h) {
                    std::copy(key.begin(), key.begin() + static_cast<std::ptrdiff_t>(rd), ex + h * qk + nope);
                }
            }
        };
        if (rows >= 64) {
            pool.parallel_for(rows, decode);
        } else {
            decode(0, rows);
        }
        // Keys and values of every head: one strip of columns per task.
        pool.parallel_for(static_cast<int64_t>(strips), [&](int64_t s0, int64_t s1) {
            for (int64_t s = s0; s < s1; ++s) {
                const auto us = static_cast<size_t>(s);
                const auto columns = static_cast<int32_t>(std::min(strip, cols - us * strip));
                expand_strip(mla_c_.data(), rows, static_cast<int64_t>(rank), mla_wp_.data() + us * rank * strip,
                             mla_ex_.data(), row, mla_col_offset_.data() + us * strip, columns);
            }
        });
        // Every (token, head) folds in the tile's positions it sees. A task
        // takes one head and a group of tokens and goes through the tile's
        // rows kAttendRows at a time, so a chunk of that head's keys and
        // values (80 KiB on DeepSeek-V2-Lite) is read from L2 once for the
        // whole group and reused from L1 by each block of kScoreQueries
        // tokens. Each token still sees its rows in ascending order, and its
        // block-mates (a last short block repeats its last token, computed,
        // not used) change no bit of it.
        const auto block = static_cast<size_t>(kScoreQueries);
        size_t group = kAttendTokens;  // fewer tokens per task when there are too few tasks for the threads
        while (group > block && heads * ((un + group - 1) / group) <
                                    static_cast<size_t>(pool.max_threads()) * ThreadPool::kChunksPerThread) {
            group /= 2;
        }
        const size_t groups = (un + group - 1) / group;
        pool.parallel_for(static_cast<int64_t>(groups * heads), [&](int64_t a, int64_t b) {
            thread_local std::vector<float> scores;                    // [token of the group][row]
            thread_local std::vector<std::array<std::pair<size_t, size_t>, 2>> ranges;  // per token: sinks, [first, pos]
            scores.resize(group * urows);
            ranges.resize(group);
            for (int64_t task = a; task < b; ++task) {
                const size_t h = static_cast<size_t>(task) % heads;
                const size_t g0 = static_cast<size_t>(task) / heads * group;
                if (head_mask != nullptr && head_mask[h] == 0) continue;  // pruned head: contributes nothing
                const size_t count = std::min(group, un - g0);
                auto index = [&](int32_t pos) {
                    return static_cast<size_t>(std::lower_bound(mla_rows_.begin(), mla_rows_.end(), pos) -
                                               mla_rows_.begin());
                };
                // Each token's visible rows, as index ranges into the ascending mla_rows_.
                size_t lo = urows, hi = 0;
                for (size_t j = 0; j < count; ++j) {
                    const int32_t pos = seg.n_past + static_cast<int32_t>(g0 + j);
                    const Seen s = seen_by(*seg.kv, pos, L.window);
                    ranges[j][0] = {index(s.sink_begin), index(s.sink_end)};
                    ranges[j][1] = {index(std::max(s.first, s.sink_end)), index(pos + 1)};
                    for (const auto& [i0, i1] : ranges[j]) {
                        if (i0 >= i1) continue;
                        lo = std::min(lo, i0);
                        hi = std::max(hi, i1);
                    }
                }
                if (lo >= hi) continue;  // the group sees nothing in this tile
                auto token = [&](size_t j0, size_t j) { return std::min(j0 + j, count - 1); };  // a short block's padding
                // Scores of every token of the group over [lo, hi), by chunks of rows.
                for (size_t c0 = lo; c0 < hi; c0 += kAttendRows) {
                    const size_t c1 = std::min(hi, c0 + kAttendRows);
                    for (size_t j0 = 0; j0 < count; j0 += block) {
                        const float* q[kScoreQueries];
                        for (size_t j = 0; j < block; ++j) q[j] = mla_q_.data() + ((first + g0 + token(j0, j)) * heads + h) * qk;
                        for (size_t i = c0; i < c1; i += kScoreKeys) {
                            const float* k[kScoreKeys];
                            for (size_t r = 0; r < static_cast<size_t>(kScoreKeys); ++r) {
                                k[r] = mla_ex_.data() + std::min(i + r, c1 - 1) * row + h * qk;
                            }
                            float dots[kScoreQueries][kScoreKeys];
                            score_block(q, k, qk, dots);
                            for (size_t j = 0; j < block && j0 + j < count; ++j) {
                                for (size_t r = 0; r < static_cast<size_t>(kScoreKeys) && i + r < c1; ++r) {
                                    float sc = dots[j][r] * scale;
                                    if (softcap > 0.0f) sc = softcap * portable_tanh(sc / softcap);
                                    if (!index_positions_.empty()) {
                                        const auto& selected = index_positions_[first + g0 + j0 + j];
                                        if (!std::binary_search(selected.begin(), selected.end(), mla_rows_[i + r])) sc = -INFINITY;
                                    }
                                    scores[(j0 + j) * urows + i + r] = sc;
                                }
                            }
                        }
                    }
                }
                // Per token: the tile's maximum, earlier tiles rescaled to it,
                // weights exp(score - maximum) in place of the scores, their sum.
                for (size_t j = 0; j < count; ++j) {
                    const size_t ut = (g0 + j) * heads + h;
                    float* o = att + ut * vd;
                    float* sc = scores.data() + j * urows;
                    float tile_max = -INFINITY;
                    for (const auto& [i0, i1] : ranges[j]) {
                        for (size_t i = i0; i < i1; ++i) tile_max = std::max(tile_max, sc[i]);
                    }
                    if (tile_max == -INFINITY) {  // an indexed token may select nothing in this tile
                        for (const auto& [i0, i1] : ranges[j]) std::fill(sc + i0, sc + i1, 0.0f);
                        continue;
                    }
                    float& mx = mla_max_[ut];
                    float& sum = mla_sum_[ut];
                    const float next = std::max(mx, tile_max);
                    if (next != mx) {  // rescale what earlier tiles added (nothing yet: 0 x 0)
                        const float f = std::exp(mx - next);
                        sum *= f;
                        for (size_t d = 0; d < vd; ++d) o[d] *= f;
                        mx = next;
                    }
                    for (const auto& [i0, i1] : ranges[j]) {
                        for (size_t i = i0; i < i1; ++i) sum += sc[i] = std::exp(sc[i] - next);
                    }
                }
                // Weighted values, by the same chunks of rows: per block, the
                // rows all of its tokens see go through together, the others per
                // token before and after them (rows ascending for every token).
                const float* values = mla_ex_.data() + heads * qk + h * vd;
                for (size_t c0 = lo; c0 < hi; c0 += kAttendRows) {
                    const size_t c1 = std::min(hi, c0 + kAttendRows);
                    auto clip = [&](size_t i0, size_t i1) {
                        return std::pair<size_t, size_t>{std::clamp(i0, c0, c1), std::clamp(i1, c0, c1)};
                    };
                    for (size_t j0 = 0; j0 < count; j0 += block) {
                        float* o[kScoreQueries];
                        const float* w[kScoreQueries];
                        const size_t in_block = std::min(block, count - j0);
                        size_t common0 = 0, common1 = 0;  // rows every token of the block sees, if no sinks
                        bool together = in_block == block;
                        for (size_t j = 0; j < in_block; ++j) {
                            o[j] = att + ((g0 + j0 + j) * heads + h) * vd;
                            w[j] = scores.data() + (j0 + j) * urows;
                            const auto& rg = ranges[j0 + j];
                            together = together && rg[0].first >= rg[0].second;
                            common0 = j == 0 ? rg[1].first : std::max(common0, rg[1].first);
                            common1 = j == 0 ? rg[1].second : std::min(common1, rg[1].second);
                        }
                        const auto [m0, m1] = clip(common0, common1);
                        if (together && m0 < m1) {
                            for (size_t j = 0; j < block; ++j) {
                                const auto [b0, b1] = clip(ranges[j0 + j][1].first, common0);
                                weigh_values<1>(&o[j], &w[j], values, row, b0, b1, vd);
                            }
                            weigh_values<kScoreQueries>(o, w, values, row, m0, m1, vd);
                            for (size_t j = 0; j < block; ++j) {
                                const auto [e0, e1] = clip(common1, ranges[j0 + j][1].second);
                                weigh_values<1>(&o[j], &w[j], values, row, e0, e1, vd);
                            }
                        } else {
                            for (size_t j = 0; j < in_block; ++j) {
                                for (const auto& [i0, i1] : ranges[j0 + j]) {
                                    const auto [r0, r1] = clip(i0, i1);
                                    weigh_values<1>(&o[j], &w[j], values, row, r0, r1, vd);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (size_t i = 0; i < un * heads; ++i) {
        if (mla_sum_[i] == 0.0f) continue;  // a pruned head stays zero
        const float inv = 1.0f / mla_sum_[i];
        float* o = att + i * vd;
        for (size_t d = 0; d < vd; ++d) o[d] *= inv;
    }
    return Status::ok();
}

// Lightning indexer: keys from the normalized block input, queries from
// the normalized query latent, learned signed weights over rectified head
// scores. Each sequence reads its own cache; tree queries read ancestors.
Status Transformer::indexer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto dim = static_cast<size_t>(c.index_dim);
    const auto heads = static_cast<size_t>(c.index_heads);
    const auto un = static_cast<size_t>(n);
    index_q_.resize(un * heads * dim);
    index_k_.resize(un * dim);
    index_weights_.resize(un * heads);
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kIndexQ], mla_qa_.data(), index_q_.data(), n));
    const TensorView* projections[] = {&w[kIndexK], &w[kIndexWeights]};
    float* outputs[] = {index_k_.data(), index_weights_.data()};
    LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, projections, xb_.data(), outputs, n));
    const RopeTable& table = rope_tables_[static_cast<size_t>(L.rope_table)];
    for (const Segment& s : segs_) {
        for (int32_t t = s.first; t < s.first + s.n; ++t) {
            float* key = index_k_.data() + static_cast<size_t>(t) * dim;
            double mean = 0.0, variance = 0.0;
            for (size_t i = 0; i < dim; ++i) mean += key[i];
            mean /= static_cast<double>(dim);
            for (size_t i = 0; i < dim; ++i) {
                const double delta = key[i] - mean;
                variance += delta * delta;
            }
            const float inv = static_cast<float>(1.0 / std::sqrt(variance / static_cast<double>(dim) + 1e-6));
            for (size_t i = 0; i < dim; ++i) {
                key[i] = (key[i] - static_cast<float>(mean)) * inv * L.index_norm[i] + L.index_bias[i];
            }
            const int32_t pos = pos_of(s, t);
            rope(key, 1, dim, pos, table, c.spec->indexer_rope);
            rope(index_q_.data() + static_cast<size_t>(t) * heads * dim, c.index_heads, dim, pos, table,
                 c.spec->indexer_rope);
            s.kv->store_auxiliary(L.kv_slot, row_of(s, t), key);
        }
    }
    index_positions_.resize(un);
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    const float weight_scale = 1.0f / std::sqrt(static_cast<float>(heads));
    pool.parallel_for(n, [&](int64_t begin, int64_t end) {
        thread_local std::vector<float> key, scores;
        thread_local std::vector<int32_t> order, slots;
        key.resize(dim);
        for (int64_t token = begin; token < end; ++token) {
            const auto t = static_cast<int32_t>(token);
            const auto ut = static_cast<size_t>(t);
            const Segment& s = segs_[segment_of(t)];
            const int32_t pos = pos_of(s, t);
            const Seen seen = seen_by(*s.kv, pos, L.window);
            const int32_t sinks = seen.sink_end - seen.sink_begin;
            const int32_t visible = sinks + pos - seen.first + 1;
            const int32_t count = std::min(c.index_topk, visible);
            const int32_t* ancestors = tree_depth_.empty() ? nullptr
                : tree_ancestors_.data() + static_cast<size_t>(t - s.first) * tree_stride_;
            slots.resize(static_cast<size_t>(visible));
            order.resize(static_cast<size_t>(visible));
            scores.resize(static_cast<size_t>(visible));
            for (int32_t i = 0; i < visible; ++i) {
                const int32_t p = i < sinks ? seen.sink_begin + i : seen.first + i - sinks;
                slots[static_cast<size_t>(i)] = ancestors == nullptr || p < s.n_past ? p
                    : s.n_past + ancestors[p - s.n_past];
                order[static_cast<size_t>(i)] = i;
                if (count == visible) continue;  // the dense identity, including its order and rounding
                std::memcpy(key.data(), s.kv->auxiliary_key(L.kv_slot, slots[static_cast<size_t>(i)], 0),
                            dim * sizeof(float));
                float score = 0.0f;
                for (size_t h = 0; h < heads; ++h) {
                    const float* q = index_q_.data() + (ut * heads + h) * dim;
                    const float dot = quant::dot_f32(q, key.data(), c.index_dim);
                    score += index_weights_[ut * heads + h] * weight_scale * std::max(0.0f, dot * scale);
                }
                scores[static_cast<size_t>(i)] = score;
            }
            if (count < visible) {
                std::partial_sort(order.begin(), order.begin() + count, order.end(), [&](int32_t a, int32_t b) {
                    return scores[static_cast<size_t>(a)] > scores[static_cast<size_t>(b)] ||
                        (scores[static_cast<size_t>(a)] == scores[static_cast<size_t>(b)] && a < b);
                });
                // Attention sums positions in causal order, independent of
                // batching, sorting implementation and the thread count.
                std::sort(order.begin(), order.begin() + count);
            }
            auto& selected = index_positions_[ut];
            selected.resize(static_cast<size_t>(count));
            for (int32_t i = 0; i < count; ++i) selected[static_cast<size_t>(i)] = slots[static_cast<size_t>(order[static_cast<size_t>(i)])];
        }
    });
    return Status::ok();
}

Status Transformer::latent_attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                           ThreadPool& pool, const uint8_t* head_mask) {
    using Clock = std::chrono::steady_clock;
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(layer)];
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const LatentAttention& m = c.mla;
    const auto un = static_cast<size_t>(n);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const auto nope = static_cast<size_t>(m.nope_dim);
    const auto vd = static_cast<size_t>(m.v_dim);
    const auto kd = static_cast<size_t>(c.head_dim);
    const size_t qk = nope + rd;
    const RopeTable& rope_table = rope_tables_[static_cast<size_t>(L.rope_table)];
    // Each segment attends in its own form (the one its pass alone would
    // take); a pass of a single sequence has one form.
    bool any_absorbed = false, any_expanded = false;
    for (const Segment& s : segs_) (s.expanded ? any_expanded : any_absorbed) = true;

    LIYAB_RETURN_IF_ERROR(latent_project(layer, w, n, route, route.attention, true));
    auto start = Clock::now();
    if (any_absorbed) LIYAB_RETURN_IF_ERROR(latent_absorb(w, n, route));  // every row: the same bits for each
    double form_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (const Segment& s : segs_) {
        for (auto t = static_cast<size_t>(s.first); t < static_cast<size_t>(s.first + s.n); ++t) {
            float* k = k_.data() + t * kd;
            const int32_t pos = pos_of(s, static_cast<int32_t>(t));
            if (s.expanded) {
                // Queries stay [q_nope | q_pe] per head; only q_pe turns. Keys are
                // stored scaled as in the absorbed form (the cache is shared).
                if (plan.rope) {
                    rope(mla_q_.data() + t * heads * qk + nope, c.n_head, qk, pos, rope_table);
                    rope(k, 1, pos, rope_table);
                }
                scale_keys(L, nullptr, k);
            } else {
                float* q = q_.data() + t * heads * kd;
                if (key_ranges_ != nullptr) add_key_ranges(L, q, k);  // unrotated: ranges cover every position
                if (plan.rope) {
                    rope(q, c.n_head, pos, rope_table);
                    rope(k, 1, pos, rope_table);
                }
                scale_keys(L, q, k);
            }
            s.kv->store(L.kv_slot, row_of(s, static_cast<int32_t>(t)), k, nullptr);  // the values are this row's latent channels
        }
    }
    if (!c.blocks[static_cast<size_t>(layer)].index_reuse) index_positions_.clear();
    if (c.blocks[static_cast<size_t>(layer)].indexed) LIYAB_RETURN_IF_ERROR(indexer(layer, w, n, route, pool));
    start = Clock::now();
    if (any_absorbed) {
        // att_: per absorbed token and head, the softmax-weighted latent
        // (value_dim = rank), rows `rank` wide per head.
        attention(L, n, pool, head_mask);

        // Per head, the weighted latent becomes the head's value: W_vb,h c. Keys
        // stored with per-channel scales carried their latent divided by them,
        // so the latent is multiplied back first (exact for each channel).
        // Every absorbed row is read before any is written in the [token]
        // [head][value] layout (the scratch holds max(head_dim, v_dim)), and expanded
        // segments write theirs after this (latent_expanded()).
        const float* scales =
            key_scale_.empty() ? nullptr : key_scale_.data() + static_cast<size_t>(L.kv_slot) * kd + rd;
        const int32_t* listed = any_expanded ? attn_rows_.data() : nullptr;  // the rows attention() attended
        const size_t m = any_expanded ? attn_rows_.size() : un;
        auto row_at = [listed](size_t i) { return listed == nullptr ? i : static_cast<size_t>(listed[i]); };
        mla_heads_in_.resize(heads * m * rank);
        for (size_t i = 0; i < m; ++i) {
            const size_t t = row_at(i);
            for (size_t h = 0; h < heads; ++h) {
                const float* src = att_.data() + (t * heads + h) * rank;
                float* dst = mla_heads_in_.data() + (h * m + i) * rank;
                if (scales == nullptr) {
                    std::copy(src, src + rank, dst);
                } else {
                    for (size_t j = 0; j < rank; ++j) dst[j] = src[j] * scales[j];
                }
            }
        }
        mla_heads_out_.resize(heads * m * vd);
        LIYAB_RETURN_IF_ERROR(per_head_matmul(route, w[kVb], mla_heads_in_.data(), mla_heads_out_.data(),
                                              static_cast<int32_t>(m)));
        for (size_t i = 0; i < m; ++i) {  // [token][head][value], the order attn_output reads
            const size_t t = row_at(i);
            for (size_t h = 0; h < heads; ++h) {
                const float* src = mla_heads_out_.data() + (h * m + i) * vd;
                std::copy(src, src + vd, att_.data() + (t * heads + h) * vd);
            }
        }
    }
    for (const Segment& s : segs_) {
        if (s.expanded) LIYAB_RETURN_IF_ERROR(latent_expanded(layer, w, s, pool, head_mask));
    }
    form_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    latent_ms_ += form_ms;
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kO], att_.data(), xb_.data(), n));
    add_bias(L.bo, xb_.data(), un);
    return Status::ok();
}

// Gated DeltaNet (Qwen3.5 / Qwen3-Next linear attention):
//   [q|k|v] = SiLU(causal_conv1d(W_qkv x)),  q, k L2-normalized, q scaled by 1/sqrt(d)
//   decay = exp(softplus(W_a x + dt_bias) * A),  b = sigmoid(W_b x)
//   S = decay * S;  S += b * (v - S k) k^T;  o = S q
//   out = W_out (RMSNorm(o) * SiLU(W_z x))      (per value head)
// Key/query heads are shared by value heads modulo their count (ggml's layout,
// into which the Qwen3.5 converter reorders the value heads), or, with
// ArchSpec::ssm_grouped_heads, by consecutive value heads (the reference's
// order, which Qwen3-Next's files keep). Qwen3-Next projects beta and alpha
// with one matrix (ssm_ba): per key head, [beta | alpha] of its value heads.
// A batch takes two parallel passes whatever its length: the convolution by
// channel heads through all tokens, then the recurrence by value head through
// all tokens (both are independent across their tasks, sequential in time).
// Token by token, a batch of n cost 2 n thread-pool rounds per block and ran
// the convolution, SiLU and L2 norms on one thread.
Status Transformer::shared_kv_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                   ThreadPool& pool, const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const BlockPlan& block = c.blocks[static_cast<size_t>(layer)];
    const CompressionPlan& compressor = block.compression;
    const auto un = static_cast<size_t>(n);
    const auto hd = static_cast<size_t>(c.head_dim), heads = static_cast<size_t>(c.n_head);
    const auto qr = static_cast<size_t>(c.shared_q_rank), id = static_cast<size_t>(c.index_dim);
    const auto ih = static_cast<size_t>(c.index_heads);
    const RopeTable& table = rope_tables_[static_cast<size_t>(L.rope_table)];
    mla_qa_.resize(un * qr);
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kQa], xb_.data(), mla_qa_.data(), n));
    for (size_t t = 0; t < un; ++t) {
        rmsnorm(mla_qa_.data() + t * qr, L.q_a_norm.data(), mla_qa_.data() + t * qr, c.shared_q_rank, c.rms_eps);
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kQb], mla_qa_.data(), q_.data(), n));
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kSharedKv], xb_.data(), k_.data(), n));
    if (compressor.stride > 0) {
        const size_t width = hd * (compressor.overlap ? 2 : 1);
        compress_kv_.resize(un * width);
        compress_gate_.resize(un * width);
        const TensorView* projections[] = {&w[kCompressKv], &w[kCompressGate]};
        float* outputs[] = {compress_kv_.data(), compress_gate_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, projections, xb_.data(), outputs, n));
        if (block.indexed) {
            index_compress_kv_.resize(un * 2 * id);
            index_compress_gate_.resize(un * 2 * id);
            index_q_.resize(un * ih * id);
            index_weights_.resize(un * ih);
            const TensorView* ip[] = {&w[kIndexCompressKv], &w[kIndexCompressGate], &w[kIndexWeights]};
            float* io[] = {index_compress_kv_.data(), index_compress_gate_.data(), index_weights_.data()};
            LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, ip, xb_.data(), io, n));
            LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kIndexQ], mla_qa_.data(), index_q_.data(), n));
        }
    }
    // Main shared K=V rows first. A row's auxiliary channels are cleared by
    // store(); a completed compressor writes them below at its closing token.
    for (const Segment& s : segs_) {
        for (int32_t t = s.first; t < s.first + s.n; ++t) {
            const int32_t pos = pos_of(s, t);
            float* query = q_.data() + static_cast<size_t>(t) * heads * hd;
            for (size_t h = 0; h < heads; ++h) {
                float* q = query + h * hd;
                const float scale = rms_scale(q, c.head_dim, c.rms_eps);
                for (size_t j = 0; j < hd; ++j) q[j] *= scale;
            }
            rope(query + hd - static_cast<size_t>(c.rope_dim), c.n_head, hd, pos, table);
            float* key = k_.data() + static_cast<size_t>(t) * hd;
            rmsnorm(key, L.kv_a_norm.data(), key, c.head_dim, c.rms_eps);
            rope(key + hd - static_cast<size_t>(c.rope_dim), 1, hd, pos, table);
            s.kv->store(L.kv_slot, row_of(s, t), key, nullptr);
            if (block.indexed) {
                rope(index_q_.data() + static_cast<size_t>(t) * ih * id + id - static_cast<size_t>(c.rope_dim),
                     c.index_heads, id, pos, table);
            }
        }
    }
    if (compressor.stride > 0) {
        const size_t width = hd * (compressor.overlap ? 2 : 1);
        const CompressionPlan index_compressor{c.index_dim, compressor.stride, true};
        compression_scratch_.resize(2 * std::max(hd, id));
        std::vector<float> emitted(hd + id, 0.0f);
        for (const Segment& s : segs_) {
            RecurrentState& state = (*s.states)[static_cast<size_t>(L.state_slot)];
            auto checkpoint = [&](int32_t i) {
                return state.checkpoints.data() + static_cast<size_t>((s.n_past + i) % s.rollback) * state.conv.size();
            };
            for (int32_t i = 0; i < s.n; ++i) {
                const int32_t t = s.first + i, pos = pos_of(s, t);
                if (!tree_depth_.empty() && i > 0 && tree_parent_[static_cast<size_t>(i)] != i - 1) {
                    std::copy_n(checkpoint(tree_parent_[static_cast<size_t>(i)]), state.conv.size(), state.conv.begin());
                }
                const auto phase = static_cast<size_t>(pos % compressor.stride);
                const bool closed = compressor.step(std::span<float>(state.conv).first(compressor.state_floats()),
                    compress_kv_.data() + static_cast<size_t>(t) * width,
                    compress_gate_.data() + static_cast<size_t>(t) * width,
                    L.compress_bias.data() + phase * width, pos, emitted.data(), compression_scratch_);
                if (block.indexed) {
                    index_compressor.step(std::span<float>(state.conv).subspan(compressor.state_floats()),
                        index_compress_kv_.data() + static_cast<size_t>(t) * 2 * id,
                        index_compress_gate_.data() + static_cast<size_t>(t) * 2 * id,
                        L.index_compress_bias.data() + phase * 2 * id, pos, emitted.data() + hd, compression_scratch_);
                }
                if (closed) {
                    rmsnorm(emitted.data(), L.compress_norm.data(), emitted.data(), c.head_dim, c.rms_eps);
                    const int32_t start = pos + 1 - compressor.stride;
                    rope(emitted.data() + hd - static_cast<size_t>(c.rope_dim), 1, hd, start, table);
                    if (block.indexed) {
                        rmsnorm(emitted.data() + hd, L.index_compress_norm.data(), emitted.data() + hd, c.index_dim, c.rms_eps);
                        rope(emitted.data() + hd + id - static_cast<size_t>(c.rope_dim), 1, id, start, table);
                    }
                    s.kv->store_auxiliary(L.kv_slot, row_of(s, t), emitted.data());
                }
                if (s.checkpoint && i + s.rollback >= s.n) {
                    std::copy(state.conv.begin(), state.conv.end(), checkpoint(i));
                }
            }
        }
    }
    // Translate a branch's logical position through its ancestry. The emitted
    // compressed entry is carried by the closing token's own row, so sibling
    // branches never share an incomplete or freshly closed window.
    auto physical = [&](const Segment& s, int32_t t, int32_t pos) {
        if (tree_depth_.empty() || pos < s.n_past) return pos;
        return s.n_past + tree_ancestors_[static_cast<size_t>(t - s.first) * static_cast<size_t>(tree_stride_) +
                                         static_cast<size_t>(pos - s.n_past)];
    };
    index_positions_.assign(un, {});
    if (compressor.stride > 0) {
        for (const Segment& s : segs_) {
            for (int32_t t = s.first; t < s.first + s.n; ++t) {
                const int32_t ready = (pos_of(s, t) + 1) / compressor.stride;
                std::vector<std::pair<float, int32_t>> scores;
                std::vector<float> key(id);
                for (int32_t entry = 0; entry < ready; ++entry) {
                    const int32_t row = physical(s, t, (entry + 1) * compressor.stride - 1);
                    float score = 0.0f;
                    if (block.indexed) {
                        std::memcpy(key.data(), s.kv->auxiliary_key(L.kv_slot, row, 0) + hd * sizeof(float), id * sizeof(float));
                        for (size_t h = 0; h < ih; ++h) {
                            const float dot = quant::dot_f32(index_q_.data() + (static_cast<size_t>(t) * ih + h) * id,
                                                             key.data(), c.index_dim);
                            score += std::max(dot, 0.0f) / std::sqrt(static_cast<float>(id)) *
                                     (index_weights_[static_cast<size_t>(t) * ih + h] / std::sqrt(static_cast<float>(ih)));
                        }
                    }
                    scores.emplace_back(score, row);
                }
                const auto count = block.indexed ? std::min(scores.size(), static_cast<size_t>(c.index_topk)) : scores.size();
                if (count < scores.size()) {
                    std::partial_sort(scores.begin(), scores.begin() + static_cast<std::ptrdiff_t>(count), scores.end(),
                        [](const auto& a, const auto& b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
                }
                auto& selected = index_positions_[static_cast<size_t>(t)];
                for (size_t i = 0; i < count; ++i) selected.push_back(scores[i].second);
                std::sort(selected.begin(), selected.end());
            }
        }
    }
    // Existing F32 dot/dequantize kernels provide NEON, AVX2 and scalar
    // paths for every main KV type. Compressed and index channels remain F32.
    pool.parallel_for(static_cast<int64_t>(un * heads), [&](int64_t begin, int64_t end) {
        thread_local std::vector<float> key, scores, values;
        key.resize(hd);
        for (int64_t job = begin; job < end; ++job) {
            const auto t = static_cast<int32_t>(static_cast<size_t>(job) / heads);
            const auto h = static_cast<size_t>(job) % heads;
            float* out = att_.data() + static_cast<size_t>(job) * hd;
            std::fill_n(out, hd, 0.0f);
            if (head_mask != nullptr && head_mask[h] == 0) continue;
            const Segment& s = segs_[segment_of(t)];
            const int32_t pos = pos_of(s, t);
            const int32_t first = L.window > 0 ? std::max(0, pos - L.window + 1) : 0;
            const auto& compressed = index_positions_[static_cast<size_t>(t)];
            const size_t raw = static_cast<size_t>(pos - first + 1), count = raw + compressed.size();
            values.resize(count * hd);
            scores.resize(count);
            const float* query = q_.data() + static_cast<size_t>(job) * hd;
            const float sink = L.sinks.empty() ? -INFINITY : L.sinks[h];
            float maximum = sink;
            for (size_t j = 0; j < count; ++j) {
                float* value = values.data() + j * hd;
                if (j < raw) {
                    quant::dequantize_row(s.kv->dtype(), s.kv->k_row(L.kv_slot, physical(s, t, first + static_cast<int32_t>(j)), 0),
                                          value, c.head_dim);
                } else {
                    std::memcpy(value, s.kv->auxiliary_key(L.kv_slot, compressed[j - raw], 0), hd * sizeof(float));
                }
                float score = quant::dot_f32(query, value, c.head_dim) * c.attn_scale;
                if (c.attn_softcap > 0.0f) score = c.attn_softcap * portable_tanh(score / c.attn_softcap);
                scores[j] = score;
                maximum = std::max(maximum, score);
            }
            float sum = L.sinks.empty() ? 0.0f : std::exp(sink - maximum);
            for (size_t j = 0; j < count; ++j) {
                scores[j] = std::exp(scores[j] - maximum);
                sum += scores[j];
            }
            for (size_t j = 0; j < count; ++j) {
                quant::axpy_row(DType::F32, values.data() + j * hd, scores[j] / sum, out, c.head_dim);
            }
            rope(out + hd - static_cast<size_t>(c.rope_dim), 1, hd, -pos, table);
        }
    });
    const auto groups = static_cast<size_t>(c.output_groups), rank = static_cast<size_t>(c.output_rank);
    const size_t cols = heads * hd / groups;
    mla_heads_in_.resize(un * cols);
    mla_heads_out_.resize(un * rank);
    mla_kv_.resize(un * groups * rank);
    for (size_t group = 0; group < groups; ++group) {
        for (size_t t = 0; t < un; ++t) std::copy_n(att_.data() + t * heads * hd + group * cols, cols, mla_heads_in_.data() + t * cols);
        TensorView projection = w[kOutputA];
        projection.data = projection.row(static_cast<int64_t>(group * rank));
        projection.ne[1] = c.output_rank;
        projection.nbytes = rank * projection.row_bytes();
        LIYAB_RETURN_IF_ERROR(route.cpu->matmul(projection, mla_heads_in_.data(), mla_heads_out_.data(), n));
        for (size_t t = 0; t < un; ++t) std::copy_n(mla_heads_out_.data() + t * rank, rank, mla_kv_.data() + (t * groups + group) * rank);
    }
    return matmul(route, route.attention, w[kOutputB], mla_kv_.data(), xb_.data(), n);
}

Status Transformer::delta_net_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                    ThreadPool& pool) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto hd = static_cast<size_t>(c.ssm_head_dim);
    const auto hk = static_cast<size_t>(c.ssm_k_heads);
    const auto hv = static_cast<size_t>(c.ssm_v_heads);
    const size_t key_dim = hk * hd;
    const size_t value_dim = hv * hd;
    const size_t conv_dim = 2 * key_dim + value_dim;
    const auto taps = static_cast<size_t>(c.ssm_conv_kernel);
    const size_t history = taps - 1;

    mix_.resize(un * conv_dim);
    z_.resize(un * value_dim);
    alpha_.resize(un * hv);
    beta_.resize(un * hv);
    dn_.resize(un * value_dim);
    if (L.w[kBa] != nullptr) {
        ba_.resize(un * 2 * hv);
        const TensorView* proj[] = {&w[kQkv], &w[kZ], &w[kBa]};
        float* outs[] = {mix_.data(), z_.data(), ba_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, proj, xb_.data(), outs, n));
        const size_t per_key = hv / hk;  // value heads per key head
        for (size_t t = 0; t < un; ++t) {
            for (size_t h = 0; h < hv; ++h) {
                const size_t g = h / per_key, j = h % per_key;
                beta_[t * hv + h] = ba_[t * 2 * hv + g * 2 * per_key + j];
                alpha_[t * hv + h] = ba_[t * 2 * hv + g * 2 * per_key + per_key + j];
            }
        }
    } else {
        const TensorView* proj[] = {&w[kQkv], &w[kZ], &w[kAlpha], &w[kBeta]};
        float* outs[] = {mix_.data(), z_.data(), alpha_.data(), beta_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, proj, xb_.data(), outs, n));
    }
    const size_t per_key_head = hv / hk;
    const bool grouped = c.spec->ssm_grouped_heads;
    auto key_head = [&](size_t h) { return grouped ? h / per_key_head : h % hk; };

    // Each segment continues its own states (segments_ in order: tasks below
    // are (segment, head) pairs; a segment's token i is batch row first + i).
    // Checkpoints: the state after each of the last `rollback` tokens of the
    // segment, for truncate(); slot = position % rollback, layout [conv | ssm].
    auto state_of = [&](const Segment& s) -> RecurrentState& { return (*s.states)[static_cast<size_t>(L.state_slot)]; };
    auto slot_of = [&](const Segment& s, RecurrentState& st, size_t i) -> float* {
        if (!s.checkpoint || i + static_cast<size_t>(s.rollback) < static_cast<size_t>(s.n)) return nullptr;
        return st.checkpoints.data() + static_cast<size_t>((s.n_past + static_cast<int32_t>(i)) % s.rollback) *
                                           (st.conv.size() + st.ssm.size());
    };

    // Tree pass (one segment): a token whose parent is not the token before
    // it (a branch other than the first) continues from its parent's state,
    // which the parent's checkpoint holds (forward_tree() keeps one for every
    // token).
    const int32_t* parent = tree_depth_.empty() ? nullptr : tree_parent_.data();
    auto branches = [&](size_t i) { return parent != nullptr && i > 0 && parent[i] != static_cast<int32_t>(i) - 1; };
    const auto segments = static_cast<int64_t>(segs_.size());

    // 1. Causal depthwise convolution over [history, current] and SiLU, every
    //    channel through all of a segment's tokens (channels are independent,
    //    tokens are not), then the L2 norm of each query / key head. One task
    //    per segment and head of channels: 2 hk query/key heads, then hv value
    //    heads.
    const auto conv_tasks = static_cast<int64_t>(2 * hk + hv);
    pool.parallel_for(conv_tasks * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / conv_tasks)];
            RecurrentState& st = state_of(s);
            const auto task = static_cast<size_t>(job % conv_tasks);
            const auto first = static_cast<size_t>(s.first);
            const auto count = static_cast<size_t>(s.n);
            const size_t ch0 = task * hd;  // query heads, key heads and value heads are hd channels each
            for (size_t ch = ch0; ch < ch0 + hd; ++ch) {
                float* hist = st.conv.data() + ch * history;
                const float* tap = L.conv1d.data() + ch * taps;
                for (size_t i = 0; i < count; ++i) {
                    if (branches(i)) {
                        const float* from = slot_of(s, st, static_cast<size_t>(parent[i])) + ch * history;
                        std::copy(from, from + history, hist);
                    }
                    float* m = mix_.data() + (first + i) * conv_dim;
                    float acc = m[ch] * tap[history];
                    for (size_t j = 0; j < history; ++j) acc += hist[j] * tap[j];
                    for (size_t j = 0; j + 1 < history; ++j) hist[j] = hist[j + 1];
                    hist[history - 1] = m[ch];
                    m[ch] = silu(acc);
                    if (float* slot = slot_of(s, st, i)) std::copy(hist, hist + history, slot + ch * history);
                }
            }
            if (task < 2 * hk) {
                for (size_t i = 0; i < count; ++i) {
                    l2norm(mix_.data() + (first + i) * conv_dim + ch0, static_cast<int32_t>(hd), c.rms_eps);
                }
            }
        }
    });

    // 2. The recurrence, one task per segment and value head through all of
    //    the segment's tokens (heads are independent), then the head's gated
    //    RMSNorm.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto heads = static_cast<int64_t>(hv);
    pool.parallel_for(heads * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / heads)];
            RecurrentState& st = state_of(s);
            const auto h = static_cast<size_t>(job % heads);
            const auto first = static_cast<size_t>(s.first);
            float* S = st.ssm.data() + h * hd * hd;
            for (size_t i = 0; i < static_cast<size_t>(s.n); ++i) {
                if (branches(i)) {
                    const float* from = slot_of(s, st, static_cast<size_t>(parent[i])) + st.conv.size() + h * hd * hd;
                    std::copy(from, from + hd * hd, S);
                }
                const size_t t = first + i;
                const float* m = mix_.data() + t * conv_dim;
                const float* q = m + key_head(h) * hd;
                const float* k = m + key_dim + key_head(h) * hd;
                const float* v = m + 2 * key_dim + h * hd;
                const float decay = std::exp(softplus(alpha_[t * hv + h] + L.ssm_dt[h]) * L.ssm_a[h]);
                const float strength = sigmoid(beta_[t * hv + h]);
                float* o = dn_.data() + t * value_dim + h * hd;
                delta_rule_step(S, q, k, v, decay, strength, q_scale, o, hd);
                const float* z = z_.data() + t * value_dim + h * hd;
                rmsnorm(o, L.ssm_norm.data(), o, static_cast<int32_t>(hd), c.rms_eps);
                for (size_t j = 0; j < hd; ++j) o[j] *= silu(z[j]);
                if (float* slot = slot_of(s, st, i)) std::copy(S, S + hd * hd, slot + st.conv.size() + h * hd * hd);
            }
        }
    });
    return matmul(route, route.attention, w[kSsmOut], dn_.data(), xb_.data(), n);
}

float* Transformer::checkpoint_slot(const Segment& s, RecurrentState& st, size_t i) noexcept {
    if (!s.checkpoint || i + static_cast<size_t>(s.rollback) < static_cast<size_t>(s.n)) return nullptr;
    return st.checkpoints.data() +
           static_cast<size_t>((s.n_past + static_cast<int32_t>(i)) % s.rollback) * (st.conv.size() + st.ssm.size());
}

// Mamba's causal depthwise convolution, one task per (segment, channel): the
// channel's history and its inputs through the segment's tokens are gathered
// into one run and convolved at once (ssm::causal_conv, vectorized over the
// tokens), then go through SiLU. The history after token i is the run's
// values i + 1 .. i + kernel - 1, which the checkpoints copy. A tree pass,
// whose tokens may continue another branch from its parent's checkpointed
// history, goes token by token.
void Transformer::mamba_conv(const Layer& L, const float* in, size_t in_stride, float* out, size_t out_stride,
                             size_t col, size_t channels, ThreadPool& pool) {
    const auto taps = static_cast<size_t>(config_.mamba.conv_kernel);
    const size_t history = taps - 1;
    const auto segments = static_cast<int64_t>(segs_.size());
    const bool tree = !tree_depth_.empty();
    pool.parallel_for(static_cast<int64_t>(channels) * segments, [&](int64_t begin, int64_t end) {
        std::vector<float> seq, conv;
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / static_cast<int64_t>(channels))];
            RecurrentState& st = (*s.states)[static_cast<size_t>(L.state_slot)];
            const auto ch = static_cast<size_t>(job % static_cast<int64_t>(channels));
            const auto first = static_cast<size_t>(s.first);
            const auto count = static_cast<size_t>(s.n);
            float* hist = history > 0 ? st.conv.data() + ch * history : nullptr;
            const float* tap = L.conv1d.data() + ch * taps;
            seq.resize(history + count);
            conv.resize(count);
            if (history > 0) std::copy_n(hist, history, seq.begin());
            for (size_t i = 0; i < count; ++i) seq[history + i] = in[(first + i) * in_stride + col + ch];
            if (!tree) {
                ssm::causal_conv(seq.data(), tap, L.conv_bias[ch], taps, count, conv.data());
                if (history > 0) std::copy(seq.end() - static_cast<std::ptrdiff_t>(history), seq.end(), hist);
                for (size_t i = 0; i < count; ++i) {
                    if (float* slot = checkpoint_slot(s, st, i)) {
                        std::copy_n(seq.data() + i + 1, history, slot + ch * history);
                    }
                }
            } else {
                // hist holds the history before token i; a token continuing
                // another branch takes its parent's instead.
                for (size_t i = 0; i < count; ++i) {
                    if (history > 0 && tree_branches(i)) {
                        const float* from =
                            checkpoint_slot(s, st, static_cast<size_t>(tree_parent_[i])) + ch * history;
                        std::copy(from, from + history, hist);
                    }
                    const float x = seq[history + i];
                    float acc = L.conv_bias[ch];
                    for (size_t j = 0; j < history; ++j) acc = std::fma(tap[j], hist[j], acc);
                    conv[i] = std::fma(tap[history], x, acc);
                    if (history > 0) {
                        std::copy(hist + 1, hist + history, hist);
                        hist[history - 1] = x;
                    }
                    if (history > 0) {
                        if (float* slot = checkpoint_slot(s, st, i)) std::copy_n(hist, history, slot + ch * history);
                    }
                }
            }
            ssm::silu(conv.data(), conv.data(), count);
            for (size_t i = 0; i < count; ++i) out[(first + i) * out_stride + col + ch] = conv[i];
        }
    });
}

// Mamba-1 (Mamba, Falcon-Mamba), per token:
//   [x | z] = W_in u;  x = SiLU(conv1d(x) + b_conv)
//   [dt | B | C] = W_x x  (Falcon-Mamba: each RMS-normalized, without weights)
//   dt = softplus(W_dt dt + b_dt)                     (one per channel)
//   h[c][j] = exp(dt[c] A[c][j]) h[c][j] + dt[c] x[c] B[j]
//   y[c] = (C . h[c] + D[c] x[c]) SiLU(z[c]);  out = W_out y
// The convolution, then the scan, run as one task per (segment, channel)
// through all of the segment's tokens; the projections between them run over
// the whole batch.
Status Transformer::mamba_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                ThreadPool& pool) {
    const ModelConfig& c = config_;
    const MambaGeometry& m = c.mamba;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto di = static_cast<size_t>(m.d_inner);
    const auto ns = static_cast<size_t>(m.d_state);
    const auto rank = static_cast<size_t>(m.dt_rank);
    const size_t dbc = rank + 2 * ns;
    mix_.resize(un * 2 * di);
    ssm_xc_.resize(un * di);
    ssm_dbc_.resize(un * dbc);
    ssm_dtl_.resize(un * rank);
    ssm_dt_.resize(un * di);
    z_.resize(un * di);
    dn_.resize(un * di);

    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kSsmIn], xb_.data(), mix_.data(), n));
    for (size_t t = 0; t < un && !L.ssm_in_bias.empty(); ++t) {
        for (size_t i = 0; i < 2 * di; ++i) mix_[t * 2 * di + i] += L.ssm_in_bias[i];
    }
    mamba_conv(L, mix_.data(), 2 * di, ssm_xc_.data(), di, 0, di, pool);
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kSsmX], ssm_xc_.data(), ssm_dbc_.data(), n));
    // Norms of dt, B and C: weightless (ArchSpec::ssm_dt_b_c_eps) or with
    // weights of their own (the file's epsilon).
    for (size_t t = 0; t < un; ++t) {
        float* row = ssm_dbc_.data() + t * dbc;
        if (m.dt_b_c_rms) {
            for (const auto& [at, len] : {std::pair{size_t{0}, rank}, std::pair{rank, ns}, std::pair{rank + ns, ns}}) {
                const float scale = rms_scale(row + at, static_cast<int32_t>(len), c.spec->ssm_dt_b_c_eps);
                for (size_t i = at; i < at + len; ++i) row[i] *= scale;
            }
        } else if (!L.dt_norm.empty()) {
            rmsnorm(row, L.dt_norm.data(), row, static_cast<int32_t>(rank), c.rms_eps);
            rmsnorm(row + rank, L.b_norm.data(), row + rank, static_cast<int32_t>(ns), c.rms_eps);
            rmsnorm(row + rank + ns, L.c_norm.data(), row + rank + ns, static_cast<int32_t>(ns), c.rms_eps);
        }
        std::copy_n(row, rank, ssm_dtl_.data() + t * rank);
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kSsmDt], ssm_dtl_.data(), ssm_dt_.data(), n));
    // The gate and the step size, token by token.
    pool.parallel_for(n, [&](int64_t begin, int64_t end) {
        for (auto t = static_cast<size_t>(begin); t < static_cast<size_t>(end); ++t) {
            ssm::silu(mix_.data() + t * 2 * di + di, z_.data() + t * di, di);
            float* dt = ssm_dt_.data() + t * di;
            for (size_t i = 0; i < di; ++i) dt[i] = softplus(dt[i] + L.ssm_dt[i]);
        }
    });

    // The scan, one task per (segment, channel). Tokens that record no
    // checkpoint go through the kernel in one run (a tree pass checkpoints
    // every token).
    const auto segments = static_cast<int64_t>(segs_.size());
    pool.parallel_for(static_cast<int64_t>(di) * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / static_cast<int64_t>(di))];
            RecurrentState& st = (*s.states)[static_cast<size_t>(L.state_slot)];
            const auto ch = static_cast<size_t>(job % static_cast<int64_t>(di));
            const auto first = static_cast<size_t>(s.first);
            const auto count = static_cast<size_t>(s.n);
            float* h = st.ssm.data() + ch * ns;
            const float* a = L.ssm_a.data() + ch * ns;
            auto scan = [&](size_t i, size_t tokens) {
                const size_t t = first + i;
                ssm::mamba_scan(h, a, ns, ssm_dt_.data() + t * di + ch, ssm_xc_.data() + t * di + ch, di,
                                ssm_dbc_.data() + t * dbc + rank, dbc, dn_.data() + t * di + ch, tokens);
            };
            const size_t recorded = s.checkpoint ? std::min(count, static_cast<size_t>(s.rollback)) : 0;
            const size_t plain = count - recorded;  // tokens before these record no checkpoint
            if (plain > 0) scan(0, plain);
            for (size_t i = plain; i < count; ++i) {
                if (tree_branches(i)) {
                    const float* from =
                        checkpoint_slot(s, st, static_cast<size_t>(tree_parent_[i])) + st.conv.size() + ch * ns;
                    std::copy(from, from + ns, h);
                }
                scan(i, 1);
                if (float* slot = checkpoint_slot(s, st, i)) std::copy(h, h + ns, slot + st.conv.size() + ch * ns);
            }
            for (size_t i = 0; i < count; ++i) {
                const size_t at = (first + i) * di + ch;
                dn_[at] = (dn_[at] + L.ssm_d[ch] * ssm_xc_[at]) * z_[at];
            }
        }
    });
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kSsmOut], dn_.data(), xb_.data(), n));
    const auto d = static_cast<size_t>(c.n_embd);
    for (size_t t = 0; t < un && !L.ssm_out_bias.empty(); ++t) {
        for (size_t i = 0; i < d; ++i) xb_[t * d + i] += L.ssm_out_bias[i];
    }
    return Status::ok();
}

// Mamba-2 (SSD: Mamba-2, Mamba-Codestral), per token, with H heads of P
// channels and G groups of heads sharing B and C:
//   [z | x B C | dt] = W_in u;  [x | B | C] = SiLU(conv1d([x | B | C]) + b_conv)
//   dt_h = softplus(dt_h + dt_bias_h);  decay_h = exp(dt_h A_h)   (A_h < 0)
//   H_h = decay_h H_h + dt_h x_h B_g^T;  y_h = H_h C_g + D_h x_h  (H_h: P x N)
//   y = RMSNorm_g(y SiLU(z)) w   (over each group of d_inner / G channels)
//   out = W_out y
// The scan runs as the recurrence it is, one task per (segment, head), every
// token an O(P N) step in place on the head's state: a prefill does exactly
// the work of its decoding steps, and every token's state is there for the
// rollback checkpoints and tree branches. The norm spans each group or the
// complete inner stream according to ArchSpec::ssm_norm_grouped: the
// converter reshapes the weights identically for both reference contracts.
Status Transformer::mamba2_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                 ThreadPool& pool) {
    const ModelConfig& c = config_;
    const MambaGeometry& m = c.mamba;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto di = static_cast<size_t>(m.d_inner);
    const auto ns = static_cast<size_t>(m.d_state);
    const auto heads = static_cast<size_t>(m.n_heads);
    const auto groups = static_cast<size_t>(m.n_groups);
    const size_t hd = di / heads;
    const size_t channels = di + 2 * groups * ns;  // [x | B | C]
    const size_t row = di + channels + heads;      // [z | x B C | dt]
    mix_.resize(un * row);
    z_.resize(un * di);
    dn_.resize(un * di);

    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kSsmIn], xb_.data(), mix_.data(), n));
    for (size_t t = 0; t < un && !L.ssm_in_bias.empty(); ++t) {
        for (size_t i = 0; i < row; ++i) mix_[t * row + i] += L.ssm_in_bias[i];
    }
    mamba_conv(L, mix_.data(), row, mix_.data(), row, di, channels, pool);

    const auto segments = static_cast<int64_t>(segs_.size());
    const size_t per_group = heads / groups;
    pool.parallel_for(static_cast<int64_t>(heads) * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / static_cast<int64_t>(heads))];
            RecurrentState& st = (*s.states)[static_cast<size_t>(L.state_slot)];
            const auto h = static_cast<size_t>(job % static_cast<int64_t>(heads));
            const auto first = static_cast<size_t>(s.first);
            const size_t g = h / per_group;
            float* state = st.ssm.data() + h * hd * ns;
            for (size_t i = 0; i < static_cast<size_t>(s.n); ++i) {
                if (tree_branches(i)) {
                    const float* from = checkpoint_slot(s, st, static_cast<size_t>(tree_parent_[i])) +
                                        st.conv.size() + h * hd * ns;
                    std::copy(from, from + hd * ns, state);
                }
                const size_t t = first + i;
                const float* r = mix_.data() + t * row;
                const float* x = r + di + h * hd;
                const float* b = r + 2 * di + g * ns;
                const float* cc = r + 2 * di + groups * ns + g * ns;
                const float dt = softplus(r[di + channels + h] + L.ssm_dt[h]);
                float* y = dn_.data() + t * di + h * hd;
                ssm::mamba2_step(state, hd, ns, std::exp(dt * L.ssm_a[h]), dt, x, b, cc, y);
                for (size_t p = 0; p < hd; ++p) y[p] += L.ssm_d[h] * x[p];
                if (float* slot = checkpoint_slot(s, st, i)) {
                    std::copy(state, state + hd * ns, slot + st.conv.size() + h * hd * ns);
                }
            }
        }
    });
    // The gated norm, token by token: y SiLU(z), then RMSNorm per group.
    const size_t norm_groups = c.spec->ssm_norm_grouped ? groups : 1;
    const size_t group = di / norm_groups;
    auto gate_norm = [&](int64_t begin, int64_t end) {
        for (auto t = static_cast<size_t>(begin); t < static_cast<size_t>(end); ++t) {
            float* y = dn_.data() + t * di;
            float* z = z_.data() + t * di;
            ssm::silu(mix_.data() + t * row, z, di);
            for (size_t i = 0; i < di; ++i) y[i] *= z[i];
            for (size_t k = 0; k < norm_groups; ++k) {
                rmsnorm(y + k * group, L.ssm_norm.data() + k * group, y + k * group, static_cast<int32_t>(group),
                        c.rms_eps);
            }
        }
    };
    if (n > 1) {
        pool.parallel_for(n, gate_norm);
    } else {
        gate_norm(0, n);
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kSsmOut], dn_.data(), xb_.data(), n));
    const auto d = static_cast<size_t>(c.n_embd);
    for (size_t t = 0; t < un && !L.ssm_out_bias.empty(); ++t) {
        for (size_t i = 0; i < d; ++i) xb_[t * d + i] += L.ssm_out_bias[i];
    }
    return Status::ok();
}


// LFM2's gated short convolution:
//   [B | C | x] = W_in h (+ b_in);  y = C * (conv(B * x) + b_conv);  out = W_out y (+ b_out)
// conv: causal depthwise over the last l_cache positions of B * x, without an
// activation. Channels are independent, so a batch takes one parallel pass:
// one task per segment and run of kShortConvChannels channels through all of
// the segment's tokens (tokens are not independent). The convolution costs
// 3 l_cache operations per channel and token, next to the 4 n_embd of the two
// projections: plain loops, whatever the CPU.
Status Transformer::short_conv_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                     ThreadPool& pool) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto d = static_cast<size_t>(c.n_embd);
    const auto taps = static_cast<size_t>(c.shortconv_l_cache);
    const size_t history = taps - 1;
    mix_.resize(un * 3 * d);
    dn_.resize(un * d);
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kConvIn], xb_.data(), mix_.data(), n));
    if (!L.in_bias.empty()) {
        for (size_t t = 0; t < un; ++t) {
            for (size_t i = 0; i < 3 * d; ++i) mix_[t * 3 * d + i] += L.in_bias[i];
        }
    }
    // Checkpoints and tree branches as in delta_net_mixer(): slot layout [conv].
    auto state_of = [&](const Segment& s) -> RecurrentState& { return (*s.states)[static_cast<size_t>(L.state_slot)]; };
    auto slot_of = [&](const Segment& s, RecurrentState& st, size_t i) -> float* {
        if (!s.checkpoint || i + static_cast<size_t>(s.rollback) < static_cast<size_t>(s.n)) return nullptr;
        return st.checkpoints.data() + static_cast<size_t>((s.n_past + static_cast<int32_t>(i)) % s.rollback) *
                                           (st.conv.size() + st.ssm.size());
    };
    const int32_t* parent = tree_depth_.empty() ? nullptr : tree_parent_.data();
    auto branches = [&](size_t i) { return parent != nullptr && i > 0 && parent[i] != static_cast<int32_t>(i) - 1; };
    constexpr size_t kShortConvChannels = 64;
    const size_t runs = (d + kShortConvChannels - 1) / kShortConvChannels;
    const auto tasks = static_cast<int64_t>(runs);
    pool.parallel_for(tasks * static_cast<int64_t>(segs_.size()), [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / tasks)];
            RecurrentState& st = state_of(s);
            const size_t ch0 = static_cast<size_t>(job % tasks) * kShortConvChannels;
            const size_t ch1 = std::min(d, ch0 + kShortConvChannels);
            const auto first = static_cast<size_t>(s.first);
            for (size_t ch = ch0; ch < ch1; ++ch) {
                float* hist = history > 0 ? st.conv.data() + ch * history : nullptr;
                const float* tap = L.conv1d.data() + ch * taps;
                const float bias = L.conv_bias.empty() ? 0.0f : L.conv_bias[ch];
                for (size_t i = 0; i < static_cast<size_t>(s.n); ++i) {
                    if (history > 0 && branches(i)) {
                        const float* from = slot_of(s, st, static_cast<size_t>(parent[i])) + ch * history;
                        std::copy(from, from + history, hist);
                    }
                    const float* m = mix_.data() + (first + i) * 3 * d;
                    const float bx = m[ch] * m[2 * d + ch];
                    float acc = 0.0f;
                    for (size_t j = 0; j < history; ++j) acc += hist[j] * tap[j];
                    acc += bx * tap[history];
                    for (size_t j = 0; j + 1 < history; ++j) hist[j] = hist[j + 1];
                    if (history > 0) hist[history - 1] = bx;
                    dn_[(first + i) * d + ch] = m[d + ch] * (acc + bias);
                    if (history > 0) {
                        if (float* slot = slot_of(s, st, i)) std::copy(hist, hist + history, slot + ch * history);
                    }
                }
            }
        }
    });
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kConvOut], dn_.data(), xb_.data(), n));
    if (!L.out_bias.empty()) {
        for (size_t t = 0; t < un; ++t) {
            for (size_t i = 0; i < d; ++i) xb_[t * d + i] += L.out_bias[i];
        }
    }
    return Status::ok();
}

// Kimi Delta Attention (Kimi-Linear), per head h of hd key and value channels:
//   q, k, v = SiLU(causal_conv1d(W_q x)), SiLU(conv(W_k x)), SiLU(conv(W_v x));
//   q, k L2-normalized (kKdaL2Eps), q scaled by 1/sqrt(hd)
//   decay_i = exp(A_h softplus((W_fb W_fa x)_i + dt_bias_i))   per key channel i
//   b = sigmoid(W_beta x)_h
//   S = S diag(decay);  S += b (v - S k) k^T;  o = S q
//   out = W_o (RMSNorm(o) * sigmoid(W_gb W_ga x))
// The reference prefills with a chunked form of the same recurrence (and
// decodes with the step); here every pass runs the step, one task per
// segment and head through all of its tokens, so a token's state and output
// do not depend on how the tokens were batched (prefill, decoding, tree
// verification and several sequences give the same bits) at O(hd^2) per
// token and head, like decoding. Prefill chunks the token scan and partitions
// independent value rows across the active pool; it does not reassociate
// the recurrence as an algebraic parallel scan. The convolutions run first,
// one task per segment and head of q, k or v channels.
Status Transformer::kda_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto hd = static_cast<size_t>(c.kda_head_dim);
    const auto heads = static_cast<size_t>(c.kda_heads);
    const size_t qkv = heads * hd;
    const auto taps = static_cast<size_t>(c.ssm_conv_kernel);
    const size_t history = taps - 1;
    // q, k and v rows one projection after another (each un x qkv), as the
    // convolution's channels are ordered.
    mix_.resize(3 * un * qkv);
    z_.resize(un * qkv);
    alpha_.resize(un * qkv);
    beta_.resize(un * heads);
    dn_.resize(un * qkv);
    low_a_.resize(un * hd);
    low_b_.resize(un * hd);
    {
        const TensorView* proj[] = {&w[kQ], &w[kK], &w[kV], &w[kBeta], &w[kFa], &w[kGa]};
        float* outs[] = {mix_.data(), mix_.data() + un * qkv, mix_.data() + 2 * un * qkv, beta_.data(),
                         low_a_.data(), low_b_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, proj, xb_.data(), outs, n));
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kFb], low_a_.data(), alpha_.data(), n));
    LIYAB_RETURN_IF_ERROR(matmul(route, route.attention, w[kGb], low_b_.data(), z_.data(), n));

    // Checkpoints and tree branches as in delta_net_mixer(): slot layout [conv | ssm].
    auto state_of = [&](const Segment& s) -> RecurrentState& { return (*s.states)[static_cast<size_t>(L.state_slot)]; };
    auto slot_of = [&](const Segment& s, RecurrentState& st, size_t i) -> float* {
        if (!s.checkpoint || i + static_cast<size_t>(s.rollback) < static_cast<size_t>(s.n)) return nullptr;
        return st.checkpoints.data() + static_cast<size_t>((s.n_past + static_cast<int32_t>(i)) % s.rollback) *
                                           (st.conv.size() + st.ssm.size());
    };
    const int32_t* parent = tree_depth_.empty() ? nullptr : tree_parent_.data();
    auto branches = [&](size_t i) { return parent != nullptr && i > 0 && parent[i] != static_cast<int32_t>(i) - 1; };
    const auto segments = static_cast<int64_t>(segs_.size());

    // 1. The causal convolutions and SiLU, then the L2 norms of q and k: one
    //    task per segment and head of q (task < heads), k or v channels.
    const auto conv_tasks = static_cast<int64_t>(3 * heads);
    pool.parallel_for(conv_tasks * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / conv_tasks)];
            RecurrentState& st = state_of(s);
            const auto task = static_cast<size_t>(job % conv_tasks);
            const size_t part = task / heads;          // 0 q, 1 k, 2 v
            const size_t col0 = (task % heads) * hd;   // the head's first channel within its projection
            float* rows = mix_.data() + part * un * qkv;
            const auto first = static_cast<size_t>(s.first);
            const auto count = static_cast<size_t>(s.n);
            for (size_t col = col0; col < col0 + hd; ++col) {
                const size_t ch = part * qkv + col;
                float* hist = st.conv.data() + ch * history;
                const float* tap = L.conv1d.data() + ch * taps;
                for (size_t i = 0; i < count; ++i) {
                    if (branches(i)) {
                        const float* from = slot_of(s, st, static_cast<size_t>(parent[i])) + ch * history;
                        std::copy(from, from + history, hist);
                    }
                    float& x = rows[(first + i) * qkv + col];
                    float acc = x * tap[history];
                    for (size_t j = 0; j < history; ++j) acc += hist[j] * tap[j];
                    for (size_t j = 0; j + 1 < history; ++j) hist[j] = hist[j + 1];
                    hist[history - 1] = x;
                    x = silu(acc);
                    if (float* slot = slot_of(s, st, i)) std::copy(hist, hist + history, slot + ch * history);
                }
            }
            if (part < 2) {
                for (size_t i = 0; i < count; ++i) {
                    l2norm(rows + (first + i) * qkv + col0, static_cast<int32_t>(hd), kKdaL2Eps);
                }
            }
        }
    });

    // 2. Decays and write strengths are shared by every value row of a head.
    // Compute them once before splitting the independent state rows into
    // tasks. Time remains ordered inside each row, including tree branches.
    for (size_t t = 0; t < un; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            float* decay = alpha_.data() + t * qkv + h * hd;
            const float* dt = L.ssm_dt.data() + h * hd;
            for (size_t j = 0; j < hd; ++j) decay[j] = std::exp(L.ssm_a[h] * softplus(decay[j] + dt[j]));
            beta_[t * heads + h] = sigmoid(beta_[t * heads + h]);
        }
    }
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    // Prefill chunks the independent value rows so even a few heads can use
    // the pool, keeping a task's rows hot over a token chunk. The partition
    // follows the active pool size, never a device/model name. Every row
    // runs exactly the decoding operations in token order: chunk boundaries,
    // thread counts and checkpoints cannot change its bits. Decoding needs
    // no temporary state whose size grows with the preceding context.
    const size_t slices = n == 1 ? 1 : std::min(hd, std::max(size_t{1},
        (static_cast<size_t>(pool.active_threads()) + heads - 1) / heads));
    const size_t rows_per_slice = (hd + slices - 1) / slices;
    const auto head_tasks = static_cast<int64_t>(heads * slices);
    constexpr size_t kTokenChunk = 64;
    pool.parallel_for(head_tasks * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / head_tasks)];
            RecurrentState& st = state_of(s);
            const size_t task = static_cast<size_t>(job % head_tasks);
            const size_t h = task / slices;
            const size_t row0 = (task % slices) * rows_per_slice;
            if (row0 >= hd) continue;
            const size_t row_count = std::min(rows_per_slice, hd - row0);
            const auto first = static_cast<size_t>(s.first);
            const size_t state_offset = h * hd * hd + row0 * hd;
            float* S = st.ssm.data() + state_offset;
            for (size_t chunk = 0; chunk < static_cast<size_t>(s.n); chunk += kTokenChunk) {
                const size_t last = std::min(static_cast<size_t>(s.n), chunk + kTokenChunk);
                for (size_t i = chunk; i < last; ++i) {
                    if (branches(i)) {
                        const float* from = slot_of(s, st, static_cast<size_t>(parent[i])) + st.conv.size() + state_offset;
                        std::copy(from, from + row_count * hd, S);
                    }
                    const size_t t = first + i;
                    const float* q = mix_.data() + t * qkv + h * hd;
                    const float* k = mix_.data() + un * qkv + t * qkv + h * hd;
                    const float* v = mix_.data() + 2 * un * qkv + t * qkv + h * hd + row0;
                    const float* decay = alpha_.data() + t * qkv + h * hd;
                    float* o = dn_.data() + t * qkv + h * hd + row0;
                    kda_rule_step(S, q, k, v, decay, beta_[t * heads + h], q_scale, o, hd, row_count);
                    if (float* slot = slot_of(s, st, i)) {
                        std::copy(S, S + row_count * hd, slot + st.conv.size() + state_offset);
                    }
                }
            }
        }
    });
    // A head's norm reduces all value rows, after the row tasks have joined.
    pool.parallel_for(static_cast<int64_t>(un * heads), [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            float* o = dn_.data() + static_cast<size_t>(job) * hd;
            const float* gate = z_.data() + static_cast<size_t>(job) * hd;
            rmsnorm(o, L.ssm_norm.data(), o, static_cast<int32_t>(hd), c.rms_eps);
            for (size_t j = 0; j < hd; ++j) o[j] *= sigmoid(gate[j]);
        }
    });
    return matmul(route, route.attention, w[kSsmOut], dn_.data(), xb_.data(), n);
}


Status Transformer::attach_expert_store(int64_t budget_option, uint64_t memory_budget, size_t reserve,
                                        int32_t requant_bits, bool repack) {
    if (budget_option == 0) return Status::ok();
    if (requant_bits != -1 && requant_bits != 0 && requant_bits != 4 && requant_bits != 5) {
        return Status(ErrorCode::InvalidArgument, "requant_bits must be -1, 0, 4 or 5");
    }
    const ModelConfig& c = config_;
    auto is_expert = [](const TensorView& t) {
        const std::string_view n = t.name;
        return n.size() > 13 && n.substr(n.size() - 12) == "_exps.weight";
    };
    size_t expert_bytes = 0;
    for (const TensorView& t : file_->tensors()) {
        if (is_expert(t)) expert_bytes += t.nbytes;
    }
    const uint64_t available = usable_memory_bytes(memory_budget);
    // Automatic mode streams only when the model does not fit.
    if (budget_option < 0 &&
        (available == 0 || static_cast<double>(file_->file_size()) <= 0.8 * static_cast<double>(available))) {
        return Status::ok();
    }

    // The expert cache gets what the resident weights leave, after room for
    // the KV cache, activations and the rest of the app: 1 GiB of free RAM,
    // or 512 MiB inside an explicit budget (the embedder already left its own
    // headroom below the platform's cap). Recurrent-state snapshots
    // (snapshot_state) come out of the cache too. Only the live context's:
    // contexts made later (make_context(), the engine's other sequences)
    // borrow theirs from the cache while they exist (ExpertStore::lend(),
    // context_state_bytes()), so the plan, and with it the requantization
    // decision, never depends on how many might be made.
    size_t legacy_scratch = 0;
    for (const Layer& l : layers_) {
        if (l.w[kKvB] == nullptr) continue;
        const auto rank = static_cast<size_t>(c.mla.kv_rank);
        const auto heads = static_cast<size_t>(c.n_head);
        legacy_scratch = std::max(legacy_scratch,
            (heads * rank * static_cast<size_t>(c.mla.nope_dim) + rank) * sizeof(float) +
            heads * static_cast<size_t>(c.mla.v_dim) * l.w[kKvB]->row_bytes());
    }
    const size_t margin = (memory_budget > 0 ? size_t{512} << 20 : size_t{1} << 30) +
                          static_cast<size_t>(kStateSnapshots) * recurrent_state_bytes() + legacy_scratch;
    // The budget came without the reserve (TransformerOptions::reserve_bytes),
    // which the loader set aside when it decided to stream; the cache plans
    // with it and hands back only what it can spare.
    const uint64_t whole = memory_budget > 0 ? usable_memory_bytes(memory_budget + reserve) : available;
    auto cache_budget = [&](size_t resident) -> size_t {
        if (budget_option > 0) return static_cast<size_t>(budget_option);
        return whole > resident + margin ? static_cast<size_t>(whole) - resident - margin : 0;
    };

    // Optional lossy conversion of the resident Q8_0 matrices (the UD quants
    // keep attention and DeltaNet projections in Q8_0, ~2/3 of the bytes read
    // per token): fewer bytes per token, and the freed memory goes to the
    // expert cache. The token embedding is skipped: decode reads one row of
    // it per token. Automatic (-1): only when the cache would hold less than
    // kTightCache of the experts, where hit rates fall off and nearly every
    // token waits for storage; Q5_K if that frees enough, else Q4_K.
    auto convertible = [&](const TensorView& t) {
        return !is_expert(t) && t.type == DType::Q8_0 && t.n_dims == 2 && t.rows() > 1 &&
               t.cols() % quant::kSuperBlock == 0 && t.name != "token_embd.weight";
    };
    const size_t resident_q8 = file_->file_size() - expert_bytes;
    int32_t bits = requant_bits;
    if (bits < 0) {
        bits = 0;
        const auto wanted = static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes));
        if (cache_budget(resident_q8) < wanted) {
            size_t q8_bytes = 0;
            for (const TensorView& t : file_->tensors()) {
                if (convertible(t)) q8_bytes += t.nbytes;
            }
            // Q8_0: 8.5 bits per weight; Q5_K 5.5, Q4_K 4.5.
            const auto saving = [&](double bpw) { return static_cast<size_t>(static_cast<double>(q8_bytes) * (1.0 - bpw / 8.5)); };
            bits = q8_bytes == 0 ? 0 : (cache_budget(resident_q8 - saving(5.5)) >= wanted ? 5 : 4);
            LIYAB_LOG_INFO("memory is tight for this model (expert cache %.2f GiB, %.0f%% of experts): %s",
                           static_cast<double>(cache_budget(resident_q8)) / (1024.0 * 1024.0 * 1024.0),
                           100.0 * static_cast<double>(cache_budget(resident_q8)) / static_cast<double>(expert_bytes),
                           bits == 0 ? "no Q8_0 matrices to convert" : bits == 5 ? "converting resident Q8_0 to Q5_K"
                                                                      : "converting resident Q8_0 to Q4_K");
        }
    }
    size_t saved = 0;
    if (bits > 0) {
        auto requantized = file_->requantize(convertible, bits == 4 ? DType::Q4_K : DType::Q5_K);
        if (!requantized) return requantized.status();
        saved = requantized.value();
        LIYAB_LOG_INFO("resident Q8_0 matrices requantized to %s: %.2f GiB saved", bits == 4 ? "Q4_K" : "Q5_K",
                       static_cast<double>(saved) / (1024.0 * 1024.0 * 1024.0));
    }
    const size_t resident_bytes = resident_q8 - saved;
    // The reserve gets only what the cache can spare above kTightCache, so it
    // never decides the conversion above nor starves the cache; with an
    // explicit cache size the rest of the budget is the embedder's anyway.
    // The cache rounds each slot size class down to whole slots: a few
    // slots' worth above the line keeps it above once rounded.
    size_t largest_expert = 0;
    for (int32_t l = 0; l < c.n_layers; ++l) {
        if (c.blocks[static_cast<size_t>(l)].ffn != FfnKind::Moe) continue;
        const Layer& L = layers_[static_cast<size_t>(l)];
        const size_t matrices = (L.w[kGateExps] != nullptr ? L.w[kGateExps]->nbytes : 0) + L.w[kUpExps]->nbytes +
                                L.w[kDownExps]->nbytes;
        largest_expert = std::max(largest_expert, matrices / static_cast<size_t>(c.n_expert));
    }
    const size_t cache_floor =
        static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes)) + 4 * largest_expert;
    size_t budget = cache_budget(resident_bytes);
    const size_t spare = budget > cache_floor ? budget - cache_floor : 0;
    memory_plan_.reserved_bytes = budget_option > 0 ? reserve : std::min(reserve, spare);
    if (budget_option <= 0) budget -= memory_plan_.reserved_bytes;
    memory_plan_.resident_bytes = resident_bytes;
    memory_plan_.expert_bytes = expert_bytes;
    memory_plan_.requant_bits = bits;
    // What the cache needs for most experts to come from RAM (~85% hits at
    // 12-15% of the experts cached on Qwen3.6-35B-A3B; the share, not the size,
    // is what matters across models).
    memory_plan_.recommended_bytes =
        resident_q8 + static_cast<size_t>(kComfortableCache * static_cast<double>(expert_bytes)) + margin;
    std::vector<std::array<const TensorView*, 3>> experts(static_cast<size_t>(c.n_layers));
    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        if (c.blocks[static_cast<size_t>(l)].ffn == FfnKind::Moe) {
            experts[static_cast<size_t>(l)] = {L.w[kGateExps], L.w[kUpExps], L.w[kDownExps]};
        }
    }
    auto store = ExpertStore::create(*file_, std::move(experts), c.n_expert, budget, 4, repack);
    if (!store) return store.status();
    expert_store_ = std::move(store).value();
    memory_plan_.expert_cache_bytes = expert_store_->capacity_bytes();
    if (memory_plan_.expert_cache_bytes < static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes))) {
        LIYAB_LOG_WARN("expert cache holds %.0f%% of the experts: most tokens will wait for storage; about %.1f GiB of "
                       "memory budget would let this model run well",
                       100.0 * static_cast<double>(memory_plan_.expert_cache_bytes) / static_cast<double>(expert_bytes),
                       static_cast<double>(memory_plan_.recommended_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    file_->keep_resident([&](const TensorView& t) { return !is_expert(t) && !file_->converted(t); });
    LIYAB_LOG_INFO("expert streaming: %.2f GiB resident weights, %.2f GiB of experts on storage",
                   static_cast<double>(resident_bytes) / (1024.0 * 1024.0 * 1024.0),
                   static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0));
    return Status::ok();
}

void Transformer::predict_experts(int32_t layer, int32_t n, const Route& route) {
    if (config_.hc_count > 0) return;  // a collapsed stream is not the next block's residual streams
    if (expert_store_ == nullptr || layer >= config_.n_layers) return;
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(layer)];
    if (plan.ffn != FfnKind::Moe) return;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const PhaseTimer timer(phases_.router);
    const auto d = static_cast<size_t>(c.n_embd);
    const auto E = static_cast<size_t>(c.n_expert);
    const auto un = static_cast<size_t>(n);
    pred_in_.resize(un * d);
    pred_logits_.resize(un * E);
    for (size_t t = 0; t < un; ++t) {
        if (plan.norm_before_ffn) norm_row(x_.data() + t * d, opt(L.ffn_norm), opt(L.ffn_norm_b), pred_in_.data() + t * d, c.n_embd);
        else std::copy_n(x_.data() + t * d, d, pred_in_.data() + t * d);
    }
    if (!L.router_scale.empty()) normalize_router_input(L, x_.data(), pred_in_.data(), n);
    if (!route.cpu->matmul(*L.w[kRouter], pred_in_.data(), pred_logits_.data(), n).is_ok()) return;
    if (!L.router_bias.empty()) {
        for (size_t t = 0; t < un; ++t) {
            for (size_t e = 0; e < E; ++e) pred_logits_[t * E + e] += L.router_bias[e];
        }
    }
    std::vector<int32_t> order(E);
    std::vector<int32_t> picks;
    std::vector<int32_t> best_rank(E, -1);
    const bool cut = expert_mass_ < 1.0f || (max_experts_ > 0 && max_experts_ < c.n_expert_used);
    const bool biased_softmax = c.moe_gating == MoeGating::Softmax && !L.expert_bias.empty();
    std::vector<float> probs(cut || biased_softmax ? E : 0);
    // Group-limited selection: the guess keeps to the groups the router will
    // allow, scored on the router's own selection scores.
    const bool grouped = c.n_expert_groups_used < c.n_expert_groups;
    const size_t per_group = E / static_cast<size_t>(c.n_expert_groups);
    std::vector<float> group_sel(grouped ? E : 0);
    std::vector<uint8_t> group_ok;
    for (size_t t = 0; t < un; ++t) {
        const float* logits = pred_logits_.data() + t * E;
        if (expert_observer_ != nullptr && n == 1) expert_observer_->predicted(layer, {logits, E});
        if (biased_softmax) {
            const float mx = *std::max_element(logits, logits + E);
            float sum = 0.0f;
            for (size_t e = 0; e < E; ++e) sum += probs[e] = std::exp(logits[e] - mx);
            for (float& p : probs) p /= sum;
        }
        if (grouped) {
            float mx = -INFINITY, sum = 0.0f;
            if (c.moe_gating == MoeGating::Softmax) {
                for (size_t e = 0; e < E; ++e) mx = std::max(mx, logits[e]);
                for (size_t e = 0; e < E; ++e) sum += group_sel[e] = std::exp(logits[e] - mx);
            }
            for (size_t e = 0; e < E; ++e) {
                const float p = c.moe_gating == MoeGating::Softmax ? group_sel[e] / sum : positive_router_score(c.moe_gating, logits[e]);
                group_sel[e] = p + (L.expert_bias.empty() ? 0.0f : L.expert_bias[e]);
            }
            select_expert_groups(group_sel.data(), E, c.n_expert_groups, c.n_expert_groups_used, !L.expert_bias.empty(),
                          group_ok);
        }
        auto score = [&](int32_t e) {  // selection corrections are in probability space, never logit space
            if (grouped && group_ok[static_cast<size_t>(e) / per_group] == 0) return -INFINITY;
            if (grouped) return group_sel[static_cast<size_t>(e)];
            const float p = biased_softmax ? probs[static_cast<size_t>(e)] :
                            c.moe_gating == MoeGating::Softmax ? logits[e] : positive_router_score(c.moe_gating, logits[e]);
            return p + (L.expert_bias.empty() ? 0.0f : L.expert_bias[static_cast<size_t>(e)]);
        };
        for (size_t e = 0; e < E; ++e) order[e] = static_cast<int32_t>(e);
        const auto k = static_cast<std::ptrdiff_t>(c.n_expert_used);
        if (!L.hash_experts.empty()) {
            if (routing_tokens_.size() != un) return;
            const int32_t token = routing_tokens_[t];
            if (token < 0 || token >= c.n_vocab) return;
            std::copy_n(L.hash_experts.data() + static_cast<size_t>(token) * static_cast<size_t>(k), k, order.begin());
        } else {
            std::partial_sort(order.begin(), order.begin() + k, order.end(),
                              [&](int32_t a, int32_t b) { return score(a) > score(b) || (score(a) == score(b) && a < b); });
        }
        size_t kept = static_cast<size_t>(k);
        if (cut) {  // guess only the experts the router would keep (unnormalized: ratios suffice)
            float sum = 0.0f;
            float top = logits[order[0]];
            if (!L.hash_experts.empty()) {
                for (std::ptrdiff_t i = 1; i < k; ++i) top = std::max(top, logits[order[static_cast<size_t>(i)]]);
            }
            for (std::ptrdiff_t i = 0; i < k; ++i) {
                const auto e = static_cast<size_t>(order[static_cast<size_t>(i)]);
                sum += probs[e] = c.moe_gating == MoeGating::Softmax ? std::exp(logits[e] - top) : positive_router_score(c.moe_gating, logits[e]);
            }
            kept = experts_to_run(probs, order, sum);
        }
        for (size_t r = 0; r < kept; ++r) {
            int32_t& best = best_rank[static_cast<size_t>(order[r])];
            if (best < 0 || static_cast<int32_t>(r) < best) best = static_cast<int32_t>(r);
        }
    }
    if (rank_guessed_.size() != static_cast<size_t>(c.n_expert_used)) {
        rank_guessed_.assign(static_cast<size_t>(c.n_expert_used), 0.0);
        rank_used_.assign(static_cast<size_t>(c.n_expert_used), 0.0);
    }
    guesses_.clear();
    std::vector<int32_t> prefetched;
    for (size_t e = 0; e < E; ++e) {
        const int32_t r = best_rank[e];
        if (r < 0) continue;
        guesses_.push_back({static_cast<int32_t>(e), r});
        picks.push_back(static_cast<int32_t>(e));
        // Laplace-smoothed precision of the rank: every rank starts as worth prefetching (2/3).
        const auto ur = static_cast<size_t>(r);
        if ((rank_used_[ur] + 2.0) / (rank_guessed_[ur] + 3.0) > 0.5) prefetched.push_back(static_cast<int32_t>(e));
    }
    expert_store_->prefetch(layer, prefetched, true);
    predicted_ = std::move(picks);  // sorted: built in expert order
    predicted_layer_ = layer;
}

Status Transformer::dense_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                              FfnMatmulHook* hook) {
    const ModelConfig& c = config_;
    const BlockPlan& plan = c.blocks[static_cast<size_t>(layer)];
    const size_t rows = static_cast<size_t>(n) * static_cast<size_t>(plan.n_ff);
    hb_.resize(rows);
    hb2_.resize(rows);
    // FFN projections: an experimental override first, else the routed backend.
    auto project = [&](FfnProjection p, const TensorView& m, const float* x, float* y) -> Status {
        if (hook != nullptr && hook->ffn_matmul(layer, p, x, y, n)) {
            // The hook replaced the base matmul only: an adapter's update still applies.
            const TensorView* ws[] = {&m};
            float* ys[] = {y};
            if (!adapters_in_pass_ || !collect_adapter_items(ws, x, ys, n)) return Status::ok();
            return adapter_runner_.apply(*route.cpu, adapter_items_, adapter_pool_);
        }
        return matmul(route, route.ffn, m, x, y, n);
    };
    const Layer& L = layers_[static_cast<size_t>(layer)];
    // A bias-free SiLU FFN can keep gate/up activations on the accelerator
    // until the down projection. Other variants need their CPU-side steps,
    // adapter updates or experimental hook between projections.
    if (route.ffn != route.cpu && hook == nullptr && !adapters_in_pass_ &&
        plan.ffn != FfnKind::Ungated && !plan.fused_gate_up && !plan.dense_fused_gate_up &&
        !plan.sparse_gate && plan.shared_swiglu_clamp == 0.0f && c.spec->ffn_activation == FfnActivation::SiLU &&
        L.b_gate.empty() && L.b_up.empty() && L.b_down.empty()) {
        Status fused = route.ffn->gated_ffn_silu(w[kGate], w[kUp], w[kDown], xb_.data(), xb_.data(), n);
        if (fused.is_ok()) return fused;
        if (fused.code() != ErrorCode::Unsupported) return fused;
    }
    if (c.blocks[static_cast<size_t>(layer)].ffn == FfnKind::Ungated) {
        // down(act(up x + b_up)) + b_down
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Up, w[kUp], xb_.data(), hb_.data()));
        add_bias(L.b_up, hb_.data(), static_cast<size_t>(n));
        ungated_activation(c.spec->ffn_activation, c.xielu.empty() ? nullptr : &c.xielu[static_cast<size_t>(layer)],
                           hb_.data(), rows);
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Down, w[kDown], hb_.data(), xb_.data()));
        add_bias(L.b_down, xb_.data(), static_cast<size_t>(n));
        return Status::ok();
    }
    if (plan.parallel_ffn ? plan.dense_fused_gate_up : plan.fused_gate_up) {
        // One [gate | up] projection per token (no per-projection hook: the
        // halves are one matrix); act(gate) * up lands in hb_.
        const auto ff = static_cast<size_t>(plan.n_ff);
        hb2_.resize(2 * rows);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kUp], xb_.data(), hb2_.data(), n));
        for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
            float* gate_up = hb2_.data() + t * 2 * ff;
            for (size_t i = 0; i < L.b_gate.size(); ++i) gate_up[i] += L.b_gate[i];
            for (size_t i = 0; i < L.b_up.size(); ++i) gate_up[ff + i] += L.b_up[i];
            gated_activation(c.spec->ffn_activation, gate_up, gate_up + ff, hb_.data() + t * ff, ff, plan.shared_swiglu_clamp);
        }
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Down, w[kDown], hb_.data(), xb_.data()));
        add_bias(L.b_down, xb_.data(), static_cast<size_t>(n));
        return Status::ok();
    }
    if (hook != nullptr) {
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Gate, w[kGate], xb_.data(), hb_.data()));
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Up, w[kUp], xb_.data(), hb2_.data()));
    } else {
        const TensorView* gate_up[] = {&w[kGate], &w[kUp]};
        float* outs[] = {hb_.data(), hb2_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.ffn, gate_up, xb_.data(), outs, n));
    }
    add_bias(L.b_gate, hb_.data(), static_cast<size_t>(n));
    add_bias(L.b_up, hb2_.data(), static_cast<size_t>(n));
    if (const BlockPlan& plan = c.blocks[static_cast<size_t>(layer)]; plan.sparse_gate) {
        // Gaussian top-k (BlockPlan::sparse_gate): each token's gate keeps
        // what exceeds mean + k * std (population std), shifted down by it.
        const auto ff = static_cast<size_t>(plan.n_ff);
        for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
            float* g = hb_.data() + t * ff;
            double sum = 0.0;
            for (size_t i = 0; i < ff; ++i) sum += g[i];
            const double mean = sum / static_cast<double>(ff);
            double var = 0.0;
            for (size_t i = 0; i < ff; ++i) var += (g[i] - mean) * (g[i] - mean);
            const auto cutoff = static_cast<float>(mean + std::sqrt(var / static_cast<double>(ff)) * plan.gate_sparsity);
            for (size_t i = 0; i < ff; ++i) g[i] = std::max(g[i] - cutoff, 0.0f);
        }
    }
    gated_activation(c.spec->ffn_activation, hb_.data(), hb2_.data(), hb_.data(), rows, plan.shared_swiglu_clamp);
    LIYAB_RETURN_IF_ERROR(project(FfnProjection::Down, w[kDown], hb_.data(), xb_.data()));
    add_bias(L.b_down, xb_.data(), static_cast<size_t>(n));
    return Status::ok();
}

size_t Transformer::experts_to_run(std::span<const float> probs, std::span<const int32_t> order, float top_sum) const {
    const auto K = static_cast<size_t>(config_.n_expert_used);
    const size_t cap = max_experts_ > 0 ? std::min(K, static_cast<size_t>(max_experts_)) : K;
    if (expert_mass_ >= 1.0f) return cap;
    float covered = 0.0f;
    for (size_t k = 0; k < cap; ++k) {
        covered += probs[static_cast<size_t>(order[k])];
        if (covered >= expert_mass_ * top_sum) return k + 1;
    }
    return cap;
}

// Mixture of experts, as llama.cpp's build_moe_ffn: router probabilities
// (softmax or sigmoid of the logits, plus the router's bias when it has one),
// top-k selection (optionally on probability + bias), optional
// renormalization and scale; each selected expert is a gated FFN (its gate
// and up projections separate or one [gate | up] matrix, each projection
// optionally with the expert's own bias).
// Tokens are grouped by expert so a batch reads every expert once. A shared
// expert, optionally scaled by sigmoid(shared_gate . x), is added on top.
// The router can read the residual through its own unweighted RMS norm,
// independently of the weighted norm that prepares the expert projections.
void Transformer::normalize_router_input(const Layer& L, const float* input, float* out, int32_t n) const {
    const auto d = static_cast<size_t>(config_.n_embd);
    const float root = 1.0f / std::sqrt(static_cast<float>(d));
    for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
        const float* row = input + t * d;
        const float scale = rms_scale(row, config_.n_embd, config_.rms_eps);
        for (size_t i = 0; i < d; ++i) out[t * d + i] = ((row[i] * scale) * L.router_scale[i]) * root;
    }
}

Status Transformer::moe_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto d = static_cast<size_t>(c.n_embd);
    const auto E = static_cast<size_t>(c.n_expert);
    const auto K = static_cast<size_t>(c.n_expert_used);
    const auto ff = static_cast<size_t>(c.n_ff_expert);
    const bool fused = c.blocks[static_cast<size_t>(layer)].fused_gate_up;  // experts as [gate | up] and down

    PhaseTimer router_timer(phases_.router);
    // A pass of several tokens of one sequence (a prompt, a speculative step)
    // runs each expert over many of them: the store repacks its reads for the
    // batched kernels and does not count those uses. Decoding steps of several
    // sequences (steps_only_) are decoding: they choose few experts in common,
    // so an expert runs over a token or two, a repack would only delay the
    // read a miss waits for, and their uses are what the cache should keep.
    if (expert_store_ != nullptr) expert_store_->set_batch(n > 1 && !steps_only_);
    router_.resize(un * E);
    const float* router_input = xb_.data();
    if (!L.router_scale.empty()) {
        router_input_.resize(un * d);
        normalize_router_input(L, x_.data(), router_input_.data(), n);
        router_input = router_input_.data();
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kRouter], router_input, router_.data(), n));
    if (!L.router_bias.empty()) {  // a router with a bias (gpt-oss): part of the logits, selection and weights alike
        for (size_t t = 0; t < un; ++t) {
            for (size_t e = 0; e < E; ++e) router_[t * E + e] += L.router_bias[e];
        }
    }

    // Routing: per expert, the tokens that selected it, with the gate weight
    // and the expert's rank in that token's top-k.
    struct Use {
        int32_t token;
        int32_t rank;
        float weight;
    };
    std::vector<std::vector<Use>> assigned(E);
    std::vector<float> probs(E);
    std::vector<int32_t> order(E);
    experts_kept_.assign(un, static_cast<int32_t>(K));
    std::vector<float> token_weight(un, 0.0f);  // per token: the weights of its kept experts (for skip_slow_)
    const bool grouped = c.n_expert_groups_used < c.n_expert_groups;
    const size_t per_group = E / static_cast<size_t>(c.n_expert_groups);
    std::vector<float> group_sel(grouped ? E : 0);
    std::vector<uint8_t> group_ok;
    for (size_t t = 0; t < un; ++t) {
        const float* logits = router_.data() + t * E;
        if (c.moe_gating == MoeGating::Softmax) {
            const float mx = *std::max_element(logits, logits + E);
            float sum = 0.0f;
            for (size_t e = 0; e < E; ++e) sum += probs[e] = std::exp(logits[e] - mx);
            for (float& p : probs) p /= sum;
        } else {
            for (size_t e = 0; e < E; ++e) probs[e] = positive_router_score(c.moe_gating, logits[e]);
        }
        // Group-limited selection: only the best groups' experts compete
        // (select_expert_groups); the others are out of the top-k.
        if (grouped) {
            for (size_t e = 0; e < E; ++e) group_sel[e] = probs[e] + (L.expert_bias.empty() ? 0.0f : L.expert_bias[e]);
            select_expert_groups(group_sel.data(), E, c.n_expert_groups, c.n_expert_groups_used, !L.expert_bias.empty(),
                          group_ok);
        }
        auto selection = [&](int32_t e) {
            if (grouped && group_ok[static_cast<size_t>(e) / per_group] == 0) return -INFINITY;
            return probs[static_cast<size_t>(e)] + (L.expert_bias.empty() ? 0.0f : L.expert_bias[static_cast<size_t>(e)]);
        };
        if (!L.hash_experts.empty()) {
            if (routing_tokens_.size() != un) return Status(ErrorCode::Internal, "hash routing lost original token IDs");
            const int32_t token = routing_tokens_[t];
            if (token < 0 || token >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "hash routing token out of range");
            std::copy_n(L.hash_experts.data() + static_cast<size_t>(token) * K, K, order.begin());
        } else {
            for (size_t e = 0; e < E; ++e) order[e] = static_cast<int32_t>(e);
            std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(K), order.end(),
                              [&](int32_t a, int32_t b) { return selection(a) > selection(b) || (selection(a) == selection(b) && a < b); });
        }
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) sum += probs[static_cast<size_t>(order[k])];
        const size_t kept = experts_to_run(probs, order, sum);
        experts_kept_[t] = static_cast<int32_t>(kept);
        if (kept < K) {
            sum = 0.0f;
            for (size_t k = 0; k < kept; ++k) sum += probs[static_cast<size_t>(order[k])];
        }
        const float norm = c.moe_norm_weights ? 1.0f / (c.moe_norm_additive ? sum + 1e-20f : std::max(sum, 6.103515625e-5f)) : 1.0f;
        for (size_t k = 0; k < kept; ++k) {
            const auto e = static_cast<size_t>(order[k]);
            assigned[e].push_back(
                {static_cast<int32_t>(t), static_cast<int32_t>(k), probs[e] * norm * c.moe_weights_scale * (L.expert_scale.empty() ? 1.0f : L.expert_scale[e])});
            token_weight[t] += assigned[e].back().weight;
        }
    }

    // With expert streaming, queue every chosen expert now (missed
    // predictions jump the queue) and compute cached ones first, so reads
    // overlap the work on the others.
    std::vector<int32_t> chosen;
    for (size_t e = 0; e < E; ++e) {
        if (!assigned[e].empty()) chosen.push_back(static_cast<int32_t>(e));
    }
    if (expert_observer_ != nullptr && n == 1 && expert_store_ != nullptr) expert_observer_->chosen(layer, chosen);
    if (expert_store_ != nullptr) {
        if (predicted_layer_ == layer) {  // both lists are sorted
            predictions_.predicted += predicted_.size();
            for (size_t i = 0, j = 0; i < predicted_.size() && j < chosen.size();) {
                if (predicted_[i] == chosen[j]) ++predictions_.used;
                if (predicted_[i] <= chosen[j]) ++i; else ++j;
            }
            // Per-rank precision, decayed so it follows the conversation.
            for (const Guess& g : guesses_) {
                const auto r = static_cast<size_t>(g.rank);
                rank_guessed_[r] += 1.0;
                if (std::binary_search(chosen.begin(), chosen.end(), g.expert)) rank_used_[r] += 1.0;
                if (rank_guessed_[r] > 2048.0) {
                    rank_guessed_[r] *= 0.5;
                    rank_used_[r] *= 0.5;
                }
            }
            predicted_layer_ = -1;
        }
        expert_store_->prefetch(layer, chosen, false);
        std::stable_partition(chosen.begin(), chosen.end(),
                              [&](int32_t e) { return expert_store_->ready(layer, e); });
    }
    router_timer.stop();
    // The next block's guess goes behind this block's own reads in the queue.
    if (expert_store_ != nullptr) predict_experts(layer + 1, n, route);

    // The shared expert (when the block has one) runs where the routed ones
    // would wait for a read, or after them; its output is added last either
    // way, so the result does not depend on when it ran.
    double shared_in_experts_ms = 0.0;  // shared-expert time inside the experts phase, moved back to its own
    bool shared_done = L.w[kUpShared] == nullptr;
    auto run_shared = [&]() -> Status {
        const auto t0 = std::chrono::steady_clock::now();
        const auto sh = static_cast<size_t>(w[kUpShared].rows());
        hb_.resize(un * sh);
        hb2_.resize(un * sh);
        const TensorView* gate_up[] = {&w[kGateShared], &w[kUpShared]};
        float* outs[] = {hb_.data(), hb2_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.ffn, gate_up, xb_.data(), outs, n));
        gated_activation(c.spec->ffn_activation, hb_.data(), hb2_.data(), hb_.data(), un * sh, c.blocks[static_cast<size_t>(layer)].shared_swiglu_clamp);
        shared_out_.resize(un * d);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kDownShared], hb_.data(), shared_out_.data(), n));
        // The gate scale is applied when the output is added (below), in one
        // multiply-add as before: scaling here first rounded differently.
        shared_scale_.assign(un, 1.0f);
        if (!L.shared_gate.empty()) {
            for (size_t t = 0; t < un; ++t) {
                float dot = 0.0f;
                for (size_t j = 0; j < d; ++j) dot += L.shared_gate[j] * xb_[t * d + j];
                shared_scale_[t] = sigmoid(dot);
            }
        }
        shared_done = true;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        phases_.shared_expert += ms;
        shared_in_experts_ms += ms;
        return Status::ok();
    };

    PhaseTimer experts_timer(phases_.experts);
    moe_out_.assign(un * d, 0.0f);
    std::vector<float> skipped_weight(un, 0.0f);  // per token, the weight of experts skipped (skip_slow_)
    std::vector<uint8_t> rank_skipped(un * K, 0);
    ranked_out_.resize(un * K * d);
    // Experts run in waves: the next expert (waiting for it if needed) plus
    // every following one already in RAM, as one batched pass for gate/up and
    // one for down, so the CPU pays two thread fork/joins per wave instead of
    // two per expert. Expert slices are transient views: they run on the CPU,
    // which reads them in place.
    constexpr size_t kMaxWave = 32;  // bounds the cache slots one wave pins
    std::vector<int32_t> wave;
    std::vector<std::array<TensorView, 3>> views;
    std::vector<size_t> first_row;  // per wave expert: its first row in the wave's scratch
    std::vector<Backend::MatmulItem> items;
    auto release_wave = [&] {
        if (expert_store_ != nullptr) {
            for (const int32_t ex : wave) expert_store_->release(layer, ex);
        }
    };
    // Queues the active adapters' updates of one expert's matmul `item` over
    // its users' rows (in token order), when they adapt the stack `stack`
    // (see set_adapter()): each segment's users with its own adapter (one
    // run for a single sequence).
    auto add_expert_update = [&](const TensorView& stack, int32_t expert, const Backend::MatmulItem& item,
                                 const std::vector<Use>& users) {
        for (size_t u0 = 0; u0 < users.size();) {
            const Segment& s = segs_[segment_of(users[u0].token)];
            size_t u1 = u0 + 1;
            while (u1 < users.size() && users[u1].token < s.first + s.n) ++u1;
            if (s.adapter != nullptr) {
                const auto it = s.adapter->find(stack.name);
                if (it != s.adapter->end() && it->second.target->experts > 0) {
                    const LoraTarget* t = it->second.target;
                    adapter_items_.push_back({t, expert, it->second.scale,
                                              item.x + u0 * static_cast<size_t>(t->a.ne[0]),
                                              item.y + u0 * static_cast<size_t>(t->rows), static_cast<int32_t>(u1 - u0)});
                }
            }
            u0 = u1;
        }
    };
    for (size_t next = 0; next < chosen.size();) {
        wave.clear();
        views.clear();
        first_row.clear();
        size_t rows_total = 0;
        do {
            const int32_t ex = chosen[next++];
            if (expert_store_ != nullptr && skip_slow_ > 0.0f && !expert_store_->ready(layer, ex)) {
                // Not in RAM yet and light for every token using it: skip it
                // rather than wait (lossy; the token's others are renormalized below).
                const auto& users = assigned[static_cast<size_t>(ex)];
                const bool light = std::all_of(users.begin(), users.end(), [&](const Use& u) {
                    return u.weight < skip_slow_ * token_weight[static_cast<size_t>(u.token)];
                });
                if (light) {
                    for (const Use& u : users) {
                        skipped_weight[static_cast<size_t>(u.token)] += u.weight;
                        rank_skipped[static_cast<size_t>(u.token) * K + static_cast<size_t>(u.rank)] = 1;
                    }
                    ++predictions_.skipped;
                    continue;
                }
            }
            if (expert_store_ != nullptr) {
                // About to wait for a read: do the shared expert's work meanwhile.
                if (!shared_done && wave.empty() && !expert_store_->ready(layer, ex)) {
                    if (Status st = run_shared(); !st.is_ok()) return st;
                }
                auto loaded = expert_store_->acquire(layer, ex);
                if (!loaded) {
                    release_wave();
                    return loaded.status();
                }
                views.push_back(*loaded);
            } else {
                views.push_back({fused ? TensorView{} : expert_slice(w[kGateExps], ex), expert_slice(w[kUpExps], ex),
                                 expert_slice(w[kDownExps], ex)});
            }
            wave.push_back(ex);
            first_row.push_back(rows_total);
            rows_total += assigned[static_cast<size_t>(ex)].size();
        } while (next < chosen.size() && wave.size() < kMaxWave &&
                 (expert_store_ == nullptr || expert_store_->ready(layer, chosen[next])));

        if (wave.empty()) continue;  // every expert of this wave was skipped
        // Inputs: an expert every token chose reads xb_ itself (always the
        // case in decode), so all of them share one quantized copy.
        ein_.resize(rows_total * d);
        eh_.resize(rows_total * ff);
        eh2_.resize(rows_total * (fused ? 2 * ff : ff));  // fused: each row's [gate | up]
        eout_.resize(rows_total * d);
        items.clear();
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto& users = assigned[static_cast<size_t>(wave[i])];
            const auto rows = static_cast<int32_t>(users.size());
            const float* in = xb_.data();
            if (users.size() != un) {
                float* gathered = ein_.data() + first_row[i] * d;
                for (size_t u = 0; u < users.size(); ++u) {
                    std::copy_n(xb_.data() + static_cast<size_t>(users[u].token) * d, d, gathered + u * d);
                }
                in = gathered;
            }
            if (fused) {
                items.push_back({&views[i][1], in, eh2_.data() + first_row[i] * 2 * ff, rows});
                continue;
            }
            items.push_back({&views[i][0], in, eh_.data() + first_row[i] * ff, rows});
            items.push_back({&views[i][1], in, eh2_.data() + first_row[i] * ff, rows});
        }
        // Per-expert adapters: expert e's slice for the tokens routed to it,
        // A·x in the wave's own batch.
        adapter_items_.clear();
        if (adapters_in_pass_) {
            for (size_t i = 0; i < wave.size(); ++i) {
                const auto& users = assigned[static_cast<size_t>(wave[i])];
                if (fused) {
                    add_expert_update(w[kUpExps], wave[i], items[i], users);
                } else {
                    add_expert_update(w[kGateExps], wave[i], items[2 * i], users);
                    add_expert_update(w[kUpExps], wave[i], items[2 * i + 1], users);
                }
            }
            adapter_runner_.add_a_matmuls(adapter_items_, items);
        }
        if (Status st = route.cpu->matmul_batch(items); !st.is_ok()) {
            release_wave();
            return st;
        }
        adapter_runner_.add_updates(adapter_items_, adapter_pool_);
        // Per-expert biases of the gate and up projections (gpt-oss), each
        // expert's own over the rows of its users.
        if (!L.gate_exps_bias.empty() || !L.up_exps_bias.empty()) {
            for (size_t i = 0; i < wave.size(); ++i) {
                const size_t users = assigned[static_cast<size_t>(wave[i])].size();
                const size_t at = static_cast<size_t>(wave[i]) * ff;
                for (size_t u = 0; u < users; ++u) {
                    const size_t r = first_row[i] + u;
                    float* gate = fused ? eh2_.data() + r * 2 * ff : eh_.data() + r * ff;
                    float* up = fused ? gate + ff : eh2_.data() + r * ff;
                    for (size_t j = 0; j < ff && !L.gate_exps_bias.empty(); ++j) gate[j] += L.gate_exps_bias[at + j];
                    for (size_t j = 0; j < ff && !L.up_exps_bias.empty(); ++j) up[j] += L.up_exps_bias[at + j];
                }
            }
        }
        if (fused) {  // the same rows as the split matrices give, so the same bits
            for (size_t r = 0; r < rows_total; ++r) {
                const float* gate_up = eh2_.data() + r * 2 * ff;
                gated_activation(c.spec->ffn_activation, gate_up, gate_up + ff, eh_.data() + r * ff, ff, c.blocks[static_cast<size_t>(layer)].expert_swiglu_clamp);
            }
        } else {
            gated_activation(c.spec->ffn_activation, eh_.data(), eh2_.data(), eh_.data(), rows_total * ff, c.blocks[static_cast<size_t>(layer)].expert_swiglu_clamp);
        }
        items.clear();
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto rows = static_cast<int32_t>(assigned[static_cast<size_t>(wave[i])].size());
            items.push_back({&views[i][2], eh_.data() + first_row[i] * ff, eout_.data() + first_row[i] * d, rows});
        }
        adapter_items_.clear();
        if (adapters_in_pass_) {
            for (size_t i = 0; i < wave.size(); ++i) {
                add_expert_update(w[kDownExps], wave[i], items[i], assigned[static_cast<size_t>(wave[i])]);
            }
            adapter_runner_.add_a_matmuls(adapter_items_, items);
        }
        const Status down_status = route.cpu->matmul_batch(items);
        release_wave();  // the outputs are in eout_; the weights are no longer read
        LIYAB_RETURN_IF_ERROR(down_status);
        adapter_runner_.add_updates(adapter_items_, adapter_pool_);
        for (size_t i = 0; i < first_row.size(); ++i) {
            const auto& users = assigned[static_cast<size_t>(wave[i])];
            const float* bias = L.down_exps_bias.empty() ? nullptr
                                                          : L.down_exps_bias.data() + static_cast<size_t>(wave[i]) * d;
            for (size_t u = 0; u < users.size(); ++u) {
                float* dst = ranked_out_.data() + (static_cast<size_t>(users[u].token) * K + users[u].rank) * d;
                float* src = eout_.data() + (first_row[i] + u) * d;
                const float weight = users[u].weight;
                if (bias != nullptr) {  // the down projection's bias, before the router weight (gpt-oss)
                    for (size_t j = 0; j < d; ++j) src[j] += bias[j];
                }
                for (size_t j = 0; j < d; ++j) dst[j] = weight * src[j];
            }
        }
    }
    // Sum each token's experts in router order. Experts run in whatever order
    // their weights arrive in RAM; adding their outputs in that order made
    // results depend on I/O timing (float addition is not associative, and
    // the int8-quantized activations downstream turn last-bit differences into
    // whole rounding steps: run-to-run perplexity varied by ~10%).
    for (size_t t = 0; t < un; ++t) {
        float* dst = moe_out_.data() + t * d;
        for (size_t k = 0; k < static_cast<size_t>(experts_kept_[t]); ++k) {
            if (rank_skipped[t * K + k]) continue;
            const float* src = ranked_out_.data() + (t * K + k) * d;
            for (size_t j = 0; j < d; ++j) dst[j] += src[j];
        }
        // Skipped experts: the others carry the token's whole weight, as the
        // model's own top-k normalization would have given it.
        if (skipped_weight[t] > 0.0f && c.moe_norm_weights && skipped_weight[t] < token_weight[t]) {
            const float scale = token_weight[t] / (token_weight[t] - skipped_weight[t]);
            for (size_t j = 0; j < d; ++j) dst[j] *= scale;
        }
    }

    experts_timer.stop();
    phases_.experts -= shared_in_experts_ms;

    if (!shared_done) LIYAB_RETURN_IF_ERROR(run_shared());
    if (L.w[kUpShared] != nullptr) {
        for (size_t t = 0; t < un; ++t) {
            const float scale = shared_scale_[t];
            for (size_t j = 0; j < d; ++j) moe_out_[t * d + j] += scale * shared_out_[t * d + j];
        }
    }
    std::copy(moe_out_.begin(), moe_out_.end(), xb_.begin());
    return Status::ok();
}

Status Transformer::hyper_collapse(const TensorView& projection, const std::vector<float>& base,
                                   const std::vector<float>& scales, int32_t n, const Route& route) {
    const ModelConfig& c = config_;
    const auto streams = static_cast<size_t>(c.hc_count), d = static_cast<size_t>(c.n_embd);
    const auto un = static_cast<size_t>(n), width = streams * (2 + streams);
    hc_normalized_.resize(un * streams * d);
    hc_coefficients_.resize(un * width);
    hc_post_.resize(un * streams);
    hc_comb_.resize(un * streams * streams);
    for (size_t t = 0; t < un; ++t) {
        const float* input = hc_streams_.data() + t * streams * d;
        const float scale = rms_scale(input, c.hc_count * c.n_embd, c.rms_eps);
        for (size_t i = 0; i < streams * d; ++i) hc_normalized_[(t * streams * d) + i] = input[i] * scale;
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.cpu, projection, hc_normalized_.data(), hc_coefficients_.data(), n));
    for (size_t t = 0; t < un; ++t) {
        const float* coefficients = hc_coefficients_.data() + t * width;
        float* post = hc_post_.data() + t * streams;
        float* comb = hc_comb_.data() + t * streams * streams;
        float* collapsed = x_.data() + t * d;
        const float* input = hc_streams_.data() + t * streams * d;
        std::fill_n(collapsed, d, 0.0f);
        for (size_t h = 0; h < streams; ++h) {
            const float pre = sigmoid(coefficients[h] * scales[0] + base[h]) + c.hc_eps;
            post[h] = 2.0f * sigmoid(coefficients[streams + h] * scales[1] + base[streams + h]);
            for (size_t j = 0; j < d; ++j) collapsed[j] += pre * input[h * d + j];
            float maximum = -INFINITY;
            for (size_t k = 0; k < streams; ++k) {
                const size_t at = 2 * streams + h * streams + k;
                comb[h * streams + k] = coefficients[at] * scales[2] + base[at];
                maximum = std::max(maximum, comb[h * streams + k]);
            }
            float sum = 0.0f;
            for (size_t k = 0; k < streams; ++k) {
                float& value = comb[h * streams + k];
                value = std::exp(value - maximum);
                sum += value;
            }
            for (size_t k = 0; k < streams; ++k) comb[h * streams + k] = comb[h * streams + k] / sum + c.hc_eps;
        }
        // Softmax already normalized rows. Normalize columns first, then
        // alternate rows/columns, always with the reference's epsilon.
        for (int32_t iteration = 0; iteration < c.hc_iterations; ++iteration) {
            if (iteration > 0) {
                for (size_t h = 0; h < streams; ++h) {
                    float sum = c.hc_eps;
                    for (size_t k = 0; k < streams; ++k) sum += comb[h * streams + k];
                    for (size_t k = 0; k < streams; ++k) comb[h * streams + k] /= sum;
                }
            }
            for (size_t k = 0; k < streams; ++k) {
                float sum = c.hc_eps;
                for (size_t h = 0; h < streams; ++h) sum += comb[h * streams + k];
                for (size_t h = 0; h < streams; ++h) comb[h * streams + k] /= sum;
            }
        }
    }
    return Status::ok();
}

void Transformer::hyper_add(int32_t n) {
    const size_t streams = static_cast<size_t>(config_.hc_count), d = static_cast<size_t>(config_.n_embd);
    hc_next_.resize(hc_streams_.size());
    for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
        for (size_t h = 0; h < streams; ++h) {
            float* out = hc_next_.data() + (t * streams + h) * d;
            const float post = hc_post_[t * streams + h];
            for (size_t j = 0; j < d; ++j) {
                float residual = 0.0f;
                // comb is consumed transposed: input stream k -> output h.
                for (size_t k = 0; k < streams; ++k) {
                    residual += hc_comb_[(t * streams + k) * streams + h] * hc_streams_[(t * streams + k) * d + j];
                }
                out[j] = post * xb_[t * d + j] + residual;
            }
        }
    }
    hc_streams_.swap(hc_next_);
}

Status Transformer::hyper_head(int32_t n, const Route& route) {
    const auto streams = static_cast<size_t>(config_.hc_count), d = static_cast<size_t>(config_.n_embd);
    const auto un = static_cast<size_t>(n);
    hc_normalized_.resize(un * streams * d);
    hc_coefficients_.resize(un * streams);
    for (size_t t = 0; t < un; ++t) {
        const float* input = hc_streams_.data() + t * streams * d;
        const float scale = rms_scale(input, config_.hc_count * config_.n_embd, config_.rms_eps);
        for (size_t i = 0; i < streams * d; ++i) hc_normalized_[t * streams * d + i] = input[i] * scale;
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.cpu, *hc_head_, hc_normalized_.data(), hc_coefficients_.data(), n));
    for (size_t t = 0; t < un; ++t) {
        float* out = x_.data() + t * d;
        std::fill_n(out, d, 0.0f);
        for (size_t h = 0; h < streams; ++h) {
            const float pre = sigmoid(hc_coefficients_[t * streams + h] * hc_head_scale_[0] + hc_head_base_[h]) + config_.hc_eps;
            for (size_t j = 0; j < d; ++j) out[j] += pre * hc_streams_[(t * streams + h) * d + j];
        }
    }
    return Status::ok();
}

void Transformer::size_batch(int32_t n) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(c.head_dim);
    const size_t kv_dim = static_cast<size_t>(c.n_head_kv) * static_cast<size_t>(c.head_dim);
    const auto un = static_cast<size_t>(n);
    x_.resize(un * d);
    xb_.resize(un * std::max(d, q_dim));
    q_.resize(un * q_dim);
    k_.resize(un * kv_dim);
    v_.resize(un * kv_dim);
    att_.resize(un * std::max(q_dim, static_cast<size_t>(c.n_head) * static_cast<size_t>(c.mla.v_dim)));
}

Status Transformer::embed(std::span<const int32_t> tokens, std::span<float> x) const {
    const ModelConfig& c = config_;
    const auto d = static_cast<size_t>(c.n_embd);
    if (x.size() < tokens.size() * d) return Status(ErrorCode::InvalidArgument, "embed: output buffer too small");
    for (size_t t = 0; t < tokens.size(); ++t) {
        if (tokens[t] < 0 || tokens[t] >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
        quant::dequantize_row(token_embd_->type, token_embd_->row(tokens[t]), x.data() + t * d, c.n_embd);
    }
    if (c.embd_scale != 1.0f) {
        for (size_t i = 0; i < tokens.size() * d; ++i) x[i] *= c.embd_scale;
    }
    if (!embd_norm_.empty() || !embd_norm_b_.empty()) {  // BLOOM's word_embeddings_layernorm
        for (size_t t = 0; t < tokens.size(); ++t) {
            norm_row(x.data() + t * d, opt(embd_norm_), opt(embd_norm_b_), x.data() + t * d, c.n_embd);
        }
    }
    return Status::ok();
}

Status Transformer::run_block(int32_t l, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool,
                              const HeadMaskHook* head_mask, FfnSkipHook* ffn_skip, FfnMatmulHook* ffn_hook) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const auto un = static_cast<size_t>(n);
    const Layer& L = layers_[static_cast<size_t>(l)];
    const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
    const bool moe = plan.ffn == FfnKind::Moe;
    const Route cpu_route{route.cpu, route.cpu, route.cpu};
    const Route& block_route = w.streamed() ? cpu_route : route;
    // The residual add: x += xb, the branch scaled first by the μP residual
    // multiplier (ModelConfig::residual_scale) when it is not 1.
    auto add_residual = [&](size_t count) {
        if (c.residual_scale != 1.0f) {
            for (size_t i = 0; i < count; ++i) x_[i] += c.residual_scale * xb_[i];
            return;
        }
        for (size_t i = 0; i < count; ++i) x_[i] += xb_[i];
    };
    const bool hyper = c.hc_count > 0;
    if (hyper) {
        if (l == 0) {
            hc_streams_.resize(un * static_cast<size_t>(c.hc_count) * d);
            for (size_t t = 0; t < un; ++t) {
                for (int32_t h = 0; h < c.hc_count; ++h) {
                    std::copy_n(x_.data() + t * d, d, hc_streams_.data() + (t * static_cast<size_t>(c.hc_count) + static_cast<size_t>(h)) * d);
                }
            }
        }
        LIYAB_RETURN_IF_ERROR(hyper_collapse(w[kHcAttn], L.hc_attn_base, L.hc_attn_scale, n, block_route));
    }
    // Norm after the mixer or the FFN (ArchSpec::post_mixer_norm / post_ffn_norm), in place on xb_.
    auto post_norm = [&](const std::vector<float>& weight) {
        if (weight.empty()) return;
        for (size_t t = 0; t < un; ++t) {
            float* row = xb_.data() + t * d;
            norm_row(row, weight.data(), nullptr, row, c.n_embd);
        }
    };

    // --- expert prefetch for the first block (the others are predicted
    // one block ahead, see predict_experts) ---
    if (expert_store_ != nullptr && l == 0) predict_experts(0, n, route);

    // --- AltUp: the block runs on the active stream's prediction ---
    const bool altup = c.altup_streams > 1;
    if (altup) altup_predict(L, n);

    // --- token mixer (attention or a recurrent mixer) ---
    if (ffn_skip != nullptr) x_block_in_.assign(x_.begin(), x_.end());
    for (size_t t = 0; t < un; ++t) {
        if (plan.norm_before_mixer) norm_row(x_.data() + t * d, opt(L.attn_norm), opt(L.attn_norm_b), xb_.data() + t * d, c.n_embd);
        else std::copy_n(x_.data() + t * d, d, xb_.data() + t * d);
    }
    if (plan.laurel) {
        // Laurel's branch of the normalized input, before the mixer
        // overwrites it: x_n + norm(laurel_r · laurel_l · x_n).
        const auto rank = static_cast<size_t>(w[kLaurelL].rows());
        laurel_mid_.resize(un * rank);
        laurel_.resize(un * d);
        LIYAB_RETURN_IF_ERROR(matmul(block_route, block_route.attention, w[kLaurelL], xb_.data(), laurel_mid_.data(), n));
        LIYAB_RETURN_IF_ERROR(matmul(block_route, block_route.attention, w[kLaurelR], laurel_mid_.data(), laurel_.data(), n));
        for (size_t t = 0; t < un; ++t) {
            float* row = laurel_.data() + t * d;
            rmsnorm(row, L.laurel_norm.data(), row, c.n_embd, c.rms_eps);
            const float* in = xb_.data() + t * d;
            for (size_t i = 0; i < d; ++i) row[i] = in[i] + row[i];
        }
    }
    // A parallel block's FFN reads the block input too: normed by its own
    // FFN norm, or the mixer's normed input itself; kept while the mixer runs.
    if (plan.parallel) {
        par_in_.resize(un * d);
        for (size_t t = 0; t < un; ++t) {
            if (L.ffn_norm.empty() && L.ffn_norm_b.empty()) {
                std::copy_n(xb_.data() + t * d, d, par_in_.data() + t * d);
            } else {
                norm_row(x_.data() + t * d, opt(L.ffn_norm), opt(L.ffn_norm_b), par_in_.data() + t * d, c.n_embd);
            }
        }
    }
    if (plan.mixer == MixerKind::None) {
        // No mixer branch: leave the residual stream exactly as it entered.
    } else if (plan.mixer == MixerKind::Linear) {
        att_.resize(un * std::max(d, static_cast<size_t>(c.n_head) * static_cast<size_t>(c.head_dim)));
        LIYAB_RETURN_IF_ERROR(matmul(block_route, block_route.attention, w[kO], xb_.data(), att_.data(), n));
        std::copy_n(att_.begin(), un * d, xb_.begin());
    } else if (plan.mixer == MixerKind::DeltaNet) {
        const PhaseTimer timer(phases_.delta_net);
        LIYAB_RETURN_IF_ERROR(delta_net_mixer(l, w, n, block_route, pool));
    } else if (plan.mixer == MixerKind::Mamba) {
        const PhaseTimer timer(phases_.delta_net);
        LIYAB_RETURN_IF_ERROR(mamba_mixer(l, w, n, block_route, pool));
    } else if (plan.mixer == MixerKind::Mamba2) {
        const PhaseTimer timer(phases_.delta_net);
        LIYAB_RETURN_IF_ERROR(mamba2_mixer(l, w, n, block_route, pool));
    } else if (plan.mixer == MixerKind::ShortConv) {
        const PhaseTimer timer(phases_.delta_net);  // the recurrent mixers' phase
        LIYAB_RETURN_IF_ERROR(short_conv_mixer(l, w, n, block_route, pool));
    } else if (plan.mixer == MixerKind::Kda) {
        const PhaseTimer timer(phases_.delta_net);
        LIYAB_RETURN_IF_ERROR(kda_mixer(l, w, n, block_route, pool));
    } else if (c.spec->shared_kv) {
        const PhaseTimer timer(phases_.attention);
        LIYAB_RETURN_IF_ERROR(shared_kv_mixer(l, w, n, block_route, pool, head_mask != nullptr ? head_mask->mask(l) : nullptr));
    } else if (plan.latent) {
        const PhaseTimer timer(phases_.attention);
        LIYAB_RETURN_IF_ERROR(latent_attention_mixer(l, w, n, block_route, pool,
                                                     head_mask != nullptr ? head_mask->mask(l) : nullptr));
    } else {
        const PhaseTimer timer(phases_.attention);
        LIYAB_RETURN_IF_ERROR(
            attention_mixer(l, w, n, block_route, pool, head_mask != nullptr ? head_mask->mask(l) : nullptr));
    }
    if (plan.mixer != MixerKind::None) {
        post_norm(L.post_mixer_norm);
        if (hyper) {
            hyper_add(n);
            LIYAB_RETURN_IF_ERROR(hyper_collapse(w[kHcFfn], L.hc_ffn_base, L.hc_ffn_scale, n, block_route));
        } else if (plan.laurel) {
            const float sqrt2 = std::sqrt(2.0f);
            for (size_t i = 0; i < un * d; ++i) x_[i] = ((x_[i] + xb_[i] * c.residual_scale) + laurel_[i]) / sqrt2;
        } else {
            add_residual(un * d);
        }
    }
    // A MoE block predicts the next one once its own experts are queued (moe_ffn).
    if (expert_store_ != nullptr && !moe) predict_experts(l + 1, n, route);
    if (plan.ffn == FfnKind::None) return Status::ok();

    // --- FFN block: dense SwiGLU or mixture of experts (skippable by an experimental hook) ---
    if (ffn_skip != nullptr && ffn_skip->skip_ffn(l, c.n_layers, x_block_in_, x_)) {
        ++last_ffn_skips_;
        if (expert_store_ != nullptr && moe) predict_experts(l + 1, n, route);
        return hyper && l + 1 == c.n_layers ? hyper_head(n, block_route) : Status::ok();
    }
    if (plan.parallel) {
        std::copy_n(par_in_.data(), un * d, xb_.data());
    } else {
        for (size_t t = 0; t < un; ++t) {
            if (plan.norm_before_ffn) norm_row(x_.data() + t * d, opt(L.ffn_norm), opt(L.ffn_norm_b), xb_.data() + t * d, c.n_embd);
            else std::copy_n(x_.data() + t * d, d, xb_.data() + t * d);
        }
    }
    if (plan.parallel_ffn) {
        {
            const PhaseTimer timer(phases_.dense_ffn);
            LIYAB_RETURN_IF_ERROR(dense_ffn(l, w, n, block_route, ffn_hook));
        }
        post_norm(L.dense_post_norm);
        parallel_ffn_out_.assign(xb_.begin(), xb_.begin() + static_cast<std::ptrdiff_t>(un * d));
        for (size_t t = 0; t < un; ++t) {
            norm_row(x_.data() + t * d, L.expert_norm.data(), nullptr, xb_.data() + t * d, c.n_embd);
        }
        LIYAB_RETURN_IF_ERROR(moe_ffn(l, w, n, block_route));
        post_norm(L.expert_post_norm);
        for (size_t i = 0; i < un * d; ++i) xb_[i] = parallel_ffn_out_[i] + xb_[i];
    } else if (moe) {
        LIYAB_RETURN_IF_ERROR(moe_ffn(l, w, n, block_route));
    } else {
        const PhaseTimer timer(phases_.dense_ffn);
        LIYAB_RETURN_IF_ERROR(dense_ffn(l, w, n, block_route, ffn_hook));
    }
    post_norm(L.post_ffn_norm);
    if (hyper) {
        hyper_add(n);
        if (l + 1 == c.n_layers) return hyper_head(n, block_route);
    } else {
        add_residual(un * d);
    }

    // --- AltUp's correction, per-layer embeddings, the block's output scale ---
    if (altup) {
        altup_correct(L, n);
        if (c.ple_dim > 0) {
            // Gated by the corrected active stream, times its scale when the block has one.
            const auto K = static_cast<size_t>(c.altup_streams);
            const auto a = static_cast<size_t>(c.altup_active);
            altup_tmp_.resize(un * d);
            for (size_t t = 0; t < un; ++t) {
                const float* src = streams_.data() + (t * K + a) * d;
                float* dst = altup_tmp_.data() + t * d;
                if (L.altup_correct_scale.empty()) {
                    std::copy_n(src, d, dst);
                } else {
                    for (size_t i = 0; i < d; ++i) dst[i] = src[i] * L.altup_correct_scale[i];
                }
            }
            LIYAB_RETURN_IF_ERROR(per_layer_input(l, w, altup_tmp_.data(), n, block_route));
        }
        // x_ follows the active stream (early-exit probes and hooks read it).
        const auto K = static_cast<size_t>(c.altup_streams);
        for (size_t t = 0; t < un; ++t) {
            std::copy_n(streams_.data() + (t * K + static_cast<size_t>(c.altup_active)) * d, d, x_.data() + t * d);
        }
    } else if (c.ple_dim > 0) {
        LIYAB_RETURN_IF_ERROR(per_layer_input(l, w, x_.data(), n, block_route));
    }
    if (L.output_scale != 1.0f) {
        for (size_t i = 0; i < un * d; ++i) x_[i] *= L.output_scale;
    }
    return Status::ok();
}

Status Transformer::per_layer_input(int32_t layer, const BlockWeights& w, const float* gate_in, int32_t n,
                                    const Route& route) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto d = static_cast<size_t>(c.n_embd);
    const auto un = static_cast<size_t>(n);
    const auto pd = static_cast<size_t>(c.ple_dim);
    const size_t per_token = static_cast<size_t>(c.n_layers) * pd;
    ple_hidden_.resize(un * pd);
    LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kPleGate], gate_in, ple_hidden_.data(), n));
    // act(gate) * the token's per-layer embedding of this block (the
    // activation of the FFN's gate: GELU on Gemma).
    for (size_t t = 0; t < un; ++t) {
        float* g = ple_hidden_.data() + t * pd;
        const float* e = ple_.data() + t * per_token + static_cast<size_t>(layer) * pd;
        for (size_t i = 0; i < pd; ++i) {
            g[i] = (c.spec->ffn_activation == FfnActivation::GeluTanh ? gelu_tanh(g[i]) : silu(g[i])) * e[i];
        }
    }
    LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kPleProj], ple_hidden_.data(), xb_.data(), n));
    for (size_t t = 0; t < un; ++t) {
        float* row = xb_.data() + t * d;
        rmsnorm(row, L.ple_norm.data(), row, c.n_embd, c.rms_eps);
    }
    if (c.altup_streams > 1) {  // every stream but the first
        const auto K = static_cast<size_t>(c.altup_streams);
        for (size_t t = 0; t < un; ++t) {
            const float* o = xb_.data() + t * d;
            for (size_t k = 1; k < K; ++k) {
                float* st = streams_.data() + (t * K + k) * d;
                for (size_t i = 0; i < d; ++i) st[i] += o[i];
            }
        }
        return Status::ok();
    }
    for (size_t i = 0; i < un * d; ++i) x_[i] += xb_[i];
    return Status::ok();
}

namespace {

// AltUp's router over one stream: tanh(router · (norm(x) / n_embd)), K
// values (the reference scales the normalized input by 1 / n_embd).
void altup_modalities(const float* x, const float* norm, const float* router, int32_t d, int32_t k, float eps,
                      float* normed, float* out) {
    rmsnorm(x, norm, normed, d, eps);
    const float scale = 1.0f / static_cast<float>(d);
    for (int32_t i = 0; i < d; ++i) normed[i] *= scale;
    for (int32_t j = 0; j < k; ++j) {
        const float* r = router + static_cast<size_t>(j) * static_cast<size_t>(d);
        double acc = 0.0;
        for (int32_t i = 0; i < d; ++i) acc += static_cast<double>(r[i]) * normed[i];
        out[j] = std::tanh(static_cast<float>(acc));
    }
}

}  // namespace

void Transformer::altup_predict(const Layer& L, int32_t n) {
    const ModelConfig& c = config_;
    const auto d = static_cast<size_t>(c.n_embd);
    const auto K = static_cast<size_t>(c.altup_streams);
    const auto a = static_cast<size_t>(c.altup_active);
    altup_tmp_.resize(std::max(d, K * d) + K * K + K);
    for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
        float* st = streams_.data() + t * K * d;
        float* pred = altup_tmp_.data();  // K * d predictions
        float* m = pred + K * d;          // K modalities
        float* coef = m + K;              // K * K coefficients
        altup_modalities(st + a * d, L.altup_router_norm.data(), L.altup_router.data(), c.n_embd, c.altup_streams,
                         c.rms_eps, pred, m);
        for (size_t r = 0; r < K * K; ++r) {
            float acc = 0.0f;
            for (size_t j = 0; j < K; ++j) acc += L.altup_predict[r * K + j] * m[j];
            coef[r] = acc;
        }
        // Stream i's prediction: itself plus sum_j coef[i][j] * stream j.
        for (size_t i = 0; i < K; ++i) {
            float* p = pred + i * d;
            for (size_t e = 0; e < d; ++e) {
                float acc = 0.0f;
                for (size_t j = 0; j < K; ++j) acc += st[j * d + e] * coef[i * K + j];
                p[e] = acc + st[i * d + e];
            }
        }
        std::copy_n(pred, K * d, st);
        std::copy_n(st + a * d, d, x_.data() + t * d);
    }
}

void Transformer::altup_correct(const Layer& L, int32_t n) {
    const ModelConfig& c = config_;
    const auto d = static_cast<size_t>(c.n_embd);
    const auto K = static_cast<size_t>(c.altup_streams);
    const auto a = static_cast<size_t>(c.altup_active);
    altup_tmp_.resize(2 * d + 2 * K);
    for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
        float* st = streams_.data() + t * K * d;  // the predictions
        const float* x = x_.data() + t * d;       // the block's output
        float* innovation = altup_tmp_.data();
        float* m = innovation + 2 * d;
        float* coef = m + K;
        altup_modalities(x, L.altup_router_norm.data(), L.altup_router.data(), c.n_embd, c.altup_streams, c.rms_eps,
                         innovation + d, m);
        for (size_t i = 0; i < K; ++i) {
            float acc = 0.0f;
            for (size_t j = 0; j < K; ++j) acc += L.altup_correct[i * K + j] * m[j];
            coef[i] = acc + 1.0f;
        }
        for (size_t e = 0; e < d; ++e) innovation[e] = x[e] - st[a * d + e];
        for (size_t i = 0; i < K; ++i) {
            float* p = st + i * d;
            for (size_t e = 0; e < d; ++e) p[e] = innovation[e] * coef[i] + p[e];
        }
    }
}

namespace {

// AltUp's rescaling of a derived stream to the first one's magnitude:
// s * target / sqrt(max(mean(s^2), 1e-5)), as the reference computes it.
void match_magnitude(float* s, float target, int32_t d) {
    double ss = 0.0;
    for (int32_t i = 0; i < d; ++i) ss += static_cast<double>(s[i]) * s[i];
    const float magnitude = std::sqrt(std::max(static_cast<float>(ss / d), 1e-5f));
    for (int32_t i = 0; i < d; ++i) s[i] = s[i] * target / magnitude;
}

float rms(const float* x, int32_t d) {
    double ss = 0.0;
    for (int32_t i = 0; i < d; ++i) ss += static_cast<double>(x[i]) * x[i];
    return std::sqrt(static_cast<float>(ss / d));
}

}  // namespace

Status Transformer::enter_blocks(std::span<const int32_t> tokens, const Route& route) {
    routing_tokens_.assign(tokens.begin(), tokens.end());
    const ModelConfig& c = config_;
    const auto d = static_cast<size_t>(c.n_embd);
    const auto n = static_cast<int32_t>(tokens.size());
    const auto un = static_cast<size_t>(n);
    if (c.ple_dim > 0) {
        // Per token and block: (norm(proj · x / sqrt(d)) + its row of the
        // table * sqrt(ple_dim)) / sqrt(2), x the scaled embedding.
        const auto pd = static_cast<size_t>(c.ple_dim);
        const size_t per_token = static_cast<size_t>(c.n_layers) * pd;
        ple_.resize(un * per_token);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, *ple_model_proj_, x_.data(), ple_.data(), n));
        const auto proj_scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(c.n_embd)));
        const auto embd_scale = static_cast<float>(std::sqrt(static_cast<double>(c.ple_dim)));
        const auto input_scale = static_cast<float>(1.0 / std::sqrt(2.0));
        std::vector<float> row(per_token);
        for (size_t t = 0; t < un; ++t) {
            float* p = ple_.data() + t * per_token;
            for (size_t i = 0; i < per_token; ++i) p[i] *= proj_scale;
            for (size_t l = 0; l < static_cast<size_t>(c.n_layers); ++l) {
                rmsnorm(p + l * pd, ple_proj_norm_.data(), p + l * pd, c.ple_dim, c.rms_eps);
            }
            // Tokens past the table (a converter pads it to the vocabulary) have zeros.
            if (tokens[t] < ple_embd_->rows()) {
                quant::dequantize_row(ple_embd_->type, ple_embd_->row(tokens[t]), row.data(),
                                      static_cast<int64_t>(per_token));
            } else {
                std::fill(row.begin(), row.end(), 0.0f);
            }
            for (size_t i = 0; i < per_token; ++i) p[i] = (p[i] + row[i] * embd_scale) * input_scale;
        }
    }
    if (c.altup_streams > 1) {
        // Stream 0: the embedding; stream i: altup_proj[i - 1] · x at x's magnitude.
        const auto K = static_cast<size_t>(c.altup_streams);
        streams_.resize(un * K * d);
        altup_tmp_.resize(un * d);
        for (size_t t = 0; t < un; ++t) std::copy_n(x_.data() + t * d, d, streams_.data() + t * K * d);
        for (size_t i = 1; i < K; ++i) {
            LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, altup_proj_[i - 1], x_.data(), altup_tmp_.data(), n));
            for (size_t t = 0; t < un; ++t) {
                float* s = streams_.data() + (t * K + i) * d;
                std::copy_n(altup_tmp_.data() + t * d, d, s);
                match_magnitude(s, rms(x_.data() + t * d, c.n_embd), c.n_embd);
            }
        }
    }
    return Status::ok();
}

Status Transformer::leave_blocks(int32_t n, const Route& route) {
    const ModelConfig& c = config_;
    if (c.altup_streams <= 1) return Status::ok();
    // Streams i > 0 through altup_unembd_proj[i - 1], at stream 0's magnitude;
    // the final hidden state is the mean of the K streams.
    const auto d = static_cast<size_t>(c.n_embd);
    const auto K = static_cast<size_t>(c.altup_streams);
    const auto un = static_cast<size_t>(n);
    altup_tmp_.resize(un * d);
    laurel_.resize(un * d);  // the stream's rows, gathered for the matmul
    for (size_t i = 1; i < K; ++i) {
        for (size_t t = 0; t < un; ++t) std::copy_n(streams_.data() + (t * K + i) * d, d, laurel_.data() + t * d);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, altup_unembd_[i - 1], laurel_.data(), altup_tmp_.data(), n));
        for (size_t t = 0; t < un; ++t) {
            float* s = streams_.data() + (t * K + i) * d;
            std::copy_n(altup_tmp_.data() + t * d, d, s);
            match_magnitude(s, rms(streams_.data() + t * K * d, c.n_embd), c.n_embd);
        }
    }
    for (size_t t = 0; t < un; ++t) {
        const float* st = streams_.data() + t * K * d;
        float* x = x_.data() + t * d;
        for (size_t e = 0; e < d; ++e) {
            float sum = st[e];
            for (size_t i = 1; i < K; ++i) sum += st[i * d + e];
            x[e] = sum / static_cast<float>(K);
        }
    }
    return Status::ok();
}

Status Transformer::run_block_at(int32_t layer, std::span<const int32_t> positions, std::span<float> x,
                                 int32_t chunk, const Route& route, ThreadPool& pool) {
    LIYAB_RETURN_IF_ERROR(reuse_support());
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    if (layer < 0 || layer >= c.n_layers || chunk < 1 || x.size() != positions.size() * d) {
        return Status(ErrorCode::InvalidArgument, "run_block_at: the block, the pass size or the rows do not fit");
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        if (positions[i] < 0 || positions[i] >= n_past_ || (i > 0 && positions[i] <= positions[i - 1])) {
            return Status(ErrorCode::InvalidArgument, "run_block_at: positions must ascend inside the context");
        }
    }
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    if (positions.empty()) return Status::ok();
    // Acquired once for every pass: a layer source streams blocks in order.
    const BlockWeights w(*this, layer);
    LIYAB_RETURN_IF_ERROR(w.status());
    predicted_layer_ = -1;  // a guess made for another batch is no use to these tokens
    // Scattered positions attend in the absorbed form: the expanded form's tiles
    // assume a contiguous pass (latent_visible), and forward() may have left it on.
    latent_expanded_ = false;
    Status status;
    for (size_t i = 0; i < positions.size() && status.is_ok(); i += static_cast<size_t>(chunk)) {
        const size_t m = std::min(static_cast<size_t>(chunk), positions.size() - i);
        size_batch(static_cast<int32_t>(m));
        std::copy_n(x.data() + i * d, m * d, x_.data());
        positions_ = positions.subspan(i, m);
        begin_live_pass(static_cast<int32_t>(m), Logits::None);
        status = run_block(layer, w, static_cast<int32_t>(m), route, pool, nullptr, nullptr, nullptr);
        positions_ = {};
        if (status.is_ok()) std::copy_n(x_.data(), m * d, x.data() + i * d);
    }
    return status;
}

Status Transformer::reuse_support() const {
    if (config_.spec->shared_kv || config_.hc_count > 0) {
        return Status(ErrorCode::Unsupported, "segment relocation does not preserve compressed windows or residual streams");
    }
    if (config_.index_heads > 0) {
        return Status(ErrorCode::Unsupported, "segment reuse with position-dependent sparse index keys is not supported");
    }
    const ModelConfig& c = config_;
    if (c.hybrid()) {
        return Status(ErrorCode::Unsupported,
                      "hybrid model: its recurrent (DeltaNet) blocks must run over the whole prompt in order, so "
                      "every token would still go through every block; reusing the attention blocks' keys and "
                      "values would only save their key and value projections");
    }
    if (kv_->window() > 0) {
        return Status(ErrorCode::Unsupported, "a sliding-window cache releases the positions it no longer shows");
    }
    if (position_embd_ != nullptr) {
        return Status(ErrorCode::Unsupported,
                      "learned absolute positions enter the residual stream, so every block's keys carry the "
                      "position they were computed at, which no rotation can move");
    }
    if (c.altup_streams > 1 || c.ple_dim > 0 || !c.causal) {
        return Status(ErrorCode::Unsupported,
                      "AltUp streams, per-layer embeddings and bidirectional attention make a block's input more than "
                      "the residual row a recomputation would start from");
    }
    if (kv_->has_windowed_layers()) {
        return Status(ErrorCode::Unsupported,
                      "windowed attention blocks keep only their window's rows, so a segment's rows of those blocks "
                      "are gone once the context moves past them (and recomputing them alone would need every "
                      "block before them over the whole segment)");
    }
    if (c.spec->qk_norm == QkNormOrder::AfterRope) {
        for (const Layer& L : layers_) {
            for (size_t i = 0; i < L.k_norm.size() && i < static_cast<size_t>(c.rope_dim); ++i) {
                if (L.k_norm[i] == 0.0f) {
                    return Status(ErrorCode::Unsupported,
                                  "a key norm after RoPE zeroes a rotated channel, so stored keys cannot be turned");
                }
            }
        }
    }
    return Status::ok();
}

Status Transformer::swap_context(std::unique_ptr<KvCache>& kv, int32_t& n_past) {
    if (config_.hybrid() || !states_.empty()) {
        return Status(ErrorCode::Unsupported, "swap_context: recurrent states are part of the context");
    }
    if (kv_->has_windowed_layers()) {
        return Status(ErrorCode::Unsupported, "swap_context: the windowed blocks' snapshots are part of the context");
    }
    if (!kv) return Status(ErrorCode::InvalidArgument, "swap_context: no cache");
    const KvCacheConfig& a = kv_->config();
    const KvCacheConfig& b = kv->config();
    if (a.n_layers != b.n_layers || a.n_head_kv != b.n_head_kv || a.head_dim != b.head_dim || a.type != b.type ||
        a.page_tokens != b.page_tokens || a.values_in_keys != b.values_in_keys || a.value_offset != b.value_offset || a.auxiliary_dim != b.auxiliary_dim ||
        a.window != b.window || n_past < 0 || n_past > b.max_positions) {
        return Status(ErrorCode::InvalidArgument, "swap_context: a cache of another layout");
    }
    std::swap(kv_, kv);
    std::swap(n_past_, n_past);
    return Status::ok();
}

Result<std::unique_ptr<KvCache>> Transformer::make_kv_cache(int32_t positions) const {
    KvCacheConfig c = kv_->config();
    c.max_positions = std::max(positions, 1);
    return KvCache::create(c);
}

Status Transformer::move_keys(uint8_t* image, int32_t slots, int32_t first, int32_t count, int32_t from, int32_t to,
                              ThreadPool* pool) const {
    if (slots < 1 || first < 0 || count < 0 || first + count > slots || from < 0 || to < 0) {
        return Status(ErrorCode::InvalidArgument, "move_keys: the slots or positions do not fit");
    }
    LIYAB_RETURN_IF_ERROR(reuse_support());
    const ModelConfig& c = config_;
    // Nothing turns: every row keeps its bits.
    if (count == 0 || from == to || c.rope_dim == 0) return Status::ok();
    const DType type = kv_->dtype();
    const bool halves = c.spec->rope == RopeLayout::Halves;
    const auto hd = static_cast<size_t>(c.head_dim);
    // The leading channels a move touches: the rotated ones, in whole blocks
    // of a quantized row (the blocks after them keep their bits).
    const int32_t block = quant::is_block_quantized(type) ? quant::kBlock : 1;
    const bool norm_after = c.spec->qk_norm == QkNormOrder::AfterRope;

    // Each table's turns, back from the stored positions and on to the new
    // ones, computed as rope() computes them (the same float operations).
    struct Turns {
        std::vector<float> cos_from, sin_from, cos_to, sin_to;  // [position][pair]
    };
    std::vector<Turns> turns(rope_tables_.size());
    auto turns_of = [&](size_t table) -> const Turns& {
        Turns& t = turns[table];
        const RopeTable& rt = rope_tables_[table];
        const auto half = static_cast<int32_t>(rt.inv_freq.size());
        if (!t.cos_from.empty() || half == 0) return t;
        const auto n = static_cast<size_t>(count) * static_cast<size_t>(half);
        t.cos_from.resize(n);
        t.sin_from.resize(n);
        t.cos_to.resize(n);
        t.sin_to.resize(n);
        for (int32_t i = 0; i < count; ++i) {
            const float pf = static_cast<float>(from + i) / rt.scale;
            const float pt = static_cast<float>(to + i) / rt.scale;
            for (int32_t j = 0; j < half; ++j) {
                const size_t at = static_cast<size_t>(i) * static_cast<size_t>(half) + static_cast<size_t>(j);
                const float tf = pf * rt.inv_freq[static_cast<size_t>(j)];
                const float tt = pt * rt.inv_freq[static_cast<size_t>(j)];
                t.cos_from[at] = std::cos(tf);
                t.sin_from[at] = std::sin(tf);
                t.cos_to[at] = std::cos(tt);
                t.sin_to[at] = std::sin(tt);
            }
        }
        return t;
    };
    // Per attention block: its table and, per KV head and touched channel,
    // what the stored value is multiplied by to get the rotated one (the key
    // scale over a norm-after-RoPE weight) and back.
    struct Slot {
        const Turns* turns = nullptr;
        std::vector<float> unscale, rescale;  // [kv_head][touched channel]
        int32_t heads = 0;
        int32_t rope_dim = 0;
        int32_t touched = 0;
    };
    std::vector<Slot> blocks;
    for (const Layer& L : layers_) {
        // Blocks sharing another's keys hold none (their slot is the other's).
        if (L.kv_slot < 0 || c.blocks[static_cast<size_t>(&L - layers_.data())].kv_source >= 0) continue;
        if (static_cast<size_t>(L.kv_slot) >= blocks.size()) blocks.resize(static_cast<size_t>(L.kv_slot) + 1);
        // A NoPE block's keys do not depend on their position: they stay as stored (no turns).
        if (!c.blocks[static_cast<size_t>(&L - layers_.data())].rope) continue;
        Slot& s = blocks[static_cast<size_t>(L.kv_slot)];
        s.heads = c.blocks[static_cast<size_t>(&L - layers_.data())].n_head_kv;
        s.rope_dim = c.blocks[static_cast<size_t>(&L - layers_.data())].rope_dim;
        s.touched = (s.rope_dim + block - 1) / block * block;
        const int32_t touched = s.touched;
        s.turns = &turns_of(static_cast<size_t>(L.rope_table));
        s.unscale.assign(static_cast<size_t>(s.heads) * static_cast<size_t>(touched), 1.0f);
        s.rescale.assign(s.unscale.size(), 1.0f);
        for (size_t g = 0; g < static_cast<size_t>(s.heads); ++g) {
            for (size_t i = 0; i < static_cast<size_t>(touched); ++i) {
                float f = key_scale_.empty() ? 1.0f : key_scale_[(static_cast<size_t>(L.kv_slot) * static_cast<size_t>(c.n_head_kv) + g) * hd + i];
                if (norm_after && !L.k_norm.empty()) f /= L.k_norm[i];
                s.unscale[g * static_cast<size_t>(touched) + i] = f;
                s.rescale[g * static_cast<size_t>(touched) + i] = 1.0f / f;
            }
        }
    }
    const auto n_blocks = static_cast<int64_t>(blocks.size());
    auto move = [&](int64_t begin, int64_t end) {
        thread_local std::vector<float> row;
        for (int64_t task = begin; task < end; ++task) {
            const auto slot = static_cast<int32_t>(task % n_blocks);
            const auto i = static_cast<int32_t>(task / n_blocks);
            const Slot& s = blocks[static_cast<size_t>(slot)];
            if (s.turns == nullptr) continue;
            const int32_t touched = s.touched, half = s.rope_dim / 2;
            row.resize(static_cast<size_t>(touched));
            const size_t at = static_cast<size_t>(i) * static_cast<size_t>(half);
            for (int32_t g = 0; g < s.heads; ++g) {
                uint8_t* r = image + kv_->image_key_offset(slots, slot, first + i, g);
                quant::dequantize_row(type, r, row.data(), touched);
                const float* un = s.unscale.data() + static_cast<size_t>(g) * static_cast<size_t>(touched);
                const float* re = s.rescale.data() + static_cast<size_t>(g) * static_cast<size_t>(touched);
                for (int32_t ch = 0; ch < s.rope_dim; ++ch) row[static_cast<size_t>(ch)] *= un[ch];
                for (int32_t j = 0; j < half; ++j) {
                    const int32_t a = halves ? j : 2 * j;
                    const int32_t b = halves ? j + half : 2 * j + 1;
                    const size_t k = at + static_cast<size_t>(j);
                    const float r0 = row[static_cast<size_t>(a)];
                    const float r1 = row[static_cast<size_t>(b)];
                    // Back by the stored position's angle, then on by the new one's.
                    const float x0 = r0 * s.turns->cos_from[k] + r1 * s.turns->sin_from[k];
                    const float x1 = r1 * s.turns->cos_from[k] - r0 * s.turns->sin_from[k];
                    row[static_cast<size_t>(a)] = x0 * s.turns->cos_to[k] - x1 * s.turns->sin_to[k];
                    row[static_cast<size_t>(b)] = x0 * s.turns->sin_to[k] + x1 * s.turns->cos_to[k];
                }
                for (int32_t ch = 0; ch < s.rope_dim; ++ch) row[static_cast<size_t>(ch)] *= re[ch];
                quant::quantize_row(type, row.data(), r, touched);
            }
        }
    };
    const int64_t tasks = int64_t{count} * n_blocks;
    if (pool != nullptr && tasks >= 64) {
        pool->parallel_for(tasks, move);
    } else {
        move(0, tasks);
    }
    return Status::ok();
}

Result<std::span<const float>> Transformer::forward(std::span<const int32_t> tokens, Logits logits,
                                                    const Route& route, ThreadPool& pool, const ForwardHooks* hooks) {
    const ModelConfig& c = config_;
    const auto n = static_cast<int32_t>(tokens.size());
    last_exit_layer_ = -1;
    predicted_layer_ = -1;  // an early exit can leave a guess for a block that never ran
    if (n == 0) return Status(ErrorCode::InvalidArgument, "forward() needs at least one token");
    if (!c.causal && n_past_ != 0) {
        return Status(ErrorCode::Unsupported, "bidirectional attention reads a complete input from an empty context");
    }
    if (tree_base_ >= 0) return Status(ErrorCode::InvalidArgument, "a tree pass awaits keep_path()");
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    for (const int32_t id : tokens) {
        if (id < 0 || id >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
    }
    LIYAB_RETURN_IF_ERROR(kv_->reserve(n_past_ + n));  // maps KV pages; ContextFull past the limit
    // The DeltaNet mixers overwrite these rollback slots; they become valid
    // only once every block has run (a failed pass leaves no half-written checkpoint).
    const int32_t recorded = checkpointing_ ? std::min(n, rollback_window_) : 0;
    for (int32_t i = 0; i < recorded; ++i) {
        checkpoint_pos_[static_cast<size_t>((n_past_ + n - 1 - i) % rollback_window_)] = -1;
    }
    // Early exit would leave the skipped blocks' recurrent states behind: attention-only stacks.
    // So would AltUp's streams and per-layer embeddings: the residual is more
    // than x_ there, and a block skipped or cut short leaves them behind.
    const bool plain_residual = c.altup_streams <= 1 && c.ple_dim == 0 && c.hc_count == 0;
    EarlyExitHook* early_exit = hooks != nullptr && n == 1 && logits == Logits::Last && !c.hybrid() && plain_residual
                                    ? hooks->early_exit
                                    : nullptr;
    const HeadMaskHook* head_mask = hooks != nullptr ? hooks->head_mask : nullptr;
    const std::atomic<bool>* cancel = hooks != nullptr ? hooks->cancel : nullptr;
    FfnSkipHook* ffn_skip = hooks != nullptr && n == 1 && plain_residual ? hooks->ffn_skip : nullptr;
    FfnMatmulHook* ffn_hook = hooks != nullptr ? hooks->ffn_matmul : nullptr;
    last_ffn_skips_ = 0;

    // Latent attention: one form for the whole pass (LatentForm). A single
    // token (decoding) is always absorbed, and so is key calibration (its
    // ranges are measured on absorbed queries); otherwise the pinned form, or
    // the one the measured costs predict faster for this batch and context.
    latent_expanded_ = false;
    latent_ms_ = 0.0;
    LatentFormChooser::Work latent_work;
    // A tree pass is absorbed too: its tokens are decoding steps of their
    // branches, and the expanded form's tiles assume a contiguous pass.
    const bool latent = c.mla.kv_rank > 0 && n >= 2 && tree_depth_.empty();
    if (latent) {
        int32_t window = 0;
        for (int32_t l = 0; l < c.n_layers; ++l) {
            if (c.blocks[static_cast<size_t>(l)].latent) {
                window = layers_[static_cast<size_t>(l)].window;
                break;
            }
        }
        const int64_t pairs = latent_visible(live_segment(0, n, logits), window);
        int64_t positions = 0;
        for (const auto& [a, b] : mla_union_) positions += b - a;
        latent_work = LatentFormChooser::work(c, n, pairs, positions);
        if (key_ranges_ == nullptr) {
            latent_expanded_ = latent_form_ == LatentForm::Expanded ||
                               (latent_form_ == LatentForm::Auto && latent_chooser_.expand(latent_work));
        }
    }
    const auto un = static_cast<size_t>(n);
    size_batch(n);
    LIYAB_RETURN_IF_ERROR(embed(tokens, std::span<float>(x_.data(), un * static_cast<size_t>(c.n_embd))));
    LIYAB_RETURN_IF_ERROR(enter_blocks(tokens, route));
    begin_live_pass(n, logits);
    LIYAB_RETURN_IF_ERROR(add_positions(n));

    for (int32_t l = 0; l < c.n_layers; ++l) {
        if (cancel != nullptr && cancel->load(std::memory_order_relaxed)) {
            return Status(ErrorCode::Cancelled, "forward pass cancelled");
        }
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        LIYAB_RETURN_IF_ERROR(run_block(l, w, n, route, pool, head_mask, ffn_skip, ffn_hook));

        // --- early-exit probe (experimental) ---
        if (early_exit != nullptr && early_exit->probe_after(l, c.n_layers)) {
            LIYAB_RETURN_IF_ERROR(compute_logits(route, 0, 1, pool));
            if (early_exit->should_exit(std::span<const float>(logits_.data(), logits_.size()))) {
                LIYAB_RETURN_IF_ERROR(propagate_kv(route, l + 1, n_past_));
                last_exit_layer_ = l;
                n_past_ += 1;
                kv_->release_unreachable(n_past_, max_batch_);
                return std::span<const float>(logits_.data(), logits_.size());
            }
        }
    }

    LIYAB_RETURN_IF_ERROR(leave_blocks(n, route));
    // A tree pass's checkpoints hold its tokens' states, not positions':
    // keep_path() marks those of the kept branch.
    for (int32_t i = 0; i < recorded && tree_depth_.empty(); ++i) {
        const int32_t pos = n_past_ + n - 1 - i;
        checkpoint_pos_[static_cast<size_t>(pos % rollback_window_)] = pos;
    }
    if (c.mla.kv_rank > 0) ++(latent_expanded_ ? latent_stats_.expanded_passes : latent_stats_.absorbed_passes);
    n_past_ += n;
    kv_->release_unreachable(n_past_, max_batch_);  // recycle pages that left the window
    if (expert_store_ != nullptr && ++decode_steps_ % 16 == 0) expert_store_->age();
    if (logits == Logits::None) return std::span<const float>();

    // --- output head ---
    const int32_t out_rows = logits == Logits::All ? n : 1;
    if (Status s = compute_logits(route, logits == Logits::All ? 0 : un - 1, out_rows, pool); !s.is_ok()) {
        n_past_ -= n;
        return s;
    }
    return std::span<const float>(logits_.data(), logits_.size());
}

Result<std::span<const float>> Transformer::forward_tree(std::span<const int32_t> tokens,
                                                         std::span<const int32_t> parents, const Route& route,
                                                         ThreadPool& pool) {
    if (!config_.causal) return Status(ErrorCode::Unsupported, "bidirectional attention has no causal tree drafts");
    const auto n = static_cast<int32_t>(tokens.size());
    if (tree_base_ >= 0) return Status(ErrorCode::InvalidArgument, "a tree pass awaits keep_path()");
    if (n == 0 || parents.size() != tokens.size() || parents[0] != -1) {
        return Status(ErrorCode::InvalidArgument, "forward_tree: a tree starts with one root token");
    }
    if (n > max_batch_) return Status(ErrorCode::InvalidArgument, "forward_tree: more tokens than max_batch()");
    if (config_.hybrid() && n > rollback_window_) {
        return Status(ErrorCode::Unsupported, "forward_tree: the rollback window cannot hold every token's state");
    }
    const auto un = static_cast<size_t>(n);
    tree_depth_.assign(un, 0);
    for (size_t t = 1; t < un; ++t) {
        const int32_t p = parents[t];
        if (p < 0 || p >= static_cast<int32_t>(t)) {
            tree_depth_.clear();
            return Status(ErrorCode::InvalidArgument, "forward_tree: a token's parent must come before it");
        }
        tree_depth_[t] = tree_depth_[static_cast<size_t>(p)] + 1;
    }
    tree_parent_.assign(parents.begin(), parents.end());
    tree_stride_ = *std::max_element(tree_depth_.begin(), tree_depth_.end()) + 1;
    tree_ancestors_.assign(un * static_cast<size_t>(tree_stride_), 0);
    for (size_t t = 0; t < un; ++t) {
        int32_t* row = tree_ancestors_.data() + t * static_cast<size_t>(tree_stride_);
        for (auto a = static_cast<int32_t>(t); a >= 0; a = tree_parent_[static_cast<size_t>(a)]) {
            row[tree_depth_[static_cast<size_t>(a)]] = a;
        }
    }
    // Every token's recurrent state is checkpointed: a later branch starts
    // from one, and keep_path() keeps the branch's.
    const bool checkpointing = checkpointing_;
    checkpointing_ = true;
    auto logits = forward(tokens, Logits::All, route, pool);
    checkpointing_ = checkpointing;
    tree_depth_.clear();
    if (!logits) {
        tree_parent_.clear();
        return logits.status();
    }
    tree_base_ = n_past_ - n;
    return logits;
}

Status Transformer::keep_path(std::span<const int32_t> path) {
    if (tree_base_ < 0) return Status(ErrorCode::InvalidArgument, "keep_path: no tree pass is pending");
    if (path.empty() || path[0] != 0) return Status(ErrorCode::InvalidArgument, "keep_path: a branch starts at the root");
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] <= 0 || path[i] >= static_cast<int32_t>(tree_parent_.size()) ||
            tree_parent_[static_cast<size_t>(path[i])] != path[i - 1]) {
            return Status(ErrorCode::InvalidArgument, "keep_path: not a branch of the tree");
        }
    }
    const int32_t base = tree_base_;
    // Ascending: token path[i] >= i of a branch is never the target of an
    // earlier move (path[j] >= j > i for every later j), so no row is
    // overwritten before it is read; the moves within the n <= window ring
    // slots below are distinct for the same reason.
    for (size_t i = 1; i < path.size(); ++i) {
        const int32_t to = base + static_cast<int32_t>(i);
        LIYAB_RETURN_IF_ERROR(kv_->copy_position(base + path[i], to));
        if (states_.empty() || rollback_window_ == 0) continue;
        const auto from_slot = static_cast<size_t>((base + path[i]) % rollback_window_);
        const auto to_slot = static_cast<size_t>(to % rollback_window_);
        for (RecurrentState& st : states_) {
            const size_t per_state = st.conv.size() + st.ssm.size();
            if (per_state > 0 && from_slot != to_slot) {
                std::copy_n(st.checkpoints.data() + from_slot * per_state, per_state,
                            st.checkpoints.data() + to_slot * per_state);
            }
        }
    }
    for (size_t i = 0; i < path.size() && rollback_window_ > 0; ++i) {
        const int32_t pos = base + static_cast<int32_t>(i);
        checkpoint_pos_[static_cast<size_t>(pos % rollback_window_)] = pos;
    }
    tree_base_ = -1;
    tree_parent_.clear();
    // The recurrent states after path.back() come back from its checkpoint
    // (none to restore when the branch is every token: they are current).
    return truncate(base + static_cast<int32_t>(path.size()));
}

Status Transformer::adopt_context(int32_t n, std::vector<OwnedSnapshot> snapshots, bool pin_end) {
    auto fail = [this](const char* why) {
        reset();
        return Status(ErrorCode::InvalidArgument, why);
    };
    const size_t floats = recurrent_state_bytes() / sizeof(float);
    tree_base_ = -1;  // the context is replaced: a pending tree pass with it
    tree_parent_.clear();
    if (n < 0 || n > context_length_ || kv_->window() > 0) return fail("the context does not fit this cache");
    if (static_cast<int32_t>(snapshots.size()) >= kStateSnapshots || (!has_position_state() && !snapshots.empty())) {
        return fail("the context's recurrent states do not fit this model");
    }
    int32_t below = 0;
    for (const OwnedSnapshot& snap : snapshots) {
        if (snap.pos <= below || snap.pos >= n || snap.data.size() != floats) {
            return fail("the context's state snapshots do not fit this model");
        }
        below = snap.pos;
    }
    for (int32_t p = 0; p * kv_->config().page_tokens < n; ++p) {
        if (!kv_->page_mapped(p)) return fail("the context's cache pages are not mapped");
    }
    // Windowed blocks: the caller wrote the rows of the window before n and
    // before each snapshot (KvCache::write_window_rows()); a context missing
    // them would attend over zeros.
    if (kv_->has_windowed_layers()) {
        if (n > 0 && kv_->restorable(n) != n) return fail("the context's window rows are not in the cache");
        for (const OwnedSnapshot& snap : snapshots) {
            if (kv_->restorable(snap.pos) != snap.pos) return fail("a snapshot's window rows are not in the cache");
        }
    }
    if (n == 0) {
        reset();
        return Status::ok();
    }
    if (Status s = kv_->truncate(n); !s.is_ok()) return fail("cannot truncate the cache");
    std::fill(checkpoint_pos_.begin(), checkpoint_pos_.end(), -1);
    // A restore normally starts from begin_restore(), which moved the
    // snapshots' buffers to spare_buffers_; any left here are kept the same way.
    for (StateSnapshot& snap : snapshots_) spare_buffers_.push_back(std::move(snap.data));
    snapshots_.clear();
    for (OwnedSnapshot& snap : snapshots) {
        snapshots_.push_back(StateSnapshot{snap.pos, std::move(snap.data), snap.pinned});
    }
    n_past_ = n;
    snapshot_state(pin_end);  // in a kept buffer when one is left; pins the snapshots' windows
    spare_buffers_.clear();   // the others: never more memory than before the restore
    return Status::ok();
}

Status Transformer::abandon_pass() {
    if (!config_.hybrid()) return kv_->truncate(n_past_);
    const auto snap = std::find_if(snapshots_.begin(), snapshots_.end(),
                                   [&](const StateSnapshot& s) { return s.pos == n_past_; });
    if (snap == snapshots_.end()) {
        if (n_past_ == 0) {
            reset();
            return Status::ok();
        }
        return Status(ErrorCode::Unsupported, "no state snapshot at the abandoned pass's start");
    }
    LIYAB_RETURN_IF_ERROR(kv_->truncate(n_past_));
    const float* saved = snap->data.data();
    for (RecurrentState& st : states_) {
        if (!st.conv.empty()) std::copy_n(saved, st.conv.size(), st.conv.begin());
        if (!st.ssm.empty()) std::copy_n(saved + st.conv.size(), st.ssm.size(), st.ssm.begin());
        if (!st.conv.empty() || !st.ssm.empty()) saved += st.conv.size() + st.ssm.size();
    }
    return Status::ok();
}

Status Transformer::truncate(int32_t n) {
    if (n < 0 || n > n_past_) return Status(ErrorCode::InvalidArgument, "truncate beyond cached positions");
    if (tree_base_ >= 0) {  // a pending tree's slots are no context: only what preceded it stays
        if (n > tree_base_) return Status(ErrorCode::InvalidArgument, "truncate into a pending tree pass");
        tree_base_ = -1;
        tree_parent_.clear();
    }
    if (n == n_past_) return Status::ok();
    if (n == 0) {
        reset();
        return Status::ok();
    }
    if (config_.hybrid()) {
        const int32_t slot = rollback_window_ > 0 ? (n - 1) % rollback_window_ : -1;
        const bool in_ring = slot >= 0 && checkpoint_pos_[static_cast<size_t>(slot)] == n - 1;
        const auto snap = std::find_if(snapshots_.begin(), snapshots_.end(),
                                       [n](const StateSnapshot& s) { return s.pos == n; });
        if (!in_ring && snap == snapshots_.end()) {
            return Status(ErrorCode::Unsupported,
                          "recurrent (DeltaNet) states can only be rolled back inside the rollback window or to a "
                          "snapshot");
        }
        LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
        const float* saved = in_ring ? nullptr : snap->data.data();
        for (RecurrentState& st : states_) {
            if (st.conv.empty() && st.ssm.empty()) continue;
            if (in_ring) saved = st.checkpoints.data() + static_cast<size_t>(slot) * (st.conv.size() + st.ssm.size());
            if (!st.conv.empty()) std::copy_n(saved, st.conv.size(), st.conv.begin());
            if (!st.ssm.empty()) std::copy_n(saved + st.conv.size(), st.ssm.size(), st.ssm.begin());
            if (!in_ring) saved += st.conv.size() + st.ssm.size();
        }
        // Checkpoints and snapshots of the discarded positions describe a
        // future that no longer exists.
        for (int32_t& p : checkpoint_pos_) {
            if (p >= n) p = -1;
        }
        std::erase_if(snapshots_, [n](const StateSnapshot& s) { return s.pos > n; });
        pin_snapshots();
        n_past_ = n;
        return Status::ok();
    }
    // Windowed blocks: refused when their rows of the window before n were
    // released (KvCache::truncate()); their snapshots past n go.
    LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
    if (!snapshots_.empty() && snapshots_.back().pos > n) {
        std::erase_if(snapshots_, [n](const StateSnapshot& s) { return s.pos > n; });
        pin_snapshots();
    }
    n_past_ = n;
    return Status::ok();
}

// ---------------------------------------------------------------------------
// Several sequences
// ---------------------------------------------------------------------------
Transformer::Segment Transformer::live_segment(int32_t first, int32_t n, Logits logits) noexcept {
    Segment s;
    s.first = first;
    s.n = n;
    s.kv = kv_.get();
    s.n_past = n_past_;
    s.states = &states_;
    s.rollback = rollback_window_;
    s.checkpoint = rollback_window_ > 0 && checkpointing_;
    s.adapter = adapter_targets_.empty() ? nullptr : &adapter_targets_;
    s.expanded = latent_expanded_;
    s.logits = logits;
    return s;
}

void Transformer::begin_live_pass(int32_t n, Logits logits) noexcept {
    segs_.assign(1, live_segment(0, n, logits));  // reuses the vector's storage: no allocation per pass
    seg_of_.clear();
    adapter_rows_.clear();
    adapters_in_pass_ = segs_.front().adapter != nullptr;
}

Result<std::unique_ptr<Transformer::Context>> Transformer::make_context() const {
    if (!kv_) return Status(ErrorCode::InvalidArgument, "make_context: no live context to lay a new one out like");
    auto kv = KvCache::create(kv_->config());
    if (!kv) return kv.status();
    auto context = std::make_unique<Context>();
    context->kv_ = std::move(kv).value();
    context->states_.resize(states_.size());
    for (size_t i = 0; i < states_.size(); ++i) {
        RecurrentState& st = context->states_[i];
        st.conv.assign(states_[i].conv.size(), 0.0f);
        st.ssm.assign(states_[i].ssm.size(), 0.0f);
        st.checkpoints.assign(static_cast<size_t>(rollback_window_) * (st.conv.size() + st.ssm.size()), 0.0f);
    }
    context->rollback_window_ = rollback_window_;
    context->checkpoint_pos_.assign(static_cast<size_t>(rollback_window_), -1);
    return context;
}

void Transformer::swap_context(Context& other) noexcept {
    std::swap(kv_, other.kv_);
    std::swap(n_past_, other.n_past_);
    states_.swap(other.states_);
    checkpoint_pos_.swap(other.checkpoint_pos_);
    std::swap(rollback_window_, other.rollback_window_);
    std::swap(checkpointing_, other.checkpointing_);
    snapshots_.swap(other.snapshots_);
    spare_buffers_.swap(other.spare_buffers_);
    std::swap(tree_base_, other.tree_base_);
    tree_parent_.swap(other.tree_parent_);
    std::swap(adapter_, other.adapter_);
    adapter_targets_.swap(other.adapter_targets_);
}

int32_t Transformer::positions_of(const Context& context) noexcept { return context.n_past_; }

Status Transformer::compute_logits_rows(const Route& route, std::span<const int32_t> rows, ThreadPool& pool) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const PhaseTimer timer(phases_.lm_head);
    file_->begin_layer(c.n_layers);  // output head slot of the prefetch window
    for (size_t i = 0; i < rows.size(); ++i) {
        norm_row(x_.data() + static_cast<size_t>(rows[i]) * d, opt(output_norm_), opt(output_norm_b_),
                 xb_.data() + i * d, c.n_embd);
    }
    logits_.resize(rows.size() * static_cast<size_t>(c.n_vocab));
    LIYAB_RETURN_IF_ERROR(
        matmul(route, route.ffn, *output_, xb_.data(), logits_.data(), static_cast<int32_t>(rows.size())));
    finish_logits(static_cast<int32_t>(rows.size()));
    if (c.final_softcap > 0.0f) apply_final_softcap(std::span<float>(logits_.data(), logits_.size()), c.final_softcap, pool);
    return Status::ok();
}

// One pass over several sequences. Everything a sequence's own state decides
// (its positions, what each token sees, its recurrent states and their
// checkpoints, its adapter, latent attention's form) is taken per segment,
// exactly as forward() takes it for that sequence alone; everything else
// runs over the rows of every segment at once, which is where the pass saves:
// each weight is read once for all of them (dense models in RAM are bound by
// that read when decoding), and a mixture of experts reads each chosen expert
// once for every token that chose it.
Status Transformer::forward_batch(std::span<const BatchItem> items, const Route& route, ThreadPool& pool,
                                  std::vector<std::span<const float>>& out, const ForwardHooks* hooks) {
    const ModelConfig& c = config_;
    out.clear();
    last_exit_layer_ = -1;
    last_ffn_skips_ = 0;
    predicted_layer_ = -1;
    if (items.empty()) return Status(ErrorCode::InvalidArgument, "forward_batch() needs at least one sequence");
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    if (hooks != nullptr && (hooks->early_exit != nullptr || hooks->ffn_skip != nullptr)) {
        return Status(ErrorCode::InvalidArgument,
                      "forward_batch(): early exit and FFN skipping apply to single-token decoding steps only");
    }
    if (!kv_) return Status(ErrorCode::InvalidArgument, "forward_batch(): no live context");
    segs_.clear();
    int32_t total = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const BatchItem& item = items[i];
        Context* ctx = item.context;
        if (item.tokens.empty()) return Status(ErrorCode::InvalidArgument, "forward_batch(): a sequence without tokens");
        for (const int32_t id : item.tokens) {
            if (id < 0 || id >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
        }
        if (ctx != nullptr && !ctx->holds()) {
            return Status(ErrorCode::InvalidArgument, "forward_batch(): an empty context");
        }
        for (size_t j = 0; j < i; ++j) {
            if (items[j].context == ctx) {
                return Status(ErrorCode::InvalidArgument, "forward_batch(): two items of one context");
            }
        }
        if ((ctx != nullptr ? ctx->tree_base_ : tree_base_) >= 0) {
            return Status(ErrorCode::InvalidArgument, "a tree pass awaits keep_path()");
        }
        const auto n = static_cast<int32_t>(item.tokens.size());
        Segment s;
        if (ctx == nullptr) {
            s = live_segment(total, n, item.logits);
        } else {
            s.first = total;
            s.n = n;
            s.kv = ctx->kv_.get();
            s.n_past = ctx->n_past_;
            s.states = &ctx->states_;
            s.rollback = ctx->rollback_window_;
            s.adapter = ctx->adapter_targets_.empty() ? nullptr : &ctx->adapter_targets_;
            s.logits = item.logits;
            s.context = ctx;
        }
        s.checkpoint = s.rollback > 0 && item.checkpoint;
        if (!c.causal && s.n_past != 0) {
            return Status(ErrorCode::Unsupported, "bidirectional attention reads a complete input from an empty context");
        }
        s.expanded = false;
        segs_.push_back(s);
        total += n;
    }
    auto n_past_of = [&](const Segment& s) -> int32_t& { return s.context != nullptr ? s.context->n_past_ : n_past_; };
    auto ring_of = [&](const Segment& s) -> std::vector<int32_t>& {
        return s.context != nullptr ? s.context->checkpoint_pos_ : checkpoint_pos_;
    };
    for (const Segment& s : segs_) {
        LIYAB_RETURN_IF_ERROR(s.kv->reserve(s.n_past + s.n));  // maps KV pages; ContextFull past the limit
    }
    // The DeltaNet mixers overwrite these rollback slots; they become valid
    // only once every block has run (as in forward()).
    for (const Segment& s : segs_) {
        const int32_t recorded = s.checkpoint ? std::min(s.n, s.rollback) : 0;
        for (int32_t i = 0; i < recorded; ++i) {
            ring_of(s)[static_cast<size_t>((s.n_past + s.n - 1 - i) % s.rollback)] = -1;
        }
    }
    // Latent attention: each segment in the form forward() would choose for
    // it alone (its own batch size and context), so its rows round as they
    // would there.
    latent_ms_ = 0.0;
    if (c.mla.kv_rank > 0) {
        int32_t window = 0;
        for (int32_t l = 0; l < c.n_layers; ++l) {
            if (c.blocks[static_cast<size_t>(l)].latent) {
                window = layers_[static_cast<size_t>(l)].window;
                break;
            }
        }
        for (Segment& s : segs_) {
            if (s.n < 2 || key_ranges_ != nullptr) continue;
            const int64_t pairs = latent_visible(s, window);
            int64_t positions = 0;
            for (const auto& [a, b] : mla_union_) positions += b - a;
            const LatentFormChooser::Work work = LatentFormChooser::work(c, s.n, pairs, positions);
            s.expanded = latent_form_ == LatentForm::Expanded ||
                         (latent_form_ == LatentForm::Auto && latent_chooser_.expand(work));
        }
    }
    batch_tokens_.clear();
    seg_of_.clear();
    adapter_rows_.clear();
    adapters_in_pass_ = false;
    for (size_t i = 0; i < items.size(); ++i) {
        batch_tokens_.insert(batch_tokens_.end(), items[i].tokens.begin(), items[i].tokens.end());
        seg_of_.insert(seg_of_.end(), items[i].tokens.size(), static_cast<int32_t>(i));
        adapters_in_pass_ = adapters_in_pass_ || segs_[i].adapter != nullptr;
    }
    const auto un = static_cast<size_t>(total);
    size_batch(total);
    LIYAB_RETURN_IF_ERROR(embed(batch_tokens_, std::span<float>(x_.data(), un * static_cast<size_t>(c.n_embd))));
    LIYAB_RETURN_IF_ERROR(add_positions(total));
    LIYAB_RETURN_IF_ERROR(enter_blocks(batch_tokens_, route));
    struct StepsOnly {
        bool& flag;
        ~StepsOnly() { flag = false; }
    } steps_only{steps_only_};
    steps_only_ = std::all_of(segs_.begin(), segs_.end(), [](const Segment& s) { return s.n == 1; });
    const HeadMaskHook* head_mask = hooks != nullptr ? hooks->head_mask : nullptr;
    const std::atomic<bool>* cancel = hooks != nullptr ? hooks->cancel : nullptr;
    FfnMatmulHook* ffn_hook = hooks != nullptr ? hooks->ffn_matmul : nullptr;

    for (int32_t l = 0; l < c.n_layers; ++l) {
        if (cancel != nullptr && cancel->load(std::memory_order_relaxed)) {
            // Every context back to where the pass started (forward()'s
            // abandon_pass(), each one made live for it in turn).
            const std::vector<Segment> segments = segs_;
            for (const Segment& s : segments) {
                if (s.context != nullptr) swap_context(*s.context);
                if (!abandon_pass().is_ok()) reset();  // a hybrid context without a snapshot here
                if (s.context != nullptr) swap_context(*s.context);
            }
            return Status(ErrorCode::Cancelled, "forward pass cancelled");
        }
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        LIYAB_RETURN_IF_ERROR(run_block(l, w, total, route, pool, head_mask, nullptr, ffn_hook));
    }
    LIYAB_RETURN_IF_ERROR(leave_blocks(total, route));

    for (const Segment& s : segs_) {
        const int32_t recorded = s.checkpoint ? std::min(s.n, s.rollback) : 0;
        for (int32_t i = 0; i < recorded; ++i) {
            const int32_t pos = s.n_past + s.n - 1 - i;
            ring_of(s)[static_cast<size_t>(pos % s.rollback)] = pos;
        }
        if (c.mla.kv_rank > 0) ++(s.expanded ? latent_stats_.expanded_passes : latent_stats_.absorbed_passes);
        n_past_of(s) = s.n_past + s.n;
        s.kv->release_unreachable(s.n_past + s.n, max_batch_);  // recycle pages that left the window
    }
    if (expert_store_ != nullptr && ++decode_steps_ % 16 == 0) expert_store_->age();

    // --- output head: the rows each segment asked for, segment after segment ---
    std::vector<int32_t> rows;
    std::vector<std::pair<size_t, size_t>> spans(segs_.size());  // per segment: its first logits row, its count
    for (size_t i = 0; i < segs_.size(); ++i) {
        const Segment& s = segs_[i];
        const size_t at = rows.size();
        if (s.logits == Logits::Last) rows.push_back(s.first + s.n - 1);
        if (s.logits == Logits::All) {
            for (int32_t t = s.first; t < s.first + s.n; ++t) rows.push_back(t);
        }
        spans[i] = {at, rows.size() - at};
        if (s.adapter != nullptr && rows.size() > at) {
            adapter_rows_.push_back({static_cast<int32_t>(at), static_cast<int32_t>(rows.size() - at), s.adapter});
        }
    }
    if (!rows.empty()) {
        logits_rows_ = true;
        const Status status = compute_logits_rows(route, rows, pool);
        logits_rows_ = false;
        adapter_rows_.clear();
        if (!status.is_ok()) {
            for (const Segment& s : segs_) n_past_of(s) = s.n_past;
            return status;
        }
    }
    const auto vocab = static_cast<size_t>(c.n_vocab);
    for (const auto& [at, count] : spans) {
        out.emplace_back(count == 0 ? std::span<const float>()
                                    : std::span<const float>(logits_.data() + at * vocab, count * vocab));
    }
    return Status::ok();
}

}  // namespace liyab
