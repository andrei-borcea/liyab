#include "core/quant.h"

#include <algorithm>
#include <cmath>

#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define LIYAB_NEON 1
#endif

namespace liyab::quant {

#if !defined(__aarch64__)
namespace {

inline float bits_to_f32(uint32_t w) noexcept {
    float f;
    std::memcpy(&f, &w, sizeof f);
    return f;
}
inline uint32_t f32_to_bits(float f) noexcept {
    uint32_t w;
    std::memcpy(&w, &f, sizeof w);
    return w;
}

}  // namespace
#endif

// Branch-free IEEE half conversions (Maratyszcza/FP16). Exact for all inputs,
// including subnormals, infinities and NaN. Must not be built with -ffast-math.
float fp16_to_fp32(uint16_t h) noexcept {
#if defined(__aarch64__)
    __fp16 v;
    std::memcpy(&v, &h, sizeof v);
    return static_cast<float>(v);
#else
    const uint32_t w = static_cast<uint32_t>(h) << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;
    const float normalized = bits_to_f32((two_w >> 4) + (0xE0u << 23)) * 0x1.0p-112f;
    const float denormalized = bits_to_f32((two_w >> 17) | (126u << 23)) - 0.5f;
    const uint32_t result =
        sign | (two_w < (1u << 27) ? f32_to_bits(denormalized) : f32_to_bits(normalized));
    return bits_to_f32(result);
#endif
}

uint16_t fp32_to_fp16(float f) noexcept {
#if defined(__aarch64__)
    const __fp16 v = static_cast<__fp16>(f);
    uint16_t h;
    std::memcpy(&h, &v, sizeof h);
    return h;
#else
    float base = (std::fabs(f) * 0x1.0p+112f) * 0x1.0p-110f;
    const uint32_t w = f32_to_bits(f);
    const uint32_t shl1_w = w + w;
    const uint32_t sign = w & 0x80000000u;
    uint32_t bias = shl1_w & 0xFF000000u;
    if (bias < 0x71000000u) bias = 0x71000000u;
    base = bits_to_f32((bias >> 1) + 0x07800000u) + base;
    const uint32_t bits = f32_to_bits(base);
    const uint32_t nonsign = ((bits >> 13) & 0x00007C00u) + (bits & 0x00000FFFu);
    return static_cast<uint16_t>((sign >> 16) | (shl1_w > 0xFF000000u ? 0x7E00u : nonsign));
#endif
}

void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * kBlock;
        float amax = 0.0f;
        for (int j = 0; j < kBlock; ++j) amax = std::max(amax, std::fabs(xb[j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < kBlock; ++j) {
            y[i].qs[j] = static_cast<int8_t>(std::lround(xb[j] * id));
        }
    }
}

void quantize_row_q4_0(const float* x, BlockQ4_0* y, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * kBlock;
        float amax = 0.0f;
        float max = 0.0f;  // signed value with the largest magnitude
        for (int j = 0; j < kBlock; ++j) {
            if (std::fabs(xb[j]) > amax) {
                amax = std::fabs(xb[j]);
                max = xb[j];
            }
        }
        const float d = max / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        for (int j = 0; j < kBlock / 2; ++j) {
            const int q0 = std::min(15, static_cast<int>(xb[j] * id + 8.5f));
            const int q1 = std::min(15, static_cast<int>(xb[j + kBlock / 2] * id + 8.5f));
            y[i].qs[j] = static_cast<uint8_t>(std::max(0, q0) | (std::max(0, q1) << 4));
        }
    }
}

void quantize_row_q4_1(const float* x, BlockQ4_1* y, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    for (int64_t i = 0; i < nb; ++i) {
        const float* xb = x + i * kBlock;
        float lo = xb[0];
        float hi = xb[0];
        for (int j = 1; j < kBlock; ++j) {
            lo = std::min(lo, xb[j]);
            hi = std::max(hi, xb[j]);
        }
        const float d = (hi - lo) / 15.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = fp32_to_fp16(d);
        y[i].m = fp32_to_fp16(lo);
        for (int j = 0; j < kBlock / 2; ++j) {
            const int q0 = std::min(15, static_cast<int>((xb[j] - lo) * id + 0.5f));
            const int q1 = std::min(15, static_cast<int>((xb[j + kBlock / 2] - lo) * id + 0.5f));
            y[i].qs[j] = static_cast<uint8_t>(std::max(0, q0) | (std::max(0, q1) << 4));
        }
    }
}

void quantize_row(DType type, const float* x, void* dst, int64_t n) noexcept {
    switch (type) {
        case DType::F32:
            std::memcpy(dst, x, static_cast<size_t>(n) * sizeof(float));
            break;
        case DType::F16: {
            auto* h = static_cast<uint16_t*>(dst);
            for (int64_t i = 0; i < n; ++i) h[i] = fp32_to_fp16(x[i]);
            break;
        }
        case DType::Q8_0:
            quantize_row_q8_0(x, static_cast<BlockQ8_0*>(dst), n);
            break;
        case DType::Q4_0:
            quantize_row_q4_0(x, static_cast<BlockQ4_0*>(dst), n);
            break;
        case DType::Q4_1:
            quantize_row_q4_1(x, static_cast<BlockQ4_1*>(dst), n);
            break;
    }
}

void dequantize_row(DType type, const void* src, float* y, int64_t n) noexcept {
    switch (type) {
        case DType::F32:
            std::memcpy(y, src, static_cast<size_t>(n) * sizeof(float));
            break;
        case DType::F16: {
            const auto* h = static_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < n; ++i) y[i] = fp16_to_fp32(h[i]);
            break;
        }
        case DType::Q8_0: {
            const auto* b = static_cast<const BlockQ8_0*>(src);
            for (int64_t i = 0; i < n / kBlock; ++i) {
                const float d = fp16_to_fp32(b[i].d);
                for (int j = 0; j < kBlock; ++j) y[i * kBlock + j] = d * b[i].qs[j];
            }
            break;
        }
        case DType::Q4_0: {
            const auto* b = static_cast<const BlockQ4_0*>(src);
            for (int64_t i = 0; i < n / kBlock; ++i) {
                const float d = fp16_to_fp32(b[i].d);
                float* yb = y + i * kBlock;
                for (int j = 0; j < kBlock / 2; ++j) {
                    yb[j] = d * static_cast<float>((b[i].qs[j] & 0x0F) - 8);
                    yb[j + kBlock / 2] = d * static_cast<float>((b[i].qs[j] >> 4) - 8);
                }
            }
            break;
        }
        case DType::Q4_1: {
            const auto* b = static_cast<const BlockQ4_1*>(src);
            for (int64_t i = 0; i < n / kBlock; ++i) {
                const float d = fp16_to_fp32(b[i].d);
                const float m = fp16_to_fp32(b[i].m);
                float* yb = y + i * kBlock;
                for (int j = 0; j < kBlock / 2; ++j) {
                    yb[j] = d * static_cast<float>(b[i].qs[j] & 0x0F) + m;
                    yb[j + kBlock / 2] = d * static_cast<float>(b[i].qs[j] >> 4) + m;
                }
            }
            break;
        }
    }
}

#if defined(LIYAB_NEON)
namespace {
// Sum of a(16 x int8) · b(16 x int8) into four int32 lanes.
inline int32x4_t dot_i8x16(int32x4_t acc, int8x16_t a, int8x16_t b) noexcept {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)));
#endif
}
}  // namespace
#endif

float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* x, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
#if defined(LIYAB_NEON)
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int64_t i = 0;
    for (; i + 1 < nb; i += 2) {
        for (int k = 0; k < 2; ++k) {
            const uint8x16_t v = vld1q_u8(w[i + k].qs);
            const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v, mask)), eight);
            const int8x16_t hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v, 4)), eight);
            int32x4_t s = dot_i8x16(vdupq_n_s32(0), lo, vld1q_s8(x[i + k].qs));
            s = dot_i8x16(s, hi, vld1q_s8(x[i + k].qs + 16));
            const float d = fp16_to_fp32(w[i + k].d) * fp16_to_fp32(x[i + k].d);
            if (k == 0) acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(s), d);
            else acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(s), d);
        }
    }
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#else
    float sum = 0.0f;
    int64_t i = 0;
#endif
    for (; i < nb; ++i) {
        int32_t s = 0;
        for (int j = 0; j < kBlock / 2; ++j) {
            s += ((w[i].qs[j] & 0x0F) - 8) * x[i].qs[j];
            s += ((w[i].qs[j] >> 4) - 8) * x[i].qs[j + kBlock / 2];
        }
        sum += static_cast<float>(s) * fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
    }
    return sum;
}

// sum_j (q_j * dw + mw) * (x_j * dx) = dw * dx * sum(q_j * x_j) + mw * dx * sum(x_j)
float dot_q4_1_q8_0(const BlockQ4_1* w, const BlockQ8_0* x, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    float sum = 0.0f;
    int64_t i = 0;
#if defined(LIYAB_NEON)
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t ones = vdupq_n_s8(1);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; i < nb; ++i) {
        const uint8x16_t v = vld1q_u8(w[i].qs);
        const int8x16_t lo = vreinterpretq_s8_u8(vandq_u8(v, mask));
        const int8x16_t hi = vreinterpretq_s8_u8(vshrq_n_u8(v, 4));
        const int8x16_t x0 = vld1q_s8(x[i].qs);
        const int8x16_t x1 = vld1q_s8(x[i].qs + 16);
        const int32x4_t qx = dot_i8x16(dot_i8x16(vdupq_n_s32(0), lo, x0), hi, x1);
        const int32x4_t sx = dot_i8x16(dot_i8x16(vdupq_n_s32(0), ones, x0), ones, x1);
        const float dx = fp16_to_fp32(x[i].d);
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(qx), fp16_to_fp32(w[i].d) * dx);
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(sx), fp16_to_fp32(w[i].m) * dx);
    }
    sum = vaddvq_f32(acc);
#endif
    for (; i < nb; ++i) {
        int32_t qx = 0;
        int32_t sx = 0;
        for (int j = 0; j < kBlock / 2; ++j) {
            qx += (w[i].qs[j] & 0x0F) * x[i].qs[j] + (w[i].qs[j] >> 4) * x[i].qs[j + kBlock / 2];
            sx += x[i].qs[j] + x[i].qs[j + kBlock / 2];
        }
        const float dx = fp16_to_fp32(x[i].d);
        sum += static_cast<float>(qx) * fp16_to_fp32(w[i].d) * dx + static_cast<float>(sx) * fp16_to_fp32(w[i].m) * dx;
    }
    return sum;
}

float dot_quantized(DType type, const void* row, const BlockQ8_0* x, int64_t n) noexcept {
    switch (type) {
        case DType::Q4_0: return dot_q4_0_q8_0(static_cast<const BlockQ4_0*>(row), x, n);
        case DType::Q4_1: return dot_q4_1_q8_0(static_cast<const BlockQ4_1*>(row), x, n);
        case DType::Q8_0: return dot_q8_0_q8_0(static_cast<const BlockQ8_0*>(row), x, n);
        default: return 0.0f;
    }
}

float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
#if defined(LIYAB_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int64_t i = 0; i < nb; ++i) {
        int32x4_t s = dot_i8x16(vdupq_n_s32(0), vld1q_s8(w[i].qs), vld1q_s8(x[i].qs));
        s = dot_i8x16(s, vld1q_s8(w[i].qs + 16), vld1q_s8(x[i].qs + 16));
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d));
    }
    return vaddvq_f32(acc);
#else
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        int32_t s = 0;
        for (int j = 0; j < kBlock; ++j) s += w[i].qs[j] * x[i].qs[j];
        sum += static_cast<float>(s) * fp16_to_fp32(w[i].d) * fp16_to_fp32(x[i].d);
    }
    return sum;
#endif
}

float dot_f16_f32(const uint16_t* w, const float* x, int64_t n) noexcept {
    int64_t i = 0;
    float sum = 0.0f;
#if defined(LIYAB_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const float16x8_t h = vld1q_f16(reinterpret_cast<const __fp16*>(w + i));
        acc0 = vfmaq_f32(acc0, vcvt_f32_f16(vget_low_f16(h)), vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, vcvt_f32_f16(vget_high_f16(h)), vld1q_f32(x + i + 4));
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
    for (; i < n; ++i) sum += fp16_to_fp32(w[i]) * x[i];
    return sum;
}

float dot_f32(const float* a, const float* b, int64_t n) noexcept {
    int64_t i = 0;
    float sum = 0.0f;
#if defined(LIYAB_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    sum = vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
#endif
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

void axpy_row(DType type, const void* row, float a, float* y, int64_t n) noexcept {
    switch (type) {
        case DType::F32: {
            const auto* r = static_cast<const float*>(row);
            for (int64_t i = 0; i < n; ++i) y[i] += a * r[i];
            break;
        }
        case DType::F16: {
            const auto* r = static_cast<const uint16_t*>(row);
            for (int64_t i = 0; i < n; ++i) y[i] += a * fp16_to_fp32(r[i]);
            break;
        }
        case DType::Q8_0:
        case DType::Q4_0:
        case DType::Q4_1: {
            const size_t block_bytes = dtype_row_bytes(type, kBlock);
            const auto* bytes = static_cast<const uint8_t*>(row);
            float tmp[kBlock];
            for (int64_t i = 0; i < n / kBlock; ++i) {
                dequantize_row(type, bytes + i * block_bytes, tmp, kBlock);
                float* yb = y + i * kBlock;
                for (int j = 0; j < kBlock; ++j) yb[j] += a * tmp[j];
            }
            break;
        }
    }
}

}  // namespace liyab::quant
