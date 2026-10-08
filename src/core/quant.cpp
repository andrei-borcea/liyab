#include "core/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>

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
        case DType::Q4_K:
            quantize_row_q4_K(x, static_cast<BlockQ4_K*>(dst), n);
            break;
        case DType::Q5_K:
            quantize_row_q5_K(x, static_cast<BlockQ5_K*>(dst), n);
            break;
        default:
            break;  // weight-only formats are never produced at runtime; callers do not request them
    }
}

namespace {

// GGML get_scale_min_k4: 6-bit scale and min of sub-block j (0..7).
inline void scale_min_k4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) noexcept {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// Rounds to nearest with llama.cpp's float trick (nearest_int).
inline int nearest_int(float f) noexcept {
    float v = f + 12582912.0f;
    int i;
    std::memcpy(&i, &v, sizeof i);
    return (i & 0x007fffff) - 0x00400000;
}

// llama.cpp make_qkx2_quants: levels L in [0, nmax] and (scale, min) for one
// 32-value sub-block, minimizing the weighted squared error over a small grid
// of scales around the plain min/max fit. Returns the scale; *the_min is the
// (non-negative) value subtracted before scaling.
float make_qkx2_quants(int n, int nmax, const float* x, const float* weights, uint8_t* L, float* the_min,
                       uint8_t* Laux, float rmin, float rdelta, int nstep) noexcept {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        min = std::min(min, x[i]);
        max = std::max(max, x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.0f;
    }
    float iscale = static_cast<float>(nmax) / (max - min);
    float scale = 1 / iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        const int l = nearest_int(iscale * (x[i] - min));
        L[i] = static_cast<uint8_t>(std::max(0, std::min(nmax, l)));
        const float diff = scale * L[i] + min - x[i];
        best_error += weights[i] * (diff * diff);  // llama.cpp's evaluation order
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * static_cast<float>(is) + static_cast<float>(nmax)) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            const int l = std::max(0, std::min(nmax, nearest_int(iscale * (x[i] - min))));
            Laux[i] = static_cast<uint8_t>(l);
            const float w = weights[i];
            sum_l += w * static_cast<float>(l);
            sum_l2 += w * static_cast<float>(l) * static_cast<float>(l);
            sum_xl += w * static_cast<float>(l) * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                const float diff = this_scale * Laux[i] + this_min - x[i];
                cur_error += weights[i] * (diff * diff);
            }
            if (cur_error < best_error) {
                std::copy_n(Laux, n, L);
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// The part Q4_K and Q5_K share: per 256-value super-block, the 8 sub-block
// scales and mins (6-bit, packed into `scales`, under fp16 d / dmin) and the
// levels L in [0, nmax] requantized against those rounded scales.
void quantize_k_super_block(const float* x, int nmax, float rmin, int nstep, uint16_t& d_out, uint16_t& dmin_out,
                            uint8_t* packed, uint8_t* L) noexcept {
    uint8_t Laux[32];
    float weights[32];
    float mins[kSuperBlock / 32];
    float scales[kSuperBlock / 32];
    float max_scale = 0;  // the min is subtracted, so scales are never negative
    float max_min = 0;
    for (int j = 0; j < kSuperBlock / 32; ++j) {
        float sum_x2 = 0;
        for (int l = 0; l < 32; ++l) sum_x2 += x[32 * j + l] * x[32 * j + l];
        const float av_x = std::sqrt(sum_x2 / 32);
        for (int l = 0; l < 32; ++l) weights[l] = av_x + std::fabs(x[32 * j + l]);
        scales[j] = make_qkx2_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, rmin, 0.1f, nstep);
        max_scale = std::max(max_scale, scales[j]);
        max_min = std::max(max_min, mins[j]);
    }
    const float inv_scale = max_scale > 0 ? 63.0f / max_scale : 0.0f;
    const float inv_min = max_min > 0 ? 63.0f / max_min : 0.0f;
    std::fill_n(packed, 12, uint8_t{0});
    for (int j = 0; j < kSuperBlock / 32; ++j) {
        const auto ls = static_cast<uint8_t>(std::min(63, nearest_int(inv_scale * scales[j])));
        const auto lm = static_cast<uint8_t>(std::min(63, nearest_int(inv_min * mins[j])));
        if (j < 4) {
            packed[j] = ls;
            packed[j + 4] = lm;
        } else {
            packed[j + 4] = static_cast<uint8_t>((ls & 0xF) | ((lm & 0xF) << 4));
            packed[j - 4] |= static_cast<uint8_t>((ls >> 4) << 6);
            packed[j] |= static_cast<uint8_t>((lm >> 4) << 6);
        }
    }
    d_out = fp32_to_fp16(max_scale / 63.0f);
    dmin_out = fp32_to_fp16(max_min / 63.0f);
    for (int j = 0; j < kSuperBlock / 32; ++j) {
        uint8_t sc, m;
        scale_min_k4(j, packed, sc, m);
        const float d = fp16_to_fp32(d_out) * sc;
        if (d == 0.0f) continue;
        const float dm = fp16_to_fp32(dmin_out) * m;
        for (int i = 0; i < 32; ++i) {
            L[32 * j + i] = static_cast<uint8_t>(std::max(0, std::min(nmax, nearest_int((x[32 * j + i] + dm) / d))));
        }
    }
}

}  // namespace

void quantize_row_q4_K(const float* x, BlockQ4_K* y, int64_t n) noexcept {
    uint8_t L[kSuperBlock];
    for (int64_t b = 0; b < n / kSuperBlock; ++b, x += kSuperBlock) {
        BlockQ4_K& blk = y[b];
        quantize_k_super_block(x, 15, -1.0f, 20, blk.d, blk.dmin, blk.scales, L);
        uint8_t* q = blk.qs;
        for (int j = 0; j < kSuperBlock; j += 64, q += 32) {
            for (int l = 0; l < 32; ++l) q[l] = static_cast<uint8_t>(L[j + l] | (L[j + l + 32] << 4));
        }
    }
}

void quantize_row_q5_K(const float* x, BlockQ5_K* y, int64_t n) noexcept {
    uint8_t L[kSuperBlock];
    for (int64_t b = 0; b < n / kSuperBlock; ++b, x += kSuperBlock) {
        BlockQ5_K& blk = y[b];
        quantize_k_super_block(x, 31, -0.5f, 15, blk.d, blk.dmin, blk.scales, L);
        std::fill_n(blk.qh, 32, uint8_t{0});
        uint8_t* ql = blk.qs;
        uint8_t m1 = 1, m2 = 2;
        for (int c = 0; c < kSuperBlock; c += 64, ql += 32) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[c + j];
                int l2 = L[c + j + 32];
                if (l1 > 15) {
                    l1 -= 16;
                    blk.qh[j] |= m1;
                }
                if (l2 > 15) {
                    l2 -= 16;
                    blk.qh[j] |= m2;
                }
                ql[j] = static_cast<uint8_t>(l1 | (l2 << 4));
            }
            m1 = static_cast<uint8_t>(m1 << 2);
            m2 = static_cast<uint8_t>(m2 << 2);
        }
    }
}

void unpack_q5_block(const uint8_t* qh_bytes, const uint8_t* qs, bool symmetric, int8_t* q) noexcept {
    uint32_t qh;
    std::memcpy(&qh, qh_bytes, 4);
    const int offset = symmetric ? 16 : 0;
    for (int j = 0; j < kBlock / 2; ++j) {
        q[j] = static_cast<int8_t>(((qs[j] & 0x0F) | (((qh >> j) & 1u) << 4)) - offset);
        q[j + 16] = static_cast<int8_t>(((qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - offset);
    }
}

void unpack_k_block(DType type, const void* block, int8_t* q, float* scale16, float* min32) noexcept {
    switch (type) {
        case DType::Q4_K: {
            const auto* b = static_cast<const BlockQ4_K*>(block);
            const float d = fp16_to_fp32(b->d);
            const float dmin = fp16_to_fp32(b->dmin);
            for (int c = 0; c < 4; ++c) {
                const uint8_t* qs = b->qs + 32 * c;
                for (int half = 0; half < 2; ++half) {
                    const int j = 2 * c + half;
                    uint8_t sc, m;
                    scale_min_k4(j, b->scales, sc, m);
                    scale16[2 * j] = scale16[2 * j + 1] = d * sc;
                    min32[j] = dmin * m;
                    for (int l = 0; l < 32; ++l) {
                        q[32 * j + l] = static_cast<int8_t>(half ? (qs[l] >> 4) : (qs[l] & 0x0F));
                    }
                }
            }
            break;
        }
        case DType::Q5_K: {
            const auto* b = static_cast<const BlockQ5_K*>(block);
            const float d = fp16_to_fp32(b->d);
            const float dmin = fp16_to_fp32(b->dmin);
            for (int c = 0; c < 4; ++c) {
                const uint8_t* qs = b->qs + 32 * c;
                for (int half = 0; half < 2; ++half) {
                    const int j = 2 * c + half;
                    uint8_t sc, m;
                    scale_min_k4(j, b->scales, sc, m);
                    scale16[2 * j] = scale16[2 * j + 1] = d * sc;
                    min32[j] = dmin * m;
                    const uint8_t bit = static_cast<uint8_t>(1u << j);  // u1 = 1 << 2c, u2 = 2 << 2c
                    for (int l = 0; l < 32; ++l) {
                        const int lo = half ? (qs[l] >> 4) : (qs[l] & 0x0F);
                        q[32 * j + l] = static_cast<int8_t>(lo + ((b->qh[l] & bit) ? 16 : 0));
                    }
                }
            }
            break;
        }
        case DType::Q6_K: {
            const auto* b = static_cast<const BlockQ6_K*>(block);
            const float d = fp16_to_fp32(b->d);
            for (int h = 0; h < 2; ++h) {  // two halves of 128 values
                const uint8_t* ql = b->ql + 64 * h;
                const uint8_t* qh = b->qh + 32 * h;
                int8_t* out = q + 128 * h;
                for (int l = 0; l < 32; ++l) {
                    out[l] = static_cast<int8_t>(((ql[l] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32);
                    out[l + 32] = static_cast<int8_t>(((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    out[l + 64] = static_cast<int8_t>(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    out[l + 96] = static_cast<int8_t>(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                }
                for (int i = 0; i < 8; ++i) scale16[8 * h + i] = d * b->scales[8 * h + i];
            }
            for (int j = 0; j < 8; ++j) min32[j] = 0.0f;
            break;
        }
        default:
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
        case DType::Q5_0:
        case DType::Q5_1: {
            const size_t block_bytes = dtype_row_bytes(type, kBlock);
            int8_t q[kBlock];
            for (int64_t i = 0; i < n / kBlock; ++i) {
                const uint8_t* blk = static_cast<const uint8_t*>(src) + i * block_bytes;
                uint16_t dh, mh = 0;
                std::memcpy(&dh, blk, 2);
                if (type == DType::Q5_1) std::memcpy(&mh, blk + 2, 2);
                const uint8_t* qh = blk + (type == DType::Q5_1 ? 4 : 2);
                unpack_q5_block(qh, qh + 4, type == DType::Q5_0, q);
                const float d = fp16_to_fp32(dh);
                const float m = type == DType::Q5_1 ? fp16_to_fp32(mh) : 0.0f;
                for (int j = 0; j < kBlock; ++j) y[i * kBlock + j] = d * q[j] + m;
            }
            break;
        }
        case DType::BF16: {
            const auto* h = static_cast<const uint16_t*>(src);
            for (int64_t i = 0; i < n; ++i) y[i] = bf16_to_fp32(h[i]);
            break;
        }
        case DType::Q2_K: case DType::Q3_K: case DType::IQ2_XXS: case DType::IQ2_XS: case DType::IQ2_S:
        case DType::IQ3_XXS: case DType::IQ3_S: case DType::IQ1_S: case DType::IQ1_M: case DType::IQ4_NL:
        case DType::IQ4_XS: case DType::TQ1_0: case DType::TQ2_0: case DType::MXFP4: case DType::NVFP4:
            dequantize_ext_row(type, src, y, n);
            break;
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K: {
            const size_t block_bytes = dtype_row_bytes(type, kSuperBlock);
            int8_t q[kSuperBlock];
            float scale16[16];
            float min32[8];
            for (int64_t i = 0; i < n / kSuperBlock; ++i) {
                unpack_k_block(type, static_cast<const uint8_t*>(src) + i * block_bytes, q, scale16, min32);
                float* yb = y + i * kSuperBlock;
                for (int k = 0; k < kSuperBlock; ++k) yb[k] = scale16[k / 16] * q[k] - min32[k / 32];
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
        case DType::Q4_K_R8:
            // Repacked layouts store 8 rows together: no single row to decode.
            // Only the CPU backend's matmuls read them; this path is never taken.
            std::fill_n(y, n, 0.0f);
            break;
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

void block_sums(const BlockQ8_0* x, int32_t* xsums, int64_t n) noexcept {
    for (int64_t i = 0; i < n / kBlock; ++i) {
        int32_t sum = 0;
        for (int j = 0; j < kBlock; ++j) sum += x[i].qs[j];
        xsums[i] = sum;
    }
}

namespace {

// Sum of 16 int8 products.
inline int32_t dot16(const int8_t* a, const int8_t* b) noexcept {
#if defined(LIYAB_NEON)
    return vaddvq_s32(dot_i8x16(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b)));
#else
    int32_t s = 0;
    for (int i = 0; i < 16; ++i) s += a[i] * b[i];
    return s;
#endif
}

#if defined(LIYAB_NEON)
// --- NEON kernels: bits are extracted in registers and fed to SDOT ---------

inline int32_t hsum(int32x4_t v) noexcept { return vaddvq_s32(v); }

// 32 int8 weights (two vectors) · one Q8_0 activation block.
inline int32_t dot32(int8x16_t w0, int8x16_t w1, const BlockQ8_0& x) noexcept {
    return hsum(dot_i8x16(dot_i8x16(vdupq_n_s32(0), w0, vld1q_s8(x.qs)), w1, vld1q_s8(x.qs + 16)));
}

// Two 16-value halves of one Q8_0 block, kept apart (Q6_K has a scale per 16).
inline void dot16x2(int8x16_t w0, int8x16_t w1, const BlockQ8_0& x, int32_t& a, int32_t& b) noexcept {
    a = hsum(dot_i8x16(vdupq_n_s32(0), w0, vld1q_s8(x.qs)));
    b = hsum(dot_i8x16(vdupq_n_s32(0), w1, vld1q_s8(x.qs + 16)));
}

float dot_q4_k_neon(const BlockQ4_K* w, const BlockQ8_0* x, const int32_t* xs, int64_t n) noexcept {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    float sum = 0.0f;
    for (int64_t i = 0; i < n / kSuperBlock; ++i) {
        const BlockQ4_K& b = w[i];
        const float d = fp16_to_fp32(b.d);
        const float dmin = fp16_to_fp32(b.dmin);
        const BlockQ8_0* xb = x + i * 8;
        const int32_t* s = xs + i * 8;
        for (int c = 0; c < 4; ++c) {  // 64 values: low nibbles = sub-block 2c, high = 2c+1
            const uint8x16_t q0 = vld1q_u8(b.qs + 32 * c);
            const uint8x16_t q1 = vld1q_u8(b.qs + 32 * c + 16);
            const int32_t lo = dot32(vreinterpretq_s8_u8(vandq_u8(q0, mask)), vreinterpretq_s8_u8(vandq_u8(q1, mask)),
                                     xb[2 * c]);
            const int32_t hi = dot32(vreinterpretq_s8_u8(vshrq_n_u8(q0, 4)), vreinterpretq_s8_u8(vshrq_n_u8(q1, 4)),
                                     xb[2 * c + 1]);
            uint8_t sc0, m0, sc1, m1;
            scale_min_k4(2 * c, b.scales, sc0, m0);
            scale_min_k4(2 * c + 1, b.scales, sc1, m1);
            sum += fp16_to_fp32(xb[2 * c].d) * (d * sc0 * static_cast<float>(lo) - dmin * m0 * static_cast<float>(s[2 * c]));
            sum += fp16_to_fp32(xb[2 * c + 1].d) *
                   (d * sc1 * static_cast<float>(hi) - dmin * m1 * static_cast<float>(s[2 * c + 1]));
        }
    }
    return sum;
}

float dot_q5_k_neon(const BlockQ5_K* w, const BlockQ8_0* x, const int32_t* xs, int64_t n) noexcept {
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const uint8x16_t sixteen = vdupq_n_u8(16);
    float sum = 0.0f;
    for (int64_t i = 0; i < n / kSuperBlock; ++i) {
        const BlockQ5_K& b = w[i];
        const float d = fp16_to_fp32(b.d);
        const float dmin = fp16_to_fp32(b.dmin);
        const BlockQ8_0* xb = x + i * 8;
        const int32_t* s = xs + i * 8;
        const uint8x16_t h0 = vld1q_u8(b.qh);
        const uint8x16_t h1 = vld1q_u8(b.qh + 16);
        for (int c = 0; c < 4; ++c) {
            const uint8x16_t q0 = vld1q_u8(b.qs + 32 * c);
            const uint8x16_t q1 = vld1q_u8(b.qs + 32 * c + 16);
            const uint8x16_t bit_lo = vdupq_n_u8(static_cast<uint8_t>(1u << (2 * c)));
            const uint8x16_t bit_hi = vdupq_n_u8(static_cast<uint8_t>(2u << (2 * c)));
            // 5th bit set -> +16 (vtst gives 0xFF lanes, masked to 16).
            const uint8x16_t l0 = vorrq_u8(vandq_u8(q0, mask), vandq_u8(vtstq_u8(h0, bit_lo), sixteen));
            const uint8x16_t l1 = vorrq_u8(vandq_u8(q1, mask), vandq_u8(vtstq_u8(h1, bit_lo), sixteen));
            const uint8x16_t u0 = vorrq_u8(vshrq_n_u8(q0, 4), vandq_u8(vtstq_u8(h0, bit_hi), sixteen));
            const uint8x16_t u1 = vorrq_u8(vshrq_n_u8(q1, 4), vandq_u8(vtstq_u8(h1, bit_hi), sixteen));
            const int32_t lo = dot32(vreinterpretq_s8_u8(l0), vreinterpretq_s8_u8(l1), xb[2 * c]);
            const int32_t hi = dot32(vreinterpretq_s8_u8(u0), vreinterpretq_s8_u8(u1), xb[2 * c + 1]);
            uint8_t sc0, m0, sc1, m1;
            scale_min_k4(2 * c, b.scales, sc0, m0);
            scale_min_k4(2 * c + 1, b.scales, sc1, m1);
            sum += fp16_to_fp32(xb[2 * c].d) * (d * sc0 * static_cast<float>(lo) - dmin * m0 * static_cast<float>(s[2 * c]));
            sum += fp16_to_fp32(xb[2 * c + 1].d) *
                   (d * sc1 * static_cast<float>(hi) - dmin * m1 * static_cast<float>(s[2 * c + 1]));
        }
    }
    return sum;
}

float dot_q6_k_neon(const BlockQ6_K* w, const BlockQ8_0* x, int64_t n) noexcept {
    const uint8x16_t mask4 = vdupq_n_u8(0x0F);
    const uint8x16_t mask2 = vdupq_n_u8(0x03);
    const int8x16_t m32 = vdupq_n_s8(32);
    float sum = 0.0f;
    for (int64_t i = 0; i < n / kSuperBlock; ++i) {
        const BlockQ6_K& b = w[i];
        const float d = fp16_to_fp32(b.d);
        for (int h = 0; h < 2; ++h) {  // 128 values = 4 activation blocks
            const uint8_t* ql = b.ql + 64 * h;
            const uint8_t* qh = b.qh + 32 * h;
            const int8_t* sc = b.scales + 8 * h;
            const BlockQ8_0* xb = x + i * 8 + 4 * h;
            const uint8x16_t ql0 = vld1q_u8(ql), ql1 = vld1q_u8(ql + 16), ql2 = vld1q_u8(ql + 32), ql3 = vld1q_u8(ql + 48);
            const uint8x16_t qh0 = vld1q_u8(qh), qh1 = vld1q_u8(qh + 16);
            auto make = [&](uint8x16_t low, uint8x16_t high_bits) {
                return vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(low, vshlq_n_u8(high_bits, 4))), m32);
            };
            // Runs of 32 values, each with two per-16 scales.
            const int8x16_t r[8] = {
                make(vandq_u8(ql0, mask4), vandq_u8(qh0, mask2)),
                make(vandq_u8(ql1, mask4), vandq_u8(qh1, mask2)),
                make(vandq_u8(ql2, mask4), vandq_u8(vshrq_n_u8(qh0, 2), mask2)),
                make(vandq_u8(ql3, mask4), vandq_u8(vshrq_n_u8(qh1, 2), mask2)),
                make(vshrq_n_u8(ql0, 4), vandq_u8(vshrq_n_u8(qh0, 4), mask2)),
                make(vshrq_n_u8(ql1, 4), vandq_u8(vshrq_n_u8(qh1, 4), mask2)),
                make(vshrq_n_u8(ql2, 4), vshrq_n_u8(qh0, 6)),
                make(vshrq_n_u8(ql3, 4), vshrq_n_u8(qh1, 6)),
            };
            for (int k = 0; k < 4; ++k) {
                int32_t a, c;
                dot16x2(r[2 * k], r[2 * k + 1], xb[k], a, c);
                sum += fp16_to_fp32(xb[k].d) * d * (static_cast<float>(sc[2 * k]) * a + static_cast<float>(sc[2 * k + 1]) * c);
            }
        }
    }
    return sum;
}

// 32 qh bits -> two byte vectors with 0x10 where the 5th bit is set.
inline void q5_high_bits(const uint8_t* qh_bytes, uint8x16_t& lo, uint8x16_t& hi) noexcept {
    static const uint8_t kSelLo[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    static const uint8_t kSelHi[16] = {2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    static const uint8_t kBits[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint32_t qh;
    std::memcpy(&qh, qh_bytes, 4);
    const uint8x16_t bytes = vreinterpretq_u8_u32(vdupq_n_u32(qh));
    const uint8x16_t bits = vld1q_u8(kBits);
    const uint8x16_t sixteen = vdupq_n_u8(16);
    lo = vandq_u8(vtstq_u8(vqtbl1q_u8(bytes, vld1q_u8(kSelLo)), bits), sixteen);
    hi = vandq_u8(vtstq_u8(vqtbl1q_u8(bytes, vld1q_u8(kSelHi)), bits), sixteen);
}
#endif

// K-quant row · Q8_0 activations, portable path: per 32-value activation block b,
//   dx * (scale16[2b] * q·x[0:16] + scale16[2b+1] * q·x[16:32] - min32[b] * sum(x)).
float dot_k_q8_0(DType type, const void* row, const BlockQ8_0* x, const int32_t* xsums, int64_t n) noexcept {
#if defined(LIYAB_NEON)
    switch (type) {
        case DType::Q4_K: return dot_q4_k_neon(static_cast<const BlockQ4_K*>(row), x, xsums, n);
        case DType::Q5_K: return dot_q5_k_neon(static_cast<const BlockQ5_K*>(row), x, xsums, n);
        case DType::Q6_K: return dot_q6_k_neon(static_cast<const BlockQ6_K*>(row), x, n);
        default: break;
    }
#endif
    const size_t block_bytes = dtype_row_bytes(type, kSuperBlock);
    alignas(16) int8_t q[kSuperBlock];
    float scale16[16];
    float min32[8];
    float sum = 0.0f;
    for (int64_t i = 0; i < n / kSuperBlock; ++i) {
        unpack_k_block(type, static_cast<const uint8_t*>(row) + i * block_bytes, q, scale16, min32);
        const BlockQ8_0* xb = x + i * 8;
        const int32_t* xs = xsums + i * 8;
        for (int b = 0; b < 8; ++b) {
            const float acc = scale16[2 * b] * static_cast<float>(dot16(q + 32 * b, xb[b].qs)) +
                              scale16[2 * b + 1] * static_cast<float>(dot16(q + 32 * b + 16, xb[b].qs + 16)) -
                              min32[b] * static_cast<float>(xs[b]);
            sum += fp16_to_fp32(xb[b].d) * acc;
        }
    }
    return sum;
}

// Q5_0 / Q5_1 row · Q8_0 activations (Q5_1 also needs the block sums for its min).
float dot_q5_q8_0(DType type, const void* row, const BlockQ8_0* x, const int32_t* xsums, int64_t n) noexcept {
    const bool symmetric = type == DType::Q5_0;
    const size_t block_bytes = symmetric ? sizeof(BlockQ5_0) : sizeof(BlockQ5_1);
    float sum = 0.0f;
#if defined(LIYAB_NEON)
    const uint8x16_t mask = vdupq_n_u8(0x0F);
    const int8x16_t offset = vdupq_n_s8(symmetric ? 16 : 0);
    for (int64_t i = 0; i < n / kBlock; ++i) {
        const uint8_t* blk = static_cast<const uint8_t*>(row) + i * block_bytes;
        uint16_t dh, mh = 0;
        std::memcpy(&dh, blk, 2);
        if (!symmetric) std::memcpy(&mh, blk + 2, 2);
        const uint8_t* qh = blk + (symmetric ? 2 : 4);
        uint8x16_t hlo, hhi;
        q5_high_bits(qh, hlo, hhi);
        const uint8x16_t qs = vld1q_u8(qh + 4);
        const int8x16_t w0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(qs, mask), hlo)), offset);
        const int8x16_t w1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(qs, 4), hhi)), offset);
        const float dot = static_cast<float>(dot32(w0, w1, x[i]));
        sum += fp16_to_fp32(x[i].d) *
               (fp16_to_fp32(dh) * dot + (symmetric ? 0.0f : fp16_to_fp32(mh) * static_cast<float>(xsums[i])));
    }
#else
    alignas(16) int8_t q[kBlock];
    for (int64_t i = 0; i < n / kBlock; ++i) {
        const uint8_t* blk = static_cast<const uint8_t*>(row) + i * block_bytes;
        uint16_t dh, mh = 0;
        std::memcpy(&dh, blk, 2);
        if (!symmetric) std::memcpy(&mh, blk + 2, 2);
        const uint8_t* qh = blk + (symmetric ? 2 : 4);
        unpack_q5_block(qh, qh + 4, symmetric, q);
        const float dot = static_cast<float>(dot16(q, x[i].qs) + dot16(q + 16, x[i].qs + 16));
        sum += fp16_to_fp32(x[i].d) *
               (fp16_to_fp32(dh) * dot + (symmetric ? 0.0f : fp16_to_fp32(mh) * static_cast<float>(xsums[i])));
    }
#endif
    return sum;
}

}  // namespace

float dot_quantized(DType type, const void* row, const BlockQ8_0* x, const int32_t* xsums, int64_t n) noexcept {
    switch (type) {
        case DType::Q4_0: return dot_q4_0_q8_0(static_cast<const BlockQ4_0*>(row), x, n);
        case DType::Q4_1: return dot_q4_1_q8_0(static_cast<const BlockQ4_1*>(row), x, n);
        case DType::Q8_0: return dot_q8_0_q8_0(static_cast<const BlockQ8_0*>(row), x, n);
        case DType::Q5_0:
        case DType::Q5_1: return dot_q5_q8_0(type, row, x, xsums, n);
        case DType::Q4_K:
        case DType::Q5_K:
        case DType::Q6_K: return dot_k_q8_0(type, row, x, xsums, n);
        case DType::IQ4_NL: return dot_iq4_nl_q8_0(row, x, n);
        case DType::MXFP4: return dot_mxfp4_q8_0(row, x, n);
        default: return is_extended(type) ? dot_ext_q8_0(type, row, x, n) : 0.0f;
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

#if defined(LIYAB_NEON)
namespace {

// dot_q8_0_q8_0 for NR activation rows: each weight block is loaded and its
// scale converted once.
template <int NR>
void dot_q8_0_rows_neon(const BlockQ8_0* w, const BlockQ8_0* const* x, int64_t nb, float* out) noexcept {
    float32x4_t acc[NR];
    for (int r = 0; r < NR; ++r) acc[r] = vdupq_n_f32(0.0f);
    for (int64_t i = 0; i < nb; ++i) {
        const int8x16_t lo = vld1q_s8(w[i].qs);
        const int8x16_t hi = vld1q_s8(w[i].qs + 16);
        const float dw = fp16_to_fp32(w[i].d);
        for (int r = 0; r < NR; ++r) {
            const int32x4_t s =
                dot_i8x16(dot_i8x16(vdupq_n_s32(0), lo, vld1q_s8(x[r][i].qs)), hi, vld1q_s8(x[r][i].qs + 16));
            acc[r] = vmlaq_n_f32(acc[r], vcvtq_f32_s32(s), dw * fp16_to_fp32(x[r][i].d));
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = vaddvq_f32(acc[r]);
}

}  // namespace
#endif

void dot_q8_0_q8_0_rows(const BlockQ8_0* w, const BlockQ8_0* const* x, int32_t rows, int64_t n, float* out) noexcept {
    int32_t r = 0;
#if defined(LIYAB_NEON)
    const int64_t nb = n / kBlock;
    for (; r + 4 <= rows; r += 4) dot_q8_0_rows_neon<4>(w, x + r, nb, out + r);
    switch (rows - r) {
        case 3: dot_q8_0_rows_neon<3>(w, x + r, nb, out + r); return;
        case 2: dot_q8_0_rows_neon<2>(w, x + r, nb, out + r); return;
        default: break;
    }
#endif
    for (; r < rows; ++r) out[r] = dot_q8_0_q8_0(w, x[r], n);
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
        default:
            break;  // KV caches never use weight-only formats
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
