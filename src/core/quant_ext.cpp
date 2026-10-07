// Liyab — extended GGML formats: Q2_K, Q3_K, the I-quants (IQ1_S/M,
// IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS), ternary TQ1_0/TQ2_0, MXFP4, NVFP4 and
// BF16.
//
// Each quantized block is decoded into a common form,
//     x[e] = scale16[e / 16] * q[e] + bias8[e / 8]     (q: int8)
// so a single integer path (SDOT against Q8_0 activations) serves every
// format. The decoders are ports of the reference implementations in
// llama.cpp's gguf-py (quants.py); tests/data/quant_vectors.bin holds blocks
// and expected values produced by that reference, checked bit-exactly by
// tests/test_engine.cpp. Lookup grids come from tools/gen_quant_tables.py.
#include <cmath>
#include <cstring>

#include "core/quant.h"

#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define LIYAB_EXT_NEON 1
#endif

namespace liyab::quant {

namespace {

#include "core/quant_tables.inc"

inline float half_at(const uint8_t* p) noexcept {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return fp16_to_fp32(h);
}
inline uint16_t u16_at(const uint8_t* p) noexcept {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint32_t u32_at(const uint8_t* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// Applies an 8-bit sign mask to 8 unsigned grid magnitudes.
inline void signed8(const uint8_t* grid, uint8_t signs, int8_t* out) noexcept {
    for (int i = 0; i < 8; ++i) {
        const int v = grid[i];
        out[i] = static_cast<int8_t>((signs >> i) & 1 ? -v : v);
    }
}

float e8m0_to_fp32_half(uint8_t e) noexcept {
    const uint32_t bits = e < 2 ? (0x00200000u << e) : (static_cast<uint32_t>(e - 1) << 23);
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float ue4m3_to_fp32_half(uint8_t x) noexcept {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const float man = static_cast<float>(x & 7);
    const float raw = exp == 0 ? man * std::ldexp(1.0f, -9) : (1.0f + man / 8.0f) * std::ldexp(1.0f, exp - 7);
    return raw * 0.5f;
}

}  // namespace

bool is_extended(DType type) noexcept {
    switch (type) {
        case DType::Q2_K: case DType::Q3_K: case DType::IQ2_XXS: case DType::IQ2_XS: case DType::IQ2_S:
        case DType::IQ3_XXS: case DType::IQ3_S: case DType::IQ1_S: case DType::IQ1_M: case DType::IQ4_NL:
        case DType::IQ4_XS: case DType::TQ1_0: case DType::TQ2_0: case DType::MXFP4: case DType::NVFP4:
            return true;
        default:
            return false;
    }
}

bool unpack_ext_block(DType type, const uint8_t* b, ExtBlock& out) noexcept {
    int8_t* q = out.q;
    float* s16 = out.scale16;
    out.has_bias = false;
    auto set32 = [&](int ib, float s) { s16[2 * ib] = s16[2 * ib + 1] = s; };
    switch (type) {
        case DType::Q2_K: {  // scales[16] qs[64] d dmin
            const float d = half_at(b + 80), dmin = half_at(b + 82);
            for (int e = 0; e < 256; ++e) {
                q[e] = static_cast<int8_t>((b[16 + 32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3);
            }
            for (int j = 0; j < 16; ++j) {
                s16[j] = d * static_cast<float>(b[j] & 0xF);
                out.bias8[2 * j] = out.bias8[2 * j + 1] = -dmin * static_cast<float>(b[j] >> 4);
            }
            out.has_bias = true;
            return true;
        }
        case DType::Q3_K: {  // hmask[32] qs[64] scales[12] d
            const uint8_t* hmask = b;
            const uint8_t* qs = b + 32;
            const uint8_t* sc = b + 96;
            const float d = half_at(b + 108);
            for (int j = 0; j < 16; ++j) {
                const int lo = j < 8 ? (sc[j] & 0xF) : (sc[j - 8] >> 4);
                const int hi = (sc[8 + j % 4] >> (2 * (j / 4))) & 3;
                s16[j] = d * static_cast<float>((lo | (hi << 4)) - 32);
            }
            for (int e = 0; e < 256; ++e) {
                const int ql = (qs[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3;
                const int h = (hmask[e % 32] >> (e / 32)) & 1;
                q[e] = static_cast<int8_t>(ql - (h ? 0 : 4));
            }
            return true;
        }
        case DType::IQ2_XXS: {  // d qs[64] (uint32 pairs per 32 values)
            const float d = half_at(b);
            for (int ib = 0; ib < 8; ++ib) {
                const uint32_t aux0 = u32_at(b + 2 + 8 * ib);
                const uint32_t aux1 = u32_at(b + 2 + 8 * ib + 4);
                set32(ib, d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f);
                for (int k = 0; k < 4; ++k) {
                    signed8(kIq2xxsGrid[(aux0 >> (8 * k)) & 0xFF], kKSigns[(aux1 >> (7 * k)) & 0x7F], q + 32 * ib + 8 * k);
                }
            }
            return true;
        }
        case DType::IQ2_XS: {  // d qs[32 x u16] scales[8]
            const float d = half_at(b);
            for (int g = 0; g < 32; ++g) {
                const uint16_t v = u16_at(b + 2 + 2 * g);
                signed8(kIq2xsGrid[v & 511], kKSigns[v >> 9], q + 8 * g);
            }
            for (int j = 0; j < 16; ++j) {
                s16[j] = d * (0.5f + static_cast<float>((b[66 + j / 2] >> (4 * (j % 2))) & 0xF)) * 0.25f;
            }
            return true;
        }
        case DType::IQ2_S: {  // d qs[32] signs[32] qh[8] scales[8]
            const float d = half_at(b);
            const uint8_t* qs = b + 2;
            const uint8_t* signs = b + 34;
            const uint8_t* qh = b + 66;
            const uint8_t* sc = b + 74;
            for (int g = 0; g < 32; ++g) {
                const int idx = qs[g] | (((qh[g / 4] >> (2 * (g % 4))) & 3) << 8);
                signed8(kIq2sGrid[idx], signs[g], q + 8 * g);
            }
            for (int j = 0; j < 16; ++j) {
                s16[j] = d * (0.5f + static_cast<float>((sc[j / 2] >> (4 * (j % 2))) & 0xF)) * 0.25f;
            }
            return true;
        }
        case DType::IQ3_XXS: {  // d qs[64] scales_and_signs[8 x u32]
            const float d = half_at(b);
            const uint8_t* qs = b + 2;
            for (int ib = 0; ib < 8; ++ib) {
                const uint32_t aux = u32_at(b + 66 + 4 * ib);
                set32(ib, d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f);
                for (int k = 0; k < 4; ++k) {
                    uint8_t mag[8];
                    std::memcpy(mag, kIq3xxsGrid[qs[8 * ib + 2 * k]], 4);
                    std::memcpy(mag + 4, kIq3xxsGrid[qs[8 * ib + 2 * k + 1]], 4);
                    signed8(mag, kKSigns[(aux >> (7 * k)) & 0x7F], q + 32 * ib + 8 * k);
                }
            }
            return true;
        }
        case DType::IQ3_S: {  // d qs[64] qh[8] signs[32] scales[4]
            const float d = half_at(b);
            const uint8_t* qs = b + 2;
            const uint8_t* qh = b + 66;
            const uint8_t* signs = b + 74;
            const uint8_t* sc = b + 106;
            for (int g = 0; g < 32; ++g) {  // 8 values = two 4-value grid rows
                uint8_t mag[8];
                for (int h = 0; h < 2; ++h) {
                    const int k = 2 * g + h;
                    const int idx = qs[k] | (((qh[k / 8] >> (k % 8)) & 1) << 8);
                    std::memcpy(mag + 4 * h, kIq3sGrid[idx], 4);
                }
                signed8(mag, signs[g], q + 8 * g);
            }
            for (int ib = 0; ib < 8; ++ib) {
                set32(ib, d * (1.0f + 2.0f * static_cast<float>((sc[ib / 2] >> (4 * (ib % 2))) & 0xF)));
            }
            return true;
        }
        case DType::IQ1_S: {  // d qs[32] qh[8 x u16]
            const float d = half_at(b);
            for (int ib = 0; ib < 8; ++ib) {
                const uint16_t h = u16_at(b + 34 + 2 * ib);
                const float dl = d * static_cast<float>(2 * ((h >> 12) & 7) + 1);
                const float delta = (h & 0x8000) ? -0.125f : 0.125f;
                set32(ib, dl);
                for (int k = 0; k < 4; ++k) {
                    const int idx = b[2 + 4 * ib + k] | (((h >> (3 * k)) & 7) << 8);
                    std::memcpy(q + 32 * ib + 8 * k, kIq1sGrid[idx], 8);
                    out.bias8[4 * ib + k] = dl * delta;
                }
            }
            out.has_bias = true;
            return true;
        }
        case DType::IQ1_M: {  // qs[32] qh[16] scales[4 x u16] (fp16 d spread over the top nibbles)
            const uint8_t* qs = b;
            const uint8_t* qh = b + 32;
            uint16_t sc[4];
            for (int i = 0; i < 4; ++i) sc[i] = u16_at(b + 48 + 2 * i);
            const uint16_t dh = static_cast<uint16_t>(((sc[0] & 0xF000) >> 12) | ((sc[1] & 0xF000) >> 8) |
                                                      ((sc[2] & 0xF000) >> 4) | (sc[3] & 0xF000));
            const float d = fp16_to_fp32(dh);
            for (int j = 0; j < 16; ++j) {
                s16[j] = d * static_cast<float>(2 * ((sc[j / 4] >> (3 * (j % 4))) & 7) + 1);
            }
            for (int g = 0; g < 32; ++g) {
                const int v = (qh[g / 2] >> (4 * (g % 2))) & 0xF;
                std::memcpy(q + 8 * g, kIq1sGrid[qs[g] | ((v & 7) << 8)], 8);
                out.bias8[g] = s16[g / 2] * ((v & 8) ? -0.125f : 0.125f);
            }
            out.has_bias = true;
            return true;
        }
        case DType::IQ4_NL: {  // d qs[16], 32 values
            set32(0, half_at(b));
            for (int j = 0; j < 16; ++j) {
                q[j] = kIq4nlValues[b[2 + j] & 0xF];
                q[j + 16] = kIq4nlValues[b[2 + j] >> 4];
            }
            return true;
        }
        case DType::IQ4_XS: {  // d scales_h(u16) scales_l[4] qs[128]
            const float d = half_at(b);
            const uint16_t sh = u16_at(b + 2);
            for (int ib = 0; ib < 8; ++ib) {
                const int lo = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF;
                const int hi = (sh >> (2 * ib)) & 3;
                set32(ib, d * static_cast<float>((lo | (hi << 4)) - 32));
                const uint8_t* qs = b + 8 + 16 * ib;
                for (int j = 0; j < 16; ++j) {
                    q[32 * ib + j] = kIq4nlValues[qs[j] & 0xF];
                    q[32 * ib + j + 16] = kIq4nlValues[qs[j] >> 4];
                }
            }
            return true;
        }
        case DType::TQ1_0: {  // qs[48] qh[4] d: base-3 digits, 5 (4 for qh) per byte
            static constexpr uint8_t kPow3[5] = {1, 3, 9, 27, 81};
            auto digit = [](uint8_t byte, uint8_t mul) {
                const uint8_t v = static_cast<uint8_t>(byte * mul);  // wraps like uint8 numpy arithmetic
                return static_cast<int8_t>(((static_cast<uint16_t>(v) * 3) >> 8) - 1);
            };
            for (int m = 0; m < 5; ++m) {
                for (int l = 0; l < 32; ++l) q[32 * m + l] = digit(b[l], kPow3[m]);
                for (int l = 0; l < 16; ++l) q[160 + 16 * m + l] = digit(b[32 + l], kPow3[m]);
            }
            for (int m = 0; m < 4; ++m) {
                for (int l = 0; l < 4; ++l) q[240 + 4 * m + l] = digit(b[48 + l], kPow3[m]);
            }
            const float d = half_at(b + 52);
            for (int j = 0; j < 16; ++j) s16[j] = d;
            return true;
        }
        case DType::TQ2_0: {  // qs[64] d
            for (int e = 0; e < 256; ++e) {
                q[e] = static_cast<int8_t>(((b[32 * (e / 128) + e % 32] >> (2 * ((e % 128) / 32))) & 3) - 1);
            }
            const float d = half_at(b + 64);
            for (int j = 0; j < 16; ++j) s16[j] = d;
            return true;
        }
        case DType::MXFP4: {  // e8m0 exponent, qs[16], 32 values (doubled e2m1)
            set32(0, e8m0_to_fp32_half(b[0]));
            for (int j = 0; j < 16; ++j) {
                q[j] = kMxfp4Values[b[1 + j] & 0xF];
                q[j + 16] = kMxfp4Values[b[1 + j] >> 4];
            }
            return true;
        }
        case DType::NVFP4: {  // 4 x ue4m3 scales, qs[32], 64 values
            for (int s = 0; s < 4; ++s) {
                s16[s] = ue4m3_to_fp32_half(b[s]);
                const uint8_t* qs = b + 4 + 8 * s;
                for (int i = 0; i < 8; ++i) {
                    q[16 * s + i] = kMxfp4Values[qs[i] & 0xF];
                    q[16 * s + 8 + i] = kMxfp4Values[qs[i] >> 4];
                }
            }
            return true;
        }
        default:
            return false;
    }
}

void dequantize_ext_row(DType type, const void* src, float* y, int64_t n) noexcept {
    const DTypeTraits t = dtype_traits(type);
    ExtBlock blk;
    for (int64_t i = 0; i < n / t.block_size; ++i) {
        unpack_ext_block(type, static_cast<const uint8_t*>(src) + i * t.block_bytes, blk);
        float* yb = y + i * t.block_size;
        for (int e = 0; e < t.block_size; ++e) {
            yb[e] = blk.scale16[e / 16] * static_cast<float>(blk.q[e]) + (blk.has_bias ? blk.bias8[e / 8] : 0.0f);
        }
    }
}

namespace {

inline int32_t dot16_i8(const int8_t* a, const int8_t* b) noexcept {
#if defined(LIYAB_EXT_NEON)
#if defined(__ARM_FEATURE_DOTPROD)
    return vaddvq_s32(vdotq_s32(vdupq_n_s32(0), vld1q_s8(a), vld1q_s8(b)));
#else
    const int8x16_t va = vld1q_s8(a), vb = vld1q_s8(b);
    const int16x8_t lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
    const int16x8_t hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));
    return vaddvq_s32(vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)));
#endif
#else
    int32_t s = 0;
    for (int i = 0; i < 16; ++i) s += a[i] * b[i];
    return s;
#endif
}

inline int32_t sum8_i8(const int8_t* a) noexcept {
    int32_t s = 0;
    for (int i = 0; i < 8; ++i) s += a[i];
    return s;
}

}  // namespace

float dot_ext_q8_0(DType type, const void* row, const BlockQ8_0* x, int64_t n) noexcept {
    const DTypeTraits t = dtype_traits(type);
    const int chunks = t.block_size / 16;
    ExtBlock blk;
    float sum = 0.0f;
    for (int64_t i = 0; i < n / t.block_size; ++i) {
        unpack_ext_block(type, static_cast<const uint8_t*>(row) + i * t.block_bytes, blk);
        const int64_t base = i * t.block_size;
        for (int c = 0; c < chunks; ++c) {
            const int64_t e = base + 16 * c;
            const BlockQ8_0& xb = x[e / 32];
            const int8_t* xq = xb.qs + (e % 32);
            float acc = blk.scale16[c] * static_cast<float>(dot16_i8(blk.q + 16 * c, xq));
            if (blk.has_bias) {
                acc += blk.bias8[2 * c] * static_cast<float>(sum8_i8(xq)) +
                       blk.bias8[2 * c + 1] * static_cast<float>(sum8_i8(xq + 8));
            }
            sum += fp16_to_fp32(xb.d) * acc;
        }
    }
    return sum;
}

float bf16_to_fp32(uint16_t h) noexcept {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

float dot_bf16_f32(const uint16_t* w, const float* x, int64_t n) noexcept {
    int64_t i = 0;
    float sum = 0.0f;
#if defined(LIYAB_EXT_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f), acc1 = vdupq_n_f32(0.0f);
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t h = vld1q_u16(w + i);
        const float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
        const float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
        acc0 = vfmaq_f32(acc0, lo, vld1q_f32(x + i));
        acc1 = vfmaq_f32(acc1, hi, vld1q_f32(x + i + 4));
    }
    sum = vaddvq_f32(vaddq_f32(acc0, acc1));
#endif
    for (; i < n; ++i) sum += bf16_to_fp32(w[i]) * x[i];
    return sum;
}

}  // namespace liyab::quant
