// Liyab — fast dot kernels for the low-bit GGML formats: Q2_K, Q3_K, TQ2_0,
// the 256-value I-quants (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
// IQ3_S, IQ4_XS) against Q8_K activations, and IQ4_NL / MXFP4 against Q8_0.
//
// The AArch64 NEON kernels are adapted from llama.cpp (ggml),
// ggml/src/ggml-cpu/arch/arm/quants.c (ggml_vec_dot_*_q8_K and friends):
//
//     Copyright (c) 2023-2026 The ggml authors
//     Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
//
// Changes from the original: C++ types and casts, Liyab's block structs and
// lookup tables (core/quant_tables.h; the +-1 sign table is derived from
// kKSigns at compile time instead of being spelled out), unaligned-safe loads,
// no SVE paths, plus throughput tweaks measured on Apple M4 (single core):
//  * packed index bytes are read with one 64-bit load and split with integer
//    shifts instead of one byte load each (the kernels are load-port bound);
//  * IQ1_S, IQ1_M, IQ2_S and IQ3_S assemble all grid indices of a super-block
//    with a few vector ops into a small stack array first;
//  * per-sub-block scales (0.5 + s) are applied in integer as 2s + 1 with the
//    1/2 folded into the final factor, accumulated in int32 vectors.
// The integer products are the same as llama.cpp's, so results agree with its
// CPU backend up to float rounding.
//
// Why a separate activation format: these weights pack 1.5-3.5 bits per value
// behind lattice grids and sign tables, so decoding dominates. Against Q8_K
// (one float scale per 256 activations) a whole super-block accumulates in
// int32 with one float multiply at the end, and sub-block minimums (Q2_K),
// TQ2_0 offsets and IQ1_S deltas come from the precomputed 16-value activation
// sums. The generic path in quant_ext.cpp (decode to an ExtBlock, then dot
// against Q8_0) remains the reference and the fallback without NEON.
#include <cmath>
#include <cstring>

#include "core/quant.h"
#include "core/quant_tables.h"

#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#define LIYAB_LOWBIT_NEON 1
#endif

namespace liyab::quant {

namespace {

// --- GGML block layouts (ggml-common.h). Rows of these blocks are read
// straight from the mapped GGUF; every block size is even, so the uint16
// members are always 2-byte aligned. ---------------------------------------
struct BlockQ2K {
    uint8_t scales[16];  // low nibble: scale, high nibble: min, per 16 values
    uint8_t qs[64];
    uint16_t d, dmin;
};
struct BlockQ3K {
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    uint16_t d;
};
struct BlockTQ2_0 {
    uint8_t qs[64];
    uint16_t d;
};
struct BlockIq2Xxs {
    uint16_t d;
    uint16_t qs[32];
};
struct BlockIq2Xs {
    uint16_t d;
    uint16_t qs[32];
    uint8_t scales[8];
};
struct BlockIq2S {
    uint16_t d;
    uint8_t qs[64];  // 32 grid indices (low 8 bits), then 32 sign bytes
    uint8_t qh[8];
    uint8_t scales[8];
};
struct BlockIq3Xxs {
    uint16_t d;
    uint8_t qs[96];  // 64 grid indices, then 8 x uint32 scales-and-signs
};
struct BlockIq3S {
    uint16_t d;
    uint8_t qs[64];
    uint8_t qh[8];
    uint8_t signs[32];
    uint8_t scales[4];
};
struct BlockIq1S {
    uint16_t d;
    uint8_t qs[32];
    uint16_t qh[8];
};
struct BlockIq1M {
    uint8_t qs[32];
    uint8_t qh[16];
    uint8_t scales[8];  // 4 x uint16; the fp16 super-block scale is spread over their top nibbles
};
struct BlockIq4Nl {
    uint16_t d;
    uint8_t qs[16];
};
struct BlockIq4Xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct BlockMxfp4 {
    uint8_t e;
    uint8_t qs[16];
};
static_assert(sizeof(BlockQ2K) == 84 && sizeof(BlockQ3K) == 110 && sizeof(BlockTQ2_0) == 66);
static_assert(sizeof(BlockIq2Xxs) == 66 && sizeof(BlockIq2Xs) == 74 && sizeof(BlockIq2S) == 82);
static_assert(sizeof(BlockIq3Xxs) == 98 && sizeof(BlockIq3S) == 110);
static_assert(sizeof(BlockIq1S) == 50 && sizeof(BlockIq1M) == 56);
static_assert(sizeof(BlockIq4Nl) == 18 && sizeof(BlockIq4Xs) == 136 && sizeof(BlockMxfp4) == 17);

// Generic (decode-based) Q8_K dot used where the NEON kernels are unavailable.
float dot_lowbit_generic(DType type, const void* row, const BlockQ8_K* x, int64_t n) noexcept {
    const DTypeTraits t = dtype_traits(type);
    ExtBlock blk;
    float sum = 0.0f;
    for (int64_t i = 0; i < n / kSuperBlock; ++i) {
        if (!unpack_ext_block(type, static_cast<const uint8_t*>(row) + i * t.block_bytes, blk)) return 0.0f;
        const int8_t* xq = x[i].qs;
        float acc = 0.0f;
        for (int c = 0; c < 16; ++c) {
            int32_t isum = 0, s0 = 0, s1 = 0;
            for (int e = 0; e < 16; ++e) isum += blk.q[16 * c + e] * xq[16 * c + e];
            acc += blk.scale16[c] * static_cast<float>(isum);
            if (blk.has_bias) {
                for (int e = 0; e < 8; ++e) {
                    s0 += xq[16 * c + e];
                    s1 += xq[16 * c + 8 + e];
                }
                acc += blk.bias8[2 * c] * static_cast<float>(s0) + blk.bias8[2 * c + 1] * static_cast<float>(s1);
            }
        }
        sum += x[i].d * acc;
    }
    return sum;
}

#if defined(LIYAB_LOWBIT_NEON)

// --- NEON helpers -------------------------------------------------------------

constexpr float kIq1Delta = 0.125f;  // IQ1S_DELTA / IQ1M_DELTA

inline uint32_t load_u32(const void* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// acc + four-lane dot of 16 int8 pairs. Without the dot-product extension the
// emulation yields the same total but a different per-lane grouping, which is
// fine: every caller reduces the lanes (or pairs of them) before scaling.
inline int32x4_t vdot(int32x4_t acc, int8x16_t a, int8x16_t b) noexcept {
#if defined(__ARM_FEATURE_DOTPROD)
    return vdotq_s32(acc, a, b);
#else
    const int16x8_t p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
#endif
}

// fp16 -> fp32 inline (the out-of-line fp16_to_fp32 is a call per super-block).
inline float h2f(uint16_t h) noexcept {
    __fp16 v;
    std::memcpy(&v, &h, sizeof v);
    return static_cast<float>(v);
}

// Loads packed index bytes into a general register and hides the value from
// the optimizer. Without the empty asm, LLVM folds every `(v >> 8k) & 0xff`
// back into a separate byte load; these kernels are bound by load-port
// throughput (grid and sign lookups), so extracting fields with integer ALU
// shifts instead is measurably faster.
inline uint64_t load_u64_opaque(const void* p) noexcept {
    uint64_t v;
    std::memcpy(&v, p, sizeof v);
    __asm__("" : "+r"(v));
    return v;
}
inline uint32_t load_u32_opaque(const void* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    __asm__("" : "+r"(v));
    return v;
}
inline uint32_t byte_at(uint64_t v, int k) noexcept { return static_cast<uint32_t>(v >> (8 * k)) & 0xff; }
inline uint32_t u16_at(uint64_t v, int k) noexcept { return static_cast<uint32_t>(v >> (16 * k)) & 0xffff; }

// sum_k scale[k] * (lanes of part[k]) for 8 / 16 partial dot vectors. Keeping
// one int32x4 per sub-block and scaling once per super-block avoids a
// horizontal add and a scalar multiply per sub-block.
inline int32_t scaled_sum8(const int32x4_t* part, uint8x8_t scales) noexcept {
    const int32x4_t t0 = vpaddq_s32(vpaddq_s32(part[0], part[1]), vpaddq_s32(part[2], part[3]));
    const int32x4_t t1 = vpaddq_s32(vpaddq_s32(part[4], part[5]), vpaddq_s32(part[6], part[7]));
    const int16x8_t s = vreinterpretq_s16_u16(vmovl_u8(scales));
    int32x4_t acc = vmulq_s32(t0, vmovl_s16(vget_low_s16(s)));
    acc = vmlaq_s32(acc, t1, vmovl_s16(vget_high_s16(s)));
    return vaddvq_s32(acc);
}
inline int32_t scaled_sum16(const int32x4_t* part, uint8x16_t scales) noexcept {
    return scaled_sum8(part, vget_low_u8(scales)) + scaled_sum8(part + 8, vget_high_u8(scales));
}

// Two 8-value grid rows -> one vector (grids of unsigned magnitudes < 128).
inline int8x16_t grid8x2(const uint8_t (*grid)[8], uint32_t i0, uint32_t i1) noexcept {
    return vcombine_s8(vld1_s8(reinterpret_cast<const int8_t*>(grid[i0])),
                       vld1_s8(reinterpret_cast<const int8_t*>(grid[i1])));
}
inline int8x16_t grid8x2s(const int8_t (*grid)[8], uint32_t i0, uint32_t i1) noexcept {
    return vcombine_s8(vld1_s8(grid[i0]), vld1_s8(grid[i1]));
}
// Four 4-value grid rows -> one vector, composed in general registers (two
// 64-bit inserts) rather than with four lane loads.
inline uint8x16_t grid4x4(const uint8_t (*grid)[4], uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3) noexcept {
    const uint64_t lo = load_u32(grid[i0]) | (static_cast<uint64_t>(load_u32(grid[i1])) << 32);
    const uint64_t hi = load_u32(grid[i2]) | (static_cast<uint64_t>(load_u32(grid[i3])) << 32);
    return vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)));
}

// kEvenSigns[k] = the 8 signs (+1 / -1) of 7-bit sign index k: bit j of
// kKSigns[k] set means element j is negative (the 8th bit is the parity bit).
// llama.cpp spells this table out as keven_signs_q2xs.
struct EvenSigns {
    alignas(16) int8_t v[128][8];
};
constexpr EvenSigns make_even_signs() {
    EvenSigns s{};
    for (int k = 0; k < 128; ++k) {
        for (int j = 0; j < 8; ++j) s.v[k][j] = ((kKSigns[k] >> j) & 1) ? -1 : 1;
    }
    return s;
}
constexpr EvenSigns kEvenSigns = make_even_signs();

inline int8x16_t signs8x2(uint32_t k0, uint32_t k1) noexcept {
    return vcombine_s8(vld1_s8(kEvenSigns.v[k0]), vld1_s8(kEvenSigns.v[k1]));
}

// Turns 32 sign bits (one per value, LSB first) into two vectors of +1 / -1.
// Used by IQ2_S and IQ3_S, whose signs are stored explicitly.
struct SignExpander {
    uint8x16x2_t mask1;
    uint8x16_t mask2;
    uint8x16_t one;
    SignExpander() noexcept {
        static const uint8_t k_mask1[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
        static const uint8_t k_mask2[16] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
                                            0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80};
        mask1 = vld1q_u8_x2(k_mask1);
        mask2 = vld1q_u8(k_mask2);
        one = vdupq_n_u8(1);
    }
    void expand(uint32_t bits, int8x16_t& lo, int8x16_t& hi) const noexcept {
        const uint8x16_t v = vreinterpretq_u8_u32(vdupq_n_u32(bits));
        const uint8x16_t b0 = vandq_u8(vqtbl1q_u8(v, mask1.val[0]), mask2);
        const uint8x16_t b1 = vandq_u8(vqtbl1q_u8(v, mask1.val[1]), mask2);
        // bit set -> 0xFF | 1 = -1, bit clear -> 0 | 1 = +1
        lo = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(b0, mask2), one));
        hi = vreinterpretq_s8_u8(vorrq_u8(vceqq_u8(b1, mask2), one));
    }
};

// --- Kernels ------------------------------------------------------------------

float dot_q2_k(const BlockQ2K* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const uint8x16_t m3 = vdupq_n_u8(0x3);
    const uint8x16_t m4 = vdupq_n_u8(0xF);
    const int32x4_t vzero = vdupq_n_s32(0);
    alignas(16) uint8_t aux[16];
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);
        const uint8_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;

        const uint8x16_t mins_and_scales = vld1q_u8(x[i].scales);
        vst1q_u8(aux, vandq_u8(mins_and_scales, m4));
        // Minimums: sum over 16-value groups of min * (activation sum), from bsums.
        const uint8x16_t mins = vshrq_n_u8(mins_and_scales, 4);
        const int16x8x2_t q8sums = vld1q_s16_x2(y[i].bsums);
        const int16x8_t mins0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(mins)));
        const int16x8_t mins1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(mins)));
        const int32x4_t s0 = vaddq_s32(vmull_s16(vget_low_s16(mins0), vget_low_s16(q8sums.val[0])),
                                       vmull_s16(vget_high_s16(mins0), vget_high_s16(q8sums.val[0])));
        const int32x4_t s1 = vaddq_s32(vmull_s16(vget_low_s16(mins1), vget_low_s16(q8sums.val[1])),
                                       vmull_s16(vget_high_s16(mins1), vget_high_s16(q8sums.val[1])));
        sum += dmin * static_cast<float>(vaddvq_s32(vaddq_s32(s0, s1)));

        int32x4_t acc = vzero;
        int is = 0;
        for (int j = 0; j < 2; ++j) {  // 128 values per 32 bytes of qs
            const uint8x16x2_t q2bits = vld1q_u8_x2(q2);
            q2 += 32;
            for (int shift = 0; shift < 8; shift += 2) {
                const int8x16x2_t q8bytes = vld1q_s8_x2(q8);
                q8 += 32;
                const int8x16_t b0 = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q2bits.val[0], vdupq_n_s8(-shift)), m3));
                const int8x16_t b1 = vreinterpretq_s8_u8(vandq_u8(vshlq_u8(q2bits.val[1], vdupq_n_s8(-shift)), m3));
                acc = vmlaq_n_s32(acc, vdot(vzero, b0, q8bytes.val[0]), aux[is + shift]);
                acc = vmlaq_n_s32(acc, vdot(vzero, b1, q8bytes.val[1]), aux[is + shift + 1]);
            }
            is += 8;
        }
        sum += d * static_cast<float>(vaddvq_s32(acc));
    }
    return sum;
}

float dot_q3_k(const BlockQ3K* x, const BlockQ8_K* y, int64_t nb) noexcept {
    constexpr uint32_t kmask1 = 0x03030303;
    constexpr uint32_t kmask2 = 0x0f0f0f0f;
    const uint8x16_t m3b = vdupq_n_u8(0x3);
    const int32x4_t vzero = vdupq_n_s32(0);
    const uint8x16_t m0 = vdupq_n_u8(1);
    const uint8x16_t m1 = vshlq_n_u8(m0, 1);
    const uint8x16_t m2 = vshlq_n_u8(m0, 2);
    const uint8x16_t m3 = vshlq_n_u8(m0, 3);
    uint32_t aux[3];
    uint32_t utmp[4];
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        const uint8_t* q3 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        uint8x16x2_t qhbits = vld1q_u8_x2(x[i].hmask);

        // 16 six-bit scales: low nibbles from bytes 0-7, high 2 bits from bytes 8-11.
        std::memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4);
        utmp[2] = ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4);
        utmp[1] = (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4);
        utmp[0] = (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4);
        int8_t scale[16];
        std::memcpy(scale, utmp, 16);
        for (int j = 0; j < 16; ++j) scale[j] = static_cast<int8_t>(scale[j] - 32);
        const int8_t* sc = scale;

        int32x4_t acc = vzero;
        for (int j = 0; j < 2; ++j) {
            const uint8x16x2_t q3bits = vld1q_u8_x2(q3);
            q3 += 32;
            const int8x16x4_t q8a = vld1q_s8_x4(q8);
            q8 += 64;
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;

            // value = (2 low bits) - 4 * (high bit clear)
            uint8x16_t h0 = vshlq_n_u8(vbicq_u8(m0, qhbits.val[0]), 2);
            uint8x16_t h1 = vshlq_n_u8(vbicq_u8(m0, qhbits.val[1]), 2);
            uint8x16_t h2 = vshlq_n_u8(vbicq_u8(m1, qhbits.val[0]), 1);
            uint8x16_t h3 = vshlq_n_u8(vbicq_u8(m1, qhbits.val[1]), 1);
            int8x16_t v0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[0], m3b)), vreinterpretq_s8_u8(h0));
            int8x16_t v1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(q3bits.val[1], m3b)), vreinterpretq_s8_u8(h1));
            int8x16_t v2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 2), m3b)), vreinterpretq_s8_u8(h2));
            int8x16_t v3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 2), m3b)), vreinterpretq_s8_u8(h3));
            acc = vmlaq_n_s32(acc, vdot(vzero, v0, q8a.val[0]), sc[0]);
            acc = vmlaq_n_s32(acc, vdot(vzero, v1, q8a.val[1]), sc[1]);
            acc = vmlaq_n_s32(acc, vdot(vzero, v2, q8a.val[2]), sc[2]);
            acc = vmlaq_n_s32(acc, vdot(vzero, v3, q8a.val[3]), sc[3]);
            sc += 4;

            h0 = vbicq_u8(m2, qhbits.val[0]);
            h1 = vbicq_u8(m2, qhbits.val[1]);
            h2 = vshrq_n_u8(vbicq_u8(m3, qhbits.val[0]), 1);
            h3 = vshrq_n_u8(vbicq_u8(m3, qhbits.val[1]), 1);
            v0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 4), m3b)), vreinterpretq_s8_u8(h0));
            v1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 4), m3b)), vreinterpretq_s8_u8(h1));
            v2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[0], 6), m3b)), vreinterpretq_s8_u8(h2));
            v3 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(q3bits.val[1], 6), m3b)), vreinterpretq_s8_u8(h3));
            acc = vmlaq_n_s32(acc, vdot(vzero, v0, q8b.val[0]), sc[0]);
            acc = vmlaq_n_s32(acc, vdot(vzero, v1, q8b.val[1]), sc[1]);
            acc = vmlaq_n_s32(acc, vdot(vzero, v2, q8b.val[2]), sc[2]);
            acc = vmlaq_n_s32(acc, vdot(vzero, v3, q8b.val[3]), sc[3]);
            sc += 4;

            if (j == 0) {  // the second 128 values use high-mask bits 4-7
                qhbits.val[0] = vshrq_n_u8(qhbits.val[0], 4);
                qhbits.val[1] = vshrq_n_u8(qhbits.val[1], 4);
            }
        }
        sum += d * static_cast<float>(vaddvq_s32(acc));
    }
    return sum;
}

float dot_tq2_0(const BlockTQ2_0* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const uint8x16_t m3 = vdupq_n_u8(3);
    float sum = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        int32x4_t sumi0 = vdupq_n_s32(0);
        int32x4_t sumi1 = vdupq_n_s32(0);
        for (int j = 0; j < 64; j += 32) {
            const uint8x16_t qx0 = vld1q_u8(x[i].qs + j);
            const uint8x16_t qx1 = vld1q_u8(x[i].qs + j + 16);
            const int8_t* qy = y[i].qs + 4 * j;
            // Stored values are q + 1 in {0, 1, 2}; the -1 is applied once via bsums below.
            sumi0 = vdot(sumi0, vreinterpretq_s8_u8(vandq_u8(qx0, m3)), vld1q_s8(qy + 0));
            sumi1 = vdot(sumi1, vreinterpretq_s8_u8(vandq_u8(qx1, m3)), vld1q_s8(qy + 16));
            sumi0 = vdot(sumi0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx0, 2), m3)), vld1q_s8(qy + 32));
            sumi1 = vdot(sumi1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx1, 2), m3)), vld1q_s8(qy + 48));
            sumi0 = vdot(sumi0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx0, 4), m3)), vld1q_s8(qy + 64));
            sumi1 = vdot(sumi1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(qx1, 4), m3)), vld1q_s8(qy + 80));
            sumi0 = vdot(sumi0, vreinterpretq_s8_u8(vshrq_n_u8(qx0, 6)), vld1q_s8(qy + 96));
            sumi1 = vdot(sumi1, vreinterpretq_s8_u8(vshrq_n_u8(qx1, 6)), vld1q_s8(qy + 112));
        }
        const int16x8_t ysum0 = vld1q_s16(y[i].bsums);
        const int16x8_t ysum1 = vld1q_s16(y[i].bsums + 8);
        const float d = h2f(x[i].d) * y[i].d;
        sumi0 = vaddq_s32(sumi0, sumi1);
        sumi0 = vsubq_s32(sumi0, vpaddlq_s16(vaddq_s16(ysum0, ysum1)));
        sum += d * static_cast<float>(vaddvq_s32(sumi0));
    }
    return sum;
}

float dot_iq2_xxs(const BlockIq2Xxs* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const auto& g = kIq2xxsGrid;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const uint16_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        int32x4_t acc = vdupq_n_s32(0);
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            // Per 32 values: 4 grid indices (bytes), then a word of 4 x 7-bit
            // sign indices + a 4-bit scale s (multiplier 0.25 * (0.5 + s)).
            const uint64_t lo = load_u64_opaque(q2), hi = load_u64_opaque(q2 + 4);
            q2 += 8;
            const uint32_t s0 = static_cast<uint32_t>(lo >> 32), s1 = static_cast<uint32_t>(hi >> 32);
            int8x16_t u0 = grid8x2(g, byte_at(lo, 0), byte_at(lo, 1));
            int8x16_t u1 = grid8x2(g, byte_at(lo, 2), byte_at(lo, 3));
            int8x16_t u2 = grid8x2(g, byte_at(hi, 0), byte_at(hi, 1));
            int8x16_t u3 = grid8x2(g, byte_at(hi, 2), byte_at(hi, 3));
            u0 = vmulq_s8(u0, signs8x2(s0 & 127, (s0 >> 7) & 127));
            u1 = vmulq_s8(u1, signs8x2((s0 >> 14) & 127, (s0 >> 21) & 127));
            u2 = vmulq_s8(u2, signs8x2(s1 & 127, (s1 >> 7) & 127));
            u3 = vmulq_s8(u3, signs8x2((s1 >> 14) & 127, (s1 >> 21) & 127));
            const int32x4_t p1 = vdot(vdot(vdupq_n_s32(0), u0, q8b.val[0]), u1, q8b.val[1]);
            const int32x4_t p2 = vdot(vdot(vdupq_n_s32(0), u2, q8b.val[2]), u3, q8b.val[3]);
            // (0.5 + s) = (2s + 1) / 2: integer accumulation, the 1/2 goes into the final factor.
            acc = vmlaq_n_s32(acc, p1, static_cast<int32_t>(2 * (s0 >> 28) + 1));
            acc = vmlaq_n_s32(acc, p2, static_cast<int32_t>(2 * (s1 >> 28) + 1));
        }
        sumf += h2f(x[i].d) * y[i].d * static_cast<float>(vaddvq_s32(acc));
    }
    return 0.125f * sumf;
}

float dot_iq2_xs(const BlockIq2Xs* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const auto& g = kIq2xsGrid;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const uint16_t* q2 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        // 16 four-bit scales -> 2s + 1, widened to int32 (four per 64 values).
        const uint8x8_t scales8 = vld1_u8(x[i].scales);
        const uint8x8_t scales_l = vand_u8(scales8, vdup_n_u8(0xf));
        const uint8x8_t scales_h = vshr_n_u8(scales8, 4);
        uint8x16_t scales = vcombine_u8(vzip1_u8(scales_l, scales_h), vzip2_u8(scales_l, scales_h));
        scales = vaddq_u8(vshlq_n_u8(scales, 1), vdupq_n_u8(1));
        const uint16x8_t scales1 = vmovl_u8(vget_low_u8(scales));
        const uint16x8_t scales2 = vmovl_u8(vget_high_u8(scales));
        int32x4_t scales32[4];
        scales32[0] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales1)));
        scales32[1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales1)));
        scales32[2] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(scales2)));
        scales32[3] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(scales2)));
        int32x4_t sumi = vdupq_n_s32(0);
        for (int ib64 = 0; ib64 < 4; ++ib64) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            // Each uint16: 9-bit grid index, 7-bit sign index.
            const uint64_t a = load_u64_opaque(q2), b = load_u64_opaque(q2 + 4);
            q2 += 8;
            auto grid = [](uint64_t v, int k) { return static_cast<uint32_t>(v >> (16 * k)) & 511; };
            auto sign = [](uint64_t v, int k) { return static_cast<uint32_t>(v >> (16 * k + 9)) & 127; };
            int8x16_t u0 = grid8x2(g, grid(a, 0), grid(a, 1));
            int8x16_t u1 = grid8x2(g, grid(a, 2), grid(a, 3));
            int8x16_t u2 = grid8x2(g, grid(b, 0), grid(b, 1));
            int8x16_t u3 = grid8x2(g, grid(b, 2), grid(b, 3));
            u0 = vmulq_s8(u0, signs8x2(sign(a, 0), sign(a, 1)));
            u1 = vmulq_s8(u1, signs8x2(sign(a, 2), sign(a, 3)));
            u2 = vmulq_s8(u2, signs8x2(sign(b, 0), sign(b, 1)));
            u3 = vmulq_s8(u3, signs8x2(sign(b, 2), sign(b, 3)));
            const int32x4_t p1 = vdot(vdupq_n_s32(0), u0, q8b.val[0]);
            const int32x4_t p2 = vdot(vdupq_n_s32(0), u1, q8b.val[1]);
            const int32x4_t p3 = vdot(vdupq_n_s32(0), u2, q8b.val[2]);
            const int32x4_t p4 = vdot(vdupq_n_s32(0), u3, q8b.val[3]);
            // Lane k of p = dot of the k-th 16 values of this 64-value chunk.
            const int32x4_t p = vpaddq_s32(vpaddq_s32(p1, p2), vpaddq_s32(p3, p4));
            sumi = vmlaq_s32(sumi, p, scales32[ib64]);
        }
        sumf += h2f(x[i].d) * y[i].d * static_cast<float>(vaddvq_s32(sumi));
    }
    return 0.125f * sumf;
}

float dot_iq2_s(const BlockIq2S* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const auto& g = kIq2sGrid;
    const SignExpander signs_x;
    // Replicates qh[2j], qh[2j + 1] four times each (as uint16 lanes) and moves
    // the 2 index bits of group 4k + m (bits 2m, 2m + 1) to bits 8-9.
    static const uint8_t k_rep[4][16] = {{0, 255, 0, 255, 0, 255, 0, 255, 1, 255, 1, 255, 1, 255, 1, 255},
                                         {2, 255, 2, 255, 2, 255, 2, 255, 3, 255, 3, 255, 3, 255, 3, 255},
                                         {4, 255, 4, 255, 4, 255, 4, 255, 5, 255, 5, 255, 5, 255, 5, 255},
                                         {6, 255, 6, 255, 6, 255, 6, 255, 7, 255, 7, 255, 7, 255, 7, 255}};
    static const int16_t k_shift[8] = {8, 6, 4, 2, 8, 6, 4, 2};
    const uint8x16x4_t rep = vld1q_u8_x4(&k_rep[0][0]);
    const int16x8_t shift = vld1q_s16(k_shift);
    const uint16x8_t m300 = vdupq_n_u16(0x300);
    const int32x4_t vzero = vdupq_n_s32(0);
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* signs = x[i].qs + 32;
        const int8_t* q8 = y[i].qs;
        // All 32 10-bit grid indices: idx[g] = qs[g] | ((qh[g / 4] >> 2 * (g % 4)) & 3) << 8.
        alignas(16) uint16_t idx[32];
        const uint8x16x2_t qs = vld1q_u8_x2(x[i].qs);
        const uint8x16_t qh = vcombine_u8(vld1_u8(x[i].qh), vdup_n_u8(0));
        const uint8x8_t lo[4] = {vget_low_u8(qs.val[0]), vget_high_u8(qs.val[0]), vget_low_u8(qs.val[1]),
                                 vget_high_u8(qs.val[1])};
        for (int j = 0; j < 4; ++j) {
            const uint16x8_t h = vreinterpretq_u16_u8(vqtbl1q_u8(qh, rep.val[j]));
            vst1q_u16(idx + 8 * j, vorrq_u16(vmovl_u8(lo[j]), vandq_u16(vshlq_u16(h, shift), m300)));
        }
        // 16 four-bit scales (one per 16 values) -> 2s + 1, in value order.
        const uint8x8_t sc8 = vld1_u8(x[i].scales);
        const uint8x16_t sc = vaddq_u8(vshlq_n_u8(vcombine_u8(vzip1_u8(vand_u8(sc8, vdup_n_u8(0xf)), vshr_n_u8(sc8, 4)),
                                                                vzip2_u8(vand_u8(sc8, vdup_n_u8(0xf)), vshr_n_u8(sc8, 4))),
                                                    1),
                                       vdupq_n_u8(1));
        int32x4_t part[16];
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            const uint64_t a = load_u64_opaque(idx + 4 * ib32), b = load_u64_opaque(idx + 4 * ib32 + 4);
            int8x16_t g0 = grid8x2(g, u16_at(a, 0), u16_at(a, 1));
            int8x16_t g1 = grid8x2(g, u16_at(a, 2), u16_at(a, 3));
            int8x16_t g2 = grid8x2(g, u16_at(b, 0), u16_at(b, 1));
            int8x16_t g3 = grid8x2(g, u16_at(b, 2), u16_at(b, 3));
            const uint64_t sg = load_u64_opaque(signs);
            signs += 8;
            int8x16_t s0, s1;
            signs_x.expand(static_cast<uint32_t>(sg), s0, s1);
            g0 = vmulq_s8(s0, g0);
            g1 = vmulq_s8(s1, g1);
            signs_x.expand(static_cast<uint32_t>(sg >> 32), s0, s1);
            g2 = vmulq_s8(s0, g2);
            g3 = vmulq_s8(s1, g3);
            part[2 * ib32 + 0] = vdot(vzero, g0, q8b.val[0]);
            part[2 * ib32 + 1] = vdot(vzero, g1, q8b.val[1]);
            part[2 * ib32 + 2] = vdot(vzero, g2, q8b.val[2]);
            part[2 * ib32 + 3] = vdot(vzero, g3, q8b.val[3]);
        }
        sumf += h2f(x[i].d) * y[i].d * static_cast<float>(scaled_sum16(part, sc));
    }
    return 0.125f * sumf;
}

float dot_iq3_xxs(const BlockIq3Xxs* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const auto& g = kIq3xxsGrid;
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* q3 = x[i].qs;
        const uint8_t* gas = x[i].qs + 64;
        const int8_t* q8 = y[i].qs;
        int32x4_t part[8];
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            // Per 32 values: 8 grid indices (4 values each) and a word of
            // 4 x 7-bit sign indices + 4-bit scale s (multiplier 0.5 * (0.5 + s)).
            const uint64_t lo = load_u64_opaque(q3), hi = load_u64_opaque(q3 + 8);
            q3 += 16;
            const uint64_t ss = load_u64_opaque(gas);
            gas += 8;
            const uint32_t w0 = static_cast<uint32_t>(ss), w1 = static_cast<uint32_t>(ss >> 32);
            const uint8x16_t a0 = grid4x4(g, byte_at(lo, 0), byte_at(lo, 1), byte_at(lo, 2), byte_at(lo, 3));
            const uint8x16_t a1 = grid4x4(g, byte_at(lo, 4), byte_at(lo, 5), byte_at(lo, 6), byte_at(lo, 7));
            const uint8x16_t a2 = grid4x4(g, byte_at(hi, 0), byte_at(hi, 1), byte_at(hi, 2), byte_at(hi, 3));
            const uint8x16_t a3 = grid4x4(g, byte_at(hi, 4), byte_at(hi, 5), byte_at(hi, 6), byte_at(hi, 7));
            const int8x16_t v0 = vmulq_s8(signs8x2(w0 & 127, (w0 >> 7) & 127), vreinterpretq_s8_u8(a0));
            const int8x16_t v1 = vmulq_s8(signs8x2((w0 >> 14) & 127, (w0 >> 21) & 127), vreinterpretq_s8_u8(a1));
            const int8x16_t v2 = vmulq_s8(signs8x2(w1 & 127, (w1 >> 7) & 127), vreinterpretq_s8_u8(a2));
            const int8x16_t v3 = vmulq_s8(signs8x2((w1 >> 14) & 127, (w1 >> 21) & 127), vreinterpretq_s8_u8(a3));
            part[ib32] = vdot(vdot(vdupq_n_s32(0), v0, q8b.val[0]), v1, q8b.val[1]);
            part[ib32 + 1] = vdot(vdot(vdupq_n_s32(0), v2, q8b.val[2]), v3, q8b.val[3]);
        }
        // (0.5 + s) = (2s + 1) / 2 per 32 values; s = top 4 bits of the 8 words.
        const uint32x4x2_t w = vld1q_u32_x2(reinterpret_cast<const uint32_t*>(x[i].qs + 64));
        const uint16x8_t s = vcombine_u16(vmovn_u32(vshrq_n_u32(w.val[0], 28)), vmovn_u32(vshrq_n_u32(w.val[1], 28)));
        const uint8x8_t s8 = vadd_u8(vshl_n_u8(vmovn_u16(s), 1), vdup_n_u8(1));
        sumf += h2f(x[i].d) * y[i].d * static_cast<float>(scaled_sum8(part, s8));
    }
    return 0.25f * sumf;
}

float dot_iq3_s(const BlockIq3S* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const auto& g = kIq3sGrid;
    const SignExpander signs_x;
    static const int16_t k_shift[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    const int16x8_t hshift = vld1q_s16(k_shift);
    const uint16x8_t m256 = vdupq_n_u16(256);
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const uint8_t* signs = x[i].signs;
        const int8_t* q8 = y[i].qs;
        // All 64 9-bit grid indices: idx[k] = qs[k] | ((qh[k / 8] >> (k % 8)) & 1) << 8.
        alignas(16) uint16_t idx[64];
        const uint8x16x4_t qs = vld1q_u8_x4(x[i].qs);
        for (int j = 0; j < 8; ++j) {
            const uint8x16_t q = qs.val[j / 2];
            const uint8x8_t lo = j % 2 ? vget_high_u8(q) : vget_low_u8(q);
            vst1q_u16(idx + 8 * j, vorrq_u16(vmovl_u8(lo), vandq_u16(vshlq_u16(vdupq_n_u16(x[i].qh[j]), hshift), m256)));
        }
        // Odd scales 2s + 1, one 4-bit s per 32 values.
        const uint32_t sc = load_u32(x[i].scales);
        int32x4_t part[8];
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            const uint16_t* ix = idx + 8 * ib32;
            const uint64_t i0 = load_u64_opaque(ix), i1 = load_u64_opaque(ix + 4), i2 = load_u64_opaque(ix + 8),
                           i3 = load_u64_opaque(ix + 12);
            const uint8x16_t a0 = grid4x4(g, u16_at(i0, 0), u16_at(i0, 1), u16_at(i0, 2), u16_at(i0, 3));
            const uint8x16_t a1 = grid4x4(g, u16_at(i1, 0), u16_at(i1, 1), u16_at(i1, 2), u16_at(i1, 3));
            const uint8x16_t a2 = grid4x4(g, u16_at(i2, 0), u16_at(i2, 1), u16_at(i2, 2), u16_at(i2, 3));
            const uint8x16_t a3 = grid4x4(g, u16_at(i3, 0), u16_at(i3, 1), u16_at(i3, 2), u16_at(i3, 3));
            const uint64_t sg = load_u64_opaque(signs);
            signs += 8;
            int8x16_t s0, s1;
            signs_x.expand(static_cast<uint32_t>(sg), s0, s1);
            const int8x16_t v0 = vmulq_s8(s0, vreinterpretq_s8_u8(a0));
            const int8x16_t v1 = vmulq_s8(s1, vreinterpretq_s8_u8(a1));
            signs_x.expand(static_cast<uint32_t>(sg >> 32), s0, s1);
            const int8x16_t v2 = vmulq_s8(s0, vreinterpretq_s8_u8(a2));
            const int8x16_t v3 = vmulq_s8(s1, vreinterpretq_s8_u8(a3));
            part[ib32] = vdot(vdot(vdupq_n_s32(0), v0, q8b.val[0]), v1, q8b.val[1]);
            part[ib32 + 1] = vdot(vdot(vdupq_n_s32(0), v2, q8b.val[2]), v3, q8b.val[3]);
        }
        // Scale of block k: nibble k of the 4 scale bytes (low nibble = even block).
        const uint8x8_t sb = vcreate_u8(sc);
        const uint8x8_t s8 = vadd_u8(vshl_n_u8(vzip1_u8(vand_u8(sb, vdup_n_u8(0xf)), vshr_n_u8(sb, 4)), 1), vdup_n_u8(1));
        sumf += h2f(x[i].d) * y[i].d * static_cast<float>(scaled_sum8(part, s8));
    }
    return sumf;
}

float dot_iq1_s(const BlockIq1S* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const auto& g = kIq1sGrid;
    // Replicates qh[2j], qh[2j + 1] four times each (as uint16 lanes) and
    // shifts the 3 index bits of group 4k + m (bits 3m..3m+2) to bits 8-10.
    static const uint8_t k_rep[4][16] = {
        {0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3},
        {4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7},
        {8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11},
        {12, 13, 12, 13, 12, 13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15}};
    static const int16_t k_shift[8] = {8, 5, 2, -1, 8, 5, 2, -1};
    const int16x8_t shift = vld1q_s16(k_shift);
    const uint16x8_t m700 = vdupq_n_u16(0x700);
    const uint8x16x4_t rep = vld1q_u8_x4(&k_rep[0][0]);
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const int8_t* q8 = y[i].qs;
        // All 32 grid indices of the super-block at once:
        // idx[g] = qs[g] | ((qh[g / 4] >> 3 * (g % 4)) & 7) << 8.
        const uint8x16x2_t qs = vld1q_u8_x2(x[i].qs);
        const uint8x16_t qh = vreinterpretq_u8_u16(vld1q_u16(x[i].qh));
        alignas(16) uint16_t idx[32];
        const uint8x16_t lo[4] = {qs.val[0], vextq_u8(qs.val[0], qs.val[0], 8), qs.val[1],
                                  vextq_u8(qs.val[1], qs.val[1], 8)};
        for (int j = 0; j < 4; ++j) {
            const uint16x8_t h = vreinterpretq_u16_u8(vqtbl1q_u8(qh, rep.val[j]));
            vst1q_u16(idx + 8 * j, vorrq_u16(vmovl_u8(vget_low_u8(lo[j])), vandq_u16(vshlq_u16(h, shift), m700)));
        }
        // Per-32 scales 2s + 1 and the delta term: sum_ib (+-ls_ib) * (activation sum of block ib).
        const int16x8_t qh16 = vreinterpretq_s16_u8(qh);
        const int16x8_t ls = vaddq_s16(vshlq_n_s16(vandq_s16(vshrq_n_s16(qh16, 12), vdupq_n_s16(7)), 1), vdupq_n_s16(1));
        const int16x8_t neg = vshrq_n_s16(qh16, 15);  // all ones where bit 15 (negative delta) is set
        const int16x8_t sls = vsubq_s16(veorq_s16(ls, neg), neg);
        const int16x8x2_t bs = vld1q_s16_x2(y[i].bsums);
        const int16x8_t bsum32 = vpaddq_s16(bs.val[0], bs.val[1]);  // 8 sums of 32 activations
        int32x4_t delta = vmull_s16(vget_low_s16(bsum32), vget_low_s16(sls));
        delta = vmlal_s16(delta, vget_high_s16(bsum32), vget_high_s16(sls));

        int32x4_t part[8];
        for (int ib = 0; ib < 8; ib += 2) {
            const uint64_t a = load_u64_opaque(idx + 4 * ib), b = load_u64_opaque(idx + 4 * ib + 4);
            auto at = [](uint64_t v, int k) { return static_cast<uint32_t>(v >> (16 * k)) & 0xffff; };
            const int8x16_t b0 = grid8x2s(g, at(a, 0), at(a, 1));
            const int8x16_t b1 = grid8x2s(g, at(a, 2), at(a, 3));
            const int8x16_t b2 = grid8x2s(g, at(b, 0), at(b, 1));
            const int8x16_t b3 = grid8x2s(g, at(b, 2), at(b, 3));
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            part[ib] = vdot(vdot(vdupq_n_s32(0), b0, q8b.val[0]), b1, q8b.val[1]);
            part[ib + 1] = vdot(vdot(vdupq_n_s32(0), b2, q8b.val[2]), b3, q8b.val[3]);
        }
        // Lane k of t0 / t1 = dot of 32-value block k / 4 + k.
        const int32x4_t t0 = vpaddq_s32(vpaddq_s32(part[0], part[1]), vpaddq_s32(part[2], part[3]));
        const int32x4_t t1 = vpaddq_s32(vpaddq_s32(part[4], part[5]), vpaddq_s32(part[6], part[7]));
        int32x4_t acc = vmulq_s32(t0, vmovl_s16(vget_low_s16(ls)));
        acc = vmlaq_s32(acc, t1, vmovl_s16(vget_high_s16(ls)));
        sumf += y[i].d * h2f(x[i].d) *
                (static_cast<float>(vaddvq_s32(acc)) + kIq1Delta * static_cast<float>(vaddvq_s32(delta)));
    }
    return sumf;
}

float dot_iq1_m(const BlockIq1M* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const auto& g = kIq1sGrid;
    // Replicates qh[4j .. 4j + 3] twice each (as uint16 lanes) and moves the
    // 3 index bits of group 2k + m (bits 4m .. 4m + 2) to bits 8-10.
    static const uint8_t k_rep[4][16] = {{0, 255, 0, 255, 1, 255, 1, 255, 2, 255, 2, 255, 3, 255, 3, 255},
                                         {4, 255, 4, 255, 5, 255, 5, 255, 6, 255, 6, 255, 7, 255, 7, 255},
                                         {8, 255, 8, 255, 9, 255, 9, 255, 10, 255, 10, 255, 11, 255, 11, 255},
                                         {12, 255, 12, 255, 13, 255, 13, 255, 14, 255, 14, 255, 15, 255, 15, 255}};
    static const int16_t k_shift[8] = {8, 4, 8, 4, 8, 4, 8, 4};
    const uint8x16x4_t rep = vld1q_u8_x4(&k_rep[0][0]);
    const int16x8_t shift = vld1q_s16(k_shift);
    const uint16x8_t m700 = vdupq_n_u16(0x700);
    const int32x4_t mask = vdupq_n_s32(0x7);
    const int32x4_t mone = vdupq_n_s32(1);
    const int32x4_t mzero = vdupq_n_s32(0);
    const int32x4_t scale_shift = {0, -3, -6, -9};
    // Delta signs per pair of 8-value groups, indexed by (bit 3, bit 7) of a qh byte.
    int8x16_t deltas[4];
    deltas[0] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(+1));
    deltas[1] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(+1));
    deltas[2] = vcombine_s8(vdup_n_s8(+1), vdup_n_s8(-1));
    deltas[3] = vcombine_s8(vdup_n_s8(-1), vdup_n_s8(-1));
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const int8_t* q8 = y[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint64_t sc = load_u64_opaque(x[i].scales);
        const uint16_t dh = static_cast<uint16_t>(((sc >> 12) & 0x000f) | ((sc >> 24) & 0x00f0) |
                                                  ((sc >> 36) & 0x0f00) | ((sc >> 48) & 0xf000));
        // All 32 11-bit grid indices: idx[g] = qs[g] | ((qh[g / 2] >> 4 * (g % 2)) & 7) << 8.
        alignas(16) uint16_t idx[32];
        const uint8x16x2_t qs = vld1q_u8_x2(x[i].qs);
        const uint8x16_t qhv = vld1q_u8(qh);
        const uint8x8_t lo[4] = {vget_low_u8(qs.val[0]), vget_high_u8(qs.val[0]), vget_low_u8(qs.val[1]),
                                 vget_high_u8(qs.val[1])};
        for (int j = 0; j < 4; ++j) {
            const uint16x8_t h = vreinterpretq_u16_u8(vqtbl1q_u8(qhv, rep.val[j]));
            vst1q_u16(idx + 8 * j, vorrq_u16(vmovl_u8(lo[j]), vandq_u16(vshlq_u16(h, shift), m700)));
        }
        int32x4_t sumi1 = mzero;
        int32x4_t sumi2 = mzero;
        for (int ib = 0; ib < 8; ib += 2) {
            const uint16_t* ix = idx + 4 * ib;
            const uint64_t a = load_u64_opaque(ix), b = load_u64_opaque(ix + 4);
            const int8x16_t b0 = grid8x2s(g, u16_at(a, 0), u16_at(a, 1));
            const int8x16_t b1 = grid8x2s(g, u16_at(a, 2), u16_at(a, 3));
            const int8x16_t b2 = grid8x2s(g, u16_at(b, 0), u16_at(b, 1));
            const int8x16_t b3 = grid8x2s(g, u16_at(b, 2), u16_at(b, 3));
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            // Lane k = dot of the k-th 16 values (one 3-bit scale each).
            const int32x4_t p1 = vpaddq_s32(vdot(mzero, b0, q8b.val[0]), vdot(mzero, b1, q8b.val[1]));
            const int32x4_t p2 = vpaddq_s32(vdot(mzero, b2, q8b.val[2]), vdot(mzero, b3, q8b.val[3]));
            const int32x4_t p12 = vpaddq_s32(p1, p2);

            const uint32_t h = load_u32_opaque(qh + 2 * ib);
            const uint32_t sel = ((h >> 3) & 0x01010101) | ((h >> 6) & 0x02020202);
            const int32x4_t p3 = vpaddq_s32(vdot(mzero, deltas[byte_at(sel, 0)], q8b.val[0]),
                                            vdot(mzero, deltas[byte_at(sel, 1)], q8b.val[1]));
            const int32x4_t p4 = vpaddq_s32(vdot(mzero, deltas[byte_at(sel, 2)], q8b.val[2]),
                                            vdot(mzero, deltas[byte_at(sel, 3)], q8b.val[3]));
            const int32x4_t p34 = vpaddq_s32(p3, p4);

            // Four 3-bit scales (2s + 1) of this 64-value chunk, from bits 0-11 of scales[ib / 2].
            const int32x4_t s4 = vshlq_s32(vdupq_n_s32(static_cast<int32_t>((sc >> (8 * ib)) & 0xffff)), scale_shift);
            const int32x4_t scales4 = vaddq_s32(vshlq_n_s32(vandq_s32(s4, mask), 1), mone);
            sumi1 = vmlaq_s32(sumi1, scales4, p12);
            sumi2 = vmlaq_s32(sumi2, scales4, p34);
        }
        sumf += y[i].d * h2f(dh) *
                (static_cast<float>(vaddvq_s32(sumi1)) + kIq1Delta * static_cast<float>(vaddvq_s32(sumi2)));
    }
    return sumf;
}

float dot_iq4_xs(const BlockIq4Xs* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const int8x16_t values = vld1q_s8(kIq4nlValues);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    float sumf = 0.0f;
    for (int64_t ibl = 0; ibl < nb; ++ibl) {
        const int8_t* q8 = y[ibl].qs;
        const uint8_t* q4 = x[ibl].qs;
        uint16_t h = x[ibl].scales_h;
        int32_t sumi1 = 0, sumi2 = 0;
        for (int ib = 0; ib < 4; ++ib) {
            const uint8x16x2_t q4bits = vld1q_u8_x2(q4);
            q4 += 32;
            const int8x16x4_t q8b = vld1q_s8_x4(q8);
            q8 += 64;
            // Non-linear 4-bit codebook lookup with a table instruction.
            const int8x16_t v0 = vqtbl1q_s8(values, vandq_u8(q4bits.val[0], m4b));
            const int8x16_t v1 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[0], 4));
            const int8x16_t v2 = vqtbl1q_s8(values, vandq_u8(q4bits.val[1], m4b));
            const int8x16_t v3 = vqtbl1q_s8(values, vshrq_n_u8(q4bits.val[1], 4));
            const int32x4_t prod1 = vdot(vdot(vdupq_n_s32(0), v0, q8b.val[0]), v1, q8b.val[1]);
            const int32x4_t prod2 = vdot(vdot(vdupq_n_s32(0), v2, q8b.val[2]), v3, q8b.val[3]);
            const int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls2 = ((x[ibl].scales_l[ib] >> 4) | ((h << 2) & 0x30)) - 32;
            h = static_cast<uint16_t>(h >> 4);
            sumi1 += vaddvq_s32(prod1) * ls1;
            sumi2 += vaddvq_s32(prod2) * ls2;
        }
        sumf += h2f(x[ibl].d) * y[ibl].d * static_cast<float>(sumi1 + sumi2);
    }
    return sumf;
}

// IQ4_NL and MXFP4: 32-value blocks with a 16-entry int8 codebook, against Q8_0.
template <typename Block, typename Scale>
float dot_codebook4_q8_0(const Block* x, const BlockQ8_0* y, int64_t nb, const int8_t* codebook, Scale scale) noexcept {
    const int8x16_t values = vld1q_s8(codebook);
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    float sumf = 0.0f;
    int64_t ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const uint8x16_t q4a = vld1q_u8(x[ib + 0].qs);
        const uint8x16_t q4b = vld1q_u8(x[ib + 1].qs);
        const int8x16_t v0 = vqtbl1q_s8(values, vandq_u8(q4a, m4b));
        const int8x16_t v1 = vqtbl1q_s8(values, vshrq_n_u8(q4a, 4));
        const int8x16_t v2 = vqtbl1q_s8(values, vandq_u8(q4b, m4b));
        const int8x16_t v3 = vqtbl1q_s8(values, vshrq_n_u8(q4b, 4));
        const int32x4_t p1 = vdot(vdot(vdupq_n_s32(0), v0, vld1q_s8(y[ib].qs)), v1, vld1q_s8(y[ib].qs + 16));
        const int32x4_t p2 = vdot(vdot(vdupq_n_s32(0), v2, vld1q_s8(y[ib + 1].qs)), v3, vld1q_s8(y[ib + 1].qs + 16));
        sumf += scale(x[ib]) * h2f(y[ib].d) * static_cast<float>(vaddvq_s32(p1)) +
                scale(x[ib + 1]) * h2f(y[ib + 1].d) * static_cast<float>(vaddvq_s32(p2));
    }
    for (; ib < nb; ++ib) {
        int32_t sumi = 0;
        for (int j = 0; j < 16; ++j) {
            sumi += y[ib].qs[j] * codebook[x[ib].qs[j] & 0xf] + y[ib].qs[j + 16] * codebook[x[ib].qs[j] >> 4];
        }
        sumf += scale(x[ib]) * h2f(y[ib].d) * static_cast<float>(sumi);
    }
    return sumf;
}

#endif  // LIYAB_LOWBIT_NEON

}  // namespace

void quantize_row_q8_K(const float* x, BlockQ8_K* y, int64_t n) noexcept {
    for (int64_t i = 0; i < n / kSuperBlock; ++i, x += kSuperBlock) {
        // llama.cpp keeps the sign of the first element of largest magnitude:
        // d = max / -127 then maps that element to exactly -127.
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < kSuperBlock; ++j) {
            const float ax = std::fabs(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        BlockQ8_K& b = y[i];
        if (amax == 0.0f) {
            b.d = 0.0f;
            std::memset(b.qs, 0, sizeof b.qs);
            std::memset(b.bsums, 0, sizeof b.bsums);
            continue;
        }
        const float iscale = -127.0f / max;
#if defined(LIYAB_LOWBIT_NEON)
        const float32x4_t vs = vdupq_n_f32(iscale);
        for (int j = 0; j < kSuperBlock; j += 16) {
            // vcvtnq rounds to nearest-even, like llama.cpp's nearest_int().
            const int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + j), vs));
            const int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + j + 4), vs));
            const int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + j + 8), vs));
            const int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + j + 12), vs));
            const int16x8_t s0 = vcombine_s16(vqmovn_s32(i0), vqmovn_s32(i1));
            const int16x8_t s1 = vcombine_s16(vqmovn_s32(i2), vqmovn_s32(i3));
            const int8x16_t q = vminq_s8(vcombine_s8(vqmovn_s16(s0), vqmovn_s16(s1)), vdupq_n_s8(127));
            vst1q_s8(b.qs + j, q);
            b.bsums[j / 16] = static_cast<int16_t>(vaddlvq_s8(q));
        }
#else
        for (int j = 0; j < kSuperBlock; ++j) {
            const long v = std::lrint(iscale * x[j]);  // default rounding mode: nearest-even
            b.qs[j] = static_cast<int8_t>(v > 127 ? 127 : (v < -128 ? -128 : v));
        }
        for (int j = 0; j < kSuperBlock / 16; ++j) {
            int sum = 0;
            for (int e = 0; e < 16; ++e) sum += b.qs[16 * j + e];
            b.bsums[j] = static_cast<int16_t>(sum);
        }
#endif
        b.d = 1.0f / iscale;
    }
}

bool uses_q8_K(DType type) noexcept {
#if defined(LIYAB_LOWBIT_NEON)
    switch (type) {
        case DType::Q2_K: case DType::Q3_K: case DType::TQ2_0: case DType::IQ1_S: case DType::IQ1_M:
        case DType::IQ2_XXS: case DType::IQ2_XS: case DType::IQ2_S: case DType::IQ3_XXS: case DType::IQ3_S:
        case DType::IQ4_XS:
            return true;
        default:
            return false;
    }
#else
    (void)type;
    return false;
#endif
}

float dot_lowbit_q8_K(DType type, const void* row, const BlockQ8_K* x, int64_t n) noexcept {
    const int64_t nb = n / kSuperBlock;
#if defined(LIYAB_LOWBIT_NEON)
    switch (type) {
        case DType::Q2_K: return dot_q2_k(static_cast<const BlockQ2K*>(row), x, nb);
        case DType::Q3_K: return dot_q3_k(static_cast<const BlockQ3K*>(row), x, nb);
        case DType::TQ2_0: return dot_tq2_0(static_cast<const BlockTQ2_0*>(row), x, nb);
        case DType::IQ2_XXS: return dot_iq2_xxs(static_cast<const BlockIq2Xxs*>(row), x, nb);
        case DType::IQ2_XS: return dot_iq2_xs(static_cast<const BlockIq2Xs*>(row), x, nb);
        case DType::IQ2_S: return dot_iq2_s(static_cast<const BlockIq2S*>(row), x, nb);
        case DType::IQ3_XXS: return dot_iq3_xxs(static_cast<const BlockIq3Xxs*>(row), x, nb);
        case DType::IQ3_S: return dot_iq3_s(static_cast<const BlockIq3S*>(row), x, nb);
        case DType::IQ1_S: return dot_iq1_s(static_cast<const BlockIq1S*>(row), x, nb);
        case DType::IQ1_M: return dot_iq1_m(static_cast<const BlockIq1M*>(row), x, nb);
        case DType::IQ4_XS: return dot_iq4_xs(static_cast<const BlockIq4Xs*>(row), x, nb);
        default: break;
    }
#endif
    switch (type) {
        case DType::Q2_K: case DType::Q3_K: case DType::TQ2_0: case DType::IQ1_S: case DType::IQ1_M:
        case DType::IQ2_XXS: case DType::IQ2_XS: case DType::IQ2_S: case DType::IQ3_XXS: case DType::IQ3_S:
        case DType::IQ4_XS:
            return dot_lowbit_generic(type, row, x, nb * kSuperBlock);
        default:
            return 0.0f;
    }
}

float dot_iq4_nl_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept {
#if defined(LIYAB_LOWBIT_NEON)
    return dot_codebook4_q8_0(static_cast<const BlockIq4Nl*>(row), x, n / kBlock, kIq4nlValues,
                              [](const BlockIq4Nl& b) { return h2f(b.d); });
#else
    return dot_ext_q8_0(DType::IQ4_NL, row, x, n);
#endif
}

float dot_mxfp4_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept {
#if defined(LIYAB_LOWBIT_NEON)
    return dot_codebook4_q8_0(static_cast<const BlockMxfp4*>(row), x, n / kBlock, kMxfp4Values,
                              [](const BlockMxfp4& b) { return e8m0_to_fp32_half(b.e); });
#else
    return dot_ext_q8_0(DType::MXFP4, row, x, n);
#endif
}

}  // namespace liyab::quant
