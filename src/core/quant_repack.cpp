// Liyab — repacked Q4_K / Q5_K / Q6_K / Q8_0 layouts (Q4_K_R8, Q5_K_R8,
// Q6_K_R8, Q8_0_R4) and their kernels.
//
// The layouts and the 4-row SMMLA kernels follow llama.cpp (ggml),
// ggml/src/ggml-cpu/repack.cpp (make_block_q4_Kx8, make_block_q5_Kx8,
// make_block_q6_Kx8, make_block_q8_0x4, ggml_quantize_mat_q8_K_4x8) and
// ggml/src/ggml-cpu/arch/arm/repack.cpp (ggml_gemm_q4_K_8x8_q8_K,
// ggml_gemm_q5_K_8x8_q8_K, ggml_gemv_q6_K_8x8_q8_K, ggml_gemm_q6_K_8x8_q8_K,
// ggml_gemv_q8_0_4x8_q8_0, ggml_gemm_q8_0_4x8_q8_0):
//
//     Copyright (c) 2023-2026 The ggml authors
//     Licensed under the MIT License (see THIRD_PARTY_NOTICES.md).
//
// Changes: Liyab's block structs, row-group ranges for the thread pool, the
// activation rows interleaved from already quantized Q8_K rows, and a
// single-row kernel of our own (llama.cpp's gemv adds per 64 values in float)
// that keeps the integer sums of a whole super-block and applies the same
// float epilogue as the 4-row kernel, so a token's logits do not depend on how
// many tokens share the matmul (speculative verification must match decoding).
// Q4_K and Q5_K share one pair of kernels (Q5_K adds the 5th bits as values
// are unpacked).
//
// Compiled with +i8mm (CMakeLists.txt); callers check the CPU first.
#include "core/quant.h"

#include <cstring>
#include <type_traits>

#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8) && \
    defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define LIYAB_REPACK 1
#endif

namespace liyab::quant {

namespace {

// One super-block of 8 Q4_K (Q5_K) rows -> one Q4_Kx8 (Q5_Kx8) block
// (llama.cpp make_block_q4_Kx8 / make_block_q5_Kx8, interleave 8): values 8
// bytes at a time (Q5_K: and the 5th bits, likewise); scales and mins of
// sub-block pair i of the 8 rows packed into 12 bytes at 24 * i (low
// nibbles' sub-block) and 24 * i + 12 (high nibbles').
template <typename Out, typename In>
Out make_k_x8(const In* in) noexcept {
    Out out{};
    if constexpr (std::is_same_v<In, BlockQ5_K>) {
        for (int i = 0; i < kSuperBlock / 8; ++i) std::memcpy(&out.qh[i * 8], &in[i % 8].qh[(i / 8) * 8], 8);
    }
    for (int i = 0; i < 8; ++i) {
        out.d[i] = in[i].d;
        out.dmin[i] = in[i].dmin;
    }
    for (int i = 0; i < kSuperBlock * 4 / 8; ++i) {
        std::memcpy(&out.qs[i * 8], &in[i % 8].qs[(i / 8) * 8], 8);
    }
    uint8_t s[8], m[8];
    auto pack = [&](int at) {
        for (int k = 0; k < 4; ++k) {
            out.scales[at + k] = static_cast<uint8_t>((s[k] & 63) + ((s[k + 4] & 48) << 2));
            out.scales[at + 4 + k] = static_cast<uint8_t>((m[k] & 63) + ((m[k + 4] & 48) << 2));
            out.scales[at + 8 + k] = static_cast<uint8_t>((s[k + 4] & 15) + ((m[k + 4] & 15) << 4));
        }
    };
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            s[j] = in[j].scales[i] & 63;
            m[j] = in[j].scales[i + 4] & 63;
        }
        pack(i * 12);
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 8; ++j) {
            s[j] = static_cast<uint8_t>(((in[j].scales[i] & 192) >> 2) | (in[j].scales[i + 8] & 15));
            m[j] = static_cast<uint8_t>(((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i + 8] & 240) >> 4));
        }
        pack(48 + i * 12);
    }
    return out;
}

// One super-block of 8 Q6_K rows -> one Q6_Kx8 block (llama.cpp
// make_block_q6_Kx8, interleave 8): low and high bits 8 bytes at a time, the
// 16 scales of the 8 rows interleaved (scale k of row i at 8 k + i).
BlockQ6_Kx8 make_q6_Kx8(const BlockQ6_K* in) noexcept {
    BlockQ6_Kx8 out{};
    for (int i = 0; i < 8; ++i) out.d[i] = in[i].d;
    for (int i = 0; i < kSuperBlock * 4 / 8; ++i) std::memcpy(&out.ql[i * 8], &in[i % 8].ql[(i / 8) * 8], 8);
    for (int i = 0; i < kSuperBlock * 2 / 8; ++i) std::memcpy(&out.qh[i * 8], &in[i % 8].qh[(i / 8) * 8], 8);
    for (int i = 0; i < 8; ++i) {
        for (int k = 0; k < kSuperBlock / 16; ++k) out.scales[k * 8 + i] = in[i].scales[k];
    }
    return out;
}

// 4 Q8_0 rows' block -> one Q8_0x4 block (llama.cpp make_block_q8_0x4,
// interleave 8); used for weights and activations alike.
BlockQ8_0x4 make_q8_0x4(const BlockQ8_0* const* in) noexcept {
    BlockQ8_0x4 out{};
    for (int i = 0; i < 4; ++i) out.d[i] = in[i]->d;
    for (int i = 0; i < kBlock * 4 / 8; ++i) std::memcpy(&out.qs[i * 8], &in[i % 4]->qs[(i / 4) * 8], 8);
    return out;
}

#if defined(LIYAB_REPACK)

// fp16 -> fp32 (exact, so gemv and gemm see the same activation scale).
inline float f16_to_f32(uint16_t h) noexcept {
    return vgetq_lane_f32(vcvt_f32_f16(vreinterpret_f16_u16(vdup_n_u16(h))), 0);
}

// Q8_0 epilogue shared by gemv and gemm, per activation row and 4 weight
// rows: acc += sum * d * d8.
inline float32x4_t epilogue_q8_0(float32x4_t acc, const BlockQ8_0x4& w, float d8, int32x4_t sum) noexcept {
    const float32x4_t scale = vmulq_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w.d))), vdupq_n_f32(d8));
    return vfmaq_f32(acc, vcvtq_f32_s32(sum), scale);
}

// Q6_K epilogue shared by gemv and gemm (no minimums): acc += sum * d * d8.
inline float32x4_t epilogue_q6(float32x4_t acc, const BlockQ6_Kx8& w, int j, float d8, int32x4_t sum) noexcept {
    const float32x4_t scale =
        vmulq_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w.d + j * 4))), vdupq_n_f32(d8));
    return vmlaq_f32(acc, vcvtq_f32_s32(sum), scale);
}

// The 12 packed bytes of 8 rows' sub-block -> 8 scales (int8) and 8 mins (int16).
inline void decode_scales_x8(const uint8_t* in, int16x8_t* mins, int8_t* scales) noexcept {
    constexpr uint32_t kmask1 = 0x3f3f3f3f, kmask2 = 0x0f0f0f0f, kmask3 = 0x03030303;
    uint32_t sm[3];
    std::memcpy(sm, in, 12);
    const uint32_t mins_0_3 = sm[1] & kmask1;
    const uint32_t mins_4_7 = ((sm[2] >> 4) & kmask2) | (((sm[1] >> 6) & kmask3) << 4);
    const uint32x2_t mins_u32 = {mins_0_3, mins_4_7};
    *mins = vreinterpretq_s16_u16(vmovl_u8(vreinterpret_u8_u32(mins_u32)));
    const uint32_t sc[2] = {sm[0] & kmask1, (sm[2] & kmask2) | (((sm[0] >> 6) & kmask3) << 4)};
    std::memcpy(scales, sc, 8);
}

// Float epilogue shared by gemv and gemm, per activation row and 4 weight
// rows (half j of the 8): acc -= bias * dmin * d8; acc += sum * d * d8.
template <typename Block>
inline float32x4_t epilogue(float32x4_t acc, const Block& w, int j, float d8, int32x4_t sum, int32x4_t bias) noexcept {
    const float32x4_t q8_d = vdupq_n_f32(d8);
    const float32x4_t dmins = vmulq_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w.dmin + j * 4))), q8_d);
    const float32x4_t scale = vmulq_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w.d + j * 4))), q8_d);
    acc = vmlsq_f32(acc, vcvtq_f32_s32(bias), dmins);
    return vmlaq_f32(acc, vcvtq_f32_s32(sum), scale);
}

// Q5_K's 5th bits of one Q5_Kx8 super-block, held in registers:
// qh[4 cp + k] is the 16 bytes matching the values at 16 cp + 64 k, shifted
// right by 2 after each sub-block pair (llama.cpp's order), so bits 0 / 1 are
// always those of the current pair's low / high nibbles. Nothing for Q4_K.
template <typename Block>
struct HighBits {
    explicit HighBits(const Block&) noexcept {}
    void apply(int, uint8x16_t&, uint8x16_t&) noexcept {}
};
template <>
struct HighBits<BlockQ5_Kx8> {
    uint8x16_t qh[16];
    explicit HighBits(const BlockQ5_Kx8& w) noexcept {
        for (int i = 0; i < 16; ++i) qh[i] = vld1q_u8(w.qh + 16 * (i / 4) + 64 * (i % 4));
    }
    void apply(int i, uint8x16_t& lo, uint8x16_t& hi) noexcept {
        lo = vsliq_n_u8(lo, vandq_u8(qh[i], vdupq_n_u8(1)), 4);
        hi = vorrq_u8(hi, vshlq_n_u8(vandq_u8(qh[i], vdupq_n_u8(2)), 3));
        qh[i] = vshrq_n_u8(qh[i], 2);
    }
};

// The values of weight rows 2 cp, 2 cp + 1 in the 16 bytes at 16 cp + 64 k
// of sub-block pair sb: the low nibbles (sub-block 2 sb) and the high ones
// (2 sb + 1), Q5_K with their 5th bit. Called for sb = 0, 1, 2, 3 in turn.
template <typename Block>
inline void unpack_k_pair(const Block& w, int sb, int cp, int k, HighBits<Block>& high, int8x16_t& lo,
                          int8x16_t& hi) noexcept {
    const uint8x16_t q = vld1q_u8(w.qs + sb * kSuperBlock + 16 * cp + 64 * k);
    uint8x16_t l = vandq_u8(q, vdupq_n_u8(0x0f));
    uint8x16_t h = vshrq_n_u8(q, 4);
    high.apply(4 * cp + k, l, h);
    lo = vreinterpretq_s8_u8(l);
    hi = vreinterpretq_s8_u8(h);
}

#endif

}  // namespace

bool repack_kernels_available() noexcept {
#if defined(LIYAB_REPACK)
    return true;
#else
    return false;
#endif
}

void repack_q4_K_r8(const BlockQ4_K* src, int64_t rows, int64_t cols, BlockQ4_Kx8* dst) noexcept {
    const int64_t nb = cols / kSuperBlock;
    for (int64_t r = 0; r + 8 <= rows; r += 8) {
        for (int64_t b = 0; b < nb; ++b) {
            BlockQ4_K group[8];
            for (int i = 0; i < 8; ++i) group[i] = src[(r + i) * nb + b];
            *dst++ = make_k_x8<BlockQ4_Kx8>(group);
        }
    }
}

void repack_q5_K_r8(const BlockQ5_K* src, int64_t rows, int64_t cols, BlockQ5_Kx8* dst) noexcept {
    const int64_t nb = cols / kSuperBlock;
    for (int64_t r = 0; r + 8 <= rows; r += 8) {
        for (int64_t b = 0; b < nb; ++b) {
            BlockQ5_K group[8];
            for (int i = 0; i < 8; ++i) group[i] = src[(r + i) * nb + b];
            *dst++ = make_k_x8<BlockQ5_Kx8>(group);
        }
    }
}

void repack_q8_0_r4(const BlockQ8_0* src, int64_t rows, int64_t cols, BlockQ8_0x4* dst) noexcept {
    const int64_t nb = cols / kBlock;
    for (int64_t r = 0; r + 4 <= rows; r += 4) {
        for (int64_t b = 0; b < nb; ++b) {
            const BlockQ8_0* group[4];
            for (int i = 0; i < 4; ++i) group[i] = &src[(r + i) * nb + b];
            *dst++ = make_q8_0x4(group);
        }
    }
}

void interleave_q8_0_x4(const BlockQ8_0* const* rows, int64_t cols, BlockQ8_0x4* dst) noexcept {
    for (int64_t b = 0; b < cols / kBlock; ++b) {
        const BlockQ8_0* group[4] = {&rows[0][b], &rows[1][b], &rows[2][b], &rows[3][b]};
        dst[b] = make_q8_0x4(group);
    }
}

void repack_q6_K_r8(const BlockQ6_K* src, int64_t rows, int64_t cols, BlockQ6_Kx8* dst) noexcept {
    const int64_t nb = cols / kSuperBlock;
    for (int64_t r = 0; r + 8 <= rows; r += 8) {
        for (int64_t b = 0; b < nb; ++b) {
            BlockQ6_K group[8];
            for (int i = 0; i < 8; ++i) group[i] = src[(r + i) * nb + b];
            *dst++ = make_q6_Kx8(group);
        }
    }
}

void interleave_q8_K_x4(const BlockQ8_K* const* rows, int64_t cols, BlockQ8_Kx4* dst) noexcept {
    for (int64_t b = 0; b < cols / kSuperBlock; ++b) {
        BlockQ8_Kx4& out = dst[b];
        for (int r = 0; r < 4; ++r) out.d[r] = rows[r][b].d;
        std::memset(out.bsums, 0, sizeof out.bsums);
        // Values 8 bytes at a time from each row in turn; bsums as llama.cpp's
        // quantize_mat (4 sums of 16 of row 0, then of row 1, ...).
        for (int j = 0; j < kSuperBlock * 4; ++j) {
            const int src_row = (j % 32) / 8;
            const int src_at = (j / 32) * 8 + (j % 8);
            const int8_t v = rows[src_row][b].qs[src_at];
            out.qs[j] = v;
            out.bsums[(((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3)] += v;
        }
    }
}

#if defined(LIYAB_REPACK)

namespace {

// Q4_K_R8 / Q5_K_R8 kernels (Block: BlockQ4_Kx8 / BlockQ5_Kx8).
template <typename Block>
void gemv_k_r8(const Block* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_K* x, float* y) noexcept {
    const int64_t nb = cols / kSuperBlock;
    for (int64_t g = g0; g < g1; ++g) {
        const Block* q4 = w + g * nb;
        float32x4_t acc_f32[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (int64_t b = 0; b < nb; ++b) {
            int32x4_t sum[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};   // weight rows 0-3, 4-7
            int32x4_t bias[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
            int16_t bsums[8];  // per 32 values
            vst1q_s16(bsums, vpaddq_s16(vld1q_s16(x[b].bsums), vld1q_s16(x[b].bsums + 8)));
            HighBits<Block> high(q4[b]);
            for (int sb = 0; sb < kSuperBlock / 64; ++sb) {
                int16x8_t mins[2];
                int8_t sc8[2][8];
                decode_scales_x8(&q4[b].scales[sb * 24], &mins[0], sc8[0]);
                decode_scales_x8(&q4[b].scales[sb * 24 + 12], &mins[1], sc8[1]);
                const int16x8_t sc_lo = vmovl_s8(vld1_s8(sc8[0]));  // low nibbles: sub-block 2 sb
                const int16x8_t sc_hi = vmovl_s8(vld1_s8(sc8[1]));  // high nibbles: sub-block 2 sb + 1
                const int8_t* q8_base = x[b].qs + sb * 64;
                int8x16_t q8[8];
                for (int i = 0; i < 8; ++i) q8[i] = vreinterpretq_s8_s64(vld1q_dup_s64(reinterpret_cast<const int64_t*>(q8_base + i * 8)));
                int32x4_t lo[4], hi[4];  // per pair of weight rows: lanes 0-1 row 2cp, 2-3 row 2cp + 1
                for (int cp = 0; cp < 4; ++cp) {
                    int32x4_t l = vdupq_n_s32(0), h = vdupq_n_s32(0);
                    for (int k = 0; k < 4; ++k) {
                        int8x16_t vl, vh;
                        unpack_k_pair(q4[b], sb, cp, k, high, vl, vh);
                        l = vdotq_s32(l, vl, q8[k]);
                        h = vdotq_s32(h, vh, q8[4 + k]);
                    }
                    lo[cp] = l;
                    hi[cp] = h;
                }
                for (int half = 0; half < 2; ++half) {  // weight rows 4 half .. 4 half + 3
                    const int32x4_t dl = vpaddq_s32(lo[2 * half], lo[2 * half + 1]);
                    const int32x4_t dh = vpaddq_s32(hi[2 * half], hi[2 * half + 1]);
                    const int16x4_t sl = half == 0 ? vget_low_s16(sc_lo) : vget_high_s16(sc_lo);
                    const int16x4_t sh = half == 0 ? vget_low_s16(sc_hi) : vget_high_s16(sc_hi);
                    sum[half] = vmlaq_s32(sum[half], dl, vmovl_s16(sl));
                    sum[half] = vmlaq_s32(sum[half], dh, vmovl_s16(sh));
                    const int16x4_t ml = half == 0 ? vget_low_s16(mins[0]) : vget_high_s16(mins[0]);
                    const int16x4_t mh = half == 0 ? vget_low_s16(mins[1]) : vget_high_s16(mins[1]);
                    bias[half] = vmlal_s16(bias[half], vdup_n_s16(bsums[2 * sb]), ml);
                    bias[half] = vmlal_s16(bias[half], vdup_n_s16(bsums[2 * sb + 1]), mh);
                }
            }
            for (int j = 0; j < 2; ++j) acc_f32[j] = epilogue(acc_f32[j], q4[b], j, x[b].d, sum[j], bias[j]);
        }
        vst1q_f32(y + 8 * g, acc_f32[0]);
        vst1q_f32(y + 8 * g + 4, acc_f32[1]);
    }
}

template <typename Block>
void gemm_k_r8(const Block* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_Kx4* x, float* y,
               int64_t ldy) noexcept {
    const int64_t nb = cols / kSuperBlock;
    for (int64_t g = g0; g < g1; ++g) {
        const Block* q4 = w + g * nb;
        float32x4_t acc_f32[8];  // [2 t + j]: activation row t, weight rows 4 j .. 4 j + 3
        for (float32x4_t& a : acc_f32) a = vdupq_n_f32(0.0f);
        for (int64_t b = 0; b < nb; ++b) {
            int16_t bsums[4][8];  // [sub-block pair][2 t + (0: low, 1: high)]
            for (int q = 0; q < 4; ++q) {
                vst1q_s16(bsums[q], vpaddq_s16(vld1q_s16(x[b].bsums + 16 * q), vld1q_s16(x[b].bsums + 16 * q + 8)));
            }
            int32x4_t acc[8];   // SMMLA tiles: [cp] rows 01 x weight pair cp, [cp + 4] rows 23
            int32x4_t bias[8];  // [2 t + j]
            for (int i = 0; i < 8; ++i) acc[i] = bias[i] = vdupq_n_s32(0);
            HighBits<Block> high(q4[b]);
            for (int sb = 0; sb < kSuperBlock / 64; ++sb) {
                int8_t sc[2][8];
                int16x8_t mins[2];
                decode_scales_x8(&q4[b].scales[sb * 24], &mins[0], sc[0]);
                decode_scales_x8(&q4[b].scales[sb * 24 + 12], &mins[1], sc[1]);
                const int8_t* q8_base = x[b].qs + sb * 256;
                int8x16_t q8[2][8];  // [row pair][i]: 8 values of 2 rows
                for (int i = 0; i < 8; ++i) {
                    q8[0][i] = vld1q_s8(q8_base + i * 32);
                    q8[1][i] = vld1q_s8(q8_base + i * 32 + 16);
                }
                for (int cp = 0; cp < 4; ++cp) {
                    int8x16_t nib[2][4];  // [0: low, 1: high nibbles][k]
                    for (int k = 0; k < 4; ++k) unpack_k_pair(q4[b], sb, cp, k, high, nib[0][k], nib[1][k]);
                    int32x4_t sb_acc[4];  // [2 rp + blk]
                    for (int rp = 0; rp < 2; ++rp) {
                        for (int blk = 0; blk < 2; ++blk) {
                            int32x4_t a = vdupq_n_s32(0);
                            for (int k = 0; k < 4; ++k) a = vmmlaq_s32(a, nib[blk][k], q8[rp][4 * blk + k]);
                            sb_acc[2 * rp + blk] = a;
                        }
                    }
                    const int32x4_t s0 = vcombine_s32(vdup_n_s32(sc[0][2 * cp]), vdup_n_s32(sc[0][2 * cp + 1]));
                    const int32x4_t s1 = vcombine_s32(vdup_n_s32(sc[1][2 * cp]), vdup_n_s32(sc[1][2 * cp + 1]));
                    acc[cp] = vmlaq_s32(acc[cp], sb_acc[0], s0);
                    acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc[2], s0);
                    acc[cp] = vmlaq_s32(acc[cp], sb_acc[1], s1);
                    acc[cp + 4] = vmlaq_s32(acc[cp + 4], sb_acc[3], s1);
                }
                for (int t = 0; t < 4; ++t) {
                    const int16x4_t lo = vdup_n_s16(bsums[sb][t * 2]);
                    const int16x4_t hi = vdup_n_s16(bsums[sb][t * 2 + 1]);
                    bias[2 * t] = vmlal_s16(bias[2 * t], lo, vget_low_s16(mins[0]));
                    bias[2 * t] = vmlal_s16(bias[2 * t], hi, vget_low_s16(mins[1]));
                    bias[2 * t + 1] = vmlal_s16(bias[2 * t + 1], lo, vget_high_s16(mins[0]));
                    bias[2 * t + 1] = vmlal_s16(bias[2 * t + 1], hi, vget_high_s16(mins[1]));
                }
            }
            // SMMLA tiles -> per activation row, 4 weight rows each.
            for (int i = 0; i < 8; ++i) {
                const int32x2x2_t z = vzip_s32(vget_low_s32(acc[i]), vget_high_s32(acc[i]));
                acc[i] = vcombine_s32(z.val[0], z.val[1]);
            }
            const int32x4_t sum[8] = {
                vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
                vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
                vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
                vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
                vcombine_s32(vget_low_s32(acc[4]), vget_low_s32(acc[5])),
                vcombine_s32(vget_low_s32(acc[6]), vget_low_s32(acc[7])),
                vcombine_s32(vget_high_s32(acc[4]), vget_high_s32(acc[5])),
                vcombine_s32(vget_high_s32(acc[6]), vget_high_s32(acc[7])),
            };
            for (int t = 0; t < 4; ++t) {
                for (int j = 0; j < 2; ++j) {
                    acc_f32[2 * t + j] = epilogue(acc_f32[2 * t + j], q4[b], j, x[b].d[t], sum[2 * t + j], bias[2 * t + j]);
                }
            }
        }
        for (int t = 0; t < 4; ++t) {
            vst1q_f32(y + t * ldy + 8 * g, acc_f32[2 * t]);
            vst1q_f32(y + t * ldy + 8 * g + 4, acc_f32[2 * t + 1]);
        }
    }
}

}  // namespace

void gemv_q4_K_r8(const BlockQ4_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_K* x, float* y) noexcept {
    gemv_k_r8(w, cols, g0, g1, x, y);
}
void gemm_q4_K_r8(const BlockQ4_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_Kx4* x, float* y,
                  int64_t ldy) noexcept {
    gemm_k_r8(w, cols, g0, g1, x, y, ldy);
}
void gemv_q5_K_r8(const BlockQ5_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_K* x, float* y) noexcept {
    gemv_k_r8(w, cols, g0, g1, x, y);
}
void gemm_q5_K_r8(const BlockQ5_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_Kx4* x, float* y,
                  int64_t ldy) noexcept {
    gemm_k_r8(w, cols, g0, g1, x, y, ldy);
}

void gemv_q8_0_r4(const BlockQ8_0x4* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_0* x, float* y) noexcept {
    const int64_t nb = cols / kBlock;
    for (int64_t g = g0; g < g1; ++g) {
        const BlockQ8_0x4* q = w + g * nb;
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int64_t b = 0; b < nb; ++b) {
            // 16 bytes = 8 values of 2 weight rows; the 8 activation values twice.
            const int8x16x4_t lo = vld1q_s8_x4(q[b].qs);
            const int8x16x4_t hi = vld1q_s8_x4(q[b].qs + 64);
            const int8x8x4_t a = vld1_s8_x4(x[b].qs);
            int32x4_t r01 = vdupq_n_s32(0), r23 = vdupq_n_s32(0);  // lanes: [row, half of 8] pairs
            for (int k = 0; k < 4; ++k) {
                const int8x16_t ak = vcombine_s8(a.val[k], a.val[k]);
                const int8x16x4_t& src = k < 2 ? lo : hi;
                r01 = vdotq_s32(r01, src.val[(k % 2) * 2], ak);
                r23 = vdotq_s32(r23, src.val[(k % 2) * 2 + 1], ak);
            }
            acc = epilogue_q8_0(acc, q[b], f16_to_f32(x[b].d), vpaddq_s32(r01, r23));
        }
        vst1q_f32(y + 4 * g, acc);
    }
}

void gemm_q8_0_r4(const BlockQ8_0x4* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_0x4* x, float* y,
                  int64_t ldy) noexcept {
    const int64_t nb = cols / kBlock;
    for (int64_t g = g0; g < g1; ++g) {
        const BlockQ8_0x4* q = w + g * nb;
        float32x4_t acc_f32[4];  // per activation row
        for (float32x4_t& a : acc_f32) a = vdupq_n_f32(0.0f);
        for (int64_t b = 0; b < nb; ++b) {
            int32x4_t acc[4];  // SMMLA tiles [activation pair][weight pair]
            for (int32x4_t& a : acc) a = vdupq_n_s32(0);
            for (int chunk = 0; chunk < 4; ++chunk) {
                const int8x16_t a01 = vld1q_s8(x[b].qs + chunk * 32);
                const int8x16_t a23 = vld1q_s8(x[b].qs + chunk * 32 + 16);
                const int8x16_t b01 = vld1q_s8(q[b].qs + chunk * 32);
                const int8x16_t b23 = vld1q_s8(q[b].qs + chunk * 32 + 16);
                acc[0] = vmmlaq_s32(acc[0], a01, b01);
                acc[1] = vmmlaq_s32(acc[1], a01, b23);
                acc[2] = vmmlaq_s32(acc[2], a23, b01);
                acc[3] = vmmlaq_s32(acc[3], a23, b23);
            }
            // 2x2 tiles -> per activation row, weight rows 0-3.
            const int32x4_t sum[4] = {
                vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
                vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
                vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
                vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
            };
            for (int t = 0; t < 4; ++t) acc_f32[t] = epilogue_q8_0(acc_f32[t], q[b], f16_to_f32(x[b].d[t]), sum[t]);
        }
        for (int t = 0; t < 4; ++t) vst1q_f32(y + t * ldy + 4 * g, acc_f32[t]);
    }
}

void gemv_q6_K_r8(const BlockQ6_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_K* x, float* y) noexcept {
    const int64_t nb = cols / kSuperBlock;
    const uint8x16_t m4b = vdupq_n_u8(0x0f), mask_lo = vdupq_n_u8(0x03), mask_hi = vdupq_n_u8(0x30);
    for (int64_t g = g0; g < g1; ++g) {
        const BlockQ6_Kx8* q6 = w + g * nb;
        float32x4_t acc_f32[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (int64_t b = 0; b < nb; ++b) {
            int16_t sc[16 * 8];
            for (int i = 0; i < 16; ++i) vst1q_s16(sc + i * 8, vmovl_s8(vld1_s8(q6[b].scales + i * 8)));
            // Values are stored as q + 32: subtract 32 * sum(scale * activation sum per 16).
            int32x4_t bias_lo = vdupq_n_s32(0), bias_hi = vdupq_n_s32(0);
            for (int i = 0; i < 16; i += 4) {
                const int16x4_t bs = vld1_s16(x[b].bsums + i);
                bias_lo = vmlal_lane_s16(bias_lo, vld1_s16(sc + (i + 0) * 8), bs, 0);
                bias_hi = vmlal_lane_s16(bias_hi, vld1_s16(sc + (i + 0) * 8 + 4), bs, 0);
                bias_lo = vmlal_lane_s16(bias_lo, vld1_s16(sc + (i + 1) * 8), bs, 1);
                bias_hi = vmlal_lane_s16(bias_hi, vld1_s16(sc + (i + 1) * 8 + 4), bs, 1);
                bias_lo = vmlal_lane_s16(bias_lo, vld1_s16(sc + (i + 2) * 8), bs, 2);
                bias_hi = vmlal_lane_s16(bias_hi, vld1_s16(sc + (i + 2) * 8 + 4), bs, 2);
                bias_lo = vmlal_lane_s16(bias_lo, vld1_s16(sc + (i + 3) * 8), bs, 3);
                bias_hi = vmlal_lane_s16(bias_hi, vld1_s16(sc + (i + 3) * 8 + 4), bs, 3);
            }
            int32x2_t acc[4] = {vdup_n_s32(0), vdup_n_s32(0), vdup_n_s32(0), vdup_n_s32(0)};  // per weight-row pair
            for (int half = 0; half < 2; ++half) {
                const uint8_t* ql_base = q6[b].ql + half * 512;
                const uint8_t* qh_base = q6[b].qh + half * 256;
                for (int sb = 0; sb < kSuperBlock / 64; ++sb) {
                    const int8_t* q8l = x[b].qs + half * 128 + sb * 16;
                    const int8_t* q8h = q8l + 64;
                    int8x16_t xl[2], xh[2];
                    for (int i = 0; i < 2; ++i) {
                        xl[i] = vreinterpretq_s8_s64(vld1q_dup_s64(reinterpret_cast<const int64_t*>(q8l + i * 8)));
                        xh[i] = vreinterpretq_s8_s64(vld1q_dup_s64(reinterpret_cast<const int64_t*>(q8h + i * 8)));
                    }
                    const int ql_off = sb * kSuperBlock / 2;
                    const int qh_off = ql_off & 255;
                    uint8x16x4_t ql0 = vld1q_u8_x4(ql_base + ql_off), ql1 = vld1q_u8_x4(ql_base + ql_off + 64);
                    uint8x16x4_t qh0 = vld1q_u8_x4(qh_base + qh_off), qh1 = vld1q_u8_x4(qh_base + qh_off + 64);
                    if (sb > 1) {
                        for (int k = 0; k < 4; ++k) {
                            qh0.val[k] = vshrq_n_u8(qh0.val[k], 2);
                            qh1.val[k] = vshrq_n_u8(qh1.val[k], 2);
                        }
                    }
                    for (int cp = 0; cp < 4; ++cp) {
                        const int8x16_t l0 = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(ql0.val[cp], m4b), vandq_u8(qh0.val[cp], mask_lo), 4));
                        const int8x16_t l1 = vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(ql1.val[cp], m4b), vandq_u8(qh1.val[cp], mask_lo), 4));
                        const int8x16_t h0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0.val[cp], 4), vandq_u8(qh0.val[cp], mask_hi)));
                        const int8x16_t h1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1.val[cp], 4), vandq_u8(qh1.val[cp], mask_hi)));
                        const int32x4_t al = vdotq_s32(vdotq_s32(vdupq_n_s32(0), l0, xl[0]), l1, xl[1]);
                        const int32x4_t ah = vdotq_s32(vdotq_s32(vdupq_n_s32(0), h0, xh[0]), h1, xh[1]);
                        const int32x2_t sl = vpadd_s32(vget_low_s32(al), vget_high_s32(al));  // [row 2cp, row 2cp + 1]
                        const int32x2_t sh = vpadd_s32(vget_low_s32(ah), vget_high_s32(ah));
                        const int il = half * 8 + sb, ih = half * 8 + sb + 4;
                        const int32x2_t vl = {sc[il * 8 + cp * 2], sc[il * 8 + cp * 2 + 1]};
                        const int32x2_t vh = {sc[ih * 8 + cp * 2], sc[ih * 8 + cp * 2 + 1]};
                        acc[cp] = vmla_s32(acc[cp], sl, vl);
                        acc[cp] = vmla_s32(acc[cp], sh, vh);
                    }
                }
            }
            const int32x4_t sum_lo = vsubq_s32(vcombine_s32(acc[0], acc[1]), vshlq_n_s32(bias_lo, 5));
            const int32x4_t sum_hi = vsubq_s32(vcombine_s32(acc[2], acc[3]), vshlq_n_s32(bias_hi, 5));
            acc_f32[0] = epilogue_q6(acc_f32[0], q6[b], 0, x[b].d, sum_lo);
            acc_f32[1] = epilogue_q6(acc_f32[1], q6[b], 1, x[b].d, sum_hi);
        }
        vst1q_f32(y + 8 * g, acc_f32[0]);
        vst1q_f32(y + 8 * g + 4, acc_f32[1]);
    }
}

void gemm_q6_K_r8(const BlockQ6_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_Kx4* x, float* y,
                  int64_t ldy) noexcept {
    const int64_t nb = cols / kSuperBlock;
    const uint8x16_t m4b = vdupq_n_u8(0x0f), mask_lo = vdupq_n_u8(0x03), mask_hi = vdupq_n_u8(0x30);
    const int8x16_t m32s = vdupq_n_s8(32);
    for (int64_t g = g0; g < g1; ++g) {
        const BlockQ6_Kx8* q6 = w + g * nb;
        float32x4_t acc_f32[8];
        for (float32x4_t& a : acc_f32) a = vdupq_n_f32(0.0f);
        for (int64_t b = 0; b < nb; ++b) {
            int32x4_t acc[8];
            for (int32x4_t& a : acc) a = vdupq_n_s32(0);
            int16_t sc[16 * 8];
            for (int i = 0; i < 16; ++i) vst1q_s16(sc + i * 8, vmovl_s8(vld1_s8(q6[b].scales + i * 8)));
            for (int half = 0; half < 2; ++half) {
                const uint8_t* ql_base = q6[b].ql + half * 512;
                const uint8_t* qh_base = q6[b].qh + half * 256;
                for (int sb = 0; sb < kSuperBlock / 64; ++sb) {
                    const int8_t* q8l = x[b].qs + half * 512 + sb * 64;
                    const int8_t* q8h = x[b].qs + half * 512 + 256 + sb * 64;
                    int8x16_t l01[2], l23[2], h01[2], h23[2];
                    for (int i = 0; i < 2; ++i) {
                        l01[i] = vld1q_s8(q8l + i * 32);
                        l23[i] = vld1q_s8(q8l + i * 32 + 16);
                        h01[i] = vld1q_s8(q8h + i * 32);
                        h23[i] = vld1q_s8(q8h + i * 32 + 16);
                    }
                    const int ql_off = sb * kSuperBlock / 2;
                    const int qh_off = ql_off & 255;
                    uint8x16_t ql0[4], ql1[4], qh0[4], qh1[4];
                    for (int k = 0; k < 4; ++k) {
                        ql0[k] = vld1q_u8(ql_base + ql_off + 16 * k);
                        ql1[k] = vld1q_u8(ql_base + ql_off + 64 + 16 * k);
                        qh0[k] = vld1q_u8(qh_base + qh_off + 16 * k);
                        qh1[k] = vld1q_u8(qh_base + qh_off + 64 + 16 * k);
                        if (sb > 1) {
                            qh0[k] = vshrq_n_u8(qh0[k], 2);
                            qh1[k] = vshrq_n_u8(qh1[k], 2);
                        }
                    }
                    for (int cp = 0; cp < 4; ++cp) {
                        const int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(ql0[cp], m4b), vandq_u8(qh0[cp], mask_lo), 4)), m32s);
                        const int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vsliq_n_u8(vandq_u8(ql1[cp], m4b), vandq_u8(qh1[cp], mask_lo), 4)), m32s);
                        const int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql0[cp], 4), vandq_u8(qh0[cp], mask_hi))), m32s);
                        const int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(ql1[cp], 4), vandq_u8(qh1[cp], mask_hi))), m32s);
                        const int32x4_t a0l = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), l0, l01[0]), l1, l01[1]);
                        const int32x4_t a0h = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), h0, h01[0]), h1, h01[1]);
                        const int32x4_t a1l = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), l0, l23[0]), l1, l23[1]);
                        const int32x4_t a1h = vmmlaq_s32(vmmlaq_s32(vdupq_n_s32(0), h0, h23[0]), h1, h23[1]);
                        const int il = half * 8 + sb, ih = half * 8 + sb + 4;
                        const int32x4_t vl = {sc[il * 8 + cp * 2], sc[il * 8 + cp * 2], sc[il * 8 + cp * 2 + 1], sc[il * 8 + cp * 2 + 1]};
                        const int32x4_t vh = {sc[ih * 8 + cp * 2], sc[ih * 8 + cp * 2], sc[ih * 8 + cp * 2 + 1], sc[ih * 8 + cp * 2 + 1]};
                        acc[cp] = vmlaq_s32(vmlaq_s32(acc[cp], a0l, vl), a0h, vh);
                        acc[cp + 4] = vmlaq_s32(vmlaq_s32(acc[cp + 4], a1l, vl), a1h, vh);
                    }
                }
            }
            for (int i = 0; i < 8; ++i) {
                const int32x2x2_t z = vzip_s32(vget_low_s32(acc[i]), vget_high_s32(acc[i]));
                acc[i] = vcombine_s32(z.val[0], z.val[1]);
            }
            const int32x4_t sum[8] = {
                vcombine_s32(vget_low_s32(acc[0]), vget_low_s32(acc[1])),
                vcombine_s32(vget_low_s32(acc[2]), vget_low_s32(acc[3])),
                vcombine_s32(vget_high_s32(acc[0]), vget_high_s32(acc[1])),
                vcombine_s32(vget_high_s32(acc[2]), vget_high_s32(acc[3])),
                vcombine_s32(vget_low_s32(acc[4]), vget_low_s32(acc[5])),
                vcombine_s32(vget_low_s32(acc[6]), vget_low_s32(acc[7])),
                vcombine_s32(vget_high_s32(acc[4]), vget_high_s32(acc[5])),
                vcombine_s32(vget_high_s32(acc[6]), vget_high_s32(acc[7])),
            };
            for (int t = 0; t < 4; ++t) {
                for (int j = 0; j < 2; ++j) acc_f32[2 * t + j] = epilogue_q6(acc_f32[2 * t + j], q6[b], j, x[b].d[t], sum[2 * t + j]);
            }
        }
        for (int t = 0; t < 4; ++t) {
            vst1q_f32(y + t * ldy + 8 * g, acc_f32[2 * t]);
            vst1q_f32(y + t * ldy + 8 * g + 4, acc_f32[2 * t + 1]);
        }
    }
}

#else

void gemv_q6_K_r8(const BlockQ6_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_K*, float*) noexcept {}
void gemm_q6_K_r8(const BlockQ6_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_Kx4*, float*, int64_t) noexcept {}
void gemv_q4_K_r8(const BlockQ4_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_K*, float*) noexcept {}
void gemm_q4_K_r8(const BlockQ4_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_Kx4*, float*, int64_t) noexcept {}
void gemv_q5_K_r8(const BlockQ5_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_K*, float*) noexcept {}
void gemm_q5_K_r8(const BlockQ5_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_Kx4*, float*, int64_t) noexcept {}
void gemv_q8_0_r4(const BlockQ8_0x4*, int64_t, int64_t, int64_t, const BlockQ8_0*, float*) noexcept {}
void gemm_q8_0_r4(const BlockQ8_0x4*, int64_t, int64_t, int64_t, const BlockQ8_0x4*, float*, int64_t) noexcept {}

#endif

}  // namespace liyab::quant
