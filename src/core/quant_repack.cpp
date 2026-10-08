// Liyab — repacked Q4_K layout (Q4_K_R8) and its kernels.
//
// The layout and the 4-row SMMLA kernel follow llama.cpp (ggml),
// ggml/src/ggml-cpu/repack.cpp (make_block_q4_Kx8, ggml_quantize_mat_q8_K_4x8)
// and ggml/src/ggml-cpu/arch/arm/repack.cpp (ggml_gemm_q4_K_8x8_q8_K):
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
//
// Compiled with +i8mm (CMakeLists.txt); callers check the CPU first.
#include "core/quant.h"

#include <cstring>

#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8) && \
    defined(__ARM_FEATURE_DOTPROD)
#include <arm_neon.h>
#define LIYAB_REPACK 1
#endif

namespace liyab::quant {

namespace {

// One super-block of 8 Q4_K rows -> one Q4_Kx8 block (llama.cpp make_block_q4_Kx8,
// interleave 8): values 8 bytes at a time; scales and mins of sub-block pair
// i of the 8 rows packed into 12 bytes at 24 * i (low nibbles' sub-block) and
// 24 * i + 12 (high nibbles').
BlockQ4_Kx8 make_q4_Kx8(const BlockQ4_K* in) noexcept {
    BlockQ4_Kx8 out{};
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

#if defined(LIYAB_REPACK)

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
inline float32x4_t epilogue(float32x4_t acc, const BlockQ4_Kx8& w, int j, float d8, int32x4_t sum,
                            int32x4_t bias) noexcept {
    const float32x4_t q8_d = vdupq_n_f32(d8);
    const float32x4_t dmins = vmulq_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w.dmin + j * 4))), q8_d);
    const float32x4_t scale = vmulq_f32(vcvt_f32_f16(vld1_f16(reinterpret_cast<const __fp16*>(w.d + j * 4))), q8_d);
    acc = vmlsq_f32(acc, vcvtq_f32_s32(bias), dmins);
    return vmlaq_f32(acc, vcvtq_f32_s32(sum), scale);
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
            *dst++ = make_q4_Kx8(group);
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

void gemv_q4_K_r8(const BlockQ4_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_K* x, float* y) noexcept {
    const int64_t nb = cols / kSuperBlock;
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    for (int64_t g = g0; g < g1; ++g) {
        const BlockQ4_Kx8* q4 = w + g * nb;
        float32x4_t acc_f32[2] = {vdupq_n_f32(0.0f), vdupq_n_f32(0.0f)};
        for (int64_t b = 0; b < nb; ++b) {
            int32x4_t sum[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};   // weight rows 0-3, 4-7
            int32x4_t bias[2] = {vdupq_n_s32(0), vdupq_n_s32(0)};
            int16_t bsums[8];  // per 32 values
            vst1q_s16(bsums, vpaddq_s16(vld1q_s16(x[b].bsums), vld1q_s16(x[b].bsums + 8)));
            for (int sb = 0; sb < kSuperBlock / 64; ++sb) {
                int16x8_t mins[2];
                int8_t sc8[2][8];
                decode_scales_x8(&q4[b].scales[sb * 24], &mins[0], sc8[0]);
                decode_scales_x8(&q4[b].scales[sb * 24 + 12], &mins[1], sc8[1]);
                const int16x8_t sc_lo = vmovl_s8(vld1_s8(sc8[0]));  // low nibbles: sub-block 2 sb
                const int16x8_t sc_hi = vmovl_s8(vld1_s8(sc8[1]));  // high nibbles: sub-block 2 sb + 1
                const uint8_t* q4_base = q4[b].qs + sb * kSuperBlock;
                const int8_t* q8_base = x[b].qs + sb * 64;
                int8x16_t q8[8];
                for (int i = 0; i < 8; ++i) q8[i] = vreinterpretq_s8_s64(vld1q_dup_s64(reinterpret_cast<const int64_t*>(q8_base + i * 8)));
                int32x4_t lo[4], hi[4];  // per pair of weight rows: lanes 0-1 row 2cp, 2-3 row 2cp + 1
                for (int cp = 0; cp < 4; ++cp) {
                    const uint8x16_t b0 = vld1q_u8(q4_base + 16 * cp);
                    const uint8x16_t b1 = vld1q_u8(q4_base + 16 * cp + 64);
                    const uint8x16_t b2 = vld1q_u8(q4_base + 16 * cp + 128);
                    const uint8x16_t b3 = vld1q_u8(q4_base + 16 * cp + 192);
                    int32x4_t l = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(b0, m4b)), q8[0]);
                    l = vdotq_s32(l, vreinterpretq_s8_u8(vandq_u8(b1, m4b)), q8[1]);
                    l = vdotq_s32(l, vreinterpretq_s8_u8(vandq_u8(b2, m4b)), q8[2]);
                    lo[cp] = vdotq_s32(l, vreinterpretq_s8_u8(vandq_u8(b3, m4b)), q8[3]);
                    int32x4_t h = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), q8[4]);
                    h = vdotq_s32(h, vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)), q8[5]);
                    h = vdotq_s32(h, vreinterpretq_s8_u8(vshrq_n_u8(b2, 4)), q8[6]);
                    hi[cp] = vdotq_s32(h, vreinterpretq_s8_u8(vshrq_n_u8(b3, 4)), q8[7]);
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

void gemm_q4_K_r8(const BlockQ4_Kx8* w, int64_t cols, int64_t g0, int64_t g1, const BlockQ8_Kx4* x, float* y,
                  int64_t ldy) noexcept {
    const int64_t nb = cols / kSuperBlock;
    const uint8x16_t m4b = vdupq_n_u8(0x0f);
    for (int64_t g = g0; g < g1; ++g) {
        const BlockQ4_Kx8* q4 = w + g * nb;
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
                    const uint8x16_t b0 = vld1q_u8(q4[b].qs + sb * kSuperBlock + 16 * cp);
                    const uint8x16_t b1 = vld1q_u8(q4[b].qs + sb * kSuperBlock + 16 * cp + 64);
                    const uint8x16_t b2 = vld1q_u8(q4[b].qs + sb * kSuperBlock + 16 * cp + 128);
                    const uint8x16_t b3 = vld1q_u8(q4[b].qs + sb * kSuperBlock + 16 * cp + 192);
                    const int8x16_t nib[2][4] = {
                        {vreinterpretq_s8_u8(vandq_u8(b0, m4b)), vreinterpretq_s8_u8(vandq_u8(b1, m4b)),
                         vreinterpretq_s8_u8(vandq_u8(b2, m4b)), vreinterpretq_s8_u8(vandq_u8(b3, m4b))},
                        {vreinterpretq_s8_u8(vshrq_n_u8(b0, 4)), vreinterpretq_s8_u8(vshrq_n_u8(b1, 4)),
                         vreinterpretq_s8_u8(vshrq_n_u8(b2, 4)), vreinterpretq_s8_u8(vshrq_n_u8(b3, 4))}};
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

#else

void gemv_q4_K_r8(const BlockQ4_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_K*, float*) noexcept {}
void gemm_q4_K_r8(const BlockQ4_Kx8*, int64_t, int64_t, int64_t, const BlockQ8_Kx4*, float*, int64_t) noexcept {}

#endif

}  // namespace liyab::quant
