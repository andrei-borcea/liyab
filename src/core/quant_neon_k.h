// Liyab — NEON helpers shared by the Q4_K / Q5_K kernels (internal).
//
// Used by the dot-product kernels (quant_lowbit.cpp) and the i8mm tile
// kernels (quant_i8mm.cpp), which are compiled with different target flags.
// Adapted from llama.cpp (ggml), ggml/src/ggml-cpu/arch/arm/quants.c
// (MIT, see THIRD_PARTY_NOTICES.md).
#ifndef LIYAB_CORE_QUANT_NEON_K_H
#define LIYAB_CORE_QUANT_NEON_K_H

#if defined(__ARM_NEON) && defined(__aarch64__)

#include <arm_neon.h>

#include <cstdint>
#include <cstring>

namespace liyab::quant::kneon {

constexpr uint32_t kKMask1 = 0x3f3f3f3f, kKMask2 = 0x0f0f0f0f, kKMask3 = 0x03030303;

// The 12 packed scale bytes of Q4_K / Q5_K -> 8 six-bit scales (bytes 0-7 of
// `scales`) and 8 six-bit mins (bytes 0-7 of `mins`), as llama.cpp's utmp.
inline void unpack_k4_scales(const uint8_t* packed, uint8x8_t& scales, uint8x8_t& mins) noexcept {
    uint32_t u[3];
    std::memcpy(u, packed, 12);
    const uint32_t mins_lo = u[1] & kKMask1;
    const uint32_t mins_hi = ((u[2] >> 4) & kKMask2) | (((u[1] >> 6) & kKMask3) << 4);
    const uint32_t scales_hi = (u[2] & kKMask2) | (((u[0] >> 6) & kKMask3) << 4);
    const uint32_t scales_lo = u[0] & kKMask1;
    scales = vcreate_u8(static_cast<uint64_t>(scales_lo) | (static_cast<uint64_t>(scales_hi) << 32));
    mins = vcreate_u8(static_cast<uint64_t>(mins_lo) | (static_cast<uint64_t>(mins_hi) << 32));
}

// sum over the 8 sub-blocks of 32 values of min * (activation sum).
inline int32_t k4_min_sum(uint8x8_t mins, const int16_t* bsums) noexcept {
    const int16x8_t sums32 = vpaddq_s16(vld1q_s16(bsums), vld1q_s16(bsums + 8));
    const int16x8_t m = vreinterpretq_s16_u16(vmovl_u8(mins));
    const int32x4_t prod = vaddq_s32(vmull_s16(vget_low_s16(sums32), vget_low_s16(m)),
                                     vmull_s16(vget_high_s16(sums32), vget_high_s16(m)));
    return vaddvq_s32(prod);
}

// fp16 -> fp32 inline (the out-of-line fp16_to_fp32 is a call per block).
inline float h2f(uint16_t h) noexcept {
    __fp16 v;
    std::memcpy(&v, &h, sizeof v);
    return static_cast<float>(v);
}

}  // namespace liyab::quant::kneon

#endif

#endif  // LIYAB_CORE_QUANT_NEON_K_H
