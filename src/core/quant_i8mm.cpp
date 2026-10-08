// Liyab — 2x2 tile kernels with the Armv8.6 int8 matrix multiply (i8mm).
//
// SMMLA (vmmlaq_s32) multiplies a 2x8 int8 matrix by an 8x2 one into a 2x2
// int32 tile: 32 multiply-adds per instruction, twice SDOT's 16. A tile here is
// two weight rows against two activation rows, so a batched matmul
// (speculative verification, prefill) does half the integer work per output of
// the dot-product kernels. Each 16-value vector of a row is split into two
// 8-value halves; zip1/zip2 on 64-bit lanes pair the halves of the two rows
// into the 2x8 operands. Sub-block scales differ per weight row, so they are
// applied to the tile as {sA, sA, sB, sB}. The integer sums equal the
// dot-product kernels' and the float epilogue repeats their expression per
// output, so a batch gives bit-identical results to one row at a time
// (speculative verification must agree with plain decoding). Q8_0 accumulates
// per lane in float in its dot kernel, which a tile cannot reproduce, so it
// keeps the multi-row dot-product kernel.
//
// This file is compiled with +i8mm (see CMakeLists.txt); callers must check
// that the CPU has the extension before calling (CpuBackend does, once).
// The scale and minimum handling follows llama.cpp's ggml_vec_dot_*_q8_K
// (ggml/src/ggml-cpu/arch/arm/quants.c, MIT, see THIRD_PARTY_NOTICES.md).
#include "core/quant.h"

#include <utility>

#if defined(__ARM_NEON) && defined(__aarch64__) && defined(__ARM_FEATURE_MATMUL_INT8)
#include <arm_neon.h>

#include "core/quant_neon_k.h"
#define LIYAB_I8MM 1
#endif

namespace liyab::quant {

#if defined(LIYAB_I8MM)

namespace {

using kneon::h2f;
using kneon::k4_min_sum;
using kneon::unpack_k4_scales;

// acc += [a·x, a·y, b·x, b·y] over 16 values: a, b weight rows, x, y activation rows.
inline int32x4_t tile16(int32x4_t acc, int8x16_t a, int8x16_t b, int8x16_t x, int8x16_t y) noexcept {
    const int64x2_t a64 = vreinterpretq_s64_s8(a), b64 = vreinterpretq_s64_s8(b);
    const int64x2_t x64 = vreinterpretq_s64_s8(x), y64 = vreinterpretq_s64_s8(y);
    acc = vmmlaq_s32(acc, vreinterpretq_s8_s64(vzip1q_s64(a64, b64)), vreinterpretq_s8_s64(vzip1q_s64(x64, y64)));
    return vmmlaq_s32(acc, vreinterpretq_s8_s64(vzip2q_s64(a64, b64)), vreinterpretq_s8_s64(vzip2q_s64(x64, y64)));
}

inline int32x4_t pair_scale(uint8_t a, uint8_t b) noexcept {
    const int32_t v[4] = {a, a, b, b};
    return vld1q_s32(v);
}

// Q4_K / Q5_K: the two weight rows decoded into 4 vectors of 16 values per
// 64-value chunk (lo0, lo1 = sub-block 2j; hi0, hi1 = sub-block 2j + 1).
struct Chunk {
    int8x16_t lo0, lo1, hi0, hi1;
};

inline Chunk q4_chunk(const uint8_t* q4, const uint8x16_t m4) noexcept {
    const uint8x16x2_t bits = vld1q_u8_x2(q4);
    return {vreinterpretq_s8_u8(vandq_u8(bits.val[0], m4)), vreinterpretq_s8_u8(vandq_u8(bits.val[1], m4)),
            vreinterpretq_s8_u8(vshrq_n_u8(bits.val[0], 4)), vreinterpretq_s8_u8(vshrq_n_u8(bits.val[1], 4))};
}

inline Chunk q5_chunk(const uint8_t* q5, uint8x16x2_t& qh, const uint8x16_t m4) noexcept {
    const uint8x16_t mone = vdupq_n_u8(1), mtwo = vdupq_n_u8(2);
    const uint8x16x2_t bits = vld1q_u8_x2(q5);
    Chunk c{vreinterpretq_s8_u8(vorrq_u8(vandq_u8(bits.val[0], m4), vshlq_n_u8(vandq_u8(mone, qh.val[0]), 4))),
            vreinterpretq_s8_u8(vorrq_u8(vandq_u8(bits.val[1], m4), vshlq_n_u8(vandq_u8(mone, qh.val[1]), 4))),
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(bits.val[0], 4), vshlq_n_u8(vandq_u8(mtwo, qh.val[0]), 3))),
            vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(bits.val[1], 4), vshlq_n_u8(vandq_u8(mtwo, qh.val[1]), 3)))};
    qh.val[0] = vshrq_n_u8(qh.val[0], 2);
    qh.val[1] = vshrq_n_u8(qh.val[1], 2);
    return c;
}

// Shared Q4_K / Q5_K tile: `decode(row, i, j)` yields the Chunk of weight row
// `row`, super-block i, chunk j.
template <typename Block, typename Decode>
void k4_tile(const Block* wa, const Block* wb, const BlockQ8_K* x, const BlockQ8_K* y, int64_t nb, Decode decode,
             float* out) noexcept {
    float sum[4] = {};
    for (int64_t i = 0; i < nb; ++i) {
        uint8x8_t sca, mna, scb, mnb;
        unpack_k4_scales(wa[i].scales, sca, mna);
        unpack_k4_scales(wb[i].scales, scb, mnb);
        uint8_t sa[8], sb[8];
        vst1_u8(sa, sca);
        vst1_u8(sb, scb);
        int32x4_t acc = vdupq_n_s32(0);
        auto chunks = decode(i);  // per-super-block decoder state (Q5_K high bits)
        for (int j = 0; j < 4; ++j) {
            const auto [ca, cb] = chunks(j);
            const int8x16x4_t qx = vld1q_s8_x4(x[i].qs + 64 * j);
            const int8x16x4_t qy = vld1q_s8_x4(y[i].qs + 64 * j);
            int32x4_t s = tile16(tile16(vdupq_n_s32(0), ca.lo0, cb.lo0, qx.val[0], qy.val[0]), ca.lo1, cb.lo1,
                                 qx.val[1], qy.val[1]);
            acc = vmlaq_s32(acc, s, pair_scale(sa[2 * j], sb[2 * j]));
            s = tile16(tile16(vdupq_n_s32(0), ca.hi0, cb.hi0, qx.val[2], qy.val[2]), ca.hi1, cb.hi1, qx.val[3],
                       qy.val[3]);
            acc = vmlaq_s32(acc, s, pair_scale(sa[2 * j + 1], sb[2 * j + 1]));
        }
        // The integer sums equal the single-row kernel's; the float epilogue
        // is the same expression per lane, so results are bit-identical to it.
        int32_t lane[4];
        vst1q_s32(lane, acc);
        const float d[2] = {h2f(wa[i].d), h2f(wb[i].d)};
        const float dmin[2] = {h2f(wa[i].dmin), h2f(wb[i].dmin)};
        const uint8x8_t mins[2] = {mna, mnb};
        const BlockQ8_K* act[2] = {&x[i], &y[i]};
        for (int l = 0; l < 4; ++l) {
            const int r = l / 2, c = l % 2;
            sum[l] += act[c]->d * (d[r] * static_cast<float>(lane[l]) -
                                   dmin[r] * static_cast<float>(k4_min_sum(mins[r], act[c]->bsums)));
        }
    }
    for (int l = 0; l < 4; ++l) out[l] = sum[l];
}

void q6_tile(const BlockQ6_K* wa, const BlockQ6_K* wb, const BlockQ8_K* x, const BlockQ8_K* y, int64_t nb,
             float* out) noexcept {
    const uint8x16_t m4 = vdupq_n_u8(0xF), m3 = vdupq_n_u8(3);
    float sum[4] = {};
    auto decode = [&](const BlockQ6_K& b, int j, int8x16_t* w) {
        const uint8x16x2_t hb = vld1q_u8_x2(b.qh + 32 * j);
        const uint8x16x4_t lb = vld1q_u8_x4(b.ql + 64 * j);
        w[0] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(lb.val[0], m4), vshlq_n_u8(vandq_u8(hb.val[0], m3), 4)));
        w[1] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(lb.val[1], m4), vshlq_n_u8(vandq_u8(hb.val[1], m3), 4)));
        w[2] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(lb.val[2], m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hb.val[0], 2), m3), 4)));
        w[3] = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(lb.val[3], m4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hb.val[1], 2), m3), 4)));
        w[4] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(lb.val[0], 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hb.val[0], 4), m3), 4)));
        w[5] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(lb.val[1], 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(hb.val[1], 4), m3), 4)));
        w[6] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(lb.val[2], 4), vshlq_n_u8(vshrq_n_u8(hb.val[0], 6), 4)));
        w[7] = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8(lb.val[3], 4), vshlq_n_u8(vshrq_n_u8(hb.val[1], 6), 4)));
    };
    // sum of scale * (16-value activation sum) over a super-block: the q + 32 offset term.
    auto min_sum = [](const int8_t* sc, const int16_t* bsums) {
        int32_t s = 0;
        for (int k = 0; k < 16; ++k) s += sc[k] * bsums[k];
        return s;
    };
    for (int64_t i = 0; i < nb; ++i) {
        int32x4_t acc = vdupq_n_s32(0);
        for (int j = 0; j < 2; ++j) {
            int8x16_t a[8], b[8];
            decode(wa[i], j, a);
            decode(wb[i], j, b);
            for (int k = 0; k < 8; ++k) {
                const int8x16_t qx = vld1q_s8(x[i].qs + 128 * j + 16 * k);
                const int8x16_t qy = vld1q_s8(y[i].qs + 128 * j + 16 * k);
                const int32_t sca = wa[i].scales[8 * j + k], scb = wb[i].scales[8 * j + k];
                const int32_t sv[4] = {sca, sca, scb, scb};
                acc = vmlaq_s32(acc, tile16(vdupq_n_s32(0), a[k], b[k], qx, qy), vld1q_s32(sv));
            }
        }
        int32_t lane[4];
        vst1q_s32(lane, acc);
        const int32_t mins[4] = {min_sum(wa[i].scales, x[i].bsums), min_sum(wa[i].scales, y[i].bsums),
                                 min_sum(wb[i].scales, x[i].bsums), min_sum(wb[i].scales, y[i].bsums)};
        const float d[2] = {h2f(wa[i].d), h2f(wb[i].d)};
        const BlockQ8_K* act[2] = {&x[i], &y[i]};
        for (int l = 0; l < 4; ++l) {  // the single-row kernel's expression: bit-identical results
            sum[l] += act[l % 2]->d * d[l / 2] * static_cast<float>(lane[l] - 32 * mins[l]);
        }
    }
    for (int l = 0; l < 4; ++l) out[l] = sum[l];
}

}  // namespace

bool i8mm_kernels_compiled() noexcept { return true; }

bool dot_q8_K_2x2(DType type, const void* wa, const void* wb, const BlockQ8_K* x, const BlockQ8_K* y, int64_t n,
                  float* out) noexcept {
    const int64_t nb = n / kSuperBlock;
    const uint8x16_t m4 = vdupq_n_u8(0xF);
    switch (type) {
        case DType::Q4_K: {
            const auto* a = static_cast<const BlockQ4_K*>(wa);
            const auto* b = static_cast<const BlockQ4_K*>(wb);
            k4_tile(a, b, x, y, nb,
                    [&](int64_t i) {
                        return [&, i](int j) {
                            return std::pair{q4_chunk(a[i].qs + 32 * j, m4), q4_chunk(b[i].qs + 32 * j, m4)};
                        };
                    },
                    out);
            return true;
        }
        case DType::Q5_K: {
            const auto* a = static_cast<const BlockQ5_K*>(wa);
            const auto* b = static_cast<const BlockQ5_K*>(wb);
            k4_tile(a, b, x, y, nb,
                    [&](int64_t i) {
                        // The high bits shift by 2 per chunk, so each row keeps its own state.
                        return [&, i, ha = vld1q_u8_x2(a[i].qh), hb = vld1q_u8_x2(b[i].qh)](int j) mutable {
                            return std::pair{q5_chunk(a[i].qs + 32 * j, ha, m4), q5_chunk(b[i].qs + 32 * j, hb, m4)};
                        };
                    },
                    out);
            return true;
        }
        case DType::Q6_K:
            q6_tile(static_cast<const BlockQ6_K*>(wa), static_cast<const BlockQ6_K*>(wb), x, y, nb, out);
            return true;
        default:
            return false;
    }
}

#else

bool i8mm_kernels_compiled() noexcept { return false; }

bool dot_q8_K_2x2(DType, const void*, const void*, const BlockQ8_K*, const BlockQ8_K*, int64_t, float*) noexcept {
    return false;
}

#endif

}  // namespace liyab::quant
