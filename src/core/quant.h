// Liyab — block quantization formats and vector kernels (internal).
//
// Block layouts are bit-identical to GGML's Q4_0 / Q8_0 so GGUF tensors are
// consumed straight from the mapped file. The same routines quantize the KV
// cache, which keeps a single, tested implementation of every format.
#ifndef LIYAB_CORE_QUANT_H
#define LIYAB_CORE_QUANT_H

#include <cstdint>
#include <cstring>

#include "liyab/types.h"

namespace liyab::quant {

inline constexpr int kBlock = 32;

struct BlockQ4_0 {
    uint16_t d;              // fp16 scale
    uint8_t qs[kBlock / 2];  // element j in low nibble of qs[j], element j+16 in high nibble
};
struct BlockQ4_1 {
    uint16_t d;              // fp16 scale
    uint16_t m;              // fp16 minimum: x = q * d + m, q in [0, 15]
    uint8_t qs[kBlock / 2];  // same nibble order as Q4_0
};
struct BlockQ5_0 {
    uint16_t d;              // fp16 scale; x = (q - 16) * d
    uint8_t qh[4];           // bit j = 5th bit of element j (little-endian uint32)
    uint8_t qs[kBlock / 2];  // low 4 bits, Q4_0 nibble order
};
struct BlockQ5_1 {
    uint16_t d;              // x = q * d + m
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[kBlock / 2];
};
struct BlockQ8_0 {
    uint16_t d;              // fp16 scale
    int8_t qs[kBlock];
};
inline constexpr int kSuperBlock = 256;  // K-quants

struct BlockQ4_K {
    uint16_t d;           // fp16 super-block scale for the sub-block scales
    uint16_t dmin;        // fp16 super-block scale for the sub-block mins
    uint8_t scales[12];   // 8 x (6-bit scale, 6-bit min), packed
    uint8_t qs[128];      // chunk c (64 values): low nibbles = sub-block 2c, high = 2c+1
};
struct BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];       // 5th bit: bit 2c (low) / 2c+1 (high) of qh[l] for chunk c
    uint8_t qs[128];
};
struct BlockQ6_K {
    uint8_t ql[128];      // low 4 bits
    uint8_t qh[64];       // high 2 bits
    int8_t scales[16];    // one per 16 values
    uint16_t d;           // fp16
};
static_assert(sizeof(BlockQ4_K) == 144, "Q4_K block must match GGML layout");
static_assert(sizeof(BlockQ5_K) == 176, "Q5_K block must match GGML layout");
static_assert(sizeof(BlockQ6_K) == 210, "Q6_K block must match GGML layout");
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block must match GGML layout");
static_assert(sizeof(BlockQ4_1) == 20, "Q4_1 block must match GGML layout");
static_assert(sizeof(BlockQ5_0) == 22, "Q5_0 block must match GGML layout");
static_assert(sizeof(BlockQ5_1) == 24, "Q5_1 block must match GGML layout");
static_assert(sizeof(BlockQ8_0) == 34, "Q8_0 block must match GGML layout");

float fp16_to_fp32(uint16_t h) noexcept;
uint16_t fp32_to_fp16(float f) noexcept;

// Row conversions. `n` must be a multiple of kBlock for quantized types.
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) noexcept;
void quantize_row_q4_0(const float* x, BlockQ4_0* y, int64_t n) noexcept;
void quantize_row_q4_1(const float* x, BlockQ4_1* y, int64_t n) noexcept;
// K-quants with 6-bit sub-block scales and mins: llama.cpp's
// quantize_row_q4_K_ref / quantize_row_q5_K_ref (identical blocks when both
// are compiled without FMA contraction; otherwise last-bit differences in a
// few scales).
// `n` must be a multiple of kSuperBlock. Slow (a weighted search per
// sub-block): meant for converting weights at load, not for activations.
void quantize_row_q4_K(const float* x, BlockQ4_K* y, int64_t n) noexcept;
void quantize_row_q5_K(const float* x, BlockQ5_K* y, int64_t n) noexcept;
// Writes a row of `type` from floats into `dst`.
void quantize_row(DType type, const float* x, void* dst, int64_t n) noexcept;
void dequantize_row(DType type, const void* src, float* y, int64_t n) noexcept;

// Dot product of one weight row with an activation row. For quantized weights
// the activation must be pre-quantized to Q8_0 (`xq`); for F32/F16 weights the
// float activation (`xf`) is used. This mirrors how integer dot-product units
// (NEON SDOT) are fed.
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* x, int64_t n) noexcept;
float dot_q4_1_q8_0(const BlockQ4_1* w, const BlockQ8_0* x, int64_t n) noexcept;
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, int64_t n) noexcept;
// The same Q8_0 weight row against `rows` activation rows (x[r] = row r's
// blocks): out[r] = dot_q8_0_q8_0(w, x[r], n), up to float summation order.
void dot_q8_0_q8_0_rows(const BlockQ8_0* w, const BlockQ8_0* const* x, int32_t rows, int64_t n, float* out) noexcept;
float dot_f16_f32(const uint16_t* w, const float* x, int64_t n) noexcept;
float dot_f32(const float* a, const float* b, int64_t n) noexcept;

// y += a * dequant(row)
void axpy_row(DType type, const void* row, float a, float* y, int64_t n) noexcept;

// Dot of a quantized row with a Q8_0 activation row. K-quants also need
// `xsums`: the sum of the int8 values of every activation block (for the
// sub-block minimums); other types accept nullptr.
float dot_quantized(DType type, const void* row, const BlockQ8_0* x, const int32_t* xsums, int64_t n) noexcept;
// Fills xsums[i] = sum of x[i].qs for `n / 32` blocks.
void block_sums(const BlockQ8_0* x, int32_t* xsums, int64_t n) noexcept;

[[nodiscard]] constexpr bool is_k_quant(DType type) noexcept {
    return type == DType::Q4_K || type == DType::Q5_K || type == DType::Q6_K;
}
// Extended formats (quant_ext.cpp): Q2_K, Q3_K, I-quants, TQ1_0/TQ2_0, MXFP4, NVFP4.
bool is_extended(DType type) noexcept;

[[nodiscard]] inline bool is_block_quantized(DType type) noexcept {
    return type == DType::Q4_0 || type == DType::Q4_1 || type == DType::Q5_0 || type == DType::Q5_1 ||
           type == DType::Q8_0 || is_k_quant(type) || is_extended(type);
}

// Common decoded form of one extended block (up to 256 values):
//   x[e] = scale16[e / 16] * q[e] + (has_bias ? bias8[e / 8] : 0)
struct ExtBlock {
    alignas(16) int8_t q[256];
    float scale16[16];
    float bias8[32];
    bool has_bias;
};
bool unpack_ext_block(DType type, const uint8_t* block, ExtBlock& out) noexcept;
void dequantize_ext_row(DType type, const void* src, float* y, int64_t n) noexcept;
float dot_ext_q8_0(DType type, const void* row, const BlockQ8_0* x, int64_t n) noexcept;
// MXFP4 block scale: 2^(e - 127) / 2 (the e2m1 grid in kMxfp4Values is doubled).
float e8m0_to_fp32_half(uint8_t e) noexcept;

// ---------------------------------------------------------------------------
// Q8_K activations and the low-bit dot kernels (quant_lowbit.cpp).
//
// The 2/3-bit K-quants and the 256-value I-quants pack their values in ways
// that cost more to decode than to multiply, so their fast kernels (ported
// from llama.cpp's ARM NEON code) consume activations quantized per 256-value
// super-block instead of per 32 values: one float scale, so all 8 sub-blocks
// of a weight super-block accumulate in int32 and are scaled once, plus int16
// sums per 16 values that fold the sub-block minimums / IQ1 deltas in without
// touching the weights.
// ---------------------------------------------------------------------------

// Bit-identical to GGML's block_q8_K: x[e] ~= d * qs[e]; bsums[j] = sum of qs[16j .. 16j+15].
struct BlockQ8_K {
    float d;
    int8_t qs[kSuperBlock];
    int16_t bsums[kSuperBlock / 16];
};
static_assert(sizeof(BlockQ8_K) == 292, "Q8_K block must match GGML layout");

// Quantizes `n` floats (a multiple of 256) like llama.cpp's quantize_row_q8_K_ref:
// d = -max/127 where max is the signed value of largest magnitude, values are
// rounded to nearest-even and clamped to [-127, 127]. An all-zero block gets
// d = 0, zero values and zero sums.
void quantize_row_q8_K(const float* x, BlockQ8_K* y, int64_t n) noexcept;

// True when `type` has a dedicated Q8_K dot kernel on this build (AArch64 NEON):
// Q2_K to Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS and
// TQ2_0. Callers should then quantize activations with quantize_row_q8_K and
// call dot_lowbit_q8_K; other types keep the Q8_0 path (dot_quantized). The
// 4-6 bit K-quants are cheap to decode, but per 32 values the Q8_0 path pays a
// horizontal add and two fp16 conversions; per 256 they accumulate in int32.
bool uses_q8_K(DType type) noexcept;

// Dot of one weight row of `type` (any type listed for uses_q8_K) with a Q8_K
// activation row; `n` is a multiple of 256. Available on every target: without
// NEON it decodes through unpack_ext_block (correct, not fast). Thread-safe
// (pure function of its inputs). Returns 0 for unsupported types.
float dot_lowbit_q8_K(DType type, const void* row, const BlockQ8_K* x, int64_t n) noexcept;
// The same weight row against `rows` activation rows (x[r] = row r's
// blocks): out[r] = dot_lowbit_q8_K(type, row, x[r], n), up to float
// summation order. Q4_K / Q5_K / Q6_K decode each super-block once for up to
// four rows (speculative verification, prefill); other types loop.
void dot_lowbit_q8_K_rows(DType type, const void* row, const BlockQ8_K* const* x, int32_t rows, int64_t n,
                          float* out) noexcept;

// NEON kernels for the 32-value I-quant / FP4 formats against Q8_0 activations
// (dispatched by dot_quantized). `n` is a multiple of 32.
float dot_iq4_nl_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept;
float dot_mxfp4_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept;

float bf16_to_fp32(uint16_t h) noexcept;
float dot_bf16_f32(const uint16_t* w, const float* x, int64_t n) noexcept;

// Unpacks one K-quant super-block into 256 int8 values with per-16 scales
// and per-32 minimums: value = scale16[i / 16] * q[i] - min32[i / 32].
void unpack_k_block(DType type, const void* block, int8_t* q, float* scale16, float* min32) noexcept;
// Unpacks one Q5_0 / Q5_1 block into 32 int8 values (Q5_0: -16..15, Q5_1: 0..31).
void unpack_q5_block(const uint8_t* qh, const uint8_t* qs, bool symmetric, int8_t* q) noexcept;

}  // namespace liyab::quant

#endif  // LIYAB_CORE_QUANT_H
