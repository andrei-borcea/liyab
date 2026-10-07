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

float bf16_to_fp32(uint16_t h) noexcept;
float dot_bf16_f32(const uint16_t* w, const float* x, int64_t n) noexcept;

// Unpacks one K-quant super-block into 256 int8 values with per-16 scales
// and per-32 minimums: value = scale16[i / 16] * q[i] - min32[i / 32].
void unpack_k_block(DType type, const void* block, int8_t* q, float* scale16, float* min32) noexcept;
// Unpacks one Q5_0 / Q5_1 block into 32 int8 values (Q5_0: -16..15, Q5_1: 0..31).
void unpack_q5_block(const uint8_t* qh, const uint8_t* qs, bool symmetric, int8_t* q) noexcept;

}  // namespace liyab::quant

#endif  // LIYAB_CORE_QUANT_H
