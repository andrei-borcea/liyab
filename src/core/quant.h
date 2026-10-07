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
struct BlockQ8_0 {
    uint16_t d;              // fp16 scale
    int8_t qs[kBlock];
};
static_assert(sizeof(BlockQ4_0) == 18, "Q4_0 block must match GGML layout");
static_assert(sizeof(BlockQ4_1) == 20, "Q4_1 block must match GGML layout");
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

// Dot of a quantized row (Q4_0 / Q4_1 / Q8_0) with a Q8_0 activation row.
float dot_quantized(DType type, const void* row, const BlockQ8_0* x, int64_t n) noexcept;
[[nodiscard]] constexpr bool is_block_quantized(DType type) noexcept {
    return type == DType::Q4_0 || type == DType::Q4_1 || type == DType::Q8_0;
}

}  // namespace liyab::quant

#endif  // LIYAB_CORE_QUANT_H
