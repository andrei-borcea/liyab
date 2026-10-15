// Liyab — x86-64 AVX2 kernels (internal; quant_x86.cpp).
//
// One binary runs on every x86-64 CPU: these kernels are compiled for
// AVX2 + FMA + F16C with per-function target attributes (the rest of the
// library keeps the x86-64 baseline), and the public entry points of
// core/quant.h call them only when avx2_enabled() is true, decided once from
// CPUID (core/cpu_features.h). The function names and contracts are those of
// the public kernels they stand in for; only the differences are noted here.
//
// Invariants they keep (tests/test_engine.cpp):
//  * a row's result does not depend on the batch: Q8_0 and the Q8_K K-quants
//    take each block's exact integer sum and the shared float epilogues of
//    quant.h in block order, in the single-row and the multi-row kernels alike;
//  * activation quantization (Q8_0, Q8_K) and every decode (F16, BF16, Q8_0,
//    Q4_0, Q4_1) give the same bits as the scalar code;
//  * the other dots are deterministic (a fixed lane grouping), but round
//    differently from the scalar and NEON kernels: their own golden hashes.
#ifndef LIYAB_CORE_QUANT_X86_H
#define LIYAB_CORE_QUANT_X86_H

#include <cstdint>

#include "core/quant.h"

#if !defined(LIYAB_USE_AVX2)
#error "core/quant_x86.h is only for builds with LIYAB_USE_AVX2"
#endif

namespace liyab::quant::x86 {

// Whether the AVX2 kernels are in use: the CPU (and OS) support AVX2, FMA
// and F16C, the environment variable LIYAB_KERNELS is not "scalar", and
// set_avx2_enabled(false) was not called. Decided on first use, then a
// relaxed atomic load.
bool avx2_enabled() noexcept;
// Whether this CPU could run them (CPUID, whatever LIYAB_KERNELS says).
bool avx2_supported() noexcept;
// Switches the kernel set (ignored when unsupported). Tests and benchmarks
// only: callers must not run kernels concurrently on other threads.
void set_avx2_enabled(bool on) noexcept;

// Activations and decodes: bit-identical to the scalar code.
void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) noexcept;
void quantize_row_q8_K(const float* x, BlockQ8_K* y, int64_t n) noexcept;
void fp32_to_fp16_row(const float* x, uint16_t* y, int64_t n) noexcept;
void fp16_to_fp32_row(const uint16_t* x, float* y, int64_t n) noexcept;
void bf16_to_fp32_row(const uint16_t* x, float* y, int64_t n) noexcept;
void dequantize_q8_0(const BlockQ8_0* x, float* y, int64_t n) noexcept;
void dequantize_q4_0(const BlockQ4_0* x, float* y, int64_t n) noexcept;
void dequantize_q4_1(const BlockQ4_1* x, float* y, int64_t n) noexcept;

// Float dots (8 lanes, fixed grouping).
float dot_f32(const float* a, const float* b, int64_t n) noexcept;
float dot_f16_f32(const uint16_t* w, const float* x, int64_t n) noexcept;
float dot_bf16_f32(const uint16_t* w, const float* x, int64_t n) noexcept;

// Against Q8_0 activations.
float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* x, int64_t n) noexcept;
float dot_q4_1_q8_0(const BlockQ4_1* w, const BlockQ8_0* x, int64_t n) noexcept;
// Q5_0 (`symmetric`) or Q5_1 rows; the activation block sums are recomputed.
float dot_q5_q8_0(bool symmetric, const void* row, const BlockQ8_0* x, int64_t n) noexcept;
float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, int64_t n) noexcept;
void dot_q8_0_q8_0_rows(const BlockQ8_0* w, const BlockQ8_0* const* x, int32_t rows, int64_t n, float* out) noexcept;
float dot_iq4_nl_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept;
float dot_mxfp4_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept;

// Against Q8_K activations: the types for which has_q8_K_kernel() is true.
bool has_q8_K_kernel(DType type) noexcept;
float dot_q8_K(DType type, const void* row, const BlockQ8_K* x, int64_t n) noexcept;
void dot_q8_K_rows(DType type, const void* row, const BlockQ8_K* const* x, int32_t rows, int64_t n,
                   float* out) noexcept;

}  // namespace liyab::quant::x86

#endif  // LIYAB_CORE_QUANT_X86_H
