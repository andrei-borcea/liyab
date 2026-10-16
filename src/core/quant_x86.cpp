// Liyab — x86-64 AVX2 kernels (AVX2 + FMA + F16C). See core/quant_x86.h.
//
// Why per-function target attributes instead of building this file with
// -mavx2: the inline functions it uses from headers (the epilogues of
// quant.h, std::fma, the standard library) would then be emitted here with
// AVX2 code, and the linker may keep that copy for the whole program, which
// then faults on a CPU without AVX2. With the attribute only the functions
// below use AVX2; everything they inline is compiled for their target.
//
// The int8 dot products are built on VPMADDUBSW (unsigned x signed bytes,
// pairs summed into saturating int16) and VPMADDWD (int16 pairs into int32):
//  * signed x signed (Q8_0, Q4_0, the codebook formats): |w| x (x with w's
//    sign). A pair sums to at most 2 x 128 x 127 = 32512 < 32767 as long as
//    the activation is never -128, which holds for every activation the
//    quantizers here produce (their range is [-127, 127]);
//  * unsigned weights (Q4_1, Q5_1, the K-quants' 4-6 bit values before their
//    offset) multiply the activation directly;
//  * VPMADDWD against a vector of a sub-block's scale applies that scale while
//    widening to int32, so a K-quant super-block accumulates exactly in int32
//    (at most 256 x 63 x 127 x 128, far below 2^31) and is scaled once.
// The choices follow native costs (VPSIGNB and VPSHUFB are single-cycle
// instructions on Haswell and Zen). Rosetta 2, where these kernels were
// tested, translates both into long sequences, so its timings say little.
#include "core/quant_x86.h"

#include <immintrin.h>

#include <atomic>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "core/cpu_features.h"
#include "core/quant_blocks.h"
#include "core/quant_tables.h"

// clang-cl takes the attribute too. cl.exe has no such attribute and needs
// none: it compiles AVX2 intrinsics in any function, while the rest of the
// library is built without /arch:AVX2, at the x86-64 baseline.
#if defined(__GNUC__) || defined(__clang__)
#define LIYAB_AVX2 __attribute__((target("avx2,fma,f16c")))
#else
#define LIYAB_AVX2
#endif

namespace liyab::quant::x86 {

// --- Dispatch state -----------------------------------------------------------

namespace {

bool cpu_supports_avx2() noexcept {
    const X86Simd f = detect_x86_simd();
    return f.avx2 && f.fma && f.f16c;
}

bool scalar_requested() noexcept {
    const char* env = std::getenv("LIYAB_KERNELS");
    return env != nullptr && std::strcmp(env, "scalar") == 0;
}

// -1: not decided yet, 0: scalar, 1: AVX2. Two threads deciding at once
// store the same value.
std::atomic<int> g_state{-1};

}  // namespace

bool avx2_supported() noexcept {
    static const bool supported = cpu_supports_avx2();
    return supported;
}

bool avx2_enabled() noexcept {
    int s = g_state.load(std::memory_order_relaxed);
    if (s < 0) {
        s = avx2_supported() && !scalar_requested() ? 1 : 0;
        g_state.store(s, std::memory_order_relaxed);
    }
    return s != 0;
}

void set_avx2_enabled(bool on) noexcept {
    g_state.store(on && avx2_supported() ? 1 : 0, std::memory_order_relaxed);
}

// --- Helpers --------------------------------------------------------------------

namespace {

#if defined(_MSC_VER) && !defined(__clang__)
// cl.exe declares neither _cvtsh_ss nor _cvtss_sh: the same instructions
// (VCVTPH2PS, VCVTPS2PH) through their vector forms, on lane 0.
inline float h2f(uint16_t h) noexcept { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(h))); }
inline uint16_t f2h(float x) noexcept {
    return static_cast<uint16_t>(_mm_extract_epi16(_mm_cvtps_ph(_mm_set_ss(x), _MM_FROUND_TO_NEAREST_INT), 0));
}
#else
LIYAB_AVX2 inline float h2f(uint16_t h) noexcept { return _cvtsh_ss(h); }
LIYAB_AVX2 inline uint16_t f2h(float x) noexcept {
    return static_cast<uint16_t>(_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT));
}
#endif

LIYAB_AVX2 inline __m256i load256(const void* p) noexcept {
    return _mm256_loadu_si256(static_cast<const __m256i*>(p));
}
LIYAB_AVX2 inline __m128i load128(const void* p) noexcept {
    return _mm_loadu_si128(static_cast<const __m128i*>(p));
}
LIYAB_AVX2 inline __m256i combine(__m128i lo, __m128i hi) noexcept {
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

// 16 bytes of packed nibbles (GGML order: element j in the low nibble of
// byte j, element j + 16 in its high nibble) -> 32 bytes 0..15 in element order.
LIYAB_AVX2 inline __m256i nibbles32(const uint8_t* qs) noexcept {
    const __m128i v = load128(qs);
    return _mm256_and_si256(combine(v, _mm_srli_epi16(v, 4)), _mm256_set1_epi8(0x0F));
}

LIYAB_AVX2 inline int32_t hsum_i32(__m256i v) noexcept {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

// The four totals of a, b, c, d (8 int32 lanes each), in that order.
LIYAB_AVX2 inline __m128i hsum4_i32(__m256i a, __m256i b, __m256i c, __m256i d) noexcept {
    const __m256i abcd = _mm256_hadd_epi32(_mm256_hadd_epi32(a, b), _mm256_hadd_epi32(c, d));
    return _mm_add_epi32(_mm256_castsi256_si128(abcd), _mm256_extracti128_si256(abcd, 1));
}

// Lane sum in a fixed order (the float dots' only cross-lane rounding).
LIYAB_AVX2 inline float hsum_ps(__m256 v) noexcept {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// 8 bf16 values -> 8 floats (the top halves of their bit patterns).
LIYAB_AVX2 inline __m256 widen_bf16(const uint16_t* p) noexcept {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(load128(p)), 16));
}

// 32 signed weights . 32 signed activations -> 8 int32 lanes (exact; the
// activations must not be -128, see the file comment).
LIYAB_AVX2 inline __m256i dot_i8(__m256i w, __m256i x) noexcept {
    const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_madd_epi16(p, _mm256_set1_epi16(1));
}
// 32 unsigned weights (< 128) . 32 signed activations -> 8 int32 lanes.
LIYAB_AVX2 inline __m256i dot_u8(__m256i w, __m256i x) noexcept {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(w, x), _mm256_set1_epi16(1));
}
// Sum of 32 signed activations in 8 int32 lanes.
LIYAB_AVX2 inline __m256i sum_i8(__m256i x) noexcept { return dot_u8(_mm256_set1_epi8(1), x); }

// Largest |x| of 8 vectors x[0..7] (NaN lanes ignored, as std::max(amax,
// |x|) in the scalar quantizers does: VMAXPS returns its second operand when
// either is NaN, and the running maximum is never NaN).
LIYAB_AVX2 inline float abs_max(const float* x, int64_t count) noexcept {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    __m256 m = _mm256_setzero_ps();
    for (int64_t j = 0; j < count; j += 8) m = _mm256_max_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), abs_mask), m);
    __m128 s = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    s = _mm_max_ps(s, _mm_movehl_ps(s, s));
    s = _mm_max_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// Four vectors of int32 (each in [-128, 127]) -> 32 int8 in element order.
// VPACKS* work within 128-bit lanes, hence the final dword permutation.
LIYAB_AVX2 inline __m256i pack_i32x4(__m256i a, __m256i b, __m256i c, __m256i d) noexcept {
    const __m256i packed = _mm256_packs_epi16(_mm256_packs_epi32(a, b), _mm256_packs_epi32(c, d));
    return _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

}  // namespace

// --- Activation quantization and decoding (bit-identical to the scalar code) ----

LIYAB_AVX2 void quantize_row_q8_0(const float* x, BlockQ8_0* y, int64_t n) noexcept {
    const __m256 sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(static_cast<int>(0x80000000u)));
    const __m256 half = _mm256_set1_ps(0.5f);
    const __m256 one = _mm256_set1_ps(1.0f);
    for (int64_t i = 0; i < n / kBlock; ++i) {
        const float* xb = x + i * kBlock;
        const float amax = abs_max(xb, kBlock);
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        y[i].d = f2h(d);
        // std::lround: to nearest, ties away from zero. VCVTPS2DQ rounds ties
        // to even, so the magnitude is truncated and bumped when the dropped
        // fraction (exact: |v| < 2^24) is at least one half.
        __m256i q[4];
        for (int k = 0; k < 4; ++k) {
            const __m256 v = _mm256_mul_ps(_mm256_loadu_ps(xb + 8 * k), _mm256_set1_ps(id));
            const __m256 a = _mm256_andnot_ps(sign_mask, v);
            __m256 t = _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
            t = _mm256_add_ps(t, _mm256_and_ps(_mm256_cmp_ps(_mm256_sub_ps(a, t), half, _CMP_GE_OQ), one));
            q[k] = _mm256_cvttps_epi32(_mm256_or_ps(t, _mm256_and_ps(v, sign_mask)));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(y[i].qs), pack_i32x4(q[0], q[1], q[2], q[3]));
    }
}

LIYAB_AVX2 void quantize_row_q8_K(const float* x, BlockQ8_K* y, int64_t n) noexcept {
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    for (int64_t i = 0; i < n / kSuperBlock; ++i, x += kSuperBlock) {
        BlockQ8_K& b = y[i];
        const float amax = abs_max(x, kSuperBlock);
        if (amax == 0.0f) {
            b.d = 0.0f;
            std::memset(b.qs, 0, sizeof b.qs);
            std::memset(b.bsums, 0, sizeof b.bsums);
            continue;
        }
        // The first element of that magnitude keeps its sign (as the scalar loop).
        float max = 0.0f;
        const __m256 target = _mm256_set1_ps(amax);
        for (int j = 0; j < kSuperBlock; j += 8) {
            const int hits = _mm256_movemask_ps(
                _mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + j), abs_mask), target, _CMP_EQ_OQ));
            if (hits != 0) {
                max = x[j + std::countr_zero(static_cast<unsigned>(hits))];
                break;
            }
        }
        const float iscale = -127.0f / max;
        const __m256 vs = _mm256_set1_ps(iscale);
        for (int j = 0; j < kSuperBlock; j += 32) {
            // VCVTPS2DQ rounds to nearest-even (default MXCSR), like std::lrint;
            // the saturating packs clamp to [-128, 127] like the scalar code.
            __m256i q[4];
            for (int k = 0; k < 4; ++k) q[k] = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + j + 8 * k), vs));
            const __m256i bytes = pack_i32x4(q[0], q[1], q[2], q[3]);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(b.qs + j), bytes);
            // Sums of the two 16-value halves: lanes 0-3 and 4-7 of sum_i8.
            __m256i s = sum_i8(bytes);
            s = _mm256_hadd_epi32(s, s);
            s = _mm256_hadd_epi32(s, s);
            b.bsums[j / 16] = static_cast<int16_t>(_mm_cvtsi128_si32(_mm256_castsi256_si128(s)));
            b.bsums[j / 16 + 1] = static_cast<int16_t>(_mm_cvtsi128_si32(_mm256_extracti128_si256(s, 1)));
        }
        b.d = 1.0f / iscale;
    }
}

LIYAB_AVX2 void fp32_to_fp16_row(const float* x, uint16_t* y, int64_t n) noexcept {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        _mm_storeu_si128(reinterpret_cast<__m128i*>(y + i),
                         _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
    }
    for (; i < n; ++i) y[i] = f2h(x[i]);
}

LIYAB_AVX2 void fp16_to_fp32_row(const uint16_t* x, float* y, int64_t n) noexcept {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_cvtph_ps(load128(x + i)));
    for (; i < n; ++i) y[i] = h2f(x[i]);
}

LIYAB_AVX2 void bf16_to_fp32_row(const uint16_t* x, float* y, int64_t n) noexcept {
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, widen_bf16(x + i));
    for (; i < n; ++i) y[i] = bf16_to_fp32(x[i]);
}

LIYAB_AVX2 void dequantize_q8_0(const BlockQ8_0* x, float* y, int64_t n) noexcept {
    for (int64_t i = 0; i < n / kBlock; ++i, y += kBlock) {
        const __m256 d = _mm256_set1_ps(h2f(x[i].d));
        for (int k = 0; k < 4; ++k) {
            const __m256i q = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(x[i].qs + 8 * k)));
            _mm256_storeu_ps(y + 8 * k, _mm256_mul_ps(d, _mm256_cvtepi32_ps(q)));
        }
    }
}

namespace {

// 32 bytes -> 32 floats d * (byte - offset) [+ m].
LIYAB_AVX2 inline void store_bytes_scaled(__m256i bytes, int offset, __m256 d, const __m256* m, float* y) noexcept {
    const __m128i lo = _mm256_castsi256_si128(bytes);
    const __m128i hi = _mm256_extracti128_si256(bytes, 1);
    const __m128i parts[4] = {lo, _mm_srli_si128(lo, 8), hi, _mm_srli_si128(hi, 8)};
    for (int k = 0; k < 4; ++k) {
        const __m256i q = _mm256_sub_epi32(_mm256_cvtepu8_epi32(parts[k]), _mm256_set1_epi32(offset));
        __m256 v = _mm256_mul_ps(d, _mm256_cvtepi32_ps(q));
        if (m != nullptr) v = _mm256_add_ps(v, *m);  // mul then add, as the scalar decode (no fusion)
        _mm256_storeu_ps(y + 8 * k, v);
    }
}

}  // namespace

LIYAB_AVX2 void dequantize_q4_0(const BlockQ4_0* x, float* y, int64_t n) noexcept {
    for (int64_t i = 0; i < n / kBlock; ++i, y += kBlock) {
        store_bytes_scaled(nibbles32(x[i].qs), 8, _mm256_set1_ps(h2f(x[i].d)), nullptr, y);
    }
}

LIYAB_AVX2 void dequantize_q4_1(const BlockQ4_1* x, float* y, int64_t n) noexcept {
    for (int64_t i = 0; i < n / kBlock; ++i, y += kBlock) {
        const __m256 m = _mm256_set1_ps(h2f(x[i].m));
        store_bytes_scaled(nibbles32(x[i].qs), 0, _mm256_set1_ps(h2f(x[i].d)), &m, y);
    }
}

// --- Float dots -------------------------------------------------------------------

LIYAB_AVX2 float dot_f32(const float* a, const float* b, int64_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    int64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    float sum = hsum_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for (; i < n; ++i) sum = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a[i]), _mm_set_ss(b[i]), _mm_set_ss(sum)));
    return sum;
}

LIYAB_AVX2 float dot_f16_f32(const uint16_t* w, const float* x, int64_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(load128(w + i)), _mm256_loadu_ps(x + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_cvtph_ps(load128(w + i + 8)), _mm256_loadu_ps(x + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(_mm256_cvtph_ps(load128(w + i)), _mm256_loadu_ps(x + i), acc0);
    float sum = hsum_ps(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) sum = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(h2f(w[i])), _mm_set_ss(x[i]), _mm_set_ss(sum)));
    return sum;
}

LIYAB_AVX2 float dot_bf16_f32(const uint16_t* w, const float* x, int64_t n) noexcept {
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0;
    int64_t i = 0;
    for (; i + 16 <= n; i += 16) {
        acc0 = _mm256_fmadd_ps(widen_bf16(w + i), _mm256_loadu_ps(x + i), acc0);
        acc1 = _mm256_fmadd_ps(widen_bf16(w + i + 8), _mm256_loadu_ps(x + i + 8), acc1);
    }
    for (; i + 8 <= n; i += 8) acc0 = _mm256_fmadd_ps(widen_bf16(w + i), _mm256_loadu_ps(x + i), acc0);
    float sum = hsum_ps(_mm256_add_ps(acc0, acc1));
    for (; i < n; ++i) {
        sum = _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(bf16_to_fp32(w[i])), _mm_set_ss(x[i]), _mm_set_ss(sum)));
    }
    return sum;
}

// --- Q8_0 activations ---------------------------------------------------------------

// Two float accumulators take alternate blocks (two independent FMA chains);
// an odd last block goes to the first. A fixed order for a given n.
LIYAB_AVX2 float dot_q4_0_q8_0(const BlockQ4_0* w, const BlockQ8_0* x, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    const __m256i eight = _mm256_set1_epi8(8);
    __m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int64_t i = 0; i < nb; ++i) {
        const __m256i q = _mm256_sub_epi8(nibbles32(w[i].qs), eight);
        const __m256i s = dot_i8(q, load256(x[i].qs));
        acc[i & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(h2f(w[i].d) * h2f(x[i].d)), acc[i & 1]);
    }
    return hsum_ps(_mm256_add_ps(acc[0], acc[1]));
}

// sum_j (q_j dw + mw)(x_j dx) = dw dx sum(q_j x_j) + mw dx sum(x_j)
LIYAB_AVX2 float dot_q4_1_q8_0(const BlockQ4_1* w, const BlockQ8_0* x, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
        const __m256i xv = load256(x[i].qs);
        const float dx = h2f(x[i].d);
        const __m256i qx = dot_u8(nibbles32(w[i].qs), xv);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(qx), _mm256_set1_ps(h2f(w[i].d) * dx), acc);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum_i8(xv)), _mm256_set1_ps(h2f(w[i].m) * dx), acc);
    }
    return hsum_ps(acc);
}

namespace {

// The 32 fifth bits of a Q5 block -> 0x10 in the bytes whose bit is set:
// byte e takes bit e % 8 of qh byte e / 8.
LIYAB_AVX2 inline __m256i q5_high_bits(const uint8_t* qh_bytes) noexcept {
    uint32_t qh;
    std::memcpy(&qh, qh_bytes, 4);
    const __m256i select = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,  //
                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ull));
    const __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(qh)), select);
    return _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit), bit), _mm256_set1_epi8(0x10));
}

}  // namespace

LIYAB_AVX2 float dot_q5_q8_0(bool symmetric, const void* row, const BlockQ8_0* x, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    __m256 acc = _mm256_setzero_ps();
    if (symmetric) {
        const auto* w = static_cast<const BlockQ5_0*>(row);
        const __m256i sixteen = _mm256_set1_epi8(16);
        for (int64_t i = 0; i < nb; ++i) {
            const __m256i q = _mm256_sub_epi8(_mm256_or_si256(nibbles32(w[i].qs), q5_high_bits(w[i].qh)), sixteen);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_i8(q, load256(x[i].qs))),
                                  _mm256_set1_ps(h2f(w[i].d) * h2f(x[i].d)), acc);
        }
    } else {
        const auto* w = static_cast<const BlockQ5_1*>(row);
        for (int64_t i = 0; i < nb; ++i) {
            const __m256i q = _mm256_or_si256(nibbles32(w[i].qs), q5_high_bits(w[i].qh));
            const __m256i xv = load256(x[i].qs);
            const float dx = h2f(x[i].d);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(dot_u8(q, xv)), _mm256_set1_ps(h2f(w[i].d) * dx), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(sum_i8(xv)), _mm256_set1_ps(h2f(w[i].m) * dx), acc);
        }
    }
    return hsum_ps(acc);
}

// Each block's exact integer sum, then the epilogue every Q8_0 kernel shares
// (quant.h), block after block. Four blocks' lanes are reduced together and
// converted in one instruction: exact (a block sum is far below 2^24), so the
// same floats as one conversion per block.
LIYAB_AVX2 float dot_q8_0_q8_0(const BlockQ8_0* w, const BlockQ8_0* x, int64_t n) noexcept {
    const int64_t nb = n / kBlock;
    float acc = 0.0f;
    int64_t i = 0;
    for (; i + 4 <= nb; i += 4) {
        __m256i s[4];
        for (int k = 0; k < 4; ++k) s[k] = dot_i8(load256(w[i + k].qs), load256(x[i + k].qs));
        alignas(16) float sums[4];
        _mm_store_ps(sums, _mm_cvtepi32_ps(hsum4_i32(s[0], s[1], s[2], s[3])));
        for (int k = 0; k < 4; ++k) acc = epilogue_scaled(acc, h2f(x[i + k].d), h2f(w[i + k].d), sums[k]);
    }
    for (; i < nb; ++i) {
        const float sum = static_cast<float>(hsum_i32(dot_i8(load256(w[i].qs), load256(x[i].qs))));
        acc = epilogue_scaled(acc, h2f(x[i].d), h2f(w[i].d), sum);
    }
    return acc;
}

namespace {

// dot_q8_0_q8_0 for NR <= 8 activation rows: lane r of `acc` runs row r's
// epilogue chain, acc_r = fma(sum_r, dw * d8_r, acc_r) block after block (the
// product epilogue_scaled forms as d * d8: one rounding, commutative), so a
// row gets the same bits as from the single-row kernel. The weight block is
// loaded, and its magnitudes taken, once for all rows.
template <int NR>
LIYAB_AVX2 void dot_q8_0_rows(const BlockQ8_0* w, const BlockQ8_0* const* x, int64_t nb, float* out) noexcept {
    static_assert(NR >= 1 && NR <= 8);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    for (int64_t i = 0; i < nb; ++i) {
        const __m256i a = load256(w[i].qs);
        const __m256i ua = _mm256_sign_epi8(a, a);
        __m256i s[8];
        for (int r = 0; r < 8; ++r) {
            s[r] = r < NR ? _mm256_madd_epi16(_mm256_maddubs_epi16(ua, _mm256_sign_epi8(load256(x[r][i].qs), a)), ones)
                          : _mm256_setzero_si256();
        }
        const __m128i lo = hsum4_i32(s[0], s[1], s[2], s[3]);
        const __m128i hi = NR > 4 ? hsum4_i32(s[4], s[5], s[6], s[7]) : _mm_setzero_si128();
        // Activation scales of the rows (0 for missing rows: their lanes stay 0).
        const __m128i d8h = _mm_setr_epi16(static_cast<int16_t>(x[0][i].d),  //
                                           static_cast<int16_t>(NR > 1 ? x[1][i].d : 0),
                                           static_cast<int16_t>(NR > 2 ? x[2][i].d : 0),
                                           static_cast<int16_t>(NR > 3 ? x[3][i].d : 0),
                                           static_cast<int16_t>(NR > 4 ? x[4][i].d : 0),
                                           static_cast<int16_t>(NR > 5 ? x[5][i].d : 0),
                                           static_cast<int16_t>(NR > 6 ? x[6][i].d : 0),
                                           static_cast<int16_t>(NR > 7 ? x[7][i].d : 0));
        const __m256 scale = _mm256_mul_ps(_mm256_cvtph_ps(d8h), _mm256_set1_ps(h2f(w[i].d)));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(combine(lo, hi)), scale, acc);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, acc);
    for (int r = 0; r < NR; ++r) out[r] = lanes[r];
}

}  // namespace

LIYAB_AVX2 void dot_q8_0_q8_0_rows(const BlockQ8_0* w, const BlockQ8_0* const* x, int32_t rows, int64_t n,
                                   float* out) noexcept {
    const int64_t nb = n / kBlock;
    int32_t r = 0;
    for (; r + 8 <= rows; r += 8) dot_q8_0_rows<8>(w, x + r, nb, out + r);
    switch (rows - r) {
        case 7: dot_q8_0_rows<7>(w, x + r, nb, out + r); break;
        case 6: dot_q8_0_rows<6>(w, x + r, nb, out + r); break;
        case 5: dot_q8_0_rows<5>(w, x + r, nb, out + r); break;
        case 4: dot_q8_0_rows<4>(w, x + r, nb, out + r); break;
        case 3: dot_q8_0_rows<3>(w, x + r, nb, out + r); break;
        case 2: dot_q8_0_rows<2>(w, x + r, nb, out + r); break;
        case 1: out[r] = x86::dot_q8_0_q8_0(w, x[r], n); break;
        default: break;
    }
}

namespace {

// IQ4_NL and MXFP4: 32-value blocks of 4-bit indices into a 16-entry int8
// codebook (one VPSHUFB), against Q8_0. Alternate blocks, two accumulators.
template <typename Block, typename Scale>
LIYAB_AVX2 float dot_codebook4(const Block* w, const BlockQ8_0* x, int64_t nb, const int8_t* codebook,
                               Scale scale) noexcept {
    const __m256i table = _mm256_broadcastsi128_si256(load128(codebook));
    __m256 acc[2] = {_mm256_setzero_ps(), _mm256_setzero_ps()};
    for (int64_t i = 0; i < nb; ++i) {
        const __m256i v = _mm256_shuffle_epi8(table, nibbles32(w[i].qs));
        const __m256i s = dot_i8(v, load256(x[i].qs));
        acc[i & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(scale(w[i]) * h2f(x[i].d)), acc[i & 1]);
    }
    return hsum_ps(_mm256_add_ps(acc[0], acc[1]));
}

struct Iq4NlScale {
    LIYAB_AVX2 float operator()(const BlockIq4Nl& b) const noexcept { return h2f(b.d); }
};
struct Mxfp4Scale {
    float operator()(const BlockMxfp4& b) const noexcept { return e8m0_to_fp32_half(b.e); }
};

}  // namespace

LIYAB_AVX2 float dot_iq4_nl_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept {
    return dot_codebook4(static_cast<const BlockIq4Nl*>(row), x, n / kBlock, kIq4nlValues, Iq4NlScale{});
}

LIYAB_AVX2 float dot_mxfp4_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept {
    return dot_codebook4(static_cast<const BlockMxfp4*>(row), x, n / kBlock, kMxfp4Values, Mxfp4Scale{});
}

// --- Q8_K activations -----------------------------------------------------------------
//
// One template per format serves 1 to 4 activation rows: each super-block is
// decoded once into byte vectors and dotted with every row. Sub-block scales
// are applied while widening (VPMADDWD), so a row needs one int32 vector, and
// a super-block ends with the epilogue of quant.h on its exact integer sums:
// the single-row kernel is the NR = 1 instance, so the batch never changes a
// row's bits.

namespace {

constexpr uint32_t kKMask1 = 0x3f3f3f3f, kKMask2 = 0x0f0f0f0f, kKMask3 = 0x03030303;

// The 12 packed scale bytes of Q4_K / Q5_K -> 8 six-bit scales and 8 mins.
inline void unpack_k4_scales(const uint8_t* packed, uint8_t* scales, uint8_t* mins) noexcept {
    uint32_t u[3];
    std::memcpy(u, packed, 12);
    const uint32_t s[2] = {u[0] & kKMask1, (u[2] & kKMask2) | (((u[0] >> 6) & kKMask3) << 4)};
    const uint32_t m[2] = {u[1] & kKMask1, ((u[2] >> 4) & kKMask2) | (((u[1] >> 6) & kKMask3) << 4)};
    std::memcpy(scales, s, 8);
    std::memcpy(mins, m, 8);
}

// sum over the 8 sub-blocks of 32 values of min * (activation sum): the
// mins duplicated per 16-value activation sum, one VPMADDWD.
LIYAB_AVX2 inline int32_t k4_min_sum(__m256i mins16, const int16_t* bsums) noexcept {
    return hsum_i32(_mm256_madd_epi16(load256(bsums), mins16));
}

template <int NR, bool kFive>
LIYAB_AVX2 void dot_q45_k(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    static_assert(NR >= 1 && NR <= 4);
    using Block = std::conditional_t<kFive, BlockQ5_K, BlockQ4_K>;
    const auto* x = static_cast<const Block*>(row);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i sixteen = _mm256_set1_epi8(0x10);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        alignas(8) uint8_t sc[8], mn[8];
        unpack_k4_scales(x[i].scales, sc, mn);
        const __m128i m8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(mn));
        const __m256i mins16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(m8, m8));
        __m256i qh = _mm256_setzero_si256();
        if constexpr (kFive) qh = load256(x[i].qh);
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {  // 64 values: low nibbles = sub-block 2j, high = 2j + 1
            const __m256i q = load256(x[i].qs + 32 * j);
            __m256i lo = _mm256_and_si256(q, m4);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if constexpr (kFive) {  // 5th bits: bit 2j of qh[l] (low nibbles), bit 2j + 1 (high)
                const __m256i bl = _mm256_set1_epi8(static_cast<char>(1 << (2 * j)));
                const __m256i bh = _mm256_set1_epi8(static_cast<char>(2 << (2 * j)));
                lo = _mm256_or_si256(lo, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, bl), bl), sixteen));
                hi = _mm256_or_si256(hi, _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(qh, bh), bh), sixteen));
            }
            const __m256i s_lo = _mm256_set1_epi16(sc[2 * j]);
            const __m256i s_hi = _mm256_set1_epi16(sc[2 * j + 1]);
            for (int r = 0; r < NR; ++r) {
                const int8_t* q8 = y[r][i].qs + 64 * j;
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(lo, load256(q8)), s_lo));
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(hi, load256(q8 + 32)), s_hi));
            }
        }
        const float d = h2f(x[i].d);
        const float dmin = h2f(x[i].dmin);
        for (int r = 0; r < NR; ++r) {
            sum[r] = epilogue_k4(sum[r], y[r][i].d, d, dmin, hsum_i32(acc[r]), k4_min_sum(mins16, y[r][i].bsums));
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = sum[r];
}

template <int NR>
LIYAB_AVX2 void dot_q6_k(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    static_assert(NR >= 1 && NR <= 4);
    const auto* x = static_cast<const BlockQ6_K*>(row);
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const __m256i m3 = _mm256_set1_epi8(0x03);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        const int8_t* sc = x[i].scales;  // one signed scale per 16 values
        // Values are stored as q + 32: 32 * sum(scale * activation sum) is subtracted.
        const __m256i sc16 = _mm256_cvtepi8_epi16(load128(sc));
        __m256i acc[NR];
        int32_t mins[NR];
        for (int r = 0; r < NR; ++r) {
            acc[r] = _mm256_setzero_si256();
            mins[r] = hsum_i32(_mm256_madd_epi16(load256(y[r][i].bsums), sc16));
        }
        for (int h = 0; h < 2; ++h) {  // 128 values: 4 vectors of 32, two scales each
            const __m256i ql0 = load256(x[i].ql + 64 * h);
            const __m256i ql1 = load256(x[i].ql + 64 * h + 32);
            const __m256i qh = load256(x[i].qh + 32 * h);
            __m256i w[4];
            w[0] = _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qh, m3), 4));
            w[1] = _mm256_or_si256(_mm256_and_si256(ql1, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m3), 4));
            w[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m3), 4));
            w[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m3), 4));
            for (int k = 0; k < 4; ++k) {
                // Low 128-bit lane: values 0-15 of the 32 (scale 2k), high lane: 16-31 (scale 2k + 1).
                const __m256i s = combine(_mm_set1_epi16(sc[8 * h + 2 * k]), _mm_set1_epi16(sc[8 * h + 2 * k + 1]));
                for (int r = 0; r < NR; ++r) {
                    const __m256i q8 = load256(y[r][i].qs + 128 * h + 32 * k);
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(w[k], q8), s));
                }
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) {
            sum[r] = epilogue_scaled(sum[r], y[r][i].d, d, static_cast<float>(hsum_i32(acc[r]) - 32 * mins[r]));
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = sum[r];
}

// IQ4_XS: 8 sub-blocks of 32 values per super-block, 4-bit indices into the
// IQ4_NL codebook, a signed 6-bit scale each (minus 32).
LIYAB_AVX2 float dot_iq4_xs(const BlockIq4Xs* x, const BlockQ8_K* y, int64_t nb) noexcept {
    const __m256i table = _mm256_broadcastsi128_si256(load128(kIq4nlValues));
    float sumf = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc = _mm256_setzero_si256();
        const uint32_t h = x[i].scales_h;
        for (int k = 0; k < 8; ++k) {
            const uint32_t low = (x[i].scales_l[k / 2] >> (4 * (k % 2))) & 0xF;
            const int ls = static_cast<int>(low | (((h >> (2 * k)) & 3) << 4)) - 32;
            const __m256i v = _mm256_shuffle_epi8(table, nibbles32(x[i].qs + 16 * k));
            const __m256i q8 = load256(y[i].qs + 32 * k);
            const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(v, v), _mm256_sign_epi8(q8, v));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(static_cast<int16_t>(ls))));
        }
        sumf += h2f(x[i].d) * y[i].d * static_cast<float>(hsum_i32(acc));
    }
    return sumf;
}

}  // namespace

bool has_q8_K_kernel(DType type) noexcept {
    return type == DType::Q4_K || type == DType::Q5_K || type == DType::Q6_K || type == DType::IQ4_XS;
}

LIYAB_AVX2 float dot_q8_K(DType type, const void* row, const BlockQ8_K* x, int64_t n) noexcept {
    const int64_t nb = n / kSuperBlock;
    float out = 0.0f;
    switch (type) {
        case DType::Q4_K: dot_q45_k<1, false>(row, &x, nb, &out); break;
        case DType::Q5_K: dot_q45_k<1, true>(row, &x, nb, &out); break;
        case DType::Q6_K: dot_q6_k<1>(row, &x, nb, &out); break;
        case DType::IQ4_XS: out = dot_iq4_xs(static_cast<const BlockIq4Xs*>(row), x, nb); break;
        default: break;
    }
    return out;
}

LIYAB_AVX2 void dot_q8_K_rows(DType type, const void* row, const BlockQ8_K* const* x, int32_t rows, int64_t n,
                              float* out) noexcept {
    const int64_t nb = n / kSuperBlock;
    // Up to 4 rows per weight decode; longer batches go 4 at a time.
    for (int32_t r = 0; r < rows;) {
        const int32_t count = rows - r >= 4 ? 4 : rows - r;
        const BlockQ8_K* const* y = x + r;
        float* o = out + r;
        switch (type) {
            case DType::Q4_K:
                switch (count) {
                    case 4: dot_q45_k<4, false>(row, y, nb, o); break;
                    case 3: dot_q45_k<3, false>(row, y, nb, o); break;
                    case 2: dot_q45_k<2, false>(row, y, nb, o); break;
                    default: dot_q45_k<1, false>(row, y, nb, o); break;
                }
                break;
            case DType::Q5_K:
                switch (count) {
                    case 4: dot_q45_k<4, true>(row, y, nb, o); break;
                    case 3: dot_q45_k<3, true>(row, y, nb, o); break;
                    case 2: dot_q45_k<2, true>(row, y, nb, o); break;
                    default: dot_q45_k<1, true>(row, y, nb, o); break;
                }
                break;
            case DType::Q6_K:
                switch (count) {
                    case 4: dot_q6_k<4>(row, y, nb, o); break;
                    case 3: dot_q6_k<3>(row, y, nb, o); break;
                    case 2: dot_q6_k<2>(row, y, nb, o); break;
                    default: dot_q6_k<1>(row, y, nb, o); break;
                }
                break;
            default:
                for (int32_t k = 0; k < count; ++k) o[k] = dot_q8_K(type, row, y[k], n);
                break;
        }
        r += count;
    }
}

}  // namespace liyab::quant::x86
