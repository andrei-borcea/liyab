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

namespace {

// The 32 sign bits of a Q1_0 block's sub-block -> +1 / -1 in the bytes: byte e takes bit e % 8 of byte e / 8 (as
// q5_high_bits), +1 where the bit is set. A byte whose bit is clear compares equal to zero (0xFF = -1); the or
// then gives -1 there and +1 elsewhere.
LIYAB_AVX2 inline __m256i q1_signs(const uint8_t* bits4) noexcept {
    uint32_t bits;
    std::memcpy(&bits, bits4, 4);
    const __m256i select = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,  //
                                            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bit = _mm256_set1_epi64x(static_cast<int64_t>(0x8040201008040201ull));
    const __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(bits)), select);
    const __m256i clear = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, bit), _mm256_setzero_si256());
    return _mm256_or_si256(clear, _mm256_set1_epi8(1));
}

}  // namespace

// Each Q8_0 block's exact integer sum, then the epilogue every Q8_0 kernel shares (quant.h), block after block; the
// four sums of a 128-value block are reduced and converted together (exact, far below 2^24).
LIYAB_AVX2 float dot_q1_0_q8_0(const void* row, const BlockQ8_0* x, int64_t n) noexcept {
    const auto* w = static_cast<const BlockQ1_0*>(row);
    const int64_t nb = n / 128;
    float acc = 0.0f;
    for (int64_t i = 0; i < nb; ++i) {
        const BlockQ8_0* a = x + 4 * i;
        __m256i s[4];
        for (int k = 0; k < 4; ++k) s[k] = dot_i8(q1_signs(w[i].qs + 4 * k), load256(a[k].qs));
        alignas(16) float sums[4];
        _mm_store_ps(sums, _mm_cvtepi32_ps(hsum4_i32(s[0], s[1], s[2], s[3])));
        const float d = h2f(w[i].d);
        for (int k = 0; k < 4; ++k) acc = epilogue_scaled(acc, h2f(a[k].d), d, sums[k]);
    }
    return acc;
}

namespace {
template <int NR>
LIYAB_AVX2 void dot_mxfp4_rows(const BlockMxfp4* w, const BlockQ8_0* const* x, int64_t nb, float* out) noexcept {
    const __m256i table = _mm256_broadcastsi128_si256(load128(kMxfp4Values));
    __m256 acc[NR][2];
    for (int r = 0; r < NR; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
    for (int64_t b = 0; b < nb; ++b) {
        const __m256i v = _mm256_shuffle_epi8(table, nibbles32(w[b].qs));
        const float d = e8m0_to_fp32_half(w[b].e);
        for (int r = 0; r < NR; ++r) {
            const __m256i p = dot_i8(v, load256(x[r][b].qs));
            acc[r][b & 1] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), _mm256_set1_ps(d * h2f(x[r][b].d)), acc[r][b & 1]);
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = hsum_ps(_mm256_add_ps(acc[r][0], acc[r][1]));
}
}  // namespace

LIYAB_AVX2 void dot_mxfp4_q8_0_rows(const void* row, const BlockQ8_0* const* x, int32_t rows, int64_t n, float* out) noexcept {
    const auto* w = static_cast<const BlockMxfp4*>(row);
    int32_t r = 0;
    for (; r + 4 <= rows; r += 4) dot_mxfp4_rows<4>(w, x + r, n / kBlock, out + r);
    switch (rows - r) {
        case 3: dot_mxfp4_rows<3>(w, x + r, n / kBlock, out + r); break;
        case 2: dot_mxfp4_rows<2>(w, x + r, n / kBlock, out + r); break;
        case 1: out[r] = x86::dot_mxfp4_q8_0(row, x[r], n); break;
        default: break;
    }
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

// Q3_K: 2 low bits per value in qs and a high bit in hmask; the value is the low bits minus 4 when the high bit
// is clear. With the high bit as +4 the byte u = value + 4 lies in [0, 7], which VPMADDUBSW takes as its
// unsigned operand; the 4 is taken out again through the activation sums, 4 * sum(scale * activation sum).
// A 32-byte vector of qs holds 32 values (a 2-bit plane of one half): 16 per scale.
template <int NR>
LIYAB_AVX2 void dot_q3_k(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    static_assert(NR >= 1 && NR <= 4);
    const auto* x = static_cast<const BlockQ3K*>(row);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i four = _mm256_set1_epi8(4);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        // 16 six-bit scales: low nibbles from bytes 0-7, high 2 bits from bytes 8-11, minus 32.
        uint32_t aux[3], utmp[4];
        std::memcpy(aux, x[i].scales, 12);
        utmp[3] = ((aux[1] >> 4) & kKMask2) | (((aux[2] >> 6) & kKMask3) << 4);
        utmp[2] = ((aux[0] >> 4) & kKMask2) | (((aux[2] >> 4) & kKMask3) << 4);
        utmp[1] = (aux[1] & kKMask2) | (((aux[2] >> 2) & kKMask3) << 4);
        utmp[0] = (aux[0] & kKMask2) | (((aux[2] >> 0) & kKMask3) << 4);
        alignas(16) int8_t sc[16];
        std::memcpy(sc, utmp, 16);
        for (int k = 0; k < 16; ++k) sc[k] = static_cast<int8_t>(sc[k] - 32);
        const __m256i sc16 = _mm256_cvtepi8_epi16(load128(sc));
        const __m256i hm = load256(x[i].hmask);
        __m256i acc[NR];
        int32_t offset[NR];
        for (int r = 0; r < NR; ++r) {
            acc[r] = _mm256_setzero_si256();
            offset[r] = 4 * hsum_i32(_mm256_madd_epi16(load256(y[r][i].bsums), sc16));
        }
        for (int h = 0; h < 2; ++h) {  // 128 values: 4 planes of 32; the high bits are bit 4h + j of hmask
            const __m256i q = load256(x[i].qs + 32 * h);
            for (int j = 0; j < 4; ++j) {
                const __m256i bit = _mm256_set1_epi8(static_cast<char>(1 << (4 * h + j)));
                const __m256i low = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * j)), m3);
                const __m256i high = _mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(hm, bit), bit), four);
                const __m256i u = _mm256_or_si256(low, high);
                const __m256i s = combine(_mm_set1_epi16(sc[8 * h + 2 * j]), _mm_set1_epi16(sc[8 * h + 2 * j + 1]));
                for (int r = 0; r < NR; ++r) {
                    const __m256i q8 = load256(y[r][i].qs + 128 * h + 32 * j);
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(u, q8), s));
                }
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) {
            sum[r] = epilogue_scaled(sum[r], y[r][i].d, d, static_cast<float>(hsum_i32(acc[r]) - offset[r]));
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = sum[r];
}

// Q2_K: 2 bits per value, a 4-bit scale and a 4-bit minimum per 16 values; the minimums go through the
// activation sums as in Q4_K (epilogue_k4).
template <int NR>
LIYAB_AVX2 void dot_q2_k(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    static_assert(NR >= 1 && NR <= 4);
    const auto* x = static_cast<const BlockQ2K*>(row);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        const __m128i packed = load128(x[i].scales);
        alignas(16) uint8_t sc[16];
        _mm_store_si128(reinterpret_cast<__m128i*>(sc), _mm_and_si128(packed, m4));
        const __m256i mins16 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(packed, 4), m4));
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {
            const __m256i q = load256(x[i].qs + 32 * h);
            for (int j = 0; j < 4; ++j) {
                const __m256i v = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * j)), m3);
                const __m256i s = combine(_mm_set1_epi16(sc[8 * h + 2 * j]), _mm_set1_epi16(sc[8 * h + 2 * j + 1]));
                for (int r = 0; r < NR; ++r) {
                    const __m256i q8 = load256(y[r][i].qs + 128 * h + 32 * j);
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(v, q8), s));
                }
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

// --- I-quants against Q8_K activations ---------------------------------------------------
//
// The values of a grid format are unsigned magnitudes (at most 62) looked up by index, and the signs come apart:
// VPMADDUBSW takes the unsigned grid bytes against the activations with the sign applied to them (VPSIGNB), and a
// pair of products stays far below 2^15. One 32-value step is then a lookup of 4 (8-byte grids) or 8 (4-byte grids)
// entries, a sign expansion and one VPMADDUBSW per activation row; the sums are exact integers, scaled once per
// super-block like the NEON kernels (which these follow, also in the constant factor each format carries).

// 32 +1 / -1 bytes from four 8-bit sign masks (the low byte of `masks` covers values 0-7): bit j of a mask set
// means value 8g + j is negative. The masks are 7-bit sign indices through kKSigns in the XXS / XS formats and
// stored bytes in the S formats.
LIYAB_AVX2 inline __m256i expand_signs(uint32_t masks) noexcept {
    alignas(32) static constexpr uint8_t kSpread[32] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                                        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3};
    alignas(32) static constexpr uint8_t kBits[32] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
                                                      1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    const __m256i bits = load256(kBits);
    const __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(static_cast<int>(masks)), load256(kSpread));
    return _mm256_or_si256(_mm256_cmpeq_epi8(_mm256_and_si256(bytes, bits), bits), _mm256_set1_epi8(1));
}

inline uint32_t load_u32(const void* p) noexcept {
    uint32_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
inline int64_t load_i64(const void* p) noexcept {
    int64_t v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
// The four 7-bit sign indices of `w` (bits 0-27) as four mask bytes.
inline uint32_t sign_masks7(uint32_t w) noexcept {
    return static_cast<uint32_t>(kKSigns[w & 127]) | (static_cast<uint32_t>(kKSigns[(w >> 7) & 127]) << 8) |
           (static_cast<uint32_t>(kKSigns[(w >> 14) & 127]) << 16) | (static_cast<uint32_t>(kKSigns[(w >> 21) & 127]) << 24);
}

// grid (unsigned magnitudes) . (activations with the signs applied): 16 int16 sums of pairs.
LIYAB_AVX2 inline __m256i grid_dot(__m256i grid, __m256i signs, __m256i q8) noexcept {
    return _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8, signs));
}

// Each of the formats below: `step(ib32, scale-or-scale-vector)` produces the 32-value group's grid and signs; the
// rows loop is shared.
template <int NR>
LIYAB_AVX2 void dot_iq2_xxs(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockIq2Xxs*>(row);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {  // 32 values: 4 grid index bytes, then 4 x 7-bit signs + a 4-bit scale
            const uint8_t* p = reinterpret_cast<const uint8_t*>(x[i].qs) + 8 * ib;
            const uint32_t a = load_u32(p), w = load_u32(p + 4);
            const __m256i grid = _mm256_set_epi64x(load_i64(kIq2xxsGrid[(a >> 24) & 255]), load_i64(kIq2xxsGrid[(a >> 16) & 255]),
                                                   load_i64(kIq2xxsGrid[(a >> 8) & 255]), load_i64(kIq2xxsGrid[a & 255]));
            const __m256i signs = expand_signs(sign_masks7(w));
            const __m256i scale = _mm256_set1_epi16(static_cast<int16_t>(2 * (w >> 28) + 1));
            for (int r = 0; r < NR; ++r) {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(grid_dot(grid, signs, load256(y[r][i].qs + 32 * ib)), scale));
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) sum[r] += d * y[r][i].d * static_cast<float>(hsum_i32(acc[r]));
    }
    for (int r = 0; r < NR; ++r) out[r] = 0.125f * sum[r];
}

// IQ2_XS and IQ2_S: two 4-bit scales per 32 values (one per 16), as 2s + 1.
LIYAB_AVX2 inline __m256i scale_pair(uint8_t packed) noexcept {
    return combine(_mm_set1_epi16(static_cast<int16_t>(2 * (packed & 15) + 1)),
                   _mm_set1_epi16(static_cast<int16_t>(2 * (packed >> 4) + 1)));
}

template <int NR>
LIYAB_AVX2 void dot_iq2_xs(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockIq2Xs*>(row);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {  // 4 uint16: 9-bit grid index, 7-bit sign index
            const uint16_t* q = x[i].qs + 4 * ib;
            const __m256i grid = _mm256_set_epi64x(load_i64(kIq2xsGrid[q[3] & 511]), load_i64(kIq2xsGrid[q[2] & 511]),
                                                   load_i64(kIq2xsGrid[q[1] & 511]), load_i64(kIq2xsGrid[q[0] & 511]));
            const uint32_t masks = static_cast<uint32_t>(kKSigns[q[0] >> 9]) | (static_cast<uint32_t>(kKSigns[q[1] >> 9]) << 8) |
                                   (static_cast<uint32_t>(kKSigns[q[2] >> 9]) << 16) | (static_cast<uint32_t>(kKSigns[q[3] >> 9]) << 24);
            const __m256i signs = expand_signs(masks);
            const __m256i scale = scale_pair(x[i].scales[ib]);
            for (int r = 0; r < NR; ++r) {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(grid_dot(grid, signs, load256(y[r][i].qs + 32 * ib)), scale));
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) sum[r] += d * y[r][i].d * static_cast<float>(hsum_i32(acc[r]));
    }
    for (int r = 0; r < NR; ++r) out[r] = 0.125f * sum[r];
}

template <int NR>
LIYAB_AVX2 void dot_iq2_s(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockIq2S*>(row);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {  // 4 grid indices: 8 bits in qs, 2 more in qh; 4 stored sign bytes
            const uint8_t* q = x[i].qs + 4 * ib;
            const uint32_t h = x[i].qh[ib];
            const __m256i grid = _mm256_set_epi64x(load_i64(kIq2sGrid[q[3] | ((h << 2) & 0x300)]), load_i64(kIq2sGrid[q[2] | ((h << 4) & 0x300)]),
                                                   load_i64(kIq2sGrid[q[1] | ((h << 6) & 0x300)]), load_i64(kIq2sGrid[q[0] | ((h << 8) & 0x300)]));
            const __m256i signs = expand_signs(load_u32(x[i].qs + 32 + 4 * ib));
            const __m256i scale = scale_pair(x[i].scales[ib]);
            for (int r = 0; r < NR; ++r) {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(grid_dot(grid, signs, load256(y[r][i].qs + 32 * ib)), scale));
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) sum[r] += d * y[r][i].d * static_cast<float>(hsum_i32(acc[r]));
    }
    for (int r = 0; r < NR; ++r) out[r] = 0.125f * sum[r];
}

LIYAB_AVX2 inline __m256i grid4(const uint8_t (*grid)[4], const uint32_t* idx) noexcept {
    return _mm256_set_epi32(static_cast<int>(load_u32(grid[idx[7]])), static_cast<int>(load_u32(grid[idx[6]])),
                            static_cast<int>(load_u32(grid[idx[5]])), static_cast<int>(load_u32(grid[idx[4]])),
                            static_cast<int>(load_u32(grid[idx[3]])), static_cast<int>(load_u32(grid[idx[2]])),
                            static_cast<int>(load_u32(grid[idx[1]])), static_cast<int>(load_u32(grid[idx[0]])));
}

template <int NR>
LIYAB_AVX2 void dot_iq3_xxs(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockIq3Xxs*>(row);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {  // 8 grid index bytes (4 values each); a word of 4 x 7-bit signs + a 4-bit scale
            uint32_t idx[8];
            for (int k = 0; k < 8; ++k) idx[k] = x[i].qs[8 * ib + k];
            const uint32_t w = load_u32(x[i].qs + 64 + 4 * ib);
            const __m256i grid = grid4(kIq3xxsGrid, idx);
            const __m256i signs = expand_signs(sign_masks7(w));
            const __m256i scale = _mm256_set1_epi16(static_cast<int16_t>(2 * (w >> 28) + 1));
            for (int r = 0; r < NR; ++r) {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(grid_dot(grid, signs, load256(y[r][i].qs + 32 * ib)), scale));
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) sum[r] += d * y[r][i].d * static_cast<float>(hsum_i32(acc[r]));
    }
    for (int r = 0; r < NR; ++r) out[r] = 0.25f * sum[r];
}

template <int NR>
LIYAB_AVX2 void dot_iq3_s(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockIq3S*>(row);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {  // 8 grid indices: 8 bits in qs, the 9th in bit k of qh[ib]; 4 stored sign bytes
            uint32_t idx[8];
            const uint32_t h = x[i].qh[ib];
            for (int k = 0; k < 8; ++k) idx[k] = x[i].qs[8 * ib + k] | (((h >> k) & 1) << 8);
            const __m256i grid = grid4(kIq3sGrid, idx);
            const __m256i signs = expand_signs(load_u32(x[i].signs + 4 * ib));
            const int nibble = (x[i].scales[ib / 2] >> (4 * (ib & 1))) & 15;
            const __m256i scale = _mm256_set1_epi16(static_cast<int16_t>(2 * nibble + 1));
            for (int r = 0; r < NR; ++r) {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(grid_dot(grid, signs, load256(y[r][i].qs + 32 * ib)), scale));
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) sum[r] += d * y[r][i].d * static_cast<float>(hsum_i32(acc[r]));
    }
    for (int r = 0; r < NR; ++r) out[r] = sum[r];
}

// IQ1_S / IQ1_M: grid values in {-1, 0, +1} (signed, so |grid| against the activations with the grid's sign), a
// 3-bit scale (2s + 1) per 32 (S) or 16 (M) values, and a delta of +-1/8 added to every value of a group, which
// comes out of the dot product as +-1/8 times the group's activation sum.
constexpr float kIq1Delta = 0.125f;

LIYAB_AVX2 inline __m256i grid_dot_signed(__m256i grid, __m256i q8) noexcept {
    return _mm256_maddubs_epi16(_mm256_sign_epi8(grid, grid), _mm256_sign_epi8(q8, grid));
}

template <int NR>
LIYAB_AVX2 void dot_iq1_s(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockIq1S*>(row);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc[NR];
        int32_t delta[NR];
        for (int r = 0; r < NR; ++r) {
            acc[r] = _mm256_setzero_si256();
            delta[r] = 0;
        }
        for (int ib = 0; ib < 8; ++ib) {  // qh[ib]: 4 x 3 index bits, a 3-bit scale (bits 12-14), the delta's sign (bit 15)
            const uint32_t h = x[i].qh[ib];
            const uint8_t* q = x[i].qs + 4 * ib;
            const __m256i grid = _mm256_set_epi64x(load_i64(kIq1sGrid[q[3] | (((h >> 9) & 7) << 8)]), load_i64(kIq1sGrid[q[2] | (((h >> 6) & 7) << 8)]),
                                                   load_i64(kIq1sGrid[q[1] | (((h >> 3) & 7) << 8)]), load_i64(kIq1sGrid[q[0] | ((h & 7) << 8)]));
            const int ls = static_cast<int>(2 * ((h >> 12) & 7) + 1);
            const __m256i scale = _mm256_set1_epi16(static_cast<int16_t>(ls));
            for (int r = 0; r < NR; ++r) {
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(grid_dot_signed(grid, load256(y[r][i].qs + 32 * ib)), scale));
                delta[r] += (h & 0x8000 ? -ls : ls) * (y[r][i].bsums[2 * ib] + y[r][i].bsums[2 * ib + 1]);
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) {
            sum[r] += y[r][i].d * d * (static_cast<float>(hsum_i32(acc[r])) + kIq1Delta * static_cast<float>(delta[r]));
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = sum[r];
}

template <int NR>
LIYAB_AVX2 void dot_iq1_m(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockIq1M*>(row);
    const __m256i ones8 = _mm256_set1_epi8(1);
    const __m256i ones16 = _mm256_set1_epi16(1);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        uint16_t words[4];
        std::memcpy(words, x[i].scales, 8);
        // The fp16 super-block scale is spread over the top nibbles of the four scale words.
        const uint16_t dh = static_cast<uint16_t>((words[0] >> 12) | ((words[1] >> 8) & 0x00f0) | ((words[2] >> 4) & 0x0f00) |
                                                  (words[3] & 0xf000));
        __m256i acc[NR], delta[NR];
        for (int r = 0; r < NR; ++r) acc[r] = delta[r] = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ++ib) {  // 4 groups of 8: 11-bit indices (qs + 3 bits of qh), 3-bit scales per 16 values
            const uint8_t* q = x[i].qs + 4 * ib;
            const uint8_t* h = x[i].qh + 2 * ib;  // a nibble per group: 3 index bits and the delta's sign (bit 3)
            const uint32_t i0 = q[0] | ((h[0] & 7u) << 8), i1 = q[1] | (((h[0] >> 4) & 7u) << 8);
            const uint32_t i2 = q[2] | ((h[1] & 7u) << 8), i3 = q[3] | (((h[1] >> 4) & 7u) << 8);
            const __m256i grid = _mm256_set_epi64x(load_i64(kIq1sGrid[i3]), load_i64(kIq1sGrid[i2]), load_i64(kIq1sGrid[i1]),
                                                   load_i64(kIq1sGrid[i0]));
            const int shift = 6 * (ib & 1);  // two 3-bit scales per 32 values, four per word
            const int s0 = static_cast<int>(2 * ((words[ib / 2] >> shift) & 7) + 1);
            const int s1 = static_cast<int>(2 * ((words[ib / 2] >> (shift + 3)) & 7) + 1);
            const __m256i scale = combine(_mm_set1_epi16(static_cast<int16_t>(s0)), _mm_set1_epi16(static_cast<int16_t>(s1)));
            // The delta term: a group's activation sum (four per 8 values below) times its sign and its scale.
            const int g0 = (h[0] & 8) ? -s0 : s0, g1 = (h[0] & 0x80) ? -s0 : s0;
            const int g2 = (h[1] & 8) ? -s1 : s1, g3 = (h[1] & 0x80) ? -s1 : s1;
            const __m256i signed_scale = _mm256_set_epi32(g3, g3, g2, g2, g1, g1, g0, g0);
            for (int r = 0; r < NR; ++r) {
                const __m256i q8 = load256(y[r][i].qs + 32 * ib);
                acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(grid_dot_signed(grid, q8), scale));
                const __m256i quad_sums = _mm256_madd_epi16(_mm256_maddubs_epi16(ones8, q8), ones16);
                delta[r] = _mm256_add_epi32(delta[r], _mm256_mullo_epi32(quad_sums, signed_scale));
            }
        }
        const float d = h2f(dh);
        for (int r = 0; r < NR; ++r) {
            sum[r] += y[r][i].d * d * (static_cast<float>(hsum_i32(acc[r])) + kIq1Delta * static_cast<float>(hsum_i32(delta[r])));
        }
    }
    for (int r = 0; r < NR; ++r) out[r] = sum[r];
}

// TQ2_0: 2 bits per value, stored as value + 1 in {0, 1, 2} in the layout of Q2_K's planes; the 1 comes out through
// the activation sums.
template <int NR>
LIYAB_AVX2 void dot_tq2_0(const void* row, const BlockQ8_K* const* y, int64_t nb, float* out) noexcept {
    const auto* x = static_cast<const BlockTQ2_0*>(row);
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i ones16 = _mm256_set1_epi16(1);
    float sum[NR] = {};
    for (int64_t i = 0; i < nb; ++i) {
        __m256i acc[NR];
        for (int r = 0; r < NR; ++r) acc[r] = _mm256_setzero_si256();
        for (int h = 0; h < 2; ++h) {
            const __m256i q = load256(x[i].qs + 32 * h);
            for (int j = 0; j < 4; ++j) {
                const __m256i v = _mm256_and_si256(_mm256_srl_epi16(q, _mm_cvtsi32_si128(2 * j)), m3);
                for (int r = 0; r < NR; ++r) {
                    const __m256i q8 = load256(y[r][i].qs + 128 * h + 32 * j);
                    acc[r] = _mm256_add_epi32(acc[r], _mm256_madd_epi16(_mm256_maddubs_epi16(v, q8), ones16));
                }
            }
        }
        const float d = h2f(x[i].d);
        for (int r = 0; r < NR; ++r) {
            const int32_t total = hsum_i32(acc[r]) - hsum_i32(_mm256_madd_epi16(load256(y[r][i].bsums), ones16));
            sum[r] += d * y[r][i].d * static_cast<float>(total);
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
    switch (type) {
        case DType::Q2_K: case DType::Q3_K: case DType::Q4_K: case DType::Q5_K: case DType::Q6_K: case DType::IQ4_XS:
        case DType::IQ2_XXS: case DType::IQ2_XS: case DType::IQ2_S: case DType::IQ3_XXS: case DType::IQ3_S:
        case DType::IQ1_S: case DType::IQ1_M: case DType::TQ2_0:
            return true;
        default:
            return false;
    }
}

LIYAB_AVX2 float dot_q8_K(DType type, const void* row, const BlockQ8_K* x, int64_t n) noexcept {
    const int64_t nb = n / kSuperBlock;
    float out = 0.0f;
    switch (type) {
        case DType::Q2_K: dot_q2_k<1>(row, &x, nb, &out); break;
        case DType::Q3_K: dot_q3_k<1>(row, &x, nb, &out); break;
        case DType::IQ2_XXS: dot_iq2_xxs<1>(row, &x, nb, &out); break;
        case DType::IQ2_XS: dot_iq2_xs<1>(row, &x, nb, &out); break;
        case DType::IQ2_S: dot_iq2_s<1>(row, &x, nb, &out); break;
        case DType::IQ3_XXS: dot_iq3_xxs<1>(row, &x, nb, &out); break;
        case DType::IQ3_S: dot_iq3_s<1>(row, &x, nb, &out); break;
        case DType::IQ1_S: dot_iq1_s<1>(row, &x, nb, &out); break;
        case DType::IQ1_M: dot_iq1_m<1>(row, &x, nb, &out); break;
        case DType::TQ2_0: dot_tq2_0<1>(row, &x, nb, &out); break;
        case DType::Q4_K: dot_q45_k<1, false>(row, &x, nb, &out); break;
        case DType::Q5_K: dot_q45_k<1, true>(row, &x, nb, &out); break;
        case DType::Q6_K: dot_q6_k<1>(row, &x, nb, &out); break;
        case DType::IQ4_XS: out = dot_iq4_xs(static_cast<const BlockIq4Xs*>(row), x, nb); break;
        default: break;
    }
    return out;
}

namespace {
// Runs `kernel(std::integral_constant<int, NR>)` for NR = count (1 to 4) activation rows.
template <class F>
LIYAB_AVX2 void with_rows(int32_t count, F&& kernel) noexcept {
    switch (count) {
        case 4: kernel(std::integral_constant<int, 4>{}); break;
        case 3: kernel(std::integral_constant<int, 3>{}); break;
        case 2: kernel(std::integral_constant<int, 2>{}); break;
        default: kernel(std::integral_constant<int, 1>{}); break;
    }
}
}  // namespace

LIYAB_AVX2 void dot_q8_K_rows(DType type, const void* row, const BlockQ8_K* const* x, int32_t rows, int64_t n,
                              float* out) noexcept {
    const int64_t nb = n / kSuperBlock;
    // Up to 4 rows per weight decode; longer batches go 4 at a time.
    for (int32_t r = 0; r < rows;) {
        const int32_t count = rows - r >= 4 ? 4 : rows - r;
        const BlockQ8_K* const* y = x + r;
        float* o = out + r;
        switch (type) {
            case DType::Q2_K: with_rows(count, [&](auto nr) { dot_q2_k<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::Q3_K: with_rows(count, [&](auto nr) { dot_q3_k<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::Q4_K: with_rows(count, [&](auto nr) { dot_q45_k<decltype(nr)::value, false>(row, y, nb, o); }); break;
            case DType::Q5_K: with_rows(count, [&](auto nr) { dot_q45_k<decltype(nr)::value, true>(row, y, nb, o); }); break;
            case DType::Q6_K: with_rows(count, [&](auto nr) { dot_q6_k<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::IQ2_XXS: with_rows(count, [&](auto nr) { dot_iq2_xxs<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::IQ2_XS: with_rows(count, [&](auto nr) { dot_iq2_xs<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::IQ2_S: with_rows(count, [&](auto nr) { dot_iq2_s<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::IQ3_XXS: with_rows(count, [&](auto nr) { dot_iq3_xxs<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::IQ3_S: with_rows(count, [&](auto nr) { dot_iq3_s<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::IQ1_S: with_rows(count, [&](auto nr) { dot_iq1_s<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::IQ1_M: with_rows(count, [&](auto nr) { dot_iq1_m<decltype(nr)::value>(row, y, nb, o); }); break;
            case DType::TQ2_0: with_rows(count, [&](auto nr) { dot_tq2_0<decltype(nr)::value>(row, y, nb, o); }); break;
            default:
                for (int32_t k = 0; k < count; ++k) o[k] = dot_q8_K(type, row, y[k], n);
                break;
        }
        r += count;
    }
}

}  // namespace liyab::quant::x86
