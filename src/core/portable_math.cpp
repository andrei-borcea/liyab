#include "core/portable_math.h"

#include <bit>
#include <cmath>
#include <cstdint>

// No fused multiply-adds here: whether a compiler contracts a * b + c differs
// between compilers and targets (and a fused result rounds once instead of
// twice), and these functions exist to give the same bits everywhere.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#elif defined(_MSC_VER)
#pragma fp_contract(off)
#endif

namespace liyab {

namespace {

// tanh(x) = x + x z P(z), z = x^2, on |x| < kPolyBound: P fitted by weighted
// least squares to tanh's relative error (reweighted towards the worst points
// until the error was level), then checked in float as evaluated below.
constexpr float kPolyBound = 0.625f;
constexpr float kP0 = -0.33333280666607856f;
constexpr float kP1 = 0.1333140583848237f;
constexpr float kP2 = -0.05373661416862567f;
constexpr float kP3 = 0.020628879500727816f;
constexpr float kP4 = -0.005693558407113511f;

// e^r = 1 + r + r^2 Q(r) on |r| <= ln 2 / 2, fitted the same way.
constexpr float kQ0 = 0.49999995351777304f;
constexpr float kQ1 = 0.16666517414631155f;
constexpr float kQ2 = 0.04166790831148158f;
constexpr float kQ3 = 0.008369060797001377f;
constexpr float kQ4 = 0.0013842695906883602f;

constexpr float kLog2e = 1.4426950408889634f;
// ln 2 split so that n * kLn2Hi is exact for the n used here (n <= 29):
// kLn2Hi has 16 significant bits.
constexpr float kLn2Hi = 0.693145751953125f;
constexpr float kLn2Lo = 1.428606765330187045e-06f;

// Past this, tanh rounds to +-1 (from |x| of about 9.1).
constexpr float kSaturate = 10.0f;

// e^y for y in [1.25, 20] (2|x| on the branch that needs it).
float exp_mid(float y) noexcept {
    const float n = std::floor(y * kLog2e + 0.5f);
    const float r = (y - n * kLn2Hi) - n * kLn2Lo;
    float q = kQ4;
    q = q * r + kQ3;
    q = q * r + kQ2;
    q = q * r + kQ1;
    q = q * r + kQ0;
    const float e = (1.0f + r) + (r * r) * q;
    // 2^n built from its exponent bits: exact, n in [2, 29].
    const float scale = std::bit_cast<float>(static_cast<uint32_t>(static_cast<int32_t>(n) + 127) << 23);
    return e * scale;
}

}  // namespace

float portable_tanh(float x) noexcept {
    if (std::isnan(x)) return x;
    const float a = std::fabs(x);
    if (a < kPolyBound) {
        const float z = x * x;
        float p = kP4;
        p = p * z + kP3;
        p = p * z + kP2;
        p = p * z + kP1;
        p = p * z + kP0;
        const float t = (x * z) * p;
        // |t| < |x|, so the sign is x's; copysign keeps tanh(-0) = -0.
        return std::copysign(x + t, x);
    }
    if (a >= kSaturate) return std::copysign(1.0f, x);
    const float e = exp_mid(a + a);
    return std::copysign(1.0f - 2.0f / (e + 1.0f), x);
}

}  // namespace liyab
