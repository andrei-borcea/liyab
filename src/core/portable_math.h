#pragma once

// Elementary functions the engine computes itself, so they give the same bits
// on every platform. The C library's are not specified to the last bit, and
// they differ: Apple's tanhf and Android's (bionic) disagree in the last bits
// for some arguments, which made Gemma 2's soft-capped logits differ between
// the Mac and the phone where test_golden pins one hash for both. These use
// only IEEE single-precision additions, multiplications and divisions in a
// fixed order, with contraction into fused multiply-adds turned off where they
// are defined (portable_math.cpp), plus exact operations (floor, copysign,
// building a power of two), so every compiler and target computes the same
// values.
//
// Thread-safe (pure functions).

namespace liyab {

// tanh(x). Within 1.32 ulp of the true value (the largest error over 13
// million arguments checked against double-precision tanh: 1.311; 0.15 ulp
// on average), odd (tanh(-0) = -0), monotone, exactly +-1 for |x| >= 10
// (infinities included), NaN for NaN. |x| < 0.625 takes an odd polynomial
// in x (fitted to tanh's relative error); above, 1 - 2 / (e^(2|x|) + 1) with
// e^y = 2^n e^r computed here too (r within ln 2 / 2, a polynomial for e^r).
[[nodiscard]] float portable_tanh(float x) noexcept;

}  // namespace liyab
