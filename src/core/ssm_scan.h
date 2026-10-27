// Liyab — the inner loops of the Mamba mixers (internal; ssm_scan.cpp).
//
// The selective scans of Mamba-1 and Mamba-2, their causal depthwise
// convolution and the elementwise SiLU, each with a NEON path (arm64), an
// AVX2 + FMA path (x86-64, chosen at run time like the matmul kernels:
// core/quant_x86.h, LIYAB_KERNELS=scalar turns it off) and a scalar path.
//
// Why kernels of their own: a Mamba-1 token costs d_inner x d_state
// exponentials (the per-(channel, state) decay exp(dt A)), as many as its
// projections cost multiply-adds on the small models, so the exponential is
// computed here as a polynomial over whole vectors instead of one C library
// call per value. The polynomial is the same on every path (Cody-Waite range
// reduction, Cephes' degree-6 polynomial, 2^n built from exponent bits):
// accurate over normal-range inputs (smaller decays clamp near zero), and
// the same bits on every device running the same
// path, where C libraries' exp() may differ in the last bit.
//
// Contracts shared by every function: results do not depend on the thread
// count or on how a caller splits its work (each output is computed by a
// fixed sequence of operations from its own inputs), but the three paths round
// differently from one another (the platforms' own golden hashes). Pure
// functions of their arguments; thread-safe on disjoint outputs.
// Convolution and SiLU use the same rounding in vector lanes and tails,
// so changing token batch boundaries does not change a channel's history.
#ifndef LIYAB_CORE_SSM_SCAN_H
#define LIYAB_CORE_SSM_SCAN_H

#include <cstddef>

namespace liyab::ssm {

// out[i] = x[i] * sigmoid(x[i]) for i < n (out may be x).
void silu(const float* x, float* out, size_t n) noexcept;

// One channel's causal depthwise convolution over a run of tokens: `seq`
// holds the channel's last k - 1 inputs before the run (oldest first) and then
// the run's n inputs; out[t] = bias + sum_j taps[j] * seq[t + j] for t < n,
// the taps in order (taps[k - 1] meets the token's own input). k >= 1.
void causal_conv(const float* seq, const float* taps, float bias, size_t k, size_t n, float* out) noexcept;

// Mamba-1's selective scan of one channel through `tokens` tokens. The state
// h holds n_state values and `a` the channel's A (negative: -exp(A_log)).
// Token t reads its step dt[t * stride] (softplus already applied), its input
// x[t * stride] and B, C at bc + t * bc_stride (B's n_state values, then C's),
// and leaves y[t * stride] = C . h after
//   h[j] = exp(dt a[j]) h[j] + (dt x) B[j].
// The D skip and the gate are the caller's.
void mamba_scan(float* h, const float* a, size_t n_state, const float* dt, const float* x, size_t stride,
                const float* bc, size_t bc_stride, float* y, size_t tokens) noexcept;

// Mamba-2's selective scan of one head for one token: the head's state is
// head_dim rows of n_state values; row p becomes
//   h[p][j] = decay h[p][j] + (dt x[p]) b[j],   y[p] = c . h[p]
// (decay = exp(dt A) of the head, computed by the caller once per token).
void mamba2_step(float* h, size_t head_dim, size_t n_state, float decay, float dt, const float* x, const float* b,
                 const float* c, float* y) noexcept;

}  // namespace liyab::ssm

#endif  // LIYAB_CORE_SSM_SCAN_H
