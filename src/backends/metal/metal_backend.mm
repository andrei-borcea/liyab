// Liyab — Apple GPU backend (Metal), built with ARC.
//
// Zero-copy on unified memory: every weight tensor is wrapped in an MTLBuffer
// created with newBufferWithBytesNoCopy over the page-aligned region of the
// read-only file mapping, so the GPU reads weights straight from the page
// cache. Only activations (a few KiB per token) are copied.
//
// Why not Core ML: the Apple Neural Engine is reachable only through compiled
// Core ML models; there is no public API to dispatch per-layer kernels over
// externally owned (mmap'd) weights, which is the core of Liyab's streaming
// design. Metal is the lowest-level public path to Apple accelerators.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <unistd.h>

#include <cstring>
#include <atomic>
#include <limits>
#include <string>
#include <map>
#include <utility>
#include <vector>

#include "backends/probes.h"
#include "liyab/backend.h"

namespace liyab {

namespace {
// One SIMD-group (32 lanes) per output element (Q1_0: per four rows); lanes
// stride over blocks and reduce with simd_sum. Block reads use byte loads so
// tensors need only the GGUF 32-byte alignment (Q1_0 reads 16-bit words: its
// blocks are 18 bytes, so every address is even).
constexpr const char* kKernelSource = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint rows;
    uint cols;
    uint row_bytes;
};

inline float load_half(device const uchar* p) {
    return float(as_type<half>(ushort(ushort(p[0]) | ushort(ushort(p[1]) << 8))));
}

// GGML K-quants: eight 32-value groups share a 256-value super-block.
// Decode the packed six-bit scales/minima directly; weights stay quantized.
inline uint2 scale_min_k(uint j, device const uchar* q) {
    return j < 4 ? uint2(q[j] & 63, q[j + 4] & 63)
                 : uint2((q[j + 4] & 15) | ((q[j - 4] >> 6) << 4),
                         (q[j + 4] >> 4) | ((q[j] >> 6) << 4));
}

inline void matvec_k45(device const uchar* w, device const float* x, device float* y,
                       constant Params& p, uint2 tg, uint lane, bool fifth) {
    device const uchar* wr = w + ulong(tg.x) * p.row_bytes;
    device const float* xr = x + ulong(tg.y) * p.cols;
    float sum = 0.0f;
    for (uint b = 0; b < p.cols / 256; ++b) {
        device const uchar* block = wr + b * (fifth ? 176 : 144);
        const float d = load_half(block), dm = load_half(block + 2);
        for (uint c = 0; c < 4; ++c) {
            const uint j = 2 * c;
            const uint2 a = scale_min_k(j, block + 4), z = scale_min_k(j + 1, block + 4);
            const uint packed = block[(fifth ? 48 : 16) + c * 32 + lane];
            const uint high = fifth ? block[16 + lane] : 0;
            const uint lo = (packed & 15) | (((high >> j) & 1) << 4);
            const uint hi = (packed >> 4) | (((high >> (j + 1)) & 1) << 4);
            sum += (d * a.x * lo - dm * a.y) * xr[b * 256 + j * 32 + lane];
            sum += (d * z.x * hi - dm * z.y) * xr[b * 256 + (j + 1) * 32 + lane];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(tg.y) * p.rows + tg.x] = sum;
}

kernel void matvec_q4_k(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    matvec_k45(w, x, y, p, tg, lane, false);
}
kernel void matvec_q5_k(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    matvec_k45(w, x, y, p, tg, lane, true);
}
kernel void matvec_q6_k(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    device const uchar* wr = w + ulong(tg.x) * p.row_bytes;
    device const float* xr = x + ulong(tg.y) * p.cols;
    float sum = 0.0f;
    for (uint b = 0; b < p.cols / 256; ++b) {
        device const uchar* block = wr + b * 210;
        const float d = load_half(block + 208);
        device const char* scales = (device const char*)(block + 192);
        for (uint h = 0; h < 2; ++h) {
            const uint a = block[h * 64 + lane], z = block[h * 64 + 32 + lane];
            const uint high = block[128 + h * 32 + lane];
            const uint si = h * 8 + lane / 16;
            const uint xi = b * 256 + h * 128 + lane;
            sum += d * scales[si] * (int((a & 15) | ((high & 3) << 4)) - 32) * xr[xi];
            sum += d * scales[si + 2] * (int((z & 15) | (((high >> 2) & 3) << 4)) - 32) * xr[xi + 32];
            sum += d * scales[si + 4] * (int((a >> 4) | (((high >> 4) & 3) << 4)) - 32) * xr[xi + 64];
            sum += d * scales[si + 6] * (int((z >> 4) | (((high >> 6) & 3) << 4)) - 32) * xr[xi + 96];
        }
    }
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(tg.y) * p.rows + tg.x] = sum;
}

kernel void matvec_q4_0(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, t = tg.y;
    device const uchar* wr = w + ulong(row) * p.row_bytes;
    device const float* xr = x + ulong(t) * p.cols;
    float sum = 0.0f;
    for (uint b = lane; b < p.cols / 32; b += 32) {
        device const uchar* blk = wr + b * 18;
        const float d = load_half(blk);
        float acc = 0.0f;
        for (uint j = 0; j < 16; ++j) {
            const uchar q = blk[2 + j];
            acc += float(int(q & 15) - 8) * xr[b * 32 + j] + float(int(q >> 4) - 8) * xr[b * 32 + j + 16];
        }
        sum += d * acc;
    }
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(t) * p.rows + row] = sum;
}

kernel void matvec_q4_1(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, t = tg.y;
    device const uchar* wr = w + ulong(row) * p.row_bytes;
    device const float* xr = x + ulong(t) * p.cols;
    float sum = 0.0f;
    for (uint b = lane; b < p.cols / 32; b += 32) {
        device const uchar* blk = wr + b * 20;
        const float d = load_half(blk);
        const float m = load_half(blk + 2);
        float acc = 0.0f;
        float xs = 0.0f;
        for (uint j = 0; j < 16; ++j) {
            const uchar q = blk[4 + j];
            const float x0 = xr[b * 32 + j], x1 = xr[b * 32 + j + 16];
            acc += float(q & 15) * x0 + float(q >> 4) * x1;
            xs += x0 + x1;
        }
        sum += d * acc + m * xs;
    }
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(t) * p.rows + row] = sum;
}

kernel void matvec_q8_0(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, t = tg.y;
    device const uchar* wr = w + ulong(row) * p.row_bytes;
    device const float* xr = x + ulong(t) * p.cols;
    float sum = 0.0f;
    for (uint b = lane; b < p.cols / 32; b += 32) {
        device const uchar* blk = wr + b * 34;
        const float d = load_half(blk);
        device const char* qs = (device const char*)(blk + 2);
        float acc = 0.0f;
        for (uint j = 0; j < 32; ++j) acc += float(qs[j]) * xr[b * 32 + j];
        sum += d * acc;
    }
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(t) * p.rows + row] = sum;
}

kernel void matvec_f16(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, t = tg.y;
    device const half* wr = (device const half*)(w + ulong(row) * p.row_bytes);
    device const float* xr = x + ulong(t) * p.cols;
    float sum = 0.0f;
    for (uint i = lane; i < p.cols; i += 32) sum += float(wr[i]) * xr[i];
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(t) * p.rows + row] = sum;
}

kernel void matvec_f32(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                       device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                       uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, t = tg.y;
    device const float* wr = (device const float*)(w + ulong(row) * p.row_bytes);
    device const float* xr = x + ulong(t) * p.cols;
    float sum = 0.0f;
    for (uint i = lane; i < p.cols; i += 32) sum += wr[i] * xr[i];
    sum = simd_sum(sum);
    if (lane == 0) y[ulong(t) * p.rows + row] = sum;
}

// Q1_0: 128 values per 18-byte block, an fp16 scale and one sign bit per value (bit j % 8 of byte j / 8 set: +d,
// clear: -d). A simdgroup computes kQ1Rows consecutive rows for one activation row, so the 32 activations a lane
// multiplies are loaded once and used for every row: the signs are 1/32 of the bytes of the activations they
// select, so a kernel that loaded them again for each row would be bound by those loads, not by the weights.
// Lane l takes the quarter (l % 4) of blocks l / 4, l / 4 + 8, ... : eight blocks per step, whatever the row
// width, and consecutive lanes read consecutive bytes. Each row's sum is the same whatever the rows around it
// or the other activation rows, so the result does not depend on the batch.
constant constexpr uint kQ1Rows = 4;

kernel void matvec_q1_0(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant Params& p [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row0 = tg.x * kQ1Rows, t = tg.y;
    const uint quarter = lane & 3;
    device const float* xr = x + ulong(t) * p.cols;
    float sum[kQ1Rows] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint b = lane >> 2; b < p.cols / 128; b += 8) {
        device const float4* xv = (device const float4*)(xr + b * 128 + quarter * 32);
        float4 xs[8];
        for (uint i = 0; i < 8; ++i) xs[i] = xv[i];
        for (uint r = 0; r < kQ1Rows; ++r) {
            if (row0 + r >= p.rows) break;
            device const uchar* blk = w + ulong(row0 + r) * p.row_bytes + b * 18;
            device const ushort* signs = (device const ushort*)(blk + 2 + quarter * 4);
            const uint bits = uint(signs[0]) | (uint(signs[1]) << 16);
            float s = 0.0f;
            for (uint i = 0; i < 8; ++i) {
                const uint nibble = bits >> (4 * i);
                s += select(-xs[i].x, xs[i].x, (nibble & 1) != 0) + select(-xs[i].y, xs[i].y, (nibble & 2) != 0) +
                     select(-xs[i].z, xs[i].z, (nibble & 4) != 0) + select(-xs[i].w, xs[i].w, (nibble & 8) != 0);
            }
            sum[r] += load_half(blk) * s;
        }
    }
    for (uint r = 0; r < kQ1Rows; ++r) {
        const float total = simd_sum(sum[r]);
        if (lane == 0 && row0 + r < p.rows) y[ulong(t) * p.rows + row0 + r] = total;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Batched products (several activation rows, e.g. prefill): out[t][r] = sum_k x[t][k] * w[r][k] as 8x8 matrix
// multiply-accumulates. A threadgroup of four simdgroups computes a tile of 64 weight rows x 32 activation rows,
// walking k in steps of 32: each thread decodes 16 weights of one row into threadgroup memory as floats and
// copies 8 activations, then each simdgroup multiplies its 32 x 16 quarter of the tile. The weights are read once
// per 32 activation rows instead of once per row, and decoded to float (not half) so the result stays within float
// rounding of the matvec kernels' (about 1e-7 of the sum of |w*x|); it is not their exact bits, which is why
// batches shorter than kMatmulMinRows keep using the matvec kernels.
// ---------------------------------------------------------------------------------------------------------------
struct MMParams {
    uint rows;
    uint cols;
    uint row_bytes;
    uint n;
};

constant constexpr uint BM = 64, BK = 32;  // a threadgroup: 64 weight rows, a tile of tokens (32 or 64), steps of 32

// Each format decodes 16 consecutive weights of a row, starting at the 16-aligned column k.
struct DqF32 {
    static void load16(device const uchar* row, uint k, thread float* o) {
        device const float* p = (device const float*)(row) + k;
        for (uint i = 0; i < 16; ++i) o[i] = p[i];
    }
};
struct DqF16 {
    static void load16(device const uchar* row, uint k, thread float* o) {
        for (uint i = 0; i < 16; ++i) o[i] = load_half(row + 2 * (k + i));
    }
};
struct DqQ4_0 {
    static void load16(device const uchar* row, uint k, thread float* o) {
        device const uchar* b = row + (k / 32) * 18;
        const float d = load_half(b);
        const bool hi = (k % 32) >= 16;
        for (uint i = 0; i < 16; ++i) {
            const uint q = b[2 + i];
            o[i] = d * float(int(hi ? (q >> 4) : (q & 15)) - 8);
        }
    }
};
struct DqQ4_1 {
    static void load16(device const uchar* row, uint k, thread float* o) {
        device const uchar* b = row + (k / 32) * 20;
        const float d = load_half(b), m = load_half(b + 2);
        const bool hi = (k % 32) >= 16;
        for (uint i = 0; i < 16; ++i) {
            const uint q = b[4 + i];
            o[i] = d * float(hi ? (q >> 4) : (q & 15)) + m;
        }
    }
};
struct DqQ8_0 {
    static void load16(device const uchar* row, uint k, thread float* o) {
        device const uchar* b = row + (k / 32) * 34;
        const float d = load_half(b);
        device const char* qs = (device const char*)(b + 2 + (k % 32));
        for (uint i = 0; i < 16; ++i) o[i] = d * float(qs[i]);
    }
};
struct DqQ1_0 {
    static void load16(device const uchar* row, uint k, thread float* o) {
        device const uchar* b = row + (k / 128) * 18;
        const float d = load_half(b);
        const uint e = k % 128;
        for (uint i = 0; i < 16; ++i) o[i] = ((b[2 + (e + i) / 8] >> ((e + i) % 8)) & 1) ? d : -d;
    }
};
template <bool Fifth>
struct DqK45 {
    static void load16(device const uchar* row, uint k, thread float* o) {
        device const uchar* b = row + (k / 256) * (Fifth ? 176 : 144);
        const uint e = k % 256, c = e / 64, off = e % 64, j0 = off % 32;
        const bool hi = off >= 32;
        const uint sub = 2 * c + (hi ? 1 : 0);
        const float d = load_half(b), dm = load_half(b + 2);
        const uint2 sm = scale_min_k(sub, b + 4);
        for (uint i = 0; i < 16; ++i) {
            const uint packed = b[(Fifth ? 48 : 16) + c * 32 + j0 + i];
            uint q = hi ? (packed >> 4) : (packed & 15);
            if (Fifth) q |= ((uint(b[16 + j0 + i]) >> sub) & 1) << 4;
            o[i] = d * sm.x * float(q) - dm * sm.y;
        }
    }
};
struct DqQ6_K {
    static void load16(device const uchar* row, uint k, thread float* o) {
        device const uchar* b = row + (k / 256) * 210;
        const uint e = k % 256, h = e / 128, part = (e % 128) / 32, lane0 = e % 32;
        const float d = load_half(b + 208);
        device const char* scales = (device const char*)(b + 192);
        for (uint i = 0; i < 16; ++i) {
            const uint lane = lane0 + i;
            const uint a = b[h * 64 + (part & 1) * 32 + lane];
            const uint high = b[128 + h * 32 + lane];
            const uint q = ((part < 2) ? (a & 15) : (a >> 4)) | (((high >> (2 * part)) & 3) << 4);
            o[i] = d * float(scales[h * 8 + lane / 16 + part * 2]) * float(int(q) - 32);
        }
    }
};

// A tile of rows x BK values is stored as 8x8 blocks of 64 consecutive floats, so that a simdgroup_load reads one
// block from 32 different banks (a row-major tile of 32 columns would put its 8 rows in the same bank).
inline uint blocked(uint row, uint k) { return ((row >> 3) * (BK / 8) + (k >> 3)) * 64 + (row & 7) * 8 + (k & 7); }

// BN tokens per threadgroup (32 or 64): the weights of a tile are decoded once and multiplied against all of them, so
// with the K-quants, whose decode is a good part of the work, 64 amortizes it over twice the tokens. Four simdgroups
// in a 2 x 2 grid, each (BM / 2) x (BN / 2) of the tile: 4 x BN / 16 matrix accumulators.
template <typename D, uint BN>
inline void mul_mm(device const uchar* w, device const float* x, device float* y, constant MMParams& p,
                   threadgroup float* shm, uint2 tg, uint tid, uint sgitg) {
    constexpr uint FT = BN / 16;             // token fragments per simdgroup
    threadgroup float* sa = shm;             // BM x BK in 8x8 blocks (blocked()): weights
    threadgroup float* sb = shm + BM * BK;   // BN x BK, likewise: activations
    const uint r0 = tg.x * BM, t0 = tg.y * BN;
    const uint sg_r = sgitg & 1, sg_t = sgitg >> 1;
    simdgroup_float8x8 mc[FT][4];
    _Pragma("clang loop unroll(full)") for (uint i = 0; i < FT; ++i)
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint lr = tid >> 1, lh = tid & 1;   // the weight row and 16-column half this thread decodes
    const uint row = r0 + lr;
    device const uchar* wrow = w + ulong(row) * p.row_bytes;
    for (uint kb = 0; kb < p.cols; kb += BK) {
        float v[16];
        if (row < p.rows) {
            D::load16(wrow, kb + lh * 16, v);
        } else {
            for (uint i = 0; i < 16; ++i) v[i] = 0.0f;
        }
        for (uint i = 0; i < 16; ++i) sa[blocked(lr, lh * 16 + i)] = v[i];
        // BN x BK activations: 4 consecutive values per step, BN * BK / 4 steps over the 128 threads.
        for (uint c = tid; c < BN * BK / 4; c += 128) {
            const uint lt = c / (BK / 4), lq = c % (BK / 4);
            const uint token = t0 + lt;
            float4 a = float4(0.0f);
            if (token < p.n) a = *reinterpret_cast<device const float4*>(x + ulong(token) * p.cols + kb + lq * 4);
            threadgroup float* dst = sb + blocked(lt, lq * 4);
            dst[0] = a.x;
            dst[1] = a.y;
            dst[2] = a.z;
            dst[3] = a.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _Pragma("clang loop unroll(full)") for (uint k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_float8x8 ma[FT], mb[4];
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < FT; ++i) simdgroup_load(ma[i], sb + ((sg_t * FT + i) * (BK / 8) + k8 / 8) * 64, 8);
            _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) simdgroup_load(mb[j], sa + ((sg_r * 4 + j) * (BK / 8) + k8 / 8) * 64, 8, ulong2(0, 0), true);
            _Pragma("clang loop unroll(full)") for (uint i = 0; i < FT; ++i)
                _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(mc[i][j], ma[i], mb[j], mc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The tile, activation-row major, through threadgroup memory so the edges are bounds-checked.
    threadgroup float* tile = shm;
    _Pragma("clang loop unroll(full)") for (uint i = 0; i < FT; ++i)
        _Pragma("clang loop unroll(full)") for (uint j = 0; j < 4; ++j) simdgroup_store(mc[i][j], tile + (sg_t * FT * 8 + i * 8) * BM + sg_r * 32 + j * 8, BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < BN * BM; e += 128) {
        const uint t = e / BM, r = e % BM;
        if (t0 + t < p.n && r0 + r < p.rows) y[ulong(t0 + t) * p.rows + r0 + r] = tile[e];
    }
}

#define LIYAB_MM(NAME, DQ)                                                                                         \
    kernel void NAME(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],                    \
                     device float* y [[buffer(2)]], constant MMParams& p [[buffer(3)]],                           \
                     threadgroup float* shm [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],       \
                     uint tid [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]) {  \
        mul_mm<DQ, 32>(w, x, y, p, shm, tg, tid, sgitg);                                                         \
    }                                                                                                            \
    kernel void NAME##_64(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],              \
                          device float* y [[buffer(2)]], constant MMParams& p [[buffer(3)]],                     \
                          threadgroup float* shm [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]], \
                          uint tid [[thread_index_in_threadgroup]], uint sgitg [[simdgroup_index_in_threadgroup]]) { \
        mul_mm<DQ, 64>(w, x, y, p, shm, tg, tid, sgitg);                                                         \
    }
LIYAB_MM(mm_f32, DqF32)
LIYAB_MM(mm_f16, DqF16)
LIYAB_MM(mm_q4_0, DqQ4_0)
LIYAB_MM(mm_q4_1, DqQ4_1)
LIYAB_MM(mm_q8_0, DqQ8_0)
LIYAB_MM(mm_q4_k, DqK45<false>)
LIYAB_MM(mm_q5_k, DqK45<true>)
LIYAB_MM(mm_q6_k, DqQ6_K)
LIYAB_MM(mm_q1_0, DqQ1_0)

// --- Attention over a prompt block (flash attention) ---
//
// One threadgroup per (query head, 32 query rows); each of its four simdgroups owns 8 rows and keeps their running
// maximum, denominator and 8 x D output tile in registers while the keys and values pass in chunks of C positions
// through threadgroup memory (shared by the four simdgroups, decoded once: half is exact for F16 caches and 2^-11 off
// for Q8_0). The scores and the weighted sum are 8x8 matrix products; the softmax works on the elements each lane
// holds of a product tile (thread_elements(): lane l holds row fm, columns fn and fn + 1, so the four lanes of a row
// are l, l^1, l^8, l^9). The keys and values were copied by the host into [position][kv head][row] arrays.
struct FAParams {
    uint n, n_past, total, n_head, n_kv_head, row_bytes;
    float scale, softcap;
};

struct KvF16 {
    static void load32(device const uchar* row, uint b, threadgroup half* dst) {
        device const half* h = reinterpret_cast<device const half*>(row) + b * 32;
        for (uint i = 0; i < 32; ++i) dst[i] = h[i];
    }
};
struct KvQ8 {
    static void load32(device const uchar* row, uint b, threadgroup half* dst) {
        device const uchar* blk = row + b * 34;
        const float d = float(*reinterpret_cast<device const half*>(blk));
        device const char* q = reinterpret_cast<device const char*>(blk + 2);
        for (uint i = 0; i < 32; ++i) dst[i] = half(d * float(q[i]));
    }
};

// The loops over fragments unroll completely (as in mul_mm): rolled, the accumulators live in memory and the kernel
// runs at a sixth of the speed (0.85 TFLOPS against 5).
template <typename KV, uint D>
inline void flash_attn(device const float* q, device const uchar* kbuf, device const uchar* vbuf, device float* out,
                       constant FAParams& p, threadgroup half* shm, uint2 tg, uint tid, uint sgitg, uint lane) {
    constexpr uint C = D > 128 ? 16 : 32;  // keys per chunk
    constexpr uint DS = D + 8;              // tile row stride in halves: 8 rows of a fragment land in different banks
    threadgroup half* sk = shm;
    threadgroup half* sv = shm + C * DS;
    threadgroup float* sp = reinterpret_cast<threadgroup float*>(shm + 2 * C * DS) + sgitg * 8 * C;  // this simdgroup's P

    const uint qid = lane >> 2;
    const uint fm = (qid & 4) + ((lane >> 1) & 3);
    const uint fn = ((qid & 2) << 1) + ((lane & 1) << 1);
    const uint h = tg.y, kvh = h / (p.n_head / p.n_kv_head);
    const uint qd = p.n_head * D;
    const uint t_base = tg.x * 32 + sgitg * 8;
    const uint row = t_base + fm;
    const uint pos = p.n_past + row;  // rows past n are padding: they read zeros and are never stored

    simdgroup_float8x8 mq[D / 8], mo[D / 8];
    _Pragma("clang loop unroll(full)") for (uint i = 0; i < D / 8; ++i) {
        simdgroup_load(mq[i], q + ulong(t_base) * qd + h * D + i * 8, qd);
        mo[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m = -INFINITY, l = 0.0f;
    const uint last_row = min(tg.x * 32 + 31, p.n - 1);
    const uint kend = p.n_past + last_row + 1;                                          // keys any row of the group sees
    const uint sg_kend = t_base < p.n ? p.n_past + min(t_base + 7, p.n - 1) + 1 : 0;    // keys any row of this simdgroup sees

    for (uint kc = 0; kc < kend; kc += C) {
        for (uint b = tid; b < C * (D / 32); b += 128) {
            const uint key = b / (D / 32), blk = b % (D / 32), idx = kc + key;
            if (idx < p.total) {
                const ulong at = (ulong(idx) * p.n_kv_head + kvh) * p.row_bytes;
                KV::load32(kbuf + at, blk, sk + key * DS + blk * 32);
                KV::load32(vbuf + at, blk, sv + key * DS + blk * 32);
            } else {
                for (uint i = 0; i < 32; ++i) {
                    sk[key * DS + blk * 32 + i] = 0.0h;
                    sv[key * DS + blk * 32 + i] = 0.0h;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kc < sg_kend) {
            simdgroup_float8x8 ms[C / 8];
            _Pragma("clang loop unroll(full)") for (uint c8 = 0; c8 < C / 8; ++c8) ms[c8] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            _Pragma("clang loop unroll(full)") for (uint d8 = 0; d8 < D / 8; ++d8) {
                _Pragma("clang loop unroll(full)") for (uint c8 = 0; c8 < C / 8; ++c8) {
                    simdgroup_half8x8 mk;
                    simdgroup_load(mk, sk + c8 * 8 * DS + d8 * 8, DS, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(ms[c8], mq[d8], mk, ms[c8]);
                }
            }
            float s[C / 4];
            float mnew = m;
            _Pragma("clang loop unroll(full)") for (uint c8 = 0; c8 < C / 8; ++c8) {
                _Pragma("clang loop unroll(full)") for (uint j = 0; j < 2; ++j) {
                    const uint key = kc + c8 * 8 + fn + j;
                    float v = ms[c8].thread_elements()[j] * p.scale;
                    if (p.softcap > 0.0f) v = p.softcap * precise::tanh(v / p.softcap);
                    v = (key <= pos && key < p.total) ? v : -INFINITY;
                    s[c8 * 2 + j] = v;
                    mnew = max(mnew, v);
                }
            }
            mnew = max(mnew, simd_shuffle_xor(mnew, 1));
            mnew = max(mnew, simd_shuffle_xor(mnew, 8));
            const float alpha = m == -INFINITY ? 0.0f : exp(m - mnew);
            l *= alpha;
            _Pragma("clang loop unroll(full)") for (uint c8 = 0; c8 < C / 8; ++c8) {
                _Pragma("clang loop unroll(full)") for (uint j = 0; j < 2; ++j) {
                    const float v = s[c8 * 2 + j];
                    const float w = v == -INFINITY ? 0.0f : exp(v - mnew);
                    l += w;
                    sp[fm * C + c8 * 8 + fn + j] = w;
                }
            }
            m = mnew;
            _Pragma("clang loop unroll(full)") for (uint d8 = 0; d8 < D / 8; ++d8) {
                mo[d8].thread_elements()[0] *= alpha;
                mo[d8].thread_elements()[1] *= alpha;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            _Pragma("clang loop unroll(full)") for (uint c8 = 0; c8 < C / 8; ++c8) {
                simdgroup_float8x8 mp;
                simdgroup_load(mp, sp + c8 * 8, C);
                _Pragma("clang loop unroll(full)") for (uint d8 = 0; d8 < D / 8; ++d8) {
                    simdgroup_half8x8 mv;
                    simdgroup_load(mv, sv + c8 * 8 * DS + d8 * 8, DS);
                    simdgroup_multiply_accumulate(mo[d8], mp, mv, mo[d8]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    l += simd_shuffle_xor(l, 1);
    l += simd_shuffle_xor(l, 8);
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    if (row < p.n) {
        _Pragma("clang loop unroll(full)") for (uint d8 = 0; d8 < D / 8; ++d8) {
            device float* o = out + ulong(row) * qd + h * D + d8 * 8 + fn;
            o[0] = mo[d8].thread_elements()[0] * inv;
            o[1] = mo[d8].thread_elements()[1] * inv;
        }
    }
}

#define LIYAB_FA(NAME, KV, D)                                                                                      \
    kernel void NAME(device const float* q [[buffer(0)]], device const uchar* kbuf [[buffer(1)]],                  \
                     device const uchar* vbuf [[buffer(2)]], device float* out [[buffer(3)]],                      \
                     constant FAParams& p [[buffer(4)]], threadgroup half* shm [[threadgroup(0)]],                \
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],        \
                     uint sgitg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {    \
        flash_attn<KV, D>(q, kbuf, vbuf, out, p, shm, tg, tid, sgitg, lane);                                        \
    }
LIYAB_FA(fa_f16_64, KvF16, 64)
LIYAB_FA(fa_f16_96, KvF16, 96)
LIYAB_FA(fa_f16_128, KvF16, 128)
LIYAB_FA(fa_f16_256, KvF16, 256)
LIYAB_FA(fa_q8_64, KvQ8, 64)
LIYAB_FA(fa_q8_96, KvQ8, 96)
LIYAB_FA(fa_q8_128, KvQ8, 128)
LIYAB_FA(fa_q8_256, KvQ8, 256)

// The gate/up products finish before this dispatch. Overwriting the gate
// output keeps the activated intermediate on the GPU for the down product.
kernel void gated_silu(device float* gate [[buffer(0)]], device const float* up [[buffer(1)]],
                       constant uint& count [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float g = gate[i];
    gate[i] = (g / (1.0f + exp(-g))) * up[i];
}
)MSL";

struct Params {
    uint32_t rows;
    uint32_t cols;
    uint32_t row_bytes;
};

class MetalBackend final : public Backend {
public:
    static Result<std::unique_ptr<Backend>> create() {
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) return Status(ErrorCode::Unsupported, "no Metal device");

            NSError* error = nil;
            MTLCompileOptions* options = [MTLCompileOptions new];
            id<MTLLibrary> library = [device newLibraryWithSource:@(kKernelSource) options:options error:&error];
            if (library == nil) {
                return Status(ErrorCode::BackendError,
                              std::string("Metal kernel compilation failed: ") +
                                  (error ? error.localizedDescription.UTF8String : "unknown error"));
            }
            auto backend = std::unique_ptr<MetalBackend>(new MetalBackend());
            backend->device_ = device;
            backend->queue_ = [device newCommandQueue];
            const char* names[] = {"matvec_f32", "matvec_f16", "matvec_q4_0", "matvec_q8_0", "matvec_q4_1", "matvec_q4_k", "matvec_q5_k", "matvec_q6_k", "gated_silu", "matvec_q1_0"};
            for (int i = 0; i < kPipelines; ++i) {
                id<MTLFunction> fn = [library newFunctionWithName:@(names[i])];
                id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
                if (pso == nil) {
                    return Status(ErrorCode::BackendError, std::string("cannot build Metal pipeline ") + names[i]);
                }
                backend->pipelines_[i] = pso;
            }
            const char* mm_names[] = {"mm_f32", "mm_f16", "mm_q4_0", "mm_q8_0", "mm_q4_1", "mm_q4_k", "mm_q5_k", "mm_q6_k", "mm_q1_0"};
            for (int i = 0; i < 9; ++i) {
                for (int wide = 0; wide < 2; ++wide) {
                    const std::string name = std::string(mm_names[i]) + (wide ? "_64" : "");
                    id<MTLFunction> fn = [library newFunctionWithName:@(name.c_str())];
                    id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
                    if (pso == nil) {
                        return Status(ErrorCode::BackendError, "cannot build Metal pipeline " + name +
                                                                   (error ? ": " + std::string(error.localizedDescription.UTF8String) : ""));
                    }
                    backend->mm_pipelines_[i][wide] = pso;
                }
            }
            const char* fa_names[] = {"fa_f16_64", "fa_f16_96", "fa_f16_128", "fa_f16_256",
                                      "fa_q8_64",  "fa_q8_96",  "fa_q8_128",  "fa_q8_256"};
            for (int i = 0; i < 8; ++i) {
                id<MTLFunction> fn = [library newFunctionWithName:@(fa_names[i])];
                id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
                if (pso == nil) {
                    return Status(ErrorCode::BackendError, std::string("cannot build Metal pipeline ") + fa_names[i] +
                                                               (error ? ": " + std::string(error.localizedDescription.UTF8String) : ""));
                }
                backend->fa_pipelines_[i] = pso;
            }
            if (backend->queue_ == nil) return Status(ErrorCode::BackendError, "cannot create Metal command queue");
            backend->make_residency_set();
            return std::unique_ptr<Backend>(std::move(backend));
        }
    }

    [[nodiscard]] BackendKind kind() const noexcept override { return BackendKind::Metal; }
    [[nodiscard]] std::string description() const override {
        return std::string("metal (") + device_.name.UTF8String + ", zero-copy weights)";
    }
    [[nodiscard]] bool supports(DType type) const noexcept override {
        return type == DType::F32 || type == DType::F16 || type == DType::Q4_0 || type == DType::Q4_1 ||
               type == DType::Q8_0 || type == DType::Q4_K || type == DType::Q5_K || type == DType::Q6_K ||
               type == DType::Q1_0;
    }

    Status matmul(const TensorView& w, const float* x, float* y, int32_t n) override {
        const TensorView* weights[] = {&w};
        float* outputs[] = {y};
        return matmul_group(weights, x, outputs, n);
    }

    // Q/K/V and gate/up share an input. Encode them into one command buffer
    // and wait once; no intermediate result leaves the GPU between products.
    Status matmul_group(std::span<const TensorView* const> ws, const float* x,
                        std::span<float* const> ys, int32_t n) override {
        if (ws.size() != ys.size()) return Status(ErrorCode::InvalidArgument, "Metal output count differs from weights");
        if (n <= 0 || ws.empty()) return Status::ok();
        for (const TensorView* w : ws) {
            if (w == nullptr || w->cols() != ws.front()->cols())
                return Status(ErrorCode::InvalidArgument, "Metal grouped matrices need the same input width");
            if (!supports(w->type)) return Status(ErrorCode::Unsupported,
                std::string(dtype_traits(w->type).name) + " runs on the CPU for now");
        }
        @autoreleasepool {
            std::vector<id<MTLBuffer>> weights;
            std::vector<NSUInteger> offsets;
            std::vector<size_t> output_offsets;
            size_t output_bytes = 0;
            for (const TensorView* w : ws) {
                NSUInteger offset = 0;
                id<MTLBuffer> buffer = weight_buffer(*w, offset);
                if (buffer == nil) return Status(ErrorCode::Unsupported, "cannot wrap Metal weights without copying");
                weights.push_back(buffer);
                offsets.push_back(offset);
                output_offsets.push_back(output_bytes);
                output_bytes += static_cast<size_t>(n) * static_cast<size_t>(w->rows()) * sizeof(float);
            }
            const size_t input_bytes = static_cast<size_t>(n) * static_cast<size_t>(ws.front()->cols()) * sizeof(float);
            if (!ensure(x_buf_, input_bytes) || !ensure(y_buf_, output_bytes))
                return Status(ErrorCode::OutOfMemory, "cannot allocate Metal activation buffers");
            std::memcpy(x_buf_.contents, x, input_bytes);
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            // The products read one input and write disjoint outputs: a concurrent encoder lets the GPU start the
            // next while the last threadgroups of the previous drain, instead of idling at the end of each.
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [enc setBuffer:x_buf_ offset:0 atIndex:1];
            for (size_t i = 0; i < ws.size(); ++i) {
                const TensorView& w = *ws[i];
                [enc setBuffer:weights[i] offset:offsets[i] atIndex:0];
                [enc setBuffer:y_buf_ offset:output_offsets[i] atIndex:2];
                dispatch_matvec(enc, w, n);
            }
            [enc endEncoding];
            commit_residency();
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) return Status(ErrorCode::BackendError,
                std::string("Metal command failed: ") + (cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown error"));
            const double gpu_seconds = cmd.GPUEndTime - cmd.GPUStartTime;
            if (gpu_seconds > 0) busy_ns_.fetch_add(static_cast<uint64_t>(gpu_seconds * 1e9), std::memory_order_relaxed);
            for (size_t i = 0; i < ws.size(); ++i) {
                const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(ws[i]->rows()) * sizeof(float);
                std::memcpy(ys[i], static_cast<const uint8_t*>(y_buf_.contents) + output_offsets[i], bytes);
            }
        }
        return Status::ok();
    }

    // Prompt-block attention (see Backend::attention). The cache's pages are heap memory the GPU cannot address as one
    // array, so the layer's rows up to the block's last position are copied into shared buffers first: a position's
    // K rows of all heads are contiguous in the cache, so that is two copies per position (a few MB per layer, against
    // the seconds the block's products take).
    Status attention(const AttentionJob& job) override {
        constexpr int32_t kMinRows = 16;  // as kMatmulMinRows: shorter passes keep the CPU's position-only rounding
        const int32_t d = job.head_dim;
        const int32_t sizes[] = {64, 96, 128, 256};
        int size_index = -1;
        for (int i = 0; i < 4; ++i) {
            if (sizes[i] == d) size_index = i;
        }
        if (job.rows == nullptr || job.n < kMinRows || size_index < 0 || job.n_head_kv <= 0 ||
            job.n_head % job.n_head_kv != 0 || (job.kv_type != DType::F16 && job.kv_type != DType::Q8_0)) {
            return Status(ErrorCode::Unsupported, "this attention job runs on the CPU");
        }
        const auto n = static_cast<size_t>(job.n);
        const auto total = static_cast<size_t>(job.n_past) + n;
        const size_t row_bytes = dtype_row_bytes(job.kv_type, d);
        const size_t position_bytes = static_cast<size_t>(job.n_head_kv) * row_bytes;
        const size_t q_floats = static_cast<size_t>(job.n_head) * static_cast<size_t>(d);
        const size_t padded_rows = (n + 31) / 32 * 32;  // the kernel reads whole 8-row fragments
        @autoreleasepool {
            if (!ensure(fa_q_, padded_rows * q_floats * sizeof(float)) || !ensure(fa_out_, n * q_floats * sizeof(float)) ||
                !ensure(fa_k_, total * position_bytes) || !ensure(fa_v_, total * position_bytes)) {
                return Status(ErrorCode::OutOfMemory, "cannot allocate Metal attention buffers");
            }
            std::memcpy(fa_q_.contents, job.q, n * q_floats * sizeof(float));
            std::memset(static_cast<uint8_t*>(fa_q_.contents) + n * q_floats * sizeof(float), 0,
                        (padded_rows - n) * q_floats * sizeof(float));
            auto* k_dst = static_cast<uint8_t*>(fa_k_.contents);
            auto* v_dst = static_cast<uint8_t*>(fa_v_.contents);
            for (size_t p = 0; p < total; ++p) {
                std::memcpy(k_dst + p * position_bytes, job.rows->k(static_cast<int32_t>(p), 0), position_bytes);
                std::memcpy(v_dst + p * position_bytes, job.rows->v(static_cast<int32_t>(p), 0), position_bytes);
            }
            struct {
                uint32_t n, n_past, total, n_head, n_kv_head, row_bytes;
                float scale, softcap;
            } params = {static_cast<uint32_t>(n),
                        static_cast<uint32_t>(job.n_past),
                        static_cast<uint32_t>(total),
                        static_cast<uint32_t>(job.n_head),
                        static_cast<uint32_t>(job.n_head_kv),
                        static_cast<uint32_t>(row_bytes),
                        job.scale,
                        job.softcap};
            const uint32_t chunk = d > 128 ? 16 : 32;
            const size_t shared = (2 * chunk * (static_cast<size_t>(d) + 8)) * sizeof(uint16_t) + 4 * 8 * chunk * sizeof(float);
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:fa_pipelines_[(job.kv_type == DType::Q8_0 ? 4 : 0) + size_index]];
            [enc setBuffer:fa_q_ offset:0 atIndex:0];
            [enc setBuffer:fa_k_ offset:0 atIndex:1];
            [enc setBuffer:fa_v_ offset:0 atIndex:2];
            [enc setBuffer:fa_out_ offset:0 atIndex:3];
            [enc setBytes:&params length:sizeof params atIndex:4];
            [enc setThreadgroupMemoryLength:shared atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((n + 31) / 32, static_cast<NSUInteger>(job.n_head), 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [enc endEncoding];
            commit_residency();
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) {
                return Status(ErrorCode::BackendError, std::string("Metal command failed: ") +
                                                           (cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown error"));
            }
            const double gpu_seconds = cmd.GPUEndTime - cmd.GPUStartTime;
            if (gpu_seconds > 0) busy_ns_.fetch_add(static_cast<uint64_t>(gpu_seconds * 1e9), std::memory_order_relaxed);
            std::memcpy(job.out, fa_out_.contents, n * q_floats * sizeof(float));
        }
        return Status::ok();
    }

    Status gated_ffn_silu(const TensorView& gate, const TensorView& up, const TensorView& down,
                          const float* x, float* y, int32_t n) override {
        if (n <= 0 || gate.cols() <= 0 || gate.rows() <= 0 || down.rows() <= 0 ||
            gate.cols() != up.cols() || gate.rows() != up.rows() || down.cols() != gate.rows())
            return Status(ErrorCode::InvalidArgument, "invalid Metal SiLU FFN shape");
        if (!supports(gate.type) || !supports(up.type) || !supports(down.type))
            return Status(ErrorCode::Unsupported, "Metal SiLU FFN needs GPU-supported weights");
        if (gate.rows() > UINT32_MAX || gate.cols() > UINT32_MAX || down.rows() > UINT32_MAX ||
            gate.row_bytes() > UINT32_MAX || up.row_bytes() > UINT32_MAX || down.row_bytes() > UINT32_MAX)
            return Status(ErrorCode::Unsupported, "Metal SiLU FFN exceeds dispatch limits");
        const uint64_t input_elems = static_cast<uint64_t>(n) * static_cast<uint64_t>(gate.cols());
        const uint64_t hidden_elems = static_cast<uint64_t>(n) * static_cast<uint64_t>(gate.rows());
        const uint64_t output_elems = static_cast<uint64_t>(n) * static_cast<uint64_t>(down.rows());
        const uint64_t buffer_elems = static_cast<uint64_t>(device_.maxBufferLength) / sizeof(float);
        if (hidden_elems > UINT32_MAX || input_elems > buffer_elems ||
            hidden_elems > buffer_elems / 2 || output_elems > buffer_elems - 2 * hidden_elems)
            return Status(ErrorCode::Unsupported, "Metal SiLU FFN exceeds buffer or dispatch limits");
        @autoreleasepool {
            NSUInteger gate_offset = 0, up_offset = 0, down_offset = 0;
            id<MTLBuffer> gate_w = weight_buffer(gate, gate_offset);
            id<MTLBuffer> up_w = weight_buffer(up, up_offset);
            id<MTLBuffer> down_w = weight_buffer(down, down_offset);
            if (gate_w == nil || up_w == nil || down_w == nil)
                return Status(ErrorCode::Unsupported, "cannot wrap Metal FFN weights without copying");

            const size_t input_bytes = static_cast<size_t>(input_elems) * sizeof(float);
            const size_t hidden_bytes = static_cast<size_t>(hidden_elems) * sizeof(float);
            const size_t output_bytes = static_cast<size_t>(output_elems) * sizeof(float);
            if (!ensure(x_buf_, input_bytes) || !ensure(y_buf_, 2 * hidden_bytes + output_bytes))
                return Status(ErrorCode::OutOfMemory, "cannot allocate Metal FFN activations");
            std::memcpy(x_buf_.contents, x, input_bytes);

            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            // Gate and up are independent (concurrent); the activation waits for both, the down product for it.
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            const TensorView* weights[] = {&gate, &up, &down};
            id<MTLBuffer> buffers[] = {gate_w, up_w, down_w};
            const NSUInteger offsets[] = {gate_offset, up_offset, down_offset};
            const NSUInteger output_offsets[] = {0, hidden_bytes, 2 * hidden_bytes};
            for (int i = 0; i < 3; ++i) {
                if (i == 2) {
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                    const uint32_t count = static_cast<uint32_t>(hidden_bytes / sizeof(float));
                    [enc setComputePipelineState:pipelines_[8]];
                    [enc setBuffer:y_buf_ offset:0 atIndex:0];
                    [enc setBuffer:y_buf_ offset:hidden_bytes atIndex:1];
                    [enc setBytes:&count length:sizeof count atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                }
                const TensorView& w = *weights[i];
                [enc setBuffer:buffers[i] offset:offsets[i] atIndex:0];
                [enc setBuffer:i == 2 ? y_buf_ : x_buf_ offset:0 atIndex:1];
                [enc setBuffer:y_buf_ offset:output_offsets[i] atIndex:2];
                dispatch_matvec(enc, w, n);
            }
            [enc endEncoding];
            commit_residency();
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) return Status(ErrorCode::BackendError,
                std::string("Metal FFN command failed: ") + (cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown error"));
            const double gpu_seconds = cmd.GPUEndTime - cmd.GPUStartTime;
            if (gpu_seconds > 0) busy_ns_.fetch_add(static_cast<uint64_t>(gpu_seconds * 1e9), std::memory_order_relaxed);
            std::memcpy(y, static_cast<const uint8_t*>(y_buf_.contents) + 2 * hidden_bytes, output_bytes);
        }
        return Status::ok();
    }

    uint8_t* allocate_shared(size_t bytes) override {
        const size_t page = static_cast<size_t>(getpagesize());
        if (bytes == 0 || bytes > device_.maxBufferLength || bytes > std::numeric_limits<size_t>::max() - page) return nullptr;
        const size_t aligned = (bytes + page - 1) & ~(page - 1);
        if (aligned > device_.maxBufferLength) return nullptr;
        id<MTLBuffer> buffer = [device_ newBufferWithLength:aligned options:MTLResourceStorageModeShared];
        auto* data = static_cast<uint8_t*>(buffer.contents);
        if (data == nullptr || reinterpret_cast<uintptr_t>(data) % page != 0) return nullptr;
        shared_.emplace(reinterpret_cast<uintptr_t>(data), buffer);
        keep_resident(buffer);
        return data;
    }
    void free_shared(uint8_t* data) noexcept override {
        const auto it = shared_.find(reinterpret_cast<uintptr_t>(data));
        if (it == shared_.end()) return;
        release_resident(it->second);
        shared_.erase(it);
    }
    [[nodiscard]] double busy_ms() const noexcept override {
        return static_cast<double>(busy_ns_.load(std::memory_order_relaxed)) / 1e6;
    }

private:
    MetalBackend() = default;

    id<MTLComputePipelineState> pipeline(DType type) const {
        switch (type) {
            case DType::F32: return pipelines_[0];
            case DType::F16: return pipelines_[1];
            case DType::Q4_0: return pipelines_[2];
            case DType::Q8_0: return pipelines_[3];
            case DType::Q4_1: return pipelines_[4];
            case DType::Q4_K: return pipelines_[5];
            case DType::Q5_K: return pipelines_[6];
            case DType::Q6_K: return pipelines_[7];
            case DType::Q1_0: return pipelines_[9];
            default: break;
        }
        return pipelines_[0];
    }

    // Batches of at least this many activation rows use the matrix kernels (and are then not bit-identical to the
    // matvec ones, see mul_mm): longer than a speculative verification pass, so those stay exact.
    static constexpr int32_t kMatmulMinRows = 16;
    static constexpr int32_t kWideTileRows = 64;  // from here on the matrix kernels take 64 tokens per tile

    int mm_index(DType type) const {
        switch (type) {
            case DType::F32: return 0;
            case DType::F16: return 1;
            case DType::Q4_0: return 2;
            case DType::Q8_0: return 3;
            case DType::Q4_1: return 4;
            case DType::Q4_K: return 5;
            case DType::Q5_K: return 6;
            case DType::Q6_K: return 7;
            case DType::Q1_0: return 8;
            default: return 0;
        }
    }

    // The matrix kernel for n >= kMatmulMinRows activation rows over whole 32-column steps, else the matvec one.
    // The weights are already bound at index 0, activations at 1, outputs at 2.
    void dispatch_matvec(id<MTLComputeCommandEncoder> enc, const TensorView& w, int32_t n) {
        if (n >= kMatmulMinRows && w.cols() % 32 == 0) {
            const uint32_t params[4] = {static_cast<uint32_t>(w.rows()), static_cast<uint32_t>(w.cols()),
                                        static_cast<uint32_t>(w.row_bytes()), static_cast<uint32_t>(n)};
            // Tiles of 64 tokens once there are enough to fill them, else of 32.
            const bool wide = n >= kWideTileRows;
            const NSUInteger tokens = wide ? 64 : 32;
            [enc setComputePipelineState:mm_pipelines_[mm_index(w.type)][wide ? 1 : 0]];
            [enc setBytes:params length:sizeof params atIndex:3];
            [enc setThreadgroupMemoryLength:(64 * 32 + tokens * 32) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(w.rows()) + 63) / 64, (static_cast<NSUInteger>(n) + tokens - 1) / tokens, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            return;
        }
        const Params params{static_cast<uint32_t>(w.rows()), static_cast<uint32_t>(w.cols()), static_cast<uint32_t>(w.row_bytes())};
        [enc setComputePipelineState:pipeline(w.type)];
        [enc setBytes:&params length:sizeof params atIndex:3];
        const NSUInteger rows_per_group = w.type == DType::Q1_0 ? 4 : 1;
        const NSUInteger groups = (static_cast<NSUInteger>(w.rows()) + rows_per_group - 1) / rows_per_group;
        [enc dispatchThreadgroups:MTLSizeMake(groups, static_cast<NSUInteger>(n), 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
    }

    // Without a residency set Metal makes every buffer a command references resident again at each submission,
    // work that grows with the bytes bound: a layer's FFN (three 36 MB weight matrices) waited 2.1 ms past its GPU
    // time per call, a 14B prompt's attention projections 0.7 ms, a layer's attention kernel (small buffers) 0.2 ms.
    // The weights and the shared regions are added to one set attached to the queue, committed before a submission
    // that follows an addition, and stay resident while the backend lives. macOS 15 and later; earlier systems and
    // a refusal simply keep the old behavior.
    void make_residency_set() {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            MTLResidencySetDescriptor* descriptor = [MTLResidencySetDescriptor new];
            descriptor.initialCapacity = 1024;
            NSError* error = nil;
            id<MTLResidencySet> set = [device_ newResidencySetWithDescriptor:descriptor error:&error];
            if (set != nil) {
                [queue_ addResidencySet:set];
                residency_ = set;
            }
        }
    }
    void keep_resident(id<MTLBuffer> buffer) {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            if (residency_ == nil) return;
            [static_cast<id<MTLResidencySet>>(residency_) addAllocation:buffer];
            residency_dirty_ = true;
        }
    }
    void release_resident(id<MTLBuffer> buffer) {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            if (residency_ == nil) return;
            [static_cast<id<MTLResidencySet>>(residency_) removeAllocation:buffer];
            residency_dirty_ = true;
        }
    }
    void commit_residency() {
        if (@available(macOS 15.0, iOS 18.0, *)) {
            if (residency_ == nil || !residency_dirty_) return;
            [static_cast<id<MTLResidencySet>>(residency_) commit];
            residency_dirty_ = false;
        }
    }

    bool ensure(id<MTLBuffer> __strong& buffer, size_t bytes) {
        if (buffer != nil && buffer.length >= bytes) return true;
        buffer = [device_ newBufferWithLength:std::max<size_t>(bytes, 4096) options:MTLResourceStorageModeShared];
        return buffer != nil;
    }

    // Wraps the pages backing `w` without copying. Cached per tensor: the
    // mapping outlives the backend's use of it (owned by the engine).
    id<MTLBuffer> weight_buffer(const TensorView& w, NSUInteger& offset) {
        const auto address = reinterpret_cast<uintptr_t>(w.data);
        auto region = shared_.upper_bound(address);
        if (region != shared_.begin()) {
            --region;
            const size_t delta = address - region->first;
            if (delta <= region->second.length && w.nbytes <= region->second.length - delta) {
                offset = delta;
                return region->second;
            }
        }
        const auto it = weights_.find({w.data, w.nbytes});
        const auto page = static_cast<uintptr_t>(getpagesize());
        const auto begin = reinterpret_cast<uintptr_t>(w.data) & ~(page - 1);
        offset = static_cast<NSUInteger>(reinterpret_cast<uintptr_t>(w.data) - begin);
        if (it != weights_.end()) return it->second;

        const uintptr_t end = (reinterpret_cast<uintptr_t>(w.data) + w.nbytes + page - 1) & ~(page - 1);
        if (end - begin > device_.maxBufferLength) return nil;
        id<MTLBuffer> buffer = [device_ newBufferWithBytesNoCopy:reinterpret_cast<void*>(begin)
                                                          length:end - begin
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
        if (buffer != nil) {
            weights_.emplace(std::make_pair(w.data, w.nbytes), buffer);
            keep_resident(buffer);
        }
        return buffer;
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    static constexpr int kPipelines = 10;
    id<MTLComputePipelineState> pipelines_[kPipelines] = {};
    id<MTLComputePipelineState> mm_pipelines_[9][2] = {};  // the batched kernels, in the order of pipeline(): f32 f16 q4_0 q8_0 q4_1 q4_k q5_k q6_k q1_0; tiles of 32 and of 64 tokens
    id<MTLComputePipelineState> fa_pipelines_[8] = {};  // attention: f16 then q8_0, each for 64 96 128 256 channels
    id residency_ = nil;  // an id<MTLResidencySet>, macOS 15+; see make_residency_set()
    bool residency_dirty_ = false;
    id<MTLBuffer> x_buf_ = nil;
    id<MTLBuffer> y_buf_ = nil;
    id<MTLBuffer> fa_q_ = nil, fa_k_ = nil, fa_v_ = nil, fa_out_ = nil;  // attention's activations and the copied cache
    std::atomic<uint64_t> busy_ns_{0};
    std::map<uintptr_t, id<MTLBuffer>> shared_;
    std::map<std::pair<const uint8_t*, size_t>, id<MTLBuffer>> weights_;  // key: (data, nbytes)
};

}  // namespace

Result<std::unique_ptr<Backend>> make_metal_backend() { return MetalBackend::create(); }

namespace probes {

AcceleratorProbe probe_metal() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) return {false, "no Metal device"};
        return {true, std::string(device.name.UTF8String) + (device.hasUnifiedMemory ? ", unified memory" : "")};
    }
}

}  // namespace probes

}  // namespace liyab
