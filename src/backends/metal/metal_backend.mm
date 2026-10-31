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

constant constexpr uint BM = 64, BN = 32, BK = 32;

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

template <typename D>
inline void mul_mm(device const uchar* w, device const float* x, device float* y, constant MMParams& p,
                   threadgroup float* shm, uint2 tg, uint tid, uint sgitg) {
    threadgroup float* sa = shm;             // BM x BK, row-major: weights
    threadgroup float* sb = shm + BM * BK;   // BN x BK, row-major: activations
    const uint r0 = tg.x * BM, t0 = tg.y * BN;
    const uint sg_r = sgitg & 1, sg_t = sgitg >> 1;
    simdgroup_float8x8 mc[2][4];
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j) mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const uint lr = tid >> 1, lh = tid & 1;   // the weight row and 16-column half this thread decodes
    const uint lt = tid >> 2, lp = tid & 3;   // the activation row and 8-column part it copies
    const uint row = r0 + lr, token = t0 + lt;
    device const uchar* wrow = w + ulong(row) * p.row_bytes;
    for (uint kb = 0; kb < p.cols; kb += BK) {
        float v[16];
        if (row < p.rows) {
            D::load16(wrow, kb + lh * 16, v);
        } else {
            for (uint i = 0; i < 16; ++i) v[i] = 0.0f;
        }
        for (uint i = 0; i < 16; ++i) sa[lr * BK + lh * 16 + i] = v[i];
        for (uint i = 0; i < 8; ++i) sb[lt * BK + lp * 8 + i] = token < p.n ? x[ulong(token) * p.cols + kb + lp * 8 + i] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < BK; k8 += 8) {
            simdgroup_float8x8 ma[2], mb[4];
            for (uint i = 0; i < 2; ++i) simdgroup_load(ma[i], sb + (sg_t * 16 + i * 8) * BK + k8, BK);
            for (uint j = 0; j < 4; ++j) simdgroup_load(mb[j], sa + (sg_r * 32 + j * 8) * BK + k8, BK, ulong2(0, 0), true);
            for (uint i = 0; i < 2; ++i)
                for (uint j = 0; j < 4; ++j) simdgroup_multiply_accumulate(mc[i][j], ma[i], mb[j], mc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The tile, activation-row major, through threadgroup memory so the edges are bounds-checked.
    threadgroup float* tile = shm;
    for (uint i = 0; i < 2; ++i)
        for (uint j = 0; j < 4; ++j) simdgroup_store(mc[i][j], tile + (sg_t * 16 + i * 8) * BM + sg_r * 32 + j * 8, BM);
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
        mul_mm<DQ>(w, x, y, p, shm, tg, tid, sgitg);                                                             \
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
                id<MTLFunction> fn = [library newFunctionWithName:@(mm_names[i])];
                id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
                if (pso == nil) {
                    return Status(ErrorCode::BackendError, std::string("cannot build Metal pipeline ") + mm_names[i]);
                }
                backend->mm_pipelines_[i] = pso;
            }
            if (backend->queue_ == nil) return Status(ErrorCode::BackendError, "cannot create Metal command queue");
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
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setBuffer:x_buf_ offset:0 atIndex:1];
            for (size_t i = 0; i < ws.size(); ++i) {
                const TensorView& w = *ws[i];
                [enc setBuffer:weights[i] offset:offsets[i] atIndex:0];
                [enc setBuffer:y_buf_ offset:output_offsets[i] atIndex:2];
                dispatch_matvec(enc, w, n);
            }
            [enc endEncoding];
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
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            const TensorView* weights[] = {&gate, &up, &down};
            id<MTLBuffer> buffers[] = {gate_w, up_w, down_w};
            const NSUInteger offsets[] = {gate_offset, up_offset, down_offset};
            const NSUInteger output_offsets[] = {0, hidden_bytes, 2 * hidden_bytes};
            for (int i = 0; i < 3; ++i) {
                if (i == 2) {
                    const uint32_t count = static_cast<uint32_t>(hidden_bytes / sizeof(float));
                    [enc setComputePipelineState:pipelines_[8]];
                    [enc setBuffer:y_buf_ offset:0 atIndex:0];
                    [enc setBuffer:y_buf_ offset:hidden_bytes atIndex:1];
                    [enc setBytes:&count length:sizeof count atIndex:2];
                    [enc dispatchThreadgroups:MTLSizeMake((count + 255) / 256, 1, 1)
                        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
                const TensorView& w = *weights[i];
                [enc setBuffer:buffers[i] offset:offsets[i] atIndex:0];
                [enc setBuffer:i == 2 ? y_buf_ : x_buf_ offset:0 atIndex:1];
                [enc setBuffer:y_buf_ offset:output_offsets[i] atIndex:2];
                dispatch_matvec(enc, w, n);
            }
            [enc endEncoding];
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
        return data;
    }
    void free_shared(uint8_t* data) noexcept override { shared_.erase(reinterpret_cast<uintptr_t>(data)); }
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
            [enc setComputePipelineState:mm_pipelines_[mm_index(w.type)]];
            [enc setBytes:params length:sizeof params atIndex:3];
            [enc setThreadgroupMemoryLength:(64 * 32 + 32 * 32) * sizeof(float) atIndex:0];
            [enc dispatchThreadgroups:MTLSizeMake((static_cast<NSUInteger>(w.rows()) + 63) / 64, (static_cast<NSUInteger>(n) + 31) / 32, 1)
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
        if (buffer != nil) weights_.emplace(std::make_pair(w.data, w.nbytes), buffer);
        return buffer;
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    static constexpr int kPipelines = 10;
    id<MTLComputePipelineState> pipelines_[kPipelines] = {};
    id<MTLComputePipelineState> mm_pipelines_[9] = {};  // the batched kernels, in the order of pipeline(): f32 f16 q4_0 q8_0 q4_1 q4_k q5_k q6_k q1_0
    id<MTLBuffer> x_buf_ = nil;
    id<MTLBuffer> y_buf_ = nil;
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
