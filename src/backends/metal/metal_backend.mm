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

// One SIMD-group (32 lanes) per output element; lanes stride over blocks and
// reduce with simd_sum. Block reads use byte loads so tensors need only the
// GGUF 32-byte alignment.
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
            const char* names[] = {"matvec_f32", "matvec_f16", "matvec_q4_0", "matvec_q8_0", "matvec_q4_1", "matvec_q4_k", "matvec_q5_k", "matvec_q6_k", "gated_silu"};
            for (int i = 0; i < 9; ++i) {
                id<MTLFunction> fn = [library newFunctionWithName:@(names[i])];
                id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&error] : nil;
                if (pso == nil) {
                    return Status(ErrorCode::BackendError, std::string("cannot build Metal pipeline ") + names[i]);
                }
                backend->pipelines_[i] = pso;
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
               type == DType::Q8_0 || type == DType::Q4_K || type == DType::Q5_K || type == DType::Q6_K;
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
                const Params params{static_cast<uint32_t>(w.rows()), static_cast<uint32_t>(w.cols()),
                                    static_cast<uint32_t>(w.row_bytes())};
                [enc setComputePipelineState:pipeline(w.type)];
                [enc setBuffer:weights[i] offset:offsets[i] atIndex:0];
                [enc setBuffer:y_buf_ offset:output_offsets[i] atIndex:2];
                [enc setBytes:&params length:sizeof params atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(w.rows()), static_cast<NSUInteger>(n), 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
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
                const Params params{static_cast<uint32_t>(w.rows()), static_cast<uint32_t>(w.cols()),
                                    static_cast<uint32_t>(w.row_bytes())};
                [enc setComputePipelineState:pipeline(w.type)];
                [enc setBuffer:buffers[i] offset:offsets[i] atIndex:0];
                [enc setBuffer:i == 2 ? y_buf_ : x_buf_ offset:0 atIndex:1];
                [enc setBuffer:y_buf_ offset:output_offsets[i] atIndex:2];
                [enc setBytes:&params length:sizeof params atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(w.rows()), static_cast<NSUInteger>(n), 1)
                    threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
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
            default: break;
        }
        return pipelines_[0];
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
    id<MTLComputePipelineState> pipelines_[9] = {nil, nil, nil, nil, nil, nil, nil, nil, nil};
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
