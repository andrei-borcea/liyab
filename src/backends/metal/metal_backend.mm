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
#include <string>
#include <map>
#include <utility>

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
            const char* names[] = {"matvec_f32", "matvec_f16", "matvec_q4_0", "matvec_q8_0", "matvec_q4_1"};
            for (int i = 0; i < 5; ++i) {
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
               type == DType::Q8_0;
    }

    Status matmul(const TensorView& w, const float* x, float* y, int32_t n) override {
        if (n <= 0) return Status::ok();
        if (!supports(w.type)) {  // K-quant kernels not written for Metal yet: CPU fallback
            return Status(ErrorCode::Unsupported, std::string(dtype_traits(w.type).name) + " runs on the CPU for now");
        }
        @autoreleasepool {
            NSUInteger weight_offset = 0;
            id<MTLBuffer> weights = weight_buffer(w, weight_offset);
            if (weights == nil) {
                return Status(ErrorCode::Unsupported, "cannot wrap tensor '" + std::string(w.name) + "' without copying");
            }
            const size_t x_bytes = static_cast<size_t>(n) * static_cast<size_t>(w.cols()) * sizeof(float);
            const size_t y_bytes = static_cast<size_t>(n) * static_cast<size_t>(w.rows()) * sizeof(float);
            if (!ensure(x_buf_, x_bytes) || !ensure(y_buf_, y_bytes)) {
                return Status(ErrorCode::OutOfMemory, "cannot allocate Metal activation buffers");
            }
            std::memcpy(x_buf_.contents, x, x_bytes);

            const Params params{static_cast<uint32_t>(w.rows()), static_cast<uint32_t>(w.cols()),
                                static_cast<uint32_t>(w.row_bytes())};
            id<MTLCommandBuffer> cmd = [queue_ commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:pipeline(w.type)];
            [enc setBuffer:weights offset:weight_offset atIndex:0];
            [enc setBuffer:x_buf_ offset:0 atIndex:1];
            [enc setBuffer:y_buf_ offset:0 atIndex:2];
            [enc setBytes:&params length:sizeof params atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(w.rows()), static_cast<NSUInteger>(n), 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            if (cmd.status != MTLCommandBufferStatusCompleted) {
                return Status(ErrorCode::BackendError,
                              std::string("Metal command failed: ") +
                                  (cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown error"));
            }
            std::memcpy(y, y_buf_.contents, y_bytes);
        }
        return Status::ok();
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
    id<MTLComputePipelineState> pipelines_[5] = {nil, nil, nil, nil, nil};
    id<MTLBuffer> x_buf_ = nil;
    id<MTLBuffer> y_buf_ = nil;
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
