// Liyab — compute backend interface.
//
// A backend executes the weight-bound matrix multiplications of a transformer
// (QKV/O projections and the FFN). Attention softmax, norms and RoPE stay on
// the CPU next to the KV cache. Weights are always read in place from the
// memory-mapped model file; backends must not copy whole tensors.
#ifndef LIYAB_BACKEND_H
#define LIYAB_BACKEND_H

#include <memory>
#include <span>
#include <string>

#include "liyab/types.h"

namespace liyab {

class ThreadPool;

class Backend {
public:
    virtual ~Backend() = default;

    [[nodiscard]] virtual BackendKind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string description() const = 0;
    [[nodiscard]] virtual bool supports(DType weight_type) const noexcept = 0;

    // y[n][w.rows()] = x[n][w.cols()] · wᵀ  (row-major activations, f32).
    // `x` and `y` must not alias. Thread-compatible: one call at a time.
    virtual Status matmul(const TensorView& w, const float* x, float* y, int32_t n) = 0;

    // Several matmuls over the same input (Q/K/V, or FFN gate/up): ys[i] =
    // x · ws[i]ᵀ. Accelerators submit them together to pay the dispatch and
    // synchronization cost once; the default runs them one by one.
    virtual Status matmul_group(std::span<const TensorView* const> ws, const float* x, std::span<float* const> ys,
                                int32_t n) {
        for (size_t i = 0; i < ws.size(); ++i) LIYAB_RETURN_IF_ERROR(matmul(*ws[i], x, ys[i], n));
        return Status::ok();
    }
};

// Always available. `pool` must outlive the backend.
LIYAB_API std::unique_ptr<Backend> make_cpu_backend(ThreadPool& pool);

// Android / Linux GPU via Vulkan compute (weights repacked into GPU memory
// on first use). Unsupported when not compiled in or without a device.
LIYAB_API Result<std::unique_ptr<Backend>> make_vulkan_backend();

// Apple GPU via Metal. Returns Unsupported when not compiled in
// (LIYAB_USE_METAL=OFF) or when no Metal device exists.
LIYAB_API Result<std::unique_ptr<Backend>> make_metal_backend();

}  // namespace liyab

#endif  // LIYAB_BACKEND_H
