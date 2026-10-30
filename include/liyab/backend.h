// Liyab — compute backend interface.
//
// A backend executes the weight-bound matrix multiplications of a transformer
// (QKV/O projections and the FFN). A backend may also fuse a complete SiLU
// FFN. Attention softmax, norms and RoPE stay on the CPU next to the KV cache.
// Weights are read in place from the memory-mapped model file; backends must
// not copy whole tensors.
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
    // Whether this backend runs `weight_type` natively in at least one placement of the weights (Vulkan, for example,
    // repacks some types from ordinary memory and reads others in place from shared memory). false: every matmul of
    // that type answers Unsupported and the caller runs it on the CPU. The CPU backend runs every format. The claim
    // is checked by running it: liyab-kernels prints the result, test_kernels enforces it.
    [[nodiscard]] virtual bool supports(DType weight_type) const noexcept = 0;

    // y[n][w.rows()] = x[n][w.cols()] · wᵀ  (row-major activations, f32).
    // `x` and `y` must not alias. Thread-compatible: one call at a time.
    // Returns ErrorCode::Unsupported when this backend has no kernel for the tensor (its type, a width it cannot
    // take, a size past a device limit): the caller then runs the product on the CPU, so `y` may have been written
    // and must not be relied on. Any other error aborts the pass.
    // A row's result should not depend on `n` or on the other rows of the call. The CPU backend guarantees the same
    // bits for any n, and for matmul_group() against matmul() (speculative verification and concurrent sequences
    // rely on it); liyab-kernels reports whether another backend does.
    virtual Status matmul(const TensorView& w, const float* x, float* y, int32_t n) = 0;

    // Unified memory (optional). Returns `bytes` of host memory that this
    // backend reads in place: a matmul whose weights lie inside such an
    // allocation runs without any copy, whatever wrote them (a direct read
    // from storage, the CPU). nullptr when unsupported (the caller then uses
    // ordinary memory). Release with free_shared(). Weights placed here
    // (streaming slots, expert caches, resident tensors) reach the
    // accelerator with zero copies and may be reused for other tensors at
    // any time, unlike ordinary weights, which backends may copy once.
    virtual uint8_t* allocate_shared(size_t bytes) {
        (void)bytes;
        return nullptr;
    }
    virtual void free_shared(uint8_t* data) noexcept { (void)data; }

    // Cumulative time this backend's device spent on submitted work, in
    // milliseconds (accelerators: from submission to completion). 0 when not
    // tracked (the CPU backend: use process CPU time instead). Thread-safe.
    [[nodiscard]] virtual double busy_ms() const noexcept { return 0.0; }

    // Several matmuls over the same input (Q/K/V, or FFN gate/up): ys[i] =
    // x · ws[i]ᵀ. Accelerators submit them together to pay the dispatch and
    // synchronization cost once; the default runs them one by one.
    virtual Status matmul_group(std::span<const TensorView* const> ws, const float* x, std::span<float* const> ys,
                                int32_t n) {
        for (size_t i = 0; i < ws.size(); ++i) LIYAB_RETURN_IF_ERROR(matmul(*ws[i], x, ys[i], n));
        return Status::ok();
    }

    // Optional fused SiLU FFN: y = down(silu(gate(x)) * up(x)). All three
    // matrices use row-major f32 activations; `x` and `y` may alias.
    // Biases, adapters and activation variants belong to the caller. Return
    // Unsupported before writing `y` when this backend cannot run the whole
    // operation; other errors abort the pass. Thread-compatible: one call at
    // a time, as for matmul().
    virtual Status gated_ffn_silu(const TensorView& gate, const TensorView& up, const TensorView& down,
                                  const float* x, float* y, int32_t n) {
        (void)gate; (void)up; (void)down; (void)x; (void)y; (void)n;
        return Status(ErrorCode::Unsupported, "fused SiLU FFN is unavailable");
    }

    // One independent matmul of a batch: y[n][w.rows()] = x[n][w.cols()] · wᵀ.
    struct MatmulItem {
        const TensorView* w = nullptr;
        const float* x = nullptr;
        float* y = nullptr;
        int32_t n = 0;
    };
    // Independent matmuls, each with its own weights, input and output (e.g.
    // the selected experts of a MoE block). Inputs may be shared between
    // items; outputs must not overlap any input or each other. Runs them as
    // one unit of work (the CPU backend: a single parallel pass over all
    // their rows, so small matrices do not each pay a thread fork/join). The
    // default runs them one by one.
    virtual Status matmul_batch(std::span<const MatmulItem> items) {
        for (const MatmulItem& it : items) LIYAB_RETURN_IF_ERROR(matmul(*it.w, it.x, it.y, it.n));
        return Status::ok();
    }
};

// Always available. `pool` must outlive the backend. `i8mm_tiles` runs Q4_K / Q5_K / Q6_K matmuls over two or more
// activation rows as 2x2 SMMLA tiles (bit-identical to the default kernels) when the CPU has i8mm and the library
// was built with the tile kernels; otherwise it has no effect.
LIYAB_API std::unique_ptr<Backend> make_cpu_backend(ThreadPool& pool, bool i8mm_tiles = false);

// Android / Linux GPU via Vulkan compute (weights repacked into GPU memory
// on first use). Unsupported when not compiled in or without a device.
LIYAB_API Result<std::unique_ptr<Backend>> make_vulkan_backend();

// Apple GPU via Metal. Returns Unsupported when not compiled in
// (LIYAB_USE_METAL=OFF) or when no Metal device exists.
LIYAB_API Result<std::unique_ptr<Backend>> make_metal_backend();

}  // namespace liyab

#endif  // LIYAB_BACKEND_H
