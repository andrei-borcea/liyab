// Liyab (experimental) — Thermal-Driven Dynamic Sparsification (TDSS).
//
// While the PowerManager reports thermal throttling (skin >= 40 °C by
// default, or a severe OS thermal status), the FFN projections (gate, up,
// down) switch from their dense weights to a 2:4 structured-sparse copy:
// in every group of 4 consecutive input weights only the 2 with the largest
// magnitude are kept (magnitude pruning, no retraining).
//
// Sparse format (per 32 input columns, 14 bytes = 3.5 bits/weight, i.e. 22%
// fewer bytes than dense Q4_0 at 4.5 bits):
//   fp16 scale | 8 bytes: 16 kept INT4 values (offset-binary, like Q4_0)
//              | 4 bytes: 16 x 2-bit position of each value inside its group
// NEON kernel: the 16 matching activations are gathered with one TBL
// (vqtbl2q_s8) from the Q8_0 activation block and reduced with one SDOT,
// instead of two SDOTs over 32 dense values.
//
// Scope and honesty: phone CPUs/GPUs have no 2:4 sparse hardware (that is
// an NVIDIA Ampere+ feature), so the gain is bounded by fewer bytes and
// half the multiply-adds, minus the gather. Magnitude pruning without
// fine-tuning changes the output; the sparse copy costs extra memory while
// the dense weights stay mapped. Power draw is not claimed: measuring watts
// needs a device on battery with a power monitor.
#ifndef LIYAB_EXPERIMENTAL_TDSS_H
#define LIYAB_EXPERIMENTAL_TDSS_H

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "liyab/experimental/forward_hooks.h"
#include "liyab/power_manager.h"
#include "liyab/types.h"

namespace liyab {
class ThreadPool;
}

namespace liyab::experimental {

// One weight matrix in 2:4 sparse INT4 form.
class LIYAB_API Sparse24Matrix {
public:
    static constexpr size_t kBlockBytes = 14;  // per 32 input columns

    // Prunes and quantizes `w` (any supported type; cols % 32 == 0).
    static Result<Sparse24Matrix> build(const TensorView& w);

    // y[n][rows] = x[n][cols] · Wᵀ using the sparse weights.
    void matmul(const float* x, float* y, int32_t n, ThreadPool& pool) const;
    // Dense float reconstruction of one row (zeros where pruned), for tests.
    void dense_row(int64_t row, float* out) const;

    [[nodiscard]] int64_t rows() const noexcept { return rows_; }
    [[nodiscard]] int64_t cols() const noexcept { return cols_; }
    [[nodiscard]] size_t bytes() const noexcept { return data_.size(); }

private:
    int64_t rows_ = 0;
    int64_t cols_ = 0;
    std::vector<uint8_t> data_;  // rows x (cols / 32) blocks
};

struct TdssConfig {
    bool force = false;  // sparse even when cool (benchmarks)
};

class LIYAB_API Tdss final : public FfnMatmulHook {
public:
    // `ffn` holds {gate, up, down} for every block. `pool` must outlive this.
    static Result<std::unique_ptr<Tdss>> create(std::span<const std::array<const TensorView*, 3>> ffn,
                                                ThreadPool& pool, TdssConfig config = {});

    // Activates sparsity while `policy.throttled` (or always with force).
    bool update(const PowerPolicy& policy);
    void set_active(bool active) noexcept { active_ = active; }
    [[nodiscard]] bool active() const noexcept { return active_; }

    bool ffn_matmul(int32_t layer, FfnProjection projection, const float* x, float* y, int32_t n) override;

    [[nodiscard]] size_t sparse_bytes() const noexcept;
    [[nodiscard]] size_t dense_bytes() const noexcept { return dense_bytes_; }

private:
    Tdss() = default;

    std::vector<std::array<Sparse24Matrix, 3>> layers_;
    ThreadPool* pool_ = nullptr;
    TdssConfig config_;
    bool active_ = false;
    size_t dense_bytes_ = 0;
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_TDSS_H
