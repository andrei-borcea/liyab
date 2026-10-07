// Liyab — portable CPU backend (ARM NEON / dot-product, scalar elsewhere).
//
// Activations are quantized to Q8_0 once per call, then every weight row is
// streamed exactly once and dotted against all `n` activation rows, so batched
// verification (speculative decoding) and prefill cost one pass over weights.
#include <string>
#include <vector>

#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/backend.h"

namespace liyab {

namespace {

class CpuBackend final : public Backend {
public:
    explicit CpuBackend(ThreadPool& pool) : pool_(pool) {}

    [[nodiscard]] BackendKind kind() const noexcept override { return BackendKind::Cpu; }

    [[nodiscard]] std::string description() const override {
        std::string isa =
#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
#if defined(__ARM_FEATURE_DOTPROD)
            "neon+dotprod";
#else
            "neon";
#endif
#else
            "scalar";
#endif
        return "cpu (" + isa + ", " + std::to_string(pool_.active_threads()) + " threads)";
    }

    [[nodiscard]] bool supports(DType) const noexcept override { return true; }

    Status matmul(const TensorView& w, const float* x, float* y, int32_t n) override {
        if (n <= 0) return Status::ok();
        const int64_t rows = w.rows();
        const int64_t cols = w.cols();
        const bool quantized = quant::is_block_quantized(w.type);
        const int64_t blocks = cols / quant::kBlock;

        if (quantized) {
            xq_.resize(static_cast<size_t>(blocks * n));
            for (int32_t t = 0; t < n; ++t) {
                quant::quantize_row_q8_0(x + t * cols, xq_.data() + t * blocks, cols);
            }
        }

        const size_t row_bytes = w.row_bytes();
        pool_.parallel_for(rows, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; ++r) {
                const uint8_t* row = w.data + static_cast<size_t>(r) * row_bytes;
                for (int32_t t = 0; t < n; ++t) {
                    float v = 0.0f;
                    switch (w.type) {
                        case DType::Q4_0:
                        case DType::Q4_1:
                        case DType::Q8_0:
                            v = quant::dot_quantized(w.type, row, xq_.data() + t * blocks, cols);
                            break;
                        case DType::F16:
                            v = quant::dot_f16_f32(reinterpret_cast<const uint16_t*>(row), x + t * cols, cols);
                            break;
                        case DType::F32:
                            v = quant::dot_f32(reinterpret_cast<const float*>(row), x + t * cols, cols);
                            break;
                    }
                    y[t * rows + r] = v;
                }
            }
        });
        return Status::ok();
    }

private:
    ThreadPool& pool_;
    std::vector<quant::BlockQ8_0> xq_;
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(ThreadPool& pool) { return std::make_unique<CpuBackend>(pool); }

}  // namespace liyab
