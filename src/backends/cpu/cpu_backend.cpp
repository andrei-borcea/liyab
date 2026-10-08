// Liyab — portable CPU backend (ARM NEON / dot-product, scalar elsewhere).
//
// Activations are quantized once per call, then every weight row is streamed
// exactly once and dotted against all `n` activation rows, so batched
// verification (speculative decoding) and prefill cost one pass over weights.
// The activation format follows the weight type: Q8_0 (32-value blocks) for
// most formats, Q8_K (256-value super-blocks with 16-value sums) for the 2/3-bit
// K-quants, TQ2_0 and the 256-value I-quants, whose NEON kernels need it
// (quant::uses_q8_K).
//
// Every entry point funnels into matmul_batch(): the rows of all the batch's
// matrices are split across the thread pool in one parallel pass. A decode
// step runs ~1000 small matmuls (a 35B MoE: 8 experts x 3 matrices x 40
// blocks, plus projections); one fork/join each cost more than the math.
#include <algorithm>
#include <string>
#include <vector>

#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/backend.h"

namespace liyab {

namespace {

// How a weight type wants its activations.
enum class ActFormat : uint8_t {
    Float,     // F32 / F16 / BF16 weights read the float input directly
    Q8_0,      // 32-value blocks
    Q8_0Sums,  // Q8_0 plus per-block sums (K-quants and Q5_1 subtract their minimums)
    Q8_K,      // 256-value super-blocks
};

ActFormat act_format(const TensorView& w) {
    if (quant::uses_q8_K(w.type) && w.cols() % quant::kSuperBlock == 0) return ActFormat::Q8_K;
    if (!quant::is_block_quantized(w.type)) return ActFormat::Float;
    return quant::is_k_quant(w.type) || w.type == DType::Q5_1 ? ActFormat::Q8_0Sums : ActFormat::Q8_0;
}

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
        const MatmulItem item{&w, x, y, n};
        return matmul_batch(std::span<const MatmulItem>(&item, 1));
    }

    Status matmul_group(std::span<const TensorView* const> ws, const float* x, std::span<float* const> ys,
                        int32_t n) override {
        items_.clear();
        for (size_t i = 0; i < ws.size(); ++i) items_.push_back({ws[i], x, ys[i], n});
        return matmul_batch(items_);
    }

    Status matmul_batch(std::span<const MatmulItem> items) override {
        // Quantize each distinct (input, format) once; items sharing an input
        // (gate/up, Q/K/V) share its quantized copy.
        jobs_.resize(items.size());
        row_end_.resize(items.size());
        size_t n_acts = 0;
        int64_t total_rows = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            const MatmulItem& it = items[i];
            const ActFormat format = it.n > 0 ? act_format(*it.w) : ActFormat::Float;
            size_t a = 0;
            while (a < n_acts && !(acts_[a].x == it.x && acts_[a].format == format && acts_[a].n == it.n &&
                                   acts_[a].cols == it.w->cols())) {
                ++a;
            }
            if (a == n_acts) {
                if (acts_.size() == n_acts) acts_.emplace_back();
                quantize(acts_[a], it.x, it.w->cols(), it.n, format);
                ++n_acts;
            }
            if (it.w->type == DType::Q4_K_R8 && it.n >= 4 && !acts_[a].q8k4_ready) interleave_x4(acts_[a]);
            jobs_[i] = {&it, a, format};  // an index: acts_ may still grow
            total_rows += it.n > 0 ? it.w->rows() : 0;
            row_end_[i] = total_rows;
        }

        // One pass over the rows of every matrix, in order; a chunk may span
        // several matrices.
        pool_.parallel_for(total_rows, [&](int64_t r0, int64_t r1) {
            size_t i = static_cast<size_t>(std::upper_bound(row_end_.begin(), row_end_.end(), r0) - row_end_.begin());
            for (int64_t r = r0; r < r1; ++i) {
                const int64_t begin = i == 0 ? 0 : row_end_[i - 1];
                const int64_t end = std::min(row_end_[i], r1);
                if (r < end) run_rows(jobs_[i], acts_[jobs_[i].acts], r - begin, end - begin);
                r = std::max(r, end);
            }
        });
        return Status::ok();
    }

private:
    // An input's activations in the format some weight type wants.
    struct Activations {
        const float* x = nullptr;
        int64_t cols = 0;
        int32_t n = 0;
        ActFormat format = ActFormat::Float;
        std::vector<quant::BlockQ8_0> q8;
        std::vector<quant::BlockQ8_K> q8k;
        std::vector<int32_t> sums;
        // Q8_K rows interleaved by 4 for the repacked kernels (Q4_K_R8 weights,
        // n >= 4); built on first use by such a weight.
        std::vector<quant::BlockQ8_Kx4> q8k4;
        bool q8k4_ready = false;
    };
    struct Job {
        const MatmulItem* item = nullptr;
        size_t acts = 0;  // index into acts_
        ActFormat format = ActFormat::Float;
    };

    // Groups of 4 Q8_K activation rows -> Q8_K_X4 (n / 4 groups; the last
    // n % 4 rows stay single).
    static void interleave_x4(Activations& a) {
        const int64_t super_blocks = a.cols / quant::kSuperBlock;
        a.q8k4.resize(static_cast<size_t>(a.n / 4 * super_blocks));
        for (int32_t t = 0; t + 4 <= a.n; t += 4) {
            const quant::BlockQ8_K* rows[4];
            for (int k = 0; k < 4; ++k) rows[k] = a.q8k.data() + (t + k) * super_blocks;
            quant::interleave_q8_K_x4(rows, a.cols, a.q8k4.data() + t / 4 * super_blocks);
        }
        a.q8k4_ready = true;
    }

    // Rows [r0, r1) of a Q4_K_R8 matrix (8-row groups): 4 activation rows at
    // a time through gemm, the rest through gemv; a group cut by the range
    // goes through a scratch tile so only rows inside it are written.
    static void run_rows_r8(const Job& job, const Activations& a, int64_t r0, int64_t r1) {
        const TensorView& w = *job.item->w;
        const int64_t rows = w.rows();
        const int64_t cols = w.cols();
        const int64_t super_blocks = cols / quant::kSuperBlock;
        const auto* packed = reinterpret_cast<const quant::BlockQ4_Kx8*>(w.data);
        const int32_t n = job.item->n;
        float* y = job.item->y;
        const int64_t g_first = r0 / 8, g_last = (r1 + 7) / 8;           // groups touched
        const int64_t g_in0 = (r0 + 7) / 8, g_in1 = std::max(g_in0, r1 / 8);  // groups fully inside
        auto partial = [&](int64_t g, int32_t t, int32_t count) {
            float tile[4 * 8];
            const quant::BlockQ4_Kx8* wg = packed + g * super_blocks;
            if (count == 4) {
                quant::gemm_q4_K_r8(wg, cols, 0, 1, a.q8k4.data() + t / 4 * super_blocks, tile, 8);
            } else {
                quant::gemv_q4_K_r8(wg, cols, 0, 1, a.q8k.data() + t * super_blocks, tile);
            }
            for (int32_t k = 0; k < count; ++k) {
                for (int64_t r = std::max(r0, 8 * g); r < std::min(r1, 8 * g + 8); ++r) {
                    y[(t + k) * rows + r] = tile[k * 8 + (r - 8 * g)];
                }
            }
        };
        for (int32_t t = 0; t < n;) {
            const int32_t count = t + 4 <= n ? 4 : 1;
            if (g_in1 > g_in0) {
                if (count == 4) {
                    quant::gemm_q4_K_r8(packed, cols, g_in0, g_in1, a.q8k4.data() + t / 4 * super_blocks, y + t * rows, rows);
                } else {
                    quant::gemv_q4_K_r8(packed, cols, g_in0, g_in1, a.q8k.data() + t * super_blocks, y + t * rows);
                }
            }
            for (int64_t g = g_first; g < g_last; ++g) {
                if (g < g_in0 || g >= g_in1) partial(g, t, count);
            }
            t += count;
        }
    }

    static void quantize(Activations& a, const float* x, int64_t cols, int32_t n, ActFormat format) {
        a.q8k4_ready = false;
        a.x = x;
        a.cols = cols;
        a.n = n;
        a.format = format;
        if (format == ActFormat::Q8_K) {
            const int64_t super_blocks = cols / quant::kSuperBlock;
            a.q8k.resize(static_cast<size_t>(super_blocks * n));
            for (int32_t t = 0; t < n; ++t) {
                quant::quantize_row_q8_K(x + t * cols, a.q8k.data() + t * super_blocks, cols);
            }
        } else if (format != ActFormat::Float) {
            const int64_t blocks = cols / quant::kBlock;
            a.q8.resize(static_cast<size_t>(blocks * n));
            for (int32_t t = 0; t < n; ++t) quant::quantize_row_q8_0(x + t * cols, a.q8.data() + t * blocks, cols);
            if (format == ActFormat::Q8_0Sums) {
                a.sums.resize(a.q8.size());
                quant::block_sums(a.q8.data(), a.sums.data(), cols * n);
            }
        }
    }

    // Rows [r0, r1) of one job's matrix against all of its input rows.
    static void run_rows(const Job& job, const Activations& a, int64_t r0, int64_t r1) {
        const TensorView& w = *job.item->w;
        const int64_t rows = w.rows();
        const int64_t cols = w.cols();
        const int64_t blocks = cols / quant::kBlock;
        const int64_t super_blocks = cols / quant::kSuperBlock;
        const size_t row_bytes = w.row_bytes();
        const int32_t n = job.item->n;
        const float* x = job.item->x;
        float* y = job.item->y;
        if (w.type == DType::Q4_K_R8) {
            run_rows_r8(job, a, r0, r1);
            return;
        }
        // Several activation rows: Q8_K and Q8_0 kernels decode each weight
        // row once for all of them (speculative verification, prefill).
        if (n > 1 && (job.format == ActFormat::Q8_K || (job.format == ActFormat::Q8_0 && w.type == DType::Q8_0))) {
            thread_local std::vector<const quant::BlockQ8_K*> q8k_rows;
            thread_local std::vector<const quant::BlockQ8_0*> q8_rows;
            thread_local std::vector<float> dots;
            dots.resize(static_cast<size_t>(n));
            if (job.format == ActFormat::Q8_K) {
                q8k_rows.resize(static_cast<size_t>(n));
                for (int32_t t = 0; t < n; ++t) q8k_rows[static_cast<size_t>(t)] = a.q8k.data() + t * super_blocks;
            } else {
                q8_rows.resize(static_cast<size_t>(n));
                for (int32_t t = 0; t < n; ++t) q8_rows[static_cast<size_t>(t)] = a.q8.data() + t * blocks;
            }
            for (int64_t r = r0; r < r1; ++r) {
                const uint8_t* row = w.data + static_cast<size_t>(r) * row_bytes;
                if (job.format == ActFormat::Q8_K) {
                    quant::dot_lowbit_q8_K_rows(w.type, row, q8k_rows.data(), n, cols, dots.data());
                } else {
                    quant::dot_q8_0_q8_0_rows(reinterpret_cast<const quant::BlockQ8_0*>(row), q8_rows.data(), n, cols,
                                              dots.data());
                }
                for (int32_t t = 0; t < n; ++t) y[t * rows + r] = dots[static_cast<size_t>(t)];
            }
            return;
        }
        for (int64_t r = r0; r < r1; ++r) {
            const uint8_t* row = w.data + static_cast<size_t>(r) * row_bytes;
            for (int32_t t = 0; t < n; ++t) {
                float v = 0.0f;
                switch (job.format) {
                    case ActFormat::Q8_K:
                        v = quant::dot_lowbit_q8_K(w.type, row, a.q8k.data() + t * super_blocks, cols);
                        break;
                    case ActFormat::Q8_0:
                    case ActFormat::Q8_0Sums:
                        v = quant::dot_quantized(w.type, row, a.q8.data() + t * blocks,
                                                 job.format == ActFormat::Q8_0Sums ? a.sums.data() + t * blocks : nullptr,
                                                 cols);
                        break;
                    case ActFormat::Float:
                        if (w.type == DType::F16) {
                            v = quant::dot_f16_f32(reinterpret_cast<const uint16_t*>(row), x + t * cols, cols);
                        } else if (w.type == DType::BF16) {
                            v = quant::dot_bf16_f32(reinterpret_cast<const uint16_t*>(row), x + t * cols, cols);
                        } else {
                            v = quant::dot_f32(reinterpret_cast<const float*>(row), x + t * cols, cols);
                        }
                        break;
                }
                y[t * rows + r] = v;
            }
        }
    }

    ThreadPool& pool_;
    std::vector<MatmulItem> items_;
    std::vector<Activations> acts_;  // grows to the largest batch's distinct inputs, then reused
    std::vector<Job> jobs_;
    std::vector<int64_t> row_end_;   // cumulative row counts: job i owns rows [row_end_[i-1], row_end_[i])
};

}  // namespace

std::unique_ptr<Backend> make_cpu_backend(ThreadPool& pool) { return std::make_unique<CpuBackend>(pool); }

}  // namespace liyab
