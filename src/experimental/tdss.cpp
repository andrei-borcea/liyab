#include "liyab/experimental/tdss.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "core/quant.h"
#include "core/thread_pool.h"

#if defined(LIYAB_USE_NEON) && defined(__ARM_NEON)
#include <arm_neon.h>
#define LIYAB_TDSS_NEON 1
#endif

namespace liyab::experimental {

namespace {

// Block layout: [0..1] fp16 scale, [2..9] 16 nibbles (value k in byte k/2,
// low nibble for even k), [10..13] little-endian 32-bit index word (value k
// at bits 2k..2k+1 = position of the value inside its group of 4).
void encode_block(const float* w, uint8_t* out) {
    float kept[16];
    uint32_t index = 0;
    for (int g = 0; g < 8; ++g) {
        const float* group = w + g * 4;
        // Two largest magnitudes; ties keep the lower index. Stored in column order.
        int a = 0;
        for (int i = 1; i < 4; ++i) {
            if (std::fabs(group[i]) > std::fabs(group[a])) a = i;
        }
        int b = a == 0 ? 1 : 0;
        for (int i = 0; i < 4; ++i) {
            if (i != a && std::fabs(group[i]) > std::fabs(group[b])) b = i;
        }
        const int lo = std::min(a, b);
        const int hi = std::max(a, b);
        kept[2 * g] = group[lo];
        kept[2 * g + 1] = group[hi];
        index |= static_cast<uint32_t>(lo) << (4 * g);
        index |= static_cast<uint32_t>(hi) << (4 * g + 2);
    }
    // Symmetric INT4 over the 16 kept values (same rule as Q4_0).
    float amax = 0.0f;
    float max = 0.0f;
    for (const float v : kept) {
        if (std::fabs(v) > amax) {
            amax = std::fabs(v);
            max = v;
        }
    }
    const float d = max / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const uint16_t dh = quant::fp32_to_fp16(d);
    std::memcpy(out, &dh, 2);
    for (int k = 0; k < 16; k += 2) {
        const int q0 = std::clamp(static_cast<int>(kept[k] * id + 8.5f), 0, 15);
        const int q1 = std::clamp(static_cast<int>(kept[k + 1] * id + 8.5f), 0, 15);
        out[2 + k / 2] = static_cast<uint8_t>(q0 | (q1 << 4));
    }
    std::memcpy(out + 10, &index, 4);
}

inline int column_of(uint32_t index, int k) { return (k / 2) * 4 + static_cast<int>((index >> (2 * k)) & 3u); }

float dot_block_scalar(const uint8_t* blk, const quant::BlockQ8_0& x) {
    uint16_t dh;
    uint32_t index;
    std::memcpy(&dh, blk, 2);
    std::memcpy(&index, blk + 10, 4);
    int32_t sum = 0;
    for (int k = 0; k < 16; ++k) {
        const uint8_t byte = blk[2 + k / 2];
        const int q = ((k & 1) ? (byte >> 4) : (byte & 0x0F)) - 8;
        sum += q * x.qs[column_of(index, k)];
    }
    return static_cast<float>(sum) * quant::fp16_to_fp32(dh) * quant::fp16_to_fp32(x.d);
}

float dot_row(const uint8_t* row, const quant::BlockQ8_0* x, int64_t blocks) {
    int64_t b = 0;
    float sum = 0.0f;
#if defined(LIYAB_TDSS_NEON)
    // Per lane k: byte k/4 of the index word, shifted right by 2*(k%4), & 3,
    // plus the group base 4*(k/2) -> absolute column 0..31.
    static const uint8_t kReplicate[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};
    static const int8_t kShift[16] = {0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6};
    static const uint8_t kBase[16] = {0, 0, 4, 4, 8, 8, 12, 12, 16, 16, 20, 20, 24, 24, 28, 28};
    const uint8x16_t replicate = vld1q_u8(kReplicate);
    const int8x16_t shift = vld1q_s8(kShift);
    const uint8x16_t base = vld1q_u8(kBase);
    const uint8x16_t three = vdupq_n_u8(3);
    const uint8x8_t mask = vdup_n_u8(0x0F);
    const int8x16_t eight = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (; b < blocks; ++b) {
        const uint8_t* blk = row + b * Sparse24Matrix::kBlockBytes;
        uint32_t index;
        uint16_t dh;
        std::memcpy(&index, blk + 10, 4);
        std::memcpy(&dh, blk, 2);
        const uint8x16_t idx_bytes = vreinterpretq_u8_u32(vdupq_n_u32(index));
        uint8x16_t cols = vqtbl1q_u8(idx_bytes, replicate);
        cols = vandq_u8(vshlq_u8(cols, shift), three);
        cols = vaddq_u8(cols, base);
        // Gather the 16 activations that meet a kept weight: one TBL over 32 bytes.
        const int8x16x2_t table = {{vld1q_s8(x[b].qs), vld1q_s8(x[b].qs + 16)}};
        const int8x16_t xg = vqtbl2q_s8(table, cols);
        // 16 INT4 weights in value order.
        const uint8x8_t packed = vld1_u8(blk + 2);
        const uint8x8x2_t zipped = vzip_u8(vand_u8(packed, mask), vshr_n_u8(packed, 4));
        const int8x16_t wv = vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(zipped.val[0], zipped.val[1])), eight);
#if defined(__ARM_FEATURE_DOTPROD)
        const int32x4_t dot = vdotq_s32(vdupq_n_s32(0), wv, xg);
#else
        const int16x8_t p0 = vmull_s8(vget_low_s8(wv), vget_low_s8(xg));
        const int16x8_t p1 = vmull_s8(vget_high_s8(wv), vget_high_s8(xg));
        const int32x4_t dot = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(dot), quant::fp16_to_fp32(dh) * quant::fp16_to_fp32(x[b].d));
    }
    sum = vaddvq_f32(acc);
#endif
    for (; b < blocks; ++b) sum += dot_block_scalar(row + b * Sparse24Matrix::kBlockBytes, x[b]);
    return sum;
}

}  // namespace

Result<Sparse24Matrix> Sparse24Matrix::build(const TensorView& w) {
    if (w.cols() % 32 != 0 || w.rows() <= 0) {
        return Status(ErrorCode::InvalidArgument, "2:4 sparsification needs rows of whole 32-column blocks");
    }
    Sparse24Matrix m;
    m.rows_ = w.rows();
    m.cols_ = w.cols();
    const int64_t blocks = m.cols_ / 32;
    m.data_.resize(static_cast<size_t>(m.rows_ * blocks) * kBlockBytes);
    std::vector<float> row(static_cast<size_t>(m.cols_));
    for (int64_t r = 0; r < m.rows_; ++r) {
        quant::dequantize_row(w.type, w.row(r), row.data(), m.cols_);
        uint8_t* out = m.data_.data() + static_cast<size_t>(r * blocks) * kBlockBytes;
        for (int64_t b = 0; b < blocks; ++b) encode_block(row.data() + b * 32, out + b * kBlockBytes);
    }
    return m;
}

void Sparse24Matrix::dense_row(int64_t row, float* out) const {
    const int64_t blocks = cols_ / 32;
    std::fill(out, out + cols_, 0.0f);
    for (int64_t b = 0; b < blocks; ++b) {
        const uint8_t* blk = data_.data() + static_cast<size_t>(row * blocks + b) * kBlockBytes;
        uint16_t dh;
        uint32_t index;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&index, blk + 10, 4);
        const float d = quant::fp16_to_fp32(dh);
        for (int k = 0; k < 16; ++k) {
            const uint8_t byte = blk[2 + k / 2];
            const int q = ((k & 1) ? (byte >> 4) : (byte & 0x0F)) - 8;
            out[b * 32 + column_of(index, k)] = d * static_cast<float>(q);
        }
    }
}

void Sparse24Matrix::matmul(const float* x, float* y, int32_t n, ThreadPool& pool) const {
    const int64_t blocks = cols_ / 32;
    std::vector<quant::BlockQ8_0> xq(static_cast<size_t>(blocks * n));
    for (int32_t t = 0; t < n; ++t) quant::quantize_row_q8_0(x + t * cols_, xq.data() + t * blocks, cols_);
    const size_t row_bytes = static_cast<size_t>(blocks) * kBlockBytes;
    pool.parallel_for(rows_, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; ++r) {
            const uint8_t* row = data_.data() + static_cast<size_t>(r) * row_bytes;
            for (int32_t t = 0; t < n; ++t) y[t * rows_ + r] = dot_row(row, xq.data() + t * blocks, blocks);
        }
    });
}

Result<std::unique_ptr<Tdss>> Tdss::create(std::span<const std::array<const TensorView*, 3>> ffn, ThreadPool& pool,
                                           TdssConfig config) {
    std::unique_ptr<Tdss> tdss(new Tdss());
    tdss->pool_ = &pool;
    tdss->config_ = config;
    tdss->layers_.resize(ffn.size());
    // Build every projection in parallel: pruning + quantization of the whole
    // FFN is the expensive one-time step.
    std::vector<Status> errors(ffn.size() * 3);
    pool.parallel_for(static_cast<int64_t>(ffn.size() * 3), [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const auto l = static_cast<size_t>(i / 3);
            const auto p = static_cast<size_t>(i % 3);
            auto m = Sparse24Matrix::build(*ffn[l][p]);
            if (m) tdss->layers_[l][p] = std::move(m).value();
            else errors[static_cast<size_t>(i)] = m.status();
        }
    });
    for (const Status& s : errors) {
        if (!s.is_ok()) return s;
    }
    for (const auto& layer : ffn) {
        for (const TensorView* t : layer) tdss->dense_bytes_ += t->nbytes;
    }
    tdss->active_ = config.force;
    return tdss;
}

bool Tdss::update(const PowerPolicy& policy) {
    active_ = config_.force || policy.throttled;
    return active_;
}

bool Tdss::ffn_matmul(int32_t layer, FfnProjection projection, const float* x, float* y, int32_t n) {
    if (!active_ || layer < 0 || static_cast<size_t>(layer) >= layers_.size()) return false;
    layers_[static_cast<size_t>(layer)][static_cast<size_t>(projection)].matmul(x, y, n, *pool_);
    return true;
}

size_t Tdss::sparse_bytes() const noexcept {
    size_t total = 0;
    for (const auto& layer : layers_) {
        for (const auto& m : layer) total += m.bytes();
    }
    return total;
}

}  // namespace liyab::experimental
