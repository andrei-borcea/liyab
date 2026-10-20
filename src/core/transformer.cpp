#include "core/transformer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <istream>
#include <limits>
#include <numbers>
#include <ostream>
#include <random>

#include "core/log.h"
#include "core/portable_math.h"
#include "core/quant.h"
#include "core/thread_pool.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace liyab {

namespace {

// The factor RMSNorm multiplies x by before its weight: 1 / sqrt(mean(x^2) + eps).
float rms_scale(const float* x, int32_t n, float eps) {
    double ss = 0.0;
    for (int32_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    return static_cast<float>(1.0 / std::sqrt(ss / n + eps));
}

void rmsnorm(const float* x, const float* w, float* out, int32_t n, float eps) {
    const float scale = rms_scale(x, n, eps);
    for (int32_t i = 0; i < n; ++i) out[i] = x[i] * scale * w[i];
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

// GELU with the tanh approximation (PyTorch's gelu(approximate="tanh")):
// 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))).
float gelu_tanh(float x) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    // The C library's tanh (not portable_tanh): Apple's and Android's agree on
    // every golden model's GELU, and switching would change pinned hashes.
    return 0.5f * x * (1.0f + std::tanh(kSqrt2OverPi * (x + 0.044715f * x * x * x)));
}

// The gated FFN's elementwise step: out[i] = act(gate[i]) * up[i] (out may be gate).
void gated_activation(FfnActivation act, const float* gate, const float* up, float* out, size_t n) {
    if (act == FfnActivation::GeluTanh) {
        for (size_t i = 0; i < n; ++i) out[i] = gelu_tanh(gate[i]) * up[i];
        return;
    }
    for (size_t i = 0; i < n; ++i) out[i] = silu(gate[i]) * up[i];
}
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }  // ggml's threshold

// x / sqrt(sum(x^2) + eps): the L2 normalization of DeltaNet queries and keys.
void l2norm(float* x, int32_t n, float eps) {
    double ss = 0.0;
    for (int32_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
    const auto scale = static_cast<float>(1.0 / std::sqrt(ss + eps));
    for (int32_t i = 0; i < n; ++i) x[i] *= scale;
}

// One token of the gated delta rule for one value head, S [hd x hd] row j
// mapping keys to value component j:
//   S = decay * S;  S += strength * (v - S k) k^T;  o = (S q) * q_scale
// Rows are independent: each is decayed and read against k, corrected, then
// read against q, in two passes over its hd values. On NEON four
// accumulators break the dependency chains of the dot products (a fixed
// order, so results do not depend on the thread count); the scalar loop
// elsewhere keeps one.
void delta_rule_step(float* S, const float* q, const float* k, const float* v, float decay, float strength,
                     float q_scale, float* o, size_t hd) {
    for (size_t j = 0; j < hd; ++j) {
        float* row = S + j * hd;
        size_t i = 0;
        float sk = 0.0f;
#if defined(__ARM_NEON)
        if (hd % 16 == 0) {
            const float32x4_t dv = vdupq_n_f32(decay);
            float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
            for (; i < hd; i += 16) {
                float32x4_t r0 = vmulq_f32(vld1q_f32(row + i), dv), r1 = vmulq_f32(vld1q_f32(row + i + 4), dv);
                float32x4_t r2 = vmulq_f32(vld1q_f32(row + i + 8), dv), r3 = vmulq_f32(vld1q_f32(row + i + 12), dv);
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(k + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(k + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(k + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(k + i + 12));
            }
            sk = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
            const float32x4_t delta = vdupq_n_f32((v[j] - sk) * strength);
            a0 = a1 = a2 = a3 = vdupq_n_f32(0.0f);
            for (i = 0; i < hd; i += 16) {
                const float32x4_t r0 = vfmaq_f32(vld1q_f32(row + i), vld1q_f32(k + i), delta);
                const float32x4_t r1 = vfmaq_f32(vld1q_f32(row + i + 4), vld1q_f32(k + i + 4), delta);
                const float32x4_t r2 = vfmaq_f32(vld1q_f32(row + i + 8), vld1q_f32(k + i + 8), delta);
                const float32x4_t r3 = vfmaq_f32(vld1q_f32(row + i + 12), vld1q_f32(k + i + 12), delta);
                vst1q_f32(row + i, r0);
                vst1q_f32(row + i + 4, r1);
                vst1q_f32(row + i + 8, r2);
                vst1q_f32(row + i + 12, r3);
                a0 = vfmaq_f32(a0, r0, vld1q_f32(q + i));
                a1 = vfmaq_f32(a1, r1, vld1q_f32(q + i + 4));
                a2 = vfmaq_f32(a2, r2, vld1q_f32(q + i + 8));
                a3 = vfmaq_f32(a3, r3, vld1q_f32(q + i + 12));
            }
            o[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * q_scale;
            continue;
        }
#endif
        for (i = 0; i < hd; ++i) {
            row[i] *= decay;
            sk += row[i] * k[i];
        }
        const float delta = (v[j] - sk) * strength;
        float oq = 0.0f;
        for (i = 0; i < hd; ++i) {
            row[i] += k[i] * delta;
            oq += row[i] * q[i];
        }
        o[j] = oq * q_scale;
    }
}

// Adds the wall time of its scope to `total_ms` (Transformer::PhaseTimes).
class PhaseTimer {
public:
    explicit PhaseTimer(double& total_ms) : total_ms_(total_ms), start_(std::chrono::steady_clock::now()) {}
    ~PhaseTimer() { stop(); }
    PhaseTimer(const PhaseTimer&) = delete;
    PhaseTimer& operator=(const PhaseTimer&) = delete;
    // Ends the measurement early (later calls and the destructor add nothing).
    void stop() {
        if (stopped_) return;
        stopped_ = true;
        total_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
    }

private:
    double& total_ms_;
    std::chrono::steady_clock::time_point start_;
    bool stopped_ = false;
};

// Slice `e` of a stacked tensor [cols, rows, n] as a 2-D view: an expert of
// a MoE block, or a head of latent attention's per-head matrices.
TensorView expert_slice(const TensorView& t, int64_t e) {
    TensorView v = t;
    v.n_dims = 2;
    v.ne = {t.ne[0], t.ne[1], 1, 1};
    v.nbytes = t.row_bytes() * static_cast<size_t>(t.ne[1]);
    v.data = t.data + static_cast<size_t>(e) * v.nbytes;
    v.file_offset = t.file_offset + static_cast<uint64_t>(e) * v.nbytes;
    return v;
}

// Positions a query at `pos` of a block with `window` (0: none) sees through
// the cache: [sink_begin, sink_end) ∪ [first, pos].
struct Seen {
    int32_t sink_begin, sink_end, first;
};
Seen seen_by(const KvCache& kv, int32_t pos, int32_t window) {
    const KvCache::Visible vis = kv.visible(pos);
    Seen s{0, vis.sink_end, vis.first};
    if (window > 0) {
        const int32_t lo = pos - window + 1;
        s.first = std::max(s.first, lo);
        s.sink_begin = std::clamp(lo, 0, s.sink_end);
    }
    return s;
}

// Latent attention's expanded form, the GEMM that expands cached latents
// into per-head keys and values: columns are taken kExpandColumns at a time
// from weights packed per column tile ([tile][kv_rank][kExpandColumns]), so
// one tile's strip (kv_rank x 16 floats, 32 KiB on DeepSeek-V2-Lite) stays in
// L1 while every latent row passes through it, and kExpandRows rows share
// each strip load. Each output is one float sum over the latent channels in
// ascending order, from zero, whatever the rows around it: the vector lanes
// run separate columns, never parts of one sum, so the bits of a position's
// keys and values do not depend on how many positions a pass expands.
constexpr int32_t kExpandColumns = 16;
constexpr int32_t kExpandRows = 4;

// Rows [0, m) of `a` (m x k, row-major) times the strip `wp` (k x
// kExpandColumns): row j's column c lands at out[j * out_stride + offset[c]]
// for the `columns` (<= kExpandColumns) columns that exist.
void expand_strip(const float* a, int64_t m, int64_t k, const float* wp, float* out, size_t out_stride,
                  const size_t* offset, int32_t columns) {
    for (int64_t j0 = 0; j0 < m; j0 += kExpandRows) {
        // A last block of fewer rows repeats the last one (computed, not stored).
        const float* rows[kExpandRows];
        for (int64_t i = 0; i < kExpandRows; ++i) rows[i] = a + std::min(j0 + i, m - 1) * k;
        float acc[kExpandRows][kExpandColumns] = {};
        for (int64_t r = 0; r < k; ++r) {
            const float* w = wp + r * kExpandColumns;
            for (int32_t i = 0; i < kExpandRows; ++i) {
                const float x = rows[i][r];
                for (int32_t c = 0; c < kExpandColumns; ++c) acc[i][c] += x * w[c];
            }
        }
        const int64_t stored = std::min<int64_t>(kExpandRows, m - j0);
        for (int64_t i = 0; i < stored; ++i) {
            float* dst = out + static_cast<size_t>(j0 + i) * out_stride;
            for (int32_t c = 0; c < columns; ++c) dst[offset[c]] = acc[i][c];
        }
    }
}

// The expanded form's attention kernels. Each score is one dot product over
// kScoreLanes interleaved partial sums reduced in a fixed tree, and each
// weighted value one multiply-add chain over the positions in ascending
// order, whatever the queries and rows computed alongside: blocks of
// kScoreQueries tokens and kScoreKeys rows share every load, and a token's
// bits do not depend on which tokens share its block.
constexpr int32_t kScoreQueries = 4;
constexpr int32_t kScoreKeys = 2;
constexpr int32_t kScoreLanes = 8;
constexpr int32_t kValueColumns = 16;
// A task attends kAttendTokens tokens of one head (fewer when the pass has
// too few tasks for the threads), kAttendRows rows of keys and values at a time.
constexpr size_t kAttendTokens = 32;
constexpr size_t kAttendRows = 64;

// out[t][r] = q[t] . k[r] over n channels, for kScoreQueries queries and kScoreKeys rows.
void score_block(const float* const* q, const float* const* k, size_t n, float (*out)[kScoreKeys]) {
#if defined(__ARM_NEON)
    // The same sums as the portable loop below, lanes 0-3 and 4-7 in two
    // vectors (compilers leave the portable loop scalar: 64 accumulators).
    if (n % kScoreLanes == 0) {
        float32x4_t lo[kScoreQueries][kScoreKeys], hi[kScoreQueries][kScoreKeys];
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) lo[t][r] = hi[t][r] = vdupq_n_f32(0.0f);
        }
        for (size_t i = 0; i < n; i += kScoreLanes) {
            float32x4_t k_lo[kScoreKeys], k_hi[kScoreKeys];
            for (int32_t r = 0; r < kScoreKeys; ++r) {
                k_lo[r] = vld1q_f32(k[r] + i);
                k_hi[r] = vld1q_f32(k[r] + i + 4);
            }
            for (int32_t t = 0; t < kScoreQueries; ++t) {
                const float32x4_t q_lo = vld1q_f32(q[t] + i);
                const float32x4_t q_hi = vld1q_f32(q[t] + i + 4);
                for (int32_t r = 0; r < kScoreKeys; ++r) {
                    lo[t][r] = vfmaq_f32(lo[t][r], q_lo, k_lo[r]);
                    hi[t][r] = vfmaq_f32(hi[t][r], q_hi, k_hi[r]);
                }
            }
        }
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) {
                const float32x4_t s = vaddq_f32(lo[t][r], hi[t][r]);  // a[l] + a[l + 4]
                out[t][r] = (vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1)) + (vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3));
            }
        }
        return;
    }
#endif
    float acc[kScoreQueries][kScoreKeys][kScoreLanes] = {};
    size_t i = 0;
    for (; i + kScoreLanes <= n; i += kScoreLanes) {
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) {
                for (int32_t l = 0; l < kScoreLanes; ++l) acc[t][r][l] += q[t][i + l] * k[r][i + l];
            }
        }
    }
    for (int32_t l = 0; i < n; ++i, ++l) {
        for (int32_t t = 0; t < kScoreQueries; ++t) {
            for (int32_t r = 0; r < kScoreKeys; ++r) acc[t][r][l] += q[t][i] * k[r][i];
        }
    }
    for (int32_t t = 0; t < kScoreQueries; ++t) {
        for (int32_t r = 0; r < kScoreKeys; ++r) {
            const float* a = acc[t][r];
            out[t][r] = ((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]));
        }
    }
}

// o[t][d] += sum over rows i in [a, b) of p[t][i] * v[i * stride + d], d in
// [0, width), for T tokens: kValueColumns columns of every token held in
// registers while the rows go by.
template <int32_t T>
void weigh_values(float* const* o, const float* const* p, const float* v, size_t stride, size_t a, size_t b,
                  size_t width) {
    if (a >= b) return;
#if defined(__ARM_NEON)
    // The same multiply-adds as the portable loop below, in vectors (which
    // compilers do not reliably make of it).
    if (width % kValueColumns == 0) {
        for (size_t d0 = 0; d0 < width; d0 += kValueColumns) {
            float32x4_t acc[T][4];
            for (int32_t t = 0; t < T; ++t) {
                for (int32_t c = 0; c < 4; ++c) acc[t][c] = vld1q_f32(o[t] + d0 + 4 * c);
            }
            for (size_t i = a; i < b; ++i) {
                const float* row = v + i * stride + d0;
                const float32x4_t r0 = vld1q_f32(row), r1 = vld1q_f32(row + 4);
                const float32x4_t r2 = vld1q_f32(row + 8), r3 = vld1q_f32(row + 12);
                for (int32_t t = 0; t < T; ++t) {
                    const float32x4_t pi = vdupq_n_f32(p[t][i]);
                    acc[t][0] = vfmaq_f32(acc[t][0], pi, r0);
                    acc[t][1] = vfmaq_f32(acc[t][1], pi, r1);
                    acc[t][2] = vfmaq_f32(acc[t][2], pi, r2);
                    acc[t][3] = vfmaq_f32(acc[t][3], pi, r3);
                }
            }
            for (int32_t t = 0; t < T; ++t) {
                for (int32_t c = 0; c < 4; ++c) vst1q_f32(o[t] + d0 + 4 * c, acc[t][c]);
            }
        }
        return;
    }
#endif
    for (size_t d0 = 0; d0 < width; d0 += kValueColumns) {
        const size_t w = std::min<size_t>(kValueColumns, width - d0);
        float acc[T][kValueColumns] = {};
        for (int32_t t = 0; t < T; ++t) std::copy(o[t] + d0, o[t] + d0 + w, acc[t]);
        if (w == static_cast<size_t>(kValueColumns)) {
            for (size_t i = a; i < b; ++i) {
                const float* row = v + i * stride + d0;
                for (int32_t t = 0; t < T; ++t) {
                    const float pi = p[t][i];
                    for (int32_t c = 0; c < kValueColumns; ++c) acc[t][c] += pi * row[c];
                }
            }
        } else {
            for (size_t i = a; i < b; ++i) {
                const float* row = v + i * stride + d0;
                for (int32_t t = 0; t < T; ++t) {
                    const float pi = p[t][i];
                    for (size_t c = 0; c < w; ++c) acc[t][c] += pi * row[c];
                }
            }
        }
        for (int32_t t = 0; t < T; ++t) std::copy(acc[t], acc[t] + w, o[t] + d0);
    }
}


}  // namespace

LatentFormChooser::Work LatentFormChooser::work(const ModelConfig& c, int64_t n, int64_t visible, int64_t positions) {
    const LatentAttention& m = c.mla;
    const auto heads = static_cast<double>(c.n_head);
    const double project = static_cast<double>(m.kv_rank) * (m.nope_dim + m.v_dim);  // per head: k_b and v_b
    Work w;
    w.absorbed = heads * (static_cast<double>(n) * project +
                          static_cast<double>(visible) * (c.rope_dim + 2.0 * m.kv_rank));
    w.expanded = heads * (static_cast<double>(positions + 1) * project +
                          static_cast<double>(visible) * (m.nope_dim + c.rope_dim + m.v_dim));
    return w;
}

Status LatentFormChooser::set_ratio(float ratio) noexcept {
    if (!(ratio >= kMinRatio && ratio <= kMaxRatio)) {
        return Status(ErrorCode::InvalidArgument, "latent form ratio out of range");
    }
    ratio_ = ratio;
    return Status::ok();
}

float LatentFormChooser::round_ratio(double measured) noexcept {
    if (!(measured > 0.0) || !std::isfinite(measured)) return 1.0f;
    const double steps = std::round(std::log2(measured) * 8.0);
    return std::clamp(static_cast<float>(std::exp2(steps / 8.0)), kMinRatio, kMaxRatio);
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------
Result<std::unique_ptr<Transformer>> Transformer::load(std::unique_ptr<MmapLoader> file,
                                                       const TransformerOptions& options) {
    if (!file) return Status(ErrorCode::InvalidArgument, "null model file");
    std::unique_ptr<Transformer> model(new Transformer());
    model->file_ = std::move(file);
    const MmapLoader& f = *model->file_;

    auto described = describe_model(f);
    if (!described) return described.status();
    model->config_ = std::move(described).value();
    const ModelConfig& c = model->config_;
    model->expert_mass_ = options.expert_mass;
    model->max_experts_ = options.max_experts;
    model->skip_slow_ = options.skip_slow;
    model->latent_form_ = options.latent_form;

    LIYAB_RETURN_IF_ERROR(model->bind_weights());
    const Layer& first = model->layers_[0];
    const TensorView* ffn_weight = c.blocks[0].ffn == FfnKind::Moe ? first.w[kUpExps] : first.w[kUp];
    LIYAB_LOG_INFO("bound %d blocks of %s weights (%s, head_dim %d, %d attention + %d DeltaNet blocks)", c.n_layers,
                   std::string(dtype_traits(ffn_weight->type).name).c_str(), c.arch.c_str(), c.head_dim,
                   c.n_attn_layers, c.n_layers - c.n_attn_layers);
    if (c.n_expert > 0) {
        LIYAB_LOG_INFO("MoE: %d experts, top-%d, expert ff %d, shared ff %d, %s gating%s", c.n_expert, c.n_expert_used,
                       c.n_ff_expert, c.n_ff_shared, c.moe_gating == MoeGating::Softmax ? "softmax" : "sigmoid",
                       c.moe_norm_weights ? ", renormalized" : "");
    }

    // Context and KV cache. Automatic: what kKvBudgetBytes of KV holds
    // (4096..32768 positions, at most the training context). The KV cache is
    // paged, so memory follows the positions actually used; a hybrid model
    // with few attention blocks gets a long context (Qwen3.6-35B-A3B: 10.6 KiB
    // per position, ~24k), where a flat 4096 forced chats with tools to drop
    // old turns, and each drop meant processing the whole history again.
    KvCacheConfig kv;
    // Only attention blocks have keys and values (at least one slot keeps the
    // cache well-formed for purely recurrent stacks).
    kv.n_layers = std::max(c.n_attn_layers, 1);
    kv.n_head_kv = c.n_head_kv;
    kv.head_dim = c.head_dim;
    kv.type = options.kv_type;
    // Latent attention: one row [k_pe | latent] per position and block, its
    // latent channels serving as the values (ModelConfig::mla).
    kv.values_in_keys = c.mla.kv_rank > 0;
    kv.value_offset = kv.values_in_keys ? c.rope_dim : 0;
    model->max_batch_ = std::max(1, options.max_batch);
    // Per-block layout: a windowed block (Gemma 2 / 3, Phi-3) keeps only the
    // rows its window still shows, so it costs a bounded amount whatever the
    // context, and a position costs only what the blocks seeing every
    // position keep for it (KvCache::kept_bytes()). Without windowed blocks
    // the automatic length is the plain per-token formula, as before.
    std::vector<int32_t> windows(static_cast<size_t>(kv.n_layers), 0);
    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = model->layers_[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        if (L.kv_slot >= 0 && plan.mixer == MixerKind::Attention && plan.windowed && c.sliding_window > 0 &&
            !options.keep_windowed_rows) {
            windows[static_cast<size_t>(L.kv_slot)] = c.sliding_window;
        }
    }
    const size_t kv_per_token = KvCache::bytes_per_token(kv);
    auto automatic = static_cast<int32_t>(std::clamp<size_t>(kKvBudgetBytes / std::max<size_t>(kv_per_token, 1),
                                                             4096, 32768));
    if (std::any_of(windows.begin(), windows.end(), [](int32_t w) { return w > 0; })) {
        // The windowed blocks' share is set aside once (their window plus the
        // rollback margin, whatever the context); the rest of the budget goes
        // to the blocks keeping every position, at their own cost per position.
        KvCacheConfig windowed_only = kv;
        int32_t every = 0;
        for (const int32_t w : windows) {
            if (w > 0) {
                windowed_only.layer_windows.push_back(w);
            } else {
                ++every;
            }
        }
        windowed_only.n_layers = static_cast<int32_t>(windowed_only.layer_windows.size());
        const size_t fixed =
            KvCache::kept_bytes(windowed_only, std::numeric_limits<int32_t>::max() / 2, model->max_batch_);
        KvCacheConfig every_only = kv;
        every_only.n_layers = every;
        const size_t per_position = every > 0 ? KvCache::bytes_per_token(every_only) : 0;
        const size_t room = kKvBudgetBytes > fixed ? kKvBudgetBytes - fixed : 0;
        automatic = per_position == 0 ? 32768
                                      : static_cast<int32_t>(std::clamp<size_t>(room / per_position, 4096, 32768));
    }
    model->context_length_ = options.context_length > 0
                                 ? options.context_length
                                 : (c.n_ctx_train > 0 ? std::min(c.n_ctx_train, automatic) : automatic);
    if (c.n_ctx_train > 0 && model->context_length_ > c.n_ctx_train) {
        LIYAB_LOG_WARN("context %d exceeds the model's training context %d", model->context_length_, c.n_ctx_train);
    }
    // A window holding the whole context keeps every position anyway: such
    // blocks stay with the others (a windowed group would only add bookkeeping).
    for (int32_t& w : windows) {
        if (w >= model->context_length_) w = 0;
    }
    if (std::any_of(windows.begin(), windows.end(), [](int32_t w) { return w > 0; })) kv.layer_windows = windows;
    kv.max_positions = model->context_length_;
    kv.window = options.sliding_window > 0 && options.sliding_window < model->context_length_ ? options.sliding_window : 0;
    kv.sink_tokens = std::clamp(options.sink_tokens, 0, kv.page_tokens);
    if (kv.type != KvCacheType::F16 && (c.head_dim % quant::kBlock != 0 || kv.value_offset % quant::kBlock != 0)) {
        LIYAB_LOG_WARN("head_dim %d (values from channel %d) is not in whole blocks of %d; using an F16 KV cache",
                       c.head_dim, kv.value_offset, quant::kBlock);
        kv.type = KvCacheType::F16;
    }
    auto cache = KvCache::create(kv);
    if (!cache) return cache.status();
    model->kv_ = std::move(cache).value();

    // RoPE inverse frequencies, with optional per-pair factors: Llama 3.x's
    // rope_freqs, or LongRoPE's (Phi-3), whose long factors apply when the
    // context is longer than the original one. The reference picks them by
    // the sequence's length instead; with one table for the whole context,
    // keys never need rotating again (a context up to the original length
    // matches the reference exactly).
    std::vector<float> factors(static_cast<size_t>(c.rope_dim / 2), 1.0f);
    const char* factor_tensor = c.rope_ctx_orig == 0                         ? "rope_freqs.weight"
                                : model->context_length_ > c.rope_ctx_orig ? "rope_factors_long.weight"
                                                                           : "rope_factors_short.weight";
    if (const TensorView* rf = f.tensor(factor_tensor)) {
        if (rf->elements() != c.rope_dim / 2 || rf->type != DType::F32) {
            return Status(ErrorCode::InvalidModel, std::string(factor_tensor) + " has an unexpected shape");
        }
        quant::dequantize_row(DType::F32, rf->data, factors.data(), rf->elements());
    }
    auto table = [&](float base, float scale, const std::vector<float>& divisors, const YarnRope* yarn) {
        RopeTable t;
        t.scale = scale;
        t.inv_freq.resize(divisors.size());
        // YaRN: pair i keeps the share e(i) of its own frequency and takes
        // 1 - e(i) of it divided by the factor; e falls linearly from 1 to 0
        // between the pairs that turn beta_fast and beta_slow times over the
        // original context (the reference's correction range), times
        // ext_factor. One table for every position: keys never need
        // rotating again, and the result does not depend on the context.
        double low = 0.0, high = 0.0;
        if (yarn != nullptr) {
            const double dims = c.rope_dim;
            auto pair_turning = [&](double turns) {  // the (fractional) pair that turns `turns` times
                return dims * std::log(yarn->ctx_orig / (turns * 2.0 * std::numbers::pi)) /
                       (2.0 * std::log(static_cast<double>(base)));
            };
            low = pair_turning(yarn->beta_fast);
            high = pair_turning(yarn->beta_slow);
            if (yarn->truncate) {
                low = std::floor(low);
                high = std::ceil(high);
            }
            low = std::max(low, 0.0);
            high = std::min(high, dims - 1.0);
            if (high == low) high += 0.001;  // as the reference: no division by zero
        }
        for (size_t i = 0; i < divisors.size(); ++i) {
            const double freq = std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / c.rope_dim);
            if (yarn == nullptr) {
                t.inv_freq[i] = static_cast<float>(freq / divisors[i]);
                continue;
            }
            const double ramp = std::clamp((static_cast<double>(i) - low) / (high - low), 0.0, 1.0);
            const double extrapolated = (1.0 - ramp) * yarn->ext_factor;
            const double original = freq / divisors[i];
            t.inv_freq[i] = static_cast<float>(original / yarn->factor * (1.0 - extrapolated) + original * extrapolated);
        }
        return t;
    };
    model->rope_tables_.push_back(table(c.rope_base, c.rope_scale, factors, c.yarn.factor > 0.0f ? &c.yarn : nullptr));
    model->rope_tables_.back().magnitude = c.rope_magnitude;
    // Windowed blocks with a base of their own rotate unscaled, without the
    // global frequency factors (Gemma 3's local RoPE).
    if (c.rope_base_swa > 0.0f) {
        model->rope_tables_.push_back(table(c.rope_base_swa, 1.0f, std::vector<float>(factors.size(), 1.0f), nullptr));
    }
    for (int32_t l = 0; l < c.n_layers; ++l) {
        Layer& L = model->layers_[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        if (plan.mixer != MixerKind::Attention || !plan.windowed) continue;
        L.window = c.sliding_window;
        if (c.rope_base_swa > 0.0f) L.rope_table = 1;
    }

    model->extra_context_bytes_ =
        static_cast<size_t>(std::max(options.extra_contexts, 0)) *
        static_cast<size_t>(1 + kStateSnapshots + std::max(options.extra_rollback_window, 0)) *
        model->recurrent_state_bytes();
    if (c.n_expert > 0) {
        LIYAB_RETURN_IF_ERROR(
            model->attach_expert_store(options.expert_cache_bytes, options.memory_budget_bytes, options.reserve_bytes,
                                       options.requant_bits, options.repack_cpu && quant::repack_kernels_available()));
    }
    model->file_->configure_layers(c.n_layers);
    LIYAB_LOG_INFO("%s: %d layers, d=%d, ff=%d, heads=%d/%d x %d, vocab=%d, ctx=%d%s, kv=%s, %.1f KiB/token",
                   c.arch.c_str(), c.n_layers, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim, c.n_vocab,
                   model->context_length_, kv.window ? " (sliding)" : "",
                   std::string(dtype_traits(model->kv_->dtype()).name).c_str(),
                   static_cast<double>(KvCache::bytes_per_token(kv)) / 1024.0);
    if (model->kv_->has_windowed_layers()) {
        const KvCacheConfig& made = model->kv_->config();
        int32_t every = 0;
        for (const int32_t w : made.layer_windows) every += w == 0 ? 1 : 0;
        const int32_t ctx = model->context_length_;
        LIYAB_LOG_INFO("KV per block: %d of %d attention blocks keep every position (%.1f KiB/token), the others their "
                       "window of %d; a full context holds %.1f MiB instead of %.1f MiB",
                       every, made.n_layers, static_cast<double>(model->kv_->rows_bytes(1)) / 1024.0, c.sliding_window,
                       static_cast<double>(KvCache::kept_bytes(made, ctx, model->max_batch_)) / (1024.0 * 1024.0),
                       static_cast<double>(KvCache::bytes_per_token(made)) * ctx / (1024.0 * 1024.0));
    }
    if (c.mla.kv_rank > 0) {
        LIYAB_LOG_INFO("latent attention: a %d-wide latent + %d RoPE channels cached per position, %d heads of %d + %d "
                       "-> %d%s",
                       c.mla.kv_rank, c.rope_dim, c.n_head, c.mla.nope_dim, c.rope_dim, c.mla.v_dim,
                       c.mla.q_rank > 0 ? ", low-rank queries" : "");
    }
    if (c.hybrid()) {
        LIYAB_LOG_INFO("DeltaNet: %d recurrent blocks, %d/%d heads x %d, conv %d, state %.1f MiB", c.n_layers - c.n_attn_layers,
                       c.ssm_k_heads, c.ssm_v_heads, c.ssm_head_dim, c.ssm_conv_kernel,
                       static_cast<double>(model->recurrent_state_bytes()) / (1024.0 * 1024.0));
    }
    // CPU-only models: the resident Q4_K / Q5_K / Q6_K / Q8_0 matrices in the
    // layout of the i8mm batched kernels (a lossless rearrangement, same
    // size). With expert streaming that is every matrix but the experts (3-D,
    // read from storage in the file's layout); a model under the layer
    // streaming window keeps the file's layout throughout.
    if (options.repack_cpu && !model->file_->streaming()) {
        struct Repack {
            DType from, to;
            int64_t rows, cols;  // required multiples
        };
        for (const Repack& r : {Repack{DType::Q4_K, DType::Q4_K_R8, 8, quant::kSuperBlock},
                                Repack{DType::Q5_K, DType::Q5_K_R8, 8, quant::kSuperBlock},
                                Repack{DType::Q6_K, DType::Q6_K_R8, 8, quant::kSuperBlock},
                                Repack{DType::Q8_0, DType::Q8_0_R4, 4, quant::kBlock}}) {
            auto repacked = model->file_->requantize(
                [&](const TensorView& t) {
                    return t.type == r.from && t.n_dims == 2 && t.rows() % r.rows == 0 && t.cols() % r.cols == 0 &&
                           t.name != "token_embd.weight" && !model->file_->converted(t);
                },
                r.to);
            if (!repacked) return repacked.status();
        }
    }
    return model;
}

Status Transformer::bind_weights() {
    const MmapLoader& f = *file_;
    const ModelConfig& c = config_;
    const int64_t q_dim = int64_t{c.n_head} * c.head_dim;
    const int64_t kv_dim = int64_t{c.n_head_kv} * c.head_dim;

    auto matrix = [&](const std::string& name, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
        if (t->cols() != cols || t->rows() != rows) {
            return Status(ErrorCode::InvalidModel, "tensor '" + name + "' has shape [" + std::to_string(t->cols()) +
                                                       ", " + std::to_string(t->rows()) + "], expected [" +
                                                       std::to_string(cols) + ", " + std::to_string(rows) + "]");
        }
        out = t;
        return Status::ok();
    };
    auto vec = [&](const std::string& name, int64_t n, bool required, std::vector<float>& out) -> Status {
        const TensorView* t = f.tensor(name);
        if (t == nullptr) {
            return required ? Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'") : Status::ok();
        }
        if (t->elements() != n) return Status(ErrorCode::InvalidModel, "tensor '" + name + "' has the wrong size");
        out.resize(static_cast<size_t>(n));
        quant::dequantize_row(t->type, t->data, out.data(), n);
        return Status::ok();
    };

    token_embd_ = f.tensor("token_embd.weight");  // shape checked by describe_model()
    if (f.tensor("output.weight") != nullptr) {
        LIYAB_RETURN_IF_ERROR(matrix("output.weight", c.n_embd, c.n_vocab, output_));
    } else {
        output_ = token_embd_;  // tied embeddings
    }
    LIYAB_RETURN_IF_ERROR(vec("output_norm.weight", c.n_embd, true, output_norm_));

    const int64_t dn_key = int64_t{c.ssm_k_heads} * c.ssm_head_dim;
    const int64_t dn_value = int64_t{c.ssm_v_heads} * c.ssm_head_dim;
    const int64_t conv_dim = 2 * dn_key + dn_value;

    layers_.resize(static_cast<size_t>(c.n_layers));
    int32_t kv_slots = 0;
    int32_t state_slots = 0;
    for (int32_t l = 0; l < c.n_layers; ++l) {
        Layer& L = layers_[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_norm.weight"), c.n_embd, true, L.attn_norm));
        if (plan.mixer == MixerKind::Attention && plan.latent) {
            L.kv_slot = kv_slots++;
            const LatentAttention& m = c.mla;
            const int64_t q_rows = int64_t{c.n_head} * (m.nope_dim + c.rope_dim);  // per head [q_nope | q_pe]
            if (m.q_rank > 0) {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q_a.weight"), c.n_embd, m.q_rank, L.w[kQa]));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_q_a_norm.weight"), m.q_rank, true, L.q_a_norm));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q_b.weight"), m.q_rank, q_rows, L.w[kQb]));
            } else {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q.weight"), c.n_embd, q_rows, L.w[kQ]));
            }
            LIYAB_RETURN_IF_ERROR(
                matrix(block_tensor(l, "attn_kv_a_mqa.weight"), c.n_embd, m.kv_rank + c.rope_dim, L.w[kKvA]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_kv_a_norm.weight"), m.kv_rank, true, L.kv_a_norm));
            // Per head: attn_k_b maps a head's q_nope onto the latent (the
            // key's projection, transposed by the converter), attn_v_b the
            // weighted latent onto the head's value.
            auto per_head = [&](const char* suffix, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
                const std::string name = block_tensor(l, suffix);
                const TensorView* t = f.tensor(name);
                if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
                if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != c.n_head || t->ne[3] != 1) {
                    return Status(ErrorCode::InvalidModel, "tensor '" + name + "' does not have the per-head shape");
                }
                out = t;
                return Status::ok();
            };
            LIYAB_RETURN_IF_ERROR(per_head("attn_k_b.weight", m.nope_dim, m.kv_rank, L.w[kKb]));
            LIYAB_RETURN_IF_ERROR(per_head("attn_v_b.weight", m.kv_rank, m.v_dim, L.w[kVb]));
            LIYAB_RETURN_IF_ERROR(
                matrix(block_tensor(l, "attn_output.weight"), int64_t{c.n_head} * m.v_dim, c.n_embd, L.w[kO]));
        } else if (plan.mixer == MixerKind::Attention) {
            L.kv_slot = kv_slots++;
            if (plan.fused_qkv) {
                LIYAB_RETURN_IF_ERROR(
                    matrix(block_tensor(l, "attn_qkv.weight"), c.n_embd, q_dim + 2 * kv_dim, L.w[kQkv]));
                // A fused bias, if any, splits like the rows.
                std::vector<float> bias;
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_qkv.bias"), q_dim + 2 * kv_dim, false, bias));
                if (!bias.empty()) {
                    L.bq.assign(bias.begin(), bias.begin() + q_dim);
                    L.bk.assign(bias.begin() + q_dim, bias.begin() + q_dim + kv_dim);
                    L.bv.assign(bias.begin() + q_dim + kv_dim, bias.end());
                }
            } else {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_q.weight"), c.n_embd,
                                             plan.attn_gate ? 2 * q_dim : q_dim, L.w[kQ]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_k.weight"), c.n_embd, kv_dim, L.w[kK]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_v.weight"), c.n_embd, kv_dim, L.w[kV]));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_q.bias"), q_dim, false, L.bq));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_k.bias"), kv_dim, false, L.bk));
                LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_v.bias"), kv_dim, false, L.bv));
            }
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_output.weight"), q_dim, c.n_embd, L.w[kO]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_q_norm.weight"), c.head_dim, false, L.q_norm));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "attn_k_norm.weight"), c.head_dim, false, L.k_norm));
        } else {
            L.state_slot = state_slots++;
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_qkv.weight"), c.n_embd, conv_dim, L.w[kQkv]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "attn_gate.weight"), c.n_embd, dn_value, L.w[kZ]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_alpha.weight"), c.n_embd, c.ssm_v_heads, L.w[kAlpha]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_beta.weight"), c.n_embd, c.ssm_v_heads, L.w[kBeta]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ssm_out.weight"), dn_value, c.n_embd, L.w[kSsmOut]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_conv1d.weight"), conv_dim * c.ssm_conv_kernel, true, L.conv1d));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_a"), c.ssm_v_heads, true, L.ssm_a));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_dt.bias"), c.ssm_v_heads, true, L.ssm_dt));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ssm_norm.weight"), c.ssm_head_dim, true, L.ssm_norm));
        }
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, std::string(c.spec->ffn_norm) + ".weight"), c.n_embd, true, L.ffn_norm));
        if (!c.spec->post_mixer_norm.empty()) {
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, std::string(c.spec->post_mixer_norm) + ".weight"), c.n_embd,
                                      false, L.post_mixer_norm));
        }
        if (!c.spec->post_ffn_norm.empty()) {
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, std::string(c.spec->post_ffn_norm) + ".weight"), c.n_embd,
                                      false, L.post_ffn_norm));
        }
        if (plan.ffn == FfnKind::Dense) {
            if (plan.fused_gate_up) {  // ffn_up is [gate | up]; no ffn_gate
                LIYAB_RETURN_IF_ERROR(
                    matrix(block_tensor(l, "ffn_up.weight"), c.n_embd, 2 * int64_t{c.n_ff}, L.w[kUp]));
            } else {
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_gate.weight"), c.n_embd, c.n_ff, L.w[kGate]));
                LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_up.weight"), c.n_embd, c.n_ff, L.w[kUp]));
            }
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_down.weight"), c.n_ff, c.n_embd, L.w[kDown]));
            continue;
        }
        LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_gate_inp.weight"), c.n_embd, c.n_expert, L.w[kRouter]));
        auto experts = [&](const char* suffix, int64_t cols, int64_t rows, const TensorView*& out) -> Status {
            const std::string name = block_tensor(l, suffix);
            const TensorView* t = f.tensor(name);
            if (t == nullptr) return Status(ErrorCode::InvalidModel, "missing tensor '" + name + "'");
            if (t->ne[0] != cols || t->ne[1] != rows || t->ne[2] != c.n_expert || t->ne[3] != 1) {
                return Status(ErrorCode::InvalidModel, "tensor '" + name + "' does not have the expert shape");
            }
            out = t;
            return Status::ok();
        };
        if (plan.fused_gate_up) {  // each expert's [gate | up] in one tensor; no gate of its own
            LIYAB_RETURN_IF_ERROR(
                experts("ffn_gate_up_exps.weight", c.n_embd, 2 * int64_t{c.n_ff_expert}, L.w[kUpExps]));
        } else {
            LIYAB_RETURN_IF_ERROR(experts("ffn_gate_exps.weight", c.n_embd, c.n_ff_expert, L.w[kGateExps]));
            LIYAB_RETURN_IF_ERROR(experts("ffn_up_exps.weight", c.n_embd, c.n_ff_expert, L.w[kUpExps]));
        }
        LIYAB_RETURN_IF_ERROR(experts("ffn_down_exps.weight", c.n_ff_expert, c.n_embd, L.w[kDownExps]));
        LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "exp_probs_b.bias"), c.n_expert, false, L.expert_bias));
        if (plan.shared_expert) {
            const int64_t sh = c.n_ff_shared > 0 ? c.n_ff_shared : f.tensor(block_tensor(l, "ffn_up_shexp.weight"))->rows();
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_gate_shexp.weight"), c.n_embd, sh, L.w[kGateShared]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_up_shexp.weight"), c.n_embd, sh, L.w[kUpShared]));
            LIYAB_RETURN_IF_ERROR(matrix(block_tensor(l, "ffn_down_shexp.weight"), sh, c.n_embd, L.w[kDownShared]));
            LIYAB_RETURN_IF_ERROR(vec(block_tensor(l, "ffn_gate_inp_shexp.weight"), c.n_embd, false, L.shared_gate));
        }
    }

    states_.resize(static_cast<size_t>(state_slots));
    for (RecurrentState& st : states_) {
        st.conv.assign(static_cast<size_t>(conv_dim) * static_cast<size_t>(c.ssm_conv_kernel - 1), 0.0f);
        st.ssm.assign(static_cast<size_t>(c.ssm_v_heads) * static_cast<size_t>(c.ssm_head_dim) *
                          static_cast<size_t>(c.ssm_head_dim),
                      0.0f);
    }
    return Status::ok();
}

size_t Transformer::recurrent_state_bytes() const noexcept {
    size_t bytes = 0;
    for (const RecurrentState& st : states_) bytes += (st.conv.size() + st.ssm.size()) * sizeof(float);
    return bytes;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
class Transformer::BlockWeights {
public:
    BlockWeights(Transformer& model, int32_t layer) : model_(model) {
        const Layer& L = model.layers_[static_cast<size_t>(layer)];
        for (size_t i = 0; i < kWeightRoles; ++i) {
            if (L.w[i] != nullptr) views_[i] = *L.w[i];
        }
        item_ = model.layer_source_ == nullptr ? -1
                : model.layer_items_.empty()  ? layer
                                              : model.layer_items_[static_cast<size_t>(layer)];
        if (item_ < 0) {
            model.file_->begin_layer(layer);  // mmap path: advance the prefetch window (if any)
            return;
        }
        auto base = model.layer_source_->acquire(item_);
        if (!base) {
            status_ = base.status();
            return;
        }
        acquired_ = true;
        // Every block tensor lies inside the block's byte range, so its slot
        // address is the slot base plus its offset within that range.
        const size_t range_begin = model.file_->layer_range(layer).first;
        for (size_t i = 0; i < kWeightRoles; ++i) {
            if (L.w[i] != nullptr) views_[i].data = base.value() + (views_[i].file_offset - range_begin);
        }
    }
    ~BlockWeights() {
        if (acquired_) model_.layer_source_->release(item_);
    }
    BlockWeights(const BlockWeights&) = delete;
    BlockWeights& operator=(const BlockWeights&) = delete;

    [[nodiscard]] const Status& status() const noexcept { return status_; }
    // True when the views point into a reused streaming slot rather than the
    // mapping. Backends that keep per-tensor device copies (keyed by address)
    // must not see such views: the next block reuses the same address.
    [[nodiscard]] bool streamed() const noexcept { return item_ >= 0 && !model_.layer_shared_; }
    // The block's matrix with role `role` (only roles the block has).
    const TensorView& operator[](WeightRole role) const { return views_[role]; }

private:
    Transformer& model_;
    int32_t item_ = -1;  // layer source item, -1: read in place
    bool acquired_ = false;
    Status status_;
    TensorView views_[kWeightRoles];
};

void Transformer::reset() noexcept {
    kv_->clear();
    for (RecurrentState& st : states_) {
        std::fill(st.conv.begin(), st.conv.end(), 0.0f);
        std::fill(st.ssm.begin(), st.ssm.end(), 0.0f);
    }
    std::fill(checkpoint_pos_.begin(), checkpoint_pos_.end(), -1);
    snapshots_.clear();
    spare_buffers_.clear();
    n_past_ = 0;
    tree_base_ = -1;
    tree_parent_.clear();
}

bool Transformer::has_position_state() const noexcept { return !states_.empty() || kv_->has_windowed_layers(); }

void Transformer::pin_snapshots() {
    if (!kv_->has_windowed_layers()) return;
    std::vector<int32_t> positions;
    positions.reserve(snapshots_.size());
    for (const StateSnapshot& snap : snapshots_) positions.push_back(snap.pos);
    kv_->pin(positions);
}

void Transformer::snapshot_state(bool pin) {
    if (!has_position_state() || n_past_ == 0) return;
    if (!snapshots_.empty() && snapshots_.back().pos == n_past_) {  // already held
        if (pin) {
            for (StateSnapshot& s : snapshots_) s.pinned = false;
            snapshots_.back().pinned = true;
        }
        return;
    }
    if (pin) {
        for (StateSnapshot& s : snapshots_) s.pinned = false;
    }
    StateSnapshot snap;
    if (static_cast<int32_t>(snapshots_.size()) >= kStateSnapshots) {
        // Keep the oldest and the pinned one; recycle the buffer of the
        // oldest other, so the newest ones stay (a draft's words and the
        // whole draft after them).
        size_t victim = 1;
        while (victim + 1 < snapshots_.size() && snapshots_[victim].pinned) ++victim;
        snap = std::move(snapshots_[victim]);
        snapshots_.erase(snapshots_.begin() + static_cast<std::ptrdiff_t>(victim));
    } else if (!spare_buffers_.empty()) {
        snap.data = std::move(spare_buffers_.back());
        spare_buffers_.pop_back();
    }
    snap.pinned = pin;
    snap.pos = n_past_;
    // The recurrent states are copied; the windowed blocks' window before
    // this position stays in the cache's pages, which the pin keeps (no copy:
    // it is usually the live window).
    snap.data.resize(recurrent_state_bytes() / sizeof(float));
    float* out = snap.data.data();
    for (const RecurrentState& st : states_) {
        out = std::copy(st.conv.begin(), st.conv.end(), out);
        out = std::copy(st.ssm.begin(), st.ssm.end(), out);
    }
    snapshots_.push_back(std::move(snap));
    pin_snapshots();
}

namespace {

constexpr uint32_t kStateMagic = 0x5453594C;  // "LYST"
// 2: the recurrent-state snapshots follow the states (1 had none).
// 3: the key scales the pages were stored with follow the fingerprint.
// 4: per-block KV layout: the cache's layout fingerprint follows the
//    fingerprint, pages are rows images (the blocks keeping every position),
//    and the windowed blocks' pages (KvCache::window_pages_of() the end and
//    the snapshots kept: a count, then each page's index and window page
//    image) end the file. Older files hold every block's rows of every
//    position in one table's pages: they load converted
//    (KvCache::write_legacy_page()).
constexpr uint32_t kStateVersion = 4;

template <typename T>
void put(std::ostream& out, const T& v) {
    out.write(reinterpret_cast<const char*>(&v), sizeof v);
}

template <typename T>
bool get(std::istream& in, T& v) {
    return static_cast<bool>(in.read(reinterpret_cast<char*>(&v), sizeof v));
}

}  // namespace

// What must match for a saved state to be valid here: the model (file size
// and shape) and how the KV cache stores rows. A LongRoPE model whose context
// picked the long factors rotates its keys otherwise than with the short
// ones, so that choice marks the first field (every other model's
// fingerprint, and so its saved states, stay as they were).
std::array<int64_t, 9> Transformer::state_fingerprint() const noexcept {
    const KvCacheConfig& kv = kv_->config();
    const bool long_rope = config_.rope_ctx_orig > 0 && context_length_ > config_.rope_ctx_orig;
    const int64_t file = static_cast<int64_t>(file_->file_size()) ^ (long_rope ? int64_t{1} << 62 : 0);
    return {file, config_.n_layers, config_.n_embd, config_.n_vocab, kv.n_layers, kv.n_head_kv, kv.head_dim,
            static_cast<int64_t>(kv_->dtype()), kv.page_tokens};
}

uint64_t Transformer::context_fingerprint() const noexcept {
    const std::array<int64_t, 9> fp = state_fingerprint();
    const uint64_t page = kv_->page_bytes();
    uint64_t seed = xxhash64(fp.data(), sizeof fp, xxhash64(&page, sizeof page));
    // Windowed blocks change what chunks hold and add their own images: the
    // layout is part of the key (0, and so no change, without them).
    if (const uint64_t layout = kv_->layout_fingerprint(); layout != 0) seed = xxhash64(&layout, sizeof layout, seed);
    return xxhash64(key_scale_.data(), key_scale_.size() * sizeof(float), seed);
}

std::vector<Transformer::KeptState> Transformer::kept_states() const {
    std::vector<KeptState> kept;
    if (!has_position_state() || n_past_ == 0) return kept;
    bool pinned_below = false;
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos < n_past_ && (&snap == &snapshots_.front() || snap.pinned)) {
            kept.push_back({snap.pos, snap.pinned});
            pinned_below = pinned_below || snap.pinned;
        }
    }
    kept.push_back({n_past_, !pinned_below});  // as read_state() pins the end
    return kept;
}

std::vector<std::span<const float>> Transformer::kept_state_parts(int32_t pos) const {
    std::vector<std::span<const float>> parts;
    if (states_.empty()) return parts;
    if (pos == n_past_) {
        for (const RecurrentState& st : states_) {
            parts.emplace_back(st.conv);
            parts.emplace_back(st.ssm);
        }
        return parts;
    }
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos == pos) parts.emplace_back(snap.data);
    }
    return parts;
}

std::vector<int32_t> Transformer::snapshot_positions() const {
    std::vector<int32_t> out;
    for (const StateSnapshot& snap : snapshots_) out.push_back(snap.pos);
    return out;
}

void Transformer::begin_restore() {
    for (StateSnapshot& snap : snapshots_) spare_buffers_.push_back(std::move(snap.data));
    snapshots_.clear();
}

std::vector<std::span<float>> Transformer::recurrent_parts() {
    std::vector<std::span<float>> parts;
    for (RecurrentState& st : states_) {
        parts.emplace_back(st.conv);
        parts.emplace_back(st.ssm);
    }
    return parts;
}

Transformer::StateBuffer Transformer::take_snapshot_buffer() {
    StateBuffer buffer;
    if (!spare_buffers_.empty()) {
        buffer = std::move(spare_buffers_.back());
        spare_buffers_.pop_back();
    }
    buffer.resize(recurrent_state_bytes() / sizeof(float));  // not zero-filled (StateBuffer)
    return buffer;
}

Status Transformer::write_state(std::ostream& out) const {
    if (kv_->window() > 0) return Status(ErrorCode::Unsupported, "a sliding-window context cannot be saved");
    // The snapshots below the end (one at the end is the states above), so a
    // restored context rewinds where the live one would: the next prompt
    // re-sends the last reply as text, which may encode differently, and
    // without the last prompt's snapshot a hybrid model would keep nothing.
    // The oldest (the system prompt's end) and the pinned one are enough; a
    // draft's is not worth its size (62.8 MiB each on a 35B).
    std::vector<SnapshotImage> kept;
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos < n_past_ && (&snap == &snapshots_.front() || snap.pinned)) {
            kept.push_back({snap.pos, snap.pinned, snap.data});
        }
    }
    // A page spread over several pool pages is copied out; otherwise written in place.
    std::vector<uint8_t> scratch(kv_->page_bytes());
    auto page = [this, &scratch](int32_t p) -> const uint8_t* {
        if (const uint8_t* data = kv_->page_data(p)) return data;
        if (!kv_->read_rows(p, kv_->config().page_tokens, scratch.data()).is_ok()) return nullptr;
        return scratch.data();
    };
    return write_image(out, n_past_, page, nullptr, {}, kept, true);
}

Status Transformer::write_state(std::ostream& out, const ContextImage& image) const {
    if (kv_->window() > 0) return Status(ErrorCode::Unsupported, "a sliding-window context cannot be saved");
    const size_t floats = recurrent_state_bytes() / sizeof(float);
    const bool windowed = kv_->has_windowed_layers();
    if (image.n < 0 || image.n > context_length_ || !image.page || image.state.size() != (states_.empty() ? 0 : floats) ||
        static_cast<int32_t>(image.snapshots.size()) >= kStateSnapshots ||
        (states_.empty() && !image.snapshots.empty()) || (windowed && (!image.window_page || !image.snapshots.empty()))) {
        return Status(ErrorCode::InvalidArgument, "the context image does not fit this model");
    }
    int32_t below = 0;
    for (const SnapshotImage& snap : image.snapshots) {
        if (snap.pos <= below || snap.pos >= image.n || snap.data.size() != floats) {
            return Status(ErrorCode::InvalidArgument, "the context image's snapshots do not fit this model");
        }
        below = snap.pos;
    }
    return write_image(out, image.n, image.page, windowed ? &image.window_page : nullptr, image.state, image.snapshots,
                       false);
}

// The body of both write_state()s: `state` holds the recurrent states at n as
// one image, or `live` takes them from states_ (no copy of the live ones).
// The windowed blocks' pages come from the live cache, or from `window_page`
// (the windowed rows of a logical page, KvCache::read_window_rows()'s image).
Status Transformer::write_image(std::ostream& out, int32_t n, const std::function<const uint8_t*(int32_t)>& page,
                                const std::function<const uint8_t*(int32_t)>* window_page,
                                std::span<const float> state, std::span<const SnapshotImage> snapshots,
                                bool live) const {
    put(out, kStateMagic);
    put(out, kStateVersion);
    for (const int64_t v : state_fingerprint()) put(out, v);
    put(out, kv_->layout_fingerprint());
    put(out, static_cast<uint64_t>(key_scale_.size()));
    out.write(reinterpret_cast<const char*>(key_scale_.data()),
              static_cast<std::streamsize>(key_scale_.size() * sizeof(float)));
    put(out, n);
    const int32_t pages = (n + kv_->config().page_tokens - 1) / kv_->config().page_tokens;
    put(out, static_cast<uint64_t>(kv_->page_bytes()));
    put(out, pages);
    for (int32_t p = 0; p < pages && kv_->page_bytes() > 0; ++p) {
        const uint8_t* data = page(p);
        if (data == nullptr) return Status(ErrorCode::Internal, "a cached page is not mapped");
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(kv_->page_bytes()));
    }
    put(out, static_cast<int32_t>(states_.size()));
    const float* from = state.data();
    for (const RecurrentState& st : states_) {
        put(out, static_cast<uint64_t>(st.conv.size()));
        put(out, static_cast<uint64_t>(st.ssm.size()));
        const float* conv = live ? st.conv.data() : from;
        const float* ssm = live ? st.ssm.data() : from + st.conv.size();
        out.write(reinterpret_cast<const char*>(conv), static_cast<std::streamsize>(st.conv.size() * sizeof(float)));
        out.write(reinterpret_cast<const char*>(ssm), static_cast<std::streamsize>(st.ssm.size() * sizeof(float)));
        if (!live) from += st.conv.size() + st.ssm.size();
    }
    put(out, static_cast<int32_t>(snapshots.size()));
    for (const SnapshotImage& snap : snapshots) {
        put(out, snap.pos);
        put(out, static_cast<uint8_t>(snap.pinned));
        put(out, static_cast<uint64_t>(snap.data.size()));
        out.write(reinterpret_cast<const char*>(snap.data.data()),
                  static_cast<std::streamsize>(snap.data.size() * sizeof(float)));
    }
    // Windowed blocks: every page of the window before the end and before
    // each snapshot kept, whole (out of the live pages, or the image's window
    // pages), so a loaded context returns where the saved one would.
    std::vector<int32_t> pins;
    for (const SnapshotImage& snap : snapshots) pins.push_back(snap.pos);
    const std::vector<int32_t> window = kv_->window_pages_of(n, pins);
    put(out, static_cast<int32_t>(window.size()));
    std::vector<uint8_t> scratch(live ? kv_->window_page_bytes() : 0);
    const int32_t t = kv_->config().page_tokens;
    for (const int32_t q : window) {
        const uint8_t* image = nullptr;
        if (live) {
            LIYAB_RETURN_IF_ERROR(kv_->read_window_rows(q, std::min(t, n - q * t), scratch.data()));
            image = scratch.data();
        } else if (window_page == nullptr || (image = (*window_page)(q)) == nullptr) {
            return Status(ErrorCode::InvalidArgument, "a window page of the context image is missing");
        }
        put(out, q);
        out.write(reinterpret_cast<const char*>(image), static_cast<std::streamsize>(kv_->window_page_bytes()));
    }
    return out ? Status::ok() : Status(ErrorCode::IoError, "cannot write the context state");
}

Status Transformer::read_state(std::istream& in) {
    reset();
    auto fail = [this](ErrorCode code, const char* why) {
        reset();
        return Status(code, why);
    };
    uint32_t magic = 0, version = 0;
    if (!get(in, magic) || !get(in, version) || magic != kStateMagic || version < 1 || version > kStateVersion) {
        return fail(ErrorCode::InvalidArgument, "not a Liyab context state (or another version)");
    }
    for (const int64_t expected : state_fingerprint()) {
        int64_t v = 0;
        if (!get(in, v) || v != expected) {
            return fail(ErrorCode::InvalidArgument, "the state was saved for another model or other cache settings");
        }
    }
    // Before version 4 every block kept every position in one table's pages:
    // such a file converts to any layout. A version 4 file holds what its
    // own layout keeps, so it must be this one.
    const bool legacy = version < 4;
    if (!legacy) {
        uint64_t layout = 0;
        if (!get(in, layout) || layout != kv_->layout_fingerprint()) {
            return fail(ErrorCode::InvalidArgument,
                        "the state was saved with other windowed attention blocks (another context length or "
                        "segment reuse setting)");
        }
    }
    // Key scales the pages were stored with (none before version 3).
    std::vector<float> stored;
    if (version >= 3) {
        uint64_t count = 0;
        if (!get(in, count) || (count != 0 && count != key_scaling_shape().factors())) {
            return fail(ErrorCode::InvalidArgument, "the state's key scales do not match");
        }
        stored.resize(static_cast<size_t>(count));
        if (!in.read(reinterpret_cast<char*>(stored.data()), static_cast<std::streamsize>(count * sizeof(float)))) {
            return fail(ErrorCode::IoError, "the state's key scales are truncated");
        }
    }
    int32_t n = 0, pages = 0;
    uint64_t page_bytes = 0;
    const int32_t t = kv_->config().page_tokens;
    if (!get(in, n) || !get(in, page_bytes) || !get(in, pages) || n < 0 || n > context_length_ ||
        page_bytes != (legacy ? kv_->legacy_page_bytes() : kv_->page_bytes()) || pages != (n + t - 1) / t) {
        return fail(ErrorCode::InvalidArgument, "the state's cache layout does not match");
    }
    // Every page of the blocks keeping every position, the windowed blocks'
    // pages of the window before n (a snapshot's are mapped when it is read).
    if (Status s = kv_->reserve_context(n, {}); !s.is_ok()) return fail(s.code(), "cannot map the state's cache pages");
    const bool convert = legacy && kv_->has_windowed_layers();
    std::vector<uint8_t> buffer;
    for (int32_t p = 0; p < pages && page_bytes > 0; ++p) {
        uint8_t* data = convert ? nullptr : kv_->mutable_page_data(p);
        if (data == nullptr) {
            buffer.resize(static_cast<size_t>(page_bytes));
            data = buffer.data();
        }
        if (!in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(page_bytes))) {
            return fail(ErrorCode::IoError, "the state's cache pages are truncated");
        }
        if (data != buffer.data()) continue;  // read in place
        const Status written = convert ? kv_->write_legacy_page(p, data) : kv_->write_rows(p, data, t, 0, t);
        if (!written.is_ok()) return fail(ErrorCode::InvalidArgument, "the state's cache pages do not fit");
    }
    int32_t count = 0;
    if (!get(in, count) || count != static_cast<int32_t>(states_.size())) {
        return fail(ErrorCode::InvalidArgument, "the state's recurrent blocks do not match");
    }
    for (RecurrentState& st : states_) {
        uint64_t conv = 0, ssm = 0;
        if (!get(in, conv) || !get(in, ssm) || conv != st.conv.size() || ssm != st.ssm.size() ||
            !in.read(reinterpret_cast<char*>(st.conv.data()), static_cast<std::streamsize>(conv * sizeof(float))) ||
            !in.read(reinterpret_cast<char*>(st.ssm.data()), static_cast<std::streamsize>(ssm * sizeof(float)))) {
            return fail(ErrorCode::IoError, "the state's recurrent states are truncated");
        }
    }
    int32_t snapshots = 0;
    if (version >= 2 && (!get(in, snapshots) || snapshots < 0 || snapshots >= kStateSnapshots)) {
        return fail(ErrorCode::InvalidArgument, "the state's snapshots do not match");
    }
    const size_t floats = recurrent_state_bytes() / sizeof(float);
    int32_t dropped = 0;
    for (int32_t i = 0; i < snapshots; ++i) {
        StateSnapshot snap;
        uint8_t pinned = 0;
        uint64_t size = 0;
        if (!get(in, snap.pos) || !get(in, pinned) || !get(in, size) || size != floats || snap.pos <= 0 ||
            snap.pos >= n || (!snapshots_.empty() && snap.pos <= snapshots_.back().pos)) {
            return fail(ErrorCode::InvalidArgument, "the state's snapshots do not match");
        }
        snap.pinned = pinned != 0;
        snap.data.resize(floats);
        if (!in.read(reinterpret_cast<char*>(snap.data.data()), static_cast<std::streamsize>(size * sizeof(float)))) {
            return fail(ErrorCode::IoError, "the state's snapshots are truncated");
        }
        if (convert) {
            // An older file kept no window of its own for a snapshot below
            // the end: its rows were in pages released here. The context
            // loads without it (it rewinds less far).
            ++dropped;
            continue;
        }
        snapshots_.push_back(std::move(snap));
    }
    // Windowed blocks (version 4): the pages of the window before n and
    // before each snapshot, whole, exactly those (older files had those rows
    // in their pages).
    if (!legacy) {
        std::vector<int32_t> pins;
        for (const StateSnapshot& snap : snapshots_) pins.push_back(snap.pos);
        if (Status s = kv_->reserve_context(n, pins); !s.is_ok()) return fail(s.code(), "cannot map the state's pages");
        const std::vector<int32_t> expected = kv_->window_pages_of(n, pins);
        int32_t count_pages = 0;
        if (!get(in, count_pages) || count_pages != static_cast<int32_t>(expected.size())) {
            return fail(ErrorCode::InvalidArgument, "the state's windowed blocks do not match");
        }
        buffer.resize(kv_->window_page_bytes());
        for (const int32_t q : expected) {
            int32_t at = 0;
            if (!get(in, at) || at != q) return fail(ErrorCode::InvalidArgument, "the state's windowed blocks do not match");
            if (!in.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()))) {
                return fail(ErrorCode::IoError, "the state's window pages are truncated");
            }
            if (!kv_->write_window_rows(q, buffer.data(), 0, std::min(t, n - q * t)).is_ok()) {
                return fail(ErrorCode::InvalidArgument, "the state's window pages do not fit");
            }
        }
    }
    // Keys stored with other scales (an older file: none, i.e. 1) are brought
    // to the ones in use: k_now = k_stored * t_stored / t_now, requantized. The
    // values and the rest of the context are unchanged.
    if (stored != key_scale_) {
        std::vector<float> ratio(key_scaling_shape().factors());
        for (size_t i = 0; i < ratio.size(); ++i) {
            ratio[i] = (stored.empty() ? 1.0f : stored[i]) / (key_scale_.empty() ? 1.0f : key_scale_[i]);
        }
        if (Status s = kv_->scale_keys(n, ratio); !s.is_ok()) return fail(s.code(), "cannot convert the state's keys");
        LIYAB_LOG_INFO("state: keys of %d positions converted to this model's key scales", n);
    }
    if (convert) {
        LIYAB_LOG_INFO("state: a version %u file converted to per-block KV (windowed blocks keep their window)%s",
                       version, dropped > 0 ? "; its snapshots below the end were dropped" : "");
    }
    n_past_ = n;
    // One at the end too, so a draft typed ahead can be dropped back to it. It
    // is where the next prompt continues: pinned, unless the last prompt's end
    // came back pinned (a state saved after a reply).
    snapshot_state(/*pin=*/std::none_of(snapshots_.begin(), snapshots_.end(),
                                         [](const StateSnapshot& snap) { return snap.pinned; }));
    return Status::ok();
}

int32_t Transformer::restorable_prefix(int32_t n) const noexcept {
    n = std::clamp(n, 0, n_past_);
    if (n == n_past_) return n;
    // Windowed blocks: only where their window's rows are still kept (the
    // rollback margin, the snapshots' pins).
    const bool windowed = kv_->has_windowed_layers();
    if (!config_.hybrid()) return windowed ? kv_->restorable(n) : n;
    auto kept = [&](int32_t m) { return !windowed || kv_->restorable(m) == m; };
    int32_t best = 0;
    for (const int32_t p : checkpoint_pos_) {  // a checkpoint at position p restores n = p + 1
        if (p >= 0 && p + 1 <= n && p + 1 > best && kept(p + 1)) best = p + 1;
    }
    for (const StateSnapshot& snap : snapshots_) {
        if (snap.pos <= n && snap.pos > best && kept(snap.pos)) best = snap.pos;
    }
    return best;
}

void Transformer::set_rollback_window(int32_t positions) {
    rollback_window_ = states_.empty() ? 0 : std::max(0, positions);
    checkpoint_pos_.assign(static_cast<size_t>(rollback_window_), -1);
    for (RecurrentState& st : states_) {
        st.checkpoints.assign(static_cast<size_t>(rollback_window_) * (st.conv.size() + st.ssm.size()), 0.0f);
        st.checkpoints.shrink_to_fit();
    }
}
void Transformer::rope(float* vec, int32_t n_heads, int32_t pos, const RopeTable& table) const {
    rope(vec, n_heads, static_cast<size_t>(config_.head_dim), pos, table);
}

void Transformer::rope(float* vec, int32_t n_heads, size_t stride, int32_t pos, const RopeTable& table) const {
    const int32_t half = config_.rope_dim / 2;
    const bool halves = config_.spec->rope == RopeLayout::Halves;
    const float p = static_cast<float>(pos) / table.scale;
    for (int32_t h = 0; h < n_heads; ++h) {
        float* x = vec + static_cast<size_t>(h) * stride;
        for (int32_t i = 0; i < half; ++i) {
            const float theta = p * table.inv_freq[static_cast<size_t>(i)];
            const float cs = std::cos(theta) * table.magnitude;  // x 1 keeps the bits
            const float sn = std::sin(theta) * table.magnitude;
            const int32_t a = halves ? i : 2 * i;
            const int32_t b = halves ? i + half : 2 * i + 1;
            const float x0 = x[a];
            const float x1 = x[b];
            x[a] = x0 * cs - x1 * sn;
            x[b] = x0 * sn + x1 * cs;
        }
    }
}

// Attention over the KV cache, for n query tokens. The query heads sharing a
// KV head (grouped query attention: 8 per KV head on Qwen3.6) are scored
// together, so each K and V row is fetched and decoded once for the whole
// group, not once per query head. A token's visible positions are cut into
// equal spans (kAttentionMinSpan, kAttentionMaxSpans); each span keeps a
// partial softmax (its maximum, sum and weighted V) and the spans are merged
// in order (log-sum-exp). The cut follows the token's own position only,
// never the batch, the threads or the load: a token's attention rounds the
// same however its prompt was split (typing ahead, chunked prefill,
// speculative verification) and however many cores the power manager left
// active. Before, the spans were sized for the active threads and the batch,
// and the same prompt could give two different greedy replies on a cool and a
// warm phone. Spans of a fixed 256 positions were deterministic too, but left
// a 1000-position decoding step 8 tasks for 8 threads: 33% slower there; the
// span bounds (transformer.h) keep the tasks even among the threads.
//
// How the spans are scheduled changes no number. Decoding, where one token
// has only n_head_kv groups, scores every span as a task of its own so every
// thread works (as in flash decoding), then merges. A batch with enough
// (token, KV head) pairs scores each pair's spans on one thread and merges
// them right away, which needs no partial results for the whole batch (a
// 1024-token pass at a 20k context would need over a GB of them). On a
// 2838-token context the per-head loop this replaced took 48 ms of a 161 ms
// decode step.
//
// A windowed block (`window` > 0) sees, of what the cache keeps visible, only
// the positions [pos - window + 1, pos]: the sinks drop out once the window
// has passed them. Its spans follow its own visible count, so they too depend
// only on the position. Its rows live in the cache's windowed group, which
// keeps only those positions (KvCache's per-block layout), so a windowed block
// saves memory as well as compute (with TransformerOptions::keep_windowed_rows,
// compute only).
void Transformer::attention(int32_t kv_slot, int32_t window, int32_t n, ThreadPool& pool, const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    // Each segment's rows of the block, their group and offsets resolved
    // once for the pass (a pass of several sequences reads several caches).
    row_views_.clear();
    bool all_rows = true;  // every segment attends here (not latent attention's expanded ones)
    for (const Segment& s : segs_) {
        row_views_.push_back(s.kv->row_view(kv_slot));
        all_rows = all_rows && !s.expanded;
    }
    // The rows attending: every row of the batch, or the absorbed segments'
    // rows of a latent pass that also has expanded ones (attn_rows_). Tasks
    // and partial results are indexed by a row's place among them.
    int32_t m = n;
    if (!all_rows) {
        attn_rows_.clear();
        for (const Segment& s : segs_) {
            if (s.expanded) continue;
            for (int32_t t = s.first; t < s.first + s.n; ++t) attn_rows_.push_back(t);
        }
        m = static_cast<int32_t>(attn_rows_.size());
        if (m == 0) return;
    }
    const int32_t* row_list = all_rows ? nullptr : attn_rows_.data();
    auto row_at = [row_list](int64_t i) { return row_list == nullptr ? static_cast<int32_t>(i) : row_list[i]; };
    const int32_t hd = c.head_dim;
    const int32_t group = c.n_head / c.n_head_kv;
    // Values may be narrower than keys (latent attention: the latent part of
    // each key row); every value-sized buffer below follows value_dim.
    const int32_t vd = c.value_dim;
    const auto uvd = static_cast<size_t>(vd);
    const size_t out_dim = static_cast<size_t>(c.n_head) * uvd;  // att_ row: every head's output
    const float scale = c.attn_scale;  // 1/sqrt(head_dim) unless the file states another (ModelConfig)
    const float softcap = c.attn_softcap;
    const DType kv_type = segs_.front().kv->dtype();  // every context's cache has the same format
    const bool quantized_kv = quant::is_block_quantized(kv_type);
    const auto uhd = static_cast<size_t>(hd);
    const auto ugroup = static_cast<size_t>(group);
    const auto n_head = static_cast<size_t>(c.n_head);
    const size_t q8_per_head = uhd / quant::kBlock;

    // Quantized caches are scored with integer dot products against a Q8_0
    // copy of each query (same kernels as the weight matmuls), made once per
    // (token, head) rather than in every span.
    if (quantized_kv) {
        const auto rows = static_cast<int64_t>(m) * c.n_head;
        q8_queries_.resize(static_cast<size_t>(n) * n_head * q8_per_head);
        auto quantize = [&](int64_t begin, int64_t end) {
            for (int64_t r = begin; r < end; ++r) {
                const size_t row =
                    static_cast<size_t>(row_at(r / c.n_head)) * n_head + static_cast<size_t>(r % c.n_head);
                quant::quantize_row_q8_0(q_.data() + row * uhd, q8_queries_.data() + row * q8_per_head, hd);
            }
        };
        if (rows >= 256) {
            pool.parallel_for(rows, quantize);
        } else {
            quantize(0, rows);
        }
    }
    auto spans_for = [](int32_t n_visible) {
        return std::clamp(n_visible / kAttentionMinSpan, 1, kAttentionMaxSpans);
    };
    // Positions a query of segment s at `pos` sees: [sink_begin, sink_end) ∪ [first, pos].
    auto seen = [&](const Segment& s, int32_t pos) { return seen_by(*s.kv, pos, window); };
    auto spans_of = [&](int32_t t) {
        const Segment& s = segs_[segment_of(t)];
        const int32_t pos = pos_of(s, t);
        const Seen v = seen(s, pos);
        return spans_for(v.sink_end - v.sink_begin + (pos - v.first + 1));
    };

    // Span sp of token t for the query heads of KV head g. For head j of the
    // group: the span's maximum score mx[j * stride], the sum of exp(score -
    // maximum) sm[j * stride] and the V rows weighted by them at out + j *
    // stride * vd (stride: span slots per head).
    auto score_span = [&](int32_t t, int32_t g, int32_t sp, float* mx, float* sm, float* out, size_t stride) {
        thread_local std::vector<float> scores;  // [group][span length]
        thread_local std::vector<float> vrow;    // one V row, decoded (vd values)
        thread_local std::vector<float> dots;    // one K row against the group's queries
        thread_local std::vector<const quant::BlockQ8_0*> queries;  // the group's Q8_0 queries
        const size_t si = segment_of(t);
        const Segment& s = segs_[si];
        const KvCache::RowView& rows = row_views_[si];
        const int32_t pos = pos_of(s, t);
        // Visible positions: attention sinks [sink_begin, sink_end) + [first, pos].
        const Seen vis = seen(s, pos);
        const int32_t n_sinks = vis.sink_end - vis.sink_begin;
        const int32_t n_visible = n_sinks + (pos - vis.first + 1);
        // A tree pass reads the rows of positions n_past + d (the batch's
        // own) from the slot of the token's ancestor at depth d: its branch.
        const int32_t* ancestors =
            tree_depth_.empty() ? nullptr : tree_ancestors_.data() + static_cast<size_t>(t - s.first) * tree_stride_;
        auto position = [&](int32_t i) {
            const int32_t p = i < n_sinks ? vis.sink_begin + i : vis.first + (i - n_sinks);
            return ancestors == nullptr || p < s.n_past ? p : s.n_past + ancestors[p - s.n_past];
        };
        const int32_t spans = spans_for(n_visible);
        const auto i0 = static_cast<int32_t>(int64_t{n_visible} * sp / spans);
        const auto i1 = static_cast<int32_t>(int64_t{n_visible} * (sp + 1) / spans);
        const auto len = static_cast<size_t>(std::max(0, i1 - i0));
        const int32_t h0 = g * group;
        const size_t first_row = static_cast<size_t>(t) * n_head + static_cast<size_t>(h0);  // the group's queries
        scores.resize(ugroup * std::max<size_t>(len, 1));
        if (kv_type == DType::Q8_0) {
            // A Q8_0 cache (the default): the group's queries against each K
            // row in multi-row passes, the same bits as one dot per head
            // (dot_q8_0_q8_0_rows) for about half the instructions: the row
            // is loaded once, and 4 heads share every vector multiply-add.
            queries.resize(ugroup);
            dots.resize(ugroup);
            for (size_t j = 0; j < ugroup; ++j) queries[j] = q8_queries_.data() + (first_row + j) * q8_per_head;
            for (size_t i = 0; i < len; ++i) {
                const auto* k = reinterpret_cast<const quant::BlockQ8_0*>(
                    rows.k(position(i0 + static_cast<int32_t>(i)), g));
                quant::dot_q8_0_q8_0_rows(k, queries.data(), group, hd, dots.data());
                for (size_t j = 0; j < ugroup; ++j) scores[j * len + i] = dots[j] * scale;
            }
        } else {
            for (size_t i = 0; i < len; ++i) {
                const uint8_t* k = rows.k(position(i0 + static_cast<int32_t>(i)), g);
                for (int32_t j = 0; j < group; ++j) {
                    const size_t row = first_row + static_cast<size_t>(j);
                    scores[static_cast<size_t>(j) * len + i] =
                        (quantized_kv
                             ? quant::dot_quantized(kv_type, k, q8_queries_.data() + row * q8_per_head, nullptr, hd)
                             : quant::dot_f16_f32(reinterpret_cast<const uint16_t*>(k), q_.data() + row * uhd, hd)) *
                        scale;
                }
            }
        }
        if (softcap > 0.0f) {  // Gemma 2: the scaled scores squashed into (-cap, cap) before the softmax
            for (size_t i = 0; i < ugroup * len; ++i) scores[i] = softcap * portable_tanh(scores[i] / softcap);
        }
        for (int32_t j = 0; j < group; ++j) {
            const size_t at = static_cast<size_t>(j) * stride;
            float* o = out + at * uvd;
            std::fill(o, o + vd, 0.0f);
            mx[at] = -INFINITY;
            sm[at] = 0.0f;
            if (len == 0 || (head_mask != nullptr && head_mask[h0 + j] == 0)) continue;
            float* sc = scores.data() + static_cast<size_t>(j) * len;
            float m = -INFINITY;
            for (size_t i = 0; i < len; ++i) m = std::max(m, sc[i]);
            float sum = 0.0f;
            for (size_t i = 0; i < len; ++i) sum += sc[i] = std::exp(sc[i] - m);
            mx[at] = m;
            sm[at] = sum;
        }
        // V: each row decoded once into floats, then added to every head of
        // the group (decoding it per head cost 8x the work).
        vrow.resize(uvd);
        for (size_t i = 0; i < len; ++i) {
            const uint8_t* v = rows.v(position(i0 + static_cast<int32_t>(i)), g);
            if (kv_type == DType::F32) {
                std::copy_n(reinterpret_cast<const float*>(v), uvd, vrow.data());
            } else {
                quant::dequantize_row(kv_type, v, vrow.data(), vd);
            }
            for (int32_t j = 0; j < group; ++j) {
                if (head_mask != nullptr && head_mask[h0 + j] == 0) continue;
                const float w = scores[static_cast<size_t>(j) * len + i];
                float* o = out + static_cast<size_t>(j) * stride * uvd;
                for (size_t d = 0; d < uvd; ++d) o[d] += w * vrow[d];
            }
        }
    };
    // Token t's spans of head h (mx[sp], sm[sp], out + sp * vd), merged in
    // order into att_: exp(m_s - M) rescales each span's sum and weighted V
    // to the overall maximum M.
    auto merge = [&](int32_t t, int32_t h, int32_t spans, const float* mx, const float* sm, const float* out) {
        float* dst = att_.data() + static_cast<size_t>(t) * out_dim + static_cast<size_t>(h) * uvd;
        std::fill(dst, dst + vd, 0.0f);
        if (head_mask != nullptr && head_mask[h] == 0) return;  // pruned head: contributes nothing
        float m = -INFINITY;
        for (int32_t sp = 0; sp < spans; ++sp) m = std::max(m, mx[sp]);
        float sum = 0.0f;
        for (int32_t sp = 0; sp < spans; ++sp) {
            if (sm[sp] == 0.0f) continue;
            const float w = std::exp(mx[sp] - m);
            sum += w * sm[sp];
            const float* src = out + static_cast<size_t>(sp) * uvd;
            for (size_t d = 0; d < uvd; ++d) dst[d] += w * src[d];
        }
        const float inv = 1.0f / sum;
        for (size_t d = 0; d < uvd; ++d) dst[d] *= inv;
    };

    // A segment's last token sees the most of its positions; in a tree
    // pass, the deepest.
    int32_t max_spans = 1;
    for (const Segment& s : segs_) {
        if (!s.expanded) max_spans = std::max(max_spans, spans_of(s.first + s.n - 1));
    }
    for (int32_t t = 0; !tree_depth_.empty() && t + 1 < n; ++t) max_spans = std::max(max_spans, spans_of(t));
    const int64_t pairs = int64_t{m} * c.n_head_kv;
    if (max_spans == 1 || pairs >= int64_t{pool.max_threads()} * ThreadPool::kChunksPerThread) {
        pool.parallel_for(pairs, [&](int64_t begin, int64_t end) {
            thread_local std::vector<float> mx, sm, out;  // [head of the group][span]
            for (int64_t task = begin; task < end; ++task) {
                const int32_t t = row_at(task / c.n_head_kv);
                const auto g = static_cast<int32_t>(task % c.n_head_kv);
                const int32_t spans = spans_of(t);
                const auto uspans = static_cast<size_t>(spans);
                mx.resize(ugroup * uspans);
                sm.resize(ugroup * uspans);
                out.resize(ugroup * uspans * uvd);
                for (int32_t sp = 0; sp < spans; ++sp) {
                    const auto usp = static_cast<size_t>(sp);
                    score_span(t, g, sp, mx.data() + usp, sm.data() + usp, out.data() + usp * uvd, uspans);
                }
                for (int32_t j = 0; j < group; ++j) {
                    const size_t at = static_cast<size_t>(j) * uspans;
                    merge(t, g * group + j, spans, mx.data() + at, sm.data() + at, out.data() + at * uvd);
                }
            }
        });
        return;
    }
    // Partial results per (attending row, query head, span slot).
    const auto slots = static_cast<size_t>(max_spans);
    part_max_.resize(static_cast<size_t>(m) * n_head * slots);
    part_sum_.resize(part_max_.size());
    part_out_.resize(part_max_.size() * uvd);
    auto base = [&](int64_t i, int32_t h) { return (static_cast<size_t>(i) * n_head + static_cast<size_t>(h)) * slots; };
    pool.parallel_for(pairs * max_spans, [&](int64_t begin, int64_t end) {
        for (int64_t task = begin; task < end; ++task) {
            const auto sp = static_cast<int32_t>(task % max_spans);
            const int64_t i = task / max_spans / c.n_head_kv;
            const int32_t t = row_at(i);
            const auto g = static_cast<int32_t>(task / max_spans % c.n_head_kv);
            if (sp >= spans_of(t)) continue;  // an earlier token of the batch sees fewer positions
            const size_t at = base(i, g * group) + static_cast<size_t>(sp);
            score_span(t, g, sp, part_max_.data() + at, part_sum_.data() + at, part_out_.data() + at * uvd, slots);
        }
    });
    // Merging many spans is worth a second fork/join (at 3000 positions on
    // Qwen3.6's shape, merging 84 spans on one thread cost 5% of the step).
    constexpr int32_t kParallelMergeSpans = 32;
    auto merge_range = [&](int64_t begin, int64_t end) {
        for (int64_t th = begin; th < end; ++th) {
            const int64_t i = th / c.n_head;
            const int32_t t = row_at(i);
            const auto h = static_cast<int32_t>(th % c.n_head);
            const size_t at = base(i, h);
            merge(t, h, spans_of(t), part_max_.data() + at, part_sum_.data() + at, part_out_.data() + at * uvd);
        }
    };
    if (max_spans >= kParallelMergeSpans) {
        pool.parallel_for(int64_t{m} * c.n_head, merge_range);
    } else {
        merge_range(0, int64_t{m} * c.n_head);
    }
}

Status Transformer::matmul(const Route& route, Backend* backend, const TensorView& w, const float* x, float* y,
                           int32_t n) {
    if (adapters_in_pass_) {
        const TensorView* ws[] = {&w};
        float* ys[] = {y};
        return matmul_group(route, backend, ws, x, ys, n);
    }
    Status s = backend->matmul(w, x, y, n);
    if (s.code() == ErrorCode::Unsupported && backend != route.cpu) s = route.cpu->matmul(w, x, y, n);
    return s;
}

Status Transformer::matmul_group(const Route& route, Backend* backend, std::span<const TensorView* const> ws,
                                 const float* x, std::span<float* const> ys, int32_t n) {
    if (adapters_in_pass_ && collect_adapter_items(ws, x, ys, n)) {
        // The A·x matmuls go with the base ones on the CPU backend (one
        // parallel pass); an accelerator's base products land in host memory
        // like the CPU's, and its A·x run on the CPU after them.
        adapter_batch_.clear();
        Status s = Status(ErrorCode::Unsupported, "");
        if (backend != route.cpu) {
            s = backend->matmul_group(ws, x, ys, n);
            if (s.is_ok()) {
                adapter_runner_.add_a_matmuls(adapter_items_, adapter_batch_);
                s = route.cpu->matmul_batch(adapter_batch_);
            }
        }
        if (s.code() == ErrorCode::Unsupported) {
            adapter_batch_.clear();
            for (size_t i = 0; i < ws.size(); ++i) adapter_batch_.push_back({ws[i], x, ys[i], n});
            adapter_runner_.add_a_matmuls(adapter_items_, adapter_batch_);
            s = route.cpu->matmul_batch(adapter_batch_);
        }
        if (s.is_ok()) adapter_runner_.add_updates(adapter_items_, adapter_pool_);
        return s;
    }
    Status s = backend->matmul_group(ws, x, ys, n);
    if (s.code() == ErrorCode::Unsupported && backend != route.cpu) s = route.cpu->matmul_group(ws, x, ys, n);
    return s;
}

bool Transformer::collect_adapter_items(std::span<const TensorView* const> ws, const float* x,
                                        std::span<float* const> ys, int32_t n) {
    adapter_items_.clear();
    // Rows [first, first + count) of x and of every y, with `adapter`.
    auto add = [&](const AdapterMap& adapter, int32_t first, int32_t count) {
        for (size_t i = 0; i < ws.size(); ++i) {
            const auto it = adapter.find(ws[i]->name);
            if (it == adapter.end() || it->second.target->experts > 0) continue;
            const LoraTarget* t = it->second.target;
            adapter_items_.push_back({t, 0, it->second.scale, x + static_cast<size_t>(first) * static_cast<size_t>(t->a.ne[0]),
                                      ys[i] + static_cast<size_t>(first) * static_cast<size_t>(t->rows), count});
        }
    };
    if (segs_.size() == 1) {  // one sequence: its adapter over every row of this matmul
        if (segs_.front().adapter != nullptr) add(*segs_.front().adapter, 0, n);
    } else if (logits_rows_) {  // forward_batch()'s logits: the rows each segment has among them
        for (const AdapterRows& r : adapter_rows_) add(*r.adapter, r.first, r.n);
    } else {
        for (const Segment& s : segs_) {
            if (s.adapter != nullptr) add(*s.adapter, s.first, s.n);
        }
    }
    return !adapter_items_.empty();
}

Status Transformer::set_adapter(const LoraAdapter* adapter, float scale, ThreadPool* pool) {
    adapter_ = nullptr;
    adapter_targets_.clear();
    adapter_pool_ = pool;
    if (adapter == nullptr) return Status::ok();
    if (!std::isfinite(scale)) return Status(ErrorCode::InvalidArgument, "adapter scale must be finite");
    // The matrices the forward pass multiplies as a whole (an adapter's
    // update is added to such a product), by name: every block role but
    // latent attention's per-head attn_k_b / attn_v_b, and the LM head unless
    // it is the token embeddings (tied).
    std::unordered_map<std::string_view, const TensorView*> applied;
    for (const Layer& L : layers_) {
        for (size_t role = 0; role < kWeightRoles; ++role) {
            if (role == kKb || role == kVb || L.w[role] == nullptr) continue;
            applied.emplace(L.w[role]->name, L.w[role]);
        }
    }
    if (output_ != nullptr && output_ != token_embd_) applied.emplace(output_->name, output_);
    std::unordered_map<std::string_view, BoundAdapter> bound;
    for (const LoraTarget& t : adapter->targets()) {
        const auto it = applied.find(t.base);
        if (it == applied.end()) {
            if (file_->tensor(t.base) == nullptr) {
                return Status(ErrorCode::InvalidArgument,
                              "the adapter changes " + t.base + ", which the model does not have");
            }
            return Status(ErrorCode::Unsupported,
                          "the adapter changes " + t.base + ", which the forward pass does not multiply as a whole");
        }
        const TensorView& w = *it->second;
        const int32_t experts = w.n_dims >= 3 ? static_cast<int32_t>(w.ne[2]) : 0;
        if (t.experts != experts || t.a.ne[0] != w.ne[0] || t.rows != w.ne[1] || w.ne[3] != 1) {
            return Status(ErrorCode::InvalidArgument, "the adapter's " + t.base + " does not fit the model's shape");
        }
        bound.emplace(t.base, BoundAdapter{&t, adapter->scale_of(t, scale)});
    }
    adapter_targets_ = std::move(bound);
    adapter_ = adapter;
    return Status::ok();
}

void apply_final_softcap(std::span<float> logits, float cap, ThreadPool& pool) {
    // Elementwise, so chunking cannot change a bit (portable_tanh: the same
    // bits on every platform, which test_golden pins). tanh costs ~5 ns per
    // element: one thread over Gemma 2's 256k vocabulary took 1-2 ms per token.
    // Below this size the fork/join (a few us) is not worth it.
    constexpr size_t kParallelMin = 16384;
    float* p = logits.data();
    auto run = [p, cap](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) p[i] = cap * portable_tanh(p[i] / cap);
    };
    if (logits.size() < kParallelMin || pool.max_threads() == 1) {
        run(0, static_cast<int64_t>(logits.size()));
    } else {
        pool.parallel_for(static_cast<int64_t>(logits.size()), run);
    }
}

Status Transformer::compute_logits(const Route& route, size_t first_row, int32_t rows, ThreadPool& pool) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const PhaseTimer timer(phases_.lm_head);
    file_->begin_layer(c.n_layers);  // output head slot of the prefetch window
    for (int32_t r = 0; r < rows; ++r) {
        rmsnorm(x_.data() + (first_row + static_cast<size_t>(r)) * d, output_norm_.data(),
                xb_.data() + static_cast<size_t>(r) * d, c.n_embd, c.rms_eps);
    }
    logits_.resize(static_cast<size_t>(rows) * static_cast<size_t>(c.n_vocab));
    LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, *output_, xb_.data(), logits_.data(), rows));
    if (c.final_softcap > 0.0f) {  // Gemma 2: logits squashed into (-cap, cap)
        apply_final_softcap(std::span<float>(logits_.data(), logits_.size()), c.final_softcap, pool);
    }
    return Status::ok();
}

Status Transformer::final_hidden(std::span<const int32_t> rows, std::span<float> out) const {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    if (out.size() < rows.size() * d) return Status(ErrorCode::InvalidArgument, "final_hidden: output buffer too small");
    // x_ holds the last pass's residual stream, one row per token (forward() sizes it to the pass).
    const size_t pass_rows = x_.size() / d;
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] < 0 || static_cast<size_t>(rows[i]) >= pass_rows) {
            return Status(ErrorCode::InvalidArgument, "final_hidden: row outside the last pass");
        }
        // The same norm compute_logits() applies before the LM head.
        rmsnorm(x_.data() + static_cast<size_t>(rows[i]) * d, output_norm_.data(), out.data() + i * d, c.n_embd,
                c.rms_eps);
    }
    return Status::ok();
}

Status Transformer::project_fused_qkv(const Route& route, Backend* backend, const BlockWeights& w, int32_t n) {
    const ModelConfig& c = config_;
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(c.head_dim);
    const size_t kv_dim = static_cast<size_t>(c.n_head_kv) * static_cast<size_t>(c.head_dim);
    const size_t row = q_dim + 2 * kv_dim;
    qkv_.resize(static_cast<size_t>(n) * row);
    LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kQkv], xb_.data(), qkv_.data(), n));
    for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
        const float* src = qkv_.data() + t * row;
        std::copy(src, src + q_dim, q_.data() + t * q_dim);
        std::copy(src + q_dim, src + q_dim + kv_dim, k_.data() + t * kv_dim);
        std::copy(src + q_dim + kv_dim, src + row, v_.data() + t * kv_dim);
    }
    return Status::ok();
}

Status Transformer::propagate_kv(const Route& route, int32_t from, int32_t pos) {
    const ModelConfig& c = config_;
    // Only reached for attention-only stacks: early exit is disabled for hybrid models.
    for (int32_t l = from; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        Backend* backend = w.streamed() ? route.cpu : route.attention;  // see BlockWeights::streamed()
        rmsnorm(x_.data(), L.attn_norm.data(), xb_.data(), c.n_embd, c.rms_eps);
        if (c.blocks[static_cast<size_t>(l)].latent) {
            // The key row only: its latent channels are the values.
            LIYAB_RETURN_IF_ERROR(latent_project(l, w, 1, route, backend, false));
        } else if (c.blocks[static_cast<size_t>(l)].fused_qkv) {
            LIYAB_RETURN_IF_ERROR(project_fused_qkv(route, backend, w, 1));
        } else {
            LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kK], xb_.data(), k_.data(), 1));
            LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kV], xb_.data(), v_.data(), 1));
        }
        for (size_t i = 0; i < L.bk.size(); ++i) k_[i] += L.bk[i];
        for (size_t i = 0; i < L.bv.size(); ++i) v_[i] += L.bv[i];
        const bool norm_after_rope = c.spec->qk_norm == QkNormOrder::AfterRope;
        if (!norm_after_rope) qk_norm(L, nullptr, k_.data());
        rope(k_.data(), c.n_head_kv, pos, rope_tables_[static_cast<size_t>(L.rope_table)]);
        if (norm_after_rope) qk_norm(L, nullptr, k_.data());
        scale_keys(L.kv_slot, nullptr, k_.data());
        kv_->store(L.kv_slot, pos, k_.data(), v_.data());
    }
    return Status::ok();
}

KeyScalingShape Transformer::key_scaling_shape() const {
    const ModelConfig& c = config_;
    // The horizon depends on the model only, never on the configured context,
    // so every process gets the same factors (and the same replies).
    const int32_t positions = c.n_ctx_train > 0 ? std::min(c.n_ctx_train, kKeyRangePositions) : kKeyRangePositions;
    KeyScalingShape shape{c.n_attn_layers, c.n_head, c.n_head_kv, c.head_dim, c.rope_dim,
                          c.spec->rope, {}, {}, positions};
    // Transformer::rope: pair i of an attention block turns by
    // inv_freq[i] / scale of the block's table per position.
    for (const Layer& L : layers_) {
        if (L.kv_slot < 0) continue;  // attention blocks only, in KV slot order
        const RopeTable& t = rope_tables_[static_cast<size_t>(L.rope_table)];
        for (const float f : t.inv_freq) shape.pair_speed.push_back(f / t.scale);
        shape.rope_magnitude.push_back(t.magnitude);
    }
    return shape;
}

void Transformer::qk_norm(const Layer& L, float* q, float* k) const {
    const ModelConfig& c = config_;
    const auto hd = static_cast<size_t>(c.head_dim);
    if (q != nullptr && !L.q_norm.empty()) {
        for (size_t h = 0; h < static_cast<size_t>(c.n_head); ++h) {
            rmsnorm(q + h * hd, L.q_norm.data(), q + h * hd, c.head_dim, c.rms_eps);
        }
    }
    if (!L.k_norm.empty()) {
        for (size_t h = 0; h < static_cast<size_t>(c.n_head_kv); ++h) {
            rmsnorm(k + h * hd, L.k_norm.data(), k + h * hd, c.head_dim, c.rms_eps);
        }
    }
}

void Transformer::add_key_ranges(const Layer& L, const float* q, const float* k) {
    const ModelConfig& c = config_;
    const bool after = c.spec->qk_norm == QkNormOrder::AfterRope;
    if (!after || (L.q_norm.empty() && L.k_norm.empty())) {
        key_ranges_->add(L.kv_slot, q, k);
        return;
    }
    // Norms after RoPE: a head's stored vector is rope(x) * s * w, s the RMS
    // factor, the same for every rotation (a rotation keeps the norm). So the
    // ranges see x * s, unrotated, and the norm weight w as a gain applied
    // after the rotation (KeyRanges::add).
    const auto hd = static_cast<size_t>(c.head_dim);
    // RoPE's magnitude (LongRoPE) scales the rotated channels before the norm.
    const float magnitude = rope_tables_[static_cast<size_t>(L.rope_table)].magnitude;
    std::vector<float> rotated(hd);
    auto scaled = [&](const float* x, int32_t heads, bool normed, std::vector<float>& out) {
        out.assign(x, x + static_cast<size_t>(heads) * hd);
        if (!normed) return;
        for (size_t h = 0; h < static_cast<size_t>(heads); ++h) {
            float* xh = out.data() + h * hd;
            std::copy(xh, xh + hd, rotated.begin());
            for (size_t i = 0; i < static_cast<size_t>(c.rope_dim); ++i) rotated[i] *= magnitude;
            const float s = rms_scale(rotated.data(), c.head_dim, c.rms_eps);
            for (size_t i = 0; i < hd; ++i) xh[i] *= s;
        }
    };
    std::vector<float> qs, ks;
    scaled(q, c.n_head, !L.q_norm.empty(), qs);
    scaled(k, c.n_head_kv, !L.k_norm.empty(), ks);
    key_ranges_->add(L.kv_slot, qs.data(), ks.data(), L.q_norm.empty() ? nullptr : L.q_norm.data(),
                     L.k_norm.empty() ? nullptr : L.k_norm.data());
}

void Transformer::scale_keys(int32_t kv_slot, float* q, float* k) const {
    if (key_scale_.empty()) return;
    const ModelConfig& c = config_;
    const auto hd = static_cast<size_t>(c.head_dim);
    const auto group = static_cast<size_t>(c.n_head / c.n_head_kv);
    const float* t = key_scale_.data() + static_cast<size_t>(kv_slot) * static_cast<size_t>(c.n_head_kv) * hd;
    for (size_t g = 0; g < static_cast<size_t>(c.n_head_kv); ++g) {
        const float* tg = t + g * hd;
        float* kg = k + g * hd;
        for (size_t i = 0; i < hd; ++i) kg[i] /= tg[i];
        if (q == nullptr) continue;
        for (size_t j = 0; j < group; ++j) {
            float* qh = q + (g * group + j) * hd;
            for (size_t i = 0; i < hd; ++i) qh[i] *= tg[i];
        }
    }
}

Status Transformer::calibrate_keys(std::span<const int32_t> tokens, const Route& route, ThreadPool& pool) {
    key_scale_.clear();
    if (!quant::is_block_quantized(kv_->dtype()) || config_.n_attn_layers == 0) return Status::ok();
    if (tokens.empty()) return Status(ErrorCode::InvalidArgument, "key calibration needs tokens");
    reset();
    key_ranges_ = std::make_unique<KeyRanges>(key_scaling_shape());
    // A probe first: the outlier channels are nearly constant, so a few
    // tokens show whether the model has them, and fewer tokens only raise the
    // ratio (lower medians). Most models stop there; the rest of the text is
    // processed only to measure their factors, and decides again.
    Status status;
    float ratio = 0.0f;
    size_t measured = 0;
    while (measured < tokens.size() && status.is_ok()) {
        const size_t limit = measured == 0 ? std::min(tokens.size(), kKeyProbeTokens) : tokens.size();
        const auto n = std::min(limit - measured, static_cast<size_t>(max_batch_));
        status = forward(tokens.subspan(measured, n), Logits::None, route, pool).status();
        measured += n;
        if (!status.is_ok() || measured < limit) continue;
        ratio = key_ranges_->outlier_ratio();
        if (ratio < kOutlierRatio) break;
    }
    if (status.is_ok() && ratio >= kOutlierRatio) key_scale_ = key_ranges_->factors();
    key_ranges_.reset();
    reset();
    if (!status.is_ok()) return status;
    if (ratio < kOutlierRatio) {
        LIYAB_LOG_INFO("keys: largest channel %.1fx its head's median over %zu tokens: no scaling needed", ratio,
                       measured);
        return Status::ok();
    }
    const auto largest = std::max_element(key_scale_.begin(), key_scale_.end());
    const auto per_block = static_cast<size_t>(config_.n_head_kv) * static_cast<size_t>(config_.head_dim);
    LIYAB_LOG_INFO("keys: largest channel %.1fx its head's median over %zu tokens: per-channel scales for the %s "
                   "cache (largest %.1f, attention block %zu)",
                   ratio, measured, std::string(dtype_traits(kv_->dtype()).name).c_str(),
                   static_cast<double>(*largest), static_cast<size_t>(largest - key_scale_.begin()) / per_block);
    return Status::ok();
}

Status Transformer::set_key_scales(std::span<const float> scales) {
    if (n_past_ != 0) return Status(ErrorCode::InvalidArgument, "key scales change only with an empty context");
    if (scales.empty()) {
        key_scale_.clear();
        return Status::ok();
    }
    if (!quant::is_block_quantized(kv_->dtype())) {
        return Status(ErrorCode::InvalidArgument, "key scales apply to quantized KV caches only");
    }
    if (scales.size() != key_scaling_shape().factors()) {
        return Status(ErrorCode::InvalidArgument, "key scales of another model");
    }
    for (const float t : scales) {
        if (!(t >= 1.0f / kMaxKeyScale && t <= kMaxKeyScale)) {
            return Status(ErrorCode::InvalidArgument, "key scale out of range");
        }
    }
    key_scale_.assign(scales.begin(), scales.end());
    return Status::ok();
}

// One synthetic pass through the first latent block in each form, on its
// real weights: cached rows of random latents (a fixed seed), random
// normalized inputs, timed over the parts that differ between the forms
// (latent_ms_, as in forward()). The ratio of the two forms' time per
// counted multiply-add is what LatentFormChooser needs; one block suffices
// because every latent block has the same shape. The best of kRepeats runs
// per form, alternated, keeps a passing burst of load out of it.
Result<float> Transformer::calibrate_latent_form(const Route& route, ThreadPool& pool) {
    const ModelConfig& c = config_;
    if (c.mla.kv_rank == 0) return 1.0f;
    int32_t layer = 0;
    while (!c.blocks[static_cast<size_t>(layer)].latent) ++layer;
    constexpr int32_t kContext = 1024, kBatch = 256, kRepeats = 3;
    const int32_t context = std::min(kContext, context_length_);
    const int32_t n = std::max(2, std::min(kBatch, context / 2));
    reset();
    LIYAB_RETURN_IF_ERROR(kv_->reserve(context));
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto kd = static_cast<size_t>(c.head_dim);
    std::mt19937 rng(20251017);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::vector<float> row(kd);
    for (int32_t pos = 0; pos < context - n; ++pos) {
        for (float& v : row) v = normal(rng);
        kv_->store(L.kv_slot, pos, row.data(), nullptr);
    }
    n_past_ = context - n;
    const auto un = static_cast<size_t>(n);
    const size_t d = static_cast<size_t>(c.n_embd);
    const size_t q_dim = static_cast<size_t>(c.n_head) * kd;
    xb_.resize(un * std::max(d, q_dim));
    q_.resize(un * q_dim);
    k_.resize(un * kd);
    att_.resize(un * q_dim);
    std::vector<float> input(un * d);
    for (float& v : input) v = normal(rng);
    for (size_t t = 0; t < un; ++t) rmsnorm(input.data() + t * d, L.attn_norm.data(), input.data() + t * d, c.n_embd, c.rms_eps);
    const int64_t pairs = latent_visible(live_segment(0, n, Logits::None), L.window);
    int64_t positions = 0;
    for (const auto& [a, b] : mla_union_) positions += b - a;
    const LatentFormChooser::Work work = LatentFormChooser::work(c, n, pairs, positions);

    Status status;
    double best[2] = {INFINITY, INFINITY};  // [absorbed, expanded], ms
    {
        const BlockWeights w(*this, layer);
        status = w.status();
        for (int32_t r = 0; r < kRepeats && status.is_ok(); ++r) {
            for (const bool expanded : {false, true}) {
                std::copy(input.begin(), input.end(), xb_.begin());
                latent_expanded_ = expanded;
                begin_live_pass(n, Logits::None);
                latent_ms_ = 0.0;
                status = latent_attention_mixer(layer, w, n, route, pool, nullptr);
                if (!status.is_ok()) break;
                best[expanded ? 1 : 0] = std::min(best[expanded ? 1 : 0], latent_ms_);
            }
        }
    }
    latent_expanded_ = false;
    reset();
    if (!status.is_ok()) return status;
    const double measured = (best[1] / work.expanded) / (best[0] / work.absorbed);
    const float ratio = LatentFormChooser::round_ratio(measured);
    LIYAB_RETURN_IF_ERROR(latent_chooser_.set_ratio(ratio));
    LIYAB_LOG_INFO("latent attention: an expanded multiply-add costs %.2fx an absorbed one (%.1f / %.1f ms on a "
                   "%d-token pass after %d positions); ratio %.3f",
                   measured, best[1], best[0], n, context - n, static_cast<double>(ratio));
    return ratio;
}

Status Transformer::attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                    ThreadPool& pool, const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const bool gated = c.blocks[static_cast<size_t>(layer)].attn_gate;
    const auto un = static_cast<size_t>(n);
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(c.head_dim);
    const size_t kv_dim = static_cast<size_t>(c.n_head_kv) * static_cast<size_t>(c.head_dim);
    const size_t hd = static_cast<size_t>(c.head_dim);
    const bool norm_after_rope = c.spec->qk_norm == QkNormOrder::AfterRope;
    const RopeTable& rope_table = rope_tables_[static_cast<size_t>(L.rope_table)];

    // Gated blocks project [q_h | gate_h] per head into qg_; split it afterwards.
    if (gated) qg_.resize(un * 2 * q_dim);
    if (c.blocks[static_cast<size_t>(layer)].fused_qkv) {
        // One [q | k | v] projection per token, split into q_, k_ and v_.
        LIYAB_RETURN_IF_ERROR(project_fused_qkv(route, route.attention, w, n));
    } else {
        const TensorView* qkv[] = {&w[kQ], &w[kK], &w[kV]};
        float* outs[] = {gated ? qg_.data() : q_.data(), k_.data(), v_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, qkv, xb_.data(), outs, n));
    }
    if (gated) {
        gate_.resize(un * q_dim);
        for (size_t t = 0; t < un; ++t) {
            for (size_t h = 0; h < static_cast<size_t>(c.n_head); ++h) {
                const float* src = qg_.data() + (t * static_cast<size_t>(c.n_head) + h) * 2 * hd;
                std::copy(src, src + hd, q_.data() + t * q_dim + h * hd);
                std::copy(src + hd, src + 2 * hd, gate_.data() + t * q_dim + h * hd);
            }
        }
    }
    for (const Segment& s : segs_) {
        for (auto t = static_cast<size_t>(s.first); t < static_cast<size_t>(s.first + s.n); ++t) {
            float* q = q_.data() + t * q_dim;
            float* k = k_.data() + t * kv_dim;
            float* v = v_.data() + t * kv_dim;
            for (size_t i = 0; i < L.bq.size(); ++i) q[i] += L.bq[i];
            for (size_t i = 0; i < L.bk.size(); ++i) k[i] += L.bk[i];
            for (size_t i = 0; i < L.bv.size(); ++i) v[i] += L.bv[i];
            if (!norm_after_rope) qk_norm(L, q, k);
            const int32_t pos = pos_of(s, static_cast<int32_t>(t));
            if (key_ranges_ != nullptr) add_key_ranges(L, q, k);  // unrotated: ranges cover every position
            rope(q, c.n_head, pos, rope_table);
            rope(k, c.n_head_kv, pos, rope_table);
            if (norm_after_rope) qk_norm(L, q, k);
            scale_keys(L.kv_slot, q, k);
            s.kv->store(L.kv_slot, row_of(s, static_cast<int32_t>(t)), k, v);
        }
    }
    attention(L.kv_slot, L.window, n, pool, head_mask);
    if (gated) {
        for (size_t i = 0; i < un * q_dim; ++i) att_[i] *= sigmoid(gate_[i]);
    }
    return matmul(route, route.attention, w[kO], att_.data(), xb_.data(), n);
}

Status Transformer::per_head_matmul(const Route& route, const TensorView& stacked, const float* in, float* out,
                                    int32_t n) {
    const auto heads = static_cast<size_t>(stacked.ne[2]);
    const auto cols = static_cast<size_t>(stacked.ne[0]);
    const auto rows = static_cast<size_t>(stacked.ne[1]);
    const auto un = static_cast<size_t>(n);
    mla_head_views_.resize(heads);
    mla_items_.clear();
    for (size_t h = 0; h < heads; ++h) {
        mla_head_views_[h] = expert_slice(stacked, static_cast<int64_t>(h));
        mla_items_.push_back({&mla_head_views_[h], in + h * un * cols, out + h * un * rows, n});
    }
    // Transient views into a 3-D tensor, as the experts' are: the CPU reads
    // them in place, one parallel pass for every head (a matmul per head
    // would pay a thread fork/join each).
    return route.cpu->matmul_batch(mla_items_);
}

Status Transformer::latent_project(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                   Backend* backend, bool queries) {
    const ModelConfig& c = config_;
    const LatentAttention& m = c.mla;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const size_t qk = static_cast<size_t>(m.nope_dim) + rd;  // a head's query as projected: [q_nope | q_pe]
    const auto kd = static_cast<size_t>(c.head_dim);          // rd + rank: a cached key row
    mla_kv_.resize(un * (rank + rd));
    if (!queries) {
        LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kKvA], xb_.data(), mla_kv_.data(), n));
    } else {
        const bool low_rank = m.q_rank > 0;  // q = W_qb RMSNorm(W_qa x), else q = W_q x
        mla_q_.resize(un * heads * qk);
        if (low_rank) mla_qa_.resize(un * static_cast<size_t>(m.q_rank));
        const TensorView* proj[] = {low_rank ? &w[kQa] : &w[kQ], &w[kKvA]};
        float* outs[] = {low_rank ? mla_qa_.data() : mla_q_.data(), mla_kv_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, backend, proj, xb_.data(), outs, n));
        if (low_rank) {
            for (size_t t = 0; t < un; ++t) {
                float* row = mla_qa_.data() + t * static_cast<size_t>(m.q_rank);
                rmsnorm(row, L.q_a_norm.data(), row, m.q_rank, c.rms_eps);
            }
            LIYAB_RETURN_IF_ERROR(matmul(route, backend, w[kQb], mla_qa_.data(), mla_q_.data(), n));
        }
    }
    // The key row: [k_pe | RMSNorm(latent)] (the projection gives [latent |
    // k_pe]). RoPE's channels come first, as in every other head this
    // engine rotates, so rope() and the key ranges treat it like any head.
    for (size_t t = 0; t < un; ++t) {
        const float* kv = mla_kv_.data() + t * (rank + rd);
        float* k = k_.data() + t * kd;
        std::copy(kv + rank, kv + rank + rd, k);
        rmsnorm(kv, L.kv_a_norm.data(), k + rd, m.kv_rank, c.rms_eps);
    }
    return Status::ok();
}

Status Transformer::latent_absorb(const BlockWeights& w, int32_t n, const Route& route) {
    const ModelConfig& c = config_;
    const LatentAttention& m = c.mla;
    const auto un = static_cast<size_t>(n);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const auto nope = static_cast<size_t>(m.nope_dim);
    const size_t qk = nope + rd;                      // a head's query as projected: [q_nope | q_pe]
    const auto kd = static_cast<size_t>(c.head_dim);  // rd + rank: an absorbed query, a cached key
    // Each head's query absorbs its key projection: W_kb,h^T q_nope,h is what
    // the latent is scored against. The q_nope of every token are gathered
    // per head so each head is one matmul over the batch.
    mla_heads_in_.resize(heads * un * nope);
    for (size_t t = 0; t < un; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            const float* src = mla_q_.data() + (t * heads + h) * qk;
            std::copy(src, src + nope, mla_heads_in_.data() + (h * un + t) * nope);
        }
    }
    mla_heads_out_.resize(heads * un * rank);
    LIYAB_RETURN_IF_ERROR(per_head_matmul(route, w[kKb], mla_heads_in_.data(), mla_heads_out_.data(), n));
    for (size_t t = 0; t < un; ++t) {
        for (size_t h = 0; h < heads; ++h) {
            float* q = q_.data() + (t * heads + h) * kd;
            const float* pe = mla_q_.data() + (t * heads + h) * qk + nope;
            std::copy(pe, pe + rd, q);
            const float* absorbed = mla_heads_out_.data() + (h * un + t) * rank;
            std::copy(absorbed, absorbed + rank, q + rd);
        }
    }
    return Status::ok();
}

int64_t Transformer::latent_visible(const Segment& seg, int32_t window) {
    mla_union_.clear();
    int64_t pairs = 0;
    for (int32_t t = 0; t < seg.n; ++t) {
        const int32_t pos = seg.n_past + t;
        const Seen s = seen_by(*seg.kv, pos, window);
        pairs += (s.sink_end - s.sink_begin) + (pos - s.first + 1);
        if (s.sink_end > s.sink_begin) mla_union_.emplace_back(s.sink_begin, s.sink_end);
        mla_union_.emplace_back(s.first, pos + 1);
    }
    std::sort(mla_union_.begin(), mla_union_.end());
    size_t kept = 0;
    for (size_t i = 0; i < mla_union_.size(); ++i) {
        if (kept > 0 && mla_union_[i].first <= mla_union_[kept - 1].second) {
            mla_union_[kept - 1].second = std::max(mla_union_[kept - 1].second, mla_union_[i].second);
        } else {
            mla_union_[kept++] = mla_union_[i];
        }
    }
    mla_union_.resize(kept);
    return pairs;
}

int32_t Transformer::latent_tile() const noexcept {
    const ModelConfig& c = config_;
    const auto row = static_cast<int64_t>(c.n_head) * (c.mla.nope_dim + c.rope_dim + c.mla.v_dim) *
                     static_cast<int64_t>(sizeof(float));
    int32_t tile = kLatentTileMax;
    while (tile > kLatentTileMin && tile * row > static_cast<int64_t>(kLatentTileBytes)) tile /= 2;
    return tile;
}

// Latent attention in the expanded form, the form the model was trained in:
// per head h, key [W_kb,h c_j | k_pe,j] and value W_vb,h c_j of every
// position j the pass sees, scored against [q_nope,h | q_pe,h] with the same
// scale. Per (token, position, head) that is nope + rope_dim channels scored
// and v_dim accumulated (192 and 128 on DeepSeek-V2-Lite) where the absorbed
// form takes rope_dim + kv_rank and kv_rank (576 and 512), for the cost of
// expanding every position the pass sees once per pass instead of absorbing
// each of its tokens: long prefills gain, decoding steps lose
// (LatentFormChooser weighs the two). The cache keeps latents only; expanded
// rows exist one tile at a time (latent_tile(), about kLatentTileBytes).
//
// Positions go by tiles at absolute bounds (multiples of latent_tile()). Per
// tile, the rows the pass sees are read from the cache (key scales multiplied
// back: stored rows were divided by them), expanded by one GEMM
// (expand_strip) into [k_nope | k_pe] and v per head, and every (token, head)
// folds the tile into its running softmax (maximum, sum, weighted values),
// tiles in ascending order. A token's result therefore depends only on its
// position, its query and the cached rows: the same bits however a prompt is
// split into passes and however many threads run (within this form; the
// absorbed form rounds differently).
Status Transformer::latent_expanded(int32_t layer, const BlockWeights& w, const Segment& seg, ThreadPool& pool,
                                    const uint8_t* head_mask) {
    const ModelConfig& c = config_;
    const LatentAttention& m = c.mla;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    // The segment's tokens, 0..un-1 here; batch row first + i in q_, mla_q_ and att_.
    const auto un = static_cast<size_t>(seg.n);
    const auto first = static_cast<size_t>(seg.first);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const auto nope = static_cast<size_t>(m.nope_dim);
    const auto vd = static_cast<size_t>(m.v_dim);
    const auto kd = static_cast<size_t>(c.head_dim);
    const size_t qk = nope + rd;            // a head's expanded key (and query): [nope | rope]
    const size_t row = heads * (qk + vd);   // an expanded row: every head's key, then every head's value
    const size_t key_cols = heads * nope;   // GEMM columns: W_kb,h's for every head, then W_vb,h's
    const size_t cols = key_cols + heads * vd;
    const auto strip = static_cast<size_t>(kExpandColumns);
    const size_t strips = (cols + strip - 1) / strip;

    // The expansion weights, decoded once per pass and packed per strip:
    // mla_wp_[s][r][i] is column s * kExpandColumns + i's weight on latent
    // channel r (zero past the last column). attn_k_b holds per head the
    // kv_rank rows of nope weights (k_nope = sum over r of c_r row_r),
    // attn_v_b the v_dim rows of kv_rank weights (v_d = row_d . c).
    mla_wp_.assign(strips * rank * strip, 0.0f);
    mla_col_offset_.assign(strips * strip, 0);
    for (size_t col = 0; col < cols; ++col) {
        mla_col_offset_[col] = col < key_cols ? col / nope * qk + col % nope : heads * qk + (col - key_cols);
    }
    const TensorView& kb = w[kKb];
    const TensorView& vb = w[kVb];
    pool.parallel_for(static_cast<int64_t>(heads), [&](int64_t h0, int64_t h1) {
        thread_local std::vector<float> buf;
        buf.resize(std::max(nope, rank));
        for (int64_t h = h0; h < h1; ++h) {
            const auto uh = static_cast<size_t>(h);
            const TensorView kh = expert_slice(kb, h);
            for (size_t r = 0; r < rank; ++r) {
                quant::dequantize_row(kh.type, kh.row(static_cast<int64_t>(r)), buf.data(), kh.cols());
                for (size_t i = 0; i < nope; ++i) {
                    const size_t col = uh * nope + i;
                    mla_wp_[(col / strip * rank + r) * strip + col % strip] = buf[i];
                }
            }
            const TensorView vh = expert_slice(vb, h);
            for (size_t d = 0; d < vd; ++d) {
                quant::dequantize_row(vh.type, vh.row(static_cast<int64_t>(d)), buf.data(), vh.cols());
                const size_t col = key_cols + uh * vd + d;
                float* dst = mla_wp_.data() + col / strip * rank * strip + col % strip;
                for (size_t r = 0; r < rank; ++r) dst[r * strip] = buf[r];
            }
        }
    });

    // Running softmax per (token, head); the weighted values accumulate in
    // att_ in the [token][head][v] order attn_output reads.
    mla_max_.assign(un * heads, -INFINITY);
    mla_sum_.assign(un * heads, 0.0f);
    float* const att = att_.data() + first * heads * vd;  // this segment's rows (others' are left as they are)
    std::fill(att, att + un * heads * vd, 0.0f);
    const float scale = c.attn_scale;
    const float softcap = c.attn_softcap;
    const DType kv_type = seg.kv->dtype();
    const float* key_scales = key_scale_.empty() ? nullptr : key_scale_.data() + static_cast<size_t>(L.kv_slot) * kd;
    const int32_t tile = latent_tile();
    latent_visible(seg, L.window);
    const int32_t lo = mla_union_.front().first;
    const int32_t hi = mla_union_.back().second;
    for (int32_t t0 = lo / tile * tile; t0 < hi; t0 += tile) {
        const int32_t t1 = t0 + tile;
        mla_rows_.clear();  // the tile's positions the pass sees, ascending
        for (const auto& [a, b] : mla_union_) {
            for (int32_t p = std::max(a, t0); p < std::min(b, t1); ++p) mla_rows_.push_back(p);
        }
        if (mla_rows_.empty()) continue;
        const auto rows = static_cast<int64_t>(mla_rows_.size());
        const auto urows = static_cast<size_t>(rows);
        mla_c_.resize(urows * rank);
        mla_ex_.resize(urows * row);

        // The cached rows, decoded: the latent into mla_c_, the RoPE key into
        // every head's key in mla_ex_ (it is shared by the heads).
        auto decode = [&](int64_t j0, int64_t j1) {
            thread_local std::vector<float> key;
            key.resize(kd);
            for (int64_t j = j0; j < j1; ++j) {
                const auto uj = static_cast<size_t>(j);
                const uint8_t* src = seg.kv->k_row(L.kv_slot, mla_rows_[uj], 0);
                quant::dequantize_row(kv_type, src, key.data(), static_cast<int64_t>(kd));
                if (key_scales != nullptr) {
                    for (size_t i = 0; i < kd; ++i) key[i] *= key_scales[i];
                }
                std::copy(key.begin() + static_cast<std::ptrdiff_t>(rd), key.end(), mla_c_.data() + uj * rank);
                float* ex = mla_ex_.data() + uj * row;
                for (size_t h = 0; h < heads; ++h) {
                    std::copy(key.begin(), key.begin() + static_cast<std::ptrdiff_t>(rd), ex + h * qk + nope);
                }
            }
        };
        if (rows >= 64) {
            pool.parallel_for(rows, decode);
        } else {
            decode(0, rows);
        }
        // Keys and values of every head: one strip of columns per task.
        pool.parallel_for(static_cast<int64_t>(strips), [&](int64_t s0, int64_t s1) {
            for (int64_t s = s0; s < s1; ++s) {
                const auto us = static_cast<size_t>(s);
                const auto columns = static_cast<int32_t>(std::min(strip, cols - us * strip));
                expand_strip(mla_c_.data(), rows, static_cast<int64_t>(rank), mla_wp_.data() + us * rank * strip,
                             mla_ex_.data(), row, mla_col_offset_.data() + us * strip, columns);
            }
        });
        // Every (token, head) folds in the tile's positions it sees. A task
        // takes one head and a group of tokens and goes through the tile's
        // rows kAttendRows at a time, so a chunk of that head's keys and
        // values (80 KiB on DeepSeek-V2-Lite) is read from L2 once for the
        // whole group and reused from L1 by each block of kScoreQueries
        // tokens. Each token still sees its rows in ascending order, and its
        // block-mates (a last short block repeats its last token, computed,
        // not used) change no bit of it.
        const auto block = static_cast<size_t>(kScoreQueries);
        size_t group = kAttendTokens;  // fewer tokens per task when there are too few tasks for the threads
        while (group > block && heads * ((un + group - 1) / group) <
                                    static_cast<size_t>(pool.max_threads()) * ThreadPool::kChunksPerThread) {
            group /= 2;
        }
        const size_t groups = (un + group - 1) / group;
        pool.parallel_for(static_cast<int64_t>(groups * heads), [&](int64_t a, int64_t b) {
            thread_local std::vector<float> scores;                    // [token of the group][row]
            thread_local std::vector<std::array<std::pair<size_t, size_t>, 2>> ranges;  // per token: sinks, [first, pos]
            scores.resize(group * urows);
            ranges.resize(group);
            for (int64_t task = a; task < b; ++task) {
                const size_t h = static_cast<size_t>(task) % heads;
                const size_t g0 = static_cast<size_t>(task) / heads * group;
                if (head_mask != nullptr && head_mask[h] == 0) continue;  // pruned head: contributes nothing
                const size_t count = std::min(group, un - g0);
                auto index = [&](int32_t pos) {
                    return static_cast<size_t>(std::lower_bound(mla_rows_.begin(), mla_rows_.end(), pos) -
                                               mla_rows_.begin());
                };
                // Each token's visible rows, as index ranges into the ascending mla_rows_.
                size_t lo = urows, hi = 0;
                for (size_t j = 0; j < count; ++j) {
                    const int32_t pos = seg.n_past + static_cast<int32_t>(g0 + j);
                    const Seen s = seen_by(*seg.kv, pos, L.window);
                    ranges[j][0] = {index(s.sink_begin), index(s.sink_end)};
                    ranges[j][1] = {index(std::max(s.first, s.sink_end)), index(pos + 1)};
                    for (const auto& [i0, i1] : ranges[j]) {
                        if (i0 >= i1) continue;
                        lo = std::min(lo, i0);
                        hi = std::max(hi, i1);
                    }
                }
                if (lo >= hi) continue;  // the group sees nothing in this tile
                auto token = [&](size_t j0, size_t j) { return std::min(j0 + j, count - 1); };  // a short block's padding
                // Scores of every token of the group over [lo, hi), by chunks of rows.
                for (size_t c0 = lo; c0 < hi; c0 += kAttendRows) {
                    const size_t c1 = std::min(hi, c0 + kAttendRows);
                    for (size_t j0 = 0; j0 < count; j0 += block) {
                        const float* q[kScoreQueries];
                        for (size_t j = 0; j < block; ++j) q[j] = mla_q_.data() + ((first + g0 + token(j0, j)) * heads + h) * qk;
                        for (size_t i = c0; i < c1; i += kScoreKeys) {
                            const float* k[kScoreKeys];
                            for (size_t r = 0; r < static_cast<size_t>(kScoreKeys); ++r) {
                                k[r] = mla_ex_.data() + std::min(i + r, c1 - 1) * row + h * qk;
                            }
                            float dots[kScoreQueries][kScoreKeys];
                            score_block(q, k, qk, dots);
                            for (size_t j = 0; j < block && j0 + j < count; ++j) {
                                for (size_t r = 0; r < static_cast<size_t>(kScoreKeys) && i + r < c1; ++r) {
                                    float sc = dots[j][r] * scale;
                                    if (softcap > 0.0f) sc = softcap * portable_tanh(sc / softcap);
                                    scores[(j0 + j) * urows + i + r] = sc;
                                }
                            }
                        }
                    }
                }
                // Per token: the tile's maximum, earlier tiles rescaled to it,
                // weights exp(score - maximum) in place of the scores, their sum.
                for (size_t j = 0; j < count; ++j) {
                    const size_t ut = (g0 + j) * heads + h;
                    float* o = att + ut * vd;
                    float* sc = scores.data() + j * urows;
                    float tile_max = -INFINITY;
                    for (const auto& [i0, i1] : ranges[j]) {
                        for (size_t i = i0; i < i1; ++i) tile_max = std::max(tile_max, sc[i]);
                    }
                    if (tile_max == -INFINITY) continue;  // nothing visible in this tile
                    float& mx = mla_max_[ut];
                    float& sum = mla_sum_[ut];
                    const float next = std::max(mx, tile_max);
                    if (next != mx) {  // rescale what earlier tiles added (nothing yet: 0 x 0)
                        const float f = std::exp(mx - next);
                        sum *= f;
                        for (size_t d = 0; d < vd; ++d) o[d] *= f;
                        mx = next;
                    }
                    for (const auto& [i0, i1] : ranges[j]) {
                        for (size_t i = i0; i < i1; ++i) sum += sc[i] = std::exp(sc[i] - next);
                    }
                }
                // Weighted values, by the same chunks of rows: per block, the
                // rows all of its tokens see go through together, the others per
                // token before and after them (rows ascending for every token).
                const float* values = mla_ex_.data() + heads * qk + h * vd;
                for (size_t c0 = lo; c0 < hi; c0 += kAttendRows) {
                    const size_t c1 = std::min(hi, c0 + kAttendRows);
                    auto clip = [&](size_t i0, size_t i1) {
                        return std::pair<size_t, size_t>{std::clamp(i0, c0, c1), std::clamp(i1, c0, c1)};
                    };
                    for (size_t j0 = 0; j0 < count; j0 += block) {
                        float* o[kScoreQueries];
                        const float* w[kScoreQueries];
                        const size_t in_block = std::min(block, count - j0);
                        size_t common0 = 0, common1 = 0;  // rows every token of the block sees, if no sinks
                        bool together = in_block == block;
                        for (size_t j = 0; j < in_block; ++j) {
                            o[j] = att + ((g0 + j0 + j) * heads + h) * vd;
                            w[j] = scores.data() + (j0 + j) * urows;
                            const auto& rg = ranges[j0 + j];
                            together = together && rg[0].first >= rg[0].second;
                            common0 = j == 0 ? rg[1].first : std::max(common0, rg[1].first);
                            common1 = j == 0 ? rg[1].second : std::min(common1, rg[1].second);
                        }
                        const auto [m0, m1] = clip(common0, common1);
                        if (together && m0 < m1) {
                            for (size_t j = 0; j < block; ++j) {
                                const auto [b0, b1] = clip(ranges[j0 + j][1].first, common0);
                                weigh_values<1>(&o[j], &w[j], values, row, b0, b1, vd);
                            }
                            weigh_values<kScoreQueries>(o, w, values, row, m0, m1, vd);
                            for (size_t j = 0; j < block; ++j) {
                                const auto [e0, e1] = clip(common1, ranges[j0 + j][1].second);
                                weigh_values<1>(&o[j], &w[j], values, row, e0, e1, vd);
                            }
                        } else {
                            for (size_t j = 0; j < in_block; ++j) {
                                for (const auto& [i0, i1] : ranges[j0 + j]) {
                                    const auto [r0, r1] = clip(i0, i1);
                                    weigh_values<1>(&o[j], &w[j], values, row, r0, r1, vd);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    for (size_t i = 0; i < un * heads; ++i) {
        if (mla_sum_[i] == 0.0f) continue;  // a pruned head stays zero
        const float inv = 1.0f / mla_sum_[i];
        float* o = att + i * vd;
        for (size_t d = 0; d < vd; ++d) o[d] *= inv;
    }
    return Status::ok();
}

Status Transformer::latent_attention_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                           ThreadPool& pool, const uint8_t* head_mask) {
    using Clock = std::chrono::steady_clock;
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const LatentAttention& m = c.mla;
    const auto un = static_cast<size_t>(n);
    const auto heads = static_cast<size_t>(c.n_head);
    const auto rd = static_cast<size_t>(c.rope_dim);
    const auto rank = static_cast<size_t>(m.kv_rank);
    const auto nope = static_cast<size_t>(m.nope_dim);
    const auto vd = static_cast<size_t>(m.v_dim);
    const auto kd = static_cast<size_t>(c.head_dim);
    const size_t qk = nope + rd;
    const RopeTable& rope_table = rope_tables_[static_cast<size_t>(L.rope_table)];
    // Each segment attends in its own form (the one its pass alone would
    // take); a pass of a single sequence has one form.
    bool any_absorbed = false, any_expanded = false;
    for (const Segment& s : segs_) (s.expanded ? any_expanded : any_absorbed) = true;

    LIYAB_RETURN_IF_ERROR(latent_project(layer, w, n, route, route.attention, true));
    auto start = Clock::now();
    if (any_absorbed) LIYAB_RETURN_IF_ERROR(latent_absorb(w, n, route));  // every row: the same bits for each
    double form_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    for (const Segment& s : segs_) {
        for (auto t = static_cast<size_t>(s.first); t < static_cast<size_t>(s.first + s.n); ++t) {
            float* k = k_.data() + t * kd;
            const int32_t pos = pos_of(s, static_cast<int32_t>(t));
            if (s.expanded) {
                // Queries stay [q_nope | q_pe] per head; only q_pe turns. Keys are
                // stored scaled as in the absorbed form (the cache is shared).
                rope(mla_q_.data() + t * heads * qk + nope, c.n_head, qk, pos, rope_table);
                rope(k, 1, pos, rope_table);
                scale_keys(L.kv_slot, nullptr, k);
            } else {
                float* q = q_.data() + t * heads * kd;
                if (key_ranges_ != nullptr) add_key_ranges(L, q, k);  // unrotated: ranges cover every position
                rope(q, c.n_head, pos, rope_table);
                rope(k, 1, pos, rope_table);
                scale_keys(L.kv_slot, q, k);
            }
            s.kv->store(L.kv_slot, row_of(s, static_cast<int32_t>(t)), k, nullptr);  // the values are this row's latent channels
        }
    }
    start = Clock::now();
    if (any_absorbed) {
        // att_: per absorbed token and head, the softmax-weighted latent
        // (value_dim = rank), rows `rank` wide per head.
        attention(L.kv_slot, L.window, n, pool, head_mask);

        // Per head, the weighted latent becomes the head's value: W_vb,h c. Keys
        // stored with per-channel scales carried their latent divided by them,
        // so the latent is multiplied back first (exact for each channel).
        // Every absorbed row is read before any is written in the [token]
        // [head][value] layout (narrower: v_dim <= rank), and expanded
        // segments write theirs after this (latent_expanded()).
        const float* scales =
            key_scale_.empty() ? nullptr : key_scale_.data() + static_cast<size_t>(L.kv_slot) * kd + rd;
        const int32_t* listed = any_expanded ? attn_rows_.data() : nullptr;  // the rows attention() attended
        const size_t m = any_expanded ? attn_rows_.size() : un;
        auto row_at = [listed](size_t i) { return listed == nullptr ? i : static_cast<size_t>(listed[i]); };
        mla_heads_in_.resize(heads * m * rank);
        for (size_t i = 0; i < m; ++i) {
            const size_t t = row_at(i);
            for (size_t h = 0; h < heads; ++h) {
                const float* src = att_.data() + (t * heads + h) * rank;
                float* dst = mla_heads_in_.data() + (h * m + i) * rank;
                if (scales == nullptr) {
                    std::copy(src, src + rank, dst);
                } else {
                    for (size_t j = 0; j < rank; ++j) dst[j] = src[j] * scales[j];
                }
            }
        }
        mla_heads_out_.resize(heads * m * vd);
        LIYAB_RETURN_IF_ERROR(per_head_matmul(route, w[kVb], mla_heads_in_.data(), mla_heads_out_.data(),
                                              static_cast<int32_t>(m)));
        for (size_t i = 0; i < m; ++i) {  // [token][head][value], the order attn_output reads
            const size_t t = row_at(i);
            for (size_t h = 0; h < heads; ++h) {
                const float* src = mla_heads_out_.data() + (h * m + i) * vd;
                std::copy(src, src + vd, att_.data() + (t * heads + h) * vd);
            }
        }
    }
    for (const Segment& s : segs_) {
        if (s.expanded) LIYAB_RETURN_IF_ERROR(latent_expanded(layer, w, s, pool, head_mask));
    }
    form_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    latent_ms_ += form_ms;
    return matmul(route, route.attention, w[kO], att_.data(), xb_.data(), n);
}

// Gated DeltaNet (Qwen3.5 / Qwen3-Next linear attention):
//   [q|k|v] = SiLU(causal_conv1d(W_qkv x)),  q, k L2-normalized, q scaled by 1/sqrt(d)
//   decay = exp(softplus(W_a x + dt_bias) * A),  b = sigmoid(W_b x)
//   S = decay * S;  S += b * (v - S k) k^T;  o = S q
//   out = W_out (RMSNorm(o) * SiLU(W_z x))      (per value head)
// Key/query heads are shared by value heads modulo their count (ggml's layout).
// A batch takes two parallel passes whatever its length: the convolution by
// channel heads through all tokens, then the recurrence by value head through
// all tokens (both are independent across their tasks, sequential in time).
// Token by token, a batch of n cost 2 n thread-pool rounds per block and ran
// the convolution, SiLU and L2 norms on one thread.
Status Transformer::delta_net_mixer(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                                    ThreadPool& pool) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto hd = static_cast<size_t>(c.ssm_head_dim);
    const auto hk = static_cast<size_t>(c.ssm_k_heads);
    const auto hv = static_cast<size_t>(c.ssm_v_heads);
    const size_t key_dim = hk * hd;
    const size_t value_dim = hv * hd;
    const size_t conv_dim = 2 * key_dim + value_dim;
    const auto taps = static_cast<size_t>(c.ssm_conv_kernel);
    const size_t history = taps - 1;

    mix_.resize(un * conv_dim);
    z_.resize(un * value_dim);
    alpha_.resize(un * hv);
    beta_.resize(un * hv);
    dn_.resize(un * value_dim);
    {
        const TensorView* proj[] = {&w[kQkv], &w[kZ], &w[kAlpha], &w[kBeta]};
        float* outs[] = {mix_.data(), z_.data(), alpha_.data(), beta_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.attention, proj, xb_.data(), outs, n));
    }

    // Each segment continues its own states (segments_ in order: tasks below
    // are (segment, head) pairs; a segment's token i is batch row first + i).
    // Checkpoints: the state after each of the last `rollback` tokens of the
    // segment, for truncate(); slot = position % rollback, layout [conv | ssm].
    auto state_of = [&](const Segment& s) -> RecurrentState& { return (*s.states)[static_cast<size_t>(L.state_slot)]; };
    auto slot_of = [&](const Segment& s, RecurrentState& st, size_t i) -> float* {
        if (!s.checkpoint || i + static_cast<size_t>(s.rollback) < static_cast<size_t>(s.n)) return nullptr;
        return st.checkpoints.data() + static_cast<size_t>((s.n_past + static_cast<int32_t>(i)) % s.rollback) *
                                           (st.conv.size() + st.ssm.size());
    };

    // Tree pass (one segment): a token whose parent is not the token before
    // it (a branch other than the first) continues from its parent's state,
    // which the parent's checkpoint holds (forward_tree() keeps one for every
    // token).
    const int32_t* parent = tree_depth_.empty() ? nullptr : tree_parent_.data();
    auto branches = [&](size_t i) { return parent != nullptr && i > 0 && parent[i] != static_cast<int32_t>(i) - 1; };
    const auto segments = static_cast<int64_t>(segs_.size());

    // 1. Causal depthwise convolution over [history, current] and SiLU, every
    //    channel through all of a segment's tokens (channels are independent,
    //    tokens are not), then the L2 norm of each query / key head. One task
    //    per segment and head of channels: 2 hk query/key heads, then hv value
    //    heads.
    const auto conv_tasks = static_cast<int64_t>(2 * hk + hv);
    pool.parallel_for(conv_tasks * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / conv_tasks)];
            RecurrentState& st = state_of(s);
            const auto task = static_cast<size_t>(job % conv_tasks);
            const auto first = static_cast<size_t>(s.first);
            const auto count = static_cast<size_t>(s.n);
            const size_t ch0 = task * hd;  // query heads, key heads and value heads are hd channels each
            for (size_t ch = ch0; ch < ch0 + hd; ++ch) {
                float* hist = st.conv.data() + ch * history;
                const float* tap = L.conv1d.data() + ch * taps;
                for (size_t i = 0; i < count; ++i) {
                    if (branches(i)) {
                        const float* from = slot_of(s, st, static_cast<size_t>(parent[i])) + ch * history;
                        std::copy(from, from + history, hist);
                    }
                    float* m = mix_.data() + (first + i) * conv_dim;
                    float acc = m[ch] * tap[history];
                    for (size_t j = 0; j < history; ++j) acc += hist[j] * tap[j];
                    for (size_t j = 0; j + 1 < history; ++j) hist[j] = hist[j + 1];
                    hist[history - 1] = m[ch];
                    m[ch] = silu(acc);
                    if (float* slot = slot_of(s, st, i)) std::copy(hist, hist + history, slot + ch * history);
                }
            }
            if (task < 2 * hk) {
                for (size_t i = 0; i < count; ++i) {
                    l2norm(mix_.data() + (first + i) * conv_dim + ch0, static_cast<int32_t>(hd), c.rms_eps);
                }
            }
        }
    });

    // 2. The recurrence, one task per segment and value head through all of
    //    the segment's tokens (heads are independent), then the head's gated
    //    RMSNorm.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto heads = static_cast<int64_t>(hv);
    pool.parallel_for(heads * segments, [&](int64_t begin, int64_t end) {
        for (int64_t job = begin; job < end; ++job) {
            const Segment& s = segs_[static_cast<size_t>(job / heads)];
            RecurrentState& st = state_of(s);
            const auto h = static_cast<size_t>(job % heads);
            const auto first = static_cast<size_t>(s.first);
            float* S = st.ssm.data() + h * hd * hd;
            for (size_t i = 0; i < static_cast<size_t>(s.n); ++i) {
                if (branches(i)) {
                    const float* from = slot_of(s, st, static_cast<size_t>(parent[i])) + st.conv.size() + h * hd * hd;
                    std::copy(from, from + hd * hd, S);
                }
                const size_t t = first + i;
                const float* m = mix_.data() + t * conv_dim;
                const float* q = m + (h % hk) * hd;
                const float* k = m + key_dim + (h % hk) * hd;
                const float* v = m + 2 * key_dim + h * hd;
                const float decay = std::exp(softplus(alpha_[t * hv + h] + L.ssm_dt[h]) * L.ssm_a[h]);
                const float strength = sigmoid(beta_[t * hv + h]);
                float* o = dn_.data() + t * value_dim + h * hd;
                delta_rule_step(S, q, k, v, decay, strength, q_scale, o, hd);
                const float* z = z_.data() + t * value_dim + h * hd;
                rmsnorm(o, L.ssm_norm.data(), o, static_cast<int32_t>(hd), c.rms_eps);
                for (size_t j = 0; j < hd; ++j) o[j] *= silu(z[j]);
                if (float* slot = slot_of(s, st, i)) std::copy(S, S + hd * hd, slot + st.conv.size() + h * hd * hd);
            }
        }
    });
    return matmul(route, route.attention, w[kSsmOut], dn_.data(), xb_.data(), n);
}

// Expert streaming: the routed experts stay on storage and are read on
// demand into a fixed RAM cache; every other tensor stays resident in the
// mapping (paged in now, never swept by the streaming window).
Status Transformer::attach_expert_store(int64_t budget_option, uint64_t memory_budget, size_t reserve,
                                        int32_t requant_bits, bool repack) {
    if (budget_option == 0) return Status::ok();
    if (requant_bits != -1 && requant_bits != 0 && requant_bits != 4 && requant_bits != 5) {
        return Status(ErrorCode::InvalidArgument, "requant_bits must be -1, 0, 4 or 5");
    }
    const ModelConfig& c = config_;
    auto is_expert = [](const TensorView& t) {
        const std::string_view n = t.name;
        return n.size() > 13 && n.substr(n.size() - 12) == "_exps.weight";
    };
    size_t expert_bytes = 0;
    for (const TensorView& t : file_->tensors()) {
        if (is_expert(t)) expert_bytes += t.nbytes;
    }
    const uint64_t available = usable_memory_bytes(memory_budget);
    // Automatic mode streams only when the model does not fit.
    if (budget_option < 0 &&
        (available == 0 || static_cast<double>(file_->file_size()) <= 0.8 * static_cast<double>(available))) {
        return Status::ok();
    }

    // The expert cache gets what the resident weights leave, after room for
    // the KV cache, activations and the rest of the app: 1 GiB of free RAM,
    // or 512 MiB inside an explicit budget (the embedder already left its own
    // headroom below the platform's cap). Recurrent-state snapshots
    // (snapshot_state) come out of the cache too.
    // Contexts besides the live one (TransformerOptions::extra_contexts)
    // each hold their recurrent states, snapshots and checkpoints too.
    const size_t margin = (memory_budget > 0 ? size_t{512} << 20 : size_t{1} << 30) +
                          static_cast<size_t>(kStateSnapshots) * recurrent_state_bytes() + extra_context_bytes_;
    // The budget came without the reserve (TransformerOptions::reserve_bytes),
    // which the loader set aside when it decided to stream; the cache plans
    // with it and hands back only what it can spare.
    const uint64_t whole = memory_budget > 0 ? usable_memory_bytes(memory_budget + reserve) : available;
    auto cache_budget = [&](size_t resident) -> size_t {
        if (budget_option > 0) return static_cast<size_t>(budget_option);
        return whole > resident + margin ? static_cast<size_t>(whole) - resident - margin : 0;
    };

    // Optional lossy conversion of the resident Q8_0 matrices (the UD quants
    // keep attention and DeltaNet projections in Q8_0, ~2/3 of the bytes read
    // per token): fewer bytes per token, and the freed memory goes to the
    // expert cache. The token embedding is skipped: decode reads one row of
    // it per token. Automatic (-1): only when the cache would hold less than
    // kTightCache of the experts, where hit rates fall off and nearly every
    // token waits for storage; Q5_K if that frees enough, else Q4_K.
    auto convertible = [&](const TensorView& t) {
        return !is_expert(t) && t.type == DType::Q8_0 && t.n_dims == 2 && t.rows() > 1 &&
               t.cols() % quant::kSuperBlock == 0 && t.name != "token_embd.weight";
    };
    const size_t resident_q8 = file_->file_size() - expert_bytes;
    int32_t bits = requant_bits;
    if (bits < 0) {
        bits = 0;
        const auto wanted = static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes));
        if (cache_budget(resident_q8) < wanted) {
            size_t q8_bytes = 0;
            for (const TensorView& t : file_->tensors()) {
                if (convertible(t)) q8_bytes += t.nbytes;
            }
            // Q8_0: 8.5 bits per weight; Q5_K 5.5, Q4_K 4.5.
            const auto saving = [&](double bpw) { return static_cast<size_t>(static_cast<double>(q8_bytes) * (1.0 - bpw / 8.5)); };
            bits = q8_bytes == 0 ? 0 : (cache_budget(resident_q8 - saving(5.5)) >= wanted ? 5 : 4);
            LIYAB_LOG_INFO("memory is tight for this model (expert cache %.2f GiB, %.0f%% of experts): %s",
                           static_cast<double>(cache_budget(resident_q8)) / (1024.0 * 1024.0 * 1024.0),
                           100.0 * static_cast<double>(cache_budget(resident_q8)) / static_cast<double>(expert_bytes),
                           bits == 0 ? "no Q8_0 matrices to convert" : bits == 5 ? "converting resident Q8_0 to Q5_K"
                                                                      : "converting resident Q8_0 to Q4_K");
        }
    }
    size_t saved = 0;
    if (bits > 0) {
        auto requantized = file_->requantize(convertible, bits == 4 ? DType::Q4_K : DType::Q5_K);
        if (!requantized) return requantized.status();
        saved = requantized.value();
        LIYAB_LOG_INFO("resident Q8_0 matrices requantized to %s: %.2f GiB saved", bits == 4 ? "Q4_K" : "Q5_K",
                       static_cast<double>(saved) / (1024.0 * 1024.0 * 1024.0));
    }
    const size_t resident_bytes = resident_q8 - saved;
    // The reserve gets only what the cache can spare above kTightCache, so it
    // never decides the conversion above nor starves the cache; with an
    // explicit cache size the rest of the budget is the embedder's anyway.
    // The cache rounds each slot size class down to whole slots: a few
    // slots' worth above the line keeps it above once rounded.
    size_t largest_expert = 0;
    for (int32_t l = 0; l < c.n_layers; ++l) {
        if (c.blocks[static_cast<size_t>(l)].ffn != FfnKind::Moe) continue;
        const Layer& L = layers_[static_cast<size_t>(l)];
        const size_t matrices = (L.w[kGateExps] != nullptr ? L.w[kGateExps]->nbytes : 0) + L.w[kUpExps]->nbytes +
                                L.w[kDownExps]->nbytes;
        largest_expert = std::max(largest_expert, matrices / static_cast<size_t>(c.n_expert));
    }
    const size_t cache_floor =
        static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes)) + 4 * largest_expert;
    size_t budget = cache_budget(resident_bytes);
    const size_t spare = budget > cache_floor ? budget - cache_floor : 0;
    memory_plan_.reserved_bytes = budget_option > 0 ? reserve : std::min(reserve, spare);
    if (budget_option <= 0) budget -= memory_plan_.reserved_bytes;
    memory_plan_.resident_bytes = resident_bytes;
    memory_plan_.expert_bytes = expert_bytes;
    memory_plan_.requant_bits = bits;
    // What the cache needs for most experts to come from RAM (~85% hits at
    // 12-15% of the experts cached on Qwen3.6-35B-A3B; the share, not the size,
    // is what matters across models).
    memory_plan_.recommended_bytes =
        resident_q8 + static_cast<size_t>(kComfortableCache * static_cast<double>(expert_bytes)) + margin;
    std::vector<std::array<const TensorView*, 3>> experts(static_cast<size_t>(c.n_layers));
    for (int32_t l = 0; l < c.n_layers; ++l) {
        const Layer& L = layers_[static_cast<size_t>(l)];
        if (c.blocks[static_cast<size_t>(l)].ffn == FfnKind::Moe) {
            experts[static_cast<size_t>(l)] = {L.w[kGateExps], L.w[kUpExps], L.w[kDownExps]};
        }
    }
    auto store = ExpertStore::create(*file_, std::move(experts), c.n_expert, budget, 4, repack);
    if (!store) return store.status();
    expert_store_ = std::move(store).value();
    memory_plan_.expert_cache_bytes = expert_store_->capacity_bytes();
    if (memory_plan_.expert_cache_bytes < static_cast<size_t>(kTightCache * static_cast<double>(expert_bytes))) {
        LIYAB_LOG_WARN("expert cache holds %.0f%% of the experts: most tokens will wait for storage; about %.1f GiB of "
                       "memory budget would let this model run well",
                       100.0 * static_cast<double>(memory_plan_.expert_cache_bytes) / static_cast<double>(expert_bytes),
                       static_cast<double>(memory_plan_.recommended_bytes) / (1024.0 * 1024.0 * 1024.0));
    }
    file_->keep_resident([&](const TensorView& t) { return !is_expert(t) && !file_->converted(t); });
    LIYAB_LOG_INFO("expert streaming: %.2f GiB resident weights, %.2f GiB of experts on storage",
                   static_cast<double>(resident_bytes) / (1024.0 * 1024.0 * 1024.0),
                   static_cast<double>(expert_bytes) / (1024.0 * 1024.0 * 1024.0));
    return Status::ok();
}

void Transformer::predict_experts(int32_t layer, int32_t n, const Route& route) {
    if (expert_store_ == nullptr || layer >= config_.n_layers) return;
    const ModelConfig& c = config_;
    if (c.blocks[static_cast<size_t>(layer)].ffn != FfnKind::Moe) return;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const PhaseTimer timer(phases_.router);
    const auto d = static_cast<size_t>(c.n_embd);
    const auto E = static_cast<size_t>(c.n_expert);
    const auto un = static_cast<size_t>(n);
    pred_in_.resize(un * d);
    pred_logits_.resize(un * E);
    for (size_t t = 0; t < un; ++t) {
        rmsnorm(x_.data() + t * d, L.ffn_norm.data(), pred_in_.data() + t * d, c.n_embd, c.rms_eps);
    }
    if (!route.cpu->matmul(*L.w[kRouter], pred_in_.data(), pred_logits_.data(), n).is_ok()) return;
    std::vector<int32_t> order(E);
    std::vector<int32_t> picks;
    std::vector<int32_t> best_rank(E, -1);
    const bool cut = expert_mass_ < 1.0f || (max_experts_ > 0 && max_experts_ < c.n_expert_used);
    std::vector<float> probs(cut ? E : 0);
    for (size_t t = 0; t < un; ++t) {
        const float* logits = pred_logits_.data() + t * E;
        auto score = [&](int32_t e) {  // softmax and sigmoid are monotonic; the selection bias is added on top
            const float p = c.moe_gating == MoeGating::Softmax ? logits[e] : sigmoid(logits[e]);
            return p + (L.expert_bias.empty() ? 0.0f : L.expert_bias[static_cast<size_t>(e)]);
        };
        for (size_t e = 0; e < E; ++e) order[e] = static_cast<int32_t>(e);
        const auto k = static_cast<std::ptrdiff_t>(c.n_expert_used);
        std::partial_sort(order.begin(), order.begin() + k, order.end(),
                          [&](int32_t a, int32_t b) { return score(a) > score(b); });
        size_t kept = static_cast<size_t>(k);
        if (cut) {  // guess only the experts the router would keep (unnormalized: ratios suffice)
            float sum = 0.0f;
            const float top = logits[order[0]];
            for (std::ptrdiff_t i = 0; i < k; ++i) {
                const auto e = static_cast<size_t>(order[static_cast<size_t>(i)]);
                sum += probs[e] = c.moe_gating == MoeGating::Softmax ? std::exp(logits[e] - top) : sigmoid(logits[e]);
            }
            kept = experts_to_run(probs, order, sum);
        }
        for (size_t r = 0; r < kept; ++r) {
            int32_t& best = best_rank[static_cast<size_t>(order[r])];
            if (best < 0 || static_cast<int32_t>(r) < best) best = static_cast<int32_t>(r);
        }
    }
    if (rank_guessed_.size() != static_cast<size_t>(c.n_expert_used)) {
        rank_guessed_.assign(static_cast<size_t>(c.n_expert_used), 0.0);
        rank_used_.assign(static_cast<size_t>(c.n_expert_used), 0.0);
    }
    guesses_.clear();
    std::vector<int32_t> prefetched;
    for (size_t e = 0; e < E; ++e) {
        const int32_t r = best_rank[e];
        if (r < 0) continue;
        guesses_.push_back({static_cast<int32_t>(e), r});
        picks.push_back(static_cast<int32_t>(e));
        // Laplace-smoothed precision of the rank: every rank starts as worth prefetching (2/3).
        const auto ur = static_cast<size_t>(r);
        if ((rank_used_[ur] + 2.0) / (rank_guessed_[ur] + 3.0) > 0.5) prefetched.push_back(static_cast<int32_t>(e));
    }
    expert_store_->prefetch(layer, prefetched, true);
    predicted_ = std::move(picks);  // sorted: built in expert order
    predicted_layer_ = layer;
}

Status Transformer::dense_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route,
                              FfnMatmulHook* hook) {
    const ModelConfig& c = config_;
    const size_t rows = static_cast<size_t>(n) * static_cast<size_t>(c.n_ff);
    hb_.resize(rows);
    hb2_.resize(rows);
    // FFN projections: an experimental override first, else the routed backend.
    auto project = [&](FfnProjection p, const TensorView& m, const float* x, float* y) -> Status {
        if (hook != nullptr && hook->ffn_matmul(layer, p, x, y, n)) {
            // The hook replaced the base matmul only: an adapter's update still applies.
            const TensorView* ws[] = {&m};
            float* ys[] = {y};
            if (!adapters_in_pass_ || !collect_adapter_items(ws, x, ys, n)) return Status::ok();
            return adapter_runner_.apply(*route.cpu, adapter_items_, adapter_pool_);
        }
        return matmul(route, route.ffn, m, x, y, n);
    };
    if (c.blocks[static_cast<size_t>(layer)].fused_gate_up) {
        // One [gate | up] projection per token (no per-projection hook: the
        // halves are one matrix); act(gate) * up lands in hb_.
        const auto ff = static_cast<size_t>(c.n_ff);
        hb2_.resize(2 * rows);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kUp], xb_.data(), hb2_.data(), n));
        for (size_t t = 0; t < static_cast<size_t>(n); ++t) {
            const float* gate_up = hb2_.data() + t * 2 * ff;
            gated_activation(c.spec->ffn_activation, gate_up, gate_up + ff, hb_.data() + t * ff, ff);
        }
        return project(FfnProjection::Down, w[kDown], hb_.data(), xb_.data());
    }
    if (hook != nullptr) {
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Gate, w[kGate], xb_.data(), hb_.data()));
        LIYAB_RETURN_IF_ERROR(project(FfnProjection::Up, w[kUp], xb_.data(), hb2_.data()));
    } else {
        const TensorView* gate_up[] = {&w[kGate], &w[kUp]};
        float* outs[] = {hb_.data(), hb2_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.ffn, gate_up, xb_.data(), outs, n));
    }
    gated_activation(c.spec->ffn_activation, hb_.data(), hb2_.data(), hb_.data(), rows);
    return project(FfnProjection::Down, w[kDown], hb_.data(), xb_.data());
}

size_t Transformer::experts_to_run(std::span<const float> probs, std::span<const int32_t> order, float top_sum) const {
    const auto K = static_cast<size_t>(config_.n_expert_used);
    const size_t cap = max_experts_ > 0 ? std::min(K, static_cast<size_t>(max_experts_)) : K;
    if (expert_mass_ >= 1.0f) return cap;
    float covered = 0.0f;
    for (size_t k = 0; k < cap; ++k) {
        covered += probs[static_cast<size_t>(order[k])];
        if (covered >= expert_mass_ * top_sum) return k + 1;
    }
    return cap;
}

// Mixture of experts, as llama.cpp's build_moe_ffn: router probabilities
// (softmax or sigmoid), top-k selection (optionally on probability + bias),
// optional renormalization and scale; each selected expert is a gated FFN
// (its gate and up projections separate or one [gate | up] matrix).
// Tokens are grouped by expert so a batch reads every expert once. A shared
// expert, optionally scaled by sigmoid(shared_gate . x), is added on top.
Status Transformer::moe_ffn(int32_t layer, const BlockWeights& w, int32_t n, const Route& route) {
    const ModelConfig& c = config_;
    const Layer& L = layers_[static_cast<size_t>(layer)];
    const auto un = static_cast<size_t>(n);
    const auto d = static_cast<size_t>(c.n_embd);
    const auto E = static_cast<size_t>(c.n_expert);
    const auto K = static_cast<size_t>(c.n_expert_used);
    const auto ff = static_cast<size_t>(c.n_ff_expert);
    const bool fused = c.blocks[static_cast<size_t>(layer)].fused_gate_up;  // experts as [gate | up] and down

    PhaseTimer router_timer(phases_.router);
    // A pass of several tokens of one sequence (a prompt, a speculative step)
    // runs each expert over many of them: the store repacks its reads for the
    // batched kernels and does not count those uses. Decoding steps of several
    // sequences (steps_only_) are decoding: they choose few experts in common,
    // so an expert runs over a token or two, a repack would only delay the
    // read a miss waits for, and their uses are what the cache should keep.
    if (expert_store_ != nullptr) expert_store_->set_batch(n > 1 && !steps_only_);
    router_.resize(un * E);
    LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kRouter], xb_.data(), router_.data(), n));

    // Routing: per expert, the tokens that selected it, with the gate weight
    // and the expert's rank in that token's top-k.
    struct Use {
        int32_t token;
        int32_t rank;
        float weight;
    };
    std::vector<std::vector<Use>> assigned(E);
    std::vector<float> probs(E);
    std::vector<int32_t> order(E);
    experts_kept_.assign(un, static_cast<int32_t>(K));
    std::vector<float> token_weight(un, 0.0f);  // per token: the weights of its kept experts (for skip_slow_)
    for (size_t t = 0; t < un; ++t) {
        const float* logits = router_.data() + t * E;
        if (c.moe_gating == MoeGating::Softmax) {
            const float mx = *std::max_element(logits, logits + E);
            float sum = 0.0f;
            for (size_t e = 0; e < E; ++e) sum += probs[e] = std::exp(logits[e] - mx);
            for (float& p : probs) p /= sum;
        } else {
            for (size_t e = 0; e < E; ++e) probs[e] = sigmoid(logits[e]);
        }
        auto selection = [&](int32_t e) {
            return probs[static_cast<size_t>(e)] + (L.expert_bias.empty() ? 0.0f : L.expert_bias[static_cast<size_t>(e)]);
        };
        for (size_t e = 0; e < E; ++e) order[e] = static_cast<int32_t>(e);
        std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(K), order.end(),
                          [&](int32_t a, int32_t b) { return selection(a) > selection(b) || (selection(a) == selection(b) && a < b); });
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) sum += probs[static_cast<size_t>(order[k])];
        const size_t kept = experts_to_run(probs, order, sum);
        experts_kept_[t] = static_cast<int32_t>(kept);
        if (kept < K) {
            sum = 0.0f;
            for (size_t k = 0; k < kept; ++k) sum += probs[static_cast<size_t>(order[k])];
        }
        const float norm = c.moe_norm_weights ? 1.0f / std::max(sum, 6.103515625e-5f) : 1.0f;
        for (size_t k = 0; k < kept; ++k) {
            const auto e = static_cast<size_t>(order[k]);
            assigned[e].push_back(
                {static_cast<int32_t>(t), static_cast<int32_t>(k), probs[e] * norm * c.moe_weights_scale});
            token_weight[t] += assigned[e].back().weight;
        }
    }

    // With expert streaming, queue every chosen expert now (missed
    // predictions jump the queue) and compute cached ones first, so reads
    // overlap the work on the others.
    std::vector<int32_t> chosen;
    for (size_t e = 0; e < E; ++e) {
        if (!assigned[e].empty()) chosen.push_back(static_cast<int32_t>(e));
    }
    if (expert_store_ != nullptr) {
        if (predicted_layer_ == layer) {  // both lists are sorted
            predictions_.predicted += predicted_.size();
            for (size_t i = 0, j = 0; i < predicted_.size() && j < chosen.size();) {
                if (predicted_[i] == chosen[j]) ++predictions_.used;
                if (predicted_[i] <= chosen[j]) ++i; else ++j;
            }
            // Per-rank precision, decayed so it follows the conversation.
            for (const Guess& g : guesses_) {
                const auto r = static_cast<size_t>(g.rank);
                rank_guessed_[r] += 1.0;
                if (std::binary_search(chosen.begin(), chosen.end(), g.expert)) rank_used_[r] += 1.0;
                if (rank_guessed_[r] > 2048.0) {
                    rank_guessed_[r] *= 0.5;
                    rank_used_[r] *= 0.5;
                }
            }
            predicted_layer_ = -1;
        }
        expert_store_->prefetch(layer, chosen, false);
        std::stable_partition(chosen.begin(), chosen.end(),
                              [&](int32_t e) { return expert_store_->ready(layer, e); });
    }
    router_timer.stop();
    // The next block's guess goes behind this block's own reads in the queue.
    if (expert_store_ != nullptr) predict_experts(layer + 1, n, route);

    // The shared expert (when the block has one) runs where the routed ones
    // would wait for a read, or after them; its output is added last either
    // way, so the result does not depend on when it ran.
    double shared_in_experts_ms = 0.0;  // shared-expert time inside the experts phase, moved back to its own
    bool shared_done = L.w[kUpShared] == nullptr;
    auto run_shared = [&]() -> Status {
        const auto t0 = std::chrono::steady_clock::now();
        const auto sh = static_cast<size_t>(w[kUpShared].rows());
        hb_.resize(un * sh);
        hb2_.resize(un * sh);
        const TensorView* gate_up[] = {&w[kGateShared], &w[kUpShared]};
        float* outs[] = {hb_.data(), hb2_.data()};
        LIYAB_RETURN_IF_ERROR(matmul_group(route, route.ffn, gate_up, xb_.data(), outs, n));
        gated_activation(c.spec->ffn_activation, hb_.data(), hb2_.data(), hb_.data(), un * sh);
        shared_out_.resize(un * d);
        LIYAB_RETURN_IF_ERROR(matmul(route, route.ffn, w[kDownShared], hb_.data(), shared_out_.data(), n));
        // The gate scale is applied when the output is added (below), in one
        // multiply-add as before: scaling here first rounded differently.
        shared_scale_.assign(un, 1.0f);
        if (!L.shared_gate.empty()) {
            for (size_t t = 0; t < un; ++t) {
                float dot = 0.0f;
                for (size_t j = 0; j < d; ++j) dot += L.shared_gate[j] * xb_[t * d + j];
                shared_scale_[t] = sigmoid(dot);
            }
        }
        shared_done = true;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        phases_.shared_expert += ms;
        shared_in_experts_ms += ms;
        return Status::ok();
    };

    PhaseTimer experts_timer(phases_.experts);
    moe_out_.assign(un * d, 0.0f);
    std::vector<float> skipped_weight(un, 0.0f);  // per token, the weight of experts skipped (skip_slow_)
    std::vector<uint8_t> rank_skipped(un * K, 0);
    ranked_out_.resize(un * K * d);
    // Experts run in waves: the next expert (waiting for it if needed) plus
    // every following one already in RAM, as one batched pass for gate/up and
    // one for down, so the CPU pays two thread fork/joins per wave instead of
    // two per expert. Expert slices are transient views: they run on the CPU,
    // which reads them in place.
    constexpr size_t kMaxWave = 32;  // bounds the cache slots one wave pins
    std::vector<int32_t> wave;
    std::vector<std::array<TensorView, 3>> views;
    std::vector<size_t> first_row;  // per wave expert: its first row in the wave's scratch
    std::vector<Backend::MatmulItem> items;
    auto release_wave = [&] {
        if (expert_store_ != nullptr) {
            for (const int32_t ex : wave) expert_store_->release(layer, ex);
        }
    };
    // Queues the active adapters' updates of one expert's matmul `item` over
    // its users' rows (in token order), when they adapt the stack `stack`
    // (see set_adapter()): each segment's users with its own adapter (one
    // run for a single sequence).
    auto add_expert_update = [&](const TensorView& stack, int32_t expert, const Backend::MatmulItem& item,
                                 const std::vector<Use>& users) {
        for (size_t u0 = 0; u0 < users.size();) {
            const Segment& s = segs_[segment_of(users[u0].token)];
            size_t u1 = u0 + 1;
            while (u1 < users.size() && users[u1].token < s.first + s.n) ++u1;
            if (s.adapter != nullptr) {
                const auto it = s.adapter->find(stack.name);
                if (it != s.adapter->end() && it->second.target->experts > 0) {
                    const LoraTarget* t = it->second.target;
                    adapter_items_.push_back({t, expert, it->second.scale,
                                              item.x + u0 * static_cast<size_t>(t->a.ne[0]),
                                              item.y + u0 * static_cast<size_t>(t->rows), static_cast<int32_t>(u1 - u0)});
                }
            }
            u0 = u1;
        }
    };
    for (size_t next = 0; next < chosen.size();) {
        wave.clear();
        views.clear();
        first_row.clear();
        size_t rows_total = 0;
        do {
            const int32_t ex = chosen[next++];
            if (expert_store_ != nullptr && skip_slow_ > 0.0f && !expert_store_->ready(layer, ex)) {
                // Not in RAM yet and light for every token using it: skip it
                // rather than wait (lossy; the token's others are renormalized below).
                const auto& users = assigned[static_cast<size_t>(ex)];
                const bool light = std::all_of(users.begin(), users.end(), [&](const Use& u) {
                    return u.weight < skip_slow_ * token_weight[static_cast<size_t>(u.token)];
                });
                if (light) {
                    for (const Use& u : users) {
                        skipped_weight[static_cast<size_t>(u.token)] += u.weight;
                        rank_skipped[static_cast<size_t>(u.token) * K + static_cast<size_t>(u.rank)] = 1;
                    }
                    ++predictions_.skipped;
                    continue;
                }
            }
            if (expert_store_ != nullptr) {
                // About to wait for a read: do the shared expert's work meanwhile.
                if (!shared_done && wave.empty() && !expert_store_->ready(layer, ex)) {
                    if (Status st = run_shared(); !st.is_ok()) return st;
                }
                auto loaded = expert_store_->acquire(layer, ex);
                if (!loaded) {
                    release_wave();
                    return loaded.status();
                }
                views.push_back(*loaded);
            } else {
                views.push_back({fused ? TensorView{} : expert_slice(w[kGateExps], ex), expert_slice(w[kUpExps], ex),
                                 expert_slice(w[kDownExps], ex)});
            }
            wave.push_back(ex);
            first_row.push_back(rows_total);
            rows_total += assigned[static_cast<size_t>(ex)].size();
        } while (next < chosen.size() && wave.size() < kMaxWave &&
                 (expert_store_ == nullptr || expert_store_->ready(layer, chosen[next])));

        if (wave.empty()) continue;  // every expert of this wave was skipped
        // Inputs: an expert every token chose reads xb_ itself (always the
        // case in decode), so all of them share one quantized copy.
        ein_.resize(rows_total * d);
        eh_.resize(rows_total * ff);
        eh2_.resize(rows_total * (fused ? 2 * ff : ff));  // fused: each row's [gate | up]
        eout_.resize(rows_total * d);
        items.clear();
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto& users = assigned[static_cast<size_t>(wave[i])];
            const auto rows = static_cast<int32_t>(users.size());
            const float* in = xb_.data();
            if (users.size() != un) {
                float* gathered = ein_.data() + first_row[i] * d;
                for (size_t u = 0; u < users.size(); ++u) {
                    std::copy_n(xb_.data() + static_cast<size_t>(users[u].token) * d, d, gathered + u * d);
                }
                in = gathered;
            }
            if (fused) {
                items.push_back({&views[i][1], in, eh2_.data() + first_row[i] * 2 * ff, rows});
                continue;
            }
            items.push_back({&views[i][0], in, eh_.data() + first_row[i] * ff, rows});
            items.push_back({&views[i][1], in, eh2_.data() + first_row[i] * ff, rows});
        }
        // Per-expert adapters: expert e's slice for the tokens routed to it,
        // A·x in the wave's own batch.
        adapter_items_.clear();
        if (adapters_in_pass_) {
            for (size_t i = 0; i < wave.size(); ++i) {
                const auto& users = assigned[static_cast<size_t>(wave[i])];
                if (fused) {
                    add_expert_update(w[kUpExps], wave[i], items[i], users);
                } else {
                    add_expert_update(w[kGateExps], wave[i], items[2 * i], users);
                    add_expert_update(w[kUpExps], wave[i], items[2 * i + 1], users);
                }
            }
            adapter_runner_.add_a_matmuls(adapter_items_, items);
        }
        if (Status st = route.cpu->matmul_batch(items); !st.is_ok()) {
            release_wave();
            return st;
        }
        adapter_runner_.add_updates(adapter_items_, adapter_pool_);
        if (fused) {  // the same rows as the split matrices give, so the same bits
            for (size_t r = 0; r < rows_total; ++r) {
                const float* gate_up = eh2_.data() + r * 2 * ff;
                gated_activation(c.spec->ffn_activation, gate_up, gate_up + ff, eh_.data() + r * ff, ff);
            }
        } else {
            gated_activation(c.spec->ffn_activation, eh_.data(), eh2_.data(), eh_.data(), rows_total * ff);
        }
        items.clear();
        for (size_t i = 0; i < wave.size(); ++i) {
            const auto rows = static_cast<int32_t>(assigned[static_cast<size_t>(wave[i])].size());
            items.push_back({&views[i][2], eh_.data() + first_row[i] * ff, eout_.data() + first_row[i] * d, rows});
        }
        adapter_items_.clear();
        if (adapters_in_pass_) {
            for (size_t i = 0; i < wave.size(); ++i) {
                add_expert_update(w[kDownExps], wave[i], items[i], assigned[static_cast<size_t>(wave[i])]);
            }
            adapter_runner_.add_a_matmuls(adapter_items_, items);
        }
        const Status down_status = route.cpu->matmul_batch(items);
        release_wave();  // the outputs are in eout_; the weights are no longer read
        LIYAB_RETURN_IF_ERROR(down_status);
        adapter_runner_.add_updates(adapter_items_, adapter_pool_);
        for (size_t i = 0; i < first_row.size(); ++i) {
            const auto& users = assigned[static_cast<size_t>(wave[i])];
            for (size_t u = 0; u < users.size(); ++u) {
                float* dst = ranked_out_.data() + (static_cast<size_t>(users[u].token) * K + users[u].rank) * d;
                const float* src = eout_.data() + (first_row[i] + u) * d;
                const float weight = users[u].weight;
                for (size_t j = 0; j < d; ++j) dst[j] = weight * src[j];
            }
        }
    }
    // Sum each token's experts in router order. Experts run in whatever order
    // their weights arrive in RAM; adding their outputs in that order made
    // results depend on I/O timing (float addition is not associative, and
    // the int8-quantized activations downstream turn last-bit differences into
    // whole rounding steps: run-to-run perplexity varied by ~10%).
    for (size_t t = 0; t < un; ++t) {
        float* dst = moe_out_.data() + t * d;
        for (size_t k = 0; k < static_cast<size_t>(experts_kept_[t]); ++k) {
            if (rank_skipped[t * K + k]) continue;
            const float* src = ranked_out_.data() + (t * K + k) * d;
            for (size_t j = 0; j < d; ++j) dst[j] += src[j];
        }
        // Skipped experts: the others carry the token's whole weight, as the
        // model's own top-k normalization would have given it.
        if (skipped_weight[t] > 0.0f && c.moe_norm_weights && skipped_weight[t] < token_weight[t]) {
            const float scale = token_weight[t] / (token_weight[t] - skipped_weight[t]);
            for (size_t j = 0; j < d; ++j) dst[j] *= scale;
        }
    }

    experts_timer.stop();
    phases_.experts -= shared_in_experts_ms;

    if (!shared_done) LIYAB_RETURN_IF_ERROR(run_shared());
    if (L.w[kUpShared] != nullptr) {
        for (size_t t = 0; t < un; ++t) {
            const float scale = shared_scale_[t];
            for (size_t j = 0; j < d; ++j) moe_out_[t * d + j] += scale * shared_out_[t * d + j];
        }
    }
    std::copy(moe_out_.begin(), moe_out_.end(), xb_.begin());
    return Status::ok();
}

void Transformer::size_batch(int32_t n) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const size_t q_dim = static_cast<size_t>(c.n_head) * static_cast<size_t>(c.head_dim);
    const size_t kv_dim = static_cast<size_t>(c.n_head_kv) * static_cast<size_t>(c.head_dim);
    const auto un = static_cast<size_t>(n);
    x_.resize(un * d);
    xb_.resize(un * std::max(d, q_dim));
    q_.resize(un * q_dim);
    k_.resize(un * kv_dim);
    v_.resize(un * kv_dim);
    att_.resize(un * q_dim);
}

Status Transformer::embed(std::span<const int32_t> tokens, std::span<float> x) const {
    const ModelConfig& c = config_;
    const auto d = static_cast<size_t>(c.n_embd);
    if (x.size() < tokens.size() * d) return Status(ErrorCode::InvalidArgument, "embed: output buffer too small");
    for (size_t t = 0; t < tokens.size(); ++t) {
        if (tokens[t] < 0 || tokens[t] >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
        quant::dequantize_row(token_embd_->type, token_embd_->row(tokens[t]), x.data() + t * d, c.n_embd);
    }
    if (c.embd_scale != 1.0f) {
        for (size_t i = 0; i < tokens.size() * d; ++i) x[i] *= c.embd_scale;
    }
    return Status::ok();
}

Status Transformer::run_block(int32_t l, const BlockWeights& w, int32_t n, const Route& route, ThreadPool& pool,
                              const HeadMaskHook* head_mask, FfnSkipHook* ffn_skip, FfnMatmulHook* ffn_hook) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const auto un = static_cast<size_t>(n);
    const Layer& L = layers_[static_cast<size_t>(l)];
    const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
    const bool moe = plan.ffn == FfnKind::Moe;
    const Route cpu_route{route.cpu, route.cpu, route.cpu};
    const Route& block_route = w.streamed() ? cpu_route : route;
    // Norm after the mixer or the FFN (ArchSpec::post_mixer_norm / post_ffn_norm), in place on xb_.
    auto post_norm = [&](const std::vector<float>& weight) {
        if (weight.empty()) return;
        for (size_t t = 0; t < un; ++t) {
            float* row = xb_.data() + t * d;
            rmsnorm(row, weight.data(), row, c.n_embd, c.rms_eps);
        }
    };

    // --- expert prefetch for the first block (the others are predicted
    // one block ahead, see predict_experts) ---
    if (expert_store_ != nullptr && l == 0) predict_experts(0, n, route);

    // --- token mixer (attention or DeltaNet) ---
    if (ffn_skip != nullptr) x_block_in_.assign(x_.begin(), x_.end());
    for (size_t t = 0; t < un; ++t) {
        rmsnorm(x_.data() + t * d, L.attn_norm.data(), xb_.data() + t * d, c.n_embd, c.rms_eps);
    }
    if (plan.mixer == MixerKind::DeltaNet) {
        const PhaseTimer timer(phases_.delta_net);
        LIYAB_RETURN_IF_ERROR(delta_net_mixer(l, w, n, block_route, pool));
    } else if (plan.latent) {
        const PhaseTimer timer(phases_.attention);
        LIYAB_RETURN_IF_ERROR(latent_attention_mixer(l, w, n, block_route, pool,
                                                     head_mask != nullptr ? head_mask->mask(l) : nullptr));
    } else {
        const PhaseTimer timer(phases_.attention);
        LIYAB_RETURN_IF_ERROR(
            attention_mixer(l, w, n, block_route, pool, head_mask != nullptr ? head_mask->mask(l) : nullptr));
    }
    post_norm(L.post_mixer_norm);
    for (size_t i = 0; i < un * d; ++i) x_[i] += xb_[i];
    // A MoE block predicts the next one once its own experts are queued (moe_ffn).
    if (expert_store_ != nullptr && !moe) predict_experts(l + 1, n, route);

    // --- FFN block: dense SwiGLU or mixture of experts (skippable by an experimental hook) ---
    if (ffn_skip != nullptr && ffn_skip->skip_ffn(l, c.n_layers, x_block_in_, x_)) {
        ++last_ffn_skips_;
        if (expert_store_ != nullptr && moe) predict_experts(l + 1, n, route);
        return Status::ok();
    }
    for (size_t t = 0; t < un; ++t) {
        rmsnorm(x_.data() + t * d, L.ffn_norm.data(), xb_.data() + t * d, c.n_embd, c.rms_eps);
    }
    if (moe) {
        LIYAB_RETURN_IF_ERROR(moe_ffn(l, w, n, block_route));
    } else {
        const PhaseTimer timer(phases_.dense_ffn);
        LIYAB_RETURN_IF_ERROR(dense_ffn(l, w, n, block_route, ffn_hook));
    }
    post_norm(L.post_ffn_norm);
    for (size_t i = 0; i < un * d; ++i) x_[i] += xb_[i];
    return Status::ok();
}

Status Transformer::run_block_at(int32_t layer, std::span<const int32_t> positions, std::span<float> x,
                                 int32_t chunk, const Route& route, ThreadPool& pool) {
    LIYAB_RETURN_IF_ERROR(reuse_support());
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    if (layer < 0 || layer >= c.n_layers || chunk < 1 || x.size() != positions.size() * d) {
        return Status(ErrorCode::InvalidArgument, "run_block_at: the block, the pass size or the rows do not fit");
    }
    for (size_t i = 0; i < positions.size(); ++i) {
        if (positions[i] < 0 || positions[i] >= n_past_ || (i > 0 && positions[i] <= positions[i - 1])) {
            return Status(ErrorCode::InvalidArgument, "run_block_at: positions must ascend inside the context");
        }
    }
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    if (positions.empty()) return Status::ok();
    // Acquired once for every pass: a layer source streams blocks in order.
    const BlockWeights w(*this, layer);
    LIYAB_RETURN_IF_ERROR(w.status());
    predicted_layer_ = -1;  // a guess made for another batch is no use to these tokens
    // Scattered positions attend in the absorbed form: the expanded form's tiles
    // assume a contiguous pass (latent_visible), and forward() may have left it on.
    latent_expanded_ = false;
    Status status;
    for (size_t i = 0; i < positions.size() && status.is_ok(); i += static_cast<size_t>(chunk)) {
        const size_t m = std::min(static_cast<size_t>(chunk), positions.size() - i);
        size_batch(static_cast<int32_t>(m));
        std::copy_n(x.data() + i * d, m * d, x_.data());
        positions_ = positions.subspan(i, m);
        begin_live_pass(static_cast<int32_t>(m), Logits::None);
        status = run_block(layer, w, static_cast<int32_t>(m), route, pool, nullptr, nullptr, nullptr);
        positions_ = {};
        if (status.is_ok()) std::copy_n(x_.data(), m * d, x.data() + i * d);
    }
    return status;
}

Status Transformer::reuse_support() const {
    const ModelConfig& c = config_;
    if (c.hybrid()) {
        return Status(ErrorCode::Unsupported,
                      "hybrid model: its recurrent (DeltaNet) blocks must run over the whole prompt in order, so "
                      "every token would still go through every block; reusing the attention blocks' keys and "
                      "values would only save their key and value projections");
    }
    if (kv_->window() > 0) {
        return Status(ErrorCode::Unsupported, "a sliding-window cache releases the positions it no longer shows");
    }
    if (kv_->has_windowed_layers()) {
        return Status(ErrorCode::Unsupported,
                      "windowed attention blocks keep only their window's rows, so a segment's rows of those blocks "
                      "are gone once the context moves past them (and recomputing them alone would need every "
                      "block before them over the whole segment)");
    }
    if (c.spec->qk_norm == QkNormOrder::AfterRope) {
        for (const Layer& L : layers_) {
            for (size_t i = 0; i < L.k_norm.size() && i < static_cast<size_t>(c.rope_dim); ++i) {
                if (L.k_norm[i] == 0.0f) {
                    return Status(ErrorCode::Unsupported,
                                  "a key norm after RoPE zeroes a rotated channel, so stored keys cannot be turned");
                }
            }
        }
    }
    return Status::ok();
}

Status Transformer::swap_context(std::unique_ptr<KvCache>& kv, int32_t& n_past) {
    if (config_.hybrid() || !states_.empty()) {
        return Status(ErrorCode::Unsupported, "swap_context: recurrent states are part of the context");
    }
    if (kv_->has_windowed_layers()) {
        return Status(ErrorCode::Unsupported, "swap_context: the windowed blocks' snapshots are part of the context");
    }
    if (!kv) return Status(ErrorCode::InvalidArgument, "swap_context: no cache");
    const KvCacheConfig& a = kv_->config();
    const KvCacheConfig& b = kv->config();
    if (a.n_layers != b.n_layers || a.n_head_kv != b.n_head_kv || a.head_dim != b.head_dim || a.type != b.type ||
        a.page_tokens != b.page_tokens || a.values_in_keys != b.values_in_keys || a.value_offset != b.value_offset ||
        a.window != b.window || n_past < 0 || n_past > b.max_positions) {
        return Status(ErrorCode::InvalidArgument, "swap_context: a cache of another layout");
    }
    std::swap(kv_, kv);
    std::swap(n_past_, n_past);
    return Status::ok();
}

Result<std::unique_ptr<KvCache>> Transformer::make_kv_cache(int32_t positions) const {
    KvCacheConfig c = kv_->config();
    c.max_positions = std::max(positions, 1);
    return KvCache::create(c);
}

Status Transformer::move_keys(uint8_t* image, int32_t slots, int32_t first, int32_t count, int32_t from, int32_t to,
                              ThreadPool* pool) const {
    if (slots < 1 || first < 0 || count < 0 || first + count > slots || from < 0 || to < 0) {
        return Status(ErrorCode::InvalidArgument, "move_keys: the slots or positions do not fit");
    }
    LIYAB_RETURN_IF_ERROR(reuse_support());
    const ModelConfig& c = config_;
    // Nothing turns: every row keeps its bits.
    if (count == 0 || from == to || c.rope_dim == 0) return Status::ok();
    const DType type = kv_->dtype();
    const int32_t half = c.rope_dim / 2;
    const bool halves = c.spec->rope == RopeLayout::Halves;
    const auto hd = static_cast<size_t>(c.head_dim);
    // The leading channels a move touches: the rotated ones, in whole blocks
    // of a quantized row (the blocks after them keep their bits).
    const int32_t block = quant::is_block_quantized(type) ? quant::kBlock : 1;
    const int32_t touched = std::min(c.head_dim, (c.rope_dim + block - 1) / block * block);
    const bool norm_after = c.spec->qk_norm == QkNormOrder::AfterRope;

    // Each table's turns, back from the stored positions and on to the new
    // ones, computed as rope() computes them (the same float operations).
    struct Turns {
        std::vector<float> cos_from, sin_from, cos_to, sin_to;  // [position][pair]
    };
    std::vector<Turns> turns(rope_tables_.size());
    auto turns_of = [&](size_t table) -> const Turns& {
        Turns& t = turns[table];
        if (!t.cos_from.empty() || half == 0) return t;
        const RopeTable& rt = rope_tables_[table];
        const auto n = static_cast<size_t>(count) * static_cast<size_t>(half);
        t.cos_from.resize(n);
        t.sin_from.resize(n);
        t.cos_to.resize(n);
        t.sin_to.resize(n);
        for (int32_t i = 0; i < count; ++i) {
            const float pf = static_cast<float>(from + i) / rt.scale;
            const float pt = static_cast<float>(to + i) / rt.scale;
            for (int32_t j = 0; j < half; ++j) {
                const size_t at = static_cast<size_t>(i) * static_cast<size_t>(half) + static_cast<size_t>(j);
                const float tf = pf * rt.inv_freq[static_cast<size_t>(j)];
                const float tt = pt * rt.inv_freq[static_cast<size_t>(j)];
                t.cos_from[at] = std::cos(tf);
                t.sin_from[at] = std::sin(tf);
                t.cos_to[at] = std::cos(tt);
                t.sin_to[at] = std::sin(tt);
            }
        }
        return t;
    };
    // Per attention block: its table and, per KV head and touched channel,
    // what the stored value is multiplied by to get the rotated one (the key
    // scale over a norm-after-RoPE weight) and back.
    struct Slot {
        const Turns* turns = nullptr;
        std::vector<float> unscale, rescale;  // [kv_head][touched channel]
    };
    std::vector<Slot> blocks;
    for (const Layer& L : layers_) {
        if (L.kv_slot < 0) continue;
        if (static_cast<size_t>(L.kv_slot) >= blocks.size()) blocks.resize(static_cast<size_t>(L.kv_slot) + 1);
        Slot& s = blocks[static_cast<size_t>(L.kv_slot)];
        s.turns = &turns_of(static_cast<size_t>(L.rope_table));
        s.unscale.assign(static_cast<size_t>(c.n_head_kv) * static_cast<size_t>(touched), 1.0f);
        s.rescale.assign(s.unscale.size(), 1.0f);
        for (size_t g = 0; g < static_cast<size_t>(c.n_head_kv); ++g) {
            for (size_t i = 0; i < static_cast<size_t>(touched); ++i) {
                float f = key_scale_.empty() ? 1.0f : key_scale_[(static_cast<size_t>(L.kv_slot) * static_cast<size_t>(c.n_head_kv) + g) * hd + i];
                if (norm_after && !L.k_norm.empty()) f /= L.k_norm[i];
                s.unscale[g * static_cast<size_t>(touched) + i] = f;
                s.rescale[g * static_cast<size_t>(touched) + i] = 1.0f / f;
            }
        }
    }
    const auto n_blocks = static_cast<int64_t>(blocks.size());
    auto move = [&](int64_t begin, int64_t end) {
        thread_local std::vector<float> row;
        row.resize(static_cast<size_t>(touched));
        for (int64_t task = begin; task < end; ++task) {
            const auto slot = static_cast<int32_t>(task % n_blocks);
            const auto i = static_cast<int32_t>(task / n_blocks);
            const Slot& s = blocks[static_cast<size_t>(slot)];
            if (s.turns == nullptr) continue;
            const size_t at = static_cast<size_t>(i) * static_cast<size_t>(half);
            for (int32_t g = 0; g < c.n_head_kv; ++g) {
                uint8_t* r = image + kv_->image_key_offset(slots, slot, first + i, g);
                quant::dequantize_row(type, r, row.data(), touched);
                const float* un = s.unscale.data() + static_cast<size_t>(g) * static_cast<size_t>(touched);
                const float* re = s.rescale.data() + static_cast<size_t>(g) * static_cast<size_t>(touched);
                for (int32_t ch = 0; ch < c.rope_dim; ++ch) row[static_cast<size_t>(ch)] *= un[ch];
                for (int32_t j = 0; j < half; ++j) {
                    const int32_t a = halves ? j : 2 * j;
                    const int32_t b = halves ? j + half : 2 * j + 1;
                    const size_t k = at + static_cast<size_t>(j);
                    const float r0 = row[static_cast<size_t>(a)];
                    const float r1 = row[static_cast<size_t>(b)];
                    // Back by the stored position's angle, then on by the new one's.
                    const float x0 = r0 * s.turns->cos_from[k] + r1 * s.turns->sin_from[k];
                    const float x1 = r1 * s.turns->cos_from[k] - r0 * s.turns->sin_from[k];
                    row[static_cast<size_t>(a)] = x0 * s.turns->cos_to[k] - x1 * s.turns->sin_to[k];
                    row[static_cast<size_t>(b)] = x0 * s.turns->sin_to[k] + x1 * s.turns->cos_to[k];
                }
                for (int32_t ch = 0; ch < c.rope_dim; ++ch) row[static_cast<size_t>(ch)] *= re[ch];
                quant::quantize_row(type, row.data(), r, touched);
            }
        }
    };
    const int64_t tasks = int64_t{count} * n_blocks;
    if (pool != nullptr && tasks >= 64) {
        pool->parallel_for(tasks, move);
    } else {
        move(0, tasks);
    }
    return Status::ok();
}

Result<std::span<const float>> Transformer::forward(std::span<const int32_t> tokens, Logits logits,
                                                    const Route& route, ThreadPool& pool, const ForwardHooks* hooks) {
    const ModelConfig& c = config_;
    const auto n = static_cast<int32_t>(tokens.size());
    last_exit_layer_ = -1;
    predicted_layer_ = -1;  // an early exit can leave a guess for a block that never ran
    if (n == 0) return Status(ErrorCode::InvalidArgument, "forward() needs at least one token");
    if (tree_base_ >= 0) return Status(ErrorCode::InvalidArgument, "a tree pass awaits keep_path()");
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    for (const int32_t id : tokens) {
        if (id < 0 || id >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
    }
    LIYAB_RETURN_IF_ERROR(kv_->reserve(n_past_ + n));  // maps KV pages; ContextFull past the limit
    // The DeltaNet mixers overwrite these rollback slots; they become valid
    // only once every block has run (a failed pass leaves no half-written checkpoint).
    const int32_t recorded = checkpointing_ ? std::min(n, rollback_window_) : 0;
    for (int32_t i = 0; i < recorded; ++i) {
        checkpoint_pos_[static_cast<size_t>((n_past_ + n - 1 - i) % rollback_window_)] = -1;
    }
    // Early exit would leave the skipped blocks' recurrent states behind: attention-only stacks.
    EarlyExitHook* early_exit =
        hooks != nullptr && n == 1 && logits == Logits::Last && !c.hybrid() ? hooks->early_exit : nullptr;
    const HeadMaskHook* head_mask = hooks != nullptr ? hooks->head_mask : nullptr;
    const std::atomic<bool>* cancel = hooks != nullptr ? hooks->cancel : nullptr;
    FfnSkipHook* ffn_skip = hooks != nullptr && n == 1 ? hooks->ffn_skip : nullptr;
    FfnMatmulHook* ffn_hook = hooks != nullptr ? hooks->ffn_matmul : nullptr;
    last_ffn_skips_ = 0;

    // Latent attention: one form for the whole pass (LatentForm). A single
    // token (decoding) is always absorbed, and so is key calibration (its
    // ranges are measured on absorbed queries); otherwise the pinned form, or
    // the one the measured costs predict faster for this batch and context.
    latent_expanded_ = false;
    latent_ms_ = 0.0;
    LatentFormChooser::Work latent_work;
    // A tree pass is absorbed too: its tokens are decoding steps of their
    // branches, and the expanded form's tiles assume a contiguous pass.
    const bool latent = c.mla.kv_rank > 0 && n >= 2 && tree_depth_.empty();
    if (latent) {
        int32_t window = 0;
        for (int32_t l = 0; l < c.n_layers; ++l) {
            if (c.blocks[static_cast<size_t>(l)].latent) {
                window = layers_[static_cast<size_t>(l)].window;
                break;
            }
        }
        const int64_t pairs = latent_visible(live_segment(0, n, logits), window);
        int64_t positions = 0;
        for (const auto& [a, b] : mla_union_) positions += b - a;
        latent_work = LatentFormChooser::work(c, n, pairs, positions);
        if (key_ranges_ == nullptr) {
            latent_expanded_ = latent_form_ == LatentForm::Expanded ||
                               (latent_form_ == LatentForm::Auto && latent_chooser_.expand(latent_work));
        }
    }
    const auto un = static_cast<size_t>(n);
    size_batch(n);
    LIYAB_RETURN_IF_ERROR(embed(tokens, std::span<float>(x_.data(), un * static_cast<size_t>(c.n_embd))));
    begin_live_pass(n, logits);

    for (int32_t l = 0; l < c.n_layers; ++l) {
        if (cancel != nullptr && cancel->load(std::memory_order_relaxed)) {
            return Status(ErrorCode::Cancelled, "forward pass cancelled");
        }
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        LIYAB_RETURN_IF_ERROR(run_block(l, w, n, route, pool, head_mask, ffn_skip, ffn_hook));

        // --- early-exit probe (experimental) ---
        if (early_exit != nullptr && early_exit->probe_after(l, c.n_layers)) {
            LIYAB_RETURN_IF_ERROR(compute_logits(route, 0, 1, pool));
            if (early_exit->should_exit(std::span<const float>(logits_.data(), logits_.size()))) {
                LIYAB_RETURN_IF_ERROR(propagate_kv(route, l + 1, n_past_));
                last_exit_layer_ = l;
                n_past_ += 1;
                kv_->release_unreachable(n_past_, max_batch_);
                return std::span<const float>(logits_.data(), logits_.size());
            }
        }
    }

    // A tree pass's checkpoints hold its tokens' states, not positions':
    // keep_path() marks those of the kept branch.
    for (int32_t i = 0; i < recorded && tree_depth_.empty(); ++i) {
        const int32_t pos = n_past_ + n - 1 - i;
        checkpoint_pos_[static_cast<size_t>(pos % rollback_window_)] = pos;
    }
    if (c.mla.kv_rank > 0) ++(latent_expanded_ ? latent_stats_.expanded_passes : latent_stats_.absorbed_passes);
    n_past_ += n;
    kv_->release_unreachable(n_past_, max_batch_);  // recycle pages that left the window
    if (expert_store_ != nullptr && ++decode_steps_ % 16 == 0) expert_store_->age();
    if (logits == Logits::None) return std::span<const float>();

    // --- output head ---
    const int32_t out_rows = logits == Logits::All ? n : 1;
    if (Status s = compute_logits(route, logits == Logits::All ? 0 : un - 1, out_rows, pool); !s.is_ok()) {
        n_past_ -= n;
        return s;
    }
    return std::span<const float>(logits_.data(), logits_.size());
}

Result<std::span<const float>> Transformer::forward_tree(std::span<const int32_t> tokens,
                                                         std::span<const int32_t> parents, const Route& route,
                                                         ThreadPool& pool) {
    const auto n = static_cast<int32_t>(tokens.size());
    if (tree_base_ >= 0) return Status(ErrorCode::InvalidArgument, "a tree pass awaits keep_path()");
    if (n == 0 || parents.size() != tokens.size() || parents[0] != -1) {
        return Status(ErrorCode::InvalidArgument, "forward_tree: a tree starts with one root token");
    }
    if (n > max_batch_) return Status(ErrorCode::InvalidArgument, "forward_tree: more tokens than max_batch()");
    if (config_.hybrid() && n > rollback_window_) {
        return Status(ErrorCode::Unsupported, "forward_tree: the rollback window cannot hold every token's state");
    }
    const auto un = static_cast<size_t>(n);
    tree_depth_.assign(un, 0);
    for (size_t t = 1; t < un; ++t) {
        const int32_t p = parents[t];
        if (p < 0 || p >= static_cast<int32_t>(t)) {
            tree_depth_.clear();
            return Status(ErrorCode::InvalidArgument, "forward_tree: a token's parent must come before it");
        }
        tree_depth_[t] = tree_depth_[static_cast<size_t>(p)] + 1;
    }
    tree_parent_.assign(parents.begin(), parents.end());
    tree_stride_ = *std::max_element(tree_depth_.begin(), tree_depth_.end()) + 1;
    tree_ancestors_.assign(un * static_cast<size_t>(tree_stride_), 0);
    for (size_t t = 0; t < un; ++t) {
        int32_t* row = tree_ancestors_.data() + t * static_cast<size_t>(tree_stride_);
        for (auto a = static_cast<int32_t>(t); a >= 0; a = tree_parent_[static_cast<size_t>(a)]) {
            row[tree_depth_[static_cast<size_t>(a)]] = a;
        }
    }
    // Every token's recurrent state is checkpointed: a later branch starts
    // from one, and keep_path() keeps the branch's.
    const bool checkpointing = checkpointing_;
    checkpointing_ = true;
    auto logits = forward(tokens, Logits::All, route, pool);
    checkpointing_ = checkpointing;
    tree_depth_.clear();
    if (!logits) {
        tree_parent_.clear();
        return logits.status();
    }
    tree_base_ = n_past_ - n;
    return logits;
}

Status Transformer::keep_path(std::span<const int32_t> path) {
    if (tree_base_ < 0) return Status(ErrorCode::InvalidArgument, "keep_path: no tree pass is pending");
    if (path.empty() || path[0] != 0) return Status(ErrorCode::InvalidArgument, "keep_path: a branch starts at the root");
    for (size_t i = 1; i < path.size(); ++i) {
        if (path[i] <= 0 || path[i] >= static_cast<int32_t>(tree_parent_.size()) ||
            tree_parent_[static_cast<size_t>(path[i])] != path[i - 1]) {
            return Status(ErrorCode::InvalidArgument, "keep_path: not a branch of the tree");
        }
    }
    const int32_t base = tree_base_;
    // Ascending: token path[i] >= i of a branch is never the target of an
    // earlier move (path[j] >= j > i for every later j), so no row is
    // overwritten before it is read; the moves within the n <= window ring
    // slots below are distinct for the same reason.
    for (size_t i = 1; i < path.size(); ++i) {
        const int32_t to = base + static_cast<int32_t>(i);
        LIYAB_RETURN_IF_ERROR(kv_->copy_position(base + path[i], to));
        if (states_.empty() || rollback_window_ == 0) continue;
        const auto from_slot = static_cast<size_t>((base + path[i]) % rollback_window_);
        const auto to_slot = static_cast<size_t>(to % rollback_window_);
        for (RecurrentState& st : states_) {
            const size_t per_state = st.conv.size() + st.ssm.size();
            if (from_slot != to_slot) {
                std::copy_n(st.checkpoints.data() + from_slot * per_state, per_state,
                            st.checkpoints.data() + to_slot * per_state);
            }
        }
    }
    for (size_t i = 0; i < path.size() && rollback_window_ > 0; ++i) {
        const int32_t pos = base + static_cast<int32_t>(i);
        checkpoint_pos_[static_cast<size_t>(pos % rollback_window_)] = pos;
    }
    tree_base_ = -1;
    tree_parent_.clear();
    // The recurrent states after path.back() come back from its checkpoint
    // (none to restore when the branch is every token: they are current).
    return truncate(base + static_cast<int32_t>(path.size()));
}

Status Transformer::adopt_context(int32_t n, std::vector<OwnedSnapshot> snapshots, bool pin_end) {
    auto fail = [this](const char* why) {
        reset();
        return Status(ErrorCode::InvalidArgument, why);
    };
    const size_t floats = recurrent_state_bytes() / sizeof(float);
    tree_base_ = -1;  // the context is replaced: a pending tree pass with it
    tree_parent_.clear();
    if (n < 0 || n > context_length_ || kv_->window() > 0) return fail("the context does not fit this cache");
    if (static_cast<int32_t>(snapshots.size()) >= kStateSnapshots || (!has_position_state() && !snapshots.empty())) {
        return fail("the context's recurrent states do not fit this model");
    }
    int32_t below = 0;
    for (const OwnedSnapshot& snap : snapshots) {
        if (snap.pos <= below || snap.pos >= n || snap.data.size() != floats) {
            return fail("the context's state snapshots do not fit this model");
        }
        below = snap.pos;
    }
    for (int32_t p = 0; p * kv_->config().page_tokens < n; ++p) {
        if (!kv_->page_mapped(p)) return fail("the context's cache pages are not mapped");
    }
    // Windowed blocks: the caller wrote the rows of the window before n and
    // before each snapshot (KvCache::write_window_rows()); a context missing
    // them would attend over zeros.
    if (kv_->has_windowed_layers()) {
        if (n > 0 && kv_->restorable(n) != n) return fail("the context's window rows are not in the cache");
        for (const OwnedSnapshot& snap : snapshots) {
            if (kv_->restorable(snap.pos) != snap.pos) return fail("a snapshot's window rows are not in the cache");
        }
    }
    if (n == 0) {
        reset();
        return Status::ok();
    }
    if (Status s = kv_->truncate(n); !s.is_ok()) return fail("cannot truncate the cache");
    std::fill(checkpoint_pos_.begin(), checkpoint_pos_.end(), -1);
    // A restore normally starts from begin_restore(), which moved the
    // snapshots' buffers to spare_buffers_; any left here are kept the same way.
    for (StateSnapshot& snap : snapshots_) spare_buffers_.push_back(std::move(snap.data));
    snapshots_.clear();
    for (OwnedSnapshot& snap : snapshots) {
        snapshots_.push_back(StateSnapshot{snap.pos, std::move(snap.data), snap.pinned});
    }
    n_past_ = n;
    snapshot_state(pin_end);  // in a kept buffer when one is left; pins the snapshots' windows
    spare_buffers_.clear();   // the others: never more memory than before the restore
    return Status::ok();
}

Status Transformer::abandon_pass() {
    if (!config_.hybrid()) return kv_->truncate(n_past_);
    const auto snap = std::find_if(snapshots_.begin(), snapshots_.end(),
                                   [&](const StateSnapshot& s) { return s.pos == n_past_; });
    if (snap == snapshots_.end()) {
        if (n_past_ == 0) {
            reset();
            return Status::ok();
        }
        return Status(ErrorCode::Unsupported, "no state snapshot at the abandoned pass's start");
    }
    LIYAB_RETURN_IF_ERROR(kv_->truncate(n_past_));
    const float* saved = snap->data.data();
    for (RecurrentState& st : states_) {
        std::copy_n(saved, st.conv.size(), st.conv.begin());
        std::copy_n(saved + st.conv.size(), st.ssm.size(), st.ssm.begin());
        saved += st.conv.size() + st.ssm.size();
    }
    return Status::ok();
}

Status Transformer::truncate(int32_t n) {
    if (n < 0 || n > n_past_) return Status(ErrorCode::InvalidArgument, "truncate beyond cached positions");
    if (tree_base_ >= 0) {  // a pending tree's slots are no context: only what preceded it stays
        if (n > tree_base_) return Status(ErrorCode::InvalidArgument, "truncate into a pending tree pass");
        tree_base_ = -1;
        tree_parent_.clear();
    }
    if (n == n_past_) return Status::ok();
    if (n == 0) {
        reset();
        return Status::ok();
    }
    if (config_.hybrid()) {
        const int32_t slot = rollback_window_ > 0 ? (n - 1) % rollback_window_ : -1;
        const bool in_ring = slot >= 0 && checkpoint_pos_[static_cast<size_t>(slot)] == n - 1;
        const auto snap = std::find_if(snapshots_.begin(), snapshots_.end(),
                                       [n](const StateSnapshot& s) { return s.pos == n; });
        if (!in_ring && snap == snapshots_.end()) {
            return Status(ErrorCode::Unsupported,
                          "recurrent (DeltaNet) states can only be rolled back inside the rollback window or to a "
                          "snapshot");
        }
        LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
        const float* saved = in_ring ? nullptr : snap->data.data();
        for (RecurrentState& st : states_) {
            if (in_ring) saved = st.checkpoints.data() + static_cast<size_t>(slot) * (st.conv.size() + st.ssm.size());
            std::copy_n(saved, st.conv.size(), st.conv.begin());
            std::copy_n(saved + st.conv.size(), st.ssm.size(), st.ssm.begin());
            if (!in_ring) saved += st.conv.size() + st.ssm.size();
        }
        // Checkpoints and snapshots of the discarded positions describe a
        // future that no longer exists.
        for (int32_t& p : checkpoint_pos_) {
            if (p >= n) p = -1;
        }
        std::erase_if(snapshots_, [n](const StateSnapshot& s) { return s.pos > n; });
        pin_snapshots();
        n_past_ = n;
        return Status::ok();
    }
    // Windowed blocks: refused when their rows of the window before n were
    // released (KvCache::truncate()); their snapshots past n go.
    LIYAB_RETURN_IF_ERROR(kv_->truncate(n));
    if (!snapshots_.empty() && snapshots_.back().pos > n) {
        std::erase_if(snapshots_, [n](const StateSnapshot& s) { return s.pos > n; });
        pin_snapshots();
    }
    n_past_ = n;
    return Status::ok();
}

// ---------------------------------------------------------------------------
// Several sequences
// ---------------------------------------------------------------------------
Transformer::Segment Transformer::live_segment(int32_t first, int32_t n, Logits logits) noexcept {
    Segment s;
    s.first = first;
    s.n = n;
    s.kv = kv_.get();
    s.n_past = n_past_;
    s.states = &states_;
    s.rollback = rollback_window_;
    s.checkpoint = rollback_window_ > 0 && checkpointing_;
    s.adapter = adapter_targets_.empty() ? nullptr : &adapter_targets_;
    s.expanded = latent_expanded_;
    s.logits = logits;
    return s;
}

void Transformer::begin_live_pass(int32_t n, Logits logits) noexcept {
    segs_.assign(1, live_segment(0, n, logits));  // reuses the vector's storage: no allocation per pass
    seg_of_.clear();
    adapter_rows_.clear();
    adapters_in_pass_ = segs_.front().adapter != nullptr;
}

Result<std::unique_ptr<Transformer::Context>> Transformer::make_context() const {
    if (!kv_) return Status(ErrorCode::InvalidArgument, "make_context: no live context to lay a new one out like");
    auto kv = KvCache::create(kv_->config());
    if (!kv) return kv.status();
    auto context = std::make_unique<Context>();
    context->kv_ = std::move(kv).value();
    context->states_.resize(states_.size());
    for (size_t i = 0; i < states_.size(); ++i) {
        RecurrentState& st = context->states_[i];
        st.conv.assign(states_[i].conv.size(), 0.0f);
        st.ssm.assign(states_[i].ssm.size(), 0.0f);
        st.checkpoints.assign(static_cast<size_t>(rollback_window_) * (st.conv.size() + st.ssm.size()), 0.0f);
    }
    context->rollback_window_ = rollback_window_;
    context->checkpoint_pos_.assign(static_cast<size_t>(rollback_window_), -1);
    return context;
}

void Transformer::swap_context(Context& other) noexcept {
    std::swap(kv_, other.kv_);
    std::swap(n_past_, other.n_past_);
    states_.swap(other.states_);
    checkpoint_pos_.swap(other.checkpoint_pos_);
    std::swap(rollback_window_, other.rollback_window_);
    std::swap(checkpointing_, other.checkpointing_);
    snapshots_.swap(other.snapshots_);
    spare_buffers_.swap(other.spare_buffers_);
    std::swap(tree_base_, other.tree_base_);
    tree_parent_.swap(other.tree_parent_);
    std::swap(adapter_, other.adapter_);
    adapter_targets_.swap(other.adapter_targets_);
}

int32_t Transformer::positions_of(const Context& context) noexcept { return context.n_past_; }

Status Transformer::compute_logits_rows(const Route& route, std::span<const int32_t> rows, ThreadPool& pool) {
    const ModelConfig& c = config_;
    const size_t d = static_cast<size_t>(c.n_embd);
    const PhaseTimer timer(phases_.lm_head);
    file_->begin_layer(c.n_layers);  // output head slot of the prefetch window
    for (size_t i = 0; i < rows.size(); ++i) {
        rmsnorm(x_.data() + static_cast<size_t>(rows[i]) * d, output_norm_.data(), xb_.data() + i * d, c.n_embd,
                c.rms_eps);
    }
    logits_.resize(rows.size() * static_cast<size_t>(c.n_vocab));
    LIYAB_RETURN_IF_ERROR(
        matmul(route, route.ffn, *output_, xb_.data(), logits_.data(), static_cast<int32_t>(rows.size())));
    if (c.final_softcap > 0.0f) apply_final_softcap(std::span<float>(logits_.data(), logits_.size()), c.final_softcap, pool);
    return Status::ok();
}

// One pass over several sequences. Everything a sequence's own state decides
// (its positions, what each token sees, its recurrent states and their
// checkpoints, its adapter, latent attention's form) is taken per segment,
// exactly as forward() takes it for that sequence alone; everything else
// runs over the rows of every segment at once, which is where the pass saves:
// each weight is read once for all of them (dense models in RAM are bound by
// that read when decoding), and a mixture of experts reads each chosen expert
// once for every token that chose it.
Status Transformer::forward_batch(std::span<const BatchItem> items, const Route& route, ThreadPool& pool,
                                  std::vector<std::span<const float>>& out, const ForwardHooks* hooks) {
    const ModelConfig& c = config_;
    out.clear();
    last_exit_layer_ = -1;
    last_ffn_skips_ = 0;
    predicted_layer_ = -1;
    if (items.empty()) return Status(ErrorCode::InvalidArgument, "forward_batch() needs at least one sequence");
    if (route.attention == nullptr || route.ffn == nullptr || route.cpu == nullptr) {
        return Status(ErrorCode::InvalidArgument, "incomplete backend route");
    }
    if (hooks != nullptr && (hooks->early_exit != nullptr || hooks->ffn_skip != nullptr)) {
        return Status(ErrorCode::InvalidArgument,
                      "forward_batch(): early exit and FFN skipping apply to single-token decoding steps only");
    }
    if (!kv_) return Status(ErrorCode::InvalidArgument, "forward_batch(): no live context");
    segs_.clear();
    int32_t total = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const BatchItem& item = items[i];
        Context* ctx = item.context;
        if (item.tokens.empty()) return Status(ErrorCode::InvalidArgument, "forward_batch(): a sequence without tokens");
        for (const int32_t id : item.tokens) {
            if (id < 0 || id >= c.n_vocab) return Status(ErrorCode::InvalidArgument, "token id out of range");
        }
        if (ctx != nullptr && !ctx->holds()) {
            return Status(ErrorCode::InvalidArgument, "forward_batch(): an empty context");
        }
        for (size_t j = 0; j < i; ++j) {
            if (items[j].context == ctx) {
                return Status(ErrorCode::InvalidArgument, "forward_batch(): two items of one context");
            }
        }
        if ((ctx != nullptr ? ctx->tree_base_ : tree_base_) >= 0) {
            return Status(ErrorCode::InvalidArgument, "a tree pass awaits keep_path()");
        }
        const auto n = static_cast<int32_t>(item.tokens.size());
        Segment s;
        if (ctx == nullptr) {
            s = live_segment(total, n, item.logits);
        } else {
            s.first = total;
            s.n = n;
            s.kv = ctx->kv_.get();
            s.n_past = ctx->n_past_;
            s.states = &ctx->states_;
            s.rollback = ctx->rollback_window_;
            s.adapter = ctx->adapter_targets_.empty() ? nullptr : &ctx->adapter_targets_;
            s.logits = item.logits;
            s.context = ctx;
        }
        s.checkpoint = s.rollback > 0 && item.checkpoint;
        s.expanded = false;
        segs_.push_back(s);
        total += n;
    }
    auto n_past_of = [&](const Segment& s) -> int32_t& { return s.context != nullptr ? s.context->n_past_ : n_past_; };
    auto ring_of = [&](const Segment& s) -> std::vector<int32_t>& {
        return s.context != nullptr ? s.context->checkpoint_pos_ : checkpoint_pos_;
    };
    for (const Segment& s : segs_) {
        LIYAB_RETURN_IF_ERROR(s.kv->reserve(s.n_past + s.n));  // maps KV pages; ContextFull past the limit
    }
    // The DeltaNet mixers overwrite these rollback slots; they become valid
    // only once every block has run (as in forward()).
    for (const Segment& s : segs_) {
        const int32_t recorded = s.checkpoint ? std::min(s.n, s.rollback) : 0;
        for (int32_t i = 0; i < recorded; ++i) {
            ring_of(s)[static_cast<size_t>((s.n_past + s.n - 1 - i) % s.rollback)] = -1;
        }
    }
    // Latent attention: each segment in the form forward() would choose for
    // it alone (its own batch size and context), so its rows round as they
    // would there.
    latent_ms_ = 0.0;
    if (c.mla.kv_rank > 0) {
        int32_t window = 0;
        for (int32_t l = 0; l < c.n_layers; ++l) {
            if (c.blocks[static_cast<size_t>(l)].latent) {
                window = layers_[static_cast<size_t>(l)].window;
                break;
            }
        }
        for (Segment& s : segs_) {
            if (s.n < 2 || key_ranges_ != nullptr) continue;
            const int64_t pairs = latent_visible(s, window);
            int64_t positions = 0;
            for (const auto& [a, b] : mla_union_) positions += b - a;
            const LatentFormChooser::Work work = LatentFormChooser::work(c, s.n, pairs, positions);
            s.expanded = latent_form_ == LatentForm::Expanded ||
                         (latent_form_ == LatentForm::Auto && latent_chooser_.expand(work));
        }
    }
    batch_tokens_.clear();
    seg_of_.clear();
    adapter_rows_.clear();
    adapters_in_pass_ = false;
    for (size_t i = 0; i < items.size(); ++i) {
        batch_tokens_.insert(batch_tokens_.end(), items[i].tokens.begin(), items[i].tokens.end());
        seg_of_.insert(seg_of_.end(), items[i].tokens.size(), static_cast<int32_t>(i));
        adapters_in_pass_ = adapters_in_pass_ || segs_[i].adapter != nullptr;
    }
    const auto un = static_cast<size_t>(total);
    size_batch(total);
    LIYAB_RETURN_IF_ERROR(embed(batch_tokens_, std::span<float>(x_.data(), un * static_cast<size_t>(c.n_embd))));
    struct StepsOnly {
        bool& flag;
        ~StepsOnly() { flag = false; }
    } steps_only{steps_only_};
    steps_only_ = std::all_of(segs_.begin(), segs_.end(), [](const Segment& s) { return s.n == 1; });
    const HeadMaskHook* head_mask = hooks != nullptr ? hooks->head_mask : nullptr;
    const std::atomic<bool>* cancel = hooks != nullptr ? hooks->cancel : nullptr;
    FfnMatmulHook* ffn_hook = hooks != nullptr ? hooks->ffn_matmul : nullptr;

    for (int32_t l = 0; l < c.n_layers; ++l) {
        if (cancel != nullptr && cancel->load(std::memory_order_relaxed)) {
            // Every context back to where the pass started (forward()'s
            // abandon_pass(), each one made live for it in turn).
            const std::vector<Segment> segments = segs_;
            for (const Segment& s : segments) {
                if (s.context != nullptr) swap_context(*s.context);
                if (!abandon_pass().is_ok()) reset();  // a hybrid context without a snapshot here
                if (s.context != nullptr) swap_context(*s.context);
            }
            return Status(ErrorCode::Cancelled, "forward pass cancelled");
        }
        const BlockWeights w(*this, l);
        LIYAB_RETURN_IF_ERROR(w.status());
        LIYAB_RETURN_IF_ERROR(run_block(l, w, total, route, pool, head_mask, nullptr, ffn_hook));
    }

    for (const Segment& s : segs_) {
        const int32_t recorded = s.checkpoint ? std::min(s.n, s.rollback) : 0;
        for (int32_t i = 0; i < recorded; ++i) {
            const int32_t pos = s.n_past + s.n - 1 - i;
            ring_of(s)[static_cast<size_t>(pos % s.rollback)] = pos;
        }
        if (c.mla.kv_rank > 0) ++(s.expanded ? latent_stats_.expanded_passes : latent_stats_.absorbed_passes);
        n_past_of(s) = s.n_past + s.n;
        s.kv->release_unreachable(s.n_past + s.n, max_batch_);  // recycle pages that left the window
    }
    if (expert_store_ != nullptr && ++decode_steps_ % 16 == 0) expert_store_->age();

    // --- output head: the rows each segment asked for, segment after segment ---
    std::vector<int32_t> rows;
    std::vector<std::pair<size_t, size_t>> spans(segs_.size());  // per segment: its first logits row, its count
    for (size_t i = 0; i < segs_.size(); ++i) {
        const Segment& s = segs_[i];
        const size_t at = rows.size();
        if (s.logits == Logits::Last) rows.push_back(s.first + s.n - 1);
        if (s.logits == Logits::All) {
            for (int32_t t = s.first; t < s.first + s.n; ++t) rows.push_back(t);
        }
        spans[i] = {at, rows.size() - at};
        if (s.adapter != nullptr && rows.size() > at) {
            adapter_rows_.push_back({static_cast<int32_t>(at), static_cast<int32_t>(rows.size() - at), s.adapter});
        }
    }
    if (!rows.empty()) {
        logits_rows_ = true;
        const Status status = compute_logits_rows(route, rows, pool);
        logits_rows_ = false;
        adapter_rows_.clear();
        if (!status.is_ok()) {
            for (const Segment& s : segs_) n_past_of(s) = s.n_past;
            return status;
        }
    }
    const auto vocab = static_cast<size_t>(c.n_vocab);
    for (const auto& [at, count] : spans) {
        out.emplace_back(count == 0 ? std::span<const float>()
                                    : std::span<const float>(logits_.data() + at * vocab, count * vocab));
    }
    return Status::ok();
}

}  // namespace liyab
