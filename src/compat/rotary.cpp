#include "compat/rotary.h"

#include <algorithm>
#include <cmath>
#include <numbers>

#include "core/quant.h"

namespace liyab {

Result<RotaryPlan> make_rotary_plan(const MmapLoader& f, const ModelConfig& c, int32_t context_length) {
    if (context_length <= 0) return Status(ErrorCode::InvalidArgument, "rotary planning needs a positive context");
    RotaryPlan out;
    out.block_tables.assign(static_cast<size_t>(c.n_layers), 0);
    // RoPE inverse frequencies, with optional per-pair factors: Llama 3.x's
    // rope_freqs, or LongRoPE's (Phi-3), whose long factors apply when the
    // context is longer than the original one. The reference picks them by
    // the sequence's length instead; with one table for the whole context,
    // keys never need rotating again (a context up to the original length
    // matches the reference exactly).
    std::vector<float> factors(static_cast<size_t>(c.rope_dim / 2), 1.0f);
    const char* factor_tensor = c.rope_ctx_orig == 0                         ? "rope_freqs.weight"
                                : context_length > c.rope_ctx_orig ? "rope_factors_long.weight"
                                                                           : "rope_factors_short.weight";
    if (const TensorView* rf = f.tensor(factor_tensor)) {
        if (rf->elements() != c.rope_dim / 2 || rf->type != DType::F32) {
            return Status(ErrorCode::InvalidModel, std::string(factor_tensor) + " has an unexpected shape");
        }
        quant::dequantize_row(DType::F32, rf->data, factors.data(), rf->elements());
    }
    auto table = [&](float base, float scale, const std::vector<float>& divisors, const YarnRope* yarn, int32_t head_dim = 0) {
        RopeTable t;
        t.head_dim = head_dim > 0 ? head_dim : c.head_dim;
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
            const double dims = 2 * divisors.size();
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
            const double freq = std::pow(static_cast<double>(base), -2.0 * static_cast<double>(i) / static_cast<double>(2 * divisors.size()));
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
    out.tables.push_back(table(c.rope_base, c.rope_scale, factors, c.yarn.factor > 0.0f ? &c.yarn : nullptr));
    out.tables.back().magnitude = c.rope_magnitude;
    // Windowed blocks with a base of their own rotate unscaled, without the
    // global frequency factors (Gemma 3's local RoPE).
    if (c.rope_base_swa > 0.0f) {
        out.tables.push_back(table(c.rope_base_swa, 1.0f, std::vector<float>(factors.size(), 1.0f), nullptr));
    }
    int32_t compression_table = 0;
    if (c.spec->shared_kv) {
        compression_table = static_cast<int32_t>(out.tables.size());
        auto compressed = table(c.compress_rope_base, c.rope_scale, factors,
                                c.yarn.factor > 0.0f ? &c.yarn : nullptr);
        compressed.magnitude = c.rope_magnitude;
        out.tables.push_back(std::move(compressed));
        // The main, sliding-only stream has plain RoPE even when compressed
        // layers use YaRN. No table is changed for existing architectures.
        out.tables[0] = table(c.rope_base, 1.0f, std::vector<float>(factors.size(), 1.0f), nullptr);
    }
    for (int32_t l = 0; l < c.n_layers; ++l) {
        int32_t& table_id = out.block_tables[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        if (plan.mixer != MixerKind::Attention || !plan.windowed) continue;
        if (c.rope_base_swa > 0.0f) table_id = 1;
        if (c.spec->shared_kv) table_id = plan.compression.stride > 0 ? compression_table : 0;
    }

    for (int32_t l = 0; l < c.n_layers; ++l) {
        int32_t& table_id = out.block_tables[static_cast<size_t>(l)];
        const BlockPlan& plan = c.blocks[static_cast<size_t>(l)];
        if (plan.mixer != MixerKind::Attention) continue;
        const RopeTable original = out.tables[static_cast<size_t>(table_id)];
        if (original.head_dim == plan.head_dim && 2 * original.inv_freq.size() == static_cast<size_t>(plan.rope_dim)) continue;
        std::vector<float> divisors(static_cast<size_t>(plan.rope_dim / 2), 1.0f);
        const bool local = plan.windowed && c.rope_base_swa > 0.0f;
        if (!local && divisors.size() == factors.size()) divisors = factors;
        RopeTable per_block = table(local ? c.rope_base_swa : c.rope_base, original.scale, divisors,
                                   !local && c.yarn.factor > 0.0f ? &c.yarn : nullptr, plan.head_dim);
        per_block.magnitude = original.magnitude;
        table_id = static_cast<int32_t>(out.tables.size());
        out.tables.push_back(std::move(per_block));
    }

    return out;
}

}  // namespace liyab
