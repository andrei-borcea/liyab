#include "liyab/speculative_decoder.h"

#include <algorithm>

#include "core/sampling.h"
#include "core/transformer.h"

namespace liyab {

SpeculativeDecoder::SpeculativeDecoder(Transformer& target, Transformer& draft, int32_t draft_tokens)
    : target_(target), draft_(draft), draft_tokens_(std::max(1, draft_tokens)) {}

Result<std::vector<int32_t>> SpeculativeDecoder::step(int32_t last, Sampler& sampler, const Route& target_route,
                                                      const Route& draft_route, ThreadPool& pool) {
    const int32_t base = target_.n_past();
    if (draft_.n_past() != base) return Status(ErrorCode::Internal, "draft and target caches diverged");
    const int32_t n_vocab = target_.config().n_vocab;

    // Room for `last` + k drafts in both caches and in one verification batch.
    int32_t k = std::min(draft_tokens_, target_.max_batch() - 1);
    if (target_.kv_cache().window() == 0) k = std::min(k, target_.context_length() - base - 1);
    if (draft_.kv_cache().window() == 0) k = std::min(k, draft_.context_length() - base - 1);

    std::vector<int32_t> out;
    if (k <= 0) {
        // No room to speculate: plain decoding step, keeping the draft in sync.
        auto logits = target_.forward(std::span<const int32_t>(&last, 1), Transformer::Logits::Last, target_route, pool);
        if (!logits) return logits.status();
        sampler.probabilities(*logits, p_);
        out.push_back(sampler.greedy() ? Sampler::argmax(p_) : sampler.sample(p_));
        if (auto d = draft_.forward(std::span<const int32_t>(&last, 1), Transformer::Logits::None, draft_route, pool); !d) {
            return d.status();
        }
        return out;
    }

    // 1. Draft k tokens.
    q_.resize(static_cast<size_t>(k));
    std::vector<int32_t> verify{last};
    int32_t cur = last;
    for (int32_t i = 0; i < k; ++i) {
        auto logits = draft_.forward(std::span<const int32_t>(&cur, 1), Transformer::Logits::Last, draft_route, pool);
        if (!logits) return logits.status();
        auto& q = q_[static_cast<size_t>(i)];
        sampler.probabilities(*logits, q);
        cur = sampler.greedy() ? Sampler::argmax(q) : sampler.sample(q);
        verify.push_back(cur);
    }
    proposed_ += k;

    // 2. Verify all drafts with one batched target pass (k + 1 logit rows).
    auto logits = target_.forward(verify, Transformer::Logits::All, target_route, pool);
    if (!logits) return logits.status();
    auto row = [&](int32_t i) {
        return logits->subspan(static_cast<size_t>(i) * static_cast<size_t>(n_vocab), static_cast<size_t>(n_vocab));
    };

    // 3. Accept / reject.
    int32_t n_accepted = 0;
    for (; n_accepted < k; ++n_accepted) {
        const int32_t d = verify[static_cast<size_t>(n_accepted) + 1];
        sampler.probabilities(row(n_accepted), p_);
        const auto& q = q_[static_cast<size_t>(n_accepted)];
        const float pd = p_[static_cast<size_t>(d)];
        const float qd = q[static_cast<size_t>(d)];
        const bool accept = sampler.greedy() ? pd > 0.0f : (qd > 0.0f && sampler.uniform() < std::min(1.0f, pd / qd));
        if (accept) {
            out.push_back(d);
            continue;
        }
        if (sampler.greedy()) {
            out.push_back(Sampler::argmax(p_));
        } else {
            for (size_t v = 0; v < p_.size(); ++v) p_[v] = std::max(0.0f, p_[v] - q[v]);
            out.push_back(sampler.sample(p_));  // sample() falls back to argmax if the residual is empty
        }
        break;
    }
    accepted_ += n_accepted;

    if (n_accepted == k) {
        // Every draft accepted: bonus token from the target's last row, and
        // feed the final draft token to the draft model to keep caches aligned.
        sampler.probabilities(row(k), p_);
        out.push_back(sampler.greedy() ? Sampler::argmax(p_) : sampler.sample(p_));
        const int32_t final_draft = verify.back();
        if (auto d = draft_.forward(std::span<const int32_t>(&final_draft, 1), Transformer::Logits::None, draft_route,
                                    pool);
            !d) {
            return d.status();
        }
    }

    // 4. Roll both caches back to prefix + last + accepted drafts.
    const int32_t keep = base + 1 + n_accepted;
    LIYAB_RETURN_IF_ERROR(target_.truncate(keep));
    LIYAB_RETURN_IF_ERROR(draft_.truncate(keep));
    return out;
}

}  // namespace liyab
