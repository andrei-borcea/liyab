#include "liyab/speculative_decoder.h"

#include <algorithm>
#include <chrono>

#include "core/draft_budget.h"
#include "core/sampling.h"
#include "core/transformer.h"

namespace liyab {

namespace {

// Longest suffix match first: a 4-token match predicts better than a 2-token
// one; single tokens repeat too often to be worth a verification.
constexpr int32_t kLookupMaxMatch = 4;
constexpr int32_t kLookupMinMatch = 2;

}  // namespace

SpeculativeDecoder::SpeculativeDecoder(Transformer& target, Transformer& draft, int32_t draft_tokens)
    : target_(target), draft_(&draft), draft_tokens_(std::max(1, draft_tokens)),
      budget_(std::make_unique<DraftBudget>(draft_tokens_)) {}

SpeculativeDecoder::SpeculativeDecoder(Transformer& target, int32_t draft_tokens)
    : target_(target), draft_tokens_(std::max(1, draft_tokens)), budget_(std::make_unique<DraftBudget>(draft_tokens_)) {}

SpeculativeDecoder::~SpeculativeDecoder() = default;

void SpeculativeDecoder::set_adaptive(bool adaptive) {
    budget_ = adaptive ? std::make_unique<DraftBudget>(draft_tokens_) : nullptr;
}

void SpeculativeDecoder::lookup(std::span<const int32_t> history, int32_t k, std::vector<int32_t>& out) {
    out.clear();
    const auto n = static_cast<int32_t>(history.size());
    for (int32_t m = std::min(kLookupMaxMatch, n - 1); m >= kLookupMinMatch; --m) {
        const std::span<const int32_t> suffix = history.last(static_cast<size_t>(m));
        // Most recent earlier occurrence: recent text is the likeliest to repeat.
        for (int32_t j = n - m - 1; j >= 0; --j) {
            if (!std::equal(suffix.begin(), suffix.end(), history.begin() + j)) continue;
            for (int32_t i = j + m; i < n && static_cast<int32_t>(out.size()) < k; ++i) out.push_back(history[static_cast<size_t>(i)]);
            return;
        }
    }
}

Result<std::vector<int32_t>> SpeculativeDecoder::step(int32_t last, std::span<const int32_t> context, Sampler& sampler,
                                                      const Route& target_route, const Route& draft_route,
                                                      ThreadPool& pool) {
    const auto t0 = std::chrono::steady_clock::now();
    const int32_t base = target_.n_past();
    if (draft_ != nullptr && draft_->n_past() != base) {
        return Status(ErrorCode::Internal, "draft and target caches diverged");
    }
    const int32_t n_vocab = target_.config().n_vocab;
    // Room for `last` + k drafts in the caches and in one verification batch.
    int32_t k = std::min(draft_tokens_, target_.max_batch() - 1);
    if (target_.kv_cache().window() == 0) k = std::min(k, target_.context_length() - base - 1);
    if (draft_ != nullptr && draft_->kv_cache().window() == 0) k = std::min(k, draft_->context_length() - base - 1);
    if (budget_ != nullptr) k = std::min(k, budget_->choose());

    // 1. Draft up to k tokens: from the draft model (with its distributions),
    // or by looking the recent tokens up in the context (deterministic).
    std::vector<int32_t> verify{last};
    if (k > 0 && draft_ != nullptr) {
        q_.resize(static_cast<size_t>(k));
        int32_t cur = last;
        for (int32_t i = 0; i < k; ++i) {
            auto logits = draft_->forward(std::span<const int32_t>(&cur, 1), Transformer::Logits::Last, draft_route, pool);
            if (!logits) return logits.status();
            auto& q = q_[static_cast<size_t>(i)];
            sampler.probabilities(*logits, q);
            cur = sampler.greedy() ? Sampler::argmax(q) : sampler.sample(q);
            verify.push_back(cur);
        }
    } else if (k > 0) {
        history_.assign(context.begin(), context.end());
        history_.push_back(last);
        lookup(history_, k, drafts_);
        verify.insert(verify.end(), drafts_.begin(), drafts_.end());
    }
    k = static_cast<int32_t>(verify.size()) - 1;
    proposed_ += k;

    // 2. Verify every draft with one batched target pass (k + 1 logit rows).
    // Only a pass with drafts can be rolled back: a plain step skips the
    // recurrent-state checkpoints.
    target_.set_checkpointing(k > 0);
    auto logits = target_.forward(verify, k > 0 ? Transformer::Logits::All : Transformer::Logits::Last, target_route, pool);
    if (!logits) return logits.status();
    auto row = [&](int32_t i) {
        return logits->subspan(static_cast<size_t>(i) * static_cast<size_t>(n_vocab), static_cast<size_t>(n_vocab));
    };

    // 3. Accept / reject (Leviathan et al.): draft d is kept with probability
    // min(1, p(d) / q(d)); on the first rejection the replacement comes from
    // norm(max(0, p - q)). A lookup draft is deterministic (q = 1 at d), so d
    // is kept with probability p(d) and the replacement drawn from p without d.
    std::vector<int32_t> out;
    int32_t n_accepted = 0;
    for (; n_accepted < k; ++n_accepted) {
        const int32_t d = verify[static_cast<size_t>(n_accepted) + 1];
        sampler.probabilities(row(n_accepted), p_);
        const float pd = p_[static_cast<size_t>(d)];
        bool accept = false;
        if (sampler.greedy()) {
            accept = pd > 0.0f;  // probabilities() leaves only the argmax nonzero
        } else if (draft_ != nullptr) {
            const float qd = q_[static_cast<size_t>(n_accepted)][static_cast<size_t>(d)];
            accept = qd > 0.0f && sampler.uniform() < std::min(1.0f, pd / qd);
        } else {
            accept = sampler.uniform() < pd;
        }
        if (accept) {
            out.push_back(d);
            continue;
        }
        if (sampler.greedy()) {
            out.push_back(Sampler::argmax(p_));
        } else if (draft_ != nullptr) {
            const auto& q = q_[static_cast<size_t>(n_accepted)];
            for (size_t v = 0; v < p_.size(); ++v) p_[v] = std::max(0.0f, p_[v] - q[v]);
            out.push_back(sampler.sample(p_));  // sample() falls back to argmax if the residual is empty
        } else {
            p_[static_cast<size_t>(d)] = 0.0f;
            out.push_back(sampler.sample(p_));
        }
        break;
    }
    accepted_ += n_accepted;
    if (n_accepted == k) {
        // Every draft accepted (or none proposed): one more token from the
        // target's last row; the draft model also reads the final draft so its
        // cache stays aligned.
        sampler.probabilities(row(k), p_);
        out.push_back(sampler.greedy() ? Sampler::argmax(p_) : sampler.sample(p_));
        if (draft_ != nullptr) {  // with no room to draft (k == 0) this is `last`
            const int32_t final_draft = verify.back();
            if (auto d = draft_->forward(std::span<const int32_t>(&final_draft, 1), Transformer::Logits::None,
                                         draft_route, pool);
                !d) {
                return d.status();
            }
        }
    }

    // 4. Roll the caches back to prefix + last + accepted drafts.
    const int32_t keep = base + 1 + n_accepted;
    LIYAB_RETURN_IF_ERROR(target_.truncate(keep));
    if (draft_ != nullptr) LIYAB_RETURN_IF_ERROR(draft_->truncate(keep));
    if (budget_ != nullptr) {
        budget_->record(k, static_cast<int32_t>(out.size()),
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return out;
}

}  // namespace liyab
