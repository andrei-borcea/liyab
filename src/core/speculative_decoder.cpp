#include "liyab/speculative_decoder.h"

#include <algorithm>
#include <chrono>

#include "core/draft_budget.h"
#include "core/sampling.h"
#include "core/suffix_drafter.h"
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

SpeculativeDecoder::SpeculativeDecoder(Transformer& target, int32_t draft_tokens, TextDrafts drafts)
    : target_(target), draft_tokens_(std::max(1, draft_tokens)), budget_(std::make_unique<DraftBudget>(draft_tokens_)),
      suffix_(drafts == TextDrafts::Suffix ? std::make_unique<SuffixDrafter>() : nullptr) {}

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
    if (suffix_ != nullptr) return suffix_step(last, context, sampler, std::max(0, k), target_route, pool, t0);
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
            // A draft with a differently padded output (the same token table, more or fewer rows after it) drafts
            // among the ids both have; q is then as long as the target's distributions.
            sampler.probabilities(logits->first(static_cast<size_t>(std::min(draft_->config().n_vocab, n_vocab))), q);
            q.resize(static_cast<size_t>(n_vocab), 0.0f);
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

Result<std::vector<int32_t>> SpeculativeDecoder::suffix_step(int32_t last, std::span<const int32_t> context,
                                                             Sampler& sampler, int32_t room, const Route& target_route,
                                                             ThreadPool& pool,
                                                             std::chrono::steady_clock::time_point t0) {
    const int32_t base = target_.n_past();
    const int32_t n_vocab = target_.config().n_vocab;
    history_.assign(context.begin(), context.end());
    history_.push_back(last);
    suffix_->sync(history_);
    const SuffixDrafter::Draft& draft = tree_ ? suffix_->propose_tree(room) : suffix_->propose(room);
    // Adaptive: as many drafts as the expected tokens per millisecond favour
    // (the likeliest first, so any first k of a tree form a tree); fixed:
    // all the drafter offers.
    int32_t k = static_cast<int32_t>(draft.tokens.size());
    if (budget_ != nullptr) k = std::min(k, budget_->choose(draft.survival));
    proposed_ += k;

    // The batch: `last`, then the drafts depth first, each node's children
    // in draft order (the likeliest first). The likeliest branch is then the
    // batch's first tokens: when it is accepted, keep_path() moves no row.
    // batch_parent_[i]: batch index of token i's parent; batch_draft_[i]:
    // its draft index (-1 for `last`).
    child_first_.assign(static_cast<size_t>(k) + 1, -1);  // per draft index + 1: first child, next sibling
    child_next_.assign(static_cast<size_t>(k) + 1, -1);
    for (int32_t i = k - 1; i >= 0; --i) {  // prepended in reverse: children in draft order
        const int32_t parent = draft.parents[static_cast<size_t>(i)] + 1;
        child_next_[static_cast<size_t>(i) + 1] = child_first_[static_cast<size_t>(parent)];
        child_first_[static_cast<size_t>(parent)] = i + 1;
    }
    std::vector<int32_t> verify{last};
    batch_of_.assign(static_cast<size_t>(k) + 1, 0);  // per draft index + 1: its batch index (`last`: 0)
    batch_parent_.assign(1, -1);
    batch_draft_.assign(1, -1);
    stack_.clear();  // siblings (draft index + 1) still to visit
    for (int32_t c = child_first_[0]; c >= 0;) {  // depth first, iteratively
        verify.push_back(draft.tokens[static_cast<size_t>(c) - 1]);
        batch_draft_.push_back(c - 1);
        batch_of_[static_cast<size_t>(c)] = static_cast<int32_t>(verify.size()) - 1;
        batch_parent_.push_back(batch_of_[static_cast<size_t>(draft.parents[static_cast<size_t>(c) - 1]) + 1]);
        if (child_next_[static_cast<size_t>(c)] >= 0) stack_.push_back(child_next_[static_cast<size_t>(c)]);
        if (child_first_[static_cast<size_t>(c)] >= 0) {
            c = child_first_[static_cast<size_t>(c)];
        } else if (!stack_.empty()) {
            c = stack_.back();
            stack_.pop_back();
        } else {
            c = -1;
        }
    }
    bool chain = true;
    for (size_t i = 1; i < batch_parent_.size(); ++i) chain = chain && batch_parent_[i] == static_cast<int32_t>(i) - 1;

    // One pass: a branch as before trees (linear), else the tree.
    if (chain) target_.set_checkpointing(k > 0);
    auto logits = chain ? target_.forward(verify, k > 0 ? Transformer::Logits::All : Transformer::Logits::Last,
                                          target_route, pool)
                        : target_.forward_tree(verify, batch_parent_, target_route, pool);
    if (!logits) return logits.status();

    // Each row on the way is sampled as plain decoding samples it (one draw
    // per token), from `last` down: a draft is kept while it is the sample,
    // the child of the last kept token whose token was drawn (siblings
    // differ, so at most one is). The tokens are plain decoding's whatever
    // was drafted, and a draft d is kept with probability p(d), as Leviathan
    // et al. keep a deterministic draft.
    std::vector<int32_t> out;
    path_.assign(1, 0);
    accepted_path_.clear();
    for (int32_t at = 0;;) {
        const auto row = logits->subspan(static_cast<size_t>(at) * static_cast<size_t>(n_vocab), static_cast<size_t>(n_vocab));
        sampler.probabilities(row, p_);
        const int32_t token = sampler.greedy() ? Sampler::argmax(p_) : sampler.sample(p_);
        out.push_back(token);
        int32_t next = -1;
        for (size_t b = static_cast<size_t>(at) + 1; b < verify.size(); ++b) {
            if (batch_parent_[b] == at && verify[b] == token) {
                next = static_cast<int32_t>(b);
                break;
            }
        }
        if (next < 0) break;
        at = next;
        path_.push_back(next);
        accepted_path_.push_back(batch_draft_[static_cast<size_t>(next)]);
    }
    const auto n_accepted = static_cast<int32_t>(accepted_path_.size());
    accepted_ += n_accepted;
    suffix_->feedback(draft, k, accepted_path_);

    if (chain) {
        LIYAB_RETURN_IF_ERROR(target_.truncate(base + 1 + n_accepted));
    } else {
        ++tree_steps_;
        if (path_.back() != static_cast<int32_t>(path_.size()) - 1) ++side_branches_;
        LIYAB_RETURN_IF_ERROR(target_.keep_path(path_));
    }
    if (budget_ != nullptr) {
        budget_->record(k, static_cast<int32_t>(out.size()),
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    return out;
}

}  // namespace liyab
