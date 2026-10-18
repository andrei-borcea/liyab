// Liyab — how many drafted tokens a speculative step should verify (internal).
//
// Whether speculation pays depends on the model and the device, not on the
// architecture: verifying n tokens costs about one token on a dense model in
// RAM, but ~n tokens on a MoE whose experts stream from flash (each token
// brings experts of its own), and draft acceptance depends on the text. So the
// budget is learned at run time instead of configured per model: for every
// draft count 0..max (0 = plain decoding) it keeps an exponential moving
// average of the tokens each step produced per millisecond and uses the best
// one. The others are tried again because heat, cache state and the text
// change, but each try that loses doubles the wait before the next one (16 up
// to 1024 steps), so a count that never pays costs almost nothing. Output
// does not depend on it (speculation is lossless).
//
// A drafter that knows how likely each draft is to be accepted (the suffix
// drafter) asks choose(survival) instead: from the measured time of a step
// per draft count it picks the count with the most expected tokens per
// millisecond for this very draft, so a confident draft is verified at length
// and a doubtful one not at all, also on a model where drafting does not pay
// on average. A count's step time is the median of its last kWindow steps,
// never below a count with fewer drafts: a step slowed by other work on the
// cores, or a lucky one, does not move it. The drafts may be the nodes of a
// tree (alternative branches, verified in one pass): a pass of j drafts costs
// about the same whether they form a branch or a tree (j + 1 rows of every
// matmul; on a MoE whose experts stream from flash, the union of the rows'
// experts either way), so the same measured costs size both.
//
// Not thread-safe: owned by one SpeculativeDecoder.
#ifndef LIYAB_CORE_DRAFT_BUDGET_H
#define LIYAB_CORE_DRAFT_BUDGET_H

#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace liyab {

class DraftBudget {
public:
    explicit DraftBudget(int32_t max_drafts);

    // Most drafts to verify in the next step that has drafts to offer.
    [[nodiscard]] int32_t choose();
    // Most drafts to verify of a draft whose survival[i] is the estimated
    // probability that draft i is accepted (non-increasing; along a branch,
    // that drafts 0..i all are; for the nodes of a tree, the likeliest first,
    // that its branch is), so the first j drafts are expected to give
    // 1 + survival[0] + ... + survival[j - 1] tokens: the count j <=
    // survival.size() maximizing
    // (1 + survival[0] + ... + survival[j - 1]) / cost(j). Plain decoding is
    // measured first. A step never costs less with more drafts, so a count
    // costs at least the dearest fewer-drafts count measured. Counts are
    // measured upwards: only the first one not measured yet is a candidate,
    // costed as that bound, so it is tried only where it could win and a
    // doubtful draft never tries a long unknown length. A count not
    // measured for longer
    // than its wait (kFirstRetry steps, doubling up to kMaxRetry with each
    // such try) is measured again when it comes within kRetryShare of the
    // best count's rate.
    [[nodiscard]] int32_t choose(std::span<const float> survival);
    // A step verified `drafts` drafted tokens (0: plain decoding), produced
    // `tokens` tokens and took `ms` milliseconds.
    void record(int32_t drafts, int32_t tokens, double ms);
    // Tokens per ms measured for `drafts`; 0 before the first sample.
    [[nodiscard]] double rate(int32_t drafts) const;
    // Milliseconds per step measured for `drafts`; 0 before the first sample.
    [[nodiscard]] double cost(int32_t drafts) const;

    // Steps before a losing count is tried again, first and at most.
    static constexpr uint64_t kFirstRetry = 16;
    static constexpr uint64_t kMaxRetry = 1024;
    static constexpr double kRetryShare = 0.9;
    // Step times per count that choose(survival) takes the median of.
    static constexpr int32_t kWindow = 8;

private:
    struct Arm {
        double rate = 0.0;
        int32_t samples = 0;
        uint64_t backoff = 0;   // current retry interval; 0 while it leads
        uint64_t retry_at = 0;  // step from which it may be tried again
        double ms = 0.0;               // step time (median of `recent`), for choose(survival)
        std::array<float, kWindow> recent{};  // the last step times, a ring
        uint8_t filled = 0, next = 0;
        uint64_t sampled_at = 0;       // step of the last sample
        uint64_t wait = kFirstRetry;   // steps after which that time is measured again if it nearly wins
    };
    [[nodiscard]] int32_t best() const;  // the count with the highest rate
    std::vector<Arm> arms_;
    uint64_t step_ = 0;
    int32_t retrying_ = -1;      // the count choose(survival) measures again this step
    std::vector<double> rates_;  // choose(survival) scratch: expected tokens per ms per count
};

}  // namespace liyab

#endif  // LIYAB_CORE_DRAFT_BUDGET_H
