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
// Not thread-safe: owned by one SpeculativeDecoder.
#ifndef LIYAB_CORE_DRAFT_BUDGET_H
#define LIYAB_CORE_DRAFT_BUDGET_H

#include <cstdint>
#include <vector>

namespace liyab {

class DraftBudget {
public:
    explicit DraftBudget(int32_t max_drafts);

    // Most drafts to verify in the next step that has drafts to offer.
    [[nodiscard]] int32_t choose();
    // A step verified `drafts` drafted tokens (0: plain decoding), produced
    // `tokens` tokens and took `ms` milliseconds.
    void record(int32_t drafts, int32_t tokens, double ms);
    // Tokens per ms measured for `drafts`; 0 before the first sample.
    [[nodiscard]] double rate(int32_t drafts) const;

    // Steps before a losing count is tried again, first and at most.
    static constexpr uint64_t kFirstRetry = 16;
    static constexpr uint64_t kMaxRetry = 1024;

private:
    struct Arm {
        double rate = 0.0;
        int32_t samples = 0;
        uint64_t backoff = 0;   // current retry interval; 0 while it leads
        uint64_t retry_at = 0;  // step from which it may be tried again
    };
    [[nodiscard]] int32_t best() const;  // the count with the highest rate
    std::vector<Arm> arms_;
    uint64_t step_ = 0;
};

}  // namespace liyab

#endif  // LIYAB_CORE_DRAFT_BUDGET_H
