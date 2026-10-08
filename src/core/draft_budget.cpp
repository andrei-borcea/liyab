#include "core/draft_budget.h"

#include <algorithm>

namespace liyab {

namespace {

// Weight of a new sample: about the last ten steps of a draft count matter.
constexpr double kDecay = 0.2;

}  // namespace

DraftBudget::DraftBudget(int32_t max_drafts) : arms_(static_cast<size_t>(std::max(0, max_drafts)) + 1) {}

int32_t DraftBudget::best() const {
    size_t best = 0;
    for (size_t i = 1; i < arms_.size(); ++i) {
        if (arms_[i].rate > arms_[best].rate) best = i;
    }
    return static_cast<int32_t>(best);
}

int32_t DraftBudget::choose() {
    ++step_;
    for (size_t i = 0; i < arms_.size(); ++i) {
        if (arms_[i].samples == 0) return static_cast<int32_t>(i);  // measure every count once first
    }
    // Re-measure a losing count when its retry step has come (oldest first).
    const int32_t leader = best();
    int32_t due = -1;
    for (size_t i = 0; i < arms_.size(); ++i) {
        if (static_cast<int32_t>(i) == leader || arms_[i].retry_at > step_) continue;
        if (due < 0 || arms_[i].retry_at < arms_[static_cast<size_t>(due)].retry_at) due = static_cast<int32_t>(i);
    }
    return due >= 0 ? due : leader;
}

void DraftBudget::record(int32_t drafts, int32_t tokens, double ms) {
    if (drafts < 0 || static_cast<size_t>(drafts) >= arms_.size() || ms <= 0.0) return;
    Arm& arm = arms_[static_cast<size_t>(drafts)];
    const double sample = static_cast<double>(tokens) / ms;
    arm.rate = arm.samples == 0 ? sample : arm.rate + kDecay * (sample - arm.rate);
    ++arm.samples;
    // A count that is not the best waits twice as long before its next try
    // each time it loses again; winning makes it the leader (always used).
    if (best() != drafts) {
        arm.backoff = std::min(arm.backoff == 0 ? kFirstRetry : 2 * arm.backoff, kMaxRetry);
        arm.retry_at = step_ + arm.backoff;
    } else {
        arm.backoff = 0;
    }
}

double DraftBudget::rate(int32_t drafts) const {
    if (drafts < 0 || static_cast<size_t>(drafts) >= arms_.size()) return 0.0;
    return arms_[static_cast<size_t>(drafts)].rate;
}

}  // namespace liyab
