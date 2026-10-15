#include "core/draft_budget.h"

#include <algorithm>
#include <array>

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

int32_t DraftBudget::choose(std::span<const float> survival) {
    ++step_;
    retrying_ = -1;
    if (arms_[0].samples == 0) return 0;  // plain decoding first: every other count is costed against it
    const int32_t offered = std::min(static_cast<int32_t>(survival.size()), static_cast<int32_t>(arms_.size()) - 1);
    rates_.assign(static_cast<size_t>(offered) + 1, 0.0);
    int32_t choice = 0;
    rates_[0] = 1.0 / std::max(arms_[0].ms, 1e-9);
    double tokens = 1.0;
    // The dearest fewer-drafts step: a step never costs less with more
    // drafts, so this bounds an unknown time and a time measured too low.
    double floor = arms_[0].ms;
    for (int32_t j = 1; j <= offered; ++j) {
        tokens += survival[static_cast<size_t>(j - 1)];
        const Arm& arm = arms_[static_cast<size_t>(j)];
        floor = std::max(floor, arm.samples == 0 ? 0.0 : arm.ms);
        const double ms = std::max(floor, 1e-9);
        rates_[static_cast<size_t>(j)] = tokens / ms;
        if (rates_[static_cast<size_t>(j)] > rates_[static_cast<size_t>(choice)]) choice = j;
    }
    // A count whose time is old and that nearly wins is measured again (its
    // time may have changed with heat or context length); one that is far
    // behind would not win anyway.
    int32_t retry = -1;
    for (int32_t j = 0; j <= offered; ++j) {
        const Arm& arm = arms_[static_cast<size_t>(j)];
        if (j == choice || arm.samples == 0 || step_ - arm.sampled_at <= arm.wait) continue;
        if (rates_[static_cast<size_t>(j)] < kRetryShare * rates_[static_cast<size_t>(choice)]) continue;
        if (retry < 0 || rates_[static_cast<size_t>(j)] > rates_[static_cast<size_t>(retry)]) retry = j;
    }
    if (retry >= 0) {
        retrying_ = retry;
        return retry;
    }
    return choice;
}

void DraftBudget::record(int32_t drafts, int32_t tokens, double ms) {
    if (drafts < 0 || static_cast<size_t>(drafts) >= arms_.size() || ms <= 0.0) return;
    Arm& arm = arms_[static_cast<size_t>(drafts)];
    const double sample = static_cast<double>(tokens) / ms;
    arm.rate = arm.samples == 0 ? sample : arm.rate + kDecay * (sample - arm.rate);
    // Step time: the median of the last kWindow steps, so neither a step
    // another process slowed nor a lucky one moves it (an average was pulled
    // around for ten steps by each).
    arm.recent[arm.next] = static_cast<float>(ms);
    arm.next = static_cast<uint8_t>((arm.next + 1) % kWindow);
    arm.filled = static_cast<uint8_t>(std::min<int32_t>(arm.filled + 1, kWindow));
    std::array<float, kWindow> sorted = arm.recent;
    std::sort(sorted.begin(), sorted.begin() + arm.filled);
    arm.ms = arm.filled % 2 ? sorted[arm.filled / 2] : 0.5 * (sorted[arm.filled / 2 - 1] + sorted[arm.filled / 2]);
    arm.sampled_at = step_;
    // A count measured again only because its time was old waits twice as
    // long before the next such try.
    if (drafts == retrying_ && arm.samples > 0) arm.wait = std::min(2 * arm.wait, kMaxRetry);
    retrying_ = -1;
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

double DraftBudget::cost(int32_t drafts) const {
    if (drafts < 0 || static_cast<size_t>(drafts) >= arms_.size()) return 0.0;
    return arms_[static_cast<size_t>(drafts)].ms;
}

double DraftBudget::rate(int32_t drafts) const {
    if (drafts < 0 || static_cast<size_t>(drafts) >= arms_.size()) return 0.0;
    return arms_[static_cast<size_t>(drafts)].rate;
}

}  // namespace liyab
