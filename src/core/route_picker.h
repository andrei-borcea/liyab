// Liyab — choosing between the accelerated route and the CPU alone by timing real passes (internal).
//
// Which is faster depends on the device, the model and the kind of pass, and not in a way a rule can state: on an
// M4 Pro a 27B Q4_K_M decodes at 4.3 tok/s through Metal (each product waits for the GPU, and its Q4_K kernel reads
// 78 GB/s) and at 12.7 on the CPU, while a 1,500-token prompt is processed at 77 tok/s through Metal and 35 on the
// CPU. So the engine measures, once per load, on the work it is asked to do anyway.
//
// Two classes of pass are decided separately: a step (decoding, forced text, a speculative verification: at most
// kStepTokens tokens, which one decision serves so that drafts are verified on the numerics of plain decoding) and a
// block (a prompt or a prefill chunk). Per class the first passes alternate between the two routes, the first pass
// of the accelerator being a warm-up that is not counted (it wraps the weights for the GPU); then the class is
// decided for the life of the picker: the accelerator only when its best time per token is below kMargin times the
// CPU's, since the CPU is the numerical reference and costs no heat. Until a class is decided, a block of fewer
// than kMinBlockTokens tokens is too short to measure and runs on the accelerated route.
//
// A CPU pass that measures a block is cut to kHostProbeTokens tokens (limit()).
//
// The schedule advances when a pass is reported, not when a route is handed out: a pass that never reports
// (cancelled, failed, served from a cache) costs nothing.
//
// Thread-safe.
#ifndef LIYAB_CORE_ROUTE_PICKER_H
#define LIYAB_CORE_ROUTE_PICKER_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <mutex>

namespace liyab {

class RoutePicker {
public:
    enum class Arm : unsigned char { Accelerator, Host };
    static constexpr size_t kStepTokens = 16;
    static constexpr size_t kMinBlockTokens = 17;
    static constexpr double kMargin = 0.8;
    // The CPU's time per token does not depend on how many tokens a pass has (it gives the same bits for any split
    // too), and a block pass of 256 tokens costs it seconds on a large model: while a block class is measured, the
    // CPU's passes take at most this many tokens of a chunk (the next pass takes the rest), the accelerator's whole
    // chunks, which is the size it runs at once decided.
    static constexpr size_t kHostProbeTokens = 32;

    // The route for a pass of `tokens` tokens; while the class is undecided, report() its time.
    Arm next(size_t tokens) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const Class& c = classes_[index(tokens)];
        if (c.decided) return c.host ? Arm::Host : Arm::Accelerator;
        if (tokens < kMinBlockTokens && tokens > kStepTokens) return Arm::Accelerator;
        return kPlan[std::min(c.reported, kPlan.size() - 1)];
    }

    // How many of a chunk's `tokens` tokens a pass on `arm` takes: all of them, except a CPU pass that measures a block.
    size_t limit(size_t tokens, Arm arm) const {
        return arm == Arm::Host && tokens > kHostProbeTokens && measuring(tokens) ? kHostProbeTokens : tokens;
    }

    // Whether next() answers by measurement still (so the pass's time is wanted).
    bool measuring(size_t tokens) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return !classes_[index(tokens)].decided;
    }

    // A pass of `tokens` tokens that ran on `arm` took `ms_per_token`.
    void report(size_t tokens, Arm arm, double ms_per_token) {
        std::lock_guard<std::mutex> lock(mutex_);
        Class& c = classes_[index(tokens)];
        if (c.decided) return;
        if (arm == Arm::Accelerator && c.accelerator_reported++ == 0) {
            ++c.reported;  // the warm-up: it wraps the weights for the GPU
            return;
        }
        double& best = arm == Arm::Host ? c.host_best : c.accelerator_best;
        best = std::min(best, ms_per_token);
        if (++c.reported >= kPlan.size()) {
            c.host = !(c.accelerator_best < kMargin * c.host_best);
            c.decided = true;
        }
    }

private:
    struct Class {
        size_t reported = 0, accelerator_reported = 0;
        double host_best = std::numeric_limits<double>::infinity();
        double accelerator_best = std::numeric_limits<double>::infinity();
        bool decided = false, host = false;
    };
    static size_t index(size_t tokens) { return tokens <= kStepTokens ? 0 : 1; }
    // Host, accelerator (warm-up), host, accelerator, accelerator.
    static constexpr std::array<Arm, 5> kPlan = {Arm::Host, Arm::Accelerator, Arm::Host, Arm::Accelerator,
                                                 Arm::Accelerator};
    mutable std::mutex mutex_;
    std::array<Class, 2> classes_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_ROUTE_PICKER_H
