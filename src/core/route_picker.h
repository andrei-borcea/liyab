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
// of the accelerator being a warm-up that is not counted (it wraps the weights for the GPU); a step is measured
// twice on each route (host, accelerator, host, accelerator, accelerator: a few tens of milliseconds), a block once
// (accelerator, host, accelerator), since a block's CPU pass costs seconds on a large model. Then the class is
// decided for the life of the picker: the accelerator only when its best time per token is below kMargin times the
// CPU's, since the CPU is the numerical reference and costs no heat. Until a class is decided, a block of fewer
// than kMinBlockTokens tokens is too short to measure and runs on the accelerated route.
//
// A CPU pass that measures a block is cut to kHostProbeTokens tokens (limit()).
//
// The schedule advances when a pass is reported, not when a route is handed out: a pass that never reports
// (cancelled, failed, served from a cache) costs nothing.
//
// A decision can be kept and adopted by a later picker for the same model and device (snapshot(), restore()), which
// then measures nothing: the first prompt of a run no longer pays for the CPU's probe pass. Only a clear decision is
// kept (kClear): one the CPU won by a small margin is measured again, since a pass timed while the device was busy
// could have swung it.
//
// Thread-safe.
#ifndef LIYAB_CORE_ROUTE_PICKER_H
#define LIYAB_CORE_ROUTE_PICKER_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>

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

    // A route is clear when the faster one takes less than 1 / kClear of the other's time per token.
    static constexpr double kClear = 1.25;

    // The decisions in a form that can be kept: trivially copyable, no padding.
    struct Snapshot {
        struct Decision {
            double host_best, accelerator_best;
            uint32_t decided, host;
        };
        std::array<Decision, 2> classes{};
    };

    // The route for a pass of `tokens` tokens; while the class is undecided, report() its time.
    Arm next(size_t tokens) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const Class& c = classes_[index(tokens)];
        if (c.decided) return c.host ? Arm::Host : Arm::Accelerator;
        if (tokens < kMinBlockTokens && tokens > kStepTokens) return Arm::Accelerator;
        return plan(index(tokens))[std::min(c.reported, plan_size(index(tokens)) - 1)];
    }

    // How many of a chunk's `tokens` tokens a pass on `arm` takes: all of them, except a CPU pass that measures a block.
    size_t limit(size_t tokens, Arm arm) const {
        return arm == Arm::Host && tokens > kHostProbeTokens && measuring(tokens) ? kHostProbeTokens : tokens;
    }

    // What the class of `tokens` measured and chose, once decided ("" before): for the log.
    std::string summary(size_t tokens) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const Class& c = classes_[index(tokens)];
        if (!c.decided) return {};
        char text[200];
        std::snprintf(text, sizeof text, "%s: CPU %.2f ms/token, accelerator %.2f; chose %s",
                      index(tokens) == 0 ? "steps" : "blocks", c.host_best, c.accelerator_best,
                      c.host ? "the CPU" : "the accelerator");
        return c.restored ? std::string(text) + " (kept from an earlier run)" : std::string(text);
    }

    // The classes decided so far whose decision is clear (the others read as undecided).
    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        Snapshot out;
        for (size_t i = 0; i < classes_.size(); ++i) {
            const Class& c = classes_[i];
            const bool clear = c.decided && (!c.host || c.accelerator_best >= kClear * c.host_best);
            out.classes[i] = {c.host_best, c.accelerator_best, clear ? 1u : 0u, c.host ? 1u : 0u};
        }
        return out;
    }

    // Adopts the decided classes of `kept` that this picker has not decided or started to measure, and ignores
    // the ones whose times are not positive numbers. Returns whether any was adopted.
    bool restore(const Snapshot& kept) {
        std::lock_guard<std::mutex> lock(mutex_);
        bool any = false;
        for (size_t i = 0; i < classes_.size(); ++i) {
            const Snapshot::Decision& d = kept.classes[i];
            Class& c = classes_[i];
            if (d.decided != 1 || c.decided || c.reported > 0 || !(d.host_best > 0.0) || !(d.accelerator_best > 0.0) ||
                !std::isfinite(d.host_best) || !std::isfinite(d.accelerator_best)) {
                continue;
            }
            c.host_best = d.host_best;
            c.accelerator_best = d.accelerator_best;
            c.host = d.host != 0;
            c.decided = c.restored = true;
            any = true;
        }
        return any;
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
        if (++c.reported >= plan_size(index(tokens))) {
            c.host = !(c.accelerator_best < kMargin * c.host_best);
            c.decided = true;
        }
    }

private:
    struct Class {
        size_t reported = 0, accelerator_reported = 0;
        double host_best = std::numeric_limits<double>::infinity();
        double accelerator_best = std::numeric_limits<double>::infinity();
        bool decided = false, host = false, restored = false;
    };
    static size_t index(size_t tokens) { return tokens <= kStepTokens ? 0 : 1; }
    // Steps: host, accelerator (warm-up), host, accelerator, accelerator. Blocks: accelerator (warm-up), host,
    // accelerator.
    static constexpr std::array<Arm, 5> kStepPlan = {Arm::Host, Arm::Accelerator, Arm::Host, Arm::Accelerator,
                                                     Arm::Accelerator};
    static constexpr std::array<Arm, 3> kBlockPlan = {Arm::Accelerator, Arm::Host, Arm::Accelerator};
    static const Arm* plan(size_t cls) { return cls == 0 ? kStepPlan.data() : kBlockPlan.data(); }
    static size_t plan_size(size_t cls) { return cls == 0 ? kStepPlan.size() : kBlockPlan.size(); }
    mutable std::mutex mutex_;
    std::array<Class, 2> classes_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_ROUTE_PICKER_H
