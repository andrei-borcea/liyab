// Liyab — how a model will run on this host (internal): its weights, whether they fit, and the decode rate the
// measured read speeds of each backend allow at most.
//
// Decoding is bound by memory bandwidth: a token reads the weights once (see model_bytes.h), so the bytes it reads
// divided by the speed a backend reads them at bound the rate. The speeds come from timing the backend's own kernels
// on the formats the model holds (conformance::check_backend, or the HostProfile that keeps such timings). The
// figure is an upper bound, not a forecast: it leaves out attention over the cache, recurrent state, norms and
// launch costs, so the engine reaches about 80% of it on the M4 Pro; automatic routing still decides by timing real
// passes.
#ifndef LIYAB_CORE_MODEL_FIT_H
#define LIYAB_CORE_MODEL_FIT_H

#include <functional>
#include <string>
#include <vector>

#include "core/model_bytes.h"
#include "liyab/device_detect.h"
#include "liyab/types.h"

namespace liyab {

// One backend's read speeds.
struct BackendSpeed {
    BackendKind kind = BackendKind::Cpu;
    std::string name;                         // Backend::description()
    std::function<double(DType)> gb_per_s;    // weight bytes read per second by a one-row call; 0: it does not run the format
};

struct ModelFit {
    std::string text;             // the report, as printed
    std::vector<double> ceiling;  // per backend: most tokens per second, 0 when a format of the model has no timing
};

// The formats worth timing for `model`: those that make up at least 0.5% of what a token reads (the norms' F32 would
// only add noise) and that the conformance check knows. Largest first.
std::vector<DType> formats_to_time(const ModelBytes& model);

// The report for `label` (the model's path). `backends` holds the CPU, which runs what a backend refuses and is the
// speed such a format is counted at, then the accelerators. `timed` are the formats the speeds cover; the others
// are left out of the ceilings and said so.
ModelFit describe_model_fit(const std::string& label, const ModelBytes& model, const DeviceInfo& device,
                            const std::vector<BackendSpeed>& backends, const std::vector<DType>& timed);

}  // namespace liyab

#endif  // LIYAB_CORE_MODEL_FIT_H
