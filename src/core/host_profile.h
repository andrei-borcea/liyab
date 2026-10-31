// Liyab — the read speeds of this host, measured once and kept (internal).
//
// What a backend reads per second for each weight format is a property of the host (the memory, the chip, the
// kernels, the thread count), not of a model: the same figures serve every model that holds the format. This keeps
// them in EngineConfig::cache_dir ("host-<id>.bin", core/measured_cache.h) as they are measured, so the first
// question about a model times only the formats it holds that no earlier question did, and later ones time nothing.
// The id covers the SoC, the memory size, the kernel set, the thread count and the backends the host offers.
// Speeds are one-row decoding calls on a matrix larger than the caches (conformance::Options::timing): short,
// indicative timings.
#ifndef LIYAB_CORE_HOST_PROFILE_H
#define LIYAB_CORE_HOST_PROFILE_H

#include <cstdint>
#include <optional>
#include <string>

#include "core/model_fit.h"
#include "liyab/device_detect.h"
#include "liyab/mmap_loader.h"
#include "liyab/types.h"

namespace liyab {

class HostProfile {
public:
    struct Entry {
        uint8_t backend = 0;
        uint8_t type = 0;
        uint16_t reserved = 0;
        float gb_per_s = 0.0f;  // 0: the backend does not run the format
    };
    static constexpr uint32_t kMaxEntries = 96;
    // The kept form: trivially copyable, no padding.
    struct Data {
        uint32_t count = 0;
        uint32_t reserved = 0;
        Entry entries[kMaxEntries];
    };

    // Reads the profile kept in `cache_dir` for this host (in memory for the process when empty).
    HostProfile(const DeviceInfo& device, int threads, std::string cache_dir);

    // The measured speed, or nothing when the backend was not timed on the format.
    [[nodiscard]] std::optional<float> speed(BackendKind backend, DType type) const;
    // Whether any format was timed on the backend (so the host is known to offer it).
    [[nodiscard]] bool knows(BackendKind backend) const;
    void set(BackendKind backend, DType type, float gb_per_s);
    // Keeps what was set since the profile was read.
    void save() const;

private:
    std::string dir_;
    uint64_t id_ = 0;
    Data data_;
    mutable bool dirty_ = false;
};

// How `model_path` will run on this host: times (on the CPU and on each accelerator the host offers) the formats the
// model holds that the profile lacks, keeps them, and describes the fit. `threads` 0: all the cores the engine would
// use. Errors when the file cannot be opened as a model.
Result<ModelFit> fit_model_file(const std::string& model_path, const std::string& cache_dir, int threads = 0);

}  // namespace liyab

#endif  // LIYAB_CORE_HOST_PROFILE_H
