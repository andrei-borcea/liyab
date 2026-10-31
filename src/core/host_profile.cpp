#include "core/host_profile.h"

#include <algorithm>
#include <memory>

#include "core/kernel_conformance.h"
#include "core/measured_cache.h"
#include "core/quant.h"
#include "core/thread_pool.h"
#include "liyab/backend.h"

namespace liyab {

namespace {

constexpr uint32_t kProfileVersion = 1;

const MeasuredCache<HostProfile::Data>& profiles() {
    static const MeasuredCache<HostProfile::Data> cache("host", "LYHP", kProfileVersion, 4, "host profile");
    return cache;
}

uint64_t profile_id(const DeviceInfo& device, int threads) {
    uint64_t id = xxhash64(device.soc.name.data(), device.soc.name.size(), kProfileVersion);
    id = xxhash64(device.soc.model.data(), device.soc.model.size(), id);
    const std::string_view isa = quant::kernel_isa();
    id = xxhash64(isa.data(), isa.size(), id);
    uint64_t parts[] = {device.total_memory, static_cast<uint64_t>(threads)};
    id = xxhash64(parts, sizeof parts, id);
    for (const BackendKind kind : device.backend_order) {
        const uint64_t k = static_cast<uint64_t>(kind);
        id = xxhash64(&k, sizeof k, id);
    }
    return id;
}

}  // namespace

HostProfile::HostProfile(const DeviceInfo& device, int threads, std::string cache_dir)
    : dir_(std::move(cache_dir)), id_(profile_id(device, threads)) {
    const auto kept = profiles().find(dir_, id_, [](const Data& d) { return d.count <= kMaxEntries; });
    if (kept) data_ = *kept;
}

std::optional<float> HostProfile::speed(BackendKind backend, DType type) const {
    for (uint32_t i = 0; i < data_.count; ++i) {
        if (data_.entries[i].backend == static_cast<uint8_t>(backend) && data_.entries[i].type == static_cast<uint8_t>(type)) {
            return data_.entries[i].gb_per_s;
        }
    }
    return std::nullopt;
}

bool HostProfile::knows(BackendKind backend) const {
    for (uint32_t i = 0; i < data_.count; ++i) {
        if (data_.entries[i].backend == static_cast<uint8_t>(backend)) return true;
    }
    return false;
}

void HostProfile::set(BackendKind backend, DType type, float gb_per_s) {
    for (uint32_t i = 0; i < data_.count; ++i) {
        if (data_.entries[i].backend == static_cast<uint8_t>(backend) && data_.entries[i].type == static_cast<uint8_t>(type)) {
            data_.entries[i].gb_per_s = gb_per_s;
            dirty_ = true;
            return;
        }
    }
    if (data_.count >= kMaxEntries) return;  // 3 backends x 27 formats fit with room to spare
    data_.entries[data_.count++] = {static_cast<uint8_t>(backend), static_cast<uint8_t>(type), 0, gb_per_s};
    dirty_ = true;
}

void HostProfile::save() const {
    if (!dirty_) return;
    profiles().keep(dir_, id_, data_);
    dirty_ = false;
}

namespace {

// The best one-row speed a report measured for `type` over the placements the backend ran, 0 when it ran none.
float best_decode_speed(const conformance::BackendReport& report, DType type) {
    float best = 0.0f;
    for (const conformance::FormatReport& f : report.formats) {
        if (f.type != type) continue;
        for (const conformance::PlacementReport& p : f.placements) {
            if (p.decode.verdict == conformance::Verdict::Ok || p.decode.verdict == conformance::Verdict::Partial) {
                best = std::max(best, static_cast<float>(p.decode.gb_per_s));
            }
        }
    }
    return best;
}

// Times the formats of `types` the profile lacks on `backend` and records them.
void measure_missing(Backend& backend, BackendKind kind, const std::vector<DType>& types, HostProfile& profile) {
    conformance::Options options;
    options.timing = true;
    for (const DType type : types) {
        if (!profile.speed(kind, type)) options.only.push_back(type);
    }
    if (options.only.empty()) return;
    const conformance::BackendReport report = conformance::check_backend(backend, options);
    for (const DType type : options.only) profile.set(kind, type, best_decode_speed(report, type));
}

}  // namespace

Result<ModelFit> fit_model_file(const std::string& model_path, const std::string& cache_dir, int threads) {
    LoaderOptions loading;
    loading.streaming = false;
    auto loader = MmapLoader::open(model_path, loading);
    if (!loader) return loader.status();
    const ModelBytes model = read_model_bytes(*loader.value());
    const std::vector<DType> timed = formats_to_time(model);

    const DeviceInfo device = detect_device();
    ThreadPool pool(threads);
    HostProfile profile(device, static_cast<int>(pool.max_threads()), cache_dir);

    // The CPU always; an accelerator when the host offers one (the profile has timed it, or it can be created now).
    std::vector<BackendKind> kinds = {BackendKind::Cpu};
    std::vector<std::unique_ptr<Backend>> made(1);
    made[0] = make_cpu_backend(pool);
    for (const BackendKind kind : {BackendKind::Metal, BackendKind::Vulkan}) {
        if (std::find(device.backend_order.begin(), device.backend_order.end(), kind) == device.backend_order.end()) continue;
        std::unique_ptr<Backend> backend;
        const bool all_known = profile.knows(kind) &&
                               std::all_of(timed.begin(), timed.end(), [&](DType t) { return profile.speed(kind, t).has_value(); });
        if (!all_known) {
            auto created = kind == BackendKind::Metal ? make_metal_backend() : make_vulkan_backend();
            if (!created) continue;
            backend = std::move(created).value();
        } else if (!profile.knows(kind)) {
            continue;
        }
        kinds.push_back(kind);
        made.push_back(std::move(backend));
    }
    // A known accelerator whose formats are all in the profile needs no instance.
    for (size_t i = 0; i < kinds.size(); ++i) {
        if (made[i]) measure_missing(*made[i], kinds[i], timed, profile);
    }
    profile.save();

    std::vector<BackendSpeed> speeds;
    for (const BackendKind kind : kinds) {
        BackendSpeed speed;
        speed.kind = kind;
        speed.name = kind == BackendKind::Cpu ? std::string("cpu (") + std::string(quant::kernel_isa()) + ", " +
                                                    std::to_string(pool.max_threads()) + " threads)"
                                              : (kind == BackendKind::Metal ? "metal" : "vulkan");
        speed.gb_per_s = [&profile, kind](DType type) { return static_cast<double>(profile.speed(kind, type).value_or(0.0f)); };
        speeds.push_back(std::move(speed));
    }
    return describe_model_fit(model_path, model, device, speeds, timed);
}

}  // namespace liyab
