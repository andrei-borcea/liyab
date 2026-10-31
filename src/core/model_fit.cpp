#include "core/model_fit.h"

#include <algorithm>
#include <cstdio>

#include "core/kernel_conformance.h"
#include "liyab/mmap_loader.h"

namespace liyab {

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

const char* backend_name(BackendKind kind) {
    switch (kind) {
        case BackendKind::Cpu: return "cpu";
        case BackendKind::Metal: return "metal";
        case BackendKind::Vulkan: return "vulkan";
        case BackendKind::Qnn: return "qnn";
        case BackendKind::NeuroPilot: return "neuropilot";
    }
    return "?";
}

std::string format(const char* pattern, double value) {
    char text[48];
    std::snprintf(text, sizeof text, pattern, value);
    return text;
}

}  // namespace

std::vector<DType> formats_to_time(const ModelBytes& model) {
    double total = 0.0;
    for (const auto& entry : model.per_token) total += entry.second;
    const std::vector<DType> known = conformance::weight_formats();
    std::vector<DType> out;
    for (const auto& [type, bytes] : model.per_token) {
        if (bytes >= 0.005 * total && std::find(known.begin(), known.end(), type) != known.end()) out.push_back(type);
    }
    return out;
}

ModelFit describe_model_fit(const std::string& label, const ModelBytes& model, const DeviceInfo& device,
                            const std::vector<BackendSpeed>& backends, const std::vector<DType>& timed) {
    ModelFit fit;
    fit.ceiling.assign(backends.size(), 0.0);
    std::string& out = fit.text;
    char line[256];
    std::snprintf(line, sizeof line, "== model %s ==\narchitecture %s, %lld blocks, %.2f GiB in the file\n", label.c_str(),
                  model.architecture.c_str(), static_cast<long long>(model.blocks),
                  static_cast<double>(model.file_bytes) / kGiB);
    out += line;
    const uint64_t capacity = resident_weight_capacity_bytes(device.available_memory);
    if (capacity >= model.file_bytes) {
        std::snprintf(line, sizeof line, "memory: fits (%.2f GiB available for weights, %.2f GiB needed)\n",
                      static_cast<double>(capacity) / kGiB, static_cast<double>(model.file_bytes) / kGiB);
    } else {
        std::snprintf(line, sizeof line,
                      "memory: %.2f GiB available for weights, %.2f GiB needed: the engine streams layers from storage\n",
                      static_cast<double>(capacity) / kGiB, static_cast<double>(model.file_bytes) / kGiB);
    }
    out += line;
    double per_token = 0.0;
    for (const auto& entry : model.per_token) per_token += entry.second;
    std::snprintf(line, sizeof line, "read per decoded token: %.2f GiB\n\n", per_token / kGiB);
    out += line;

    const auto cpu = std::find_if(backends.begin(), backends.end(), [](const BackendSpeed& b) { return b.kind == BackendKind::Cpu; });
    std::snprintf(line, sizeof line, "%-8s %9s %6s", "format", "GiB/token", "share");
    out += line;
    for (const BackendSpeed& b : backends) {
        std::snprintf(line, sizeof line, "  %-10s", (std::string(backend_name(b.kind)) + " GB/s").c_str());
        out += line;
    }
    out += "\n";
    std::vector<double> seconds(backends.size(), 0.0);
    std::vector<bool> known(backends.size(), true);
    double untimed = 0.0;
    for (const auto& [type, bytes] : model.per_token) {
        if (std::find(timed.begin(), timed.end(), type) == timed.end()) {
            untimed += bytes;
            continue;
        }
        std::snprintf(line, sizeof line, "%-8s %9.3f %5.1f%%", std::string(dtype_traits(type).name).c_str(), bytes / kGiB,
                      100.0 * bytes / per_token);
        out += line;
        for (size_t i = 0; i < backends.size(); ++i) {
            double speed = backends[i].gb_per_s(type);
            std::string text = format("%.1f", speed);
            if (speed <= 0.0 && cpu != backends.end()) {
                speed = cpu->gb_per_s(type);
                text = speed > 0.0 ? "cpu " + format("%.1f", speed) : "none";
            }
            if (speed > 0.0) seconds[i] += bytes / (speed * 1e9);
            else known[i] = false;
            std::snprintf(line, sizeof line, "  %-10s", text.c_str());
            out += line;
        }
        out += "\n";
    }
    if (untimed > 0.0) {
        std::snprintf(line, sizeof line, "(%.1f%% of the bytes are in formats too small to time, left out of the ceilings)\n",
                      100.0 * untimed / per_token);
        out += line;
    }
    out += "\n";
    for (size_t i = 0; i < backends.size(); ++i) {
        if (known[i] && seconds[i] > 0.0) {
            fit.ceiling[i] = 1.0 / seconds[i];
            std::snprintf(line, sizeof line, "decode ceiling, %s: %.1f tokens/s\n", backends[i].name.c_str(), fit.ceiling[i]);
        } else {
            std::snprintf(line, sizeof line, "decode ceiling, %s: unknown (a format of the model has no timing)\n",
                          backends[i].name.c_str());
        }
        out += line;
    }
    return fit;
}

}  // namespace liyab
