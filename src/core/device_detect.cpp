#include "liyab/device_detect.h"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <thread>

#include "backends/probes.h"
#include "core/thread_pool.h"
#include "liyab/mmap_loader.h"

#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#if defined(__linux__)
#include <sys/auxv.h>
#if defined(__aarch64__)
#include <asm/hwcap.h>
#endif
#endif
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace liyab {

const char* soc_vendor_name(SocVendor vendor) noexcept {
    switch (vendor) {
        case SocVendor::Qualcomm: return "Qualcomm";
        case SocVendor::MediaTek: return "MediaTek";
        case SocVendor::Google: return "Google";
        case SocVendor::Samsung: return "Samsung";
        case SocVendor::Apple: return "Apple";
        case SocVendor::Unknown: break;
    }
    return "Unknown";
}

namespace {

std::string lower_trim(std::string_view s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.remove_prefix(1);
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.remove_suffix(1);
    std::string out(s);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
    return out;
}

bool starts_with_digits_after(std::string_view s, std::string_view prefix) {
    return s.size() > prefix.size() && s.substr(0, prefix.size()) == prefix &&
           std::isdigit(static_cast<unsigned char>(s[prefix.size()]));
}

struct KnownSoc {
    std::string_view id;
    bool exact;  // codenames must match exactly ("sun" is a substring of "samsung")
    SocVendor vendor;
    std::string_view name;
    std::string_view npu;
};

constexpr KnownSoc kKnownSocs[] = {
    {"sm8850", false, SocVendor::Qualcomm, "Snapdragon 8 Elite Gen 5", "Hexagon v81"},
    {"sm8750", false, SocVendor::Qualcomm, "Snapdragon 8 Elite", "Hexagon v79"},
    {"sun", true, SocVendor::Qualcomm, "Snapdragon 8 Elite", "Hexagon v79"},
    {"sm8650", false, SocVendor::Qualcomm, "Snapdragon 8 Gen 3", "Hexagon v75"},
    {"pineapple", true, SocVendor::Qualcomm, "Snapdragon 8 Gen 3", "Hexagon v75"},
    {"sm8550", false, SocVendor::Qualcomm, "Snapdragon 8 Gen 2", "Hexagon v73"},
    {"kalama", true, SocVendor::Qualcomm, "Snapdragon 8 Gen 2", "Hexagon v73"},
    {"mt6993", false, SocVendor::MediaTek, "Dimensity 9500", "MediaTek APU"},
    {"mt6991", false, SocVendor::MediaTek, "Dimensity 9400", "MediaTek APU 890"},
    {"mt6989", false, SocVendor::MediaTek, "Dimensity 9300", "MediaTek APU 790"},
    {"mt6985", false, SocVendor::MediaTek, "Dimensity 9200", "MediaTek APU 690"},
    {"zumapro", true, SocVendor::Google, "Tensor G4", "Google TPU"},
    {"zuma", true, SocVendor::Google, "Tensor G3", "Google TPU"},
    {"gs201", true, SocVendor::Google, "Tensor G2", "Google TPU"},
    {"gs101", true, SocVendor::Google, "Tensor G1", "Google TPU"},
    {"s5e9955", false, SocVendor::Samsung, "Exynos 2500", "Samsung NPU"},
    {"s5e9945", false, SocVendor::Samsung, "Exynos 2400", "Samsung NPU"},
    {"s5e9925", false, SocVendor::Samsung, "Exynos 2200", "Samsung NPU"},
};

SocVendor vendor_fallback(const std::string& id) {
    if (id == "qti" || id == "qcom" || id == "qualcomm" || starts_with_digits_after(id, "sm") ||
        starts_with_digits_after(id, "sdm")) {
        return SocVendor::Qualcomm;
    }
    if (id == "mediatek" || id == "mtk" || starts_with_digits_after(id, "mt")) return SocVendor::MediaTek;
    if (id == "google" || id.find("tensor") != std::string::npos) return SocVendor::Google;
    if (id == "samsung" || id.find("exynos") != std::string::npos || starts_with_digits_after(id, "s5e")) {
        return SocVendor::Samsung;
    }
    if (id == "apple" || id.rfind("apple ", 0) == 0) return SocVendor::Apple;
    return SocVendor::Unknown;
}

#if defined(__ANDROID__)
std::string system_property(const char* name) {
    char value[PROP_VALUE_MAX] = {};
    const int len = __system_property_get(name, value);
    return len > 0 ? std::string(value, static_cast<size_t>(len)) : std::string();
}
#endif

#if defined(__linux__)
std::string cpuinfo_hardware() {
    std::FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (!f) return {};
    char line[512];
    std::string result;
    while (std::fgets(line, sizeof line, f)) {
        if (std::strncmp(line, "Hardware", 8) == 0) {
            if (const char* colon = std::strchr(line, ':')) {
                result = colon + 1;
                while (!result.empty() && (result.back() == '\n' || result.back() == ' ')) result.pop_back();
                while (!result.empty() && result.front() == ' ') result.erase(result.begin());
            }
            break;
        }
    }
    std::fclose(f);
    return result;
}
#endif

#if defined(__APPLE__)
std::string sysctl_string(const char* name) {
    char buffer[256] = {};
    size_t len = sizeof buffer;
    if (sysctlbyname(name, buffer, &len, nullptr, 0) != 0 || len == 0) return {};
    return std::string(buffer, strnlen(buffer, sizeof buffer));
}
[[maybe_unused]] bool sysctl_flag(const char* name) {
    int32_t value = 0;
    size_t len = sizeof value;
    return sysctlbyname(name, &value, &len, nullptr, 0) == 0 && value != 0;
}
#endif


uint64_t total_memory_bytes() {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof mem;
    return sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0 ? mem : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGESIZE);
    return pages > 0 && page > 0 ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(page) : 0;
#endif
}

}  // namespace

SocInfo classify_soc(std::string_view manufacturer, std::string_view model, std::string_view platform,
                     std::string_view hardware) {
    const std::string_view originals[] = {model, platform, hardware, manufacturer};
    std::string ids[4];
    for (size_t i = 0; i < 4; ++i) ids[i] = lower_trim(originals[i]);

    for (const KnownSoc& known : kKnownSocs) {
        for (size_t i = 0; i < 4; ++i) {
            const std::string& id = ids[i];
            const bool match = known.exact ? id == known.id : id.rfind(known.id, 0) == 0;
            if (match) {
                return {known.vendor, std::string(known.name), std::string(originals[i]), std::string(known.npu)};
            }
        }
    }

    SocInfo info;
    for (size_t i = 0; i < 4 && info.vendor == SocVendor::Unknown; ++i) info.vendor = vendor_fallback(ids[i]);
    // Most specific non-empty identifier wins for display.
    for (const std::string_view original : originals) {
        if (!original.empty()) {
            info.model = std::string(original);
            break;
        }
    }
    switch (info.vendor) {
        case SocVendor::Qualcomm: info.name = "Snapdragon (" + info.model + ")"; info.npu = "Hexagon"; break;
        case SocVendor::MediaTek: info.name = "Dimensity (" + info.model + ")"; info.npu = "MediaTek APU"; break;
        case SocVendor::Google: info.name = "Tensor (" + info.model + ")"; info.npu = "Google TPU"; break;
        case SocVendor::Samsung: info.name = "Exynos (" + info.model + ")"; info.npu = "Samsung NPU"; break;
        case SocVendor::Apple:
            info.name = std::string(model.empty() ? "Apple Silicon" : model);
            info.npu = "Apple Neural Engine (Core ML only)";
            break;
        case SocVendor::Unknown: info.name = info.model.empty() ? "unknown" : info.model; break;
    }
    return info;
}

std::vector<BackendKind> rank_backends(const SocInfo& soc, const Accelerators& acc) {
    std::vector<BackendKind> order;
    // NPU first: highest TOPS/W for the integer FFN projections.
    if (soc.vendor == SocVendor::Qualcomm && acc.qnn_htp.available) order.push_back(BackendKind::Qnn);
    if (soc.vendor == SocVendor::MediaTek && acc.neuropilot.available) order.push_back(BackendKind::NeuroPilot);
    // GPU next.
    if (acc.metal.available) order.push_back(BackendKind::Metal);
    if (acc.vulkan.available && soc.vendor != SocVendor::Apple) order.push_back(BackendKind::Vulkan);
    order.push_back(BackendKind::Cpu);
    return order;
}

CpuFeatures detect_cpu() {
    CpuFeatures f;
    f.cores = static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
    f.performance_cores = performance_core_count();
#if defined(__APPLE__) && defined(__aarch64__)
    f.neon = true;
    f.dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    f.i8mm = sysctl_flag("hw.optional.arm.FEAT_I8MM");
    f.fp16 = sysctl_flag("hw.optional.arm.FEAT_FP16");
#elif defined(__linux__) && defined(__aarch64__)
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
    const unsigned long hwcap = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    f.neon = (hwcap & HWCAP_ASIMD) != 0;
    f.dotprod = (hwcap & HWCAP_ASIMDDP) != 0;
    f.fp16 = (hwcap & HWCAP_ASIMDHP) != 0;
    f.i8mm = (hwcap2 & HWCAP2_I8MM) != 0;
#endif
    return f;
}

DeviceInfo detect_device() {
    DeviceInfo info;
#if defined(__ANDROID__)
    const std::string hardware = system_property("ro.hardware");
    info.soc = classify_soc(system_property("ro.soc.manufacturer"), system_property("ro.soc.model"),
                            system_property("ro.board.platform"),
                            hardware.empty() ? cpuinfo_hardware() : hardware);
#elif defined(__APPLE__)
    std::string brand = sysctl_string("machdep.cpu.brand_string");
    const std::string machine = sysctl_string("hw.machine");
    if (brand.empty() && !machine.empty()) brand = "Apple Silicon (" + machine + ")";
    info.soc = classify_soc("Apple", brand, machine, "");
#elif defined(__linux__)
    info.soc = classify_soc("", "", "", cpuinfo_hardware());
#endif
    info.cpu = detect_cpu();
    info.total_memory = total_memory_bytes();
    info.available_memory = available_memory_bytes();
    info.page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));

    info.accelerators.qnn_htp = probes::probe_qnn_htp();
    info.accelerators.neuropilot = probes::probe_neuropilot();
    info.accelerators.vulkan = probes::probe_vulkan();
    info.accelerators.metal = probes::probe_metal();
    info.backend_order = rank_backends(info.soc, info.accelerators);
    return info;
}

std::string describe_device(const DeviceInfo& info) {
    auto gib = [](uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); };
    auto probe = [](const AcceleratorProbe& p) { return std::string(p.available ? "yes" : "no") + " (" + p.detail + ")"; };
    char mem[128];
    std::snprintf(mem, sizeof mem, "%.1f GiB total, %.1f GiB available, %zu KiB pages", gib(info.total_memory),
                  gib(info.available_memory), info.page_size / 1024);

    std::string out;
    out += "SoC:        " + info.soc.name + " [" + soc_vendor_name(info.soc.vendor) + ", " + info.soc.model + "]\n";
    out += "NPU:        " + (info.soc.npu.empty() ? std::string("unknown") : info.soc.npu) + "\n";
    out += "CPU:        " + std::to_string(info.cpu.cores) + " cores (" + std::to_string(info.cpu.performance_cores) +
           " performance), neon=" + (info.cpu.neon ? "1" : "0") + " dotprod=" + (info.cpu.dotprod ? "1" : "0") +
           " i8mm=" + (info.cpu.i8mm ? "1" : "0") + " fp16=" + (info.cpu.fp16 ? "1" : "0") + "\n";
    out += std::string("Memory:     ") + mem + "\n";
    out += "QNN HTP:    " + probe(info.accelerators.qnn_htp) + "\n";
    out += "NeuroPilot: " + probe(info.accelerators.neuropilot) + "\n";
    out += "Vulkan:     " + probe(info.accelerators.vulkan) + "\n";
    out += "Metal:      " + probe(info.accelerators.metal) + "\n";
    out += "Backends:   ";
    for (size_t i = 0; i < info.backend_order.size(); ++i) {
        out += (i ? " -> " : "") + std::string(backend_kind_name(info.backend_order[i]));
    }
    out += "\n";
    return out;
}

}  // namespace liyab
