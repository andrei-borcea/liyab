// Liyab — runtime SoC identification and backend ranking.
#ifndef LIYAB_DEVICE_DETECT_H
#define LIYAB_DEVICE_DETECT_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "liyab/types.h"

namespace liyab {

enum class SocVendor : int32_t { Unknown = 0, Qualcomm, MediaTek, Google, Samsung, Apple };

LIYAB_API const char* soc_vendor_name(SocVendor vendor) noexcept;

struct SocInfo {
    SocVendor vendor = SocVendor::Unknown;
    std::string name;   // marketing name, e.g. "Snapdragon 8 Elite"
    std::string model;  // raw identifier that matched, e.g. "SM8750"
    std::string npu;    // NPU generation if known, e.g. "Hexagon v79"
};

// What the CPU offers for the kernels. On x86-64 the AVX flags are true only
// when the OS also saves the 256-bit registers (XCR0), so they mean "usable".
// The kernels themselves choose by these at run time (quant::kernel_isa()).
struct CpuFeatures {
    bool neon = false;
    bool dotprod = false;  // SDOT/UDOT (ARMv8.2), used by the Q4_0/Q8_0 kernels
    bool i8mm = false;     // SMMLA (ARMv8.6 int8 matrix multiply): repacked Q4_K kernels and the 2x2 tile kernels
    bool fp16 = false;
    bool avx2 = false;     // x86-64 AVX2: the "avx2" kernel set needs it with fma and f16c
    bool fma = false;      // x86-64 FMA3
    bool f16c = false;     // x86-64 F16C (fp16 <-> fp32 conversions, the F16 KV cache)
    int32_t cores = 0;              // logical processors (on Windows, over every processor group)
    // The cores the thread pool sizes itself by: on Linux / Android the CPUs
    // within 70% of the fastest one's max frequency; on Apple and Windows the
    // physical cores of the performance level / highest efficiency class.
    int32_t performance_cores = 0;
};

struct AcceleratorProbe {
    bool available = false;
    std::string detail;  // device name or reason it is unavailable
};

struct Accelerators {
    AcceleratorProbe qnn_htp;
    AcceleratorProbe neuropilot;
    AcceleratorProbe vulkan;
    AcceleratorProbe metal;
};

struct DeviceInfo {
    SocInfo soc;
    CpuFeatures cpu;
    Accelerators accelerators;
    uint64_t total_memory = 0;
    uint64_t available_memory = 0;
    size_t page_size = 0;
    // Best first; always ends with BackendKind::Cpu.
    std::vector<BackendKind> backend_order;
};

// Inspects the running device (system properties, sysctl, /proc, runtime
// libraries; on Windows the processor name the registry records, the memory
// status and the processor topology). Safe to call from any thread; takes a
// few milliseconds.
LIYAB_API DeviceInfo detect_device();
// The CPU part only (feature registers / sysctl and core counts): cheap,
// thread-safe.
LIYAB_API CpuFeatures detect_cpu();

// Pure classifier over identifiers as exposed by the OS. Any argument may be
// empty. On Android pass ro.soc.manufacturer, ro.soc.model, ro.board.platform
// and ro.hardware (or /proc/cpuinfo "Hardware"); on Apple pass "Apple", the
// CPU brand string, hw.machine and ""; on Windows "", the processor name, ""
// and "".
LIYAB_API SocInfo classify_soc(std::string_view manufacturer, std::string_view model, std::string_view platform,
                               std::string_view hardware);

// NPU → GPU → CPU ordering restricted to accelerators present on the device.
LIYAB_API std::vector<BackendKind> rank_backends(const SocInfo& soc, const Accelerators& accelerators);

LIYAB_API std::string describe_device(const DeviceInfo& info);

}  // namespace liyab

#endif  // LIYAB_DEVICE_DETECT_H
