// Liyab — x86-64 SIMD feature detection shared by device_detect (reporting)
// and the CPU kernels (run-time dispatch, core/quant_x86.cpp). Internal.
#ifndef LIYAB_CORE_CPU_FEATURES_H
#define LIYAB_CORE_CPU_FEATURES_H

namespace liyab {

// The x86-64 extensions this process can use: the CPUID bit, and for the
// 256-bit ones also the OS saving the YMM state across context switches
// (OSXSAVE and XCR0 bits 1-2): a CPU with AVX2 under an OS that does not save
// those registers would corrupt them on every preemption. All false on other
// architectures. Cheap (a few CPUID instructions: <cpuid.h> with GCC and
// Clang, __cpuidex / _xgetbv with MSVC and clang-cl) and thread-safe.
struct X86Simd {
    bool avx2 = false;
    bool fma = false;
    bool f16c = false;
};
X86Simd detect_x86_simd() noexcept;

}  // namespace liyab

#endif  // LIYAB_CORE_CPU_FEATURES_H
