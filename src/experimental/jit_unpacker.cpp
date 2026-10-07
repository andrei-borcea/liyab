#include "liyab/experimental/jit_unpacker.h"

#include <cstring>
#include <vector>

#if defined(__aarch64__) && !(defined(__APPLE__) && defined(__IPHONE_OS_VERSION_MIN_REQUIRED))
#define LIYAB_JIT_AVAILABLE 1
#include <sys/mman.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif
#endif

namespace liyab::experimental {

namespace {

// --- AArch64 instruction encoders (Arm ARM, A64 SIMD&FP / base) -------------
// Vd/Vn/Vm and Xn are register numbers 0..31.
constexpr uint32_t movi_16b(uint32_t vd, uint8_t imm) {  // MOVI Vd.16B, #imm
    return 0x4F00E400u | ((imm >> 5) & 7u) << 16 | (imm & 31u) << 5 | vd;
}
constexpr uint32_t ld1_post16(uint32_t vt, uint32_t xn) { return 0x4CDF7000u | xn << 5 | vt; }  // LD1 {Vt.16B}, [Xn], #16
constexpr uint32_t ld1_post_reg(uint32_t vt, uint32_t xn, uint32_t xm) {  // LD1 {Vt.16B}, [Xn], Xm
    return 0x4CC07000u | xm << 16 | xn << 5 | vt;
}
constexpr uint32_t st1_post16(uint32_t vt, uint32_t xn) { return 0x4C9F7000u | xn << 5 | vt; }  // ST1 {Vt.16B}, [Xn], #16
constexpr uint32_t and_16b(uint32_t vd, uint32_t vn, uint32_t vm) { return 0x4E201C00u | vm << 16 | vn << 5 | vd; }
constexpr uint32_t ushr4_16b(uint32_t vd, uint32_t vn) { return 0x6F0C0400u | vn << 5 | vd; }  // USHR Vd.16B, Vn.16B, #4
constexpr uint32_t zip1_16b(uint32_t vd, uint32_t vn, uint32_t vm) { return 0x4E003800u | vm << 16 | vn << 5 | vd; }
constexpr uint32_t zip2_16b(uint32_t vd, uint32_t vn, uint32_t vm) { return 0x4E007800u | vm << 16 | vn << 5 | vd; }
constexpr uint32_t sub_16b(uint32_t vd, uint32_t vn, uint32_t vm) { return 0x6E208400u | vm << 16 | vn << 5 | vd; }
constexpr uint32_t movz_x(uint32_t xd, uint16_t imm) { return 0xD2800000u | uint32_t{imm} << 5 | xd; }  // MOVZ Xd, #imm
constexpr uint32_t add_x_imm(uint32_t xd, uint32_t xn, uint32_t imm12) { return 0x91000000u | imm12 << 10 | xn << 5 | xd; }
constexpr uint32_t kRet = 0xD65F03C0u;

// Register plan (AAPCS64: x0 = src, x1 = dst; v0-v7 and v16-v31 are
// caller-saved, so no spills are needed). Two register sets alternate so
// consecutive groups have no false dependencies and can overlap.
constexpr uint32_t kMask = 30;   // v30 = 0x0F
constexpr uint32_t kEight = 31;  // v31 = 8

std::vector<uint32_t> emit(JitUnpacker::Layout layout, size_t elements) {
    std::vector<uint32_t> code;
    const size_t groups = elements / 32;
    code.reserve(groups * 9 + 6);
    code.push_back(movi_16b(kMask, 0x0F));
    code.push_back(movi_16b(kEight, 8));
    if (layout == JitUnpacker::Layout::Q4_0) {
        code.push_back(movz_x(9, 18));      // x9 = block stride
        code.push_back(add_x_imm(0, 0, 2)); // skip the first block's fp16 scale
    }
    for (size_t g = 0; g < groups; ++g) {
        const uint32_t base = (g & 1) ? 16 : 0;  // v0-v4 or v16-v20
        const uint32_t in = base, lo = base + 1, hi = base + 2, a = base + 3, b = base + 4;
        if (layout == JitUnpacker::Layout::Interleaved) {
            code.push_back(ld1_post16(in, 0));
            code.push_back(and_16b(lo, in, kMask));
            code.push_back(ushr4_16b(hi, in));
            code.push_back(zip1_16b(a, lo, hi));  // restore element order
            code.push_back(zip2_16b(b, lo, hi));
            code.push_back(sub_16b(a, a, kEight));
            code.push_back(sub_16b(b, b, kEight));
            code.push_back(st1_post16(a, 1));
            code.push_back(st1_post16(b, 1));
        } else {
            code.push_back(ld1_post_reg(in, 0, 9));  // 16 nibble bytes, then advance one 18-byte block
            code.push_back(and_16b(lo, in, kMask));  // elements 0..15
            code.push_back(ushr4_16b(hi, in));       // elements 16..31
            code.push_back(sub_16b(lo, lo, kEight));
            code.push_back(sub_16b(hi, hi, kEight));
            code.push_back(st1_post16(lo, 1));
            code.push_back(st1_post16(hi, 1));
        }
    }
    code.push_back(kRet);
    return code;
}

}  // namespace

bool JitUnpacker::supported() noexcept {
#if defined(LIYAB_JIT_AVAILABLE)
    return true;
#else
    return false;
#endif
}

Result<std::unique_ptr<JitUnpacker>> JitUnpacker::compile(Layout layout, size_t num_elements) {
    if (num_elements == 0 || num_elements % 32 != 0 || num_elements > kMaxElements) {
        return Status(ErrorCode::InvalidArgument, "JIT unpacker needs a multiple of 32 elements up to kMaxElements");
    }
#if !defined(LIYAB_JIT_AVAILABLE)
    return Status(ErrorCode::Unsupported, "runtime code generation is unavailable on this platform (needs AArch64; "
                                          "iOS forbids it)");
#else
    const std::vector<uint32_t> code = emit(layout, num_elements);
    const size_t bytes = code.size() * sizeof(uint32_t);
    const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const size_t mapped = (bytes + page - 1) / page * page;

#if defined(__APPLE__)
    // Apple Silicon: JIT pages must be MAP_JIT; writability is toggled per
    // thread, so the region is never W and X for this thread simultaneously.
    void* mem = mmap(nullptr, mapped, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (mem == MAP_FAILED) return Status(ErrorCode::Unsupported, "mmap(MAP_JIT) failed");
    pthread_jit_write_protect_np(0);
    std::memcpy(mem, code.data(), bytes);
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(mem, bytes);
#else
    void* mem = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mem == MAP_FAILED) return Status(ErrorCode::OutOfMemory, "mmap for JIT code failed");
    std::memcpy(mem, code.data(), bytes);
    if (mprotect(mem, mapped, PROT_READ | PROT_EXEC) != 0) {  // W^X: drop write before allowing execute
        munmap(mem, mapped);
        return Status(ErrorCode::Unsupported, "mprotect(PROT_READ | PROT_EXEC) refused (SELinux execmem policy?)");
    }
    __builtin___clear_cache(static_cast<char*>(mem), static_cast<char*>(mem) + bytes);
#endif

    std::unique_ptr<JitUnpacker> jit(new JitUnpacker());
    jit->memory_ = mem;
    jit->mapped_bytes_ = mapped;
    jit->code_bytes_ = bytes;
    jit->elements_ = num_elements;
    jit->layout_ = layout;
    jit->fn_ = reinterpret_cast<Fn>(mem);
    return jit;
#endif
}

JitUnpacker::~JitUnpacker() {
#if defined(LIYAB_JIT_AVAILABLE)
    if (memory_ != nullptr) munmap(memory_, mapped_bytes_);
#endif
}

}  // namespace liyab::experimental
