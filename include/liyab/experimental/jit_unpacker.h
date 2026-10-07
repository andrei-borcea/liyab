// Liyab (experimental) — runtime-generated ARM64 NEON INT4 unpackers.
//
// For a fixed element count and weight layout, emits a straight-line AArch64
// routine (fully unrolled: no loop counter, no branches except the final
// `ret`) and maps it executable. Two layouts are supported:
//
//  * Interleaved — byte i holds elements 2i (low nibble) and 2i+1 (high);
//    same semantics as MmapLoader::unpack_int4_to_int8_neon.
//  * Q4_0 — GGML blocks of 18 bytes (fp16 scale + 16 nibble bytes, element j
//    in the low nibble of byte j, element j+16 in the high nibble). The code
//    skips the scales and writes 32 int8 values per block, so no zip is
//    needed: this is the "tailored to the loaded layout" case.
// Output values are offset-binary nibbles minus 8, in [-8, 7].
//
// Memory protection is W^X: the buffer is written while writable and then
// switched to read+execute (Linux/Android: mprotect; macOS: MAP_JIT with
// pthread_jit_write_protect_np). It is never writable and executable at once.
// iOS forbids runtime code generation for App Store apps; compile() returns
// Unsupported there and on non-ARM64 targets.
#ifndef LIYAB_EXPERIMENTAL_JIT_UNPACKER_H
#define LIYAB_EXPERIMENTAL_JIT_UNPACKER_H

#include <cstddef>
#include <cstdint>
#include <memory>

#include "liyab/types.h"

namespace liyab::experimental {

class LIYAB_API JitUnpacker {
public:
    enum class Layout { Interleaved, Q4_0 };
    static constexpr size_t kMaxElements = size_t{1} << 20;  // bounds code size (~1.2 MiB)

    // `num_elements` must be a positive multiple of 32 and <= kMaxElements.
    static Result<std::unique_ptr<JitUnpacker>> compile(Layout layout, size_t num_elements);
    [[nodiscard]] static bool supported() noexcept;
    ~JitUnpacker();
    JitUnpacker(const JitUnpacker&) = delete;
    JitUnpacker& operator=(const JitUnpacker&) = delete;

    // Unpacks num_elements() values. `src` holds num_elements()/2 bytes
    // (Interleaved) or num_elements()/32 blocks (Q4_0).
    void run(const uint8_t* src, int8_t* dst) const noexcept { fn_(src, dst); }

    [[nodiscard]] size_t num_elements() const noexcept { return elements_; }
    [[nodiscard]] size_t code_bytes() const noexcept { return code_bytes_; }
    [[nodiscard]] Layout layout() const noexcept { return layout_; }

private:
    using Fn = void (*)(const uint8_t*, int8_t*);
    JitUnpacker() = default;

    Fn fn_ = nullptr;
    void* memory_ = nullptr;
    size_t mapped_bytes_ = 0;
    size_t code_bytes_ = 0;
    size_t elements_ = 0;
    Layout layout_ = Layout::Interleaved;
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_JIT_UNPACKER_H
