// Liyab (experimental) — page-cache-bypassing direct I/O for weight streaming.
//
// Reads model blocks from flash into application-owned, 4 KiB-aligned
// buffers (posix_memalign) with O_DIRECT, so streamed weights do not evict
// other data from the OS page cache and no kernel→user copy is made from
// cached pages. On Linux the reads are issued through io_uring with up to
// `queue_depth` requests in flight (raw syscalls, no liburing dependency).
//
// Platform reality (the reader degrades automatically and reports why):
//  * Android 14+ blocks io_uring for regular apps (SELinux), and seccomp
//    filters may block it earlier; the probe then falls back to O_DIRECT
//    pread on a worker thread. Shell / rooted / Linux-SBC processes keep it.
//  * O_DIRECT is unsupported on some filesystems (tmpfs): buffered pread.
//  * Apple platforms have neither: pread with F_NOCACHE (no caching).
//  * DMA-BUF: apps cannot allocate from /dev/dma_heap directly; GPU-visible
//    memory goes through AHardwareBuffer / Vulkan imports. This module stops
//    at aligned host buffers, which those APIs can import.
// This is an alternative to the mmap path, not wired into the engine yet:
// TensorViews would have to point into rotating buffers instead of the map.
#ifndef LIYAB_EXPERIMENTAL_IO_URING_LOADER_H
#define LIYAB_EXPERIMENTAL_IO_URING_LOADER_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "liyab/types.h"

namespace liyab::experimental {

// Owning, aligned host buffer suitable for O_DIRECT and DMA imports.
class LIYAB_API AlignedBuffer {
public:
    static Result<AlignedBuffer> allocate(size_t bytes, size_t alignment = 4096);
    AlignedBuffer() = default;
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;
    ~AlignedBuffer();

    [[nodiscard]] uint8_t* data() noexcept { return data_; }
    [[nodiscard]] const uint8_t* data() const noexcept { return data_; }
    [[nodiscard]] size_t size() const noexcept { return size_; }

private:
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
};

enum class IoMethod { IoUring, PreadDirect, PreadNoCache, PreadBuffered };
LIYAB_API const char* io_method_name(IoMethod method) noexcept;

struct DirectReaderOptions {
    uint32_t queue_depth = 8;          // reads in flight (io_uring)
    size_t chunk_bytes = size_t{1} << 20;  // rounded up to 4 KiB
    bool use_io_uring = true;
    bool bypass_page_cache = true;     // O_DIRECT / F_NOCACHE
};

struct ReadChunk {
    uint64_t offset = 0;          // file offset of data[0]
    size_t length = 0;            // bytes valid in data
    const uint8_t* data = nullptr;  // valid only during the callback
};

class LIYAB_API DirectReader {
public:
    static Result<std::unique_ptr<DirectReader>> open(const std::string& path, const DirectReaderOptions& options = {});
    ~DirectReader();
    DirectReader(const DirectReader&) = delete;
    DirectReader& operator=(const DirectReader&) = delete;

    [[nodiscard]] IoMethod method() const noexcept { return method_; }
    // Why a faster method was not used (empty when the requested one works).
    [[nodiscard]] const std::string& fallback_reason() const noexcept { return fallback_reason_; }
    [[nodiscard]] uint64_t file_size() const noexcept { return file_size_; }

    // Streams [offset, offset + length) (clamped to the file) in chunks.
    // With io_uring, chunks arrive in completion order — use ReadChunk::offset
    // to place them. Return false from `on_chunk` to stop early.
    Status stream(uint64_t offset, uint64_t length, const std::function<bool(const ReadChunk&)>& on_chunk);

    // Reads [offset, offset + length) straight into caller memory — the DMA
    // target for O_DIRECT, no intermediate copy. `offset`, `length` and `dst`
    // must be 4 KiB aligned. Bytes past EOF are left untouched.
    Status read_into(uint64_t offset, size_t length, uint8_t* dst);

private:
    struct Ring;
    DirectReader() = default;
    Status stream_pread(uint64_t begin, uint64_t end, uint64_t want_begin, uint64_t want_end,
                        const std::function<bool(const ReadChunk&)>& on_chunk);
    Status stream_uring(uint64_t begin, uint64_t end, uint64_t want_begin, uint64_t want_end,
                        const std::function<bool(const ReadChunk&)>& on_chunk);

    int fd_ = -1;
    uint64_t file_size_ = 0;
    IoMethod method_ = IoMethod::PreadBuffered;
    std::string fallback_reason_;
    DirectReaderOptions options_;
    std::unique_ptr<Ring> ring_;
    std::vector<AlignedBuffer> slots_;
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_IO_URING_LOADER_H
