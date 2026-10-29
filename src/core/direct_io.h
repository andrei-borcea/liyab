// Liyab — direct (uncached) file reads for weight streaming (internal).
//
// Streamed weights are read once per use, so going through the page cache
// only costs memory and evicts weights that are meant to stay resident.
// DirectFile opens with O_DIRECT (Linux / Android), F_NOCACHE (Apple) or
// FILE_FLAG_NO_BUFFERING (Windows) and falls back to buffered reads where
// direct I/O is refused or untrustworthy: on FUSE (Android's
// /storage/emulated, where direct reads were seen to return success without
// the file's bytes), whenever a direct read of the first and middle block
// differs from a buffered one, and on Windows when the volume refuses an
// unbuffered read of the first block (sectors larger than kAlign).
//
// Measured on a Snapdragon 8 Elite phone (UFS 4, f2fs): one reader peaks at
// ~2.6 GB/s with 1 MiB requests and ~3.6 GB/s with 4 MiB; two to four
// concurrent >= 256 KiB requests reach the device's ~4.4 GB/s, random or
// sequential alike, while 16 KiB requests stay below 1.2 GB/s. Hence
// read_parallel(): large requests, a few in flight.
#ifndef LIYAB_CORE_DIRECT_IO_H
#define LIYAB_CORE_DIRECT_IO_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "liyab/types.h"

namespace liyab {

class DirectFile {
public:
    // Alignment of offsets, lengths and buffers for direct I/O: a multiple of
    // the 512 B and 4 KiB logical blocks of phone flash and PC disks alike
    // (Windows' unbuffered reads ask the same of offset, length and buffer).
    static constexpr size_t kAlign = 4096;

    // `path` is UTF-8.
    static Result<std::unique_ptr<DirectFile>> open(const std::string& path);
#if !defined(_WIN32)
    // Ordinary buffered reads: through the page cache, with no read-ahead beyond the requested range. direct() is
    // false. For ExpertStore's page-cache tier, which copies experts the kernel already holds.
    static Result<std::unique_ptr<DirectFile>> open_buffered(const std::string& path);
#endif
    ~DirectFile();
    DirectFile(const DirectFile&) = delete;
    DirectFile& operator=(const DirectFile&) = delete;

    // Reads `length` bytes at `offset` into `dst`. `offset`, `length` and
    // `dst` must be kAlign-aligned; a read reaching past the end of the file
    // stops there (callers size their requests from the file layout).
    // Thread-safe: pread, or on Windows one overlapped ReadFile per request,
    // so concurrent requests on one file run in parallel there too.
    Status read(uint64_t offset, size_t length, uint8_t* dst) const;
    // Same, split into `parts` concurrent requests of >= 4 MiB each.
    Status read_parallel(uint64_t offset, size_t length, uint8_t* dst, int32_t parts) const;

    [[nodiscard]] bool direct() const noexcept { return direct_; }
    [[nodiscard]] const std::string& path() const noexcept { return path_; }

private:
    DirectFile() = default;
#if defined(_WIN32)
    void* handle_ = nullptr;  // HANDLE, opened FILE_FLAG_OVERLAPPED
#else
    int fd_ = -1;
#endif
    bool direct_ = false;
    std::string path_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_DIRECT_IO_H
