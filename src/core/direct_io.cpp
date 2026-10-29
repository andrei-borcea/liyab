#include "core/direct_io.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/vfs.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <thread>
#include <vector>

#include "core/log.h"
#include "core/os.h"

#if defined(_WIN32)
#include "core/windows_api.h"
#endif

namespace liyab {

namespace {

#if defined(__linux__)
constexpr long kFuseSuperMagic = 0x65735546;

// Direct reads that "succeed" without delivering the file's bytes have been
// seen through FUSE (Android's /storage/emulated): compare a direct and a
// buffered read of the first and the middle block before trusting O_DIRECT.
bool direct_reads_match(int direct_fd, const std::string& path) {
    const int plain = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (plain < 0) return false;
    struct stat st {};
    bool ok = fstat(plain, &st) == 0;
    const off_t size = ok ? st.st_size : 0;
    constexpr size_t kBlock = DirectFile::kAlign;
    void* a = nullptr;
    void* b = nullptr;
    ok = ok && posix_memalign(&a, kBlock, kBlock) == 0 && posix_memalign(&b, kBlock, kBlock) == 0;
    const off_t offsets[] = {0, size / 2 / static_cast<off_t>(kBlock) * static_cast<off_t>(kBlock)};
    for (const off_t off : offsets) {
        if (!ok || off + static_cast<off_t>(kBlock) > size) break;
        std::memset(a, 0x5A, kBlock);
        const ssize_t rd = pread(direct_fd, a, kBlock, off);
        const ssize_t rb = pread(plain, b, kBlock, off);
        ok = rd == static_cast<ssize_t>(kBlock) && rb == rd && std::memcmp(a, b, kBlock) == 0;
    }
    std::free(a);
    std::free(b);
    ::close(plain);
    return ok;
}
#endif

#if defined(_WIN32)
// Reads up to `length` bytes at `offset` into `dst` and waits for them.
// Returns the bytes read (fewer at the end of the file, 0 at or past it), or
// -1 with the Win32 error in `error`.
//
// Windows serializes every request made through a synchronous handle (the I/O
// manager holds the file object's lock for the whole read), which would run
// read_parallel()'s requests one after another: the handle is opened
// FILE_FLAG_OVERLAPPED, and each request carries its offset and waits on an
// event of its own, since the handle alone cannot tell concurrent requests
// apart.
int64_t overlapped_read(HANDLE file, uint64_t offset, uint8_t* dst, DWORD length, DWORD& error) {
    OVERLAPPED request{};
    request.Offset = static_cast<DWORD>(offset);
    request.OffsetHigh = static_cast<DWORD>(offset >> 32);
    request.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (request.hEvent == nullptr) {
        error = GetLastError();
        return -1;
    }
    DWORD read = 0;
    BOOL ok = ReadFile(file, dst, length, nullptr, &request);
    if (ok || GetLastError() == ERROR_IO_PENDING) ok = GetOverlappedResult(file, &request, &read, TRUE);
    const DWORD status = ok ? ERROR_SUCCESS : GetLastError();
    CloseHandle(request.hEvent);
    if (ok) return read;
    if (status == ERROR_HANDLE_EOF) return 0;
    error = status;
    return -1;
}

// Whether the volume serves unbuffered reads of kAlign-aligned blocks: a
// sector size above kAlign, or a filter driver that refuses them, makes the
// first read fail (ERROR_INVALID_PARAMETER), not the open.
bool unbuffered_reads_work(HANDLE file) {
    constexpr size_t kBlock = DirectFile::kAlign;
    void* block = os::aligned_alloc(kBlock, kBlock);
    if (block == nullptr) return false;
    DWORD error = ERROR_SUCCESS;
    const bool ok = overlapped_read(file, 0, static_cast<uint8_t*>(block), static_cast<DWORD>(kBlock), error) >= 0;
    os::aligned_free(block);
    return ok;
}

HANDLE open_for_reads(const std::wstring& path, DWORD flags) {
    // FILE_SHARE_DELETE: the model stays deletable while it is read, as on POSIX.
    return CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                       FILE_FLAG_OVERLAPPED | flags, nullptr);
}
#endif

}  // namespace

#if defined(_WIN32)
Result<std::unique_ptr<DirectFile>> DirectFile::open(const std::string& path) {
    std::unique_ptr<DirectFile> f(new DirectFile());
    f->path_ = path;
    const std::wstring wide = os::widen(path);
    // FILE_FLAG_NO_BUFFERING bypasses the file cache, like O_DIRECT. Every
    // read's file offset, length and buffer address must then be a multiple of
    // the volume's sector size (512 B or 4 KiB): read()'s contract (kAlign)
    // already asks that of the callers.
    HANDLE h = open_for_reads(wide, FILE_FLAG_NO_BUFFERING);
    DWORD error = h == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
    std::string reason;
    if (h == INVALID_HANDLE_VALUE) {
        reason = "refused: " + os::error_text(error);
    } else if (!unbuffered_reads_work(h)) {
        reason = "unbuffered reads fail on this volume";
        CloseHandle(h);
        h = INVALID_HANDLE_VALUE;
    }
    f->direct_ = h != INVALID_HANDLE_VALUE;
    if (!f->direct_) {
        h = open_for_reads(wide, 0);
        error = h == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
        if (h != INVALID_HANDLE_VALUE) {
            LIYAB_LOG_WARN("FILE_FLAG_NO_BUFFERING not used for %s (%s): streamed weights go through the file cache",
                           path.c_str(), reason.c_str());
        }
    }
    if (h == INVALID_HANDLE_VALUE) return Status(ErrorCode::IoError, "cannot open " + path + ": " + os::error_text(error));
    f->handle_ = h;
    return f;
}

DirectFile::~DirectFile() {
    if (handle_ != nullptr) CloseHandle(static_cast<HANDLE>(handle_));
}

Status DirectFile::read(uint64_t offset, size_t length, uint8_t* dst) const {
    // ReadFile takes a 32-bit length: requests are cut at 1 GiB (a multiple of kAlign).
    constexpr size_t kMaxRequest = size_t{1} << 30;
    size_t done = 0;
    while (done < length) {
        const auto want = static_cast<DWORD>(std::min(length - done, kMaxRequest));
        DWORD error = ERROR_SUCCESS;
        const int64_t r = overlapped_read(static_cast<HANDLE>(handle_), offset + done, dst + done, want, error);
        if (r < 0) return Status(ErrorCode::IoError, "read failed in " + path_ + ": " + os::error_text(error));
        done += static_cast<size_t>(r);
        // Short means the end of the file: the caller asked for an aligned
        // tail. Stop here, since an unbuffered read from the unaligned end
        // would be refused rather than return 0.
        if (static_cast<DWORD>(r) < want) break;
    }
    return Status::ok();
}
#else
Result<std::unique_ptr<DirectFile>> DirectFile::open(const std::string& path) {
    std::unique_ptr<DirectFile> f(new DirectFile());
    f->path_ = path;
#if defined(__linux__)
    f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    f->direct_ = f->fd_ >= 0;
    std::string reason = f->fd_ < 0 ? std::string("refused: ") + std::strerror(errno) : std::string();
    if (f->direct_) {
        struct statfs fs {};
        if (fstatfs(f->fd_, &fs) == 0 && static_cast<long>(fs.f_type) == kFuseSuperMagic) {
            reason = "FUSE filesystem";
        } else if (!direct_reads_match(f->fd_, path)) {
            reason = "direct reads disagree with buffered reads";
        }
        if (!reason.empty()) {
            ::close(f->fd_);
            f->fd_ = -1;
            f->direct_ = false;
        }
    }
    if (!f->direct_) {
        LIYAB_LOG_WARN("O_DIRECT not used for %s (%s): streamed weights go through the page cache", path.c_str(),
                       reason.c_str());
        f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    }
#else
    f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
#if defined(__APPLE__)
    f->direct_ = f->fd_ >= 0 && fcntl(f->fd_, F_NOCACHE, 1) == 0;  // bypass the unified buffer cache
#endif
#endif
    if (f->fd_ < 0) return Status(ErrorCode::IoError, "cannot open " + path + ": " + std::strerror(errno));
    return f;
}

Result<std::unique_ptr<DirectFile>> DirectFile::open_buffered(const std::string& path) {
    std::unique_ptr<DirectFile> f(new DirectFile());
    f->path_ = path;
    f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (f->fd_ < 0) return Status(ErrorCode::IoError, "cannot open " + path + ": " + std::strerror(errno));
#if defined(__linux__)
    posix_fadvise(f->fd_, 0, 0, POSIX_FADV_RANDOM);  // experts are read at random: no read-ahead past the range
#endif
    return f;
}

DirectFile::~DirectFile() {
    if (fd_ >= 0) ::close(fd_);
}

Status DirectFile::read(uint64_t offset, size_t length, uint8_t* dst) const {
    size_t done = 0;
    while (done < length) {
        const ssize_t r = pread(fd_, dst + done, length - done, static_cast<off_t>(offset + done));
        if (r < 0 && errno == EINTR) continue;
        if (r < 0) return Status(ErrorCode::IoError, "read failed in " + path_ + ": " + std::strerror(errno));
        if (r == 0) break;  // end of file: the caller asked for an aligned tail
        done += static_cast<size_t>(r);
    }
    return Status::ok();
}
#endif

Status DirectFile::read_parallel(uint64_t offset, size_t length, uint8_t* dst, int32_t parts) const {
    constexpr size_t kMinPart = size_t{4} << 20;
    const auto n = static_cast<size_t>(std::clamp<int64_t>(std::min<int64_t>(parts, static_cast<int64_t>(length / kMinPart)), 1, 16));
    if (n == 1) return read(offset, length, dst);
    const size_t part = (length / n + kAlign - 1) / kAlign * kAlign;
    std::vector<Status> results(n);
    std::vector<std::thread> threads;
    for (size_t i = 1; i < n; ++i) {
        const size_t begin = i * part;
        if (begin >= length) break;
        threads.emplace_back([&, i, begin] {
            results[i] = read(offset + begin, std::min(part, length - begin), dst + begin);
        });
    }
    results[0] = read(offset, std::min(part, length), dst);
    for (std::thread& t : threads) t.join();
    for (const Status& s : results) LIYAB_RETURN_IF_ERROR(s);
    return Status::ok();
}

}  // namespace liyab
