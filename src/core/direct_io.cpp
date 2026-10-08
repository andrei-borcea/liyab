#include "core/direct_io.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
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

}  // namespace

Result<std::unique_ptr<DirectFile>> DirectFile::open(const std::string& path, bool direct) {
    std::unique_ptr<DirectFile> f(new DirectFile());
    f->path_ = path;
    if (!direct) {
        f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (f->fd_ < 0) return Status(ErrorCode::IoError, "cannot open " + path + ": " + std::strerror(errno));
#if defined(__linux__)
        posix_fadvise(f->fd_, 0, 0, POSIX_FADV_RANDOM);
#endif
        return f;
    }
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
