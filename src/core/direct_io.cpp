#include "core/direct_io.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <thread>
#include <vector>

#include "core/log.h"

namespace liyab {

Result<std::unique_ptr<DirectFile>> DirectFile::open(const std::string& path) {
    std::unique_ptr<DirectFile> f(new DirectFile());
    f->path_ = path;
#if defined(__linux__)
    f->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
    f->direct_ = f->fd_ >= 0;
    if (f->fd_ < 0) {
        LIYAB_LOG_WARN("O_DIRECT refused for %s (%s): streamed weights go through the page cache", path.c_str(),
                       std::strerror(errno));
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
