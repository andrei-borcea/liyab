#include "liyab/experimental/io_uring_loader.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#ifndef __NR_io_uring_setup
#define __NR_io_uring_setup 425  // identical on every architecture since Linux 5.1
#endif
#ifndef __NR_io_uring_enter
#define __NR_io_uring_enter 426
#endif
#endif

namespace liyab::experimental {

namespace {

constexpr size_t kAlign = 4096;  // covers 512 B and 4 KiB logical blocks (UFS uses 4 KiB)

uint64_t align_down(uint64_t v) { return v & ~uint64_t{kAlign - 1}; }
uint64_t align_up(uint64_t v) { return (v + kAlign - 1) & ~uint64_t{kAlign - 1}; }

std::string errno_text(const char* what, int err) { return std::string(what) + ": " + std::strerror(err); }

}  // namespace

const char* io_method_name(IoMethod method) noexcept {
    switch (method) {
        case IoMethod::IoUring: return "io_uring";
        case IoMethod::PreadDirect: return "pread+O_DIRECT";
        case IoMethod::PreadNoCache: return "pread+F_NOCACHE";
        case IoMethod::PreadBuffered: return "pread (page cache)";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// AlignedBuffer
// ---------------------------------------------------------------------------
Result<AlignedBuffer> AlignedBuffer::allocate(size_t bytes, size_t alignment) {
    if (bytes == 0 || alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return Status(ErrorCode::InvalidArgument, "aligned buffer needs a size and a power-of-two alignment");
    }
    void* p = nullptr;
    if (posix_memalign(&p, std::max(alignment, sizeof(void*)), bytes) != 0) {
        return Status(ErrorCode::OutOfMemory, "posix_memalign failed");
    }
    AlignedBuffer buffer;
    buffer.data_ = static_cast<uint8_t*>(p);
    buffer.size_ = bytes;
    return buffer;
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this != &other) {
        std::free(data_);
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

AlignedBuffer::~AlignedBuffer() { std::free(data_); }

// ---------------------------------------------------------------------------
// io_uring ring (raw syscalls; layout per include/uapi/linux/io_uring.h)
// ---------------------------------------------------------------------------
#if defined(__linux__)
struct DirectReader::Ring {
    int fd = -1;
    void* sq_ptr = MAP_FAILED;
    size_t sq_len = 0;
    void* cq_ptr = MAP_FAILED;
    size_t cq_len = 0;
    io_uring_sqe* sqes = static_cast<io_uring_sqe*>(MAP_FAILED);
    size_t sqes_len = 0;
    unsigned* sq_tail = nullptr;
    unsigned* sq_mask = nullptr;
    unsigned* sq_array = nullptr;
    unsigned* cq_head = nullptr;
    unsigned* cq_tail = nullptr;
    unsigned* cq_mask = nullptr;
    io_uring_cqe* cqes = nullptr;

    ~Ring() {
        if (sqes != MAP_FAILED) munmap(sqes, sqes_len);
        if (cq_ptr != MAP_FAILED && cq_ptr != sq_ptr) munmap(cq_ptr, cq_len);
        if (sq_ptr != MAP_FAILED) munmap(sq_ptr, sq_len);
        if (fd >= 0) close(fd);
    }

    // Returns 0 or an errno value.
    int init(unsigned entries) {
        io_uring_params p{};
        fd = static_cast<int>(syscall(__NR_io_uring_setup, entries, &p));
        if (fd < 0) return errno;
        sq_len = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_len = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        const bool single = (p.features & IORING_FEAT_SINGLE_MMAP) != 0;
        if (single) sq_len = cq_len = std::max(sq_len, cq_len);
        sq_ptr = mmap(nullptr, sq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
        if (sq_ptr == MAP_FAILED) return errno;
        cq_ptr = single ? sq_ptr
                        : mmap(nullptr, cq_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
        if (cq_ptr == MAP_FAILED) return errno;
        sqes_len = p.sq_entries * sizeof(io_uring_sqe);
        sqes = static_cast<io_uring_sqe*>(
            mmap(nullptr, sqes_len, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES));
        if (sqes == MAP_FAILED) return errno;

        auto* sq = static_cast<uint8_t*>(sq_ptr);
        auto* cq = static_cast<uint8_t*>(cq_ptr);
        sq_tail = reinterpret_cast<unsigned*>(sq + p.sq_off.tail);
        sq_mask = reinterpret_cast<unsigned*>(sq + p.sq_off.ring_mask);
        sq_array = reinterpret_cast<unsigned*>(sq + p.sq_off.array);
        cq_head = reinterpret_cast<unsigned*>(cq + p.cq_off.head);
        cq_tail = reinterpret_cast<unsigned*>(cq + p.cq_off.tail);
        cq_mask = reinterpret_cast<unsigned*>(cq + p.cq_off.ring_mask);
        cqes = reinterpret_cast<io_uring_cqe*>(cq + p.cq_off.cqes);
        return 0;
    }

    // Queues one read; the caller keeps at most `entries` requests in flight.
    void push_read(int file, void* buf, unsigned len, uint64_t offset, uint64_t user_data) {
        const unsigned tail = *sq_tail;  // only this thread writes the SQ tail
        const unsigned index = tail & *sq_mask;
        io_uring_sqe* sqe = &sqes[index];
        std::memset(sqe, 0, sizeof *sqe);
        sqe->opcode = IORING_OP_READ;
        sqe->fd = file;
        sqe->addr = reinterpret_cast<uint64_t>(buf);
        sqe->len = len;
        sqe->off = offset;
        sqe->user_data = user_data;
        sq_array[index] = index;
        __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);  // publish the SQE to the kernel
    }

    // Submits `to_submit` SQEs and waits for at least `wait_nr` completions.
    int enter(unsigned to_submit, unsigned wait_nr) {
        for (;;) {
            const long r = syscall(__NR_io_uring_enter, fd, to_submit, wait_nr, IORING_ENTER_GETEVENTS, nullptr, 0);
            if (r >= 0) return 0;
            if (errno != EINTR) return errno;
        }
    }

    bool pop(io_uring_cqe& out) {
        const unsigned head = *cq_head;
        if (head == __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE)) return false;
        out = cqes[head & *cq_mask];
        __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);  // hand the slot back
        return true;
    }
};
#else
struct DirectReader::Ring {};
#endif

// ---------------------------------------------------------------------------
// DirectReader
// ---------------------------------------------------------------------------
Result<std::unique_ptr<DirectReader>> DirectReader::open(const std::string& path, const DirectReaderOptions& options) {
    std::unique_ptr<DirectReader> reader(new DirectReader());
    reader->options_ = options;
    reader->options_.queue_depth = std::clamp<uint32_t>(options.queue_depth, 1, 256);
    reader->options_.chunk_bytes = static_cast<size_t>(align_up(std::max<size_t>(options.chunk_bytes, kAlign)));

#if defined(__linux__)
    if (options.bypass_page_cache) {
        reader->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (reader->fd_ >= 0) {
            reader->method_ = IoMethod::PreadDirect;
        } else if (errno == EINVAL) {
            reader->fallback_reason_ = "O_DIRECT not supported by this filesystem";
        } else {
            return Status(ErrorCode::IoError, errno_text(("cannot open '" + path + "'").c_str(), errno));
        }
    }
#endif
    if (reader->fd_ < 0) {
        reader->fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (reader->fd_ < 0) return Status(ErrorCode::IoError, errno_text(("cannot open '" + path + "'").c_str(), errno));
#if defined(__APPLE__)
        if (options.bypass_page_cache && fcntl(reader->fd_, F_NOCACHE, 1) == 0) reader->method_ = IoMethod::PreadNoCache;
#endif
    }
    struct stat st {};
    if (fstat(reader->fd_, &st) != 0) return Status(ErrorCode::IoError, errno_text("fstat", errno));
    reader->file_size_ = static_cast<uint64_t>(st.st_size);

    for (uint32_t i = 0; i < reader->options_.queue_depth; ++i) {
        auto slot = AlignedBuffer::allocate(reader->options_.chunk_bytes, kAlign);
        if (!slot) return slot.status();
        reader->slots_.push_back(std::move(slot).value());
    }

#if defined(__linux__)
    if (options.use_io_uring) {
        auto ring = std::make_unique<Ring>();
        if (const int err = ring->init(reader->options_.queue_depth); err != 0) {
            reader->fallback_reason_ = errno_text("io_uring_setup", err);
        } else {
            // Probe one read: catches kernels without IORING_OP_READ (< 5.6)
            // and LSM policies that allow setup but deny I/O.
            ring->push_read(reader->fd_, reader->slots_[0].data(), static_cast<unsigned>(kAlign), 0, 0);
            io_uring_cqe cqe{};
            int err2 = ring->enter(1, 1);
            if (err2 == 0 && !ring->pop(cqe)) err2 = EIO;
            if (err2 == 0 && cqe.res < 0) err2 = -cqe.res;
            if (err2 == 0) {
                reader->ring_ = std::move(ring);
                reader->method_ = IoMethod::IoUring;
            } else {
                reader->fallback_reason_ = errno_text("io_uring read", err2);
            }
        }
    }
#else
    if (options.use_io_uring) reader->fallback_reason_ = "io_uring is Linux-only";
#endif
    return reader;
}

DirectReader::~DirectReader() {
    ring_.reset();  // before closing the file it may reference
    if (fd_ >= 0) close(fd_);
}

Status DirectReader::stream(uint64_t offset, uint64_t length, const std::function<bool(const ReadChunk&)>& on_chunk) {
    const uint64_t want_begin = std::min(offset, file_size_);
    const uint64_t want_end = length > file_size_ - want_begin ? file_size_ : want_begin + length;
    if (want_begin >= want_end) return Status::ok();
    // O_DIRECT needs block-aligned offsets and sizes: read the aligned
    // superset and trim each chunk before handing it out.
    const uint64_t begin = align_down(want_begin);
    const uint64_t end = align_up(want_end);
    if (method_ == IoMethod::IoUring) return stream_uring(begin, end, want_begin, want_end, on_chunk);
    return stream_pread(begin, end, want_begin, want_end, on_chunk);
}

Status DirectReader::read_into(uint64_t offset, size_t length, uint8_t* dst) {
    if (offset % kAlign != 0 || length % kAlign != 0 || reinterpret_cast<uintptr_t>(dst) % kAlign != 0) {
        return Status(ErrorCode::InvalidArgument, "read_into needs 4 KiB aligned offset, length and buffer");
    }
#if defined(__linux__)
    if (method_ == IoMethod::IoUring) {
        // Up to queue_depth chunk reads in flight, each DMA-ing into its part of dst.
        Ring& ring = *ring_;
        std::vector<std::pair<uint64_t, unsigned>> pending(options_.queue_depth);  // (dst offset, length) per tag
        std::vector<uint32_t> free_tags;
        for (uint32_t i = options_.queue_depth; i-- > 0;) free_tags.push_back(i);
        uint64_t next = 0;
        unsigned inflight = 0;
        unsigned to_submit = 0;
        Status status = Status::ok();
        auto queue = [&](uint32_t tag, uint64_t at, unsigned len) {
            pending[tag] = {at, len};
            ring.push_read(fd_, dst + at, len, offset + at, tag);
            ++inflight;
            ++to_submit;
        };
        while (inflight > 0 || (status.is_ok() && next < length && offset + next < file_size_)) {
            while (status.is_ok() && !free_tags.empty() && next < length && offset + next < file_size_) {
                const uint32_t tag = free_tags.back();
                free_tags.pop_back();
                const auto len = static_cast<unsigned>(std::min<uint64_t>(options_.chunk_bytes, length - next));
                queue(tag, next, len);
                next += len;
            }
            if (const int err = ring.enter(to_submit, 1); err != 0) {
                std::fprintf(stderr, "liyab: io_uring_enter failed with %u requests in flight\n", inflight);
                std::abort();  // the kernel still owns buffers of in-flight reads
            }
            to_submit = 0;
            io_uring_cqe cqe{};
            while (ring.pop(cqe)) {
                --inflight;
                const auto tag = static_cast<uint32_t>(cqe.user_data);
                const auto [at, len] = pending[tag];
                if (cqe.res < 0) {
                    if (status.is_ok()) status = Status(ErrorCode::IoError, errno_text("io_uring read", -cqe.res));
                    free_tags.push_back(tag);
                    continue;
                }
                const auto n = static_cast<unsigned>(cqe.res);
                if (status.is_ok() && n > 0 && n < len && offset + at + n < file_size_) {
                    queue(tag, at + n, len - n);  // short read: fetch the remainder
                } else {
                    free_tags.push_back(tag);
                }
            }
        }
        return status;
    }
#endif
    for (size_t pos = 0; pos < length && offset + pos < file_size_;) {
        const size_t len = std::min(options_.chunk_bytes, length - pos);
        const ssize_t n = pread(fd_, dst + pos, len, static_cast<off_t>(offset + pos));
        if (n < 0) {
            if (errno == EINTR) continue;
            return Status(ErrorCode::IoError, errno_text("pread", errno));
        }
        if (n == 0) break;
        pos += static_cast<size_t>(n);
    }
    return Status::ok();
}

namespace {

// Delivers the part of [pos, pos + n) inside [want_begin, want_end).
bool deliver(const uint8_t* data, uint64_t pos, uint64_t n, uint64_t want_begin, uint64_t want_end,
             const std::function<bool(const ReadChunk&)>& on_chunk) {
    const uint64_t lo = std::max(pos, want_begin);
    const uint64_t hi = std::min(pos + n, want_end);
    if (lo >= hi) return true;
    return on_chunk(ReadChunk{lo, static_cast<size_t>(hi - lo), data + (lo - pos)});
}

}  // namespace

Status DirectReader::stream_pread(uint64_t begin, uint64_t end, uint64_t want_begin, uint64_t want_end,
                                  const std::function<bool(const ReadChunk&)>& on_chunk) {
    AlignedBuffer& buffer = slots_[0];
    for (uint64_t pos = begin; pos < end && pos < file_size_;) {
        const size_t len = static_cast<size_t>(std::min<uint64_t>(options_.chunk_bytes, end - pos));
        const ssize_t n = pread(fd_, buffer.data(), len, static_cast<off_t>(pos));
        if (n < 0) {
            if (errno == EINTR) continue;
            return Status(ErrorCode::IoError, errno_text("pread", errno));
        }
        if (n == 0) break;
        const bool keep_going = deliver(buffer.data(), pos, static_cast<uint64_t>(n), want_begin, want_end, on_chunk);
#if defined(__linux__)
        // Buffered fallback: drop what we just streamed so it does not
        // displace the rest of the page cache.
        if (method_ == IoMethod::PreadBuffered && options_.bypass_page_cache) {
            posix_fadvise(fd_, static_cast<off_t>(pos), n, POSIX_FADV_DONTNEED);
        }
#endif
        if (!keep_going) break;
        pos += static_cast<uint64_t>(n);
    }
    return Status::ok();
}

Status DirectReader::stream_uring(uint64_t begin, uint64_t end, uint64_t want_begin, uint64_t want_end,
                                  const std::function<bool(const ReadChunk&)>& on_chunk) {
#if defined(__linux__)
    Ring& ring = *ring_;
    struct Slot {
        uint64_t offset = 0;
        unsigned length = 0;
    };
    std::vector<Slot> slots(slots_.size());
    std::vector<uint32_t> free_slots;
    for (uint32_t i = static_cast<uint32_t>(slots_.size()); i-- > 0;) free_slots.push_back(i);

    uint64_t next = begin;
    unsigned inflight = 0;
    unsigned to_submit = 0;
    Status status = Status::ok();
    bool stop = false;

    auto queue = [&](uint32_t slot, uint64_t offset, unsigned length) {
        slots[slot] = {offset, length};
        ring.push_read(fd_, slots_[slot].data(), length, offset, slot);
        ++inflight;
        ++to_submit;
    };

    while (inflight > 0 || (!stop && next < end && next < file_size_)) {
        while (!stop && !free_slots.empty() && next < end && next < file_size_) {
            const uint32_t slot = free_slots.back();
            free_slots.pop_back();
            const auto len = static_cast<unsigned>(std::min<uint64_t>(options_.chunk_bytes, end - next));
            queue(slot, next, len);
            next += len;
        }
        if (const int err = ring.enter(to_submit, 1); err != 0) {
            // Requests already queued may still complete; the kernel owns their
            // buffers, so we must not return while any are in flight. Without a
            // working enter() we cannot reap them: fail hard.
            std::fprintf(stderr, "liyab: io_uring_enter failed with %d requests in flight\n", inflight);
            std::abort();
        }
        to_submit = 0;

        io_uring_cqe cqe{};
        while (ring.pop(cqe)) {
            --inflight;
            const auto slot = static_cast<uint32_t>(cqe.user_data);
            const Slot s = slots[slot];
            if (cqe.res < 0) {
                if (status.is_ok()) status = Status(ErrorCode::IoError, errno_text("io_uring read", -cqe.res));
                stop = true;
                free_slots.push_back(slot);
                continue;
            }
            const auto n = static_cast<unsigned>(cqe.res);
            if (!stop && !deliver(slots_[slot].data(), s.offset, n, want_begin, want_end, on_chunk)) stop = true;
            if (!stop && n > 0 && n < s.length && s.offset + n < file_size_) {
                queue(slot, s.offset + n, s.length - n);  // short read: fetch the rest into the same slot
            } else {
                free_slots.push_back(slot);
            }
        }
    }
    return status;
#else
    (void)begin;
    (void)end;
    (void)want_begin;
    (void)want_end;
    (void)on_chunk;
    return Status(ErrorCode::Unsupported, "io_uring is Linux-only");
#endif
}

}  // namespace liyab::experimental
