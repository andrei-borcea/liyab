#include "liyab/triple_buffer_loader.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>

namespace liyab {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

}  // namespace

Result<std::unique_ptr<TripleBufferLoader>> TripleBufferLoader::create(std::vector<Item> items, FetchFn fetch,
                                                                       TransformFn transform, size_t alignment,
                                                                       SlotMemory memory) {
    if (items.empty() || !fetch) return Status(ErrorCode::InvalidArgument, "triple buffer needs items and a fetch stage");
    // Page alignment satisfies O_DIRECT (4 KiB blocks) and zero-copy GPU
    // wrapping (Metal requires page-aligned host memory; 16 KiB on Apple).
    const auto page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    if (alignment == 0) alignment = std::max<size_t>(4096, page);
    if ((alignment & (alignment - 1)) != 0) return Status(ErrorCode::InvalidArgument, "alignment must be a power of two");

    std::unique_ptr<TripleBufferLoader> loader(new TripleBufferLoader());
    loader->alignment_ = alignment;
    for (const Item& item : items) {
        if (item.length == 0) return Status(ErrorCode::InvalidArgument, "empty triple-buffer item");
        const uint64_t head = item.offset % alignment;
        const uint64_t span = (head + item.length + alignment - 1) / alignment * alignment;
        loader->slot_bytes_ = std::max<size_t>(loader->slot_bytes_, static_cast<size_t>(span));
    }
    loader->memory_ = std::move(memory);
    for (Slot& slot : loader->slots_) {
        if (loader->memory_.allocate) {
            slot.buffer = loader->memory_.allocate(loader->slot_bytes_);
            if (slot.buffer == nullptr || reinterpret_cast<uintptr_t>(slot.buffer) % alignment != 0) {
                return Status(ErrorCode::OutOfMemory, "cannot allocate triple-buffer slots in the provided memory");
            }
            continue;
        }
        void* p = nullptr;
        if (posix_memalign(&p, alignment, loader->slot_bytes_) != 0) {
            return Status(ErrorCode::OutOfMemory, "cannot allocate triple-buffer slots");
        }
        slot.buffer = static_cast<uint8_t*>(p);
    }
    loader->items_ = std::move(items);
    loader->fetch_ = std::move(fetch);
    loader->transform_ = std::move(transform);
    loader->fetch_thread_ = std::thread([l = loader.get()] { l->fetch_loop(); });
    loader->transform_thread_ = std::thread([l = loader.get()] { l->transform_loop(); });
    return loader;
}

TripleBufferLoader::~TripleBufferLoader() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (fetch_thread_.joinable()) fetch_thread_.join();
    if (transform_thread_.joinable()) transform_thread_.join();
    for (Slot& slot : slots_) {
        if (slot.buffer == nullptr) continue;
        if (memory_.release) memory_.release(slot.buffer);
        else std::free(slot.buffer);
    }
}

TripleBufferLoader::Slot* TripleBufferLoader::find(int32_t item) {
    for (Slot& slot : slots_) {
        if (slot.state != State::Empty && slot.item == item) return &slot;
    }
    return nullptr;
}

void TripleBufferLoader::fetch_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        Slot* slot = nullptr;
        cv_.wait(lock, [&] {
            if (stop_) return true;
            if (find(fetch_cursor_) != nullptr) return false;  // fewer items than slots: all resident
            for (Slot& s : slots_) {
                if (s.state == State::Empty) {
                    slot = &s;
                    return true;
                }
            }
            return false;
        });
        if (stop_) return;

        const int32_t item = fetch_cursor_;
        fetch_cursor_ = (fetch_cursor_ + 1) % static_cast<int32_t>(items_.size());
        slot->state = State::Fetching;
        slot->item = item;
        slot->sequence = ++sequence_;
        slot->generation = generation_;
        const Item& it = items_[static_cast<size_t>(item)];
        const uint64_t aligned = it.offset / alignment_ * alignment_;
        const size_t head = static_cast<size_t>(it.offset - aligned);
        const size_t length = (head + static_cast<size_t>(it.length) + alignment_ - 1) / alignment_ * alignment_;
        uint8_t* buffer = slot->buffer;

        lock.unlock();  // stage 1 runs concurrently with stages 2 and 3
        const auto t0 = Clock::now();
        Status status = fetch_(aligned, length, buffer);
        const double dt = ms_since(t0);
        lock.lock();

        stats_.fetch_ms += dt;
        ++stats_.items_fetched;
        if (slot->generation != generation_) {  // resynchronized meanwhile: stale
            slot->state = State::Empty;
            slot->item = -1;
        } else {
            slot->head = head;
            slot->error = std::move(status);
            slot->state = State::Fetched;
        }
        cv_.notify_all();
    }
}

void TripleBufferLoader::transform_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        Slot* slot = nullptr;
        cv_.wait(lock, [&] {
            if (stop_) return true;
            for (Slot& s : slots_) {  // oldest fetched slot first
                if (s.state == State::Fetched && (slot == nullptr || s.sequence < slot->sequence)) slot = &s;
            }
            return slot != nullptr;
        });
        if (stop_) return;

        slot->state = State::Transforming;
        const int32_t item = slot->item;
        uint8_t* data = slot->buffer + slot->head;
        const auto length = static_cast<size_t>(items_[static_cast<size_t>(item)].length);
        const bool run = transform_ && slot->error.is_ok();

        lock.unlock();
        const auto t0 = Clock::now();
        Status status = run ? transform_(item, data, length) : Status::ok();
        const double dt = ms_since(t0);
        lock.lock();

        stats_.transform_ms += dt;
        if (slot->generation != generation_) {
            slot->state = State::Empty;
            slot->item = -1;
        } else {
            if (!status.is_ok()) slot->error = std::move(status);
            slot->state = State::Ready;
        }
        cv_.notify_all();
    }
}

Result<const uint8_t*> TripleBufferLoader::acquire(int32_t item) {
    if (item < 0 || item >= static_cast<int32_t>(items_.size())) {
        return Status(ErrorCode::InvalidArgument, "triple-buffer item out of range");
    }
    std::unique_lock<std::mutex> lock(mutex_);
    const auto t0 = Clock::now();
    bool waited = false;
    for (;;) {
        Slot* slot = find(item);
        if (slot != nullptr && slot->state == State::InUse) {
            return Status(ErrorCode::InvalidArgument, "triple-buffer item acquired twice");
        }
        if (slot != nullptr && slot->state == State::Ready && slot->generation == generation_) {
            if (waited) {
                ++stats_.stalls;
                stats_.consumer_wait_ms += ms_since(t0);
            }
            if (!slot->error.is_ok()) {
                Status error = slot->error;
                slot->state = State::Empty;
                slot->item = -1;
                cv_.notify_all();
                return error;
            }
            slot->state = State::InUse;
            return static_cast<const uint8_t*>(slot->buffer + slot->head);
        }
        if (slot == nullptr) {
            // Out-of-order request: drop prefetched work and restart at `item`.
            ++generation_;
            ++stats_.resyncs;
            for (Slot& s : slots_) {
                if (s.state == State::Fetched || s.state == State::Ready) {
                    s.state = State::Empty;
                    s.item = -1;
                }
            }
            fetch_cursor_ = item;
            cv_.notify_all();
        }
        waited = true;
        cv_.wait(lock);
        if (stop_) return Status(ErrorCode::Cancelled, "triple-buffer loader stopped");
    }
}

void TripleBufferLoader::release(int32_t item) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (Slot& slot : slots_) {
            if (slot.state == State::InUse && slot.item == item) {
                slot.state = State::Empty;
                slot.item = -1;
            }
        }
    }
    cv_.notify_all();
}

TripleBufferLoader::Stats TripleBufferLoader::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}  // namespace liyab
