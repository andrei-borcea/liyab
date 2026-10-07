// Liyab — triple-buffered weight streaming pipeline.
//
// Three slots rotate through three concurrent stages, so while block i
// executes, block i+1 is being prepared and block i+2 fetched:
//
//   stage 1  fetch thread      storage -> slot  (O_DIRECT / io_uring DMA, or
//                                                a copy out of the mmap)
//   stage 2  transform thread  optional in-place pass over the slot, e.g.
//                              INT4 -> INT8 unpacking for an accelerator
//                              that cannot read packed weights
//   stage 3  consumer          acquire(i) ... execute ... release(i)
//
// Items are consumed cyclically (block 0..N-1, then block 0 again for the
// next token); requesting an item out of order resynchronizes the pipeline.
// Memory is bounded at 3 x the largest item regardless of model size.
//
// What buffering can and cannot do: it hides storage latency behind compute
// as long as the device reads a block at least as fast as it computes one.
// If flash bandwidth is lower than the compute rate, the consumer must wait;
// stats() reports every such stall so it is visible rather than silent.
//
// Liyab's CPU and Metal kernels consume Q4_0/Q4_1/Q8_0 blocks directly
// (nibbles are unpacked in registers), so the engine runs stage 2 as a
// pass-through: an explicit INT4 -> INT8 pass would double the bytes the
// kernels have to read.
#ifndef LIYAB_TRIPLE_BUFFER_LOADER_H
#define LIYAB_TRIPLE_BUFFER_LOADER_H

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "liyab/types.h"

namespace liyab {

class LIYAB_API TripleBufferLoader {
public:
    static constexpr int kSlots = 3;

    struct Item {
        uint64_t offset = 0;  // file offset of the item's first byte
        uint64_t length = 0;
    };
    // Stage 1: fill `dst` with `length` file bytes starting at `offset`.
    // `offset` and `dst` are aligned to `alignment`; `length` is rounded up
    // to it (reading past EOF must be tolerated).
    using FetchFn = std::function<Status(uint64_t offset, size_t length, uint8_t* dst)>;
    // Stage 2: optional in-place transform of one fetched item.
    using TransformFn = std::function<Status(int32_t item, uint8_t* data, size_t length)>;

    struct Stats {
        uint64_t items_fetched = 0;
        double fetch_ms = 0.0;
        double transform_ms = 0.0;
        double consumer_wait_ms = 0.0;  // time stage 3 spent blocked in acquire()
        uint64_t stalls = 0;            // acquire() calls that had to wait
        uint64_t resyncs = 0;           // out-of-order requests
    };

    static Result<std::unique_ptr<TripleBufferLoader>> create(std::vector<Item> items, FetchFn fetch,
                                                              TransformFn transform = {}, size_t alignment = 0);
    ~TripleBufferLoader();
    TripleBufferLoader(const TripleBufferLoader&) = delete;
    TripleBufferLoader& operator=(const TripleBufferLoader&) = delete;

    // Blocks until `item` passed stages 1 and 2; returns its first byte. The
    // pointer stays valid until release(item). At most two items may be
    // held at once (the third slot keeps the pipeline moving).
    Result<const uint8_t*> acquire(int32_t item);
    void release(int32_t item);

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] size_t slot_bytes() const noexcept { return slot_bytes_; }
    [[nodiscard]] int32_t item_count() const noexcept { return static_cast<int32_t>(items_.size()); }

private:
    enum class State { Empty, Fetching, Fetched, Transforming, Ready, InUse };
    struct Slot {
        uint8_t* buffer = nullptr;
        State state = State::Empty;
        int32_t item = -1;
        uint64_t sequence = 0;     // fetch order, so stage 2 keeps it
        uint64_t generation = 0;   // pipeline generation when the fetch started
        size_t head = 0;           // item offset - aligned fetch offset
        Status error;
    };

    TripleBufferLoader() = default;
    void fetch_loop();
    void transform_loop();
    Slot* find(int32_t item);

    std::vector<Item> items_;
    FetchFn fetch_;
    TransformFn transform_;
    size_t alignment_ = 4096;
    size_t slot_bytes_ = 0;
    Slot slots_[kSlots];

    mutable std::mutex mutex_;
    std::condition_variable cv_;
    int32_t fetch_cursor_ = 0;   // next item stage 1 will fetch
    uint64_t sequence_ = 0;
    uint64_t generation_ = 0;    // bumped on resync; stale fetches are dropped
    bool stop_ = false;
    Stats stats_;
    std::thread fetch_thread_;
    std::thread transform_thread_;
};

}  // namespace liyab

#endif  // LIYAB_TRIPLE_BUFFER_LOADER_H
