// Liyab — routed-expert streaming for mixture-of-experts models (internal).
//
// A MoE model's routed experts are most of its bytes, but a token uses only
// top-k of them per block. ExpertStore keeps those tensors out of the page
// cache and serves them from a fixed RAM budget instead:
//
//  * Experts are read with direct I/O (O_DIRECT on Linux/Android, F_NOCACHE
//    on Apple) by background threads, one large aligned read per matrix, so
//    the kernel neither duplicates them in the page cache nor evicts the
//    resident (non-expert) weights to make room for them.
//  * A cache entry is one (block, expert): its gate, up and down matrices.
//    Eviction is LFU with periodic halving (frequency + recency), skipping
//    entries in use.
//  * prefetch() queues predicted experts (Transformer applies a block's
//    router to the current hidden state ahead of time); acquire() blocks only
//    for experts that are neither cached nor already loaded in time.
//
// Thread-safety: prefetch/acquire/release/stats may be called from any
// thread; the I/O threads are internal.
#ifndef LIYAB_CORE_EXPERT_STORE_H
#define LIYAB_CORE_EXPERT_STORE_H

#include <array>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#include "core/direct_io.h"
#include "liyab/mmap_loader.h"
#include "liyab/types.h"

namespace liyab {

class ExpertStore {
public:
    struct Stats {
        uint64_t hits = 0;    // in RAM when the forward pass needed it
        uint64_t late = 0;    // predicted, but still in flight when needed (partial stall)
        uint64_t misses = 0;  // not predicted: read once the router chose it
        uint64_t loads = 0;          // experts read from storage
        uint64_t bytes_read = 0;
        double stall_ms = 0.0;       // compute time spent waiting for experts
    };

    // `experts[l]` = {gate, up, down} stacked expert tensors of block l, or
    // all nullptr for a dense block. `budget_bytes` bounds the cache; at
    // least a few entries per block are kept whatever the budget.
    static Result<std::unique_ptr<ExpertStore>> create(const MmapLoader& file,
                                                       std::vector<std::array<const TensorView*, 3>> experts,
                                                       int32_t n_expert, size_t budget_bytes, int32_t io_threads = 2);
    ~ExpertStore();
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;

    // Queues background loads of `experts` of block `layer` (cached or queued
    // ones are skipped). `predicted`: a guess made ahead of time (queued
    // behind earlier work); false: the router already chose them (front of
    // the queue, counted as misses). Never blocks on I/O.
    void prefetch(int32_t layer, std::span<const int32_t> experts, bool predicted = true);
    // The {gate, up, down} views of one expert, pinned until release().
    // Blocks until the expert is in RAM.
    Result<std::array<TensorView, 3>> acquire(int32_t layer, int32_t expert);
    void release(int32_t layer, int32_t expert);
    // Whether acquire() would return without waiting.
    [[nodiscard]] bool ready(int32_t layer, int32_t expert) const;
    // Halves every use counter (call once per generated token): old
    // popularity fades so the cache follows the conversation.
    void age();

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] size_t capacity_bytes() const noexcept { return slot_bytes_ * slots_.size(); }
    [[nodiscard]] size_t entries() const noexcept { return slots_.size(); }
    [[nodiscard]] uint64_t expert_bytes_total() const noexcept { return expert_bytes_total_; }

private:
    enum class State : uint8_t { Empty, Queued, Loading, Ready };
    struct Entry {
        State state = State::Empty;
        int32_t slot = -1;
        int32_t pins = 0;
        bool on_demand = false;  // queued after the router chose it (a miss, even if it arrives in time)
        bool failed = false;     // the last read failed (reported to the waiting acquire())
        float uses = 0.0f;
        uint64_t last_use = 0;
    };
    struct Slot {
        int32_t entry = -1;  // key = layer * n_expert + expert
    };
    struct Segment {      // one matrix of one expert in storage
        uint32_t shard = 0;
        uint64_t offset = 0;  // file offset of the matrix
        size_t bytes = 0;
    };

    ExpertStore() = default;
    void io_loop();
    int32_t take_slot_locked(std::unique_lock<std::mutex>& lock);  // evicts if needed; may wait
    Status read_entry(int32_t key, uint8_t* dst);
    [[nodiscard]] Segment segment(int32_t key, int32_t matrix) const;

    const MmapLoader* file_ = nullptr;
    std::vector<std::array<const TensorView*, 3>> experts_;
    int32_t n_expert_ = 0;
    size_t slot_bytes_ = 0;
    std::array<size_t, 3> slot_offsets_{};  // where each matrix starts inside a slot (max over blocks)
    uint64_t expert_bytes_total_ = 0;
    std::vector<std::unique_ptr<DirectFile>> files_;  // one per file part

    uint8_t* arena_ = nullptr;
    size_t arena_bytes_ = 0;
    std::vector<Slot> slots_;
    std::vector<Entry> entries_;

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;   // I/O threads: queue not empty / stop
    std::condition_variable ready_cv_;  // consumers: an entry became Ready or a slot freed
    std::deque<int32_t> queue_;
    bool stop_ = false;
    uint64_t clock_ = 0;
    Stats stats_;
    std::vector<std::thread> io_threads_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_EXPERT_STORE_H
