// Liyab — routed-expert streaming for mixture-of-experts models (internal).
//
// A MoE model's routed experts are most of its bytes, but a token uses only
// top-k of them per block. ExpertStore keeps those tensors out of the page
// cache and serves them from a fixed RAM budget instead:
//
//  * Experts are read with direct I/O (O_DIRECT on Linux/Android, F_NOCACHE
//    on Apple) by background threads, one large aligned read per matrix, so
//    the kernel neither duplicates them in the page cache nor evicts the
//    resident (non-expert) weights to make room for them. The thread that
//    starts an expert reads its first matrix and hands the other two to idle
//    threads, which take them before any new expert: an expert the forward
//    pass waits for arrives after one matrix's latency, not three.
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
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <utility>
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
        uint64_t unused = 0;         // loaded experts evicted before any use (wrong predictions)
        uint64_t dropped = 0;        // wrong guesses removed from the queue before they were read
        uint64_t bytes_read = 0;
        double stall_ms = 0.0;       // compute time spent waiting for experts
    };

    // `experts[l]` = {gate, up, down} stacked expert tensors of block l, or
    // all nullptr for a dense block. `repack`: rearrange the matrices of an
    // expert read for a multi-token pass (set_batch) into the CPU's i8mm
    // layout as they arrive (I/O thread; Q4_K/Q5_K/Q6_K to *_R8, Q8_0 to
    // Q8_0_R4; lossless, same size), as the resident weights are at load:
    // the batched kernels then multiply it by many tokens at once instead of
    // decoding each weight row again per token (prefill). Experts read for
    // single-token decoding stay in the file's layout: there the repacking
    // only delayed the read a miss waits for (decode was 6% slower). `budget_bytes` bounds the cache; at
    // least a few entries per block are kept whatever the budget.
    static Result<std::unique_ptr<ExpertStore>> create(const MmapLoader& file,
                                                       std::vector<std::array<const TensorView*, 3>> experts,
                                                       int32_t n_expert, size_t budget_bytes, int32_t io_threads = 2,
                                                       bool repack = false);
    ~ExpertStore();
    ExpertStore(const ExpertStore&) = delete;
    ExpertStore& operator=(const ExpertStore&) = delete;

    // Queues background loads of `experts` of block `layer` (cached or queued
    // ones are skipped). `predicted`: a guess made ahead of time (queued
    // behind earlier work); false: the router already chose them (front of
    // the queue, counted as misses), which also settles the block's guesses:
    // those not chosen are dropped from the queue if still waiting, and
    // become the first to evict if already read (a wrong guess kept its
    // prefetch bonus and pushed out experts that were in use). Never blocks
    // on I/O.
    void prefetch(int32_t layer, std::span<const int32_t> experts, bool predicted = true);
    // The {gate, up, down} views of one expert, pinned until release().
    // Blocks until the expert is in RAM.
    Result<std::array<TensorView, 3>> acquire(int32_t layer, int32_t expert);
    void release(int32_t layer, int32_t expert);
    // Whether acquire() would return without waiting.
    [[nodiscard]] bool ready(int32_t layer, int32_t expert) const;
    // Whether the forward pass now running covers several tokens (see `repack`).
    void set_batch(bool batch) noexcept { batch_.store(batch, std::memory_order_relaxed); }

    // Halves every use counter (call once per generated token): old
    // popularity fades so the cache follows the conversation.
    void age();
    // Empties the cache and returns its pages to the OS, e.g. while the app
    // sits in the background: queued guesses are dropped, reads in flight and
    // pinned experts are waited for. The arena stays allocated; its pages
    // come back as experts load again. Returns the bytes of cached experts
    // released.
    size_t trim();

    // The hot list: the cached experts by use (most used first), recorded by
    // trim() and kept across it, so warm() can refill the cache in the
    // background (behind any demand read) when the model is used again: one
    // pass of large reads instead of a miss per expert. Keys are
    // layer * n_expert + expert; set_hot() restores a saved list.
    [[nodiscard]] std::vector<int32_t> hot_keys() const;
    void set_hot(std::vector<int32_t> keys);
    size_t warm();

    [[nodiscard]] Stats stats() const;
    [[nodiscard]] size_t capacity_bytes() const noexcept { return arena_bytes_; }
    [[nodiscard]] size_t entries() const noexcept { return slots_.size(); }
    [[nodiscard]] uint64_t expert_bytes_total() const noexcept { return expert_bytes_total_; }

private:
    enum class State : uint8_t { Empty, Queued, Loading, Ready };
    struct Entry {
        State state = State::Empty;
        int32_t slot = -1;
        int32_t pins = 0;
        bool on_demand = false;  // queued after the router chose it (a miss, even if it arrives in time)
        int8_t parts_left = 0;   // Loading: matrices not read yet
        bool part_failed = false;
        bool failed = false;     // the last read failed (reported to the waiting acquire())
        bool untouched = false;  // loaded, not acquired since (evicting it wasted the read)
        bool guess = false;      // queued by a prediction its block's router has not confirmed yet
        bool for_batch = false;  // queued while a multi-token pass ran: repacked as it is read
        bool repacked = false;   // its matrices are in the repacked types (served_type)
        float uses = 0.0f;
        uint64_t last_use = 0;
    };
    struct Slot {
        int32_t entry = -1;  // key = layer * n_expert + expert
        size_t offset = 0;   // in the arena
        int32_t size_class = 0;
    };
    // Slots for the experts of blocks with one matrix layout (sizes and offsets).
    struct SizeClass {
        size_t bytes = 0;
        std::array<size_t, 3> offsets{};  // gate, up, down inside a slot
        size_t first_slot = 0;
        size_t n_slots = 0;
    };
    struct Segment {      // one matrix of one expert in storage
        uint32_t shard = 0;
        uint64_t offset = 0;  // file offset of the matrix
        size_t bytes = 0;
    };

    ExpertStore() = default;
    void io_loop();
    // A free slot of `size_class`, evicting within the class if needed; may wait.
    int32_t take_slot_locked(std::unique_lock<std::mutex>& lock, int32_t size_class);
    // Reads matrix `m` of entry `key` (Loading, slot assigned) without the
    // lock, then marks the entry Ready (or Empty on failure) after its last part.
    void read_part(std::unique_lock<std::mutex>& lock, int32_t key, int32_t m);
    [[nodiscard]] Segment segment(int32_t key, int32_t matrix) const;
    // The type matrix `matrix` of block `layer` is served as (its repacked type, or its own).
    [[nodiscard]] DType served_type(int32_t layer, int32_t matrix) const;
    // Drops or demotes block `layer`'s guesses that its router did not confirm.
    void settle_locked(int32_t layer);

    const MmapLoader* file_ = nullptr;
    std::vector<std::array<const TensorView*, 3>> experts_;
    int32_t n_expert_ = 0;
    std::vector<SizeClass> classes_;
    std::vector<int32_t> block_class_;  // per block: its size class, -1 without experts
    uint64_t expert_bytes_total_ = 0;
    std::vector<std::unique_ptr<DirectFile>> files_;  // one per file part

    uint8_t* arena_ = nullptr;
    size_t arena_bytes_ = 0;
    bool locked_ = false;  // mlock succeeded (until trim() unlocks it)
    std::vector<Slot> slots_;
    std::vector<Entry> entries_;
    std::vector<int32_t> hot_;  // see hot_keys()
    bool repack_ = false;
    std::atomic<bool> batch_{false};

    mutable std::mutex mutex_;
    std::condition_variable work_cv_;   // I/O threads: queue or parts not empty / stop
    std::condition_variable ready_cv_;  // consumers: an entry became Ready or a slot freed
    std::deque<int32_t> queue_;                   // entries to load, by priority
    std::deque<std::pair<int32_t, int32_t>> parts_;  // {entry, matrix} of entries being loaded
    bool stop_ = false;
    uint64_t clock_ = 0;
    Stats stats_;
    std::vector<std::thread> io_threads_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_EXPERT_STORE_H
