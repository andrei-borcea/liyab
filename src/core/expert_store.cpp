#include "core/expert_store.h"

#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>

#include "core/log.h"

namespace liyab {

namespace {

constexpr size_t kAlign = DirectFile::kAlign;

// Acquires (8 per block for a top-8 model) during which a freshly loaded,
// not yet used expert is not evicted.
constexpr uint64_t kFreshTicks = 48;

size_t align_up(size_t v) { return (v + kAlign - 1) / kAlign * kAlign; }

}  // namespace

Result<std::unique_ptr<ExpertStore>> ExpertStore::create(const MmapLoader& file,
                                                         std::vector<std::array<const TensorView*, 3>> experts,
                                                         int32_t n_expert, size_t budget_bytes, int32_t io_threads) {
    if (n_expert <= 0) return Status(ErrorCode::InvalidArgument, "ExpertStore needs n_expert > 0");
    std::unique_ptr<ExpertStore> s(new ExpertStore());
    s->file_ = &file;
    s->experts_ = std::move(experts);
    s->n_expert_ = n_expert;

    // A slot holds the three matrices of any block's expert, each in its own
    // aligned area (an expert's offset is not block-aligned, so one extra
    // alignment unit is reserved per matrix).
    std::array<size_t, 3> area{};
    for (const auto& block : s->experts_) {
        if (block[0] == nullptr) continue;
        for (size_t m = 0; m < 3; ++m) {
            const size_t bytes = block[m]->row_bytes() * static_cast<size_t>(block[m]->ne[1]);
            area[m] = std::max(area[m], align_up(bytes + kAlign));
            s->expert_bytes_total_ += static_cast<uint64_t>(bytes) * static_cast<uint64_t>(n_expert);
        }
    }
    if (area[0] == 0) return Status(ErrorCode::InvalidArgument, "no expert tensors");
    s->slot_offsets_ = {0, area[0], area[0] + area[1]};
    s->slot_bytes_ = area[0] + area[1] + area[2];

    // Enough slots for a block's top-k plus the in-flight loads, whatever the budget.
    const size_t total_entries = s->experts_.size() * static_cast<size_t>(n_expert);
    const size_t min_slots = static_cast<size_t>(std::max(io_threads, 1)) + 34;
    const size_t n_slots = std::clamp(budget_bytes / s->slot_bytes_, std::min(min_slots, total_entries), total_entries);
    s->arena_bytes_ = n_slots * s->slot_bytes_;
    void* arena = nullptr;
    if (posix_memalign(&arena, kAlign, s->arena_bytes_) != 0) {
        return Status(ErrorCode::OutOfMemory, "cannot allocate the expert cache");
    }
    s->arena_ = static_cast<uint8_t*>(arena);
    // Best effort: locked pages are never swapped or compressed (Android
    // usually caps RLIMIT_MEMLOCK, in which case the cache stays pageable).
    s->locked_ = mlock(s->arena_, s->arena_bytes_) == 0;
    s->slots_.resize(n_slots);
    s->entries_.resize(total_entries);

    for (size_t i = 0; i < file.shard_count(); ++i) {
        auto part = DirectFile::open(file.shard(i).path());
        if (!part) return part.status();
        s->files_.push_back(std::move(part).value());
    }
    for (int32_t i = 0; i < std::max(io_threads, 1); ++i) s->io_threads_.emplace_back([p = s.get()] { p->io_loop(); });
    LIYAB_LOG_INFO("expert cache: %zu slots x %.2f MiB = %.2f GiB (%.0f%% of %.2f GiB of experts)%s", n_slots,
                   static_cast<double>(s->slot_bytes_) / (1024.0 * 1024.0),
                   static_cast<double>(s->arena_bytes_) / (1024.0 * 1024.0 * 1024.0),
                   100.0 * static_cast<double>(n_slots) / static_cast<double>(total_entries),
                   static_cast<double>(s->expert_bytes_total_) / (1024.0 * 1024.0 * 1024.0),
                   s->locked_ ? ", locked in RAM" : "");
    return s;
}

ExpertStore::~ExpertStore() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    work_cv_.notify_all();
    ready_cv_.notify_all();
    for (std::thread& t : io_threads_) t.join();
    if (arena_ != nullptr) {
        if (locked_) munlock(arena_, arena_bytes_);
        std::free(arena_);
    }
}

ExpertStore::Segment ExpertStore::segment(int32_t key, int32_t matrix) const {
    const auto& block = experts_[static_cast<size_t>(key / n_expert_)];
    const TensorView& t = *block[static_cast<size_t>(matrix)];
    const size_t bytes = t.row_bytes() * static_cast<size_t>(t.ne[1]);
    return {t.shard, t.file_offset + static_cast<uint64_t>(key % n_expert_) * bytes, bytes};
}

void ExpertStore::read_part(std::unique_lock<std::mutex>& lock, int32_t key, int32_t m) {
    Entry& e = entries_[static_cast<size_t>(key)];
    const Segment seg = segment(key, m);
    const uint64_t begin = seg.offset / kAlign * kAlign;
    const size_t length = align_up(static_cast<size_t>(seg.offset - begin) + seg.bytes);
    uint8_t* dst = arena_ + static_cast<size_t>(e.slot) * slot_bytes_ + slot_offsets_[static_cast<size_t>(m)];
    lock.unlock();
    // One read per matrix: smaller parts (tried: 2 and 4 per matrix) cost
    // the flash more than the extra concurrency saves.
    const Status status = files_[seg.shard]->read(begin, length, dst);
    lock.lock();
    stats_.bytes_read += length;
    if (!status.is_ok()) {
        LIYAB_LOG_ERROR("%s", status.to_string().c_str());
        e.part_failed = true;
    }
    if (--e.parts_left > 0) return;
    if (e.part_failed) {
        e.failed = true;
        e.state = State::Empty;
        slots_[static_cast<size_t>(e.slot)].entry = -1;
        e.slot = -1;
    } else {
        e.state = State::Ready;
        e.untouched = true;
        e.last_use = clock_;
        ++stats_.loads;
    }
    ready_cv_.notify_all();
}

int32_t ExpertStore::take_slot_locked(std::unique_lock<std::mutex>& lock) {
    for (;;) {
        if (stop_) return -1;
        // A fresh load has no uses yet, so plain LFU would evict it first,
        // before the acquire() it was loaded for (which then reads it again).
        // Fresh entries are spared until a few blocks' worth of acquires
        // passed, unless nothing else can go.
        int32_t victim = -1;
        int32_t fresh_victim = -1;
        auto better = [&](size_t candidate, int32_t current) {
            if (current < 0) return true;
            const Entry& e = entries_[static_cast<size_t>(slots_[candidate].entry)];
            const Entry& v = entries_[static_cast<size_t>(slots_[static_cast<size_t>(current)].entry)];
            return e.uses < v.uses || (e.uses == v.uses && e.last_use < v.last_use);
        };
        for (size_t i = 0; i < slots_.size(); ++i) {
            const int32_t key = slots_[i].entry;
            if (key < 0) return static_cast<int32_t>(i);
            const Entry& e = entries_[static_cast<size_t>(key)];
            if (e.state != State::Ready || e.pins > 0) continue;
            int32_t& best = e.untouched && clock_ - e.last_use < kFreshTicks ? fresh_victim : victim;
            if (better(i, best)) best = static_cast<int32_t>(i);
        }
        if (victim < 0) victim = fresh_victim;
        if (victim >= 0) {
            Entry& old = entries_[static_cast<size_t>(slots_[static_cast<size_t>(victim)].entry)];
            if (old.untouched) ++stats_.unused;
            old.untouched = false;
            old.state = State::Empty;
            old.slot = -1;
            slots_[static_cast<size_t>(victim)].entry = -1;
            return victim;
        }
        ready_cv_.wait(lock);  // every slot pinned or loading: wait for a release
    }
}

void ExpertStore::io_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        work_cv_.wait(lock, [this] { return stop_ || !queue_.empty() || !parts_.empty(); });
        if (stop_) return;
        if (!parts_.empty()) {  // finish started experts first
            const auto [key, m] = parts_.front();
            parts_.pop_front();
            read_part(lock, key, m);
            continue;
        }
        const int32_t key = queue_.front();
        queue_.pop_front();
        Entry& e = entries_[static_cast<size_t>(key)];
        if (e.state != State::Queued) continue;
        // Loading from here on: taking a slot may wait (unlocked), and the
        // entry must not be queued or dropped again meanwhile.
        e.state = State::Loading;
        const int32_t slot = take_slot_locked(lock);
        if (slot < 0) return;
        e.slot = slot;
        e.parts_left = 3;
        e.part_failed = false;
        slots_[static_cast<size_t>(slot)].entry = key;
        parts_.emplace_back(key, 1);
        parts_.emplace_back(key, 2);
        work_cv_.notify_all();
        read_part(lock, key, 0);
    }
}

void ExpertStore::prefetch(int32_t layer, std::span<const int32_t> experts, bool predicted) {
    bool queued = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const int32_t x : experts) {
            const int32_t key = layer * n_expert_ + x;
            Entry& e = entries_[static_cast<size_t>(key)];
            if (!predicted) e.guess = false;  // confirmed: settle_locked keeps it
            if (e.state == State::Queued && !predicted) {  // needed now: move ahead of guesses
                queue_.erase(std::remove(queue_.begin(), queue_.end(), key), queue_.end());
                queue_.push_front(key);
                continue;
            }
            if (e.state != State::Empty) continue;
            e.state = State::Queued;
            e.on_demand = !predicted;
            e.guess = predicted;
            if (predicted) {
                e.uses += 0.5f;  // a predicted expert is worth keeping until it is used
                queue_.push_back(key);
            } else {
                queue_.push_front(key);
            }
            queued = true;
        }
        if (!predicted) settle_locked(layer);
    }
    if (queued) work_cv_.notify_all();
}

void ExpertStore::settle_locked(int32_t layer) {
    const auto first = static_cast<size_t>(layer) * static_cast<size_t>(n_expert_);
    for (size_t key = first; key < first + static_cast<size_t>(n_expert_); ++key) {
        Entry& e = entries_[key];
        if (!e.guess) continue;
        e.guess = false;
        if (e.state == State::Queued) {
            e.state = State::Empty;  // io_loop skips keys that are no longer Queued
            ++stats_.dropped;
        } else if (e.state == State::Ready && e.untouched) {
            e.uses = 0.0f;  // no prefetch bonus, and old: the next eviction takes it
            e.last_use = 0;
        }
    }
}

bool ExpertStore::ready(int32_t layer, int32_t expert) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_[static_cast<size_t>(layer * n_expert_ + expert)].state == State::Ready;
}

Result<std::array<TensorView, 3>> ExpertStore::acquire(int32_t layer, int32_t expert) {
    const int32_t key = layer * n_expert_ + expert;
    std::unique_lock<std::mutex> lock(mutex_);
    Entry& e = entries_[static_cast<size_t>(key)];
    if (e.on_demand || e.state == State::Empty) {
        ++stats_.misses;
    } else if (e.state == State::Ready) {
        ++stats_.hits;
    } else {
        ++stats_.late;
    }
    e.on_demand = false;
    if (e.state != State::Ready) {
        const auto t0 = std::chrono::steady_clock::now();
        if (e.state == State::Empty) {
            e.state = State::Queued;
            queue_.push_front(key);  // demand loads jump the prefetch queue
            work_cv_.notify_one();
        } else if (e.state == State::Queued) {
            queue_.erase(std::remove(queue_.begin(), queue_.end(), key), queue_.end());
            queue_.push_front(key);
        }
        e.failed = false;
        // A loaded expert can be evicted again before this thread wakes up
        // (other loads need slots): then it is simply queued again.
        for (;;) {
            ready_cv_.wait(lock, [&] { return stop_ || e.state == State::Ready || e.state == State::Empty; });
            if (stop_ || e.state == State::Ready || e.failed) break;
            e.state = State::Queued;
            queue_.push_front(key);
            work_cv_.notify_one();
        }
        stats_.stall_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (e.state != State::Ready) return Status(ErrorCode::IoError, "expert could not be loaded");
    }
    ++e.pins;
    e.untouched = false;
    e.uses += 1.0f;
    e.last_use = ++clock_;
    const uint8_t* base = arena_ + static_cast<size_t>(e.slot) * slot_bytes_;
    std::array<TensorView, 3> views{};
    const auto& block = experts_[static_cast<size_t>(layer)];
    for (int32_t m = 0; m < 3; ++m) {
        const TensorView& t = *block[static_cast<size_t>(m)];
        const Segment seg = segment(key, m);
        TensorView v = t;
        v.n_dims = 2;
        v.ne = {t.ne[0], t.ne[1], 1, 1};
        v.nbytes = seg.bytes;
        v.file_offset = seg.offset;
        v.data = base + slot_offsets_[static_cast<size_t>(m)] + static_cast<size_t>(seg.offset % kAlign);
        views[static_cast<size_t>(m)] = v;
    }
    return views;
}

void ExpertStore::release(int32_t layer, int32_t expert) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        Entry& e = entries_[static_cast<size_t>(layer * n_expert_ + expert)];
        if (e.pins > 0) --e.pins;
    }
    ready_cv_.notify_all();
}

void ExpertStore::age() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (Entry& e : entries_) e.uses *= 0.5f;
}

size_t ExpertStore::trim() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (const int32_t key : queue_) {
        Entry& e = entries_[static_cast<size_t>(key)];
        if (e.state == State::Queued) e.state = State::Empty;
    }
    queue_.clear();
    ready_cv_.wait(lock, [this] {
        if (stop_) return true;
        if (!parts_.empty()) return false;
        return std::none_of(entries_.begin(), entries_.end(),
                            [](const Entry& e) { return e.state == State::Loading || e.pins > 0; });
    });
    size_t released = 0;
    for (Slot& slot : slots_) {
        if (slot.entry < 0) continue;
        Entry& e = entries_[static_cast<size_t>(slot.entry)];
        e.state = State::Empty;
        e.slot = -1;
        e.untouched = false;
        e.uses = 0.0f;
        slot.entry = -1;
        released += slot_bytes_;
    }
    // Locked pages cannot be dropped; the cache stays pageable from now on
    // (re-locking would fault the whole arena back in).
    if (locked_) {
        munlock(arena_, arena_bytes_);
        locked_ = false;
    }
#if defined(__linux__)
    // Private anonymous pages: freed at once, and read back as zeros (then
    // overwritten by the next expert read) when touched again.
    madvise(arena_, arena_bytes_, MADV_DONTNEED);
#endif
    return released;
}

ExpertStore::Stats ExpertStore::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}  // namespace liyab
