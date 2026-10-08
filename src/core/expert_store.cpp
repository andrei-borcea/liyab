#include "core/expert_store.h"

#include <sys/mman.h>
#include <unistd.h>

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
    const bool locked = mlock(s->arena_, s->arena_bytes_) == 0;
    s->slots_.resize(n_slots);
    s->entries_.resize(total_entries);

    for (size_t i = 0; i < file.shard_count(); ++i) {
        auto part = DirectFile::open(file.shard(i).path());
        if (!part) return part.status();
        s->files_.push_back(std::move(part).value());
        auto buffered = DirectFile::open(file.shard(i).path(), false);
        if (!buffered) return buffered.status();
        s->buffered_.push_back(std::move(buffered).value());
    }
    for (int32_t i = 0; i < std::max(io_threads, 1); ++i) s->io_threads_.emplace_back([p = s.get()] { p->io_loop(); });
    LIYAB_LOG_INFO("expert cache: %zu slots x %.2f MiB = %.2f GiB (%.0f%% of %.2f GiB of experts)%s", n_slots,
                   static_cast<double>(s->slot_bytes_) / (1024.0 * 1024.0),
                   static_cast<double>(s->arena_bytes_) / (1024.0 * 1024.0 * 1024.0),
                   100.0 * static_cast<double>(n_slots) / static_cast<double>(total_entries),
                   static_cast<double>(s->expert_bytes_total_) / (1024.0 * 1024.0 * 1024.0),
                   locked ? ", locked in RAM" : "");
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
        munlock(arena_, arena_bytes_);
        std::free(arena_);
    }
}

ExpertStore::Segment ExpertStore::segment(int32_t key, int32_t matrix) const {
    const auto& block = experts_[static_cast<size_t>(key / n_expert_)];
    const TensorView& t = *block[static_cast<size_t>(matrix)];
    const size_t bytes = t.row_bytes() * static_cast<size_t>(t.ne[1]);
    return {t.shard, t.file_offset + static_cast<uint64_t>(key % n_expert_) * bytes, bytes};
}

bool ExpertStore::in_page_cache(const Segment& seg) const {
#if defined(__APPLE__)
    using Residency = char;
#else
    using Residency = unsigned char;
#endif
    static const auto page = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    const uint64_t begin = seg.offset / page * page;
    const size_t length = static_cast<size_t>(seg.offset + seg.bytes - begin);
    thread_local std::vector<Residency> pages;
    pages.resize((length + page - 1) / page);
    auto* addr = const_cast<uint8_t*>(file_->shard(seg.shard).data()) + begin;
    if (mincore(addr, length, pages.data()) != 0) return false;
    return std::all_of(pages.begin(), pages.end(), [](Residency r) { return (r & 1) != 0; });
}

Status ExpertStore::read_entry(int32_t key, uint8_t* dst, bool on_demand) {
    static const int mode = std::getenv("LIYAB_PC_MODE") ? std::atoi(std::getenv("LIYAB_PC_MODE")) : 1;  // TEMP
    bool all_cached = true;
    uint64_t bytes = 0;
    for (int32_t m = 0; m < 3; ++m) {
        const Segment seg = segment(key, m);
        const uint64_t begin = seg.offset / kAlign * kAlign;
        const size_t length = align_up(static_cast<size_t>(seg.offset - begin) + seg.bytes);
        const bool cached = mode != 0 && in_page_cache(seg);
        all_cached = all_cached && cached;
        const bool buffered = cached || mode == 1 || (mode == 2 && !on_demand);
        const DirectFile& f = buffered ? *buffered_[seg.shard] : *files_[seg.shard];
        LIYAB_RETURN_IF_ERROR(f.read(begin, length, dst + slot_offsets_[static_cast<size_t>(m)]));
        if (!cached) bytes += length;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    stats_.bytes_read += bytes;
    if (all_cached) ++stats_.page_cache_loads;
    return Status::ok();
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
        work_cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
        if (stop_) return;
        const int32_t key = queue_.front();
        queue_.pop_front();
        Entry& e = entries_[static_cast<size_t>(key)];
        if (e.state != State::Queued) continue;
        const int32_t slot = take_slot_locked(lock);
        if (slot < 0) return;
        e.state = State::Loading;
        e.slot = slot;
        slots_[static_cast<size_t>(slot)].entry = key;
        uint8_t* dst = arena_ + static_cast<size_t>(slot) * slot_bytes_;
        const bool urgent = e.urgent;
        lock.unlock();
        const Status status = read_entry(key, dst, urgent);
        lock.lock();
        e.urgent = false;
        if (!status.is_ok()) {
            LIYAB_LOG_ERROR("%s", status.to_string().c_str());
            e.failed = true;
            e.state = State::Empty;
            e.slot = -1;
            slots_[static_cast<size_t>(slot)].entry = -1;
        } else {
            e.state = State::Ready;
            e.untouched = true;
            e.last_use = clock_;
            ++stats_.loads;
        }
        ready_cv_.notify_all();
    }
}

void ExpertStore::prefetch(int32_t layer, std::span<const int32_t> experts, bool predicted) {
    bool queued = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const int32_t x : experts) {
            const int32_t key = layer * n_expert_ + x;
            Entry& e = entries_[static_cast<size_t>(key)];
            if (e.state == State::Queued && !predicted) {  // needed now: move ahead of guesses
                queue_.erase(std::remove(queue_.begin(), queue_.end(), key), queue_.end());
                queue_.push_front(key);
                e.urgent = true;
                continue;
            }
            if (e.state != State::Empty) continue;
            e.state = State::Queued;
            e.on_demand = !predicted;
            e.urgent = !predicted;
            if (predicted) {
                e.uses += 0.5f;  // a predicted expert is worth keeping until it is used
                queue_.push_back(key);
            } else {
                queue_.push_front(key);
            }
            queued = true;
        }
    }
    if (queued) work_cv_.notify_all();
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
        e.urgent = true;
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

ExpertStore::Stats ExpertStore::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

}  // namespace liyab
