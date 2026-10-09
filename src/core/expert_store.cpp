#include "core/expert_store.h"

#include <sys/mman.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#include "core/log.h"
#include "core/quant.h"

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
                                                         int32_t n_expert, size_t budget_bytes, int32_t io_threads,
                                                         bool repack) {
    if (n_expert <= 0) return Status(ErrorCode::InvalidArgument, "ExpertStore needs n_expert > 0");
    std::unique_ptr<ExpertStore> s(new ExpertStore());
    s->file_ = &file;
    s->experts_ = std::move(experts);
    s->n_expert_ = n_expert;
    s->repack_ = repack;

    // A slot holds the three matrices of one expert, each in its own aligned
    // area (an expert's offset is not block-aligned, so one extra alignment
    // unit is reserved per matrix). Blocks quantized alike share a slot size
    // class; mixed quantizations (Unsloth's UD files: Q5_K or Q6_K down
    // matrices in a few blocks) would otherwise size every slot for the
    // largest expert. Each class gets a share of the budget proportional to
    // its blocks, since every block runs top-k experts per token.
    s->block_class_.assign(s->experts_.size(), -1);
    std::vector<size_t> class_blocks;
    for (size_t l = 0; l < s->experts_.size(); ++l) {
        const auto& block = s->experts_[l];
        if (block[0] == nullptr) continue;
        std::array<size_t, 3> area{};
        for (size_t m = 0; m < 3; ++m) {
            const size_t bytes = block[m]->row_bytes() * static_cast<size_t>(block[m]->ne[1]);
            area[m] = align_up(bytes + kAlign);
            s->expert_bytes_total_ += static_cast<uint64_t>(bytes) * static_cast<uint64_t>(n_expert);
        }
        const std::array<size_t, 3> offsets = {0, area[0], area[0] + area[1]};
        const size_t bytes = area[0] + area[1] + area[2];
        size_t c = 0;
        while (c < s->classes_.size() && (s->classes_[c].bytes != bytes || s->classes_[c].offsets != offsets)) ++c;
        if (c == s->classes_.size()) {
            s->classes_.push_back({bytes, offsets, 0, 0});
            class_blocks.push_back(0);
        }
        ++class_blocks[c];
        s->block_class_[l] = static_cast<int32_t>(c);
    }
    if (s->classes_.empty()) return Status(ErrorCode::InvalidArgument, "no expert tensors");

    // Per class: its share of the budget, and at least enough slots for a
    // block's top-k plus the in-flight loads.
    const size_t total_entries = s->experts_.size() * static_cast<size_t>(n_expert);
    size_t moe_blocks = 0;
    for (const size_t b : class_blocks) moe_blocks += b;
    const size_t min_slots = static_cast<size_t>(std::max(io_threads, 1)) + 34;
    size_t n_slots = 0;
    for (size_t c = 0; c < s->classes_.size(); ++c) {
        SizeClass& sc = s->classes_[c];
        const size_t entries = class_blocks[c] * static_cast<size_t>(n_expert);
        const double share = static_cast<double>(budget_bytes) * static_cast<double>(class_blocks[c]) /
                             static_cast<double>(moe_blocks);
        sc.first_slot = n_slots;
        sc.n_slots = std::clamp(static_cast<size_t>(share / static_cast<double>(sc.bytes)), std::min(min_slots, entries),
                                entries);
        n_slots += sc.n_slots;
        s->arena_bytes_ += sc.n_slots * sc.bytes;
    }
    void* arena = nullptr;
    if (posix_memalign(&arena, kAlign, s->arena_bytes_) != 0) {
        return Status(ErrorCode::OutOfMemory, "cannot allocate the expert cache");
    }
    s->arena_ = static_cast<uint8_t*>(arena);
    // Best effort: locked pages are never swapped or compressed (Android
    // usually caps RLIMIT_MEMLOCK, in which case the cache stays pageable).
    s->locked_ = mlock(s->arena_, s->arena_bytes_) == 0;
    s->slots_.resize(n_slots);
    size_t offset = 0;
    for (size_t c = 0; c < s->classes_.size(); ++c) {
        for (size_t i = 0; i < s->classes_[c].n_slots; ++i) {
            Slot& slot = s->slots_[s->classes_[c].first_slot + i];
            slot.offset = offset;
            slot.size_class = static_cast<int32_t>(c);
            offset += s->classes_[c].bytes;
        }
    }
    s->entries_.resize(total_entries);

    for (size_t i = 0; i < file.shard_count(); ++i) {
        auto part = DirectFile::open(file.shard(i).path());
        if (!part) return part.status();
        s->files_.push_back(std::move(part).value());
    }
    for (int32_t i = 0; i < std::max(io_threads, 1); ++i) s->io_threads_.emplace_back([p = s.get()] { p->io_loop(); });
    std::string classes;
    for (const SizeClass& sc : s->classes_) {
        char part[48];
        std::snprintf(part, sizeof part, "%s%zu x %.2f MiB", classes.empty() ? "" : " + ", sc.n_slots,
                      static_cast<double>(sc.bytes) / (1024.0 * 1024.0));
        classes += part;
    }
    LIYAB_LOG_INFO("expert cache: %s = %.2f GiB (%.0f%% of %.2f GiB of experts)%s", classes.c_str(),
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

DType ExpertStore::served_type(int32_t layer, int32_t matrix) const {
    const TensorView& t = *experts_[static_cast<size_t>(layer)][static_cast<size_t>(matrix)];
    if (!repack_) return t.type;
    const int64_t rows = t.ne[1], cols = t.ne[0];
    switch (t.type) {
        case DType::Q4_K: return rows % 8 == 0 && cols % quant::kSuperBlock == 0 ? DType::Q4_K_R8 : t.type;
        case DType::Q5_K: return rows % 8 == 0 && cols % quant::kSuperBlock == 0 ? DType::Q5_K_R8 : t.type;
        case DType::Q6_K: return rows % 8 == 0 && cols % quant::kSuperBlock == 0 ? DType::Q6_K_R8 : t.type;
        case DType::Q8_0: return rows % 4 == 0 && cols % quant::kBlock == 0 ? DType::Q8_0_R4 : t.type;
        default: return t.type;
    }
}

namespace {

// Rearranges one expert matrix in place into its repacked type (same size),
// through a per-thread copy of the file layout.
void repack_in_place(uint8_t* data, size_t bytes, DType from, DType to, int64_t rows, int64_t cols) {
    thread_local std::vector<uint8_t> scratch;
    scratch.assign(data, data + bytes);
    switch (to) {
        case DType::Q4_K_R8:
            quant::repack_q4_K_r8(reinterpret_cast<const quant::BlockQ4_K*>(scratch.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ4_Kx8*>(data));
            break;
        case DType::Q5_K_R8:
            quant::repack_q5_K_r8(reinterpret_cast<const quant::BlockQ5_K*>(scratch.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ5_Kx8*>(data));
            break;
        case DType::Q6_K_R8:
            quant::repack_q6_K_r8(reinterpret_cast<const quant::BlockQ6_K*>(scratch.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ6_Kx8*>(data));
            break;
        case DType::Q8_0_R4:
            quant::repack_q8_0_r4(reinterpret_cast<const quant::BlockQ8_0*>(scratch.data()), rows, cols,
                                  reinterpret_cast<quant::BlockQ8_0x4*>(data));
            break;
        default:
            (void)from;
            break;
    }
}

}  // namespace

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
    const Slot& slot = slots_[static_cast<size_t>(e.slot)];
    uint8_t* dst = arena_ + slot.offset + classes_[static_cast<size_t>(slot.size_class)].offsets[static_cast<size_t>(m)];
    e.repacked = repack_ && e.for_batch;
    const bool repack = e.repacked;
    lock.unlock();
    // One read per matrix: smaller parts (tried: 2 and 4 per matrix) cost
    // the flash more than the extra concurrency saves.
    Status status = files_[seg.shard]->read(begin, length, dst);
    if (status.is_ok()) {
        const int32_t layer = key / n_expert_;
        const DType to = served_type(layer, m);
        const TensorView& t = *experts_[static_cast<size_t>(layer)][static_cast<size_t>(m)];
        if (repack && to != t.type) repack_in_place(dst + (seg.offset - begin), seg.bytes, t.type, to, t.ne[1], t.ne[0]);
    }
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

int32_t ExpertStore::take_slot_locked(std::unique_lock<std::mutex>& lock, int32_t size_class) {
    const SizeClass& sc = classes_[static_cast<size_t>(size_class)];
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
        for (size_t i = sc.first_slot; i < sc.first_slot + sc.n_slots; ++i) {
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
        const int32_t slot = take_slot_locked(lock, block_class_[static_cast<size_t>(key / n_expert_)]);
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
            e.for_batch = batch_.load(std::memory_order_relaxed);
            if (predicted) {
                if (!batch_.load(std::memory_order_relaxed)) e.uses += 0.5f;  // worth keeping until used
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
            e.for_batch = batch_.load(std::memory_order_relaxed);
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
            e.for_batch = batch_.load(std::memory_order_relaxed);
            queue_.push_front(key);
            work_cv_.notify_one();
        }
        stats_.stall_ms +=
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (e.state != State::Ready) return Status(ErrorCode::IoError, "expert could not be loaded");
    }
    ++e.pins;
    e.untouched = false;
    // A multi-token pass (prefill) uses most experts once: crediting those
    // uses evicted the experts the conversation keeps using, and decoding
    // after a 1900-token prompt found 73% of its experts cached instead of 85%.
    if (!batch_.load(std::memory_order_relaxed)) e.uses += 1.0f;
    e.last_use = ++clock_;
    const Slot& slot = slots_[static_cast<size_t>(e.slot)];
    const uint8_t* base = arena_ + slot.offset;
    const std::array<size_t, 3>& offsets = classes_[static_cast<size_t>(slot.size_class)].offsets;
    std::array<TensorView, 3> views{};
    const auto& block = experts_[static_cast<size_t>(layer)];
    for (int32_t m = 0; m < 3; ++m) {
        const TensorView& t = *block[static_cast<size_t>(m)];
        const Segment seg = segment(key, m);
        TensorView v = t;
        v.type = e.repacked ? served_type(layer, m) : t.type;
        v.n_dims = 2;
        v.ne = {t.ne[0], t.ne[1], 1, 1};
        v.nbytes = seg.bytes;
        v.file_offset = seg.offset;
        v.data = base + offsets[static_cast<size_t>(m)] + static_cast<size_t>(seg.offset % kAlign);
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

std::vector<int32_t> ExpertStore::hot_keys() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int32_t> keys;
    for (const Slot& slot : slots_) {
        if (slot.entry >= 0 && entries_[static_cast<size_t>(slot.entry)].state == State::Ready) keys.push_back(slot.entry);
    }
    std::stable_sort(keys.begin(), keys.end(), [&](int32_t a, int32_t b) {
        const Entry& x = entries_[static_cast<size_t>(a)];
        const Entry& y = entries_[static_cast<size_t>(b)];
        return x.uses > y.uses || (x.uses == y.uses && x.last_use > y.last_use);
    });
    return keys.empty() ? hot_ : keys;  // after a trim, the list it recorded
}

void ExpertStore::set_hot(std::vector<int32_t> keys) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::erase_if(keys, [&](int32_t k) {
        return k < 0 || static_cast<size_t>(k) >= entries_.size() || block_class_[static_cast<size_t>(k / n_expert_)] < 0;
    });
    hot_ = std::move(keys);
}

size_t ExpertStore::warm() {
    size_t queued = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const int32_t key : hot_) {
            if (queued >= slots_.size()) break;  // beyond the cache's size the list would evict itself
            Entry& e = entries_[static_cast<size_t>(key)];
            if (e.state != State::Empty) continue;
            e.state = State::Queued;
            e.on_demand = false;
            e.guess = false;
            e.uses = 0.25f;  // kept like a light prefetch until used
            queue_.push_back(key);
            ++queued;
        }
    }
    if (queued > 0) work_cv_.notify_all();
    return queued;
}

size_t ExpertStore::trim() {
    const std::vector<int32_t> hot = hot_keys();
    std::unique_lock<std::mutex> lock(mutex_);
    if (!hot.empty()) hot_ = hot;
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
        released += classes_[static_cast<size_t>(slot.size_class)].bytes;
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
