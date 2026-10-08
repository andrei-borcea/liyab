// Liyab — fixed-size fork/join pool for data-parallel kernels (internal).
#ifndef LIYAB_CORE_THREAD_POOL_H
#define LIYAB_CORE_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace liyab {

// Workers spin between jobs for a short while before they sleep: a decode
// step issues about a thousand parallel_for calls of 10–300 us each, and a
// condition-variable wake-up costs ~80–100 us on a Snapdragon 8 Elite, which
// made small matmuls slower on 8 threads than on one. Spinning workers pick
// a job up in about a microsecond. After kSpinMicros without work they block
// on a condition variable, so an idle engine costs no CPU.
//
// A job is split into kChunksPerThread chunks per participating thread, and
// every participant (the caller included) claims the next unclaimed chunk
// until none is left. With one fixed chunk per thread, the job ended when the
// slowest thread did: the prime cores run ~20% faster than the performance
// cores, and a worker the OS preempts (I/O threads, the app's UI) stalled the
// whole job until it ran again. Now the others take over its chunks.
class ThreadPool {
public:
    // `n_threads` includes the calling thread; values < 1 select
    // default_thread_count().
    explicit ThreadPool(int32_t n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] int32_t max_threads() const noexcept { return static_cast<int32_t>(workers_.size()) + 1; }
    [[nodiscard]] int32_t active_threads() const noexcept { return active_.load(std::memory_order_relaxed); }
    // Lets the power manager shed cores under thermal pressure (clamped to [1, max]).
    void set_active_threads(int32_t n) noexcept;

    // Runs fn(begin, end) over [0, n) split into contiguous chunks and blocks
    // until every chunk finished. Chunks run concurrently and in any order, on
    // any participating thread. Not reentrant: kernels must not nest calls,
    // and only one thread may call it at a time.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);

    // How long an idle worker spins before it sleeps.
    static constexpr int64_t kSpinMicros = 300;
    static constexpr int32_t kMaxThreads = 255;  // fits the low byte of generation_
    // Chunks per participating thread: enough to rebalance around a slow or
    // preempted thread, few enough that claiming one (a CAS) stays negligible.
    static constexpr int32_t kChunksPerThread = 4;

private:
    void worker_loop(int32_t index);
    // Waits for a generation other than `seen` (or stop); returns it.
    uint64_t wait_for_job(uint64_t seen);
    // Claims and runs chunks of job `sequence` until none is left.
    void run_chunks(uint64_t sequence);

    std::vector<std::thread> workers_;
    std::atomic<int32_t> active_{1};

    // Job: written by parallel_for() before it publishes the job's ticket,
    // read by a participant only after it claimed a chunk. parallel_for()
    // returns once every chunk is done_, so a claimed chunk keeps these valid;
    // a thread that claims nothing never reads them.
    const std::function<void(int64_t, int64_t)>* job_ = nullptr;
    int64_t job_size_ = 0;
    // (job sequence << 32) | (chunks << 16) | next unclaimed chunk. Claimed by
    // compare-and-swap, never fetch_add: a worker still looking at a finished
    // job must fail on the sequence instead of taking a chunk of the next one.
    std::atomic<uint64_t> ticket_{0};
    std::atomic<int32_t> done_{0};  // chunks finished
    // (job sequence << 8) | participating threads: wakes the workers and tells
    // each one whether to take part.
    std::atomic<uint64_t> generation_{0};
    std::atomic<bool> stop_{false};

    // Sleeping workers. The publisher stores the generation, then reads
    // sleepers_; a worker increments sleepers_, then re-reads the generation
    // (both sequentially consistent), so one of them always sees the other.
    std::atomic<int32_t> sleepers_{0};
    std::mutex mutex_;
    std::condition_variable wake_;
};

// Number of "big" cores (highest max frequency) on this device; falls back to
// std::thread::hardware_concurrency().
int32_t performance_core_count() noexcept;

// Default pool size: the performance cores, minus one when every core is a
// performance core (Snapdragon 8 Elite: 8 Oryon cores). With no efficiency
// cluster, one free core is what the expert/block I/O threads, the OS and the
// app's UI run on: 8 workers spinning on 8 cores kept the I/O threads waiting
// for a core, and Qwen3.6-35B-A3B decoded 21% slower than with 7.
int32_t default_thread_count() noexcept;

}  // namespace liyab

#endif  // LIYAB_CORE_THREAD_POOL_H
