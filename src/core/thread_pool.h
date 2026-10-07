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

class ThreadPool {
public:
    // `n_threads` includes the calling thread; values < 1 select the number of
    // performance-class cores reported by the OS.
    explicit ThreadPool(int32_t n_threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] int32_t max_threads() const noexcept { return static_cast<int32_t>(workers_.size()) + 1; }
    [[nodiscard]] int32_t active_threads() const noexcept { return active_.load(std::memory_order_relaxed); }
    // Lets the power manager shed cores under thermal pressure (clamped to [1, max]).
    void set_active_threads(int32_t n) noexcept;

    // Runs fn(begin, end) over [0, n) split into contiguous chunks and blocks
    // until every chunk finished. Not reentrant: kernels must not nest calls.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);

private:
    void worker_loop(int32_t index);

    std::vector<std::thread> workers_;
    std::atomic<int32_t> active_{1};

    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable done_;
    const std::function<void(int64_t, int64_t)>* job_ = nullptr;
    int64_t job_size_ = 0;
    int32_t job_threads_ = 0;
    uint64_t generation_ = 0;
    int32_t pending_ = 0;
    bool stop_ = false;
};

// Number of "big" cores (highest max frequency) on this device; falls back to
// std::thread::hardware_concurrency().
int32_t performance_core_count() noexcept;

}  // namespace liyab

#endif  // LIYAB_CORE_THREAD_POOL_H
