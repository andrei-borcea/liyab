#include "core/thread_pool.h"

#include <algorithm>
#include <cstdio>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace liyab {

ThreadPool::ThreadPool(int32_t n_threads) {
    if (n_threads < 1) n_threads = performance_core_count();
    n_threads = std::max<int32_t>(1, n_threads);
    workers_.reserve(static_cast<size_t>(n_threads - 1));
    for (int32_t i = 0; i < n_threads - 1; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
    active_.store(n_threads, std::memory_order_relaxed);
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : workers_) t.join();
}

void ThreadPool::set_active_threads(int32_t n) noexcept {
    active_.store(std::clamp<int32_t>(n, 1, max_threads()), std::memory_order_relaxed);
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    const auto threads = static_cast<int32_t>(std::min<int64_t>(active_threads(), n));
    if (threads <= 1) {
        fn(0, n);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = &fn;
        job_size_ = n;
        job_threads_ = threads;
        pending_ = threads - 1;
        ++generation_;
    }
    wake_.notify_all();

    fn(0, n / threads);  // the caller takes chunk 0

    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
    job_ = nullptr;
}

void ThreadPool::worker_loop(int32_t index) {
    uint64_t seen = 0;
    for (;;) {
        const std::function<void(int64_t, int64_t)>* job = nullptr;
        int64_t size = 0;
        int32_t threads = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            const int32_t chunk = index + 1;
            if (chunk >= job_threads_) continue;  // not a participant this round
            job = job_;
            size = job_size_;
            threads = job_threads_;
        }
        const int32_t chunk = index + 1;
        (*job)(size * chunk / threads, size * (chunk + 1) / threads);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--pending_ == 0) done_.notify_one();
        }
    }
}

int32_t performance_core_count() noexcept {
    const auto hw = static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
#if defined(__APPLE__)
    int32_t perf = 0;
    size_t len = sizeof perf;
    if (sysctlbyname("hw.perflevel0.physicalcpu", &perf, &len, nullptr, 0) == 0 && perf > 0) {
        return perf;
    }
    return hw;
#elif defined(__linux__)
    // Big cores: max frequency within 70% of the fastest core. Excludes the
    // efficiency cluster whose throughput does not pay for its sync cost.
    std::vector<long> freqs;
    for (int cpu = 0; cpu < hw; ++cpu) {
        const std::string path =
            "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cpufreq/cpuinfo_max_freq";
        std::FILE* f = std::fopen(path.c_str(), "r");
        if (!f) continue;
        long khz = 0;
        if (std::fscanf(f, "%ld", &khz) == 1 && khz > 0) freqs.push_back(khz);
        std::fclose(f);
    }
    if (freqs.empty()) return hw;
    const long fastest = *std::max_element(freqs.begin(), freqs.end());
    const auto big = std::count_if(freqs.begin(), freqs.end(),
                                   [&](long khz) { return khz * 10 >= fastest * 7; });
    return std::max<int32_t>(1, static_cast<int32_t>(big));
#else
    return hw;
#endif
}

}  // namespace liyab
