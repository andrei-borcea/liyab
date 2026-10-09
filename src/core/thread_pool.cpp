#include "core/thread_pool.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace liyab {

namespace {

// Tells the core it is in a spin-wait loop (lower power, yields to an SMT sibling).
inline void cpu_relax() noexcept {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

}  // namespace

ThreadPool::ThreadPool(int32_t n_threads) {
    if (n_threads < 1) n_threads = default_thread_count();
    n_threads = std::clamp<int32_t>(n_threads, 1, kMaxThreads);
    workers_.reserve(static_cast<size_t>(n_threads - 1));
    worker_tids_ = std::vector<std::atomic<int32_t>>(static_cast<size_t>(n_threads - 1));
    for (int32_t i = 0; i < n_threads - 1; ++i) {
        workers_.emplace_back([this, i] { worker_loop(i); });
    }
    active_.store(n_threads, std::memory_order_relaxed);
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_.store(true);
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
    const auto chunks = static_cast<int32_t>(std::min<int64_t>(n, int64_t{threads} * kChunksPerThread));
    job_ = &fn;
    job_size_ = n;
    done_.store(0, std::memory_order_relaxed);
    const uint64_t sequence = ((generation_.load(std::memory_order_relaxed) >> 8) + 1) & 0xFFFFFFFFu;
    ticket_.store(sequence << 32 | static_cast<uint64_t>(chunks) << 16, std::memory_order_release);
    // seq_cst: ordered after the ticket, and before the sleepers_ read.
    generation_.store(sequence << 8 | static_cast<uint64_t>(threads));
    if (sleepers_.load() > 0) {
        std::lock_guard<std::mutex> lock(mutex_);  // a worker between its check and its wait holds the mutex
        wake_.notify_all();
    }

    run_chunks(sequence);
    // The remaining chunks are running on other threads: spin until they end.
    while (done_.load(std::memory_order_acquire) != chunks) cpu_relax();
    job_ = nullptr;
}

void ThreadPool::run_chunks(uint64_t sequence) {
    uint64_t ticket = ticket_.load(std::memory_order_acquire);
    for (;;) {
        const auto next = static_cast<int64_t>(ticket & 0xFFFF);
        const auto chunks = static_cast<int64_t>(ticket >> 16 & 0xFFFF);
        if ((ticket >> 32) != sequence || next >= chunks) return;
        if (!ticket_.compare_exchange_weak(ticket, ticket + 1, std::memory_order_acquire,
                                           std::memory_order_acquire)) {
            continue;  // `ticket` now holds the current value
        }
        const int64_t size = job_size_;
        (*job_)(size * next / chunks, size * (next + 1) / chunks);
        done_.fetch_add(1, std::memory_order_release);
        ticket = ticket_.load(std::memory_order_acquire);
    }
}

uint64_t ThreadPool::wait_for_job(uint64_t seen) {
    using Clock = std::chrono::steady_clock;
    const auto spin_until = Clock::now() + std::chrono::microseconds(kSpinMicros);
    for (uint32_t i = 1;; ++i) {
        const uint64_t g = generation_.load(std::memory_order_acquire);
        if (g != seen || stop_.load(std::memory_order_relaxed)) return g;
        cpu_relax();
        if ((i & 255) == 0 && Clock::now() >= spin_until) break;  // read the clock rarely
    }
    std::unique_lock<std::mutex> lock(mutex_);
    sleepers_.fetch_add(1);  // seq_cst: ordered before the generation re-read below
    wake_.wait(lock, [&] { return generation_.load() != seen || stop_.load(); });
    sleepers_.fetch_sub(1);
    return generation_.load(std::memory_order_acquire);
}

std::vector<int32_t> ThreadPool::worker_thread_ids() const {
    std::vector<int32_t> ids;
    for (const auto& tid : worker_tids_) {
        if (const int32_t id = tid.load(std::memory_order_acquire); id > 0) ids.push_back(id);
    }
    return ids;
}

int32_t ThreadPool::current_thread_id() noexcept {
#if defined(__linux__)
    return static_cast<int32_t>(syscall(SYS_gettid));
#else
    return 0;
#endif
}

void ThreadPool::worker_loop(int32_t index) {
    worker_tids_[static_cast<size_t>(index)].store(current_thread_id(), std::memory_order_release);
    uint64_t seen = 0;
    const int32_t participant = index + 1;  // the caller is participant 0
    for (;;) {
        seen = wait_for_job(seen);
        if (stop_.load(std::memory_order_relaxed)) return;
        const auto threads = static_cast<int32_t>(seen & 0xFF);
        if (participant >= threads) continue;  // not a participant this round
        run_chunks(seen >> 8);
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

int32_t default_thread_count() noexcept {
    const int32_t big = performance_core_count();
    const auto all = static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
    return big >= all && big > 2 ? big - 1 : big;
}

}  // namespace liyab
