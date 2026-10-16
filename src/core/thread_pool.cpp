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
#if defined(__linux__) || defined(__APPLE__)
#include <pthread.h>
#endif
#if defined(_WIN32)
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>  // _mm_pause
#endif
#include "core/windows_api.h"
#endif

namespace liyab {

namespace {

// Tells the core it is in a spin-wait loop (lower power, yields to an SMT sibling).
[[maybe_unused]] inline void cpu_relax() noexcept {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(_M_X64) || defined(_M_IX86)
    _mm_pause();  // MSVC, which defines neither of the above: the same PAUSE instruction
#endif
}

// One step of waiting for `a` to differ from `old`. On ARM64 the core arms
// its exclusive monitor on a's cache line and sleeps in WFE until another
// core writes that line, or until the next event (Linux and Android send one
// about every 100 us): no instructions issue meanwhile, so the core idles at
// low power. A `yield` spin keeps it at full clock (yield only helps SMT
// siblings, which phone and Apple cores do not have), and the pool waits
// between ~400 jobs per decoded token. Elsewhere a pause / yield hint.
template <typename T>
inline void wait_step(const std::atomic<T>& a, T old) noexcept {
#if defined(__aarch64__)
    static_assert(sizeof(T) == 4 || sizeof(T) == 8);
    T v;
    if constexpr (sizeof(T) == 8) {
        __asm__ __volatile__("ldaxr %x0, [%1]" : "=r"(v) : "r"(&a) : "memory");
    } else {
        __asm__ __volatile__("ldaxr %w0, [%1]" : "=r"(v) : "r"(&a) : "memory");
    }
    if (v == old) __asm__ __volatile__("wfe" ::: "memory");
#else
    (void)a;
    (void)old;
    cpu_relax();
#endif
}

#if defined(_WIN32)
// The processors as Windows' scheduler presents them, read once.
//  * Machines with more than 64 logical processors are split into processor
//    groups. Before Windows 11 a thread runs only in the group of the thread
//    that created it unless it is moved, so a pool sized from every group's
//    cores would crowd them all onto one group's (see join_processor_group).
//  * Cores of a hybrid CPU (Intel P- and E-cores) report an EfficiencyClass,
//    higher for the faster ones; homogeneous CPUs report 0 for every core.
struct Topology {
    std::vector<KAFFINITY> group_masks;            // active processors, by group number
    std::vector<int32_t> group_performance_cores;  // physical cores of the top class, by group
    int32_t performance_cores = 0;                 // physical cores of the top class, all groups
    int32_t logical = 0;                           // active logical processors, all groups
};

Topology read_topology() {
    Topology t;
    DWORD bytes = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &bytes);  // fails, and reports the size it needs
    if (bytes == 0) return t;
    std::vector<uint8_t> buffer(bytes);
    if (!GetLogicalProcessorInformationEx(RelationAll,
                                          reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data()),
                                          &bytes)) {
        return t;
    }
    // Records of varying size, in no promised order: one for the groups, one
    // per physical core, and others (caches, NUMA nodes, packages) skipped.
    auto each = [&](LOGICAL_PROCESSOR_RELATIONSHIP kind, const auto& fn) {
        for (DWORD at = 0; at + sizeof(LOGICAL_PROCESSOR_RELATIONSHIP) + sizeof(DWORD) <= bytes;) {
            const auto* info = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buffer.data() + at);
            if (info->Size == 0 || info->Size > bytes - at) break;
            if (info->Relationship == kind) fn(*info);
            at += info->Size;
        }
    };
    each(RelationGroup, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& info) {
        for (WORD g = 0; g < info.Group.ActiveGroupCount; ++g) {
            t.group_masks.push_back(info.Group.GroupInfo[g].ActiveProcessorMask);
            t.logical += info.Group.GroupInfo[g].ActiveProcessorCount;
        }
    });
    BYTE top = 0;
    each(RelationProcessorCore, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& info) {
        top = std::max(top, info.Processor.EfficiencyClass);
    });
    t.group_performance_cores.assign(t.group_masks.size(), 0);
    each(RelationProcessorCore, [&](const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX& info) {
        if (info.Processor.EfficiencyClass != top) return;
        ++t.performance_cores;
        const WORD group = info.Processor.GroupMask[0].Group;  // a core lies in one group
        if (group < t.group_performance_cores.size()) ++t.group_performance_cores[group];
    });
    return t;
}

const Topology& topology() noexcept {
    static const Topology t = [] {
        try {
            return read_topology();
        } catch (...) {  // allocation failure: report nothing, callers fall back
            return Topology{};
        }
    }();
    return t;
}

// Moves the calling worker, pool participant `participant` (the thread that
// calls parallel_for() is participant 0), to the processor group holding the
// participant-th performance core, counting from the group the pool was
// created in: the pool's threads spread over the groups as its size was
// counted over them. With a single group (nearly every PC) nothing is set and
// the scheduler places the workers, as on other systems. Windows 11 lets
// threads use every group by default; one group per worker still keeps each
// on one NUMA node there.
void join_processor_group(int32_t participant) noexcept {
    const Topology& t = topology();
    if (t.group_masks.size() < 2 || t.performance_cores <= 0) return;
    GROUP_AFFINITY current{};
    if (!GetThreadGroupAffinity(GetCurrentThread(), &current)) return;
    const auto groups = static_cast<WORD>(t.group_masks.size());
    int32_t slot = participant % t.performance_cores;
    for (WORD i = 0; i < groups; ++i) {
        const auto g = static_cast<WORD>((current.Group + i) % groups);
        if (slot >= t.group_performance_cores[g]) {
            slot -= t.group_performance_cores[g];
            continue;
        }
        if (g != current.Group) {
            GROUP_AFFINITY target{};
            target.Mask = t.group_masks[g];
            target.Group = g;
            SetThreadGroupAffinity(GetCurrentThread(), &target, nullptr);
        }
        return;
    }
}

// SetThreadDescription (Windows 10 1607 and later), looked up at run time so
// the library still loads on systems without it. The name shows in debuggers
// and ETW traces, as a pthread name does in /proc and Instruments.
void set_thread_name(const std::string& name) noexcept {
    using SetDescription = HRESULT(WINAPI*)(HANDLE, PCWSTR);
    static const SetDescription set = [] {
        const HMODULE kernel = GetModuleHandleW(L"kernel32.dll");
        // Through void (*)(): a cast from FARPROC to another function type
        // needs an intermediate one compilers accept without a warning.
        return kernel == nullptr ? nullptr
                                 : reinterpret_cast<SetDescription>(reinterpret_cast<void (*)()>(
                                       GetProcAddress(kernel, "SetThreadDescription")));
    }();
    if (set == nullptr) return;
    const std::wstring wide(name.begin(), name.end());  // ASCII
    set(GetCurrentThread(), wide.c_str());
}
#endif

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
        generation_.fetch_add(uint64_t{1} << 8);  // a write to the line wakes workers waiting in WFE
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
    // The remaining chunks are running on other threads: wait until they end.
    for (int32_t d; (d = done_.load(std::memory_order_acquire)) != chunks;) wait_step(done_, d);
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
    for (;;) {
        const uint64_t g = generation_.load(std::memory_order_acquire);
        if (g != seen || stop_.load(std::memory_order_relaxed)) return g;
        wait_step(generation_, seen);
        if (Clock::now() >= spin_until) break;  // a wait step lasts long enough for a clock read each
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
    const int32_t participant = index + 1;  // the caller is participant 0
    // A name of its own (a thread otherwise inherits its creator's: the CLI's,
    // or a Dart isolate's in the app), so per-thread CPU readings tell the
    // decode workers apart (scripts/thread_cpu.py). At most 15 characters.
    {
        const std::string name = "liyab-worker-" + std::to_string(index + 1);
#if defined(__linux__)
        pthread_setname_np(pthread_self(), name.c_str());
#elif defined(__APPLE__)
        pthread_setname_np(name.c_str());
#elif defined(_WIN32)
        set_thread_name(name);
        join_processor_group(participant);
#endif
    }
    uint64_t seen = 0;
    for (;;) {
        seen = wait_for_job(seen);
        if (stop_.load(std::memory_order_relaxed)) return;
        const auto threads = static_cast<int32_t>(seen & 0xFF);
        if (participant >= threads) continue;  // not a participant this round
        run_chunks(seen >> 8);
    }
}

int32_t logical_core_count() noexcept {
#if defined(_WIN32)
    // Every group's processors: depending on the C++ runtime,
    // hardware_concurrency() may count the current group's only.
    if (const int32_t n = topology().logical; n > 0) return n;
#endif
    return static_cast<int32_t>(std::max(1u, std::thread::hardware_concurrency()));
}

int32_t performance_core_count() noexcept {
    const int32_t hw = logical_core_count();
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
#elif defined(_WIN32)
    // Physical cores of the highest efficiency class, one worker each as on
    // Apple (hw.perflevel0.physicalcpu): SMT siblings share a core's load
    // ports and the memory bandwidth decoding is bound by, and two workers
    // spinning on one core slow each other down.
    const int32_t big = topology().performance_cores;
    return big > 0 ? std::min(big, hw) : hw;
#else
    return hw;
#endif
}

int32_t default_thread_count() noexcept {
    const int32_t big = performance_core_count();
    const int32_t all = logical_core_count();
    return big >= all && big > 2 ? big - 1 : big;
}

}  // namespace liyab
