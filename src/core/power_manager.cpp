#include "liyab/power_manager.h"

#include <dirent.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string_view>

#include "core/log.h"

#if defined(__ANDROID__)
#include <dlfcn.h>
#endif
#if defined(__APPLE__)
#include <objc/message.h>
#include <objc/runtime.h>
#endif

namespace liyab {

const char* thermal_status_name(ThermalStatus status) noexcept {
    switch (status) {
        case ThermalStatus::Unknown: return "unknown";
        case ThermalStatus::None: return "none";
        case ThermalStatus::Light: return "light";
        case ThermalStatus::Moderate: return "moderate";
        case ThermalStatus::Severe: return "severe";
        case ThermalStatus::Critical: return "critical";
        case ThermalStatus::Emergency: return "emergency";
        case ThermalStatus::Shutdown: return "shutdown";
    }
    return "unknown";
}

namespace {

bool read_file(const std::string& path, char* buf, size_t size) {
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return false;
    size_t n = std::fread(buf, 1, size - 1, f);
    std::fclose(f);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == ' ')) --n;
    buf[n] = '\0';
    return n > 0;
}

bool contains_any(std::string_view s, std::initializer_list<std::string_view> needles) {
    return std::any_of(needles.begin(), needles.end(), [&](std::string_view n) { return s.find(n) != std::string_view::npos; });
}

void keep_max(std::optional<float>& slot, float v) {
    if (!slot || v > *slot) slot = v;
}

ThermalStatus os_thermal_status() {
#if defined(__ANDROID__)
    // AThermal_* (API 30+) resolved at runtime so the library still loads on
    // older releases.
    struct Api {
        void* manager = nullptr;
        int (*get_status)(void*) = nullptr;
    };
    static const Api api = [] {
        Api a;
        void* lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) return a;
        auto acquire = reinterpret_cast<void* (*)()>(dlsym(lib, "AThermal_acquireManager"));
        a.get_status = reinterpret_cast<int (*)(void*)>(dlsym(lib, "AThermal_getCurrentThermalStatus"));
        if (acquire && a.get_status) a.manager = acquire();  // held for the process lifetime
        return a;
    }();
    if (api.manager == nullptr) return ThermalStatus::Unknown;
    const int status = api.get_status(api.manager);
    return status >= 0 && status <= 6 ? static_cast<ThermalStatus>(status) : ThermalStatus::Unknown;
#elif defined(__APPLE__)
    // [[NSProcessInfo processInfo] thermalState] through the ObjC runtime so
    // this file stays plain C++.
    auto cls = reinterpret_cast<id>(objc_getClass("NSProcessInfo"));
    if (cls == nullptr) return ThermalStatus::Unknown;
    using SendId = id (*)(id, SEL);
    using SendLong = long (*)(id, SEL);
    id info = reinterpret_cast<SendId>(objc_msgSend)(cls, sel_registerName("processInfo"));
    if (info == nullptr) return ThermalStatus::Unknown;
    switch (reinterpret_cast<SendLong>(objc_msgSend)(info, sel_registerName("thermalState"))) {
        case 0: return ThermalStatus::None;
        case 1: return ThermalStatus::Light;
        case 2: return ThermalStatus::Severe;
        case 3: return ThermalStatus::Critical;
        default: return ThermalStatus::Unknown;
    }
#else
    return ThermalStatus::Unknown;
#endif
}

}  // namespace

ThermalSample read_sysfs_thermal(const std::string& root) {
    ThermalSample sample;
    DIR* dir = opendir(root.c_str());
    if (dir == nullptr) return sample;  // commonly denied to apps by SELinux
    while (const dirent* entry = readdir(dir)) {
        if (std::strncmp(entry->d_name, "thermal_zone", 12) != 0) continue;
        const std::string zone = root + "/" + entry->d_name;
        char type[128];
        char temp[64];
        if (!read_file(zone + "/type", type, sizeof type) || !read_file(zone + "/temp", temp, sizeof temp)) continue;
        char* end = nullptr;
        double value = std::strtod(temp, &end);
        if (end == temp) continue;
        if (std::abs(value) >= 1000.0) value /= 1000.0;  // millidegrees (the usual unit)
        if (value < -40.0 || value > 150.0) continue;     // disconnected / bogus sensors

        std::string lower(type);
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
        if (contains_any(lower, {"skin", "shell", "case", "quiet", "xo-therm", "xo_therm"})) {
            keep_max(sample.skin_c, static_cast<float>(value));
        } else if (contains_any(lower, {"cpu", "gpu", "soc", "tsens", "npu", "apu", "big", "mid", "little"})) {
            keep_max(sample.soc_c, static_cast<float>(value));
        }
    }
    closedir(dir);
    return sample;
}

ThermalSample read_platform_thermal() {
#if defined(__linux__)
    ThermalSample sample = read_sysfs_thermal();
#else
    ThermalSample sample;
#endif
    sample.status = os_thermal_status();
    return sample;
}

// ---------------------------------------------------------------------------
// PowerManager
// ---------------------------------------------------------------------------
PowerManager::PowerManager(PowerConfig config, ThermalSensor sensor)
    : config_(config), sensor_(sensor ? std::move(sensor) : ThermalSensor(read_platform_thermal)) {}

PowerManager::~PowerManager() { stop(); }

void PowerManager::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (thread_.joinable()) return;
    stop_ = false;
    thread_ = std::thread([this] { poll_loop(); });
}

void PowerManager::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void PowerManager::poll_once() {
    const ThermalSample sample = sensor_();
    std::lock_guard<std::mutex> lock(mutex_);
    const bool was_throttled = throttled_locked(sample_);
    sample_ = sample;
    if (throttled_locked(sample_) != was_throttled) {
        LIYAB_LOG_INFO("thermal %s: status=%s skin=%.1fC soc=%.1fC", was_throttled ? "recovered" : "throttle",
                       thermal_status_name(sample_.status), sample_.skin_c.value_or(-1.0f),
                       sample_.soc_c.value_or(-1.0f));
    }
}

void PowerManager::poll_loop() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!stop_) {
        lock.unlock();
        poll_once();
        lock.lock();
        cv_.wait_for(lock, config_.poll_interval, [this] { return stop_; });
    }
}

ThermalSample PowerManager::last_sample() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sample_;
}

PowerProfile PowerManager::profile() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_.profile;
}

void PowerManager::set_profile(PowerProfile profile) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.profile = profile;
}

bool PowerManager::throttled_locked(const ThermalSample& s) const {
    // Status at which the OS itself starts limiting us, per profile.
    ThermalStatus limit = ThermalStatus::Moderate;
    if (config_.profile == PowerProfile::Performance) limit = ThermalStatus::Severe;
    if (config_.profile == PowerProfile::LowPower) limit = ThermalStatus::Light;
    return (s.status != ThermalStatus::Unknown && s.status >= limit) ||
           (s.skin_c && *s.skin_c >= config_.skin_threshold_c) ||
           (s.soc_c && *s.soc_c >= config_.soc_threshold_c);
}

PowerPolicy PowerManager::policy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PowerPolicy p;
    switch (config_.profile) {
        case PowerProfile::Performance: p.target_tps = 0.0; break;
        case PowerProfile::Balanced: p.target_tps = 12.0; break;
        case PowerProfile::LowPower: p.target_tps = 6.0; p.thread_fraction = 0.5f; break;
    }
    if (config_.target_tps > 0.0) p.target_tps = config_.target_tps;

    p.throttled = throttled_locked(sample_);
    if (p.throttled) {
        p.target_tps = p.target_tps > 0.0 ? p.target_tps * 0.5 : 8.0;
        p.thread_fraction = std::min(p.thread_fraction, 0.5f);
    }
    if (sample_.status >= ThermalStatus::Critical) {
        p.target_tps = 2.0;
        p.thread_fraction = 0.25f;
    }
    return p;
}

std::chrono::microseconds PowerManager::pace_token() {
    using clock = std::chrono::steady_clock;
    const double tps = policy().target_tps;
    auto now = clock::now();
    if (tps <= 0.0) {
        has_deadline_ = false;
        return std::chrono::microseconds(0);
    }
    const auto period = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(1.0 / tps));
    std::chrono::microseconds slept{0};
    if (!has_deadline_ || now > deadline_ + period) {
        deadline_ = now;  // first token, or fell behind: re-anchor, no burst
    } else if (now < deadline_) {
        std::this_thread::sleep_until(deadline_);
        slept = std::chrono::duration_cast<std::chrono::microseconds>(deadline_ - now);
    }
    has_deadline_ = true;
    deadline_ += period;
    return slept;
}

void PowerManager::reset_pacing() { has_deadline_ = false; }

}  // namespace liyab
