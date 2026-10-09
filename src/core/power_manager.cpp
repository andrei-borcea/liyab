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

#if defined(__ANDROID__)
// AThermal_* (status: API 30, headroom: API 31) resolved at runtime so the
// library still loads on older releases.
struct AThermalApi {
    void* manager = nullptr;
    int (*get_status)(void*) = nullptr;
    float (*get_headroom)(void*, int) = nullptr;
};

const AThermalApi& athermal() {
    static const AThermalApi api = [] {
        AThermalApi a;
        void* lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib) return a;
        auto acquire = reinterpret_cast<void* (*)()>(dlsym(lib, "AThermal_acquireManager"));
        a.get_status = reinterpret_cast<int (*)(void*)>(dlsym(lib, "AThermal_getCurrentThermalStatus"));
        a.get_headroom = reinterpret_cast<float (*)(void*, int)>(dlsym(lib, "AThermal_getThermalHeadroom"));
        if (acquire && a.get_status) a.manager = acquire();  // held for the process lifetime
        return a;
    }();
    return api;
}
#endif

// The OS forecast of thermal headroom 10 s ahead (NaN until it has data, and
// at most one fresh value per second, which the 1 s poll respects).
std::optional<float> os_thermal_headroom() {
#if defined(__ANDROID__)
    const AThermalApi& api = athermal();
    if (api.manager == nullptr || api.get_headroom == nullptr) return std::nullopt;
    const float h = api.get_headroom(api.manager, 10);
    if (std::isnan(h) || h < 0.0f) return std::nullopt;
    return h;
#else
    return std::nullopt;
#endif
}

ThermalStatus os_thermal_status() {
#if defined(__ANDROID__)
    const AThermalApi& api = athermal();
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
    sample.headroom = os_thermal_headroom();
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
    const float before = pressure_locked(sample_);
    sample_ = sample;
    if (sample.headroom) {
        headroom_ema_ = headroom_ema_ ? 0.7f * *headroom_ema_ + 0.3f * *sample.headroom : *sample.headroom;
    } else {
        headroom_ema_.reset();
    }
    // Log when the response changes band: none, shedding cores, throttled.
    const float after = pressure_locked(sample_);
    const auto band = [](float p) { return p >= 1.0f ? 2 : (p > 0.0f ? 1 : 0); };
    if (band(after) != band(before)) {
        static constexpr const char* kBand[] = {"cool", "shedding cores", "throttle"};
        LIYAB_LOG_INFO("thermal %s: pressure=%.2f headroom=%.2f status=%s skin=%.1fC soc=%.1fC", kBand[band(after)],
                       after, headroom_ema_.value_or(-1.0f), thermal_status_name(sample_.status),
                       sample_.skin_c.value_or(-1.0f), sample_.soc_c.value_or(-1.0f));
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

float PowerManager::pressure_locked(const ThermalSample& s) const {
    // The forecast ramps the response: nothing below 0.75, full at 0.95.
    float p = 0.0f;
    if (headroom_ema_) p = std::clamp((*headroom_ema_ - 0.75f) / 0.20f, 0.0f, 1.0f);
    // Guards: the status at which the OS itself limits us (per profile), the
    // skin threshold, and the SoC's emergency temperature.
    ThermalStatus limit = ThermalStatus::Moderate;
    if (config_.profile == PowerProfile::Performance) limit = ThermalStatus::Severe;
    if (config_.profile == PowerProfile::LowPower) limit = ThermalStatus::Light;
    if ((s.status != ThermalStatus::Unknown && s.status >= limit) || (s.skin_c && *s.skin_c >= config_.skin_threshold_c) ||
        (s.soc_c && *s.soc_c >= config_.soc_threshold_c)) {
        p = 1.0f;
    }
    return p;
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

    p.pressure = pressure_locked(sample_);
    p.thread_fraction = std::min(p.thread_fraction, 1.0f - 0.5f * p.pressure);  // shed up to half the cores
    p.throttled = p.pressure >= 1.0f;
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
