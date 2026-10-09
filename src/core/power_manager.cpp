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
// APerformanceHint_* (API 33), resolved at runtime like AThermal_*. A session
// covers the decode threads; its target is the token period, and each report
// is one token's work time.
struct PowerManager::HintSession {
    std::vector<int32_t> threads;
    void* session = nullptr;
    int64_t target_ns = 0;
    bool failed = false;  // creation refused: do not retry for these threads

#if defined(__ANDROID__)
    struct Api {
        void* manager = nullptr;
        void* (*create)(void*, const int32_t*, size_t, int64_t) = nullptr;
        int (*update_target)(void*, int64_t) = nullptr;
        int (*report)(void*, int64_t) = nullptr;
        void (*close)(void*) = nullptr;
    };
    static const Api& api() {
        static const Api a = [] {
            Api r;
            void* lib = dlopen("libandroid.so", RTLD_NOW | RTLD_LOCAL);
            if (!lib) return r;
            auto get_manager = reinterpret_cast<void* (*)()>(dlsym(lib, "APerformanceHint_getManager"));
            r.create = reinterpret_cast<decltype(r.create)>(dlsym(lib, "APerformanceHint_createSession"));
            r.update_target = reinterpret_cast<decltype(r.update_target)>(dlsym(lib, "APerformanceHint_updateTargetWorkDuration"));
            r.report = reinterpret_cast<decltype(r.report)>(dlsym(lib, "APerformanceHint_reportActualWorkDuration"));
            r.close = reinterpret_cast<decltype(r.close)>(dlsym(lib, "APerformanceHint_closeSession"));
            if (get_manager && r.create && r.update_target && r.report && r.close) r.manager = get_manager();
            return r;
        }();
        return a;
    }
#endif

    ~HintSession() { close(); }

    void close() {
#if defined(__ANDROID__)
        if (session != nullptr) api().close(session);
#endif
        session = nullptr;
    }

    // One token's work against a period of `period_ns`.
    void report(int64_t work_ns, int64_t period_ns) {
#if defined(__ANDROID__)
        if (threads.empty() || failed || api().manager == nullptr || work_ns <= 0) return;
        if (session == nullptr) {
            session = api().create(api().manager, threads.data(), threads.size(), period_ns);
            if (session == nullptr) {
                failed = true;
                LIYAB_LOG_WARN("performance hints unavailable for %zu decode threads", threads.size());
                return;
            }
            target_ns = period_ns;
            LIYAB_LOG_INFO("performance hints: %zu decode threads, %.1f ms per token", threads.size(), period_ns / 1e6);
        } else if (period_ns != target_ns) {
            api().update_target(session, period_ns);
            target_ns = period_ns;
        }
        api().report(session, work_ns);
#else
        (void)work_ns;
        (void)period_ns;
#endif
    }
};

PowerManager::PowerManager(PowerConfig config, ThermalSensor sensor)
    : config_(config),
      sensor_(sensor ? std::move(sensor) : ThermalSensor(read_platform_thermal)),
      hint_(std::make_unique<HintSession>()) {}

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
    // The hottest SoC zone jumps by 25-30 °C for a single sample under load
    // (65 -> 97 -> 70 °C on a Snapdragon 8 Elite): the emergency guard reads
    // a smoothed value, so one spike does not throttle the next second.
    if (sample.soc_c) {
        soc_ema_ = soc_ema_ ? 0.7f * *soc_ema_ + 0.3f * *sample.soc_c : *sample.soc_c;
    } else {
        soc_ema_.reset();
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

namespace {

// What each profile trades: where on the headroom forecast the response
// starts and reaches full strength, the token-rate cap (0: none), the share
// of cores, and the OS status at which the OS itself limits us.
struct ProfileTraits {
    float ramp_start;
    float ramp_end;
    double cap_tps;
    float thread_fraction;
    ThermalStatus os_limit;
};

ProfileTraits traits(PowerProfile profile) {
    switch (profile) {
        case PowerProfile::Performance: return {0.75f, 0.95f, 0.0, 1.0f, ThermalStatus::Severe};
        case PowerProfile::Balanced: return {0.65f, 0.90f, 12.0, 1.0f, ThermalStatus::Moderate};
        case PowerProfile::LowPower: return {0.35f, 0.60f, 6.0, 0.5f, ThermalStatus::Light};
    }
    return {0.75f, 0.95f, 0.0, 1.0f, ThermalStatus::Severe};
}

}  // namespace

float PowerManager::pressure_locked(const ThermalSample& s) const {
    // The forecast ramps the response over the profile's band of headroom.
    const ProfileTraits t = traits(config_.profile);
    float p = 0.0f;
    if (headroom_ema_) p = std::clamp((*headroom_ema_ - t.ramp_start) / (t.ramp_end - t.ramp_start), 0.0f, 1.0f);
    // Guards: the status at which the OS itself limits us (per profile), the
    // skin threshold, and the SoC's emergency temperature. With a forecast,
    // the OS already weighs the device maker's calibrated skin and SoC
    // sensors, so the sysfs "skin" reading (often a board thermistor near the
    // SoC: xo-therm reads 45-50 °C under load on a Snapdragon 8 Elite) only
    // guards emergencies, kSkinEmergencyMargin above the threshold: as a
    // plain threshold it throttled every token of a phone the OS rated cool.
    const ThermalStatus limit = t.os_limit;
    const float skin_limit = config_.skin_threshold_c + (headroom_ema_ ? kSkinEmergencyMargin : 0.0f);
    if ((s.status != ThermalStatus::Unknown && s.status >= limit) || (s.skin_c && *s.skin_c >= skin_limit) ||
        (soc_ema_ && *soc_ema_ >= config_.soc_threshold_c)) {
        p = 1.0f;
    }
    return p;
}

PowerPolicy PowerManager::policy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    PowerPolicy p;
    const ProfileTraits t = traits(config_.profile);
    p.target_tps = config_.target_tps > 0.0 ? config_.target_tps : t.cap_tps;
    p.thread_fraction = t.thread_fraction;

    p.pressure = pressure_locked(sample_);
    p.thread_fraction = std::min(p.thread_fraction, 1.0f - 0.5f * p.pressure);  // shed up to half the cores
    // Up to twice the full-speed time per token, from kPacingPressure on: at a
    // pressure of 0.0001 (a charging phone resting at the band's start) the
    // performance hint already asked for the current, slowest pace, and the
    // governor held the clocks there, capping the app at ~6 tok/s.
    p.slowdown = p.pressure >= kPacingPressure ? 1.0f + p.pressure : 1.0f;
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
    const PowerPolicy policy = this->policy();
    auto now = clock::now();
    // Full-speed work time per token, learned only while not slowed down: a
    // slowed token takes longer at the lower clocks the hint lets the
    // governor pick, and learning from it would slow the pace further.
    if (has_work_start_ && policy.slowdown <= 1.0f) {
        const double work = std::chrono::duration<double>(now - work_start_).count();
        work_ema_s_ = work_ema_s_ > 0.0 ? 0.8 * work_ema_s_ + 0.2 * work : work;
    }
    double period_s = policy.target_tps > 0.0 ? 1.0 / policy.target_tps : 0.0;
    if (policy.slowdown > 1.0f && work_ema_s_ > 0.0) period_s = std::max(period_s, work_ema_s_ * policy.slowdown);
    if (period_s <= 0.0) {
        has_deadline_ = false;
        has_work_start_ = true;
        work_start_ = now;
        hint_->close();  // full speed: no target to hint
        return std::chrono::microseconds(0);
    }
    const auto period = std::chrono::duration_cast<clock::duration>(std::chrono::duration<double>(period_s));
    if (has_work_start_) {
        hint_->report(std::chrono::duration_cast<std::chrono::nanoseconds>(now - work_start_).count(),
                      std::chrono::duration_cast<std::chrono::nanoseconds>(period).count());
    }
    std::chrono::microseconds slept{0};
    if (!has_deadline_ || now > deadline_ + period) {
        deadline_ = now;  // first token, or fell behind: re-anchor, no burst
    } else if (now < deadline_) {
        std::this_thread::sleep_until(deadline_);
        slept = std::chrono::duration_cast<std::chrono::microseconds>(deadline_ - now);
    }
    has_deadline_ = true;
    deadline_ += period;
    has_work_start_ = true;
    work_start_ = clock::now();  // the next token's work starts after the sleep
    return slept;
}

void PowerManager::reset_pacing() {
    has_deadline_ = false;
    has_work_start_ = false;  // the first token's work includes the prompt: not a per-token time
}

void PowerManager::set_hint_threads(std::vector<int32_t> thread_ids) {
    std::erase(thread_ids, 0);
    std::sort(thread_ids.begin(), thread_ids.end());
    if (thread_ids == hint_->threads) return;
    hint_->close();
    hint_->threads = std::move(thread_ids);
    hint_->failed = false;
}

}  // namespace liyab
