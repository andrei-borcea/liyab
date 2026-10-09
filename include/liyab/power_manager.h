// Liyab — energy and thermal governor.
//
// * Duty cycling: pace_token() holds token emission to a target rate so the
//   CPU/GPU/NPU reach idle states between tokens instead of racing and
//   heating up (reading speed is ~5–10 tok/s; anything faster is wasted heat
//   on a phone).
// * Thermal telemetry: a polling thread samples the OS thermal status
//   (Android AThermal API, Apple NSProcessInfo.thermalState) and the sysfs
//   thermal zones (skin and SoC temperatures, where the sandbox allows it).
// * Thermal control: the main input is the OS's thermal-headroom forecast
//   (Android AThermal_getThermalHeadroom, 10 s ahead; 1.0 = severe
//   throttling), smoothed, turned into a pressure from 0 to 1 over a band of
//   headroom that depends on the profile: Performance 0.75-0.95, Balanced
//   0.55-0.80, LowPower 0.35-0.60 (a cooler profile reacts earlier). With
//   pressure the engine sheds cores (down to half) and paces tokens at up to
//   twice their full-speed work time, which the performance hint (ADPF) turns
//   into lower clocks: the same tokens for fewer watts, instead of racing and
//   then halving. At pressure 1, at the OS's own limiting status, above the
//   skin threshold, or with the SoC past its emergency temperature, policy()
//   reports throttled: work moves off the GPU, half the cores, and the token
//   rate is capped. Devices without a forecast (iOS, older Android) use the
//   status and temperature guards only.
#ifndef LIYAB_POWER_MANAGER_H
#define LIYAB_POWER_MANAGER_H

#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "liyab/types.h"

namespace liyab {

// Ordered by severity; mirrors Android's AThermalStatus. Apple states map as
// nominal→None, fair→Light, serious→Severe, critical→Critical.
enum class ThermalStatus : int32_t {
    Unknown = -1,
    None = 0,
    Light = 1,
    Moderate = 2,
    Severe = 3,
    Critical = 4,
    Emergency = 5,
    Shutdown = 6,
};

LIYAB_API const char* thermal_status_name(ThermalStatus status) noexcept;

struct ThermalSample {
    ThermalStatus status = ThermalStatus::Unknown;
    std::optional<float> skin_c;    // device surface temperature
    std::optional<float> soc_c;     // hottest CPU/GPU/SoC zone
    std::optional<float> headroom;  // OS forecast 10 s ahead: 0 cool .. 1 severe throttling (Android 12+)
};

using ThermalSensor = std::function<ThermalSample()>;

// OS thermal status combined with sysfs zones. Default sensor.
LIYAB_API ThermalSample read_platform_thermal();
// Parses Linux thermal zones under `root` (thermal_zone*/type, thermal_zone*/temp).
LIYAB_API ThermalSample read_sysfs_thermal(const std::string& root = "/sys/class/thermal");

struct PowerConfig {
    PowerProfile profile = PowerProfile::Balanced;
    float skin_threshold_c = 40.0f;
    float soc_threshold_c = 95.0f;  // emergency only: flagship SoCs run at 85-90 °C under load
    double target_tps = 0.0;  // > 0 overrides the profile's pacing rate
    std::chrono::milliseconds poll_interval{1000};
};

struct PowerPolicy {
    double target_tps = 0.0;       // 0 = unpaced
    bool throttled = false;        // route GPU work away and shed cores
    float thread_fraction = 1.0f;  // share of the thread pool to use
    float pressure = 0.0f;         // thermal pressure, 0 none .. 1 throttled
    // >= 1: tokens are paced to this multiple of their full-speed work time
    // (1 + pressure), so under thermal pressure the performance hint lets the
    // governor lower clocks: the same tokens, more slowly, for fewer watts.
    float slowdown = 1.0f;
};

class LIYAB_API PowerManager {
public:
    explicit PowerManager(PowerConfig config, ThermalSensor sensor = read_platform_thermal);
    ~PowerManager();
    PowerManager(const PowerManager&) = delete;
    PowerManager& operator=(const PowerManager&) = delete;

    // Background polling every config.poll_interval. Idempotent.
    void start();
    void stop();
    // Takes one sample synchronously.
    void poll_once();

    [[nodiscard]] ThermalSample last_sample() const;
    [[nodiscard]] PowerPolicy policy() const;
    [[nodiscard]] PowerProfile profile() const;
    void set_profile(PowerProfile profile);

    // Duty cycling: call right before emitting each token. Sleeps until the
    // token's slot at policy().target_tps (the first token is never delayed).
    // Falling behind never causes a catch-up burst. Returns the time slept.
    // Generation thread only (not synchronized).
    //
    // While pacing on Android 13+, it also reports each token's work time to
    // a performance-hint session (ADPF) over the threads given to
    // set_hint_threads(), with the token period as the target: the CPU
    // governor then picks the lowest clocks that still meet the rate, rather
    // than sprinting and sleeping (lower voltage, less energy per token).
    // Unpaced decoding wants the highest clocks and opens no session.
    std::chrono::microseconds pace_token();
    // Starts a new run: the next token is not delayed and its work time is not reported.
    void reset_pacing();
    // The OS ids of the threads that decode (the generation thread and its
    // workers); a change closes the open hint session. Generation thread only.
    void set_hint_threads(std::vector<int32_t> thread_ids);

private:
    struct HintSession;  // ADPF session; a no-op off Android

    void poll_loop();
    // 0 (no thermal pressure) .. 1 (throttle), from the smoothed headroom
    // forecast and the status / temperature guards.
    // With a headroom forecast, the skin guard is an emergency this far above the threshold (°C).
    static constexpr float kSkinEmergencyMargin = 8.0f;
    [[nodiscard]] float pressure_locked(const ThermalSample& sample) const;

    mutable std::mutex mutex_;
    PowerConfig config_;
    ThermalSensor sensor_;
    ThermalSample sample_;
    std::optional<float> headroom_ema_;  // smoothed forecast: one noisy sample does not shed cores

    std::thread thread_;
    std::condition_variable cv_;
    bool stop_ = false;

    bool has_deadline_ = false;
    std::chrono::steady_clock::time_point deadline_{};
    bool has_work_start_ = false;
    std::chrono::steady_clock::time_point work_start_{};  // when the current token's work began
    double work_ema_s_ = 0.0;  // full-speed work time per token (seconds), see pace_token()
    std::unique_ptr<HintSession> hint_;
};

}  // namespace liyab

#endif  // LIYAB_POWER_MANAGER_H
