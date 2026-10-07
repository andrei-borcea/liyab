// Liyab — energy and thermal governor.
//
// * Duty cycling: pace_token() holds token emission to a target rate so the
//   CPU/GPU/NPU reach idle states between tokens instead of racing and
//   heating up (reading speed is ~5–10 tok/s; anything faster is wasted heat
//   on a phone).
// * Thermal telemetry: a polling thread samples the OS thermal status
//   (Android AThermal API, Apple NSProcessInfo.thermalState) and the sysfs
//   thermal zones (skin and SoC temperatures, where the sandbox allows it).
// * Throttle guard: above the skin threshold (40 °C by default) or at a
//   severe OS thermal status, policy() tells the engine to move GPU work to
//   the NPU / CPU, shed cores and lower the token-rate cap.
#ifndef LIYAB_POWER_MANAGER_H
#define LIYAB_POWER_MANAGER_H

#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

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
    std::optional<float> skin_c;  // device surface temperature
    std::optional<float> soc_c;   // hottest CPU/GPU/SoC zone
};

using ThermalSensor = std::function<ThermalSample()>;

// OS thermal status combined with sysfs zones. Default sensor.
LIYAB_API ThermalSample read_platform_thermal();
// Parses Linux thermal zones under `root` (thermal_zone*/type, thermal_zone*/temp).
LIYAB_API ThermalSample read_sysfs_thermal(const std::string& root = "/sys/class/thermal");

struct PowerConfig {
    PowerProfile profile = PowerProfile::Balanced;
    float skin_threshold_c = 40.0f;
    float soc_threshold_c = 85.0f;
    double target_tps = 0.0;  // > 0 overrides the profile's pacing rate
    std::chrono::milliseconds poll_interval{1000};
};

struct PowerPolicy {
    double target_tps = 0.0;       // 0 = unpaced
    bool throttled = false;        // route GPU work away and shed cores
    float thread_fraction = 1.0f;  // share of the thread pool to use
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
    std::chrono::microseconds pace_token();
    void reset_pacing();

private:
    void poll_loop();
    [[nodiscard]] bool throttled_locked(const ThermalSample& sample) const;

    mutable std::mutex mutex_;
    PowerConfig config_;
    ThermalSensor sensor_;
    ThermalSample sample_;

    std::thread thread_;
    std::condition_variable cv_;
    bool stop_ = false;

    bool has_deadline_ = false;
    std::chrono::steady_clock::time_point deadline_{};
};

}  // namespace liyab

#endif  // LIYAB_POWER_MANAGER_H
