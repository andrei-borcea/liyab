// Tests for SoC classification, backend ranking, live device detection and
// the thermal side of the power manager (sysfs parsing, policy, pacing).
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

#include "liyab/device_detect.h"
#include "liyab/power_manager.h"
#include "test_util.h"

using namespace liyab;

TEST_CASE("classify_soc identifies flagship mobile SoCs") {
    SocInfo s = classify_soc("QTI", "SM8750", "sun", "qcom");
    CHECK(s.vendor == SocVendor::Qualcomm);
    CHECK(s.name == "Snapdragon 8 Elite");
    CHECK(s.npu == "Hexagon v79");

    s = classify_soc("", "", "pineapple", "");
    CHECK(s.name == "Snapdragon 8 Gen 3");

    s = classify_soc("Mediatek", "MT6991", "mt6991", "mt6991");
    CHECK(s.vendor == SocVendor::MediaTek);
    CHECK(s.name == "Dimensity 9400");
    CHECK(classify_soc("", "MT6989V/CZA", "", "").name == "Dimensity 9300");

    CHECK(classify_soc("Google", "", "zumapro", "").name == "Tensor G4");
    CHECK(classify_soc("Google", "", "zuma", "").name == "Tensor G3");
    CHECK(classify_soc("Samsung", "s5e9945", "", "").name == "Exynos 2400");

    s = classify_soc("Apple", "Apple M4 Pro", "Mac16,8", "");
    CHECK(s.vendor == SocVendor::Apple);
    CHECK(s.name == "Apple M4 Pro");
}

TEST_CASE("classify_soc uses exact matching for codenames") {
    // "sun" (Snapdragon 8 Elite codename) is a substring of "samsung".
    const SocInfo s = classify_soc("Samsung", "", "", "");
    CHECK(s.vendor == SocVendor::Samsung);
    CHECK(s.name != "Snapdragon 8 Elite");
}

TEST_CASE("classify_soc falls back to vendor families and unknown") {
    CHECK(classify_soc("", "SM7675", "", "").vendor == SocVendor::Qualcomm);
    CHECK(classify_soc("", "", "mt6878", "").vendor == SocVendor::MediaTek);
    CHECK(classify_soc("", "", "", "Exynos 1480").vendor == SocVendor::Samsung);
    const SocInfo u = classify_soc("", "", "", "");
    CHECK(u.vendor == SocVendor::Unknown);
    CHECK(u.name == "unknown");
}

TEST_CASE("rank_backends orders NPU -> GPU -> CPU among available accelerators") {
    Accelerators acc;
    acc.qnn_htp.available = true;
    acc.vulkan.available = true;
    SocInfo snapdragon{SocVendor::Qualcomm, "Snapdragon 8 Elite", "SM8750", "Hexagon v79"};
    auto order = rank_backends(snapdragon, acc);
    REQUIRE(order.size() == 3);
    CHECK(order[0] == BackendKind::Qnn);
    CHECK(order[1] == BackendKind::Vulkan);
    CHECK(order[2] == BackendKind::Cpu);

    // QNN runtime on a non-Qualcomm SoC is ignored; NeuroPilot used on MediaTek.
    Accelerators mtk;
    mtk.qnn_htp.available = true;
    mtk.neuropilot.available = true;
    order = rank_backends({SocVendor::MediaTek, "Dimensity 9400", "MT6991", ""}, mtk);
    CHECK(order.front() == BackendKind::NeuroPilot);
    CHECK(std::find(order.begin(), order.end(), BackendKind::Qnn) == order.end());

    Accelerators apple;
    apple.metal.available = true;
    order = rank_backends({SocVendor::Apple, "Apple M4", "", ""}, apple);
    CHECK(order == (std::vector<BackendKind>{BackendKind::Metal, BackendKind::Cpu}));

    CHECK(rank_backends({}, {}) == std::vector<BackendKind>{BackendKind::Cpu});
}

TEST_CASE("detect_device inspects the running machine") {
    const DeviceInfo info = detect_device();
    CHECK(!info.backend_order.empty());
    CHECK(info.backend_order.back() == BackendKind::Cpu);
    CHECK(info.total_memory > 0);
    CHECK(info.page_size >= 4096);
    CHECK(info.cpu.cores >= 1);
    CHECK(info.cpu.performance_cores >= 1 && info.cpu.performance_cores <= info.cpu.cores);
#if defined(__aarch64__)
    CHECK(info.cpu.neon);
#endif
    const std::string text = describe_device(info);
    CHECK(text.find("Backends:") != std::string::npos);
    std::printf("%s", text.c_str());
}

TEST_CASE("read_sysfs_thermal classifies skin and SoC zones") {
    const std::string root = test::temp_dir() + "/liyab_thermal_" + std::to_string(getpid());
    auto zone = [&](int i, const char* type, const char* temp) {
        const std::string dir = root + "/thermal_zone" + std::to_string(i);
        mkdir(dir.c_str(), 0755);
        std::ofstream(dir + "/type") << type << "\n";
        std::ofstream(dir + "/temp") << temp << "\n";
    };
    mkdir(root.c_str(), 0755);
    zone(0, "skin-therm", "41500");      // millidegrees
    zone(1, "cpu-1-0-usr", "72000");
    zone(2, "gpuss-0", "78");             // already in degrees
    zone(3, "battery", "30000");          // neither skin nor SoC
    zone(4, "cpu-bogus", "-273000");      // disconnected sensor

    const ThermalSample s = read_sysfs_thermal(root);
    REQUIRE(s.skin_c.has_value());
    REQUIRE(s.soc_c.has_value());
    CHECK_NEAR(*s.skin_c, 41.5, 1e-3);
    CHECK_NEAR(*s.soc_c, 78.0, 1e-3);
    CHECK(!read_sysfs_thermal(root + "/missing").skin_c.has_value());
    std::system(("rm -rf '" + root + "'").c_str());
}

TEST_CASE("PowerManager throttles on skin temperature and OS thermal status") {
    ThermalSample sample;
    PowerConfig config;
    config.profile = PowerProfile::Balanced;
    PowerManager pm(config, [&] { return sample; });

    sample.skin_c = 35.0f;
    pm.poll_once();
    PowerPolicy p = pm.policy();
    CHECK(!p.throttled);
    CHECK_NEAR(p.target_tps, 12.0, 1e-9);
    CHECK_NEAR(p.thread_fraction, 1.0, 1e-9);

    sample.skin_c = 40.5f;  // above the 40 °C guard
    pm.poll_once();
    p = pm.policy();
    CHECK(p.throttled);
    CHECK_NEAR(p.target_tps, 6.0, 1e-9);
    CHECK_NEAR(p.thread_fraction, 0.5, 1e-9);

    sample.skin_c = 30.0f;
    sample.status = ThermalStatus::Severe;
    pm.poll_once();
    CHECK(pm.policy().throttled);

    pm.set_profile(PowerProfile::Performance);
    sample.status = ThermalStatus::Moderate;  // below the Performance limit
    pm.poll_once();
    p = pm.policy();
    CHECK(!p.throttled);
    CHECK(p.target_tps == 0.0);  // unpaced

    sample.status = ThermalStatus::Critical;
    pm.poll_once();
    p = pm.policy();
    CHECK(p.throttled);
    CHECK_NEAR(p.target_tps, 2.0, 1e-9);
}

TEST_CASE("PowerManager polling thread samples the sensor") {
    std::atomic<int> calls{0};
    PowerConfig config;
    config.poll_interval = std::chrono::milliseconds(5);
    PowerManager pm(config, [&] {
        ++calls;
        return ThermalSample{};
    });
    pm.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    pm.stop();
    CHECK(calls.load() >= 3);
}

TEST_CASE("pace_token duty-cycles output to the target rate without bursts") {
    PowerConfig config;
    config.target_tps = 100.0;  // 10 ms per token
    PowerManager pm(config, [] { return ThermalSample{}; });

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < 6; ++i) pm.pace_token();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    CHECK(elapsed_ms >= 45.0);   // 5 paced gaps of 10 ms (first token is immediate)
    CHECK(elapsed_ms < 200.0);

    // After a long stall the schedule re-anchors instead of bursting.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto slept = pm.pace_token();
    CHECK(slept.count() == 0);
    const auto t0 = std::chrono::steady_clock::now();
    pm.pace_token();
    const double gap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    CHECK(gap_ms >= 5.0);
}

TEST_MAIN()
