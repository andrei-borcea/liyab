import Cocoa
import FlutterMacOS
import IOKit
import IOKit.ps

/// The "liyab/device" channel on macOS, with the same methods and replies as
/// the Android one (android/.../DeviceChannel.kt): power draw, thermal state,
/// whether Liyab opens from any app, plus the process's CPU time (Android
/// reads /proc/self/stat from Dart, which macOS does not have).
enum DeviceChannel {
  static func attach(_ messenger: FlutterBinaryMessenger, hotKeyActive: @escaping () -> Bool) {
    FlutterMethodChannel(name: "liyab/device", binaryMessenger: messenger).setMethodCallHandler { call, result in
      switch call.method {
      case "power": result(power())
      case "thermal": result(thermal())
      case "cpuTicks": result(cpuTicks())
      case "isAssistant": result(hotKeyActive())
      default: result(FlutterMethodNotImplemented)
      }
    }
  }

  /// Battery power in watts and whether a charger is connected, from the
  /// battery controller's registry entry (laptops). Amperage is mA (negative
  /// while discharging), Voltage mV, Temperature hundredths of °C. A desktop
  /// Mac has no battery: zeros, as the Dart side expects when unknown.
  private static func power() -> [String: Any] {
    let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSmartBattery"))
    guard service != 0 else { return ["watts": 0.0, "charging": true, "level": -1, "batteryC": 0.0] }
    defer { IOObjectRelease(service) }
    func number(_ key: String) -> Double {
      (IORegistryEntryCreateCFProperty(service, key as CFString, kCFAllocatorDefault, 0)?
        .takeRetainedValue() as? NSNumber)?.doubleValue ?? 0
    }
    func flag(_ key: String) -> Bool {
      (IORegistryEntryCreateCFProperty(service, key as CFString, kCFAllocatorDefault, 0)?
        .takeRetainedValue() as? Bool) ?? false
    }
    let watts = abs(number("Amperage") / 1000 * number("Voltage") / 1000)
    return [
      "watts": watts,
      "charging": flag("ExternalConnected"),
      "level": Int(number("CurrentCapacity")),
      "batteryC": number("Temperature") / 100,
    ]
  }

  /// macOS reports four thermal states and no forecast. They map onto
  /// Android's status scale (0 none, 1 light, 2 moderate, 3 severe, 4
  /// critical); headroom stays unknown (NaN), so the flame's heat follows the
  /// status.
  private static func thermal() -> [String: Any] {
    let status: Int
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: status = 0
    case .fair: status = 1
    case .serious: status = 3
    case .critical: status = 4
    @unknown default: status = 0
    }
    return ["status": status, "headroom": Double.nan]
  }

  /// User + system CPU time of this process in 1/100 s, the unit of
  /// /proc/self/stat's utime + stime that the Dart side divides by.
  private static func cpuTicks() -> Int {
    var usage = rusage()
    guard getrusage(RUSAGE_SELF, &usage) == 0 else { return 0 }
    func centis(_ t: timeval) -> Int { t.tv_sec * 100 + Int(t.tv_usec) / 10_000 }
    return centis(usage.ru_utime) + centis(usage.ru_stime)
  }
}
