// What the phone is doing while Liyab runs: CPU, power, heat, speed and
// storage reads, sampled once a second while the app is in the foreground.
import 'dart:async';
import 'dart:io';
import 'dart:math' as math;

import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';

import '../engine/engine_service.dart';
import '../engine/liyab_ffi.dart';

class DeviceSample {
  const DeviceSample({
    required this.cpuPercent,
    required this.watts,
    required this.charging,
    required this.batteryC,
    required this.thermalStatus,
    required this.headroom,
    required this.tokensPerSecond,
    required this.flashMBps,
    required this.gpuPercent,
  });

  /// This app's CPU time over all cores, 0..100.
  final double cpuPercent;

  /// Battery power; while charging it is the net flow into the battery.
  final double watts;
  final bool charging;
  final double batteryC;

  /// Android thermal status: 0 none, 1 light, 2 moderate, 3 severe, 4 critical…
  final int thermalStatus;

  /// Forecast thermal headroom 10 s ahead (1.0 = severe throttling); NaN if unknown.
  final double headroom;
  final double tokensPerSecond;
  final double flashMBps;

  /// GPU time on this engine's work, 0..100 (Vulkan submit-to-fence).
  final double gpuPercent;

  /// Energy per generated token, when tokens are flowing on battery.
  double? get joulesPerToken => tokensPerSecond > 0.5 && !charging ? watts / tokensPerSecond : null;
}

class DeviceMonitor extends ChangeNotifier with WidgetsBindingObserver {
  DeviceMonitor(this._engine, this._onLogLines) {
    WidgetsBinding.instance.addObserver(this);
    _start();
  }

  final EngineService _engine;
  final void Function(List<String>) _onLogLines;
  static const _channel = MethodChannel('liyab/device');

  DeviceSample? latest;
  final List<DeviceSample> history = []; // last 120 s
  Timer? _timer;
  int? _cpuTicks;
  EngineCounters? _counters;
  DateTime _at = DateTime.now();

  /// The flame's heat: the forecast headroom when Android gives one, else the thermal status.
  double get heat {
    final s = latest;
    if (s == null) return 0.3;
    if (!s.headroom.isNaN) return s.headroom.clamp(0.0, 1.0);
    return (0.3 + s.thermalStatus * 0.2).clamp(0.0, 1.0);
  }

  void _start() {
    _timer ??= Timer.periodic(const Duration(seconds: 1), (_) => _sample());
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _start();
    } else {
      _timer?.cancel(); // nothing to show: no sampling in the background
      _timer = null;
    }
  }

  static int? _readCpuTicks() {
    try {
      // /proc/self/stat: fields 14 and 15 (utime, stime) after the ")" of the name.
      final stat = File('/proc/self/stat').readAsStringSync();
      final f = stat.substring(stat.lastIndexOf(')') + 2).split(' ');
      return int.parse(f[11]) + int.parse(f[12]);
    } on Object {
      return null;
    }
  }

  Future<void> _sample() async {
    final now = DateTime.now();
    final dt = now.difference(_at).inMicroseconds / 1e6;
    _at = now;
    final ticks = _readCpuTicks();
    final cpu = ticks != null && _cpuTicks != null && dt > 0
        ? ((ticks - _cpuTicks!) / 100 / dt / Platform.numberOfProcessors * 100).clamp(0.0, 100.0)
        : 0.0;
    _cpuTicks = ticks;
    final counters = _engine.counters();
    final prev = _counters;
    _counters = counters;
    double rate(num Function(EngineCounters) f) =>
        counters != null && prev != null && dt > 0 ? math.max(0, (f(counters) - f(prev)) / dt) : 0;

    Map<Object?, Object?> power = const {}, thermal = const {};
    if (Platform.isAndroid) {
      try {
        power = await _channel.invokeMethod<Map<Object?, Object?>>('power') ?? const {};
        thermal = await _channel.invokeMethod<Map<Object?, Object?>>('thermal') ?? const {};
      } on PlatformException {
        // keep the defaults
      }
    }
    final sample = DeviceSample(
      cpuPercent: cpu.toDouble(),
      watts: (power['watts'] as num?)?.toDouble() ?? 0,
      charging: power['charging'] as bool? ?? false,
      batteryC: (power['batteryC'] as num?)?.toDouble() ?? 0,
      thermalStatus: (thermal['status'] as num?)?.toInt() ?? 0,
      headroom: (thermal['headroom'] as num?)?.toDouble() ?? double.nan,
      tokensPerSecond: rate((c) => c.tokensGenerated).toDouble(),
      flashMBps: rate((c) => c.storageBytesRead).toDouble() / 1e6,
      gpuPercent: (rate((c) => c.acceleratorBusyMs) / 10).clamp(0, 100).toDouble(),
    );
    latest = sample;
    history.add(sample);
    if (history.length > 120) history.removeAt(0);
    _onLogLines(LiyabLib.instance.takeLogLines());
    notifyListeners();
  }

  @override
  void dispose() {
    _timer?.cancel();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }
}
