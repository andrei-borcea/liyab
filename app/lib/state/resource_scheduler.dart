// Headroom policy for desktop model budgets. Engine thermal scheduling remains
// authoritative; memory layout changes happen only between engine requests.
import 'dart:math' as math;

class ResourceBudget {
  static const mib = 1024 * 1024;
  int currentMb = 0;
  int _low = 0, _high = 0;
  DateTime? _lastChange;

  int reserveMb(int totalBytes) => math.max(3072, (totalBytes / mib * 0.10).round());

  int initial(int availableBytes, int totalBytes) {
    _low = _high = 0;
    if (availableBytes <= 0 || totalBytes <= 0) return currentMb = 2048;
    currentMb = (availableBytes ~/ mib - reserveMb(totalBytes)).clamp(2048, math.max(2048, totalBytes ~/ mib ~/ 2));
    return currentMb;
  }

  /// Sustained pressure shrinks the budget; sustained spare memory grows it.
  /// Hysteresis and cooldown prevent reload oscillations from transient spikes.
  int? sample(int availableBytes, int totalBytes, DateTime now) {
    if (availableBytes <= 0 || totalBytes <= 0 || currentMb == 0) return null;
    final free = availableBytes ~/ mib;
    final reserve = reserveMb(totalBytes);
    _low = free < reserve ? _low + 1 : 0;
    _high = free > reserve + 2048 ? _high + 1 : 0;
    if (_lastChange != null && now.difference(_lastChange!) < const Duration(minutes: 2)) return null;
    if (_low >= 3 && currentMb > 2048) return math.max(2048, currentMb - math.max(512, currentMb ~/ 4));
    final ceiling = math.max(2048, totalBytes ~/ mib ~/ 2);
    if (_high >= 12 && currentMb < ceiling) {
      return math.min(ceiling, currentMb + math.min(2048, (free - reserve) ~/ 2));
    }
    return null;
  }

  /// Keep reloads tied to a placement change. A fully resident dense model
  /// cannot gain from a larger cap; under pressure the next cap must cross
  /// the resident/streaming boundary to release any weight memory.
  int? forLayout(int proposedMb, {required int fileBytes, required bool blockStreaming,
      required bool mmapStreaming,
      required int expertBytes, required bool expertRoom, required int totalBytes}) {
    final thresholdMb = (fileBytes + (1 << 30) + mib - 1) ~/ mib;
    if (proposedMb > currentMb && !blockStreaming && !expertRoom &&
        (!mmapStreaming || proposedMb < thresholdMb)) {
      return null;
    }
    var next = proposedMb;
    if (next < currentMb && !blockStreaming && expertBytes == 0) {
      if (next >= thresholdMb) next = thresholdMb - 1;
    }
    final ceiling = math.max(2048, totalBytes ~/ mib ~/ 2);
    next = next.clamp(2048, next < currentMb ? currentMb - 1 : ceiling);
    return next == currentMb ? null : next;
  }

  void applied(int budgetMb, DateTime now) {
    currentMb = budgetMb;
    _lastChange = now;
    _low = _high = 0;
  }

  /// A failed load must not trigger another unload on the next 5 s sample.
  void defer(DateTime now) {
    _lastChange = now;
    _low = _high = 0;
  }
}

class ResourceEvent {
  ResourceEvent(this.message, {this.degraded = true}) : at = DateTime.now();
  final String message;
  final bool degraded;
  final DateTime at;
}
