import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/state/resource_scheduler.dart';

void main() {
  const gib = 1024 * 1024 * 1024;
  test('automatic budget leaves OS headroom and caps own share of RAM', () {
    final policy = ResourceBudget();
    final mb = policy.initial(14 * gib, 48 * gib);
    expect(mb, lessThan(14 * 1024));
    expect(mb, lessThanOrEqualTo(24 * 1024));
    expect(14 * 1024 - mb, greaterThanOrEqualTo(policy.reserveMb(48 * gib)));
    expect(policy.initial(60 * gib, 64 * gib), 32 * 1024);
  });

  test('sustained pressure degrades and cooldown prevents repeated reloads', () {
    final policy = ResourceBudget()..initial(20 * gib, 48 * gib);
    final now = DateTime(2026);
    expect(policy.sample(gib, 48 * gib, now), isNull);
    expect(policy.sample(gib, 48 * gib, now), isNull);
    final reduced = policy.sample(gib, 48 * gib, now)!;
    expect(reduced, lessThan(policy.currentMb));
    policy.applied(reduced, now);
    for (var i = 0; i < 20; i++) {
      expect(policy.sample(gib, 48 * gib, now.add(Duration(seconds: i * 5))), isNull);
    }
    expect(policy.sample(gib, 48 * gib, now.add(const Duration(minutes: 2))), lessThan(reduced));
  });

  test('transient pressure resets; recovery grows in bounded steps', () {
    final policy = ResourceBudget()..initial(12 * gib, 48 * gib);
    final now = DateTime(2026);
    policy.sample(gib, 48 * gib, now);
    policy.sample(gib, 48 * gib, now);
    expect(policy.sample(10 * gib, 48 * gib, now), isNull);
    expect(policy.sample(gib, 48 * gib, now), isNull);
    int? increase;
    for (var i = 0; i < 12; i++) { increase = policy.sample(15 * gib, 48 * gib, now); }
    expect(increase, greaterThan(policy.currentMb));
    expect(increase! - policy.currentMb, lessThanOrEqualTo(2048));
  });

  test('unknown memory counters remain conservative and do not cause reload loops', () {
    final policy = ResourceBudget();
    expect(policy.initial(0, 0), 2048);
    expect(policy.sample(0, 0, DateTime(2026)), isNull);
  });

  test('resident dense weights do not reload for extra budget and pressure crosses placement threshold', () {
    final policy = ResourceBudget()..initial(30 * gib, 48 * gib);
    expect(policy.currentMb, 24 * 1024);
    expect(policy.forLayout(26 * 1024, fileBytes: 15 * gib, blockStreaming: false, mmapStreaming: false,
      expertBytes: 0, expertRoom: false, totalBytes: 48 * gib), isNull);
    final smaller = policy.forLayout(18 * 1024, fileBytes: 15 * gib, blockStreaming: false, mmapStreaming: false,
      expertBytes: 0, expertRoom: false, totalBytes: 48 * gib);
    expect(smaller, 16 * 1024 - 1);
  });

  test('split GGUF mmap hint reloads only when the larger cap can fit its file', () {
    final policy = ResourceBudget()..initial(14 * gib, 48 * gib);
    expect(policy.forLayout(12 * 1024, fileBytes: 15 * gib, blockStreaming: false, mmapStreaming: true,
      expertBytes: 0, expertRoom: false, totalBytes: 48 * gib), isNull);
    expect(policy.forLayout(17 * 1024, fileBytes: 15 * gib, blockStreaming: false, mmapStreaming: true,
      expertBytes: 0, expertRoom: false, totalBytes: 48 * gib), 17 * 1024);
  });
}
