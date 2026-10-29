import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/state/settings.dart';

void main() {
  test('the i8mm tiles, the prerouter trace and the expert page cache are off by default', () {
    final d = ExperimentalSettings();
    expect(d.i8mmTiles, isFalse);
    expect(d.prerouterTrace, isFalse);
    expect(d.pageCacheMode, 0);
    expect(d.active, isEmpty);
  });

  test('they survive a save and a load, and show in the active list', () {
    final saved = ExperimentalSettings()
      ..i8mmTiles = true
      ..prerouterTrace = true
      ..pageCacheMode = 2;
    final loaded = ExperimentalSettings.fromJson(saved.toJson());
    expect(loaded.i8mmTiles, isTrue);
    expect(loaded.prerouterTrace, isTrue);
    expect(loaded.pageCacheMode, 2);
    expect(loaded.active, containsAll(['i8mm tiles', 'prerouter trace', 'expert page cache prefetches']));
  });

  test('settings saved before they existed load with them off', () {
    final loaded = ExperimentalSettings.fromJson({'v': 2, 'earlyExit': true});
    expect(loaded.earlyExit, isTrue);
    expect(loaded.i8mmTiles, isFalse);
    expect(loaded.prerouterTrace, isFalse);
    expect(loaded.pageCacheMode, 0);
  });
}
