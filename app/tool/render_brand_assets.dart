// Run from app/: flutter test tool/render_brand_assets.dart
// Rasterizes the repository's SVGs for native icon formats; no artwork edits.
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter_svg/flutter_svg.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('render native brand assets from the canonical vectors', () async {
    for (final entry in {'mark': 512, 'tray': 32, 'icon': 1024}.entries) {
      final asset = entry.key == 'tray' ? 'mark' : entry.key;
      final info = await vg.loadPicture(SvgStringLoader(await File('assets/brand/$asset.svg').readAsString()), null);
      final recorder = ui.PictureRecorder();
      final canvas = ui.Canvas(recorder)..scale(entry.value / info.size.width);
      canvas.drawPicture(info.picture);
      final picture = recorder.endRecording();
      final image = await picture.toImage(entry.value, entry.value);
      final data = await image.toByteData(format: ui.ImageByteFormat.png);
      await File('assets/brand/${entry.key}.png').writeAsBytes(data!.buffer.asUint8List());
      image.dispose(); picture.dispose(); info.picture.dispose();
    }
  });
}
