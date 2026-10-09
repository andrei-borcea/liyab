// Where models live on the device.
import 'dart:io';

import 'package:path_provider/path_provider.dart';

class LocalModel {
  const LocalModel(this.file, this.bytes, {required this.shared});

  /// The first file of the model (split models: part 1).
  final File file;

  /// All parts together.
  final int bytes;

  /// In the shared (FUSE) folder: readable, but without direct I/O, so
  /// streaming a model larger than RAM is several times slower.
  final bool shared;

  String get name => file.uri.pathSegments.last;
}

final _part = RegExp(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$');

abstract final class ModelsStore {
  /// App-private storage (plain f2fs: direct reads work, streaming at full speed).
  static Future<Directory> modelsDir() async {
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/models');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    return dir;
  }

  /// The shared app folder (`/sdcard/Android/data/<package>/files`), where adb can push models.
  static Future<Directory?> sharedDir() async => Platform.isAndroid ? await getExternalStorageDirectory() : null;

  /// GGUF models on the device, largest first; split models appear once.
  static Future<List<LocalModel>> list() async {
    final found = <LocalModel>[];
    Future<void> scan(Directory? dir, {required bool shared}) async {
      if (dir == null || !dir.existsSync()) return;
      final files = dir.listSync().whereType<File>().where((f) => f.path.endsWith('.gguf')).toList();
      for (final f in files) {
        final m = _part.firstMatch(f.uri.pathSegments.last);
        if (m != null && m.group(2) != '00001') continue;
        var bytes = 0;
        if (m == null) {
          bytes = f.lengthSync();
        } else {
          for (final p in files) {
            final pm = _part.firstMatch(p.uri.pathSegments.last);
            if (pm != null && pm.group(1) == m.group(1)) bytes += p.lengthSync();
          }
        }
        found.add(LocalModel(f, bytes, shared: shared));
      }
    }

    await scan(await modelsDir(), shared: false);
    await scan(await sharedDir(), shared: true);
    found.sort((a, b) => b.bytes.compareTo(a.bytes));
    return found;
  }
}

/// Every file of the model whose first file is `first` (one, or all parts of a split model).
List<File> modelParts(File first) {
  final m = _part.firstMatch(first.uri.pathSegments.last);
  if (m == null) return [first];
  return first.parent
      .listSync()
      .whereType<File>()
      .where((f) => _part.firstMatch(f.uri.pathSegments.last)?.group(1) == m.group(1))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));
}

String formatBytes(int bytes) =>
    bytes >= 1e9 ? '${(bytes / 1e9).toStringAsFixed(2)} GB' : '${(bytes / 1e6).toStringAsFixed(0)} MB';
