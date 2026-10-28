// Where models live on the device.
import 'dart:io';
import 'package:shared_preferences/shared_preferences.dart';

import 'package:path_provider/path_provider.dart';

class LocalModel {
  const LocalModel(this.file, this.bytes, {required this.shared, this.external = false});

  /// The first file of the model (split models: part 1).
  final File file;

  /// All parts together.
  final int bytes;

  /// In the shared (FUSE) folder: readable, but without direct I/O, so
  /// streaming a model larger than RAM is several times slower.
  final bool shared;

  /// User-owned file: removing its library entry must never delete the file.
  final bool external;
  bool get available => file.existsSync();

  String get name => file.uri.pathSegments.last;
}

final _part = RegExp(r'^(.*)-(\d{5})-of-(\d{5})\.gguf$');

abstract final class ModelsStore {
  static const _references = 'externalModelPaths';

  /// Registers a readable GGUF in place. Split models must include every shard.
  /// No model bytes are copied, moved, or deleted on desktop.
  static Future<void> register(File file) async {
    if (!file.path.toLowerCase().endsWith('.gguf')) {
      throw const FormatException('Choose a .gguf model file.');
    }
    final canonical = await file.resolveSymbolicLinks();
    final first = File(canonical);
    final match = _part.firstMatch(first.uri.pathSegments.last);
    if (match != null) {
      if (match.group(2) != '00001') {
        throw const FormatException('Choose the first shard (00001) of the model.');
      }
      final count = int.parse(match.group(3)!);
      for (var i = 1; i <= count; i++) {
        final shard = File('${first.parent.path}/${match.group(1)}-${i.toString().padLeft(5, '0')}-of-${match.group(3)}.gguf');
        if (!await shard.exists()) throw FormatException('Missing model shard: ${shard.uri.pathSegments.last}');
      }
    }
    final handle = await first.open();
    try {
      final magic = await handle.read(4);
      if (magic.length != 4 || magic[0] != 0x47 || magic[1] != 0x47 || magic[2] != 0x55 || magic[3] != 0x46) {
        throw const FormatException('This file does not contain a GGUF model.');
      }
    } finally {
      await handle.close();
    }
    final prefs = await SharedPreferences.getInstance();
    final paths = {...?prefs.getStringList(_references), canonical};
    if (!await prefs.setStringList(_references, paths.toList())) {
      throw const FileSystemException('Could not save the model reference');
    }
  }

  /// Removes a reference, leaving all user-owned model shards untouched.
  static Future<void> forget(File file) async {
    final prefs = await SharedPreferences.getInstance();
    final paths = prefs.getStringList(_references) ?? [];
    paths.remove(file.path);
    if (!await prefs.setStringList(_references, paths)) {
      throw const FileSystemException('Could not remove the model reference');
    }
  }

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
      final files = dir.listSync().whereType<File>().where((f) => f.path.toLowerCase().endsWith('.gguf')).toList();
      for (final f in files) {
        final m = _part.firstMatch(f.uri.pathSegments.last);
        if (m != null && m.group(2) != '00001') continue;
        var bytes = 0;
        if (m == null) {
          bytes = f.lengthSync();
        } else {
          for (final p in files) {
            final pm = _part.firstMatch(p.uri.pathSegments.last);
            if (pm != null && pm.group(1) == m.group(1) && pm.group(3) == m.group(3)) bytes += p.lengthSync();
          }
        }
        found.add(LocalModel(f, bytes, shared: shared));
      }
    }

    await scan(await modelsDir(), shared: false);
    await scan(await sharedDir(), shared: true);
    final prefs = await SharedPreferences.getInstance();
    final known = <String>{};
    for (final model in found) {
      known.add(await model.file.resolveSymbolicLinks());
    }
    for (final path in prefs.getStringList(_references) ?? <String>[]) {
      if (!known.add(path)) continue;
      final file = File(path);
      var bytes = 0;
      if (await file.exists()) {
        for (final part in modelParts(file)) { bytes += await part.length(); }
      }
      found.add(LocalModel(file, bytes, shared: false, external: true));
    }
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
      .where((f) => _part.firstMatch(f.uri.pathSegments.last)?.group(1) == m.group(1) && _part.firstMatch(f.uri.pathSegments.last)?.group(3) == m.group(3))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));
}

String formatBytes(int bytes) =>
    bytes >= 1e9 ? '${(bytes / 1e9).toStringAsFixed(2)} GB' : '${(bytes / 1e6).toStringAsFixed(0)} MB';
