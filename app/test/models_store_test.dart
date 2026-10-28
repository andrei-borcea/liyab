import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/state/models_store.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late Directory folder;
  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    folder = await Directory.systemTemp.createTemp('liyab-library-');
  });
  tearDown(() async => folder.delete(recursive: true));

  Future<File> model(String name) => File('${folder.path}/$name').writeAsBytes([0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0]);

  test('opening a local model persists one canonical reference without moving bytes', () async {
    final file = await model('local.gguf');
    final original = await file.readAsBytes();
    await ModelsStore.register(file);
    await ModelsStore.register(file);
    final prefs = await SharedPreferences.getInstance();
    expect(prefs.getStringList('externalModelPaths'), [await file.resolveSymbolicLinks()]);
    expect(await file.readAsBytes(), original);
    await ModelsStore.forget(File(await file.resolveSymbolicLinks()));
    expect(prefs.getStringList('externalModelPaths'), isEmpty);
    expect(await file.readAsBytes(), original);
  });

  test('invalid and truncated files are rejected before persisting', () async {
    for (final bytes in [[0x47], [0, 0, 0, 0]]) {
      final file = await File('${folder.path}/invalid.gguf').writeAsBytes(bytes);
      await expectLater(ModelsStore.register(file), throwsFormatException);
    }
    expect((await SharedPreferences.getInstance()).getStringList('externalModelPaths'), isNull);
  });

  test('split registration requires the first shard and all declared parts', () async {
    final first = await model('model-00001-of-00002.gguf');
    await expectLater(ModelsStore.register(first), throwsFormatException);
    final second = await model('model-00002-of-00002.gguf');
    await expectLater(ModelsStore.register(second), throwsFormatException);
    await ModelsStore.register(first);
    await ModelsStore.forget(File(await first.resolveSymbolicLinks()));
    expect(await first.exists(), isTrue);
    expect(await second.exists(), isTrue);
  });

  test('model parts never mix exports with different shard counts', () async {
    final first = await model('model-00001-of-00002.gguf');
    final second = await model('model-00002-of-00002.gguf');
    await model('model-00001-of-00003.gguf');
    expect(modelParts(first).map((file) => file.path), [first.path, second.path]);
  });

  test('missing models report file errors without leaving a reference', () async {
    await expectLater(ModelsStore.register(File('${folder.path}/missing.gguf')), throwsA(isA<FileSystemException>()));
    expect((await SharedPreferences.getInstance()).getStringList('externalModelPaths'), isNull);
  });
}
