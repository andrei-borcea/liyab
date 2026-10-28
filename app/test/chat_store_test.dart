import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/state/chat_store.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  sqfliteFfiInit();
  late Directory directory;
  late ChatStore store;
  setUp(() async {
    directory = await Directory.systemTemp.createTemp('liyab-chats-');
    store = await ChatStore.open(path: '${directory.path}/chats.sqlite', factory: databaseFactoryFfi);
  });
  tearDown(() async { await store.close(); await directory.delete(recursive: true); });

  Future<void> save(String id, String question, {String model = '/models/a.gguf', String answer = 'A reply'}) =>
    store.save(id: id, modelPath: model, modelName: model.split('/').last,
      messages: [{'user': question, 'raw': answer, 'prefix': ''}], historyStart: 0);

  test('messages survive reopening SQLite and model filters isolate histories', () async {
    await save('one', 'First');
    await save('two', 'Second', model: '/models/b.gguf');
    await store.close();
    store = await ChatStore.open(path: '${directory.path}/chats.sqlite', factory: databaseFactoryFfi);
    expect((await store.get('one'))!.messages.single['user'], 'First');
    expect((await store.list(modelPath: '/models/b.gguf')).single.id, 'two');
  });

  test('search matches message content and treats SQL wildcards literally', () async {
    await save('one', 'A title', answer: 'The CPU is at 50% _busy_');
    await save('two', 'Another', answer: 'Other text');
    expect((await store.list(query: 'CPU')).single.id, 'one');
    expect((await store.list(query: '%')).single.id, 'one');
    expect((await store.list(query: '_busy_')).single.id, 'one');
    expect(await store.list(query: "' OR 1=1 --"), isEmpty);
  });

  test('pinning has priority over recency and later saves preserve pin and title', () async {
    await save('one', 'Old chat');
    await store.pin('one', true);
    await store.rename('one', 'My saved title');
    await save('two', 'New chat');
    await save('one', 'Old chat', answer: 'Updated reply');
    expect((await store.list()).first.id, 'one');
    expect((await store.list(pinnedOnly: true)).single.title, 'My saved title');
    expect((await store.get('one'))!.messages.single['raw'], 'Updated reply');
    await store.pin('one', false);
    expect(await store.list(pinnedOnly: true), isEmpty);
  });

  test('list queries omit message bodies and deletion removes search results', () async {
    await save('one', 'Keep private', answer: 'A long transcript');
    expect((await store.list()).single.messages, isEmpty);
    expect((await store.get('one'))!.messages, hasLength(1));
    await store.delete('one');
    expect(await store.get('one'), isNull);
    expect(await store.list(query: 'private'), isEmpty);
  });

  test('empty chats do not create rows and invalid titles do not overwrite', () async {
    await store.save(id: 'empty', modelPath: 'a', modelName: 'a', messages: [], historyStart: 0);
    expect(await store.list(), isEmpty);
    await save('one', 'Valid title');
    await expectLater(store.rename('one', '  '), throwsFormatException);
    expect((await store.get('one'))!.title, 'Valid title');
  });
}
