// Local conversation archive. SQLite work runs on the driver's worker isolate.
import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart' as native;
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

class SavedChat {
  const SavedChat({required this.id, required this.title, required this.modelPath,
    required this.modelName, required this.updated, required this.pinned,
    required this.messages, required this.historyStart});
  final String id, title, modelPath, modelName;
  final DateTime updated;
  final bool pinned;
  final List<Map<String, Object?>> messages;
  final int historyStart;

  factory SavedChat.fromRow(Map<String, Object?> row) => SavedChat(
    id: row['id'] as String, title: row['title'] as String,
    modelPath: row['model_path'] as String, modelName: row['model_name'] as String,
    updated: DateTime.fromMillisecondsSinceEpoch(row['updated'] as int), pinned: row['pinned'] == 1,
    messages: (jsonDecode(row['messages'] as String? ?? '[]') as List).cast<Map<String, Object?>>(),
    historyStart: row['history_start'] as int,
  );
}

class ChatStore extends ChangeNotifier {
  ChatStore._(this._db);
  final Database _db;

  static String newId() {
    final random = Random.secure();
    return List.generate(16, (_) => random.nextInt(256).toRadixString(16).padLeft(2, '0')).join();
  }

  static Future<ChatStore> open({String? path, DatabaseFactory? factory}) async {
    if (factory == null) {
      if (Platform.isWindows || Platform.isLinux) {
        sqfliteFfiInit();
        factory = databaseFactoryFfi;
      } else {
        factory = native.databaseFactory;
      }
    }
    final location = path ?? '${(await getApplicationSupportDirectory()).path}/chats.sqlite';
    final db = await factory.openDatabase(location, options: OpenDatabaseOptions(
      version: 1,
      onConfigure: (db) async { await db.execute('PRAGMA foreign_keys = ON'); },
      onCreate: (db, _) async {
        await db.execute('''CREATE TABLE chats (
          id TEXT PRIMARY KEY, title TEXT NOT NULL, model_path TEXT NOT NULL,
          model_name TEXT NOT NULL, messages TEXT NOT NULL, search_text TEXT NOT NULL,
          history_start INTEGER NOT NULL DEFAULT 0, pinned INTEGER NOT NULL DEFAULT 0,
          updated INTEGER NOT NULL)''');
        await db.execute('CREATE INDEX chats_recency ON chats(pinned DESC, updated DESC)');
        await db.execute('CREATE INDEX chats_model ON chats(model_path)');
      },
    ));
    return ChatStore._(db);
  }

  /// Literal, case-insensitive substring search across title and message text.
  /// Bound parameters keep user input out of SQL. Pins retain priority.
  Future<List<SavedChat>> list({String query = '', bool pinnedOnly = false, String? modelPath}) async {
    final where = <String>[];
    final args = <Object?>[];
    if (query.trim().isNotEmpty) {
      final escaped = query.trim().toLowerCase().replaceAll('\\', '\\\\').replaceAll('%', '\\%').replaceAll('_', '\\_');
      where.add("(LOWER(title) LIKE ? ESCAPE '\\' OR search_text LIKE ? ESCAPE '\\')");
      args.addAll(['%$escaped%', '%$escaped%']);
    }
    if (pinnedOnly) where.add('pinned = 1');
    if (modelPath != null) { where.add('model_path = ?'); args.add(modelPath); }
    final rows = await _db.query('chats', columns: ['id', 'title', 'model_path', 'model_name', 'history_start', 'pinned', 'updated'], where: where.isEmpty ? null : where.join(' AND '),
      whereArgs: args, orderBy: 'pinned DESC, updated DESC');
    return rows.map(SavedChat.fromRow).toList();
  }

  Future<SavedChat?> get(String id) async {
    final rows = await _db.query('chats', where: 'id = ?', whereArgs: [id], limit: 1);
    return rows.isEmpty ? null : SavedChat.fromRow(rows.single);
  }

  /// Preserve user title/pin when saving a reply. One transaction makes an
  /// interrupted update leave either the previous or the complete new chat.
  Future<void> save({required String id, required String modelPath, required String modelName,
    required List<Map<String, Object?>> messages, required int historyStart}) async {
    if (messages.isEmpty) return;
    final first = '${messages.first['user'] ?? 'Untitled chat'}'.replaceAll(RegExp(r'\s+'), ' ').trim();
    final title = first.length > 72 ? '${first.substring(0, 72)}…' : first;
    final text = messages.map((m) => '${m['user'] ?? ''}\n${m['raw'] ?? ''}\n${m['done'] ?? ''}').join('\n').toLowerCase();
    await _db.transaction((txn) async {
      await txn.rawInsert('''INSERT INTO chats(id, title, model_path, model_name, messages, search_text, history_start, updated)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET messages=excluded.messages, search_text=excluded.search_text,
          history_start=excluded.history_start, updated=excluded.updated''',
        [id, title.isEmpty ? 'Untitled chat' : title, modelPath, modelName, jsonEncode(messages), text,
          historyStart, DateTime.now().millisecondsSinceEpoch]);
    });
    notifyListeners();
  }

  Future<void> pin(String id, bool value) async {
    await _db.update('chats', {'pinned': value ? 1 : 0}, where: 'id = ?', whereArgs: [id]);
    notifyListeners();
  }

  Future<void> rename(String id, String title) async {
    final clean = title.trim();
    if (clean.isEmpty || clean.length > 200) throw const FormatException('Use a title between 1 and 200 characters.');
    await _db.update('chats', {'title': clean}, where: 'id = ?', whereArgs: [id]);
    notifyListeners();
  }

  Future<void> delete(String id) async {
    await _db.delete('chats', where: 'id = ?', whereArgs: [id]);
    notifyListeners();
  }

  Future<void> close() => _db.close();
}
