// Model downloads into app storage (the folder the engine reads with direct
// I/O). background_downloader runs them outside the app (Android: a
// foreground data-sync job with a progress notification), resumes interrupted
// transfers, and keeps its own database, so downloads survive the app being
// closed. Every part is checked against the SHA-256 Hugging Face publishes.
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';

import 'package:background_downloader/background_downloader.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';

import '../state/models_store.dart';
import 'hugging_face.dart';

const _group = 'models';

enum DownloadPhase { queued, running, paused, checking, done, failed }

/// One model being downloaded (all its parts).
class ModelDownload {
  ModelDownload(this.key, this.name, this.totalBytes);

  final String key; // repo/path of part 1
  final String name; // file name of part 1
  final int totalBytes;
  final Map<String, Task> tasks = {}; // taskId -> task, one per part
  final Map<String, int> doneBytes = {}; // taskId -> bytes received
  final Set<String> completed = {}; // parts downloaded and checked
  DownloadPhase phase = DownloadPhase.queued;
  double speedMBps = 0;
  String? error;

  int get done => doneBytes.values.fold(0, (s, v) => s + v);
  double get fraction => totalBytes == 0 ? 0 : (done / totalBytes).clamp(0, 1);
}

class Downloads extends ChangeNotifier {
  Downloads._();

  final Map<String, ModelDownload> active = {};
  final _fd = FileDownloader();

  static Future<Downloads> start() async {
    final d = Downloads._();
    await d._fd.configure(androidConfig: [(Config.runInForeground, Config.always)]);
    d._fd.configureNotificationForGroup(
      _group,
      running: const TaskNotification('{displayName}', 'Downloading, {progress}, {networkSpeed}'),
      complete: const TaskNotification('{displayName}', 'Downloaded'),
      error: const TaskNotification('{displayName}', 'Download stopped; open Liyab to retry'),
      paused: const TaskNotification('{displayName}', 'Paused'),
      progressBar: true,
    );
    d._fd.updates.listen(d._onUpdate);
    await d._fd.start();
    // Downloads still in the database from an earlier run.
    for (final r in await d._fd.database.allRecords(group: _group)) {
      if (r.status == TaskStatus.complete || r.status == TaskStatus.canceled) continue;
      final m = d._track(r.task);
      if (m == null) continue;
      m.doneBytes[r.task.taskId] = (r.progress.clamp(0, 1) * _meta(r.task)['size']).round();
      m.phase = switch (r.status) {
        TaskStatus.paused => DownloadPhase.paused,
        TaskStatus.failed || TaskStatus.notFound => DownloadPhase.failed,
        _ => DownloadPhase.running,
      };
    }
    return d;
  }

  static Map<String, dynamic> _meta(Task t) => jsonDecode(t.metaData) as Map<String, dynamic>;

  ModelDownload? _track(Task task) {
    if (task.metaData.isEmpty) return null;
    final meta = _meta(task);
    final m = active.putIfAbsent(meta['model'] as String,
        () => ModelDownload(meta['model'] as String, meta['name'] as String, meta['total'] as int));
    m.tasks[task.taskId] = task;
    return m;
  }

  /// Whether notifications may be shown (progress of a running download).
  Future<bool> ensureNotificationPermission() async {
    var status = await _fd.permissions.status(PermissionType.notifications);
    if (status != PermissionStatus.granted) status = await _fd.permissions.request(PermissionType.notifications);
    return status == PermissionStatus.granted;
  }

  Future<PermissionStatus> notificationPermission() => _fd.permissions.status(PermissionType.notifications);

  /// Starts downloading every part of `model` into app storage.
  Future<void> download(HubModel model) async {
    final key = '${model.repo}/${model.path}';
    if (active.containsKey(key)) return;
    await ensureNotificationPermission(); // downloads also run without it, silently
    final dir = await ModelsStore.modelsDir();
    final m = ModelDownload(key, model.fileName, model.size);
    active[key] = m;
    for (final (i, part) in model.parts.indexed) {
      if (File('${dir.path}/${part.fileName}').existsSync() &&
          File('${dir.path}/${part.fileName}').lengthSync() == part.size) {
        m.doneBytes[part.path] = part.size; // already here (e.g. an earlier attempt)
        continue;
      }
      final task = ParallelDownloadTask(
        url: part.url.toString(),
        chunks: 4, // parallel range requests: one connection rarely fills the link
        filename: part.fileName,
        directory: 'models',
        baseDirectory: BaseDirectory.applicationSupport,
        group: _group,
        updates: Updates.statusAndProgress,
        allowPause: true,
        retries: 5,
        displayName: model.parts.length == 1 ? model.fileName : '${model.fileName} (part ${i + 1} of ${model.parts.length})',
        metaData: jsonEncode({
          'model': key,
          'name': model.fileName,
          'total': model.size,
          'size': part.size,
          'sha256': part.sha256,
        }),
      );
      m.tasks[task.taskId] = task;
      m.doneBytes[task.taskId] = 0;
      await _fd.enqueue(task);
    }
    m.phase = m.tasks.isEmpty ? DownloadPhase.done : DownloadPhase.running;
    notifyListeners();
  }

  Future<void> pause(ModelDownload m) async {
    for (final t in m.tasks.values.whereType<DownloadTask>()) {
      if (!m.completed.contains(t.taskId)) await _fd.pause(t);
    }
  }

  Future<void> resume(ModelDownload m) async {
    for (final t in m.tasks.values.whereType<DownloadTask>()) {
      if (m.completed.contains(t.taskId)) continue;
      if (!await _fd.resume(t)) await _fd.enqueue(t); // nothing to resume from: start over
    }
    m
      ..phase = DownloadPhase.running
      ..error = null;
    notifyListeners();
  }

  /// Stops the download and deletes what was received.
  Future<void> cancel(ModelDownload m) async {
    await _fd.cancelTasksWithIds(m.tasks.keys);
    final dir = await ModelsStore.modelsDir();
    for (final t in m.tasks.values) {
      final f = File('${dir.path}/${t.filename}');
      if (f.existsSync()) f.deleteSync();
    }
    active.remove(m.key);
    notifyListeners();
  }

  void dismiss(ModelDownload m) {
    active.remove(m.key);
    notifyListeners();
  }

  Future<void> _onUpdate(TaskUpdate update) async {
    final m = _track(update.task);
    if (m == null) return;
    final size = _meta(update.task)['size'] as int;
    switch (update) {
      case TaskProgressUpdate(:final progress, :final networkSpeed):
        if (progress >= 0) m.doneBytes[update.task.taskId] = (progress * size).round();
        if (networkSpeed > 0) m.speedMBps = networkSpeed;
      case TaskStatusUpdate(:final status, :final exception):
        switch (status) {
          case TaskStatus.complete:
            m.doneBytes[update.task.taskId] = size;
            await _check(m, update.task);
          case TaskStatus.paused:
            m.phase = DownloadPhase.paused;
          case TaskStatus.failed || TaskStatus.notFound:
            m
              ..phase = DownloadPhase.failed
              ..error = exception?.description ?? 'the server refused the file';
          case TaskStatus.canceled:
            break;
          default:
            if (m.phase != DownloadPhase.checking) m.phase = DownloadPhase.running;
        }
    }
    notifyListeners();
  }

  /// Verifies a finished part against its published SHA-256 (on another isolate).
  Future<void> _check(ModelDownload m, Task task) async {
    final expected = _meta(task)['sha256'] as String?;
    final path = await task.filePath();
    if (expected != null && expected.length == 64) {
      m.phase = DownloadPhase.checking;
      notifyListeners();
      final actual = await Isolate.run(() async => (await sha256.bind(File(path).openRead()).first).toString());
      if (actual != expected.toLowerCase()) {
        File(path).deleteSync();
        m
          ..phase = DownloadPhase.failed
          ..error = 'the file arrived damaged (checksum mismatch); retry to download it again';
        return;
      }
    }
    m.completed.add(task.taskId);
    m.phase = m.completed.length == m.tasks.length ? DownloadPhase.done : DownloadPhase.running;
  }
}
