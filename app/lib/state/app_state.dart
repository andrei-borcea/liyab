// The loaded model and the conversation, shared by every screen. Engine calls
// go through EngineService, which runs them one at a time on its isolate.
import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../chat/chat_template.dart';
import '../engine/engine_service.dart';
import '../engine/liyab_ffi.dart';
import '../models/downloads.dart';
import 'device_monitor.dart';
import 'models_store.dart';
import 'settings.dart';

class ChatMessage {
  ChatMessage(this.user, this.prefix);

  final String user;

  /// What the template put at the start of the reply (thinking control).
  final String prefix;

  /// Everything the model generated, reasoning included.
  String raw = '';
  GenerationStats? stats;
  String? error;
  bool streaming = true;

  /// The reply as the next prompt replays it (token for token what the engine saw).
  String get exact => prefix + raw;

  /// The model's reasoning, when it opened a `<think>` block itself or thinking is on.
  String? get reasoning {
    final open = prefix.endsWith('<think>\n') ? 0 : raw.indexOf('<think>');
    if (open < 0) return null;
    final start = prefix.endsWith('<think>\n') ? 0 : open + '<think>'.length;
    final close = raw.indexOf('</think>', start);
    return (close < 0 ? raw.substring(start) : raw.substring(start, close)).trim();
  }

  /// The answer without the reasoning block.
  String get answer {
    final close = raw.indexOf('</think>');
    if (close >= 0) return raw.substring(close + '</think>'.length).trimLeft();
    return reasoning != null ? '' : raw;
  }

  bool get thinkingNow => streaming && reasoning != null && !raw.contains('</think>');
}

/// One line of the activity log.
class LogLine {
  LogLine(this.text, {this.engine = false}) : at = DateTime.now();
  final DateTime at;
  final String text;
  final bool engine; // from libliyab's log (level letter first), else the app
}

class AppState extends ChangeNotifier {
  AppState._(this.engine, this.prefs, this.downloads) : device = DeviceSettings(prefs) {
    monitor = DeviceMonitor(engine, (lines) {
      for (final l in lines) {
        _log(l, engine: true);
      }
    });
  }

  final EngineService engine;
  final SharedPreferences prefs;
  final DeviceSettings device;
  final Downloads downloads;
  late final DeviceMonitor monitor;

  String status = 'No model loaded';
  bool loading = false;
  bool generating = false;
  String? modelPath;
  String modelName = '';
  String description = '';
  ChatTemplate template = ChatTemplate.zephyr;
  bool thinkingSupported = false;
  ModelSettings settings = ModelSettings();
  final List<ChatMessage> messages = [];

  /// Name of the model being moved into app storage, if any.
  String? moving;
  final List<LogLine> log = [];

  /// Whether the welcome screen was shown.
  bool get welcomed => prefs.getBool('welcomed') ?? false;
  set welcomed(bool v) {
    prefs.setBool('welcomed', v);
    notifyListeners();
  }

  static Future<AppState> create() async {
    LiyabLib.instance
      ..setLogLevel(1) // info: loads, streaming, warnings
      ..logBufferEnable(256 * 1024);
    final state = AppState._(
        await EngineService.start(), await SharedPreferences.getInstance(), await Downloads.start());
    final last = state.device.lastModel;
    if (last != null && File(last).existsSync()) unawaited(state.load(last));
    return state;
  }

  void _log(String line, {bool engine = false}) {
    log.add(LogLine(line, engine: engine));
    if (log.length > 1000) log.removeAt(0);
  }

  void clearLog() {
    log.clear();
    notifyListeners();
  }

  Future<void> unload() async {
    if (loading || generating) return;
    await engine.unload();
    _log('Unloaded $modelName');
    modelPath = null;
    modelName = '';
    description = '';
    messages.clear();
    status = 'No model loaded';
    device.lastModel = null;
    notifyListeners();
  }

  Future<void> load(String path) async {
    if (loading) return;
    final name = File(path).uri.pathSegments.last;
    loading = true;
    status = 'Loading $name…';
    messages.clear();
    notifyListeners();
    try {
      final gpu = device.useGpu;
      final model = await engine.load(LoadOptions(
        modelPath: path,
        backend: gpu ? LiyabBackend.vulkan : LiyabBackend.cpu,
        contextLength: ModelSettings.contextFor(prefs, name),
        skinThresholdC: device.thermalLimitC,
        memoryBudgetMb: device.memoryBudgetMb,
      ));
      final (detected, thinks) = await ChatTemplate.detect(engine);
      template = detected;
      thinkingSupported = thinks;
      settings = ModelSettings.resolve(prefs, name, model.metadata);
      modelPath = path;
      modelName = name;
      description = model.description;
      device.lastModel = path;
      status = '$name · ${gpu ? 'GPU' : 'CPU'} · ready in ${model.loadSeconds.toStringAsFixed(1)} s';
      _log('Loaded $name in ${model.loadSeconds.toStringAsFixed(2)} s (${template.label})');
      for (final line in description.split('\n')) {
        _log('  $line');
      }
      unawaited(prepareSystemPrompt());
    } on EngineException catch (e) {
      status = 'Failed to load $name: $e';
      _log('ERROR: $e');
    } finally {
      loading = false;
      notifyListeners();
    }
  }

  /// Processes the system block in the background, so the first message only
  /// costs its own tokens (the engine keeps that context and a snapshot of it).
  Future<void> prepareSystemPrompt() async {
    if (!engine.loaded) return;
    final watch = Stopwatch()..start();
    try {
      await engine.prefill(template.system(settings.systemPrompt));
      _log('System prompt prepared in ${(watch.elapsedMilliseconds / 1000).toStringAsFixed(1)} s');
    } on EngineException catch (e) {
      _log('System prompt not prepared: $e');
    }
  }

  Thinking get _thinking => !thinkingSupported ? Thinking.none : (settings.thinking ? Thinking.on : Thinking.off);

  /// The prompt for `user` with as much history as fits the context (oldest
  /// turns dropped first, counted with the model's own tokenizer).
  Future<String> _prompt(String user) async {
    final history = [
      for (final m in messages)
        if (m.error == null && !m.streaming) Turn(user: m.user, exact: m.exact)
    ];
    final budget = settings.contextLength - settings.maxTokens;
    while (true) {
      final prompt = template.build(settings.systemPrompt, history, user, _thinking);
      if (history.isEmpty || await engine.countTokens(prompt) <= budget) return prompt;
      history.removeAt(0);
    }
  }

  Future<void> send(String text) async {
    if (generating || !engine.loaded || text.trim().isEmpty) return;
    generating = true;
    final prompt = await _prompt(text);
    final message = ChatMessage(text, template.assistantPrefix(_thinking));
    messages.add(message);
    notifyListeners();
    // Streaming redraws are batched (~15 a second): the engine runs on all but
    // one core, and rebuilding the reply for every token would take CPU time
    // from it.
    var lastPaint = DateTime.fromMillisecondsSinceEpoch(0);
    try {
      await for (final event in engine.generate(
          prompt,
          SamplingOptions(
              temperature: settings.temperature,
              topP: settings.topP,
              topK: settings.topK,
              maxTokens: settings.maxTokens))) {
        switch (event) {
          case TextPiece(:final text):
            message.raw += text;
            final now = DateTime.now();
            if (now.difference(lastPaint) < const Duration(milliseconds: 66)) continue;
            lastPaint = now;
          case GenerationDone(:final stats):
            message.stats = stats;
        }
        notifyListeners();
      }
    } on EngineException catch (e) {
      message.error = e.message;
      _log('ERROR: $e');
    } finally {
      message.streaming = false;
      generating = false;
      notifyListeners();
    }
  }

  void stop() => engine.cancel();

  /// Starts a new conversation (the engine keeps the system prompt's context).
  void newChat() {
    if (generating) return;
    messages.clear();
    notifyListeners();
  }

  Future<void> saveSettings() async {
    await settings.save(prefs, modelName);
    notifyListeners();
  }

  Future<List<LocalModel>> localModels() => ModelsStore.list();

  /// Moves a model (all parts) from the shared folder into app storage, where
  /// the engine's direct reads work and streaming runs at full speed.
  Future<void> moveToAppStorage(LocalModel model) async {
    if (moving != null) return;
    moving = model.name;
    notifyListeners();
    final watch = Stopwatch()..start();
    try {
      final dir = await ModelsStore.modelsDir();
      for (final part in modelParts(model.file)) {
        final name = part.uri.pathSegments.last;
        final tmp = await part.copy('${dir.path}/$name.moving');
        await tmp.rename('${dir.path}/$name');
        await part.delete();
      }
      final moved = '${dir.path}/${model.name}';
      if (device.lastModel == model.file.path) device.lastModel = moved;
      _log('Moved ${model.name} to app storage in ${watch.elapsed.inSeconds} s');
    } on FileSystemException catch (e) {
      _log('Move failed: ${e.message}');
      status = 'Could not move ${model.name}: ${e.message}';
    } finally {
      moving = null;
      notifyListeners();
    }
  }
}
