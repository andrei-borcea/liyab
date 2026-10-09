// The loaded model and the conversation, shared by every screen. Engine calls
// go through EngineService, which runs them one at a time on its isolate.
import 'dart:async';
import 'dart:io';

import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/widgets.dart' show AppLifecycleListener, AppLifecycleState;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../agent/tool_format.dart';
import '../agent/tools.dart';
import '../chat/chat_template.dart';
import '../engine/engine_service.dart';
import '../engine/liyab_ffi.dart';
import '../models/downloads.dart';
import 'device_monitor.dart';
import 'models_store.dart';
import 'settings.dart';

class ChatMessage {
  ChatMessage(this.user, this.prefix, {String? promptUser}) : promptUser = promptUser ?? user;

  /// What the user wrote (shown).
  final String user;

  /// The user turn as the model saw it (with the current date and time).
  final String promptUser;

  /// What the template put at the start of the current reply segment (thinking control).
  String prefix;

  /// The current segment as generated, reasoning included. A reply that calls
  /// tools has several segments: each call ends one, its result follows.
  String raw = '';

  /// Earlier segments with their tool results, exactly as the model saw them.
  String done = '';

  /// The tools this reply used, shown above the answer.
  final List<ToolStep> steps = [];
  GenerationStats? stats;
  String? error;
  bool streaming = true;

  /// The reply as the next prompt replays it (token for token what the engine saw).
  String get exact => done + prefix + raw;

  /// The model's reasoning, when it opened a `<think>` block itself or thinking is on.
  String? get reasoning {
    final open = prefix.endsWith('<think>\n') ? 0 : raw.indexOf('<think>');
    if (open < 0) return null;
    final start = prefix.endsWith('<think>\n') ? 0 : open + '<think>'.length;
    final close = raw.indexOf('</think>', start);
    return (close < 0 ? raw.substring(start) : raw.substring(start, close)).trim();
  }

  /// The answer without the reasoning block, or a tool call being written.
  String get answer {
    final close = raw.indexOf('</think>');
    var text = close >= 0 ? raw.substring(close + '</think>'.length).trimLeft() : (reasoning != null ? '' : raw);
    final call = text.indexOf('<tool_call>');
    if (call >= 0) text = text.substring(0, call).trimRight();
    return text;
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
  AppState._(this.engine, this.prefs, this.downloads) : device = DeviceSettings(prefs), toolbox = Toolbox(prefs) {
    monitor = DeviceMonitor(engine, (lines) {
      for (final l in lines) {
        _log(l, engine: true);
      }
    });
    // States, not transitions: Android may destroy the activity in the
    // background, and a new one then goes from detached straight to resumed,
    // which onShow never reports.
    _lifecycle = AppLifecycleListener(onStateChange: (state) {
      if (state == AppLifecycleState.resumed) {
        _backOnScreen();
      } else if ((state == AppLifecycleState.hidden || state == AppLifecycleState.detached) && !_away) {
        _away = true;
        _scheduleRelease();
      }
    });
  }

  // Idle release: a loaded model holds its memory (GBs) while Liyab sits in
  // the background, and the process outlives the window (the notification
  // listener keeps it). After a minute hidden the expert cache is emptied
  // (2.4 GB on a 35B MoE; the model stays loaded, so the assistant answers at
  // once, only its first tokens are slower); after device.releaseAfterMinutes
  // the model is unloaded, and loads again, chat kept, once Liyab is back.
  late final AppLifecycleListener _lifecycle;
  bool _away = false; // off screen since the timers were started
  Timer? _trimTimer;
  Timer? _releaseTimer;
  String? _released; // the model unloaded while idle

  static const _trimAfter = Duration(minutes: 1);

  void _scheduleRelease() {
    _trimTimer?.cancel();
    _releaseTimer?.cancel();
    _trimTimer = Timer(_trimAfter, _trimIdle);
    final minutes = device.releaseAfterMinutes;
    if (minutes > 0) _releaseTimer = Timer(Duration(minutes: minutes), _releaseIdle);
  }

  Future<void> _trimIdle() async {
    if (!engine.loaded) return;
    if (generating || loading) {
      _trimTimer = Timer(_trimAfter, _trimIdle); // busy: try again later
      return;
    }
    final bytes = await engine.trimMemory();
    if (bytes > 0) _log('Freed ${(bytes / (1 << 30)).toStringAsFixed(2)} GB of cached experts in the background');
  }

  Future<void> _releaseIdle() async {
    final path = modelPath;
    if (path == null || _released != null) return;
    if (generating || loading || moving != null) return _scheduleRelease(); // busy: try again later
    _released = path; // set first: coming back during the unload reloads it (commands run in order)
    if (messages.isNotEmpty) {
      // The conversation's context, so the reload continues it instead of processing the chat again.
      try {
        await engine.saveState(await _conversationFile());
      } on EngineException catch (e) {
        _log('Conversation not saved: $e');
      }
    }
    _prepared = null;
    await engine.unload();
    if (_released != path) return; // back on screen meanwhile: already reloading
    status = 'Model unloaded while idle';
    _log('Unloaded $modelName after ${device.releaseAfterMinutes} min in the background');
    notifyListeners();
  }

  @override
  void dispose() {
    _lifecycle.dispose();
    _trimTimer?.cancel();
    _releaseTimer?.cancel();
    super.dispose();
  }

  void _backOnScreen() {
    _trimTimer?.cancel();
    _releaseTimer?.cancel();
    _trimTimer = _releaseTimer = null;
    _away = false;
    final path = _released;
    _released = null;
    if (path != null && File(path).existsSync()) unawaited(load(path, keepChat: true));
  }

  final EngineService engine;
  final SharedPreferences prefs;
  final DeviceSettings device;
  final Downloads downloads;
  final Toolbox toolbox;
  late final DeviceMonitor monitor;

  /// How the loaded model calls tools (none: no tools offered).
  ToolDialect toolDialect = ToolDialect.none;

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
    // App events in logcat too (adb logcat -s flutter), without tool arguments: they can hold the user's data.
    if (!engine) debugPrint('liyab-app: ${line.startsWith('Tool ') ? line.split('(').first : line}');
    if (log.length > 1000) log.removeAt(0);
  }

  void clearLog() {
    log.clear();
    notifyListeners();
  }

  Future<void> unload() async {
    if (loading || generating) return;
    _released = null;
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

  /// Loads `path`; `keepChat` keeps the conversation (a reload after an idle
  /// release). While a load runs, returns that load.
  Future<void> load(String path, {bool keepChat = false}) {
    if (loading) return _pendingLoad ?? Future.value();
    return _pendingLoad = _load(path, keepChat: keepChat);
  }

  Future<void>? _pendingLoad;

  Future<void> _load(String path, {required bool keepChat}) async {
    _released = null;
    final name = File(path).uri.pathSegments.last;
    loading = true;
    status = 'Loading $name…';
    if (!keepChat) messages.clear();
    notifyListeners();
    try {
      final gpu = device.useGpu;
      _prepared = null;
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
      toolDialect =
          template == ChatTemplate.chatml ? ToolDialect.of(model.metadata['tokenizer.chat_template'] ?? '') : ToolDialect.none;
      modelPath = path;
      modelName = name;
      description = model.description;
      device.lastModel = path;
      status = '$name · ${gpu ? 'GPU' : 'CPU'} · ready in ${model.loadSeconds.toStringAsFixed(1)} s';
      _log('Loaded $name in ${model.loadSeconds.toStringAsFixed(2)} s (${template.label}, tools: ${toolDialect.name})');
      for (final line in description.split('\n')) {
        _log('  $line');
      }
      _preparing = keepChat ? _restoreConversation() : prepareSystemPrompt();
      if (!keepChat) _discardConversation();
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
  String? _prepared; // the system block the engine last processed

  /// Where the processed system block of this model is kept: one file per
  /// model, named with the block's hash (a changed prompt or tool list is a
  /// different file; the older one is deleted).
  /// The context being prepared after a load (system prompt or saved conversation).
  Future<void>? _preparing;

  /// Where the conversation's context is kept while its model is unloaded for being idle.
  Future<String> _conversationFile() async {
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/states');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    return '${dir.path}/$modelName.conversation.state';
  }

  Future<void> _discardConversation() async {
    final file = File(await _conversationFile());
    if (file.existsSync()) file.deleteSync();
  }

  /// After an idle reload: the conversation's saved context (system prompt
  /// included), else the system prompt alone. The file is used once.
  Future<void> _restoreConversation() async {
    final file = File(await _conversationFile());
    if (file.existsSync()) {
      final watch = Stopwatch()..start();
      try {
        final n = await engine.loadState(file.path);
        _prepared = await _systemBlock(); // the conversation starts with it
        _log('Conversation restored ($n tokens) in ${watch.elapsedMilliseconds} ms');
        return;
      } on EngineException catch (e) {
        _log('Saved conversation not usable ($e)');
      } finally {
        file.deleteSync();
      }
    }
    await prepareSystemPrompt();
  }

  Future<(Directory, String)> _stateFile(String block) async {
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/states');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    final hash = sha1.convert(utf8.encode('$block|${device.useGpu}')).toString().substring(0, 16);
    return (dir, '${dir.path}/$modelName.$hash.state');
  }

  /// Gets the system block into the engine's context before the first message:
  /// restored from its saved state when there is one (a fraction of a second),
  /// else processed (about a minute on a 35B MoE with tools) and saved for the
  /// next start. The first message then only costs its own tokens.
  Future<void> prepareSystemPrompt() async {
    if (!engine.loaded) return;
    final block = await _systemBlock();
    if (block == _prepared) return;
    _prepared = block;
    final watch = Stopwatch()..start();
    final (dir, file) = await _stateFile(block);
    if (File(file).existsSync()) {
      try {
        final n = await engine.loadState(file);
        _log('System prompt restored ($n tokens) in ${watch.elapsedMilliseconds} ms');
        return;
      } on EngineException catch (e) {
        _log('Saved system prompt not usable ($e); processing it again');
        File(file).deleteSync();
      }
    }
    try {
      await engine.prefill(block);
      _log('System prompt prepared in ${(watch.elapsedMilliseconds / 1000).toStringAsFixed(1)} s');
      await engine.saveState(file);
      for (final old in dir.listSync().whereType<File>()) {
        if (old.path != file && old.uri.pathSegments.last.startsWith('$modelName.')) old.deleteSync();
      }
    } on EngineException catch (e) {
      _log('System prompt not prepared: $e');
    }
  }

  /// The system block: the system prompt, and the enabled tools when the model can call them.
  Future<String> _systemBlock() async => toolDialect == ToolDialect.none
      ? template.system(settings.systemPrompt)
      : toolDialect.system(settings.systemPrompt, await toolbox.enabled());

  Thinking get _thinking => !thinkingSupported ? Thinking.none : (settings.thinking ? Thinking.on : Thinking.off);

  /// The prompt for `user` with as much history as fits the context (oldest
  /// turns dropped first, counted with the model's own tokenizer).
  Future<String> _prompt(String user) async {
    final history = [
      for (final m in messages)
        if (m.error == null && !m.streaming) Turn(user: m.promptUser, exact: m.exact)
    ];
    final budget = settings.contextLength - settings.maxTokens;
    final system = await _systemBlock();
    while (true) {
      final prompt = template.build(system, history, user, _thinking);
      if (history.isEmpty || await engine.countTokens(prompt) <= budget) return prompt;
      history.removeAt(0);
    }
  }

  static const _weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  static const _months = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
  ];

  /// "[Now: Friday 9 October 2026, 13:40, UTC+02:00]": appended to each user
  /// turn so "today" or "in two hours" mean something (the system prompt stays
  /// fixed, so its context is reused).
  static String nowLine(DateTime t) {
    final o = t.timeZoneOffset;
    final sign = o.isNegative ? '-' : '+';
    String two(int v) => v.abs().toString().padLeft(2, '0');
    return '[Now: ${_weekdays[t.weekday - 1]} ${t.day} ${_months[t.month - 1]} ${t.year}, '
        '${two(t.hour)}:${two(t.minute)}, UTC$sign${two(o.inHours)}:${two(o.inMinutes % 60)}]';
  }

  /// At most this many tool calls per reply.
  static const _maxToolCalls = 4;

  Future<void> send(String text) async {
    if (generating || text.trim().isEmpty || (!engine.loaded && _released == null && !loading)) return;
    generating = true;
    final promptUser = '$text\n\n${nowLine(DateTime.now())}';
    final message = ChatMessage(text, template.assistantPrefix(_thinking), promptUser: promptUser);
    messages.add(message); // shown at once, also while the model wakes up
    notifyListeners();
    // Unloaded while idle, or still loading: wait for the model, then for its context.
    final released = _released;
    if (released != null) {
      await load(released, keepChat: true);
    } else if (loading) {
      await _pendingLoad;
    }
    await _preparing;
    if (!engine.loaded) {
      message
        ..error = 'The model could not be loaded'
        ..streaming = false;
      generating = false;
      notifyListeners();
      return;
    }
    final prompt = await _prompt(promptUser);
    // The prompt without the reply's opening: each step appends the reply so far.
    final head = prompt.substring(0, prompt.length - message.prefix.length);
    final sampling = SamplingOptions(
        temperature: settings.temperature, topP: settings.topP, topK: settings.topK, maxTokens: settings.maxTokens);
    try {
      for (var calls = 0;; ++calls) {
        await _stream(head + message.exact, sampling, message);
        final call = calls < _maxToolCalls && message.stats?.cancelled != true ? toolDialect.parse(message.raw) : null;
        final tool = call == null ? null : toolbox.byName(call.name);
        if (call == null) break;
        final (result, summary) = tool == null || !toolbox.isOn(tool)
            ? ('{"error": "No tool named ${call.name} is available."}', 'Unknown tool')
            : await _runTool(tool, call);
        if (tool != null) message.steps.add(ToolStep(tool.label, summary));
        _log('Tool ${call.name}(${call.arguments}): $summary');
        // The call ends this segment; its result and the next reply follow.
        message
          ..done = message.exact + ToolDialect.response(result)
          ..prefix = template.assistantPrefix(_thinking)
          ..raw = '';
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

  Future<(String, String)> _runTool(AgentTool tool, ToolCall call) async {
    try {
      return await tool.run(call.arguments);
    } on Exception catch (e) {
      return ('{"error": "The tool failed: $e"}', 'Failed');
    }
  }

  /// Streams one generation into the message's current segment. Redraws are
  /// batched (~15 a second): the engine runs on all but one core, and
  /// rebuilding the reply for every token would take CPU time from it.
  Future<void> _stream(String prompt, SamplingOptions sampling, ChatMessage message) async {
    var lastPaint = DateTime.fromMillisecondsSinceEpoch(0);
    await for (final event in engine.generate(prompt, sampling)) {
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
  }

  Timer? _toolsSettle;

  /// Tools changed (turned on or off): the system block that lists them
  /// changes too. Prefilled once the switches settle, not once per switch.
  Future<void> toolsChanged() async {
    notifyListeners();
    _toolsSettle?.cancel();
    _toolsSettle = Timer(const Duration(seconds: 2), prepareSystemPrompt);
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
