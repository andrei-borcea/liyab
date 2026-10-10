// The loaded model and the conversation, shared by every screen. Engine calls
// go through EngineService, which runs them one at a time on its isolate.
import 'dart:async';
import 'dart:io';

import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart' show MethodChannel;
import 'package:flutter/widgets.dart' show AppLifecycleListener, AppLifecycleState;
import 'package:path_provider/path_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../agent/tool_format.dart';
import '../api/api_core.dart';
import '../api/local_api.dart';
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

  /// A finished message as saved with its conversation (stats are not kept).
  Map<String, Object?> toJson() => {
        'user': user,
        'promptUser': promptUser,
        'prefix': prefix,
        'raw': raw,
        'done': done,
        'steps': [for (final s in steps) [s.label, s.summary]],
        if (error != null) 'error': error,
      };

  factory ChatMessage.fromJson(Map<String, Object?> o) => ChatMessage(o['user'] as String, o['prefix'] as String,
      promptUser: o['promptUser'] as String?)
    ..raw = o['raw'] as String? ?? ''
    ..done = o['done'] as String? ?? ''
    ..steps.addAll([
      for (final s in (o['steps'] as List<Object?>? ?? const []).cast<List<Object?>>()) ToolStep('${s[0]}', '${s[1]}')
    ])
    ..error = o['error'] as String?
    ..streaming = false;
}

/// One line of the activity log.
class LogLine {
  LogLine(this.text, {this.engine = false}) : at = DateTime.now();
  final DateTime at;
  final String text;
  final bool engine; // from libliyab's log (level letter first), else the app
}

class AppState extends ChangeNotifier implements ApiBackend {
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
        _leftScreen();
      }
    });
    addListener(_updateWork); // the busy flags below change with a notification
  }

  // Leaving the screen. HyperOS stops a background app that holds several GB
  // within a minute (kill_bg_proc), and the process outlives the window anyway
  // (the notification listener keeps it), so the moment Liyab is hidden it
  // parks: the expert cache is emptied (2.4 GB on a 35B MoE; the model stays
  // loaded and answers at once, its first tokens a little slower) and the
  // conversation is saved, messages and context, so a stopped process loses
  // nothing. After device.releaseAfterMinutes the model is unloaded too; it
  // loads again, chat kept, once Liyab is back.
  //
  // Work in hand while hidden (a reply, a load, a model being moved, a request
  // of another app) would get four of the eight cores on HyperOS, and soon none:
  // its cgroup freezer stops hidden apps. On Android the work service
  // (WorkService.kt) keeps the process running at full speed until that work
  // and the park after it are done. It starts only with work in hand, so
  // leaving an idle Liyab costs nothing.
  late final AppLifecycleListener _lifecycle;
  bool _away = false; // off screen since the last park
  Timer? _parkRetry;
  Timer? _releaseTimer;
  String? _released; // the model unloaded while idle
  bool _parkPending = false; // hidden, and the park not done yet

  static const _work = MethodChannel('liyab/work');
  bool _working = false; // the work service is asked for

  /// Work that keeps the model loaded, and the work service up while Liyab is hidden.
  bool get _busy => generating || loading || moving != null || _apiGenerations > 0;

  void _updateWork() {
    if (!Platform.isAndroid) return;
    final want = _away && (_busy || (_working && _parkPending));
    if (want == _working) return;
    _working = want;
    final text = generating
        ? 'Finishing a reply'
        : loading
            ? 'Loading the model'
            : moving != null
                ? 'Moving $moving'
                : 'Answering another app';
    unawaited(_work.invokeMethod<bool>(want ? 'start' : 'stop', want ? text : null).then((started) {
      // Android allows it only just after Liyab was on screen (or with its battery use unrestricted).
      if (started == false) _log('Working hidden without the work service: Android did not allow it now');
    }, onError: (Object e) => _log('Work service: $e')));
  }

  void _leftScreen() {
    _parkPending = true;
    _updateWork();
    unawaited(_park());
    _armRelease();
  }

  /// While hidden, the model is unloaded device.releaseAfterMinutes after the last use: leaving the screen, or a
  /// request of another app through the local API (which loads the model again if it was released).
  void _armRelease() {
    _releaseTimer?.cancel();
    final minutes = device.releaseAfterMinutes;
    if (minutes > 0) _releaseTimer = Timer(Duration(minutes: minutes), _releaseIdle);
  }

  /// Generations of the local API queued or running: the model is not released under them.
  int _apiGenerations = 0;

  Future<void> _park() async {
    final waiting = _parkRetry != null;
    _parkRetry?.cancel();
    _parkRetry = null;
    if (_away && engine.loaded && (generating || loading)) {
      if (!waiting) _log('In the background while ${generating ? 'answering' : 'loading'}: parking once done');
      // Checked every second: the work service stays up until this park is done.
      _parkRetry = Timer(const Duration(seconds: 1), _park);
      return;
    }
    try {
      if (!_away || !engine.loaded) return;
      final watch = Stopwatch()..start();
      try {
        await engine.saveExpertProfile(await _expertProfileFile()); // what was hot, to warm up again
      } on EngineException catch (e) {
        _log('Expert profile not saved: $e');
      }
      final bytes = await engine.trimMemory();
      _cacheCold = bytes > 0;
      _log('Freed ${(bytes / (1 << 30)).toStringAsFixed(2)} GB of cached experts in ${watch.elapsedMilliseconds} ms');
      await _saveConversation(withContext: true);
      if (messages.isNotEmpty) _log('Conversation saved in ${watch.elapsedMilliseconds} ms');
    } finally {
      _parkPending = false;
      _updateWork();
    }
  }

  Future<void> _releaseIdle() async {
    final path = modelPath;
    if (path == null || _released != null) return;
    if (_busy) {
      _releaseTimer = Timer(const Duration(minutes: 1), _releaseIdle); // busy: try again later
      return;
    }
    _released = path; // set first: coming back during the unload reloads it (commands run in order)
    await _saveConversation(withContext: true);
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
    _parkRetry?.cancel();
    _releaseTimer?.cancel();
    super.dispose();
  }

  void _backOnScreen() {
    _parkRetry?.cancel();
    _parkRetry = null;
    _releaseTimer?.cancel();
    _away = false;
    _parkPending = false;
    _updateWork();
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
  @override
  String modelName = '';
  String description = '';
  @override
  ChatTemplate template = ChatTemplate.zephyr;
  @override
  bool thinkingSupported = false;
  ModelSettings settings = ModelSettings();

  /// The loaded model's context length (positions), as the engine resolved it.
  int contextLength = 4096;

  /// How the loaded model uses memory (expert streaming); zero for models that fit.
  MemoryPlan memory = const MemoryPlan(0, 0, 0, 0, 0);
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
    // keepChat: the last conversation comes back if the OS stopped Liyab.
    if (last != null && File(last).existsSync()) unawaited(state.load(last, keepChat: true));
    if (state.device.apiEnabled) unawaited(state._startApi());
    return state;
  }

  // The local API (lib/api/local_api.dart): this state is its backend.
  LocalApi? _api;

  /// Whether the local API is listening, and why not when it should be.
  bool get apiRunning => _api != null;
  String apiError = '';

  Future<void> setApiEnabled(bool on) async {
    device.apiEnabled = on;
    on ? await _startApi() : await _stopApi();
  }

  /// Replaces the token: clients holding the old one are refused from now on.
  Future<void> newApiToken() async {
    device.apiToken = LocalApi.newToken();
    if (_api != null) await _startApi();
  }

  Future<void> _startApi() async {
    await _stopApi();
    final token = device.apiToken ?? (device.apiToken = LocalApi.newToken());
    final api = LocalApi(this, token);
    try {
      await api.start();
      _api = api;
      apiError = '';
      _log('Local API on 127.0.0.1:${LocalApi.httpPort} (HTTP) and :${LocalApi.grpcPort} (gRPC)');
    } on SocketException catch (e) {
      apiError = 'Port in use (${e.osError?.message ?? e.message})';
      _log('Local API not started: $apiError');
    }
    notifyListeners();
  }

  Future<void> _stopApi() async {
    final api = _api;
    _api = null;
    await api?.stop();
    notifyListeners();
  }

  @override
  SamplingOptions get defaults => SamplingOptions(
      temperature: settings.temperature, topP: settings.topP, topK: settings.topK, maxTokens: settings.maxTokens);

  @override
  Future<void> ensureLoaded() async {
    final released = _released;
    if (released != null) {
      await load(released, keepChat: true);
    } else if (loading) {
      await _pendingLoad;
    }
    // Hidden, the model would otherwise stay loaded (several GB in the background, which HyperOS stops apps for).
    if (_away) _armRelease();
    if (!engine.loaded) throw const ApiError(503, 'no model is loaded in Liyab');
  }

  @override
  Future<int> countTokens(String text) => engine.countTokens(text);

  @override
  Generation generate(String prompt, SamplingOptions sampling) {
    _chatContext = false; // queued after anything the chat queued: the engine's context is this request's once it ran
    final generation = engine.generate(prompt, sampling);
    ++_apiGenerations;
    _updateWork();
    var ended = false;
    void end() {
      if (ended) return;
      ended = true;
      --_apiGenerations;
      _updateWork();
      if (_away) _armRelease(); // counted from the last request
    }

    // The events pass through unchanged; their end (or the client going away) ends the request.
    final events = StreamController<GenerationEvent>();
    final source = generation.events.listen(events.add, onError: events.addError, onDone: () {
      end();
      events.close();
    });
    events.onCancel = () {
      end();
      return source.cancel();
    };
    return Generation(events.stream, generation.stop);
  }

  /// Whether the engine's context is this chat's once the commands queued so far have run (another app's request
  /// replaces it; the engine keeps the chat's aside and restores it on the chat's next prompt). Parking saves the
  /// context as the conversation's only then: another one saved in its place would be restored with the chat
  /// after a restart, and the whole conversation processed again.
  bool _chatContext = true;

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
    if (!keepChat) {
      messages.clear();
      _historyStart = 0;
    }
    notifyListeners();
    try {
      final gpu = device.useGpu;
      _prepared = null;
      final experimental = device.experimental;
      final model = await engine.load(LoadOptions(
        modelPath: path,
        backend: gpu ? LiyabBackend.vulkan : LiyabBackend.cpu,
        contextLength: ModelSettings.contextFor(prefs, name),
        skinThresholdC: device.thermalLimitC,
        memoryBudgetMb: device.memoryBudgetMb,
        powerProfile: device.powerProfile,
        // A context another conversation replaces (other apps through the local API, a chat left for a new one)
        // goes to storage, not RAM: it comes back in a fraction of a second when its conversation goes on, instead
        // of being processed again (minutes on a large MoE), and the memory budget stays with the model.
        prefixCacheDir: '${(await getApplicationSupportDirectory()).path}/prefix-cache',
        // Bounded, and the engine removes the files as contexts come back, and all of them when the model unloads:
        // room for the largest context of a 35B MoE (~320 MB at 24k tokens) and a smaller one.
        prefixCacheDiskMb: 512,
        experimental: {
          ...experimental.toJson(),
          if (experimental.kvDedup) 'kvDedupDir': (await _kvDedupDir()).path,
        },
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
      memory = model.memory;
      contextLength = model.contextLength;
      status = '$name · ${gpu ? 'GPU' : 'CPU'} · ready in ${model.loadSeconds.toStringAsFixed(1)} s';
      if (model.memory.tight) {
        status = '$name · needs about ${(model.memory.recommendedBytes / (1 << 30)).toStringAsFixed(1)} GB of memory to '
            'answer quickly (Settings, Memory for the model)';
        _log('Memory too tight for $name: ${(model.memory.expertCacheBytes / (1 << 30)).toStringAsFixed(2)} GB of expert '
            'cache; about ${(model.memory.recommendedBytes / (1 << 30)).toStringAsFixed(1)} GB recommended');
      }
      if (model.memory.requantBits > 0) {
        _log('Resident weights converted to ${model.memory.requantBits} bits to make room (memory was tight)');
      }
      _log('Loaded $name in ${model.loadSeconds.toStringAsFixed(2)} s (${template.label}, tools: ${toolDialect.name})');
      if (experimental.active.isNotEmpty) _log('Experimental: ${experimental.active.join(', ')}');
      for (final line in description.split('\n')) {
        _log('  $line');
      }
      _preparing = keepChat ? _restoreConversation() : prepareSystemPrompt();
      unawaited(_loadExpertProfile());
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

  /// After a load: the hot list a previous process saved, warmed at once (a fresh process has an empty cache).
  Future<void> _loadExpertProfile() async {
    final file = File(await _expertProfileFile());
    if (!file.existsSync()) return;
    try {
      await engine.loadExpertProfile(file.path);
      _cacheCold = true;
      _warmIfCold();
    } on EngineException catch (e) {
      _log('Expert profile not usable: $e');
    }
  }

  /// Whether the expert cache was emptied while parked (its hot list is kept to warm it up again).
  bool _cacheCold = false;

  /// Refills the expert cache in the background with what was hot before it
  /// was emptied, as soon as the user starts writing: one pass of large reads
  /// while they type, instead of a wait per expert in the first answer.
  void _warmIfCold() {
    if (!_cacheCold || !engine.loaded) return;
    _cacheCold = false;
    unawaited(engine.warmMemory().then((n) {
      if (n > 0) _log('Warming the expert cache: $n experts');
    }, onError: (Object e) => _log('Expert cache not warmed: $e')));
  }

  /// The expert cache's hot list of this model.
  Future<String> _expertProfileFile() async {
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/states');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    return '${dir.path}/$modelName.experts';
  }

  /// Where the experimental persistent prefix KV cache lives (the engine needs an existing directory).
  Future<Directory> _kvDedupDir() async =>
      Directory('${(await getApplicationSupportDirectory()).path}/kv-dedup').create(recursive: true);

  /// The context being prepared after a load (system prompt or saved conversation).
  Future<void>? _preparing;

  /// Where this model's conversation is kept: the engine's context and the messages.
  Future<(String, String)> _conversationFiles() async {
    final dir = Directory('${(await getApplicationSupportDirectory()).path}/states');
    if (!dir.existsSync()) dir.createSync(recursive: true);
    return ('${dir.path}/$modelName.conversation.state', '${dir.path}/$modelName.conversation.json');
  }

  /// Saves the conversation (removes it when there is none): the messages,
  /// and with `withContext` the engine's context (KV pages and recurrent
  /// states, ~200 MB on a 35B MoE: written when parking, not after every
  /// reply), when it is the chat's (_chatContext). A context older than the
  /// messages is still a prefix of the next prompt, so it only saves less
  /// work. Between generations only.
  Future<void> _saveConversation({bool withContext = false}) async {
    final (state, chat) = await _conversationFiles();
    if (messages.isEmpty) return _discardConversation();
    await File(chat).writeAsString(
        jsonEncode({'start': _historyStart, 'messages': [for (final m in messages) m.toJson()]}));
    if (!withContext || !_chatContext) return; // checked as the save is queued: nothing can come in between
    try {
      await engine.saveState(state);
    } on EngineException catch (e) {
      _log('Conversation context not saved: $e');
    }
  }

  Future<void> _discardConversation() async {
    final (state, chat) = await _conversationFiles();
    for (final path in [state, chat]) {
      if (File(path).existsSync()) File(path).deleteSync();
    }
  }

  /// When a load keeps the chat (back from an idle unload, or a restart after
  /// the OS stopped Liyab): the saved messages if none are shown, and the
  /// conversation's context (system prompt included), else the system prompt alone.
  Future<void> _restoreConversation() async {
    final (state, chat) = await _conversationFiles();
    if (messages.isEmpty && File(chat).existsSync()) {
      try {
        final saved = jsonDecode(await File(chat).readAsString());
        // Earlier saves were the bare list of messages.
        final list = saved is Map ? saved['messages'] as List<Object?> : saved as List<Object?>;
        messages.addAll([for (final o in list) ChatMessage.fromJson(o as Map<String, Object?>)]);
        _historyStart = saved is Map ? (saved['start'] as num? ?? 0).toInt() : 0;
        notifyListeners();
      } on Object catch (e) {
        _log('Saved conversation not readable ($e)');
      }
    }
    if (messages.isNotEmpty && File(state).existsSync()) {
      final watch = Stopwatch()..start();
      try {
        _chatContext = true;
        final n = await engine.loadState(state);
        _prepared = await _systemBlock(); // the conversation starts with it
        _log('Conversation restored (${messages.length} messages, $n tokens) in ${watch.elapsedMilliseconds} ms');
        return;
      } on EngineException catch (e) {
        _log('Saved context not usable ($e)');
      }
    }
    await prepareSystemPrompt();
  }

  /// Where the processed system block of this model is kept: one file per
  /// model, named with the block's hash (a changed prompt or tool list is a
  /// different file; the older one is deleted).
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
        _chatContext = true;
        final n = await engine.loadState(file);
        _log('System prompt restored ($n tokens) in ${watch.elapsedMilliseconds} ms');
        return;
      } on EngineException catch (e) {
        _log('Saved system prompt not usable ($e); processing it again');
        File(file).deleteSync();
      }
    }
    try {
      // In the background: a message sent meanwhile goes first, and its
      // prompt (which starts with this block) continues the work done.
      _chatContext = true;
      await engine.prefill(block, background: true);
      _log('System prompt prepared in ${(watch.elapsedMilliseconds / 1000).toStringAsFixed(1)} s');
      // Saved only while the context is still the chat's: a request of another app queued meanwhile has replaced it.
      if (!_chatContext) return;
      await engine.saveState(file);
      // Older states of this model's system prompt go; its conversation and expert profile stay.
      final older = RegExp('^${RegExp.escape(modelName)}\\.[0-9a-f]{16}\\.state\$');
      for (final old in dir.listSync().whereType<File>()) {
        if (old.path != file && older.hasMatch(old.uri.pathSegments.last)) old.deleteSync();
      }
    } on EngineException catch (e) {
      _prepared = null; // prepared again (and saved) when the engine is free
      if (!_cancelled(e)) _log('System prompt not prepared: $e');
    }
  }

  /// A background prefill the scheduler stopped for a request of the user.
  static bool _cancelled(EngineException e) => e.message.contains('cancel');

  /// The system block: the system prompt, and the enabled tools when the model can call them.
  Future<String> _systemBlock() async => toolDialect == ToolDialect.none
      ? template.system(settings.systemPrompt)
      : toolDialect.system(settings.systemPrompt, await toolbox.enabled());

  Thinking get _thinking => !thinkingSupported ? Thinking.none : (settings.thinking ? Thinking.on : Thinking.off);

  /// Messages before this index are no longer replayed to the model (they stay on screen).
  int _historyStart = 0;
  int get historyStart => _historyStart;

  /// The prompt for `user` with the history from _historyStart. When it no
  /// longer fits the context, the history is compacted in one step: the
  /// oldest turns go until it fills at most half of the room, counted with
  /// the model's own tokenizer. Dropping one turn per message instead would
  /// change the start of every prompt, and the engine would process the
  /// whole conversation again for each message (minutes on a large MoE);
  /// this way the following messages extend the same prompt, and the cost
  /// comes once every many messages.
  Future<String> _prompt(String user) async {
    final budget = contextLength - settings.maxTokens;
    final system = await _systemBlock();
    List<Turn> history() => [
          for (final m in messages.skip(_historyStart.clamp(0, messages.length)))
            if (m.error == null && !m.streaming) Turn(user: m.promptUser, exact: m.exact)
        ];
    var turns = history();
    var prompt = template.build(system, turns, user, _thinking);
    if (turns.isEmpty || await engine.countTokens(prompt) <= budget) return prompt;
    final before = await engine.countTokens(prompt);
    final dropped = _historyStart;
    final base = await engine.countTokens(template.build(system, const [], user, _thinking));
    while (turns.isNotEmpty) {
      final rest = await engine.countTokens(prompt) - base;
      if (rest <= (budget - base) ~/ 2) break;
      ++_historyStart;
      turns = history();
      prompt = template.build(system, turns, user, _thinking);
    }
    _log('Conversation compacted: ${_historyStart - dropped} older messages left out of the model\'s context '
        '($before -> ${await engine.countTokens(prompt)} prompt tokens)');
    return prompt;
  }

  // Typing ahead: while the user writes, the stable part of the draft (whole
  // words, cut where its tokens are a prefix of the text so far) is processed
  // in the background, so sending only costs the last words and the time line.
  // On a MoE whose experts stream from storage a prompt token costs about as
  // much as a generated one, so this hides most of the wait before an answer.
  // On hybrid models the engine reuses a context only up to its snapshot at
  // the end of a prefill, hence the exact token-prefix check.
  Timer? _draftTimer;
  String _drafted = ''; // the prompt text last processed for the draft
  bool _drafting = false;

  /// Called on every edit of the composer's text.
  void draftChanged(String text) {
    _draftTimer?.cancel();
    if (text.trim().isEmpty) return;
    _warmIfCold();
    _draftTimer = Timer(const Duration(milliseconds: 600), () => unawaited(_prefillDraft(text)));
  }

  Future<void> _prefillDraft(String text) async {
    if (_drafting || generating || loading || _away || !engine.loaded) return;
    _drafting = true;
    try {
      await _preparing;
      const marker = '\u0000';
      final template = await _prompt(marker);
      final head = template.substring(0, template.indexOf(marker));
      final whole = await engine.tokenIds(head + text, background: true);
      var cut = text.length;
      for (var tries = 0; tries < 3; ++tries) {
        // Whole words; before the first one, the history alone.
        cut = cut <= 0 ? 0 : text.lastIndexOf(RegExp(r'\s'), cut - 1);
        if (cut < 0) cut = 0;
        final stable = head + text.substring(0, cut);
        if (_drafted.startsWith(stable)) return; // nothing new to process
        final ids = await engine.tokenIds(stable, background: true);
        if (ids.length < whole.length && _isPrefix(ids, whole)) {
          final watch = Stopwatch()..start();
          _chatContext = true;
          await engine.prefill(stable, background: true);
          _log('Draft prepared (${ids.length} tokens) in ${(watch.elapsedMilliseconds / 1000).toStringAsFixed(2)} s');
          _drafted = stable;
          return;
        }
        if (cut == 0) return;
      }
    } on EngineException catch (e) {
      if (!_cancelled(e)) _log('Draft not prepared: $e');
    } finally {
      _drafting = false;
    }
  }

  static bool _isPrefix(List<int> a, List<int> b) {
    for (var i = 0; i < a.length; ++i) {
      if (a[i] != b[i]) return false;
    }
    return true;
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
    _warmIfCold();
    _draftTimer?.cancel();
    _drafted = '';
    // The time line ends the turn: placed first (where typing ahead could
    // process it too), Qwen3.6 copied it at the start of its replies.
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
    engine.preempt(); // a system prompt still being prepared yields to the message
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
    final tools = toolDialect == ToolDialect.none ? const <String>[] : [for (final t in await toolbox.enabled()) t.name];
    try {
      for (var calls = 0;; ++calls) {
        // Tool calls are structured output: their format and names are forced, not sampled.
        final force = tools.isEmpty || calls >= _maxToolCalls
            ? null
            : toolDialect.forcer(tools, thinking: message.prefix.endsWith('<think>\n'));
        await _stream(head + message.exact, sampling, message, force);
        final call = calls < _maxToolCalls && message.stats?.cancelled != true ? toolDialect.parse(message.raw) : null;
        final tool = call == null ? null : toolbox.byName(call.name);
        if (call == null) break;
        final (result, summary) = tool == null || !toolbox.isOn(tool)
            ? ('No tool named ${call.name} is available.', 'Unknown tool')
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
      // The OS may stop Liyab at any time in the background: the messages are on disk after every reply.
      unawaited(_saveConversation());
    }
  }

  Future<(String, String)> _runTool(AgentTool tool, ToolCall call) async {
    try {
      return await tool.run(call.arguments);
    } on Exception catch (e) {
      return ('The tool failed: $e', 'Failed');
    }
  }

  /// Streams one generation into the message's current segment. Redraws are
  /// batched (~15 a second): the engine runs on all but one core, and
  /// rebuilding the reply for every token would take CPU time from it.
  Future<void> _stream(String prompt, SamplingOptions sampling, ChatMessage message,
      String Function(String)? force) async {
    var lastPaint = DateTime.fromMillisecondsSinceEpoch(0);
    _chatContext = true;
    await for (final event in engine.generate(prompt, sampling, force: force).events) {
      switch (event) {
        case TextPiece(:final text):
          message.raw += text;
          final now = DateTime.now();
          if (now.difference(lastPaint) < const Duration(milliseconds: 66)) continue;
          lastPaint = now;
        case GenerationDone(:final stats):
          message.stats = stats;
          // One line per generation (a reply with tool calls has several), so the time of each step shows.
          final fresh = stats.promptTokens - stats.cachedPrefixTokens;
          _log('Step ${message.steps.length + 1}: $fresh new prompt tokens (${stats.cachedPrefixTokens} reused), '
              'first token ${(stats.ttftMs / 1000).toStringAsFixed(2)} s, ${stats.generatedTokens} tokens at '
              '${stats.tokensPerSecond.toStringAsFixed(1)} tok/s${_thinking == Thinking.on ? ', thinking' : ''}'
              '${stats.thermalReroutes > 0 ? ', ${stats.thermalReroutes} throttled steps' : ''}${_forced(stats)}');
          if (stats.generatedTokens >= 16) {
            final phases = stats.phasesMs.entries.where((e) => e.value >= 0.5).map((e) => '${e.key} ${e.value.toStringAsFixed(1)}');
            final total = 1000 / (stats.tokensPerSecond > 0 ? stats.tokensPerSecond : 1);
            _log('  decode ms/token: ${phases.join(' | ')} | total ${total.toStringAsFixed(0)}'
                '${stats.expertHitRate > 0 ? '; experts ${(stats.expertHitRate * 100).round()}% cached, '
                    '${(stats.expertStallMs / stats.generatedTokens).toStringAsFixed(1)} ms/token waiting' : ''}');
          }
      }
      notifyListeners();
    }
  }

  /// The forced tokens of a step, what their batched passes took, and what
  /// sampling them one by one would have taken at the step's own decode
  /// speed: the time structured output saved (or cost) in that step.
  static String _forced(GenerationStats stats) {
    if (stats.forcedTokens == 0) return '';
    final passes = ', ${stats.forcedTokens} forced in ${(stats.forcedMs / 1000).toStringAsFixed(2)} s';
    if (stats.generatedTokens == 0 || stats.tokensPerSecond <= 0) return passes;
    final decodeMs = stats.generatedTokens * 1000 / stats.tokensPerSecond; // forced passes included
    final perToken = (decodeMs - stats.forcedMs) / stats.generatedTokens;
    return '$passes (sampled: ~${(stats.forcedTokens * perToken / 1000).toStringAsFixed(2)} s)';
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

  /// Saves the power profile and applies it to the loaded model at once.
  void setPowerProfile(int profile) {
    device.powerProfile = profile;
    engine.setPowerProfile(profile);
    notifyListeners();
  }

  /// Starts a new conversation (the engine keeps the system prompt's context).
  void newChat() {
    if (generating) return;
    messages.clear();
    _drafted = '';
    _historyStart = 0;
    unawaited(_discardConversation());
    // A hybrid model rewinds only to its state snapshots, and the old
    // conversation's context has none at the system prompt's end once it was
    // restored from a file: the system prompt comes back from its own saved
    // state instead, so the first message costs only its own tokens.
    _prepared = null;
    _preparing = prepareSystemPrompt();
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
