// The engine on a background isolate.
//
// liyab_engine_create / generate / prefill block for seconds, so they run on
// one long-lived worker isolate that owns the engine; the UI isolate sends it
// commands and receives replies and streamed pieces. Cancelling and reading
// the counters are thread-safe in the C API, so the UI isolate calls those
// directly through the engine's address, without queuing behind a generation.
import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:isolate';

import 'package:ffi/ffi.dart';

import 'liyab_ffi.dart';

/// How to load a model (the engine's own defaults apply to anything not set here).
class LoadOptions {
  const LoadOptions({
    required this.modelPath,
    this.backend = LiyabBackend.cpu,
    this.contextLength = 4096,
    this.skinThresholdC = 50,
    this.memoryBudgetMb = 5500,
  });

  final String modelPath;
  final int backend;
  final int contextLength;
  final double skinThresholdC;
  final int memoryBudgetMb;
}

class SamplingOptions {
  const SamplingOptions({this.temperature = 0.7, this.topP = 0.9, this.topK = 40, this.maxTokens = 1024});

  final double temperature;
  final double topP;
  final int topK;
  final int maxTokens;
}

/// What the engine reported about a loaded model.
class LoadedModel {
  const LoadedModel({required this.description, required this.loadSeconds, required this.metadata});

  final String description;
  final double loadSeconds;

  /// The GGUF keys the app reads (missing ones absent).
  final Map<String, String> metadata;
}

/// One finished generation, from liyab_generation_stats.
class GenerationStats {
  const GenerationStats({
    required this.promptTokens,
    required this.generatedTokens,
    required this.tokensPerSecond,
    required this.ttftMs,
    required this.cachedPrefixTokens,
    required this.thermalReroutes,
    required this.cancelled,
  });

  final int promptTokens;
  final int generatedTokens;
  final double tokensPerSecond;
  final double ttftMs;
  final int cachedPrefixTokens;
  final int thermalReroutes;
  final bool cancelled;
}

sealed class GenerationEvent {}

class TextPiece extends GenerationEvent {
  TextPiece(this.text);
  final String text;
}

class GenerationDone extends GenerationEvent {
  GenerationDone(this.stats);
  final GenerationStats stats;
}

class EngineException implements Exception {
  EngineException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Cumulative engine counters (see liyab_engine_counters).
class EngineCounters {
  const EngineCounters(this.acceleratorBusyMs, this.storageBytesRead, this.tokensGenerated);
  final double acceleratorBusyMs;
  final int storageBytesRead;
  final int tokensGenerated;
}

const _metadataKeys = [
  'general.sampling.temp',
  'general.sampling.top_p',
  'general.sampling.top_k',
  'tokenizer.chat_template', // how the model calls tools
];

class EngineService {
  EngineService._(this._commands, this._replies);

  final SendPort _commands;
  final ReceivePort _replies;
  final Map<int, void Function(Object?)> _pending = {};
  final Map<int, StreamController<GenerationEvent>> _streams = {};
  int _nextId = 0;
  int _engine = 0; // address of the loaded liyab_engine, 0 when none

  bool get loaded => _engine != 0;

  static Future<EngineService> start() async {
    final replies = ReceivePort();
    final ready = Completer<SendPort>();
    late final EngineService service;
    replies.listen((message) {
      if (!ready.isCompleted) {
        ready.complete(message as SendPort);
        return;
      }
      service._onReply(message as List<Object?>);
    });
    await Isolate.spawn(_worker, replies.sendPort, debugName: 'liyab-engine');
    service = EngineService._(await ready.future, replies);
    return service;
  }

  void _onReply(List<Object?> m) {
    final kind = m[0] as String, id = m[1] as int;
    switch (kind) {
      case 'piece':
        _streams[id]?.add(TextPiece(m[2] as String));
      case 'done':
        final s = m[2] as List<Object?>;
        _streams.remove(id)
          ?..add(GenerationDone(GenerationStats(
            promptTokens: s[0] as int,
            generatedTokens: s[1] as int,
            tokensPerSecond: s[2] as double,
            ttftMs: s[3] as double,
            cachedPrefixTokens: s[4] as int,
            thermalReroutes: s[5] as int,
            cancelled: s[6] as bool,
          )))
          ..close();
      case 'error':
        final error = EngineException(m[2] as String);
        final stream = _streams.remove(id);
        if (stream != null) {
          stream
            ..addError(error)
            ..close();
        } else {
          _pending.remove(id)?.call(error);
        }
      default: // 'ok'
        _pending.remove(id)?.call(m.length > 2 ? m[2] : null);
    }
  }

  Future<T> _call<T>(List<Object?> command) {
    final id = _nextId++;
    final completer = Completer<T>();
    _pending[id] = (value) => value is EngineException ? completer.completeError(value) : completer.complete(value as T);
    _commands.send([command[0], id, ...command.skip(1)]);
    return completer.future;
  }

  /// Loads a model, replacing the current one.
  Future<LoadedModel> load(LoadOptions o) async {
    _engine = 0;
    final r = await _call<List<Object?>>(
        ['load', o.modelPath, o.backend, o.contextLength, o.skinThresholdC, o.memoryBudgetMb]);
    _engine = r[0] as int;
    return LoadedModel(
        description: r[1] as String,
        loadSeconds: r[2] as double,
        metadata: Map<String, String>.from(r[3] as Map));
  }

  Future<void> unload() async {
    _engine = 0;
    await _call<Object?>(['unload']);
  }

  /// Number of tokens `text` encodes to (special-token texts count as one).
  Future<int> countTokens(String text) => _call<int>(['tokenize', text]);

  /// Processes `text` into the context so a later prompt starting with it skips that work.
  Future<void> prefill(String text) => _call<Object?>(['prefill', text]);

  /// Saves the context (e.g. the prefilled system prompt) to `path`.
  Future<void> saveState(String path) => _call<Object?>(['saveState', path]);

  /// Restores a context saveState() wrote for this model; the tokens restored.
  Future<int> loadState(String path) => _call<int>(['loadState', path]);

  /// Streams the reply to `prompt` (already chat-formatted).
  Stream<GenerationEvent> generate(String prompt, SamplingOptions s) {
    final id = _nextId++;
    final controller = StreamController<GenerationEvent>();
    _streams[id] = controller;
    _commands.send(['generate', id, prompt, s.temperature, s.topP, s.topK, s.maxTokens]);
    return controller.stream;
  }

  /// Stops the running generation at the next token (thread-safe; no queueing).
  void cancel() {
    if (_engine != 0) LiyabLib.instance.cancel(Pointer<Void>.fromAddress(_engine));
  }

  /// Live counters, read directly (thread-safe in the C API); null without a model.
  EngineCounters? counters() {
    if (_engine == 0) return null;
    final out = calloc<LiyabEngineCounters>();
    try {
      if (LiyabLib.instance.getCounters(Pointer<Void>.fromAddress(_engine), out) != LiyabStatus.ok) return null;
      return EngineCounters(out.ref.acceleratorBusyMs, out.ref.storageBytesRead, out.ref.tokensGenerated);
    } finally {
      calloc.free(out);
    }
  }

  static String deviceDescription() =>
      LiyabLib.readText((buffer, size) => LiyabLib.instance.describeDevice(buffer, size));

  void dispose() => _replies.close();
}

// ---------------------------------------------------------------------------
// Worker isolate: owns the engine and runs one command at a time.

void _worker(SendPort replies) {
  final commands = ReceivePort();
  replies.send(commands.sendPort);
  final lib = LiyabLib.instance;
  var engine = Pointer<Void>.fromAddress(0);

  void destroy() {
    if (engine.address != 0) lib.destroy(engine);
    engine = Pointer<Void>.fromAddress(0);
  }

  commands.listen((message) {
    final m = message as List<Object?>;
    final kind = m[0] as String, id = m[1] as int;
    void fail([String? text]) => replies.send(['error', id, text ?? lib.errorMessage()]);
    try {
      switch (kind) {
        case 'load':
          destroy();
          using((arena) {
            final config = arena<LiyabEngineConfig>();
            lib.configDefault(config);
            config.ref
              ..modelPath = (m[2] as String).toNativeUtf8(allocator: arena)
              ..backend = m[3] as int
              ..contextLength = m[4] as int
              ..powerProfile = LiyabPowerProfile.performance // a chat wants full speed; the thermal guard still applies
              ..skinThresholdC = m[5] as double
              ..memoryBudgetMb = m[6] as int; // per-app OS caps (HyperOS: 6 GiB PSS) are invisible to the engine
            final out = arena<Pointer<Void>>();
            final watch = Stopwatch()..start();
            if (lib.create(config, out) != LiyabStatus.ok) return fail();
            engine = out.value;
            final description = LiyabLib.readText((b, s) => lib.describe(engine, b, s));
            final metadata = <String, String>{};
            for (final key in _metadataKeys) {
              final k = key.toNativeUtf8(allocator: arena);
              final value = LiyabLib.readText((b, s) => lib.metadata(engine, k, b, s));
              if (value.isNotEmpty) metadata[key] = value;
            }
            replies.send([
              'ok',
              id,
              [engine.address, description, watch.elapsedMicroseconds / 1e6, metadata]
            ]);
          });
        case 'unload':
          destroy();
          replies.send(['ok', id]);
        case 'tokenize':
          if (engine.address == 0) return fail('no model loaded');
          using((arena) {
            final n = lib.tokenize(engine, (m[2] as String).toNativeUtf8(allocator: arena), 0, nullptr, 0);
            n < 0 ? fail() : replies.send(['ok', id, n]);
          });
        case 'prefill':
          if (engine.address == 0) return fail('no model loaded');
          using((arena) {
            lib.prefill(engine, (m[2] as String).toNativeUtf8(allocator: arena), 1) == LiyabStatus.ok
                ? replies.send(['ok', id])
                : fail();
          });
        case 'saveState':
          if (engine.address == 0) return fail('no model loaded');
          using((arena) {
            lib.saveState(engine, (m[2] as String).toNativeUtf8(allocator: arena)) == LiyabStatus.ok
                ? replies.send(['ok', id])
                : fail();
          });
        case 'loadState':
          if (engine.address == 0) return fail('no model loaded');
          using((arena) {
            final n = arena<Int32>();
            lib.loadState(engine, (m[2] as String).toNativeUtf8(allocator: arena), n) == LiyabStatus.ok
                ? replies.send(['ok', id, n.value])
                : fail();
          });
        case 'generate':
          if (engine.address == 0) return fail('no model loaded');
          _generate(lib, engine, m, replies);
      }
    } catch (e) {
      fail('$e');
    }
  });
}

void _generate(LiyabLib lib, Pointer<Void> engine, List<Object?> m, SendPort replies) {
  final id = m[1] as int;
  // Called synchronously on this isolate's thread during liyab_engine_generate;
  // each piece is complete UTF-8.
  final callback = NativeCallable<LiyabTokenCallbackNative>.isolateLocal(
    (Pointer<Utf8> piece, int len, int token, Pointer<Void> user) {
      replies.send(['piece', id, utf8.decode(piece.cast<Uint8>().asTypedList(len), allowMalformed: true)]);
      return 1;
    },
    exceptionalReturn: 0,
  );
  try {
    using((arena) {
      final params = arena<LiyabSamplingParams>();
      lib.samplingDefault(params);
      params.ref
        ..temperature = m[3] as double
        ..topP = m[4] as double
        ..topK = m[5] as int
        ..maxTokens = m[6] as int;
      final stats = arena<LiyabGenerationStats>();
      final status = lib.generate(
          engine, (m[2] as String).toNativeUtf8(allocator: arena), params, callback.nativeFunction, nullptr, stats);
      if (status != LiyabStatus.ok) {
        replies.send(['error', id, lib.errorMessage()]);
        return;
      }
      final s = stats.ref;
      replies.send([
        'done',
        id,
        [
          s.promptTokens,
          s.generatedTokens,
          s.tokensPerSecond,
          s.ttftMs,
          s.cachedPrefixTokens,
          s.thermalReroutes,
          s.cancelled != 0
        ]
      ]);
    });
  } finally {
    callback.close();
  }
}
