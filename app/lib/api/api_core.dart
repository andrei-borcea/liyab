// The local API's protocol-neutral core: what a request asks for (ApiRequest),
// what a reply streams (ApiEvent), and the one place that turns a request into
// a prompt in the loaded model's own chat format, runs it, separates the
// reasoning from the answer and applies stop sequences. The OpenAI, Anthropic
// and gRPC front ends (openai_api.dart, anthropic_api.dart, grpc_api.dart)
// only translate their wire formats to and from these types.
import 'dart:async';

import '../chat/chat_template.dart';
import '../engine/engine_service.dart';

/// One message of a conversation; `role` is system, user or assistant.
class ApiMessage {
  const ApiMessage(this.role, this.content);
  final String role;
  final String content;
}

class ApiRequest {
  const ApiRequest({
    this.messages = const [],
    this.prompt,
    this.temperature,
    this.topP,
    this.topK,
    this.maxTokens,
    this.stop = const [],
    this.thinking = false,
  });

  /// A conversation, formatted with the model's chat template. A last
  /// assistant message is continued (prefilled reply), not closed.
  final List<ApiMessage> messages;

  /// Raw text completion instead of `messages`: the model continues `prompt` as is.
  final String? prompt;

  /// Sampling; null keeps the loaded model's settings.
  final double? temperature;
  final double? topP;
  final int? topK;
  final int? maxTokens;

  /// Text that ends the reply where it first appears (not included).
  final List<String> stop;

  /// Reason before answering, on models with a thinking mode (ignored otherwise).
  final bool thinking;
}

/// A request the API refuses, with the HTTP status that says why.
class ApiError implements Exception {
  const ApiError(this.status, this.message);
  final int status;
  final String message;

  @override
  String toString() => message;
}

sealed class ApiEvent {}

/// First event: the prompt was accepted and has this many tokens.
class ApiStart extends ApiEvent {
  ApiStart(this.promptTokens);
  final int promptTokens;
}

/// Part of the answer.
class ApiText extends ApiEvent {
  ApiText(this.text);
  final String text;
}

/// Part of the reasoning (thinking mode), before the answer.
class ApiReasoning extends ApiEvent {
  ApiReasoning(this.text);
  final String text;
}

enum FinishReason { stop, length, stopSequence }

/// Last event.
class ApiDone extends ApiEvent {
  ApiDone(this.reason, this.stats, {this.stopSequence});
  final FinishReason reason;
  final GenerationStats stats;
  final String? stopSequence; // with FinishReason.stopSequence

  /// Every token of the reply, sampled or forced.
  int get completionTokens => stats.generatedTokens + stats.forcedTokens;
}

/// What the core needs from the app: the loaded model and the engine.
abstract class ApiBackend {
  /// The loaded model's name ('' when none).
  String get modelName;
  ChatTemplate get template;
  bool get thinkingSupported;

  /// The loaded model's sampling settings, for what a request leaves out.
  SamplingOptions get defaults;

  /// Loads the model again if it was released while idle; throws ApiError when no model is chosen.
  Future<void> ensureLoaded();
  Future<int> countTokens(String text);
  Generation generate(String prompt, SamplingOptions sampling);
}

class ApiCore {
  ApiCore(this.backend);
  final ApiBackend backend;

  /// The prompt for `r` and whether the reply starts inside a reasoning block.
  (String, bool) prompt(ApiRequest r) {
    final raw = r.prompt;
    if (raw != null) return (raw, false);
    if (r.messages.isEmpty) throw const ApiError(400, 'messages must not be empty');
    final t = backend.template;
    final continued = r.messages.last.role == 'assistant';
    final thinking = !backend.thinkingSupported
        ? Thinking.none
        : (r.thinking && !continued ? Thinking.on : Thinking.off);
    final p = StringBuffer();
    for (final (i, m) in r.messages.indexed) {
      if (!const {'system', 'user', 'assistant'}.contains(m.role)) throw ApiError(400, 'unsupported role: ${m.role}');
      if (continued && i == r.messages.length - 1) {
        p.write(t.reply(thinking) + m.content);
      } else {
        p.write(t.message(m.role, m.content));
      }
    }
    if (!continued) p.write(t.reply(thinking));
    return (p.toString(), thinking == Thinking.on);
  }

  Future<int> countTokens(ApiRequest r) async {
    await backend.ensureLoaded();
    return backend.countTokens(prompt(r).$1);
  }

  /// Runs `r`: ApiStart, then reasoning and answer pieces, then ApiDone.
  /// Cancelling the subscription stops the generation.
  Stream<ApiEvent> run(ApiRequest r) async* {
    await backend.ensureLoaded();
    final (text, reasoning) = prompt(r);
    yield ApiStart(await backend.countTokens(text));
    final d = backend.defaults;
    final sampling = SamplingOptions(
      temperature: r.temperature ?? d.temperature,
      topP: r.topP ?? d.topP,
      topK: r.topK ?? d.topK,
      maxTokens: (r.maxTokens ?? d.maxTokens).clamp(1, 1 << 20),
    );
    // A chat reply's leading blank lines are the template's; a raw completion's text is kept as generated.
    final out = ReplyShaper(reasoning: reasoning, stop: r.stop, trimStart: r.prompt == null);
    final generation = backend.generate(text, sampling);
    var stopping = false;
    try {
      // Leaving this loop early (the client went away) cancels the subscription, which stops the generation.
      await for (final event in generation.events) {
        switch (event) {
          case TextPiece(:final text):
            for (final e in out.feed(text)) {
              yield e;
            }
            if (out.stopped != null && !stopping) {
              stopping = true;
              generation.stop(); // it still ends with GenerationDone, which has the counts
            }
          case GenerationDone(:final stats):
            for (final e in out.flush()) {
              yield e;
            }
            final reason = out.stopped != null
                ? FinishReason.stopSequence
                : (stats.generatedTokens >= sampling.maxTokens ? FinishReason.length : FinishReason.stop);
            yield ApiDone(reason, stats, stopSequence: out.stopped);
        }
      }
    } on EngineException catch (e) {
      throw ApiError(400, e.message);
    }
  }
}

/// Splits a reply into reasoning and answer and cuts it at the first stop
/// sequence. Text that may be the start of `</think>` or of a stop sequence
/// is held back until the next piece tells.
class ReplyShaper {
  ReplyShaper({required this._reasoning, this.stop = const [], bool trimStart = true}) : _answerStarted = !trimStart;

  final List<String> stop;
  bool _reasoning;
  bool _answerStarted;
  String _pending = '';

  /// The stop sequence that ended the reply, once one did.
  String? stopped;

  static const _endThink = '</think>';

  List<ApiEvent> feed(String piece) {
    if (stopped != null) return const [];
    _pending += piece;
    final events = <ApiEvent>[];
    if (_reasoning) {
      final end = _pending.indexOf(_endThink);
      if (end < 0) {
        final keep = _heldBack(_pending, const [_endThink]);
        final ready = _pending.substring(0, _pending.length - keep);
        if (ready.isNotEmpty) events.add(ApiReasoning(ready));
        _pending = _pending.substring(ready.length);
        return events;
      }
      final thought = _pending.substring(0, end).trimRight();
      if (thought.isNotEmpty) events.add(ApiReasoning(thought));
      _reasoning = false;
      _pending = _pending.substring(end + _endThink.length);
    }
    if (!_answerStarted) {
      _pending = _pending.trimLeft();
      if (_pending.isEmpty) return events;
      _answerStarted = true;
    }
    for (final s in stop) {
      final at = s.isEmpty ? -1 : _pending.indexOf(s);
      if (at >= 0) {
        if (at > 0) events.add(ApiText(_pending.substring(0, at)));
        stopped = s;
        _pending = '';
        return events;
      }
    }
    final keep = _heldBack(_pending, stop);
    final ready = _pending.substring(0, _pending.length - keep);
    if (ready.isNotEmpty) events.add(ApiText(ready));
    _pending = _pending.substring(ready.length);
    return events;
  }

  /// What was held back, at the end of the reply.
  List<ApiEvent> flush() {
    final rest = _pending;
    _pending = '';
    if (stopped != null || rest.isEmpty) return const [];
    return [_reasoning ? ApiReasoning(rest) : ApiText(rest)];
  }

  /// The length of the longest end of `text` that starts one of `marks`.
  static int _heldBack(String text, List<String> marks) {
    var keep = 0;
    for (final m in marks) {
      for (var n = m.length - 1; n > keep; --n) {
        if (n <= text.length && text.endsWith(m.substring(0, n))) {
          keep = n;
          break;
        }
      }
    }
    return keep;
  }
}
