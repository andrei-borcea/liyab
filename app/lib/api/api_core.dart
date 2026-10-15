// The local API's protocol-neutral core: what a request asks for (ApiRequest),
// what a reply streams (ApiEvent), and the one place that turns a request into
// a prompt in the loaded model's own chat format, runs it, separates the
// reasoning from the answer and the tool calls from the text, and applies stop
// sequences. Tool calls and response formats are constrained decoding: the
// engine only lets the model write a call that parses, with arguments valid
// against the tool's schema (tool_calls.dart), or an answer valid against the
// requested schema. The OpenAI, Anthropic and gRPC front ends (openai_api.dart,
// anthropic_api.dart, grpc_api.dart) only translate their wire formats to and
// from these types.
import 'dart:async';
import 'dart:convert';

import '../agent/tool_format.dart';
import '../chat/chat_template.dart';
import '../engine/engine_service.dart';
import 'tool_calls.dart';

/// A tool call an assistant turn made (in a conversation's history).
class ApiCall {
  const ApiCall(this.name, this.arguments);
  final String name;
  final Map<String, Object?> arguments;
}

/// One message of a conversation; `role` is system, user, assistant or tool
/// (a call's result, answered in the model's own tool-response format).
class ApiMessage {
  const ApiMessage(this.role, this.content, {this.calls = const []});
  final String role;
  final String content;

  /// An assistant turn's tool calls, after its text.
  final List<ApiCall> calls;
}

/// Whether and which tools a reply may call.
enum ToolChoice {
  /// The model decides.
  auto,

  /// No call: the tools are not offered.
  none,

  /// At least one call (the first one right away).
  required,
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
    this.tools = const [],
    this.toolChoice = ToolChoice.auto,
    this.toolName,
    this.parallelToolCalls = true,
    this.responseSchema,
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

  /// Tools the model may call; `toolName` set: it must call that one.
  final List<ToolSpec> tools;
  final ToolChoice toolChoice;
  final String? toolName;

  /// false: the reply ends after its first call.
  final bool parallelToolCalls;

  /// The answer is a JSON value valid against this JSON Schema (response_format).
  final Map<String, Object?>? responseSchema;
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

/// A tool call's name, once the reply wrote it (calls are numbered from 0).
class ApiToolCallStart extends ApiEvent {
  ApiToolCallStart(this.index, this.name);
  final int index;
  final String name;
}

/// More of a tool call's arguments, as JSON text: a call's deltas put
/// together are its arguments object.
class ApiToolCallArguments extends ApiEvent {
  ApiToolCallArguments(this.index, this.delta);
  final int index;
  final String delta;
}

enum FinishReason { stop, length, stopSequence, toolCalls }

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

  /// How the loaded model calls tools (none: it cannot, as far as Liyab knows).
  ToolDialect get toolDialect;

  /// Why grammar text does not compile (null when it does): see EngineService.grammarError.
  String? grammarError(String grammar);

  /// Loads the model again if it was released while idle; throws ApiError when no model is chosen.
  Future<void> ensureLoaded();
  Future<int> countTokens(String text);
  /// See EngineService.generate for `grammar` and `structure` (constrained decoding).
  Generation generate(String prompt, SamplingOptions sampling,
      {String? grammar, Continuation Function(String generated)? structure});
}

class ApiCore {
  ApiCore(this.backend);
  final ApiBackend backend;

  /// The tools `r` offers the model (none with tool_choice none).
  List<ToolSpec> _offered(ApiRequest r) {
    if (r.toolChoice == ToolChoice.none) return const [];
    final name = r.toolName;
    if (name != null && !r.tools.any((t) => t.name == name)) throw ApiError(400, 'tool_choice names an unknown tool: $name');
    return r.tools;
  }

  /// The tool-call format of the loaded model, for a request that needs one.
  ToolDialect _dialect(ApiRequest r) {
    final usesTools = r.tools.isNotEmpty || r.messages.any((m) => m.role == 'tool' || m.calls.isNotEmpty);
    if (!usesTools) return ToolDialect.none;
    final d = backend.toolDialect;
    if (d == ToolDialect.none) {
      throw const ApiError(400, 'the loaded model has no tool-call format Liyab knows '
          '(ChatML models with Hermes JSON or Qwen XML calls, such as Qwen2.5, Qwen3, Qwen3.5 and Qwen3.6)');
    }
    return d;
  }

  /// The prompt for `r` and whether the reply starts inside a reasoning block.
  (String, bool) prompt(ApiRequest r) {
    final raw = r.prompt;
    if (raw != null) return (raw, false);
    if (r.messages.isEmpty) throw const ApiError(400, 'messages must not be empty');
    final t = backend.template;
    final dialect = _dialect(r);
    final tools = _offered(r);
    final continued = r.messages.last.role == 'assistant';
    final thinking = !backend.thinkingSupported
        ? Thinking.none
        : (r.thinking && !continued ? Thinking.on : Thinking.off);
    final p = StringBuffer();
    var system = false, users = 0;
    // The tools are described in the system block, as the model's template does.
    if (tools.isNotEmpty) {
      final first = r.messages.first;
      p.write(dialect.systemFor(first.role == 'system' ? first.content : '', tools));
      system = true;
    }
    final results = <String>[]; // consecutive tool results: one user turn
    void flushResults() {
      if (results.isEmpty) return;
      p.write(t.user(ToolDialect.responses(results), first: users++ == 0, afterSystem: system));
      results.clear();
    }

    for (final (i, m) in r.messages.indexed) {
      if (m.role != 'tool') flushResults();
      switch (m.role) {
        case 'system':
          if (tools.isNotEmpty && i == 0) continue;
          p.write(t.system(m.content));
          system = system || i == 0;
        case 'user':
          p.write(t.user(m.content, first: users++ == 0, afterSystem: system));
        case 'tool':
          results.add(m.content);
        case 'assistant':
          final text = m.calls.isEmpty
              ? m.content
              : dialect.assistantText(m.content, [for (final c in m.calls) dialect.callText(c.name, c.arguments)]);
          p.write(continued && i == r.messages.length - 1 ? t.reply(thinking) + text : t.assistant(text));
        default:
          throw ApiError(400, 'unsupported role: ${m.role}');
      }
    }
    flushResults();
    if (!continued) p.write(t.reply(thinking));
    return (p.toString(), thinking == Thinking.on);
  }

  /// Constrained decoding for `r`: the whole answer's grammar (a response
  /// schema), or the structure of a reply that may call tools; both null for
  /// free text. ApiError when a response schema uses what the engine cannot
  /// enforce; a tool whose schema it cannot enforce gets arguments of any
  /// shape instead (a JSON object, or any parameters), so a client's long
  /// tool list keeps working.
  (String?, Continuation Function(String)?) constraints(ApiRequest r, {required bool reasoning}) {
    final schema = r.responseSchema;
    final tools = r.prompt != null ? const <ToolSpec>[] : _offered(r);
    if (schema != null && tools.isNotEmpty) {
      throw const ApiError(400, 'response_format with tools is not supported: use one or the other');
    }
    if (schema != null) {
      final body = '%schema ${jsonEncode(schema)}';
      final error = backend.grammarError('root ::= $body');
      if (error != null) throw ApiError(400, 'response_format: $error');
      // A reasoning reply thinks freely first, then answers.
      return (reasoning ? 'root ::= %until "</think>" "\\n\\n" $body' : 'root ::= $body', null);
    }
    if (tools.isEmpty) return (null, null);
    final dialect = _dialect(r);
    final name = r.toolName;
    final callable = name == null ? tools : [for (final t in tools) if (t.name == name) t];
    final loose = {
      for (final t in callable)
        if (backend.grammarError(dialect.callGrammar([t])) != null) t.name,
    };
    return (
      null,
      toolStructure(
        grammar: dialect.callGrammar(callable, loose: loose),
        grammarAfterNewline: dialect.callGrammar(callable, newline: false, loose: loose),
        mustCall: r.toolChoice == ToolChoice.required || name != null,
        thinking: reasoning,
      )
    );
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
    final (grammar, structure) = constraints(r, reasoning: reasoning);
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
    final calls = structure == null ? null : ToolCallParser(backend.toolDialect, r.tools);
    // The answer's text goes through the call parser when the reply may call tools.
    Iterable<ApiEvent> split(List<ApiEvent> events) sync* {
      for (final e in events) {
        if (calls != null && e is ApiText) {
          yield* calls.feed(e.text);
        } else {
          yield e;
        }
      }
    }

    final generation = backend.generate(text, sampling, grammar: grammar, structure: structure);
    var stopping = false;
    try {
      // Leaving this loop early (the client went away) cancels the subscription, which stops the generation.
      await for (final event in generation.events) {
        switch (event) {
          case TextPiece(:final text):
            for (final e in split(out.feed(text))) {
              yield e;
            }
            final enough = calls != null && !r.parallelToolCalls && calls.completed > 0;
            if ((out.stopped != null || enough) && !stopping) {
              stopping = true;
              generation.stop(); // it still ends with GenerationDone, which has the counts
            }
          case GenerationDone(:final stats):
            for (final e in [...split(out.flush()), ...?calls?.flush()]) {
              yield e;
            }
            final cut = stats.generatedTokens >= sampling.maxTokens;
            final reason = out.stopped != null
                ? FinishReason.stopSequence
                : (calls != null && calls.completed > 0 && !cut)
                    ? FinishReason.toolCalls
                    : (cut ? FinishReason.length : FinishReason.stop);
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
