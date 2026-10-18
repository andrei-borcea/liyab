// Anthropic Messages API (/v1/messages, /v1/messages/count_tokens, models)
// over the local API's core. Thinking comes back as `thinking` blocks (with an
// empty signature: there is nothing to verify locally); a last assistant
// message is continued, as the API's prefill. Tools (`tools`, `tool_choice`
// with disable_parallel_tool_use; calls as `tool_use` blocks, also streamed as
// input_json_delta events; results as `tool_result` blocks) are constrained
// decoding: calls always parse and their input is valid against the tool's
// input_schema. Not supported: server tools, images, documents.
import 'dart:async';
import 'dart:convert';

import '../agent/tool_format.dart';
import 'api_core.dart';
import 'local_api.dart';

class AnthropicApi {
  AnthropicApi(this.core);
  final ApiCore core;

  static Map<String, Object?> error(int status, String message) => {
        'type': 'error',
        'error': {
          'type': switch (status) {
            401 => 'authentication_error',
            404 => 'not_found_error',
            413 => 'request_too_large',
            503 => 'overloaded_error',
            >= 500 => 'api_error',
            _ => 'invalid_request_error',
          },
          'message': message,
        },
      };

  /// The model, then the LoRA adapters loaded next to it (a request names one as its `model` to use it).
  static Map<String, Object?> models(String model, [List<String> adapters = const []]) => {
        'data': [
          if (model.isNotEmpty)
            {'type': 'model', 'id': model, 'display_name': model, 'created_at': '1970-01-01T00:00:00Z'},
          for (final a in adapters)
            {'type': 'model', 'id': a, 'display_name': a, 'created_at': '1970-01-01T00:00:00Z'},
        ],
        'has_more': false,
        'first_id': model.isEmpty ? (adapters.isEmpty ? null : adapters.first) : model,
        'last_id': adapters.isNotEmpty ? adapters.last : (model.isEmpty ? null : model),
      };

  static List<ToolSpec> _tools(Object? raw) {
    if (raw == null) return const [];
    if (raw is! List) throw const ApiError(400, 'tools must be a list');
    return [
      for (final t in raw)
        switch (t) {
          {'name': String name, 'input_schema': Map schema} when t['type'] == null || t['type'] == 'custom' =>
            ToolSpec(name, '${t['description'] ?? ''}', schema.cast<String, Object?>()),
          {'type': String type} => throw ApiError(400, 'server tools are not supported: $type'),
          _ => throw const ApiError(400, 'each tool needs a name and an input_schema'),
        },
    ];
  }

  /// A message's turns: tool_result blocks become tool results (before the
  /// text that follows them, as the user wrote it), tool_use blocks the
  /// assistant's calls. Earlier thinking is not replayed, as Anthropic's own
  /// models drop it.
  static List<ApiMessage> _messages(Map<String, Object?> m) {
    final role = m['role'];
    if (role != 'user' && role != 'assistant') throw const ApiError(400, 'each message must have the role user or assistant');
    final content = m['content'];
    if (content is! List) return [ApiMessage(role as String, textContent(content))];
    final text = <Object?>[], calls = <ApiCall>[], results = <ApiMessage>[];
    for (final block in content) {
      switch (block) {
        case {'type': 'tool_use', 'name': String name}:
          final input = block['input'];
          if (role != 'assistant' || input is! Map) throw const ApiError(400, 'tool_use blocks belong to assistant turns, with an input object');
          calls.add(ApiCall(name, input.cast<String, Object?>()));
        case {'type': 'tool_result'}:
          if (role != 'user') throw const ApiError(400, 'tool_result blocks belong to user turns');
          final result = textContent(block['content']);
          results.add(ApiMessage('tool', block['is_error'] == true ? 'Error: $result' : result));
        default:
          text.add(block);
      }
    }
    final rest = textContent(text, skip: const {'thinking', 'redacted_thinking'});
    return [
      ...results,
      if (role == 'assistant') ApiMessage('assistant', rest, calls: calls) else if (rest.isNotEmpty || results.isEmpty) ApiMessage('user', rest),
    ];
  }

  /// tool_choice: {"type": "auto" | "any" | "tool" (with "name") | "none"}.
  static (ToolChoice, String?, bool) _toolChoice(Object? raw) {
    if (raw == null) return (ToolChoice.auto, null, true);
    if (raw is! Map) throw const ApiError(400, 'tool_choice must be an object');
    final parallel = raw['disable_parallel_tool_use'] != true;
    return switch (raw['type']) {
      'auto' => (ToolChoice.auto, null, parallel),
      'any' => (ToolChoice.required, null, parallel),
      'none' => (ToolChoice.none, null, parallel),
      'tool' when raw['name'] is String => (ToolChoice.required, raw['name'] as String, parallel),
      _ => throw const ApiError(400, 'tool_choice.type must be auto, any, tool (with a name) or none'),
    };
  }

  static ApiRequest request(Map<String, Object?> body, {bool requireMaxTokens = true}) {
    final maxTokens = intField(body, 'max_tokens');
    if (requireMaxTokens && maxTokens == null) throw const ApiError(400, 'max_tokens is required');
    final raw = body['messages'];
    if (raw is! List) throw const ApiError(400, 'messages is required');
    final system = textContent(body['system']);
    final thinking = body['thinking'];
    final (choice, toolName, parallel) = _toolChoice(body['tool_choice']);
    return ApiRequest(
      messages: [
        if (system.isNotEmpty) ApiMessage('system', system),
        for (final m in raw)
          if (m is Map<String, Object?>) ..._messages(m) else throw const ApiError(400, 'each message must be an object'),
      ],
      tools: _tools(body['tools']),
      toolChoice: choice,
      toolName: toolName,
      parallelToolCalls: parallel,
      temperature: numberField(body, 'temperature'),
      topP: numberField(body, 'top_p'),
      topK: intField(body, 'top_k'),
      maxTokens: maxTokens,
      stop: stopList(body['stop_sequences']),
      thinking: thinking is Map && thinking['type'] != 'disabled',
      model: body['model'] as String?,
      adapter: body['adapter'] as String?,
    );
  }

  /// A call's input: its arguments object (empty when they never finished: max_tokens).
  static Object? _input(String args) {
    try {
      return jsonDecode(args);
    } on FormatException {
      return const <String, Object?>{};
    }
  }

  Future<void> countTokens(Map<String, Object?> body, ApiConnection conn) async =>
      conn.json(200, {'input_tokens': await core.countTokens(request(body, requireMaxTokens: false))});

  static String _stopReason(ApiDone d) => switch (d.reason) {
        FinishReason.stop => 'end_turn',
        FinishReason.length => 'max_tokens',
        FinishReason.stopSequence => 'stop_sequence',
        FinishReason.toolCalls => 'tool_use',
      };

  Future<void> messages(Map<String, Object?> body, ApiConnection conn) async {
    final r = request(body);
    final id = responseId('msg_');
    final model = core.backend.modelName;
    final it = await startRun(core, r, conn);
    final promptTokens = (it.current as ApiStart).promptTokens;
    final callIds = <int, String>{};
    String callId(int index) => callIds.putIfAbsent(index, () => responseId('toolu_'));
    Map<String, Object?> usage(int output, ApiDone? d) => {
          'input_tokens': promptTokens,
          'output_tokens': output,
          'cache_read_input_tokens': d?.stats.cachedPrefixTokens ?? 0,
        };

    if (body['stream'] != true) {
      final text = StringBuffer(), reasoning = StringBuffer();
      final calls = <(String, StringBuffer)>[];
      ApiDone? done;
      while (await it.moveNext()) {
        switch (it.current) {
          case ApiText(text: final t):
            text.write(t);
          case ApiReasoning(text: final t):
            reasoning.write(t);
          case ApiToolCallStart(:final name):
            calls.add((name, StringBuffer()));
          case ApiToolCallArguments(:final delta):
            calls.last.$2.write(delta);
          case ApiDone d:
            done = d;
          case ApiStart():
        }
      }
      if (done == null) throw const ApiError(500, 'the engine ended the reply early');
      return conn.json(200, {
        'id': id,
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': [
          if (reasoning.isNotEmpty) {'type': 'thinking', 'thinking': reasoning.toString(), 'signature': ''},
          if (text.isNotEmpty || calls.isEmpty) {'type': 'text', 'text': text.toString()},
          for (final (i, (name, args)) in calls.indexed)
            {'type': 'tool_use', 'id': callId(i), 'name': name, 'input': _input(args.toString())},
        ],
        'stop_reason': _stopReason(done),
        'stop_sequence': done.stopSequence,
        'usage': usage(done.completionTokens, done),
      });
    }

    final sse = conn;
    Future<void> send(String event, Map<String, Object?> data) => sse.send({'type': event, ...data}, event: event);
    var index = -1;
    String? open; // the type of the content block being streamed
    Future<void> block(String type, {Map<String, Object?>? start}) async {
      if (open == type && start == null) return;
      if (open != null) await send('content_block_stop', {'index': index});
      open = type;
      await send('content_block_start', {
        'index': ++index,
        'content_block': start ??
            (type == 'thinking' ? {'type': 'thinking', 'thinking': '', 'signature': ''} : {'type': 'text', 'text': ''}),
      });
    }

    try {
      await send('message_start', {
        'message': {
          'id': id,
          'type': 'message',
          'role': 'assistant',
          'model': model,
          'content': const [],
          'stop_reason': null,
          'stop_sequence': null,
          'usage': usage(0, null),
        },
      });
      while (await it.moveNext()) {
        switch (it.current) {
          case ApiReasoning(text: final t):
            await block('thinking');
            await send('content_block_delta', {
              'index': index,
              'delta': {'type': 'thinking_delta', 'thinking': t},
            });
          case ApiText(text: final t):
            await block('text');
            await send('content_block_delta', {
              'index': index,
              'delta': {'type': 'text_delta', 'text': t},
            });
          case ApiToolCallStart(index: final call, :final name):
            await block('tool_use', start: {'type': 'tool_use', 'id': callId(call), 'name': name, 'input': const {}});
          case ApiToolCallArguments(:final delta):
            await send('content_block_delta', {
              'index': index,
              'delta': {'type': 'input_json_delta', 'partial_json': delta},
            });
          case ApiDone d:
            if (open == null) await block('text'); // a reply has at least one (possibly empty) text block
            await send('content_block_stop', {'index': index});
            await send('message_delta', {
              'delta': {'stop_reason': _stopReason(d), 'stop_sequence': d.stopSequence},
              'usage': {'output_tokens': d.completionTokens},
            });
            await send('message_stop', const {});
          case ApiStart():
        }
      }
    } on ApiError catch (e) {
      await sse.send(error(e.status, e.message), event: 'error');
    } finally {
      await it.cancel(); // stops the generation if the client went away
      await sse.close();
    }
  }
}
