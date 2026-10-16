// OpenAI API (chat completions, completions, embeddings, models) over the
// local API's core. Reasoning is returned as `reasoning_content`, as vLLM does. Tools
// (`tools`, `tool_choice`, `parallel_tool_calls`; calls in `tool_calls`, also
// streamed as deltas; results in `tool` messages) and `response_format`
// (json_object, json_schema) are constrained decoding: calls always parse and
// their arguments, like a json_schema answer, are valid against the schema.
// Embeddings come from the loaded embedding model, as float lists or base64
// (the official clients' default), with `dimensions` for models trained for
// it. Not supported: images and other non-text content, n > 1, logprobs,
// token arrays as embedding input.
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import '../agent/tool_format.dart';
import 'api_core.dart';
import 'local_api.dart';

class OpenAiApi {
  OpenAiApi(this.core);
  final ApiCore core;

  static Map<String, Object?> error(int status, String message) => {
        'error': {
          'message': message,
          'type': switch (status) {
            401 => 'authentication_error',
            404 => 'not_found_error',
            503 => 'service_unavailable',
            >= 500 => 'server_error',
            _ => 'invalid_request_error',
          },
          'code': status,
        },
      };

  static Map<String, Object?> models(String model) => {
        'object': 'list',
        'data': [
          if (model.isNotEmpty) {'id': model, 'object': 'model', 'created': 0, 'owned_by': 'liyab'},
        ],
      };

  static void _refuseUnsupported(Map<String, Object?> body) {
    if ((body['n'] as num? ?? 1) != 1) throw const ApiError(400, 'n must be 1');
    if (body['logprobs'] == true) throw const ApiError(400, 'logprobs are not supported');
  }

  static List<ToolSpec> _tools(Object? raw) {
    if (raw == null) return const [];
    if (raw is! List) throw const ApiError(400, 'tools must be a list');
    return [
      for (final t in raw)
        if (t is Map && t['type'] == 'function' && t['function'] is Map && (t['function'] as Map)['name'] is String)
          ToolSpec(
            (t['function'] as Map)['name'] as String,
            '${(t['function'] as Map)['description'] ?? ''}',
            switch ((t['function'] as Map)['parameters']) {
              null => const {'type': 'object', 'properties': <String, Object?>{}},
              Map p => p.cast<String, Object?>(),
              _ => throw const ApiError(400, 'a function\'s parameters must be a JSON schema object'),
            },
          )
        else
          throw const ApiError(400, 'each tool must be {"type": "function", "function": {"name": ...}}'),
    ];
  }

  /// tool_choice: "auto", "none", "required" or {"type": "function", "function": {"name": ...}}.
  static (ToolChoice, String?) _toolChoice(Object? raw) => switch (raw) {
        null || 'auto' => (ToolChoice.auto, null),
        'none' => (ToolChoice.none, null),
        'required' => (ToolChoice.required, null),
        {'type': 'function', 'function': {'name': String name}} => (ToolChoice.required, name),
        _ => throw const ApiError(400, 'tool_choice must be "auto", "none", "required" or a function'),
      };

  /// response_format: text, json_object (any JSON object) or json_schema.
  static Map<String, Object?>? _responseSchema(Object? raw) {
    if (raw == null) return null;
    if (raw is! Map) throw const ApiError(400, 'response_format must be an object');
    switch (raw['type']) {
      case 'text':
        return null;
      case 'json_object':
        return const {'type': 'object'};
      case 'json_schema':
        final spec = raw['json_schema'];
        final schema = spec is Map ? spec['schema'] : null;
        if (schema == null || schema == true) return const <String, Object?>{};
        if (schema is Map) return schema.cast<String, Object?>();
        throw const ApiError(400, 'response_format.json_schema.schema must be a JSON schema object');
      default:
        throw const ApiError(400, 'response_format must be text, json_object or json_schema');
    }
  }

  static ApiMessage _message(Map<String, Object?> m) {
    final role = m['role'];
    switch (role) {
      case 'system' || 'developer':
        return ApiMessage('system', textContent(m['content']));
      case 'user':
        return ApiMessage('user', textContent(m['content']));
      case 'tool':
        return ApiMessage('tool', textContent(m['content']));
      case 'assistant':
        final calls = m['tool_calls'];
        if (calls != null && calls is! List) throw const ApiError(400, 'tool_calls must be a list');
        return ApiMessage('assistant', textContent(m['content']), calls: [
          for (final c in (calls as List?) ?? const [])
            switch (c) {
              {'function': {'name': String name, 'arguments': final Object? args}} => ApiCall(name, _arguments(args)),
              _ => throw const ApiError(400, 'each tool call must have a function name and arguments'),
            },
        ]);
      default:
        throw ApiError(400, 'unsupported message: role $role');
    }
  }

  /// A past call's arguments: a JSON object, or the JSON text of one (as OpenAI sends them).
  static Map<String, Object?> _arguments(Object? raw) {
    try {
      final value = raw is String ? (raw.trim().isEmpty ? <String, Object?>{} : jsonDecode(raw)) : raw;
      if (value is Map) return value.cast<String, Object?>();
    } on FormatException {
      // refused below
    }
    throw const ApiError(400, 'tool call arguments must be a JSON object');
  }

  static ApiRequest chatRequest(Map<String, Object?> body) {
    _refuseUnsupported(body);
    final raw = body['messages'];
    if (raw is! List) throw const ApiError(400, 'messages is required');
    final messages = [
      for (final m in raw)
        if (m is Map<String, Object?>) _message(m) else throw const ApiError(400, 'each message must be an object'),
    ];
    final (choice, toolName) = _toolChoice(body['tool_choice']);
    // vLLM's chat_template_kwargs.enable_thinking, else OpenAI's reasoning_effort.
    final kwargs = body['chat_template_kwargs'];
    final enable = kwargs is Map ? kwargs['enable_thinking'] : null;
    final effort = body['reasoning_effort'];
    return ApiRequest(
      messages: messages,
      temperature: numberField(body, 'temperature'),
      topP: numberField(body, 'top_p'),
      topK: intField(body, 'top_k'),
      maxTokens: intField(body, 'max_completion_tokens') ?? intField(body, 'max_tokens'),
      stop: stopList(body['stop']),
      thinking: enable is bool ? enable : (effort is String && effort != 'none' && effort != 'minimal'),
      tools: _tools(body['tools']),
      toolChoice: choice,
      toolName: toolName,
      parallelToolCalls: body['parallel_tool_calls'] != false,
      responseSchema: _responseSchema(body['response_format']),
    );
  }

  static ApiRequest completionRequest(Map<String, Object?> body) {
    _refuseUnsupported(body);
    final tools = body['tools'];
    if (tools is List && tools.isNotEmpty) throw const ApiError(400, 'tools need /v1/chat/completions');
    final prompt = switch (body['prompt']) {
      String p => p,
      [String p] => p,
      _ => throw const ApiError(400, 'prompt must be one string'),
    };
    return ApiRequest(
      prompt: prompt,
      temperature: numberField(body, 'temperature'),
      topP: numberField(body, 'top_p'),
      topK: intField(body, 'top_k'),
      maxTokens: intField(body, 'max_tokens') ?? 16, // the OpenAI default for completions
      stop: stopList(body['stop']),
    );
  }

  Future<void> chat(Map<String, Object?> body, ApiConnection conn) =>
      _respond(chatRequest(body), body, conn, chat: true);

  /// /v1/embeddings: `input` (a string or a list of strings), `encoding_format` ("float" or "base64"),
  /// `dimensions`; `model` and `user` are accepted and ignored. The texts, the dimensions (0: all) and whether the
  /// vectors go out as base64.
  static (List<String>, int, bool) embeddingRequest(Map<String, Object?> body) {
    final input = switch (body['input']) {
      String text => [text],
      List list when list.every((x) => x is String) => List<String>.from(list),
      List _ => throw const ApiError(400, 'input must be a string or a list of strings (token arrays are not supported)'),
      _ => throw const ApiError(400, 'input is required: a string or a list of strings'),
    };
    if (input.isEmpty) throw const ApiError(400, 'input must not be empty');
    final format = body['encoding_format'] ?? 'float';
    if (format != 'float' && format != 'base64') throw const ApiError(400, 'encoding_format must be float or base64');
    final dimensions = body['dimensions'];
    if (dimensions != null && (dimensions is! int || dimensions <= 0)) {
      throw const ApiError(400, 'dimensions must be a positive integer');
    }
    return (input, dimensions as int? ?? 0, format == 'base64');
  }

  Future<void> embeddings(Map<String, Object?> body, ApiConnection conn) async {
    final (input, dimensions, base64) = embeddingRequest(body);
    final result = await core.embed(input, dimensions: dimensions);
    return conn.json(200, {
      'object': 'list',
      'data': [
        for (final (i, v) in result.vectors.indexed)
          {
            'object': 'embedding',
            'index': i,
            // base64: the vector's float32 values, little-endian, as OpenAI sends them.
            'embedding': base64 ? base64Encode(Uint8List.view(v.buffer, v.offsetInBytes, v.lengthInBytes)) : v,
          },
      ],
      'model': core.backend.modelName,
      'usage': {'prompt_tokens': result.tokens, 'total_tokens': result.tokens},
    });
  }

  Future<void> completion(Map<String, Object?> body, ApiConnection conn) =>
      _respond(completionRequest(body), body, conn, chat: false);

  Future<void> _respond(ApiRequest r, Map<String, Object?> body, ApiConnection conn, {required bool chat}) async {
    final id = responseId(chat ? 'chatcmpl-' : 'cmpl-');
    final created = DateTime.now().millisecondsSinceEpoch ~/ 1000;
    final model = core.backend.modelName;
    final object = chat ? 'chat.completion' : 'text_completion';
    final it = await startRun(core, r, conn);
    final promptTokens = (it.current as ApiStart).promptTokens;

    Map<String, Object?> usage(ApiDone d) => {
          'prompt_tokens': promptTokens,
          'completion_tokens': d.completionTokens,
          'total_tokens': promptTokens + d.completionTokens,
          'prompt_tokens_details': {'cached_tokens': d.stats.cachedPrefixTokens},
        };
    String finish(ApiDone d) => switch (d.reason) {
          FinishReason.length => 'length',
          FinishReason.toolCalls => 'tool_calls',
          _ => 'stop',
        };
    // Call ids: one per call of this response.
    final callIds = <int, String>{};
    String callId(int index) => callIds.putIfAbsent(index, () => responseId('call_'));

    if (body['stream'] != true) {
      final text = StringBuffer(), reasoning = StringBuffer();
      final calls = <(String, StringBuffer)>[];
      ApiDone? done;
      while (await it.moveNext()) {
        switch (it.current) {
          case ApiText(text: final text_):
            text.write(text_);
          case ApiReasoning(text: final text_):
            reasoning.write(text_);
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
        'object': object,
        'created': created,
        'model': model,
        'choices': [
          {
            'index': 0,
            if (chat)
              'message': {
                'role': 'assistant',
                'content': calls.isNotEmpty && text.isEmpty ? null : text.toString(),
                if (reasoning.isNotEmpty) 'reasoning_content': reasoning.toString(),
                if (calls.isNotEmpty)
                  'tool_calls': [
                    for (final (i, (name, args)) in calls.indexed)
                      {
                        'id': callId(i),
                        'type': 'function',
                        'function': {'name': name, 'arguments': args.toString()},
                      },
                  ],
              }
            else
              'text': text.toString(),
            'logprobs': null,
            'finish_reason': finish(done),
          },
        ],
        'usage': usage(done),
      });
    }

    final sse = conn;
    final options = body['stream_options'];
    final includeUsage = options is Map && options['include_usage'] == true;
    Map<String, Object?> chunk(Map<String, Object?> choice) => {
          'id': id,
          'object': chat ? 'chat.completion.chunk' : 'text_completion',
          'created': created,
          'model': model,
          'choices': [
            {'index': 0, ...choice},
          ],
        };
    try {
      if (chat) await sse.send(chunk({'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': null}));
      while (await it.moveNext()) {
        switch (it.current) {
          case ApiText(text: final text_):
            await sse.send(chunk(chat
                ? {'delta': {'content': text_}, 'finish_reason': null}
                : {'text': text_, 'logprobs': null, 'finish_reason': null}));
          case ApiReasoning(text: final text_):
            if (chat) await sse.send(chunk({'delta': {'reasoning_content': text_}, 'finish_reason': null}));
          case ApiToolCallStart(:final index, :final name):
            await sse.send(chunk({
              'delta': {
                'tool_calls': [
                  {
                    'index': index,
                    'id': callId(index),
                    'type': 'function',
                    'function': {'name': name, 'arguments': ''},
                  },
                ],
              },
              'finish_reason': null,
            }));
          case ApiToolCallArguments(:final index, :final delta):
            await sse.send(chunk({
              'delta': {
                'tool_calls': [
                  {
                    'index': index,
                    'function': {'arguments': delta},
                  },
                ],
              },
              'finish_reason': null,
            }));
          case ApiDone d:
            await sse.send(chunk(chat ? {'delta': {}, 'finish_reason': finish(d)} : {'text': '', 'finish_reason': finish(d)}));
            if (includeUsage) {
              await sse.send({
                'id': id,
                'object': chat ? 'chat.completion.chunk' : 'text_completion',
                'created': created,
                'model': model,
                'choices': const [],
                'usage': usage(d),
              });
            }
          case ApiStart():
        }
      }
      await sse.send('[DONE]');
    } on ApiError catch (e) {
      await sse.send(error(e.status, e.message));
    } finally {
      await it.cancel(); // stops the generation if the client went away
      await sse.close();
    }
  }
}
