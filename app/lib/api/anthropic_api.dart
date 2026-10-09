// Anthropic Messages API (/v1/messages, /v1/messages/count_tokens, models)
// over the local API's core. Thinking comes back as `thinking` blocks (with an
// empty signature: there is nothing to verify locally); a last assistant
// message is continued, as the API's prefill. Not supported: tools, images,
// documents.
import 'dart:async';

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

  static Map<String, Object?> models(String model) => {
        'data': [
          if (model.isNotEmpty)
            {'type': 'model', 'id': model, 'display_name': model, 'created_at': '1970-01-01T00:00:00Z'},
        ],
        'has_more': false,
        'first_id': model.isEmpty ? null : model,
        'last_id': model.isEmpty ? null : model,
      };

  static ApiRequest request(Map<String, Object?> body, {bool requireMaxTokens = true}) {
    final tools = body['tools'];
    if (tools is List && tools.isNotEmpty) throw const ApiError(400, 'tools are not supported');
    final maxTokens = intField(body, 'max_tokens');
    if (requireMaxTokens && maxTokens == null) throw const ApiError(400, 'max_tokens is required');
    final raw = body['messages'];
    if (raw is! List) throw const ApiError(400, 'messages is required');
    final system = textContent(body['system']);
    final thinking = body['thinking'];
    return ApiRequest(
      messages: [
        if (system.isNotEmpty) ApiMessage('system', system),
        for (final m in raw)
          if (m is Map<String, Object?> && (m['role'] == 'user' || m['role'] == 'assistant'))
            // Earlier thinking is not replayed, as Anthropic's own models drop it.
            ApiMessage(m['role'] as String, textContent(m['content'], skip: const {'thinking', 'redacted_thinking'}))
          else
            throw const ApiError(400, 'each message must have the role user or assistant'),
      ],
      temperature: numberField(body, 'temperature'),
      topP: numberField(body, 'top_p'),
      topK: intField(body, 'top_k'),
      maxTokens: maxTokens,
      stop: stopList(body['stop_sequences']),
      thinking: thinking is Map && thinking['type'] != 'disabled',
    );
  }

  Future<void> countTokens(Map<String, Object?> body, ApiConnection conn) async =>
      conn.json(200, {'input_tokens': await core.countTokens(request(body, requireMaxTokens: false))});

  static String _stopReason(ApiDone d) => switch (d.reason) {
        FinishReason.stop => 'end_turn',
        FinishReason.length => 'max_tokens',
        FinishReason.stopSequence => 'stop_sequence',
      };

  Future<void> messages(Map<String, Object?> body, ApiConnection conn) async {
    final r = request(body);
    final id = responseId('msg_');
    final model = core.backend.modelName;
    final it = await startRun(core, r, conn);
    final promptTokens = (it.current as ApiStart).promptTokens;
    Map<String, Object?> usage(int output, ApiDone? d) => {
          'input_tokens': promptTokens,
          'output_tokens': output,
          'cache_read_input_tokens': d?.stats.cachedPrefixTokens ?? 0,
        };

    if (body['stream'] != true) {
      final text = StringBuffer(), reasoning = StringBuffer();
      ApiDone? done;
      while (await it.moveNext()) {
        switch (it.current) {
          case ApiText(text: final t):
            text.write(t);
          case ApiReasoning(text: final t):
            reasoning.write(t);
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
          {'type': 'text', 'text': text.toString()},
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
    Future<void> block(String type) async {
      if (open == type) return;
      if (open != null) await send('content_block_stop', {'index': index});
      open = type;
      await send('content_block_start', {
        'index': ++index,
        'content_block': type == 'thinking' ? {'type': 'thinking', 'thinking': '', 'signature': ''} : {'type': 'text', 'text': ''},
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
