// OpenAI API (chat completions, completions, models) over the local API's
// core. Reasoning is returned as `reasoning_content`, as vLLM does. Not
// supported: tools, images and other non-text content, n > 1, logprobs.
import 'dart:async';

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
    final tools = body['tools'];
    if (tools is List && tools.isNotEmpty) throw const ApiError(400, 'tools are not supported');
    if ((body['n'] as num? ?? 1) != 1) throw const ApiError(400, 'n must be 1');
    if (body['logprobs'] == true) throw const ApiError(400, 'logprobs are not supported');
  }

  static ApiRequest chatRequest(Map<String, Object?> body) {
    _refuseUnsupported(body);
    final raw = body['messages'];
    if (raw is! List) throw const ApiError(400, 'messages is required');
    final messages = [
      for (final m in raw)
        if (m is Map<String, Object?>)
          ApiMessage(
            switch (m['role']) {
              'system' || 'developer' => 'system',
              'user' => 'user',
              'assistant' when m['tool_calls'] == null => 'assistant',
              _ => throw ApiError(400, 'unsupported message: role ${m['role']}${m['tool_calls'] != null ? ' with tool calls' : ''}'),
            },
            textContent(m['content']),
          )
        else
          throw const ApiError(400, 'each message must be an object'),
    ];
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
    );
  }

  static ApiRequest completionRequest(Map<String, Object?> body) {
    _refuseUnsupported(body);
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
    String finish(ApiDone d) => d.reason == FinishReason.length ? 'length' : 'stop';

    if (body['stream'] != true) {
      final text = StringBuffer(), reasoning = StringBuffer();
      ApiDone? done;
      while (await it.moveNext()) {
        switch (it.current) {
          case ApiText(text: final text_):
            text.write(text_);
          case ApiReasoning(text: final text_):
            reasoning.write(text_);
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
                'content': text.toString(),
                if (reasoning.isNotEmpty) 'reasoning_content': reasoning.toString(),
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
