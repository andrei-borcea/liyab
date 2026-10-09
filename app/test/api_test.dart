// The local API: prompt building, reasoning and stop sequences, and the
// OpenAI, Anthropic and gRPC front ends over real sockets with a scripted model.
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:grpc/grpc.dart' as grpc;
import 'package:liyab/api/api_core.dart';
import 'package:liyab/api/local_api.dart';
import 'package:liyab/chat/chat_template.dart';
import 'package:liyab/engine/engine_service.dart';
import 'package:protobuf/protobuf.dart';

/// Replies with `pieces`, then GenerationDone; records prompts and stops.
class FakeBackend implements ApiBackend {
  List<String> pieces = ['Hello', ' there'];
  Duration pieceDelay = Duration.zero; // a real model takes ~0.1 s a token
  final prompts = <String>[];
  var stops = 0;

  @override
  String get modelName => 'test-model.gguf';
  @override
  ChatTemplate get template => ChatTemplate.chatml;
  @override
  bool get thinkingSupported => true;
  @override
  SamplingOptions get defaults => const SamplingOptions(maxTokens: 50);
  @override
  Future<void> ensureLoaded() async {}
  @override
  Future<int> countTokens(String text) async => text.length;

  @override
  Generation generate(String prompt, SamplingOptions sampling) {
    prompts.add(prompt);
    var stopped = false;
    void stop() {
      if (!stopped) ++stops; // like EngineService's, stopping twice is stopping once
      stopped = true;
    }

    // As EngineService's: cancelling the subscription stops the generation.
    final controller = StreamController<GenerationEvent>(onCancel: stop);
    () async {
      var n = 0;
      for (final p in pieces) {
        await Future<void>.delayed(pieceDelay);
        if (stopped) break;
        controller.add(TextPiece(p));
        ++n;
      }
      controller.add(GenerationDone(GenerationStats(
        promptTokens: prompt.length,
        generatedTokens: n,
        tokensPerSecond: 10,
        ttftMs: 1,
        cachedPrefixTokens: 3,
        thermalReroutes: 0,
        cancelled: stopped,
      )));
      await controller.close();
    }();
    return Generation(controller.stream, stop);
  }
}

List<String> texts(List<ApiEvent> events) => [
      for (final e in events)
        if (e is ApiText) 'T:${e.text}' else if (e is ApiReasoning) 'R:${e.text}',
    ];

void main() {
  test('reasoning is split from the answer and stop sequences cut it, across pieces', () {
    final s = ReplyShaper(reasoning: true, stop: const ['END']);
    final events = [
      for (final p in ['Let me', ' think.</th', 'ink>\n\nThe answer', ' is 4E', 'ND and more']) ...s.feed(p),
      ...s.flush(),
    ];
    expect(texts(events), ['R:Let me', 'R: think.', 'T:The answer', 'T: is 4']);
    expect(s.stopped, 'END');

    final plain = ReplyShaper(reasoning: false, stop: const ['\n\n']);
    expect(texts([...plain.feed('\nA line\n'), ...plain.feed('more'), ...plain.flush()]), ['T:A line', 'T:\nmore']);
    expect(plain.stopped, isNull);

    // A raw completion keeps its leading whitespace.
    final raw = ReplyShaper(reasoning: false, trimStart: false);
    expect(texts([...raw.feed(' blue'), ...raw.flush()]), ['T: blue']);
  });

  test('prompts use the model\'s template; thinking and a prefilled reply', () {
    final core = ApiCore(FakeBackend());
    final (p, reasoning) = core.prompt(const ApiRequest(
      messages: [ApiMessage('system', 'Be brief.'), ApiMessage('user', 'Hi')],
      thinking: true,
    ));
    expect(p, '<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n');
    expect(reasoning, isTrue);
    final (q, r2) = core.prompt(const ApiRequest(messages: [ApiMessage('user', 'Hi'), ApiMessage('assistant', 'Sure,')]));
    expect(q, '<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nSure,');
    expect(r2, isFalse);
    expect(core.prompt(const ApiRequest(prompt: 'raw text')).$1, 'raw text');
    expect(() => core.prompt(const ApiRequest(messages: [ApiMessage('tool', 'x')])), throwsA(isA<ApiError>()));
  });

  group('servers', () {
    late FakeBackend backend;
    late LocalApi api;
    late HttpClient http;
    const token = 'liyab-test-token';

    setUp(() async {
      backend = FakeBackend();
      api = LocalApi(backend, token);
      await api.start(http: 0, grpcPort: 0);
      http = HttpClient();
    });
    tearDown(() async {
      http.close(force: true);
      await api.stop();
    });

    Future<(int, String)> call(String method, String path,
        {Object? body, Map<String, String> headers = const {'authorization': 'Bearer $token'}}) async {
      final req = await http.open(method, '127.0.0.1', api.boundHttpPort!, path);
      headers.forEach(req.headers.set);
      if (body != null) {
        req.headers.contentType = ContentType.json;
        req.write(jsonEncode(body));
      }
      final res = await req.close();
      return (res.statusCode, await utf8.decodeStream(res));
    }

    test('without the token nothing runs', () async {
      final (status, body) = await call('POST', '/v1/chat/completions', body: {'messages': []}, headers: const {});
      expect(status, 401);
      expect(jsonDecode(body)['error']['type'], 'authentication_error');
      final (wrong, _) = await call('GET', '/v1/models', headers: const {'authorization': 'Bearer liyab-other-token'});
      expect(wrong, 401);
      expect(backend.prompts, isEmpty);
      expect((await call('GET', '/health', headers: const {})).$1, 200);
    });

    test('OpenAI: models, chat completion, reasoning, usage, streaming', () async {
      final (_, models) = await call('GET', '/v1/models');
      expect(jsonDecode(models)['data'][0]['id'], 'test-model.gguf');

      backend.pieces = ['Hmm', '</think>\n\n', 'Four', '.'];
      final (status, body) = await call('POST', '/v1/chat/completions', body: {
        'model': 'any',
        'messages': [
          {'role': 'developer', 'content': 'Be brief.'},
          {
            'role': 'user',
            'content': [
              {'type': 'text', 'text': '2+2?'},
            ],
          },
        ],
        'reasoning_effort': 'low',
        'max_tokens': 20,
      });
      expect(status, 200);
      final o = jsonDecode(body);
      expect(o['object'], 'chat.completion');
      expect(o['choices'][0]['message'], {'role': 'assistant', 'content': 'Four.', 'reasoning_content': 'Hmm'});
      expect(o['choices'][0]['finish_reason'], 'stop');
      expect(o['usage']['completion_tokens'], 4);
      expect(backend.prompts.last, endsWith('2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n'));

      backend.pieces = ['Hello', ' there'];
      final (_, sse) = await call('POST', '/v1/chat/completions', body: {
        'messages': [
          {'role': 'user', 'content': 'Hi'},
        ],
        'stream': true,
        'stream_options': {'include_usage': true},
      });
      final data = [for (final l in sse.split('\n')) if (l.startsWith('data: ')) l.substring(6)];
      expect(data.last, '[DONE]');
      final chunks = [for (final d in data.take(data.length - 1)) jsonDecode(d) as Map<String, Object?>];
      final content = chunks
          .where((c) => (c['choices'] as List).isNotEmpty)
          .map((c) => ((c['choices'] as List)[0]['delta'] as Map)['content'] ?? '')
          .join();
      expect(content, 'Hello there');
      expect(((chunks[chunks.length - 2]['choices'] as List)[0] as Map)['finish_reason'], 'stop');
      expect((chunks.last['usage'] as Map)['completion_tokens'], 2);
    });

    test('OpenAI: a stop sequence stops the generation; refused requests', () async {
      backend.pieces = ['one', ' two', ' STOP', ' three', ' four'];
      final (_, body) = await call('POST', '/v1/completions', body: {'prompt': 'count:', 'stop': ' STOP'});
      final o = jsonDecode(body);
      expect(o['choices'][0]['text'], 'one two');
      expect(o['choices'][0]['finish_reason'], 'stop');
      expect(backend.stops, 1);
      expect(backend.prompts.last, 'count:');

      final (tools, _) = await call('POST', '/v1/chat/completions', body: {
        'messages': [
          {'role': 'user', 'content': 'x'},
        ],
        'tools': [
          {'type': 'function'},
        ],
      });
      expect(tools, 400);
      final (bad, err) = await call('POST', '/v1/chat/completions', body: {'messages': 'nope'});
      expect(bad, 400);
      expect(jsonDecode(err)['error']['type'], 'invalid_request_error');
    });

    test('a client that goes away mid-stream stops its generation', () async {
      backend
        ..pieces = List.filled(2000, 'word ')
        ..pieceDelay = const Duration(milliseconds: 2);
      final req = await http.open('POST', '127.0.0.1', api.boundHttpPort!, '/v1/chat/completions');
      req.headers.set('authorization', 'Bearer $token');
      req.write(jsonEncode({
        'messages': [
          {'role': 'user', 'content': 'Talk'},
        ],
        'stream': true,
      }));
      final res = await req.close();
      await res.first; // the stream has started
      http.close(force: true);
      for (var i = 0; i < 200 && backend.stops == 0; ++i) {
        await Future<void>.delayed(const Duration(milliseconds: 10));
      }
      expect(backend.stops, 1);
    });

    test('Anthropic: messages with thinking, streaming events, count_tokens', () async {
      final anthropic = {'x-api-key': token, 'anthropic-version': '2023-06-01'};
      backend.pieces = ['Hmm', '</think>', '\n\nFour'];
      final (status, body) = await call('POST', '/v1/messages', headers: anthropic, body: {
        'model': 'any',
        'max_tokens': 30,
        'system': 'Be brief.',
        'thinking': {'type': 'enabled', 'budget_tokens': 1024},
        'messages': [
          {'role': 'user', 'content': '2+2?'},
        ],
      });
      expect(status, 200);
      final o = jsonDecode(body);
      expect(o['type'], 'message');
      expect(o['content'], [
        {'type': 'thinking', 'thinking': 'Hmm', 'signature': ''},
        {'type': 'text', 'text': 'Four'},
      ]);
      expect(o['stop_reason'], 'end_turn');
      expect(o['usage']['output_tokens'], 3);

      backend.pieces = ['Hi', '!'];
      final (_, sse) = await call('POST', '/v1/messages', headers: anthropic, body: {
        'max_tokens': 30,
        'stream': true,
        'messages': [
          {'role': 'user', 'content': 'Hello'},
        ],
      });
      final events = [for (final l in sse.split('\n')) if (l.startsWith('event: ')) l.substring(7)];
      expect(events, [
        'message_start',
        'content_block_start',
        'content_block_delta',
        'content_block_delta',
        'content_block_stop',
        'message_delta',
        'message_stop',
      ]);

      final (_, count) = await call('POST', '/v1/messages/count_tokens', headers: anthropic, body: {
        'messages': [
          {'role': 'user', 'content': 'Hello'},
        ],
      });
      expect(jsonDecode(count)['input_tokens'], greaterThan(0));
      final (missing, err) = await call('POST', '/v1/messages', headers: anthropic, body: {
        'messages': [
          {'role': 'user', 'content': 'Hello'},
        ],
      });
      expect(missing, 400);
      expect(jsonDecode(err)['error']['type'], 'invalid_request_error');
    });

    test('gRPC: Generate streams pieces and Done; the token is required', () async {
      final channel = grpc.ClientChannel('127.0.0.1',
          port: api.boundGrpcPort!,
          options: const grpc.ChannelOptions(credentials: grpc.ChannelCredentials.insecure()));
      addTearDown(channel.shutdown);
      final generate = grpc.ClientMethod<List<int>, List<int>>('/liyab.v1.Liyab/Generate', (b) => b, (b) => b);
      final message = (CodedBufferWriter()
            ..writeField(1, PbFieldType.OS, 'user')
            ..writeField(2, PbFieldType.OS, 'Hi'))
          .toBuffer();
      final request = (CodedBufferWriter()
            ..writeField(1, PbFieldType.OY, message)
            ..writeField(6, PbFieldType.O3, 10))
          .toBuffer();
      final client = grpc.Client(channel);

      final replies = await client
          .$createStreamingCall(generate, Stream.value(request),
              options: grpc.CallOptions(metadata: {'authorization': 'Bearer $token'}))
          .toList();
      final text = StringBuffer();
      var completion = -1;
      for (final r in replies) {
        final reader = CodedBufferReader(r);
        final tag = reader.readTag();
        if (tag >> 3 == 1) text.write(reader.readString());
        if (tag >> 3 == 3) {
          final done = CodedBufferReader(reader.readBytes());
          while (!done.isAtEnd()) {
            final t = done.readTag();
            t >> 3 == 4 ? completion = done.readInt32() : done.skipField(t);
          }
        }
      }
      expect(text.toString(), 'Hello there');
      expect(completion, 2);

      await expectLater(
        client.$createStreamingCall(generate, Stream.value(request)).toList(),
        throwsA(isA<grpc.GrpcError>().having((e) => e.code, 'code', grpc.StatusCode.unauthenticated)),
      );
    });
  });
}
