// The local API: prompt building, reasoning and stop sequences, and the
// OpenAI, Anthropic and gRPC front ends over real sockets with a scripted model.
import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:grpc/grpc.dart' as grpc;
import 'package:liyab/agent/tool_format.dart';
import 'package:liyab/api/api_core.dart';
import 'package:liyab/api/local_api.dart';
import 'package:liyab/api/tool_calls.dart';
import 'package:liyab/chat/chat_template.dart';
import 'package:liyab/engine/engine_service.dart';
import 'package:protobuf/protobuf.dart';

/// Replies with `pieces`, then GenerationDone; records prompts and stops.
/// Like the engine, it asks a structure callback before the first piece and
/// after each one while no constrained span is active, and streams the
/// forced text it returns; a call's span ends with its `</tool_call>`. The
/// grammars themselves are the engine's (test_grammar checks these).
class FakeBackend implements ApiBackend {
  List<String> pieces = ['Hello', ' there'];
  Duration pieceDelay = Duration.zero; // a real model takes ~0.1 s a token
  final prompts = <String>[];
  var stops = 0;
  ToolDialect dialect = ToolDialect.hermesJson;
  final grammars = <String?>[]; // the whole-output grammar of each generation
  final spans = <String>[]; // grammars the structure callbacks opened
  final forced = <String>[]; // text they forced

  @override
  ToolDialect get toolDialect => dialect;
  // The engine refuses keywords it cannot enforce, naming them.
  @override
  String? grammarError(String grammar) =>
      grammar.contains('"pattern"') ? "JSON schema at #/properties/x: keyword 'pattern' is not supported" : null;

  bool loaded = true; // false: released while idle, as after a start without a window
  @override
  String get modelName => loaded ? 'test-model.gguf' : '';
  @override
  String get servedModel => 'test-model.gguf';
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
  Generation generate(String prompt, SamplingOptions sampling,
      {String? grammar, Continuation Function(String generated)? structure}) {
    prompts.add(prompt);
    grammars.add(grammar);
    var stopped = false;
    void stop() {
      if (!stopped) ++stops; // like EngineService's, stopping twice is stopping once
      stopped = true;
    }

    // As EngineService's: cancelling the subscription stops the generation.
    final controller = StreamController<GenerationEvent>(onCancel: stop);
    () async {
      var n = 0;
      var generated = '';
      var span = false;
      void ask() {
        if (structure == null || span) return;
        final c = structure(generated);
        if (c.text.isNotEmpty) {
          forced.add(c.text);
          controller.add(TextPiece(c.text));
          generated += c.text;
        }
        if (c.grammar != null) {
          spans.add(c.grammar!);
          span = true;
        }
      }

      ask();
      for (final p in pieces) {
        await Future<void>.delayed(pieceDelay);
        if (stopped) break;
        controller.add(TextPiece(p));
        generated += p;
        ++n;
        if (span && generated.endsWith('</tool_call>')) span = false;
        ask();
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

const weather = ToolSpec('get_weather', 'Weather', {
  'type': 'object',
  'properties': {
    'city': {'type': 'string'},
    'unit': {
      'enum': ['celsius', 'fahrenheit'],
    },
    'days': {'type': 'integer', 'minimum': 1, 'maximum': 14},
  },
  'required': ['city'],
});
const clock = ToolSpec('now', 'Time', {'type': 'object', 'properties': <String, Object?>{}});

/// The tools as an OpenAI request lists them.
List<Map<String, Object?>> openAiTools(List<ToolSpec> tools) => [for (final t in tools) t.schema];

/// The parts of a parser's events, as text.
List<String> parsed(List<ApiEvent> events) => [
      for (final e in events)
        switch (e) {
          ApiText(:final text) => 'T:$text',
          ApiToolCallStart(:final index, :final name) => 'C$index:$name',
          ApiToolCallArguments(:final index, :final delta) => 'A$index:$delta',
          _ => '?',
        },
    ];

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
    expect(() => core.prompt(const ApiRequest(messages: [ApiMessage('critic', 'x')])), throwsA(isA<ApiError>()));
    // Tool results need a model with a tool-call format.
    final plain = FakeBackend()..dialect = ToolDialect.none;
    expect(() => ApiCore(plain).prompt(const ApiRequest(messages: [ApiMessage('tool', 'x')])), throwsA(isA<ApiError>()));
  });

  test('tool-call grammars, as the engine checks them (tests/test_grammar.cpp has the same texts)', () {
    expect(ToolDialect.hermesJson.callGrammar([weather, clock]), r'''
root ::= "\n" "{\"name\": " (t0 | t1) "}\n</tool_call>"
t0 ::= "\"get_weather\", \"arguments\": " %schema {"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"days":{"type":"integer","minimum":1,"maximum":14}},"required":["city"]}
t1 ::= "\"now\", \"arguments\": " "{}"''');
    expect(ToolDialect.qwenXml.callGrammar([weather, clock], newline: false), r'''
root ::= "<function=" (t0 | t1) "</function>\n</tool_call>"
p0-0 ::= "<parameter=city>\n" %until "\n</parameter>" "\n"
p0-1 ::= "<parameter=unit>\n" %string {"enum":["celsius","fahrenheit"]} "\n</parameter>\n"
p0-2 ::= "<parameter=days>\n" %schema {"type":"integer","minimum":1,"maximum":14} "\n</parameter>\n"
t0 ::= "get_weather>\n" p0-0 p0-1? p0-2?
t1 ::= "now>\n" ""''');
    // A tool whose schema the engine refuses takes arguments of any shape.
    expect(ToolDialect.hermesJson.callGrammar([weather], loose: {'get_weather'}),
        contains('"\\"get_weather\\", \\"arguments\\": " %schema {"type": "object"}'));
    expect(ToolDialect.qwenXml.callGrammar([weather], loose: {'get_weather'}),
        contains('("<parameter=" [^>\\n]+ ">\\n" %until "\\n</parameter>" "\\n")*'));
  });

  test('the structure of a reply with tools: calls are spans; reasoning first; a required call is forced', () {
    final free = toolStructure(grammar: 'G', grammarAfterNewline: 'G2', mustCall: false, thinking: false);
    expect(free('').grammar, isNull);
    expect(free('Let me check.').text, '');
    expect(free('Let me check.\n<tool_call>').grammar, 'G');
    expect(free('<tool_call>\n').grammar, 'G2');
    expect(free('<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>').grammar, isNull);
    expect(free('<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>\n<tool_call>').grammar, 'G');

    final thinking = toolStructure(grammar: 'G', grammarAfterNewline: 'G2', mustCall: true, thinking: true);
    expect(thinking('I could use <tool_call>').grammar, isNull); // a thought, not a call
    final afterThought = thinking('Weather, then.</think>');
    expect((afterThought.text, afterThought.grammar), ('\n\n<tool_call>', 'G'));
    expect(thinking('Ok.</think>\n\nIt is').grammar, isNull); // the answer started: too late to force

    final required = toolStructure(grammar: 'G', grammarAfterNewline: 'G2', mustCall: true, thinking: false);
    final first = required('');
    expect((first.text, first.grammar), ('<tool_call>', 'G'));
  });

  test('replies split into text and calls as they stream, in both formats', () {
    final json = ToolCallParser(ToolDialect.hermesJson, [weather, clock]);
    final events = [
      for (final p in [
        'Let me look.\n',
        '<tool_',
        'call>\n{"name": "get_wea',
        'ther", "arguments": {"city": "Par',
        'is", "days": 3}}\n</tool_call>\n<tool_call>\n{"name": "now", "arguments": {}}\n</tool_call>',
      ])
        ...json.feed(p),
      ...json.flush(),
    ];
    final calls = parsed(events);
    expect(calls.first, 'T:Let me look.');
    expect(calls[1], 'C0:get_weather');
    final args0 = [for (final c in calls) if (c.startsWith('A0:')) c.substring(3)].join();
    expect(jsonDecode(args0), {'city': 'Paris', 'days': 3});
    expect(calls.contains('C1:now'), isTrue);
    expect([for (final c in calls) if (c.startsWith('A1:')) c.substring(3)].join(), '{}');
    expect(json.completed, 2);

    final xml = ToolCallParser(ToolDialect.qwenXml, [weather]);
    final xmlEvents = [
      for (final p in [
        '<tool_call>\n<function=get_weather>\n<parameter=city>\nSão Paulo\n</par',
        'ameter>\n<parameter=days>\n7\n</parameter>\n</function>\n</tool_call>',
      ])
        ...xml.feed(p),
    ];
    final xmlCalls = parsed(xmlEvents);
    expect(xmlCalls.first, 'C0:get_weather');
    final args = [for (final c in xmlCalls) if (c.startsWith('A0:')) c.substring(3)].join();
    expect(jsonDecode(args), {'city': 'São Paulo', 'days': 7});

    // Plain text passes through; a tag's start is held back until the next piece tells.
    final plain = ToolCallParser(ToolDialect.hermesJson, const []);
    expect(parsed([...plain.feed('a <tool'), ...plain.feed('box> b'), ...plain.flush()]), ['T:a', 'T: <toolbox> b']);
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

    test('models lists the model requests load, also while it is released', () async {
      backend.loaded = false;
      final (_, openai) = await call('GET', '/v1/models');
      expect(jsonDecode(openai)['data'][0]['id'], 'test-model.gguf');
      final (_, anthropic) =
          await call('GET', '/v1/models', headers: const {'x-api-key': token, 'anthropic-version': '2023-06-01'});
      expect(jsonDecode(anthropic)['data'][0]['id'], 'test-model.gguf');
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

    test('OpenAI: tool calls, constrained, in the model\'s format; streamed as deltas; history replayed', () async {
      backend.pieces = [
        'Checking.',
        '<tool_call>',
        '\n{"name": "get_weather", "arguments": {"city": "Paris", ',
        '"days": 2}}\n</tool_call>',
      ];
      final request = {
        'messages': [
          {'role': 'system', 'content': 'Be brief.'},
          {'role': 'user', 'content': 'Weather in Paris?'},
        ],
        'tools': openAiTools([weather, clock]),
      };
      final (status, body) = await call('POST', '/v1/chat/completions', body: request);
      expect(status, 200);
      final o = jsonDecode(body);
      final message = o['choices'][0]['message'] as Map;
      expect(message['content'], 'Checking.');
      final call0 = (message['tool_calls'] as List).single as Map;
      expect(call0['type'], 'function');
      expect((call0['id'] as String).startsWith('call_'), isTrue);
      expect(call0['function']['name'], 'get_weather');
      expect(jsonDecode(call0['function']['arguments'] as String), {'city': 'Paris', 'days': 2});
      expect(o['choices'][0]['finish_reason'], 'tool_calls');
      // The tools are described in the system block, as the template does; the call is a constrained span.
      expect(backend.prompts.last, startsWith('<|im_start|>system\nBe brief.\n\n# Tools\n'));
      expect(backend.prompts.last, contains('"name": "get_weather"'));
      expect(backend.spans.last, ToolDialect.hermesJson.callGrammar([weather, clock]));

      // Streaming: the call's start, then its arguments in pieces.
      final (_, sse) = await call('POST', '/v1/chat/completions', body: {...request, 'stream': true});
      final data = [for (final l in sse.split('\n')) if (l.startsWith('data: ')) l.substring(6)];
      final deltas = [
        for (final d in data.take(data.length - 1))
          for (final c in (jsonDecode(d)['choices'] as List)) c as Map,
      ];
      final callDeltas = [
        for (final c in deltas)
          if ((c['delta'] as Map)['tool_calls'] != null) ((c['delta'] as Map)['tool_calls'] as List).single as Map,
      ];
      expect(callDeltas.first['function']['name'], 'get_weather');
      expect(callDeltas.first['index'], 0);
      expect(jsonDecode(callDeltas.map((c) => c['function']['arguments']).join()), {'city': 'Paris', 'days': 2});
      expect(callDeltas.length, greaterThan(2)); // the arguments came in more than one piece
      expect(deltas.last['finish_reason'], 'tool_calls');

      // The history: the call and its result, in the model's own format.
      backend.pieces = ['Sunny.'];
      final (_, after) = await call('POST', '/v1/chat/completions', body: {
        'messages': [
          {'role': 'user', 'content': 'Weather in Paris?'},
          {
            'role': 'assistant',
            'content': null,
            'tool_calls': [
              {
                'id': 'call_1',
                'type': 'function',
                'function': {'name': 'get_weather', 'arguments': '{"city":"Paris"}'},
              },
            ],
          },
          {'role': 'tool', 'tool_call_id': 'call_1', 'content': '21 C, sunny'},
        ],
        'tools': openAiTools([weather]),
      });
      expect(jsonDecode(after)['choices'][0]['message'], {'role': 'assistant', 'content': 'Sunny.'});
      expect(
          backend.prompts.last,
          endsWith('<|im_start|>user\nWeather in Paris?<|im_end|>\n'
              '<|im_start|>assistant\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><|im_end|>\n'
              '<|im_start|>user\n<tool_response>\n21 C, sunny\n</tool_response><|im_end|>\n'
              '<|im_start|>assistant\n<think>\n\n</think>\n\n'));
    });

    test('OpenAI: tool_choice, parallel_tool_calls, response_format and what is refused', () async {
      Map<String, Object?> ask(Map<String, Object?> extra) => {
            'messages': [
              {'role': 'user', 'content': 'Hi'},
            ],
            ...extra,
          };
      // A named tool: the call is forced at once, with only that tool in its grammar.
      backend.pieces = ['\n{"name": "now", "arguments": {}}\n</tool_call>'];
      final (_, named) = await call('POST', '/v1/chat/completions', body: ask({
        'tools': openAiTools([weather, clock]),
        'tool_choice': {
          'type': 'function',
          'function': {'name': 'now'},
        },
      }));
      expect(backend.forced.last, '<tool_call>');
      expect(backend.spans.last, ToolDialect.hermesJson.callGrammar([clock]));
      expect(((jsonDecode(named)['choices'][0]['message']['tool_calls'] as List).single as Map)['function']['name'], 'now');

      // none: the tools are not offered at all.
      backend.pieces = ['Hello'];
      await call('POST', '/v1/chat/completions', body: ask({'tools': openAiTools([weather]), 'tool_choice': 'none'}));
      expect(backend.prompts.last, isNot(contains('# Tools')));

      // parallel_tool_calls false: the reply ends after its first call.
      backend.pieces = [
        '<tool_call>',
        '\n{"name": "now", "arguments": {}}\n</tool_call>',
        '<tool_call>',
        '\n{"name": "now", "arguments": {}}\n</tool_call>',
      ];
      final stopsBefore = backend.stops;
      final (_, single) =
          await call('POST', '/v1/chat/completions', body: ask({'tools': openAiTools([clock]), 'parallel_tool_calls': false}));
      expect((jsonDecode(single)['choices'][0]['message']['tool_calls'] as List).length, 1);
      expect(backend.stops, stopsBefore + 1);

      // A tool whose schema the engine cannot enforce still works, with arguments of any shape.
      const strict = ToolSpec('lookup', 'Lookup', {
        'type': 'object',
        'properties': {
          'x': {'type': 'string', 'pattern': '^a'},
        },
      });
      backend.pieces = ['Hi'];
      final (loose, _) = await call('POST', '/v1/chat/completions', body: ask({'tools': openAiTools([strict])}));
      expect(loose, 200);

      // response_format: the whole answer is constrained by the schema.
      backend.pieces = ['{"ok": true}'];
      final schema = {
        'type': 'object',
        'properties': {
          'ok': {'type': 'boolean'},
        },
        'required': ['ok'],
      };
      final (_, json) = await call('POST', '/v1/chat/completions', body: ask({
        'response_format': {
          'type': 'json_schema',
          'json_schema': {'name': 'r', 'schema': schema},
        },
      }));
      expect(jsonDecode(jsonDecode(json)['choices'][0]['message']['content'] as String), {'ok': true});
      expect(backend.grammars.last, 'root ::= %schema ${jsonEncode(schema)}');
      await call('POST', '/v1/chat/completions', body: ask({'response_format': {'type': 'json_object'}, 'reasoning_effort': 'high'}));
      expect(backend.grammars.last, 'root ::= %until "</think>" "\\n\\n" %schema {"type":"object"}');

      final (refused, why) = await call('POST', '/v1/chat/completions', body: ask({
        'response_format': {
          'type': 'json_schema',
          'json_schema': {
            'name': 'r',
            'schema': {'type': 'string', 'pattern': 'a+'},
          },
        },
      }));
      expect(refused, 400);
      expect(jsonDecode(why)['error']['message'], contains('pattern'));
      final (both, _) = await call('POST', '/v1/chat/completions',
          body: ask({'tools': openAiTools([clock]), 'response_format': {'type': 'json_object'}}));
      expect(both, 400);
      final (unknown, _) = await call('POST', '/v1/chat/completions', body: ask({
        'tools': openAiTools([clock]),
        'tool_choice': {
          'type': 'function',
          'function': {'name': 'missing'},
        },
      }));
      expect(unknown, 400);
      backend.dialect = ToolDialect.none;
      final (noFormat, message) = await call('POST', '/v1/chat/completions', body: ask({'tools': openAiTools([clock])}));
      expect(noFormat, 400);
      expect(jsonDecode(message)['error']['message'], contains('tool-call format'));
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

    test('Anthropic: tool_use blocks, streamed input_json_delta, tool_result history, XML calls', () async {
      final anthropic = {'x-api-key': token, 'anthropic-version': '2023-06-01'};
      final tools = [
        for (final t in [weather, clock]) {'name': t.name, 'description': t.description, 'input_schema': t.parameters},
      ];
      backend
        ..dialect = ToolDialect.qwenXml
        ..pieces = [
          'On it.',
          '<tool_call>',
          '\n<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n',
          '<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>',
        ];
      final request = {
        'max_tokens': 100,
        'tools': tools,
        'messages': [
          {'role': 'user', 'content': 'Weather in Rome?'},
        ],
      };
      final (status, body) = await call('POST', '/v1/messages', headers: anthropic, body: request);
      expect(status, 200);
      final o = jsonDecode(body);
      expect(o['stop_reason'], 'tool_use');
      final content = o['content'] as List;
      expect(content[0], {'type': 'text', 'text': 'On it.'});
      expect(content[1]['type'], 'tool_use');
      expect((content[1]['id'] as String).startsWith('toolu_'), isTrue);
      expect(content[1]['name'], 'get_weather');
      expect(content[1]['input'], {'city': 'Rome', 'unit': 'celsius'});
      expect(backend.spans.last, ToolDialect.qwenXml.callGrammar([weather, clock]));
      expect(backend.prompts.last, startsWith('<|im_start|>system\n# Tools\n'));

      final (_, sse) = await call('POST', '/v1/messages', headers: anthropic, body: {...request, 'stream': true});
      final events = [
        for (final l in sse.split('\n'))
          if (l.startsWith('data: ')) jsonDecode(l.substring(6)) as Map,
      ];
      final start = events.firstWhere((e) => e['type'] == 'content_block_start' && e['content_block']['type'] == 'tool_use');
      expect(start['content_block']['name'], 'get_weather');
      final partial = [
        for (final e in events)
          if (e['type'] == 'content_block_delta' && e['delta']['type'] == 'input_json_delta') e['delta']['partial_json'],
      ].join();
      expect(jsonDecode(partial), {'city': 'Rome', 'unit': 'celsius'});
      expect(events.firstWhere((e) => e['type'] == 'message_delta')['delta']['stop_reason'], 'tool_use');

      // History: tool_use and tool_result blocks, in the model's XML format; tool_choice any forces a call.
      backend.pieces = ['\n<function=now>\n</function>\n</tool_call>'];
      await call('POST', '/v1/messages', headers: anthropic, body: {
        'max_tokens': 100,
        'tools': tools,
        'tool_choice': {'type': 'any'},
        'messages': [
          {'role': 'user', 'content': 'Weather in Rome?'},
          {
            'role': 'assistant',
            'content': [
              {'type': 'text', 'text': 'On it.'},
              {
                'type': 'tool_use',
                'id': 'toolu_1',
                'name': 'get_weather',
                'input': {'city': 'Rome', 'days': 2},
              },
            ],
          },
          {
            'role': 'user',
            'content': [
              {'type': 'tool_result', 'tool_use_id': 'toolu_1', 'content': '19 C'},
              {'type': 'text', 'text': 'And the time?'},
            ],
          },
        ],
      });
      expect(
          backend.prompts.last,
          contains('<|im_start|>assistant\nOn it.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nRome\n'
              '</parameter>\n<parameter=days>\n2\n</parameter>\n</function>\n</tool_call><|im_end|>\n'
              '<|im_start|>user\n<tool_response>\n19 C\n</tool_response><|im_end|>\n'
              '<|im_start|>user\nAnd the time?<|im_end|>\n'));
      expect(backend.forced.last, '<tool_call>');
      final (server, _) = await call('POST', '/v1/messages', headers: anthropic, body: {
        ...request,
        'tools': [
          {'type': 'web_search_20250305', 'name': 'web_search'},
        ],
      });
      expect(server, 400);
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
