// The app's own tool calls, constrained like the local API's: each tool's
// arguments schema and call grammar in both formats, the prompt that describes
// the tools (byte for byte as before, so saved contexts stay valid), a
// scripted model calling through the structure the app generates with, and,
// with the engine built on this host, every call grammar compiled and matched
// by the engine itself.
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

import 'package:crypto/crypto.dart';
import 'package:ffi/ffi.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/agent/tool_format.dart';
import 'package:liyab/agent/tools.dart';
import 'package:liyab/engine/engine_service.dart';
import 'package:liyab/state/app_state.dart';

/// The engine's grammar compiler and matcher (include/liyab/grammar.h), from
/// the library built on this host (`cmake --build build`; LIYAB_LIBRARY
/// names another). Null when it is not built: the tests that need it are
/// skipped, saying so.
class Engine {
  Engine._(DynamicLibrary lib)
      : _parse = lib.lookupFunction<Int32 Function(Pointer<Utf8>, Pointer<Pointer<Void>>),
            int Function(Pointer<Utf8>, Pointer<Pointer<Void>>)>('liyab_grammar_parse'),
        _destroy = lib.lookupFunction<Void Function(Pointer<Void>), void Function(Pointer<Void>)>('liyab_grammar_destroy'),
        _matches = lib.lookupFunction<Int32 Function(Pointer<Void>, Pointer<Uint8>, Size),
            int Function(Pointer<Void>, Pointer<Uint8>, int)>('liyab_grammar_matches'),
        _lastError = lib.lookupFunction<Pointer<Utf8> Function(), Pointer<Utf8> Function()>('liyab_last_error');

  final int Function(Pointer<Utf8>, Pointer<Pointer<Void>>) _parse;
  final void Function(Pointer<Void>) _destroy;
  final int Function(Pointer<Void>, Pointer<Uint8>, int) _matches;
  final Pointer<Utf8> Function() _lastError;

  static final path = Platform.environment['LIYAB_LIBRARY'] ??
      '${Directory.current.parent.path}/build/libliyab.${Platform.isMacOS ? 'dylib' : 'so'}';
  static final Engine? instance = File(path).existsSync() ? Engine._(DynamicLibrary.open(path)) : null;
  static final String? skip = instance == null ? 'the engine is not built on this host ($path)' : null;

  /// Runs `body` with `grammar` compiled; throws with the engine's reason when it does not compile.
  T _with<T>(String grammar, T Function(Pointer<Void> g) body) => using((arena) {
        final out = arena<Pointer<Void>>();
        if (_parse(grammar.toNativeUtf8(allocator: arena), out) != 0) throw StateError(_lastError().toDartString());
        try {
          return body(out.value);
        } finally {
          _destroy(out.value);
        }
      });

  /// Why `grammar` does not compile; null when it does.
  String? error(String grammar) {
    try {
      return _with(grammar, (_) => null);
    } on StateError catch (e) {
      return e.message;
    }
  }

  /// Whether `text` is a complete match of `grammar`.
  bool matches(String grammar, String text) => _with(grammar, (g) {
        return using((arena) {
          final bytes = utf8.encode(text);
          final buffer = arena<Uint8>(bytes.length + 1);
          buffer.asTypedList(bytes.length).setAll(0, bytes);
          return _matches(g, buffer, bytes.length) == 1;
        });
      });
}

final allTools = <AgentTool>[
  CalendarTool(),
  NotificationsTool(),
  MessagesTool(),
  CallsTool(),
  ContactsTool(),
  ClipboardTool(),
  EmailTool(),
];

/// The tools each platform offers (Toolbox.all).
final android = [for (final n in ['calendar_events', 'read_notifications', 'read_sms', 'recent_calls', 'find_contact', 'read_clipboard']) tool(n)];
final macos = [for (final n in ['calendar_events', 'read_email', 'read_sms', 'recent_calls', 'find_contact', 'read_clipboard']) tool(n)];

AgentTool tool(String name) => allTools.firstWhere((t) => t.name == name);

const dialects = [ToolDialect.qwenXml, ToolDialect.hermesJson];

final _localDateTime = RegExp(r'^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])(T([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?)?$');

/// Why `args` are not what `t` takes (null when they are): declared
/// parameters only, the required ones present, text values, date-times in
/// the shape run() reads as local time. Checked here on its own, not with
/// the grammar it checks.
String? invalid(AgentTool t, Map<String, Object?> args) {
  for (final k in args.keys) {
    if (!t.parameters.containsKey(k)) return 'undeclared parameter $k';
  }
  for (final r in t.required) {
    if (!args.containsKey(r)) return 'missing $r';
  }
  for (final MapEntry(:key, :value) in args.entries) {
    if (value is! String) return '$key is not text: $value';
    if (!t.dateTimes.contains(key)) continue;
    if (!_localDateTime.hasMatch(value)) return '$key is not a local date-time: $value';
    final time = DateTime.tryParse(value);
    if (time == null || time.isUtc) return '$key is not read as local time: $value';
  }
  return null;
}

/// Example arguments of every tool, in its declared order: required ones,
/// optional ones left out, every date-time shape, text that JSON escapes.
final examples = <String, List<Map<String, Object?>>>{
  'calendar_events': [
    {'start': '2026-10-09T00:00', 'end': '2026-10-10T00:00'},
    {'start': '2026-10-09', 'end': '2026-12-31T23:59:59'},
  ],
  'read_notifications': [
    {},
    {'since': '2026-10-09T08:00', 'app': 'WhatsApp'},
    {'app': 'Gmail'},
  ],
  'read_sms': [
    {'since': '2026-10-09T08:00:30'},
    {'from': '+39 333 123 4567'},
    {'since': '2026-01-31T07:05', 'from': 'Zoë "Zo" Müller'},
  ],
  'recent_calls': [
    {},
    {'since': '2026-10-08'},
  ],
  'find_contact': [
    {'name': 'Anna'},
    {'name': 'Jean-Luc Picard\\'},
  ],
  'read_clipboard': [{}],
  'read_email': [
    {'since': '2026-10-08T18:00', 'query': 'invoice'},
    {'query': 'Q3 "final"'},
  ],
};

/// A scripted model, run as the engine runs a structure callback: asked
/// before the first piece and, while no span is active, after each piece;
/// its text joins the output and its grammar opens a span, which ends with
/// the call's `</tool_call>` (where every call grammar ends). The output, and
/// each span's grammar with the text written under it.
({String output, List<(String, String)> spans, List<String> forced}) script(
    Continuation Function(String) structure, List<String> pieces) {
  var output = '';
  final spans = <(String, String)>[];
  final forced = <String>[];
  String? grammar;
  var start = 0;
  void ask() {
    if (grammar != null) return;
    final next = structure(output);
    if (next.text.isNotEmpty) forced.add(next.text);
    output += next.text;
    if (next.grammar != null) {
      grammar = next.grammar;
      start = output.length;
    }
  }

  ask();
  for (final piece in pieces) {
    output += piece;
    if (grammar != null && output.endsWith('</tool_call>')) {
      spans.add((grammar!, output.substring(start)));
      grammar = null;
    }
    ask();
  }
  return (output: output, spans: spans, forced: forced);
}

void main() {
  test('the prompt describes the tools byte for byte as before, and calls follow the schema it shows', () {
    // The system block's hash names its saved context (AppState._stateFile), and a conversation's saved context
    // starts with it: a change here would make every user's saved states useless (minutes of prefill on a MoE).
    String hash(ToolDialect d) => sha1.convert(utf8.encode(d.system('You are Liyab.', allTools))).toString();
    expect(hash(ToolDialect.qwenXml), '847b0b85fd3837d5c6f3ab4f05e845296c98beb3');
    expect(hash(ToolDialect.hermesJson), 'a2e165dd33b9121e95ab5faa6791c2ae3611ceaf');
    expect(jsonEncode(tool('calendar_events').schema),
        '{"type":"function","function":{"name":"calendar_events","description":"The user\'s calendar events between '
        'two local times.","parameters":{"type":"object","properties":{"start":{"type":"string","description":"e.g. '
        '2026-10-09T00:00"},"end":{"type":"string","description":"e.g. 2026-10-10T00:00"}},"required":["start","end"]}}}');
    for (final t in allTools) {
      final spec = t.spec;
      expect((spec.name, spec.description), (t.name, t.description));
      expect(spec.parameters, (t.schema['function'] as Map)['parameters']);
      expect(spec.values.keys.toSet(), t.dateTimes);
      expect(t.dateTimes.every(t.parameters.containsKey), isTrue);
      expect(examples[t.name], isNotEmpty, reason: '${t.name} needs examples below');
    }
  });

  test('each tool\'s call grammar: names and keys forced, values of their types, date-times in their shape', () {
    const date = r'([0-9]{4} "-" ("0" [1-9] | "1" [0-2]) "-" ("0" [1-9] | [12] [0-9] | "3" [01]) '
        r'("T" ([01] [0-9] | "2" [0-3]) ":" [0-5] [0-9] (":" [0-5] [0-9])?)?)';
    expect(localDateTime, date.substring(1, date.length - 1));
    expect(ToolDialect.qwenXml.callGrammar([tool('calendar_events').spec, tool('read_sms').spec]), '''
root ::= "\\n" "<function=" (t0 | t1) "</function>\\n</tool_call>"
p0-0 ::= "<parameter=start>\\n" $date "\\n</parameter>\\n"
p0-1 ::= "<parameter=end>\\n" $date "\\n</parameter>\\n"
t0 ::= "calendar_events>\\n" p0-0 p0-1
p1-0 ::= "<parameter=since>\\n" $date "\\n</parameter>\\n"
p1-1 ::= "<parameter=from>\\n" %until "\\n</parameter>" "\\n"
t1 ::= "read_sms>\\n" p1-0? p1-1?''');
    // JSON: the object is written out where a value has a grammar; the others keep their schemas.
    expect(ToolDialect.hermesJson.callGrammar([tool('calendar_events').spec, tool('read_sms').spec], newline: false), '''
root ::= "{\\"name\\": " (t0 | t1) "}\\n</tool_call>"
p0-0 ::= "\\"start\\": " "\\"" $date "\\""
p0-1 ::= "\\"end\\": " "\\"" $date "\\""
t0 ::= "\\"calendar_events\\", \\"arguments\\": " "{" p0-0 ", " p0-1 "}"
p1-0 ::= "\\"since\\": " "\\"" $date "\\""
p1-1 ::= "\\"from\\": " %schema {"type":"string","description":"name or number"}
t1 ::= "\\"read_sms\\", \\"arguments\\": " "{" (p1-0 (", " p1-1)? | p1-1)? "}"''');
    expect(ToolDialect.hermesJson.callGrammar([tool('recent_calls').spec, tool('find_contact').spec, tool('read_clipboard').spec]),
        endsWith('t0 ::= "\\"recent_calls\\", \\"arguments\\": " "{" (p0-0)? "}"\n'
            't1 ::= "\\"find_contact\\", \\"arguments\\": " %schema {"type":"object","properties":{"name":{"type":"string"}},'
            '"required":["name"]}\n'
            't2 ::= "\\"read_clipboard\\", \\"arguments\\": " "{}"'));
    // An optional value before a required one, as a value grammar may come with any schema.
    const mixed = ToolSpec('m', '', {
      'type': 'object',
      'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}, 'c': {'type': 'integer'}, 'd': {'type': 'string'}},
      'required': ['b', 'd'],
    }, values: {'a': '"x"', 'b': '"y"'});
    expect(ToolDialect.hermesJson.callGrammar([mixed]),
        endsWith('t0 ::= "\\"m\\", \\"arguments\\": " "{" (p0-0 ", ")? p0-1 (", " p0-2)? ", " p0-3 "}"'));
  });

  test('every tool\'s call grammar compiles in the engine, alone and with each platform\'s tools', () {
    final engine = Engine.instance!;
    for (final d in dialects) {
      for (final tools in [android, macos, ...[for (final t in allTools) [t]]]) {
        final specs = [for (final t in tools) t.spec];
        for (final newline in [true, false]) {
          expect(engine.error(d.callGrammar(specs, newline: newline)), isNull, reason: '${d.name} ${tools.map((t) => t.name)}');
        }
      }
    }
  }, skip: Engine.skip);

  test('each tool\'s calls in both formats: the grammar admits them, the parser reads them back, they are valid', () {
    final engine = Engine.instance!;
    for (final d in dialects) {
      final grammar = d.callGrammar([for (final t in allTools) t.spec]);
      for (final t in allTools) {
        for (final args in examples[t.name]!) {
          expect(invalid(t, args), isNull);
          // The format the model's template writes past calls in (and the model was trained on).
          final text = d.callText(t.name, args);
          expect(text, startsWith('<tool_call>\n'));
          expect(engine.matches(grammar, text.substring('<tool_call>'.length)), isTrue, reason: '${d.name}: $text');
          final call = d.parse('Let me check.\n\n$text')!;
          expect(call.name, t.name);
          expect(call.arguments, args, reason: '${d.name}: $text');
        }
      }
      // Exactly this layout: no white space of the model's own, one tool's name only.
      final calendar = d.callText('calendar_events', examples['calendar_events']![0]).substring('<tool_call>'.length);
      expect(calendar, d == ToolDialect.qwenXml
          ? '\n<function=calendar_events>\n<parameter=start>\n2026-10-09T00:00\n</parameter>\n<parameter=end>\n'
              '2026-10-10T00:00\n</parameter>\n</function>\n</tool_call>'
          : '\n{"name": "calendar_events", "arguments": {"start": "2026-10-09T00:00", "end": "2026-10-10T00:00"}}\n'
              '</tool_call>');
      expect(engine.matches(grammar, calendar.replaceFirst('\n', '')), isFalse); // the newline after <tool_call>
      expect(engine.matches(d.callGrammar([for (final t in allTools) t.spec], newline: false), calendar.substring(1)), isTrue);
    }
  }, skip: Engine.skip);

  test('what a call cannot be: another tool, a parameter undeclared, missing or out of order, a time out of shape', () {
    final engine = Engine.instance!;
    final json = ToolDialect.hermesJson.callGrammar([for (final t in android) t.spec]);
    String j(String name, String args) => '\n{"name": "$name", "arguments": $args}\n</tool_call>';
    const start = '"start": "2026-10-09T00:00"', end = '"end": "2026-10-10T00:00"';
    expect(engine.matches(json, j('calendar_events', '{$start, $end}')), isTrue);
    for (final bad in [
      j('read_email', '{}'), // not one of Android's tools
      j('get_weather', '{}'),
      j('calendar_events', '{$start}'), // a required parameter missing
      j('calendar_events', '{$end, $start}'), // the declared order
      j('calendar_events', '{$start, $end, "calendar": "Work"}'), // undeclared
      j('calendar_events', '{"start": "today", $end}'), // date-times: the shape the prompt shows
      j('calendar_events', '{"start": "2026-10-09T08:00:00Z", $end}'),
      j('calendar_events', '{"start": "2026-10-09T08:00+02:00", $end}'),
      j('calendar_events', '{"start": "2026-10-09 08:00", $end}'),
      j('calendar_events', '{"start": "2026-10-09T24:00", $end}'),
      j('calendar_events', '{"start": "2026-13-01", $end}'),
      j('calendar_events', '{"start": "09/10/2026", $end}'),
      j('read_sms', '{"since": "yesterday"}'),
      j('read_sms', '{"from": 3}'), // text
      j('read_sms', '{"from": "Marco", "since": "2026-10-09"}'),
      j('read_clipboard', '{"text": ""}'),
      j('find_contact', '{}'),
      j('find_contact', '{"name":"Anna"}'), // json.dumps' layout
    ]) {
      expect(engine.matches(json, bad), isFalse, reason: bad);
    }
    final xml = ToolDialect.qwenXml.callGrammar([for (final t in android) t.spec]);
    String x(String name, Map<String, String> params) =>
        '\n<function=$name>\n${params.entries.map((e) => '<parameter=${e.key}>\n${e.value}\n</parameter>\n').join()}'
        '</function>\n</tool_call>';
    expect(engine.matches(xml, x('read_sms', {'since': '2026-10-09T08:00', 'from': 'Marco\nRossi'})), isTrue);
    for (final bad in [
      x('read_email', {}),
      x('calendar_events', {'start': '2026-10-09T00:00'}),
      x('calendar_events', {'date': '2026-10-09'}),
      x('calendar_events', {'start': 'today', 'end': '2026-10-10T00:00'}),
      x('calendar_events', {'start': ' 2026-10-09', 'end': '2026-10-10'}),
      x('read_sms', {'from': 'Marco', 'since': '2026-10-09T08:00'}),
      x('read_sms', {'since': 'this morning'}),
      x('recent_calls', {'since': '2026-10-09T08:00', 'limit': '5'}),
    ]) {
      expect(engine.matches(xml, bad), isFalse, reason: bad);
    }
  }, skip: Engine.skip);

  test('a scripted model calls through the structure the app generates with, in both formats', () async {
    final engine = Engine.instance;
    for (final d in dialects) {
      final grammar = d.callGrammar([for (final t in android) t.spec]);
      final structure = AppState.callStructure(d, android, thinking: false);
      // Nothing is forced before a call: whether to call is the model's choice.
      final none = structure('');
      expect((none.text, none.grammar), ('', null));
      final body = d == ToolDialect.qwenXml
          ? ['\n<function=', 'read_sms>\n<parameter=since>\n2026-10', '-09T08:00\n</parameter>\n', '<parameter=from>\n',
              'Marco\n</parameter>\n</function>\n</tool_call>']
          : ['\n{"name": "', 'read_sms", "arguments": {"since": "2026-10', '-09T08:00", "from": "Marco"}}\n</tool_call>'];
      final run = script(structure, ['Let me look.\n\n', '<tool_call>', ...body]);
      // One span, right after <tool_call>, with the grammar of every enabled tool.
      expect(run.forced, isEmpty);
      expect(run.spans.length, 1);
      expect(run.spans.single.$1, grammar);
      if (engine != null) expect(engine.matches(grammar, run.spans.single.$2), isTrue);
      final call = d.parse(run.output)!;
      expect(call.name, 'read_sms');
      expect(call.arguments, {'since': '2026-10-09T08:00', 'from': 'Marco'});
      expect(invalid(tool(call.name), call.arguments), isNull);
      // The newline after <tool_call> in the same piece: the span starts after it.
      final split = script(structure, ['<tool_call>\n', body.first.substring(1), ...body.skip(1)]);
      expect(split.spans.single.$1, d.callGrammar([for (final t in android) t.spec], newline: false));
      if (engine != null) expect(engine.matches(split.spans.single.$1, split.spans.single.$2), isTrue);

      // With reasoning, a call mentioned while thinking is a thought; the one after </think> is constrained.
      final thinking = AppState.callStructure(d, android, thinking: true);
      final reasoned = script(thinking, ['I could use <tool_call>', ' read_sms.', '</think>\n\n', '<tool_call>', ...body]);
      expect(reasoned.spans.length, 1);
      expect(reasoned.spans.single.$2, run.spans.single.$2);

      // A reply without a call is generated as without tools: no span, nothing forced.
      final plain = script(structure, ['Hello', ' there. A <tool', 'box> is a box.']);
      expect(plain.output, 'Hello there. A <toolbox> is a box.');
      expect(plain.spans, isEmpty);
      expect(plain.forced, isEmpty);

      // A second call in the same reply is constrained too.
      final twice = script(structure, ['<tool_call>', ...body, '\n', '<tool_call>', ...body]);
      expect(twice.spans.length, 2);

      // The structure runs on the engine's isolate: it must reach another isolate intact.
      expect(await Isolate.run(() => structure('Sure.<tool_call>').grammar), grammar);
    }
  });

  test('the step log line gives what constrained decoding cost', () {
    // AppState logs it per step; the stats are the engine's (liyab_grammar_stats).
    const stats = GenerationStats(
        promptTokens: 10,
        generatedTokens: 30,
        tokensPerSecond: 10,
        ttftMs: 1,
        cachedPrefixTokens: 0,
        thermalReroutes: 0,
        cancelled: false,
        forcedTokens: 12,
        forcedMs: 200,
        constrainedTokens: 9,
        grammarMs: 4.25);
    expect(AppState.stepCost(stats), ', 12 forced in 0.20 s (sampled: ~1.12 s), 9 constrained, grammar 4.3 ms');
    expect(
        AppState.stepCost(const GenerationStats(
            promptTokens: 1,
            generatedTokens: 5,
            tokensPerSecond: 10,
            ttftMs: 1,
            cachedPrefixTokens: 0,
            thermalReroutes: 0,
            cancelled: false)),
        '');
  });
}
