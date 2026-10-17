// How a model family describes tools and calls them, taken from its own chat
// template (GGUF tokenizer.chat_template), so prompts match what it was
// trained on:
//  - Qwen3.5 / Qwen3.6 / Qwen3-Coder: XML calls,
//      <tool_call>\n<function=NAME>\n<parameter=ARG>\nvalue\n</parameter>\n</function>\n</tool_call>
//  - Qwen2.5 / Qwen3 (Hermes): JSON calls,
//      <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>
// Both answer tool results in a user turn wrapped in <tool_response>.
//
// Calls are constrained decoding (include/liyab/grammar.h): from `<tool_call>`
// on, a call is one span of the grammar callGrammar writes, which admits only
// the names of the tools offered and arguments valid against the chosen
// tool's schema. The local API's clients (api/tool_calls.dart) and the app's
// own tools (AgentTool.spec, state/app_state.dart) take the same path.
import 'dart:convert';

import 'tools.dart';

/// A tool as a call's grammar sees it: its name, description and the JSON
/// Schema of its arguments (an object). Clients of the local API describe
/// theirs; the app's own tools give theirs through AgentTool.spec.
///
/// `values`: grammar text (include/liyab/grammar.h) for the values of some
/// string parameters, which calls follow instead of the parameter's schema,
/// for shapes JSON Schema cannot express in the engine's subset (a local
/// date-time, see localDateTime in tools.dart). Each key names a declared
/// property, every required name is a declared property, and the grammar
/// admits only characters JSON writes without escapes: it matches the value's
/// characters as they are, between the quotes of a JSON call or the tags of
/// an XML one.
class ToolSpec {
  const ToolSpec(this.name, this.description, this.parameters, {this.values = const {}});
  final String name;
  final String description;
  final Map<String, Object?> parameters;
  final Map<String, String> values;

  /// The OpenAI-style function description chat templates render.
  Map<String, Object?> get schema => {
        'type': 'function',
        'function': {'name': name, 'description': description, 'parameters': parameters},
      };
}

/// JSON as Python's json.dumps writes it (", " and ": ", non-ASCII kept), the
/// layout chat templates' `tojson` gives tool calls and the layout constrained
/// decoding writes JSON in.
String pyJson(Object? v) => switch (v) {
      null => 'null',
      bool b => '$b',
      int i => '$i',
      double d => d.isFinite && d == d.truncateToDouble() && d.abs() < 1e16 ? '${d.toInt()}.0' : jsonEncode(d),
      String s => jsonEncode(s),
      List l => '[${l.map(pyJson).join(', ')}]',
      Map m => '{${m.entries.map((e) => '${jsonEncode('${e.key}')}: ${pyJson(e.value)}').join(', ')}}',
      _ => jsonEncode('$v'),
    };

/// `text` as a literal of grammar text (include/liyab/grammar.h).
String grammarLiteral(String text) =>
    '"${text.replaceAll(r'\', r'\\').replaceAll('"', r'\"').replaceAll('\n', r'\n').replaceAll('\r', r'\r').replaceAll('\t', r'\t')}"';

/// How a parameter value is written in an XML call (ToolDialect.xmlValueKind).
enum XmlValue { text, raw, json }

enum ToolDialect {
  none,
  qwenXml,
  hermesJson;

  /// The dialect a model's chat template uses (none when it has no tool support).
  static ToolDialect of(String chatTemplate) {
    if (chatTemplate.contains('<function=')) return qwenXml;
    if (chatTemplate.contains('<tool_call>')) return hermesJson;
    return none;
  }

  /// The ChatML system block with the tool descriptions, as the template renders it.
  String system(String text, List<AgentTool> tools) {
    if (this == none || tools.isEmpty) return '<|im_start|>system\n$text<|im_end|>\n';
    return _withTools(text, tools.map((t) => jsonEncode(t.schema)).join('\n'));
  }

  /// The system block describing a local API client's tools. With no system
  /// text, the block holds only the tools, as the templates render it.
  /// Each tool is written as the templates' `tojson` writes it.
  String systemFor(String text, List<ToolSpec> tools) {
    if (this == none || tools.isEmpty) return system(text, const []);
    final full = _withTools(text, tools.map((t) => pyJson(t.schema)).join('\n'));
    if (text.isNotEmpty) return full;
    return switch (this) {
      qwenXml => full.replaceFirst('</IMPORTANT>\n\n<|im_end|>', '</IMPORTANT><|im_end|>'),
      _ => full.replaceFirst('<|im_start|>system\n\n\n# Tools', '<|im_start|>system\n# Tools'),
    };
  }

  String _withTools(String text, String list) => switch (this) {
        qwenXml => '<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n$list\n</tools>'
            '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n'
            '<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n'
            '<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n'
            '</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified '
            'format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n'
            '- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in '
            'natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer '
            'the question like normal with your current knowledge and do not tell the user about function calls\n'
            '</IMPORTANT>\n\n$text<|im_end|>\n',
        _ => '<|im_start|>system\n$text\n\n# Tools\n\nYou may call one or more functions to assist with the user query.'
            '\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n$list\n</tools>\n\n'
            'For each function call, return a json object with function name and arguments within <tool_call></tool_call> '
            'XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n',
      };

  /// A call as the model's template writes it in a past assistant turn.
  String callText(String name, Map<String, Object?> arguments) => switch (this) {
        qwenXml => '<tool_call>\n<function=$name>\n'
            '${arguments.entries.map((e) => '<parameter=${e.key}>\n${e.value is String ? e.value : pyJson(e.value)}\n</parameter>\n').join()}'
            '</function>\n</tool_call>',
        _ => '<tool_call>\n{"name": ${pyJson(name)}, "arguments": ${pyJson(arguments)}}\n</tool_call>',
      };

  /// A past assistant turn's text: what it wrote, then its calls.
  String assistantText(String content, List<String> calls) {
    if (calls.isEmpty) return content;
    return switch (this) {
      qwenXml => '${content.trim().isEmpty ? '' : '$content\n\n'}${calls.join('\n')}',
      _ => '${content.isEmpty ? '' : '$content\n'}${calls.join('\n')}',
    };
  }

  /// The user turn that answers a turn's calls, one result per call.
  static String responses(List<String> results) =>
      results.map((r) => '<tool_response>\n$r\n</tool_response>').join('\n');

  /// Grammar text (include/liyab/grammar.h) of one call right after
  /// `<tool_call>` (`newline`: the newline that follows it is still to
  /// come): the call's syntax around the name of one of `tools` and
  /// arguments valid against that tool's schema (and its value grammars,
  /// ToolSpec.values), then `</tool_call>`. With constrained decoding the
  /// model chooses only the tool and the values; names, keys and every
  /// separator are fixed (forced, or sampled under the mask where a token of
  /// the model's covers them: the engine's token healing). Tools in `loose`
  /// (a schema the engine cannot enforce) take arguments of any shape: a JSON
  /// object, or parameters with any names and text values.
  String callGrammar(List<ToolSpec> tools, {bool newline = true, Set<String> loose = const {}}) {
    final nl = newline ? r'"\n" ' : '';
    final rules = <String>[];
    final alts = <String>[];
    for (final (i, t) in tools.indexed) {
      alts.add('t$i');
      switch (this) {
        case qwenXml:
          rules.add('t$i ::= ${grammarLiteral('${t.name}>\n')} ${_xmlParameters(t, i, loose.contains(t.name), rules)}');
        default:
          final args = loose.contains(t.name)
              ? '%schema {"type": "object"}'
              : t.values.isEmpty
                  ? _jsonArguments(t.parameters)
                  : _jsonObject(t, i, rules);
          rules.add('t$i ::= ${grammarLiteral('${pyJson(t.name)}, "arguments": ')} $args');
      }
    }
    final root = switch (this) {
      qwenXml => 'root ::= $nl"<function=" (${alts.join(' | ')}) "</function>\\n</tool_call>"',
      _ => 'root ::= $nl"{\\"name\\": " (${alts.join(' | ')}) "}\\n</tool_call>"',
    };
    return [root, ...rules].join('\n');
  }

  /// The arguments of a JSON call: a value of the schema; `{}` for a tool without parameters.
  static String _jsonArguments(Map<String, Object?> parameters) {
    final props = parameters['properties'];
    final open = parameters.containsKey('additionalProperties') || parameters.containsKey('required');
    if ((props is! Map || props.isEmpty) && !open) return '"{}"';
    return '%schema ${jsonEncode({'type': 'object', ...parameters})}';
  }

  /// The arguments of a JSON call whose tool has value grammars: the object
  /// written out the way the engine writes a schema's (the declared
  /// properties in order, required ones always and optional ones when the
  /// model chooses, json.dumps' separators), so that a value can follow its
  /// grammar, in quotes; the other values follow their schemas.
  static String _jsonObject(ToolSpec t, int i, List<String> rules) {
    final props = t.parameters['properties'] as Map<Object?, Object?>? ?? const {};
    final required = {for (final r in (t.parameters['required'] as List?) ?? const []) '$r'};
    assert(required.every(props.containsKey) && t.values.keys.every(props.containsKey), 'see ToolSpec.values');
    final names = [for (final k in props.keys) '$k'];
    for (final (j, name) in names.indexed) {
      final grammar = t.values[name];
      final value = grammar == null ? '%schema ${jsonEncode(props[name])}' : '"\\"" ($grammar) "\\""';
      rules.add('p$i-$j ::= ${grammarLiteral('${jsonEncode(name)}: ')} $value');
    }
    String seq(Iterable<String> parts) => parts.where((p) => p.isNotEmpty).join(' ');
    // The properties from `j` on, after one was written: each after ", ".
    String rest(int j) =>
        seq([for (var k = j; k < names.length; ++k) required.contains(names[k]) ? '", " p$i-$k' : '(", " p$i-$k)?']);
    final first = names.indexWhere(required.contains);
    if (first >= 0) {
      // Optional properties before the first required one are each followed by ", ".
      return seq(['"{"', for (var k = 0; k < first; ++k) '(p$i-$k ", ")?', 'p$i-$first', rest(first + 1), '"}"']);
    }
    // All optional: none, or the first one written and any of those after it.
    return '"{" (${[for (var k = 0; k < names.length; ++k) seq(['p$i-$k', rest(k + 1)])].join(' | ')})? "}"';
  }

  /// XML parameters: declared ones in order (required ones always), each
  /// value its grammar when the tool gives one (ToolSpec.values), else raw
  /// text for strings and JSON for every other type.
  static String _xmlParameters(ToolSpec t, int i, bool loose, List<String> rules) {
    const free = r'%until "\n</parameter>" "\n"';
    if (loose) return '("<parameter=" [^>\\n]+ ">\\n" $free)*';
    final props = t.parameters['properties'];
    final required = [for (final r in (t.parameters['required'] as List?) ?? const []) '$r'];
    final entries = <(String, Object?)>[
      if (props is Map) for (final e in props.entries) ('${e.key}', e.value),
      for (final r in required)
        if (props is! Map || !props.containsKey(r)) (r, null),
    ];
    final parts = <String>[];
    for (final (j, (name, schema)) in entries.indexed) {
      final grammar = t.values[name];
      final value = grammar != null
          ? '($grammar) "\\n</parameter>\\n"'
          : switch (xmlValueKind(schema)) {
              XmlValue.text => free,
              XmlValue.raw => '${_rawValue(schema as Map)} "\\n</parameter>\\n"',
              XmlValue.json => '%schema ${jsonEncode(schema)} "\\n</parameter>\\n"',
            };
      rules.add('p$i-$j ::= ${grammarLiteral('<parameter=$name>\n')} $value');
      parts.add(required.contains(name) ? 'p$i-$j' : 'p$i-$j?');
    }
    return parts.isEmpty ? '""' : parts.join(' ');
  }

  /// The formats whose text the engine checks (include/liyab/grammar.h): none
  /// has a newline, so none can hold a parameter's closing tag.
  static const _checkedFormats = {'date', 'time', 'date-time', 'uuid'};

  /// A constrained string parameter's raw value (XmlValue.raw). %string writes
  /// literals (enum, const) and the formats the engine checks as the schema
  /// says. Any other raw value, under a length bound or a format the engine
  /// does not check, is free characters, which %string would let run over the
  /// closing `\n</parameter>`: the grammar would also read the call's closing
  /// tags as the value going on, and the call's span would stay open to the
  /// end of generation. Such a value takes every character but `<` instead,
  /// with the schema's length, so the first `\n</parameter>` closes it, as
  /// the parsers read it. A value too short to hold the closing tag
  /// (maxLength under its 13 characters) cannot run over it, and keeps
  /// %string. Bounds the engine refuses (above 10000, a maximum below the
  /// minimum) are refused here too, so the tool takes arguments of any shape,
  /// as when %string refused them.
  static String _rawValue(Map<Object?, Object?> schema) {
    final string = '%string ${jsonEncode(schema)}';
    final format = schema['format'];
    if (schema['enum'] is List || schema.containsKey('const') || _checkedFormats.contains(format)) return string;
    int? count(Object? v) => v is num && v >= 0 && v == v.roundToDouble() && v <= 1e9 ? v.toInt() : null;
    final min = schema.containsKey('minLength') ? count(schema['minLength']) : 0;
    final max = schema.containsKey('maxLength') ? count(schema['maxLength']) : null;
    if (min == null || (schema.containsKey('maxLength') && max == null)) return string; // the engine says why
    if (max != null && max < '\n</parameter>'.length) return string;
    return max == null
        ? (min == 0 ? '[^<]*' : '[^<]{$min,}')
        : (min == max ? '[^<]{$min}' : '[^<]{$min,$max}');
  }

  /// How an XML parameter's value is written: free text (a string without
  /// constraints), raw text bounded by its schema (a constrained string), or
  /// JSON (every other type), as Qwen's parser reads them.
  static XmlValue xmlValueKind(Object? schema) {
    if (schema is! Map) return XmlValue.text;
    final type = schema['type'];
    final values = [if (schema['enum'] is List) ...schema['enum'] as List, if (schema.containsKey('const')) schema['const']];
    if (values.isNotEmpty) {
      return values.every((v) => v is String) && (type == null || type == 'string') ? XmlValue.raw : XmlValue.json;
    }
    if (type != 'string') return type == null && schema.isEmpty ? XmlValue.text : XmlValue.json;
    const constraints = {'format', 'minLength', 'maxLength'};
    return schema.keys.any(constraints.contains) ? XmlValue.raw : XmlValue.text;
  }

  /// What follows an assistant segment that called a tool: the end of that
  /// turn, the result in a user turn, and the opening of the next assistant turn.
  static String response(String result) =>
      '<|im_end|>\n<|im_start|>user\n<tool_response>\n$result\n</tool_response><|im_end|>\n<|im_start|>assistant\n';

  static final _xmlCall = RegExp(r'<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>', dotAll: true);
  static final _xmlParam = RegExp(r'<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>', dotAll: true);
  static final _jsonCall = RegExp(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', dotAll: true);

  /// The first complete tool call in `text` (after any reasoning), or null.
  ToolCall? parse(String text) {
    final body = text.contains('</think>') ? text.substring(text.lastIndexOf('</think>')) : text;
    switch (this) {
      case qwenXml:
        final m = _xmlCall.firstMatch(body);
        if (m == null) return null;
        return ToolCall(m.group(1)!.trim(), {
          for (final p in _xmlParam.allMatches(m.group(2)!)) p.group(1)!.trim(): p.group(2)!.trim(),
        });
      case hermesJson:
        final m = _jsonCall.firstMatch(body);
        if (m == null) return null;
        try {
          final o = jsonDecode(m.group(1)!) as Map<String, Object?>;
          final args = o['arguments'];
          return ToolCall('${o['name']}',
              args is Map<String, Object?> ? args : (args is String ? jsonDecode(args) as Map<String, Object?> : {}));
        } on FormatException {
          return null;
        }
      case none:
        return null;
    }
  }
}
