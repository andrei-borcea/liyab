// How a model family describes tools and calls them, taken from its own chat
// template (GGUF tokenizer.chat_template), so prompts match what it was
// trained on:
//  - Qwen3.5 / Qwen3.6 / Qwen3-Coder: XML calls,
//      <tool_call>\n<function=NAME>\n<parameter=ARG>\nvalue\n</parameter>\n</function>\n</tool_call>
//  - Qwen2.5 / Qwen3 (Hermes): JSON calls,
//      <tool_call>\n{"name": NAME, "arguments": {...}}\n</tool_call>
// Both answer tool results in a user turn wrapped in <tool_response>.
import 'dart:convert';

import 'tools.dart';

/// A tool described by a client of the local API: its name, description and
/// the JSON Schema of its arguments (an object).
class ToolSpec {
  const ToolSpec(this.name, this.description, this.parameters);
  final String name;
  final String description;
  final Map<String, Object?> parameters;

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
  /// arguments valid against that tool's schema, then `</tool_call>`. With
  /// constrained decoding the model chooses only the tool and the values;
  /// names, keys and every separator are forced. Tools in `loose` (a schema
  /// the engine cannot enforce) take arguments of any shape: a JSON object,
  /// or parameters with any names and text values.
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
          final args = loose.contains(t.name) ? '%schema {"type": "object"}' : _jsonArguments(t.parameters);
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

  /// XML parameters: declared ones in order (required ones always), each
  /// value raw text for strings, JSON for every other type.
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
      final value = xmlValueKind(schema) == XmlValue.text
          ? free
          : '${xmlValueKind(schema) == XmlValue.raw ? '%string' : '%schema'} ${jsonEncode(schema)} "\\n</parameter>\\n"';
      rules.add('p$i-$j ::= ${grammarLiteral('<parameter=$name>\n')} $value');
      parts.add(required.contains(name) ? 'p$i-$j' : 'p$i-$j?');
    }
    return parts.isEmpty ? '""' : parts.join(' ');
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

  /// Structured output: the text a reply must continue with at this point of a
  /// tool call ('' for none), given the reply so far (`generated`, after the
  /// template's opening; `thinking` when that opening is a `<think>` block).
  /// The fixed parts of the call format are forced, and a function name as
  /// soon as one of `names` is the only one that fits, so the model samples
  /// only what it actually chooses (which tool, the arguments) and cannot
  /// misspell a name or break the format around it. Tags inside reasoning are
  /// left alone. The result already includes every continuation that follows
  /// from it (after `<tool_call>`, with a single tool, its whole opening).
  String continuation(String generated, List<String> names, {bool thinking = false}) {
    var forced = '';
    for (var step = _forced(generated, names, thinking); step.isNotEmpty; step = _forced(generated, names, thinking)) {
      forced += step;
      generated += step;
    }
    return forced;
  }

  /// continuation() as a function to hand to EngineService.generate: it
  /// captures only sendable values, so it can run on the engine's isolate.
  String Function(String generated) forcer(List<String> names, {bool thinking = false}) =>
      (generated) => continuation(generated, names, thinking: thinking);

  String _forced(String generated, List<String> names, bool thinking) {
    if (this == none || names.isEmpty) return '';
    final close = generated.lastIndexOf('</think>');
    if (close < 0 && (thinking || generated.contains('<think>'))) return '';
    final body = close < 0 ? generated : generated.substring(close + '</think>'.length);
    final open = body.lastIndexOf('<tool_call>');
    if (open < 0) return '';
    final call = body.substring(open + '<tool_call>'.length);
    if (call.contains('</tool_call>')) return '';
    final newline = call.isEmpty ? '\n' : '';
    switch (this) {
      case qwenXml:
        if (call.trim().isEmpty) return '$newline<function=';
        final end = call.trimRight();
        if (end.endsWith('</function>')) return '${end.length == call.length ? '\n' : ''}</tool_call>';
        const fn = '<function=';
        final at = call.indexOf(fn);
        if (at < 0 || call.contains('>', at + fn.length)) return '';
        return _completeName(call.substring(at + fn.length), names, '>\n');
      case hermesJson:
        if (call.trim().isEmpty) return '$newline{"name": "';
        final m = _jsonName.firstMatch(call);
        return m == null ? '' : _completeName(m.group(1)!, names, '", "arguments": ');
      case none:
        return '';
    }
  }

  /// The rest of the only name in `names` that starts with `partial`, then `after`; '' when none or several fit.
  static String _completeName(String partial, List<String> names, String after) {
    final fits = names.where((n) => n.startsWith(partial)).toList();
    return fits.length == 1 ? fits.single.substring(partial.length) + after : '';
  }

  static final _jsonName = RegExp(r'^\s*\{\s*"name"\s*:\s*"([^"]*)$');
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
