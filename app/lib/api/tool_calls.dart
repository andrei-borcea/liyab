// Tool calls through the local API: the structure a reply follows (which
// text is forced, which spans are constrained by a call's grammar), and the
// parser that splits the streamed answer into text and calls.
//
// A call is written in the loaded model's own format (ToolDialect, from its
// chat template): `<tool_call>` is the model's choice; from there the call is
// one constrained span whose grammar (ToolDialect.callGrammar) admits only the
// format around the name of one of the request's tools and arguments valid
// against that tool's schema, up to `</tool_call>`. So a call always parses,
// and its arguments always validate.
import 'dart:convert';

import '../agent/tool_format.dart';
import '../engine/engine_service.dart';
import 'api_core.dart';

/// The structure callback of a reply that may call tools (see
/// EngineService.generate). `grammar` / `grammarAfterNewline` are the call's
/// grammar before and after the newline that follows `<tool_call>`.
/// `mustCall`: the reply starts with a call (tool_choice required or a named
/// tool), right after its reasoning when it reasons (`thinking`). Captures
/// only sendable values: it runs on the engine's isolate.
Continuation Function(String) toolStructure(
    {required String grammar, required String grammarAfterNewline, required bool mustCall, required bool thinking}) {
  return (generated) {
    // Nothing inside reasoning: tags there are thoughts, not calls.
    final close = generated.lastIndexOf('</think>');
    if (close < 0 && (thinking || generated.contains('<think>'))) return Continuation.none;
    final body = close < 0 ? generated : generated.substring(close + '</think>'.length);
    final open = body.lastIndexOf('<tool_call>');
    if (open < 0) {
      if (!mustCall || body.trim().isNotEmpty) return Continuation.none;
      return Continuation(close >= 0 && body.isEmpty ? '\n\n<tool_call>' : '<tool_call>', grammar);
    }
    final call = body.substring(open + '<tool_call>'.length);
    if (call.isEmpty) return Continuation('', grammar);
    if (call == '\n') return Continuation('', grammarAfterNewline);
    return Continuation.none; // a finished call (or one written without a span): the model goes on
  };
}

/// Splits a reply's answer into text and tool calls as it streams. Text that
/// may be the start of `<tool_call>` (and white space before it) is held back
/// until the next piece tells. JSON calls stream their arguments as written;
/// XML calls one parameter at a time, converted to JSON (raw text for string
/// parameters, as Qwen's parser reads them).
class ToolCallParser {
  ToolCallParser(this.dialect, List<ToolSpec> tools) : _tools = {for (final t in tools) t.name: t};

  final ToolDialect dialect;
  final Map<String, ToolSpec> _tools;
  String _pending = '';
  _Where _where = _Where.text;
  int _index = -1;
  int _params = 0;
  String? _name;

  /// Calls finished so far.
  int completed = 0;

  static const _open = '<tool_call>';
  static const _close = '</tool_call>';
  static final _jsonHead = RegExp(r'^\s*\{"name": "((?:[^"\\]|\\.)*)", "arguments": ');
  static final _xmlHead = RegExp(r'^\s*<function=([^>\n]+)>\n?');
  static final _xmlParam = RegExp(r'^<parameter=([^>\n]+)>\n(.*?)\n</parameter>\n?', dotAll: true);

  List<ApiEvent> feed(String text) {
    _pending += text;
    final events = <ApiEvent>[];
    for (var progress = true; progress;) {
      progress = switch (_where) {
        _Where.text => _text(events),
        _Where.head => _head(events),
        _Where.arguments => dialect == ToolDialect.qwenXml ? _xmlArguments(events) : _jsonArguments(events),
      };
    }
    return events;
  }

  /// The end of the reply: held-back text, unless it is only white space or
  /// the start of a call that never came.
  List<ApiEvent> flush() {
    final rest = _pending;
    _pending = '';
    if (_where != _Where.text || rest.trim().isEmpty || _open.startsWith(rest.trim())) return const [];
    return [ApiText(rest)];
  }

  bool _text(List<ApiEvent> events) {
    final at = _pending.indexOf(_open);
    if (at < 0) {
      // Hold back trailing white space and any start of `<tool_call>`.
      var keep = 0;
      for (var n = _open.length - 1; n > 0; --n) {
        if (_pending.endsWith(_open.substring(0, n))) {
          keep = n;
          break;
        }
      }
      final head = _pending.substring(0, _pending.length - keep);
      final trimmed = head.trimRight();
      if (trimmed.isNotEmpty) events.add(ApiText(trimmed));
      _pending = _pending.substring(trimmed.length);
      return false;
    }
    final before = _pending.substring(0, at).trimRight();
    if (before.isNotEmpty) events.add(ApiText(before));
    _pending = _pending.substring(at + _open.length);
    _where = _Where.head;
    return true;
  }

  bool _head(List<ApiEvent> events) {
    final m = (dialect == ToolDialect.qwenXml ? _xmlHead : _jsonHead).firstMatch(_pending);
    if (m == null) {
      final end = _pending.indexOf(_close);
      if (end < 0) return false;
      _pending = _pending.substring(end + _close.length); // not a call this parser can read: dropped
      _where = _Where.text;
      return true;
    }
    final name = dialect == ToolDialect.qwenXml ? m.group(1)!.trim() : jsonDecode('"${m.group(1)}"') as String;
    _name = name;
    events.add(ApiToolCallStart(++_index, name));
    _pending = _pending.substring(m.end);
    _where = _Where.arguments;
    _params = 0;
    return true;
  }

  // The arguments as written, up to the `}` that closes the call's object.
  bool _jsonArguments(List<ApiEvent> events) {
    const tail = '}\n$_close';
    final end = _pending.indexOf(_close);
    if (end < 0) {
      final ready = _pending.length - tail.length;
      if (ready > 0) {
        events.add(ApiToolCallArguments(_index, _pending.substring(0, ready)));
        _pending = _pending.substring(ready);
      }
      return false;
    }
    var args = _pending.substring(0, end).trimRight();
    if (args.endsWith('}')) args = args.substring(0, args.length - 1);
    if (args.isNotEmpty) events.add(ApiToolCallArguments(_index, args));
    _finish(end);
    return true;
  }

  bool _xmlArguments(List<ApiEvent> events) {
    final m = _xmlParam.firstMatch(_pending);
    if (m != null) {
      final key = m.group(1)!.trim();
      final props = _tools[_name]?.parameters['properties'];
      final value = xmlArgument(m.group(2)!, props is Map ? props[key] : null);
      events.add(ApiToolCallArguments(_index, '${_params++ == 0 ? '{' : ', '}${jsonEncode(key)}: ${pyJson(value)}'));
      _pending = _pending.substring(m.end);
      return true;
    }
    final end = _pending.indexOf(_close);
    if (end < 0) return false;
    events.add(ApiToolCallArguments(_index, _params == 0 ? '{}' : '}'));
    _finish(end);
    return true;
  }

  void _finish(int end) {
    _pending = _pending.substring(end + _close.length);
    _where = _Where.text;
    ++completed;
  }
}

enum _Where { text, head, arguments }

/// An XML parameter's value as JSON: the raw text for string parameters,
/// the JSON it holds for others (raw text when it holds none).
Object? xmlArgument(String raw, Object? schema) {
  if (ToolDialect.xmlValueKind(schema) != XmlValue.json) return raw;
  try {
    return jsonDecode(raw);
  } on FormatException {
    return raw;
  }
}
