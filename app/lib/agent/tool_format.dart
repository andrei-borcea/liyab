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
    final list = tools.map((t) => jsonEncode(t.schema)).join('\n');
    return switch (this) {
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
