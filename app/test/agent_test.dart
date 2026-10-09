// Tool calling: the dialects' parsing, and replies with tool steps replayed exactly.
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/agent/tool_format.dart';
import 'package:liyab/agent/tools.dart';
import 'package:liyab/chat/chat_template.dart';
import 'package:liyab/state/app_state.dart';

void main() {
  test('the dialect comes from the chat template', () {
    expect(ToolDialect.of('... <tool_call>\n<function=example_function_name> ...'), ToolDialect.qwenXml);
    expect(ToolDialect.of('... <tool_call>\n{"name": <function-name> ...'), ToolDialect.hermesJson);
    expect(ToolDialect.of('{% for message in messages %}'), ToolDialect.none);
  });

  test('Qwen3.5/3.6 XML calls parse, after any reasoning', () {
    const out = 'Let me check.\n\n<tool_call>\n<function=calendar_events>\n<parameter=start>\n2026-10-09T00:00\n'
        '</parameter>\n<parameter=end>\n2026-10-10T00:00\n</parameter>\n</function>\n</tool_call>';
    final call = ToolDialect.qwenXml.parse(out)!;
    expect(call.name, 'calendar_events');
    expect(call.arguments, {'start': '2026-10-09T00:00', 'end': '2026-10-10T00:00'});
    // A call mentioned inside the reasoning is not a call.
    expect(ToolDialect.qwenXml.parse('<tool_call>\n<function=x>\n</function>\n</tool_call></think>\n\nNo.'), isNull);
  });

  test('Qwen3 JSON calls parse', () {
    const out = '<tool_call>\n{"name": "read_sms", "arguments": {"from": "Bank"}}\n</tool_call>';
    final call = ToolDialect.hermesJson.parse(out)!;
    expect(call.name, 'read_sms');
    expect(call.arguments, {'from': 'Bank'});
    expect(ToolDialect.hermesJson.parse('<tool_call>\n{broken\n</tool_call>'), isNull);
  });

  test('a reply with a tool step replays exactly and hides the call from the answer', () {
    const t = ChatTemplate.chatml;
    final prefix = t.assistantPrefix(Thinking.off);
    final m = ChatMessage('Today?', prefix, promptUser: 'Today?\n\n[Now: …]')
      ..raw = '<tool_call>\n<function=calendar_events>\n</function>\n</tool_call>';
    expect(m.answer, isEmpty);
    m
      ..done = m.exact + ToolDialect.response('{"events": []}')
      ..prefix = prefix
      ..raw = 'Nothing today.';
    expect(m.answer, 'Nothing today.');
    final next = t.build(t.system('sys'), [Turn(user: m.promptUser, exact: m.exact)], 'Tomorrow?', Thinking.off);
    expect(next, contains('<tool_response>\n{"events": []}\n</tool_response><|im_end|>\n<|im_start|>assistant\n$prefix'
        'Nothing today.<|im_end|>'));
  });

  test('the time line names the day, date, time and offset', () {
    expect(AppState.nowLine(DateTime(2026, 10, 9, 13, 5)), startsWith('[Now: Friday 9 October 2026, 13:05, UTC'));
  });

  test('a saved message comes back as the model saw it', () {
    final m = ChatMessage('What do I have today?', '<think>\n\n</think>\n\n', promptUser: 'What do I have today?\n\n[Now: …]')
      ..done = 'call<tool_response>{}</tool_response>'
      ..raw = 'Nothing today.'
      ..streaming = false;
    m.steps.add(const ToolStep('Read your calendar', 'No events'));
    final back = ChatMessage.fromJson(jsonDecode(jsonEncode(m.toJson())) as Map<String, Object?>);
    expect(back.exact, m.exact);
    expect(back.promptUser, m.promptUser);
    expect(back.answer, 'Nothing today.');
    expect(back.steps.single.summary, 'No events');
    expect(back.streaming, isFalse);
  });
}
