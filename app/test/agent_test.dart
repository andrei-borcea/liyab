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

  test('tool calls are forced: the format, and a name once only one fits', () {
    const xml = ToolDialect.qwenXml, json = ToolDialect.hermesJson;
    const names = ['calendar_events', 'read_sms', 'read_notifications'];
    expect(xml.continuation('Let me check.\n<tool_call>', names), '\n<function=');
    expect(xml.continuation('<tool_call>\n<function=cal', names), 'endar_events>\n');
    expect(xml.continuation('<tool_call>\n<function=read_', names), ''); // two fit: the model chooses
    expect(xml.continuation('<tool_call>\n<function=read_s', names), 'ms>\n');
    expect(xml.continuation('<tool_call>\n<function=weather', names), ''); // unknown: left to the model
    expect(xml.continuation('<tool_call>\n<function=read_sms>\n<parameter=from', names), '');
    expect(xml.continuation('<tool_call>\n<function=read_sms>\n</function>', names), '\n</tool_call>');
    expect(xml.continuation('<tool_call>\n<function=read_sms>\n</function>\n', names), '</tool_call>');
    expect(xml.continuation('<tool_call>\n<function=read_sms>\n</function>\n</tool_call>', names), '');
    // With one tool the whole opening follows <tool_call>.
    expect(xml.continuation('<tool_call>', const ['read_sms']), '\n<function=read_sms>\n');
    // Inside reasoning nothing is forced; after it, calls are.
    expect(xml.continuation('<think>\nmaybe <tool_call>', names), '');
    expect(xml.continuation('maybe <tool_call>', names, thinking: true), '');
    expect(xml.continuation('hm</think>\n\n<tool_call>', names, thinking: true), '\n<function=');
    expect(xml.continuation('<tool_call>', const []), '');
    expect(json.continuation('<tool_call>', names), '\n{"name": "');
    expect(json.continuation('<tool_call>\n{"name": "read_n', names), 'otifications", "arguments": ');
    expect(json.continuation('<tool_call>\n{"name": "read_sms", "arguments": {', names), '');
    expect(ToolDialect.none.continuation('<tool_call>', names), '');
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

  test('tool results are short lines, days shown only when not today', () {
    final now = DateTime(2026, 10, 9, 13, 0);
    int ms(DateTime t) => t.millisecondsSinceEpoch;
    final events = ToolText.events([
      {'title': 'Standup', 'begin': ms(DateTime(2026, 10, 9, 9)), 'end': ms(DateTime(2026, 10, 9, 9, 30)), 'calendar': 'Work'},
      {'title': 'Dentist', 'begin': ms(DateTime(2026, 10, 10, 14)), 'end': ms(DateTime(2026, 10, 10, 15)), 'location': 'Via Roma 1'},
      {'title': 'Holiday', 'begin': ms(DateTime(2026, 10, 9)), 'end': ms(DateTime(2026, 10, 10)), 'allDay': true},
    ], DateTime(2026, 10, 9), DateTime(2026, 10, 11), now);
    expect(events.split('\n'), [
      'Calendar from 2026-10-09 00:00 to 2026-10-11 00:00, 3 events:',
      '- 09:00-09:30 Standup [Work]',
      '- 2026-10-10 14:00-15:00 Dentist @ Via Roma 1',
      '- 2026-10-09 all day Holiday',
    ]);
    expect(ToolText.events([], DateTime(2026, 10, 9), DateTime(2026, 10, 10), now), endsWith(': no events.'));
  });

  test('notifications drop reposts and keep the newest', () {
    final now = DateTime(2026, 10, 9, 13, 0);
    final raw = [
      for (var i = 0; i < 30; ++i)
        {'app': 'WhatsApp', 'title': 'Marco', 'text': 'message $i', 'time': DateTime(2026, 10, 9, 12, i).millisecondsSinceEpoch},
      {'app': 'WhatsApp', 'title': 'Marco', 'text': 'message 29', 'time': DateTime(2026, 10, 9, 12, 31).millisecondsSinceEpoch},
    ];
    final lines = ToolText.notifications(raw, DateTime(2026, 10, 9, 8), '', now, limit: 5).split('\n');
    expect(lines.first, 'Notifications since 08:00 (newest 5 of 30), newest first:');
    expect(lines[1], '- 12:31 WhatsApp: Marco - message 29');
    expect(lines.length, 6);
    expect(ToolText.notifications(raw, DateTime(2026, 10, 9, 8), 'gmail', now), endsWith('from gmail: none.'));
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
