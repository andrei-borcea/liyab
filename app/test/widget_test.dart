// Chat prompts: the parts the engine's context reuse depends on.
import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/chat/chat_template.dart';
import 'package:liyab/state/app_state.dart';

void main() {
  test('a new prompt continues the previous prompt plus the exact reply', () {
    const t = ChatTemplate.chatml;
    final first = t.build('sys', const [], 'hi', Thinking.off);
    final reply = ChatMessage('hi', t.assistantPrefix(Thinking.off))..raw = 'Hello!';
    final second = t.build('sys', [Turn(user: 'hi', exact: reply.exact)], 'more', Thinking.off);
    expect(second.startsWith(first + reply.raw), isTrue);
  });

  test('reasoning and answer are split at </think>', () {
    final m = ChatMessage('q', ChatTemplate.chatml.assistantPrefix(Thinking.on))..raw = 'let me see</think>\n\nIt is 4.';
    expect(m.reasoning, 'let me see');
    expect(m.answer, 'It is 4.');
    final plain = ChatMessage('q', ChatTemplate.chatml.assistantPrefix(Thinking.off))..raw = 'It is 4.';
    expect(plain.reasoning, isNull);
    expect(plain.answer, 'It is 4.');
  });
}
