import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/chat/chat_template.dart';

// The pieces Engine::chat_format finds in a model's template (tests/test_text.cpp checks them against the real
// templates): prompts built from them must equal the built-in format where they are the same model family.
void main() {
  // Qwen3.6's template with preserve_thinking, as the engine splits it (thinking switch after "<think>").
  const qwen36 = ChatTemplate(
    label: 'model template',
    systemPrefix: '<|im_start|>system\n',
    systemSuffix: '<|im_end|>\n',
    firstUserPrefix: '<|im_start|>user\n',
    noSystemHead: '<|im_start|>user\n',
    userPrefix: '<|im_start|>user\n',
    userSuffix: '<|im_end|>\n',
    opening: '<|im_start|>assistant\n',
    assistantEnd: '<|im_end|>\n',
    thinkingOn: '<think>\n',
    thinkingOff: '<think>\n\n</think>\n\n',
    isModelTemplate: true,
  );

  test('a ChatML model template builds the same bytes as the built-in ChatML (saved contexts stay valid)', () {
    const history = [Turn(user: 'Hi\n\nMonday 14 October 2026, 09:00 (+02:00)', exact: '<think>\n\n</think>\n\nHello!')];
    for (final thinking in Thinking.values) {
      final system = qwen36.system('You are Liyab.');
      expect(system, ChatTemplate.chatml.system('You are Liyab.'));
      expect(qwen36.build(system, history, 'And now?', thinking),
          ChatTemplate.chatml.build(system, history, 'And now?', thinking));
    }
    expect(qwen36.isChatML, isTrue);
    expect(ChatTemplate.llama3.isChatML, isFalse);
  });

  test('a template whose system text opens the first user turn (Gemma) keeps every prompt a prefix of the next', () {
    const gemma = ChatTemplate(
      label: 'model template',
      systemPrefix: '<bos><start_of_turn>user\n',
      systemSuffix: '\n\n',
      firstUserPrefix: '',
      noSystemHead: '<bos><start_of_turn>user\n',
      userPrefix: '<start_of_turn>user\n',
      userSuffix: '<end_of_turn>\n',
      opening: '<start_of_turn>model\n',
      assistantEnd: '<end_of_turn>\n',
      isModelTemplate: true,
    );
    final system = gemma.system('Be brief.');
    final first = gemma.build(system, const [], 'Hi', Thinking.none);
    expect(first, '<bos><start_of_turn>user\nBe brief.\n\nHi<end_of_turn>\n<start_of_turn>model\n');
    final second = gemma.build(system, const [Turn(user: 'Hi', exact: 'Hello.')], 'More?', Thinking.none);
    expect(second.startsWith('${first}Hello.'), isTrue);
    expect(second.substring(first.length),
        'Hello.<end_of_turn>\n<start_of_turn>user\nMore?<end_of_turn>\n<start_of_turn>model\n');
    // Without a system block the first turn opens with the head.
    expect(gemma.build('', const [], 'Hi', Thinking.none),
        '<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n');
  });
}
