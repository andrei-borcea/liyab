// Typing ahead: when a draft's words and the whole message are prepared.
import 'package:flutter_test/flutter_test.dart';
import 'package:liyab/state/typing_ahead.dart';

void main() {
  // testWidgets runs in a fake-time zone: pump() advances the pause timer.
  late List<String> calls;
  TypingAhead create() {
    calls = [];
    return TypingAhead((text, {required message}) => calls.add('${message ? 'message' : 'words'}:$text'));
  }

  // Types `text` one character at a time after the first `from` ones.
  void type(TypingAhead t, String text, {int from = 0}) {
    for (var i = from + 1; i <= text.length; ++i) {
      t.changed(text.substring(0, i));
    }
  }

  testWidgets('typed text: words as each one ends, the message after a pause', (tester) async {
    final t = create();
    type(t, 'hello wor');
    expect(calls, ['words:hello ']);
    await tester.pump(const Duration(milliseconds: 599));
    expect(calls, ['words:hello ']);
    await tester.pump(const Duration(milliseconds: 1));
    expect(calls, ['words:hello ', 'message:hello wor']);
  });

  testWidgets('a paste prepares the whole message at once, and not again at the pause', (tester) async {
    final t = create();
    const pasted = 'Summarize this article about the new bridge';
    t.changed(pasted);
    expect(calls, ['message:$pasted']);
    await tester.pump(const Duration(seconds: 1));
    expect(calls, ['message:$pasted']);
  });

  testWidgets('a paste into a draft counts what it added, wherever it went', (tester) async {
    final t = create();
    type(t, 'Translate: ');
    expect(calls, ['words:Translate: ']);
    calls.clear();
    const draft = 'Translate: le pont est ouvert depuis hier';
    t.changed(draft); // pasted after the typed words
    expect(calls, ['message:$draft']);
    calls.clear();
    const inside = 'Translate to English: le pont est ouvert depuis hier';
    t.changed(inside); // " to English" is 11 characters: typed or suggested, not a paste
    expect(calls, isEmpty);
    await tester.pump(const Duration(milliseconds: 600));
    expect(calls, ['message:$inside']);
  });

  testWidgets('typing on after a paste waits for the pause again', (tester) async {
    final t = create();
    const pasted = 'Summarize this article about the new bridge';
    t.changed(pasted);
    calls.clear();
    type(t, '$pasted please', from: pasted.length);
    expect(calls, ['words:$pasted ']); // the space ended a word
    await tester.pump(const Duration(milliseconds: 600));
    expect(calls, ['words:$pasted ', 'message:$pasted please']);
  });

  testWidgets('cursor moves do not restart the pause; deleting is never a paste', (tester) async {
    final t = create();
    type(t, 'what time');
    calls.clear();
    await tester.pump(const Duration(milliseconds: 400));
    t.changed('what time'); // the same text: a cursor move
    await tester.pump(const Duration(milliseconds: 200));
    expect(calls, ['message:what time']);
    calls.clear();
    t.changed('wh'); // a selection deleted
    t.changed('');
    expect(calls, isEmpty);
    await tester.pump(const Duration(seconds: 1));
    expect(calls, isEmpty); // blank text prepares nothing
  });

  testWidgets('after a send the next paste is seen as one, and no pause is pending', (tester) async {
    final t = create();
    const first = 'Summarize this article about the new bridge';
    type(t, first);
    t.cancel(); // sent; the composer may be cleared later or not at all
    await tester.pump(const Duration(seconds: 1));
    expect(calls.where((c) => c.startsWith('message:')), isEmpty);
    calls.clear();
    const second = 'And this one about the tunnel';
    t.changed(second);
    expect(calls, ['message:$second']);
  });

  test('a paste is more than a keystroke or a word suggestion adds', () {
    expect(TypingAhead.pasteChars, greaterThan('hello '.length));
  });
}
