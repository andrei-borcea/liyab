import 'dart:async';

/// Prepares a draft while it is being written: the composer's text, each word
/// as it ends, and the whole message once typing pauses or right away when a
/// paste added it. Decides only *when*; what preparing means (processing the
/// draft's words, then the message as it would be sent, in the engine) is
/// `prepare`'s job (AppState._prefillDraft).
///
/// Not thread-safe: call it from the UI isolate, as text fields notify.
class TypingAhead {
  TypingAhead(this.prepare, {this.pause = const Duration(milliseconds: 600)});

  /// Prepares `text`: its whole words, and with `message` the whole message as
  /// it would be sent now. Called synchronously from [changed] or from the
  /// pause timer; it is expected to start background work and return.
  final void Function(String text, {required bool message}) prepare;

  /// How long typing must pause before the whole message is prepared: long
  /// enough not to process a message that keeps changing, short enough to be
  /// done before a typical send.
  final Duration pause;

  /// Text that grows by at least this many UTF-16 code units in one edit was
  /// not typed: a paste, dictation, or text shared into the composer. No
  /// keystroke adds that much and a word suggestion rarely does (a longer one
  /// only costs preparing the message early). Such a message is usually sent
  /// as it is, so it is prepared at once instead of after the pause.
  static const pasteChars = 12;

  Timer? _timer;
  String _text = ''; // as last seen

  /// Called on every edit of a composer's text (blank text prepares nothing).
  /// The same text again (a text field also notifies cursor moves) changes
  /// nothing: the pause runs from the last edit.
  void changed(String text) {
    if (text == _text) return;
    _timer?.cancel();
    final grew = text.length - _text.length;
    _text = text;
    if (text.trim().isEmpty) return;
    if (grew >= pasteChars) {
      prepare(text, message: true); // words and message now: nothing left for the pause to do
      return;
    }
    // A word just ended: its words at once (a message sent without a pause
    // then costs only its last word and the turn's end); the whole message
    // once typing pauses.
    if (text.endsWith(' ') || text.endsWith('\n')) prepare(text, message: false);
    _timer = Timer(pause, () => prepare(text, message: true));
  }

  /// The message was sent (or dropped): nothing more to prepare for it, and
  /// the next text starts from an empty composer.
  void cancel() {
    _timer?.cancel();
    _timer = null;
    _text = '';
  }
}
