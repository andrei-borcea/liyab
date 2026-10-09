// Prompt formats for chat models, picked from the model's vocabulary: a
// special token that encodes as one id identifies the family (ChatML for Qwen,
// Llama 3 headers), otherwise the Zephyr / TinyLlama format is used. Liyab maps
// special-token texts to their ids when tokenizing.
//
// Thinking (ChatML models with <think> tokens, as the Qwen3 / Qwen3.5
// templates do): on opens the reasoning block, off closes it empty so the model
// answers directly. Past turns are replayed exactly as generated, reasoning
// included: the engine then sees each new prompt continue the text it already
// processed and only works on the new message.
import '../engine/engine_service.dart';

enum Thinking { none, on, off }

/// One finished turn: what the user wrote and the reply exactly as generated
/// (assistant prefix included), which the next prompt replays.
class Turn {
  const Turn({required this.user, required this.exact});
  final String user;
  final String exact;
}

enum ChatTemplate {
  chatml('ChatML'),
  llama3('Llama 3'),
  zephyr('Zephyr');

  const ChatTemplate(this.label);
  final String label;

  /// The system block that starts every prompt (prefilled right after loading).
  String system(String text) => switch (this) {
        chatml => '<|im_start|>system\n$text<|im_end|>\n',
        llama3 => '${_header('system')}$text<|eot_id|>',
        zephyr => '<|system|>\n$text</s>\n',
      };

  String _turn(Turn t) => switch (this) {
        chatml => '<|im_start|>user\n${t.user}<|im_end|>\n<|im_start|>assistant\n${t.exact}<|im_end|>\n',
        llama3 => '${_header('user')}${t.user}<|eot_id|>${_header('assistant')}${t.exact}<|eot_id|>',
        zephyr => '<|user|>\n${t.user}</s>\n<|assistant|>\n${t.exact}</s>\n',
      };

  String _open(String user, Thinking thinking) => switch (this) {
        chatml => '<|im_start|>user\n$user<|im_end|>\n<|im_start|>assistant\n${assistantPrefix(thinking)}',
        llama3 => '${_header('user')}$user<|eot_id|>${_header('assistant')}',
        zephyr => '<|user|>\n$user</s>\n<|assistant|>\n',
      };

  /// What the reply starts with before the model writes (ChatML thinking control).
  String assistantPrefix(Thinking thinking) => this == chatml
      ? switch (thinking) {
          Thinking.on => '<think>\n',
          Thinking.off => '<think>\n\n</think>\n\n',
          Thinking.none => '',
        }
      : '';

  /// The full prompt for a new message after `history`, starting with
  /// `systemBlock` (system(), or a block that also lists tools).
  String build(String systemBlock, List<Turn> history, String user, Thinking thinking) {
    final p = StringBuffer(systemBlock);
    for (final t in history) {
      p.write(_turn(t));
    }
    p.write(_open(user, thinking));
    return p.toString();
  }

  static String _header(String role) => '<|start_header_id|>$role<|end_header_id|>\n\n';

  /// The template the loaded model's vocabulary implies, and whether it has a thinking mode.
  static Future<(ChatTemplate, bool)> detect(EngineService engine) async {
    if (await engine.countTokens('<|im_start|>') == 1) {
      final thinking = await engine.countTokens('<think>') == 1 && await engine.countTokens('</think>') == 1;
      return (chatml, thinking);
    }
    if (await engine.countTokens('<|start_header_id|>') == 1) return (llama3, false);
    return (zephyr, false);
  }
}
