// Prompt formats for chat models. The model's own chat template (GGUF
// tokenizer.chat_template, rendered by the engine) is used when it lays out a
// conversation so that each turn's prompt is a prefix of the next one's
// (Engine::chat_format): the engine renders probe conversations once at load
// and the pieces it finds (system block, user turn, the reply's opening and
// end) build every prompt. Otherwise a built-in format is selected when the
// vocabulary identifies ChatML or Llama 3 headers. Models without an author
// template retain the legacy Zephyr / TinyLlama fallback. A present incompatible
// template without a recognized fallback disables chat; the engine remains
// available for raw completions or pooling. Liyab maps special-token texts
// to their ids when tokenizing.
//
// Why pieces and not the whole conversation rendered each turn: the engine
// reuses the context a prompt continues, and past replies are replayed exactly
// as generated (reasoning included), so each new prompt continues the text
// already processed and only the new message is worked on. Templates that
// render earlier turns differently once the conversation goes on (Qwen3 and
// Qwen3.5 drop the reasoning of past replies) would make every turn start
// over; for those the built-in format is kept. For Qwen3.6 (with
// preserve_thinking) the template's pieces are exactly the built-in ChatML
// ones, so prompts and saved contexts are unchanged.
//
// Thinking (models with <think> tokens whose template has a thinking switch):
// on opens the reasoning block, off closes it empty so the model answers
// directly.
import '../engine/engine_service.dart';

enum Thinking { none, on, off }

/// One finished turn: what the user wrote and the reply exactly as generated
/// (assistant prefix included), which the next prompt replays.
class Turn {
  const Turn({required this.user, required this.exact});
  final String user;
  final String exact;
}

/// The pieces a prompt is made of. A conversation is
///   system(text) + firstUserPrefix   (or noSystemHead without a system block)
///   + user + userSuffix + opening + assistantPrefix(thinking) + reply + assistantEnd
///   + userPrefix + user + userSuffix + opening + ...
class ChatTemplate {
  const ChatTemplate({
    required this.label,
    required this.systemPrefix,
    required this.systemSuffix,
    required this.firstUserPrefix,
    required this.noSystemHead,
    required this.userPrefix,
    required this.userSuffix,
    required this.opening,
    required this.assistantEnd,
    this.thinkingOn = '',
    this.thinkingOff = '',
    this.isModelTemplate = false,
  });

  static const chatml = ChatTemplate(
    label: 'ChatML',
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
  );
  static const llama3 = ChatTemplate(
    label: 'Llama 3',
    systemPrefix: '<|start_header_id|>system<|end_header_id|>\n\n',
    systemSuffix: '<|eot_id|>',
    firstUserPrefix: '<|start_header_id|>user<|end_header_id|>\n\n',
    noSystemHead: '<|start_header_id|>user<|end_header_id|>\n\n',
    userPrefix: '<|start_header_id|>user<|end_header_id|>\n\n',
    userSuffix: '<|eot_id|>',
    opening: '<|start_header_id|>assistant<|end_header_id|>\n\n',
    assistantEnd: '<|eot_id|>',
  );
  static const zephyr = ChatTemplate(
    label: 'Zephyr',
    systemPrefix: '<|system|>\n',
    systemSuffix: '</s>\n',
    firstUserPrefix: '<|user|>\n',
    noSystemHead: '<|user|>\n',
    userPrefix: '<|user|>\n',
    userSuffix: '</s>\n',
    opening: '<|assistant|>\n',
    assistantEnd: '</s>\n',
  );

  final String label;
  final String systemPrefix, systemSuffix;

  /// Between the system block and the first user's text (empty when the
  /// system text opens the first user turn, as in Gemma).
  final String firstUserPrefix;

  /// Before the first user's text when there is no system block.
  final String noSystemHead;
  final String userPrefix, userSuffix;

  /// The reply's opening (the generation prompt without its thinking control).
  final String opening;
  final String assistantEnd;

  /// The thinking control after the opening (models with a thinking mode).
  final String thinkingOn, thinkingOff;

  /// Built from the model's own chat template.
  final bool isModelTemplate;

  /// Same pieces as the built-in ChatML, so the ChatML tool formats apply.
  bool get isChatML =>
      systemPrefix == chatml.systemPrefix &&
      systemSuffix == chatml.systemSuffix &&
      userPrefix == chatml.userPrefix &&
      userSuffix == chatml.userSuffix &&
      opening == chatml.opening &&
      assistantEnd == chatml.assistantEnd;

  /// The system block that starts every prompt (prefilled right after loading).
  String system(String text) => '$systemPrefix$text$systemSuffix';

  /// A user turn; `first` with the block before it (system block or nothing).
  String user(String content, {bool first = false, bool afterSystem = true}) =>
      (first ? (afterSystem ? firstUserPrefix : noSystemHead) : userPrefix) + content + userSuffix;

  /// A finished reply (no thinking control: as the API's history carries it).
  String assistant(String content) => '$opening$content$assistantEnd';

  /// The opening of the assistant's reply, which the model continues.
  String reply(Thinking thinking) => opening + assistantPrefix(thinking);

  /// What the reply starts with before the model writes (thinking control).
  String assistantPrefix(Thinking thinking) => switch (thinking) {
        Thinking.on => thinkingOn,
        Thinking.off => thinkingOff,
        Thinking.none => '',
      };

  /// The full prompt for a new message after `history`, starting with
  /// `systemBlock` (system(), or a block that also lists tools; '' for none).
  String build(String systemBlock, List<Turn> history, String user, Thinking thinking) {
    final p = StringBuffer(systemBlock);
    var first = true;
    String userTurn(String text) {
      final block = this.user(text, first: first, afterSystem: systemBlock.isNotEmpty);
      first = false;
      return block;
    }

    for (final t in history) {
      p.write(userTurn(t.user) + opening + t.exact + assistantEnd);
    }
    p.write(userTurn(user) + reply(thinking));
    return p.toString();
  }

  /// The built-in format the loaded model's vocabulary implies, and whether it has a thinking mode.
  static Future<(ChatTemplate, bool)> detect(EngineService engine) async {
    if (await engine.countTokens('<|im_start|>') == 1) {
      return (chatml, await _thinkTokens(engine));
    }
    if (await engine.countTokens('<|start_header_id|>') == 1) return (llama3, false);
    return (zephyr, false);
  }

  /// Selects the author format or a verified vocabulary fallback. A present
  /// incompatible template must not silently become an unrelated protocol.
  /// Null means chat is unavailable; raw completion and pooling do not need it.
  static Future<(ChatTemplate?, bool, String)> resolve(EngineService engine, String source) async {
    final (own, thinking, why) = await fromModel(engine, source);
    if (own != null) return (own, thinking, '');
    final (fallback, fallbackThinking) = await detect(engine);
    if (source.isNotEmpty && identical(fallback, zephyr)) return (null, false, why);
    return (fallback, fallbackThinking, why);
  }

  static Future<bool> _thinkTokens(EngineService engine) async =>
      await engine.countTokens('<think>') == 1 && await engine.countTokens('</think>') == 1;

  /// Probe variables: past replies keep their reasoning (preserve_thinking,
  /// Qwen3.6), as the app replays them; templates without the switch ignore it.
  static Map<String, Object?> _probe(bool? thinking, {String replyPrefix = ''}) => {
        'preserve_thinking': true,
        'enable_thinking': ?thinking,
        if (replyPrefix.isNotEmpty) 'reply_prefix': replyPrefix,
      };

  /// The format of the model's own chat template, when it has one and it is
  /// prefix-stable in every mode the app uses; else null and why.
  static Future<(ChatTemplate?, bool, String)> fromModel(EngineService engine, String source) async {
    if (source.isEmpty) return (null, false, 'the model has no chat template');
    try {
      final thinks = source.contains('enable_thinking') && await _thinkTokens(engine);
      final plain = await engine.chatFormat(_probe(thinks ? false : null));
      if (plain['prefix_stable'] != true) return (null, false, '${plain['reason']}');
      final off = plain['generation_prompt'] as String;
      var opening = off, thinkingOff = '', thinkingOn = '';
      if (thinks) {
        // The reply's reasoning, as the model writes it after the opening.
        final on = await engine.chatFormat(_probe(true, replyPrefix: 'LiyabProbeReasoning\n</think>\n\n'));
        if (on['prefix_stable'] != true) return (null, false, 'thinking on: ${on['reason']}');
        final onPrompt = on['generation_prompt'] as String;
        final at = off.indexOf('<think>'), atOn = onPrompt.indexOf('<think>');
        if (at < 0 || atOn < 0 || off.substring(0, at) != onPrompt.substring(0, atOn)) {
          return (null, false, 'the thinking switch changes more than the reasoning block');
        }
        opening = off.substring(0, at);
        thinkingOff = off.substring(at);
        thinkingOn = onPrompt.substring(atOn);
      }
      String s(String key) => plain[key] as String;
      return (
        ChatTemplate(
          label: 'model template',
          systemPrefix: s('system_prefix'),
          systemSuffix: s('system_suffix'),
          firstUserPrefix: s('first_user_prefix'),
          noSystemHead: s('no_system_head'),
          userPrefix: s('user_prefix'),
          userSuffix: s('user_suffix'),
          opening: opening,
          assistantEnd: s('assistant_end'),
          thinkingOn: thinkingOn,
          thinkingOff: thinkingOff,
          isModelTemplate: true,
        ),
        thinks,
        ''
      );
    } on EngineException catch (e) {
      return (null, false, e.message);
    }
  }
}
