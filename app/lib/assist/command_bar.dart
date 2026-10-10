// The assistant on a computer: a command bar, opened from any app with a
// global shortcut (Option-Space on macOS), in the manner of Spotlight, Raycast
// or PowerToys Run, so it reads as native on macOS, Windows and Linux alike.
// A large input with the living flame; under it the conversation started
// here, then a footer with the model and the keys.
//
// It is alive the way the phone's sheet is: the aura turns around it with the
// assistant's state, the flame ignites when the bar opens and leans in as the
// user types, idle suggestions drift through the empty field, and the answer
// grows the bar smoothly. The native window is exactly the bar plus a margin
// for the aura's glow (AssistMode.resize), and keeps its top edge fixed.
// Phones keep the sheet (assist_sheet.dart).

import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';

import '../state/app_state.dart';
import '../ui/aura.dart';
import '../ui/living_flame.dart';
import '../ui/theme.dart';
import '../ui/waking_up.dart';
import 'assist_sheet.dart';

class CommandBar extends StatefulWidget {
  const CommandBar({super.key, required this.app, required this.mode});
  final AppState app;
  final AssistMode mode;

  /// Width of the bar itself.
  static const width = 720.0;

  /// Transparent room around the bar for the aura's glow; the native window
  /// is the bar plus this on every side (macos/Runner/AssistWindow.swift).
  static const margin = 30.0;

  /// Tallest the conversation area grows before it scrolls.
  static const maxConversation = 460.0;

  @override
  State<CommandBar> createState() => _CommandBarState();
}

class _CommandBarState extends State<CommandBar> with TickerProviderStateMixin {
  final _input = TextEditingController();
  final _focus = FocusNode();
  final _scroll = ScrollController();
  late int _firstMessage = widget.app.messages.length; // the bar shows only what was asked here

  // The bar rises into place and the flame ignites a moment after it.
  late final AnimationController _enter = AnimationController(vsync: this, duration: const Duration(milliseconds: 520))
    ..forward();

  // Typing: each keystroke is a small pulse the flame leans into, fading out.
  double _typing = 0;
  Timer? _typingDecay;

  // Idle suggestions drifting through the empty field.
  static const _suggestions = [
    'Ask Liyab anything',
    'Summarize what I copied',
    'Draft a reply to this email',
    'Explain this in simple words',
    'Turn these notes into a list',
  ];
  int _suggestion = 0;
  Timer? _suggestionTimer;

  AppState get app => widget.app;

  // macOS draws the system's blurred material behind the bar
  // (AssistWindow.swift): the bar only tints it. Elsewhere the night ground
  // is painted, nearly opaque.
  static final Color _ground = Platform.isMacOS
      ? Palette.kiln.withValues(alpha: 0.55)
      : Palette.kiln.withValues(alpha: 0.97);
  static const _radius = BorderRadius.all(Radius.circular(18));

  @override
  void initState() {
    super.initState();
    _input.addListener(_typed);
    _suggestionTimer = Timer.periodic(const Duration(milliseconds: 3200), (_) {
      if (_input.text.isEmpty && _asked.isEmpty && mounted) {
        setState(() => _suggestion = (_suggestion + 1) % _suggestions.length);
      }
    });
  }

  void _typed() {
    app.draftChanged(_input.text); // the engine processes the draft while the user writes
    setState(() => _typing = 1);
    _typingDecay?.cancel();
    _typingDecay = Timer.periodic(const Duration(milliseconds: 50), (t) {
      if (!mounted) return t.cancel();
      setState(() => _typing *= 0.82);
      if (_typing < 0.05) {
        t.cancel();
        setState(() => _typing = 0);
      }
    });
  }

  @override
  void dispose() {
    _typingDecay?.cancel();
    _suggestionTimer?.cancel();
    _enter.dispose();
    _input.dispose();
    _focus.dispose();
    _scroll.dispose();
    super.dispose();
  }

  List<ChatMessage> get _asked =>
      app.messages.length > _firstMessage ? app.messages.sublist(_firstMessage) : const <ChatMessage>[];

  bool get _ready => app.modelPath != null && !app.loading;

  FlameState get _flame {
    final last = _asked.isEmpty ? null : _asked.last;
    if (app.loading) return FlameState.thinking;
    if (app.generating && last != null) {
      return last.raw.isEmpty || last.thinkingNow ? FlameState.thinking : FlameState.answering;
    }
    return _typing > 0 ? FlameState.listening : FlameState.resting;
  }

  void _send() {
    final text = _input.text.trim();
    if (text.isEmpty || !_ready || app.generating) return;
    _input.clear();
    app.send(text);
  }

  void _newChat() {
    if (app.generating) return;
    app.newChat();
    setState(() => _firstMessage = 0);
    _focus.requestFocus();
  }

  @override
  Widget build(BuildContext context) {
    final rise = CurvedAnimation(parent: _enter, curve: Curves.easeOutCubic);
    return CallbackShortcuts(
      bindings: {
        const SingleActivator(LogicalKeyboardKey.escape): () => app.generating ? app.stop() : widget.mode.close(),
        SingleActivator(LogicalKeyboardKey.keyN, meta: Platform.isMacOS, control: !Platform.isMacOS): _newChat,
        SingleActivator(LogicalKeyboardKey.enter, meta: Platform.isMacOS, control: !Platform.isMacOS):
            widget.mode.openApp,
      },
      child: Material(
        type: MaterialType.transparency,
        // Laid out at its natural height whatever the window's current size,
        // so it can ask the window to grow.
        child: OverflowBox(
          alignment: Alignment.topCenter,
          minHeight: 0,
          maxHeight: double.infinity,
          child: _ReportSize(
            onSize: (size) => widget.mode.resize(size.height),
            child: Padding(
              padding: const EdgeInsets.all(CommandBar.margin),
              child: AnimatedBuilder(
                animation: rise,
                builder: (context, child) => Opacity(
                  opacity: rise.value,
                  child: Transform.translate(
                    offset: Offset(0, -10 * (1 - rise.value)),
                    child: Transform.scale(scale: 0.97 + 0.03 * rise.value, child: child),
                  ),
                ),
                child: SizedBox(
                  width: CommandBar.width,
                  child: ListenableBuilder(listenable: Listenable.merge([app, app.monitor]), builder: _bar),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _bar(BuildContext context, Widget? _) => Aura(
    state: _flame,
    borderRadius: _radius,
    spread: CommandBar.margin,
    child: DecoratedBox(
      decoration: BoxDecoration(
        color: _ground,
        borderRadius: _radius,
        border: Border.all(color: Colors.white.withValues(alpha: 0.06)),
      ),
      child: AnimatedSize(
        duration: const Duration(milliseconds: 260),
        curve: Curves.easeOutCubic,
        alignment: Alignment.topCenter,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _inputRow(context),
            if (_body(context) case final body?) ...[_hairline(), body],
            _hairline(),
            _footer(context),
          ],
        ),
      ),
    ),
  );

  Widget _hairline() => Container(height: 1, color: Colors.white.withValues(alpha: 0.06));

  Widget _inputRow(BuildContext context) {
    final theme = Theme.of(context);
    final style = theme.textTheme.bodyLarge?.copyWith(fontSize: 23, height: 1.25, letterSpacing: -0.2);
    final ignite = CurvedAnimation(parent: _enter, curve: const Interval(0.25, 1, curve: Curves.elasticOut));
    final hint = _ready
        ? (_asked.isEmpty ? _suggestions[_suggestion] : 'Ask a follow-up')
        : (app.loading ? 'Waking up…' : 'No model loaded');
    return SizedBox(
      height: 76,
      child: Row(
        children: [
          const SizedBox(width: 14),
          // The flame, larger, on a halo of its own light; it ignites as the
          // bar opens and swells while the user types.
          ScaleTransition(
            scale: ignite,
            child: SizedBox(
              width: 52,
              height: 52,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  DecoratedBox(
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      gradient: RadialGradient(
                        colors: [
                          Palette.ember.withValues(alpha: 0.30 + 0.25 * _typing),
                          Palette.ember.withValues(alpha: 0),
                        ],
                      ),
                    ),
                    child: const SizedBox.expand(),
                  ),
                  Transform.scale(
                    scale: 1 + 0.08 * _typing,
                    child: LivingFlame(size: 46, state: _flame, heat: app.monitor.heat, voiceLevel: _typing),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Stack(
              alignment: Alignment.centerLeft,
              children: [
                // The hint drifts: each suggestion rises in as the last fades out.
                if (_input.text.isEmpty)
                  IgnorePointer(
                    child: AnimatedSwitcher(
                      duration: const Duration(milliseconds: 450),
                      transitionBuilder: (child, a) => FadeTransition(
                        opacity: a,
                        child: SlideTransition(
                          position: Tween(begin: const Offset(0, 0.35), end: Offset.zero).animate(a),
                          child: child,
                        ),
                      ),
                      layoutBuilder: (current, previous) =>
                          Stack(alignment: Alignment.centerLeft, children: [...previous, ?current]),
                      child: Text(
                        hint,
                        key: ValueKey(hint),
                        style: style?.copyWith(color: Palette.smoke.withValues(alpha: 0.75)),
                      ),
                    ),
                  ),
                TextField(
                  controller: _input,
                  focusNode: _focus,
                  autofocus: true,
                  enabled: _ready,
                  style: style,
                  cursorColor: Palette.ember,
                  cursorWidth: 2.2,
                  cursorRadius: const Radius.circular(2),
                  decoration: const InputDecoration(border: InputBorder.none, isCollapsed: true),
                  onSubmitted: (_) {
                    _send();
                    _focus.requestFocus(); // Enter would otherwise leave the field
                  },
                ),
              ],
            ),
          ),
          AnimatedSwitcher(
            duration: const Duration(milliseconds: 200),
            transitionBuilder: (child, a) => ScaleTransition(scale: a, child: FadeTransition(opacity: a, child: child)),
            child: app.generating
                ? _IconKey(key: const ValueKey('stop'), icon: Icons.stop_rounded, tooltip: 'Stop', onPressed: app.stop)
                : _asked.isNotEmpty
                ? _IconKey(key: const ValueKey('new'), icon: Icons.add_rounded, tooltip: 'New chat', onPressed: _newChat)
                : const SizedBox(key: ValueKey('none')),
          ),
          const SizedBox(width: 16),
        ],
      ),
    );
  }

  /// What sits between the input and the footer, if anything.
  Widget? _body(BuildContext context) {
    final theme = Theme.of(context);
    if (app.modelPath == null && !app.loading) {
      return Padding(
        padding: const EdgeInsets.fromLTRB(22, 18, 18, 18),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Choose a model to talk to', style: theme.textTheme.titleMedium),
                  const SizedBox(height: 2),
                  Text(
                    'It runs here, on this computer. Nothing you ask leaves it.',
                    style: theme.textTheme.bodyMedium?.copyWith(color: Palette.smoke),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 16),
            FilledButton.icon(
              onPressed: widget.mode.openApp,
              icon: const Icon(Icons.download_rounded, size: 18),
              label: const Text('Get a model'),
            ),
          ],
        ),
      );
    }
    if (_asked.isEmpty) {
      if (!app.loading) return null;
      return Padding(
        padding: const EdgeInsets.symmetric(vertical: 14),
        child: WakingUp(status: app.status, heat: app.monitor.heat, size: 56),
      );
    }
    // Follows the answer as it streams.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients) _scroll.jumpTo(_scroll.position.maxScrollExtent);
    });
    return ConstrainedBox(
      constraints: const BoxConstraints(maxHeight: CommandBar.maxConversation),
      child: SingleChildScrollView(
        controller: _scroll,
        padding: const EdgeInsets.fromLTRB(24, 4, 24, 18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [for (final m in _asked) _Exchange(key: ObjectKey(m), message: m)],
        ),
      ),
    );
  }

  Widget _footer(BuildContext context) {
    final theme = Theme.of(context);
    final stats = _asked.isEmpty ? null : _asked.last.stats;
    final meta = Platform.isMacOS ? '⌘' : 'Ctrl ';
    final label = theme.textTheme.labelSmall?.copyWith(color: Palette.smoke);
    return SizedBox(
      height: 38,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16),
        child: Row(
          children: [
            Icon(Icons.lock_outline_rounded, size: 13, color: Palette.smoke.withValues(alpha: 0.8)),
            const SizedBox(width: 6),
            Flexible(
              child: Text(
                [
                  app.modelName.isEmpty ? 'Private, on this computer' : app.modelName,
                  if (stats != null) '${stats.tokensPerSecond.toStringAsFixed(1)} tok/s',
                ].join('  ·  '),
                style: label,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const Spacer(),
            _Hint(keys: '↩', label: 'Ask', style: label),
            _Hint(keys: '${meta}N', label: 'New', style: label),
            _Hint(keys: '$meta↩', label: 'Open Liyab', style: label),
            _Hint(keys: 'esc', label: app.generating ? 'Stop' : 'Close', style: label),
          ],
        ),
      ),
    );
  }
}

/// One question and its answer: the question as a muted line, the answer as
/// reading text fading in, a shimmering "Thinking" while the model reasons.
class _Exchange extends StatelessWidget {
  const _Exchange({super.key, required this.message});
  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final answer = message.answer;
    final reading = theme.textTheme.bodyLarge?.copyWith(fontSize: 15.5, height: 1.6);
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: const Duration(milliseconds: 380),
      curve: Curves.easeOutCubic,
      builder: (context, v, child) => Opacity(
        opacity: v,
        child: Transform.translate(offset: Offset(0, 8 * (1 - v)), child: child),
      ),
      child: Padding(
        padding: const EdgeInsets.only(top: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              message.user,
              style: theme.textTheme.bodyMedium?.copyWith(color: Palette.smoke, fontWeight: FontWeight.w600),
            ),
            const SizedBox(height: 8),
            for (final step in message.steps)
              Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Row(
                  children: [
                    const Icon(Icons.auto_awesome_rounded, size: 14, color: Palette.gold),
                    const SizedBox(width: 6),
                    Flexible(
                      child: Text(
                        '${step.label}: ${step.summary}',
                        style: theme.textTheme.labelSmall?.copyWith(color: Palette.gold.withValues(alpha: 0.9)),
                      ),
                    ),
                  ],
                ),
              ),
            AnimatedSwitcher(
              duration: const Duration(milliseconds: 300),
              child: answer.isEmpty && message.streaming
                  ? _Shimmer(
                      key: const ValueKey('thinking'),
                      child: Text(message.thinkingNow ? 'Thinking…' : 'Reading…', style: reading),
                    )
                  : SelectableText(answer, key: const ValueKey('answer'), style: reading),
            ),
            if (message.error != null)
              Padding(
                padding: const EdgeInsets.only(top: 6),
                child: Text(
                  'The reply stopped: ${message.error}',
                  style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.error),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

/// A band of light sweeping across its child, for text that waits.
class _Shimmer extends StatefulWidget {
  const _Shimmer({super.key, required this.child});
  final Widget child;

  @override
  State<_Shimmer> createState() => _ShimmerState();
}

class _ShimmerState extends State<_Shimmer> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this, duration: const Duration(milliseconds: 1600))
    ..repeat();

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
    animation: _c,
    child: widget.child,
    builder: (context, child) => ShaderMask(
      blendMode: BlendMode.srcIn,
      shaderCallback: (rect) => LinearGradient(
        colors: [Palette.smoke.withValues(alpha: 0.55), Palette.gold, Palette.smoke.withValues(alpha: 0.55)],
        stops: const [0.35, 0.5, 0.65],
        begin: Alignment(3 * _c.value - 2, 0),
        end: Alignment(3 * _c.value, 0),
      ).createShader(rect),
      child: child,
    ),
  );
}

/// A key and what it does, as in Raycast's footer.
class _Hint extends StatelessWidget {
  const _Hint({required this.keys, required this.label, this.style});
  final String keys;
  final String label;
  final TextStyle? style;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.only(left: 14),
    child: Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(label, style: style),
        const SizedBox(width: 6),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 1),
          decoration: BoxDecoration(
            color: Colors.white.withValues(alpha: 0.07),
            borderRadius: BorderRadius.circular(4),
            border: Border.all(color: Colors.white.withValues(alpha: 0.10)),
          ),
          child: Text(keys, style: style?.copyWith(fontSize: 11, color: Palette.ash.withValues(alpha: 0.85))),
        ),
      ],
    ),
  );
}

class _IconKey extends StatelessWidget {
  const _IconKey({super.key, required this.icon, required this.tooltip, required this.onPressed});
  final IconData icon;
  final String tooltip;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) => IconButton(
    tooltip: tooltip,
    onPressed: onPressed,
    icon: Icon(icon, size: 20),
    style: IconButton.styleFrom(
      foregroundColor: Palette.ash,
      backgroundColor: Colors.white.withValues(alpha: 0.08),
      fixedSize: const Size(34, 34),
      minimumSize: const Size(34, 34),
      padding: EdgeInsets.zero,
    ),
  );
}

/// Calls `onSize` after layout whenever its child's size changes, so the
/// native window can match the content.
class _ReportSize extends SingleChildRenderObjectWidget {
  const _ReportSize({required this.onSize, required super.child});
  final ValueChanged<Size> onSize;

  @override
  RenderObject createRenderObject(BuildContext context) => _RenderReportSize(onSize);

  @override
  void updateRenderObject(BuildContext context, _RenderReportSize renderObject) => renderObject.onSize = onSize;
}

class _RenderReportSize extends RenderProxyBox {
  _RenderReportSize(this.onSize);
  ValueChanged<Size> onSize;
  Size? _reported;

  @override
  void performLayout() {
    super.performLayout();
    final s = child?.size ?? Size.zero;
    if (s == _reported) return;
    _reported = s;
    WidgetsBinding.instance.addPostFrameCallback((_) => onSize(s));
  }
}
