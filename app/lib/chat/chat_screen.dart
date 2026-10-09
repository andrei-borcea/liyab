// The conversation: the living flame at rest in an empty chat, then the
// messages; a composer that glows while the model works.
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../state/app_state.dart';
import '../ui/living_flame.dart';
import '../ui/theme.dart';
import 'model_sheet.dart';
import 'settings_sheet.dart';

class ChatScreen extends StatefulWidget {
  const ChatScreen({super.key, required this.app});
  final AppState app;

  @override
  State<ChatScreen> createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final _input = TextEditingController();
  final _scroll = ScrollController();
  final _focus = FocusNode();

  AppState get app => widget.app;

  @override
  void initState() {
    super.initState();
    app.addListener(_changed);
  }

  @override
  void dispose() {
    app.removeListener(_changed);
    _input.dispose();
    _scroll.dispose();
    _focus.dispose();
    super.dispose();
  }

  void _changed() {
    setState(() {});
    // Follow the reply while it streams, unless the reader scrolled up.
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scroll.hasClients && _scroll.position.extentAfter < 120) {
        _scroll.jumpTo(_scroll.position.maxScrollExtent);
      }
    });
  }

  void _send([String? text]) {
    final t = (text ?? _input.text).trim();
    if (t.isEmpty) return;
    HapticFeedback.lightImpact();
    _input.clear();
    app.send(t);
  }

  FlameState get _flameState {
    if (!app.generating || app.messages.isEmpty) return FlameState.resting;
    final m = app.messages.last;
    return m.raw.isEmpty || m.thinkingNow ? FlameState.thinking : FlameState.answering;
  }

  double get _heat {
    final reroutes = app.messages.isEmpty ? 0 : (app.messages.last.stats?.thermalReroutes ?? 0);
    return reroutes > 0 ? 0.85 : 0.3;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        titleSpacing: 12,
        title: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () => showModelSheet(context, app),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
            child: Row(children: [
              LivingFlame(size: 34, state: _flameState, heat: _heat),
              const SizedBox(width: 10),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Liyab', style: theme.textTheme.titleLarge),
                  Text(
                    app.loading ? app.status : (app.modelName.isEmpty ? 'Choose a model' : app.modelName),
                    style: theme.textTheme.labelSmall,
                    overflow: TextOverflow.ellipsis,
                  ),
                ]),
              ),
              Icon(Icons.expand_more, color: theme.colorScheme.onSurfaceVariant, size: 20),
            ]),
          ),
        ),
        actions: [
          IconButton(
            tooltip: 'New chat',
            onPressed: app.messages.isEmpty || app.generating ? null : app.newChat,
            icon: const Icon(Icons.edit_note_rounded),
          ),
          IconButton(
            tooltip: 'Settings',
            onPressed: () => showSettingsSheet(context, app),
            icon: const Icon(Icons.tune_rounded),
          ),
          const SizedBox(width: 4),
        ],
      ),
      body: Column(children: [
        Expanded(
          child: AnimatedSwitcher(
            duration: const Duration(milliseconds: 350),
            child: app.messages.isEmpty
                ? _EmptyChat(app: app, onSuggestion: _send, flameState: _flameState, heat: _heat)
                : ListView.builder(
                    key: const ValueKey('messages'),
                    controller: _scroll,
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 24),
                    itemCount: app.messages.length,
                    itemBuilder: (context, i) => _MessageView(message: app.messages[i]),
                  ),
          ),
        ),
        _Composer(
          controller: _input,
          focus: _focus,
          enabled: app.modelPath != null && !app.loading,
          generating: app.generating,
          onSend: _send,
          onStop: app.stop,
        ),
      ]),
    );
  }
}

class _EmptyChat extends StatelessWidget {
  const _EmptyChat({required this.app, required this.onSuggestion, required this.flameState, required this.heat});
  final AppState app;
  final void Function(String) onSuggestion;
  final FlameState flameState;
  final double heat;

  static const _suggestions = [
    'Explain how a refrigerator works',
    'Write a short poem about the sea',
    'Plan a three-day trip to Rome',
  ];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final ready = app.modelPath != null && !app.loading;
    return Center(
      key: const ValueKey('empty'),
      child: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24),
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          LivingFlame(size: 168, state: app.loading ? FlameState.thinking : flameState, heat: heat),
          const SizedBox(height: 8),
          Text(
            ready ? 'How can I help?' : (app.loading ? 'Waking up' : 'Choose a model to begin'),
            style: theme.textTheme.headlineMedium,
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 6),
          Text(
            ready ? 'Runs on this phone. Nothing you write leaves it.' : app.status,
            style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 24),
          if (ready)
            Wrap(
              alignment: WrapAlignment.center,
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final s in _suggestions)
                  ActionChip(
                    label: Text(s),
                    onPressed: () => onSuggestion(s),
                    backgroundColor: theme.colorScheme.surfaceContainerHighest,
                    side: BorderSide(color: theme.colorScheme.outline),
                    shape: const StadiumBorder(),
                  ),
              ],
            )
          else if (!app.loading)
            FilledButton.icon(
              onPressed: () => showModelSheet(context, app),
              icon: const Icon(Icons.folder_open_rounded),
              label: const Text('Choose a model'),
            ),
        ]),
      ),
    );
  }
}

class _MessageView extends StatelessWidget {
  const _MessageView({required this.message});
  final ChatMessage message;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final reasoning = message.reasoning;
    final stats = message.stats;
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0, end: 1),
      duration: const Duration(milliseconds: 320),
      curve: Curves.easeOutCubic,
      builder: (context, v, child) => Opacity(
        opacity: v,
        child: Transform.translate(offset: Offset(0, (1 - v) * 12), child: child),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
        const SizedBox(height: 16),
        // The user's message: a pill on the right.
        Align(
          alignment: Alignment.centerRight,
          child: ConstrainedBox(
            constraints: BoxConstraints(maxWidth: MediaQuery.sizeOf(context).width * 0.8),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(22),
              ),
              child: SelectableText(message.user, style: theme.textTheme.bodyLarge),
            ),
          ),
        ),
        const SizedBox(height: 14),
        if (reasoning != null) _Reasoning(text: reasoning, active: message.thinkingNow),
        // The reply: plain text, full width, like a page.
        if (message.answer.isNotEmpty || (message.streaming && reasoning == null))
          AnimatedSize(
            duration: const Duration(milliseconds: 120),
            alignment: Alignment.topLeft,
            child: SelectableText.rich(
              TextSpan(children: [
                TextSpan(text: message.answer),
                if (message.streaming && !message.thinkingNow)
                  const WidgetSpan(alignment: PlaceholderAlignment.middle, child: _Caret()),
              ]),
              style: theme.textTheme.bodyLarge,
            ),
          ),
        if (message.error != null)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text('The reply stopped: ${message.error}',
                style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.error)),
          ),
        if (stats != null)
          Padding(
            padding: const EdgeInsets.only(top: 8),
            child: Text(
              '${stats.generatedTokens} tokens, ${stats.tokensPerSecond.toStringAsFixed(1)} tok/s, '
              'first token ${(stats.ttftMs / 1000).toStringAsFixed(2)} s, '
              '${stats.cachedPrefixTokens} of ${stats.promptTokens} prompt tokens reused'
              '${stats.thermalReroutes > 0 ? ', cooling down' : ''}'
              '${stats.cancelled ? ', stopped' : ''}',
              style: theme.textTheme.labelSmall,
            ),
          ),
      ]),
    );
  }
}

/// The model's reasoning, folded under one line once it is done.
class _Reasoning extends StatefulWidget {
  const _Reasoning({required this.text, required this.active});
  final String text;
  final bool active;

  @override
  State<_Reasoning> createState() => _ReasoningState();
}

class _ReasoningState extends State<_Reasoning> {
  bool _open = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    final show = widget.active || _open;
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        InkWell(
          borderRadius: BorderRadius.circular(8),
          onTap: widget.active ? null : () => setState(() => _open = !_open),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(mainAxisSize: MainAxisSize.min, children: [
              Text(widget.active ? 'Thinking' : 'Reasoning', style: muted?.copyWith(fontWeight: FontWeight.w600)),
              if (!widget.active) Icon(show ? Icons.expand_less : Icons.expand_more, size: 18, color: muted?.color),
            ]),
          ),
        ),
        AnimatedSize(
          duration: const Duration(milliseconds: 220),
          alignment: Alignment.topLeft,
          child: show
              ? Container(
                  padding: const EdgeInsets.only(left: 12),
                  decoration: BoxDecoration(border: Border(left: BorderSide(color: theme.colorScheme.outline, width: 2))),
                  child: Text(widget.text, style: muted),
                )
              : const SizedBox(width: double.infinity),
        ),
      ]),
    );
  }
}

/// A soft ember dot at the end of a streaming reply.
class _Caret extends StatefulWidget {
  const _Caret();
  @override
  State<_Caret> createState() => _CaretState();
}

class _CaretState extends State<_Caret> with SingleTickerProviderStateMixin {
  late final AnimationController _c =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 900))..repeat(reverse: true);

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => FadeTransition(
        opacity: Tween(begin: 0.35, end: 1.0).animate(_c),
        child: Container(
          width: 9,
          height: 9,
          margin: const EdgeInsets.only(left: 4),
          decoration: const BoxDecoration(
            shape: BoxShape.circle,
            gradient: LinearGradient(colors: [Palette.flare, Palette.gold]),
          ),
        ),
      );
}

class _Composer extends StatelessWidget {
  const _Composer({
    required this.controller,
    required this.focus,
    required this.enabled,
    required this.generating,
    required this.onSend,
    required this.onStop,
  });

  final TextEditingController controller;
  final FocusNode focus;
  final bool enabled;
  final bool generating;
  final void Function([String?]) onSend;
  final VoidCallback onStop;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Stack(clipBehavior: Clip.none, children: [
      // The flame's glow rises over the composer while the model works.
      Positioned(
        left: 0,
        right: 0,
        top: -48,
        height: 48,
        child: IgnorePointer(
          child: AnimatedOpacity(
            opacity: generating ? 1 : 0,
            duration: const Duration(milliseconds: 500),
            child: const DecoratedBox(
              decoration: BoxDecoration(
                gradient: RadialGradient(
                  center: Alignment.bottomCenter,
                  radius: 1.6,
                  colors: [Color(0x55FF7A3D), Color(0x00FF7A3D)],
                ),
              ),
            ),
          ),
        ),
      ),
      SafeArea(
        top: false,
        minimum: const EdgeInsets.fromLTRB(12, 6, 12, 10),
        child: Container(
          padding: const EdgeInsets.fromLTRB(18, 4, 6, 4),
          decoration: BoxDecoration(
            color: theme.colorScheme.surfaceContainerHighest,
            borderRadius: BorderRadius.circular(28),
            border: Border.all(color: theme.colorScheme.outline),
          ),
          child: Row(crossAxisAlignment: CrossAxisAlignment.end, children: [
            Expanded(
              child: TextField(
                controller: controller,
                focusNode: focus,
                enabled: enabled,
                minLines: 1,
                maxLines: 6,
                textCapitalization: TextCapitalization.sentences,
                style: theme.textTheme.bodyLarge,
                decoration: InputDecoration(
                  hintText: enabled ? 'Ask Liyab' : 'Load a model first',
                  border: InputBorder.none,
                  isDense: true,
                  contentPadding: const EdgeInsets.symmetric(vertical: 12),
                ),
                onSubmitted: enabled && !generating ? (_) => onSend() : null,
              ),
            ),
            const SizedBox(width: 6),
            Padding(
              padding: const EdgeInsets.only(bottom: 2),
              child: _RoundAction(
                tooltip: generating ? 'Stop' : 'Send',
                icon: generating ? Icons.stop_rounded : Icons.arrow_upward_rounded,
                onPressed: !enabled ? null : (generating ? onStop : () => onSend()),
              ),
            ),
          ]),
        ),
      ),
    ]);
  }
}

class _RoundAction extends StatelessWidget {
  const _RoundAction({required this.tooltip, required this.icon, required this.onPressed});
  final String tooltip;
  final IconData icon;
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) => Tooltip(
        message: tooltip,
        child: AnimatedOpacity(
          opacity: onPressed == null ? 0.4 : 1,
          duration: const Duration(milliseconds: 200),
          child: Material(
            shape: const CircleBorder(),
            clipBehavior: Clip.antiAlias,
            child: Ink(
              width: 44,
              height: 44,
              decoration: const BoxDecoration(
                gradient: LinearGradient(begin: Alignment.bottomLeft, end: Alignment.topRight, colors: [Palette.flare, Palette.ember]),
              ),
              child: InkWell(
                onTap: onPressed,
                child: AnimatedSwitcher(
                  duration: const Duration(milliseconds: 180),
                  transitionBuilder: (c, a) => ScaleTransition(scale: a, child: c),
                  child: Icon(icon, key: ValueKey(icon), color: Colors.white),
                ),
              ),
            ),
          ),
        ),
      );
}
