// The assistant sheet: what the system assist gesture opens over any app.
// A see-through window whose sheet rises from the bottom with the living flame;
// the conversation is the app's own, so "Continue in Liyab" picks it up there.
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../state/app_state.dart';
import '../ui/living_flame.dart';
import '../ui/theme.dart';

/// Whether the assistant sheet is showing, driven by AssistActivity.
class AssistMode extends ValueNotifier<bool> {
  AssistMode() : super(false) {
    _channel.setMethodCallHandler((call) async {
      if (call.method == 'opened') value = true;
      if (call.method == 'closed') value = false;
    });
  }

  static const _channel = MethodChannel('liyab/assist');

  Future<void> close() async {
    value = false;
    await _channel.invokeMethod<void>('close');
  }

  Future<void> openApp() async {
    value = false;
    await _channel.invokeMethod<void>('openApp');
  }
}

class AssistSheet extends StatefulWidget {
  const AssistSheet({super.key, required this.app, required this.mode});
  final AppState app;
  final AssistMode mode;

  @override
  State<AssistSheet> createState() => _AssistSheetState();
}

class _AssistSheetState extends State<AssistSheet> with SingleTickerProviderStateMixin {
  final _input = TextEditingController();
  late final AnimationController _rise =
      AnimationController(vsync: this, duration: const Duration(milliseconds: 380))..forward();
  late final int _firstMessage = widget.app.messages.length; // the sheet shows only what was asked here

  AppState get app => widget.app;

  @override
  void dispose() {
    _input.dispose();
    _rise.dispose();
    super.dispose();
  }

  Future<void> _close() async {
    await _rise.reverse();
    await widget.mode.close();
  }

  void _send([String? text]) {
    final t = (text ?? _input.text).trim();
    if (t.isEmpty) return;
    HapticFeedback.lightImpact();
    _input.clear();
    app.send(t);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      backgroundColor: Colors.transparent,
      resizeToAvoidBottomInset: true,
      body: Stack(children: [
        // Tapping outside the sheet dismisses it, like a system sheet.
        Positioned.fill(
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: _close,
            child: FadeTransition(
              opacity: CurvedAnimation(parent: _rise, curve: Curves.easeOut),
              child: const ColoredBox(color: Color(0x66000000)),
            ),
          ),
        ),
        Align(
          alignment: Alignment.bottomCenter,
          child: SlideTransition(
            position: Tween(begin: const Offset(0, 1), end: Offset.zero)
                .animate(CurvedAnimation(parent: _rise, curve: Curves.easeOutCubic)),
            child: ListenableBuilder(
              listenable: Listenable.merge([app, app.monitor]),
              builder: (context, _) => _sheet(theme),
            ),
          ),
        ),
      ]),
    );
  }

  Widget _sheet(ThemeData theme) {
    final asked = app.messages.length > _firstMessage ? app.messages.sublist(_firstMessage) : const <ChatMessage>[];
    final last = asked.isEmpty ? null : asked.last;
    final flame = !app.generating || last == null
        ? (app.loading ? FlameState.thinking : FlameState.resting)
        : (last.raw.isEmpty || last.thinkingNow ? FlameState.thinking : FlameState.answering);
    final ready = app.modelPath != null && !app.loading;
    return Stack(clipBehavior: Clip.none, children: [
      // The flame's glow above the sheet.
      Positioned(
        left: 0,
        right: 0,
        top: -70,
        height: 70,
        child: IgnorePointer(
          child: DecoratedBox(
            decoration: BoxDecoration(
              gradient: RadialGradient(
                center: Alignment.bottomCenter,
                radius: 1.4,
                colors: [Palette.ember.withValues(alpha: app.generating ? 0.5 : 0.25), Palette.ember.withValues(alpha: 0)],
              ),
            ),
          ),
        ),
      ),
      Container(
        width: double.infinity,
        constraints: BoxConstraints(maxHeight: MediaQuery.sizeOf(context).height * 0.75),
        decoration: const BoxDecoration(
          color: Palette.kiln,
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: SafeArea(
          top: false,
          minimum: const EdgeInsets.fromLTRB(18, 16, 18, 14),
          child: Column(mainAxisSize: MainAxisSize.min, crossAxisAlignment: CrossAxisAlignment.stretch, children: [
            Row(children: [
              LivingFlame(size: 44, state: flame, heat: app.monitor.heat),
              const SizedBox(width: 10),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(
                    last == null ? (ready ? 'How can I help?' : (app.loading ? 'Waking up' : 'No model loaded')) : last.user,
                    style: theme.textTheme.titleLarge,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  Text(
                    ready ? 'On this phone. Nothing leaves it.' : app.status,
                    style: theme.textTheme.labelSmall,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                  ),
                ]),
              ),
              IconButton(
                tooltip: 'Continue in Liyab',
                icon: const Icon(Icons.open_in_full_rounded),
                onPressed: widget.mode.openApp,
              ),
            ]),
            if (last != null)
              Flexible(
                child: SingleChildScrollView(
                  reverse: true,
                  padding: const EdgeInsets.only(top: 12, bottom: 4),
                  child: Text(
                    last.answer.isNotEmpty
                        ? last.answer
                        : (last.thinkingNow ? 'Thinking…' : (last.error != null ? 'The reply stopped: ${last.error}' : '')),
                    style: theme.textTheme.bodyLarge,
                  ),
                ),
              ),
            if (last == null && ready) ...[
              const SizedBox(height: 12),
              Wrap(spacing: 8, runSpacing: 8, children: [
                for (final s in const ['Summarize what I paste', 'Draft a short reply', 'Explain a word'])
                  ActionChip(
                    label: Text(s),
                    onPressed: () => _input.text = '$s: ',
                    backgroundColor: theme.colorScheme.surfaceContainerHighest,
                    side: BorderSide(color: theme.colorScheme.outline),
                    shape: const StadiumBorder(),
                  ),
              ]),
            ],
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.fromLTRB(16, 2, 6, 2),
              decoration: BoxDecoration(
                color: theme.colorScheme.surfaceContainerHighest,
                borderRadius: BorderRadius.circular(26),
              ),
              child: Row(children: [
                Expanded(
                  child: TextField(
                    controller: _input,
                    enabled: ready,
                    autofocus: ready,
                    minLines: 1,
                    maxLines: 4,
                    textCapitalization: TextCapitalization.sentences,
                    decoration: const InputDecoration(hintText: 'Ask Liyab', border: InputBorder.none, isDense: true),
                    onSubmitted: ready && !app.generating ? (_) => _send() : null,
                  ),
                ),
                IconButton(
                  tooltip: app.generating ? 'Stop' : 'Send',
                  onPressed: !ready ? null : (app.generating ? app.stop : _send),
                  icon: Icon(app.generating ? Icons.stop_rounded : Icons.arrow_upward_rounded, color: Palette.ember),
                ),
              ]),
            ),
          ]),
        ),
      ),
    ]);
  }
}
