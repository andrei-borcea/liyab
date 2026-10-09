// The assistant sheet: what the system assist gesture opens over any app.
// It rises from the bottom with the living flame and the latest answer, and
// expands in place into a full-height conversation, still over the app the
// user was in (as Gemini's overlay does); closing returns to that app. The
// full Liyab app is never opened from here.
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../chat/chat_screen.dart';
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

  /// Finishes the sheet's window. The mode turns off on 'closed' (the window's
  /// onPause), not here: switching now would draw the app's chat in the
  /// see-through window for the moment before it disappears.
  Future<void> close() => _channel.invokeMethod<void>('close');
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
  late final AnimationController _rise = AnimationController(vsync: this, duration: const Duration(milliseconds: 380))
    ..forward();
  late final int _firstMessage = widget.app.messages.length; // the sheet shows only what was asked here
  bool _expanded = false;

  // Resizing: the sheet follows a drag on its handle or header and stays where
  // it is let go; reaching the top edge turns it into the app.
  final _sheetKey = GlobalKey();
  double? _height; // set once dragged; null: fit the content
  bool _dragging = false;
  double _screen = 800;

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

  void _setExpanded(bool v) {
    HapticFeedback.selectionClick();
    setState(() {
      _expanded = v;
      _height = null;
    });
  }

  List<ChatMessage> get _asked =>
      app.messages.length > _firstMessage ? app.messages.sublist(_firstMessage) : const <ChatMessage>[];

  FlameState get _flame {
    final last = _asked.isEmpty ? null : _asked.last;
    if (app.loading) return FlameState.thinking;
    if (!app.generating || last == null) return FlameState.resting;
    return last.raw.isEmpty || last.thinkingNow ? FlameState.thinking : FlameState.answering;
  }

  @override
  Widget build(BuildContext context) {
    final height = _screen = MediaQuery.sizeOf(context).height;
    return PopScope(
      canPop: false,
      // Back (once pages pushed from the drawer are closed): collapse, then close.
      onPopInvokedWithResult: (didPop, _) {
        if (didPop) return;
        _expanded ? _setExpanded(false) : _close();
      },
      child: Scaffold(
        backgroundColor: Colors.transparent,
        resizeToAvoidBottomInset: true,
        body: Stack(
          children: [
            // Outside the sheet: a dim layer that closes it (compact only).
            Positioned.fill(
              child: GestureDetector(
                behavior: HitTestBehavior.opaque,
                onTap: _expanded ? null : _close,
                child: FadeTransition(
                  opacity: CurvedAnimation(parent: _rise, curve: Curves.easeOut),
                  child: const ColoredBox(color: Color(0x66000000)),
                ),
              ),
            ),
            Align(
              alignment: Alignment.bottomCenter,
              child: SlideTransition(
                position: Tween(
                  begin: const Offset(0, 1),
                  end: Offset.zero,
                ).animate(CurvedAnimation(parent: _rise, curve: Curves.easeOutCubic)),
                child: ListenableBuilder(
                  listenable: Listenable.merge([app, app.monitor]),
                  builder: (context, _) => _EmberEdge(
                    state: _flame,
                    heat: app.monitor.heat,
                    radius: _expanded ? 0 : 28,
                    child: AnimatedContainer(
                      key: _sheetKey,
                      // Follows the finger with no delay; snaps animate.
                      duration: _dragging ? Duration.zero : const Duration(milliseconds: 360),
                      curve: Curves.easeOutCubic,
                      // Constraints animate (a null height cannot): compact fits its
                      // content up to 60 % of the screen, a dragged sheet keeps its
                      // height, expanded fills the screen.
                      constraints: _expanded
                          ? BoxConstraints.tightFor(height: height)
                          : _height != null
                          ? BoxConstraints.tightFor(height: _height)
                          : BoxConstraints(maxHeight: height * 0.6),
                      decoration: BoxDecoration(
                        color: Palette.kiln,
                        borderRadius: BorderRadius.vertical(top: Radius.circular(_expanded ? 0 : 28)),
                      ),
                      // Expanded, the sheet becomes the app itself: its chat screen
                      // with the whole conversation and the navigation drawer.
                      child: _expanded
                          ? ChatScreen(app: app, onCollapse: () => _setExpanded(false))
                          : SafeArea(
                              top: false,
                              minimum: const EdgeInsets.fromLTRB(14, 6, 14, 12),
                              child: _compact(context),
                            ),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _dragStart(DragStartDetails _) {
    final box = _sheetKey.currentContext?.findRenderObject() as RenderBox?;
    setState(() {
      _dragging = true;
      _height = box?.size.height ?? _screen * 0.4;
    });
  }

  void _dragUpdate(DragUpdateDetails d) =>
      setState(() => _height = ((_height ?? 0) - d.delta.dy).clamp(120.0, _screen));

  /// Let go near the top (or flung up): the app; flung down or nearly gone:
  /// closed; anywhere else: the sheet stays at that height.
  void _dragEnd(DragEndDetails d) {
    final v = d.primaryVelocity ?? 0;
    final h = _height ?? 0;
    setState(() => _dragging = false);
    if (h >= _screen * 0.9 || v < -1500) {
      _setExpanded(true);
    } else if (v > 1200 || h < 150) {
      _close();
    }
  }

  /// The part of the sheet that resizes it when dragged.
  Widget _dragArea(Widget child) => GestureDetector(
    behavior: HitTestBehavior.opaque,
    onVerticalDragStart: _dragStart,
    onVerticalDragUpdate: _dragUpdate,
    onVerticalDragEnd: _dragEnd,
    child: child,
  );

  Widget _handle() => GestureDetector(
    onTap: () => _setExpanded(!_expanded),
    child: Center(
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 8),
        width: 40,
        height: 4,
        decoration: BoxDecoration(color: Palette.hairline, borderRadius: BorderRadius.circular(2)),
      ),
    ),
  );

  Widget _header(ThemeData theme, {required String title}) {
    final ready = app.modelPath != null && !app.loading;
    return Row(
      children: [
        LivingFlame(size: 42, state: _flame, heat: app.monitor.heat),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title, style: theme.textTheme.titleLarge, maxLines: 2, overflow: TextOverflow.ellipsis),
              Text(
                ready ? 'On this phone. Nothing leaves it.' : app.status,
                style: theme.textTheme.labelSmall,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
            ],
          ),
        ),
        IconButton(
          tooltip: _expanded ? 'Collapse' : 'Expand',
          icon: Icon(_expanded ? Icons.close_fullscreen_rounded : Icons.open_in_full_rounded),
          onPressed: () => _setExpanded(!_expanded),
        ),
        IconButton(tooltip: 'Close', icon: const Icon(Icons.close_rounded), onPressed: _close),
      ],
    );
  }

  Widget _compact(BuildContext context) {
    final theme = Theme.of(context);
    final last = _asked.isEmpty ? null : _asked.last;
    final ready = app.modelPath != null && !app.loading;
    final title = last?.user ?? (ready ? 'How can I help?' : (app.loading ? 'Waking up' : 'No model loaded'));
    return Column(
      mainAxisSize: _height != null ? MainAxisSize.max : MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _dragArea(
          Column(
            children: [
              _handle(),
              _header(theme, title: title),
            ],
          ),
        ),
        if (last != null)
          Flexible(
            child: SingleChildScrollView(
              reverse: true,
              padding: const EdgeInsets.only(top: 10, bottom: 4),
              child: Text(
                last.answer.isNotEmpty
                    ? last.answer
                    : (last.thinkingNow ? 'Thinking…' : (last.error != null ? 'The reply stopped: ${last.error}' : '')),
                style: theme.textTheme.bodyLarge,
              ),
            ),
          ),
        if (last == null && ready) ...[
          const SizedBox(height: 10),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final s in const ['Summarize what I paste', 'Draft a short reply', 'Explain a word'])
                ActionChip(
                  label: Text(s),
                  onPressed: () => _input.text = '$s: ',
                  backgroundColor: theme.colorScheme.surfaceContainerHighest,
                  side: BorderSide(color: theme.colorScheme.outline),
                  shape: const StadiumBorder(),
                ),
            ],
          ),
        ],
        // A resized sheet keeps its composer at the bottom.
        if (_height != null && last == null) const Spacer(),
        const SizedBox(height: 12),
        _composer(theme),
      ],
    );
  }

  Widget _composer(ThemeData theme) {
    final ready = app.modelPath != null && !app.loading;
    return Container(
      padding: const EdgeInsets.fromLTRB(16, 2, 6, 2),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(26),
      ),
      child: Row(
        children: [
          Expanded(
            child: TextField(
              controller: _input,
              enabled: ready,
              autofocus: ready && _asked.isEmpty,
              minLines: 1,
              maxLines: 4,
              textCapitalization: TextCapitalization.sentences,
              decoration: const InputDecoration(hintText: 'Ask Liyab', border: InputBorder.none, isDense: true),
              onSubmitted: ready && !app.generating ? (_) => _send() : null,
            ),
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: RoundAction(
              tooltip: app.generating ? 'Stop' : 'Send',
              icon: app.generating ? Icons.stop_rounded : Icons.arrow_upward_rounded,
              onPressed: !ready ? null : (app.generating ? app.stop : _send),
            ),
          ),
        ],
      ),
    );
  }
}

/// The sheet's lit edge: a glow in the flame's colours (cool to hot with the
/// phone's heat) and a light running along the top edge. It breathes faster
/// while the model thinks and calmly while it answers; at rest it stands still,
/// so the animation only costs power while the model works anyway.
class _EmberEdge extends StatefulWidget {
  const _EmberEdge({required this.state, required this.heat, required this.radius, required this.child});
  final FlameState state;
  final double heat;
  final double radius;
  final Widget child;

  @override
  State<_EmberEdge> createState() => _EmberEdgeState();
}

class _EmberEdgeState extends State<_EmberEdge> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this);

  @override
  void initState() {
    super.initState();
    _sync();
  }

  @override
  void didUpdateWidget(_EmberEdge old) {
    super.didUpdateWidget(old);
    if (old.state != widget.state) _sync();
  }

  void _sync() {
    final period = switch (widget.state) {
      FlameState.thinking => const Duration(milliseconds: 1300),
      FlameState.answering || FlameState.listening => const Duration(milliseconds: 2400),
      FlameState.resting => Duration.zero,
    };
    if (period == Duration.zero) {
      _c.stop();
      _c.value = 0;
    } else {
      _c
        ..duration = period
        ..repeat();
    }
  }

  @override
  void dispose() {
    _c.dispose();
    super.dispose();
  }

  // Cool (headroom) -> ember -> flare (near throttling), as the flame.
  Color get _glow => widget.heat < 0.5
      ? Color.lerp(const Color(0xFF8F9CFF), Palette.ember, widget.heat / 0.5)!
      : Color.lerp(Palette.ember, Palette.flare, (widget.heat - 0.5) / 0.5)!;

  @override
  Widget build(BuildContext context) => AnimatedBuilder(
    animation: _c,
    child: widget.child,
    builder: (context, child) {
      final working = widget.state != FlameState.resting;
      final pulse = working ? 0.5 - 0.5 * math.cos(2 * math.pi * _c.value) : 0.0;
      final glow = _glow;
      final radius = BorderRadius.vertical(top: Radius.circular(widget.radius));
      return DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: radius,
          boxShadow: [
            BoxShadow(
              color: glow.withValues(alpha: 0.28 + 0.32 * pulse),
              blurRadius: 44 + 36 * pulse,
              spreadRadius: -14 + 8 * pulse,
            ),
          ],
        ),
        child: ClipRRect(
          borderRadius: radius,
          child: Stack(
            children: [
              child!,
              if (working)
                Positioned(
                  left: 0,
                  right: 0,
                  top: 0,
                  height: 2.5,
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          glow.withValues(alpha: 0),
                          Palette.flare,
                          Palette.gold,
                          Palette.core,
                          glow.withValues(alpha: 0),
                        ],
                        stops: const [0, 0.3, 0.5, 0.7, 1],
                        transform: _Slide(_c.value),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      );
    },
  );
}

/// Moves a gradient across its box: phase 0..1 sweeps it from left to right.
class _Slide extends GradientTransform {
  const _Slide(this.phase);
  final double phase;

  @override
  Matrix4 transform(Rect bounds, {TextDirection? textDirection}) =>
      Matrix4.translationValues(bounds.width * (phase * 2 - 1), 0, 0);
}
