// The flame's lit edge around a surface while the model works: the assistant
// sheet, and the chat's composer.
import 'dart:math' as math;

import 'package:flutter/material.dart';

import 'living_flame.dart';
import 'theme.dart';

/// A lit edge: a glow in the flame's colours (cool to hot with the
/// phone's heat) and a light running along the top edge. It breathes faster
/// while the model thinks and calmly while it answers; at rest it stands still,
/// so the animation only costs power while the model works anyway.
class EmberEdge extends StatefulWidget {
  const EmberEdge({
    super.key,
    required this.state,
    required this.heat,
    required this.borderRadius,
    required this.child,
    this.outerGlow = true,
  });
  final FlameState state;

  /// 0 cool (plenty of thermal headroom) .. 1 hot (throttling).
  final double heat;

  /// The shape of `child`, which the glow and the running light follow.
  final BorderRadius borderRadius;
  final Widget child;

  /// The glow around the shape. Off where nothing can be drawn outside it
  /// (the desktop command bar's window is exactly its size): the running
  /// light along the top edge remains.
  final bool outerGlow;

  @override
  State<EmberEdge> createState() => _EmberEdgeState();
}

class _EmberEdgeState extends State<EmberEdge> with SingleTickerProviderStateMixin {
  late final AnimationController _c = AnimationController(vsync: this);

  @override
  void initState() {
    super.initState();
    _sync();
  }

  @override
  void didUpdateWidget(EmberEdge old) {
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
      final radius = widget.borderRadius;
      return DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: radius,
          boxShadow: [
            if (widget.outerGlow)
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
