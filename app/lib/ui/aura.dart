// The aura: a ring of the flame's colours turning around a rounded shape, with
// a soft glow outside it. It says what the assistant is doing, like the living
// flame it surrounds: at rest it barely breathes; while the user types it
// wakes; while the model reasons it turns fast and bright; while the answer
// streams it flows. Used around the desktop command bar.
//
// Like the flame it redraws at most ~30 times a second (24 at rest), never at
// the display's full rate: the engine uses all but one core.

import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/scheduler.dart';
import 'package:flutter/widgets.dart';

import 'living_flame.dart';
import 'theme.dart';

class Aura extends StatefulWidget {
  const Aura({super.key, required this.state, required this.borderRadius, required this.child, this.spread = 26});

  final FlameState state;
  final BorderRadius borderRadius;
  final Widget child;

  /// Room the glow takes outside the shape; the caller leaves this much margin.
  final double spread;

  @override
  State<Aura> createState() => _AuraState();
}

/// How the ring moves in each state: turns per second, brightness, glow.
typedef _Motion = ({double turns, double alpha, double glow});

class _AuraState extends State<Aura> with SingleTickerProviderStateMixin {
  late final Ticker _ticker = createTicker(_tick)..start();
  double _angle = 0, _t = 0;
  _Motion _now = _motion(FlameState.resting);
  Duration _last = Duration.zero, _painted = Duration.zero;

  static _Motion _motion(FlameState s) => switch (s) {
    FlameState.resting => (turns: 0.04, alpha: 0.38, glow: 0.22),
    FlameState.listening => (turns: 0.16, alpha: 0.75, glow: 0.42),
    FlameState.answering => (turns: 0.22, alpha: 0.85, glow: 0.5),
    FlameState.thinking => (turns: 0.55, alpha: 1.0, glow: 0.7),
  };

  void _tick(Duration elapsed) {
    final dt = ((elapsed - _last).inMicroseconds / 1e6).clamp(0.0, 0.05);
    _last = elapsed;
    if (MediaQuery.maybeDisableAnimationsOf(context) ?? false) return;
    final target = _motion(widget.state);
    // Eases toward the state's motion, so changes swell instead of jumping.
    final k = 1 - math.exp(-dt * 4);
    _now = (
      turns: _now.turns + (target.turns - _now.turns) * k,
      alpha: _now.alpha + (target.alpha - _now.alpha) * k,
      glow: _now.glow + (target.glow - _now.glow) * k,
    );
    _angle += 2 * math.pi * _now.turns * dt;
    _t += dt;
    final interval = widget.state == FlameState.resting ? 42 : 33;
    if (elapsed - _painted < Duration(milliseconds: interval)) return;
    _painted = elapsed;
    setState(() {});
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final breath = 0.5 + 0.5 * math.sin(_t * 2 * math.pi / 4.5); // a slow breath under every state
    _AuraPainter painter(bool ring) => _AuraPainter(
      ring: ring,
      angle: _angle,
      radius: widget.borderRadius,
      alpha: ring ? _now.alpha * (0.82 + 0.18 * breath) : _now.glow * (0.75 + 0.25 * breath),
      spread: widget.spread,
    );
    // The glow under the child (it shows through a translucent ground as an
    // inner light too), the ring over its edge.
    return CustomPaint(painter: painter(false), foregroundPainter: painter(true), child: widget.child);
  }
}

class _AuraPainter extends CustomPainter {
  _AuraPainter({
    required this.ring,
    required this.angle,
    required this.radius,
    required this.alpha,
    required this.spread,
  });

  final bool ring;
  final double angle, alpha, spread;
  final BorderRadius radius;

  static const _colors = [Palette.flare, Palette.ember, Palette.gold, Palette.core, Palette.ember, Palette.flare];

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Offset.zero & size;
    final shader = SweepGradient(
      colors: [for (final c in _colors) c.withValues(alpha: alpha.clamp(0.0, 1.0))],
      transform: GradientRotation(angle),
    ).createShader(rect.inflate(spread));
    final paint = Paint()
      ..shader = shader
      ..style = PaintingStyle.stroke;
    if (ring) {
      // A hairline of light on the edge.
      canvas.drawRRect(radius.toRRect(rect).deflate(0.75), paint..strokeWidth = 1.5);
    } else {
      // The same ring, wide and blurred, mostly outside the shape.
      canvas.drawRRect(
        radius.toRRect(rect).inflate(2),
        paint
          ..strokeWidth = 10
          ..maskFilter = ui.MaskFilter.blur(BlurStyle.normal, spread * 0.55),
      );
    }
  }

  @override
  bool shouldRepaint(_AuraPainter old) => old.angle != angle || old.alpha != alpha || old.radius != radius;
}
