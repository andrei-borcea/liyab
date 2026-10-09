// The living flame: Liyab's mark (docs/brand/liyab-mark.svg) drawn live as the
// assistant's presence. Its motion shows what the assistant is doing; its
// colour shows how warm the phone is (cool when there is thermal headroom,
// hot near throttling), so the work happening on the device stays visible.
import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/scheduler.dart';

enum FlameState { resting, listening, thinking, answering }

class LivingFlame extends StatefulWidget {
  const LivingFlame({
    super.key,
    this.size = 120,
    this.state = FlameState.resting,
    this.heat = 0.3,
    this.voiceLevel = 0,
  });

  final double size;
  final FlameState state;

  /// 0 cool (plenty of thermal headroom) .. 1 hot (throttling).
  final double heat;

  /// 0..1 microphone level while listening.
  final double voiceLevel;

  @override
  State<LivingFlame> createState() => _LivingFlameState();
}

class _Spark {
  _Spark(this.x, this.y, this.vx, this.vy);
  double x, y, vx, vy, life = 1;
}

class _LivingFlameState extends State<LivingFlame> with SingleTickerProviderStateMixin {
  late final Ticker _ticker;
  final _random = math.Random();
  final _sparks = <_Spark>[];
  double _t = 0;
  double _voice = 0;
  double _heat = 0.3;
  Duration _last = Duration.zero;
  Duration _painted = Duration.zero;

  @override
  void initState() {
    super.initState();
    _heat = widget.heat;
    _ticker = createTicker(_tick)..start();
  }

  void _tick(Duration elapsed) {
    final dt = ((elapsed - _last).inMicroseconds / 1e6).clamp(0.0, 0.05);
    _last = elapsed;
    if (MediaQuery.maybeDisableAnimationsOf(context) ?? false) return;
    _t += dt;
    // At rest the flame only breathes (~24 frames a second); while the model
    // works ~30 are enough. The engine uses all but one core, so the UI should
    // not draw at the display's full rate.
    final interval = widget.state == FlameState.resting && _sparks.isEmpty ? 42 : 33;
    if (elapsed - _painted < Duration(milliseconds: interval)) return;
    _painted = elapsed;
    setState(() {
      _voice += ((widget.state == FlameState.listening ? widget.voiceLevel : 0) - _voice) * 0.25;
      _heat += (widget.heat - _heat) * 0.03; // colour drifts, never jumps
      final emit = switch (widget.state) {
        FlameState.thinking => 1.6,
        FlameState.answering => 0.5,
        FlameState.listening => 0.12,
        FlameState.resting => 0.03,
      };
      if (_random.nextDouble() < emit) {
        final n = widget.state == FlameState.thinking ? 3 : 1;
        for (var i = 0; i < n; i++) {
          _sparks.add(_Spark(262 + (_random.nextDouble() - 0.5) * 30, 356, (_random.nextDouble() - 0.5) * 1.6,
              -(1.6 + _random.nextDouble() * 3.2)));
        }
      }
      for (final s in _sparks) {
        s.x += s.vx + math.sin((s.y + _t * 40) / 18) * 0.6;
        s.y += s.vy;
        s.life -= 0.012;
      }
      _sparks.removeWhere((s) => s.life <= 0);
    });
  }

  @override
  void dispose() {
    _ticker.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => RepaintBoundary(
        child: CustomPaint(
          size: Size.square(widget.size),
          painter: _FlamePainter(t: _t, state: widget.state, voice: _voice, heat: _heat, sparks: _sparks),
        ),
      );
}

class _FlamePainter extends CustomPainter {
  _FlamePainter({required this.t, required this.state, required this.voice, required this.heat, required this.sparks});

  final double t, voice, heat;
  final FlameState state;
  final List<_Spark> sparks;

  // The mark's paths in its 512-unit viewport.
  static final Path _left = Path()
    ..moveTo(246, 48)
    ..cubicTo(166, 144, 132, 220, 132, 302)
    ..cubicTo(132, 386, 192, 446, 262, 458)
    ..cubicTo(212, 428, 186, 378, 190, 322)
    ..cubicTo(196, 252, 252, 196, 246, 48)
    ..close();
  static final Path _right = Path()
    ..moveTo(304, 144)
    ..cubicTo(364, 210, 386, 262, 386, 322)
    ..cubicTo(386, 396, 332, 450, 262, 458)
    ..cubicTo(308, 430, 332, 390, 328, 344)
    ..cubicTo(324, 290, 296, 236, 304, 144)
    ..close();

  // Colour ramps: cool -> warm -> hot, per gradient stop.
  static const _ramps = <List<Color>>[
    [Color(0xFF6C7BFF), Color(0xFFFF3D6E), Color(0xFFE3263B)], // left, base
    [Color(0xFF9C8CFF), Color(0xFFFF8A3D), Color(0xFFFF5A2A)], // left, tip
    [Color(0xFF8FA6FF), Color(0xFFFF7A3D), Color(0xFFFF4A2E)], // right, base
    [Color(0xFFB9D4FF), Color(0xFFFFC24D), Color(0xFFFF9A3D)], // right, tip
    [Color(0xFFBFE6FF), Color(0xFF9ED8FF), Color(0xFFFFE0B0)], // core edge
  ];

  Color _ramp(int i) => heat < 0.5
      ? Color.lerp(_ramps[i][0], _ramps[i][1], heat / 0.5)!
      : Color.lerp(_ramps[i][1], _ramps[i][2], (heat - 0.5) / 0.5)!;

  @override
  void paint(Canvas canvas, Size size) {
    final k = size.width / 512;
    canvas.scale(k);
    final rate = switch (state) {
      FlameState.resting => 0.7,
      FlameState.listening => 1.4,
      FlameState.thinking => 2.2,
      FlameState.answering => 3.4,
    };
    final amp = switch (state) {
      FlameState.resting => 1.2,
      FlameState.listening => 2.2,
      FlameState.thinking => 2.6,
      FlameState.answering => 3.2,
    };
    final lean = state == FlameState.listening ? 3 + voice * 4 : 0.0;
    final flick = state == FlameState.answering ? math.sin(t * 17) * 1.4 : 0.0;
    final sway = math.sin(t * rate) * amp;
    final coreR = 36 + (state == FlameState.listening ? voice * 16 : 0) +
        math.sin(t * rate * 2) * (state == FlameState.resting ? 1.5 : 2.5);

    // Halo
    final haloR = 110 + coreR * 1.2 + heat * 30;
    canvas.drawCircle(
      const Offset(262, 356),
      haloR,
      Paint()
        ..shader = RadialGradient(colors: [
          _ramp(1).withValues(alpha: 0.30 + heat * 0.25),
          _ramp(1).withValues(alpha: 0),
        ]).createShader(Rect.fromCircle(center: const Offset(262, 356), radius: haloR)),
    );

    void blade(Path path, double degrees, int base, int tip, Rect bounds) {
      canvas.save();
      canvas.translate(262, 458);
      canvas.rotate(degrees * math.pi / 180);
      canvas.translate(-262, -458);
      canvas.drawPath(
        path,
        Paint()
          ..shader = LinearGradient(
            begin: Alignment.bottomLeft,
            end: Alignment.topCenter,
            colors: [_ramp(base), _ramp(tip)],
          ).createShader(bounds),
      );
      canvas.restore();
    }

    blade(_left, sway - lean + flick, 0, 1, const Rect.fromLTRB(132, 48, 262, 458));
    blade(_right, -sway * 0.8 + lean - flick, 2, 3, const Rect.fromLTRB(262, 144, 386, 458));

    canvas.drawCircle(
      const Offset(262, 356),
      coreR,
      Paint()
        ..shader = RadialGradient(
          center: const Alignment(-0.2, -0.3),
          colors: [Colors.white, _ramp(4)],
        ).createShader(Rect.fromCircle(center: const Offset(262, 356), radius: coreR)),
    );

    for (final s in sparks) {
      final c = heat > 0.5
          ? Color.fromRGBO(255, (150 + 80 * s.life).round(), 90, s.life)
          : Color.fromRGBO((170 + 70 * s.life).round(), (190 + 50 * s.life).round(), 255, s.life);
      canvas.drawCircle(Offset(s.x, s.y), 2.4 + s.life * 3.6, Paint()..color = c);
    }
  }

  @override
  bool shouldRepaint(_FlamePainter old) => true;
}
