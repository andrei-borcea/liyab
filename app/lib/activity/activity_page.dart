// What the phone is doing: live performance and the activity log.
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../state/app_state.dart';
import '../ui/theme.dart';

class ActivityPage extends StatelessWidget {
  const ActivityPage({super.key, required this.app});
  final AppState app;

  static const _thermal = ['cool', 'light', 'moderate', 'severe', 'critical', 'emergency', 'shutdown'];

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      appBar: AppBar(
        title: Text('Activity', style: theme.textTheme.titleLarge),
        actions: [
          IconButton(
            tooltip: 'Copy log',
            icon: const Icon(Icons.copy_rounded),
            onPressed: () {
              Clipboard.setData(ClipboardData(text: app.log.map((l) => '${_time(l.at)} ${l.text}').join('\n')));
              ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Log copied')));
            },
          ),
          IconButton(tooltip: 'Clear log', icon: const Icon(Icons.delete_sweep_outlined), onPressed: app.clearLog),
        ],
      ),
      body: ListenableBuilder(
        listenable: Listenable.merge([app, app.monitor]),
        builder: (context, _) {
          final s = app.monitor.latest;
          final history = app.monitor.history;
          return CustomScrollView(slivers: [
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 8),
              sliver: SliverGrid.count(
                crossAxisCount: 2,
                mainAxisSpacing: 10,
                crossAxisSpacing: 10,
                childAspectRatio: 1.7,
                children: [
                  _Meter('Speed', s == null ? '—' : '${s.tokensPerSecond.toStringAsFixed(1)} tok/s',
                      history.map((e) => e.tokensPerSecond).toList(), Palette.ember),
                  _Meter(
                      'Power',
                      s == null ? '—' : '${s.watts.toStringAsFixed(1)} W${s.charging ? ' (charging)' : ''}',
                      history.map((e) => e.watts).toList(),
                      Palette.flare),
                  _Meter('CPU', s == null ? '—' : '${s.cpuPercent.toStringAsFixed(0)} %',
                      history.map((e) => e.cpuPercent).toList(), Palette.gold),
                  _Meter('Storage reads', s == null ? '—' : '${s.flashMBps.toStringAsFixed(0)} MB/s',
                      history.map((e) => e.flashMBps).toList(), Palette.core),
                  _Meter(
                      'Heat',
                      s == null
                          ? '—'
                          : '${_thermal[s.thermalStatus.clamp(0, 6)]}, battery ${s.batteryC.toStringAsFixed(0)} °C',
                      history.map((e) => e.headroom.isNaN ? 0.0 : e.headroom).toList(),
                      Palette.flare),
                  _Meter('Energy', s?.joulesPerToken == null ? '—' : '${s!.joulesPerToken!.toStringAsFixed(2)} J/token',
                      history.map((e) => e.joulesPerToken ?? 0).toList(), Palette.ember),
                ],
              ),
            ),
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
                child: Text(
                  s?.charging ?? false
                      ? 'While charging, power is the flow into the battery, not what the phone uses.'
                      : 'GPU busy ${s?.gpuPercent.toStringAsFixed(0) ?? '0'} %. Energy is shown on battery while text is generated.',
                  style: theme.textTheme.labelSmall,
                ),
              ),
            ),
            SliverToBoxAdapter(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
                child: Text('Log', style: theme.textTheme.titleLarge),
              ),
            ),
            SliverList.builder(
              itemCount: app.log.length,
              itemBuilder: (context, i) {
                final l = app.log[app.log.length - 1 - i]; // newest first
                final warn = l.engine && (l.text.startsWith('W ') || l.text.startsWith('E '));
                return Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 3),
                  child: Text.rich(
                    TextSpan(children: [
                      TextSpan(text: '${_time(l.at)}  ', style: theme.textTheme.labelSmall),
                      TextSpan(
                        text: l.engine ? l.text.substring(2) : l.text,
                        style: theme.textTheme.bodyMedium?.copyWith(
                          fontSize: 13,
                          color: warn
                              ? theme.colorScheme.error
                              : l.engine
                                  ? theme.colorScheme.onSurfaceVariant
                                  : theme.colorScheme.onSurface,
                        ),
                      ),
                    ]),
                  ),
                );
              },
            ),
            const SliverPadding(padding: EdgeInsets.only(bottom: 32)),
          ]);
        },
      ),
    );
  }

  static String _time(DateTime t) =>
      '${t.hour.toString().padLeft(2, '0')}:${t.minute.toString().padLeft(2, '0')}:${t.second.toString().padLeft(2, '0')}';
}

class _Meter extends StatelessWidget {
  const _Meter(this.label, this.value, this.series, this.color);
  final String label;
  final String value;
  final List<double> series;
  final Color color;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.fromLTRB(14, 12, 14, 10),
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(18),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Text(label, style: theme.textTheme.labelSmall),
        const SizedBox(height: 2),
        Text(value,
            style: theme.textTheme.bodyLarge?.copyWith(
                fontWeight: FontWeight.w600, fontFeatures: const [FontFeature.tabularFigures()]),
            maxLines: 1,
            overflow: TextOverflow.ellipsis),
        const Spacer(),
        SizedBox(height: 26, width: double.infinity, child: CustomPaint(painter: _Spark(series, color))),
      ]),
    );
  }
}

/// A sparkline with a soft area fill and an emphasized latest point.
class _Spark extends CustomPainter {
  _Spark(this.values, this.color);
  final List<double> values;
  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    if (values.length < 2) return;
    final max = values.reduce((a, b) => a > b ? a : b);
    if (max <= 0) return;
    final dx = size.width / (DeviceMonitorWindow.points - 1);
    final start = size.width - dx * (values.length - 1);
    Offset at(int i) => Offset(start + i * dx, size.height - values[i] / max * (size.height - 2) - 1);
    final line = Path()..moveTo(at(0).dx, at(0).dy);
    for (var i = 1; i < values.length; i++) {
      line.lineTo(at(i).dx, at(i).dy);
    }
    final area = Path.from(line)
      ..lineTo(at(values.length - 1).dx, size.height)
      ..lineTo(start, size.height)
      ..close();
    canvas.drawPath(area, Paint()..color = color.withValues(alpha: 0.16));
    canvas.drawPath(
        line,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1.6);
    canvas.drawCircle(at(values.length - 1), 2.6, Paint()..color = color);
  }

  @override
  bool shouldRepaint(_Spark old) => true;
}

/// The sparkline width in samples (the monitor keeps 120 s).
abstract final class DeviceMonitorWindow {
  static const points = 120;
}
