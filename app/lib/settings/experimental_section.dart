// Settings, Experimental: the engine's lossy, unfinished or measurement-only
// options (ExperimentalSettings), applied the next time the model loads.
import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../state/settings.dart';

class ExperimentalSection extends StatefulWidget {
  const ExperimentalSection({super.key, required this.app});
  final AppState app;

  @override
  State<ExperimentalSection> createState() => _ExperimentalSectionState();
}

class _ExperimentalSectionState extends State<ExperimentalSection> {
  late ExperimentalSettings _x = widget.app.device.experimental;
  bool _changed = false; // since the model loaded

  void _set(void Function(ExperimentalSettings x) change) => setState(() {
        change(_x);
        widget.app.device.experimental = _x;
        _changed = true;
      });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    final app = widget.app;

    Widget toggle(String title, String subtitle, bool value, void Function(bool) set) => SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: Text(title),
          subtitle: Text(subtitle, style: muted),
          value: value,
          onChanged: (v) => _set((_) => set(v)),
        );

    Widget slider(String label, double value, double min, double max, int divisions, String Function(double) show,
            void Function(double) set) =>
        Padding(
          padding: const EdgeInsets.only(left: 16),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [Expanded(child: Text(label, style: muted)), Text(show(value), style: theme.textTheme.labelSmall)]),
            Slider(value: value.clamp(min, max), min: min, max: max, divisions: divisions, onChanged: (v) => _set((_) => set(v))),
          ]),
        );

    Widget choice(String title, String subtitle, int value, List<(int, String)> options, void Function(int) set) =>
        ListTile(
          contentPadding: EdgeInsets.zero,
          title: Text(title),
          subtitle: Text(subtitle, style: muted),
          trailing: DropdownButton<int>(
            value: options.any((o) => o.$1 == value) ? value : options.first.$1,
            underline: const SizedBox.shrink(),
            items: [for (final (v, label) in options) DropdownMenuItem(value: v, child: Text(label))],
            onChanged: (v) => _set((_) => set(v ?? options.first.$1)),
          ),
        );

    return Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
      Text(
        'Engine features that trade quality for speed, are unfinished, or exist for measurements. They apply the next '
        'time the model loads; the Activity log lists the active ones.',
        style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant),
      ),
      const SizedBox(height: 8),
      Text('Speed versus quality (lossy)', style: theme.textTheme.titleSmall),
      toggle('Early exit', 'Stops the layer stack early on tokens the model is already sure about.', _x.earlyExit,
          (v) => _x.earlyExit = v),
      if (_x.earlyExit)
        slider('Confidence needed', _x.earlyExitThreshold, 0.90, 0.995, 19, (v) => v.toStringAsFixed(3),
            (v) => _x.earlyExitThreshold = v),
      toggle('Head pruning', 'While the phone is hot, skips the least important attention heads.', _x.headPruning,
          (v) => _x.headPruning = v),
      if (_x.headPruning)
        slider('Heads kept', _x.headKeepRatio, 0.5, 0.95, 9, (v) => '${(v * 100).round()}%', (v) => _x.headKeepRatio = v),
      toggle('Entropy-guided FFN skipping (EGLS)', 'Skips a block\'s feed-forward work when it barely changes the '
          'prediction.', _x.egls, (v) => _x.egls = v),
      if (_x.egls)
        slider('Skip below', _x.eglsThreshold, 0.0005, 0.01, 19, (v) => v.toStringAsFixed(4), (v) => _x.eglsThreshold = v),
      choice('2:4 sparse FFN (TDSS)', 'Half the FFN weights. Slower on phones today (no sparse units); for tests.',
          _x.tdss, const [(0, 'Off'), (1, 'While hot'), (2, 'Always')], (v) => _x.tdss = v),
      Text('Mixture of experts', style: theme.textTheme.titleSmall),
      slider('Expert mass (lossy below 1)', _x.expertMass, 0.5, 1.0, 10, (v) => v >= 1 ? 'All' : v.toStringAsFixed(2),
          (v) => _x.expertMass = v),
      choice('Experts per token', 'Fewer experts read less per token (lossy).', _x.maxExperts,
          const [(0, 'Model'), (2, '2'), (4, '4'), (6, '6')], (v) => _x.maxExperts = v),
      toggle('Skip slow, light experts', 'When an expert the router chose is still being read and weighs little for the '
          'token, answer without it instead of waiting (lossy).', _x.skipSlow > 0, (v) => _x.skipSlow = v ? 0.1 : 0),
      if (_x.skipSlow > 0)
        slider('Skip below', _x.skipSlow, 0.05, 0.3, 5, (v) => '${(v * 100).round()}% of the token', (v) => _x.skipSlow = v),
      choice('Requantize resident matrices', 'Q8_0 attention and shared weights to 4 or 5 bits at load (lossy). '
          'Auto: only when memory is too tight for the expert cache.',
          _x.requantBits, const [(-1, 'Auto'), (0, 'Off'), (4, 'Q4_K'), (5, 'Q5_K')], (v) => _x.requantBits = v),
      Text('Decoding', style: theme.textTheme.titleSmall),
      toggle('Lookup speculative decoding', 'Drafts tokens from the conversation\'s own text and verifies them in one '
          'pass. Exact output. The lossy features above do not apply with it.', _x.lookupDrafts, (v) => _x.lookupDrafts = v),
      if (_x.lookupDrafts)
        slider('Tokens per draft', _x.draftTokens.toDouble(), 1, 7, 6, (v) => '${v.round()}', (v) => _x.draftTokens = v.round()),
      choice('KV cache', 'Smaller caches hold longer contexts, with some loss. A change re-prepares the system prompt.',
          _x.kvCacheType, const [(1, 'Q8_0'), (0, 'F16'), (2, 'Q4_0'), (3, 'Q4_1')], (v) => _x.kvCacheType = v),
      choice('Threads', 'Cores the engine computes on.', _x.threads,
          const [(0, 'Auto'), (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8')], (v) => _x.threads = v),
      toggle('Persistent prefix KV cache', 'Keeps processed prompt prefixes on storage across runs (KV dedup).',
          _x.kvDedup, (v) => _x.kvDedup = v),
      const SizedBox(height: 8),
      Row(children: [
        FilledButton.tonal(
          onPressed: app.modelPath == null || app.loading || app.generating
              ? null
              : () {
                  setState(() => _changed = false);
                  app.load(app.modelPath!);
                },
          child: Text(_changed ? 'Reload the model to apply' : 'Reload the model'),
        ),
        const SizedBox(width: 12),
        TextButton(
          onPressed: () => _set((x) => _x = ExperimentalSettings()),
          child: const Text('Reset'),
        ),
      ]),
    ]);
  }
}
