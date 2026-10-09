// Generation settings of the loaded model, and device settings.
import 'package:flutter/material.dart';

import '../engine/engine_service.dart';
import '../state/app_state.dart';

Future<void> showSettingsSheet(BuildContext context, AppState app) => showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => _SettingsSheet(app: app),
    );

class _SettingsSheet extends StatefulWidget {
  const _SettingsSheet({required this.app});
  final AppState app;

  @override
  State<_SettingsSheet> createState() => _SettingsSheetState();
}

class _SettingsSheetState extends State<_SettingsSheet> {
  late final _system = TextEditingController(text: widget.app.settings.systemPrompt);

  @override
  void dispose() {
    _system.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final app = widget.app;
    final systemChanged = app.settings.systemPrompt != _system.text;
    app.settings.systemPrompt = _system.text;
    await app.saveSettings();
    if (systemChanged) await app.prepareSystemPrompt();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final app = widget.app;
    final s = app.settings;
    final muted = theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant);

    Widget slider(String label, double value, double min, double max, int divisions, String Function(double) show,
            void Function(double) set) =>
        Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [Expanded(child: Text(label)), Text(show(value), style: theme.textTheme.labelSmall)]),
          Slider(value: value.clamp(min, max), min: min, max: max, divisions: divisions, onChanged: (v) => setState(() => set(v))),
        ]);

    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.8,
      maxChildSize: 0.95,
      builder: (context, scroll) => ListView(controller: scroll, padding: const EdgeInsets.fromLTRB(20, 0, 20, 32), children: [
        Text('Settings', style: theme.textTheme.headlineMedium),
        const SizedBox(height: 4),
        Text(app.modelName.isEmpty ? 'No model loaded' : 'For ${app.modelName}', style: muted),
        const SizedBox(height: 16),
        if (app.thinkingSupported)
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Think before answering'),
            subtitle: const Text('Slower, better on hard questions.'),
            value: s.thinking,
            onChanged: (v) => setState(() => s.thinking = v),
          ),
        slider('Temperature', s.temperature, 0, 2, 40, (v) => v.toStringAsFixed(2), (v) => s.temperature = v),
        slider('Top-p', s.topP, 0.05, 1, 19, (v) => v.toStringAsFixed(2), (v) => s.topP = v),
        slider('Top-k', s.topK.toDouble(), 1, 100, 99, (v) => v.round().toString(), (v) => s.topK = v.round()),
        slider('Longest reply', s.maxTokens.toDouble(), 64, 4096, 63, (v) => '${v.round()} tokens',
            (v) => s.maxTokens = v.round()),
        slider('Context length', s.contextLength.toDouble(), 1024, 32768, 31, (v) => '${v.round()} tokens',
            (v) => s.contextLength = (v / 1024).round() * 1024),
        Text('Context length applies the next time the model loads.', style: muted),
        const SizedBox(height: 16),
        TextField(
          controller: _system,
          minLines: 2,
          maxLines: 6,
          decoration: const InputDecoration(labelText: 'System prompt', border: OutlineInputBorder()),
        ),
        const SizedBox(height: 24),
        Text('This phone', style: theme.textTheme.titleLarge),
        const SizedBox(height: 8),
        slider('Cool down above', app.device.thermalLimitC, 38, 60, 22, (v) => '${v.round()} °C',
            (v) => app.device.thermalLimitC = v.roundToDouble()),
        slider('Memory for the model', app.device.memoryBudgetMb.toDouble(), 2048, 12288, 40,
            (v) => '${(v / 1024).toStringAsFixed(1)} GB', (v) => app.device.memoryBudgetMb = (v / 256).round() * 256),
        Text('HyperOS and MIUI close apps above 6 GB. Both apply the next time the model loads.', style: muted),
        const SizedBox(height: 16),
        ExpansionTile(
          tilePadding: EdgeInsets.zero,
          title: const Text('About this device'),
          children: [
            Align(
              alignment: Alignment.centerLeft,
              child: SelectableText(
                app.description.isEmpty ? EngineService.deviceDescription() : app.description,
                style: theme.textTheme.labelSmall,
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),
        FilledButton(
          onPressed: () async {
            await _save();
            if (context.mounted) Navigator.pop(context);
          },
          child: const Text('Save'),
        ),
      ]),
    );
  }
}
