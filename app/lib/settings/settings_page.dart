// Settings: the loaded model's generation settings, this phone's limits, and
// the permissions Liyab uses.
import 'package:background_downloader/background_downloader.dart';
import 'package:flutter/material.dart';

import '../engine/engine_service.dart';
import '../state/app_state.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key, required this.app});
  final AppState app;

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late final _system = TextEditingController(text: widget.app.settings.systemPrompt);
  late Future<PermissionStatus> _notifications = widget.app.downloads.notificationPermission();
  bool _dirty = false;

  AppState get app => widget.app;

  @override
  void dispose() {
    _system.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    final systemChanged = app.settings.systemPrompt != _system.text;
    app.settings.systemPrompt = _system.text;
    await app.saveSettings();
    if (systemChanged) await app.prepareSystemPrompt();
    if (!mounted) return;
    setState(() => _dirty = false);
    ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Settings saved')));
  }

  void _set(VoidCallback change) => setState(() {
        change();
        _dirty = true;
      });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final s = app.settings;
    final muted = theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant);

    Widget slider(String label, double value, double min, double max, int divisions, String Function(double) show,
            void Function(double) set) =>
        Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [Expanded(child: Text(label)), Text(show(value), style: theme.textTheme.labelSmall)]),
          Slider(value: value.clamp(min, max), min: min, max: max, divisions: divisions, onChanged: (v) => _set(() => set(v))),
        ]);

    Widget heading(String text) => Padding(
          padding: const EdgeInsets.only(top: 24, bottom: 8),
          child: Text(text, style: theme.textTheme.titleLarge),
        );

    return Scaffold(
      appBar: AppBar(title: Text('Settings', style: theme.textTheme.titleLarge)),
      floatingActionButton: _dirty
          ? FloatingActionButton.extended(onPressed: _save, icon: const Icon(Icons.check_rounded), label: const Text('Save'))
          : null,
      body: ListView(padding: const EdgeInsets.fromLTRB(20, 0, 20, 96), children: [
        heading('Answers'),
        Text(app.modelName.isEmpty ? 'Load a model to change how it answers.' : 'For ${app.modelName}', style: muted),
        if (app.modelName.isNotEmpty) ...[
          const SizedBox(height: 8),
          if (app.thinkingSupported)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('Think before answering'),
              subtitle: const Text('Slower, better on hard questions.'),
              value: s.thinking,
              onChanged: (v) => _set(() => s.thinking = v),
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
            onChanged: (_) => setState(() => _dirty = true),
            decoration: const InputDecoration(labelText: 'System prompt', border: OutlineInputBorder()),
          ),
        ],
        heading('This phone'),
        SwitchListTile(
          contentPadding: EdgeInsets.zero,
          title: const Text('Run on the GPU'),
          subtitle: const Text('Vulkan. The CPU is usually faster for mixture-of-experts models. Reloads the model.'),
          value: app.device.useGpu,
          onChanged: app.loading
              ? null
              : (v) {
                  setState(() => app.device.useGpu = v);
                  if (app.modelPath != null) app.load(app.modelPath!);
                },
        ),
        slider('Cool down above', app.device.thermalLimitC, 38, 60, 22, (v) => '${v.round()} °C',
            (v) => app.device.thermalLimitC = v.roundToDouble()),
        slider('Memory for the model', app.device.memoryBudgetMb.toDouble(), 2048, 12288, 40,
            (v) => '${(v / 1024).toStringAsFixed(1)} GB', (v) => app.device.memoryBudgetMb = (v / 256).round() * 256),
        Text('HyperOS and MIUI close apps that use more than 6 GB. Both apply the next time the model loads.',
            style: muted),
        heading('Permissions'),
        FutureBuilder<PermissionStatus>(
          future: _notifications,
          builder: (context, snap) {
            final granted = snap.data == PermissionStatus.granted;
            return ListTile(
              contentPadding: EdgeInsets.zero,
              leading: Icon(granted ? Icons.notifications_active_outlined : Icons.notifications_off_outlined),
              title: const Text('Notifications'),
              subtitle: Text(granted
                  ? 'Allowed: downloads show their progress.'
                  : 'Off: downloads still run, without a progress notification.'),
              trailing: granted
                  ? null
                  : TextButton(
                      onPressed: () async {
                        await app.downloads.ensureNotificationPermission();
                        setState(() => _notifications = app.downloads.notificationPermission());
                      },
                      child: const Text('Allow'),
                    ),
            );
          },
        ),
        const ListTile(
          contentPadding: EdgeInsets.zero,
          leading: Icon(Icons.folder_outlined),
          title: Text('Files'),
          subtitle: Text('Not needed. Models live in Liyab\'s own storage; importing uses the system file picker.'),
        ),
        const ListTile(
          contentPadding: EdgeInsets.zero,
          leading: Icon(Icons.public_outlined),
          title: Text('Internet'),
          subtitle: Text('Only to search and download models from Hugging Face. Conversations never leave the phone.'),
        ),
        heading('About this device'),
        SelectableText(
          app.description.isEmpty ? EngineService.deviceDescription() : app.description,
          style: theme.textTheme.labelSmall,
        ),
      ]),
    );
  }
}
