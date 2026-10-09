// Settings: the loaded model's generation settings, this phone's limits, and
// the permissions Liyab uses.
import 'package:background_downloader/background_downloader.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart' show openAppSettings;

import '../agent/tools.dart';
import '../engine/engine_service.dart';
import '../state/app_state.dart';
import 'experimental_section.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({super.key, required this.app});
  final AppState app;

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> with WidgetsBindingObserver {
  /// Per tool: whether its permission is granted (re-checked when the user comes back from Android's settings).
  final Map<String, bool> _permitted = {};

  Future<void> _checkTools() async {
    for (final t in app.toolbox.all) {
      _permitted[t.name] = await t.permitted();
    }
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _checkTools();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) {
      _checkTools();
      setState(() => _assistant = _isAssistant());
    }
  }

  Future<void> _toggleTool(AgentTool tool, bool on) async {
    if (on && !(_permitted[tool.name] ?? false)) {
      final granted = await tool.requestPermission();
      _permitted[tool.name] = granted;
      if (!granted && mounted && tool is! NotificationsTool) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(
          content: const Text('Android did not grant access. You can allow it in the app settings.'),
          action: SnackBarAction(label: 'Open', onPressed: openAppSettings),
        ));
      }
    }
    await app.toolbox.setOn(tool, on);
    setState(() {});
    await app.toolsChanged();
  }
  late final _system = TextEditingController(text: widget.app.settings.systemPrompt);
  late Future<PermissionStatus> _notifications = widget.app.downloads.notificationPermission();
  static const _device = MethodChannel('liyab/device');
  late Future<bool> _assistant = _isAssistant();

  static Future<bool> _isAssistant() async {
    try {
      return await _device.invokeMethod<bool>('isAssistant') ?? false;
    } on PlatformException {
      return false;
    } on MissingPluginException {
      return false;
    }
  }
  bool _dirty = false;

  static const _releaseChoices = [1, 5, 15, 60, 0]; // minutes; 0 = never
  static String _releaseLabel(int minutes) => switch (minutes) {
        0 => 'Never',
        60 => 'After 1 hour',
        _ => 'After $minutes min',
      };

  AppState get app => widget.app;

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
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
        const SizedBox(height: 8),
        Text('Speed or coolness'),
        const SizedBox(height: 8),
        SegmentedButton<int>(
          segments: const [
            ButtonSegment(value: 0, label: Text('Fastest')),
            ButtonSegment(value: 1, label: Text('Balanced')),
            ButtonSegment(value: 2, label: Text('Coolest')),
          ],
          selected: {app.device.powerProfile},
          onSelectionChanged: (v) => setState(() => app.setPowerProfile(v.first)),
        ),
        const SizedBox(height: 4),
        Text(
          const [
            'Full speed; slows down only near Android\'s own heat limit.',
            'Answers at reading speed or faster; as the phone warms up, tokens come a little slower at lower clocks.',
            'Reacts to heat earliest, on half the cores: the phone stays cool, answers come slower.',
          ][app.device.powerProfile.clamp(0, 2)],
          style: muted,
        ),
        const SizedBox(height: 8),
        slider('Cool down above', app.device.thermalLimitC, 38, 60, 22, (v) => '${v.round()} °C',
            (v) => app.device.thermalLimitC = v.roundToDouble()),
        slider('Memory for the model', app.device.memoryBudgetMb.toDouble(), 2048, 12288, 40,
            (v) => '${(v / 1024).toStringAsFixed(1)} GB', (v) => app.device.memoryBudgetMb = (v / 256).round() * 256),
        Text('HyperOS and MIUI close apps that use more than 6 GB. Both apply the next time the model loads.',
            style: muted),
        const SizedBox(height: 12),
        slider(
            'Free memory when idle',
            _releaseChoices.indexOf(app.device.releaseAfterMinutes).clamp(0, _releaseChoices.length - 1).toDouble(),
            0,
            _releaseChoices.length - 1.0,
            _releaseChoices.length - 1,
            (v) => _releaseLabel(_releaseChoices[v.round()]),
            (v) => app.device.releaseAfterMinutes = _releaseChoices[v.round()]),
        Text('After this long in the background Liyab unloads the model; it loads again when you open Liyab.',
            style: muted),
        heading('Assistant'),
        FutureBuilder<bool>(
          future: _assistant,
          builder: (context, snap) {
            final on = snap.data ?? false;
            return ListTile(
              contentPadding: EdgeInsets.zero,
              leading: Icon(on ? Icons.assistant_rounded : Icons.assistant_outlined),
              title: const Text('Open Liyab from any app'),
              subtitle: Text(on
                  ? 'Liyab is your digital assistant: hold the power button to ask it anything.'
                  : 'Make Liyab the default digital assistant, then hold the power button to open it over any app.'),
              trailing: on
                  ? null
                  : TextButton(
                      onPressed: () async {
                        await _device.invokeMethod<void>('openAssistantSettings');
                        setState(() => _assistant = _isAssistant());
                      },
                      child: const Text('Set up'),
                    ),
            );
          },
        ),
        heading('What Liyab can read'),
        Text(
          app.toolDialect.name == 'none' && app.modelName.isNotEmpty
              ? 'The loaded model cannot call tools; Qwen3, Qwen3.5 and Qwen3.6 can.'
              : 'Turn on what the assistant may read to answer about your day. It reads on this phone, only when a '
                  'question needs it, and nothing leaves the device.',
          style: muted,
        ),
        for (final t in app.toolbox.all)
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            secondary: Icon(_toolIcon(t)),
            title: Text(_toolTitle(t)),
            subtitle: Text(app.toolbox.isOn(t) && !(_permitted[t.name] ?? true)
                ? 'Waiting for Android\'s permission: tap to ask again.'
                : _toolExample(t)),
            value: app.toolbox.isOn(t) && (_permitted[t.name] ?? false),
            onChanged: (v) => _toggleTool(t, v),
          ),
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
        heading('Experimental'),
        ExperimentalSection(app: app),
        heading('About this device'),
        SelectableText(
          app.description.isEmpty ? EngineService.deviceDescription() : app.description,
          style: theme.textTheme.labelSmall,
        ),
      ]),
    );
  }
}

IconData _toolIcon(AgentTool t) => switch (t) {
      CalendarTool() => Icons.event_outlined,
      NotificationsTool() => Icons.notifications_outlined,
      MessagesTool() => Icons.sms_outlined,
      CallsTool() => Icons.call_outlined,
      ContactsTool() => Icons.contacts_outlined,
      _ => Icons.content_paste_outlined,
    };

String _toolTitle(AgentTool t) => switch (t) {
      CalendarTool() => 'Calendar',
      NotificationsTool() => 'Notifications (chats, email previews)',
      MessagesTool() => 'Text messages',
      CallsTool() => 'Calls',
      ContactsTool() => 'Contacts',
      _ => 'Clipboard',
    };

String _toolExample(AgentTool t) => switch (t) {
      CalendarTool() => '"What do I have today?" "Am I free at 5?"',
      NotificationsTool() => '"What did Marco write me?" "Any new email?" Android asks in its settings.',
      MessagesTool() => '"What did the bank\'s SMS say?"',
      CallsTool() => '"Who called me this morning?"',
      ContactsTool() => '"What is Anna\'s number?"',
      _ => '"Summarize what I copied." Android shows a notice when it is read.',
    };
