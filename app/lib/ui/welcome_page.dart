// First launch: what Liyab is, the one permission it asks for, and a first model.
import 'package:flutter/material.dart';

import '../models/models_page.dart';
import '../state/app_state.dart';
import 'living_flame.dart';

class WelcomePage extends StatefulWidget {
  const WelcomePage({super.key, required this.app});
  final AppState app;

  @override
  State<WelcomePage> createState() => _WelcomePageState();
}

class _WelcomePageState extends State<WelcomePage> {
  FlameState _flame = FlameState.resting;

  Future<void> _start() async {
    setState(() => _flame = FlameState.thinking);
    await widget.app.downloads.ensureNotificationPermission();
    widget.app.welcomed = true;
    if (!mounted) return;
    final models = await widget.app.localModels();
    if (models.isEmpty && mounted) {
      await Navigator.push(context, MaterialPageRoute<void>(builder: (_) => ModelsPage(app: widget.app)));
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final muted = theme.textTheme.bodyLarge?.copyWith(color: theme.colorScheme.onSurfaceVariant);
    Widget point(IconData icon, String title, String body) => Padding(
          padding: const EdgeInsets.only(bottom: 18),
          child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Icon(icon, color: theme.colorScheme.primary),
            const SizedBox(width: 14),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(title, style: theme.textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
                Text(body, style: muted),
              ]),
            ),
          ]),
        );
    return Scaffold(
      body: SafeArea(
        child: ListView(padding: const EdgeInsets.fromLTRB(28, 32, 28, 32), children: [
          Center(child: LivingFlame(size: 180, state: _flame, heat: widget.app.monitor.heat)),
          const SizedBox(height: 8),
          Text('Liyab', style: theme.textTheme.headlineMedium?.copyWith(fontSize: 44), textAlign: TextAlign.center),
          const SizedBox(height: 4),
          Text('Your assistant, on your phone.', style: muted, textAlign: TextAlign.center),
          const SizedBox(height: 36),
          point(Icons.lock_outline_rounded, 'Private by design',
              'The model runs on this phone. What you write and what it answers never leave it.'),
          point(Icons.download_rounded, 'Bring your own model',
              'Download open models from Hugging Face, or import a GGUF file. Large ones stream from storage.'),
          point(Icons.notifications_none_rounded, 'One permission',
              'Notifications, so a download can show its progress while you use other apps.'),
          const SizedBox(height: 12),
          FilledButton(
            onPressed: _start,
            style: FilledButton.styleFrom(minimumSize: const Size.fromHeight(52)),
            child: const Text('Get started'),
          ),
        ]),
      ),
    );
  }
}
