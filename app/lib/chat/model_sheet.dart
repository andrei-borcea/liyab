// Choose and load a model stored on the phone; switch between CPU and GPU.
import 'package:flutter/material.dart';

import '../state/app_state.dart';
import '../state/models_store.dart';

Future<void> showModelSheet(BuildContext context, AppState app) => showModalBottomSheet<void>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => _ModelSheet(app: app),
    );

class _ModelSheet extends StatefulWidget {
  const _ModelSheet({required this.app});
  final AppState app;

  @override
  State<_ModelSheet> createState() => _ModelSheetState();
}

class _ModelSheetState extends State<_ModelSheet> {
  late Future<List<LocalModel>> _models = widget.app.localModels();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final app = widget.app;
    return DraggableScrollableSheet(
      expand: false,
      initialChildSize: 0.6,
      maxChildSize: 0.92,
      builder: (context, scroll) => ListenableBuilder(
        listenable: app,
        builder: (context, _) => ListView(controller: scroll, padding: const EdgeInsets.fromLTRB(20, 0, 20, 24), children: [
          Text('Models', style: theme.textTheme.headlineMedium),
          const SizedBox(height: 4),
          Text(app.status, style: theme.textTheme.bodyMedium?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
          const SizedBox(height: 12),
          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Run on the GPU'),
            subtitle: const Text('Vulkan. The CPU is usually faster for mixture-of-experts models.'),
            value: app.device.useGpu,
            onChanged: app.loading
                ? null
                : (v) {
                    app.device.useGpu = v;
                    if (app.modelPath != null) app.load(app.modelPath!);
                    setState(() {});
                  },
          ),
          const Divider(height: 24),
          FutureBuilder<List<LocalModel>>(
            future: _models,
            builder: (context, snap) {
              if (!snap.hasData) return const Padding(padding: EdgeInsets.all(24), child: Center(child: CircularProgressIndicator()));
              final models = snap.data!;
              if (models.isEmpty) {
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  child: Text(
                    'No models on this phone yet. Downloads from Hugging Face come in the next update; '
                    'for now, push a GGUF file with adb into the app folder.',
                    style: theme.textTheme.bodyMedium,
                  ),
                );
              }
              return Column(children: [
                for (final m in models)
                  ListTile(
                    contentPadding: EdgeInsets.zero,
                    leading: Icon(
                      app.modelPath == m.file.path ? Icons.radio_button_checked : Icons.radio_button_off,
                      color: app.modelPath == m.file.path ? theme.colorScheme.primary : theme.colorScheme.onSurfaceVariant,
                    ),
                    title: Text(m.name, maxLines: 2, overflow: TextOverflow.ellipsis),
                    subtitle: Text(app.moving == m.name
                        ? 'Moving to app storage…'
                        : m.shared
                            ? '${formatBytes(m.bytes)}, shared folder (slower streaming)'
                            : formatBytes(m.bytes)),
                    trailing: !m.shared
                        ? null
                        : app.moving == m.name
                            ? const SizedBox.square(dimension: 22, child: CircularProgressIndicator(strokeWidth: 2.5))
                            : IconButton(
                                tooltip: 'Move to app storage',
                                icon: const Icon(Icons.drive_file_move_outline),
                                onPressed: app.moving != null || app.modelPath == m.file.path
                                    ? null
                                    : () async {
                                        await app.moveToAppStorage(m);
                                        if (mounted) setState(() => _models = app.localModels());
                                      },
                              ),
                    onTap: app.loading || app.moving == m.name
                        ? null
                        : () {
                            Navigator.pop(context);
                            app.load(m.file.path);
                          },
                  ),
              ]);
            },
          ),
          TextButton.icon(
            onPressed: () => setState(() => _models = app.localModels()),
            icon: const Icon(Icons.refresh_rounded),
            label: const Text('Refresh'),
          ),
        ]),
      ),
    );
  }
}
