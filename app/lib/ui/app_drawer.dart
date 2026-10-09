// The navigation drawer: the model in use, then the app's pages.
import 'package:flutter/material.dart';

import '../activity/activity_page.dart';
import '../models/models_page.dart';
import '../settings/settings_page.dart';
import '../state/app_state.dart';
import 'living_flame.dart';

class AppDrawer extends StatelessWidget {
  const AppDrawer({super.key, required this.app});
  final AppState app;

  void _open(BuildContext context, Widget page) {
    Navigator.pop(context);
    Navigator.push(context, MaterialPageRoute<void>(builder: (_) => page));
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return NavigationDrawer(
      backgroundColor: theme.colorScheme.surface,
      onDestinationSelected: (i) => switch (i) {
        0 => Navigator.pop(context),
        1 => _open(context, ModelsPage(app: app)),
        2 => _open(context, ActivityPage(app: app)),
        _ => _open(context, SettingsPage(app: app)),
      },
      selectedIndex: 0,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 16),
          child: Row(children: [
            LivingFlame(size: 44, heat: app.monitor.heat),
            const SizedBox(width: 12),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Liyab', style: theme.textTheme.headlineMedium),
                Text(app.modelName.isEmpty ? 'No model loaded' : app.modelName,
                    style: theme.textTheme.labelSmall, overflow: TextOverflow.ellipsis),
              ]),
            ),
          ]),
        ),
        const NavigationDrawerDestination(icon: Icon(Icons.chat_bubble_outline_rounded), label: Text('Chat')),
        const NavigationDrawerDestination(icon: Icon(Icons.inventory_2_outlined), label: Text('Models')),
        const NavigationDrawerDestination(icon: Icon(Icons.monitor_heart_outlined), label: Text('Activity')),
        const NavigationDrawerDestination(icon: Icon(Icons.tune_rounded), label: Text('Settings')),
        Padding(
          padding: const EdgeInsets.fromLTRB(28, 24, 28, 16),
          child: Text('Everything runs on this phone. Nothing you write leaves it.', style: theme.textTheme.labelSmall),
        ),
      ],
    );
  }
}
