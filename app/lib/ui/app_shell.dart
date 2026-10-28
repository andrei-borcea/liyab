// Shared navigation and layout for every desktop platform and compact screens.
import 'package:flutter/material.dart';

import '../desktop/desktop_host.dart';
import '../activity/activity_page.dart';
import '../chat/chat_screen.dart';
import '../chat/chats_page.dart';
import '../models/models_page.dart';
import '../settings/settings_page.dart';
import '../state/app_state.dart';
import 'theme.dart';
import 'liyab_mark.dart';

class AppShell extends StatefulWidget {
  const AppShell({super.key, required this.app});
  final AppState app;

  @override
  State<AppShell> createState() => _AppShellState();
}

class _AppShellState extends State<AppShell> {
  int _selected = 0;
  // Build pages on first visit: opening chat must not query the model hub.
  final Map<int, Widget> _pages = {};
  static const _labels = ['Workspace', 'Chats', 'Models', 'Activity', 'Settings'];
  static const _icons = [Icons.chat_bubble_outline_rounded, Icons.forum_outlined, Icons.layers_outlined,
    Icons.insights_rounded, Icons.tune_rounded];

  @override
  void initState() {
    super.initState();
    DesktopHost.navigate = _select;
  }

  @override
  void dispose() {
    DesktopHost.navigate = null;
    super.dispose();
  }

  void _select(int value) => setState(() => _selected = value);

  Widget _page(int index) => _pages.putIfAbsent(index, () => switch (index) {
    0 => ChatScreen(app: widget.app, embedded: true, onModels: () => _select(2)),
    1 => ChatsPage(app: widget.app, onOpen: () => _select(0)),
    2 => ModelsPage(app: widget.app, onLoaded: () => _select(0)),
    3 => ActivityPage(app: widget.app),
    _ => SettingsPage(app: widget.app),
  });

  @override
  Widget build(BuildContext context) {
    _page(_selected);
    return LayoutBuilder(builder: (context, constraints) {
      final wide = constraints.maxWidth >= 760;
      final content = IndexedStack(index: _selected, children: [
        for (var i = 0; i < 5; i++) TickerMode(enabled: i == _selected,
          child: _pages[i] ?? const SizedBox.shrink()),
      ]);
      return Scaffold(
        body: Row(children: [
          if (wide) _sidebar(context),
          Expanded(child: Center(child: ConstrainedBox(constraints: const BoxConstraints(maxWidth: 1200), child: content))),
        ]),
        bottomNavigationBar: wide ? null : NavigationBar(
          selectedIndex: _selected,
          onDestinationSelected: _select,
          destinations: [for (var i = 0; i < 5; i++)
            NavigationDestination(icon: Icon(_icons[i]), label: _labels[i])],
        ),
      );
    });
  }

  Widget _sidebar(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      width: 216,
      decoration: BoxDecoration(
        color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.45),
        border: Border(right: BorderSide(color: theme.colorScheme.outline)),
      ),
      child: SafeArea(child: Padding(
        padding: const EdgeInsets.fromLTRB(16, 24, 16, 18),
        child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
          Expanded(child: ListView(children: [
          Row(children: [const LiyabMark(size: 36), const SizedBox(width: 10),
            Expanded(child: Text('liyab', style: theme.textTheme.headlineMedium, maxLines: 1, overflow: TextOverflow.ellipsis)),
          ]),
          const SizedBox(height: 6),
          Text('YOUR LOCAL WORKSPACE', style: theme.textTheme.labelSmall?.copyWith(fontSize: 10, letterSpacing: 1.5)),
          const SizedBox(height: 30),
          ListenableBuilder(listenable: widget.app, builder: (context, _) => OutlinedButton.icon(
            onPressed: widget.app.generating || widget.app.loading || widget.app.switchingChat ? null : () { widget.app.newChat(); _select(0); },
            icon: const Icon(Icons.add_rounded, size: 18), label: const Text('New chat'),
            style: OutlinedButton.styleFrom(padding: const EdgeInsets.all(16),
              side: BorderSide(color: theme.colorScheme.outline),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
          )),
          const SizedBox(height: 24),
          for (var i = 0; i < 5; i++) Padding(
            padding: const EdgeInsets.only(bottom: 6),
            child: Material(color: i == _selected ? Palette.ember.withValues(alpha: 0.12) : Colors.transparent,
              borderRadius: BorderRadius.circular(10),
              child: ListTile(dense: true, selected: i == _selected,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                selectedColor: Palette.ember,
                leading: Icon(_icons[i], size: 19), title: Text(_labels[i]), onTap: () => _select(i)),
            ),
          ),
          ])),
          ValueListenableBuilder<String?>(valueListenable: DesktopHost.problem,
            builder: (context, message, _) => message == null ? const SizedBox.shrink() :
              Padding(padding: const EdgeInsets.only(bottom: 12), child: Text(message, style: theme.textTheme.labelSmall))),
          ListenableBuilder(listenable: widget.app, builder: (context, _) {
            final app = widget.app;
            return Container(padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: theme.colorScheme.surfaceContainerHighest,
                border: Border.all(color: theme.colorScheme.outline), borderRadius: BorderRadius.circular(12)),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(children: [Icon(Icons.circle, size: 7, color: app.modelPath == null ? Palette.smoke : Palette.ember),
                  const SizedBox(width: 7), Expanded(child: Text(app.generating ? 'Generating' : app.loading ? 'Loading model' : 'Local engine', maxLines: 1, overflow: TextOverflow.ellipsis, style: theme.textTheme.labelSmall))]),
                const SizedBox(height: 8),
                Text(app.modelName.isEmpty ? 'No model loaded' : app.modelName,
                  maxLines: 2, overflow: TextOverflow.ellipsis, style: theme.textTheme.bodySmall),
                if (app.modelPath != null) TextButton(onPressed: app.generating || app.loading ? null : app.unload,
                  child: const Text('Unload model')),
              ]),
            );
          }),
          const SizedBox(height: 16),
          Row(children: [Icon(Icons.lock_outline_rounded, size: 13, color: theme.colorScheme.onSurfaceVariant),
            const SizedBox(width: 6), Expanded(child: Text('Inference on your device', style: theme.textTheme.labelSmall?.copyWith(fontSize: 11)))]),
        ]),
      )),
    );
  }
}
