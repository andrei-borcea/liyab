// Desktop lifecycle shared by macOS, Windows and Linux. Native plugins own
// system tray integration; Flutter owns menu actions and model state.
import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:dbus/dbus.dart';
import 'package:flutter/services.dart';
import 'package:tray_manager/tray_manager.dart';
import 'package:window_manager/window_manager.dart';

import '../state/app_state.dart';

class DesktopHost with TrayListener, WindowListener {
  static bool get supported => Platform.isMacOS || Platform.isWindows || Platform.isLinux;
  static final navigatorKey = GlobalKey<NavigatorState>();
  static final problem = ValueNotifier<String?>(null);
  static void Function(int)? navigate;
  static DesktopHost? _instance;
  AppState? _app;
  String _lastStatus = '';
  Future<void> _updates = Future.value();
  bool _quitting = false;

  static Future<void> initialize() async {
    if (!supported) return;
    final host = _instance = DesktopHost();
    try {
      await windowManager.ensureInitialized();
      await windowManager.setMinimumSize(const Size(420, 560));
      windowManager.addListener(host);
      trayManager.addListener(host);
      await trayManager.setIcon(Platform.isWindows ? 'assets/brand/tray.ico' : 'assets/brand/tray.png');
      await host._menu('Starting engine');
      // Only intercept close after tray creation succeeds: a tray-less Linux
      // session must not leave an invisible process the user cannot reopen.
      var keepInTray = true;
      if (Platform.isLinux) {
        final bus = DBusClient.session();
        try { keepInTray = await bus.nameHasOwner('org.kde.StatusNotifierWatcher'); }
        finally { await bus.close(); }
      }
      await windowManager.setPreventClose(keepInTray);
      if (!keepInTray) problem.value = 'Enable a system tray extension to keep Liyab running after closing the window.';
    } catch (error) {
      problem.value = 'System tray unavailable. You can still use the app window.';
      debugPrint('Desktop integration: $error');
    }
  }

  static void attach(AppState app) {
    final host = _instance;
    if (host == null) return;
    host._app = app;
    app.addListener(host._changed);
    host._changed();
  }

  void _changed() {
    final app = _app!;
    final status = app.loading ? 'Loading model' : app.generating ? 'Generating' :
      app.modelPath == null ? 'No model loaded' : 'Ready · ${app.modelName}';
    if (status == _lastStatus) return;
    _lastStatus = status;
    // Serialize native menu updates; streaming tokens do not rebuild the menu.
    _updates = _updates.then((_) => _menu(status)).catchError((Object error) {
      debugPrint('Tray status: $error');
    });
  }

  Future<void> _menu(String status) async {
    await trayManager.setToolTip('Liyab · $status');
    await trayManager.setContextMenu(Menu(items: [
      MenuItem(label: status, disabled: true),
      MenuItem.separator(),
      MenuItem(key: 'show', label: 'Open Liyab'),
      MenuItem(key: 'chat', label: 'Chat'),
      MenuItem(key: 'history', label: 'Saved chats'),
      MenuItem(key: 'models', label: 'Local models'),
      MenuItem(key: 'settings', label: 'Settings & API'),
      MenuItem.separator(),
      MenuItem(key: 'quit', label: 'Quit Liyab'),
    ]));
  }

  Future<void> _show() async {
    if (Platform.isMacOS) {
      // Option-Space temporarily transforms the same NSWindow into a panel.
      // Restore it through its owner before showing the normal workspace.
      await const MethodChannel('liyab/assist').invokeMethod<void>('openApp');
    }
    if (await windowManager.isMinimized()) await windowManager.restore();
    await windowManager.show();
    await windowManager.focus();
  }

  @override
  void onWindowClose() => unawaited(windowManager.hide());

  @override
  void onTrayIconMouseDown() => unawaited(_show());

  @override
  void onTrayIconRightMouseDown() => unawaited(trayManager.popUpContextMenu());

  @override
  void onTrayMenuItemClick(MenuItem menuItem) {
    if (menuItem.key == 'quit') { unawaited(_quit()); return; }
    navigate?.call(switch (menuItem.key) { 'history' => 1, 'models' => 2, 'settings' => 4, _ => 0 });
    unawaited(_show());
  }

  Future<void> _quit() async {
    if (_quitting) return;
    _quitting = true;
    try {
      await _show();
      final context = navigatorKey.currentContext;
      if (context == null || !context.mounted) return;
      final confirmed = await showDialog<bool>(context: context, builder: (context) => AlertDialog(
        title: const Text('Quit Liyab?'),
        content: const Text('This stops local inference, downloads and the API server. Closing the window keeps Liyab in the tray.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('Keep running')),
          FilledButton(onPressed: () => Navigator.pop(context, true), child: const Text('Quit')),
        ],
      ));
      if (confirmed != true) return;
      await _app?.prepareToQuit();
      await _updates;
      _app?.removeListener(_changed);
      trayManager.removeListener(this);
      windowManager.removeListener(this);
      await trayManager.destroy();
      await windowManager.destroy();
    } catch (error) {
      problem.value = 'Could not quit cleanly: $error';
    } finally {
      _quitting = false;
    }
  }
}
