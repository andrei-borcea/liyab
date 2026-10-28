// Deliver scheduler changes through exactly one channel: in-app while focused,
// OS notification while hidden. Keep the last event in the app's activity log.
import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import '../state/app_state.dart';
import '../state/resource_scheduler.dart';

class ResourceNotices extends StatefulWidget {
  const ResourceNotices({super.key, required this.app, required this.child});
  final AppState app;
  final Widget child;
  @override
  State<ResourceNotices> createState() => _ResourceNoticesState();
}

class _ResourceNoticesState extends State<ResourceNotices> with WidgetsBindingObserver {
  final _notifications = FlutterLocalNotificationsPlugin();
  late final Future<void> _ready;
  ResourceEvent? _seen;
  bool _foreground = true;
  String? _notice;
  Timer? _dismiss;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    widget.app.addListener(_changed);
    _foreground = WidgetsBinding.instance.lifecycleState == AppLifecycleState.resumed;
    _ready = _initialize();
  }

  Future<void> _initialize() async {
    if (!Platform.isMacOS && !Platform.isWindows && !Platform.isLinux) return;
    try {
      await _notifications.initialize(settings: const InitializationSettings(
        macOS: DarwinInitializationSettings(requestAlertPermission: true,
          requestSoundPermission: false, requestBadgePermission: false),
        linux: LinuxInitializationSettings(defaultActionName: 'Open Liyab'),
        windows: WindowsInitializationSettings(appName: 'Liyab', appUserModelId: 'com.liyab.chat',
          guid: 'e11a5296-50d6-4d12-bbde-08c19436f1da'),
      ));
    } catch (error) { debugPrint('Scheduler notifications unavailable: $error'); }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    _foreground = state == AppLifecycleState.resumed;
  }

  void _changed() {
    final event = widget.app.resourceEvent;
    if (event == null || identical(event, _seen)) return;
    _seen = event;
    if (_foreground) {
      setState(() => _notice = event.message);
      _dismiss?.cancel();
      _dismiss = Timer(const Duration(seconds: 18), () { if (mounted) setState(() => _notice = null); });
    } else {
      unawaited(_notify(event));
    }
  }

  Future<void> _notify(ResourceEvent event) async {
    await _ready;
    try {
      await _notifications.show(id: 41, title: event.degraded ? 'Liyab · Resources adjusted' : 'Liyab · Resources restored',
        body: event.message,
        notificationDetails: const NotificationDetails(macOS: DarwinNotificationDetails(presentSound: false)));
    } catch (error) { debugPrint('Scheduler notification: $error'); }
  }

  @override
  void dispose() {
    widget.app.removeListener(_changed);
    WidgetsBinding.instance.removeObserver(this);
    _dismiss?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) => Column(children: [
    if (_notice != null) SafeArea(bottom: false, child: Material(color: Theme.of(context).colorScheme.surfaceContainerHighest,
      child: Padding(padding: const EdgeInsets.fromLTRB(16, 8, 8, 8), child: Row(children: [
        Icon(Icons.speed_rounded, color: Theme.of(context).colorScheme.primary, size: 20),
        const SizedBox(width: 10), Expanded(child: Semantics(liveRegion: true, child: Text(_notice!))),
        IconButton(tooltip: 'Dismiss resource notice', onPressed: () => setState(() => _notice = null), icon: const Icon(Icons.close_rounded, size: 18)),
      ])),
    )),
    Expanded(child: widget.child),
  ]);
}
