// HyperOS / MIUI's Autostart switch, for what Liyab does while it is closed.
//
// Those systems refuse to start an app's process by themselves unless the user
// allowed its Autostart: after a reboot Android cannot start Liyab for the
// notification listener, so the local API stays closed (and notifications are
// not kept) until Liyab is opened. No API turns the switch on; this hint shows
// it is off and opens the system screen where it is (DeviceChannel.kt). Shown
// only where the switch exists and reads off.
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class AutostartHint extends StatefulWidget {
  const AutostartHint({super.key});

  @override
  State<AutostartHint> createState() => _AutostartHintState();
}

class _AutostartHintState extends State<AutostartHint> with WidgetsBindingObserver {
  static const _device = MethodChannel('liyab/device');
  bool _needed = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _check();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  // Back from the system's Autostart screen: read the switch again.
  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed) _check();
  }

  Future<void> _check() async {
    if (!Platform.isAndroid) return;
    bool needed = false;
    try {
      final state = await _device.invokeMapMethod<String, Object?>('autostart');
      // Unknown (the switch could not be read) is not shown: a hint must not
      // ask for something that may already be allowed.
      needed = state?['hyperos'] == true && state?['allowed'] == false;
    } on PlatformException {
      needed = false;
    } on MissingPluginException {
      needed = false;
    }
    if (mounted && needed != _needed) setState(() => _needed = needed);
  }

  @override
  Widget build(BuildContext context) {
    if (!_needed) return const SizedBox.shrink();
    final theme = Theme.of(context);
    return ListTile(
      contentPadding: EdgeInsets.zero,
      leading: Icon(Icons.restart_alt_rounded, color: theme.colorScheme.error),
      title: const Text('Allow Autostart'),
      subtitle: Text(
        'HyperOS does not let Liyab start by itself, so after a reboot other apps reach the local API only once you '
        'open Liyab. Turn on Liyab in Autostart.',
        style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
      ),
      trailing: TextButton(
        onPressed: () async {
          try {
            await _device.invokeMethod<bool>('openAutostartSettings');
          } on PlatformException {
            // Nothing to open on this system: the hint stays.
          }
        },
        child: const Text('Allow'),
      ),
    );
  }
}
