// Liyab: a private assistant that runs entirely on the device.
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'assist/assist_sheet.dart';
import 'assist/command_bar.dart';
import 'ui/app_shell.dart';
import 'ui/resource_notices.dart';
import 'desktop/desktop_host.dart';
import 'state/app_state.dart';
import 'ui/living_flame.dart';
import 'ui/theme.dart';
import 'ui/welcome_page.dart';

/// `args` holds "hidden" when Android started the engine without a window
/// (LiyabEngine.startForApi in the Android app: the local API is on and the
/// system started the process for the notification listener).
Future<void> main(List<String> args) async {
  WidgetsFlutterBinding.ensureInitialized();
  final hidden = args.contains('hidden');
  await DesktopHost.initialize();
  void edgeToEdge() => SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  if (hidden) {
    // The system UI mode belongs to a window: set once one shows Liyab
    // (without a window the call has nothing to reach).
    var shown = false;
    AppLifecycleListener(onStateChange: (state) {
      if (state != AppLifecycleState.resumed || shown) return;
      shown = true;
      edgeToEdge();
    });
  } else {
    edgeToEdge();
  }
  // Created here, not by a widget: without a window no frame may ever build
  // the tree, and the local API must start all the same.
  final app = AppState.create(hidden: hidden);
  // A failed engine initialization must still leave a usable window and Quit.
  app.then(DesktopHost.attach, onError: (Object error, StackTrace stack) {});
  runApp(LiyabApp(app: app));
}

class LiyabApp extends StatefulWidget {
  const LiyabApp({super.key, required this.app});

  /// The app's state, being created (model, settings, the local API).
  final Future<AppState> app;

  @override
  State<LiyabApp> createState() => _LiyabAppState();
}

class _LiyabAppState extends State<LiyabApp> {
  late final Future<AppState> _app = widget.app;
  final _assist = AssistMode(); // the system assist gesture opened the sheet

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Liyab',
        navigatorKey: DesktopHost.navigatorKey,
        debugShowCheckedModeBanner: false,
        theme: liyabTheme(Brightness.light),
        darkTheme: liyabTheme(Brightness.dark),
        themeMode: ThemeMode.dark, // Liyab lives at night, by its own light
        builder: (context, child) => ListenableBuilder(
          listenable: _assist,
          builder: (context, _) => _titleBarInset(context, child!),
        ),
        home: FutureBuilder<AppState>(
          future: _app,
          builder: (context, snap) {
            final app = snap.data;
            if (snap.hasError) {
              return Scaffold(body: Center(child: Padding(padding: const EdgeInsets.all(32),
                child: SelectableText('Could not start the local engine.\n${snap.error}'))));
            }
            if (app == null) {
              return const Scaffold(body: Center(child: LivingFlame(size: 140, state: FlameState.thinking)));
            }
            return ResourceNotices(app: app, child: ListenableBuilder(
              listenable: Listenable.merge([app, _assist]),
              builder: (context, _) => _assist.value
                  ? (Platform.isAndroid ? AssistSheet(app: app, mode: _assist) : CommandBar(app: app, mode: _assist))
                  : (app.welcomed || DesktopHost.supported)
                      ? AppShell(app: app)
                      : WelcomePage(app: app),
            ));
          },
        ),
      );

  /// macOS: the window's content runs under its transparent title bar
  /// (macos/Runner/MainFlutterWindow.swift). The bar's height becomes top
  /// padding, as a status bar's is on a phone: app bars extend under the
  /// traffic lights and their content starts below them. The assistant sheet
  /// (a borderless panel) has no title bar.
  Widget _titleBarInset(BuildContext context, Widget child) {
    if (!Platform.isMacOS || _assist.value) return child;
    final media = MediaQuery.of(context);
    const inset = EdgeInsets.only(top: 28);
    return MediaQuery(
      data: media.copyWith(padding: media.padding + inset, viewPadding: media.viewPadding + inset),
      child: child,
    );
  }
}
