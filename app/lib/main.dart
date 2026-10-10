// Liyab: a private assistant that runs entirely on the device.
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'assist/assist_sheet.dart';
import 'assist/command_bar.dart';
import 'chat/chat_screen.dart';
import 'state/app_state.dart';
import 'ui/living_flame.dart';
import 'ui/theme.dart';
import 'ui/welcome_page.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setEnabledSystemUIMode(SystemUiMode.edgeToEdge);
  runApp(const LiyabApp());
}

class LiyabApp extends StatefulWidget {
  const LiyabApp({super.key});

  @override
  State<LiyabApp> createState() => _LiyabAppState();
}

class _LiyabAppState extends State<LiyabApp> {
  final Future<AppState> _app = AppState.create();
  final _assist = AssistMode(); // the system assist gesture opened the sheet

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Liyab',
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
            if (app == null) {
              return const Scaffold(body: Center(child: LivingFlame(size: 140, state: FlameState.thinking)));
            }
            return ListenableBuilder(
              listenable: Listenable.merge([app, _assist]),
              builder: (context, _) => _assist.value
                  ? (Platform.isAndroid ? AssistSheet(app: app, mode: _assist) : CommandBar(app: app, mode: _assist))
                  : app.welcomed
                      ? ChatScreen(app: app)
                      : WelcomePage(app: app),
            );
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
