// Liyab: a private assistant that runs entirely on the phone.
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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

  @override
  Widget build(BuildContext context) => MaterialApp(
        title: 'Liyab',
        debugShowCheckedModeBanner: false,
        theme: liyabTheme(Brightness.light),
        darkTheme: liyabTheme(Brightness.dark),
        themeMode: ThemeMode.dark, // Liyab lives at night, by its own light
        home: FutureBuilder<AppState>(
          future: _app,
          builder: (context, snap) {
            final app = snap.data;
            if (app == null) {
              return const Scaffold(body: Center(child: LivingFlame(size: 140, state: FlameState.thinking)));
            }
            return ListenableBuilder(
              listenable: app,
              builder: (context, _) => app.welcomed ? ChatScreen(app: app) : WelcomePage(app: app),
            );
          },
        ),
      );
}
