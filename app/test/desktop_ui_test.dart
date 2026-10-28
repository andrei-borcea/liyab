import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:liyab/models/downloads.dart';
import 'package:liyab/state/app_state.dart';
import 'package:liyab/state/device_monitor.dart';
import 'package:liyab/state/models_store.dart';
import 'package:liyab/state/resource_scheduler.dart';
import 'package:liyab/state/settings.dart';
import 'package:liyab/ui/app_shell.dart';
import 'package:liyab/ui/resource_notices.dart';
import 'package:liyab/ui/theme.dart';
import 'package:shared_preferences/shared_preferences.dart';

class PreviewMonitor extends ChangeNotifier implements DeviceMonitor {
  @override
  double get heat => 0.3;
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class PreviewDownloads extends ChangeNotifier implements Downloads {
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

class PreviewApp extends ChangeNotifier implements AppState {
  PreviewApp(this.device);
  @override
  final DeviceSettings device;
  @override
  final monitor = PreviewMonitor();
  @override
  final downloads = PreviewDownloads();
  @override
  final List<ChatMessage> messages = [];
  @override
  bool loading = false, generating = false, switchingChat = false;
  @override
  String? modelPath, moving;
  @override
  String modelName = '', status = 'No model loaded';
  @override
  ResourceEvent? resourceEvent;
  int newChats = 0;
  int libraryReads = 0;
  @override
  void newChat() => newChats++;
  @override
  void draftChanged(String text) {}
  @override
  Future<List<LocalModel>> localModels() async { libraryReads++; return []; }
  @override
  dynamic noSuchMethod(Invocation invocation) => super.noSuchMethod(invocation);
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  late PreviewApp app;
  setUp(() async {
    SharedPreferences.setMockInitialValues({});
    app = PreviewApp(DeviceSettings(await SharedPreferences.getInstance()));
  });

  testWidgets('desktop workspace and compact navigation fit minimum window sizes', (tester) async {
    addTearDown(() => tester.view.resetPhysicalSize());
    addTearDown(() => tester.view.resetDevicePixelRatio());
    tester.view.devicePixelRatio = 1;
    for (final size in [const Size(1280, 800), const Size(760, 560), const Size(420, 560)]) {
      tester.view.physicalSize = size;
      await tester.pumpWidget(MaterialApp(theme: liyabTheme(Brightness.dark), home: AppShell(app: app)));
      await tester.pumpAndSettle();
      expect(tester.takeException(), isNull, reason: 'Layout at $size');
      expect(find.text('A little space to think.'), findsOneWidget);
    }
    expect(app.libraryReads, 0, reason: 'Opening the workspace must not scan files or contact the hub');
  });

  testWidgets('new chat uses shared state and is disabled during generation', (tester) async {
    tester.view.physicalSize = const Size(1280, 800);
    tester.view.devicePixelRatio = 1;
    addTearDown(() => tester.view.resetPhysicalSize());
    addTearDown(() => tester.view.resetDevicePixelRatio());
    await tester.pumpWidget(MaterialApp(theme: liyabTheme(Brightness.dark), home: AppShell(app: app)));
    await tester.pumpAndSettle();
    await tester.tap(find.text('New chat'));
    expect(app.newChats, 1);
    app.generating = true;
    app.notifyListeners();
    await tester.pump();
    final button = tester.widget<OutlinedButton>(find.widgetWithText(OutlinedButton, 'New chat'));
    expect(button.onPressed, isNull);
    await tester.pumpWidget(const SizedBox());
  });

  testWidgets('scheduler notices use the in-app banner in foreground and OS channel in background', (tester) async {
    debugDefaultTargetPlatformOverride = TargetPlatform.macOS;
    addTearDown(() => debugDefaultTargetPlatformOverride = null);
    MacOSFlutterLocalNotificationsPlugin.registerWith();
    final calls = <MethodCall>[];
    tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(const MethodChannel('dexterous.com/flutter/local_notifications'), (call) async {
      calls.add(call);
      return true;
    });
    addTearDown(() => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(const MethodChannel('dexterous.com/flutter/local_notifications'), null));
    tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
    await tester.pumpWidget(MaterialApp(home: ResourceNotices(app: app, child: const Scaffold(body: Text('Workspace')))));
    await tester.pump();
    app.resourceEvent = ResourceEvent('CPU share 100% → 50% due to heat pressure.');
    app.notifyListeners();
    await tester.pump();
    expect(find.text('CPU share 100% → 50% due to heat pressure.'), findsOneWidget);
    expect(calls.where((c) => c.method == 'show'), isEmpty);
    tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.hidden);
    app.resourceEvent = ResourceEvent('Memory budget 8 → 6 GiB due to memory pressure.');
    app.notifyListeners();
    await tester.pump();
    await tester.pump();
    expect(calls.where((c) => c.method == 'show'), hasLength(1));
    await tester.pumpWidget(const SizedBox());
    tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
    debugDefaultTargetPlatformOverride = null;
  });
}
