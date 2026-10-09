// Agent tools: what the model may read on the phone to answer about the
// user's own data. Each tool is off until the user turns it on in Settings
// (and Android grants its permission); only enabled tools are offered to the
// model. Results stay on the device and go back to the model as data.
import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// A request the model wrote: a tool name and its arguments.
class ToolCall {
  const ToolCall(this.name, this.arguments);
  final String name;
  final Map<String, Object?> arguments;
}

/// What a tool did, shown above the answer ("Read your calendar").
class ToolStep {
  const ToolStep(this.label, this.summary);
  final String label;
  final String summary;
}

abstract class AgentTool {
  String get name;
  String get description;

  /// JSON schema of the arguments (object properties).
  Map<String, Object?> get parameters;
  List<String> get required => const [];

  /// What the step is called in the chat.
  String get label;

  /// The preference that turns the tool on.
  String get setting => 'tool.$name';

  /// Whether the OS lets the tool run (a granted permission, if it needs one).
  Future<bool> permitted() async => true;

  /// Asks for the tool's permission; whether it is granted.
  Future<bool> requestPermission() async => true;

  /// Runs the tool; the result is text for the model (JSON), and a short summary for the chat.
  Future<(String, String)> run(Map<String, Object?> args);

  /// The OpenAI-style function description chat templates render.
  Map<String, Object?> get schema => {
        'type': 'function',
        'function': {
          'name': name,
          'description': description,
          'parameters': {'type': 'object', 'properties': parameters, if (required.isNotEmpty) 'required': required},
        },
      };
}

const _data = MethodChannel('liyab/data');

String _two(int v) => v.toString().padLeft(2, '0');
String _local(DateTime t) => '${t.year}-${_two(t.month)}-${_two(t.day)} ${_two(t.hour)}:${_two(t.minute)}';

class CalendarTool extends AgentTool {
  @override
  String get name => 'calendar_events';

  @override
  String get label => 'Read your calendar';


  @override
  String get description =>
      "The user's calendar events between two local times.";

  @override
  Map<String, Object?> get parameters => {
        'start': {'type': 'string', 'description': 'e.g. 2026-10-09T00:00'},
        'end': {'type': 'string', 'description': 'e.g. 2026-10-10T00:00'},
      };

  @override
  List<String> get required => const ['start', 'end'];

  @override
  Future<bool> permitted() => Permission.calendarFullAccess.isGranted;

  @override
  Future<bool> requestPermission() async => (await Permission.calendarFullAccess.request()).isGranted;

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final start = DateTime.tryParse('${args['start'] ?? ''}') ?? today;
    var end = DateTime.tryParse('${args['end'] ?? ''}') ?? today.add(const Duration(days: 1));
    if (!end.isAfter(start)) end = start.add(const Duration(days: 1));
    if (!await permitted()) {
      return (jsonEncode({'error': 'Calendar access is off. The user can allow it in Liyab Settings.'}), 'Calendar access is off');
    }
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>(
            'calendarEvents', {'start': start.millisecondsSinceEpoch, 'end': end.millisecondsSinceEpoch}) ??
        const [];
    final events = [
      for (final e in raw)
        {
          'title': e['title'] ?? '(no title)',
          if (e['allDay'] == true)
            'all_day': true
          else ...{
            'start': _local(DateTime.fromMillisecondsSinceEpoch((e['begin'] as num).toInt())),
            'end': _local(DateTime.fromMillisecondsSinceEpoch((e['end'] as num).toInt())),
          },
          if ((e['location'] as String?)?.isNotEmpty ?? false) 'location': e['location'],
          if ((e['calendar'] as String?)?.isNotEmpty ?? false) 'calendar': e['calendar'],
          if ((e['description'] as String?)?.isNotEmpty ?? false) 'notes': e['description'],
        }
    ];
    final result = jsonEncode({'from': _local(start), 'to': _local(end), 'events': events});
    return (result, events.isEmpty ? 'No events' : '${events.length} event${events.length == 1 ? '' : 's'}');
  }
}

class ClipboardTool extends AgentTool {
  @override
  String get name => 'read_clipboard';

  @override
  String get label => 'Read what you copied';

  @override
  String get description =>
      'The text the user last copied.';

  @override
  Map<String, Object?> get parameters => const {};

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final text = (await Clipboard.getData(Clipboard.kTextPlain))?.text ?? '';
    if (text.isEmpty) return (jsonEncode({'text': '', 'note': 'The clipboard is empty.'}), 'Nothing copied');
    const limit = 6000;
    final clipped = text.length > limit ? '${text.substring(0, limit)}…' : text;
    return (jsonEncode({'text': clipped}), '${text.length} characters');
  }
}

/// "since" arguments: a local ISO 8601 date-time; default: 24 hours ago.
DateTime _since(Object? v) => DateTime.tryParse('${v ?? ''}') ?? DateTime.now().subtract(const Duration(hours: 24));

const _sinceParam = {
  'since': {'type': 'string', 'description': 'e.g. 2026-10-09T08:00; default 24 h ago'},
};

String _plural(int n, String one) => '$n $one${n == 1 ? '' : 's'}';

class ContactsTool extends AgentTool {
  @override
  String get name => 'find_contact';
  @override
  String get label => 'Looked up your contacts';
  @override
  String get description => "Finds the user's contacts by name: numbers and emails.";
  @override
  Map<String, Object?> get parameters => {
        'name': {'type': 'string'},
      };
  @override
  List<String> get required => const ['name'];
  @override
  Future<bool> permitted() => Permission.contacts.isGranted;
  @override
  Future<bool> requestPermission() async => (await Permission.contacts.request()).isGranted;

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final found = await _data.invokeListMethod<Object?>('contacts', {'query': '${args['name'] ?? ''}'}) ?? const [];
    return (jsonEncode({'contacts': found}), _plural(found.length, 'contact'));
  }
}

class MessagesTool extends AgentTool {
  @override
  String get name => 'read_sms';
  @override
  String get label => 'Read your text messages';
  @override
  String get description =>
      'SMS sent or received since a time, optionally with one person. Chat apps and email: read_notifications.';
  @override
  Map<String, Object?> get parameters => {
        ..._sinceParam,
        'from': {'type': 'string', 'description': 'name or number'},
      };
  @override
  Future<bool> permitted() => Permission.sms.isGranted;
  @override
  Future<bool> requestPermission() async => (await Permission.sms.request()).isGranted;

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final since = _since(args['since']);
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>(
            'sms', {'since': since.millisecondsSinceEpoch, 'from': args['from']}) ??
        const [];
    final messages = [
      for (final m in raw)
        {
          (m['sent'] == true ? 'to' : 'from'): m['from'],
          'time': _local(DateTime.fromMillisecondsSinceEpoch((m['time'] as num).toInt())),
          'text': m['text'],
        }
    ];
    return (jsonEncode({'since': _local(since), 'messages': messages}), _plural(messages.length, 'message'));
  }
}

class CallsTool extends AgentTool {
  @override
  String get name => 'recent_calls';
  @override
  String get label => 'Checked your calls';
  @override
  String get description => 'Phone calls since a time: who, when, missed or not, length.';
  @override
  Map<String, Object?> get parameters => _sinceParam;
  @override
  Future<bool> permitted() => Permission.phone.isGranted;
  @override
  Future<bool> requestPermission() async => (await Permission.phone.request()).isGranted;

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final since = _since(args['since']);
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>('calls', {'since': since.millisecondsSinceEpoch}) ??
        const [];
    final calls = [
      for (final c in raw)
        {
          'who': c['who'],
          'type': c['type'],
          'time': _local(DateTime.fromMillisecondsSinceEpoch((c['time'] as num).toInt())),
          'seconds': c['seconds'],
        }
    ];
    return (jsonEncode({'since': _local(since), 'calls': calls}), _plural(calls.length, 'call'));
  }
}

class NotificationsTool extends AgentTool {
  @override
  String get name => 'read_notifications';
  @override
  String get label => 'Read your notifications';
  @override
  String get description =>
      'Notifications since a time: chat messages (WhatsApp, Telegram…), email previews, app alerts.';
  @override
  Map<String, Object?> get parameters => {
        ..._sinceParam,
        'app': {'type': 'string', 'description': 'e.g. WhatsApp, Gmail'},
      };
  @override
  Future<bool> permitted() async => await _data.invokeMethod<bool>('notificationAccess') ?? false;

  /// Notification access is a special permission: Android's settings page, not a dialog.
  @override
  Future<bool> requestPermission() async {
    await _data.invokeMethod<void>('openNotificationAccess');
    return false; // granted (or not) in Android's settings; checked again later
  }

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final since = _since(args['since']);
    final app = '${args['app'] ?? ''}'.toLowerCase();
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>(
            'notifications', {'since': since.millisecondsSinceEpoch}) ??
        const [];
    final items = [
      for (final n in raw.reversed)
        if (app.isEmpty || '${n['app']}'.toLowerCase().contains(app))
          {
            'app': n['app'],
            'title': n['title'],
            'text': n['text'],
            'time': _local(DateTime.fromMillisecondsSinceEpoch((n['time'] as num).toInt())),
          }
    ].take(60).toList();
    return (jsonEncode({'since': _local(since), 'notifications': items}), _plural(items.length, 'notification'));
  }
}

/// The tools Liyab has, and which ones the user turned on.
class Toolbox {
  Toolbox(this._prefs);
  final SharedPreferences _prefs;

  final List<AgentTool> all = [CalendarTool(), NotificationsTool(), MessagesTool(), CallsTool(), ContactsTool(), ClipboardTool()];

  bool isOn(AgentTool t) => _prefs.getBool(t.setting) ?? false;
  Future<void> setOn(AgentTool t, bool on) => _prefs.setBool(t.setting, on);

  /// Tools turned on whose permission is granted: the ones offered to the model.
  Future<List<AgentTool>> enabled() async => [
        for (final t in all)
          if (isOn(t) && await t.permitted()) t
      ];

  AgentTool? byName(String name) => all.where((t) => t.name == name).firstOrNull;
}
