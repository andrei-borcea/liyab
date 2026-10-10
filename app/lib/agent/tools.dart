// Agent tools: what the model may read on the phone to answer about the
// user's own data. Each tool is off until the user turns it on in Settings
// (and Android grants its permission); only enabled tools are offered to the
// model. Results stay on the device and go back to the model as data, as
// short text (ToolText).

import 'dart:io';

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

  /// Where the user grants the permission by hand once it was refused.
  Future<void> openSettings() async {}

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

/// A permission-guarded source: Android's runtime permission, or on macOS the
/// runner's liyab/data channel (macos/Runner/DataChannel.swift), where `kind`
/// is calendar, contacts, or a Full Disk Access source (messages, calls, email).
mixin _OsPermission on AgentTool {
  Permission get android;
  String get kind;

  @override
  Future<bool> permitted() async => Platform.isAndroid
      ? android.isGranted
      : await _data.invokeMethod<bool>('permission', {'kind': kind}) ?? false;

  @override
  Future<bool> requestPermission() async => Platform.isAndroid
      ? (await android.request()).isGranted
      : await _data.invokeMethod<bool>('requestPermission', {'kind': kind}) ?? false;

  @override
  Future<void> openSettings() async =>
      Platform.isAndroid ? await openAppSettings() : await _data.invokeMethod<void>('openSettings', {'kind': kind});
}

String _two(int v) => v.toString().padLeft(2, '0');

/// Tool results as short text, one line per item. Before the answer starts the
/// model processes every token of a result, and on a 35B MoE that costs about
/// as much as generating one: JSON repeats every key on every item, so the same
/// items as lines take about half the tokens. Lists are capped, newest first,
/// and long texts cut. Dates are ISO 8601 (no day or month names, which the
/// model could carry into an answer in another language); today's times stand
/// alone, since each user turn ends with the current date and weekday.
abstract final class ToolText {
  static String day(DateTime t) => '${t.year}-${_two(t.month)}-${_two(t.day)}';
  static String hm(DateTime t) => '${_two(t.hour)}:${_two(t.minute)}';
  static bool _sameDay(DateTime a, DateTime b) => a.year == b.year && a.month == b.month && a.day == b.day;

  /// `t` as a time, with its day unless it falls on `today`.
  static String at(DateTime t, DateTime today) => _sameDay(t, today) ? hm(t) : '${day(t)} ${hm(t)}';

  /// One line, at most `max` characters.
  static String clip(Object? text, int max) {
    final s = '${text ?? ''}'.replaceAll(RegExp(r'\s+'), ' ').trim();
    return s.length <= max ? s : '${s.substring(0, max - 1)}…';
  }

  static DateTime _time(Object? ms) => DateTime.fromMillisecondsSinceEpoch((ms as num).toInt());

  /// Calendar instances (DataChannel.calendarEvents) between `from` and `to`.
  static String events(List<Map<Object?, Object?>> raw, DateTime from, DateTime to, DateTime now) {
    final head = 'Calendar from ${day(from)} ${hm(from)} to ${day(to)} ${hm(to)}';
    if (raw.isEmpty) return '$head: no events.';
    final lines = [
      for (final e in raw)
        () {
          final begin = _time(e['begin']), end = _time(e['end']);
          final when = e['allDay'] == true
              ? '${day(begin)} all day'
              : '${at(begin, now)}-${_sameDay(begin, end) ? hm(end) : at(end, now)}';
          final title = clip(e['title'] ?? '(no title)', 120);
          final location = clip(e['location'], 80), calendar = clip(e['calendar'], 40), notes = clip(e['description'], 120);
          return '- $when $title${location.isEmpty ? '' : ' @ $location'}${calendar.isEmpty ? '' : ' [$calendar]'}'
              '${notes.isEmpty ? '' : ' (notes: $notes)'}';
        }()
    ];
    return '$head, ${raw.length} event${raw.length == 1 ? '' : 's'}:\n${lines.join('\n')}';
  }

  /// Notifications (newest last in `raw`), duplicates dropped, newest first, at most `limit`.
  static String notifications(List<Map<Object?, Object?>> raw, DateTime since, String app, DateTime now,
      {int limit = 25}) {
    final seen = <String>{};
    final lines = <String>[];
    var total = 0;
    for (final n in raw.reversed) {
      if (app.isNotEmpty && !'${n['app']}'.toLowerCase().contains(app.toLowerCase())) continue;
      final title = clip(n['title'], 80), text = clip(n['text'], 200);
      if (!seen.add('${n['app']}|$title|$text')) continue; // chat apps repost the same lines
      ++total;
      if (lines.length < limit) {
        lines.add('- ${at(_time(n['time']), now)} ${clip(n['app'], 30)}: $title${text.isEmpty ? '' : ' - $text'}');
      }
    }
    final head = 'Notifications since ${at(since, now)}${app.isEmpty ? '' : ' from $app'}';
    if (lines.isEmpty) return '$head: none.';
    final more = total > lines.length ? ' (newest $limit of $total)' : '';
    return '$head$more, newest first:\n${lines.join('\n')}';
  }

  /// SMS (DataChannel.sms, newest first), at most `limit`.
  static String sms(List<Map<Object?, Object?>> raw, DateTime since, DateTime now, {int limit = 30}) {
    final head = 'Text messages since ${at(since, now)}';
    if (raw.isEmpty) return '$head: none.';
    final lines = [
      for (final m in raw.take(limit))
        '- ${at(_time(m['time']), now)} ${m['sent'] == true ? 'to' : 'from'} ${clip(m['from'], 40)}: ${clip(m['text'], 300)}'
    ];
    final more = raw.length > limit ? ' (newest $limit of ${raw.length})' : '';
    return '$head$more, newest first:\n${lines.join('\n')}';
  }

  /// Calls (DataChannel.calls, newest first), at most `limit`.
  static String calls(List<Map<Object?, Object?>> raw, DateTime since, DateTime now, {int limit = 30}) {
    final head = 'Calls since ${at(since, now)}';
    if (raw.isEmpty) return '$head: none.';
    String length(Object? s) {
      final v = (s as num?)?.toInt() ?? 0;
      return v == 0 ? '' : ' (${v ~/ 60}:${_two(v % 60)})';
    }

    final lines = [
      for (final c in raw.take(limit)) '- ${at(_time(c['time']), now)} ${c['type']} ${clip(c['who'], 40)}${length(c['seconds'])}'
    ];
    final more = raw.length > limit ? ' (newest $limit of ${raw.length})' : '';
    return '$head$more, newest first:\n${lines.join('\n')}';
  }

  /// Emails (DataChannel.email on macOS, newest first), at most `limit`.
  static String emails(List<Map<Object?, Object?>> raw, DateTime since, String query, DateTime now, {int limit = 25}) {
    final head = 'Emails since ${at(since, now)}${query.isEmpty ? '' : ' matching "$query"'}';
    if (raw.isEmpty) return '$head: none.';
    final lines = [
      for (final m in raw.take(limit))
        '- ${at(_time(m['time']), now)}${m['read'] == true ? '' : ' (unread)'} ${clip(m['from'], 40)}: '
            '${clip(m['subject'] ?? '(no subject)', 120)}${clip(m['preview'], 200).isEmpty ? '' : ' - ${clip(m['preview'], 200)}'}'
    ];
    final more = raw.length > limit ? ' (newest $limit of ${raw.length})' : '';
    return '$head$more, newest first:\n${lines.join('\n')}';
  }

  /// Contacts (DataChannel.contacts): name, numbers, email addresses.
  static String contacts(List<Object?> raw, String query) {
    if (raw.isEmpty) return 'No contact matches "$query".';
    return [
      for (final c in raw.cast<Map<Object?, Object?>>())
        '- ${c['name']}: ${[...(c['phones'] as List? ?? const []), ...(c['emails'] as List? ?? const [])].join(', ')}'
    ].join('\n');
  }
}

class CalendarTool extends AgentTool with _OsPermission {
  @override
  Permission get android => Permission.calendarFullAccess;
  @override
  String get kind => 'calendar';

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
  Future<(String, String)> run(Map<String, Object?> args) async {
    final now = DateTime.now();
    final today = DateTime(now.year, now.month, now.day);
    final start = DateTime.tryParse('${args['start'] ?? ''}') ?? today;
    var end = DateTime.tryParse('${args['end'] ?? ''}') ?? today.add(const Duration(days: 1));
    if (!end.isAfter(start)) end = start.add(const Duration(days: 1));
    if (!await permitted()) {
      return ('Calendar access is off. The user can allow it in Liyab Settings.', 'Calendar access is off');
    }
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>(
            'calendarEvents', {'start': start.millisecondsSinceEpoch, 'end': end.millisecondsSinceEpoch}) ??
        const [];
    return (ToolText.events(raw, start, end, now), raw.isEmpty ? 'No events' : _plural(raw.length, 'event'));
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
    if (text.isEmpty) return ('The clipboard is empty.', 'Nothing copied');
    const limit = 6000;
    final clipped = text.length > limit ? '${text.substring(0, limit)}…' : text;
    return ('Copied text:\n$clipped', '${text.length} characters');
  }
}

/// "since" arguments: a local ISO 8601 date-time; default: 24 hours ago.
DateTime _since(Object? v) => DateTime.tryParse('${v ?? ''}') ?? DateTime.now().subtract(const Duration(hours: 24));

const _sinceParam = {
  'since': {'type': 'string', 'description': 'e.g. 2026-10-09T08:00; default 24 h ago'},
};

String _plural(int n, String one) => '$n $one${n == 1 ? '' : 's'}';

class ContactsTool extends AgentTool with _OsPermission {
  @override
  Permission get android => Permission.contacts;
  @override
  String get kind => 'contacts';
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
  Future<(String, String)> run(Map<String, Object?> args) async {
    final query = '${args['name'] ?? ''}';
    final found = await _data.invokeListMethod<Object?>('contacts', {'query': query}) ?? const [];
    return (ToolText.contacts(found, query), _plural(found.length, 'contact'));
  }
}

class MessagesTool extends AgentTool with _OsPermission {
  @override
  Permission get android => Permission.sms;
  @override
  String get kind => 'messages';
  @override
  String get name => 'read_sms';
  @override
  String get label => 'Read your text messages';
  @override
  String get description => Platform.isAndroid
      ? 'SMS sent or received since a time, optionally with one person. Chat apps and email: read_notifications.'
      : 'iMessage and SMS sent or received since a time, optionally with one person.';
  @override
  Map<String, Object?> get parameters => {
        ..._sinceParam,
        'from': {'type': 'string', 'description': 'name or number'},
      };

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final since = _since(args['since']);
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>(
            'sms', {'since': since.millisecondsSinceEpoch, 'from': args['from']}) ??
        const [];
    return (ToolText.sms(raw, since, DateTime.now()), _plural(raw.length, 'message'));
  }
}

class CallsTool extends AgentTool with _OsPermission {
  @override
  Permission get android => Permission.phone;
  @override
  String get kind => 'calls';
  @override
  String get name => 'recent_calls';
  @override
  String get label => 'Checked your calls';
  @override
  String get description => 'Phone calls since a time: who, when, missed or not, length.';
  @override
  Map<String, Object?> get parameters => _sinceParam;

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final since = _since(args['since']);
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>('calls', {'since': since.millisecondsSinceEpoch}) ??
        const [];
    return (ToolText.calls(raw, since, DateTime.now()), _plural(raw.length, 'call'));
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
    final app = '${args['app'] ?? ''}';
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>(
            'notifications', {'since': since.millisecondsSinceEpoch}) ??
        const [];
    final text = ToolText.notifications(raw, since, app, DateTime.now());
    final shown = '\n'.allMatches(text).length;
    return (text, _plural(shown, 'notification'));
  }
}

class EmailTool extends AgentTool with _OsPermission {
  @override
  Permission get android => Permission.unknown; // macOS only (Apple Mail)
  @override
  String get kind => 'email';
  @override
  String get name => 'read_email';
  @override
  String get label => 'Read your email';
  @override
  String get description =>
      'Emails received since a time (every account in Mail): sender, subject, preview; optionally only those whose '
      'sender or subject contains a word.';
  @override
  Map<String, Object?> get parameters => {
        ..._sinceParam,
        'query': {'type': 'string', 'description': 'a sender or a word of the subject'},
      };

  @override
  Future<(String, String)> run(Map<String, Object?> args) async {
    final since = _since(args['since']);
    final query = '${args['query'] ?? ''}';
    final raw = await _data.invokeListMethod<Map<Object?, Object?>>(
            'email', {'since': since.millisecondsSinceEpoch, 'query': query}) ??
        const [];
    return (ToolText.emails(raw, since, query, DateTime.now()), _plural(raw.length, 'email'));
  }
}

/// The tools Liyab has, and which ones the user turned on.
class Toolbox {
  Toolbox(this._prefs);
  final SharedPreferences _prefs;

  /// The sources each platform's liyab/data channel reads. macOS has no API
  /// for other apps' notifications; its email comes from Apple Mail (on
  /// Android, email previews arrive as notifications). Elsewhere, only what
  /// Flutter reaches on every platform.
  final List<AgentTool> all = Platform.isAndroid
      ? [CalendarTool(), NotificationsTool(), MessagesTool(), CallsTool(), ContactsTool(), ClipboardTool()]
      : Platform.isMacOS
      ? [CalendarTool(), EmailTool(), MessagesTool(), CallsTool(), ContactsTool(), ClipboardTool()]
      : [ClipboardTool()];

  bool isOn(AgentTool t) => _prefs.getBool(t.setting) ?? false;
  Future<void> setOn(AgentTool t, bool on) => _prefs.setBool(t.setting, on);

  /// Tools turned on whose permission is granted: the ones offered to the model.
  /// A tool whose check fails counts as not granted: it must not take the
  /// other tools out of the system prompt with it.
  Future<List<AgentTool>> enabled() async => [
        for (final t in all)
          if (isOn(t) && await t.permitted().catchError((Object _) => false)) t
      ];

  AgentTool? byName(String name) => all.where((t) => t.name == name).firstOrNull;
}
