import Contacts
import EventKit
import FlutterMacOS
import Foundation
import SQLite3

/// The "liyab/data" channel on macOS: the user's own data the agent tools read
/// (lib/agent/tools.dart), with the replies of Android's DataChannel.kt, so the
/// Dart tools and their text formats stay the same. Everything is read on this
/// Mac, only when a tool runs, and only from sources the user turned on.
///
/// Permissions:
///  * calendar, contacts: the system's privacy prompts (EventKit, Contacts);
///  * messages, calls, email (Apple Mail): macOS has no API for them; their databases are readable
///    only with Full Disk Access, which the user grants in System Settings
///    (there is no prompt), so "request" opens that settings pane.
/// Notifications of other apps cannot be read on macOS at all.
enum DataChannel {
  private static let events = EKEventStore()
  private static let contactStore = CNContactStore()

  static func attach(_ messenger: FlutterBinaryMessenger) {
    FlutterMethodChannel(name: "liyab/data", binaryMessenger: messenger).setMethodCallHandler { call, result in
      let args = call.arguments as? [String: Any] ?? [:]
      switch call.method {
      case "permission": result(permitted(args["kind"] as? String ?? ""))
      case "requestPermission": request(args["kind"] as? String ?? "", result)
      case "openSettings":
        openSettings(args["kind"] as? String ?? "")
        result(nil)
      case "calendarEvents":
        result(calendarEvents(from: millis(args["start"]), to: millis(args["end"])))
      case "contacts": result(contacts(matching: args["query"] as? String ?? ""))
      case "sms":
        // SQLite reads off the main thread: the Messages database can be large.
        DispatchQueue.global(qos: .userInitiated).async {
          let rows = messages(since: millis(args["since"]), from: args["from"] as? String)
          DispatchQueue.main.async { result(rows) }
        }
      case "calls":
        DispatchQueue.global(qos: .userInitiated).async {
          let rows = calls(since: millis(args["since"]))
          DispatchQueue.main.async { result(rows) }
        }
      case "email":
        DispatchQueue.global(qos: .userInitiated).async {
          let rows = emails(since: millis(args["since"]), matching: args["query"] as? String)
          DispatchQueue.main.async { result(rows) }
        }
      default: result(FlutterMethodNotImplemented)
      }
    }
  }

  private static func millis(_ v: Any?) -> Date {
    Date(timeIntervalSince1970: ((v as? NSNumber)?.doubleValue ?? 0) / 1000)
  }

  // MARK: Permissions

  private static let messagesDB = NSHomeDirectory() + "/Library/Messages/chat.db"
  private static let callsDB = NSHomeDirectory() + "/Library/Application Support/CallHistoryDB/CallHistory.storedata"

  private static func permitted(_ kind: String) -> Bool {
    switch kind {
    case "calendar": return EKEventStore.authorizationStatus(for: .event) == .fullAccess
    case "contacts": return CNContactStore.authorizationStatus(for: .contacts) == .authorized
    // Full Disk Access shows only as whether the file opens.
    case "messages": return FileManager.default.isReadableFile(atPath: messagesDB) && canOpen(messagesDB)
    case "calls": return FileManager.default.isReadableFile(atPath: callsDB) && canOpen(callsDB)
    case "email": return mailDB.map(canOpen) ?? false
    default: return false
    }
  }

  private static func request(_ kind: String, _ result: @escaping FlutterResult) {
    switch kind {
    case "calendar":
      events.requestFullAccessToEvents { granted, _ in DispatchQueue.main.async { result(granted) } }
    case "contacts":
      contactStore.requestAccess(for: .contacts) { granted, _ in DispatchQueue.main.async { result(granted) } }
    case "messages", "calls", "email":
      if permitted(kind) { return result(true) }
      openSettings(kind)
      result(false)  // granted (or not) in System Settings; Dart checks again when Liyab is back in front
    default:
      result(false)
    }
  }

  /// The Privacy & Security pane where the user allows `kind` by hand.
  private static func openSettings(_ kind: String) {
    let pane = switch kind {
    case "calendar": "Privacy_Calendars"
    case "contacts": "Privacy_Contacts"
    default: "Privacy_AllFiles"
    }
    NSWorkspace.shared.open(URL(string: "x-apple.systempreferences:com.apple.preference.security?\(pane)")!)
  }

  // MARK: Calendar and contacts

  private static func calendarEvents(from start: Date, to end: Date) -> [[String: Any]] {
    guard permitted("calendar") else { return [] }
    let predicate = events.predicateForEvents(withStart: start, end: end, calendars: nil)
    return events.events(matching: predicate).sorted { $0.startDate < $1.startDate }.prefix(100).map { e in
      [
        "title": e.title ?? "",
        "begin": Int(e.startDate.timeIntervalSince1970 * 1000),
        "end": Int(e.endDate.timeIntervalSince1970 * 1000),
        "allDay": e.isAllDay,
        "location": e.location ?? "",
        "calendar": e.calendar?.title ?? "",
        "description": String((e.notes ?? "").prefix(300)),
      ]
    }
  }

  private static let contactKeys: [CNKeyDescriptor] = [
    CNContactFormatter.descriptorForRequiredKeys(for: .fullName),
    CNContactPhoneNumbersKey as CNKeyDescriptor,
    CNContactEmailAddressesKey as CNKeyDescriptor,
  ]

  private static func contacts(matching query: String) -> [[String: Any]] {
    guard permitted("contacts"), !query.isEmpty else { return [] }
    let found = (try? contactStore.unifiedContacts(
      matching: CNContact.predicateForContacts(matchingName: query), keysToFetch: contactKeys)) ?? []
    return found.prefix(20).map { c in
      [
        "name": CNContactFormatter.string(from: c, style: .fullName) ?? "",
        "phones": c.phoneNumbers.map { $0.value.stringValue },
        "emails": c.emailAddresses.map { $0.value as String },
      ]
    }
  }

  /// Name for a phone number or email address, when Contacts is allowed;
  /// the address itself otherwise. Cached per lookup batch by the caller.
  private static func name(for address: String) -> String? {
    guard permitted("contacts"), !address.isEmpty else { return nil }
    let predicate = address.contains("@")
      ? CNContact.predicateForContacts(matchingEmailAddress: address)
      : CNContact.predicateForContacts(matching: CNPhoneNumber(stringValue: address))
    guard let c = try? contactStore.unifiedContacts(matching: predicate, keysToFetch: contactKeys).first else { return nil }
    return CNContactFormatter.string(from: c, style: .fullName)
  }

  // MARK: Messages and calls (SQLite, read-only)

  /// Apple's databases count time from 2001-01-01 (Messages in nanoseconds).
  private static let appleEpoch: TimeInterval = 978_307_200

  private static func canOpen(_ path: String) -> Bool {
    var db: OpaquePointer?
    defer { sqlite3_close(db) }
    guard sqlite3_open_v2(path, &db, SQLITE_OPEN_READONLY, nil) == SQLITE_OK else { return false }
    // Opening can succeed without access; reading the schema cannot.
    return sqlite3_exec(db, "SELECT count(*) FROM sqlite_master", nil, nil, nil) == SQLITE_OK
  }

  /// Runs `sql` read-only, binding one Int64 parameter; calls `row` per result row.
  private static func query(_ path: String, _ sql: String, _ param: Int64, row: (OpaquePointer) -> Bool) {
    var db: OpaquePointer?
    defer { sqlite3_close(db) }
    // Immutable read: never touches the live database's locks or WAL.
    let uri = "file:\(path.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? path)?mode=ro&immutable=1"
    guard sqlite3_open_v2(uri, &db, SQLITE_OPEN_READONLY | SQLITE_OPEN_URI, nil) == SQLITE_OK else { return }
    var stmt: OpaquePointer?
    defer { sqlite3_finalize(stmt) }
    guard sqlite3_prepare_v2(db, sql, -1, &stmt, nil) == SQLITE_OK, let stmt else { return }
    sqlite3_bind_int64(stmt, 1, param)
    while sqlite3_step(stmt) == SQLITE_ROW, row(stmt) {}
  }

  private static func text(_ stmt: OpaquePointer, _ col: Int32) -> String? {
    sqlite3_column_text(stmt, col).map { String(cString: $0) }
  }

  /// iMessage and SMS since `since`, newest first, optionally only with a
  /// person whose name or address contains `from`.
  private static func messages(since: Date, from: String?) -> [[String: Any]] {
    guard permitted("messages") else { return [] }
    let sinceNs = Int64((since.timeIntervalSince1970 - appleEpoch) * 1_000_000_000)
    let filter = (from ?? "").lowercased()
    var names: [String: String] = [:]
    var out: [[String: Any]] = []
    query(messagesDB, """
      SELECT m.text, m.attributedBody, m.date, m.is_from_me, h.id
      FROM message m LEFT JOIN handle h ON h.ROWID = m.handle_id
      WHERE m.date >= ? AND m.item_type = 0 ORDER BY m.date DESC LIMIT 400
      """, sinceNs) { stmt in
      let address = text(stmt, 4) ?? ""
      let who = names[address] ?? {
        let n = name(for: address) ?? address
        names[address] = n
        return n
      }()
      if !filter.isEmpty && !who.lowercased().contains(filter) && !address.lowercased().contains(filter) { return true }
      guard let body = text(stmt, 0) ?? attributedText(stmt, 1), !body.isEmpty else { return true }
      out.append([
        "from": who,
        "number": address,
        "text": String(body.prefix(1000)),
        "time": Int((Double(sqlite3_column_int64(stmt, 2)) / 1e9 + appleEpoch) * 1000),
        "sent": sqlite3_column_int(stmt, 3) != 0,
      ])
      return out.count < 50
    }
    return out
  }

  /// Recent macOS versions keep a message's text only in `attributedBody`, an
  /// NSAttributedString in the old typedstream archive format.
  private static func attributedText(_ stmt: OpaquePointer, _ col: Int32) -> String? {
    guard let bytes = sqlite3_column_blob(stmt, col) else { return nil }
    let data = Data(bytes: bytes, count: Int(sqlite3_column_bytes(stmt, col)))
    return (NSUnarchiver.unarchiveObject(with: data) as? NSAttributedString)?.string
  }

  /// Phone and FaceTime calls since `since`, newest first.
  private static func calls(since: Date) -> [[String: Any]] {
    guard permitted("calls") else { return [] }
    var names: [String: String] = [:]
    var out: [[String: Any]] = []
    query(callsDB, """
      SELECT ZADDRESS, ZNAME, ZORIGINATED, ZANSWERED, ZDATE, ZDURATION
      FROM ZCALLRECORD WHERE ZDATE >= ? ORDER BY ZDATE DESC LIMIT 50
      """, Int64(since.timeIntervalSince1970 - appleEpoch)) { stmt in
      let address = text(stmt, 0) ?? ""
      let who = text(stmt, 1).flatMap { $0.isEmpty ? nil : $0 } ?? names[address] ?? {
        let n = name(for: address) ?? address
        names[address] = n
        return n
      }()
      let outgoing = sqlite3_column_int(stmt, 2) != 0, answered = sqlite3_column_int(stmt, 3) != 0
      out.append([
        "who": who,
        "number": address,
        "type": outgoing ? "outgoing" : (answered ? "incoming" : "missed"),
        "time": Int((sqlite3_column_double(stmt, 4) + appleEpoch) * 1000),
        "seconds": Int(sqlite3_column_double(stmt, 5)),
      ])
      return true
    }
    return out
  }

  /// Apple Mail's index of every message of every account set up in Mail:
  /// ~/Library/Mail/V<n>/MailData/Envelope Index (the newest V<n>). Listing
  /// ~/Library/Mail already needs Full Disk Access, so nil also means "not
  /// allowed" (or Mail never used).
  private static var mailDB: String? {
    let root = NSHomeDirectory() + "/Library/Mail"
    let versions = ((try? FileManager.default.contentsOfDirectory(atPath: root)) ?? [])
      .filter { $0.hasPrefix("V") }
      .sorted { (Int($0.dropFirst()) ?? 0) < (Int($1.dropFirst()) ?? 0) }
    guard let newest = versions.last else { return nil }
    let path = "\(root)/\(newest)/MailData/Envelope Index"
    return FileManager.default.fileExists(atPath: path) ? path : nil
  }

  /// Emails received since `since`, newest first, optionally only those whose
  /// sender or subject contains `search`; with Mail's own preview of the body.
  private static func emails(since: Date, matching search: String?) -> [[String: Any]] {
    guard let db = mailDB else { return [] }
    let filter = (search ?? "").lowercased()
    var out: [[String: Any]] = []
    let sql = """
      SELECT a.address, a.comment, s.subject, m.date_received, m.read, sm.summary, mb.url
      FROM messages m
      LEFT JOIN addresses a ON a.ROWID = m.sender
      LEFT JOIN subjects s ON s.ROWID = m.subject
      LEFT JOIN summaries sm ON sm.ROWID = m.summary
      LEFT JOIN mailboxes mb ON mb.ROWID = m.mailbox
      WHERE m.date_received >= ? AND m.deleted = 0
      ORDER BY m.date_received DESC LIMIT 400
      """
    query(db, sql, Int64(since.timeIntervalSince1970)) { stmt in
      let address = text(stmt, 0) ?? "", name = text(stmt, 1) ?? ""
      let subject = text(stmt, 2) ?? ""
      let mailbox = (text(stmt, 6) ?? "").removingPercentEncoding ?? ""
      // Sent, drafts, junk and trash are not what "my email" means.
      let skipped = ["sent", "draft", "junk", "spam", "trash", "deleted", "bin"]
      if skipped.contains(where: { mailbox.lowercased().contains($0) }) { return true }
      if !filter.isEmpty
        && ![address, name, subject].contains(where: { $0.lowercased().contains(filter) }) { return true }
      out.append([
        "from": name.isEmpty ? address : name,
        "address": address,
        "subject": subject,
        "time": Int(sqlite3_column_int64(stmt, 3)) * 1000,
        "read": sqlite3_column_int(stmt, 4) != 0,
        "preview": String((text(stmt, 5) ?? "").prefix(400)),
      ])
      return out.count < 40
    }
    return out
  }
}
