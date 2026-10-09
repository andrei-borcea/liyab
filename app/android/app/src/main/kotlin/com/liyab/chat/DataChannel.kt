package com.liyab.chat

import android.content.Context
import android.content.Intent
import android.provider.CalendarContract
import android.provider.CallLog
import android.provider.ContactsContract
import android.provider.Settings
import android.provider.Telephony
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * The "liyab/data" channel: the user's own data that agent tools read, on the
 * phone, after the user enabled the tool and Android granted its permission
 * (the Dart side checks both before calling). Nothing here leaves the device.
 */
object DataChannel {
    fun attach(context: Context, engine: FlutterEngine) {
        MethodChannel(engine.dartExecutor.binaryMessenger, "liyab/data").setMethodCallHandler { call, result ->
            when (call.method) {
                "calendarEvents" -> guarded(result) {
                    calendarEvents(context, (call.argument<Number>("start") ?: 0).toLong(), (call.argument<Number>("end") ?: 0).toLong())
                }
                "contacts" -> guarded(result) { contacts(context, call.argument<String>("query").orEmpty()) }
                "sms" -> guarded(result) {
                    sms(context, (call.argument<Number>("since") ?: 0).toLong(), call.argument<String>("from"))
                }
                "calls" -> guarded(result) { calls(context, (call.argument<Number>("since") ?: 0).toLong()) }
                "notifications" -> result.success(
                    LiyabNotificationListener.recent((call.argument<Number>("since") ?: 0).toLong()).map {
                        mapOf("app" to it.app, "title" to it.title, "text" to it.text, "time" to it.time)
                    })
                "notificationAccess" -> result.success(LiyabNotificationListener.granted(context))
                "openNotificationAccess" -> {
                    context.startActivity(
                        Intent(Settings.ACTION_NOTIFICATION_LISTENER_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }
    }

    private fun guarded(result: MethodChannel.Result, read: () -> Any?) {
        try {
            result.success(read())
        } catch (e: SecurityException) {
            result.error("permission", "permission not granted", null)
        }
    }

    /** Contacts whose name contains `query` (up to 10), with their numbers and email addresses. Needs READ_CONTACTS. */
    private fun contacts(context: Context, query: String): List<Map<String, Any?>> {
        val out = linkedMapOf<Long, MutableMap<String, Any?>>()
        val cr = context.contentResolver
        cr.query(
            ContactsContract.Contacts.CONTENT_URI,
            arrayOf(ContactsContract.Contacts._ID, ContactsContract.Contacts.DISPLAY_NAME_PRIMARY),
            "${ContactsContract.Contacts.DISPLAY_NAME_PRIMARY} LIKE ?", arrayOf("%$query%"),
            "${ContactsContract.Contacts.DISPLAY_NAME_PRIMARY} ASC",
        )?.use { c ->
            while (c.moveToNext() && out.size < 10) {
                out[c.getLong(0)] = mutableMapOf("name" to c.getString(1), "phones" to mutableListOf<String>(), "emails" to mutableListOf<String>())
            }
        }
        if (out.isEmpty()) return emptyList()
        val ids = out.keys.joinToString(",")
        cr.query(ContactsContract.CommonDataKinds.Phone.CONTENT_URI,
            arrayOf(ContactsContract.CommonDataKinds.Phone.CONTACT_ID, ContactsContract.CommonDataKinds.Phone.NUMBER),
            "${ContactsContract.CommonDataKinds.Phone.CONTACT_ID} IN ($ids)", null, null)?.use { c ->
            while (c.moveToNext()) @Suppress("UNCHECKED_CAST") (out[c.getLong(0)]?.get("phones") as? MutableList<String>)?.add(c.getString(1))
        }
        cr.query(ContactsContract.CommonDataKinds.Email.CONTENT_URI,
            arrayOf(ContactsContract.CommonDataKinds.Email.CONTACT_ID, ContactsContract.CommonDataKinds.Email.ADDRESS),
            "${ContactsContract.CommonDataKinds.Email.CONTACT_ID} IN ($ids)", null, null)?.use { c ->
            while (c.moveToNext()) @Suppress("UNCHECKED_CAST") (out[c.getLong(0)]?.get("emails") as? MutableList<String>)?.add(c.getString(1))
        }
        return out.values.toList()
    }

    /** The name saved for a phone number, or null. */
    private fun contactName(context: Context, number: String?): String? {
        if (number.isNullOrEmpty()) return null
        val uri = android.net.Uri.withAppendedPath(ContactsContract.PhoneLookup.CONTENT_FILTER_URI, android.net.Uri.encode(number))
        return try {
            context.contentResolver.query(uri, arrayOf(ContactsContract.PhoneLookup.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) c.getString(0) else null
            }
        } catch (e: SecurityException) {
            null // contacts not allowed: numbers only
        }
    }

    /**
     * SMS since `since` (epoch ms), newest first, up to 50; `from` keeps only
     * messages whose address or contact name contains it. Needs READ_SMS.
     */
    private fun sms(context: Context, since: Long, from: String?): List<Map<String, Any?>> {
        val out = mutableListOf<Map<String, Any?>>()
        context.contentResolver.query(
            Telephony.Sms.CONTENT_URI,
            arrayOf(Telephony.Sms.ADDRESS, Telephony.Sms.BODY, Telephony.Sms.DATE, Telephony.Sms.TYPE),
            "${Telephony.Sms.DATE} >= ?", arrayOf(since.toString()), "${Telephony.Sms.DATE} DESC",
        )?.use { c ->
            while (c.moveToNext() && out.size < 50) {
                val address = c.getString(0)
                val name = contactName(context, address)
                if (!from.isNullOrBlank() && listOfNotNull(address, name).none { it.contains(from, ignoreCase = true) }) continue
                out += mapOf(
                    "from" to (name ?: address),
                    "number" to address,
                    "text" to c.getString(1)?.take(1000),
                    "time" to c.getLong(2),
                    "sent" to (c.getInt(3) == Telephony.Sms.MESSAGE_TYPE_SENT),
                )
            }
        }
        return out
    }

    /** Calls since `since` (epoch ms), newest first, up to 50. Needs READ_CALL_LOG. */
    private fun calls(context: Context, since: Long): List<Map<String, Any?>> {
        val out = mutableListOf<Map<String, Any?>>()
        context.contentResolver.query(
            CallLog.Calls.CONTENT_URI,
            arrayOf(CallLog.Calls.NUMBER, CallLog.Calls.CACHED_NAME, CallLog.Calls.TYPE, CallLog.Calls.DATE, CallLog.Calls.DURATION),
            "${CallLog.Calls.DATE} >= ?", arrayOf(since.toString()), "${CallLog.Calls.DATE} DESC",
        )?.use { c ->
            while (c.moveToNext() && out.size < 50) {
                out += mapOf(
                    "who" to (c.getString(1)?.takeIf { it.isNotEmpty() } ?: c.getString(0)),
                    "number" to c.getString(0),
                    "type" to when (c.getInt(2)) {
                        CallLog.Calls.INCOMING_TYPE -> "incoming"
                        CallLog.Calls.OUTGOING_TYPE -> "outgoing"
                        CallLog.Calls.MISSED_TYPE -> "missed"
                        CallLog.Calls.REJECTED_TYPE -> "rejected"
                        else -> "other"
                    },
                    "time" to c.getLong(3),
                    "seconds" to c.getLong(4),
                )
            }
        }
        return out
    }

    /**
     * Event instances overlapping [start, end) (epoch ms), recurring events
     * expanded, in start order: title, begin / end (epoch ms), all-day flag,
     * location, calendar name, and the first 300 characters of the description.
     * Needs READ_CALENDAR.
     */
    private fun calendarEvents(context: Context, start: Long, end: Long): List<Map<String, Any?>> {
        val uri = CalendarContract.Instances.CONTENT_URI.buildUpon()
            .appendPath(start.toString())
            .appendPath(end.toString())
            .build()
        val projection = arrayOf(
            CalendarContract.Instances.TITLE,
            CalendarContract.Instances.BEGIN,
            CalendarContract.Instances.END,
            CalendarContract.Instances.ALL_DAY,
            CalendarContract.Instances.EVENT_LOCATION,
            CalendarContract.Instances.CALENDAR_DISPLAY_NAME,
            CalendarContract.Instances.DESCRIPTION,
        )
        val events = mutableListOf<Map<String, Any?>>()
        context.contentResolver.query(uri, projection, null, null, "${CalendarContract.Instances.BEGIN} ASC")?.use { c ->
            while (c.moveToNext() && events.size < 100) {
                events += mapOf(
                    "title" to c.getString(0),
                    "begin" to c.getLong(1),
                    "end" to c.getLong(2),
                    "allDay" to (c.getInt(3) != 0),
                    "location" to c.getString(4),
                    "calendar" to c.getString(5),
                    "description" to c.getString(6)?.take(300),
                )
            }
        }
        return events
    }
}
