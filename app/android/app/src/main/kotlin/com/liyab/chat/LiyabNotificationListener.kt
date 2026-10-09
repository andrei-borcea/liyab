package com.liyab.chat

import android.app.Notification
import android.content.ComponentName
import android.content.Context
import android.provider.Settings
import android.service.notification.NotificationListenerService
import android.service.notification.StatusBarNotification

/**
 * Keeps the notifications the phone receives (messages, email previews, app
 * alerts) so the assistant can answer "what did Marco write me today?". Runs
 * only after the user grants Notification access in Android's settings; the
 * list lives in memory (latest 300, none of Liyab's own), is never written to
 * disk, and is gone when the process ends.
 */
class LiyabNotificationListener : NotificationListenerService() {
    data class Item(val app: String, val title: String, val text: String, val time: Long)

    companion object {
        private const val MAX = 300
        private val items = ArrayDeque<Item>()

        @Synchronized
        fun recent(since: Long): List<Item> = items.filter { it.time >= since }

        @Synchronized
        private fun add(item: Item) {
            items.removeAll { it.app == item.app && it.title == item.title && it.text == item.text }
            items.addLast(item)
            while (items.size > MAX) items.removeFirst()
        }

        fun granted(context: Context): Boolean {
            val enabled = Settings.Secure.getString(context.contentResolver, "enabled_notification_listeners") ?: return false
            return enabled.split(':').any { ComponentName.unflattenFromString(it)?.packageName == context.packageName }
        }
    }

    override fun onListenerConnected() {
        // What is already in the shade when access is granted or the service restarts.
        activeNotifications?.forEach { record(it) }
    }

    override fun onNotificationPosted(sbn: StatusBarNotification) = record(sbn)

    private fun record(sbn: StatusBarNotification) {
        if (sbn.packageName == packageName || sbn.isOngoing) return
        val extras = sbn.notification.extras
        val title = extras.getCharSequence(Notification.EXTRA_TITLE)?.toString().orEmpty()
        // Messaging apps put the whole conversation in EXTRA_TEXT_LINES / big text.
        val lines = extras.getCharSequenceArray(Notification.EXTRA_TEXT_LINES)?.joinToString("\n")
        val text = (extras.getCharSequence(Notification.EXTRA_BIG_TEXT) ?: lines
            ?: extras.getCharSequence(Notification.EXTRA_TEXT))?.toString().orEmpty()
        if (title.isEmpty() && text.isEmpty()) return
        val app = try {
            packageManager.getApplicationLabel(packageManager.getApplicationInfo(sbn.packageName, 0)).toString()
        } catch (e: Exception) {
            sbn.packageName
        }
        add(Item(app, title, text.take(1500), sbn.postTime))
    }
}
