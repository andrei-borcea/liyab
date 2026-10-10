package com.liyab.chat

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * A foreground service held only while Liyab works with its window hidden: a
 * reply still being written and the save after it, a model loading, a request
 * of another app through the local API. While it runs, Android treats the
 * process as in use: on HyperOS it keeps all eight cores (a hidden app gets
 * four, and while another app is in use a reply drops from ~7 to 1.3 tokens/s
 * on a 35B) and is not frozen (a frozen process runs no code, so the reply
 * stops where it was). A partial wake lock lets the reply finish with the
 * screen off. The service computes nothing itself and the app stops it as soon
 * as the work is done, so an idle Liyab holds neither.
 *
 * Its notification asks every Android for the same things, through standard
 * means, and each system shows what it supports:
 *  - a Live Update (Android 16): the system may promote it to a chip in the
 *    status bar (or its island) and a card on the lock screen;
 *  - one alert, silent and still, when the service starts, so the system's
 *    new-notification effects (the always-on display's pulse or edge light,
 *    where the phone has them) mark the start; updates do not alert again.
 *
 * Android allows starting it only while, or just after, Liyab is on screen, or
 * at any time when the user exempted Liyab from battery optimisation: work that
 * begins with Liyab hidden otherwise runs without it, and [start] reports that
 * by returning false.
 */
class WorkService : Service() {
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // startForeground first, always: a service started with
        // startForegroundService() must call it within seconds, even when
        // stop() came in between, or Android stops the app.
        val notification = notification(this, intent?.getStringExtra(EXTRA_TEXT) ?: "")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(NOTIFICATION_ID, notification, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE)
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
        running = true
        if (!wanted) {
            stopSelf()
        } else if (wakeLock == null) {
            // Not reference counted: release() is then safe after the timeout.
            wakeLock = getSystemService(PowerManager::class.java)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "liyab:work")
                .apply { setReferenceCounted(false); acquire(MAX_WAKE_MS) }
        }
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        wakeLock?.release()
        wakeLock = null
        running = false
        super.onDestroy()
    }

    companion object {
        private const val TAG = "liyab-work"
        private const val CHANNEL_ID = "working"
        private const val LIGHT_COLOR = 0xFFFF7A3D.toInt() // the flame's orange
        private const val NOTIFICATION_ID = 42
        private const val EXTRA_TEXT = "text"

        // Notification.EXTRA_REQUEST_PROMOTED_ONGOING, public from SDK 36.1 on
        // (the key androidx's NotificationCompat writes); compileSdk is 36.
        private const val EXTRA_REQUEST_PROMOTED_ONGOING = "android.requestPromotedOngoing"

        // Longer than any reply on a phone: it bounds the wake lock should the
        // app never stop the service.
        private const val MAX_WAKE_MS = 30 * 60 * 1000L

        // Both touched on the main thread only (channel calls and service
        // callbacks), so a stop() that arrives before onStartCommand still wins.
        private var wanted = false
        private var running = false

        /** Starts the service, or updates its text; false when Android does not allow it now. */
        fun start(context: Context, text: String): Boolean {
            wanted = true
            return try {
                context.startForegroundService(Intent(context, WorkService::class.java).putExtra(EXTRA_TEXT, text))
                true
            } catch (e: IllegalStateException) {
                // ForegroundServiceStartNotAllowedException (Android 12+): Liyab is in the background.
                Log.w(TAG, "Not started: ${e.message}")
                wanted = false
                false
            }
        }

        fun stop(context: Context) {
            wanted = false
            if (running) context.stopService(Intent(context, WorkService::class.java))
        }

        private fun notification(context: Context, text: String): Notification {
            val manager = context.getSystemService(NotificationManager::class.java)
            if (manager.getNotificationChannel(CHANNEL_ID) == null) {
                manager.deleteNotificationChannel("work") // an earlier build's channel, created without the light
                // Default importance and a notification light, so the system plays its new-notification effects
                // with the screen off; no sound, no vibration. Android lets only the user change a channel's
                // settings once it exists.
                manager.createNotificationChannel(
                    NotificationChannel(CHANNEL_ID, "Working in the background", NotificationManager.IMPORTANCE_DEFAULT)
                        .apply {
                            description = "Shown while Liyab finishes a reply with its window closed"
                            setSound(null, null)
                            enableVibration(false)
                            enableLights(true)
                            lightColor = LIGHT_COLOR
                        })
            }
            val open = PendingIntent.getActivity(
                context, 0, Intent(context, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE)
            return Notification.Builder(context, CHANNEL_ID)
                .setSmallIcon(R.drawable.ic_launcher_monochrome)
                .setContentTitle("Liyab")
                .setContentText(text)
                .setContentIntent(open)
                .setCategory(Notification.CATEGORY_PROGRESS)
                .setOngoing(true)
                .setOnlyAlertOnce(true)
                // At once: by default Android holds a service's notification back for 10 s.
                .setForegroundServiceBehavior(Notification.FOREGROUND_SERVICE_IMMEDIATE)
                .addExtras(Bundle().apply { putBoolean(EXTRA_REQUEST_PROMOTED_ONGOING, true) })
                .apply { if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.BAKLAVA) setShortCriticalText("Liyab") }
                .build()
        }
    }
}

/** The "liyab/work" channel: Dart holds the service while it has work and Liyab is hidden. */
object WorkChannel {
    fun attach(context: Context, engine: FlutterEngine) {
        MethodChannel(engine.dartExecutor.binaryMessenger, "liyab/work").setMethodCallHandler { call, result ->
            when (call.method) {
                "start" -> result.success(WorkService.start(context, call.arguments as? String ?: ""))
                "stop" -> {
                    WorkService.stop(context)
                    result.success(null)
                }
                else -> result.notImplemented()
            }
        }
    }
}
