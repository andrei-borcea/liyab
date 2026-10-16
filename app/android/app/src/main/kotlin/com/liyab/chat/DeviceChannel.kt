package com.liyab.chat

import android.app.AppOpsManager
import android.app.role.RoleManager
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.net.Uri
import android.os.Process
import android.provider.Settings
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlin.math.abs

/**
 * The "liyab/device" channel: what only the Android SDK exposes (power draw,
 * thermal state, the default-assistant role, HyperOS's Autostart switch). The engine itself is reached
 * from Dart through dart:ffi. Attached once, when the shared engine is created
 * (LiyabEngine), with the application context: it needs no activity, so it
 * also serves an engine running without a window.
 */
object DeviceChannel {
    fun attach(context: Context, engine: FlutterEngine) {
        MethodChannel(engine.dartExecutor.binaryMessenger, "liyab/device").setMethodCallHandler { call, result ->
            when (call.method) {
                "power" -> result.success(power(context))
                "thermal" -> result.success(thermal(context))
                "isAssistant" -> result.success(isAssistant(context))
                "openAssistantSettings" -> {
                    // The assistant role cannot be requested with a dialog; the
                    // user picks it in the system's default-apps settings.
                    // (From the application context: a task of its own.)
                    context.startActivity(
                        Intent(Settings.ACTION_MANAGE_DEFAULT_APPS_SETTINGS).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                    result.success(null)
                }
                "autostart" -> result.success(autostart(context))
                "openAutostartSettings" -> result.success(openAutostartSettings(context))
                else -> result.notImplemented()
            }
        }
    }

    /**
     * Battery power in watts (positive while discharging) and whether a
     * charger is connected. CURRENT_NOW is µA on most phones and mA on some:
     * values of 20000 or more are read as µA. While charging it is the net
     * flow into the battery, not what the phone draws.
     */
    private fun power(context: Context): Map<String, Any> {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val raw = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val amps = if (abs(raw) >= 20000) raw / 1e6 else raw / 1e3
        val battery = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val volts = (battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0) / 1000.0
        val plugged = (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val tenths = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return mapOf("watts" to abs(amps * volts), "charging" to plugged, "level" to level, "batteryC" to tenths / 10.0)
    }

    /**
     * The OS thermal status (0 none .. 6 shutdown) and, on Android 11+, the
     * headroom forecast 10 s ahead (1.0 = severe throttling; NaN when unknown).
     */
    private fun thermal(context: Context): Map<String, Any> {
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val status = if (Build.VERSION.SDK_INT >= 29) pm.currentThermalStatus else 0
        val headroom = if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(10) else Float.NaN
        return mapOf("status" to status, "headroom" to headroom.toDouble())
    }

    /**
     * MIUI / HyperOS (Xiaomi, Redmi, POCO) refuse to start an app's process
     * by themselves unless the user allowed its "Autostart": after a reboot
     * the notification listener is then never bound, so Liyab's process does
     * not start and its local API stays closed until Liyab is opened (seen in
     * the log as "AutoStartManagerService: Reject service …
     * LiyabNotificationListener"). No API turns it on; the user can, in the
     * system's Autostart screen (openAutostartSettings).
     *
     * Returns "hyperos": whether this is such a system, and "allowed": the
     * switch read through MIUI's own app-op (10008, not in the public SDK,
     * read by reflection), or null when it cannot be read.
     */
    private fun autostart(context: Context): Map<String, Any?> {
        if (!isMiui()) return mapOf("hyperos" to false, "allowed" to null)
        val allowed = try {
            val ops = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
            val check = AppOpsManager::class.java.getMethod(
                "checkOpNoThrow", Int::class.javaPrimitiveType, Int::class.javaPrimitiveType, String::class.java)
            check.invoke(ops, MIUI_OP_AUTO_START, Process.myUid(), context.packageName) == AppOpsManager.MODE_ALLOWED
        } catch (e: Exception) {
            null
        }
        return mapOf("hyperos" to true, "allowed" to allowed)
    }

    /**
     * Opens the system's Autostart screen (HyperOS / MIUI Security app), or
     * this app's settings page where it is missing. Returns false when
     * neither could be opened.
     */
    private fun openAutostartSettings(context: Context): Boolean {
        val screens = listOf(
            Intent().setComponent(
                ComponentName("com.miui.securitycenter", "com.miui.permcenter.autostart.AutoStartManagementActivity")),
            Intent(Settings.ACTION_APPLICATION_DETAILS_SETTINGS, Uri.parse("package:${context.packageName}")),
        )
        for (intent in screens) {
            try {
                context.startActivity(intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                return true
            } catch (e: Exception) {
                // Not on this system (or not exported to us): try the next one.
            }
        }
        return false
    }

    private fun isMiui(): Boolean {
        val manufacturer = Build.MANUFACTURER.lowercase()
        if (manufacturer == "xiaomi" || manufacturer == "redmi" || manufacturer == "poco") return true
        return systemProperty("ro.mi.os.version.name").isNotEmpty() || systemProperty("ro.miui.ui.version.name").isNotEmpty()
    }

    private fun systemProperty(name: String): String = try {
        Class.forName("android.os.SystemProperties").getMethod("get", String::class.java).invoke(null, name) as String
    } catch (e: Exception) {
        ""
    }

    /** MIUI's app-op for Autostart (android.app.AppOpsManager has no constant for it). */
    private const val MIUI_OP_AUTO_START = 10008

    private fun isAssistant(context: Context): Boolean {
        val roles = context.getSystemService(RoleManager::class.java) ?: return false
        return roles.isRoleAvailable(RoleManager.ROLE_ASSISTANT) && roles.isRoleHeld(RoleManager.ROLE_ASSISTANT)
    }
}
