package com.liyab.chat

import android.app.Activity
import android.app.role.RoleManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.provider.Settings
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlin.math.abs

/**
 * The "liyab/device" channel: what only the Android SDK exposes (power draw,
 * thermal state, the default-assistant role). The engine itself is reached
 * from Dart through dart:ffi. Attached by every activity that hosts the shared
 * engine, so calls go to the visible one.
 */
object DeviceChannel {
    fun attach(activity: Activity, engine: FlutterEngine) {
        MethodChannel(engine.dartExecutor.binaryMessenger, "liyab/device").setMethodCallHandler { call, result ->
            when (call.method) {
                "power" -> result.success(power(activity))
                "thermal" -> result.success(thermal(activity))
                "isAssistant" -> result.success(isAssistant(activity))
                "openAssistantSettings" -> {
                    // The assistant role cannot be requested with a dialog; the
                    // user picks it in the system's default-apps settings.
                    activity.startActivity(Intent(Settings.ACTION_MANAGE_DEFAULT_APPS_SETTINGS))
                    result.success(null)
                }
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

    private fun isAssistant(context: Context): Boolean {
        val roles = context.getSystemService(RoleManager::class.java) ?: return false
        return roles.isRoleAvailable(RoleManager.ROLE_ASSISTANT) && roles.isRoleHeld(RoleManager.ROLE_ASSISTANT)
    }
}
