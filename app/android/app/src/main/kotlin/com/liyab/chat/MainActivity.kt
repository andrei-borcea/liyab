package com.liyab.chat

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel
import kotlin.math.abs

/**
 * Hosts the Flutter UI and answers the "liyab/device" channel: power draw and
 * thermal state, which only the Android SDK exposes. The engine itself is
 * reached from Dart through dart:ffi, not through this activity.
 */
class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "liyab/device").setMethodCallHandler { call, result ->
            when (call.method) {
                "power" -> result.success(power())
                "thermal" -> result.success(thermal())
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
    private fun power(): Map<String, Any> {
        val bm = getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val raw = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
        val amps = if (abs(raw) >= 20000) raw / 1e6 else raw / 1e3
        val battery = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val volts = (battery?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0) ?: 0) / 1000.0
        val plugged = (battery?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0
        val level = battery?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val tenths = battery?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0
        return mapOf(
            "watts" to abs(amps * volts),
            "charging" to plugged,
            "level" to level,
            "batteryC" to tenths / 10.0,
        )
    }

    /**
     * The OS thermal status (0 none .. 6 shutdown) and, on Android 11+, the
     * headroom forecast 10 s ahead (1.0 = severe throttling; NaN when unknown).
     */
    private fun thermal(): Map<String, Any> {
        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        val status = if (Build.VERSION.SDK_INT >= 29) pm.currentThermalStatus else 0
        val headroom = if (Build.VERSION.SDK_INT >= 30) pm.getThermalHeadroom(10) else Float.NaN
        return mapOf("status" to status, "headroom" to headroom.toDouble())
    }
}
