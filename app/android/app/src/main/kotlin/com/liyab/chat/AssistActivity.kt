package com.liyab.chat

import android.content.Context
import android.content.Intent
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.android.FlutterActivityLaunchConfigs.BackgroundMode
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * The assistant sheet, opened by the system assist gesture (ACTION_ASSIST:
 * long press on power or home, when Liyab is the default digital assistant).
 * A see-through window over the current app, drawn by the shared Flutter
 * engine in "assist" mode; it never loads a second model.
 */
class AssistActivity : FlutterActivity() {
    private var channel: MethodChannel? = null

    override fun provideFlutterEngine(context: Context): FlutterEngine = LiyabEngine.get(context)

    override fun shouldDestroyEngineWithHost(): Boolean = false

    override fun getBackgroundMode(): BackgroundMode = BackgroundMode.transparent

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        DeviceChannel.attach(this, flutterEngine)
        channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "liyab/assist").also {
            it.setMethodCallHandler { call, result ->
                when (call.method) {
                    "close" -> {
                        finish()
                        result.success(null)
                    }
                    "openApp" -> {
                        startActivity(Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
                        finish()
                        result.success(null)
                    }
                    else -> result.notImplemented()
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        channel?.invokeMethod("opened", null)
    }

    override fun onPause() {
        super.onPause()
        channel?.invokeMethod("closed", null)
        if (!isChangingConfigurations) finish() // the sheet never lingers behind other apps
    }
}
