package com.liyab.chat

import io.flutter.embedding.android.FlutterActivityLaunchConfigs.BackgroundMode
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * The assistant sheet, opened by the system assist gesture (ACTION_ASSIST:
 * long press on power or home, when Liyab is the default digital assistant).
 * A see-through window over the current app, drawn by the shared Flutter
 * engine in "assist" mode; it never loads a second model. Plugins stay bound
 * to the app window (the sheet needs none), and the sheet gives the engine's
 * surface up as soon as it pauses.
 */
class AssistActivity : SharedEngineActivity() {
    private var channel: MethodChannel? = null

    override fun getBackgroundMode(): BackgroundMode = BackgroundMode.transparent

    override fun shouldAttachEngineToActivity(): Boolean = false

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        channel = MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "liyab/assist").also {
            it.setMethodCallHandler { call, result ->
                when (call.method) {
                    "close" -> {
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
        channel?.invokeMethod("closed", null)
        super.onPause()
        releaseSurface()
        if (!isChangingConfigurations) finish() // the sheet never lingers behind other apps
    }
}
