package com.liyab.chat

import android.content.Context
import android.os.Bundle
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.android.FlutterView
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.embedding.engine.FlutterEngineCache
import io.flutter.embedding.engine.dart.DartExecutor

/**
 * The one Flutter engine (and so one Dart state, one loaded model) shared by
 * the app window and the assistant sheet. Created on first use by an activity,
 * not at process start: download jobs start the process too, and must not
 * load a model. The one exception is [startForApi].
 *
 * Main thread only.
 */
object LiyabEngine {
    private const val ID = "liyab"

    /** Dart entrypoint argument (main's args): the engine was started without a window, see [startForApi]. */
    private const val HIDDEN_ARG = "hidden"

    /** The activity in front: the only one that reports lifecycle states to the engine. */
    var host: SharedEngineActivity? = null

    /** The activity whose window the engine draws into. */
    var drawingInto: SharedEngineActivity? = null

    fun get(context: Context): FlutterEngine = cached() ?: create(context, hidden = false)

    /**
     * Android started the process without a window, for a component of its own
     * (today the notification listener, which the system binds while the user
     * grants notification access, and so starts the process at boot or after
     * it was stopped): with the local API on, starts the Dart side now, so the
     * API listens instead of waiting for Liyab to be opened. Dart then loads no
     * model until a request needs one (AppState.create's `hidden`). No-op when
     * the engine runs already or the API is off; costs nothing otherwise.
     *
     * The listening foreground service (WorkService) that Dart then starts is
     * allowed although no activity was ever shown: the system binds the
     * listener with BIND_FOREGROUND_SERVICE from system_server, which puts this
     * process at PROCESS_STATE_BOUND_FOREGROUND_SERVICE with the
     * background-FGS-launch capability (OomAdjuster), and ActiveServices lets
     * such a process start foreground services (REASON_PROC_STATE_BFGS;
     * AOSP 14 and 15). Should an OEM build refuse it, WorkService.start
     * reports it and the API still listens while the process is not frozen.
     */
    fun startForApi(context: Context) {
        if (cached() != null || !apiEnabled(context)) return
        create(context, hidden = true)
    }

    private fun cached(): FlutterEngine? = FlutterEngineCache.getInstance().get(ID)

    private fun create(context: Context, hidden: Boolean): FlutterEngine {
        val app = context.applicationContext
        val engine = FlutterEngine(app)
        // Channels that need no activity are attached once, here, so they also
        // serve an engine that runs without a window.
        DeviceChannel.attach(app, engine)
        DataChannel.attach(app, engine)
        WorkChannel.attach(app, engine)
        engine.dartExecutor.executeDartEntrypoint(
            DartExecutor.DartEntrypoint.createDefault(), if (hidden) listOf(HIDDEN_ARG) else null)
        FlutterEngineCache.getInstance().put(ID, engine)
        return engine
    }

    /**
     * Settings → Local API, as Dart's shared_preferences stores it on Android
     * (its legacy store: the "FlutterSharedPreferences" file, keys prefixed
     * "flutter."; DeviceSettings.apiEnabled in Dart).
     */
    private fun apiEnabled(context: Context): Boolean =
        context.getSharedPreferences("FlutterSharedPreferences", Context.MODE_PRIVATE)
            .getBoolean("flutter.api_enabled", false)
}

/**
 * A window on the shared engine. Two of them can overlap during a switch (the
 * sheet closing while the app opens), and a Flutter engine draws into one
 * surface and keeps one lifecycle state, so:
 *  - only the activity in front reports lifecycle states; the other's late
 *    onStop would otherwise pause the engine under the visible window;
 *  - the activity coming to the front takes the engine's surface back, since
 *    the other window may have drawn into its own meanwhile.
 */
abstract class SharedEngineActivity : FlutterActivity() {
    override fun provideFlutterEngine(context: Context): FlutterEngine = LiyabEngine.get(context)

    override fun shouldDestroyEngineWithHost(): Boolean = false

    override fun shouldDispatchAppLifecycleState(): Boolean = LiyabEngine.host === this

    override fun onCreate(savedInstanceState: Bundle?) {
        LiyabEngine.host = this
        super.onCreate(savedInstanceState)
        LiyabEngine.drawingInto = this // the delegate attached this window's view
    }

    override fun onResume() {
        LiyabEngine.host = this
        val engine = flutterEngine
        val view = flutterView()
        if (engine != null && view != null && LiyabEngine.drawingInto !== this) {
            if (view.isAttachedToFlutterEngine) view.detachFromFlutterEngine()
            view.attachToFlutterEngine(engine)
            LiyabEngine.drawingInto = this
        }
        super.onResume()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (LiyabEngine.host === this) LiyabEngine.host = null
        if (LiyabEngine.drawingInto === this) LiyabEngine.drawingInto = null
    }

    /** Stops drawing into this window, so a later detach cannot stop the other window's surface. */
    protected fun releaseSurface() {
        flutterView()?.let { if (it.isAttachedToFlutterEngine) it.detachFromFlutterEngine() }
        if (LiyabEngine.drawingInto === this) LiyabEngine.drawingInto = null
    }

    private fun flutterView(): FlutterView? = findViewById(FLUTTER_VIEW_ID)
}
