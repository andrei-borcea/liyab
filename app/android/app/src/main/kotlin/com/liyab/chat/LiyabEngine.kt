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
 * load a model.
 */
object LiyabEngine {
    private const val ID = "liyab"

    /** The activity in front: the only one that reports lifecycle states to the engine. */
    var host: SharedEngineActivity? = null

    /** The activity whose window the engine draws into. */
    var drawingInto: SharedEngineActivity? = null

    fun get(context: Context): FlutterEngine {
        FlutterEngineCache.getInstance().get(ID)?.let { return it }
        val engine = FlutterEngine(context.applicationContext)
        engine.dartExecutor.executeDartEntrypoint(DartExecutor.DartEntrypoint.createDefault())
        FlutterEngineCache.getInstance().put(ID, engine)
        return engine
    }
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

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        DeviceChannel.attach(this, flutterEngine)
        DataChannel.attach(applicationContext, flutterEngine)
        WorkChannel.attach(applicationContext, flutterEngine)
    }

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
