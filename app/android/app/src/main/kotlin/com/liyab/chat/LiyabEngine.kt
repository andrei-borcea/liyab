package com.liyab.chat

import android.content.Context
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

    fun get(context: Context): FlutterEngine {
        FlutterEngineCache.getInstance().get(ID)?.let { return it }
        val engine = FlutterEngine(context.applicationContext)
        engine.dartExecutor.executeDartEntrypoint(DartExecutor.DartEntrypoint.createDefault())
        FlutterEngineCache.getInstance().put(ID, engine)
        return engine
    }
}
