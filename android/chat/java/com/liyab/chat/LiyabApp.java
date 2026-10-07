package com.liyab.chat;

import android.app.Application;

/** Process-wide setup: engine log capture, the debug-log pump and the shared engine holder. */
public final class LiyabApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        LiyabNative.enableLogs(1);  // info and above into the debug log
        EngineHolder.init(this);
        DebugLog.startPump();
        DebugLog.add("Liyab Chat started; models folder: " + EngineHolder.modelsDir());
    }
}
