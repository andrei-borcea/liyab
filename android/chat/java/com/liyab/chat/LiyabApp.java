package com.liyab.chat;

import android.app.Activity;
import android.app.Application;
import android.os.Bundle;

/**
 * Process-wide setup: engine log capture, the debug-log pump, the shared engine holder, and the
 * performance overlay, which follows the user onto every screen while it is enabled.
 */
public final class LiyabApp extends Application {
    @Override
    public void onCreate() {
        super.onCreate();
        LiyabNative.enableLogs(1);  // info and above into the debug log
        EngineHolder.init(this);
        DebugLog.startPump();
        DebugLog.add("Liyab Chat started; models folder: " + EngineHolder.modelsDir());
        registerActivityLifecycleCallbacks(new ActivityLifecycleCallbacks() {
            @Override public void onActivityResumed(Activity a) { PerfOverlay.attach(a); }
            @Override public void onActivityPaused(Activity a) { PerfOverlay.detach(a); }
            @Override public void onActivityCreated(Activity a, Bundle b) {}
            @Override public void onActivityStarted(Activity a) {}
            @Override public void onActivityStopped(Activity a) {}
            @Override public void onActivitySaveInstanceState(Activity a, Bundle b) {}
            @Override public void onActivityDestroyed(Activity a) {}
        });
    }
}
