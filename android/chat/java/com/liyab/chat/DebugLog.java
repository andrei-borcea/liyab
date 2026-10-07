package com.liyab.chat;

import android.os.Handler;
import android.os.Looper;

import java.text.SimpleDateFormat;
import java.util.ArrayDeque;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * App-wide debug log shared by every screen: timestamped app events plus the engine's own log lines
 * (pumped from the native buffer twice per second). Bounded to the last MAX_LINES lines.
 */
final class DebugLog {
    interface Listener {
        void onLine(String line);  // main thread
    }

    private static final int MAX_LINES = 2000;
    private static final ArrayDeque<String> lines = new ArrayDeque<>();
    private static final List<Listener> listeners = new CopyOnWriteArrayList<>();
    private static final Handler main = new Handler(Looper.getMainLooper());
    private static final SimpleDateFormat clock = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);
    private static boolean pumping;

    private DebugLog() {}

    /** Appends a line; safe from any thread. */
    static void add(String line) {
        final String stamped;
        synchronized (lines) {
            stamped = clock.format(new Date()) + "  " + line;
            lines.addLast(stamped);
            while (lines.size() > MAX_LINES) lines.removeFirst();
        }
        main.post(() -> {
            for (Listener l : listeners) l.onLine(stamped);
        });
    }

    static String snapshot() {
        synchronized (lines) {
            return String.join("\n", lines);
        }
    }

    static void clear() {
        synchronized (lines) {
            lines.clear();
        }
    }

    static void addListener(Listener l) {
        listeners.add(l);
    }

    static void removeListener(Listener l) {
        listeners.remove(l);
    }

    /** Moves buffered engine log lines into the app log. */
    static void drainNative() {
        String text = LiyabNative.takeLogs();
        if (text == null || text.isEmpty()) return;
        for (String l : text.split("\n")) {
            if (!l.isEmpty()) add("  liyab " + l);
        }
    }

    /** Starts the periodic native-log pump (idempotent). */
    static void startPump() {
        if (pumping) return;
        pumping = true;
        main.post(new Runnable() {
            @Override
            public void run() {
                drainNative();
                main.postDelayed(this, 500);
            }
        });
    }
}
