package com.liyab.chat;

import android.content.Context;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;

import java.io.File;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * The (single) running model download, shared by all screens, so it keeps going while the user moves
 * between Home, Chat and Models. A partial wake lock keeps the CPU awake with the screen off.
 */
final class Downloads {
    interface Listener {
        void onDownloadChanged();  // main thread; progress is polled from `state`
    }

    static final ExecutorService net = Executors.newSingleThreadExecutor();

    static volatile HuggingFace.DownloadState state;  // non-null while running
    static volatile HuggingFace.GgufFile file;
    static volatile long startNanos;
    static volatile String lastMessage = "";
    static volatile boolean discardRequested;

    private static final List<Listener> listeners = new CopyOnWriteArrayList<>();
    private static final Handler main = new Handler(Looper.getMainLooper());

    private Downloads() {}

    static void addListener(Listener l) {
        listeners.add(l);
    }

    static void removeListener(Listener l) {
        listeners.remove(l);
    }

    private static void changed() {
        main.post(() -> {
            for (Listener l : listeners) l.onDownloadChanged();
        });
    }

    static boolean running() {
        return state != null;
    }

    /** Starts (or resumes) `f`; loads it when finished. No-op if another download runs. */
    static void start(Context context, HuggingFace.GgufFile f) {
        if (state != null) return;
        final HuggingFace.DownloadState s = new HuggingFace.DownloadState();
        state = s;
        file = f;
        discardRequested = false;
        startNanos = System.nanoTime();
        lastMessage = "";
        PowerManager pm = (PowerManager) context.getApplicationContext().getSystemService(Context.POWER_SERVICE);
        PowerManager.WakeLock wake = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "liyab:download");
        wake.acquire(6 * 60 * 60 * 1000L);  // safety timeout: 6 h
        DebugLog.add("Download started: " + f.repo + "/" + f.path + " (" + Ui.gb(f.size) + ")");
        changed();
        net.execute(() -> {
            File result = null;
            String error = null;
            try {
                result = HuggingFace.downloadModel(f, EngineHolder.modelsDir(), s);
            } catch (Exception e) {
                error = e.getMessage();
            }
            double seconds = (System.nanoTime() - startNanos) / 1e9;
            if (wake.isHeld()) wake.release();
            if (result != null) {
                lastMessage = "Downloaded " + result.getName();
                DebugLog.add(String.format(Locale.US, "Downloaded %s in %s (%.1f MB/s avg, %d retries)%s",
                        result.getName(), Ui.duration(seconds), f.size / 1e6 / Math.max(seconds, 0.001),
                        s.retries.get(), f.sha256 != null ? ", SHA-256 verified" : ""));
                EngineHolder.load(result, null);
            } else if (discardRequested) {
                for (HuggingFace.GgufFile part : f.parts.isEmpty() ? java.util.Collections.singletonList(f) : f.parts) {
                    HuggingFace.discard(EngineHolder.modelsDir(), part);
                }
                lastMessage = "Download cancelled";
                DebugLog.add("Download cancelled, partial file deleted: " + f.fileName());
            } else {
                lastMessage = String.format(Locale.US, "Paused at %.0f%%%s", 100.0 * s.done.get() / Math.max(1, s.total),
                        error != null && !error.contains("paused") ? " — " + error : "");
                DebugLog.add("Download stopped: " + error);
            }
            state = null;
            changed();
        });
    }

    static void pause() {
        HuggingFace.DownloadState s = state;
        if (s != null) s.cancelled = true;  // the partial file is kept for resuming
    }

    static void cancel() {
        HuggingFace.DownloadState s = state;
        if (s != null) {
            discardRequested = true;
            s.cancelled = true;
        }
    }

    /** Looks the file up in the repository listing (for its size and hash) and downloads it. */
    static void startByName(Context context, String repo, String path) {
        DebugLog.add("Requested download " + repo + "/" + path);
        net.execute(() -> {
            try {
                for (HuggingFace.GgufFile f : HuggingFace.files(repo)) {
                    if (f.path.equals(path)) {
                        main.post(() -> start(context, f));
                        return;
                    }
                }
                DebugLog.add("ERROR: " + path + " not found in " + repo);
            } catch (Exception e) {
                DebugLog.add("ERROR: " + e.getMessage());
            }
        });
    }
}
