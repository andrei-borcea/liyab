package com.liyab.chat;

import android.content.Context;
import android.content.SharedPreferences;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * The loaded model, shared by all screens: load / unload / backend switch, and the chat history so a
 * conversation survives leaving the chat screen. Every engine call runs on the single `worker`
 * thread, so loads, unloads and generations never overlap.
 */
final class EngineHolder {
    interface Listener {
        void onEngineChanged();  // main thread
    }

    static final int THREADS = 4;  // fastest setting measured on Snapdragon 8 Elite
    static final ExecutorService worker = Executors.newSingleThreadExecutor();

    static volatile long handle;
    static volatile boolean busy;        // loading or unloading
    static volatile boolean generating;
    static volatile String status = "No model loaded";
    static String modelLabel = "";
    static File modelFile;
    static Uri modelUri;
    static String description = "";
    static volatile ChatTemplate template = ChatTemplate.ZEPHYR;
    static volatile boolean thinkingSupported;  // the vocabulary has <think> / </think> tokens
    static volatile ModelSettings settings = ModelSettings.defaults(0);
    static String modelKey = "";  // settings key: the model's file name
    static final List<String[]> history = new ArrayList<>();  // chat turns {user, assistant}

    private static final List<Listener> listeners = new CopyOnWriteArrayList<>();
    private static final Handler main = new Handler(Looper.getMainLooper());
    private static Context app;

    private EngineHolder() {}

    static void init(Context context) {
        app = context.getApplicationContext();
    }

    static SharedPreferences prefs() {
        return app.getSharedPreferences("liyab", Context.MODE_PRIVATE);
    }

    static boolean useGpu() {
        return prefs().getBoolean("gpu", false);
    }

    static void addListener(Listener l) {
        listeners.add(l);
    }

    static void removeListener(Listener l) {
        listeners.remove(l);
    }

    private static void changed() {
        main.post(() -> {
            for (Listener l : listeners) l.onEngineChanged();
        });
    }

    static File modelsDir() {
        File dir = app.getExternalFilesDir(null);
        return dir != null ? dir : app.getFilesDir();
    }

    static boolean isLoaded(File f) {
        return handle != 0 && modelFile != null && modelFile.getAbsolutePath().equals(f.getAbsolutePath());
    }

    /** Reloads the last used model, if it still exists. */
    static void restoreLast() {
        String last = prefs().getString("model", null);
        if (last == null || handle != 0 || busy) return;
        if (last.startsWith("content://")) {
            load(null, Uri.parse(last));
        } else if (new File(last).canRead()) {
            load(new File(last), null);
        }
    }

    static void setGpu(boolean gpu) {
        prefs().edit().putBoolean("gpu", gpu).apply();
        DebugLog.add("Backend switched to " + (gpu ? "GPU (Vulkan)" : "CPU (NEON)"));
        if (modelFile != null || modelUri != null) load(modelFile, modelUri);
        else changed();
    }

    static void unload() {
        busy = true;
        changed();
        worker.execute(() -> {
            unloadOnWorker();
            status = "No model loaded";
            busy = false;
            changed();
        });
    }

    private static void unloadOnWorker() {
        long old = handle;
        if (old == 0) return;
        handle = 0;
        long t0 = System.nanoTime();
        LiyabNative.destroy(old);
        DebugLog.add(String.format(Locale.US, "Unloaded %s in %.0f ms", modelLabel, (System.nanoTime() - t0) / 1e6));
    }

    static String labelOf(File f) {
        return String.format(Locale.US, "%s (%.2f GB)", f.getName(), sizeOf(f) / 1e9);
    }

    /** Bytes of the model whose first file is `f` (all parts of a split model). */
    static long sizeOf(File f) {
        long total = 0;
        for (File part : HuggingFace.localParts(f)) total += part.length();
        return total;
    }

    /** Loads `file` or the document `uri` (exactly one non-null), unloading the current model first. */
    static void load(File file, Uri uri) {
        final boolean gpu = useGpu();
        final String label = file != null ? labelOf(file) : String.valueOf(uri.getLastPathSegment());
        final String key = file != null ? file.getName() : String.valueOf(uri.getLastPathSegment());
        final int context = ModelSettings.contextFor(key);
        busy = true;
        status = "Loading " + label + "…";
        changed();
        DebugLog.add("Selected " + label + (uri != null ? " via file picker" : " from " + file.getParent()));
        File cache = new File(app.getCacheDir(), "kv");
        worker.execute(() -> {
            unloadOnWorker();
            deleteRecursively(cache);  // KV snapshots belong to the previous model
            cache.mkdirs();
            ParcelFileDescriptor pfd = null;
            try {
                String path;
                if (file != null) {
                    path = file.getAbsolutePath();
                } else {
                    // Map the picked document through its descriptor: no copy, no storage permission.
                    pfd = app.getContentResolver().openFileDescriptor(uri, "r");
                    if (pfd == null) throw new RuntimeException("the provider returned no file descriptor");
                    path = "/proc/self/fd/" + pfd.getFd();
                }
                DebugLog.add("Creating engine on " + (gpu ? "GPU (Vulkan)" : "CPU") + "…");
                long t0 = System.nanoTime();
                long h = LiyabNative.create(path, cache.getAbsolutePath(), THREADS,
                        gpu ? LiyabNative.BACKEND_VULKAN : LiyabNative.BACKEND_CPU, context);
                if (gpu) {
                    DebugLog.add("Uploading weights to GPU memory (one-time warm-up)…");
                    LiyabNative.generate(h, "Hi", 1, 0f, 1f, 0, bytes -> true);
                }
                double seconds = (System.nanoTime() - t0) / 1e9;
                DebugLog.drainNative();
                description = LiyabNative.describe(h);
                for (String line : description.split("\n")) DebugLog.add("  " + line);
                DebugLog.add(String.format(Locale.US, "Ready in %.2f s", seconds));
                template = ChatTemplate.detect(h);
                thinkingSupported = template == ChatTemplate.CHATML && LiyabNative.countTokens(h, "<think>") == 1
                        && LiyabNative.countTokens(h, "</think>") == 1;
                settings = ModelSettings.load(key, h);
                modelKey = key;
                DebugLog.add("Chat template: " + template.label + (thinkingSupported ? " (with thinking mode)" : ""));
                DebugLog.add("Settings: " + settings.summary(thinkingSupported) + ", context " + context);
                handle = h;
                modelLabel = label;
                modelFile = file;
                modelUri = uri;
                synchronized (history) {
                    history.clear();
                }
                prefs().edit().putString("model", file != null ? file.getAbsolutePath() : uri.toString()).apply();
                status = String.format(Locale.US, "%s · %s · ready in %.1fs", label, gpu ? "GPU" : "CPU", seconds);
            } catch (Exception e) {
                DebugLog.drainNative();
                DebugLog.add("ERROR: " + e.getMessage());
                status = "Failed to load " + label + ": " + e.getMessage();
            } finally {
                if (pfd != null) {
                    try {
                        pfd.close();  // the engine keeps its own mapping
                    } catch (java.io.IOException ignored) {
                    }
                }
                busy = false;
                changed();
            }
        });
    }

    static void deleteRecursively(File f) {
        File[] children = f.listFiles();
        if (children != null) for (File c : children) deleteRecursively(c);
        f.delete();
    }
}
