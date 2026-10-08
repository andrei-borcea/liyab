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

    static final int THREADS = 0;  // engine default: performance cores, one left for I/O and the UI (7 on an 8 Elite)
    static final ExecutorService worker = Executors.newSingleThreadExecutor();

    static volatile long handle;
    /** Held while an engine is created or destroyed; monitors take it to read a live engine safely. */
    static final Object LIFECYCLE = new Object();
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

    /**
     * Skin/board temperature (°C) above which the engine halves its CPU threads. Device-wide setting.
     * 50 by default: on some phones the only "skin" sensor is a board thermistor that reads 45+ while
     * charging, which would halve speed at a comfortable temperature. The OS thermal status (severe,
     * critical) throttles regardless of this value.
     */
    static float thermalLimitC() {
        return prefs().getFloat("thermal_limit", 50f);
    }

    /**
     * Memory (MiB) the engine may keep resident: weights, expert cache, streaming slots. Device-wide.
     * 5500 by default because HyperOS / MIUI stop any app above 6 GiB of PSS, whatever RAM is free;
     * raise it on phones without such a cap.
     */
    static long memoryBudgetMb() {
        return prefs().getLong("memory_budget_mb", 5500);
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

    static void changed() {
        main.post(() -> {
            for (Listener l : listeners) l.onEngineChanged();
        });
    }

    /**
     * Where downloads go: app-private internal storage. It is plain f2fs, so the engine's direct
     * (O_DIRECT) reads work and streaming runs at full speed.
     */
    static File modelsDir() {
        File dir = new File(app.getFilesDir(), "models");
        if (!dir.isDirectory()) dir.mkdirs();
        return dir;
    }

    /**
     * The shared app folder (/sdcard/Android/data/<package>/files), where adb can push models. Android
     * serves it through FUSE: direct reads are unavailable there, so streaming models larger than RAM
     * is several times slower. Null when external storage is unavailable.
     */
    static File sharedDir() {
        return app.getExternalFilesDir(null);
    }

    /** True for files in the shared (FUSE) folder. */
    static boolean inSharedDir(File f) {
        File shared = sharedDir();
        return shared != null && f.getAbsolutePath().startsWith(shared.getAbsolutePath() + "/");
    }

    static volatile String moving;  // name of the model being moved, or null

    /** Moves a model (all parts) from the shared folder into modelsDir() in the background. */
    static void moveToAppStorage(File first) {
        if (moving != null) return;
        moving = first.getName();
        changed();
        new Thread(() -> {
            long t0 = System.nanoTime();
            long bytes = 0;
            String error = null;
            try {
                for (File part : HuggingFace.localParts(first)) {
                    File dst = new File(modelsDir(), part.getName());
                    File tmp = new File(modelsDir(), part.getName() + ".moving");
                    if (modelsDir().getUsableSpace() < part.length()) throw new java.io.IOException("not enough free space");
                    try (java.nio.channels.FileChannel in = new java.io.FileInputStream(part).getChannel();
                         java.nio.channels.FileChannel out = new java.io.FileOutputStream(tmp).getChannel()) {
                        long done = 0, size = in.size(), next = 1L << 30;
                        while (done < size) {
                            done += in.transferTo(done, size - done, out);
                            if (done >= next) {
                                DebugLog.add(String.format(Locale.US, "Moving %s: %.1f / %.1f GB", part.getName(),
                                        done / 1e9, size / 1e9));
                                next += 1L << 30;
                            }
                        }
                        bytes += size;
                    }
                    if (!tmp.renameTo(dst)) throw new java.io.IOException("cannot rename " + tmp);
                    if (!part.delete()) DebugLog.add("Could not delete the shared copy of " + part.getName());
                }
                String last = prefs().getString("model", "");
                if (last.equals(first.getAbsolutePath())) {
                    prefs().edit().putString("model", new File(modelsDir(), first.getName()).getAbsolutePath()).apply();
                }
            } catch (java.io.IOException e) {
                error = e.getMessage();
            }
            double s = (System.nanoTime() - t0) / 1e9;
            DebugLog.add(error == null
                    ? String.format(Locale.US, "Moved %s to app storage: %.1f GB in %.0f s", first.getName(), bytes / 1e9, s)
                    : "Move failed: " + error);
            moving = null;
            changed();
        }, "liyab-move").start();
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
        long t0 = System.nanoTime();
        synchronized (LIFECYCLE) {
            handle = 0;
            LiyabNative.destroy(old);
        }
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
                        gpu ? LiyabNative.BACKEND_VULKAN : LiyabNative.BACKEND_CPU, context, thermalLimitC(),
                        memoryBudgetMb());
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
                prepareSystemPrompt();  // queued after this load finishes; also the GPU warm-up
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

    /**
     * Processes the chat's system block in the background, so the first message only costs its own
     * tokens: the engine keeps that context and the chat prompt starts with exactly this text.
     */
    static void prepareSystemPrompt() {
        worker.execute(EngineHolder::prepareSystemPromptOnWorker);  // no-op if nothing is loaded by then
    }

    private static void prepareSystemPromptOnWorker() {
        long h = handle;
        if (h == 0) return;
        long t0 = System.nanoTime();
        try {
            LiyabNative.prefill(h, template.system(settings.systemPrompt));
            DebugLog.add(String.format(Locale.US, "System prompt prepared in %.1f s",
                    (System.nanoTime() - t0) / 1e9));
        } catch (RuntimeException e) {
            DebugLog.add("System prompt not prepared: " + e.getMessage());
        }
    }

    static void deleteRecursively(File f) {
        File[] children = f.listFiles();
        if (children != null) for (File c : children) deleteRecursively(c);
        f.delete();
    }
}
