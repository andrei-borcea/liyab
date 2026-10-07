package com.liyab.chat;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.SharedPreferences;
import android.database.Cursor;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.ParcelFileDescriptor;
import android.provider.OpenableColumns;
import android.text.InputType;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Single-screen chat with a local model.
 *
 * Models: any *.gguf in the app's folder (/sdcard/Android/data/com.liyab.chat/files/, fill it
 * with `adb push`) or any file picked with the system file picker ("Browse…"), which needs no
 * storage permission. Picked files are mapped through /proc/self/fd, so nothing is copied.
 * Loading a model first unloads the previous one. The debug panel shows each load step with
 * timestamps plus Liyab's own log lines.
 *
 * Prompts use the Zephyr/TinyLlama chat template; every turn re-sends the conversation, and
 * Liyab's KV dedup restores the already-computed prefix.
 */
public final class MainActivity extends Activity {
    private static final String SYSTEM_PROMPT =
            "You are Liyab, a helpful assistant running entirely on this phone, without internet. "
                    + "Answer clearly and concisely.";
    private static final int MAX_TOKENS = 384;
    private static final float TEMPERATURE = 0.7f;
    private static final int THREADS = 4;    // fastest setting measured on Snapdragon 8 Elite
    private static final int MAX_TURNS = 6;  // history kept in the prompt (2048-token context)
    private static final int PICK_MODEL = 1;

    private static final int BG = Color.rgb(16, 17, 20);
    private static final int USER_BG = Color.rgb(46, 86, 160);
    private static final int BOT_BG = Color.rgb(36, 38, 44);
    private static final int MUTED = Color.rgb(150, 155, 165);
    private static final int ACCENT = Color.rgb(255, 122, 48);
    private static final int DEBUG_BG = Color.rgb(8, 9, 11);
    private static final int DEBUG_FG = Color.rgb(120, 220, 140);

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final List<String[]> history = new ArrayList<>();  // {user, assistant}
    private final SimpleDateFormat clock = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);

    private volatile long engine;
    private volatile boolean generating;
    private String modelName = "";
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private TextView status;
    private ScrollView debugScroll;
    private TextView debugLog;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LiyabNative.enableLogs(1);  // info and above into the debug panel
        buildUi();
        log("Liyab Chat started; models folder: " + modelsDir());
        restoreLastModelOrAsk();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (engine != 0) LiyabNative.cancel(engine);
        worker.execute(this::unloadOnWorker);
        worker.shutdown();
    }

    // ------------------------------------------------------------------ UI

    private int dp(float v) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, getResources().getDisplayMetrics());
    }

    private void buildUi() {
        getWindow().setStatusBarColor(BG);
        getWindow().setNavigationBarColor(BG);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        // Edge-to-edge (targetSdk 35): pad for system bars and the keyboard.
        root.setOnApplyWindowInsetsListener((v, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.ime());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return WindowInsets.CONSUMED;
        });

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.HORIZONTAL);
        header.setGravity(Gravity.CENTER_VERTICAL);
        header.setPadding(dp(16), dp(10), dp(8), 0);
        TextView title = new TextView(this);
        title.setText("🔥 Liyab Chat");
        title.setTextColor(Color.WHITE);
        title.setTextSize(20);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        header.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        header.addView(smallButton("Model", v -> showModelPicker()));
        header.addView(smallButton("Debug", v -> toggleDebug()));
        header.addView(smallButton("New", v -> newChat()));
        root.addView(header);

        status = new TextView(this);
        status.setTextColor(MUTED);
        status.setTextSize(12);
        status.setPadding(dp(16), dp(2), dp(16), dp(8));
        status.setText("No model loaded");
        root.addView(status);

        scroll = new ScrollView(this);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(dp(12), dp(4), dp(12), dp(12));
        scroll.addView(messages);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        debugScroll = new ScrollView(this);
        debugScroll.setBackgroundColor(DEBUG_BG);
        debugLog = new TextView(this);
        debugLog.setTextColor(DEBUG_FG);
        debugLog.setTextSize(10.5f);
        debugLog.setTypeface(Typeface.MONOSPACE);
        debugLog.setTextIsSelectable(true);
        debugLog.setPadding(dp(10), dp(6), dp(10), dp(6));
        debugScroll.addView(debugLog);
        debugScroll.setVisibility(View.GONE);
        root.addView(debugScroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(220)));

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setPadding(dp(12), dp(8), dp(12), dp(12));
        bar.setGravity(Gravity.CENTER_VERTICAL);

        input = new EditText(this);
        input.setHint("Message");
        input.setHintTextColor(MUTED);
        input.setTextColor(Color.WHITE);
        input.setMaxLines(5);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        input.setBackground(rounded(BOT_BG, 22));
        input.setPadding(dp(14), dp(10), dp(14), dp(10));
        bar.addView(input, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));

        send = new Button(this);
        send.setText("Send");
        send.setTextColor(Color.WHITE);
        send.setAllCaps(false);
        send.setBackground(rounded(ACCENT, 22));
        send.setEnabled(false);
        LinearLayout.LayoutParams sendParams =
                new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(44));
        sendParams.leftMargin = dp(8);
        bar.addView(send, sendParams);
        root.addView(bar);

        send.setOnClickListener(v -> onSendOrStop());
        input.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                onSendOrStop();
                return true;
            }
            return false;
        });
        setContentView(root);
    }

    private Button smallButton(String label, View.OnClickListener listener) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(13);
        b.setTextColor(Color.WHITE);
        b.setBackground(rounded(BOT_BG, 16));
        b.setMinWidth(0);
        b.setMinimumWidth(0);
        b.setPadding(dp(12), 0, dp(12), 0);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(36));
        p.leftMargin = dp(6);
        b.setLayoutParams(p);
        b.setOnClickListener(listener);
        return b;
    }

    private static GradientDrawable rounded(int color, int radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(radiusDp * 2.75f);
        return d;
    }

    private TextView addBubble(String text, boolean user) {
        TextView bubble = new TextView(this);
        bubble.setText(text);
        bubble.setTextColor(Color.WHITE);
        bubble.setTextSize(15);
        bubble.setTextIsSelectable(true);
        bubble.setPadding(dp(14), dp(10), dp(14), dp(10));
        bubble.setBackground(rounded(user ? USER_BG : BOT_BG, 16));
        LinearLayout.LayoutParams params =
                new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        params.gravity = user ? Gravity.END : Gravity.START;
        params.topMargin = dp(8);
        if (user) params.leftMargin = dp(48);
        else params.rightMargin = dp(48);
        messages.addView(bubble, params);
        scrollToBottom();
        return bubble;
    }

    private TextView addNote(String text) {
        TextView note = new TextView(this);
        note.setText(text);
        note.setTextColor(MUTED);
        note.setTextSize(11);
        note.setPadding(dp(4), dp(2), dp(4), 0);
        messages.addView(note);
        scrollToBottom();
        return note;
    }

    private void scrollToBottom() {
        scroll.post(() -> scroll.fullScroll(View.FOCUS_DOWN));
    }

    private void toggleDebug() {
        debugScroll.setVisibility(debugScroll.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
        drainNativeLogs();
    }

    private void newChat() {
        if (generating) return;
        history.clear();
        messages.removeAllViews();
        log("New chat: history cleared");
    }

    // --------------------------------------------------------------- debug

    /** Appends a timestamped line to the debug panel; safe from any thread. */
    private void log(String line) {
        final String stamped = clock.format(new Date()) + "  " + line + "\n";
        ui.post(() -> {
            debugLog.append(stamped);
            debugScroll.post(() -> debugScroll.fullScroll(View.FOCUS_DOWN));
        });
    }

    /** Moves buffered engine log lines into the panel. */
    private void drainNativeLogs() {
        String lines = LiyabNative.takeLogs();
        if (lines == null || lines.isEmpty()) return;
        for (String l : lines.split("\n")) {
            if (!l.isEmpty()) log("  liyab " + l);
        }
    }

    // --------------------------------------------------------------- models

    private File modelsDir() {
        File dir = getExternalFilesDir(null);
        return dir != null ? dir : getFilesDir();
    }

    private File[] localModels() {
        File[] files = modelsDir().listFiles((d, name) -> name.endsWith(".gguf"));
        return files != null ? files : new File[0];
    }

    private void restoreLastModelOrAsk() {
        SharedPreferences prefs = getPreferences(MODE_PRIVATE);
        String last = prefs.getString("model", null);
        if (last != null) {
            if (last.startsWith("content://")) {
                loadModel(null, Uri.parse(last));
                return;
            }
            File f = new File(last);
            if (f.canRead()) {
                loadModel(f, null);
                return;
            }
        }
        File[] local = localModels();
        if (local.length == 1) loadModel(local[0], null);
        else showModelPicker();
    }

    private void showModelPicker() {
        if (generating) return;
        File[] local = localModels();
        List<String> labels = new ArrayList<>();
        for (File f : local) labels.add(String.format(Locale.US, "%s  (%.2f GB)", f.getName(), f.length() / 1e9));
        labels.add("Browse phone storage…");
        if (engine != 0) labels.add("Unload current model");
        new AlertDialog.Builder(this)
                .setTitle("Choose a model")
                .setItems(labels.toArray(new String[0]), (dialog, which) -> {
                    if (which < local.length) {
                        loadModel(local[which], null);
                    } else if (which == local.length) {
                        Intent pick = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                        pick.addCategory(Intent.CATEGORY_OPENABLE);
                        pick.setType("*/*");
                        startActivityForResult(pick, PICK_MODEL);
                    } else {
                        unload();
                    }
                })
                .show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_MODEL || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        try {
            getContentResolver().takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION);
        } catch (SecurityException ignored) {
            // Provider does not offer persistable grants: works for this session only.
        }
        loadModel(null, uri);
    }

    private String displayName(Uri uri) {
        try (Cursor c = getContentResolver().query(uri, new String[] {OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE},
                null, null, null)) {
            if (c != null && c.moveToFirst()) {
                return String.format(Locale.US, "%s (%.2f GB)", c.getString(0), c.getLong(1) / 1e9);
            }
        } catch (RuntimeException ignored) {
        }
        return uri.getLastPathSegment();
    }

    private void unload() {
        send.setEnabled(false);
        worker.execute(() -> {
            unloadOnWorker();
            ui.post(() -> status.setText("No model loaded"));
        });
    }

    /** Destroys the current engine; runs on the worker so it never races a generation. */
    private void unloadOnWorker() {
        long old = engine;
        if (old == 0) return;
        engine = 0;
        long t0 = System.nanoTime();
        LiyabNative.destroy(old);
        log(String.format(Locale.US, "Unloaded %s in %.0f ms (memory released)", modelName,
                (System.nanoTime() - t0) / 1e6));
    }

    /** Loads `file` or the document `uri` (exactly one is non-null), unloading the previous model. */
    private void loadModel(File file, Uri uri) {
        final String label = file != null
                ? String.format(Locale.US, "%s (%.2f GB)", file.getName(), file.length() / 1e9)
                : displayName(uri);
        send.setEnabled(false);
        status.setText("Loading " + label + "…");
        if (debugScroll.getVisibility() != View.VISIBLE) toggleDebug();
        log("Selected " + label + (uri != null ? " via file picker" : " from " + file.getParent()));

        File cache = new File(getCacheDir(), "kv");
        worker.execute(() -> {
            unloadOnWorker();
            deleteRecursively(cache);  // KV snapshots belong to the previous model/session
            cache.mkdirs();
            ParcelFileDescriptor pfd = null;
            String path;
            try {
                if (file != null) {
                    path = file.getAbsolutePath();
                } else {
                    // Map the picked document through its descriptor: no copy, no storage permission.
                    pfd = getContentResolver().openFileDescriptor(uri, "r");
                    if (pfd == null) throw new RuntimeException("the provider returned no file descriptor");
                    path = "/proc/self/fd/" + pfd.getFd();
                    log("Opened document as " + path);
                }
                log("Creating engine: mmap + GGUF parse + device detection + backend selection…");
                long t0 = System.nanoTime();
                long handle = LiyabNative.create(path, cache.getAbsolutePath(), THREADS);
                double seconds = (System.nanoTime() - t0) / 1e9;
                drainNativeLogs();
                String info = LiyabNative.describe(handle);
                for (String line : info.split("\n")) log("  " + line);
                log(String.format(Locale.US, "Ready in %.2f s", seconds));
                engine = handle;
                modelName = label;
                getPreferences(MODE_PRIVATE).edit()
                        .putString("model", file != null ? file.getAbsolutePath() : uri.toString()).apply();
                ui.post(() -> {
                    history.clear();
                    status.setText(String.format(Locale.US, "%s · ready in %.1fs", label, seconds));
                    send.setEnabled(true);
                    addNote("Model loaded: " + label);
                });
            } catch (Exception e) {
                drainNativeLogs();
                log("ERROR: " + e.getMessage());
                ui.post(() -> {
                    status.setText("Failed to load " + label);
                    addBubble("Could not load the model:\n" + e.getMessage()
                            + "\n\nTap \"Model\" to choose another file.", false);
                });
            } finally {
                if (pfd != null) {
                    try {
                        pfd.close();  // the engine keeps its own mapping of the file
                    } catch (java.io.IOException ignored) {
                    }
                }
            }
        });
    }

    private static void deleteRecursively(File f) {
        File[] children = f.listFiles();
        if (children != null) for (File c : children) deleteRecursively(c);
        f.delete();
    }

    // ---------------------------------------------------------------- chat

    private String buildPrompt(String userMessage) {
        StringBuilder p = new StringBuilder("<|system|>\n").append(SYSTEM_PROMPT).append("</s>\n");
        for (String[] turn : history) {
            p.append("<|user|>\n").append(turn[0]).append("</s>\n<|assistant|>\n").append(turn[1]).append("</s>\n");
        }
        return p.append("<|user|>\n").append(userMessage).append("</s>\n<|assistant|>\n").toString();
    }

    private void onSendOrStop() {
        if (generating) {
            LiyabNative.cancel(engine);
            return;
        }
        String message = input.getText().toString().trim();
        if (message.isEmpty() || engine == 0) return;
        input.setText("");
        addBubble(message, true);
        TextView reply = addBubble("…", false);
        TextView note = addNote("");
        generating = true;
        send.setText("Stop");

        final String prompt = buildPrompt(message);
        final StringBuilder text = new StringBuilder();
        worker.execute(() -> {
            String error = null;
            double[] stats = null;
            try {
                stats = LiyabNative.generate(engine, prompt, MAX_TOKENS, TEMPERATURE, bytes -> {
                    String piece = new String(bytes, StandardCharsets.UTF_8);
                    synchronized (text) {
                        text.append(piece);
                    }
                    ui.post(() -> {
                        synchronized (text) {
                            reply.setText(text.toString().replaceFirst("^\\s+", ""));
                        }
                        scrollToBottom();
                    });
                    return true;
                });
            } catch (RuntimeException e) {
                error = e.getMessage();
            }
            drainNativeLogs();
            final double[] s = stats;
            final String err = error;
            if (s != null) {
                log(String.format(Locale.US, "Generated %d tok at %.1f tok/s, TTFT %.0f ms, prompt %d tok (%d reused)",
                        (int) s[1], s[2], s[3], (int) s[0], (int) s[4]));
            } else {
                log("ERROR: " + err);
            }
            ui.post(() -> {
                generating = false;
                send.setText("Send");
                String answer;
                synchronized (text) {
                    answer = text.toString().trim();
                }
                if (err != null) {
                    reply.setText("Error: " + err);
                    return;
                }
                if (answer.isEmpty()) reply.setText("(no answer)");
                history.add(new String[] {message, answer});
                while (history.size() > MAX_TURNS) history.remove(0);
                if (s != null) {
                    note.setText(String.format(Locale.US,
                            "%d tokens · %.1f tok/s · first token %.2fs · prompt %d tok (%d reused)%s%s",
                            (int) s[1], s[2], s[3] / 1000.0, (int) s[0], (int) s[4],
                            s[6] > 0 ? " · thermal throttle" : "", s[7] > 0 ? " · stopped" : ""));
                }
                scrollToBottom();
            });
        });
    }
}
