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
import android.widget.ArrayAdapter;
import android.widget.LinearLayout;
import android.widget.ListView;
import android.widget.ProgressBar;
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
    private static final long FRAME_MS = 33;  // streaming redraw interval (~30 fps)

    private static final int BG = Color.rgb(16, 17, 20);
    private static final int USER_BG = Color.rgb(46, 86, 160);
    private static final int BOT_BG = Color.rgb(36, 38, 44);
    private static final int MUTED = Color.rgb(150, 155, 165);
    private static final int ACCENT = Color.rgb(255, 122, 48);
    private static final int DEBUG_BG = Color.rgb(8, 9, 11);
    private static final int DEBUG_FG = Color.rgb(120, 220, 140);

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final ExecutorService net = Executors.newSingleThreadExecutor();  // Hugging Face API + downloads
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final List<String[]> history = new ArrayList<>();  // {user, assistant}
    private final SimpleDateFormat clock = new SimpleDateFormat("HH:mm:ss.SSS", Locale.US);

    private volatile long engine;
    private boolean useGpu;        // Vulkan backend instead of CPU; persisted
    private File loadedFile;       // current model source, to reload on a backend switch
    private Uri loadedUri;
    private Button modeButton;
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
        useGpu = getPreferences(MODE_PRIVATE).getBoolean("gpu", false);
        buildUi();
        modeButton.setText(useGpu ? "GPU" : "CPU");
        log("Liyab Chat started; models folder: " + modelsDir());
        // Scriptable download (testing / automation):
        //   adb shell am start -n com.liyab.chat/.MainActivity --es download "owner/repo|file.gguf"
        String request = getIntent().getStringExtra("download");
        if (request != null && request.contains("|")) {
            String[] parts = request.split("\\|", 2);
            downloadByName(parts[0], parts[1]);
        } else {
            restoreLastModelOrAsk();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (engine != 0) LiyabNative.cancel(engine);
        worker.execute(this::unloadOnWorker);
        worker.shutdown();
        if (download != null) download.cancelled = true;
        net.shutdownNow();
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
        title.setText("🔥 Liyab");
        title.setTextColor(Color.WHITE);
        title.setTextSize(20);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        header.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        modeButton = smallButton("CPU", v -> toggleBackend());
        header.addView(modeButton);
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
        // Model replies grow while streaming: give them a fixed (full) width so
        // only the height changes; user messages are static and wrap their text.
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                user ? ViewGroup.LayoutParams.WRAP_CONTENT : ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        params.gravity = user ? Gravity.END : Gravity.START;
        params.topMargin = dp(8);
        if (user) params.leftMargin = dp(48);
        else params.rightMargin = dp(24);
        if (!user) bubble.setMinHeight(dp(40));
        messages.addView(bubble, params);
        scrollToBottom(true);
        return bubble;
    }

    private TextView addNote(String text) {
        TextView note = new TextView(this);
        note.setText(text);
        note.setTextColor(MUTED);
        note.setTextSize(11);
        note.setPadding(dp(4), dp(2), dp(4), 0);
        messages.addView(note);
        scrollToBottom(true);
        return note;
    }

    /** True when the conversation is scrolled (almost) to the end. */
    private boolean atBottom() {
        View content = scroll.getChildAt(0);
        return content == null || content.getBottom() - (scroll.getScrollY() + scroll.getHeight()) < dp(48);
    }

    /**
     * Scrolls to the end without moving focus (fullScroll(FOCUS_DOWN) steals focus from the input and makes the
     * view jump). While streaming, only follow the text if the user has not scrolled up to read.
     */
    private void scrollToBottom(boolean force) {
        if (!force && !atBottom()) return;
        scroll.post(() -> scroll.scrollTo(0, Math.max(0, messages.getBottom() - scroll.getHeight())));
    }

    private void toggleDebug() {
        debugScroll.setVisibility(debugScroll.getVisibility() == View.VISIBLE ? View.GONE : View.VISIBLE);
        drainNativeLogs();
    }

    /** Switches between the CPU and the Vulkan GPU backend and reloads the current model. */
    private void toggleBackend() {
        if (generating) return;
        useGpu = !useGpu;
        modeButton.setText(useGpu ? "GPU" : "CPU");
        getPreferences(MODE_PRIVATE).edit().putBoolean("gpu", useGpu).apply();
        log("Backend switched to " + (useGpu ? "GPU (Vulkan)" : "CPU (NEON)"));
        if (loadedFile != null || loadedUri != null) loadModel(loadedFile, loadedUri);
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
            debugScroll.post(() -> debugScroll.scrollTo(0, debugLog.getBottom()));
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
        List<HuggingFace.Partial> partials = HuggingFace.partials(modelsDir());
        List<String> labels = new ArrayList<>();
        List<Runnable> actions = new ArrayList<>();
        for (File f : local) {
            labels.add(String.format(Locale.US, "%s  (%.2f GB)", f.getName(), f.length() / 1e9));
            actions.add(() -> loadModel(f, null));
        }
        for (HuggingFace.Partial p : partials) {
            labels.add(String.format(Locale.US, "⏸ %s — %.0f%% downloaded, tap to resume", p.file.fileName(),
                    100.0 * p.done / Math.max(1, p.file.size)));
            actions.add(() -> startDownload(p.file));
        }
        labels.add("⬇ Download from Hugging Face…");
        actions.add(this::showHuggingFaceSearch);
        if (local.length > 0 || !partials.isEmpty()) {
            labels.add("🗑 Manage downloaded models…");
            actions.add(this::showManageModels);
        }
        labels.add("Browse phone storage…");
        actions.add(() -> {
            Intent pick = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            pick.addCategory(Intent.CATEGORY_OPENABLE);
            pick.setType("*/*");
            startActivityForResult(pick, PICK_MODEL);
        });
        if (engine != 0) {
            labels.add("Unload current model");
            actions.add(this::unload);
        }
        new AlertDialog.Builder(this)
                .setTitle("Choose a model")
                .setItems(labels.toArray(new String[0]), (dialog, which) -> actions.get(which).run())
                .show();
    }

    /** True when `f` is the model currently loaded (it is memory-mapped and cannot be deleted). */
    private boolean inUse(File f) {
        return engine != 0 && loadedFile != null && loadedFile.getAbsolutePath().equals(f.getAbsolutePath());
    }

    /** Lists downloaded and partial models with their sizes; tap one to delete it. */
    private void showManageModels() {
        File[] local = localModels();
        List<HuggingFace.Partial> partials = HuggingFace.partials(modelsDir());
        List<String> labels = new ArrayList<>();
        List<Runnable> actions = new ArrayList<>();
        long total = 0;
        for (File f : local) {
            total += f.length();
            boolean used = inUse(f);
            labels.add(String.format(Locale.US, "%s%s\n%.2f GB%s", used ? "● " : "", f.getName(), f.length() / 1e9,
                    used ? " · in use (unload or load another model to delete)" : " · tap to delete"));
            actions.add(() -> {
                if (inUse(f)) {
                    addNote(f.getName() + " is in use: unload it or load another model first.");
                    return;
                }
                confirmDelete(f.getName(), f.length(), () -> {
                    boolean ok = f.delete();
                    if (ok && f.getAbsolutePath().equals(getPreferences(MODE_PRIVATE).getString("model", ""))) {
                        getPreferences(MODE_PRIVATE).edit().remove("model").apply();
                    }
                    log((ok ? "Deleted " : "Could not delete ") + f.getName());
                });
            });
        }
        for (HuggingFace.Partial p : partials) {
            File part = new File(modelsDir(), p.file.fileName() + ".part");
            total += part.length();
            labels.add(String.format(Locale.US, "⏸ %s (partial, %.0f%%)\n%.2f GB on disk · tap to delete",
                    p.file.fileName(), 100.0 * p.done / Math.max(1, p.file.size), part.length() / 1e9));
            actions.add(() -> {
                if (download != null) {
                    addNote("Stop the running download before deleting partial files.");
                    return;
                }
                confirmDelete(p.file.fileName() + " (partial)", part.length(), () -> {
                    HuggingFace.discard(modelsDir(), p.file);
                    log("Deleted partial download " + p.file.fileName());
                });
            });
        }
        new AlertDialog.Builder(this)
                .setTitle(String.format(Locale.US, "Models · %.2f GB used · %.1f GB free", total / 1e9,
                        modelsDir().getUsableSpace() / 1e9))
                .setItems(labels.toArray(new String[0]), (d, which) -> actions.get(which).run())
                .setNegativeButton("Close", null)
                .show();
    }

    private void confirmDelete(String name, long bytes, Runnable delete) {
        new AlertDialog.Builder(this)
                .setTitle("Delete model?")
                .setMessage(String.format(Locale.US, "%s\n\nThis frees %.2f GB. You can download it again later.", name,
                        bytes / 1e9))
                .setPositiveButton("Delete", (d, w) -> {
                    delete.run();
                    showManageModels();  // refreshed list
                })
                .setNegativeButton("Cancel", null)
                .show();
    }

    // ---------------------------------------------------------- Hugging Face

    private void showHuggingFaceSearch() {
        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(16), dp(8), dp(16), 0);
        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        EditText query = new EditText(this);
        query.setHint("Search GGUF models (e.g. tinyllama, mistral 7b)");
        query.setSingleLine(true);
        query.setImeOptions(EditorInfo.IME_ACTION_SEARCH);
        row.addView(query, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Button go = new Button(this);
        go.setText("Search");
        go.setAllCaps(false);
        row.addView(go);
        box.addView(row);
        TextView hint = new TextView(this);
        hint.setTextColor(MUTED);
        hint.setTextSize(12);
        hint.setText("Most downloaded GGUF repositories");
        box.addView(hint);
        ListView list = new ListView(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<>(this, android.R.layout.simple_list_item_1);
        list.setAdapter(adapter);
        box.addView(list, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(380)));

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("Hugging Face")
                .setView(box)
                .setNegativeButton("Close", null)
                .show();
        final List<HuggingFace.Repo> results = new ArrayList<>();
        Runnable search = () -> {
            String q = query.getText().toString().trim();
            hint.setText("Searching…");
            net.execute(() -> {
                try {
                    List<HuggingFace.Repo> repos = HuggingFace.search(q);
                    ui.post(() -> {
                        results.clear();
                        results.addAll(repos);
                        adapter.clear();
                        for (HuggingFace.Repo r : repos) {
                            adapter.add(String.format(Locale.US, "%s\n⬇ %s   ♥ %d", r.id, compact(r.downloads), r.likes));
                        }
                        hint.setText(repos.isEmpty() ? "No GGUF repositories found"
                                : repos.size() + " repositories · tap one to see its files");
                    });
                } catch (Exception e) {
                    ui.post(() -> hint.setText("Search failed: " + e.getMessage()));
                }
            });
        };
        go.setOnClickListener(v -> search.run());
        query.setOnEditorActionListener((v, actionId, event) -> {
            search.run();
            return true;
        });
        list.setOnItemClickListener((parent, view, position, id) -> {
            dialog.dismiss();
            showHuggingFaceFiles(results.get(position).id);
        });
        search.run();
    }

    private void showHuggingFaceFiles(String repo) {
        AlertDialog loading = new AlertDialog.Builder(this).setTitle(repo).setMessage("Listing files…").show();
        net.execute(() -> {
            try {
                List<HuggingFace.GgufFile> files = HuggingFace.files(repo);
                ui.post(() -> {
                    loading.dismiss();
                    if (files.isEmpty()) {
                        new AlertDialog.Builder(this).setTitle(repo).setMessage("No .gguf files in this repository.")
                                .setPositiveButton("OK", null).show();
                        return;
                    }
                    String[] labels = new String[files.size()];
                    for (int i = 0; i < files.size(); i++) {
                        HuggingFace.GgufFile f = files.get(i);
                        labels[i] = String.format(Locale.US, "%s\n%.2f GB · %s · %s", f.fileName(), f.size / 1e9,
                                f.quant, f.compat.label);
                    }
                    new AlertDialog.Builder(this)
                            .setTitle(repo)
                            .setItems(labels, (d, which) -> confirmDownload(files.get(which)))
                            .setNegativeButton("Back", (d, w) -> showHuggingFaceSearch())
                            .show();
                });
            } catch (Exception e) {
                ui.post(() -> {
                    loading.dismiss();
                    addNote("Could not list " + repo + ": " + e.getMessage());
                });
            }
        });
    }

    private void confirmDownload(HuggingFace.GgufFile f) {
        String message = String.format(Locale.US, "%s\n%.2f GB, free space %.1f GB\n\n%s", f.fileName(),
                f.size / 1e9, modelsDir().getUsableSpace() / 1e9,
                f.compat == HuggingFace.Compat.OK ? "Liyab can run this format."
                        : f.compat == HuggingFace.Compat.MAYBE
                        ? "Q4_0/Q4_1 files usually keep the output head in Q6_K, which Liyab does not run yet. "
                        + "Q8_0 is the safe choice today."
                        : "Liyab cannot run " + f.quant + " yet (K-quants / IQ formats). Pick a Q8_0 or F16 file.");
        new AlertDialog.Builder(this)
                .setTitle("Download model?")
                .setMessage(message)
                .setPositiveButton("Download", (d, w) -> startDownload(f))
                .setNegativeButton("Cancel", null)
                .show();
    }

    /** Looks the file up in the repository listing (for its size) and downloads it. */
    private void downloadByName(String repo, String path) {
        log("Requested download " + repo + "/" + path);
        net.execute(() -> {
            try {
                for (HuggingFace.GgufFile f : HuggingFace.files(repo)) {
                    if (f.path.equals(path)) {
                        ui.post(() -> startDownload(f));
                        return;
                    }
                }
                log("ERROR: " + path + " not found in " + repo);
            } catch (Exception e) {
                log("ERROR: " + e.getMessage());
            }
        });
    }

    private static String compact(long n) {
        if (n >= 1_000_000) return String.format(Locale.US, "%.1fM", n / 1e6);
        if (n >= 1_000) return String.format(Locale.US, "%.1fk", n / 1e3);
        return Long.toString(n);
    }

    private static String duration(double seconds) {
        if (Double.isNaN(seconds) || Double.isInfinite(seconds) || seconds < 0) return "--:--";
        long s = Math.round(seconds);
        return s >= 3600 ? String.format(Locale.US, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
                : String.format(Locale.US, "%d:%02d", s / 60, s % 60);
    }

    private volatile HuggingFace.DownloadState download;

    /** Runs (or resumes) a download with a live progress dialog, then loads the model. */
    private void startDownload(HuggingFace.GgufFile f) {
        if (download != null) {
            addNote("A download is already running.");
            return;
        }
        final HuggingFace.DownloadState state = new HuggingFace.DownloadState();
        download = state;

        LinearLayout box = new LinearLayout(this);
        box.setOrientation(LinearLayout.VERTICAL);
        box.setPadding(dp(20), dp(12), dp(20), dp(4));
        TextView name = new TextView(this);
        name.setText(f.repo + "\n" + f.fileName());
        name.setTextSize(13);
        box.addView(name);
        ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(1000);
        LinearLayout.LayoutParams barParams =
                new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(18));
        barParams.topMargin = dp(12);
        box.addView(bar, barParams);
        TextView percent = new TextView(this);
        percent.setTextSize(22);
        percent.setTypeface(Typeface.DEFAULT_BOLD);
        box.addView(percent);
        TextView details = new TextView(this);
        details.setTypeface(Typeface.MONOSPACE);
        details.setTextSize(12);
        box.addView(details);

        AlertDialog dialog = new AlertDialog.Builder(this)
                .setTitle("Downloading model")
                .setView(box)
                .setCancelable(false)
                .setPositiveButton("Pause", null)
                .setNegativeButton("Cancel", null)
                .show();
        android.os.PowerManager pm = (android.os.PowerManager) getSystemService(POWER_SERVICE);
        android.os.PowerManager.WakeLock wake = pm.newWakeLock(android.os.PowerManager.PARTIAL_WAKE_LOCK, "liyab:download");
        wake.acquire(6 * 60 * 60 * 1000L);  // safety timeout: 6 h
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        final boolean[] discard = {false};
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener(v -> {
            state.cancelled = true;  // keeps the partial file; resume from the model picker
            ((Button) v).setEnabled(false);
            details.append("\nPausing…");
        });
        dialog.getButton(AlertDialog.BUTTON_NEGATIVE).setOnClickListener(v -> {
            discard[0] = true;
            state.cancelled = true;
        });

        // Live stats: progress, smoothed speed, elapsed time and ETA, refreshed 4x per second.
        final long startNanos = System.nanoTime();
        final long[] last = {-1, startNanos};
        final double[] speed = {0};
        final Runnable[] ticker = new Runnable[1];
        ticker[0] = () -> {
            long done = state.done.get();
            long total = Math.max(1, state.total);
            long now = System.nanoTime();
            if (last[0] >= 0) {
                double dt = (now - last[1]) / 1e9;
                if (dt > 0) {
                    double instant = (done - last[0]) / dt;
                    speed[0] = speed[0] == 0 ? instant : 0.8 * speed[0] + 0.2 * instant;  // moving average
                }
            }
            last[0] = done;
            last[1] = now;
            if (state.verifying) {
                long checked = state.verified.get();
                bar.setProgress((int) (1000 * checked / total));
                percent.setText(String.format(Locale.US, "Verifying %.0f%%", 100.0 * checked / total));
                details.setText("Checking SHA-256 against Hugging Face…");
                if (download == state) ui.postDelayed(ticker[0], 250);
                return;
            }
            bar.setProgress((int) (1000 * done / total));
            percent.setText(String.format(Locale.US, "%.1f%%", 100.0 * done / total));
            double elapsed = (now - startNanos) / 1e9;
            double eta = speed[0] > 1 ? (total - done) / speed[0] : Double.NaN;
            StringBuilder sb = new StringBuilder(String.format(Locale.US,
                    "%.2f / %.2f GB\n%.1f MB/s · %d connections\nelapsed %s · remaining %s",
                    done / 1e9, total / 1e9, speed[0] / 1e6, state.activeConnections.get(), duration(elapsed),
                    duration(eta)));
            if (state.retries.get() > 0) {
                sb.append(String.format(Locale.US, "\nretries: %d (last: %s)", state.retries.get(), state.lastError));
            }
            details.setText(sb.toString());
            if (download == state) ui.postDelayed(ticker[0], 250);
        };
        ui.post(ticker[0]);
        log("Download started: " + f.repo + "/" + f.path + String.format(Locale.US, " (%.2f GB)", f.size / 1e9));

        net.execute(() -> {
            File result = null;
            String error = null;
            try {
                result = HuggingFace.download(f, modelsDir(), state);
            } catch (Exception e) {
                error = e.getMessage();
            }
            final File done = result;
            final String err = error;
            final double seconds = (System.nanoTime() - startNanos) / 1e9;
            ui.post(() -> {
                download = null;
                if (wake.isHeld()) wake.release();
                getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
                dialog.dismiss();
                if (done != null) {
                    log(String.format(Locale.US, "Downloaded %s in %s (%.1f MB/s avg, %d retries)%s", done.getName(),
                            duration(seconds), f.size / 1e6 / Math.max(seconds, 0.001), state.retries.get(),
                            f.sha256 != null ? ", SHA-256 verified" : ""));
                    loadModel(done, null);
                } else if (discard[0]) {
                    HuggingFace.discard(modelsDir(), f);
                    log("Download cancelled and partial file deleted: " + f.fileName());
                } else {
                    log("Download stopped: " + err + " — resume it from the Model menu");
                    addNote("Download of " + f.fileName() + " paused at "
                            + String.format(Locale.US, "%.0f%%", 100.0 * state.done.get() / Math.max(1, state.total))
                            + (err != null && !err.contains("paused") ? " (" + err + ")" : "")
                            + ". Open Model to resume.");
                }
            });
        });
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
                log("Creating engine on " + (useGpu ? "GPU (Vulkan)" : "CPU")
                        + ": mmap + GGUF parse + device detection + backend selection…");
                long t0 = System.nanoTime();
                long handle = LiyabNative.create(path, cache.getAbsolutePath(), THREADS,
                        useGpu ? LiyabNative.BACKEND_VULKAN : LiyabNative.BACKEND_CPU);
                if (useGpu) {
                    // Vulkan copies and repacks weights on first use: do it now, not on the first message.
                    log("Uploading weights to GPU memory (one-time warm-up)…");
                    long tw = System.nanoTime();
                    LiyabNative.generate(handle, "Hi", 1, 0f, bytes -> true);
                    log(String.format(Locale.US, "GPU warm-up done in %.2f s", (System.nanoTime() - tw) / 1e9));
                }
                double seconds = (System.nanoTime() - t0) / 1e9;
                drainNativeLogs();
                String info = LiyabNative.describe(handle);
                for (String line : info.split("\n")) log("  " + line);
                log(String.format(Locale.US, "Ready in %.2f s", seconds));
                engine = handle;
                modelName = label;
                loadedFile = file;
                loadedUri = uri;
                getPreferences(MODE_PRIVATE).edit()
                        .putString("model", file != null ? file.getAbsolutePath() : uri.toString()).apply();
                ui.post(() -> {
                    history.clear();
                    status.setText(String.format(Locale.US, "%s · %s · ready in %.1fs", label,
                            useGpu ? "GPU" : "CPU", seconds));
                    send.setEnabled(true);
                    addNote("Model loaded on " + (useGpu ? "GPU (Vulkan)" : "CPU") + ": " + label);
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
        // Tokens arrive faster than the screen refreshes: render at most every
        // FRAME_MS, so the bubble re-lays out ~30 times per second, not per token.
        final boolean[] renderPending = {false};
        final Runnable render = () -> {
            boolean follow = atBottom();
            synchronized (text) {
                renderPending[0] = false;
                reply.setText(text.toString().replaceFirst("^\\s+", ""));
            }
            if (follow) scrollToBottom(true);
        };
        worker.execute(() -> {
            String error = null;
            double[] stats = null;
            try {
                stats = LiyabNative.generate(engine, prompt, MAX_TOKENS, TEMPERATURE, bytes -> {
                    String piece = new String(bytes, StandardCharsets.UTF_8);
                    synchronized (text) {
                        text.append(piece);
                        if (!renderPending[0]) {
                            renderPending[0] = true;
                            ui.postDelayed(render, FRAME_MS);
                        }
                    }
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
                ui.removeCallbacks(render);
                render.run();  // final text, even if a frame was pending
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
                scrollToBottom(false);
            });
        });
    }
}
