package com.liyab.chat;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.Intent;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.util.Locale;

/**
 * Home: model status, quick actions (chat, models, CPU/GPU, unload) and the always-visible debug log.
 *
 * Scriptable entry points (tests / automation / deep links):
 *   adb shell am start -n com.liyab.chat/.HomeActivity --es download "owner/repo|file.gguf"
 *   adb shell am start -n com.liyab.chat/.HomeActivity --es load model.gguf     (file in the models folder)
 *   adb shell am start -n com.liyab.chat/.HomeActivity --es open models|hf|chat|settings
 */
public final class HomeActivity extends Activity
        implements EngineHolder.Listener, Downloads.Listener, DebugLog.Listener {
    private final Handler ui = new Handler(Looper.getMainLooper());
    private TextView modelLine;
    private TextView backendLine;
    private TextView deviceLine;
    private TextView downloadLine;
    private Button chatButton;
    private Button backendButton;
    private Button unloadButton;
    private Button settingsButton;
    private TextView log;
    private ScrollView logScroll;
    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            refreshDownload();
            ui.postDelayed(this, 500);
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        buildUi();
        handleIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        handleIntent(intent);
    }

    private void handleIntent(Intent intent) {
        String download = intent.getStringExtra("download");
        String load = intent.getStringExtra("load");
        String open = intent.getStringExtra("open");
        if (download != null && download.contains("|")) {
            String[] parts = download.split("\\|", 2);
            Downloads.startByName(this, parts[0], parts[1]);
        } else if (load != null) {
            java.io.File f = new java.io.File(EngineHolder.modelsDir(), load);
            if (f.canRead()) EngineHolder.load(f, null);
            else DebugLog.add("ERROR: cannot read " + f);
        } else {
            EngineHolder.restoreLast();
        }
        if ("models".equals(open)) startActivity(new Intent(this, ModelsActivity.class));
        else if ("hf".equals(open)) startActivity(new Intent(this, ModelsActivity.class).putExtra("hf", true));
        else if ("chat".equals(open)) startActivity(new Intent(this, ChatActivity.class));
        else if ("settings".equals(open)) startActivity(new Intent(this, SettingsActivity.class));
    }

    @Override
    protected void onResume() {
        super.onResume();
        EngineHolder.addListener(this);
        Downloads.addListener(this);
        DebugLog.addListener(this);
        log.setText(DebugLog.snapshot());
        scrollLogToEnd();
        refresh();
        ui.post(tick);
    }

    @Override
    protected void onPause() {
        super.onPause();
        EngineHolder.removeListener(this);
        Downloads.removeListener(this);
        DebugLog.removeListener(this);
        ui.removeCallbacks(tick);
    }

    private void buildUi() {
        LinearLayout root = Ui.screen(this);
        root.addView(Ui.topBar(this, "🔥 Liyab", false));

        LinearLayout status = Ui.card(this);
        status.addView(Ui.text(this, "MODEL", 11, Ui.MUTED));
        modelLine = Ui.bold(this, "", 16);
        status.addView(modelLine);
        backendLine = Ui.text(this, "", 13, Ui.MUTED);
        status.addView(backendLine);
        deviceLine = Ui.text(this, "", 12, Ui.MUTED);
        status.addView(deviceLine);
        downloadLine = Ui.text(this, "", 12, Ui.ACCENT);
        downloadLine.setOnClickListener(v -> startActivity(new Intent(this, ModelsActivity.class)));
        status.addView(downloadLine);
        root.addView(status);

        LinearLayout actions = Ui.card(this);
        actions.addView(Ui.text(this, "QUICK ACTIONS", 11, Ui.MUTED));
        chatButton = Ui.primary(this, "💬  Open chat", v -> startActivity(new Intent(this, ChatActivity.class)));
        Button models = Ui.pill(this, "⬇  Models", v -> startActivity(new Intent(this, ModelsActivity.class)));
        actions.addView(Ui.buttonRow(this, chatButton, models));
        backendButton = Ui.pill(this, "", v -> EngineHolder.setGpu(!EngineHolder.useGpu()));
        unloadButton = Ui.pill(this, "⏏  Unload", v -> EngineHolder.unload());
        actions.addView(Ui.buttonRow(this, backendButton, unloadButton));
        settingsButton = Ui.pill(this, "⚙  Model settings", v -> startActivity(new Intent(this, SettingsActivity.class)));
        actions.addView(Ui.buttonRow(this, settingsButton));
        root.addView(actions);

        LinearLayout debug = Ui.card(this);
        LinearLayout header = new LinearLayout(this);
        header.setGravity(Gravity.CENTER_VERTICAL);
        TextView title = Ui.text(this, "DEBUG LOG", 11, Ui.MUTED);
        header.addView(title, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        header.addView(Ui.pill(this, "Copy", v -> {
            ClipboardManager cm = (ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            cm.setPrimaryClip(ClipData.newPlainText("liyab log", DebugLog.snapshot()));
        }));
        header.addView(Ui.pill(this, "Clear", v -> {
            DebugLog.clear();
            log.setText("");
        }));
        debug.addView(header);
        logScroll = new ScrollView(this);
        logScroll.setBackground(Ui.rounded(this, Ui.DEBUG_BG, 10));
        log = new TextView(this);
        log.setTextColor(Ui.DEBUG_FG);
        log.setTextSize(10.5f);
        log.setTypeface(Typeface.MONOSPACE);
        log.setTextIsSelectable(true);
        log.setPadding(Ui.dp(this, 8), Ui.dp(this, 6), Ui.dp(this, 8), Ui.dp(this, 6));
        logScroll.addView(log);
        LinearLayout.LayoutParams logParams = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f);
        logParams.topMargin = Ui.dp(this, 6);
        debug.addView(logScroll, logParams);
        LinearLayout.LayoutParams debugParams = (LinearLayout.LayoutParams) debug.getLayoutParams();
        debugParams.height = 0;
        debugParams.weight = 1f;
        debugParams.bottomMargin = Ui.dp(this, 12);
        root.addView(debug, debugParams);
        setContentView(root);
    }

    private void refresh() {
        boolean loaded = EngineHolder.handle != 0;
        modelLine.setText(loaded ? EngineHolder.modelLabel : EngineHolder.busy ? "Loading…" : "No model loaded");
        backendLine.setText(!loaded && !EngineHolder.busy && EngineHolder.status.equals("No model loaded")
                ? "Open Models to load one from the phone or download it from Hugging Face" : EngineHolder.status);
        String desc = EngineHolder.description;
        int nl = desc.indexOf('\n');
        deviceLine.setText(nl > 0 ? desc.substring(0, nl).replace("SoC:", "").trim() : "");
        chatButton.setEnabled(loaded && !EngineHolder.busy);
        chatButton.setAlpha(chatButton.isEnabled() ? 1f : 0.4f);
        unloadButton.setEnabled(loaded && !EngineHolder.busy);
        unloadButton.setAlpha(unloadButton.isEnabled() ? 1f : 0.4f);
        settingsButton.setEnabled(loaded && !EngineHolder.busy);
        settingsButton.setAlpha(settingsButton.isEnabled() ? 1f : 0.4f);
        backendButton.setText(EngineHolder.useGpu() ? "⚡  Backend: GPU" : "🧠  Backend: CPU");
        backendButton.setEnabled(!EngineHolder.busy);
        refreshDownload();
    }

    private void refreshDownload() {
        HuggingFace.DownloadState s = Downloads.state;
        if (s != null && Downloads.file != null) {
            downloadLine.setVisibility(View.VISIBLE);
            downloadLine.setText(String.format(Locale.US, "⬇ %s — %.0f%% (tap for details)", Downloads.file.fileName(),
                    s.verifying ? 100.0 : 100.0 * s.done.get() / Math.max(1, s.total)));
        } else {
            downloadLine.setVisibility(View.GONE);
        }
    }

    private void scrollLogToEnd() {
        logScroll.post(() -> logScroll.scrollTo(0, log.getBottom()));
    }

    @Override
    public void onEngineChanged() {
        refresh();
    }

    @Override
    public void onDownloadChanged() {
        refresh();
    }

    @Override
    public void onLine(String line) {
        boolean follow = logScroll.getChildAt(0).getBottom() - (logScroll.getScrollY() + logScroll.getHeight()) < 80;
        log.append((log.length() > 0 ? "\n" : "") + line);
        if (follow) scrollLogToEnd();
    }
}
