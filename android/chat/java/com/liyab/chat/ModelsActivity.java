package com.liyab.chat;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Typeface;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * Models page with two tabs — "On this phone" (load, delete, resume partial downloads, pick a file)
 * and "Hugging Face" (search, browse a repository's files, download) — plus a live download card at
 * the top that stays visible on both tabs while a download runs.
 */
public final class ModelsActivity extends Activity implements EngineHolder.Listener, Downloads.Listener {
    private static final int PICK_MODEL = 1;

    private final Handler ui = new Handler(Looper.getMainLooper());
    private boolean hfTab;
    private Button localTabButton;
    private Button hfTabButton;
    private LinearLayout content;

    // Download card
    private LinearLayout downloadCard;
    private TextView downloadTitle;
    private ProgressBar downloadBar;
    private TextView downloadPercent;
    private TextView downloadDetails;
    private Button pauseButton;
    private Button cancelButton;
    private long lastDone = -1;
    private long lastTick;
    private double speed;
    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            refreshDownload();
            ui.postDelayed(this, 250);
        }
    };

    // Hugging Face tab state (survives tab switches)
    private String query = "";
    private final List<HuggingFace.Repo> repos = new ArrayList<>();
    private String openRepo;
    private String openRepoArch;  // general.architecture read from a file header, or null
    private final List<HuggingFace.GgufFile> repoFiles = new ArrayList<>();
    private String hfMessage = "Most downloaded GGUF repositories";

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = Ui.screen(this);
        root.addView(Ui.topBar(this, "Models", true));

        LinearLayout tabs = new LinearLayout(this);
        tabs.setPadding(Ui.dp(this, 12), Ui.dp(this, 4), Ui.dp(this, 12), Ui.dp(this, 4));
        localTabButton = Ui.pill(this, "On this phone", v -> showTab(false));
        hfTabButton = Ui.pill(this, "Hugging Face", v -> showTab(true));
        tabs.addView(localTabButton, new LinearLayout.LayoutParams(0, Ui.dp(this, 40), 1f));
        LinearLayout.LayoutParams hp = new LinearLayout.LayoutParams(0, Ui.dp(this, 40), 1f);
        hp.leftMargin = Ui.dp(this, 8);
        tabs.addView(hfTabButton, hp);
        root.addView(tabs);

        buildDownloadCard();
        root.addView(downloadCard);

        ScrollView scroll = new ScrollView(this);
        content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        content.setPadding(0, 0, 0, Ui.dp(this, 16));
        scroll.addView(content);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);
        showTab(getIntent().getBooleanExtra("hf", false));
    }

    @Override
    protected void onResume() {
        super.onResume();
        EngineHolder.addListener(this);
        Downloads.addListener(this);
        ui.post(tick);
        render();
    }

    @Override
    protected void onPause() {
        super.onPause();
        EngineHolder.removeListener(this);
        Downloads.removeListener(this);
        ui.removeCallbacks(tick);
    }

    @Override
    public void onEngineChanged() {
        if (!hfTab) render();
    }

    @Override
    public void onDownloadChanged() {
        render();
    }

    private void showTab(boolean hf) {
        hfTab = hf;
        localTabButton.setBackground(Ui.rounded(this, hf ? Ui.CARD : Ui.ACCENT, 18));
        hfTabButton.setBackground(Ui.rounded(this, hf ? Ui.ACCENT : Ui.CARD, 18));
        render();
        if (hf && repos.isEmpty() && openRepo == null) search("");
    }

    private void render() {
        content.removeAllViews();
        if (hfTab) renderHuggingFace();
        else renderLocal();
        refreshDownload();
    }

    // ------------------------------------------------------------- download card

    private void buildDownloadCard() {
        downloadCard = Ui.card(this);
        downloadTitle = Ui.bold(this, "", 13);
        downloadCard.addView(downloadTitle);
        downloadBar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        downloadBar.setMax(1000);
        LinearLayout.LayoutParams bp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, Ui.dp(this, 16));
        bp.topMargin = Ui.dp(this, 8);
        downloadCard.addView(downloadBar, bp);
        downloadPercent = Ui.bold(this, "", 22);
        downloadCard.addView(downloadPercent);
        downloadDetails = Ui.text(this, "", 12, Ui.MUTED);
        downloadDetails.setTypeface(Typeface.MONOSPACE);
        downloadCard.addView(downloadDetails);
        pauseButton = Ui.pill(this, "Pause", v -> {
            Downloads.pause();
            v.setEnabled(false);
        });
        cancelButton = Ui.pill(this, "Cancel", v -> Downloads.cancel());
        downloadCard.addView(Ui.buttonRow(this, pauseButton, cancelButton));
    }

    private void refreshDownload() {
        HuggingFace.DownloadState s = Downloads.state;
        HuggingFace.GgufFile f = Downloads.file;
        if (s == null || f == null) {
            boolean show = !Downloads.lastMessage.isEmpty();
            downloadCard.setVisibility(show ? View.VISIBLE : View.GONE);
            if (show) {
                downloadTitle.setText(Downloads.lastMessage);
                downloadBar.setVisibility(View.GONE);
                downloadPercent.setVisibility(View.GONE);
                downloadDetails.setVisibility(View.GONE);
                pauseButton.setVisibility(View.GONE);
                cancelButton.setVisibility(View.GONE);
            }
            lastDone = -1;
            speed = 0;
            return;
        }
        downloadCard.setVisibility(View.VISIBLE);
        downloadBar.setVisibility(View.VISIBLE);
        downloadPercent.setVisibility(View.VISIBLE);
        downloadDetails.setVisibility(View.VISIBLE);
        pauseButton.setVisibility(View.VISIBLE);
        cancelButton.setVisibility(View.VISIBLE);
        downloadTitle.setText(f.repo + "\n" + f.fileName());
        long total = Math.max(1, s.total > 0 ? s.total : f.size);
        if (s.verifying) {
            long checked = s.verified.get();
            downloadBar.setProgress((int) (1000 * checked / total));
            downloadPercent.setText(String.format(Locale.US, "Verifying %.0f%%", 100.0 * checked / total));
            downloadDetails.setText("Checking SHA-256 against Hugging Face…");
            return;
        }
        long done = s.done.get();
        long now = System.nanoTime();
        if (lastDone >= 0 && now > lastTick) {
            double instant = (done - lastDone) / ((now - lastTick) / 1e9);
            speed = speed == 0 ? instant : 0.8 * speed + 0.2 * instant;  // moving average
        }
        lastDone = done;
        lastTick = now;
        downloadBar.setProgress((int) (1000 * done / total));
        downloadPercent.setText(String.format(Locale.US, "%.1f%%", 100.0 * done / total));
        double elapsed = (now - Downloads.startNanos) / 1e9;
        double eta = speed > 1 ? (total - done) / speed : Double.NaN;
        StringBuilder sb = new StringBuilder(String.format(Locale.US,
                "%.2f / %.2f GB\n%.1f MB/s · %d connections\nelapsed %s · remaining %s", done / 1e9, total / 1e9,
                speed / 1e6, s.activeConnections.get(), Ui.duration(elapsed), Ui.duration(eta)));
        if (s.retries.get() > 0) {
            sb.append(String.format(Locale.US, "\nretries: %d (last: %s)", s.retries.get(), s.lastError));
        }
        downloadDetails.setText(sb.toString());
        pauseButton.setEnabled(!s.cancelled);
    }

    // ------------------------------------------------------------- local tab

    private void renderLocal() {
        File dir = EngineHolder.modelsDir();
        // Split models are listed once, through part 1; a part without its part 1 stays visible so it can be deleted.
        java.io.FilenameFilter models = (d, name) -> name.endsWith(".gguf")
                && (!HuggingFace.isLaterPart(name) || !new File(d, HuggingFace.firstPart(name)).exists());
        File[] own = dir.listFiles(models);
        File shared = EngineHolder.sharedDir();
        File[] pushed = shared != null ? shared.listFiles(models) : null;  // adb-pushed / older downloads
        List<File> all = new ArrayList<>();
        if (own != null) all.addAll(java.util.Arrays.asList(own));
        if (pushed != null) all.addAll(java.util.Arrays.asList(pushed));
        File[] local = all.toArray(new File[0]);
        List<HuggingFace.Partial> partials = HuggingFace.partials(dir);
        long used = 0;
        if (local != null) for (File f : local) used += EngineHolder.sizeOf(f);
        TextView summary = Ui.text(this, String.format(Locale.US, "%s used · %s free", Ui.gb(used),
                Ui.gb(dir.getUsableSpace())), 12, Ui.MUTED);
        summary.setPadding(Ui.dp(this, 16), Ui.dp(this, 6), Ui.dp(this, 16), 0);
        content.addView(summary);

        if ((local == null || local.length == 0) && partials.isEmpty()) {
            LinearLayout empty = Ui.card(this);
            empty.addView(Ui.bold(this, "No models yet", 15));
            empty.addView(Ui.text(this, "Download one from Hugging Face, or pick a .gguf file from the phone.", 13, Ui.MUTED));
            empty.addView(Ui.buttonRow(this, Ui.primary(this, "Browse Hugging Face", v -> showTab(true))));
            content.addView(empty);
        }
        if (local != null) {
            for (File f : local) content.addView(localCard(f));
        }
        for (HuggingFace.Partial p : partials) {
            if (Downloads.running() && Downloads.file != null
                    && HuggingFace.firstPart(Downloads.file.path).equals(HuggingFace.firstPart(p.file.path))) continue;
            content.addView(partialCard(p));
        }
        LinearLayout pick = Ui.card(this);
        pick.addView(Ui.bold(this, "Model stored elsewhere?", 14));
        pick.addView(Ui.text(this, "Open any .gguf with the system file picker (no copy, no storage permission).", 12, Ui.MUTED));
        pick.addView(Ui.buttonRow(this, Ui.pill(this, "Browse phone storage…", v -> {
            Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
            intent.addCategory(Intent.CATEGORY_OPENABLE);
            intent.setType("*/*");
            startActivityForResult(intent, PICK_MODEL);
        })));
        content.addView(pick);
    }

    private View localCard(File f) {
        boolean loaded = EngineHolder.isLoaded(f);
        LinearLayout card = Ui.card(this);
        card.addView(Ui.bold(this, f.getName(), 14));
        List<File> parts = HuggingFace.localParts(f);
        int missing = 0;
        for (File part : parts) if (!part.exists()) missing++;
        TextView meta = Ui.text(this, Ui.gb(EngineHolder.sizeOf(f))
                + (parts.size() > 1 ? " · " + parts.size() + " parts" + (missing > 0 ? ", " + missing + " missing" : "") : "")
                + (loaded ? " · ● loaded on "
                + (EngineHolder.useGpu() ? "GPU" : "CPU") : ""), 12, loaded ? Ui.GOOD : Ui.MUTED);
        card.addView(meta);
        final boolean shared = EngineHolder.inSharedDir(f);
        if (shared) {
            card.addView(Ui.text(this, "In the shared folder: Android routes it through FUSE, so models larger than RAM "
                    + "stream several times slower. Move it to app storage (fast direct reads).", 12, Ui.BAD));
        }
        Button load = Ui.primary(this, loaded ? "Loaded" : "Load", v -> {
            EngineHolder.load(f, null);
            render();
        });
        load.setEnabled(!loaded && !EngineHolder.busy);
        load.setAlpha(load.isEnabled() ? 1f : 0.4f);
        Button chat = Ui.pill(this, "Chat", v -> startActivity(new Intent(this, ChatActivity.class)));
        chat.setEnabled(loaded && !EngineHolder.busy);
        chat.setAlpha(chat.isEnabled() ? 1f : 0.4f);
        Button delete = Ui.pill(this, "Delete", v -> confirmDelete(f.getName(), EngineHolder.sizeOf(f), () -> {
            boolean ok = true;
            for (File part : parts) ok &= !part.exists() || part.delete();
            if (ok && f.getAbsolutePath().equals(EngineHolder.prefs().getString("model", ""))) {
                EngineHolder.prefs().edit().remove("model").apply();
            }
            DebugLog.add((ok ? "Deleted " : "Could not delete ") + f.getName());
        }));
        delete.setEnabled(!loaded);  // the loaded model is memory-mapped
        delete.setAlpha(delete.isEnabled() ? 1f : 0.4f);
        if (shared) {
            boolean busy = EngineHolder.moving != null;
            Button move = Ui.pill(this, busy && f.getName().equals(EngineHolder.moving) ? "Moving…" : "Move to app storage",
                    v -> {
                        EngineHolder.moveToAppStorage(f);
                        render();
                    });
            move.setEnabled(!loaded && !busy);
            move.setAlpha(move.isEnabled() ? 1f : 0.4f);
            card.addView(Ui.buttonRow(this, load, chat, delete));
            card.addView(Ui.buttonRow(this, move));
            return card;
        }
        card.addView(Ui.buttonRow(this, load, chat, delete));
        return card;
    }

    private View partialCard(HuggingFace.Partial p) {
        LinearLayout card = Ui.card(this);
        card.addView(Ui.bold(this, "⏸ " + p.file.fileName(), 14));
        card.addView(Ui.text(this, String.format(Locale.US, "%.0f%% of %s downloaded · %s", 100.0 * p.done / Math.max(1, p.file.size),
                Ui.gb(p.file.size), p.file.repo), 12, Ui.MUTED));
        // A part of a split model resumes the whole set (finished parts are skipped).
        Button resume = Ui.primary(this, "Resume", v -> {
            if (p.file.split) Downloads.startByName(this, p.file.repo, HuggingFace.firstPart(p.file.path));
            else Downloads.start(this, p.file);
        });
        resume.setEnabled(!Downloads.running());
        resume.setAlpha(resume.isEnabled() ? 1f : 0.4f);
        Button delete = Ui.pill(this, "Delete", v -> confirmDelete(p.file.fileName() + " (partial)",
                new File(EngineHolder.modelsDir(), p.file.fileName() + ".part").length(), () -> {
                    HuggingFace.discard(EngineHolder.modelsDir(), p.file);
                    DebugLog.add("Deleted partial download " + p.file.fileName());
                }));
        card.addView(Ui.buttonRow(this, resume, delete));
        return card;
    }

    private void confirmDelete(String name, long bytes, Runnable delete) {
        new AlertDialog.Builder(this)
                .setTitle("Delete model?")
                .setMessage(String.format(Locale.US, "%s\n\nThis frees %s.", name, Ui.gb(bytes)))
                .setPositiveButton("Delete", (d, w) -> {
                    delete.run();
                    render();
                })
                .setNegativeButton("Cancel", null)
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
            // provider without persistable grants: works for this session only
        }
        EngineHolder.load(null, uri);
    }

    // ------------------------------------------------------------- Hugging Face tab

    private void renderHuggingFace() {
        LinearLayout searchCard = Ui.card(this);
        LinearLayout row = new LinearLayout(this);
        EditText field = new EditText(this);
        field.setText(query);
        field.setHint("Search (e.g. qwen2.5 0.5b, tinyllama, llama 3.2)");
        field.setHintTextColor(Ui.MUTED);
        field.setTextColor(Ui.TEXT);
        field.setSingleLine(true);
        field.setImeOptions(EditorInfo.IME_ACTION_SEARCH);
        row.addView(field, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        Button go = Ui.primary(this, "Search", v -> search(field.getText().toString().trim()));
        row.addView(go);
        field.setOnEditorActionListener((v, actionId, event) -> {
            search(field.getText().toString().trim());
            return true;
        });
        searchCard.addView(row);
        TextView hint = Ui.text(this, hfMessage, 12, Ui.MUTED);
        searchCard.addView(hint);
        content.addView(searchCard);

        if (openRepo != null) {
            LinearLayout header = Ui.card(this);
            header.addView(Ui.bold(this, openRepo, 15));
            if (openRepoArch != null && !HuggingFace.SUPPORTED_ARCHS.contains(openRepoArch)) {
                header.addView(Ui.text(this, "✗ Architecture '" + openRepoArch + "' is not supported yet (supported: "
                        + String.join(", ", HuggingFace.SUPPORTED_ARCHS) + "). None of these files can run.", 13, Ui.BAD));
            } else {
                header.addView(Ui.text(this, (openRepoArch != null ? "✓ Architecture " + openRepoArch + " · " : "")
                        + "RAM fit is based on currently free memory.", 12, openRepoArch != null ? Ui.GOOD : Ui.MUTED));
            }
            header.addView(Ui.buttonRow(this, Ui.pill(this, "← Back to results", v -> {
                openRepo = null;
                repoFiles.clear();
                render();
            })));
            content.addView(header);
            for (HuggingFace.GgufFile f : repoFiles) content.addView(fileCard(f));
            return;
        }
        for (HuggingFace.Repo r : repos) {
            LinearLayout card = Ui.card(this);
            card.addView(Ui.bold(this, r.id, 14));
            card.addView(Ui.text(this, String.format(Locale.US, "⬇ %s   ♥ %d", compact(r.downloads), r.likes), 12, Ui.MUTED));
            card.setOnClickListener(v -> openRepository(r.id));
            content.addView(card);
        }
    }

    private View fileCard(HuggingFace.GgufFile f) {
        boolean fits = fitsInRam(f.size);
        LinearLayout card = Ui.card(this);
        card.addView(Ui.bold(this, f.fileName(), 14));
        card.addView(Ui.text(this, Ui.gb(f.size) + " · " + f.quant
                + (f.parts.isEmpty() ? "" : " · " + f.parts.size() + " parts"), 12, Ui.MUTED));
        card.addView(Ui.text(this, fits ? "Fits in memory" : "Larger than free RAM: runs from flash, well under 1 token/s",
                12, fits ? Ui.GOOD : Ui.BAD));
        boolean present = f.presentIn(EngineHolder.modelsDir());
        Button download = Ui.primary(this, present ? "Downloaded" : "Download", v -> {
            Downloads.start(this, f);
            render();
        });
        boolean archOk = openRepoArch == null || HuggingFace.SUPPORTED_ARCHS.contains(openRepoArch);
        download.setEnabled(archOk && !present && !Downloads.running()
                && EngineHolder.modelsDir().getUsableSpace() > f.size);
        download.setAlpha(download.isEnabled() ? 1f : 0.4f);
        card.addView(Ui.buttonRow(this, download));
        return card;
    }

    private void search(String q) {
        query = q;
        hfMessage = "Searching…";
        hideKeyboard();
        render();
        Downloads.net.execute(() -> {
            try {
                List<HuggingFace.Repo> found = HuggingFace.search(q);
                ui.post(() -> {
                    repos.clear();
                    repos.addAll(found);
                    openRepo = null;
                    hfMessage = found.isEmpty() ? "No GGUF repositories found"
                            : found.size() + " repositories · tap one to see its files";
                    render();
                });
            } catch (Exception e) {
                ui.post(() -> {
                    hfMessage = "Search failed: " + e.getMessage();
                    render();
                });
            }
        });
    }

    private void openRepository(String repo) {
        hfMessage = "Listing " + repo + "…";
        render();
        Downloads.net.execute(() -> {
            try {
                List<HuggingFace.GgufFile> files = HuggingFace.files(repo);
                List<HuggingFace.GgufFile> supported = new ArrayList<>();
                for (HuggingFace.GgufFile f : files) if (f.compat == HuggingFace.Compat.OK) supported.add(f);
                final String arch = supported.isEmpty() ? null : HuggingFace.architecture(supported.get(0));
                DebugLog.add("Repository " + repo + ": architecture " + (arch != null ? arch : "unknown"));
                ui.post(() -> {
                    openRepo = repo;
                    openRepoArch = arch;
                    repoFiles.clear();
                    repoFiles.addAll(supported);
                    int hidden = files.size() - supported.size();
                    hfMessage = supported.size() + " model files" + (hidden > 0 ? " (" + hidden
                            + " hidden: single parts of split models, vision projectors, imatrix or unsupported formats)" : "");
                    render();
                });
            } catch (Exception e) {
                ui.post(() -> {
                    hfMessage = "Could not list " + repo + ": " + e.getMessage();
                    render();
                });
            }
        });
    }

    private void hideKeyboard() {
        View focus = getCurrentFocus();
        if (focus != null) {
            ((InputMethodManager) getSystemService(INPUT_METHOD_SERVICE)).hideSoftInputFromWindow(focus.getWindowToken(), 0);
        }
    }

    private boolean fitsInRam(long bytes) {
        android.app.ActivityManager am = (android.app.ActivityManager) getSystemService(ACTIVITY_SERVICE);
        android.app.ActivityManager.MemoryInfo info = new android.app.ActivityManager.MemoryInfo();
        am.getMemoryInfo(info);
        return bytes + 1_500_000_000L < info.availMem;
    }

    private static String compact(long n) {
        if (n >= 1_000_000) return String.format(Locale.US, "%.1fM", n / 1e6);
        if (n >= 1_000) return String.format(Locale.US, "%.1fk", n / 1e3);
        return Long.toString(n);
    }
}
