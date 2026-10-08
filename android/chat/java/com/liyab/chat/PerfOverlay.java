package com.liyab.chat;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Path;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.util.Locale;

/**
 * Floating performance panel drawn over every screen of the app while enabled (Home: "Overlay"):
 * one 60-second sparkline per metric of PerfMonitor with its current value, plus battery level,
 * temperature, thermal status and energy per token. Drag the header to move it; tap it to fold the
 * panel into a one-line summary. Position and folding are shared by all screens.
 */
final class PerfOverlay extends LinearLayout implements PerfMonitor.Listener {
    // Series colors: the dark-mode steps of a CVD-validated categorical palette (text stays in ink colors).
    private static final int[] COLORS = {
        Color.rgb(0x39, 0x87, 0xe5), Color.rgb(0xd9, 0x59, 0x26), Color.rgb(0x19, 0x9e, 0x70),
        Color.rgb(0xc9, 0x85, 0x00), Color.rgb(0x90, 0x85, 0xe9)};
    private static float posX = -1, posY = -1;  // shared across screens
    private static boolean folded;

    private final TextView header;
    private final LinearLayout body;
    private final TextView[] values = new TextView[PerfMonitor.METRICS];
    private final Sparkline[] lines = new Sparkline[PerfMonitor.METRICS];
    private final TextView footer;

    static boolean enabled() {
        return EngineHolder.prefs().getBoolean("overlay", false);
    }

    static void setEnabled(Context context, boolean on) {
        EngineHolder.prefs().edit().putBoolean("overlay", on).apply();
        if (on) PerfMonitor.start(context);
        else PerfMonitor.stop();
    }

    /** Adds the panel to `activity` (no-op when disabled or already attached). */
    static void attach(Activity activity) {
        if (!enabled()) return;
        ViewGroup root = activity.findViewById(android.R.id.content);
        if (root == null || root.findViewWithTag("perf-overlay") != null) return;
        PerfMonitor.start(activity);
        PerfOverlay overlay = new PerfOverlay(activity);
        overlay.setTag("perf-overlay");
        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(Ui.dp(activity, 230),
                ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP | Gravity.START);
        root.addView(overlay, lp);
        overlay.post(() -> overlay.place(root));
    }

    static void detach(Activity activity) {
        ViewGroup root = activity.findViewById(android.R.id.content);
        View v = root == null ? null : root.findViewWithTag("perf-overlay");
        if (v != null) root.removeView(v);
    }

    @SuppressLint("ClickableViewAccessibility")
    private PerfOverlay(Context c) {
        super(c);
        setOrientation(VERTICAL);
        setBackground(Ui.rounded(c, Color.argb(225, 12, 13, 16), 12));
        setPadding(Ui.dp(c, 10), Ui.dp(c, 6), Ui.dp(c, 10), Ui.dp(c, 8));
        setElevation(Ui.dp(c, 12));
        header = Ui.text(c, "", 12, Ui.TEXT);
        header.setPadding(0, Ui.dp(c, 2), 0, Ui.dp(c, 4));
        addView(header);
        body = new LinearLayout(c);
        body.setOrientation(VERTICAL);
        for (int m = 0; m < PerfMonitor.METRICS; ++m) {
            LinearLayout row = new LinearLayout(c);
            row.setGravity(Gravity.CENTER_VERTICAL);
            TextView label = Ui.text(c, PerfMonitor.NAMES[m], 11, Ui.MUTED);
            row.addView(label, new LayoutParams(Ui.dp(c, 44), ViewGroup.LayoutParams.WRAP_CONTENT));
            lines[m] = new Sparkline(c, m, COLORS[m]);
            row.addView(lines[m], new LayoutParams(0, Ui.dp(c, 26), 1f));
            values[m] = Ui.text(c, "", 11, Ui.TEXT);
            values[m].setGravity(Gravity.END);
            row.addView(values[m], new LayoutParams(Ui.dp(c, 64), ViewGroup.LayoutParams.WRAP_CONTENT));
            body.addView(row);
        }
        footer = Ui.text(c, "", 10, Ui.MUTED);
        footer.setPadding(0, Ui.dp(c, 4), 0, 0);
        body.addView(footer);
        addView(body);
        body.setVisibility(folded ? GONE : VISIBLE);

        final float[] down = new float[4];  // touch x, y and panel x, y at ACTION_DOWN
        header.setOnTouchListener((v, e) -> {
            switch (e.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    down[0] = e.getRawX();
                    down[1] = e.getRawY();
                    down[2] = getX();
                    down[3] = getY();
                    return true;
                case MotionEvent.ACTION_MOVE:
                    posX = down[2] + e.getRawX() - down[0];
                    posY = down[3] + e.getRawY() - down[1];
                    place((View) getParent());
                    return true;
                case MotionEvent.ACTION_UP:
                    if (Math.abs(e.getRawX() - down[0]) < Ui.dp(c, 6) && Math.abs(e.getRawY() - down[1]) < Ui.dp(c, 6)) {
                        folded = !folded;  // a tap folds or unfolds
                        body.setVisibility(folded ? GONE : VISIBLE);
                        render();
                    }
                    return true;
                default:
                    return false;
            }
        });
        render();
    }

    /** Clamps the shared position into `root` (default: top-right, below the status bar). */
    private void place(View root) {
        if (root == null) return;
        if (posX < 0 && posY < 0) {
            posX = root.getWidth() - getWidth() - Ui.dp(getContext(), 8);
            posY = Ui.dp(getContext(), 90);
        }
        posX = Math.max(0, Math.min(posX, root.getWidth() - getWidth()));
        posY = Math.max(0, Math.min(posY, root.getHeight() - getHeight()));
        setX(posX);
        setY(posY);
    }

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        PerfMonitor.addListener(this);
    }

    @Override
    protected void onDetachedFromWindow() {
        PerfMonitor.removeListener(this);
        super.onDetachedFromWindow();
    }

    @Override
    public void onSample() {
        render();
    }

    private void render() {
        float[] v = PerfMonitor.latest;
        String speed = v[PerfMonitor.SPEED] > 0 ? String.format(Locale.US, " · %.1f tok/s", v[PerfMonitor.SPEED]) : "";
        header.setText(String.format(Locale.US, "📈 CPU %.0f%% · GPU %.0f%% · %.1f W%s  %s", v[PerfMonitor.CPU],
                v[PerfMonitor.GPU], v[PerfMonitor.POWER], speed, folded ? "▸" : "▾"));
        if (folded) return;
        for (int m = 0; m < PerfMonitor.METRICS; ++m) {
            values[m].setText(String.format(Locale.US, m == PerfMonitor.POWER ? "%.2f %s" : "%.1f %s", v[m],
                    PerfMonitor.UNITS[m]));
            lines[m].invalidate();
        }
        StringBuilder f = new StringBuilder();
        if (PerfMonitor.batteryPercent >= 0) f.append(String.format(Locale.US, "🔋 %.0f%%", PerfMonitor.batteryPercent));
        if (PerfMonitor.batteryTempC > -1) f.append(String.format(Locale.US, " · %.1f °C", PerfMonitor.batteryTempC));
        if (PerfMonitor.thermalStatus > 0) f.append(" · thermal ").append(PerfMonitor.thermalStatus);
        if (PerfMonitor.joulesPerToken > 0) f.append(String.format(Locale.US, " · %.2f J/token", PerfMonitor.joulesPerToken));
        if (PerfMonitor.charging) f.append("\n⚡ charging: power = net battery flow, not consumption");
        f.append("\nGPU = time the GPU spends on Liyab's work");
        footer.setText(f.toString());
    }

    /** 60-second line + area of one metric, newest sample on the right. */
    private static final class Sparkline extends View {
        private final int metric;
        private final Paint line = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint fill = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint grid = new Paint();
        private final Path path = new Path();

        Sparkline(Context c, int metric, int color) {
            super(c);
            this.metric = metric;
            line.setColor(color);
            line.setStyle(Paint.Style.STROKE);
            line.setStrokeWidth(Ui.dp(c, 1.5f));
            line.setStrokeJoin(Paint.Join.ROUND);
            fill.setColor(Color.argb(60, Color.red(color), Color.green(color), Color.blue(color)));
            grid.setColor(Color.rgb(0x2c, 0x2c, 0x2a));
            grid.setStrokeWidth(1);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            final float w = getWidth(), h = getHeight(), pad = 2;
            canvas.drawLine(0, h - 1, w, h - 1, grid);
            final int n = PerfMonitor.count;
            if (n < 2) return;
            float max = PerfMonitor.FIXED_MAX[metric];
            if (max == 0) {
                for (int i = 0; i < n; ++i) max = Math.max(max, PerfMonitor.history[metric][i]);
                max = max <= 0 ? 1 : max * 1.15f;
            }
            final float step = w / (PerfMonitor.CAPACITY - 1);
            final float x0 = w - step * (n - 1);
            path.reset();
            for (int i = 0; i < n; ++i) {
                final float x = x0 + step * i;
                final float y = pad + (h - 2 * pad) * (1 - Math.min(1f, PerfMonitor.history[metric][i] / max));
                if (i == 0) path.moveTo(x, y);
                else path.lineTo(x, y);
            }
            canvas.drawPath(path, line);
            path.lineTo(w, h);
            path.lineTo(x0, h);
            path.close();
            canvas.drawPath(path, fill);
        }
    }
}
