package com.liyab.chat;

import android.app.Activity;
import android.content.Context;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.util.TypedValue;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowInsets;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

/** Shared look and small view factories (plain Android views, no AndroidX). */
final class Ui {
    static final int BG = Color.rgb(16, 17, 20);
    static final int CARD = Color.rgb(28, 30, 35);
    static final int CARD_HI = Color.rgb(40, 43, 50);
    static final int USER_BG = Color.rgb(46, 86, 160);
    static final int TEXT = Color.WHITE;
    static final int MUTED = Color.rgb(150, 155, 165);
    static final int ACCENT = Color.rgb(255, 122, 48);
    static final int GOOD = Color.rgb(90, 200, 120);
    static final int BAD = Color.rgb(235, 90, 80);
    static final int DEBUG_BG = Color.rgb(8, 9, 11);
    static final int DEBUG_FG = Color.rgb(120, 220, 140);

    private Ui() {}

    static int dp(Context c, float v) {
        return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, c.getResources().getDisplayMetrics());
    }

    static GradientDrawable rounded(Context c, int color, float radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(color);
        d.setCornerRadius(dp(c, radiusDp));
        return d;
    }

    /** Root column with edge-to-edge insets (targetSdk 35) and the dark background. */
    static LinearLayout screen(Activity a) {
        a.getWindow().setStatusBarColor(BG);
        a.getWindow().setNavigationBarColor(BG);
        LinearLayout root = new LinearLayout(a);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setBackgroundColor(BG);
        root.setOnApplyWindowInsetsListener((v, insets) -> {
            android.graphics.Insets bars = insets.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.ime());
            v.setPadding(bars.left, bars.top, bars.right, bars.bottom);
            return WindowInsets.CONSUMED;
        });
        return root;
    }

    /** Top bar: optional back arrow, title, and trailing action buttons. */
    static LinearLayout topBar(Activity a, String title, boolean back, View... actions) {
        LinearLayout bar = new LinearLayout(a);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setGravity(Gravity.CENTER_VERTICAL);
        bar.setPadding(dp(a, back ? 4 : 16), dp(a, 8), dp(a, 8), dp(a, 4));
        if (back) {
            Button arrow = pill(a, "←", v -> a.finish());
            arrow.setTextSize(20);
            arrow.setBackground(null);
            bar.addView(arrow);
        }
        TextView t = new TextView(a);
        t.setText(title);
        t.setTextColor(TEXT);
        t.setTextSize(20);
        t.setTypeface(Typeface.DEFAULT_BOLD);
        bar.addView(t, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        for (View v : actions) bar.addView(v);
        return bar;
    }

    static Button pill(Context c, String label, View.OnClickListener l) {
        Button b = new Button(c);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(13);
        b.setTextColor(TEXT);
        b.setBackground(rounded(c, CARD_HI, 18));
        b.setMinWidth(0);
        b.setMinimumWidth(0);
        b.setPadding(dp(c, 12), 0, dp(c, 12), 0);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, dp(c, 36));
        p.leftMargin = dp(c, 6);
        b.setLayoutParams(p);
        b.setOnClickListener(l);
        return b;
    }

    static Button primary(Context c, String label, View.OnClickListener l) {
        Button b = pill(c, label, l);
        b.setBackground(rounded(c, ACCENT, 18));
        return b;
    }

    static LinearLayout card(Context c) {
        LinearLayout card = new LinearLayout(c);
        card.setOrientation(LinearLayout.VERTICAL);
        card.setBackground(rounded(c, CARD, 16));
        card.setPadding(dp(c, 14), dp(c, 12), dp(c, 14), dp(c, 12));
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        p.setMargins(dp(c, 12), dp(c, 6), dp(c, 12), dp(c, 6));
        card.setLayoutParams(p);
        return card;
    }

    static TextView text(Context c, String s, float size, int color) {
        TextView t = new TextView(c);
        t.setText(s);
        t.setTextSize(size);
        t.setTextColor(color);
        return t;
    }

    static TextView bold(Context c, String s, float size) {
        TextView t = text(c, s, size, TEXT);
        t.setTypeface(Typeface.DEFAULT_BOLD);
        return t;
    }

    /** Horizontal row of buttons with equal weight. */
    static LinearLayout buttonRow(Context c, Button... buttons) {
        LinearLayout row = new LinearLayout(c);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(0, dp(c, 8), 0, 0);
        for (Button b : buttons) {
            LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, dp(c, 40), 1f);
            p.leftMargin = row.getChildCount() == 0 ? 0 : dp(c, 8);
            row.addView(b, p);
        }
        return row;
    }

    static String gb(long bytes) {
        return String.format(java.util.Locale.US, "%.2f GB", bytes / 1e9);
    }

    static String duration(double seconds) {
        if (Double.isNaN(seconds) || Double.isInfinite(seconds) || seconds < 0) return "--:--";
        long s = Math.round(seconds);
        return s >= 3600 ? String.format(java.util.Locale.US, "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
                : String.format(java.util.Locale.US, "%d:%02d", s / 60, s % 60);
    }
}
