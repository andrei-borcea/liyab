package com.liyab.chat;

import android.app.Activity;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.inputmethod.EditorInfo;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.nio.charset.StandardCharsets;
import java.util.Locale;

/**
 * Chat with the loaded model. The conversation lives in EngineHolder.history, so leaving and
 * re-opening the chat keeps it. Leaving while a reply streams stops that reply.
 */
public final class ChatActivity extends Activity implements EngineHolder.Listener {
    private static final String SYSTEM_PROMPT =
            "You are Liyab, a helpful assistant running entirely on this phone, without internet. "
                    + "Answer clearly and concisely.";
    private static final int MAX_TOKENS = 384;
    private static final float TEMPERATURE = 0.7f;
    private static final int MAX_TURNS = 6;   // history kept in the prompt (2048-token context)
    private static final long FRAME_MS = 33;  // streaming redraw interval (~30 fps)

    private final Handler ui = new Handler(Looper.getMainLooper());
    private LinearLayout messages;
    private ScrollView scroll;
    private EditText input;
    private Button send;
    private TextView status;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = Ui.screen(this);
        root.addView(Ui.topBar(this, "Chat", true, Ui.pill(this, "New", v -> newChat())));
        status = Ui.text(this, "", 12, Ui.MUTED);
        status.setPadding(Ui.dp(this, 16), 0, Ui.dp(this, 16), Ui.dp(this, 6));
        root.addView(status);

        scroll = new ScrollView(this);
        messages = new LinearLayout(this);
        messages.setOrientation(LinearLayout.VERTICAL);
        messages.setPadding(Ui.dp(this, 12), Ui.dp(this, 4), Ui.dp(this, 12), Ui.dp(this, 12));
        scroll.addView(messages);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));

        LinearLayout bar = new LinearLayout(this);
        bar.setOrientation(LinearLayout.HORIZONTAL);
        bar.setPadding(Ui.dp(this, 12), Ui.dp(this, 8), Ui.dp(this, 12), Ui.dp(this, 12));
        bar.setGravity(Gravity.CENTER_VERTICAL);
        input = new EditText(this);
        input.setHint("Message");
        input.setHintTextColor(Ui.MUTED);
        input.setTextColor(Ui.TEXT);
        input.setMaxLines(5);
        input.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        input.setImeOptions(EditorInfo.IME_ACTION_SEND);
        input.setBackground(Ui.rounded(this, Ui.CARD, 22));
        input.setPadding(Ui.dp(this, 14), Ui.dp(this, 10), Ui.dp(this, 14), Ui.dp(this, 10));
        bar.addView(input, new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f));
        send = Ui.primary(this, "Send", v -> onSendOrStop());
        LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, Ui.dp(this, 44));
        sp.leftMargin = Ui.dp(this, 8);
        bar.addView(send, sp);
        root.addView(bar);
        input.setOnEditorActionListener((v, actionId, event) -> {
            if (actionId == EditorInfo.IME_ACTION_SEND) {
                onSendOrStop();
                return true;
            }
            return false;
        });
        setContentView(root);

        synchronized (EngineHolder.history) {
            for (String[] turn : EngineHolder.history) {
                addBubble(turn[0], true);
                addBubble(turn[1], false);
            }
        }
        scrollToBottom(true);
    }

    @Override
    protected void onResume() {
        super.onResume();
        EngineHolder.addListener(this);
        onEngineChanged();
    }

    @Override
    protected void onPause() {
        super.onPause();
        EngineHolder.removeListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (EngineHolder.generating && EngineHolder.handle != 0) LiyabNative.cancel(EngineHolder.handle);
    }

    @Override
    public void onEngineChanged() {
        status.setText(EngineHolder.handle != 0 ? EngineHolder.modelLabel + " · " + (EngineHolder.useGpu() ? "GPU" : "CPU")
                + " · " + EngineHolder.template.label : EngineHolder.status);
        send.setEnabled(EngineHolder.handle != 0 && !EngineHolder.busy);
        send.setAlpha(send.isEnabled() ? 1f : 0.4f);
    }

    private void newChat() {
        if (EngineHolder.generating) return;
        synchronized (EngineHolder.history) {
            EngineHolder.history.clear();
        }
        messages.removeAllViews();
        DebugLog.add("New chat: history cleared");
    }

    private TextView addBubble(String text, boolean user) {
        TextView bubble = new TextView(this);
        bubble.setText(text);
        bubble.setTextColor(Ui.TEXT);
        bubble.setTextSize(15);
        bubble.setTextIsSelectable(true);
        bubble.setPadding(Ui.dp(this, 14), Ui.dp(this, 10), Ui.dp(this, 14), Ui.dp(this, 10));
        bubble.setBackground(Ui.rounded(this, user ? Ui.USER_BG : Ui.CARD, 16));
        // Model replies grow while streaming: fixed (full) width so only the height changes.
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(
                user ? ViewGroup.LayoutParams.WRAP_CONTENT : ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT);
        p.gravity = user ? Gravity.END : Gravity.START;
        p.topMargin = Ui.dp(this, 8);
        if (user) p.leftMargin = Ui.dp(this, 48);
        else p.rightMargin = Ui.dp(this, 24);
        if (!user) bubble.setMinHeight(Ui.dp(this, 40));
        messages.addView(bubble, p);
        return bubble;
    }

    private TextView addNote(String text) {
        TextView note = Ui.text(this, text, 11, Ui.MUTED);
        note.setPadding(Ui.dp(this, 4), Ui.dp(this, 2), Ui.dp(this, 4), 0);
        messages.addView(note);
        return note;
    }

    private boolean atBottom() {
        View content = scroll.getChildAt(0);
        return content == null || content.getBottom() - (scroll.getScrollY() + scroll.getHeight()) < Ui.dp(this, 48);
    }

    private void scrollToBottom(boolean force) {
        if (!force && !atBottom()) return;
        scroll.post(() -> scroll.scrollTo(0, Math.max(0, messages.getBottom() - scroll.getHeight())));
    }

    private void onSendOrStop() {
        long engine = EngineHolder.handle;
        if (EngineHolder.generating) {
            if (engine != 0) LiyabNative.cancel(engine);
            return;
        }
        String message = input.getText().toString().trim();
        if (message.isEmpty() || engine == 0) return;
        input.setText("");
        addBubble(message, true);
        TextView reply = addBubble("…", false);
        TextView note = addNote("");
        scrollToBottom(true);
        EngineHolder.generating = true;
        send.setText("Stop");

        final String prompt;
        synchronized (EngineHolder.history) {
            prompt = EngineHolder.template.build(SYSTEM_PROMPT, EngineHolder.history, message);
        }
        final StringBuilder text = new StringBuilder();
        // Tokens arrive faster than the screen refreshes: redraw at most every FRAME_MS.
        final boolean[] pending = {false};
        final Runnable render = () -> {
            boolean follow = atBottom();
            synchronized (text) {
                pending[0] = false;
                reply.setText(text.toString().replaceFirst("^\\s+", ""));
            }
            if (follow) scrollToBottom(true);
        };
        EngineHolder.worker.execute(() -> {
            String error = null;
            double[] stats = null;
            try {
                stats = LiyabNative.generate(engine, prompt, MAX_TOKENS, TEMPERATURE, bytes -> {
                    String piece = new String(bytes, StandardCharsets.UTF_8);
                    synchronized (text) {
                        text.append(piece);
                        if (!pending[0]) {
                            pending[0] = true;
                            ui.postDelayed(render, FRAME_MS);
                        }
                    }
                    return true;
                });
            } catch (RuntimeException e) {
                error = e.getMessage();
            }
            EngineHolder.generating = false;
            final double[] s = stats;
            final String err = error;
            final String answer;
            synchronized (text) {
                answer = text.toString().trim();
            }
            if (s != null) {
                synchronized (EngineHolder.history) {
                    EngineHolder.history.add(new String[] {message, answer});
                    while (EngineHolder.history.size() > MAX_TURNS) EngineHolder.history.remove(0);
                }
                DebugLog.add(String.format(Locale.US, "Generated %d tok at %.1f tok/s, TTFT %.0f ms, prompt %d tok (%d reused)",
                        (int) s[1], s[2], s[3], (int) s[0], (int) s[4]));
            } else {
                DebugLog.add("ERROR: " + err);
            }
            ui.post(() -> {
                ui.removeCallbacks(render);
                render.run();
                send.setText("Send");
                if (err != null) {
                    reply.setText("Error: " + err);
                    return;
                }
                if (answer.isEmpty()) reply.setText("(no answer)");
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
