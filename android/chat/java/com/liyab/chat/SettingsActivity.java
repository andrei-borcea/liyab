package com.liyab.chat;

import android.app.Activity;
import android.os.Bundle;
import android.text.InputType;
import android.view.ViewGroup;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;

import java.util.Locale;

/**
 * Settings of the loaded model: thinking mode, sampling (temperature, top-p, top-k), reply length,
 * context length and system prompt. Saved per model file; "Reset" fills in the defaults (the model
 * file's recommendations when it carries them) for Save. A new context length reloads the model.
 */
public final class SettingsActivity extends Activity {
    private Switch thinking;
    private EditText temperature;
    private EditText topP;
    private EditText topK;
    private EditText maxTokens;
    private EditText context;
    private EditText system;
    private TextView error;
    private EditText memoryBudget;
    private EditText thermalLimit;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        LinearLayout root = Ui.screen(this);
        root.addView(Ui.topBar(this, "Settings", true));
        ScrollView scroll = new ScrollView(this);
        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        scroll.addView(content);
        root.addView(scroll, new LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f));
        setContentView(root);

        // Device-wide settings first: they apply to every model and need no loaded engine.
        LinearLayout device = Ui.card(this);
        device.addView(Ui.text(this, "DEVICE (ALL MODELS)", 11, Ui.MUTED));
        memoryBudget = field(device, "Memory limit (MiB)", String.valueOf(EngineHolder.memoryBudgetMb()), false,
                "Weights + expert cache + streaming buffers. HyperOS / MIUI close any app above 6 GiB, so keep it "
                        + "around 5500 there; raise it on other phones. 0 = use all free RAM.");
        thermalLimit = field(device, "Thermal limit (°C)", fmt(EngineHolder.thermalLimitC()), true,
                "Above this skin/board temperature the engine halves its CPU threads. The OS thermal status "
                        + "(severe, critical) slows it down regardless. Default 50.");
        device.addView(Ui.buttonRow(this, Ui.pill(this, "Save device settings", v -> saveDevice(true))));
        content.addView(device);

        if (EngineHolder.handle == 0) {
            LinearLayout card = Ui.card(this);
            card.addView(Ui.bold(this, "No model loaded", 15));
            card.addView(Ui.text(this, "Model settings belong to a model: load one from Models first.", 13, Ui.MUTED));
            content.addView(card);
            return;
        }
        ModelSettings s = EngineHolder.settings;
        ModelSettings d = ModelSettings.defaults(EngineHolder.handle);

        LinearLayout header = Ui.card(this);
        header.addView(Ui.bold(this, EngineHolder.modelLabel, 14));
        header.addView(Ui.text(this, "Saved for this model file. Template: " + EngineHolder.template.label, 12, Ui.MUTED));
        content.addView(header);

        LinearLayout think = Ui.card(this);
        think.addView(Ui.text(this, "THINKING", 11, Ui.MUTED));
        thinking = new Switch(this);
        thinking.setText("Reason before answering");
        thinking.setTextColor(Ui.TEXT);
        thinking.setChecked(s.thinking);
        thinking.setEnabled(EngineHolder.thinkingSupported);
        think.addView(thinking);
        think.addView(Ui.text(this, EngineHolder.thinkingSupported
                ? "The model writes its reasoning first (shown dimmed above the answer, tap to expand). Better answers "
                        + "on hard questions, but many more tokens: raise Max reply tokens (2048+)."
                : "This model has no thinking mode.", 12, Ui.MUTED));
        content.addView(think);

        LinearLayout sampling = Ui.card(this);
        sampling.addView(Ui.text(this, "SAMPLING", 11, Ui.MUTED));
        temperature = field(sampling, "Temperature", fmt(s.temperature), true,
                String.format(Locale.US, "0 = always the most likely token (deterministic). Default %s (%s).",
                        fmt(d.temperature), d.temperatureSource));
        topP = field(sampling, "Top-p", fmt(s.topP), true,
                String.format(Locale.US, "Keep the smallest set of tokens covering this probability (1 = off). "
                        + "Default %s (%s).", fmt(d.topP), d.topPSource));
        topK = field(sampling, "Top-k", String.valueOf(s.topK), false,
                String.format(Locale.US, "Keep only the k most likely tokens (0 = off). Default %d (%s).", d.topK,
                        d.topKSource));
        content.addView(sampling);

        LinearLayout lengths = Ui.card(this);
        lengths.addView(Ui.text(this, "LENGTHS", 11, Ui.MUTED));
        maxTokens = field(lengths, "Max reply tokens", String.valueOf(s.maxTokens), false,
                "Upper bound per reply, reasoning included. Default " + d.maxTokens + ".");
        context = field(lengths, "Context length (tokens)", String.valueOf(s.contextLength), false,
                "Prompt + history + reply. Changing it reloads the model. Default " + d.contextLength + ".");
        content.addView(lengths);

        LinearLayout prompt = Ui.card(this);
        prompt.addView(Ui.text(this, "SYSTEM PROMPT", 11, Ui.MUTED));
        system = new EditText(this);
        system.setText(s.systemPrompt);
        system.setTextColor(Ui.TEXT);
        system.setTextSize(14);
        system.setMinLines(3);
        system.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_FLAG_MULTI_LINE
                | InputType.TYPE_TEXT_FLAG_CAP_SENTENCES);
        prompt.addView(system);
        content.addView(prompt);

        LinearLayout actions = Ui.card(this);
        error = Ui.text(this, "", 13, Ui.BAD);
        actions.addView(error);
        actions.addView(Ui.buttonRow(this, Ui.primary(this, "Save", v -> save()),
                Ui.pill(this, "Reset to defaults", v -> {
                    apply(ModelSettings.defaults(EngineHolder.handle));
                    error.setTextColor(Ui.MUTED);
                    error.setText("Defaults filled in: Save to keep them.");
                })));
        content.addView(actions);
    }

    private EditText field(LinearLayout parent, String label, String value, boolean decimal, String help) {
        parent.addView(Ui.bold(this, label, 14));
        EditText e = new EditText(this);
        e.setText(value);
        e.setTextColor(Ui.TEXT);
        e.setInputType(InputType.TYPE_CLASS_NUMBER | (decimal ? InputType.TYPE_NUMBER_FLAG_DECIMAL : 0));
        parent.addView(e);
        parent.addView(Ui.text(this, help, 12, Ui.MUTED));
        return e;
    }

    private static String fmt(float v) {
        return String.format(Locale.US, "%.2f", v).replaceFirst("\\.?0+$", "");
    }

    /** Saves the device-wide settings; returns whether they changed (the model must reload to apply them). */
    private boolean saveDevice(boolean announce) {
        long mem;
        float thermal;
        try {
            mem = Long.parseLong(memoryBudget.getText().toString().trim());
            thermal = Float.parseFloat(thermalLimit.getText().toString().trim());
        } catch (NumberFormatException e) {
            android.widget.Toast.makeText(this, "Memory and thermal limits need numbers", android.widget.Toast.LENGTH_SHORT).show();
            return false;
        }
        if (mem < 0 || (mem > 0 && mem < 1024) || thermal < 30 || thermal > 80) {
            android.widget.Toast.makeText(this, "Memory limit: 0 or >= 1024 MiB; thermal limit: 30..80 °C",
                    android.widget.Toast.LENGTH_LONG).show();
            return false;
        }
        boolean changed = mem != EngineHolder.memoryBudgetMb() || thermal != EngineHolder.thermalLimitC();
        EngineHolder.prefs().edit().putLong("memory_budget_mb", mem).putFloat("thermal_limit", thermal).apply();
        if (changed) {
            DebugLog.add(String.format(Locale.US, "Device settings: memory limit %d MiB, thermal limit %.0f °C", mem, thermal));
            if (announce && EngineHolder.handle != 0) {
                DebugLog.add("Reloading the model to apply them");
                EngineHolder.load(EngineHolder.modelFile, EngineHolder.modelUri);
            }
        }
        return changed;
    }

    private void save() {
        final boolean deviceChanged = saveDevice(false);
        ModelSettings s = new ModelSettings();
        try {
            s.thinking = thinking.isChecked();
            s.temperature = Float.parseFloat(temperature.getText().toString().trim());
            s.topP = Float.parseFloat(topP.getText().toString().trim());
            s.topK = Integer.parseInt(topK.getText().toString().trim());
            s.maxTokens = Integer.parseInt(maxTokens.getText().toString().trim());
            s.contextLength = Integer.parseInt(context.getText().toString().trim());
        } catch (NumberFormatException e) {
            error.setTextColor(Ui.BAD);
            error.setText("Every field needs a number.");
            return;
        }
        s.systemPrompt = system.getText().toString().trim();
        String problem = s.temperature < 0 || s.temperature > 2 ? "Temperature must be between 0 and 2."
                : s.topP <= 0 || s.topP > 1 ? "Top-p must be in (0, 1]."
                : s.topK < 0 ? "Top-k cannot be negative."
                : s.maxTokens < 1 || s.maxTokens > 32768 ? "Max reply tokens must be 1..32768."
                : s.contextLength < 256 || s.contextLength > 131072 ? "Context length must be 256..131072."
                : s.maxTokens >= s.contextLength ? "Max reply tokens must be below the context length."
                : null;
        if (problem != null) {
            error.setTextColor(Ui.BAD);
            error.setText(problem);
            return;
        }
        boolean reload = s.contextLength != EngineHolder.settings.contextLength || deviceChanged;
        s.save(EngineHolder.modelKey);
        EngineHolder.settings = ModelSettings.load(EngineHolder.modelKey, EngineHolder.handle);
        DebugLog.add("Settings saved: " + EngineHolder.settings.summary(EngineHolder.thinkingSupported) + ", context "
                + s.contextLength + (reload ? " (reloading the model)" : ""));
        if (reload) EngineHolder.load(EngineHolder.modelFile, EngineHolder.modelUri);
        else EngineHolder.prepareSystemPrompt();  // the system prompt may have changed
        finish();
    }

    private void apply(ModelSettings s) {
        thinking.setChecked(s.thinking);
        temperature.setText(fmt(s.temperature));
        topP.setText(fmt(s.topP));
        topK.setText(String.valueOf(s.topK));
        maxTokens.setText(String.valueOf(s.maxTokens));
        context.setText(String.valueOf(s.contextLength));
        system.setText(s.systemPrompt);
    }
}
