package com.liyab.chat;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.Locale;

/**
 * Generation settings of one model, stored per model file name. A value the user never changed
 * falls back to the publisher's recommendation embedded in the GGUF (general.sampling.*), then to
 * the app default; `source` says which one applies, for the settings screen.
 */
final class ModelSettings {
    static final String DEFAULT_SYSTEM_PROMPT =
            "You are Liyab, a helpful assistant running entirely on this phone, without internet. "
                    + "Answer clearly and concisely.";
    static final float DEFAULT_TEMPERATURE = 0.7f;
    static final float DEFAULT_TOP_P = 0.9f;
    static final int DEFAULT_TOP_K = 40;
    static final int DEFAULT_MAX_TOKENS = 1024;
    static final int DEFAULT_CONTEXT = 4096;

    boolean thinking;  // only used when the model has a thinking mode
    float temperature;
    float topP;
    int topK;
    int maxTokens;
    int contextLength;  // takes effect on the next load
    String systemPrompt;

    /** Where the defaults come from ("model file" or "app"), per sampling field. */
    String temperatureSource = "app";
    String topPSource = "app";
    String topKSource = "app";

    private static String key(String model) {
        return "settings:" + model;
    }

    /** Defaults for the loaded `engine`: the GGUF's sampling recommendations, else app values. */
    static ModelSettings defaults(long engine) {
        ModelSettings s = new ModelSettings();
        s.temperature = DEFAULT_TEMPERATURE;
        s.topP = DEFAULT_TOP_P;
        s.topK = DEFAULT_TOP_K;
        s.maxTokens = DEFAULT_MAX_TOKENS;
        s.contextLength = DEFAULT_CONTEXT;
        s.systemPrompt = DEFAULT_SYSTEM_PROMPT;
        s.thinking = false;
        if (engine != 0) {
            Float t = number(LiyabNative.metadata(engine, "general.sampling.temp"));
            Float p = number(LiyabNative.metadata(engine, "general.sampling.top_p"));
            Float k = number(LiyabNative.metadata(engine, "general.sampling.top_k"));
            if (t != null) {
                s.temperature = t;
                s.temperatureSource = "model file";
            }
            if (p != null) {
                s.topP = p;
                s.topPSource = "model file";
            }
            if (k != null) {
                s.topK = Math.round(k);
                s.topKSource = "model file";
            }
        }
        return s;
    }

    /** The settings of `model`: defaults overlaid with what the user saved. */
    static ModelSettings load(String model, long engine) {
        ModelSettings s = defaults(engine);
        String json = EngineHolder.prefs().getString(key(model), null);
        if (json == null) return s;
        try {
            JSONObject o = new JSONObject(json);
            s.thinking = o.optBoolean("thinking", s.thinking);
            s.temperature = (float) o.optDouble("temperature", s.temperature);
            s.topP = (float) o.optDouble("top_p", s.topP);
            s.topK = o.optInt("top_k", s.topK);
            s.maxTokens = o.optInt("max_tokens", s.maxTokens);
            s.contextLength = o.optInt("context", s.contextLength);
            s.systemPrompt = o.optString("system", s.systemPrompt);
        } catch (JSONException e) {
            DebugLog.add("Ignoring unreadable settings of " + model);
        }
        return s;
    }

    /** The context length to load `model` with (saved value or the default), before an engine exists. */
    static int contextFor(String model) {
        String json = EngineHolder.prefs().getString(key(model), null);
        if (json == null) return DEFAULT_CONTEXT;
        try {
            return new JSONObject(json).optInt("context", DEFAULT_CONTEXT);
        } catch (JSONException e) {
            return DEFAULT_CONTEXT;
        }
    }

    void save(String model) {
        try {
            JSONObject o = new JSONObject();
            o.put("thinking", thinking);
            o.put("temperature", temperature);
            o.put("top_p", topP);
            o.put("top_k", topK);
            o.put("max_tokens", maxTokens);
            o.put("context", contextLength);
            o.put("system", systemPrompt);
            EngineHolder.prefs().edit().putString(key(model), o.toString()).apply();
        } catch (JSONException e) {
            throw new IllegalStateException(e);  // only finite numbers and strings are put
        }
    }

    /** One-line summary for the chat header and the log. */
    String summary(boolean thinkingSupported) {
        return String.format(Locale.US, "%stemp %.2f · top-p %.2f · top-k %d · max %d tok",
                thinkingSupported ? (thinking ? "thinking on · " : "thinking off · ") : "", temperature, topP, topK,
                maxTokens);
    }

    private static Float number(String s) {
        if (s == null) return null;
        try {
            float v = Float.parseFloat(s);
            return Float.isFinite(v) ? v : null;
        } catch (NumberFormatException e) {
            return null;
        }
    }
}
