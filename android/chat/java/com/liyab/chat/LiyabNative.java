package com.liyab.chat;

/** JNI entry points into libliyab (see android/chat/jni/liyab_jni.cpp). */
final class LiyabNative {
    static {
        System.loadLibrary("liyab");
        System.loadLibrary("liyab_chat");
    }

    /** Receives complete UTF-8 pieces as they are generated; return false to stop. */
    interface PieceCallback {
        boolean onPiece(byte[] utf8);
    }

    private LiyabNative() {}

    /** Routes engine diagnostics (0 debug .. 4 off) into a buffer drained by takeLogs(). */
    static native void enableLogs(int level);

    /** Returns and clears the buffered engine log lines ("I message\n"). */
    static native String takeLogs();

    static final int BACKEND_CPU = 0;     // LIYAB_BACKEND_CPU
    static final int BACKEND_VULKAN = 2;  // LIYAB_BACKEND_VULKAN

    /**
     * Loads the model on `backend` with a `contextLength`-token context (0: engine default) and a
     * thermal guard that halves the CPU threads above `skinThresholdC` (the OS thermal status applies
     * regardless), keeping weights and caches within `memoryBudgetMb` (0: free RAM only); throws
     * RuntimeException with Liyab's error message on failure.
     */
    static native long create(String modelPath, String cacheDir, int threads, int backend, int contextLength,
                              float skinThresholdC, long memoryBudgetMb);

    /**
     * Blocking generation on the calling thread. Returns {prompt_tokens, generated_tokens,
     * tokens_per_second, ttft_ms, cached_prefix_tokens, decode_ms, thermal_reroutes, cancelled}.
     */
    static native double[] generate(long engine, String prompt, int maxTokens, float temperature, float topP, int topK,
                                    PieceCallback cb);

    /** Thread-safe: stops an in-flight generate() at the next token. */
    static native void cancel(long engine);

    static native void destroy(long engine);

    static native String describe(long engine);

    /**
     * Processes `text` into the context without generating (e.g. the system prompt right after
     * loading); a later generate() whose prompt starts with it skips that work. Blocking.
     */
    static native void prefill(long engine, String text);

    /**
     * Live cumulative counters {gpu_busy_ms, storage_bytes_read, tokens_generated}, or null. Safe while a
     * generation runs; the caller must keep the engine alive (EngineHolder.LIFECYCLE).
     */
    static native double[] counters(long engine);

    /** GGUF architectures the engine runs, comma-separated (e.g. "llama,qwen2,..."). */
    static native String supportedArchitectures();

    /** A scalar GGUF metadata value of the loaded model as text (e.g. "general.sampling.temp"), or null. */
    static native String metadata(long engine, String key);

    /** Number of tokens `text` encodes to (no BOS), or -1 on error. */
    static native int countTokens(long engine, String text);
}
