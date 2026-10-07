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

    /** Loads the model; throws RuntimeException with Liyab's error message on failure. */
    static native long create(String modelPath, String cacheDir, int threads);

    /**
     * Blocking generation on the calling thread. Returns {prompt_tokens, generated_tokens,
     * tokens_per_second, ttft_ms, cached_prefix_tokens, decode_ms, thermal_reroutes, cancelled}.
     */
    static native double[] generate(long engine, String prompt, int maxTokens, float temperature, PieceCallback cb);

    /** Thread-safe: stops an in-flight generate() at the next token. */
    static native void cancel(long engine);

    static native void destroy(long engine);

    static native String describe(long engine);
}
