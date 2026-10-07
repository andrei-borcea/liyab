/*
 * Liyab — stable C ABI for Android (JNI / Kotlin) and iOS (Swift).
 *
 * Conventions
 *  - Every function is safe to call with NULL handles (it then fails with
 *    LIYAB_ERR_INVALID_ARGUMENT); no C++ exception crosses this boundary.
 *  - On failure, liyab_last_error() returns a description of the most recent
 *    error on the calling thread. The pointer stays valid until the next
 *    Liyab call on that thread.
 *  - Structs are initialized with their *_default() function before use, so
 *    new fields can be appended without breaking callers.
 *  - Strings passed in are UTF-8 and only borrowed for the call's duration.
 */
#ifndef LIYAB_C_API_H
#define LIYAB_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define LIYAB_C_API __declspec(dllexport)
#else
#define LIYAB_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct liyab_engine liyab_engine;

typedef enum liyab_status {
    LIYAB_OK = 0,
    LIYAB_ERR_INVALID_ARGUMENT = 1,
    LIYAB_ERR_IO = 2,
    LIYAB_ERR_INVALID_MODEL = 3,
    LIYAB_ERR_UNSUPPORTED = 4,
    LIYAB_ERR_OUT_OF_MEMORY = 5,
    LIYAB_ERR_CONTEXT_FULL = 6,
    LIYAB_ERR_CANCELLED = 7,
    LIYAB_ERR_BACKEND = 8,
    LIYAB_ERR_BUSY = 9,
    LIYAB_ERR_INTERNAL = 10
} liyab_status;

typedef enum liyab_backend {
    LIYAB_BACKEND_AUTO = -1, /* device ranking: NPU -> GPU -> CPU */
    LIYAB_BACKEND_CPU = 0,
    LIYAB_BACKEND_METAL = 1,
    LIYAB_BACKEND_VULKAN = 2,
    LIYAB_BACKEND_QNN = 3,
    LIYAB_BACKEND_NEUROPILOT = 4
} liyab_backend;

typedef enum liyab_power_profile {
    LIYAB_POWER_PERFORMANCE = 0,
    LIYAB_POWER_BALANCED = 1,
    LIYAB_POWER_LOW_POWER = 2
} liyab_power_profile;

typedef enum liyab_kv_cache_type {
    LIYAB_KV_F16 = 0,
    LIYAB_KV_Q8_0 = 1,
    LIYAB_KV_Q4_0 = 2, /* symmetric INT4 */
    LIYAB_KV_Q4_1 = 3  /* asymmetric INT4 */
} liyab_kv_cache_type;

typedef struct liyab_engine_config {
    const char* model_path;       /* required: GGUF file */
    const char* draft_model_path; /* optional (NULL): enables speculative decoding */
    int32_t n_threads;            /* 0: performance cores */
    int32_t context_length;       /* 0: min(model context, 4096) */
    int32_t sliding_window;       /* 0: full attention */
    int32_t kv_sink_tokens;       /* attention sinks kept with a sliding window (default 8) */
    int32_t draft_tokens;         /* tokens proposed per speculative step */
    liyab_kv_cache_type kv_cache_type;
    liyab_backend backend;
    liyab_power_profile power_profile;
    float skin_threshold_c;       /* thermal throttle threshold, default 40 */
    double target_tps;            /* > 0 overrides the profile's pacing rate */
    int32_t thermal_polling;      /* nonzero: background thermal sampling */
    int32_t triple_buffer_loading; /* nonzero: stream blocks through 3 rotating buffers */
    /* Experimental (build with LIYAB_ENABLE_EXPERIMENTAL=ON, else LIYAB_ERR_UNSUPPORTED). */
    int32_t early_exit;           /* nonzero: confidence-based early exit (lossy) */
    float early_exit_threshold;   /* default 0.98 */
    int32_t head_pruning;         /* nonzero: prune attention heads when hot / low power (lossy) */
    float head_keep_ratio;        /* default 0.75 */
    int32_t egls;                 /* nonzero: entropy-guided FFN skipping (lossy) */
    float egls_threshold;         /* default 0.002 */
    int32_t tdss;                 /* 1: 2:4 sparse FFN while throttled, 2: always (lossy) */
    const char* kv_dedup_dir;     /* non-NULL: persistent prefix KV cache directory */
} liyab_engine_config;

typedef struct liyab_sampling_params {
    float temperature; /* <= 0: greedy */
    int32_t top_k;     /* <= 0: disabled */
    float top_p;       /* >= 1: disabled */
    uint64_t seed;     /* 0: random */
    int32_t max_tokens;
    int32_t add_bos;   /* nonzero: prepend BOS when the model uses one */
} liyab_sampling_params;

typedef struct liyab_generation_stats {
    int32_t prompt_tokens;
    int32_t generated_tokens;
    double prefill_ms;
    double decode_ms;
    double tokens_per_second;
    int32_t draft_tokens_proposed;
    int32_t draft_tokens_accepted;
    int32_t thermal_reroutes;
    double paced_idle_ms;
    int32_t cancelled;
    int32_t early_exits;
    int32_t early_exit_layers_skipped;
    int32_t head_pruned_steps;
    int32_t weight_stalls;
    double weight_wait_ms;
    uint64_t kv_cache_bytes;
    int32_t ffn_blocks_skipped;
    int32_t sparse_ffn_steps;
    double ttft_ms;
    int32_t cached_prefix_tokens;
} liyab_generation_stats;

/*
 * Streaming callback: `piece` holds `piece_len` bytes of complete UTF-8 (not
 * NUL-terminated); `token` is the token that completed it. Return nonzero to
 * continue, 0 to stop. Invoked on the thread that called liyab_engine_generate.
 */
typedef int32_t (*liyab_token_callback)(const char* piece, size_t piece_len, int32_t token, void* user_data);

LIYAB_C_API const char* liyab_version(void);
LIYAB_C_API const char* liyab_last_error(void);

LIYAB_C_API void liyab_engine_config_default(liyab_engine_config* config);
LIYAB_C_API void liyab_sampling_params_default(liyab_sampling_params* params);

/* Loads the model(s) and selects backends. *out_engine is set on success. */
LIYAB_C_API liyab_status liyab_engine_create(const liyab_engine_config* config, liyab_engine** out_engine);
LIYAB_C_API void liyab_engine_destroy(liyab_engine* engine);

/* Blocking generation; `stats` may be NULL. Cancellation is not an error:
 * it returns LIYAB_OK with stats->cancelled = 1. */
LIYAB_C_API liyab_status liyab_engine_generate(liyab_engine* engine, const char* prompt,
                                               const liyab_sampling_params* params, liyab_token_callback callback,
                                               void* user_data, liyab_generation_stats* stats);
LIYAB_C_API liyab_status liyab_engine_generate_tokens(liyab_engine* engine, const int32_t* tokens, size_t n_tokens,
                                                      const liyab_sampling_params* params,
                                                      liyab_token_callback callback, void* user_data,
                                                      liyab_generation_stats* stats);

/* Thread-safe; stops an in-flight generation at the next token boundary. */
LIYAB_C_API void liyab_engine_cancel(liyab_engine* engine);

LIYAB_C_API liyab_status liyab_engine_set_power_profile(liyab_engine* engine, liyab_power_profile profile);

/*
 * Writes a NUL-terminated, human-readable device/model summary into `buffer`
 * (truncated to `size`). Returns the length the full text needs, excluding NUL.
 */
LIYAB_C_API size_t liyab_engine_describe(const liyab_engine* engine, char* buffer, size_t size);
/* Same for the device only; no engine required. */
LIYAB_C_API size_t liyab_describe_device(char* buffer, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* LIYAB_C_API_H */
