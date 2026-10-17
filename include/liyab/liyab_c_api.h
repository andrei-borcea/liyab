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

/* Windows: dllexport while building liyab.dll (LIYAB_DLL_EXPORT), dllimport
 * for its users that define LIYAB_DLL_IMPORT (the CMake target `liyab` does);
 * loading liyab.dll at run time (FFI, LoadLibrary) needs neither. */
#if defined(_WIN32)
#if defined(LIYAB_DLL_EXPORT)
#define LIYAB_C_API __declspec(dllexport)
#elif defined(LIYAB_DLL_IMPORT)
#define LIYAB_C_API __declspec(dllimport)
#else
#define LIYAB_C_API
#endif
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
    int32_t n_threads;            /* 0: performance cores, minus one when there is no efficiency cluster */
    int32_t context_length;       /* 0: automatic, what 256 MiB of KV holds (4096..32768), at most the model's */
    int32_t sliding_window;       /* 0: full attention */
    int32_t kv_sink_tokens;       /* attention sinks kept with a sliding window (default 8) */
    int32_t draft_tokens;         /* tokens proposed per speculative step */
    int32_t lookup_drafts;        /* nonzero, no draft model: draft from the text itself (suffix drafts or context lookup, see suffix_drafts) */
    int32_t adaptive_drafts;      /* nonzero (default): learn how many drafts per step pay off; 0: always draft_tokens */
    liyab_kv_cache_type kv_cache_type;
    liyab_backend backend;
    liyab_power_profile power_profile;
    float skin_threshold_c;       /* thermal throttle threshold, default 40 */
    double target_tps;            /* > 0 overrides the profile's pacing rate */
    int32_t thermal_polling;      /* nonzero: background thermal sampling while prefilling or generating */
    int32_t triple_buffer_loading; /* nonzero: stream blocks through 3 rotating buffers */
    int64_t expert_cache_mb;      /* MoE: -1 auto (default), 0 never stream experts, > 0 cache size in MiB */
    int64_t memory_budget_mb;     /* > 0: cap on resident memory in MiB (e.g. per-app OS limits); 0: free RAM only */
    int32_t requant_bits;         /* MoE expert streaming: resident Q8_0 matrices -> Q4_K (4) / Q5_K (5) at load (lossy); 0 off, -1 only when memory is tight */
    float moe_expert_mass;        /* MoE: run the fewest top experts covering this router mass (lossy, e.g. 0.9); >= 1 all (default 1) */
    int32_t moe_max_experts;      /* MoE: at most this many experts per token (lossy); 0: the model's top-k */
    /* Experimental (build with LIYAB_ENABLE_EXPERIMENTAL=ON, else LIYAB_ERR_UNSUPPORTED). */
    int32_t early_exit;           /* nonzero: confidence-based early exit (lossy) */
    float early_exit_threshold;   /* default 0.98 */
    int32_t head_pruning;         /* nonzero: prune attention heads when hot / low power (lossy) */
    float head_keep_ratio;        /* default 0.75 */
    int32_t egls;                 /* nonzero: entropy-guided FFN skipping (lossy) */
    float egls_threshold;         /* default 0.002 */
    int32_t tdss;                 /* 1: 2:4 sparse FFN while throttled, 2: always (lossy) */
    const char* kv_dedup_dir;     /* retired (KV dedup; the prefix cache's store replaces it): ignored, except that the snapshot files it left in this directory are removed; kept for the struct's layout */
    float moe_skip_slow;          /* MoE streaming: skip a chosen expert not yet in RAM whose router weight share is below this for every token using it (lossy; output then depends on I/O timing); 0 off */
    int64_t prefix_cache_mb;      /* RAM at most for contexts other conversations replaced, restored when a prompt continues one (a store of KV chunks: shared prefixes kept once); 0 off (see EngineConfig) */
    const char* prefix_cache_dir; /* non-NULL with prefix_cache_disk_mb > 0: what RAM does not hold goes to files here, which outlive the engine (it sets its context aside there when destroyed) and later engines restore from */
    int64_t prefix_cache_disk_mb; /* storage at most for them; 0 off */
    const char* cache_dir;        /* non-NULL: what the engine measures once per model is kept here (today the key scales of a quantized KV cache, a few KB per model); NULL: measured at every load */
    int32_t suffix_drafts;        /* with lookup_drafts: nonzero (the default) drafts from the suffix drafter (this and earlier conversations of the engine, ~2 MiB at most), 0 by context lookup (see EngineConfig); a zero-filled config without liyab_engine_config_default gets context lookup */
    liyab_kv_cache_type pooling_kv_cache_type; /* KV cache of a pooling model (embeddings, reranker: its GGUF declares a pooling_type), instead of kv_cache_type; LIYAB_KV_F16 by default (see EngineConfig) */
    int32_t prefix_cache_lossy;   /* nonzero: prefix-cache chunks on storage re-quantized to 4 bits instead of compressed losslessly (lossy; 0 default, see EngineConfig) */
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
    /* MoE expert streaming (zero when experts are read in place) */
    int32_t expert_hits;          /* routed experts already cached in RAM */
    int32_t expert_late;          /* prefetched but still loading when needed */
    int32_t expert_misses;        /* read after the router chose them */
    uint64_t expert_bytes_read;
    double expert_stall_ms;       /* time spent waiting for expert reads */
    int32_t expert_unused;        /* experts read and evicted again without being used */
    /* Decode time by phase, ms (decode steps only; the rest of decode_ms is
     * norms, residuals, sampling, callbacks and pacing) */
    double attention_ms;          /* softmax-attention mixers */
    double delta_net_ms;          /* Gated DeltaNet mixers */
    double router_ms;             /* MoE routers and expert prediction */
    double experts_ms;            /* routed experts, expert_stall_ms included */
    double shared_expert_ms;      /* MoE shared experts */
    double dense_ffn_ms;          /* dense FFNs */
    double lm_head_ms;            /* output projection */
    /* MoE expert prediction (one block ahead): precision = used / predicted */
    int32_t expert_predicted;     /* experts guessed ahead of the router, cached ones included */
    int32_t expert_predicted_used; /* guesses the router then chose */
    int32_t expert_dropped;       /* wrong guesses removed from the read queue before being read */
    int32_t expert_skipped;       /* chosen experts skipped instead of waited for (moe_skip_slow) */
    int32_t forced_tokens;        /* tokens of forced continuations (liyab_engine_generate_forced) */
    double forced_ms;             /* time of the batched passes over them (part of decode_ms) */
    /* Before ttft_ms: matching the context, a prefix-cache restore, a rewind */
    double context_ms;
    double ttft_stall_ms;         /* of ttft_ms: waiting for expert reads */
} liyab_generation_stats;

/*
 * Streaming callback: `piece` holds `piece_len` bytes of complete UTF-8 (not
 * NUL-terminated); `token` is the token that completed it. Return nonzero to
 * continue, 0 to stop. Invoked on the thread that called liyab_engine_generate.
 */
typedef int32_t (*liyab_token_callback)(const char* piece, size_t piece_len, int32_t token, void* user_data);

LIYAB_C_API const char* liyab_version(void);

/*
 * Diagnostics. Levels: 0 debug, 1 info, 2 warn (default), 3 error, 4 off.
 * The callback receives every message at or above the level, from any thread
 * (load, prefetch and worker threads included); NULL removes it. Messages also
 * go to logcat (Android) or stderr.
 */
typedef void (*liyab_log_callback)(int32_t level, const char* message, void* user_data);
LIYAB_C_API void liyab_set_log_level(int32_t level);
LIYAB_C_API void liyab_set_log_callback(liyab_log_callback callback, void* user_data);
LIYAB_C_API const char* liyab_last_error(void);
/*
 * Log buffer for UIs that poll instead of taking callbacks (e.g. Dart, whose
 * asynchronous callbacks cannot read `message` after the call returns).
 * liyab_log_buffer_enable(max_bytes) installs a sink that appends each message
 * as "<level letter> <message>\n" (D, I, W, E), keeping at most `max_bytes`
 * (oldest lines dropped); it replaces any callback, and 0 removes it.
 * liyab_log_buffer_take copies up to size - 1 bytes of whole lines into
 * `buffer` (NUL-terminated), removes them, and returns the bytes copied; 0
 * when empty. A single line longer than the buffer is cut. Thread-safe.
 */
LIYAB_C_API void liyab_log_buffer_enable(size_t max_bytes);
LIYAB_C_API size_t liyab_log_buffer_take(char* buffer, size_t size);

LIYAB_C_API void liyab_engine_config_default(liyab_engine_config* config);
LIYAB_C_API void liyab_sampling_params_default(liyab_sampling_params* params);

/* Loads the model(s) and selects backends. *out_engine is set on success. */
LIYAB_C_API liyab_status liyab_engine_create(const liyab_engine_config* config, liyab_engine** out_engine);
LIYAB_C_API void liyab_engine_destroy(liyab_engine* engine);

/* Structured output (liyab_engine_generate_forced): called after each piece
 * with the output so far; writes up to out_cap bytes the output must continue
 * with into `out` and returns their length (0: none). The engine runs that
 * text through the model in one batched pass and emits it through `callback`
 * like generated text. See Engine's ForceCallback. */
typedef size_t (*liyab_force_callback)(const char* generated, size_t generated_len, char* out, size_t out_cap,
                                       void* user_data);

/* Blocking generation; `stats` may be NULL. Cancellation is not an error:
 * it returns LIYAB_OK with stats->cancelled = 1. On the CPU backend greedy
 * output is reproducible: it does not depend on how the prompt was split,
 * the threads, the expert cache or I/O timing (see Engine::generate). */
LIYAB_C_API liyab_status liyab_engine_generate(liyab_engine* engine, const char* prompt,
                                               const liyab_sampling_params* params, liyab_token_callback callback,
                                               void* user_data, liyab_generation_stats* stats);
/* liyab_engine_generate with forced continuations (`force` may be NULL). */
LIYAB_C_API liyab_status liyab_engine_generate_forced(liyab_engine* engine, const char* prompt,
                                                      const liyab_sampling_params* params,
                                                      liyab_token_callback callback, liyab_force_callback force,
                                                      void* user_data, liyab_generation_stats* stats);
LIYAB_C_API liyab_status liyab_engine_generate_tokens(liyab_engine* engine, const int32_t* tokens, size_t n_tokens,
                                                      const liyab_sampling_params* params,
                                                      liyab_token_callback callback, void* user_data,
                                                      liyab_generation_stats* stats);

/*
 * Tokenizes `text` (UTF-8). Writes up to `capacity` ids into `out` (may be
 * NULL with capacity 0) and returns the total number of tokens, or -1 on
 * error (see liyab_last_error()).
 */
LIYAB_C_API int32_t liyab_engine_tokenize(const liyab_engine* engine, const char* text, int32_t add_bos, int32_t* out,
                                          int32_t capacity);

/* Thread-safe; stops an in-flight generation at the next token boundary. */
LIYAB_C_API void liyab_engine_cancel(liyab_engine* engine);

LIYAB_C_API liyab_status liyab_engine_set_power_profile(liyab_engine* engine, liyab_power_profile profile);

/*
 * Writes a NUL-terminated, human-readable device/model summary into `buffer`
 * (truncated to `size`). Returns the length the full text needs, excluding NUL.
 */
LIYAB_C_API size_t liyab_engine_describe(const liyab_engine* engine, char* buffer, size_t size);
/*
 * Copies the loaded model's GGUF metadata value `key` as NUL-terminated text
 * (strings as-is, numbers in decimal, booleans "true"/"false") into `buffer`
 * (truncated to `size`). Returns the length the full text needs, excluding
 * NUL, or -1 when the key is absent or holds an array.
 */
LIYAB_C_API int64_t liyab_engine_metadata(const liyab_engine* engine, const char* key, char* buffer, size_t size);
/* Same as liyab_engine_describe for the device only; no engine required. */
LIYAB_C_API size_t liyab_describe_device(char* buffer, size_t size);
/* Processes `text` into the context without generating, so a later generate
 * whose prompt starts with it skips that work (e.g. a chat's system prompt,
 * prepared right after loading). generate reuses the context whenever its
 * prompt continues the tokens already processed. */
LIYAB_C_API liyab_status liyab_engine_prefill(liyab_engine* engine, const char* text, int32_t add_bos);
/* Thread-safe: stops an in-flight liyab_engine_prefill between two blocks of
 * the model; it returns LIYAB_ERR_CANCELLED and keeps the context processed so
 * far. Never affects liyab_engine_generate. See Engine::cancel_prefill. */
LIYAB_C_API void liyab_engine_cancel_prefill(liyab_engine* engine);
/* Forgets the context: the next call starts from scratch (or from what the
 * prefix cache keeps of its prompt). */
LIYAB_C_API liyab_status liyab_engine_reset_context(liyab_engine* engine);
/* Saves the context (e.g. a prefilled system prompt) to `path`, and restores
 * it for the same model file loaded with the same cache settings, so a later
 * prompt continuing its text starts at once (on hybrid models with its state
 * snapshots, so a diverging prompt keeps as much). load writes the tokens restored
 * to *out_tokens (may be NULL). See Engine::save_state / load_state. */
LIYAB_C_API liyab_status liyab_engine_save_state(liyab_engine* engine, const char* path);
LIYAB_C_API liyab_status liyab_engine_load_state(liyab_engine* engine, const char* path, int32_t* out_tokens);
/* Saves the context that the `text_len` bytes of `text` continue (a
 * conversation's text so far): the live one, or the longest one the prefix
 * cache set aside for it, written from its store as kept, so a conversation
 * another prompt replaced can still be saved. LIYAB_ERR_INVALID_ARGUMENT when
 * none does. See Engine::save_state. */
LIYAB_C_API liyab_status liyab_engine_save_state_of(liyab_engine* engine, const char* path, const char* text,
                                                    size_t text_len);
/* Gives memory back to the OS while the engine is idle (an app in the
 * background): empties the MoE expert cache (and constrained decoding's
 * caches); the model and its context stay
 * loaded, and the next generation refills the cache from storage. Writes the
 * bytes released to *out_bytes (may be NULL; 0 for models without streamed
 * experts). Not while a generation runs on the engine. See Engine::trim_memory. */
LIYAB_C_API liyab_status liyab_engine_trim_memory(liyab_engine* engine, uint64_t* out_bytes);

/* How a MoE model with streamed experts uses memory (all zero when experts
 * are read in place): an app can tell the user that its memory budget is too
 * small for the model, and how much would let it run well. */
typedef struct liyab_memory_plan {
    uint64_t resident_bytes;     /* weights kept in RAM (after any requantization) */
    uint64_t expert_bytes;       /* routed experts read from storage */
    uint64_t expert_cache_bytes; /* RAM caching experts */
    uint64_t recommended_bytes;  /* memory budget at which most expert uses would come from RAM */
    int32_t requant_bits;        /* resident Q8_0 matrices converted at load (4 / 5); 0: none */
} liyab_memory_plan;
LIYAB_C_API liyab_status liyab_engine_memory_plan(const liyab_engine* engine, liyab_memory_plan* out);
/* The context length the engine was created with (resolved when the config asked for 0). */
LIYAB_C_API int32_t liyab_engine_context_length(const liyab_engine* engine);

/* The expert cache's hot list (MoE with streamed experts; no-ops otherwise),
 * recorded by liyab_engine_trim_memory: warm_memory queues those experts
 * behind any demand read so the cache refills in the background once the
 * model is used again (*out_queued may be NULL); save / load keep the list in
 * a small file for a later process (load fails for another model). Not while
 * a generation runs on the engine. See Engine::warm_memory. */
LIYAB_C_API liyab_status liyab_engine_warm_memory(liyab_engine* engine, uint32_t* out_queued);
LIYAB_C_API liyab_status liyab_engine_save_expert_profile(liyab_engine* engine, const char* path);
LIYAB_C_API liyab_status liyab_engine_load_expert_profile(liyab_engine* engine, const char* path);

/* Live counters for monitoring UIs, cumulative since creation: sample them
 * periodically and divide deltas by the elapsed time. Thread-safe (may be
 * called while a generation runs). */
typedef struct liyab_engine_counters {
    double accelerator_busy_ms;  /* GPU time spent on this engine's work (0: none / not tracked) */
    uint64_t storage_bytes_read; /* weights streamed from storage */
    uint64_t tokens_generated;   /* decoded tokens, all generations */
} liyab_engine_counters;
LIYAB_C_API liyab_status liyab_engine_get_counters(const liyab_engine* engine, liyab_engine_counters* out);
/* GGUF `general.architecture` values this build runs, comma-separated; same
 * buffer contract as liyab_engine_describe. */
LIYAB_C_API size_t liyab_supported_architectures(char* buffer, size_t size);

/*
 * Chat template: the loaded model's tokenizer.chat_template, rendered as
 * transformers' apply_chat_template renders it (Engine::render_chat).
 * `request_json` is a JSON object: "messages" (role / content objects),
 * optional "tools", optional "add_generation_prompt" (default false), any
 * other key a template variable (e.g. "enable_thinking"). Writes the
 * NUL-terminated prompt into `buffer` (truncated to `size`) and returns the
 * length the full text needs, excluding NUL; -1 on failure (no template, bad
 * JSON, or the template refused the conversation: liyab_last_error says
 * which). Thread-safe; does not touch the context.
 */
LIYAB_C_API int64_t liyab_engine_render_chat(const liyab_engine* engine, const char* request_json, char* buffer,
                                             size_t size);
/*
 * How the template lays out a conversation (Engine::chat_format), as a JSON
 * object: {"prefix_stable": bool, "reason", "system_prefix", "system_suffix",
 * "first_user_prefix", "no_system_head", "user_prefix", "user_suffix",
 * "generation_prompt", "assistant_end"} (strings). `probe_json`: template
 * variables to render with plus an optional "reply_prefix" (NULL or "" for
 * none). Same buffer contract as liyab_engine_render_chat; -1 when the model
 * has no usable template.
 */
LIYAB_C_API int64_t liyab_engine_chat_format(const liyab_engine* engine, const char* probe_json, char* buffer,
                                             size_t size);

/*
 * Constrained decoding (structured output; liyab/grammar.h has the JSON
 * Schema subset and the grammar syntax). While a grammar is active, tokens
 * it cannot accept are masked before sampling, so the output matches it.
 *
 * A grammar is independent of engines: compile it once, use it with any
 * engine and from any thread, destroy it when no generation still needs it
 * (an engine keeps its own reference while it uses one).
 */
typedef struct liyab_grammar liyab_grammar;

/* A JSON value valid against `schema_json`. any_whitespace 0: json.dumps'
 * layout (", " and ": " only), which lets every character that is the only
 * one possible be forced; 1: any JSON white space. On failure *out_grammar
 * is NULL and liyab_last_error() says why (LIYAB_ERR_UNSUPPORTED names a
 * keyword Liyab does not enforce, and where it is). */
LIYAB_C_API liyab_status liyab_grammar_from_json_schema(const char* schema_json, int32_t any_whitespace,
                                                        liyab_grammar** out_grammar);
/* Grammar text (GBNF-like rules from `root`, with %json, %schema and %until). */
LIYAB_C_API liyab_status liyab_grammar_parse(const char* text, liyab_grammar** out_grammar);
LIYAB_C_API void liyab_grammar_destroy(liyab_grammar* grammar);
/* 1 when the `len` bytes of `text` are a complete match, else 0. */
LIYAB_C_API int32_t liyab_grammar_matches(const liyab_grammar* grammar, const char* text, size_t len);

/* Called with the output so far before the first token and, while no
 * constrained span is active, after every token that ends a whole
 * character. Writes up to out_cap (4096) bytes of text the output must
 * continue with into `out` and returns their length (0: none); may set
 * *out_grammar to a grammar the output continues with after that text (a
 * span, which ends when its match is complete and nothing can follow; then
 * the callback is asked again). See Engine's ContinuationCallback. */
typedef size_t (*liyab_continuation_callback)(const char* generated, size_t generated_len, char* out, size_t out_cap,
                                              const liyab_grammar** out_grammar, void* user_data);

typedef struct liyab_grammar_stats {
    int32_t constrained_tokens; /* tokens sampled under a grammar (forced ones: stats->forced_tokens) */
    double grammar_ms;          /* masks and grammar bookkeeping, part of decode_ms */
} liyab_grammar_stats;

/* Generation with constrained decoding. `grammar` (may be NULL) constrains
 * the whole output: generation ends once its match is complete and nothing
 * can follow, or at an end-of-generation token where it accepts.
 * `continuation` (may be NULL) forces text and opens constrained spans at
 * points it chooses. At most one of the two (both: LIYAB_ERR_INVALID_ARGUMENT);
 * neither: liyab_engine_generate. Speculative decoding is off while a grammar
 * is active. `stats` and `grammar_stats` may be NULL. */
LIYAB_C_API liyab_status liyab_engine_generate_structured(liyab_engine* engine, const char* prompt,
                                                          const liyab_sampling_params* params,
                                                          liyab_token_callback callback, const liyab_grammar* grammar,
                                                          liyab_continuation_callback continuation, void* user_data,
                                                          liyab_generation_stats* stats,
                                                          liyab_grammar_stats* grammar_stats);

/*
 * Pooling models: embeddings and reranking (Engine::embed / rerank and
 * Pooling in liyab/engine.h have the whole contract). An embedding model
 * (Qwen3-Embedding) turns each text into one vector, a reranker
 * (Qwen3-Reranker) scores each document against a query; what a model does
 * comes from its GGUF (pooling_type, add_eos_token, ranking head, rerank
 * prompt). An engine serves one call at a time (LIYAB_ERR_BUSY while a
 * generation, a prefill or another call runs); liyab_engine_cancel stops a
 * call between passes (LIYAB_ERR_CANCELLED). The context is used and left
 * empty (a conversation is set aside in the prefix cache when one is
 * configured).
 */
typedef enum liyab_pooling {
    LIYAB_POOLING_AUTO = -1, /* embed: the model's own pooling_type (it must be MEAN or LAST) */
    LIYAB_POOLING_NONE = 0,  /* the model declares none: a generative model */
    LIYAB_POOLING_MEAN = 1,  /* mean of every token's final hidden state */
    LIYAB_POOLING_CLS = 2,   /* first token: encoder models (refused for embedding by a decoder) */
    LIYAB_POOLING_LAST = 3,  /* last token: decoder embedding models (Qwen3-Embedding) */
    LIYAB_POOLING_RANK = 4   /* ranking head over the last token: rerankers (liyab_engine_rerank) */
} liyab_pooling;

typedef struct liyab_embed_params {
    liyab_pooling pooling; /* LIYAB_POOLING_AUTO (default), _MEAN or _LAST */
    int32_t dimensions;    /* > 0: keep the first N values (Matryoshka models: Qwen3-Embedding 32..1024); 0 (default): all */
    int32_t normalize;     /* nonzero (default): unit-length vectors; 0: as pooled */
    int32_t truncate;      /* nonzero: cut inputs longer than the context (keeping BOS/EOS); 0 (default): fail */
} liyab_embed_params;

typedef struct liyab_rerank_params {
    const char* prompt_template; /* NULL (default): the model's rerank prompt; else one with {query} and {document} */
    const char* const* labels;   /* NULL (default): the model's ranking head; else n_labels one-token texts scored */
    size_t n_labels;             /*   with the LM head, the first positive (a reranker without a head: "yes", "no") */
} liyab_rerank_params;

typedef struct liyab_pooling_stats {
    int32_t tokens;        /* tokens of every input, special tokens included */
    int32_t reused_tokens; /* of them, shared with another input and not processed again */
    double ms;             /* wall time of the call */
} liyab_pooling_stats;

LIYAB_C_API void liyab_embed_params_default(liyab_embed_params* params);
LIYAB_C_API void liyab_rerank_params_default(liyab_rerank_params* params);
/* The pooling the loaded model declares (a liyab_pooling value: NONE for a
 * generative model); -1 for a NULL engine. Thread-safe. */
LIYAB_C_API int32_t liyab_engine_pooling(const liyab_engine* engine);
/* Values per vector of liyab_engine_embed without `dimensions`: the model's
 * hidden size (1024 for Qwen3-Embedding-0.6B); 0 for a NULL engine. */
LIYAB_C_API int32_t liyab_engine_embedding_size(const liyab_engine* engine);
/*
 * Embeds `n_texts` UTF-8 texts: texts[i] holds lengths[i] bytes (`lengths`
 * NULL: each text is NUL-terminated). Writes the vectors one after another to
 * `out`: n_texts * dims floats, dims being params->dimensions when set, else
 * liyab_engine_embedding_size(); `out_capacity` (in floats) must hold them
 * (else LIYAB_ERR_INVALID_ARGUMENT before any work). `params` and `stats`
 * may be NULL (defaults / not reported). An instruction is part of the text
 * (Qwen3-Embedding queries: "Instruct: {task}\nQuery:{query}").
 */
LIYAB_C_API liyab_status liyab_engine_embed(liyab_engine* engine, const char* const* texts, const size_t* lengths,
                                            size_t n_texts, const liyab_embed_params* params, float* out,
                                            size_t out_capacity, liyab_pooling_stats* stats);
/*
 * Scores `n_documents` UTF-8 documents (lengths as for liyab_engine_embed)
 * against the NUL-terminated `query`, writing one score in (0, 1) per
 * document to `out_scores` (higher: more relevant; in input order, not
 * sorted). `params` and `stats` may be NULL.
 */
LIYAB_C_API liyab_status liyab_engine_rerank(liyab_engine* engine, const char* query, const char* const* documents,
                                             const size_t* lengths, size_t n_documents,
                                             const liyab_rerank_params* params, float* out_scores,
                                             liyab_pooling_stats* stats);

/*
 * Prefix cache (prefix_cache_mb, prefix_cache_dir in liyab_engine_config):
 * what its store holds and how it was used since the engine was created
 * (Engine::PrefixCacheStats has the whole contract). All zero without a
 * prefix cache. Thread-safe: may be read while a generation runs.
 */
typedef struct liyab_prefix_cache_stats {
    uint64_t ram_bytes;          /* KV chunks and state images in RAM */
    uint64_t ram_cap_bytes;      /* prefix_cache_mb as granted (0: no RAM tier) */
    uint64_t storage_bytes;      /* the store's files in prefix_cache_dir, every model's */
    uint64_t storage_cap_bytes;  /* prefix_cache_disk_mb (0: no storage tier) */
    int32_t chunks;              /* this model's, in RAM and on storage */
    int32_t state_images;        /* recurrent-state images (hybrid models) */
    int32_t contexts;            /* stored contexts */
    uint64_t set_asides;         /* contexts set aside */
    uint64_t chunks_written;     /* chunks they added */
    uint64_t chunks_reused;      /* chunks they found stored already (shared prefixes) */
    double set_aside_ms;         /* total time setting aside */
    uint64_t restores;           /* prompts that restored a prefix */
    uint64_t restored_tokens;    /* positions restored */
    double restore_ms;           /* total time restoring */
} liyab_prefix_cache_stats;

LIYAB_C_API liyab_status liyab_engine_prefix_cache_stats(const liyab_engine* engine, liyab_prefix_cache_stats* out_stats);

#ifdef __cplusplus
}
#endif

#endif /* LIYAB_C_API_H */
