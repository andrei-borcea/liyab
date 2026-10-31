// Liyab — C ABI implementation. Translates C structs to the C++ API and makes
// sure no exception or C++ type escapes across the boundary.
#include "liyab/liyab_c_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "core/chat_template.h"
#include "core/kv_store.h"
#include "core/log.h"
#include "liyab/liyab.h"

struct liyab_engine {
    std::unique_ptr<liyab::Engine> engine;
};

struct liyab_grammar {
    liyab::Grammar grammar;
};

struct liyab_sequence {
    std::unique_ptr<liyab::Engine::Sequence> sequence;
};

namespace {

thread_local std::string g_last_error;

liyab_status to_c(liyab::ErrorCode code) { return static_cast<liyab_status>(code); }

liyab_status fail(liyab_status status, std::string message) {
    g_last_error = std::move(message);
    return status;
}

liyab_status fail(const liyab::Status& status) { return fail(to_c(status.code()), status.to_string()); }

// Runs `fn`, converting any escaping exception to an error status.
template <typename Fn>
liyab_status guarded(Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        return fail(LIYAB_ERR_OUT_OF_MEMORY, "out of memory");
    } catch (const std::exception& e) {
        return fail(LIYAB_ERR_INTERNAL, std::string("internal error: ") + e.what());
    } catch (...) {
        return fail(LIYAB_ERR_INTERNAL, "internal error");
    }
}

size_t copy_out(const std::string& text, char* buffer, size_t size) {
    if (buffer != nullptr && size > 0) {
        const size_t n = std::min(size - 1, text.size());
        std::memcpy(buffer, text.data(), n);
        buffer[n] = '\0';
    }
    return text.size();
}

liyab::SamplingParams to_cpp(const liyab_sampling_params* p) {
    liyab::SamplingParams params;
    if (p != nullptr) {
        params.temperature = p->temperature;
        params.top_k = p->top_k;
        params.top_p = p->top_p;
        params.seed = p->seed;
        params.max_tokens = p->max_tokens;
        params.add_bos = p->add_bos != 0;
    }
    return params;
}

liyab_status finish_generation(const liyab::Result<liyab::GenerationStats>& result, liyab_generation_stats* stats) {
    if (!result) return fail(result.status());
    if (stats != nullptr) {
        const liyab::GenerationStats& s = result.value();
        *stats = liyab_generation_stats{s.prompt_tokens,         s.generated_tokens,      s.prefill_ms,
                                        s.decode_ms,             s.tokens_per_second,     s.draft_tokens_proposed,
                                        s.draft_tokens_accepted, s.thermal_reroutes,      s.paced_idle_ms,
                                        s.cancelled ? 1 : 0,     s.early_exits,           s.early_exit_layers_skipped,
                                        s.head_pruned_steps,     s.weight_stalls,         s.weight_wait_ms,
                                        s.kv_cache_bytes,        s.ffn_blocks_skipped,    s.sparse_ffn_steps,
                                        s.ttft_ms,               s.cached_prefix_tokens,  s.expert_hits,
                                        s.expert_late,           s.expert_misses,         s.expert_bytes_read,
                                        s.expert_stall_ms,       s.expert_unused,
                                        s.decode_phases.attention_ms,
                                        s.decode_phases.delta_net_ms,
                                        s.decode_phases.router_ms,
                                        s.decode_phases.experts_ms,
                                        s.decode_phases.shared_expert_ms,
                                        s.decode_phases.dense_ffn_ms,
                                        s.decode_phases.lm_head_ms,
                                        s.expert_predicted,
                                        s.expert_predicted_used,
                                        s.expert_dropped,
                                        s.expert_skipped,
                                        s.forced_tokens,
                                        s.forced_ms,
                                        s.context_ms,
                                        s.ttft_stall_ms,
                                        s.segment_reused_tokens,
                                        s.segment_recomputed_tokens,
                                        s.segment_ms,
                                        s.expert_page_cache_loads};
    }
    return LIYAB_OK;
}

liyab::TokenCallback wrap(liyab_token_callback callback, void* user_data) {
    if (callback == nullptr) return {};
    return [callback, user_data](std::string_view piece, int32_t token) {
        return callback(piece.data(), piece.size(), token, user_data) != 0;
    };
}

// Borrowed views of a C array of texts (lengths NULL: NUL-terminated).
std::vector<std::string_view> text_views(const char* const* texts, const size_t* lengths, size_t n) {
    std::vector<std::string_view> views(n);
    for (size_t i = 0; i < n; ++i) {
        views[i] = lengths != nullptr ? std::string_view(texts[i], lengths[i]) : std::string_view(texts[i]);
    }
    return views;
}

bool texts_valid(const char* const* texts, const size_t* lengths, size_t n) {
    if (n > 0 && texts == nullptr) return false;
    for (size_t i = 0; i < n; ++i) {
        if (texts[i] == nullptr && (lengths == nullptr || lengths[i] > 0)) return false;
    }
    return true;
}

void put_stats(const liyab::PoolingStats& s, liyab_pooling_stats* out) {
    if (out != nullptr) *out = liyab_pooling_stats{s.tokens, s.reused_tokens, s.ms};
}

// liyab_log_buffer_*: messages kept for UIs that poll.
std::mutex g_log_mutex;
std::string g_log;
size_t g_log_cap = 0;

void buffer_sink(int level, const char* message, void*) {
    static constexpr char kLetters[] = {'D', 'I', 'W', 'E'};
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log += kLetters[std::clamp(level, 0, 3)];
    g_log += ' ';
    g_log += message;
    g_log += '\n';
    if (g_log.size() > g_log_cap) {  // drop whole lines from the front
        const size_t cut = g_log.find('\n', g_log.size() - g_log_cap);
        g_log.erase(0, cut == std::string::npos ? g_log.size() : cut + 1);
    }
}

}  // namespace

extern "C" {

const char* liyab_version(void) { return LIYAB_VERSION_STRING; }

const char* liyab_last_error(void) { return g_last_error.c_str(); }

void liyab_set_log_level(int32_t level) {
    liyab::log::set_level(static_cast<liyab::log::Level>(std::clamp(level, 0, 4)));
}

void liyab_set_log_callback(liyab_log_callback callback, void* user_data) {
    liyab::log::set_sink(reinterpret_cast<liyab::log::Sink>(callback), user_data);
}

void liyab_log_buffer_enable(size_t max_bytes) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log_cap = max_bytes;
        g_log.clear();
    }
    liyab::log::set_sink(max_bytes > 0 ? buffer_sink : nullptr, nullptr);
}

size_t liyab_log_buffer_take(char* buffer, size_t size) {
    if (buffer == nullptr || size == 0) return 0;
    buffer[0] = '\0';
    if (size < 2) return 0;
    std::lock_guard<std::mutex> lock(g_log_mutex);
    size_t n = std::min(size - 1, g_log.size());
    if (n < g_log.size()) {  // whole lines only, unless one line alone is too long
        const size_t end = g_log.rfind('\n', n == 0 ? 0 : n - 1);
        if (end != std::string::npos) n = end + 1;
    }
    std::memcpy(buffer, g_log.data(), n);
    buffer[n] = '\0';
    g_log.erase(0, n);
    return n;
}

void liyab_engine_config_default(liyab_engine_config* config) {
    if (config == nullptr) return;
    const liyab::EngineConfig d;
    *config = liyab_engine_config{};
    config->n_threads = d.n_threads;
    config->context_length = d.context_length;
    config->sliding_window = d.sliding_window;
    config->kv_sink_tokens = d.kv_sink_tokens;
    config->draft_tokens = d.draft_tokens;
    config->lookup_drafts = d.lookup_drafts ? 1 : 0;
    config->adaptive_drafts = d.adaptive_drafts ? 1 : 0;
    config->suffix_drafts = d.suffix_drafts ? 1 : 0;
    config->kv_cache_type = static_cast<liyab_kv_cache_type>(d.kv_cache_type);
    config->backend = LIYAB_BACKEND_AUTO;
    config->power_profile = static_cast<liyab_power_profile>(d.power.profile);
    config->skin_threshold_c = d.power.skin_threshold_c;
    config->target_tps = d.power.target_tps;
    config->thermal_polling = d.thermal_polling ? 1 : 0;
    config->triple_buffer_loading = d.triple_buffer_loading ? 1 : 0;
    config->expert_cache_mb = d.expert_cache_mb;
    config->memory_budget_mb = d.memory_budget_mb;
    config->requant_bits = d.requant_bits;
    config->moe_expert_mass = d.moe_expert_mass;
    config->moe_max_experts = d.moe_max_experts;
    config->moe_skip_slow = d.moe_skip_slow;
    config->prefix_cache_mb = d.prefix_cache_mb;
    config->prefix_cache_dir = nullptr;
    config->prefix_cache_disk_mb = d.prefix_cache_disk_mb;
    config->cache_dir = nullptr;
    config->early_exit = d.experimental.early_exit ? 1 : 0;
    config->early_exit_threshold = d.experimental.early_exit_threshold;
    config->head_pruning = d.experimental.head_pruning ? 1 : 0;
    config->head_keep_ratio = d.experimental.head_keep_ratio;
    config->egls = d.experimental.egls ? 1 : 0;
    config->egls_threshold = d.experimental.egls_threshold;
    config->tdss = d.experimental.tdss ? (d.experimental.tdss_force ? 2 : 1) : 0;
    config->pooling_kv_cache_type = static_cast<liyab_kv_cache_type>(d.pooling_kv_cache_type);
    config->segment_reuse = d.segment_reuse ? 1 : 0;
    config->segment_reuse_max_kl = d.segment_reuse_max_kl;
    config->prefix_cache_lossy = d.prefix_cache_lossy ? 1 : 0;
    config->adapter_memory_mb = d.adapter_memory_mb;
    config->draft_tree = d.draft_tree ? 1 : 0;
    config->max_sequences = d.max_sequences;
    config->i8mm_tiles = d.experimental.i8mm_tiles ? 1 : 0;
    config->prerouter_trace = d.experimental.prerouter_trace ? 1 : 0;
    config->page_cache_mode = d.experimental.page_cache_mode;
    config->converted_weights_mb = d.converted_weights_mb;
}

void liyab_sampling_params_default(liyab_sampling_params* params) {
    if (params == nullptr) return;
    const liyab::SamplingParams d;
    *params = liyab_sampling_params{d.temperature, d.top_k, d.top_p, d.seed, d.max_tokens, d.add_bos ? 1 : 0};
}

liyab_status liyab_engine_create(const liyab_engine_config* config, liyab_engine** out_engine) {
    return guarded([&]() -> liyab_status {
        if (config == nullptr || out_engine == nullptr || config->model_path == nullptr) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "config, config->model_path and out_engine are required");
        }
        if (config->kv_cache_type < LIYAB_KV_F16 || config->kv_cache_type > LIYAB_KV_Q4_1 ||
            config->pooling_kv_cache_type < LIYAB_KV_F16 || config->pooling_kv_cache_type > LIYAB_KV_Q4_1 ||
            config->backend < LIYAB_BACKEND_AUTO || config->backend > LIYAB_BACKEND_NEUROPILOT ||
            config->power_profile < LIYAB_POWER_PERFORMANCE || config->power_profile > LIYAB_POWER_LOW_POWER) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "enum value out of range in liyab_engine_config");
        }
        *out_engine = nullptr;
        liyab::EngineConfig c;
        c.model_path = config->model_path;
        if (config->draft_model_path != nullptr) c.draft_model_path = config->draft_model_path;
        c.n_threads = config->n_threads;
        c.context_length = config->context_length;
        c.sliding_window = config->sliding_window;
        c.kv_sink_tokens = config->kv_sink_tokens;
        c.draft_tokens = config->draft_tokens;
        c.lookup_drafts = config->lookup_drafts != 0;
        c.adaptive_drafts = config->adaptive_drafts != 0;
        c.suffix_drafts = config->suffix_drafts != 0;
        c.kv_cache_type = static_cast<liyab::KvCacheType>(config->kv_cache_type);
        if (config->backend != LIYAB_BACKEND_AUTO) c.backend = static_cast<liyab::BackendKind>(config->backend);
        c.power.profile = static_cast<liyab::PowerProfile>(config->power_profile);
        if (config->skin_threshold_c > 0.0f) c.power.skin_threshold_c = config->skin_threshold_c;
        c.power.target_tps = config->target_tps;
        c.thermal_polling = config->thermal_polling != 0;
        c.triple_buffer_loading = config->triple_buffer_loading != 0;
        c.expert_cache_mb = config->expert_cache_mb;
        c.memory_budget_mb = config->memory_budget_mb;
        c.requant_bits = config->requant_bits;
        c.moe_expert_mass = config->moe_expert_mass;
        c.moe_max_experts = config->moe_max_experts;
        c.experimental.early_exit = config->early_exit != 0;
        c.experimental.early_exit_threshold = config->early_exit_threshold;
        c.experimental.head_pruning = config->head_pruning != 0;
        c.experimental.head_keep_ratio = config->head_keep_ratio;
        c.experimental.egls = config->egls != 0;
        c.experimental.egls_threshold = config->egls_threshold;
        c.experimental.tdss = config->tdss != 0;
        c.experimental.tdss_force = config->tdss == 2;
        // KV dedup is retired (the prefix cache's store replaces it): the
        // field stays for the struct's layout, and what it left there goes.
        if (config->kv_dedup_dir != nullptr) liyab::remove_kv_dedup_files(config->kv_dedup_dir);
        c.moe_skip_slow = config->moe_skip_slow;
        c.prefix_cache_mb = config->prefix_cache_mb;
        if (config->prefix_cache_dir != nullptr) c.prefix_cache_dir = config->prefix_cache_dir;
        c.prefix_cache_disk_mb = config->prefix_cache_disk_mb;
        if (config->cache_dir != nullptr) c.cache_dir = config->cache_dir;
        c.pooling_kv_cache_type = static_cast<liyab::KvCacheType>(config->pooling_kv_cache_type);
        c.segment_reuse = config->segment_reuse != 0;
        c.segment_reuse_max_kl = config->segment_reuse_max_kl;
        c.prefix_cache_lossy = config->prefix_cache_lossy != 0;
        c.adapter_memory_mb = config->adapter_memory_mb;
        c.draft_tree = config->draft_tree != 0;
        c.experimental.i8mm_tiles = config->i8mm_tiles != 0;
        c.experimental.prerouter_trace = config->prerouter_trace != 0;
        c.experimental.page_cache_mode = config->page_cache_mode;
        c.converted_weights_mb = config->converted_weights_mb;
        c.max_sequences = std::max(config->max_sequences, 1);  // a zero-filled config: one sequence

        auto engine = liyab::Engine::create(c);
        if (!engine) return fail(engine.status());
        *out_engine = new liyab_engine{std::move(engine).value()};
        return LIYAB_OK;
    });
}

void liyab_engine_destroy(liyab_engine* engine) {
    // Destruction joins worker threads; it does not throw.
    delete engine;
}

liyab_status liyab_engine_generate(liyab_engine* engine, const char* prompt, const liyab_sampling_params* params,
                                   liyab_token_callback callback, void* user_data, liyab_generation_stats* stats) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || prompt == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and prompt are required");
        return finish_generation(engine->engine->generate(prompt, to_cpp(params), wrap(callback, user_data)), stats);
    });
}

liyab_status liyab_engine_generate_forced(liyab_engine* engine, const char* prompt, const liyab_sampling_params* params,
                                          liyab_token_callback callback, liyab_force_callback force, void* user_data,
                                          liyab_generation_stats* stats) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || prompt == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and prompt are required");
        if (force == nullptr) {
            return finish_generation(engine->engine->generate(prompt, to_cpp(params), wrap(callback, user_data)), stats);
        }
        const liyab::ForceCallback forced = [force, user_data](std::string_view generated) {
            char out[512];
            const size_t n = force(generated.data(), generated.size(), out, sizeof out, user_data);
            return std::string(out, std::min(n, sizeof out));
        };
        return finish_generation(engine->engine->generate(prompt, to_cpp(params), wrap(callback, user_data), forced),
                                 stats);
    });
}

liyab_status liyab_engine_generate_tokens(liyab_engine* engine, const int32_t* tokens, size_t n_tokens,
                                          const liyab_sampling_params* params, liyab_token_callback callback,
                                          void* user_data, liyab_generation_stats* stats) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || tokens == nullptr || n_tokens == 0) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and a non-empty token array are required");
        }
        return finish_generation(engine->engine->generate_tokens(std::span<const int32_t>(tokens, n_tokens),
                                                                 to_cpp(params), wrap(callback, user_data)),
                                 stats);
    });
}

int32_t liyab_engine_tokenize(const liyab_engine* engine, const char* text, int32_t add_bos, int32_t* out,
                              int32_t capacity) {
    if (engine == nullptr || text == nullptr || capacity < 0 || (out == nullptr && capacity > 0)) {
        fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and text are required");
        return -1;
    }
    try {
        auto ids = engine->engine->tokenize(text, add_bos != 0);
        if (!ids) {
            fail(ids.status());
            return -1;
        }
        const auto n = static_cast<int32_t>(ids->size());
        std::copy_n(ids->begin(), std::min(n, capacity), out);
        return n;
    } catch (...) {
        fail(LIYAB_ERR_INTERNAL, "internal error");
        return -1;
    }
}

void liyab_engine_cancel(liyab_engine* engine) {
    if (engine != nullptr) engine->engine->cancel();
}

void liyab_engine_cancel_prefill(liyab_engine* engine) {
    if (engine != nullptr) engine->engine->cancel_prefill();
}

liyab_status liyab_engine_set_power_profile(liyab_engine* engine, liyab_power_profile profile) {
    if (engine == nullptr || profile < LIYAB_POWER_PERFORMANCE || profile > LIYAB_POWER_LOW_POWER) {
        return fail(LIYAB_ERR_INVALID_ARGUMENT, "invalid engine or power profile");
    }
    engine->engine->power().set_profile(static_cast<liyab::PowerProfile>(profile));
    return LIYAB_OK;
}

size_t liyab_engine_describe(const liyab_engine* engine, char* buffer, size_t size) {
    if (engine == nullptr) return copy_out("", buffer, size);
    try {
        return copy_out(engine->engine->describe(), buffer, size);
    } catch (...) {
        return copy_out("", buffer, size);
    }
}

int64_t liyab_engine_metadata(const liyab_engine* engine, const char* key, char* buffer, size_t size) {
    if (engine == nullptr || key == nullptr) return -1;
    try {
        const auto value = engine->engine->model_metadata(key);
        if (!value) return -1;
        return static_cast<int64_t>(copy_out(*value, buffer, size));
    } catch (...) {
        return -1;
    }
}

liyab_status liyab_engine_prefill(liyab_engine* engine, const char* text, int32_t add_bos) {
    if (engine == nullptr || text == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const liyab::Status s = engine->engine->prefill(text, add_bos != 0);
        return s.is_ok() ? LIYAB_OK : fail(s);
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_engine_reset_context(liyab_engine* engine) {
    if (engine == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    engine->engine->reset_context();
    return LIYAB_OK;
}

liyab_status liyab_engine_save_state(liyab_engine* engine, const char* path) {
    if (engine == nullptr || path == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const liyab::Status s = engine->engine->save_state(path);
        return s.is_ok() ? LIYAB_OK : fail(s);
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_engine_save_state_of(liyab_engine* engine, const char* path, const char* text, size_t text_len) {
    if (engine == nullptr || path == nullptr || (text == nullptr && text_len > 0)) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const liyab::Status s = engine->engine->save_state(path, std::string_view(text, text_len));
        return s.is_ok() ? LIYAB_OK : fail(s);
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

int32_t liyab_engine_context_length(const liyab_engine* engine) {
    return engine == nullptr ? 0 : engine->engine->context_length();
}

liyab_status liyab_engine_memory_plan(const liyab_engine* engine, liyab_memory_plan* out) {
    if (engine == nullptr || out == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    const liyab::Engine::MemoryPlan p = engine->engine->memory_plan();
    *out = liyab_memory_plan{p.resident_bytes, p.expert_bytes, p.expert_cache_bytes, p.recommended_bytes, p.requant_bits};
    return LIYAB_OK;
}

liyab_status liyab_engine_weight_layout(const liyab_engine* engine, liyab_weight_layout* out) {
    if (engine == nullptr || out == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    const liyab::Engine::WeightLayout layout = engine->engine->weight_layout();
    *out = {layout.file_bytes, layout.streamed_bytes, layout.streamed_blocks,
            layout.total_blocks, layout.mmap_streaming ? 1 : 0};
    return LIYAB_OK;
}

liyab_status liyab_engine_warm_memory(liyab_engine* engine, uint32_t* out_queued) {
    if (engine == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const size_t queued = engine->engine->warm_memory();
        if (out_queued != nullptr) *out_queued = static_cast<uint32_t>(queued);
        return LIYAB_OK;
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_engine_save_expert_profile(liyab_engine* engine, const char* path) {
    if (engine == nullptr || path == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const liyab::Status s = engine->engine->save_expert_profile(path);
        return s.is_ok() ? LIYAB_OK : fail(s);
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_engine_load_expert_profile(liyab_engine* engine, const char* path) {
    if (engine == nullptr || path == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const liyab::Status s = engine->engine->load_expert_profile(path);
        return s.is_ok() ? LIYAB_OK : fail(s);
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_engine_trim_memory(liyab_engine* engine, uint64_t* out_bytes) {
    if (engine == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const size_t released = engine->engine->trim_memory();
        if (out_bytes != nullptr) *out_bytes = released;
        return LIYAB_OK;
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_engine_load_state(liyab_engine* engine, const char* path, int32_t* out_tokens) {
    if (engine == nullptr || path == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        auto r = engine->engine->load_state(path);
        if (!r) return fail(r.status());
        if (out_tokens != nullptr) *out_tokens = r.value();
        return LIYAB_OK;
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_engine_get_counters(const liyab_engine* engine, liyab_engine_counters* out) {
    if (engine == nullptr || out == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    try {
        const liyab::Engine::Counters c = engine->engine->counters();
        *out = liyab_engine_counters{c.accelerator_busy_ms, c.storage_bytes_read, c.tokens_generated};
        return LIYAB_OK;
    } catch (...) {
        return LIYAB_ERR_INTERNAL;
    }
}

liyab_status liyab_get_resource_snapshot(const liyab_engine* engine, liyab_resource_snapshot* out) {
    if (out == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    return guarded([&]() -> liyab_status {
        // Cache hardware capacity, sample reclaimable memory and policy anew.
        static const uint64_t total = liyab::detect_device().total_memory;
        const liyab::PowerPolicy policy = engine != nullptr ? engine->engine->power().policy() : liyab::PowerPolicy{};
        *out = {liyab::available_memory_bytes(), total, policy.thread_fraction,
                policy.pressure, policy.slowdown, policy.throttled ? 1 : 0};
        return LIYAB_OK;
    });
}

size_t liyab_supported_architectures(char* buffer, size_t size) {
    try {
        return copy_out(liyab::supported_architectures(), buffer, size);
    } catch (...) {
        return copy_out("", buffer, size);
    }
}

size_t liyab_describe_device(char* buffer, size_t size) {
    try {
        return copy_out(liyab::describe_device(liyab::detect_device()), buffer, size);
    } catch (...) {
        return copy_out("", buffer, size);
    }
}

int64_t liyab_engine_render_chat(const liyab_engine* engine, const char* request_json, char* buffer, size_t size) {
    if (engine == nullptr || request_json == nullptr) {
        fail(LIYAB_ERR_INVALID_ARGUMENT, "null argument");
        return -1;
    }
    int64_t length = -1;
    const liyab_status status = guarded([&] {
        auto text = engine->engine->render_chat(request_json);
        if (!text) return fail(text.status());
        length = static_cast<int64_t>(copy_out(*text, buffer, size));
        return LIYAB_OK;
    });
    return status == LIYAB_OK ? length : -1;
}

int64_t liyab_engine_chat_format(const liyab_engine* engine, const char* probe_json, char* buffer, size_t size) {
    if (engine == nullptr) {
        fail(LIYAB_ERR_INVALID_ARGUMENT, "null engine");
        return -1;
    }
    int64_t length = -1;
    const liyab_status status = guarded([&] {
        auto format = engine->engine->chat_format(probe_json == nullptr ? "" : probe_json);
        if (!format) return fail(format.status());
        using liyab::jinja::Value;
        Value out = Value::object();
        auto& o = out.fields();
        o.set("prefix_stable", Value(format->prefix_stable));
        o.set("reason", Value(format->reason));
        o.set("system_prefix", Value(format->system_prefix));
        o.set("system_suffix", Value(format->system_suffix));
        o.set("first_user_prefix", Value(format->first_user_prefix));
        o.set("no_system_head", Value(format->no_system_head));
        o.set("user_prefix", Value(format->user_prefix));
        o.set("user_suffix", Value(format->user_suffix));
        o.set("generation_prompt", Value(format->generation_prompt));
        o.set("assistant_end", Value(format->assistant_end));
        length = static_cast<int64_t>(copy_out(out.to_json(), buffer, size));
        return LIYAB_OK;
    });
    return status == LIYAB_OK ? length : -1;
}

liyab_status liyab_grammar_from_json_schema(const char* schema_json, int32_t any_whitespace, liyab_grammar** out_grammar) {
    return guarded([&]() -> liyab_status {
        if (out_grammar == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "out_grammar is required");
        *out_grammar = nullptr;
        if (schema_json == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "schema_json is required");
        auto g = liyab::Grammar::from_json_schema(schema_json, liyab::JsonFormat{.any_whitespace = any_whitespace != 0});
        if (!g) return fail(g.status());
        *out_grammar = new liyab_grammar{std::move(g).value()};
        return LIYAB_OK;
    });
}

liyab_status liyab_grammar_parse(const char* text, liyab_grammar** out_grammar) {
    return guarded([&]() -> liyab_status {
        if (out_grammar == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "out_grammar is required");
        *out_grammar = nullptr;
        if (text == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "text is required");
        auto g = liyab::Grammar::parse(text);
        if (!g) return fail(g.status());
        *out_grammar = new liyab_grammar{std::move(g).value()};
        return LIYAB_OK;
    });
}

void liyab_grammar_destroy(liyab_grammar* grammar) { delete grammar; }

int32_t liyab_grammar_matches(const liyab_grammar* grammar, const char* text, size_t len) {
    if (grammar == nullptr || (text == nullptr && len > 0)) return 0;
    int32_t matched = 0;
    (void)guarded([&] {
        matched = grammar->grammar.matches(std::string_view(text == nullptr ? "" : text, len)) ? 1 : 0;
        return LIYAB_OK;
    });
    return matched;
}

liyab_status liyab_engine_generate_structured(liyab_engine* engine, const char* prompt,
                                              const liyab_sampling_params* params, liyab_token_callback callback,
                                              const liyab_grammar* grammar, liyab_continuation_callback continuation,
                                              void* user_data, liyab_generation_stats* stats,
                                              liyab_grammar_stats* grammar_stats) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || prompt == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and prompt are required");
        liyab::GenerateOptions options;
        if (grammar != nullptr) options.grammar = grammar->grammar;
        if (continuation != nullptr) {
            options.continuation = [continuation, user_data](std::string_view generated) {
                // Room for a tool call's whole fixed opening; a callback whose
                // text does not fit returns 0 rather than cut it.
                char out[4096];
                const liyab_grammar* next = nullptr;
                const size_t n = continuation(generated.data(), generated.size(), out, sizeof out, &next, user_data);
                liyab::Continuation c;
                c.text.assign(out, std::min(n, sizeof out));
                if (next != nullptr) c.grammar = next->grammar;
                return c;
            };
        }
        auto result = engine->engine->generate(prompt, to_cpp(params), wrap(callback, user_data), options);
        if (result && grammar_stats != nullptr) {
            *grammar_stats = liyab_grammar_stats{result->constrained_tokens, result->grammar_ms};
        }
        return finish_generation(result, stats);
    });
}

void liyab_embed_params_default(liyab_embed_params* params) {
    if (params == nullptr) return;
    const liyab::EmbedOptions d;
    *params = liyab_embed_params{static_cast<liyab_pooling>(d.pooling), d.dimensions, d.normalize ? 1 : 0,
                                 d.truncate ? 1 : 0};
}

void liyab_rerank_params_default(liyab_rerank_params* params) {
    if (params == nullptr) return;
    *params = liyab_rerank_params{nullptr, nullptr, 0};
}

int32_t liyab_engine_pooling(const liyab_engine* engine) {
    return engine == nullptr ? -1 : static_cast<int32_t>(engine->engine->pooling());
}

int32_t liyab_engine_embedding_size(const liyab_engine* engine) {
    return engine == nullptr ? 0 : engine->engine->embedding_size();
}

liyab_status liyab_engine_embed(liyab_engine* engine, const char* const* texts, const size_t* lengths, size_t n_texts,
                                const liyab_embed_params* params, float* out, size_t out_capacity,
                                liyab_pooling_stats* stats) {
    return guarded([&]() -> liyab_status {
        if (stats != nullptr) *stats = liyab_pooling_stats{};
        if (engine == nullptr || !texts_valid(texts, lengths, n_texts) || (out == nullptr && n_texts > 0)) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine, texts and out are required");
        }
        liyab::EmbedOptions options;
        if (params != nullptr) {
            if (params->pooling < LIYAB_POOLING_AUTO || params->pooling > LIYAB_POOLING_RANK) {
                return fail(LIYAB_ERR_INVALID_ARGUMENT, "pooling out of range in liyab_embed_params");
            }
            options.pooling = static_cast<liyab::Pooling>(params->pooling);
            options.dimensions = params->dimensions;
            options.normalize = params->normalize != 0;
            options.truncate = params->truncate != 0;
        }
        const int32_t size = engine->engine->embedding_size();
        const auto dims = static_cast<size_t>(options.dimensions > 0 && options.dimensions <= size ? options.dimensions
                                                                                                     : size);
        if (out_capacity < n_texts * dims) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "out holds " + std::to_string(out_capacity) + " floats; " +
                                                        std::to_string(n_texts * dims) + " are needed");
        }
        liyab::PoolingStats s;
        const auto views = text_views(texts, lengths, n_texts);
        auto vectors = engine->engine->embed(views, options, &s);
        put_stats(s, stats);
        if (!vectors) return fail(vectors.status());
        for (size_t i = 0; i < vectors->size(); ++i) std::copy((*vectors)[i].begin(), (*vectors)[i].end(), out + i * dims);
        return LIYAB_OK;
    });
}

liyab_status liyab_engine_rerank(liyab_engine* engine, const char* query, const char* const* documents,
                                 const size_t* lengths, size_t n_documents, const liyab_rerank_params* params,
                                 float* out_scores, liyab_pooling_stats* stats) {
    return guarded([&]() -> liyab_status {
        if (stats != nullptr) *stats = liyab_pooling_stats{};
        if (engine == nullptr || query == nullptr || !texts_valid(documents, lengths, n_documents) ||
            (out_scores == nullptr && n_documents > 0)) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine, query, documents and out_scores are required");
        }
        liyab::RerankOptions options;
        if (params != nullptr) {
            if (params->prompt_template != nullptr) options.prompt_template = params->prompt_template;
            if (params->n_labels > 0 && params->labels == nullptr) {
                return fail(LIYAB_ERR_INVALID_ARGUMENT, "liyab_rerank_params: n_labels without labels");
            }
            for (size_t i = 0; i < params->n_labels; ++i) {
                if (params->labels[i] == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "a rerank label is NULL");
                options.labels.emplace_back(params->labels[i]);
            }
        }
        liyab::PoolingStats s;
        const auto views = text_views(documents, lengths, n_documents);
        auto scores = engine->engine->rerank(query, views, options, &s);
        put_stats(s, stats);
        if (!scores) return fail(scores.status());
        std::copy(scores->begin(), scores->end(), out_scores);
        return LIYAB_OK;
    });
}

liyab_status liyab_engine_prefix_cache_stats(const liyab_engine* engine, liyab_prefix_cache_stats* out_stats) {
    if (engine == nullptr || out_stats == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    const liyab::Engine::PrefixCacheStats s = engine->engine->prefix_cache_stats();
    *out_stats = liyab_prefix_cache_stats{s.ram_bytes,     s.ram_cap_bytes,  s.storage_bytes,   s.storage_cap_bytes,
                                          s.chunks,        s.state_images,   s.contexts,        s.set_asides,
                                          s.chunks_written, s.chunks_reused, s.set_aside_ms,    s.restores,
                                          s.restored_tokens, s.restore_ms,    s.chunk_tokens,    s.segments,
                                          s.segments_kept, s.segment_reuses, s.segment_reused_tokens,
                                          s.segment_recomputed_tokens};
    return LIYAB_OK;
}

namespace {

// Marked parts of a C caller (NULL with none).
std::vector<liyab::PromptSegment> to_cpp(const liyab_prompt_segment* segments, size_t n) {
    std::vector<liyab::PromptSegment> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back({segments[i].begin, segments[i].end});
    return out;
}

}  // namespace

liyab_status liyab_engine_generate_segmented(liyab_engine* engine, const char* prompt,
                                             const liyab_prompt_segment* segments, size_t n_segments,
                                             const liyab_sampling_params* params, liyab_token_callback callback,
                                             void* user_data, liyab_generation_stats* stats) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || prompt == nullptr || (segments == nullptr && n_segments > 0)) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and prompt are required, and segments when n_segments > 0");
        }
        liyab::GenerateOptions options;
        options.segments = to_cpp(segments, n_segments);
        return finish_generation(engine->engine->generate(prompt, to_cpp(params), wrap(callback, user_data), options),
                                 stats);
    });
}

liyab_status liyab_engine_prefill_segmented(liyab_engine* engine, const char* text,
                                            const liyab_prompt_segment* segments, size_t n_segments, int32_t add_bos) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || text == nullptr || (segments == nullptr && n_segments > 0)) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and text are required, and segments when n_segments > 0");
        }
        const std::vector<liyab::PromptSegment> parts = to_cpp(segments, n_segments);
        const liyab::Status s = engine->engine->prefill(text, parts, add_bos != 0);
        return s.is_ok() ? LIYAB_OK : fail(s);
    });
}

liyab_status liyab_engine_segment_reuse(const liyab_engine* engine, liyab_segment_reuse_info* out) {
    if (engine == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine is required");
    const liyab::Engine::SegmentReuseInfo info = engine->engine->segment_reuse();
    if (out != nullptr) {
        *out = liyab_segment_reuse_info{info.fraction, info.kl, info.top1, info.from_cache ? 1 : 0,
                                        info.calibrating ? 1 : 0, info.progress, info.bound};
    }
    return info.status.is_ok() ? LIYAB_OK : fail(info.status);
}

liyab_status liyab_engine_load_adapter(liyab_engine* engine, const char* name, const char* path) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || name == nullptr || path == nullptr) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine, name and path are required");
        }
        // liyab_adapter_info holds the name whole.
        if (std::strlen(name) >= sizeof(liyab_adapter_info::name)) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "an adapter name is at most 127 bytes");
        }
        const liyab::Status s = engine->engine->load_adapter(name, path);
        return s.is_ok() ? LIYAB_OK : fail(s);
    });
}

liyab_status liyab_engine_unload_adapter(liyab_engine* engine, const char* name) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || name == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and name are required");
        const liyab::Status s = engine->engine->unload_adapter(name);
        return s.is_ok() ? LIYAB_OK : fail(s);
    });
}

liyab_status liyab_engine_set_adapter(liyab_engine* engine, const char* name, float scale) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine is required");
        const liyab::Status s = engine->engine->set_adapter(name != nullptr ? name : "", scale);
        return s.is_ok() ? LIYAB_OK : fail(s);
    });
}

int32_t liyab_engine_adapter_count(const liyab_engine* engine) {
    if (engine == nullptr) return 0;
    try {
        return static_cast<int32_t>(engine->engine->adapters().size());
    } catch (...) {
        return 0;
    }
}

liyab_status liyab_engine_adapter_info(const liyab_engine* engine, int32_t index, liyab_adapter_info* out) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || out == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and out are required");
        const std::vector<liyab::AdapterInfo> all = engine->engine->adapters();
        if (index < 0 || static_cast<size_t>(index) >= all.size()) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "no adapter at this index");
        }
        const liyab::AdapterInfo& a = all[static_cast<size_t>(index)];
        *out = liyab_adapter_info{};
        copy_out(a.name, out->name, sizeof out->name);
        out->alpha = a.alpha;
        out->rank = a.rank;
        out->matrices = a.matrices;
        out->bytes = a.bytes;
        out->resident = a.resident ? 1 : 0;
        out->active = a.active ? 1 : 0;
        out->scale = a.scale;
        return LIYAB_OK;
    });
}

liyab_status liyab_sequence_open(liyab_engine* engine, liyab_sequence_priority priority, liyab_sequence** out_sequence) {
    return guarded([&]() -> liyab_status {
        if (out_sequence == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "out_sequence is required");
        *out_sequence = nullptr;
        if (engine == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine is required");
        if (priority != LIYAB_SEQUENCE_FOREGROUND && priority != LIYAB_SEQUENCE_BACKGROUND) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "priority out of range");
        }
        auto seq = engine->engine->open_sequence(static_cast<liyab::SequencePriority>(priority));
        if (!seq) return fail(seq.status());
        *out_sequence = new liyab_sequence{std::move(seq).value()};
        return LIYAB_OK;
    });
}

void liyab_sequence_close(liyab_sequence* sequence) {
    // Closing waits for a call on it and sets its context aside; it does not throw out.
    (void)guarded([&] {
        delete sequence;
        return LIYAB_OK;
    });
}

int32_t liyab_sequence_id(const liyab_sequence* sequence) { return sequence != nullptr ? sequence->sequence->id() : -1; }

liyab_status liyab_sequence_set_priority(liyab_sequence* sequence, liyab_sequence_priority priority) {
    if (sequence == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "sequence is required");
    if (priority != LIYAB_SEQUENCE_FOREGROUND && priority != LIYAB_SEQUENCE_BACKGROUND) {
        return fail(LIYAB_ERR_INVALID_ARGUMENT, "priority out of range");
    }
    sequence->sequence->set_priority(static_cast<liyab::SequencePriority>(priority));
    return LIYAB_OK;
}

void liyab_sequence_cancel(liyab_sequence* sequence) {
    if (sequence != nullptr) sequence->sequence->cancel();
}

void liyab_sequence_cancel_prefill(liyab_sequence* sequence) {
    if (sequence != nullptr) sequence->sequence->cancel_prefill();
}

liyab_status liyab_sequence_generate(liyab_sequence* sequence, const char* prompt, const liyab_sampling_params* params,
                                     liyab_token_callback callback, liyab_force_callback force,
                                     const liyab_grammar* grammar, liyab_continuation_callback continuation,
                                     void* user_data, liyab_generation_stats* stats,
                                     liyab_grammar_stats* grammar_stats) {
    return guarded([&]() -> liyab_status {
        if (sequence == nullptr || prompt == nullptr) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "sequence and prompt are required");
        }
        if ((force != nullptr) + (grammar != nullptr) + (continuation != nullptr) > 1) {
            return fail(LIYAB_ERR_INVALID_ARGUMENT, "set at most one of force, grammar and continuation");
        }
        liyab::Engine::Sequence& seq = *sequence->sequence;
        liyab::Result<liyab::GenerationStats> result = liyab::GenerationStats{};
        if (force != nullptr) {
            const liyab::ForceCallback forced = [force, user_data](std::string_view generated) {
                char out[512];
                const size_t n = force(generated.data(), generated.size(), out, sizeof out, user_data);
                return std::string(out, std::min(n, sizeof out));
            };
            result = seq.generate(prompt, to_cpp(params), wrap(callback, user_data), forced);
        } else {
            liyab::GenerateOptions options;
            if (grammar != nullptr) options.grammar = grammar->grammar;
            if (continuation != nullptr) {
                options.continuation = [continuation, user_data](std::string_view generated) {
                    char out[4096];  // as liyab_engine_generate_structured
                    const liyab_grammar* next = nullptr;
                    const size_t n = continuation(generated.data(), generated.size(), out, sizeof out, &next, user_data);
                    liyab::Continuation c;
                    c.text.assign(out, std::min(n, sizeof out));
                    if (next != nullptr) c.grammar = next->grammar;
                    return c;
                };
            }
            result = seq.generate(prompt, to_cpp(params), wrap(callback, user_data), options);
        }
        if (result && grammar_stats != nullptr) {
            *grammar_stats = liyab_grammar_stats{result->constrained_tokens, result->grammar_ms};
        }
        return finish_generation(result, stats);
    });
}

liyab_status liyab_sequence_prefill(liyab_sequence* sequence, const char* text, int32_t add_bos) {
    return guarded([&]() -> liyab_status {
        if (sequence == nullptr || text == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "sequence and text are required");
        const liyab::Status s = sequence->sequence->prefill(text, add_bos != 0);
        return s.is_ok() ? LIYAB_OK : fail(s);
    });
}

liyab_status liyab_sequence_reset_context(liyab_sequence* sequence) {
    return guarded([&]() -> liyab_status {
        if (sequence == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "sequence is required");
        sequence->sequence->reset_context();
        return LIYAB_OK;
    });
}

liyab_status liyab_sequence_set_adapter(liyab_sequence* sequence, const char* name, float scale) {
    return guarded([&]() -> liyab_status {
        if (sequence == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "sequence is required");
        const liyab::Status s = sequence->sequence->set_adapter(name == nullptr ? "" : name, scale);
        return s.is_ok() ? LIYAB_OK : fail(s);
    });
}

liyab_status liyab_engine_scheduler_stats(const liyab_engine* engine, liyab_scheduler_stats* out) {
    return guarded([&]() -> liyab_status {
        if (engine == nullptr || out == nullptr) return fail(LIYAB_ERR_INVALID_ARGUMENT, "engine and out are required");
        const liyab::Engine::SchedulerStats s = engine->engine->scheduler_stats();
        *out = liyab_scheduler_stats{s.passes,         s.shared_passes,    s.shared_rows, s.preempted,
                                     s.open_sequences, s.active_sequences, s.pass_ms,     s.gather_ms};
        return LIYAB_OK;
    });
}

}  // extern "C"
