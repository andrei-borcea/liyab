// Liyab — C ABI implementation. Translates C structs to the C++ API and makes
// sure no exception or C++ type escapes across the boundary.
#include "liyab/liyab_c_api.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <mutex>
#include <new>
#include <string>

#include "core/log.h"
#include "liyab/liyab.h"

struct liyab_engine {
    std::unique_ptr<liyab::Engine> engine;
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
                                        s.expert_dropped};
    }
    return LIYAB_OK;
}

liyab::TokenCallback wrap(liyab_token_callback callback, void* user_data) {
    if (callback == nullptr) return {};
    return [callback, user_data](std::string_view piece, int32_t token) {
        return callback(piece.data(), piece.size(), token, user_data) != 0;
    };
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
    config->early_exit = d.experimental.early_exit ? 1 : 0;
    config->early_exit_threshold = d.experimental.early_exit_threshold;
    config->head_pruning = d.experimental.head_pruning ? 1 : 0;
    config->head_keep_ratio = d.experimental.head_keep_ratio;
    config->egls = d.experimental.egls ? 1 : 0;
    config->egls_threshold = d.experimental.egls_threshold;
    config->tdss = d.experimental.tdss ? (d.experimental.tdss_force ? 2 : 1) : 0;
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
        if (config->kv_dedup_dir != nullptr) c.experimental.kv_dedup_dir = config->kv_dedup_dir;

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

liyab_status liyab_engine_memory_plan(const liyab_engine* engine, liyab_memory_plan* out) {
    if (engine == nullptr || out == nullptr) return LIYAB_ERR_INVALID_ARGUMENT;
    const liyab::Engine::MemoryPlan p = engine->engine->memory_plan();
    *out = liyab_memory_plan{p.resident_bytes, p.expert_bytes, p.expert_cache_bytes, p.recommended_bytes, p.requant_bits};
    return LIYAB_OK;
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

}  // extern "C"
