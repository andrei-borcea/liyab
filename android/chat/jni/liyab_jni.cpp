// JNI bridge between com.liyab.chat.LiyabNative and Liyab's C ABI.
// All calls are synchronous; generate() runs on the caller's (background)
// Java thread and streams pieces back through the callback on that thread.
#include <jni.h>

#include <mutex>
#include <string>

#include "liyab/liyab_c_api.h"

namespace {

void throw_runtime(JNIEnv* env, const char* message) {
    jclass cls = env->FindClass("java/lang/RuntimeException");
    if (cls != nullptr) env->ThrowNew(cls, message);
}

std::string to_string(JNIEnv* env, jstring s) {
    if (s == nullptr) return {};
    const char* chars = env->GetStringUTFChars(s, nullptr);
    std::string out(chars != nullptr ? chars : "");
    if (chars != nullptr) env->ReleaseStringUTFChars(s, chars);
    return out;
}

liyab_engine* handle(jlong h) { return reinterpret_cast<liyab_engine*>(h); }

// Engine log messages arrive on any native thread; buffer them and let the
// UI drain the buffer (no JNI calls from threads the JVM does not know).
std::mutex g_log_mutex;
std::string g_log;

void on_log(int32_t level, const char* message, void*) {
    static const char kLevel[] = {'D', 'I', 'W', 'E', '-'};
    std::lock_guard<std::mutex> lock(g_log_mutex);
    if (g_log.size() > (1 << 20)) g_log.clear();  // bound memory if nobody drains
    g_log += kLevel[level >= 0 && level <= 4 ? level : 4];
    g_log += ' ';
    g_log += message;
    g_log += '\n';
}

struct StreamContext {
    JNIEnv* env;
    jobject callback;
    jmethodID on_piece;
};

int32_t on_piece(const char* piece, size_t length, int32_t /*token*/, void* user) {
    auto* ctx = static_cast<StreamContext*>(user);
    JNIEnv* env = ctx->env;
    jbyteArray bytes = env->NewByteArray(static_cast<jsize>(length));
    if (bytes == nullptr) return 0;
    env->SetByteArrayRegion(bytes, 0, static_cast<jsize>(length), reinterpret_cast<const jbyte*>(piece));
    const jboolean keep_going = env->CallBooleanMethod(ctx->callback, ctx->on_piece, bytes);
    env->DeleteLocalRef(bytes);
    if (env->ExceptionCheck()) return 0;  // the Java exception propagates after generate returns
    return keep_going ? 1 : 0;
}

}  // namespace

extern "C" {

JNIEXPORT void JNICALL Java_com_liyab_chat_LiyabNative_enableLogs(JNIEnv*, jclass, jint level) {
    liyab_set_log_level(level);
    liyab_set_log_callback(on_log, nullptr);
}

JNIEXPORT jstring JNICALL Java_com_liyab_chat_LiyabNative_takeLogs(JNIEnv* env, jclass) {
    std::string out;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        out.swap(g_log);
    }
    return env->NewStringUTF(out.c_str());
}

JNIEXPORT jlong JNICALL Java_com_liyab_chat_LiyabNative_create(JNIEnv* env, jclass, jstring model_path,
                                                               jstring cache_dir, jint threads, jint backend,
                                                               jint context_length, jfloat skin_threshold_c,
                                                               jlong memory_budget_mb) {
    const std::string model = to_string(env, model_path);
    const std::string cache = to_string(env, cache_dir);
    liyab_engine_config config;
    liyab_engine_config_default(&config);
    config.model_path = model.c_str();
    config.n_threads = threads;
    config.backend = static_cast<liyab_backend>(backend);  // LIYAB_BACKEND_CPU or LIYAB_BACKEND_VULKAN
    config.power_profile = LIYAB_POWER_PERFORMANCE;  // a chat UI wants full speed; thermal guard still applies
    config.skin_threshold_c = skin_threshold_c;  // app setting; the OS thermal status still applies
    config.memory_budget_mb = memory_budget_mb;  // per-app OS caps (HyperOS: 6 GiB PSS) are invisible to the engine
    config.context_length = context_length;  // 0: the engine's default
    config.kv_dedup_dir = cache.empty() ? nullptr : cache.c_str();  // multi-turn: reuse earlier turns' KV

    liyab_engine* engine = nullptr;
    if (liyab_engine_create(&config, &engine) != LIYAB_OK) {
        throw_runtime(env, liyab_last_error());
        return 0;
    }
    return reinterpret_cast<jlong>(engine);
}

// Returns {prompt_tokens, generated_tokens, tokens_per_second, ttft_ms,
//          cached_prefix_tokens, decode_ms, thermal_reroutes, cancelled}.
JNIEXPORT jdoubleArray JNICALL Java_com_liyab_chat_LiyabNative_generate(JNIEnv* env, jclass, jlong h, jstring prompt,
                                                                       jint max_tokens, jfloat temperature,
                                                                       jfloat top_p, jint top_k, jobject callback) {
    if (h == 0 || callback == nullptr) {
        throw_runtime(env, "engine not created");
        return nullptr;
    }
    jclass cls = env->GetObjectClass(callback);
    StreamContext ctx{env, callback, env->GetMethodID(cls, "onPiece", "([B)Z")};
    if (ctx.on_piece == nullptr) return nullptr;  // NoSuchMethodError pending

    liyab_sampling_params params;
    liyab_sampling_params_default(&params);
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    params.top_k = top_k;
    params.top_p = top_p;

    const std::string text = to_string(env, prompt);
    liyab_generation_stats stats{};
    const liyab_status status = liyab_engine_generate(handle(h), text.c_str(), &params, on_piece, &ctx, &stats);
    if (env->ExceptionCheck()) return nullptr;
    if (status != LIYAB_OK) {
        throw_runtime(env, liyab_last_error());
        return nullptr;
    }
    const jdouble values[] = {static_cast<jdouble>(stats.prompt_tokens),
                              static_cast<jdouble>(stats.generated_tokens),
                              stats.tokens_per_second,
                              stats.ttft_ms,
                              static_cast<jdouble>(stats.cached_prefix_tokens),
                              stats.decode_ms,
                              static_cast<jdouble>(stats.thermal_reroutes),
                              static_cast<jdouble>(stats.cancelled)};
    jdoubleArray out = env->NewDoubleArray(8);
    if (out != nullptr) env->SetDoubleArrayRegion(out, 0, 8, values);
    return out;
}

JNIEXPORT void JNICALL Java_com_liyab_chat_LiyabNative_cancel(JNIEnv*, jclass, jlong h) {
    if (h != 0) liyab_engine_cancel(handle(h));
}

JNIEXPORT void JNICALL Java_com_liyab_chat_LiyabNative_destroy(JNIEnv*, jclass, jlong h) {
    if (h != 0) liyab_engine_destroy(handle(h));
}

JNIEXPORT jint JNICALL Java_com_liyab_chat_LiyabNative_countTokens(JNIEnv* env, jclass, jlong h, jstring text) {
    if (h == 0) return -1;
    const std::string s = to_string(env, text);
    return liyab_engine_tokenize(handle(h), s.c_str(), 0, nullptr, 0);
}

JNIEXPORT jstring JNICALL Java_com_liyab_chat_LiyabNative_metadata(JNIEnv* env, jclass, jlong h, jstring key) {
    if (h == 0) return nullptr;
    const std::string k = to_string(env, key);
    const int64_t n = liyab_engine_metadata(handle(h), k.c_str(), nullptr, 0);
    if (n < 0) return nullptr;
    std::string value(static_cast<size_t>(n) + 1, '\0');
    liyab_engine_metadata(handle(h), k.c_str(), value.data(), value.size());
    return env->NewStringUTF(value.c_str());
}

// Returns {accelerator_busy_ms, storage_bytes_read, tokens_generated}, or null.
JNIEXPORT jdoubleArray JNICALL Java_com_liyab_chat_LiyabNative_counters(JNIEnv* env, jclass, jlong h) {
    liyab_engine_counters c{};
    if (h == 0 || liyab_engine_get_counters(handle(h), &c) != LIYAB_OK) return nullptr;
    const jdouble values[] = {c.accelerator_busy_ms, static_cast<jdouble>(c.storage_bytes_read),
                              static_cast<jdouble>(c.tokens_generated)};
    jdoubleArray out = env->NewDoubleArray(3);
    if (out != nullptr) env->SetDoubleArrayRegion(out, 0, 3, values);
    return out;
}

JNIEXPORT jstring JNICALL Java_com_liyab_chat_LiyabNative_supportedArchitectures(JNIEnv* env, jclass) {
    std::string text(liyab_supported_architectures(nullptr, 0) + 1, '\0');
    liyab_supported_architectures(text.data(), text.size());
    return env->NewStringUTF(text.c_str());
}

JNIEXPORT jstring JNICALL Java_com_liyab_chat_LiyabNative_describe(JNIEnv* env, jclass, jlong h) {
    if (h == 0) return env->NewStringUTF("");
    std::string text(liyab_engine_describe(handle(h), nullptr, 0) + 1, '\0');
    liyab_engine_describe(handle(h), text.data(), text.size());
    return env->NewStringUTF(text.c_str());
}

}  // extern "C"
