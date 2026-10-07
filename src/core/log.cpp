#include "core/log.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace liyab::log {

namespace {

Level initial_level() noexcept {
    const char* env = std::getenv("LIYAB_LOG");
    if (env == nullptr) return Level::Warn;
    if (std::strcmp(env, "debug") == 0) return Level::Debug;
    if (std::strcmp(env, "info") == 0) return Level::Info;
    if (std::strcmp(env, "error") == 0) return Level::Error;
    if (std::strcmp(env, "off") == 0) return Level::Off;
    return Level::Warn;
}

std::atomic<int>& level_ref() noexcept {
    static std::atomic<int> level{static_cast<int>(initial_level())};
    return level;
}

struct SinkSlot {
    std::atomic<Sink> sink{nullptr};
    std::atomic<void*> user{nullptr};
};

SinkSlot& sink_slot() noexcept {
    static SinkSlot slot;
    return slot;
}

}  // namespace

void set_sink(Sink sink, void* user_data) noexcept {
    sink_slot().user.store(user_data, std::memory_order_relaxed);
    sink_slot().sink.store(sink, std::memory_order_release);
}

void set_level(Level level) noexcept { level_ref().store(static_cast<int>(level), std::memory_order_relaxed); }

bool enabled(Level level) noexcept {
    return static_cast<int>(level) >= level_ref().load(std::memory_order_relaxed);
}

void write(Level level, const char* fmt, ...) noexcept {
    char buffer[1024];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buffer, sizeof buffer, fmt, args);
    va_end(args);
    if (Sink sink = sink_slot().sink.load(std::memory_order_acquire)) {
        sink(static_cast<int>(level), buffer, sink_slot().user.load(std::memory_order_relaxed));
    }
#if defined(__ANDROID__)
    static constexpr int kPriority[] = {ANDROID_LOG_DEBUG, ANDROID_LOG_INFO, ANDROID_LOG_WARN, ANDROID_LOG_ERROR,
                                        ANDROID_LOG_SILENT};
    __android_log_write(kPriority[static_cast<int>(level)], "liyab", buffer);
#else
    static constexpr const char* kTag[] = {"debug", "info", "warn", "error", "off"};
    std::fprintf(stderr, "[liyab %s] %s\n", kTag[static_cast<int>(level)], buffer);
#endif
}

}  // namespace liyab::log
