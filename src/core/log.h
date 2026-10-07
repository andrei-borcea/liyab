// Liyab — minimal logging (internal). Android routes to logcat, elsewhere stderr.
#ifndef LIYAB_CORE_LOG_H
#define LIYAB_CORE_LOG_H

namespace liyab::log {

enum class Level : int { Debug = 0, Info = 1, Warn = 2, Error = 3, Off = 4 };

using Sink = void (*)(int level, const char* message, void* user_data);

void set_level(Level level) noexcept;
// Additional sink (e.g. an app's debug console). Called from any thread.
void set_sink(Sink sink, void* user_data) noexcept;
bool enabled(Level level) noexcept;
void write(Level level, const char* fmt, ...) noexcept __attribute__((format(printf, 2, 3)));

}  // namespace liyab::log

#define LIYAB_LOG_(level, ...)                                                      \
    do {                                                                            \
        if (::liyab::log::enabled(level)) ::liyab::log::write(level, __VA_ARGS__);  \
    } while (0)
#define LIYAB_LOG_DEBUG(...) LIYAB_LOG_(::liyab::log::Level::Debug, __VA_ARGS__)
#define LIYAB_LOG_INFO(...) LIYAB_LOG_(::liyab::log::Level::Info, __VA_ARGS__)
#define LIYAB_LOG_WARN(...) LIYAB_LOG_(::liyab::log::Level::Warn, __VA_ARGS__)
#define LIYAB_LOG_ERROR(...) LIYAB_LOG_(::liyab::log::Level::Error, __VA_ARGS__)

#endif  // LIYAB_CORE_LOG_H
