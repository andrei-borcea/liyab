// Liyab — small values measured once per model and device, kept in EngineConfig::cache_dir (internal).
//
// A measurement that costs a fraction of a second to seconds (latent attention's form ratio, what a verification
// pass costs, which route a class of passes takes) is kept as "<prefix>-<id>.bin" = magic, version, id, a
// trivially copyable payload. The id is the caller's hash of everything the measurement depends on (the model file's
// fingerprint, the KV format, the kernel set, the route, the threads...), so a model is told apart from another by
// its file and measured again when any of that changes. A file that is missing, damaged or of another version reads
// as "not measured".
//
// The directory is bounded: after a write, the `max_files` most recently used files of this prefix stay (a read
// refreshes the file's time), the older ones are removed. Without a directory a value measured in this process is
// kept in memory, so every engine of the process takes the same decision. Failures to write only cost a new
// measurement at the next load and are logged, not returned.
//
// Thread-safe; writes are a temporary file renamed over the final one, so a reader never sees half a file.
#ifndef LIYAB_CORE_MEASURED_CACHE_H
#define LIYAB_CORE_MEASURED_CACHE_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/log.h"
#include "core/os.h"

namespace liyab {

template <class T>
class MeasuredCache {
    static_assert(std::is_trivially_copyable_v<T>, "the payload is written as raw bytes");

public:
    // `prefix`: the files' name before the id ("verify"); `magic`: 4 bytes; `version`: bump when the payload's
    // meaning changes; `what`: for the log ("lookup drafts").
    MeasuredCache(const char* prefix, const char (&magic)[5], uint32_t version, size_t max_files, const char* what)
        : prefix_(prefix), version_(version), max_files_(max_files), what_(what) {
        std::memcpy(magic_, magic, sizeof magic_);
    }

    // The value kept for `id` (in `dir`, else in this process), or nothing. `valid` rejects a payload whose values are
    // out of range.
    template <class Valid>
    std::optional<T> find(const std::string& dir, uint64_t id, Valid valid) const {
        if (dir.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto it = memory_.find(id);
            return it != memory_.end() ? std::optional<T>(it->second) : std::nullopt;
        }
        const std::string path = path_of(dir, id);
        std::ifstream in(os::utf8_path(path), std::ios::binary);
        char magic[4] = {};
        uint32_t version = 0;
        uint64_t stored_id = 0;
        T value{};
        in.read(magic, sizeof magic);
        in.read(reinterpret_cast<char*>(&version), sizeof version);
        in.read(reinterpret_cast<char*>(&stored_id), sizeof stored_id);
        in.read(reinterpret_cast<char*>(&value), sizeof value);
        if (!in || std::memcmp(magic, magic_, sizeof magic) != 0 || version != version_ || stored_id != id ||
            !valid(value)) {
            return std::nullopt;
        }
        std::error_code ignored;
        std::filesystem::last_write_time(os::utf8_path(path), std::filesystem::file_time_type::clock::now(), ignored);
        return value;
    }

    // Keeps `value` for `id`.
    void keep(const std::string& dir, uint64_t id, const T& value) const {
        if (dir.empty()) {
            std::lock_guard<std::mutex> lock(mutex_);
            memory_[id] = value;
            return;
        }
        std::error_code ec;
        std::filesystem::create_directories(os::utf8_path(dir), ec);
        const std::string path = path_of(dir, id);
        // A time-based suffix, so two processes writing the same id do not share a temporary file.
        const std::string tmp = path + "." + std::to_string(static_cast<unsigned long long>(
                                                  std::filesystem::file_time_type::clock::now().time_since_epoch().count())) +
                                ".tmp";
        {
            std::ofstream out(os::utf8_path(tmp), std::ios::binary | std::ios::trunc);
            out.write(magic_, sizeof magic_);
            out.write(reinterpret_cast<const char*>(&version_), sizeof version_);
            out.write(reinterpret_cast<const char*>(&id), sizeof id);
            out.write(reinterpret_cast<const char*>(&value), sizeof value);
            if (!out) {
                LIYAB_LOG_WARN("%s: cannot write %s", what_, tmp.c_str());
                std::filesystem::remove(os::utf8_path(tmp), ec);
                return;
            }
        }
        std::filesystem::rename(os::utf8_path(tmp), os::utf8_path(path), ec);
        if (ec) {
            LIYAB_LOG_WARN("%s: cannot keep the measurement in %s: %s", what_, dir.c_str(), ec.message().c_str());
            std::filesystem::remove(os::utf8_path(tmp), ec);
            return;
        }
        std::vector<std::pair<std::filesystem::file_time_type, std::filesystem::path>> kept;
        std::filesystem::directory_iterator it(os::utf8_path(dir), ec), end;
        const std::string head = prefix_ + "-";
        for (; !ec && it != end; it.increment(ec)) {
            const std::string name = os::path_utf8(it->path().filename());
            if (!name.starts_with(head)) continue;
            std::error_code ignored;
            const auto time = std::filesystem::last_write_time(it->path(), ignored);
            if (name.ends_with(".bin")) {
                kept.emplace_back(time, it->path());
            } else if (name.ends_with(".tmp") && time < std::filesystem::file_time_type::clock::now() - std::chrono::hours(1)) {
                std::filesystem::remove(it->path(), ignored);  // left by a process that died while writing
            }
        }
        if (kept.size() <= max_files_) return;
        std::sort(kept.begin(), kept.end());  // oldest first
        for (size_t i = 0; i + max_files_ < kept.size(); ++i) std::filesystem::remove(kept[i].second, ec);
    }

private:
    std::string path_of(const std::string& dir, uint64_t id) const {
        char name[48];
        std::snprintf(name, sizeof name, "%s-%016llx.bin", prefix_.c_str(), static_cast<unsigned long long>(id));
        return os::path_utf8(os::utf8_path(dir) / name);
    }

    std::string prefix_;
    char magic_[4];
    uint32_t version_;
    size_t max_files_;
    const char* what_;
    mutable std::mutex mutex_;
    mutable std::map<uint64_t, T> memory_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_MEASURED_CACHE_H
