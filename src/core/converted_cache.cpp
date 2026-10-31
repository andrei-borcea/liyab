#include "core/converted_cache.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <utility>
#include <vector>

#include "core/log.h"
#include "core/os.h"

namespace liyab::converted_cache {

namespace {
namespace fs = std::filesystem;
constexpr uint64_t kFreeFloor = uint64_t{1} << 30;  // never fill the volume
}  // namespace

std::string path(const std::string& dir, uint64_t id) {
    char name[40];
    std::snprintf(name, sizeof name, "weights-%016" PRIx64 ".bin", id);
    return os::path_utf8(os::utf8_path(dir) / name);
}

void touch(const std::string& file) {
    std::error_code ec;
    fs::last_write_time(os::utf8_path(file), fs::file_time_type::clock::now(), ec);
}

bool make_room(const std::string& dir, uint64_t need, uint64_t cap_bytes) {
    if (need > cap_bytes) return false;
    std::error_code ec;
    fs::create_directories(os::utf8_path(dir), ec);
    if (ec) return false;
    std::vector<std::pair<fs::file_time_type, fs::path>> kept;
    uint64_t total = 0;
    const auto now = fs::file_time_type::clock::now();
    fs::directory_iterator it(os::utf8_path(dir), ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string name = os::path_utf8(it->path().filename());
        if (!name.starts_with("weights-")) continue;
        std::error_code ignored;
        const auto when = fs::last_write_time(it->path(), ignored);
        if (name.ends_with(".tmp")) {
            if (now - when > std::chrono::hours(1)) fs::remove(it->path(), ignored);  // a writer that died
        } else if (name.ends_with(".bin")) {
            kept.emplace_back(when, it->path());
            total += fs::file_size(it->path(), ignored);
        }
    }
    std::sort(kept.begin(), kept.end());  // least recently used first
    std::vector<fs::path> evict;
    uint64_t freed = 0;
    for (size_t i = 0; total - freed + need > cap_bytes && i < kept.size(); ++i) {
        std::error_code ignored;
        freed += fs::file_size(kept[i].second, ignored);
        evict.push_back(kept[i].second);
    }
    const auto space = fs::space(os::utf8_path(dir), ec);
    if (!ec && space.available + freed < need + kFreeFloor) return false;
    for (const fs::path& p : evict) {
        std::error_code ignored;
        fs::remove(p, ignored);
        LIYAB_LOG_INFO("converted weights: evicted %s", os::path_utf8(p.filename()).c_str());
    }
    return true;
}

}  // namespace liyab::converted_cache
