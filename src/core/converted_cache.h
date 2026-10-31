// The converted-weights cache: files written by MmapLoader::save_converted() in EngineConfig::cache_dir, one per
// (model, recipe), "weights-<id>.bin". Each is about the size of the model, so unlike the other cache_dir files
// (a few KB) the directory is bounded by a byte cap, least recently used out first.
#pragma once

#include <cstdint>
#include <string>

namespace liyab::converted_cache {

// "<dir>/weights-<id>.bin".
std::string path(const std::string& dir, uint64_t id);

// Marks the file used now (a hit), so eviction removes the files nobody loads first.
void touch(const std::string& path);

// Makes room for a file of `need` bytes: creates `dir`, removes abandoned "*.tmp" files older than an hour, then the
// least recently used weights files until the directory holds at most `cap_bytes` with the new file. Returns false
// (removing nothing) when the file alone exceeds the cap, or the volume would be left under 1 GiB free.
bool make_room(const std::string& dir, uint64_t need, uint64_t cap_bytes);

}  // namespace liyab::converted_cache
