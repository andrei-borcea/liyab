#include "liyab/experimental/kv_dedup.h"

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

namespace liyab::experimental {

// ---------------------------------------------------------------------------
// Snapshot files
// ---------------------------------------------------------------------------
namespace {

constexpr uint32_t kMagic = 0x564B594C;  // "LYKV"
constexpr uint32_t kVersion = 1;

struct Header {
    uint32_t magic;
    uint32_t version;
    uint64_t key;
    uint32_t n_tokens;
    uint32_t page_tokens;
    uint64_t page_bytes;
    uint32_t n_pages;
    uint32_t header_bytes;  // page data starts here (OS-page aligned)
};

size_t os_page() { return static_cast<size_t>(sysconf(_SC_PAGESIZE)); }

}  // namespace

Result<std::unique_ptr<KvDedup>> KvDedup::create(std::string directory, uint64_t fingerprint) {
    struct stat st {};
    if (stat(directory.c_str(), &st) != 0 || !S_ISDIR(st.st_mode) || access(directory.c_str(), W_OK) != 0) {
        return Status(ErrorCode::InvalidArgument, "KV dedup directory '" + directory + "' is not a writable directory");
    }
    std::unique_ptr<KvDedup> dedup(new KvDedup());
    dedup->directory_ = std::move(directory);
    dedup->fingerprint_ = fingerprint;
    return dedup;
}

uint64_t KvDedup::key(std::span<const int32_t> prefix, const KvCache& cache) const noexcept {
    const KvCacheConfig& c = cache.config();
    const uint64_t layout[] = {fingerprint_,
                               static_cast<uint64_t>(cache.dtype()),
                               static_cast<uint64_t>(c.page_tokens),
                               static_cast<uint64_t>(c.n_layers),
                               static_cast<uint64_t>(c.n_head_kv),
                               static_cast<uint64_t>(c.head_dim),
                               cache.page_bytes()};
    const uint64_t seed = xxhash64(layout, sizeof layout);
    return xxhash64(prefix.data(), prefix.size_bytes(), seed);
}

std::string KvDedup::path_for(std::span<const int32_t> prefix, const KvCache& cache) const {
    char name[40];
    std::snprintf(name, sizeof name, "/%016llx.lykv", static_cast<unsigned long long>(key(prefix, cache)));
    return directory_ + name;
}

int32_t KvDedup::restore(std::span<const int32_t> tokens, int32_t max_tokens, KvCache& cache) {
    attached_.reset();
    const int32_t page = cache.config().page_tokens;
    if (cache.window() > 0) return 0;
    const auto limit = std::min<int64_t>(max_tokens, static_cast<int64_t>(tokens.size()));
    for (int32_t n = static_cast<int32_t>(limit / page * page); n >= page; n -= page) {
        const auto prefix = tokens.first(static_cast<size_t>(n));
        auto file = MappedFile::open(path_for(prefix, cache));
        if (!file) continue;
        Header h{};
        if (file.value()->size() < sizeof h) continue;
        std::memcpy(&h, file.value()->data(), sizeof h);
        const uint64_t expected_size = h.header_bytes + static_cast<uint64_t>(h.n_pages) * h.page_bytes;
        if (h.magic != kMagic || h.version != kVersion || h.key != key(prefix, cache) ||
            h.n_tokens != static_cast<uint32_t>(n) || h.page_tokens != static_cast<uint32_t>(page) ||
            h.page_bytes != cache.page_bytes() || h.n_pages != static_cast<uint32_t>(n / page) ||
            file.value()->size() != expected_size) {
            continue;  // stale or foreign file: ignore
        }
        std::vector<const uint8_t*> pages(h.n_pages);
        for (uint32_t p = 0; p < h.n_pages; ++p) {
            pages[p] = file.value()->data() + h.header_bytes + static_cast<size_t>(p) * h.page_bytes;
        }
        if (!cache.attach_external_pages(pages).is_ok()) return 0;
        // Pages are read in place on the next forward pass: prefault them now
        // so the first token does not pay for page faults.
        file.value()->advise_willneed(h.header_bytes, static_cast<size_t>(h.n_pages) * h.page_bytes);
        attached_ = std::move(file).value();
        return n;
    }
    return 0;
}

Result<int32_t> KvDedup::save(std::span<const int32_t> tokens, int32_t n, const KvCache& cache) {
    const int32_t page = cache.config().page_tokens;
    if (cache.window() > 0) return 0;
    n = std::min<int32_t>(n, static_cast<int32_t>(tokens.size())) / page * page;
    if (n <= 0) return 0;
    const auto prefix = tokens.first(static_cast<size_t>(n));
    const std::string path = path_for(prefix, cache);
    if (access(path.c_str(), F_OK) == 0) return 0;  // already snapshotted

    Header h{kMagic, kVersion, key(prefix, cache), static_cast<uint32_t>(n), static_cast<uint32_t>(page),
             cache.page_bytes(), static_cast<uint32_t>(n / page), static_cast<uint32_t>(std::max<size_t>(os_page(), 4096))};
    const std::string tmp = path + ".tmp" + std::to_string(getpid());
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (f == nullptr) return Status(ErrorCode::IoError, "cannot create '" + tmp + "'");
    std::vector<uint8_t> header(h.header_bytes, 0);
    std::memcpy(header.data(), &h, sizeof h);
    bool ok = std::fwrite(header.data(), 1, header.size(), f) == header.size();
    for (uint32_t p = 0; ok && p < h.n_pages; ++p) {
        const uint8_t* data = cache.page_data(static_cast<int32_t>(p));
        ok = data != nullptr && std::fwrite(data, 1, h.page_bytes, f) == h.page_bytes;
    }
    ok = std::fflush(f) == 0 && ok;
    ok = std::fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return Status(ErrorCode::IoError, "cannot write KV snapshot '" + path + "'");
    }
    return n;
}

}  // namespace liyab::experimental
