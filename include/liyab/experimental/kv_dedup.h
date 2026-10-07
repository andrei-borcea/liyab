// Liyab (experimental) — cross-session KV-cache deduplication (persistent
// prefix cache).
//
// Chat sessions usually start with the same tokens (system prompt, tool
// descriptions, few-shot examples). After a prefill, the KV pages that are
// completely filled by the prompt are written to a snapshot file whose name
// is the xxHash64 of (model fingerprint, KV layout, prefix tokens). A later
// session whose prompt starts with the same tokens maps that file with mmap
// and attaches its pages directly to the paged KV cache — read-only and
// without copying — so only the remaining tokens are prefilled.
//
// Invariants that keep this correct:
//  * Only whole pages are shared (prefix length = k * page_tokens); new
//    positions always land in fresh, owned pages, never in the mapping.
//  * The key covers everything the KV bytes depend on: model file
//    fingerprint, KV dtype, page size and geometry, and every prefix token.
//  * Snapshots are written to a temporary file and renamed, so a crash never
//    leaves a truncated snapshot under a valid name; headers are validated.
// Scope: full-attention caches only (sliding windows release pages), no
// speculative decoding (the draft keeps its own cache), no eviction policy
// for the directory yet.
#ifndef LIYAB_EXPERIMENTAL_KV_DEDUP_H
#define LIYAB_EXPERIMENTAL_KV_DEDUP_H

#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "liyab/kv_cache.h"
#include "liyab/mmap_loader.h"
#include "liyab/types.h"

namespace liyab::experimental {

// xxHash64 (Yann Collet's XXH64), for snapshot keys.
LIYAB_API uint64_t xxhash64(const void* data, size_t length, uint64_t seed = 0) noexcept;

// Fingerprint of a model file: size + hash of its first MiB (GGUF header,
// metadata and tensor table, which change whenever the weights do).
LIYAB_API uint64_t model_fingerprint(const MappedFile& file) noexcept;

class LIYAB_API KvDedup {
public:
    // `directory` must exist and be writable.
    static Result<std::unique_ptr<KvDedup>> create(std::string directory, uint64_t model_fingerprint);

    // Attaches the longest snapshotted page-aligned prefix of `tokens`, at
    // most `max_tokens` long, to the (empty) `cache`. Returns the number of
    // positions restored (0 when nothing matched). The mapping stays alive
    // until the next restore() or destruction.
    int32_t restore(std::span<const int32_t> tokens, int32_t max_tokens, KvCache& cache);

    // Snapshots the first `n` positions (rounded down to whole pages) unless
    // a snapshot for that prefix already exists. Returns positions saved.
    Result<int32_t> save(std::span<const int32_t> tokens, int32_t n, const KvCache& cache);

    [[nodiscard]] std::string path_for(std::span<const int32_t> prefix, const KvCache& cache) const;
    [[nodiscard]] const std::string& directory() const noexcept { return directory_; }

private:
    KvDedup() = default;
    [[nodiscard]] uint64_t key(std::span<const int32_t> prefix, const KvCache& cache) const noexcept;

    std::string directory_;
    uint64_t fingerprint_ = 0;
    std::unique_ptr<MappedFile> attached_;
};

}  // namespace liyab::experimental

#endif  // LIYAB_EXPERIMENTAL_KV_DEDUP_H
