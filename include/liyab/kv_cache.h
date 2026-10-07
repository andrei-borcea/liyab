// Liyab — paged, quantized KV cache with attention sinks.
//
// Paging (PagedAttention-style): K/V live in fixed-size pages of
// `page_tokens` positions (all layers per page) taken from a pool on demand
// and mapped through a logical→physical page table. Memory therefore grows
// with the tokens actually cached instead of being reserved for the whole
// context up front, and because every page has the same size, freed pages
// are reused exactly — no fragmentation, no reallocation, no copying.
//
// Sliding window with attention sinks (StreamingLLM): with `window` > 0 a
// query at position p attends to the first `sink_tokens` positions (the
// "anchors" that soak up attention mass) plus the last `window` positions.
// Pages that no future query can see are returned to the pool, so the cache
// stays bounded for arbitrarily long generations. Positions keep their
// absolute RoPE angles: quality beyond the model's training context is not
// guaranteed.
//
// Quantization: rows are stored in GGML block formats — F16, Q8_0,
// Q4_0 (symmetric INT4: scale only, 4.5 bits/value, 72% smaller than F16) or
// Q4_1 (asymmetric INT4: scale + minimum, 5 bits/value, 69% smaller), which
// tracks shifted distributions such as K channels with large biases better.
// Per-block scales make 75–80% vs F16 unreachable at 4 bits; vs F32 the
// savings are 86% / 84%.
#ifndef LIYAB_KV_CACHE_H
#define LIYAB_KV_CACHE_H

#include <cstdint>
#include <memory>
#include <vector>

#include "liyab/types.h"

namespace liyab {

struct KvCacheConfig {
    int32_t n_layers = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim = 0;   // must be a multiple of 32 for quantized types
    KvCacheType type = KvCacheType::Q8_0;
    int32_t max_positions = 4096;  // context limit when window == 0
    int32_t window = 0;            // 0: full attention
    int32_t sink_tokens = 8;       // anchors kept with a sliding window
    int32_t page_tokens = 64;      // positions per page
};

class LIYAB_API KvCache {
public:
    static Result<std::unique_ptr<KvCache>> create(const KvCacheConfig& config);

    // Bytes per cached position across all layers, K and V.
    static size_t bytes_per_token(int32_t n_layers, int32_t n_head_kv, int32_t head_dim, KvCacheType type) noexcept;

    [[nodiscard]] const KvCacheConfig& config() const noexcept { return config_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] int32_t window() const noexcept { return config_.window; }
    [[nodiscard]] int32_t sink_tokens() const noexcept { return config_.window > 0 ? config_.sink_tokens : 0; }

    // Positions visible to a query at `pos`: [0, sink_end) ∪ [first, pos].
    struct Visible {
        int32_t sink_end;
        int32_t first;
    };
    [[nodiscard]] Visible visible(int32_t pos) const noexcept;

    // Maps pages for every position < end. ContextFull past max_positions
    // without a window; OutOfMemory when a page cannot be allocated.
    Status reserve(int32_t end);
    // Row access; the position's page must be mapped (reserve()).
    void store(int32_t layer, int32_t pos, const float* k, const float* v);
    [[nodiscard]] const uint8_t* k_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept;
    [[nodiscard]] const uint8_t* v_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept;

    // Forgets positions >= n (rollback). Fails when a query at n would need
    // positions whose pages were already released.
    Status truncate(int32_t n);
    // Returns pages to the pool that no query at position >= n_past - keep_back
    // can see (keep_back = largest rollback the caller may still perform).
    void release_unreachable(int32_t n_past, int32_t keep_back);
    void clear() noexcept;

    [[nodiscard]] int32_t pages_in_use() const noexcept;
    [[nodiscard]] int32_t pages_allocated() const noexcept { return static_cast<int32_t>(pool_.size()); }
    [[nodiscard]] size_t page_bytes() const noexcept { return page_bytes_; }
    [[nodiscard]] size_t bytes_in_use() const noexcept { return static_cast<size_t>(pages_in_use()) * page_bytes_; }
    [[nodiscard]] size_t bytes_allocated() const noexcept { return pool_.size() * page_bytes_; }

private:
    KvCache() = default;
    [[nodiscard]] uint8_t* row(int32_t layer, int32_t pos, int32_t kv_head, bool value) const noexcept;
    void free_page(size_t logical) noexcept;
    [[nodiscard]] int32_t sink_page_count() const noexcept;

    KvCacheConfig config_;
    DType dtype_ = DType::Q8_0;
    size_t row_bytes_ = 0;
    size_t page_bytes_ = 0;
    std::vector<std::unique_ptr<uint8_t[]>> pool_;  // physical pages (never shrinks; reused)
    std::vector<int32_t> free_;                     // free physical page ids
    std::vector<int32_t> table_;                    // logical page -> physical id, -1 = unmapped
    int32_t released_below_ = 0;                    // logical pages < this (beyond sinks) were released
};

}  // namespace liyab

#endif  // LIYAB_KV_CACHE_H
