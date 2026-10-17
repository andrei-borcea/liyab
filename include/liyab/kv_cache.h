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
#include <span>
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
    // Values are not stored on their own: a value row is channels
    // [value_offset, head_dim) of its key row. Multi-head latent attention
    // (DeepSeek) attends over keys [RoPE part | latent] whose values are the
    // latent itself, so one row serves both and the cache holds half the
    // bytes. value_offset must keep value rows block-aligned in a quantized
    // cache (a multiple of 32) and leave at least one channel.
    bool values_in_keys = false;
    int32_t value_offset = 0;
};

class LIYAB_API KvCache {
public:
    static Result<std::unique_ptr<KvCache>> create(const KvCacheConfig& config);

    // Bytes per cached position across all layers, K and V.
    static size_t bytes_per_token(int32_t n_layers, int32_t n_head_kv, int32_t head_dim, KvCacheType type) noexcept;
    // The same for a whole configuration: K rows only when values_in_keys.
    static size_t bytes_per_token(const KvCacheConfig& config) noexcept;

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
    // Row access; the position's page must be mapped (reserve()). With
    // values_in_keys, `v` is ignored (it may be nullptr) and v_row() points
    // into the key row at channel value_offset.
    void store(int32_t layer, int32_t pos, const float* k, const float* v);
    [[nodiscard]] const uint8_t* k_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept;
    [[nodiscard]] const uint8_t* v_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept;

    // Multiplies channel c of the keys of positions [0, end) of layer l, KV
    // head h by factors[(l * n_head_kv + h) * head_dim + c], requantizing each
    // row (a saved context brought to other per-channel key scales; with
    // values_in_keys the values in those rows move with them).
    // InvalidArgument when a position's page is unmapped, or for the wrong
    // number of factors.
    Status scale_keys(int32_t end, std::span<const float> factors);
    // Forgets positions >= n (rollback). Fails when a query at n would need
    // positions whose pages were already released.
    Status truncate(int32_t n);
    // Returns pages to the pool that no query at position >= n_past - keep_back
    // can see (keep_back = largest rollback the caller may still perform).
    void release_unreachable(int32_t n_past, int32_t keep_back);
    void clear() noexcept;

    // --- Pages as bytes (saved contexts, the engine's KV store) ------------
    // Bytes of logical page `logical` (all layers, K then V; K only with
    // values_in_keys), or nullptr when it is not mapped.
    [[nodiscard]] const uint8_t* page_data(int32_t logical) const noexcept;
    // The same page, writable, for restoring a saved state: a page that
    // reserve() mapped; nullptr when it is not mapped.
    [[nodiscard]] uint8_t* mutable_page_data(int32_t logical) noexcept;
    // Rows of positions as a chunk image: the engine's KV store keeps a
    // context as chunks of one page each (core/kv_store.h), the last one
    // possibly partial. An image of n positions of one page is laid out as the
    // page is, with n slots instead of page_tokens: [K | V][layer][slot <
    // n][kv_head][row] (K only with values_in_keys), so the image of a whole
    // page is that page's bytes. Its size, rows_bytes(n), grows linearly with n.
    [[nodiscard]] size_t rows_bytes(int32_t n) const noexcept;
    // Copies the rows of the first n positions of logical page `logical` into
    // `out` (rows_bytes(n) bytes). InvalidArgument when the page is not mapped
    // or n is outside [1, page_tokens].
    Status read_rows(int32_t logical, int32_t n, uint8_t* out) const;
    // Writes slots [from, to) of `image`, an image of n positions as
    // read_rows() writes it, to the same slots of logical page `logical`,
    // which reserve() mapped. InvalidArgument when the page is not mapped,
    // the range is not within [0, n] or n is outside [1, page_tokens].
    Status write_rows(int32_t logical, const uint8_t* image, int32_t n, int32_t from, int32_t to);
    // Expands an image of n positions into a whole page's layout at `page`
    // (page_bytes(); the slots past n zeroed), as write_state() writes pages.
    void image_to_page(const uint8_t* image, int32_t n, uint8_t* page) const noexcept;
    // Rows at any position, for KV reused away from where it was computed
    // (the engine's segment reuse): a segment of a prompt rarely starts on
    // a page boundary. Copies the rows of positions [pos, pos + count) into
    // slots [first, first + count) of `image`, an image of `slots` positions
    // laid out as read_rows() writes one, across as many pages as they span.
    // InvalidArgument when a position's page is not mapped or the slots are
    // not inside the image.
    Status read_rows_at(int32_t pos, int32_t count, uint8_t* image, int32_t slots, int32_t first) const;
    // Writes slots [first, first + count) of `image` (an image of `slots`
    // positions) to positions [pos, pos + count), whose pages reserve()
    // mapped. InvalidArgument otherwise.
    Status write_rows_at(int32_t pos, const uint8_t* image, int32_t slots, int32_t first, int32_t count);
    // Byte offset, in an image of `slots` positions, of the key row of
    // `kv_head` at `slot` of `layer`: head_row_bytes() long (with
    // values_in_keys its value channels follow from value_offset).
    [[nodiscard]] size_t image_key_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept;
    // The same for its value row (with values_in_keys, inside the key row).
    [[nodiscard]] size_t image_value_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept;
    // Bytes of one cached row: head_dim values in the cache's format.
    [[nodiscard]] size_t head_row_bytes() const noexcept { return row_bytes_; }

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
    [[nodiscard]] uint8_t* page_ptr(size_t logical) const noexcept;
    // Copies `count` slots of every plane and layer from an image of
    // `src_slots` positions (from slot src_first) to one of `dst_slots`
    // (to slot dst_first); a page is an image of page_tokens slots.
    void copy_slots(const uint8_t* src, int32_t src_slots, int32_t src_first, uint8_t* dst, int32_t dst_slots,
                    int32_t dst_first, int32_t count) const noexcept;

    // table_ entries: >= 0 pool page, or kUnmapped.
    static constexpr int32_t kUnmapped = -1;

    KvCacheConfig config_;
    DType dtype_ = DType::Q8_0;
    size_t row_bytes_ = 0;
    size_t value_skip_ = 0;  // values_in_keys: bytes of a key row before its value channels
    size_t page_bytes_ = 0;
    std::vector<std::unique_ptr<uint8_t[]>> pool_;  // physical pages (never shrinks; reused)
    std::vector<int32_t> free_;                     // free physical page ids
    std::vector<int32_t> table_;                    // logical page -> physical id (see kUnmapped)
    int32_t released_below_ = 0;                    // logical pages < this (beyond sinks) were released
};

}  // namespace liyab

#endif  // LIYAB_KV_CACHE_H
