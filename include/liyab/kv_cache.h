// Liyab — paged, quantized KV cache with attention sinks, laid out per block.
//
// Paging (PagedAttention-style): K/V live in fixed-size pages of
// `page_tokens` positions taken from a pool on demand and mapped through
// logical→physical page tables. Memory therefore grows with the tokens
// actually cached instead of being reserved for the whole context up front,
// and because every page has the same size, freed pages are reused exactly —
// no fragmentation, no reallocation, no copying.
//
// Per-block layout. Attention blocks that see only the last W positions
// (KvCacheConfig::layer_windows: Gemma 2 / 3's windowed blocks, Phi-3's) need
// no rows older than their window, so layers are grouped by window: the
// layers that see every position form one group, the layers of each window
// another, and each group maps its own pages through its own table. A
// windowed group lets go of the pages no future query can see (after a
// rollback of at most `keep_back` positions, see release_unreachable()), so
// its memory is bounded by its window while the others grow with the
// context. All pages still come from one pool and have one size: a pool page
// holds `page_tokens` positions of U layers, U being the largest number that
// divides every group's layer count (every layer when there is one group),
// so a group's logical page is (its layers / U) pool pages and a page one
// group frees is reused by any other. With one group (no windowed layer) the
// layout, the bytes and the memory are exactly those of a single table.
//
// Sliding window with attention sinks (StreamingLLM, the engine's own
// window: `window` > 0): a query at position p attends to the first
// `sink_tokens` positions (the "anchors" that soak up attention mass) plus
// the last `window` positions, in every group (a windowed layer sees what
// both windows let it see). Pages no future query can see are returned to the
// pool, so the cache stays bounded for arbitrarily long generations; the
// sink pages are kept in every group. Positions keep their absolute RoPE
// angles: quality beyond the model's training context is not guaranteed.
//
// What leaves the cache (saved contexts, the engine's KV store, segment
// reuse) comes in two images:
//  * rows images (read_rows() and the functions after it): the rows of a run
//    of positions of the layers that keep every position (layer window 0),
//    [K | V][layer][slot][kv_head][row] over those layers in cache order;
//    with no windowed layer, the image of a whole page is that page's bytes;
//  * window pages (read_window_rows()): the windowed layers' rows of one
//    page. A context continues at position p only with the windowed rows of
//    [p - W + 1, p), and those pages are restored whole (window_pages_of()),
//    so a windowed page that is mapped always holds the rows of every
//    position of the context in it: the cache can tell where a context may
//    return to from its page tables alone. pin() keeps the pages of a
//    position's window while the context goes on (the Transformer's state
//    snapshots).
//
// Quantization: rows are stored in GGML block formats — F16, Q8_0,
// Q4_0 (symmetric INT4: scale only, 4.5 bits/value, 72% smaller than F16) or
// Q4_1 (asymmetric INT4: scale + minimum, 5 bits/value, 69% smaller), which
// tracks shifted distributions such as K channels with large biases better.
// Per-block scales make 75–80% vs F16 unreachable at 4 bits; vs F32 the
// savings are 86% / 84%.
//
// Thread-compatible: concurrent const calls (row reads during attention) are
// safe; anything that maps, frees or writes pages needs exclusive access.
#ifndef LIYAB_KV_CACHE_H
#define LIYAB_KV_CACHE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "liyab/types.h"

namespace liyab {

struct KvCacheConfig {
    int32_t n_layers = 0;
    int32_t n_head_kv = 0;
    int32_t head_dim = 0;   // must be a multiple of 32 for quantized types
    KvCacheType type = KvCacheType::Q8_0;
    int32_t max_positions = 4096;  // context limit when window == 0
    int32_t window = 0;            // the engine's sliding window, every layer; 0: full attention
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
    // Per layer, in cache order: the positions a query of that layer sees,
    // its own included (a model's windowed attention block), whose older rows
    // the cache does not keep; 0: every position. Empty: 0 for every layer.
    // Layers sharing a window share a group of pages (see the header
    // comment). A window that holds every position of the context should be
    // given as 0: it would only cost the bookkeeping of a windowed group.
    std::vector<int32_t> layer_windows;
};

class LIYAB_API KvCache {
public:
    // InvalidArgument for non-positive dimensions, a quantized type whose
    // rows are not whole blocks, a misplaced value_offset, sinks that do not
    // fit the first page, or layer_windows of the wrong size or negative.
    static Result<std::unique_ptr<KvCache>> create(const KvCacheConfig& config);

    // Bytes per cached position across all layers, K and V: what a cache
    // keeping every position of every layer holds per position.
    static size_t bytes_per_token(int32_t n_layers, int32_t n_head_kv, int32_t head_dim, KvCacheType type) noexcept;
    // The same for a whole configuration: K rows only when values_in_keys.
    // Every layer counts, windowed ones included (see kept_bytes()).
    static size_t bytes_per_token(const KvCacheConfig& config) noexcept;
    // What a cache of `config` holds at most once a context of `positions`
    // positions has been processed and release_unreachable(positions,
    // keep_back) has run, with no pins: whole pages, the layers that keep
    // every position for all of them, each windowed group for its window
    // plus `keep_back` (and the sink pages with the engine's window; the
    // engine's window bounds every group as it bounds windowed ones). The
    // automatic context length is sized with it.
    static size_t kept_bytes(const KvCacheConfig& config, int32_t positions, int32_t keep_back) noexcept;

    [[nodiscard]] const KvCacheConfig& config() const noexcept { return config_; }
    [[nodiscard]] DType dtype() const noexcept { return dtype_; }
    [[nodiscard]] int32_t window() const noexcept { return config_.window; }
    [[nodiscard]] int32_t sink_tokens() const noexcept { return config_.window > 0 ? config_.sink_tokens : 0; }
    // Whether some layer has a window of its own (layer_windows), i.e. the
    // cache keeps only the recent rows of those layers.
    [[nodiscard]] bool has_windowed_layers() const noexcept { return windowed_groups_ > 0; }
    // A hash of the per-block layout (which layers keep which window); 0
    // without windowed layers, so it changes nothing that hashes it for the
    // caches that have none.
    [[nodiscard]] uint64_t layout_fingerprint() const noexcept { return layout_fingerprint_; }

    // Positions visible to a query at `pos` through the engine's window:
    // [0, sink_end) ∪ [first, pos]. A windowed layer sees, of these, only
    // its last layer window positions.
    struct Visible {
        int32_t sink_end;
        int32_t first;
    };
    [[nodiscard]] Visible visible(int32_t pos) const noexcept;

    // Maps pages for every position < end that a query may still see:
    // pages released by release_unreachable() stay unmapped. ContextFull past
    // max_positions without the engine's window; OutOfMemory when a page
    // cannot be allocated. A page is zeroed when it is mapped.
    Status reserve(int32_t end);
    // Prepares the pages of a context of `n` positions about to be written as
    // a whole (a restored or loaded context), and pins `pins` (see pin()):
    // maps every page of [0, n) of the layers that keep every position
    // (mapped pages keep their rows), and of each windowed group only the
    // pages of the window before n and before each pin, freeing its others
    // and every page from n on: the pages window_pages_of(n, pins) lists. The
    // caller then writes each of those windowed pages whole (every position
    // of it below n: write_window_rows() or write_legacy_page()), since a
    // mapped windowed page is taken to hold them. ContextFull / OutOfMemory
    // as reserve(); InvalidArgument for a pin outside (0, n].
    Status reserve_context(int32_t n, std::span<const int32_t> pins);
    // The logical pages the windowed groups keep for a context of `n`
    // positions returning to `pins`: those of the window before n and before
    // each pin, ascending (every windowed group has the same window: see
    // window_of_windowed(); empty without windowed layers).
    [[nodiscard]] std::vector<int32_t> window_pages_of(int32_t n, std::span<const int32_t> pins) const;
    // Row access; the position's page must be mapped (reserve()). With
    // values_in_keys, `v` is ignored (it may be nullptr) and v_row() points
    // into the key row at channel value_offset.
    void store(int32_t layer, int32_t pos, const float* k, const float* v);
    [[nodiscard]] const uint8_t* k_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept;
    [[nodiscard]] const uint8_t* v_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept;

    // Multiplies channel c of the keys of positions [0, end) of layer l, KV
    // head h by factors[(l * n_head_kv + h) * head_dim + c], requantizing each
    // row (a saved context brought to other per-channel key scales; with
    // values_in_keys the values in those rows move with them). Windowed
    // groups: the positions whose pages are mapped. InvalidArgument when a
    // page of a layer keeping every position is unmapped, or for the wrong
    // number of factors.
    Status scale_keys(int32_t end, std::span<const float> factors);
    // Forgets positions >= n (rollback). Fails when a query at n would need
    // positions whose pages were already released (the engine's window, or a
    // windowed group's rows older than its window and not pinned).
    Status truncate(int32_t n);
    // The largest m <= n that truncate(m) accepts, for a context of at least
    // n positions (0 always is).
    [[nodiscard]] int32_t restorable(int32_t n) const;
    // Returns pages to the pool that no query at position >= n_past - keep_back
    // can see (keep_back = largest rollback the caller may still perform);
    // windowed groups keep the pages of their pins.
    void release_unreachable(int32_t n_past, int32_t keep_back);
    // Positions whose windows must stay (the Transformer's state snapshots):
    // each windowed group keeps the pages of [p - W + 1, p) of every pin p,
    // so truncate(p) keeps working while the context goes on. Replaces the
    // previous pins; pages only they held are
    // freed when no query can see them. Positions whose pages are already
    // released stay without them. No effect without windowed layers.
    void pin(std::span<const int32_t> positions);
    [[nodiscard]] const std::vector<int32_t>& pins() const noexcept { return pins_; }
    void clear() noexcept;

    // --- Rows images (saved contexts, the engine's KV store, segment reuse) --
    // A logical page here is the page of positions [p * page_tokens, (p + 1)
    // * page_tokens) of the layers that keep every position.
    //
    // Bytes of logical page `logical` laid out as a rows image of
    // page_tokens positions, in place: only when those layers form one pool
    // page per logical page (always without windowed layers); nullptr when
    // it is not mapped or spread over several pool pages (read_rows() then
    // copies it). reserve() zeroes a page when it maps it, so rows no
    // position has written hold zeros.
    [[nodiscard]] const uint8_t* page_data(int32_t logical) const noexcept;
    // The same page, writable, for restoring a saved state.
    [[nodiscard]] uint8_t* mutable_page_data(int32_t logical) noexcept;
    // Whether logical page `logical` of every layer keeping every position is
    // mapped (true when there is no such layer).
    [[nodiscard]] bool page_mapped(int32_t logical) const noexcept;
    // Rows of positions as a chunk image: the engine's KV store keeps a
    // context as chunks of whole pages (core/kv_store.h), the last one
    // possibly partial. An image of n positions of one page is laid out as a
    // rows image page, with n slots instead of page_tokens: [K | V][layer][slot
    // < n][kv_head][row] over the layers keeping every position (K only with
    // values_in_keys). Its size, rows_bytes(n), grows linearly with n; 0
    // when every layer is windowed.
    [[nodiscard]] size_t rows_bytes(int32_t n) const noexcept;
    // Bytes of a rows image of page_tokens positions (rows_bytes(page_tokens)).
    [[nodiscard]] size_t page_bytes() const noexcept { return page_bytes_; }
    // Copies the rows of the first n positions of logical page `logical` into
    // `out` (rows_bytes(n) bytes). InvalidArgument when the page is not mapped
    // or n is outside [1, page_tokens].
    Status read_rows(int32_t logical, int32_t n, uint8_t* out) const;
    // Writes slots [from, to) of `image`, an image of n positions as
    // read_rows() writes it, to the same slots of logical page `logical`,
    // which reserve() mapped. InvalidArgument when the page is not mapped,
    // the range is not within [0, n] or n is outside [1, page_tokens].
    Status write_rows(int32_t logical, const uint8_t* image, int32_t n, int32_t from, int32_t to);
    // Expands an image of n positions into a whole rows image page at `page`
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
    // `kv_head` at `slot` of `layer`, a layer that keeps every position:
    // head_row_bytes() long (with values_in_keys its value channels follow
    // from value_offset).
    [[nodiscard]] size_t image_key_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept;
    // The same for its value row (with values_in_keys, inside the key row).
    [[nodiscard]] size_t image_value_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept;
    // Whether `layer` keeps every position (it is in rows images).
    [[nodiscard]] bool keeps_every_position(int32_t layer) const noexcept;
    // Bytes of one cached row: head_dim values in the cache's format.
    [[nodiscard]] size_t head_row_bytes() const noexcept { return row_bytes_; }

    // --- Window pages (saved contexts; the engine's KV store keeps windowed rows page by page) --
    // The rows of the windowed layers of one logical page: per windowed group
    // (ascending window), [K | V][layer of the group][slot < page_tokens]
    // [kv_head][row]; window_page_bytes() in all (0 without windowed layers).
    [[nodiscard]] size_t window_page_bytes() const noexcept { return window_page_bytes_; }
    // The window of the windowed layers when they all have the same one, 0
    // without windowed layers, -1 when they have several.
    [[nodiscard]] int32_t window_of_windowed() const noexcept;
    // Whether logical page `logical` of every windowed group is mapped: its
    // rows are kept (the window before the end, a pin's window), those of
    // every position of the context in it.
    [[nodiscard]] bool window_page_mapped(int32_t logical) const noexcept;
    // Copies the windowed rows of the first n positions of logical page
    // `logical` into `out`, a window page image whose slots from n on are
    // zeroed (so the image depends on those n positions only).
    // InvalidArgument when a windowed group's page is not mapped or n is
    // outside [1, page_tokens].
    Status read_window_rows(int32_t logical, int32_t n, uint8_t* out) const;
    // Writes slots [from, to) of a window page image to logical page
    // `logical` of every windowed group, which reserve_context() mapped.
    // InvalidArgument otherwise.
    Status write_window_rows(int32_t logical, const uint8_t* image, int32_t from, int32_t to);
    // --- Older saved contexts -----------------------------------------------
    // Bytes of a page in the layout of a single table (every layer, the
    // layout before per-block groups: [K | V][layer][slot][kv_head][row]).
    [[nodiscard]] size_t legacy_page_bytes() const noexcept { return legacy_page_bytes_; }
    // Writes such a page of logical page `logical` (saved before per-block
    // groups) into the groups: every layer's rows of the positions whose
    // pages are mapped (the layers keeping every position must be).
    // InvalidArgument otherwise.
    Status write_legacy_page(int32_t logical, const uint8_t* page);

    // Pool pages (one size for every group; unit_bytes() each).
    [[nodiscard]] int32_t pages_in_use() const noexcept;
    [[nodiscard]] int32_t pages_allocated() const noexcept { return static_cast<int32_t>(pool_.size()); }
    [[nodiscard]] size_t unit_bytes() const noexcept { return unit_bytes_; }
    [[nodiscard]] size_t bytes_in_use() const noexcept { return static_cast<size_t>(pages_in_use()) * unit_bytes_; }
    [[nodiscard]] size_t bytes_allocated() const noexcept { return pool_.size() * unit_bytes_; }

private:
    KvCache() = default;

    // Layers sharing a window, and the pool pages holding their rows.
    struct Group {
        int32_t window = 0;           // 0: every position
        std::vector<int32_t> members;  // its layers, ascending
        int32_t units = 1;            // pool pages per logical page (layers / unit_layers_)
        std::vector<int32_t> table;   // [logical * units + unit] -> pool page, kUnmapped
        int32_t released_below = 0;   // logical pages in [sink pages, this) were let go (but pinned ones)
        std::vector<std::pair<int32_t, int32_t>> pinned;  // windowed: page ranges [a, b] of the pins
    };
    // Where a layer's rows are.
    struct Place {
        int32_t group = 0;
        int32_t unit = 0;         // pool page of the group's logical page
        int32_t layer = 0;        // inside that pool page
        int32_t image_layer = -1; // in rows images; -1: windowed
    };

    [[nodiscard]] uint8_t* row(int32_t layer, int32_t pos, int32_t kv_head, bool value) const noexcept;
    [[nodiscard]] uint8_t* unit_ptr(const Group& g, size_t logical, int32_t unit) const noexcept;
    [[nodiscard]] bool mapped(const Group& g, size_t logical) const noexcept;
    Status map_page(Group& g, size_t logical);
    void free_page(Group& g, size_t logical) noexcept;
    [[nodiscard]] bool releases(const Group& g) const noexcept { return config_.window > 0 || g.window > 0; }
    [[nodiscard]] bool is_pinned(const Group& g, int32_t logical) const noexcept;
    // Oldest position (sinks aside) a query at `pos` of group g sees.
    [[nodiscard]] int32_t first_seen(const Group& g, int32_t pos) const noexcept;
    // Whether every page of the positions a query at n of group g needs
    // (before n) is mapped.
    [[nodiscard]] bool can_continue(const Group& g, int32_t n) const noexcept;
    [[nodiscard]] int32_t sink_page_count() const noexcept;
    void set_pinned_ranges(Group& g) const;
    // The run of plane `plane` of a layer inside its pool page: slot 0's rows.
    [[nodiscard]] size_t run_offset(const Place& p, int32_t plane) const noexcept;
    // Copies `count` slots of every plane and rows-image layer between logical
    // page `logical` (from slot `page_first`) and an image of `slots`
    // positions (from slot `first`); to the image when `out`.
    void copy_page_slots(size_t logical, int32_t page_first, uint8_t* image, int32_t slots, int32_t first,
                         int32_t count, bool out) const noexcept;
    // The same between two images (a page image is one of page_tokens slots).
    void copy_slots(const uint8_t* src, int32_t src_slots, int32_t src_first, uint8_t* dst, int32_t dst_slots,
                    int32_t dst_first, int32_t count) const noexcept;
    // Copies `count` slots of every windowed group's planes and layers
    // between logical page `logical` (from slot `page_first`) and a window
    // rows image of `slots` positions (from slot `first`).
    void copy_window_slots(size_t logical, int32_t page_first, uint8_t* image, int32_t slots, int32_t first,
                           int32_t count, bool out) const noexcept;

    // table entries: >= 0 pool page, or kUnmapped.
    static constexpr int32_t kUnmapped = -1;

    KvCacheConfig config_;
    DType dtype_ = DType::Q8_0;
    size_t row_bytes_ = 0;
    size_t value_skip_ = 0;     // values_in_keys: bytes of a key row before its value channels
    size_t slot_bytes_ = 0;     // n_head_kv rows: one position of one layer
    int32_t planes_ = 2;        // 1 with values_in_keys
    int32_t unit_layers_ = 0;   // layers per pool page
    size_t unit_bytes_ = 0;     // a pool page
    size_t page_bytes_ = 0;     // a rows image of page_tokens positions
    size_t legacy_page_bytes_ = 0;
    size_t window_page_bytes_ = 0;
    int32_t image_layers_ = 0;  // layers keeping every position
    int32_t full_group_ = -1;   // their group, or -1
    int32_t windowed_groups_ = 0;
    uint64_t layout_fingerprint_ = 0;
    std::vector<Group> groups_;  // the every-position group first, then windows ascending
    std::vector<Place> places_;  // per layer
    std::vector<int32_t> image_order_;  // rows-image layer -> layer
    std::vector<int32_t> pins_;
    std::vector<std::unique_ptr<uint8_t[]>> pool_;  // physical pages (never shrinks; reused)
    std::vector<int32_t> free_;                     // free physical page ids
};

}  // namespace liyab

#endif  // LIYAB_KV_CACHE_H
