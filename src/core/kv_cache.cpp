#include "liyab/kv_cache.h"

#include <algorithm>
#include <cstring>
#include <new>
#include <numeric>
#include <string>
#include <vector>

#include "core/quant.h"

namespace liyab {

namespace {

DType to_dtype(KvCacheType type) noexcept {
    switch (type) {
        case KvCacheType::F16: return DType::F16;
        case KvCacheType::Q8_0: return DType::Q8_0;
        case KvCacheType::Q4_0: return DType::Q4_0;
        case KvCacheType::Q4_1: return DType::Q4_1;
    }
    return DType::F16;
}

int32_t layer_window(const KvCacheConfig& config, int32_t layer) noexcept {
    return config.layer_windows.empty() ? 0 : config.layer_windows[static_cast<size_t>(layer)];
}

}  // namespace

size_t KvCache::bytes_per_token(int32_t n_layers, int32_t n_head_kv, int32_t head_dim, KvCacheType type) noexcept {
    return 2 * static_cast<size_t>(n_layers) * static_cast<size_t>(n_head_kv) *
           dtype_row_bytes(to_dtype(type), head_dim);
}

size_t KvCache::bytes_per_token(const KvCacheConfig& config) noexcept {
    const size_t both = bytes_per_token(config.n_layers, config.n_head_kv, config.head_dim, config.type);
    return config.values_in_keys ? both / 2 : both;
}

size_t KvCache::kept_bytes(const KvCacheConfig& config, int32_t positions, int32_t keep_back) noexcept {
    auto made = create(config);
    if (!made) return static_cast<size_t>(std::max(positions, 0)) * bytes_per_token(config);
    const KvCache& kv = *made.value();
    const int32_t t = config.page_tokens;
    const int32_t pages = positions > 0 ? (positions - 1) / t + 1 : 0;
    size_t units = 0;
    for (const Group& g : kv.groups_) {
        int32_t kept = pages;
        if (kv.releases(g)) {
            // The oldest position a query after a rollback of keep_back sees,
            // to the end: at most window - 1 + keep_back positions, over one
            // page more than they fill when they start inside one.
            const int32_t reach = g.window > 0 && config.window > 0 ? std::min(g.window, config.window)
                                                                    : std::max(g.window, config.window);
            const int32_t span = reach - 1 + std::max(keep_back, 0);
            kept = std::min(pages, (span + t - 1) / t + 1 + kv.sink_page_count());
        }
        units += static_cast<size_t>(kept) * static_cast<size_t>(g.units);
    }
    return units * kv.unit_bytes_;
}

Result<std::unique_ptr<KvCache>> KvCache::create(const KvCacheConfig& config) {
    if (config.n_layers <= 0 || config.n_head_kv <= 0 || config.head_dim <= 0 || config.page_tokens <= 0 ||
        config.max_positions <= 0 || config.window < 0 || config.sink_tokens < 0) {
        return Status(ErrorCode::InvalidArgument, "invalid KV cache dimensions");
    }
    const DType dtype = to_dtype(config.type);
    if (quant::is_block_quantized(dtype) && config.head_dim % quant::kBlock != 0) {
        return Status(ErrorCode::InvalidArgument, "quantized KV cache needs head_dim to be a multiple of 32");
    }
    if (config.values_in_keys &&
        (config.value_offset < 0 || config.value_offset >= config.head_dim ||
         (quant::is_block_quantized(dtype) && config.value_offset % quant::kBlock != 0))) {
        return Status(ErrorCode::InvalidArgument, "values in keys need a block-aligned value_offset inside the row");
    }
    if (config.window > 0 && config.sink_tokens > config.page_tokens) {
        return Status(ErrorCode::InvalidArgument, "sink_tokens must fit in the first page");
    }
    if (!config.layer_windows.empty() &&
        (config.layer_windows.size() != static_cast<size_t>(config.n_layers) ||
         std::any_of(config.layer_windows.begin(), config.layer_windows.end(), [](int32_t w) { return w < 0; }))) {
        return Status(ErrorCode::InvalidArgument, "one non-negative window per KV cache layer");
    }
    std::unique_ptr<KvCache> cache(new KvCache());
    KvCache& kv = *cache;
    kv.config_ = config;
    kv.dtype_ = dtype;
    kv.row_bytes_ = dtype_row_bytes(dtype, config.head_dim);
    kv.value_skip_ = config.values_in_keys ? dtype_row_bytes(dtype, config.value_offset) : 0;
    kv.slot_bytes_ = static_cast<size_t>(config.n_head_kv) * kv.row_bytes_;
    kv.planes_ = config.values_in_keys ? 1 : 2;

    // Groups: the layers keeping every position first, then one per window,
    // ascending; a pool page holds U layers, U dividing every group's size.
    std::vector<int32_t> windows;
    for (int32_t l = 0; l < config.n_layers; ++l) windows.push_back(layer_window(config, l));
    std::vector<int32_t> distinct = windows;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    kv.groups_.resize(distinct.size());
    for (size_t i = 0; i < distinct.size(); ++i) {
        kv.groups_[i].window = distinct[i];
        if (distinct[i] == 0) kv.full_group_ = static_cast<int32_t>(i);
    }
    kv.windowed_groups_ = static_cast<int32_t>(distinct.size()) - (kv.full_group_ >= 0 ? 1 : 0);
    kv.places_.resize(static_cast<size_t>(config.n_layers));
    for (int32_t l = 0; l < config.n_layers; ++l) {
        const auto g = static_cast<int32_t>(std::lower_bound(distinct.begin(), distinct.end(), windows[static_cast<size_t>(l)]) -
                                            distinct.begin());
        Group& group = kv.groups_[static_cast<size_t>(g)];
        Place& p = kv.places_[static_cast<size_t>(l)];
        p.group = g;
        p.layer = static_cast<int32_t>(group.members.size());  // index in the group for now
        group.members.push_back(l);
        if (group.window == 0) {
            p.image_layer = kv.image_layers_++;
            kv.image_order_.push_back(l);
        }
    }
    int32_t unit = 0;
    for (const Group& g : kv.groups_) unit = std::gcd(unit, static_cast<int32_t>(g.members.size()));
    kv.unit_layers_ = unit;
    for (Group& g : kv.groups_) g.units = static_cast<int32_t>(g.members.size()) / unit;
    for (Place& p : kv.places_) {
        p.unit = p.layer / unit;
        p.layer %= unit;
    }
    const auto planes = static_cast<size_t>(kv.planes_);
    const auto tokens = static_cast<size_t>(config.page_tokens);
    kv.unit_bytes_ = planes * static_cast<size_t>(unit) * tokens * kv.slot_bytes_;
    kv.page_bytes_ = planes * static_cast<size_t>(kv.image_layers_) * tokens * kv.slot_bytes_;
    kv.legacy_page_bytes_ = planes * static_cast<size_t>(config.n_layers) * tokens * kv.slot_bytes_;
    for (const Group& g : kv.groups_) {
        if (g.window == 0) continue;
        kv.window_page_bytes_ += planes * g.members.size() * tokens * kv.slot_bytes_;
    }
    if (kv.windowed_groups_ > 0) {
        uint64_t h = 0xcbf29ce484222325ULL;  // FNV-1a over the windows, layer by layer
        for (const int32_t w : windows) {
            for (int b = 0; b < 4; ++b) {
                h ^= static_cast<uint64_t>((static_cast<uint32_t>(w) >> (8 * b)) & 0xFFu);
                h *= 0x100000001b3ULL;
            }
        }
        kv.layout_fingerprint_ = h == 0 ? 1 : h;
    }
    return cache;
}

KvCache::Visible KvCache::visible(int32_t pos) const noexcept {
    if (config_.window <= 0) return {0, 0};
    const int32_t first = std::max(0, pos - config_.window + 1);
    return {std::min(config_.sink_tokens, first), first};
}

int32_t KvCache::first_seen(const Group& g, int32_t pos) const noexcept {
    int32_t first = config_.window > 0 ? pos - config_.window + 1 : 0;
    if (g.window > 0) first = std::max(first, pos - g.window + 1);
    return std::max(first, 0);
}

uint8_t* KvCache::unit_ptr(const Group& g, size_t logical, int32_t unit) const noexcept {
    return pool_[static_cast<size_t>(g.table[logical * static_cast<size_t>(g.units) + static_cast<size_t>(unit)])].get();
}

bool KvCache::mapped(const Group& g, size_t logical) const noexcept {
    // Logical pages are mapped and freed whole: the first unit tells.
    const size_t at = logical * static_cast<size_t>(g.units);
    return at < g.table.size() && g.table[at] != kUnmapped;
}

Status KvCache::map_page(Group& g, size_t logical) {
    const auto units = static_cast<size_t>(g.units);
    if (g.table.size() < (logical + 1) * units) g.table.resize((logical + 1) * units, kUnmapped);
    for (size_t u = 0; u < units; ++u) {
        int32_t& slot = g.table[logical * units + u];
        if (slot != kUnmapped) continue;
        if (free_.empty()) {
            uint8_t* page = new (std::nothrow) uint8_t[unit_bytes_];
            if (page == nullptr) {
                free_page(g, logical);  // never half a logical page
                return Status(ErrorCode::OutOfMemory, "cannot allocate a KV cache page");
            }
            pool_.emplace_back(page);
            free_.push_back(static_cast<int32_t>(pool_.size() - 1));
        }
        slot = free_.back();
        free_.pop_back();
        // Zeroed when mapped, new or recycled. Attention never reads a row
        // before it is written, but whole pages leave the process (save_state,
        // the prefix cache's chunks): rows past the context's end must hold
        // zeros, not another context's rows or the heap's leftovers, so the
        // same context always gives the same bytes and nothing else is written
        // to storage. One page per page_tokens positions: noise next to
        // computing them.
        std::memset(pool_[static_cast<size_t>(slot)].get(), 0, unit_bytes_);
    }
    return Status::ok();
}

void KvCache::free_page(Group& g, size_t logical) noexcept {
    const auto units = static_cast<size_t>(g.units);
    for (size_t u = 0; u < units; ++u) {
        const size_t at = logical * units + u;
        if (at >= g.table.size() || g.table[at] == kUnmapped) continue;
        free_.push_back(g.table[at]);
        g.table[at] = kUnmapped;
    }
}

bool KvCache::is_pinned(const Group& g, int32_t logical) const noexcept {
    for (const auto& [a, b] : g.pinned) {
        if (logical >= a && logical <= b) return true;
    }
    return false;
}

void KvCache::set_pinned_ranges(Group& g) const {
    g.pinned.clear();
    if (g.window == 0) return;
    for (const int32_t p : pins_) {
        if (p > 0) g.pinned.emplace_back(first_seen(g, p) / config_.page_tokens, (p - 1) / config_.page_tokens);
    }
}

bool KvCache::can_continue(const Group& g, int32_t n) const noexcept {
    if (n <= 0 || !releases(g)) return true;
    const int32_t last = (n - 1) / config_.page_tokens;
    for (int32_t p = first_seen(g, n) / config_.page_tokens; p <= last; ++p) {
        if (!mapped(g, static_cast<size_t>(p))) return false;
    }
    return true;
}

Status KvCache::reserve(int32_t end) {
    if (config_.window <= 0 && end > config_.max_positions) {
        return Status(ErrorCode::ContextFull, "context of " + std::to_string(config_.max_positions) + " tokens is full");
    }
    if (end <= 0) return Status::ok();
    const auto pages = static_cast<size_t>((end - 1) / config_.page_tokens + 1);
    const auto sink_pages = static_cast<size_t>(sink_page_count());
    for (Group& g : groups_) {
        const bool release = releases(g);
        for (size_t p = 0; p < pages; ++p) {
            // Released pages stay unmapped: no query can see them any more.
            if (mapped(g, p) || (release && p >= sink_pages && p < static_cast<size_t>(g.released_below))) continue;
            LIYAB_RETURN_IF_ERROR(map_page(g, p));
        }
    }
    return Status::ok();
}

Status KvCache::reserve_context(int32_t n, std::span<const int32_t> pins) {
    if (config_.window <= 0 && n > config_.max_positions) {
        return Status(ErrorCode::ContextFull, "context of " + std::to_string(config_.max_positions) + " tokens is full");
    }
    for (const int32_t p : pins) {
        if (p <= 0 || p > n) return Status(ErrorCode::InvalidArgument, "a pinned position outside the context");
    }
    pins_.assign(pins.begin(), pins.end());
    std::sort(pins_.begin(), pins_.end());
    pins_.erase(std::unique(pins_.begin(), pins_.end()), pins_.end());
    const int32_t pages = n > 0 ? (n - 1) / config_.page_tokens + 1 : 0;
    const int32_t sink_pages = sink_page_count();
    for (Group& g : groups_) {
        set_pinned_ranges(g);
        // Only windowed groups keep less than [0, n): a group bounded by the
        // engine's window alone maps every page, as reserve() after clear()
        // does, and lets go of them once the context moves on.
        const bool windowed = g.window > 0;
        const int32_t first = windowed && n > 0 ? first_seen(g, n) / config_.page_tokens : 0;
        const auto table_pages = static_cast<int32_t>(g.table.size() / static_cast<size_t>(g.units));
        for (int32_t p = 0; p < std::max(pages, table_pages); ++p) {
            const bool needed = p < pages && (!windowed || p < sink_pages || p >= first || is_pinned(g, p));
            if (needed) {
                if (!mapped(g, static_cast<size_t>(p))) LIYAB_RETURN_IF_ERROR(map_page(g, static_cast<size_t>(p)));
            } else {
                free_page(g, static_cast<size_t>(p));
            }
        }
        g.released_below = first;
    }
    return Status::ok();
}

int32_t KvCache::sink_page_count() const noexcept {
    const int32_t sinks = sink_tokens();
    return sinks > 0 ? (sinks - 1) / config_.page_tokens + 1 : 0;
}

uint8_t* KvCache::row(int32_t layer, int32_t pos, int32_t kv_head, bool value) const noexcept {
    const Place& place = places_[static_cast<size_t>(layer)];
    const Group& g = groups_[static_cast<size_t>(place.group)];
    const auto logical = static_cast<size_t>(pos / config_.page_tokens);
    const auto slot = static_cast<size_t>(pos % config_.page_tokens);
    uint8_t* page = unit_ptr(g, logical, place.unit);
    // Pool page layout: [K | V], each [layer][slot][kv_head][row] over its U
    // layers; only K with values_in_keys, where a value row starts
    // value_skip_ bytes into its key row (whole blocks, so it decodes on its
    // own). With one group, a pool page is the whole logical page: the
    // single-table layout, byte for byte.
    const size_t index = (static_cast<size_t>(place.layer) * static_cast<size_t>(config_.page_tokens) + slot) *
                             static_cast<size_t>(config_.n_head_kv) + static_cast<size_t>(kv_head);
    if (config_.values_in_keys) return page + index * row_bytes_ + (value ? value_skip_ : 0);
    return page + (value ? unit_bytes_ / 2 : 0) + index * row_bytes_;
}

void KvCache::store(int32_t layer, int32_t pos, const float* k, const float* v) {
    for (int32_t g = 0; g < config_.n_head_kv; ++g) {
        quant::quantize_row(dtype_, k + g * config_.head_dim, row(layer, pos, g, false), config_.head_dim);
        if (config_.values_in_keys) continue;
        quant::quantize_row(dtype_, v + g * config_.head_dim, row(layer, pos, g, true), config_.head_dim);
    }
}

const uint8_t* KvCache::k_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept {
    return row(layer, pos, kv_head, false);
}

const uint8_t* KvCache::v_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept {
    return row(layer, pos, kv_head, true);
}

Status KvCache::scale_keys(int32_t end, std::span<const float> factors) {
    const auto hd = static_cast<size_t>(config_.head_dim);
    if (factors.size() != static_cast<size_t>(config_.n_layers) * static_cast<size_t>(config_.n_head_kv) * hd) {
        return Status(ErrorCode::InvalidArgument, "one key factor per layer, KV head and channel");
    }
    const int32_t t = config_.page_tokens;
    for (int32_t p = 0; p * t < end; ++p) {
        if (!page_mapped(p)) return Status(ErrorCode::InvalidArgument, "keys on a page that is not mapped");
    }
    std::vector<float> row_values(hd);
    for (int32_t layer = 0; layer < config_.n_layers; ++layer) {
        const Group& group = groups_[static_cast<size_t>(places_[static_cast<size_t>(layer)].group)];
        for (int32_t g = 0; g < config_.n_head_kv; ++g) {
            const float* f = factors.data() + (static_cast<size_t>(layer) * static_cast<size_t>(config_.n_head_kv) +
                                               static_cast<size_t>(g)) * hd;
            for (int32_t pos = 0; pos < end; ++pos) {
                // A windowed group's released rows are gone: only its mapped pages.
                if (!mapped(group, static_cast<size_t>(pos / t))) continue;
                uint8_t* r = row(layer, pos, g, false);
                quant::dequantize_row(dtype_, r, row_values.data(), config_.head_dim);
                for (size_t i = 0; i < hd; ++i) row_values[i] *= f[i];
                quant::quantize_row(dtype_, row_values.data(), r, config_.head_dim);
            }
        }
    }
    return Status::ok();
}

Status KvCache::truncate(int32_t n) {
    if (n < 0) return Status(ErrorCode::InvalidArgument, "negative KV cache length");
    for (const Group& g : groups_) {
        if (!can_continue(g, n)) {
            return Status(ErrorCode::InvalidArgument, "rollback reaches KV pages that were already released");
        }
    }
    // Pages that start at or after n hold no surviving position.
    const auto keep = static_cast<size_t>((n + config_.page_tokens - 1) / config_.page_tokens);
    for (Group& g : groups_) {
        for (size_t p = keep; p * static_cast<size_t>(g.units) < g.table.size(); ++p) free_page(g, p);
        // Pages between the window before n and the old watermark come back
        // into use as positions from n on are written again.
        g.released_below = n == 0 ? 0 : std::min(g.released_below, releases(g) ? first_seen(g, n) / config_.page_tokens : 0);
    }
    // Pins past n describe positions that no longer exist.
    if (!pins_.empty() && pins_.back() > n) {
        std::erase_if(pins_, [n](int32_t p) { return p > n; });
        for (Group& g : groups_) set_pinned_ranges(g);
    }
    return Status::ok();
}

int32_t KvCache::restorable(int32_t n) const {
    if (n <= 0) return 0;
    bool all = true;
    for (const Group& g : groups_) all = all && can_continue(g, n);
    if (all) return n;
    // Per group, the first page of the run of mapped pages ending at each
    // page (-1 when that page is unmapped): m is accepted when the run ending
    // at m's last page reaches back to the oldest page a query at m sees.
    const int32_t t = config_.page_tokens;
    const auto last = static_cast<size_t>((n - 1) / t);
    std::vector<std::vector<int32_t>> run_start;
    std::vector<const Group*> checked;
    for (const Group& g : groups_) {
        if (!releases(g)) continue;
        std::vector<int32_t> starts(last + 1, -1);
        for (size_t e = 0; e <= last; ++e) {
            if (!mapped(g, e)) continue;
            starts[e] = e > 0 && starts[e - 1] >= 0 ? starts[e - 1] : static_cast<int32_t>(e);
        }
        run_start.push_back(std::move(starts));
        checked.push_back(&g);
    }
    for (int32_t m = n - 1; m > 0; --m) {
        bool ok = true;
        for (size_t i = 0; i < checked.size() && ok; ++i) {
            const int32_t start = run_start[i][static_cast<size_t>((m - 1) / t)];
            ok = start >= 0 && start <= first_seen(*checked[i], m) / t;
        }
        if (ok) return m;
    }
    return 0;
}

void KvCache::release_unreachable(int32_t n_past, int32_t keep_back) {
    const int32_t sink_pages = sink_page_count();
    for (Group& g : groups_) {
        if (!releases(g)) continue;
        // Oldest non-sink position any future query (after a rollback of at
        // most keep_back tokens) can still see.
        const int32_t oldest = first_seen(g, std::max(0, n_past - keep_back));
        const int32_t limit = oldest / config_.page_tokens;  // pages strictly below are unreachable
        for (int32_t p = std::max(sink_pages, g.released_below); p < limit; ++p) {
            if (!is_pinned(g, p)) free_page(g, static_cast<size_t>(p));
        }
        g.released_below = std::max(g.released_below, limit);
    }
}

void KvCache::pin(std::span<const int32_t> positions) {
    pins_.clear();
    for (const int32_t p : positions) {
        if (p > 0) pins_.push_back(p);
    }
    std::sort(pins_.begin(), pins_.end());
    pins_.erase(std::unique(pins_.begin(), pins_.end()), pins_.end());
    const int32_t sink_pages = sink_page_count();
    for (Group& g : groups_) {
        if (g.window == 0) continue;
        set_pinned_ranges(g);
        // Pages below the watermark were kept for a pin only: those of pins
        // gone go now.
        for (int32_t p = sink_pages; p < g.released_below; ++p) {
            if (!is_pinned(g, p)) free_page(g, static_cast<size_t>(p));
        }
    }
}

void KvCache::clear() noexcept {
    for (Group& g : groups_) {
        for (size_t p = 0; p * static_cast<size_t>(g.units) < g.table.size(); ++p) free_page(g, p);
        g.released_below = 0;
        g.pinned.clear();
    }
    pins_.clear();
}

int32_t KvCache::pages_in_use() const noexcept {
    return static_cast<int32_t>(pool_.size() - free_.size());
}

// ---------------------------------------------------------------------------
// Rows images
// ---------------------------------------------------------------------------

bool KvCache::keeps_every_position(int32_t layer) const noexcept {
    return layer >= 0 && layer < config_.n_layers && places_[static_cast<size_t>(layer)].image_layer >= 0;
}

bool KvCache::page_mapped(int32_t logical) const noexcept {
    if (logical < 0) return false;
    return full_group_ < 0 || mapped(groups_[static_cast<size_t>(full_group_)], static_cast<size_t>(logical));
}

const uint8_t* KvCache::page_data(int32_t logical) const noexcept {
    if (full_group_ < 0 || !page_mapped(logical)) return nullptr;
    const Group& g = groups_[static_cast<size_t>(full_group_)];
    // One pool page per logical page: it is the rows image page itself.
    return g.units == 1 ? unit_ptr(g, static_cast<size_t>(logical), 0) : nullptr;
}

uint8_t* KvCache::mutable_page_data(int32_t logical) noexcept {
    return const_cast<uint8_t*>(static_cast<const KvCache*>(this)->page_data(logical));
}

size_t KvCache::rows_bytes(int32_t n) const noexcept {
    return page_bytes_ / static_cast<size_t>(config_.page_tokens) * static_cast<size_t>(std::max(n, 0));
}

size_t KvCache::run_offset(const Place& p, int32_t plane) const noexcept {
    return (static_cast<size_t>(plane) * static_cast<size_t>(unit_layers_) + static_cast<size_t>(p.layer)) *
           static_cast<size_t>(config_.page_tokens) * slot_bytes_;
}

void KvCache::copy_page_slots(size_t logical, int32_t page_first, uint8_t* image, int32_t slots, int32_t first,
                              int32_t count, bool out) const noexcept {
    if (full_group_ < 0 || count <= 0) return;
    const Group& g = groups_[static_cast<size_t>(full_group_)];
    const int32_t t = config_.page_tokens;
    if (g.units == 1 && page_first == 0 && first == 0 && count == t && slots == t) {
        // A whole page of a one-pool-page group: the image is its bytes.
        uint8_t* page = unit_ptr(g, logical, 0);
        std::memcpy(out ? image : page, out ? page : image, page_bytes_);
        return;
    }
    // Every plane and layer is a run of contiguous slots on both sides.
    const size_t bytes = static_cast<size_t>(count) * slot_bytes_;
    for (int32_t plane = 0; plane < planes_; ++plane) {
        for (int32_t f = 0; f < image_layers_; ++f) {
            const Place& p = places_[static_cast<size_t>(image_order_[static_cast<size_t>(f)])];
            uint8_t* page = unit_ptr(g, logical, p.unit) + run_offset(p, plane) + static_cast<size_t>(page_first) * slot_bytes_;
            uint8_t* im = image + ((static_cast<size_t>(plane) * static_cast<size_t>(image_layers_) + static_cast<size_t>(f)) *
                                       static_cast<size_t>(slots) + static_cast<size_t>(first)) * slot_bytes_;
            std::memcpy(out ? im : page, out ? page : im, bytes);
        }
    }
}

void KvCache::copy_slots(const uint8_t* src, int32_t src_slots, int32_t src_first, uint8_t* dst, int32_t dst_slots,
                         int32_t dst_first, int32_t count) const noexcept {
    // A slot is n_head_kv rows; each plane (K, then V) holds a run of slots
    // per layer, so one copy per plane and layer moves `count` contiguous slots.
    for (int32_t run = 0; run < planes_ * image_layers_; ++run) {
        std::memcpy(dst + (static_cast<size_t>(run) * static_cast<size_t>(dst_slots) + static_cast<size_t>(dst_first)) * slot_bytes_,
                    src + (static_cast<size_t>(run) * static_cast<size_t>(src_slots) + static_cast<size_t>(src_first)) * slot_bytes_,
                    static_cast<size_t>(count) * slot_bytes_);
    }
}

Status KvCache::read_rows(int32_t logical, int32_t n, uint8_t* out) const {
    if (!page_mapped(logical) || n < 1 || n > config_.page_tokens) {
        return Status(ErrorCode::InvalidArgument, "read_rows: the page is not mapped or the count is out of range");
    }
    copy_page_slots(static_cast<size_t>(logical), 0, out, n, 0, n, true);
    return Status::ok();
}

Status KvCache::write_rows(int32_t logical, const uint8_t* image, int32_t n, int32_t from, int32_t to) {
    if (!page_mapped(logical) || n < 1 || n > config_.page_tokens || from < 0 || to > n || from > to) {
        return Status(ErrorCode::InvalidArgument, "write_rows: the page is not mapped or the range is out of bounds");
    }
    if (from < to) {
        copy_page_slots(static_cast<size_t>(logical), from, const_cast<uint8_t*>(image), n, from, to - from, false);
    }
    return Status::ok();
}

void KvCache::image_to_page(const uint8_t* image, int32_t n, uint8_t* page) const noexcept {
    if (n >= config_.page_tokens) {
        std::memcpy(page, image, page_bytes_);
        return;
    }
    std::memset(page, 0, page_bytes_);
    copy_slots(image, n, 0, page, config_.page_tokens, 0, n);
}

Status KvCache::read_rows_at(int32_t pos, int32_t count, uint8_t* image, int32_t slots, int32_t first) const {
    if (pos < 0 || count < 0 || first < 0 || slots < 1 || first + count > slots) {
        return Status(ErrorCode::InvalidArgument, "read_rows_at: the slots are not inside the image");
    }
    // One copy per run of positions inside a page.
    for (int32_t done = 0; done < count;) {
        const int32_t p = pos + done;
        const int32_t slot = p % config_.page_tokens;
        const int32_t run = std::min(count - done, config_.page_tokens - slot);
        if (!page_mapped(p / config_.page_tokens)) return Status(ErrorCode::InvalidArgument, "read_rows_at: a page is not mapped");
        copy_page_slots(static_cast<size_t>(p / config_.page_tokens), slot, image, slots, first + done, run, true);
        done += run;
    }
    return Status::ok();
}

Status KvCache::write_rows_at(int32_t pos, const uint8_t* image, int32_t slots, int32_t first, int32_t count) {
    if (pos < 0 || count < 0 || first < 0 || slots < 1 || first + count > slots) {
        return Status(ErrorCode::InvalidArgument, "write_rows_at: the slots are not inside the image");
    }
    for (int32_t done = 0; done < count;) {
        const int32_t p = pos + done;
        const int32_t slot = p % config_.page_tokens;
        const int32_t run = std::min(count - done, config_.page_tokens - slot);
        if (!page_mapped(p / config_.page_tokens)) return Status(ErrorCode::InvalidArgument, "write_rows_at: a page is not mapped");
        copy_page_slots(static_cast<size_t>(p / config_.page_tokens), slot, const_cast<uint8_t*>(image), slots, first + done,
                        run, false);
        done += run;
    }
    return Status::ok();
}

size_t KvCache::image_key_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept {
    // The K plane: [layer][slot][kv_head][row] over the rows-image layers.
    const auto f = static_cast<size_t>(places_[static_cast<size_t>(layer)].image_layer);
    return (f * static_cast<size_t>(slots) + static_cast<size_t>(slot)) * slot_bytes_ +
           static_cast<size_t>(kv_head) * row_bytes_;
}

size_t KvCache::image_value_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept {
    const size_t key = image_key_offset(slots, layer, slot, kv_head);
    if (config_.values_in_keys) return key + value_skip_;
    // The V plane follows the K plane of the whole image.
    return static_cast<size_t>(image_layers_) * static_cast<size_t>(slots) * slot_bytes_ + key;
}

// ---------------------------------------------------------------------------
// Window pages
// ---------------------------------------------------------------------------

std::vector<int32_t> KvCache::window_pages_of(int32_t n, std::span<const int32_t> pins) const {
    std::vector<int32_t> pages;
    const int32_t t = config_.page_tokens;
    for (const Group& g : groups_) {
        if (g.window == 0 || n <= 0) continue;
        auto add = [&](int32_t pos) {
            if (pos <= 0) return;
            for (int32_t p = first_seen(g, pos) / t; p <= (pos - 1) / t; ++p) pages.push_back(p);
        };
        add(n);
        for (const int32_t p : pins) add(p);
    }
    std::sort(pages.begin(), pages.end());
    pages.erase(std::unique(pages.begin(), pages.end()), pages.end());
    return pages;
}
int32_t KvCache::window_of_windowed() const noexcept {
    if (windowed_groups_ == 0) return 0;
    return windowed_groups_ == 1 ? groups_.back().window : -1;
}

bool KvCache::window_page_mapped(int32_t logical) const noexcept {
    if (logical < 0) return false;
    for (const Group& g : groups_) {
        if (g.window > 0 && !mapped(g, static_cast<size_t>(logical))) return false;
    }
    return true;
}

void KvCache::copy_window_slots(size_t logical, int32_t page_first, uint8_t* image, int32_t slots, int32_t first,
                                int32_t count, bool out) const noexcept {
    if (count <= 0) return;
    const size_t bytes = static_cast<size_t>(count) * slot_bytes_;
    uint8_t* at = image;
    for (const Group& g : groups_) {
        if (g.window == 0) continue;
        const auto layers = static_cast<int32_t>(g.members.size());
        for (int32_t plane = 0; plane < planes_; ++plane) {
            for (int32_t j = 0; j < layers; ++j) {
                const Place& p = places_[static_cast<size_t>(g.members[static_cast<size_t>(j)])];
                uint8_t* page = unit_ptr(g, logical, p.unit) + run_offset(p, plane) + static_cast<size_t>(page_first) * slot_bytes_;
                uint8_t* im = at + ((static_cast<size_t>(plane) * static_cast<size_t>(layers) + static_cast<size_t>(j)) *
                                        static_cast<size_t>(slots) + static_cast<size_t>(first)) * slot_bytes_;
                std::memcpy(out ? im : page, out ? page : im, bytes);
            }
        }
        at += static_cast<size_t>(planes_) * g.members.size() * static_cast<size_t>(slots) * slot_bytes_;
    }
}

Status KvCache::read_window_rows(int32_t logical, int32_t n, uint8_t* out) const {
    if (!window_page_mapped(logical) || n < 1 || n > config_.page_tokens) {
        return Status(ErrorCode::InvalidArgument, "read_window_rows: the page is not kept or the count is out of range");
    }
    const int32_t t = config_.page_tokens;
    if (n < t) std::memset(out, 0, window_page_bytes_);  // slots past n: zeros, whatever the page holds there
    copy_window_slots(static_cast<size_t>(logical), 0, out, t, 0, n, true);
    return Status::ok();
}

Status KvCache::write_window_rows(int32_t logical, const uint8_t* image, int32_t from, int32_t to) {
    if (!window_page_mapped(logical) || from < 0 || to > config_.page_tokens || from > to) {
        return Status(ErrorCode::InvalidArgument, "write_window_rows: the page is not mapped or the range is out of bounds");
    }
    copy_window_slots(static_cast<size_t>(logical), from, const_cast<uint8_t*>(image), config_.page_tokens, from,
                      to - from, false);
    return Status::ok();
}

Status KvCache::write_legacy_page(int32_t logical, const uint8_t* page) {
    if (logical < 0 || !page_mapped(logical)) {
        return Status(ErrorCode::InvalidArgument, "write_legacy_page: the page is not mapped");
    }
    const size_t run = static_cast<size_t>(config_.page_tokens) * slot_bytes_;
    for (int32_t plane = 0; plane < planes_; ++plane) {
        for (int32_t layer = 0; layer < config_.n_layers; ++layer) {
            const Place& p = places_[static_cast<size_t>(layer)];
            const Group& g = groups_[static_cast<size_t>(p.group)];
            if (!mapped(g, static_cast<size_t>(logical))) continue;  // a windowed group's released rows
            const uint8_t* src = page + (static_cast<size_t>(plane) * static_cast<size_t>(config_.n_layers) +
                                         static_cast<size_t>(layer)) * run;
            std::memcpy(unit_ptr(g, static_cast<size_t>(logical), p.unit) + run_offset(p, plane), src, run);
        }
    }
    return Status::ok();
}

}  // namespace liyab
