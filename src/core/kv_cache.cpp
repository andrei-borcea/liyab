#include "liyab/kv_cache.h"

#include <algorithm>
#include <cstring>
#include <new>
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

}  // namespace

size_t KvCache::bytes_per_token(int32_t n_layers, int32_t n_head_kv, int32_t head_dim, KvCacheType type) noexcept {
    return 2 * static_cast<size_t>(n_layers) * static_cast<size_t>(n_head_kv) *
           dtype_row_bytes(to_dtype(type), head_dim);
}

size_t KvCache::bytes_per_token(const KvCacheConfig& config) noexcept {
    const size_t both = bytes_per_token(config.n_layers, config.n_head_kv, config.head_dim, config.type);
    return config.values_in_keys ? both / 2 : both;
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
    std::unique_ptr<KvCache> cache(new KvCache());
    cache->config_ = config;
    cache->dtype_ = dtype;
    cache->row_bytes_ = dtype_row_bytes(dtype, config.head_dim);
    cache->value_skip_ = config.values_in_keys ? dtype_row_bytes(dtype, config.value_offset) : 0;
    cache->page_bytes_ = (config.values_in_keys ? 1 : 2) * static_cast<size_t>(config.n_layers) *
                         static_cast<size_t>(config.page_tokens) * static_cast<size_t>(config.n_head_kv) *
                         cache->row_bytes_;
    return cache;
}

KvCache::Visible KvCache::visible(int32_t pos) const noexcept {
    if (config_.window <= 0) return {0, 0};
    const int32_t first = std::max(0, pos - config_.window + 1);
    return {std::min(config_.sink_tokens, first), first};
}

Status KvCache::reserve(int32_t end) {
    if (config_.window <= 0 && end > config_.max_positions) {
        return Status(ErrorCode::ContextFull, "context of " + std::to_string(config_.max_positions) + " tokens is full");
    }
    if (end <= 0) return Status::ok();
    const auto pages = static_cast<size_t>((end - 1) / config_.page_tokens + 1);
    if (table_.size() < pages) table_.resize(pages, -1);
    const auto sink_pages = static_cast<size_t>(sink_page_count());
    for (size_t p = 0; p < pages; ++p) {
        // Released pages stay unmapped: no query can see them any more.
        if (table_[p] != kUnmapped || (p >= sink_pages && p < static_cast<size_t>(released_below_))) continue;
        if (free_.empty()) {
            uint8_t* page = new (std::nothrow) uint8_t[page_bytes_];
            if (page == nullptr) return Status(ErrorCode::OutOfMemory, "cannot allocate a KV cache page");
            pool_.emplace_back(page);
            free_.push_back(static_cast<int32_t>(pool_.size() - 1));
        }
        table_[p] = free_.back();
        free_.pop_back();
        // Zeroed when mapped, new or recycled. Attention never reads a row
        // before it is written, but whole pages leave the process (save_state,
        // the prefix cache's chunks): rows past the context's end must hold
        // zeros, not another context's rows or the heap's leftovers, so the
        // same context always gives the same bytes and nothing else is written
        // to storage. One page per page_tokens positions: noise next to
        // computing them.
        std::memset(page_ptr(p), 0, page_bytes_);
    }
    return Status::ok();
}

uint8_t* KvCache::page_ptr(size_t logical) const noexcept { return pool_[static_cast<size_t>(table_[logical])].get(); }

const uint8_t* KvCache::page_data(int32_t logical) const noexcept {
    if (logical < 0 || static_cast<size_t>(logical) >= table_.size() || table_[static_cast<size_t>(logical)] == kUnmapped) {
        return nullptr;
    }
    return page_ptr(static_cast<size_t>(logical));
}

uint8_t* KvCache::mutable_page_data(int32_t logical) noexcept {
    return const_cast<uint8_t*>(static_cast<const KvCache*>(this)->page_data(logical));
}

size_t KvCache::rows_bytes(int32_t n) const noexcept {
    return page_bytes_ / static_cast<size_t>(config_.page_tokens) * static_cast<size_t>(std::max(n, 0));
}

void KvCache::copy_slots(const uint8_t* src, int32_t src_slots, int32_t src_first, uint8_t* dst, int32_t dst_slots,
                         int32_t dst_first, int32_t count) const noexcept {
    // A slot is n_head_kv rows; each plane (K, then V) holds n_layers runs of
    // slots, so one copy per plane and layer moves `count` contiguous slots.
    const size_t slot = static_cast<size_t>(config_.n_head_kv) * row_bytes_;
    const int32_t planes = config_.values_in_keys ? 1 : 2;
    for (int32_t run = 0; run < planes * config_.n_layers; ++run) {
        std::memcpy(dst + (static_cast<size_t>(run) * static_cast<size_t>(dst_slots) + static_cast<size_t>(dst_first)) * slot,
                    src + (static_cast<size_t>(run) * static_cast<size_t>(src_slots) + static_cast<size_t>(src_first)) * slot,
                    static_cast<size_t>(count) * slot);
    }
}

Status KvCache::read_rows(int32_t logical, int32_t n, uint8_t* out) const {
    const uint8_t* page = page_data(logical);
    if (page == nullptr || n < 1 || n > config_.page_tokens) {
        return Status(ErrorCode::InvalidArgument, "read_rows: the page is not mapped or the count is out of range");
    }
    if (n == config_.page_tokens) {
        std::memcpy(out, page, page_bytes_);
    } else {
        copy_slots(page, config_.page_tokens, 0, out, n, 0, n);
    }
    return Status::ok();
}

Status KvCache::write_rows(int32_t logical, const uint8_t* image, int32_t n, int32_t from, int32_t to) {
    uint8_t* page = mutable_page_data(logical);
    if (page == nullptr || n < 1 || n > config_.page_tokens || from < 0 || to > n || from > to) {
        return Status(ErrorCode::InvalidArgument, "write_rows: the page is not mapped or the range is out of bounds");
    }
    if (from == 0 && to == config_.page_tokens) {
        std::memcpy(page, image, page_bytes_);
    } else if (from < to) {
        copy_slots(image, n, from, page, config_.page_tokens, from, to - from);
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
    // One copy per run of positions inside a page (copy_slots moves every
    // plane and layer of the run).
    for (int32_t done = 0; done < count;) {
        const int32_t p = pos + done;
        const int32_t slot = p % config_.page_tokens;
        const int32_t run = std::min(count - done, config_.page_tokens - slot);
        const uint8_t* page = page_data(p / config_.page_tokens);
        if (page == nullptr) return Status(ErrorCode::InvalidArgument, "read_rows_at: a page is not mapped");
        copy_slots(page, config_.page_tokens, slot, image, slots, first + done, run);
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
        uint8_t* page = mutable_page_data(p / config_.page_tokens);
        if (page == nullptr) return Status(ErrorCode::InvalidArgument, "write_rows_at: a page is not mapped");
        copy_slots(image, slots, first + done, page, config_.page_tokens, slot, run);
        done += run;
    }
    return Status::ok();
}

size_t KvCache::image_key_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept {
    // The K plane: [layer][slot][kv_head][row], as a page with `slots` slots.
    return ((static_cast<size_t>(layer) * static_cast<size_t>(slots) + static_cast<size_t>(slot)) *
                static_cast<size_t>(config_.n_head_kv) +
            static_cast<size_t>(kv_head)) *
           row_bytes_;
}

size_t KvCache::image_value_offset(int32_t slots, int32_t layer, int32_t slot, int32_t kv_head) const noexcept {
    const size_t key = image_key_offset(slots, layer, slot, kv_head);
    if (config_.values_in_keys) return key + value_skip_;
    // The V plane follows the K plane of the whole image.
    return static_cast<size_t>(config_.n_layers) * static_cast<size_t>(slots) * static_cast<size_t>(config_.n_head_kv) *
               row_bytes_ +
           key;
}

uint8_t* KvCache::row(int32_t layer, int32_t pos, int32_t kv_head, bool value) const noexcept {
    const auto logical = static_cast<size_t>(pos / config_.page_tokens);
    const auto slot = static_cast<size_t>(pos % config_.page_tokens);
    uint8_t* page = page_ptr(logical);
    // Page layout: [K | V], each [layer][slot][kv_head][row]; only K with
    // values_in_keys, where a value row starts value_skip_ bytes into its
    // key row (whole blocks, so it decodes on its own).
    const size_t index = (static_cast<size_t>(layer) * static_cast<size_t>(config_.page_tokens) + slot) *
                             static_cast<size_t>(config_.n_head_kv) + static_cast<size_t>(kv_head);
    if (config_.values_in_keys) return page + index * row_bytes_ + (value ? value_skip_ : 0);
    return page + (value ? page_bytes_ / 2 : 0) + index * row_bytes_;
}

void KvCache::store(int32_t layer, int32_t pos, const float* k, const float* v) {
    for (int32_t g = 0; g < config_.n_head_kv; ++g) {
        quant::quantize_row(dtype_, k + g * config_.head_dim, row(layer, pos, g, false), config_.head_dim);
        if (config_.values_in_keys) continue;
        quant::quantize_row(dtype_, v + g * config_.head_dim, row(layer, pos, g, true), config_.head_dim);
    }
}

Status KvCache::scale_keys(int32_t end, std::span<const float> factors) {
    const auto hd = static_cast<size_t>(config_.head_dim);
    if (factors.size() != static_cast<size_t>(config_.n_layers) * static_cast<size_t>(config_.n_head_kv) * hd) {
        return Status(ErrorCode::InvalidArgument, "one key factor per layer, KV head and channel");
    }
    for (int32_t p = 0; p * config_.page_tokens < end; ++p) {
        if (mutable_page_data(p) == nullptr) return Status(ErrorCode::InvalidArgument, "keys on a page that is not mapped");
    }
    std::vector<float> row_values(hd);
    for (int32_t layer = 0; layer < config_.n_layers; ++layer) {
        for (int32_t g = 0; g < config_.n_head_kv; ++g) {
            const float* f = factors.data() + (static_cast<size_t>(layer) * static_cast<size_t>(config_.n_head_kv) +
                                               static_cast<size_t>(g)) * hd;
            for (int32_t pos = 0; pos < end; ++pos) {
                uint8_t* r = row(layer, pos, g, false);
                quant::dequantize_row(dtype_, r, row_values.data(), config_.head_dim);
                for (size_t i = 0; i < hd; ++i) row_values[i] *= f[i];
                quant::quantize_row(dtype_, row_values.data(), r, config_.head_dim);
            }
        }
    }
    return Status::ok();
}

const uint8_t* KvCache::k_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept {
    return row(layer, pos, kv_head, false);
}

const uint8_t* KvCache::v_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept {
    return row(layer, pos, kv_head, true);
}

void KvCache::free_page(size_t logical) noexcept {
    if (logical >= table_.size() || table_[logical] == kUnmapped) return;
    free_.push_back(table_[logical]);
    table_[logical] = kUnmapped;
}

Status KvCache::truncate(int32_t n) {
    if (n < 0) return Status(ErrorCode::InvalidArgument, "negative KV cache length");
    if (config_.window > 0 && n > 0) {
        const Visible v = visible(n);
        if (v.first / config_.page_tokens < released_below_) {
            return Status(ErrorCode::InvalidArgument, "rollback reaches KV pages that were already released");
        }
    }
    // Pages that start at or after n hold no surviving position.
    for (size_t p = static_cast<size_t>((n + config_.page_tokens - 1) / config_.page_tokens); p < table_.size(); ++p) {
        free_page(p);
    }
    if (n == 0) released_below_ = 0;
    return Status::ok();
}

void KvCache::release_unreachable(int32_t n_past, int32_t keep_back) {
    if (config_.window <= 0) return;
    // Oldest non-sink position any future query (after a rollback of at most
    // keep_back tokens) can still see.
    const int32_t oldest = visible(std::max(0, n_past - keep_back)).first;
    const int32_t sink_pages = sink_page_count();
    const int32_t limit = oldest / config_.page_tokens;  // pages strictly below are unreachable
    for (int32_t p = std::max(sink_pages, released_below_); p < limit; ++p) free_page(static_cast<size_t>(p));
    released_below_ = std::max(released_below_, limit);
}

int32_t KvCache::sink_page_count() const noexcept {
    const int32_t sinks = sink_tokens();
    return sinks > 0 ? (sinks - 1) / config_.page_tokens + 1 : 0;
}

void KvCache::clear() noexcept {
    for (size_t p = 0; p < table_.size(); ++p) free_page(p);
    released_below_ = 0;
}

int32_t KvCache::pages_in_use() const noexcept {
    return static_cast<int32_t>(pool_.size() - free_.size());
}

}  // namespace liyab
