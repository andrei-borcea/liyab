#include "liyab/kv_cache.h"

#include <algorithm>
#include <new>

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

Result<std::unique_ptr<KvCache>> KvCache::create(const KvCacheConfig& config) {
    if (config.n_layers <= 0 || config.n_head_kv <= 0 || config.head_dim <= 0 || config.page_tokens <= 0 ||
        config.max_positions <= 0 || config.window < 0 || config.sink_tokens < 0) {
        return Status(ErrorCode::InvalidArgument, "invalid KV cache dimensions");
    }
    const DType dtype = to_dtype(config.type);
    if (quant::is_block_quantized(dtype) && config.head_dim % quant::kBlock != 0) {
        return Status(ErrorCode::InvalidArgument, "quantized KV cache needs head_dim to be a multiple of 32");
    }
    if (config.window > 0 && config.sink_tokens > config.page_tokens) {
        return Status(ErrorCode::InvalidArgument, "sink_tokens must fit in the first page");
    }
    std::unique_ptr<KvCache> cache(new KvCache());
    cache->config_ = config;
    cache->dtype_ = dtype;
    cache->row_bytes_ = dtype_row_bytes(dtype, config.head_dim);
    cache->page_bytes_ = 2 * static_cast<size_t>(config.n_layers) * static_cast<size_t>(config.page_tokens) *
                         static_cast<size_t>(config.n_head_kv) * cache->row_bytes_;
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
        if (table_[p] >= 0 || (p >= sink_pages && p < static_cast<size_t>(released_below_))) continue;
        if (free_.empty()) {
            // Uninitialized on purpose: every row is written before it is read.
            uint8_t* page = new (std::nothrow) uint8_t[page_bytes_];
            if (page == nullptr) return Status(ErrorCode::OutOfMemory, "cannot allocate a KV cache page");
            pool_.emplace_back(page);
            free_.push_back(static_cast<int32_t>(pool_.size() - 1));
        }
        table_[p] = free_.back();
        free_.pop_back();
    }
    return Status::ok();
}

uint8_t* KvCache::row(int32_t layer, int32_t pos, int32_t kv_head, bool value) const noexcept {
    const auto logical = static_cast<size_t>(pos / config_.page_tokens);
    const auto slot = static_cast<size_t>(pos % config_.page_tokens);
    uint8_t* page = pool_[static_cast<size_t>(table_[logical])].get();
    // Page layout: [K | V], each [layer][slot][kv_head][row].
    const size_t half = page_bytes_ / 2;
    const size_t index = (static_cast<size_t>(layer) * static_cast<size_t>(config_.page_tokens) + slot) *
                             static_cast<size_t>(config_.n_head_kv) + static_cast<size_t>(kv_head);
    return page + (value ? half : 0) + index * row_bytes_;
}

void KvCache::store(int32_t layer, int32_t pos, const float* k, const float* v) {
    for (int32_t g = 0; g < config_.n_head_kv; ++g) {
        quant::quantize_row(dtype_, k + g * config_.head_dim, row(layer, pos, g, false), config_.head_dim);
        quant::quantize_row(dtype_, v + g * config_.head_dim, row(layer, pos, g, true), config_.head_dim);
    }
}

const uint8_t* KvCache::k_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept {
    return row(layer, pos, kv_head, false);
}

const uint8_t* KvCache::v_row(int32_t layer, int32_t pos, int32_t kv_head) const noexcept {
    return row(layer, pos, kv_head, true);
}

void KvCache::free_page(size_t logical) noexcept {
    if (logical < table_.size() && table_[logical] >= 0) {
        free_.push_back(table_[logical]);
        table_[logical] = -1;
    }
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
