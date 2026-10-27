#include "compat/weights.h"

#include <algorithm>
#include <cstring>
#include <limits>

#include "core/quant.h"

namespace liyab {

Status CombinedLatentWeights::bind(const TensorView& source, const LatentAttention& m, int32_t n_heads) {
    if (m.kv_rank <= 0 || m.nope_dim <= 0 || m.v_dim <= 0 || n_heads <= 0 || source.n_dims != 2 ||
        source.ne[2] != 1 || source.ne[3] != 1 ||
        source.cols() != m.kv_rank || source.rows() != int64_t{n_heads} * (int64_t{m.nope_dim} + m.v_dim)) {
        return Status(ErrorCode::InvalidModel, "combined latent expansion has inconsistent dimensions");
    }
    DType type;
    if (!dtype_from_ggml(static_cast<uint32_t>(source.type), type) || type == DType::I32 ||
        m.kv_rank % dtype_traits(type).block_size != 0) {
        return Status(ErrorCode::Unsupported, "combined latent expansion needs ordinary on-disk weight rows");
    }
    if (source.data == nullptr || source.nbytes / source.row_bytes() < static_cast<size_t>(source.rows())) {
        return Status(ErrorCode::InvalidModel, "combined latent expansion has incomplete weight rows");
    }
    const auto rank = static_cast<size_t>(m.kv_rank);
    const auto nope = static_cast<size_t>(m.nope_dim);
    const auto vd = static_cast<size_t>(m.v_dim);
    const auto heads = static_cast<size_t>(n_heads);
    if (nope > std::numeric_limits<size_t>::max() / heads / rank / sizeof(float)) {
        return Status(ErrorCode::InvalidModel, "combined latent expansion is too large");
    }
    keys_.resize(heads * rank * nope);
    values_.resize(heads * vd * source.row_bytes());
    std::vector<float> row(rank);
    for (size_t h = 0; h < heads; ++h) {
        const auto start = static_cast<int64_t>(h * (nope + vd));
        for (size_t i = 0; i < nope; ++i) {
            quant::dequantize_row(source.type, source.row(start + static_cast<int64_t>(i)), row.data(), m.kv_rank);
            for (size_t r = 0; r < rank; ++r) keys_[(h * rank + r) * nope + i] = row[r];
        }
        std::memcpy(values_.data() + h * vd * source.row_bytes(),
                    source.row(start + m.nope_dim), vd * source.row_bytes());
    }
    key_ = source;
    key_.type = DType::F32;
    key_.n_dims = 3;
    key_.ne = {m.nope_dim, m.kv_rank, n_heads, 1};
    key_.data = reinterpret_cast<const uint8_t*>(keys_.data());
    key_.nbytes = keys_.size() * sizeof(float);
    value_ = source;
    value_.n_dims = 3;
    value_.ne = {m.kv_rank, m.v_dim, n_heads, 1};
    value_.data = values_.data();
    value_.nbytes = values_.size();
    return Status::ok();
}

uint64_t compatibility_fingerprint(const MmapLoader& file, const ModelConfig& config) noexcept {
    if (!config.full_header_identity) return 0;
    size_t header = file.file().size();
    for (const TensorView& tensor : file.tensors()) {
        if (tensor.shard == 0) header = std::min(header, static_cast<size_t>(tensor.file_offset));
    }
    uint64_t identity = xxhash64(file.file().data(), header, model_fingerprint(file.file()));
    // Route tables define model behavior despite being stored as tensors.
    // Equal-sized changes beyond the first MiB must invalidate saved states.
    for (const TensorView& tensor : file.tensors()) {
        if (tensor.type == DType::I32) identity = xxhash64(tensor.data, tensor.nbytes, identity);
    }
    return identity;
}

Result<std::vector<int32_t>> adapt_expert_routes(const TensorView& source, int32_t vocab, int32_t top_k, int32_t experts) {
    if (source.type != DType::I32) return Status(ErrorCode::Unsupported, "hash expert routes require I32 indices");
    if (vocab <= 0 || top_k <= 0 || experts <= 0 || source.n_dims != 2 ||
        source.ne != std::array<int64_t, 4>{top_k, vocab, 1, 1}) {
        return Status(ErrorCode::InvalidModel, "invalid hash expert routing table shape");
    }
    if (static_cast<size_t>(vocab) > std::numeric_limits<size_t>::max() / static_cast<size_t>(top_k)) {
        return Status(ErrorCode::InvalidModel, "hash expert routing table is too large");
    }
    const size_t count = static_cast<size_t>(vocab) * static_cast<size_t>(top_k);
    if (count > std::numeric_limits<size_t>::max() / sizeof(int32_t) || source.data == nullptr ||
        source.nbytes != count * sizeof(int32_t)) {
        return Status(ErrorCode::InvalidModel, "invalid hash expert routing table payload");
    }
    std::vector<int32_t> routes(count);
    std::memcpy(routes.data(), source.data, source.nbytes);
    for (const int32_t expert : routes) {
        if (expert < 0 || expert >= experts) return Status(ErrorCode::InvalidModel, "hash route selects an out-of-range expert");
    }
    return routes;
}

}  // namespace liyab
