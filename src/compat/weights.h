// Liyab — bounded format adaptation of mapped weight tensors (internal).
#ifndef LIYAB_COMPAT_WEIGHTS_H
#define LIYAB_COMPAT_WEIGHTS_H

#include <vector>

#include "compat/architecture.h"
#include "liyab/mmap_loader.h"

namespace liyab {

// Owns the split execution views of one old combined MLA projection.
// bind() validates [head][nope + value][rank], decodes/transposes only keys
// to F32 and copies value bytes unchanged. Source bytes are needed only
// during bind(); its name must outlive these views. Views remain valid
// until the next bind() or destruction. Single-thread ownership; do not
// bind while another thread reads its views. InvalidModel for bad shapes,
// Unsupported for formats that are not ordinary on-disk weight rows.
class CombinedLatentWeights {
public:
    Status bind(const TensorView& source, const LatentAttention& geometry, int32_t heads);
    [[nodiscard]] const TensorView& key() const noexcept { return key_; }
    [[nodiscard]] const TensorView& value() const noexcept { return value_; }
    CombinedLatentWeights() = default;
    CombinedLatentWeights(const CombinedLatentWeights&) = delete;
    CombinedLatentWeights& operator=(const CombinedLatentWeights&) = delete;

private:
    std::vector<float> keys_;
    std::vector<uint8_t> values_;
    TensorView key_, value_;
};

// Extended identity for metadata-dependent plans and integer route tables.
// Returns zero for legacy plans, which retain model_fingerprint(file.file()).
// Reads the complete first header and auxiliary routing bytes without copying
// them. Requires a validated description and read-only mapped file; thread-safe.
uint64_t compatibility_fingerprint(const MmapLoader& file, const ModelConfig& config) noexcept;

// Copies an I32 [vocabulary, top-k] routing table after checking its shape,
// byte bounds and expert IDs. The small auxiliary table stays resident so
// hashing and prefetch prediction do not depend on a streamed weight slot.
// Source bytes are borrowed only during this call. InvalidModel on malformed
// shape/data/indices; Unsupported for a non-I32 table. Safe on independent outputs.
Result<std::vector<int32_t>> adapt_expert_routes(const TensorView& source, int32_t vocab, int32_t top_k, int32_t experts);

}  // namespace liyab

#endif  // LIYAB_COMPAT_WEIGHTS_H
