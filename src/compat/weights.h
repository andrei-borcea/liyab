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

}  // namespace liyab

#endif  // LIYAB_COMPAT_WEIGHTS_H
