// SentencePiece's compiled character maps (internal). The map and returned
// spans are borrowed from static data or a GGUF mapping that must outlive this
// value. Loading validates offsets and UTF-8 replacement terminators; immutable
// normalizers are thread-safe. An empty map is identity normalization.
#ifndef LIYAB_CORE_SPM_NORMALIZER_H
#define LIYAB_CORE_SPM_NORMALIZER_H

#include <span>
#include <string>
#include <string_view>
#include "liyab/types.h"

namespace liyab {
class SpmNormalizer {
public:
    // Exact registered map, or an empty span for an unknown/empty identity ID.
    static std::span<const uint8_t> registered_map(std::string_view id);
    static std::span<const std::string_view> registered_user_defined(std::string_view id);
    static Result<SpmNormalizer> load(std::span<const uint8_t> data);
    [[nodiscard]] bool empty() const noexcept { return trie_.empty(); }
    // Longest-prefix replacements; unmatched UTF-8 bytes retain their values.
    [[nodiscard]] std::string normalize(std::string_view text) const;
private:
    std::span<const uint8_t> trie_, values_;
    [[nodiscard]] uint32_t unit(size_t index) const;
};
}  // namespace liyab
#endif
