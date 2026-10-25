#include "core/spm_normalizer.h"
#include <cstring>
#include "core/unicode.h"

namespace liyab {
namespace {
#include "core/spm_normalizer.inc"
uint32_t little_u32(const uint8_t* data) {
    return uint32_t{data[0]} | (uint32_t{data[1]} << 8) | (uint32_t{data[2]} << 16) | (uint32_t{data[3]} << 24);
}
// Darts double-array units: 8-bit label, leaf flag, extended-offset flag,
// 22-bit offset. Leaf values have bit31 set and store a string-table offset.
uint32_t offset(uint32_t word) { return (word >> 10) << ((word & 0x200) >> 6); }
uint32_t label(uint32_t word) { return word & 0x800000ffu; }
}

std::span<const uint8_t> SpmNormalizer::registered_map(std::string_view id) {
    return id == kSpmCharsmapId ? std::span<const uint8_t>(kSpmCharsmap) : std::span<const uint8_t>();
}

std::span<const std::string_view> SpmNormalizer::registered_user_defined(std::string_view id) {
    return id == kSpmUserDefinedId ? std::span<const std::string_view>(kSpmUserDefined) : std::span<const std::string_view>();
}

uint32_t SpmNormalizer::unit(size_t index) const { return little_u32(trie_.data() + index * 4); }

Result<SpmNormalizer> SpmNormalizer::load(std::span<const uint8_t> data) {
    SpmNormalizer out;
    if (data.empty()) return out;
    auto invalid = [] { return Status(ErrorCode::InvalidModel, "invalid SentencePiece precompiled character map"); };
    if (data.size() < 8) return invalid();
    const size_t bytes = little_u32(data.data());
    if (bytes < 4 || bytes % 4 != 0 || bytes > data.size() - 4) return invalid();
    out.trie_ = data.subspan(4, bytes);
    out.values_ = data.subspan(4 + bytes);
    const size_t count = bytes / 4;
    if (out.values_.empty()) return invalid();
    for (size_t i = 0; i < count; ++i) {
        const uint32_t word = out.unit(i);
        if (word & 0x80000000u) continue;  // a leaf's value, not a transition
        const size_t node = i ^ offset(word);
        if (node >= count) return invalid();
        if (word & 0x100) {
            const uint32_t value = out.unit(node);
            if (!(value & 0x80000000u)) return invalid();
            const size_t at = value & 0x7fffffffu;
            if (at >= out.values_.size() ||
                std::memchr(out.values_.data() + at, 0, out.values_.size() - at) == nullptr) return invalid();
        }
    }
    return out;
}

std::string SpmNormalizer::normalize(std::string_view text) const {
    if (trie_.empty()) return std::string(text);
    std::string result;
    result.reserve(text.size());
    const size_t count = trie_.size() / 4;
    for (size_t pos = 0; pos < text.size();) {
        size_t node = offset(unit(0)), matched = 0, replacement = 0;
        for (size_t i = pos; i < text.size(); ++i) {
            const uint8_t byte = static_cast<uint8_t>(text[i]);
            if (byte == 0) break;  // Darts labels zero mark leaves, never input NUL
            const size_t next = node ^ byte;
            if (next >= count) break;
            const uint32_t word = unit(next);
            if (label(word) != byte) break;
            node = next ^ offset(word);
            if (word & 0x100) {
                matched = i + 1 - pos;
                replacement = unit(node) & 0x7fffffffu;
            }
        }
        if (matched) {
            result += reinterpret_cast<const char*>(values_.data() + replacement);
            pos += matched;
        } else {
            // Consume an entire unmatched character: a later byte of its UTF-8
            // encoding cannot start a different normalization rule.
            size_t end = pos;
            (void)unicode::decode_utf8(text, end);
            result.append(text.substr(pos, end - pos));
            pos = end;
        }
    }
    return result;
}
}  // namespace liyab
