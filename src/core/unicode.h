// Liyab — Unicode support for the tokenizers (internal): UTF-8 decoding,
// general categories, White_Space, and NFC / NFKC normalization.
//
// The data is generated from the Unicode Character Database by
// tools/gen_unicode_tables.py into unicode_tables.inc (the version is in its
// header). Everything here is stateless and thread-safe.
#ifndef LIYAB_CORE_UNICODE_H
#define LIYAB_CORE_UNICODE_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace liyab::unicode {

// Unicode general categories. The order is the one of
// tools/gen_unicode_tables.py (CATEGORIES); a regex character class keeps one
// bit per category, so the count must stay below 32.
enum Category : uint8_t {
    Lu, Ll, Lt, Lm, Lo,           // letters
    Mn, Mc, Me,                   // marks
    Nd, Nl, No,                   // numbers
    Pc, Pd, Ps, Pe, Pi, Pf, Po,   // punctuation
    Sm, Sc, Sk, So,               // symbols
    Zs, Zl, Zp,                   // separators
    Cc, Cf, Cs, Co, Cn,           // other (Cn: unassigned)
    kCategoryCount
};
static_assert(kCategoryCount <= 32);

// The category of a code point (Cn above U+10FFFF).
[[nodiscard]] Category category(uint32_t cp) noexcept;
// The Unicode White_Space property (what \s matches in the tokenizers'
// regexes: U+001C..U+001F, U+180E, U+200B and U+FEFF are not white space).
[[nodiscard]] bool is_whitespace(uint32_t cp) noexcept;
// Canonical combining class (0 for starters).
[[nodiscard]] uint8_t combining_class(uint32_t cp) noexcept;
// The ASCII lower-case letter `cp` matches case-insensitively under simple
// case folding, or 0: 'A'..'Z', 'a'..'z', U+017F (long s) and U+212A (Kelvin).
[[nodiscard]] char ascii_fold(uint32_t cp) noexcept;
// Every code point whose ascii_fold() is `lower` ('a'..'z'): both ASCII
// cases plus the non-ASCII ones folding to it.
[[nodiscard]] std::vector<uint32_t> ascii_case_variants(char lower);

// Byte length of a UTF-8 sequence from its lead byte (1 for an invalid lead).
[[nodiscard]] size_t utf8_length(unsigned char lead) noexcept;
// Decodes the code point at text[i] and advances `i`. A truncated sequence
// decodes as U+FFFD and advances one byte, so every byte is consumed once
// and slicing by the returned offsets keeps the original bytes.
[[nodiscard]] uint32_t decode_utf8(std::string_view text, size_t& i) noexcept;
void append_utf8(std::string& out, uint32_t cp);

// Whether NFC leaves `text` unchanged for certain, from a scan that only
// looks up code points at or above U+0300 (no allocation). False means it may
// change, not that it will.
[[nodiscard]] bool nfc_quick_check(std::string_view text) noexcept;
// Canonical composition (Unicode Normalization Form C). Malformed UTF-8 bytes
// are kept as they are. Returns `text` unchanged when nfc_quick_check holds.
[[nodiscard]] std::string to_nfc(std::string_view text);
// Unicode 9 compatibility decomposition followed by canonical composition
// (NFKC), matching the tokenizers normalization dependency. Characters
// unassigned in Unicode 9 remain inert.
// Malformed UTF-8 bytes are preserved; ASCII returns unchanged. Stateless
// and thread-safe; existing NFC uses its own current Unicode tables.
[[nodiscard]] std::string to_nfkc(std::string_view text);

}  // namespace liyab::unicode

#endif  // LIYAB_CORE_UNICODE_H
