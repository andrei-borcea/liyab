#include "core/unicode.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <unordered_map>
#include <vector>

namespace liyab::unicode {

namespace {

#include "core/unicode_tables.inc"

// Value of the run holding `cp` in a table of (first << 8) | value entries
// sorted by first code point, the first one starting at U+0000.
template <size_t N>
uint8_t run_value(const uint32_t (&runs)[N], uint32_t cp) noexcept {
    if (cp > 0x10FFFF) cp = 0x10FFFF;
    const uint32_t key = (cp << 8) | 0xFF;
    const uint32_t* it = std::upper_bound(std::begin(runs), std::end(runs), key);
    return static_cast<uint8_t>(*(it - 1) & 0xFF);
}

// Latin-1 categories, looked up directly: most text is mostly ASCII.
const std::array<uint8_t, 256>& latin1_categories() {
    static const std::array<uint8_t, 256> table = [] {
        std::array<uint8_t, 256> t{};
        for (uint32_t cp = 0; cp < 256; ++cp) t[cp] = run_value(kCategoryRuns, cp);
        return t;
    }();
    return table;
}

// Hangul syllables (Unicode 3.12, conjoining jamo behavior).
constexpr uint32_t kSBase = 0xAC00, kLBase = 0x1100, kVBase = 0x1161, kTBase = 0x11A7;
constexpr uint32_t kLCount = 19, kVCount = 21, kTCount = 28, kNCount = kVCount * kTCount, kSCount = kLCount * kNCount;

constexpr size_t kDecompositionCount = std::size(kDecompositions) / 3;
constexpr uint32_t kComposesFlag = 0x80000000u;

// Index of `cp` in a decomposition table of triples, or -1.
template <size_t N>
std::ptrdiff_t find_decomposition(uint32_t cp, const uint32_t (&rows)[N]) noexcept {
    size_t lo = 0, hi = N / 3;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const uint32_t at = rows[mid * 3] & ~kComposesFlag;
        if (at < cp) lo = mid + 1;
        else if (at > cp) hi = mid;
        else return static_cast<std::ptrdiff_t>(mid);
    }
    return -1;
}

// Primary composites by (first << 21 | second), built once.
const std::unordered_map<uint64_t, uint32_t>& compositions() {
    static const std::unordered_map<uint64_t, uint32_t> table = [] {
        std::unordered_map<uint64_t, uint32_t> m;
        for (size_t i = 0; i < kDecompositionCount; ++i) {
            const uint32_t cp = kDecompositions[i * 3];
            if ((cp & kComposesFlag) == 0) continue;
            m.emplace((uint64_t{kDecompositions[i * 3 + 1]} << 21) | kDecompositions[i * 3 + 2], cp & ~kComposesFlag);
        }
        return m;
    }();
    return table;
}

// Malformed bytes travel through normalization as these values: never a
// valid code point, combining class 0, no decomposition or composition.
constexpr uint32_t kRawByte = 0x40000000u;

// Normalization must preserve malformed bytes without consuming a later
// valid character (which may itself compose). General regex decoding has
// a different fast path; keep this validation local to normalization.
uint32_t normalization_cp(std::string_view text, size_t& at) noexcept {
    const auto lead = static_cast<unsigned char>(text[at]);
    if (lead < 0x80) { ++at; return lead; }
    const size_t length = utf8_length(lead);
    auto raw = [&] { ++at; return kRawByte | lead; };
    if (length == 1 || length > text.size() - at) return raw();
    uint32_t cp = lead & (0x7F >> length);
    for (size_t i = 1; i < length; ++i) {
        const auto byte = static_cast<unsigned char>(text[at + i]);
        if ((byte & 0xC0) != 0x80) return raw();
        cp = (cp << 6) | (byte & 0x3F);
    }
    if ((length == 2 && cp < 0x80) || (length == 3 && cp < 0x800) ||
        (length == 4 && cp < 0x10000) || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return raw();
    at += length;
    return cp;
}

bool nfkc_assigned(uint32_t cp) noexcept {
    return cp < kRawByte && run_value(kNfkcAssignedRuns, cp) != 0;
}

void decompose(uint32_t cp, std::vector<uint32_t>& out, bool compatibility) {
    if (compatibility && !nfkc_assigned(cp)) { out.push_back(cp); return; }
    if (cp >= kSBase && cp < kSBase + kSCount) {
        const uint32_t s = cp - kSBase;
        out.push_back(kLBase + s / kNCount);
        out.push_back(kVBase + (s % kNCount) / kTCount);
        if (s % kTCount != 0) out.push_back(kTBase + s % kTCount);
        return;
    }
    if (compatibility && cp >= kCompatibilityDecompositions[0] && cp < kRawByte) {
        const auto index = find_decomposition(cp, kCompatibilityDecompositions);
        if (index >= 0) {
            const size_t offset = kCompatibilityDecompositions[index * 3 + 1];
            const size_t count = kCompatibilityDecompositions[index * 3 + 2];
            for (size_t part = 0; part < count; ++part) decompose(kCompatibilityParts[offset + part], out, true);
            return;
        }
    }
    const std::ptrdiff_t i = cp < 0xC0 || cp >= kRawByte ? -1 : find_decomposition(cp, kDecompositions);
    if (i < 0) {
        out.push_back(cp);
        return;
    }
    decompose(kDecompositions[i * 3 + 1], out, compatibility);
    if (const uint32_t second = kDecompositions[i * 3 + 2]) decompose(second, out, compatibility);
}

uint8_t ccc(uint32_t cp) noexcept { return cp < 0x300 || cp >= kRawByte ? 0 : run_value(kCombiningClassRuns, cp); }

uint32_t compose(uint32_t a, uint32_t b) {
    if (a >= kLBase && a < kLBase + kLCount && b >= kVBase && b < kVBase + kVCount) {
        return kSBase + ((a - kLBase) * kVCount + (b - kVBase)) * kTCount;
    }
    if (a >= kSBase && a < kSBase + kSCount && (a - kSBase) % kTCount == 0 && b > kTBase && b < kTBase + kTCount) {
        return a + (b - kTBase);
    }
    if (a >= kRawByte || b >= kRawByte) return 0;
    const auto& table = compositions();
    const auto it = table.find((uint64_t{a} << 21) | b);
    return it == table.end() ? 0 : it->second;
}

}  // namespace

Category category(uint32_t cp) noexcept {
    if (cp < 256) return static_cast<Category>(latin1_categories()[cp]);
    return static_cast<Category>(run_value(kCategoryRuns, cp));
}

bool is_whitespace(uint32_t cp) noexcept {
    // PropList.txt White_Space: 25 code points, stable since Unicode 6.3.
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 ||
           (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F ||
           cp == 0x3000;
}

uint8_t combining_class(uint32_t cp) noexcept { return ccc(cp); }

char ascii_fold(uint32_t cp) noexcept {
    if (cp >= 'A' && cp <= 'Z') return static_cast<char>(cp + 32);
    if (cp >= 'a' && cp <= 'z') return static_cast<char>(cp);
    for (size_t i = 0; i + 1 < std::size(kAsciiFoldExtras); i += 2) {
        if (kAsciiFoldExtras[i] == cp) return static_cast<char>(kAsciiFoldExtras[i + 1]);
    }
    return 0;
}

std::vector<uint32_t> ascii_case_variants(char lower) {
    std::vector<uint32_t> out = {static_cast<uint32_t>(lower), static_cast<uint32_t>(lower - 32)};
    for (size_t i = 0; i + 1 < std::size(kAsciiFoldExtras); i += 2) {
        if (kAsciiFoldExtras[i + 1] == static_cast<uint32_t>(lower)) out.push_back(kAsciiFoldExtras[i]);
    }
    return out;
}

size_t utf8_length(unsigned char lead) noexcept {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;  // a continuation or invalid lead byte: one byte on its own
}

uint32_t decode_utf8(std::string_view s, size_t& i) noexcept {
    const auto lead = static_cast<unsigned char>(s[i]);
    if (lead < 0x80) {
        ++i;
        return lead;
    }
    const size_t len = utf8_length(lead);
    if (len == 1 || i + len > s.size()) {
        ++i;
        return 0xFFFD;
    }
    uint32_t cp = lead & (0x7F >> len);
    for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3F);
    i += len;
    return cp;
}

void append_utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

bool nfc_quick_check(std::string_view text) noexcept {
    for (size_t i = 0; i < text.size();) {
        if (static_cast<unsigned char>(text[i]) < 0x80) {
            ++i;
            continue;
        }
        const uint32_t cp = normalization_cp(text, i);
        if (cp >= kRawByte || (cp >= 0x300 && run_value(kNfcMaybeRuns, cp) != 0)) return false;
    }
    return true;
}

static std::string to_composed(std::string_view text, bool compatibility) {
    // Full canonical decomposition, keeping malformed bytes as they are.
    std::vector<uint32_t> cps;
    cps.reserve(text.size());
    for (size_t i = 0; i < text.size();) {
        const uint32_t cp = normalization_cp(text, i);
        if (cp >= kRawByte) cps.push_back(cp);
        else decompose(cp, cps, compatibility);
    }
    const auto combining = [compatibility](uint32_t cp) {
        return compatibility && !nfkc_assigned(cp) ? uint8_t{0} : ccc(cp);
    };
    // Canonical ordering: each run of non-starters, stably by combining class.
    for (size_t i = 0; i < cps.size();) {
        if (combining(cps[i]) == 0) {
            ++i;
            continue;
        }
        size_t end = i;
        while (end < cps.size() && combining(cps[end]) != 0) ++end;
        std::stable_sort(cps.begin() + static_cast<std::ptrdiff_t>(i), cps.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](uint32_t a, uint32_t b) { return combining(a) < combining(b); });
        i = end;
    }
    // Canonical composition: a character joins the last starter unless a
    // character between them has a class of 0 or one at least its own.
    std::vector<uint32_t> composed;
    composed.reserve(cps.size());
    std::ptrdiff_t starter = -1;
    uint8_t last_class = 0;  // class of the last character kept after the starter (0: none)
    for (const uint32_t cp : cps) {
        const uint8_t cls = combining(cp);
        if (starter >= 0 && (last_class == 0 || last_class < cls)) {
            if (const uint32_t c = compose(composed[static_cast<size_t>(starter)], cp);
                c != 0 && (!compatibility || nfkc_assigned(c))) {
                composed[static_cast<size_t>(starter)] = c;
                continue;
            }
        }
        if (cls == 0) {
            starter = static_cast<std::ptrdiff_t>(composed.size());
            last_class = 0;
        } else {
            last_class = cls;
        }
        composed.push_back(cp);
    }
    std::string out;
    out.reserve(text.size());
    for (const uint32_t cp : composed) {
        if (cp >= kRawByte) out += static_cast<char>(cp & 0xFF);
        else append_utf8(out, cp);
    }
    return out;
}

std::string to_nfc(std::string_view text) {
    return nfc_quick_check(text) ? std::string(text) : to_composed(text, false);
}

std::string to_nfkc(std::string_view text) {
    const bool ascii = std::all_of(text.begin(), text.end(), [](unsigned char byte) { return byte < 0x80; });
    return ascii ? std::string(text) : to_composed(text, true);
}

}  // namespace liyab::unicode
