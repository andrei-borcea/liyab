// Liyab — a small regex engine for tokenizer pre-tokenizer patterns (internal).
//
// Byte-level BPE tokenizers cut text into pieces with a regular expression
// before merging (Hugging Face `Split` / `ByteLevel` pre-tokenizers, written
// for Oniguruma and Python's `regex`). This engine runs those patterns on
// code points. Supported syntax:
//
//   * literals, `.` (anything but "\n"), escapes \n \r \t \f \v \e \a \0
//     \xHH \x{H...} \uHHHH \UHHHHHHHH and escaped punctuation;
//   * classes [...] and [^...] with ranges, and inside or outside them
//     \p{X} / \P{X} / \pX for the general categories (L Lu Ll Lt Lm Lo LC M
//     Mn Mc Me N Nd Nl No P Pc Pd Ps Pe Pi Pf Po S Sm Sc Sk So Z Zs Zl Zp C
//     Cc Cf Cs Co Cn, or their long names such as Letter), \s \S (Unicode
//     White_Space), \d \D (Nd) and \w \W (letters, marks, Nd, Pc);
//   * alternation, groups ( ) and (?: ), case-insensitive groups (?i: ) and
//     (?i), lookahead (?= ) (?! ), atomic groups (?> );
//   * quantifiers * + ? {n} {n,} {n,m}, lazy (suffix ?) and possessive
//     (suffix +);
//   * anchors ^ and $ at line boundaries (as Oniguruma, which the Hugging
//     Face tokenizers use), \A, \z and \Z.
//
// Matching is leftmost-first with backtracking, like both reference engines.
// Case-insensitive matching folds ASCII letters (with U+017F and U+212A,
// which fold to 's' and 'k'); other literals match exactly. Lookbehind,
// backreferences, word boundaries and repeated groups that can match the
// empty string are rejected at compile time with an error naming them.
//
// Speed matters (tokenization is on the typing path), so the subject is
// decoded once into code points plus one class key per code point (general
// category and White_Space), and every character class is a 64-bit set over
// those keys plus an ASCII bitmap and sorted ranges: testing a code point
// costs a shift, not a table search. A single class under a quantifier (the
// common \p{L}+, \s*, \p{N}{1,3}) runs as one instruction whose backtracking
// costs one stack entry, not one per character.
//
// A compiled Regex is immutable and safe to use from several threads.
#ifndef LIYAB_CORE_REGEX_H
#define LIYAB_CORE_REGEX_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "liyab/types.h"

namespace liyab {

// Text to match: code points and their class keys (regex_key()), both
// `size` long. Positions are code-point indices.
struct RegexSubject {
    const uint32_t* cps = nullptr;
    const uint8_t* keys = nullptr;
    size_t size = 0;
};

// The class key of a code point: its general category, plus 32 when it is
// white space.
[[nodiscard]] uint8_t regex_key(uint32_t cp) noexcept;

class Regex {
public:
    // Compiles `pattern` (UTF-8). InvalidArgument names what is wrong or
    // unsupported and where.
    static Result<Regex> compile(std::string_view pattern);

    // The leftmost match in s[from, size): its start and end. A match may be
    // empty. False when there is none.
    [[nodiscard]] bool search(const RegexSubject& s, size_t from, size_t& begin, size_t& end) const;
    // A match starting exactly at `at`: its end.
    [[nodiscard]] bool match_at(const RegexSubject& s, size_t at, size_t& end) const;

    [[nodiscard]] const std::string& pattern() const noexcept { return pattern_; }

    // Internal representation, public for the compiler in regex.cpp.
    struct Class {
        uint64_t keys = 0;                               // members by class key (non-ASCII code points)
        uint64_t ascii[2] = {0, 0};                      // members below U+0080, negation applied
        std::vector<std::pair<uint32_t, uint32_t>> ranges;  // sorted, merged, non-ASCII
        bool negated = false;

        [[nodiscard]] bool contains(uint32_t cp, uint8_t key) const noexcept;
    };
    enum class Op : uint8_t { Class, Repeat, Split, Jump, Match, Assert, Look, Atomic };
    enum class Mode : uint8_t { Greedy, Lazy, Possessive };
    enum class Anchor : uint8_t { LineStart, LineEnd, TextStart, TextEnd, TextEndNewline };
    struct Inst {
        Op op = Op::Match;
        Mode mode = Mode::Greedy;  // Repeat
        Anchor anchor = Anchor::TextStart;  // Assert
        bool negate = false;       // Look
        uint32_t a = 0;            // Class/Repeat: class index; Split: preferred target; Jump: target; Look/Atomic: end of the sub-program
        uint32_t b = 0;            // Split: other target
        uint32_t min = 0, max = 0;  // Repeat
    };

private:
    Regex() = default;
    bool run(const RegexSubject& s, uint32_t pc, size_t pos, size_t& end) const;

    std::string pattern_;
    std::vector<Inst> program_;
    std::vector<Class> classes_;
    // Code points a match can start with (prefilter for search), when the
    // pattern cannot match the empty string or start with an assertion.
    bool has_first_ = false;
    Class first_;

    friend class RegexCompiler;
};

}  // namespace liyab

#endif  // LIYAB_CORE_REGEX_H
