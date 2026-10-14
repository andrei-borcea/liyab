#include "core/regex.h"

#include <algorithm>
#include <limits>

#include "core/unicode.h"

namespace liyab {

namespace {

constexpr uint32_t kUnbounded = std::numeric_limits<uint32_t>::max();
// Upper bound on a counted repetition of a group, which compiles to copies.
constexpr uint32_t kMaxCopies = 1000;

constexpr uint64_t kWhiteSpaceKeys = 0xFFFFFFFF00000000ull;  // keys with the White_Space bit
constexpr uint64_t kAllKeys = ~0ull;

// Key bits of the categories in `mask` (one bit per unicode::Category),
// with and without White_Space.
uint64_t category_keys(uint32_t mask) { return uint64_t{mask} | (uint64_t{mask} << 32); }

uint32_t cat_bit(unicode::Category c) { return 1u << c; }

// Category masks for \p{...} names: one-letter groups, two-letter
// categories, and the long names Unicode gives the groups.
bool category_mask(std::string_view name, uint32_t& mask) {
    using namespace unicode;
    struct Group {
        std::string_view short_name, long_name;
        uint32_t mask;
    };
    static const Group groups[] = {
        {"L", "Letter", cat_bit(Lu) | cat_bit(Ll) | cat_bit(Lt) | cat_bit(Lm) | cat_bit(Lo)},
        {"LC", "Cased_Letter", cat_bit(Lu) | cat_bit(Ll) | cat_bit(Lt)},
        {"L&", "", cat_bit(Lu) | cat_bit(Ll) | cat_bit(Lt)},
        {"M", "Mark", cat_bit(Mn) | cat_bit(Mc) | cat_bit(Me)},
        {"N", "Number", cat_bit(Nd) | cat_bit(Nl) | cat_bit(No)},
        {"P", "Punctuation",
         cat_bit(Pc) | cat_bit(Pd) | cat_bit(Ps) | cat_bit(Pe) | cat_bit(Pi) | cat_bit(Pf) | cat_bit(Po)},
        {"S", "Symbol", cat_bit(Sm) | cat_bit(Sc) | cat_bit(Sk) | cat_bit(So)},
        {"Z", "Separator", cat_bit(Zs) | cat_bit(Zl) | cat_bit(Zp)},
        {"C", "Other", cat_bit(Cc) | cat_bit(Cf) | cat_bit(Cs) | cat_bit(Co) | cat_bit(Cn)},
    };
    static const std::string_view names[] = {"Lu", "Ll", "Lt", "Lm", "Lo", "Mn", "Mc", "Me", "Nd", "Nl",
                                             "No", "Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", "Sm", "Sc",
                                             "Sk", "So", "Zs", "Zl", "Zp", "Cc", "Cf", "Cs", "Co", "Cn"};
    static_assert(std::size(names) == kCategoryCount);
    static const std::string_view long_names[] = {
        "Uppercase_Letter", "Lowercase_Letter", "Titlecase_Letter", "Modifier_Letter", "Other_Letter",
        "Nonspacing_Mark", "Spacing_Mark", "Enclosing_Mark", "Decimal_Number", "Letter_Number", "Other_Number",
        "Connector_Punctuation", "Dash_Punctuation", "Open_Punctuation", "Close_Punctuation",
        "Initial_Punctuation", "Final_Punctuation", "Other_Punctuation", "Math_Symbol", "Currency_Symbol",
        "Modifier_Symbol", "Other_Symbol", "Space_Separator", "Line_Separator", "Paragraph_Separator", "Control",
        "Format", "Surrogate", "Private_Use", "Unassigned"};
    static_assert(std::size(long_names) == kCategoryCount);
    for (const Group& g : groups) {
        if (name == g.short_name || (!g.long_name.empty() && name == g.long_name)) {
            mask = g.mask;
            return true;
        }
    }
    for (uint32_t c = 0; c < kCategoryCount; ++c) {
        if (name == names[c] || name == long_names[c]) {
            mask = 1u << c;
            return true;
        }
    }
    return false;
}

// A character class while it is being parsed.
struct ClassBuilder {
    uint64_t keys = 0;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;  // any code points, unsorted
    bool negated = false;

    void add(uint32_t first, uint32_t last, bool fold) {
        ranges.emplace_back(first, last);
        if (!fold) return;
        for (uint32_t cp = first; cp <= last && cp < 0x80; ++cp) {
            if (const char lower = unicode::ascii_fold(cp)) {
                for (const uint32_t v : unicode::ascii_case_variants(lower)) ranges.emplace_back(v, v);
            }
        }
        for (const uint32_t extra : {0x17Fu, 0x212Au}) {  // non-ASCII members fold to ASCII letters too
            if (extra >= first && extra <= last) {
                for (const uint32_t v : unicode::ascii_case_variants(unicode::ascii_fold(extra))) {
                    ranges.emplace_back(v, v);
                }
            }
        }
    }

    Regex::Class build() const {
        Regex::Class c;
        c.negated = negated;
        c.keys = keys;
        std::vector<std::pair<uint32_t, uint32_t>> sorted = ranges;
        std::sort(sorted.begin(), sorted.end());
        for (const auto& [first, last] : sorted) {
            for (uint32_t cp = first; cp <= last && cp < 0x80; ++cp) c.ascii[cp >> 6] |= 1ull << (cp & 63);
            if (last < 0x80) continue;
            const uint32_t from = std::max<uint32_t>(first, 0x80);
            if (!c.ranges.empty() && from <= c.ranges.back().second + 1) {
                c.ranges.back().second = std::max(c.ranges.back().second, last);
            } else {
                c.ranges.emplace_back(from, last);
            }
        }
        for (uint32_t cp = 0; cp < 0x80; ++cp) {
            const bool in = ((c.ascii[cp >> 6] >> (cp & 63)) & 1) || ((keys >> regex_key(cp)) & 1);
            if (in != negated) c.ascii[cp >> 6] |= 1ull << (cp & 63);
            else c.ascii[cp >> 6] &= ~(1ull << (cp & 63));
        }
        return c;
    }
};

struct Node {
    enum class Kind { Empty, Chars, Concat, Alt, Repeat, Look, Atomic, Assert } kind = Kind::Empty;
    uint32_t cls = 0;  // Chars
    std::vector<Node> kids;
    uint32_t min = 0, max = 0;  // Repeat
    Regex::Mode mode = Regex::Mode::Greedy;
    bool negate = false;  // Look
    Regex::Anchor anchor = Regex::Anchor::TextStart;
};

bool nullable(const Node& n) {
    switch (n.kind) {
        case Node::Kind::Empty: case Node::Kind::Look: case Node::Kind::Assert: return true;
        case Node::Kind::Chars: return false;
        case Node::Kind::Concat:
            return std::all_of(n.kids.begin(), n.kids.end(), [](const Node& k) { return nullable(k); });
        case Node::Kind::Alt:
            return std::any_of(n.kids.begin(), n.kids.end(), [](const Node& k) { return nullable(k); });
        case Node::Kind::Repeat: return n.min == 0 || nullable(n.kids[0]);
        case Node::Kind::Atomic: return nullable(n.kids[0]);
    }
    return true;
}

}  // namespace

uint8_t regex_key(uint32_t cp) noexcept {
    return static_cast<uint8_t>(unicode::category(cp) | (unicode::is_whitespace(cp) ? 32 : 0));
}

bool Regex::Class::contains(uint32_t cp, uint8_t key) const noexcept {
    if (cp < 0x80) return (ascii[cp >> 6] >> (cp & 63)) & 1;
    bool in = (keys >> key) & 1;
    if (!in && !ranges.empty()) {
        size_t lo = 0, hi = ranges.size();
        while (lo < hi) {
            const size_t mid = (lo + hi) / 2;
            if (cp < ranges[mid].first) hi = mid;
            else if (cp > ranges[mid].second) lo = mid + 1;
            else {
                in = true;
                break;
            }
        }
    }
    return in != negated;
}

// ---------------------------------------------------------------------------
// Parser and compiler
// ---------------------------------------------------------------------------
class RegexCompiler {
public:
    explicit RegexCompiler(std::string_view pattern) : source_(pattern) {
        for (size_t i = 0; i < pattern.size();) {
            offsets_.push_back(i);
            p_.push_back(unicode::decode_utf8(pattern, i));
        }
        offsets_.push_back(pattern.size());
    }

    Result<Regex> compile() {
        Node root;
        if (!parse_alt(root, false)) return error_status();
        if (pos_ < p_.size()) {
            fail("unbalanced ')'");
            return error_status();
        }
        Regex re;
        re.pattern_ = std::string(source_);
        for (const ClassBuilder& b : classes_) re.classes_.push_back(b.build());
        program_ = &re.program_;
        if (!emit(root)) return error_status();
        re.program_.push_back({});  // Match
        compute_first(re);
        return re;
    }

private:
    std::string_view source_;
    std::vector<uint32_t> p_;
    std::vector<size_t> offsets_;
    size_t pos_ = 0;
    bool fold_ = false;  // inside a case-insensitive group
    std::vector<ClassBuilder> classes_;
    std::vector<Regex::Inst>* program_ = nullptr;
    std::string error_;

    bool fail(const std::string& what) {
        if (error_.empty()) {
            error_ = "regex '" + std::string(source_) + "': " + what + " at byte " +
                     std::to_string(offsets_[std::min(pos_, offsets_.size() - 1)]);
        }
        return false;
    }
    Status error_status() const { return Status(ErrorCode::InvalidArgument, error_); }

    bool at_end() const { return pos_ >= p_.size(); }
    uint32_t peek(size_t ahead = 0) const { return pos_ + ahead < p_.size() ? p_[pos_ + ahead] : 0; }
    bool eat(uint32_t c) {
        if (at_end() || p_[pos_] != c) return false;
        ++pos_;
        return true;
    }

    uint32_t new_class(ClassBuilder b) {
        classes_.push_back(std::move(b));
        return static_cast<uint32_t>(classes_.size() - 1);
    }
    Node chars(ClassBuilder b) {
        Node n;
        n.kind = Node::Kind::Chars;
        n.cls = new_class(std::move(b));
        return n;
    }

    // alternation := concat ('|' concat)*
    bool parse_alt(Node& out, bool in_group) {
        Node alt;
        alt.kind = Node::Kind::Alt;
        for (;;) {
            Node seq;
            if (!parse_concat(seq)) return false;
            alt.kids.push_back(std::move(seq));
            if (!eat('|')) break;
        }
        if (!in_group && !at_end() && peek() == ')') return fail("unbalanced ')'");
        out = alt.kids.size() == 1 ? std::move(alt.kids[0]) : std::move(alt);
        return true;
    }

    bool parse_concat(Node& out) {
        Node seq;
        seq.kind = Node::Kind::Concat;
        while (!at_end() && peek() != '|' && peek() != ')') {
            Node atom;
            if (!parse_atom(atom)) return false;
            if (!parse_quantifier(atom)) return false;
            if (atom.kind != Node::Kind::Empty) seq.kids.push_back(std::move(atom));
        }
        if (seq.kids.empty()) out = Node{};
        else if (seq.kids.size() == 1) out = std::move(seq.kids[0]);
        else out = std::move(seq);
        return true;
    }

    bool parse_number(uint32_t& v) {
        if (at_end() || peek() < '0' || peek() > '9') return false;
        uint64_t n = 0;
        while (!at_end() && peek() >= '0' && peek() <= '9') {
            n = n * 10 + (p_[pos_++] - '0');
            if (n > kMaxCopies * 100) return false;
        }
        v = static_cast<uint32_t>(n);
        return true;
    }

    bool parse_quantifier(Node& atom) {
        uint32_t min = 0, max = 0;
        if (eat('*')) {
            max = kUnbounded;
        } else if (eat('+')) {
            min = 1;
            max = kUnbounded;
        } else if (eat('?')) {
            max = 1;
        } else if (peek() == '{') {
            // {n}, {n,}, {n,m}, {,m}; anything else is a literal '{' (as both reference engines read it).
            const size_t save = pos_;
            ++pos_;
            const bool has_min = parse_number(min);
            if (eat(',')) {
                if (!parse_number(max)) max = kUnbounded;
            } else {
                max = min;
            }
            if (!eat('}') || (!has_min && max == kUnbounded)) {
                pos_ = save;
                return true;
            }
            if (max < min) return fail("repetition {n,m} with m < n");
        } else {
            return true;
        }
        if (atom.kind == Node::Kind::Assert || atom.kind == Node::Kind::Look || atom.kind == Node::Kind::Empty) {
            return fail("quantifier without something to repeat");
        }
        Regex::Mode mode = Regex::Mode::Greedy;
        if (eat('?')) mode = Regex::Mode::Lazy;
        else if (eat('+')) mode = Regex::Mode::Possessive;
        if (peek() == '*' || peek() == '+' || peek() == '?' || peek() == '{') {
            if (peek() != '{') return fail("nested quantifier");
        }
        Node rep;
        rep.kind = Node::Kind::Repeat;
        rep.min = min;
        rep.max = max;
        rep.mode = mode;
        rep.kids.push_back(std::move(atom));
        atom = std::move(rep);
        return true;
    }

    bool parse_atom(Node& out) {
        const uint32_t c = p_[pos_++];
        switch (c) {
            case '(': return parse_group(out);
            case '[': {
                ClassBuilder b;
                if (!parse_class(b)) return false;
                out = chars(std::move(b));
                return true;
            }
            case '.': {
                ClassBuilder b;
                b.negated = true;
                b.add('\n', '\n', false);
                out = chars(std::move(b));
                return true;
            }
            case '^': out = anchor(Regex::Anchor::LineStart); return true;
            case '$': out = anchor(Regex::Anchor::LineEnd); return true;
            case '\\': return parse_escape_atom(out);
            case '*': case '+': case '?': --pos_; return fail("quantifier without something to repeat");
            default: {
                ClassBuilder b;
                b.add(c, c, fold_);
                out = chars(std::move(b));
                return true;
            }
        }
    }

    static Node anchor(Regex::Anchor a) {
        Node n;
        n.kind = Node::Kind::Assert;
        n.anchor = a;
        return n;
    }

    bool parse_group(Node& out) {
        const bool saved_fold = fold_;
        Node::Kind kind = Node::Kind::Concat;  // plain group
        bool negate = false;
        if (eat('?')) {
            if (eat(':')) {
            } else if (eat('=')) {
                kind = Node::Kind::Look;
            } else if (eat('!')) {
                kind = Node::Kind::Look;
                negate = true;
            } else if (eat('>')) {
                kind = Node::Kind::Atomic;
            } else if (peek() == '<' && (peek(1) == '=' || peek(1) == '!')) {
                return fail("lookbehind is not supported");
            } else if (peek() == 'i') {
                ++pos_;
                if (eat(')')) {  // (?i): the rest of the enclosing group
                    fold_ = true;
                    out = Node{};
                    return true;
                }
                if (!eat(':')) return fail("unsupported group flags");
                fold_ = true;
            } else {
                return fail("unsupported group syntax");
            }
        }
        Node body;
        if (!parse_alt(body, true)) return false;
        fold_ = saved_fold;
        if (!eat(')')) return fail("missing ')'");
        if (kind == Node::Kind::Look || kind == Node::Kind::Atomic) {
            out.kind = kind;
            out.negate = negate;
            out.kids.push_back(std::move(body));
        } else {
            out = std::move(body);
        }
        return true;
    }

    // \p{..} \P{..} \pX: true and the class when the escape is one.
    bool parse_property(uint32_t letter, ClassBuilder& b, bool& ok) {
        if (letter != 'p' && letter != 'P') return false;
        std::string name;
        if (eat('{')) {
            bool negate_inner = eat('^');
            while (!at_end() && peek() != '}') {
                const uint32_t ch = p_[pos_++];
                if (ch != ' ' && ch != '_') name += static_cast<char>(ch < 0x80 ? ch : '?');
                else if (ch == '_') name += '_';
            }
            if (!eat('}')) {
                ok = fail("missing '}' in \\p{...}");
                return true;
            }
            if (negate_inner) letter = letter == 'p' ? 'P' : 'p';
        } else if (!at_end()) {
            name = static_cast<char>(p_[pos_++]);
        }
        uint32_t mask = 0;
        if (!category_mask(name, mask)) {
            ok = fail("unsupported property \\p{" + name + "} (general categories only)");
            return true;
        }
        if (letter == 'P') mask = ~mask & ((1u << unicode::kCategoryCount) - 1);
        b.keys |= category_keys(mask);
        ok = true;
        return true;
    }

    // Shorthand classes \s \S \d \D \w \W.
    static bool shorthand(uint32_t letter, ClassBuilder& b) {
        using namespace unicode;
        const uint32_t word = cat_bit(Lu) | cat_bit(Ll) | cat_bit(Lt) | cat_bit(Lm) | cat_bit(Lo) | cat_bit(Mn) |
                              cat_bit(Mc) | cat_bit(Me) | cat_bit(Nd) | cat_bit(Pc);
        switch (letter) {
            case 's': b.keys |= kWhiteSpaceKeys; return true;
            case 'S': b.keys |= kAllKeys & ~kWhiteSpaceKeys; return true;
            case 'd': b.keys |= category_keys(cat_bit(Nd)); return true;
            case 'D': b.keys |= category_keys(~cat_bit(Nd)); return true;
            case 'w': b.keys |= category_keys(word); return true;
            case 'W': b.keys |= category_keys(~word); return true;
            default: return false;
        }
    }

    // A single code point after '\' (escape sequences and escaped punctuation).
    bool parse_escaped_char(uint32_t letter, uint32_t& cp) {
        auto hex = [&](size_t digits, bool braced) {
            uint32_t v = 0;
            size_t n = 0;
            while (!at_end() && (braced || n < digits)) {
                const uint32_t h = peek();
                uint32_t d;
                if (h >= '0' && h <= '9') d = h - '0';
                else if (h >= 'a' && h <= 'f') d = h - 'a' + 10;
                else if (h >= 'A' && h <= 'F') d = h - 'A' + 10;
                else break;
                v = v * 16 + d;
                ++pos_;
                ++n;
                if (v > 0x10FFFF) return fail("code point out of range");
            }
            if (n == 0 || (!braced && n != digits)) return fail("bad hexadecimal escape");
            cp = v;
            return true;
        };
        switch (letter) {
            case 'n': cp = '\n'; return true;
            case 'r': cp = '\r'; return true;
            case 't': cp = '\t'; return true;
            case 'f': cp = '\f'; return true;
            case 'v': cp = '\v'; return true;
            case 'e': cp = 0x1B; return true;
            case 'a': cp = 0x07; return true;
            case '0': cp = 0; return true;
            case 'x':
                if (eat('{')) return hex(0, true) && (eat('}') || fail("missing '}'"));
                return hex(2, false);
            case 'u': return hex(4, false);
            case 'U': return hex(8, false);
            default:
                if (letter < 0x80 && ((letter >= 'a' && letter <= 'z') || (letter >= 'A' && letter <= 'Z') ||
                                      (letter >= '0' && letter <= '9'))) {
                    return fail(std::string("unsupported escape \\") + static_cast<char>(letter));
                }
                cp = letter;  // escaped punctuation or any other character: itself
                return true;
        }
    }

    bool parse_escape_atom(Node& out) {
        if (at_end()) return fail("pattern ends with '\\'");
        const uint32_t letter = p_[pos_++];
        switch (letter) {
            case 'A': out = anchor(Regex::Anchor::TextStart); return true;
            case 'z': out = anchor(Regex::Anchor::TextEnd); return true;
            case 'Z': out = anchor(Regex::Anchor::TextEndNewline); return true;
            case 'b': case 'B': return fail("word boundaries are not supported");
            default: break;
        }
        if (letter >= '1' && letter <= '9') return fail("backreferences are not supported");
        ClassBuilder b;
        bool ok = true;
        if (parse_property(letter, b, ok)) {
            if (!ok) return false;
        } else if (!shorthand(letter, b)) {
            uint32_t cp = 0;
            if (!parse_escaped_char(letter, cp)) return false;
            b.add(cp, cp, fold_);
        }
        out = chars(std::move(b));
        return true;
    }

    // After '[': members up to the closing ']'.
    bool parse_class(ClassBuilder& b) {
        b.negated = eat('^');
        bool first = true;
        for (;;) {
            if (at_end()) return fail("missing ']'");
            uint32_t c = p_[pos_++];
            if (c == ']' && !first) return true;
            first = false;
            if (c == '[') {
                if (peek() == ':') return fail("POSIX classes are not supported");
                return fail("nested classes are not supported");
            }
            uint32_t lo = c;
            if (c == '\\') {
                if (at_end()) return fail("missing ']'");
                const uint32_t letter = p_[pos_++];
                bool ok = true;
                if (parse_property(letter, b, ok)) {
                    if (!ok) return false;
                    continue;
                }
                if (shorthand(letter, b)) continue;
                if (letter == 'b') {
                    lo = 0x08;  // backspace inside a class
                } else if (!parse_escaped_char(letter, lo)) {
                    return false;
                }
            }
            // A range lo-hi, unless '-' is last or the end is a class.
            if (peek() == '-' && peek(1) != ']' && pos_ + 1 < p_.size()) {
                ++pos_;
                uint32_t hi = p_[pos_++];
                if (hi == '\\') {
                    if (at_end()) return fail("missing ']'");
                    const uint32_t letter = p_[pos_++];
                    if (letter == 'p' || letter == 'P' || letter == 's' || letter == 'S' || letter == 'd' ||
                        letter == 'D' || letter == 'w' || letter == 'W') {
                        return fail("a range cannot end with a class");
                    }
                    if (!parse_escaped_char(letter, hi)) return false;
                }
                if (hi < lo) return fail("range out of order");
                b.add(lo, hi, fold_);
            } else {
                b.add(lo, lo, fold_);
            }
        }
    }

    // ---- code generation --------------------------------------------------
    uint32_t here() const { return static_cast<uint32_t>(program_->size()); }
    uint32_t push(Regex::Inst in) {
        program_->push_back(in);
        return here() - 1;
    }

    bool emit(const Node& n) {
        using Op = Regex::Op;
        switch (n.kind) {
            case Node::Kind::Empty: return true;
            case Node::Kind::Chars: {
                Regex::Inst in;
                in.op = Op::Class;
                in.a = n.cls;
                push(in);
                return true;
            }
            case Node::Kind::Concat:
                for (const Node& k : n.kids) {
                    if (!emit(k)) return false;
                }
                return true;
            case Node::Kind::Alt: {
                std::vector<uint32_t> jumps;
                for (size_t i = 0; i < n.kids.size(); ++i) {
                    uint32_t split = 0;
                    if (i + 1 < n.kids.size()) {
                        Regex::Inst s;
                        s.op = Op::Split;
                        split = push(s);
                        (*program_)[split].a = here();
                    }
                    if (!emit(n.kids[i])) return false;
                    if (i + 1 < n.kids.size()) {
                        Regex::Inst j;
                        j.op = Op::Jump;
                        jumps.push_back(push(j));
                        (*program_)[split].b = here();
                    }
                }
                for (const uint32_t j : jumps) (*program_)[j].a = here();
                return true;
            }
            case Node::Kind::Assert: {
                Regex::Inst in;
                in.op = Op::Assert;
                in.anchor = n.anchor;
                push(in);
                return true;
            }
            case Node::Kind::Look:
            case Node::Kind::Atomic: {
                Regex::Inst in;
                in.op = n.kind == Node::Kind::Look ? Op::Look : Op::Atomic;
                in.negate = n.negate;
                const uint32_t at = push(in);
                if (!emit(n.kids[0])) return false;
                push(Regex::Inst{});  // Match ends the sub-program
                (*program_)[at].a = here();
                return true;
            }
            case Node::Kind::Repeat: return emit_repeat(n);
        }
        return false;
    }

    bool emit_repeat(const Node& n) {
        using Op = Regex::Op;
        const Node& body = n.kids[0];
        if (body.kind == Node::Kind::Chars) {
            Regex::Inst in;
            in.op = Op::Repeat;
            in.a = body.cls;
            in.min = n.min;
            in.max = n.max;
            in.mode = n.mode;
            push(in);
            return true;
        }
        if (n.mode == Regex::Mode::Possessive) {  // X*+ is (?>X*)
            Node atomic;
            atomic.kind = Node::Kind::Atomic;
            Node greedy = n;
            greedy.mode = Regex::Mode::Greedy;
            atomic.kids.push_back(std::move(greedy));
            return emit(atomic);
        }
        if (n.max == kUnbounded && nullable(body)) {
            return fail("a repeated group that can match the empty string is not supported");
        }
        if (n.min > kMaxCopies || (n.max != kUnbounded && n.max > kMaxCopies)) {
            return fail("repetition count too large");
        }
        for (uint32_t i = 0; i < n.min; ++i) {
            if (!emit(body)) return false;
        }
        const bool lazy = n.mode == Regex::Mode::Lazy;
        auto split_to = [&](uint32_t split, uint32_t body_at, uint32_t out) {
            (*program_)[split].a = lazy ? out : body_at;
            (*program_)[split].b = lazy ? body_at : out;
        };
        if (n.max == kUnbounded) {
            Regex::Inst s;
            s.op = Op::Split;
            const uint32_t split = push(s);
            const uint32_t body_at = here();
            if (!emit(body)) return false;
            Regex::Inst j;
            j.op = Op::Jump;
            j.a = split;
            push(j);
            split_to(split, body_at, here());
            return true;
        }
        std::vector<uint32_t> splits;
        for (uint32_t i = n.min; i < n.max; ++i) {
            Regex::Inst s;
            s.op = Op::Split;
            splits.push_back(push(s));
            (*program_)[splits.back()].a = here();  // body; out patched below
            if (!emit(body)) return false;
        }
        for (const uint32_t s : splits) split_to(s, s + 1, here());
        return true;
    }

    // First characters of a match, as one superset class (exact below U+0080):
    // lets search() skip positions where no match can start.
    void compute_first(Regex& re) {
        using Op = Regex::Op;
        Regex::Class first;
        std::vector<bool> seen(re.program_.size(), false);
        std::vector<uint32_t> work = {0};
        auto add = [&](const Regex::Class& c) {
            first.ascii[0] |= c.ascii[0];
            first.ascii[1] |= c.ascii[1];
            if (c.negated) {
                first.keys |= ~c.keys;
            } else {
                first.keys |= c.keys;
                first.ranges.insert(first.ranges.end(), c.ranges.begin(), c.ranges.end());
            }
        };
        while (!work.empty()) {
            const uint32_t pc = work.back();
            work.pop_back();
            if (seen[pc]) continue;
            seen[pc] = true;
            const Regex::Inst& in = re.program_[pc];
            switch (in.op) {
                case Op::Class: add(re.classes_[in.a]); break;
                case Op::Repeat:
                    if (in.max == 0) {
                        work.push_back(pc + 1);
                        break;
                    }
                    add(re.classes_[in.a]);
                    if (in.min == 0) work.push_back(pc + 1);
                    break;
                case Op::Split: work.push_back(in.a); work.push_back(in.b); break;
                case Op::Jump: work.push_back(in.a); break;
                case Op::Match: case Op::Assert: case Op::Look: case Op::Atomic: return;  // no prefilter
            }
        }
        if (first.ascii[0] == ~0ull && first.ascii[1] == ~0ull && first.keys == kAllKeys) return;  // useless
        std::sort(first.ranges.begin(), first.ranges.end());
        std::vector<std::pair<uint32_t, uint32_t>> merged;
        for (const auto& r : first.ranges) {
            if (!merged.empty() && r.first <= merged.back().second + 1) {
                merged.back().second = std::max(merged.back().second, r.second);
            } else {
                merged.push_back(r);
            }
        }
        first.ranges = std::move(merged);
        re.first_ = std::move(first);
        re.has_first_ = true;
    }
};

Result<Regex> Regex::compile(std::string_view pattern) { return RegexCompiler(pattern).compile(); }

// ---------------------------------------------------------------------------
// Matching: backtracking over the program with an explicit stack.
// ---------------------------------------------------------------------------
namespace {

struct Frame {
    enum Kind : uint8_t { Resume, RepeatGreedy, RepeatLazy };
    uint32_t pc;
    uint32_t count;  // RepeatGreedy / RepeatLazy: code points the repeat consumed
    size_t pos;
    Kind kind;
};

// One stack per thread: a compiled Regex is shared, and allocating a stack
// per match would cost more than most matches.
std::vector<Frame>& frame_stack() {
    thread_local std::vector<Frame> stack;
    return stack;
}

}  // namespace

bool Regex::run(const RegexSubject& s, uint32_t pc, size_t pos, size_t& end) const {
    std::vector<Frame>& stack = frame_stack();
    const size_t base = stack.size();
    const size_t n = s.size;
    auto member = [&](const Class& c, size_t at) { return at < n && c.contains(s.cps[at], s.keys[at]); };
    for (;;) {
        const Inst& in = program_[pc];
        bool ok = true;
        switch (in.op) {
            case Op::Class:
                if (member(classes_[in.a], pos)) {
                    ++pos;
                    ++pc;
                } else {
                    ok = false;
                }
                break;
            case Op::Repeat: {
                const Class& c = classes_[in.a];
                uint32_t k = 0;
                if (in.mode == Mode::Lazy) {
                    while (k < in.min && member(c, pos + k)) ++k;
                    if (k < in.min) {
                        ok = false;
                        break;
                    }
                    pos += k;
                    if (k < in.max) stack.push_back({pc, k, pos, Frame::RepeatLazy});
                } else {
                    while (k < in.max && member(c, pos + k)) ++k;
                    if (k < in.min) {
                        ok = false;
                        break;
                    }
                    pos += k;
                    if (in.mode == Mode::Greedy && k > in.min) stack.push_back({pc, k, pos, Frame::RepeatGreedy});
                }
                ++pc;
                break;
            }
            case Op::Split:
                stack.push_back({in.b, 0, pos, Frame::Resume});
                pc = in.a;
                break;
            case Op::Jump: pc = in.a; break;
            case Op::Match:
                end = pos;
                stack.resize(base);
                return true;
            case Op::Assert: {
                switch (in.anchor) {
                    case Anchor::LineStart: ok = pos == 0 || s.cps[pos - 1] == '\n'; break;
                    case Anchor::LineEnd: ok = pos == n || s.cps[pos] == '\n'; break;
                    case Anchor::TextStart: ok = pos == 0; break;
                    case Anchor::TextEnd: ok = pos == n; break;
                    case Anchor::TextEndNewline: ok = pos == n || (pos + 1 == n && s.cps[pos] == '\n'); break;
                }
                if (ok) ++pc;
                break;
            }
            case Op::Look: {
                size_t sub_end = 0;
                if (run(s, pc + 1, pos, sub_end) != in.negate) pc = in.a;
                else ok = false;
                break;
            }
            case Op::Atomic: {
                size_t sub_end = 0;
                if (run(s, pc + 1, pos, sub_end)) {
                    pos = sub_end;
                    pc = in.a;
                } else {
                    ok = false;
                }
                break;
            }
        }
        if (ok) continue;
        // Backtrack to the most recent choice that is left.
        for (;;) {
            if (stack.size() == base) return false;
            const Frame f = stack.back();
            stack.pop_back();
            if (f.kind == Frame::Resume) {
                pc = f.pc;
                pos = f.pos;
                break;
            }
            const Inst& rep = program_[f.pc];
            if (f.kind == Frame::RepeatGreedy) {  // give back one code point
                const uint32_t k = f.count - 1;
                pos = f.pos - 1;
                if (k > rep.min) stack.push_back({f.pc, k, pos, Frame::RepeatGreedy});
                pc = f.pc + 1;
                break;
            }
            if (member(classes_[rep.a], f.pos)) {  // lazy: take one more
                const uint32_t k = f.count + 1;
                pos = f.pos + 1;
                if (k < rep.max) stack.push_back({f.pc, k, pos, Frame::RepeatLazy});
                pc = f.pc + 1;
                break;
            }
        }
    }
}

bool Regex::match_at(const RegexSubject& s, size_t at, size_t& end) const { return run(s, 0, at, end); }

bool Regex::search(const RegexSubject& s, size_t from, size_t& begin, size_t& end) const {
    for (size_t p = from; p <= s.size; ++p) {
        if (has_first_ && (p == s.size || !first_.contains(s.cps[p], s.keys[p]))) continue;
        if (run(s, 0, p, end)) {
            begin = p;
            return true;
        }
    }
    return false;
}

}  // namespace liyab
