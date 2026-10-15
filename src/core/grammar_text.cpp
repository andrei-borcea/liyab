// Grammar text (GBNF-like, see liyab/grammar.h) → Builder.
#include <algorithm>
#include <optional>
#include <string>

#include "core/grammar.h"

namespace liyab::grammar {
namespace {

using Node = Builder::Node;

class TextParser {
public:
    TextParser(Builder& b, std::string_view text) : b_(b), text_(text) {}

    Result<uint32_t> run() {
        skip();
        if (pos_ == text_.size()) return error("empty grammar");
        while (pos_ < text_.size()) {
            const std::string name = read_name();
            if (name.empty()) return error("expected a rule name");
            skip();
            if (!text_.substr(pos_).starts_with("::=")) return error("expected '::=' after '" + name + "'");
            pos_ += 3;
            const uint32_t rule = b_.named_rule(name);
            if (b_.defined(rule)) return error("rule '" + name + "' is defined twice");
            auto body = alternatives();
            if (!body) return body.status();
            b_.define(rule, *body);
            skip();
        }
        return b_.named_rule("root");
    }

private:
    Status error(const std::string& what) const {
        size_t line = 1;
        for (size_t i = 0; i < pos_ && i < text_.size(); ++i) line += text_[i] == '\n';
        return Status(ErrorCode::InvalidArgument, "grammar: line " + std::to_string(line) + ": " + what);
    }

    void skip() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
                ++pos_;
            } else if (c == '#') {
                while (pos_ < text_.size() && text_[pos_] != '\n') ++pos_;
            } else {
                return;
            }
        }
    }

    static bool name_char(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    }

    std::string read_name() {
        const size_t start = pos_;
        while (pos_ < text_.size() && name_char(text_[pos_])) ++pos_;
        return std::string(text_.substr(start, pos_ - start));
    }

    // Whether a rule definition (`name ::=`) starts at the current position.
    bool at_definition() const {
        size_t p = pos_;
        while (p < text_.size() && name_char(text_[p])) ++p;
        if (p == pos_) return false;
        while (p < text_.size() && (text_[p] == ' ' || text_[p] == '\t' || text_[p] == '\n' || text_[p] == '\r')) ++p;
        return text_.substr(p).starts_with("::=");
    }

    Result<Node> alternatives() {
        std::vector<Node> alts;
        for (;;) {
            auto s = sequence();
            if (!s) return s;
            alts.push_back(*s);
            skip();
            if (pos_ < text_.size() && text_[pos_] == '|') {
                ++pos_;
                continue;
            }
            return b_.alt(std::move(alts));
        }
    }

    Result<Node> sequence() {
        std::vector<Node> items;
        for (;;) {
            skip();
            if (pos_ == text_.size() || text_[pos_] == '|' || text_[pos_] == ')' || at_definition()) break;
            auto item = primary();
            if (!item) return item;
            Node node = *item;
            for (;;) {  // postfix operators
                if (pos_ >= text_.size()) break;
                const char c = text_[pos_];
                if (c == '*') {
                    node = b_.repeat(node, 0, kUnbounded);
                } else if (c == '+') {
                    node = b_.repeat(node, 1, kUnbounded);
                } else if (c == '?') {
                    node = b_.repeat(node, 0, 1);
                } else if (c == '{') {
                    auto bounds = repetition();
                    if (!bounds) return bounds.status();
                    node = b_.repeat(node, bounds->first, bounds->second);
                    continue;
                } else {
                    break;
                }
                ++pos_;
            }
            items.push_back(node);
        }
        return b_.seq(std::move(items));
    }

    Result<std::pair<uint32_t, uint32_t>> repetition() {
        ++pos_;  // '{'
        auto number = [&]() -> std::optional<uint32_t> {
            skip();
            const size_t start = pos_;
            uint64_t v = 0;
            while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9' && v <= kMaxRepeat) {
                v = v * 10 + static_cast<uint64_t>(text_[pos_++] - '0');
            }
            if (pos_ == start) return std::nullopt;
            return static_cast<uint32_t>(std::min<uint64_t>(v, kMaxRepeat + 1));
        };
        const auto min = number();
        if (!min) return error("expected a number after '{'");
        uint32_t max = *min;
        skip();
        if (pos_ < text_.size() && text_[pos_] == ',') {
            ++pos_;
            skip();
            if (pos_ < text_.size() && text_[pos_] == '}') {
                max = kUnbounded;
            } else {
                const auto m = number();
                if (!m) return error("expected a number or '}' after ','");
                max = *m;
            }
            skip();
        }
        if (pos_ >= text_.size() || text_[pos_] != '}') return error("expected '}'");
        ++pos_;
        if (*min > kMaxRepeat || (max != kUnbounded && max > kMaxRepeat)) {
            return error("repetition bounds above " + std::to_string(kMaxRepeat) + " are not supported");
        }
        if (max != kUnbounded && max < *min) return error("repetition maximum below its minimum");
        return std::pair<uint32_t, uint32_t>{*min, max};
    }

    static void append_utf8(std::string& out, uint32_t cp) {
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

    // One character of a literal or class (an escape or a UTF-8 code point).
    Result<uint32_t> character() {
        if (pos_ >= text_.size()) return error("unexpected end of the grammar");
        const auto c = static_cast<unsigned char>(text_[pos_]);
        if (c == '\\') {
            if (++pos_ >= text_.size()) return error("unfinished escape");
            const char e = text_[pos_++];
            auto hex = [&](int digits) -> Result<uint32_t> {
                uint32_t v = 0;
                for (int i = 0; i < digits; ++i) {
                    if (pos_ >= text_.size()) return error("unfinished escape");
                    const char h = text_[pos_++];
                    v <<= 4;
                    if (h >= '0' && h <= '9') v |= static_cast<uint32_t>(h - '0');
                    else if (h >= 'a' && h <= 'f') v |= static_cast<uint32_t>(h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F') v |= static_cast<uint32_t>(h - 'A' + 10);
                    else return error("bad hexadecimal digit in an escape");
                }
                if (v > 0x10FFFF || (v >= 0xD800 && v <= 0xDFFF)) return error("escape is not a Unicode scalar value");
                return v;
            };
            switch (e) {
                case 'n': return uint32_t{'\n'};
                case 'r': return uint32_t{'\r'};
                case 't': return uint32_t{'\t'};
                case 'x': return hex(2);
                case 'u': return hex(4);
                case 'U': return hex(8);
                case '\\': case '"': case '\'': case '[': case ']': case '-': case '^': case '/':
                    return static_cast<uint32_t>(e);
                default: return error(std::string("unknown escape '\\") + e + "'");
            }
        }
        // A UTF-8 code point of the source.
        int n = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
        if (n == 0 || pos_ + static_cast<size_t>(n) > text_.size()) return error("invalid UTF-8 in the grammar");
        uint32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
        for (int i = 1; i < n; ++i) {
            const auto cc = static_cast<unsigned char>(text_[pos_ + static_cast<size_t>(i)]);
            if ((cc & 0xC0) != 0x80) return error("invalid UTF-8 in the grammar");
            cp = cp << 6 | (cc & 0x3F);
        }
        pos_ += static_cast<size_t>(n);
        return cp;
    }

    Result<std::string> string_literal() {
        ++pos_;  // opening quote
        std::string out;
        while (pos_ < text_.size() && text_[pos_] != '"') {
            if (text_[pos_] == '\n') return error("newline inside a string (write \\n)");
            auto cp = character();
            if (!cp) return cp.status();
            append_utf8(out, *cp);
        }
        if (pos_ >= text_.size()) return error("unterminated string");
        ++pos_;
        return out;
    }

    Result<Node> char_class() {
        ++pos_;  // '['
        bool negated = false;
        if (pos_ < text_.size() && text_[pos_] == '^') {
            negated = true;
            ++pos_;
        }
        std::vector<CpRange> ranges;
        while (pos_ < text_.size() && text_[pos_] != ']') {
            auto lo = character();
            if (!lo) return lo.status();
            uint32_t hi = *lo;
            if (pos_ + 1 < text_.size() && text_[pos_] == '-' && text_[pos_ + 1] != ']') {
                ++pos_;
                auto h = character();
                if (!h) return h.status();
                hi = *h;
                if (hi < *lo) return error("character range out of order");
            }
            ranges.push_back({*lo, hi});
        }
        if (pos_ >= text_.size()) return error("unterminated character class");
        ++pos_;
        return b_.chars(std::move(ranges), negated);
    }

    // The extent of a JSON value starting at pos_ (objects / arrays
    // balanced, strings skipped; or a bare word such as true).
    Result<std::string_view> json_extent() {
        skip();
        const size_t start = pos_;
        if (pos_ >= text_.size()) return error("expected a JSON schema after %schema");
        if (text_[pos_] != '{' && text_[pos_] != '[') {
            while (pos_ < text_.size() && name_char(text_[pos_])) ++pos_;
            if (pos_ == start) return error("expected a JSON schema after %schema");
            return text_.substr(start, pos_ - start);
        }
        int depth = 0;
        bool in_string = false;
        for (; pos_ < text_.size(); ++pos_) {
            const char c = text_[pos_];
            if (in_string) {
                if (c == '\\') ++pos_;
                else if (c == '"') in_string = false;
                continue;
            }
            if (c == '"') in_string = true;
            else if (c == '{' || c == '[') ++depth;
            else if ((c == '}' || c == ']') && --depth == 0) {
                ++pos_;
                return text_.substr(start, pos_ - start);
            }
        }
        return error("unterminated JSON schema");
    }

    Result<Node> primary() {
        const char c = text_[pos_];
        if (c == '"') {
            auto s = string_literal();
            if (!s) return s.status();
            return b_.literal(*s);
        }
        if (c == '[') return char_class();
        if (c == '.') {
            ++pos_;
            return b_.any_char();
        }
        if (c == '(') {
            ++pos_;
            auto inner = alternatives();
            if (!inner) return inner;
            skip();
            if (pos_ >= text_.size() || text_[pos_] != ')') return error("expected ')'");
            ++pos_;
            return inner;
        }
        if (c == '%') {
            ++pos_;
            const std::string word = read_name();
            if (word == "json") {
                if (!json_) json_ = json_value(b_, JsonFormat{});
                return *json_;
            }
            if (word == "schema" || word == "string") {
                auto extent = json_extent();
                if (!extent) return extent.status();
                auto node = word == "schema" ? json_schema(b_, *extent, JsonFormat{}) : raw_string(b_, *extent);
                if (!node) return Status(node.status().code(), error(node.status().message()).message());
                return node;
            }
            if (word == "until") {
                skip();
                if (pos_ >= text_.size() || text_[pos_] != '"') return error("expected a string after %until");
                auto s = string_literal();
                if (!s) return s.status();
                if (s->empty()) return error("%until needs a non-empty string");
                return until(*s);
            }
            return error("unknown builtin '%" + word + "'");
        }
        if (name_char(c)) {
            const std::string name = read_name();
            return b_.ref(b_.named_rule(name));
        }
        return error(std::string("unexpected character '") + c + "'");
    }

    // Any text without `stop`, then `stop`: a string-matching automaton over
    // the stop's code points (state k: its first k characters were just
    // read). From state k, a character goes to the longest prefix of the stop
    // that ends the text read so far; one completing the stop ends the match.
    Node until(const std::string& stop) {
        std::vector<uint32_t> cps;
        for (size_t i = 0; i < stop.size();) {
            const auto c = static_cast<unsigned char>(stop[i]);
            const int n = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
            uint32_t cp = n == 1 ? c : c & (0xFF >> (n + 1));
            for (int k = 1; k < n; ++k) cp = cp << 6 | (static_cast<unsigned char>(stop[i + static_cast<size_t>(k)]) & 0x3F);
            cps.push_back(cp);
            i += static_cast<size_t>(n);
        }
        const size_t m = cps.size();
        std::vector<uint32_t> distinct(cps);
        std::sort(distinct.begin(), distinct.end());
        distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
        // Next state after reading `c` in state k.
        auto next = [&](size_t k, uint32_t c) {
            for (size_t len = std::min(k + 1, m); len > 0; --len) {
                // Does cps[0..len) equal (cps[k-len+1..k) + c)?
                if (cps[len - 1] != c) continue;
                bool ok = true;
                for (size_t j = 0; j + 1 < len; ++j) {
                    if (cps[j] != cps[k - (len - 1) + j]) {
                        ok = false;
                        break;
                    }
                }
                if (ok) return len;
            }
            return size_t{0};
        };
        std::vector<uint32_t> rules(m);
        for (size_t k = 0; k < m; ++k) rules[k] = b_.add_rule("until");
        for (size_t k = 0; k < m; ++k) {
            std::vector<std::vector<CpRange>> to(m + 1);  // characters per target state
            for (const uint32_t c : distinct) to[next(k, c)].push_back({c, c});
            std::vector<Node> alts;
            for (size_t t = 0; t <= m; ++t) {
                if (t == 0) {
                    // Every character outside the stop's own, plus those that lead back to 0.
                    std::vector<CpRange> exclude;
                    for (const uint32_t c : distinct) {
                        if (next(k, c) != 0) exclude.push_back({c, c});
                    }
                    alts.push_back(b_.seq({b_.chars(exclude, true), b_.ref(rules[0])}));
                    continue;
                }
                if (to[t].empty()) continue;
                const Node cls = b_.chars(to[t]);
                alts.push_back(t == m ? cls : b_.seq({cls, b_.ref(rules[t])}));
            }
            b_.define(rules[k], b_.alt(std::move(alts)));
        }
        return b_.ref(rules[0]);
    }

    Builder& b_;
    std::string_view text_;
    size_t pos_ = 0;
    std::optional<Node> json_;
};

}  // namespace

Result<uint32_t> parse_text(Builder& b, std::string_view text) {
    TextParser parser(b, text);
    auto root = parser.run();
    if (!root) return root;
    if (!b.defined(*root)) return Status(ErrorCode::InvalidArgument, "grammar: no rule named 'root'");
    return root;
}

}  // namespace liyab::grammar
