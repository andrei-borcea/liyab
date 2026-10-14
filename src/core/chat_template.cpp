#include "core/chat_template.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <map>
#include <optional>
#include <stdexcept>
#include <unordered_map>

#include "core/unicode.h"

namespace liyab {

namespace jinja {

namespace {

// Errors inside the interpreter travel as exceptions and become a Status at
// the API boundary: evaluation is deeply recursive, and every level would
// otherwise have to forward a Result.
struct TemplateError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
[[noreturn]] void fail(const std::string& message) { throw TemplateError(message); }

const char* type_name(Value::Type t) {
    switch (t) {
        case Value::Type::Undefined: return "undefined";
        case Value::Type::None: return "NoneType";
        case Value::Type::Bool: return "bool";
        case Value::Type::Int: return "int";
        case Value::Type::Float: return "float";
        case Value::Type::String: return "str";
        case Value::Type::Array: return "list";
        case Value::Type::Object: return "dict";
        case Value::Type::Callable: return "function";
    }
    return "?";
}

// Python's repr() of a float: the shortest digits that round-trip, in fixed
// notation for exponents -4..15 and scientific (e+XX) otherwise.
std::string float_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string sci(buf, res.ptr);
    const size_t e = sci.find('e');
    std::string mantissa = sci.substr(0, e);
    const int exponent = std::stoi(sci.substr(e + 1));
    std::string sign;
    if (mantissa[0] == '-') {
        sign = "-";
        mantissa.erase(0, 1);
    }
    std::string digits;
    for (const char c : mantissa) {
        if (c != '.') digits += c;
    }
    if (exponent >= -4 && exponent < 16) {
        std::string out;
        if (exponent < 0) {
            out = "0." + std::string(static_cast<size_t>(-exponent - 1), '0') + digits;
        } else if (static_cast<size_t>(exponent) + 1 >= digits.size()) {
            out = digits + std::string(static_cast<size_t>(exponent) + 1 - digits.size(), '0') + ".0";
        } else {
            out = digits.substr(0, static_cast<size_t>(exponent) + 1) + "." + digits.substr(static_cast<size_t>(exponent) + 1);
        }
        return sign + out;
    }
    std::string out = digits.substr(0, 1);
    if (digits.size() > 1) out += "." + digits.substr(1);
    char exp[16];
    std::snprintf(exp, sizeof exp, "e%c%02d", exponent < 0 ? '-' : '+', std::abs(exponent));
    return sign + out + exp;
}

// Whether Python's repr() shows a code point as is (str.isprintable()).
bool printable(uint32_t cp) {
    if (cp == ' ') return true;
    const auto c = unicode::category(cp);
    return !(c == unicode::Cc || c == unicode::Cf || c == unicode::Cs || c == unicode::Co || c == unicode::Cn ||
             c == unicode::Zl || c == unicode::Zp || c == unicode::Zs);
}

std::string string_repr(const std::string& s) {
    const bool single = s.find('\'') == std::string::npos || s.find('"') != std::string::npos;
    const char quote = single ? '\'' : '"';
    std::string out(1, quote);
    for (size_t i = 0; i < s.size();) {
        const size_t at = i;
        const uint32_t cp = unicode::decode_utf8(s, i);
        if (cp == static_cast<uint32_t>(quote) || cp == '\\') {
            out += '\\';
            out += static_cast<char>(cp);
        } else if (cp == '\n') {
            out += "\\n";
        } else if (cp == '\r') {
            out += "\\r";
        } else if (cp == '\t') {
            out += "\\t";
        } else if (printable(cp)) {
            out.append(s, at, i - at);
        } else {
            char buf[16];
            if (cp < 0x100) std::snprintf(buf, sizeof buf, "\\x%02x", cp);
            else if (cp < 0x10000) std::snprintf(buf, sizeof buf, "\\u%04x", cp);
            else std::snprintf(buf, sizeof buf, "\\U%08x", cp);
            out += buf;
        }
    }
    return out + quote;
}

void json_string(std::string& out, const std::string& s, bool ensure_ascii) {
    out += '"';
    for (size_t i = 0; i < s.size();) {
        const size_t at = i;
        const uint32_t cp = unicode::decode_utf8(s, i);
        switch (cp) {
            case '"': out += "\\\""; continue;
            case '\\': out += "\\\\"; continue;
            case '\n': out += "\\n"; continue;
            case '\r': out += "\\r"; continue;
            case '\t': out += "\\t"; continue;
            case '\b': out += "\\b"; continue;
            case '\f': out += "\\f"; continue;
            default: break;
        }
        char buf[16];
        if (cp < 0x20 || (ensure_ascii && cp >= 0x80)) {
            if (cp >= 0x10000) {
                const uint32_t v = cp - 0x10000;
                std::snprintf(buf, sizeof buf, "\\u%04x\\u%04x", 0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF));
            } else {
                std::snprintf(buf, sizeof buf, "\\u%04x", cp);
            }
            out += buf;
        } else {
            out.append(s, at, i - at);
        }
    }
    out += '"';
}

}  // namespace

// ---------------------------------------------------------------------------
// Value
// ---------------------------------------------------------------------------
Value Value::undefined(std::string name) {
    Value v;
    v.s_ = std::move(name);
    return v;
}
Value Value::none() {
    Value v;
    v.type_ = Type::None;
    return v;
}
Value::Value(bool b) : type_(Type::Bool), b_(b), i_(b ? 1 : 0) {}
Value::Value(int64_t i) : type_(Type::Int), i_(i) {}
Value::Value(double f) : type_(Type::Float), f_val_(f) {}
Value::Value(std::string s) : type_(Type::String), s_(std::move(s)) {}
Value Value::array(Array items) {
    Value v;
    v.type_ = Type::Array;
    v.a_ = std::make_shared<Array>(std::move(items));
    return v;
}
Value Value::object() {
    Value v;
    v.type_ = Type::Object;
    v.o_ = std::make_shared<Object>();
    return v;
}
Value Value::callable(std::string name, std::function<Value(Array&, Kwargs&)> fn) {
    Value v;
    v.type_ = Type::Callable;
    v.f_ = std::make_shared<Callable>(Callable{std::move(name), std::move(fn)});
    return v;
}

int64_t Value::as_int() const noexcept {
    if (type_ == Type::Float) return static_cast<int64_t>(f_val_);
    return i_;
}
double Value::as_float() const noexcept { return type_ == Type::Float ? f_val_ : static_cast<double>(i_); }

bool Value::truthy() const {
    switch (type_) {
        case Type::Undefined: case Type::None: return false;
        case Type::Bool: return b_;
        case Type::Int: return i_ != 0;
        case Type::Float: return f_val_ != 0.0;
        case Type::String: return !s_.empty();
        case Type::Array: return !a_->empty();
        case Type::Object: return !o_->entries.empty() || o_->is_namespace;
        case Type::Callable: return true;
    }
    return false;
}

std::string Value::str() const {
    switch (type_) {
        case Type::Undefined: return "";
        case Type::String: return s_;
        default: return repr();
    }
}

std::string Value::repr() const {
    switch (type_) {
        case Type::Undefined: return "";
        case Type::None: return "None";
        case Type::Bool: return b_ ? "True" : "False";
        case Type::Int: return std::to_string(i_);
        case Type::Float: return float_repr(f_val_);
        case Type::String: return string_repr(s_);
        case Type::Array: {
            std::string out = "[";
            for (size_t i = 0; i < a_->size(); ++i) out += (i ? ", " : "") + (*a_)[i].repr();
            return out + "]";
        }
        case Type::Object: {
            std::string out = o_->is_namespace ? "<Namespace {" : "{";
            for (size_t i = 0; i < o_->entries.size(); ++i) {
                out += (i ? ", " : "") + string_repr(o_->entries[i].first) + ": " + o_->entries[i].second.repr();
            }
            return out + (o_->is_namespace ? "}>" : "}");
        }
        case Type::Callable: return "<function " + f_->name + ">";
    }
    return "";
}

std::string Value::to_json(int indent, bool ensure_ascii, std::string_view item_sep, std::string_view key_sep,
                           bool sort_keys) const {
    // json.dumps defaults: ", " between items without indent, "," with it.
    const std::string isep(item_sep.empty() ? (indent >= 0 ? "," : ", ") : item_sep);
    const std::string ksep(key_sep.empty() ? ": " : key_sep);
    std::string out;
    std::function<void(const Value&, int)> dump = [&](const Value& v, int level) {
        auto newline = [&](int lvl) {
            if (indent < 0) return;
            out += '\n';
            out.append(static_cast<size_t>(indent * lvl), ' ');
        };
        switch (v.type_) {
            case Type::None: out += "null"; return;
            case Type::Bool: out += v.b_ ? "true" : "false"; return;
            case Type::Int: out += std::to_string(v.i_); return;
            case Type::Float:
                if (std::isnan(v.f_val_)) out += "NaN";
                else if (std::isinf(v.f_val_)) out += v.f_val_ < 0 ? "-Infinity" : "Infinity";
                else out += float_repr(v.f_val_);
                return;
            case Type::String: json_string(out, v.s_, ensure_ascii); return;
            case Type::Array: {
                if (v.a_->empty()) {
                    out += "[]";
                    return;
                }
                out += '[';
                for (size_t i = 0; i < v.a_->size(); ++i) {
                    if (i) out += isep;
                    newline(level + 1);
                    dump((*v.a_)[i], level + 1);
                }
                newline(level);
                out += ']';
                return;
            }
            case Type::Object: {
                if (v.o_->entries.empty()) {
                    out += "{}";
                    return;
                }
                std::vector<const std::pair<std::string, Value>*> entries;
                for (const auto& e : v.o_->entries) entries.push_back(&e);
                if (sort_keys) {
                    std::stable_sort(entries.begin(), entries.end(), [](auto* a, auto* b) { return a->first < b->first; });
                }
                out += '{';
                for (size_t i = 0; i < entries.size(); ++i) {
                    if (i) out += isep;
                    newline(level + 1);
                    json_string(out, entries[i]->first, ensure_ascii);
                    out += ksep;
                    dump(entries[i]->second, level + 1);
                }
                newline(level);
                out += '}';
                return;
            }
            case Type::Undefined: fail("Object of type Undefined is not JSON serializable");
            case Type::Callable: fail("Object of type function is not JSON serializable");
        }
    };
    dump(*this, 0);
    return out;
}

bool operator==(const Value& a, const Value& b) {
    using T = Value::Type;
    if (a.is_number() && b.is_number()) {
        if (a.type_ == T::Float || b.type_ == T::Float) return a.as_float() == b.as_float();
        return a.i_ == b.i_;
    }
    if (a.type_ != b.type_) return false;
    switch (a.type_) {
        case T::Undefined: case T::None: return true;
        case T::String: return a.s_ == b.s_;
        case T::Array: {
            if (a.a_ == b.a_) return true;
            if (a.a_->size() != b.a_->size()) return false;
            for (size_t i = 0; i < a.a_->size(); ++i) {
                if (!((*a.a_)[i] == (*b.a_)[i])) return false;
            }
            return true;
        }
        case T::Object: {
            if (a.o_ == b.o_) return true;
            if (a.o_->entries.size() != b.o_->entries.size()) return false;
            for (const auto& [k, v] : a.o_->entries) {
                const Value* other = b.o_->find(k);
                if (other == nullptr || !(v == *other)) return false;
            }
            return true;
        }
        case T::Callable: return a.f_ == b.f_;
        default: return false;
    }
}

const Value* Object::find(std::string_view key) const {
    for (const auto& [k, v] : entries) {
        if (k == key) return &v;
    }
    return nullptr;
}

void Object::set(const std::string& key, Value value) {
    for (auto& [k, v] : entries) {
        if (k == key) {
            v = std::move(value);
            return;
        }
    }
    entries.emplace_back(key, std::move(value));
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------
namespace {

struct JsonParser {
    std::string_view s;
    size_t pos = 0;
    std::string error;

    void skip() {
        while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) ++pos;
    }
    bool bad(const std::string& what) {
        if (error.empty()) error = "JSON: " + what + " at offset " + std::to_string(pos);
        return false;
    }
    bool hex4(uint32_t& v) {
        if (pos + 4 > s.size()) return bad("truncated \\u escape");
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[pos++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else return bad("bad \\u escape");
        }
        return true;
    }
    bool string(std::string& out) {
        ++pos;  // opening quote
        while (pos < s.size() && s[pos] != '"') {
            if (s[pos] != '\\') {
                out += s[pos++];
                continue;
            }
            if (++pos >= s.size()) return bad("truncated escape");
            const char c = s[pos++];
            switch (c) {
                case '"': out += '"'; break;
                case '\\': out += '\\'; break;
                case '/': out += '/'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;
                case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) return false;
                    if (cp >= 0xD800 && cp < 0xDC00 && pos + 1 < s.size() && s[pos] == '\\' && s[pos + 1] == 'u') {
                        pos += 2;
                        uint32_t low = 0;
                        if (!hex4(low)) return false;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    }
                    unicode::append_utf8(out, cp);
                    break;
                }
                default: return bad("bad escape");
            }
        }
        if (pos >= s.size()) return bad("unterminated string");
        ++pos;
        return true;
    }
    bool value(Value& out, int depth) {
        if (depth > 256) return bad("nested too deeply");
        skip();
        if (pos >= s.size()) return bad("unexpected end");
        const char c = s[pos];
        if (c == '{') {
            ++pos;
            out = Value::object();
            skip();
            if (pos < s.size() && s[pos] == '}') {
                ++pos;
                return true;
            }
            for (;;) {
                skip();
                if (pos >= s.size() || s[pos] != '"') return bad("expected a key");
                std::string key;
                if (!string(key)) return false;
                skip();
                if (pos >= s.size() || s[pos++] != ':') return bad("expected ':'");
                Value v;
                if (!value(v, depth + 1)) return false;
                out.fields().set(key, std::move(v));
                skip();
                if (pos < s.size() && s[pos] == ',') {
                    ++pos;
                    continue;
                }
                if (pos < s.size() && s[pos] == '}') {
                    ++pos;
                    return true;
                }
                return bad("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++pos;
            out = Value::array();
            skip();
            if (pos < s.size() && s[pos] == ']') {
                ++pos;
                return true;
            }
            for (;;) {
                Value v;
                if (!value(v, depth + 1)) return false;
                out.items().push_back(std::move(v));
                skip();
                if (pos < s.size() && s[pos] == ',') {
                    ++pos;
                    continue;
                }
                if (pos < s.size() && s[pos] == ']') {
                    ++pos;
                    return true;
                }
                return bad("expected ',' or ']'");
            }
        }
        if (c == '"') {
            std::string str;
            if (!string(str)) return false;
            out = Value(std::move(str));
            return true;
        }
        auto word = [&](std::string_view w) { return s.substr(pos, w.size()) == w; };
        if (word("true")) {
            pos += 4;
            out = Value(true);
            return true;
        }
        if (word("false")) {
            pos += 5;
            out = Value(false);
            return true;
        }
        if (word("null")) {
            pos += 4;
            out = Value::none();
            return true;
        }
        const size_t start = pos;
        bool is_float = false;
        if (pos < s.size() && s[pos] == '-') ++pos;
        while (pos < s.size() && ((s[pos] >= '0' && s[pos] <= '9') || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E' ||
                                  s[pos] == '+' || s[pos] == '-')) {
            is_float = is_float || s[pos] == '.' || s[pos] == 'e' || s[pos] == 'E';
            ++pos;
        }
        if (pos == start) return bad("unexpected character");
        const std::string num(s.substr(start, pos - start));
        if (!is_float) {
            int64_t i = 0;
            const auto res = std::from_chars(num.data(), num.data() + num.size(), i);
            if (res.ec == std::errc() && res.ptr == num.data() + num.size()) {
                out = Value(i);
                return true;
            }
        }
        char* end = nullptr;
        const double f = std::strtod(num.c_str(), &end);
        if (end != num.c_str() + num.size()) return bad("bad number");
        out = Value(f);
        return true;
    }
};

}  // namespace

Result<Value> parse_json(std::string_view text) {
    JsonParser p{text, 0, {}};
    Value v;
    if (!p.value(v, 0)) return Status(ErrorCode::InvalidArgument, p.error);
    p.skip();
    if (p.pos != text.size()) return Status(ErrorCode::InvalidArgument, "JSON: trailing text at offset " + std::to_string(p.pos));
    return v;
}

}  // namespace jinja

// ---------------------------------------------------------------------------
// Template syntax tree
// ---------------------------------------------------------------------------
using jinja::Array;
using jinja::Object;
using jinja::Kwargs;
using jinja::Value;
using jinja::TemplateError;
using jinja::fail;

namespace {

struct Expr;
using ExprPtr = std::unique_ptr<Expr>;

struct Expr {
    enum class Kind { Literal, Name, List, Tuple, Dict, Attr, Index, Slice, Call, Filter, Test, Not, Neg, Pos, Binary,
                      And, Or, Ternary };
    Kind kind = Kind::Literal;
    Value literal;
    std::string name;  // Name, Attr, Filter, Test, Binary operator
    ExprPtr a, b, c;   // operands; Slice: a target, b/c/d bounds
    ExprPtr d;
    std::vector<ExprPtr> args;
    std::vector<std::pair<std::string, ExprPtr>> kwargs;  // Call / Filter / Test keyword arguments; Dict entries
    std::vector<std::pair<ExprPtr, ExprPtr>> entries;     // Dict
    bool negate = false;                                  // Test: "is not"; Binary "in": "not in"
};

}  // namespace

struct ChatTemplate::Node {
    enum class Kind { Text, Output, If, For, Set, SetBlock, Macro, Break, Continue, Block };
    Kind kind = Kind::Text;
    std::string text;                    // Text; Set / SetBlock / Macro: name
    std::string attr;                    // Set on a namespace attribute: name.attr
    ExprPtr expr;                        // Output / Set value / For iterable
    ExprPtr filter;                      // For ... if filter
    std::vector<std::string> targets;    // For variables
    std::vector<std::pair<ExprPtr, std::vector<Node>>> branches;  // If
    std::vector<Node> body, else_body;   // For / If else / SetBlock / Macro / Block
    std::vector<std::pair<std::string, ExprPtr>> params;           // Macro
};

namespace {

using Node = ChatTemplate::Node;

// ---------------------------------------------------------------------------
// Lexing: text and tags, with Jinja's whitespace rules as transformers sets
// them (trim_blocks, lstrip_blocks, keep_trailing_newline off).
// ---------------------------------------------------------------------------
struct Piece {
    enum class Kind { Text, Output, Statement, Comment } kind;
    std::string text;  // text, or the tag's inside
    size_t line = 1;
};

std::vector<Piece> split_template(std::string_view raw) {
    // Jinja joins the source's lines with "\n" (CR LF and CR become LF) and
    // drops one trailing newline.
    std::string src;
    src.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\r') {
            src += '\n';
            if (i + 1 < raw.size() && raw[i + 1] == '\n') ++i;
        } else {
            src += raw[i];
        }
    }
    if (!src.empty() && src.back() == '\n') src.pop_back();

    struct RawTag {
        Piece::Kind kind;
        std::string inner;
        bool strip_before, strip_after, plus;
        size_t line;
    };
    std::vector<std::string> texts;
    std::vector<RawTag> tags;
    size_t pos = 0, line = 1;
    std::string text;
    while (pos < src.size()) {
        if (src[pos] == '{' && pos + 1 < src.size() && (src[pos + 1] == '{' || src[pos + 1] == '%' || src[pos + 1] == '#')) {
            const char open = src[pos + 1];
            const Piece::Kind kind = open == '{' ? Piece::Kind::Output : open == '%' ? Piece::Kind::Statement : Piece::Kind::Comment;
            const char close = open == '{' ? '}' : open == '%' ? '%' : '#';
            size_t p = pos + 2;
            const bool strip_before = p < src.size() && src[p] == '-';
            const bool plus = p < src.size() && src[p] == '+';
            if (strip_before || plus) ++p;
            const size_t inner_start = p;
            size_t end = std::string::npos;
            char quote = 0;
            int braces = 0;  // a dict literal's "}}" does not close the tag (Jinja balances them)
            for (; p + 1 < src.size(); ++p) {
                const char c = src[p];
                if (kind != Piece::Kind::Comment) {
                    if (quote) {
                        if (c == '\\') {
                            ++p;
                        } else if (c == quote) {
                            quote = 0;
                        }
                        continue;
                    }
                    if (c == '\'' || c == '"') {
                        quote = c;
                        continue;
                    }
                    if (c == '{') ++braces;
                    if (c == '}' && braces > 0) {
                        --braces;
                        continue;
                    }
                }
                if (c == close && src[p + 1] == '}') {
                    end = p;
                    break;
                }
            }
            if (end == std::string::npos) fail("unclosed tag at line " + std::to_string(line));
            bool strip_after = false;
            size_t inner_end = end;
            if (inner_end > inner_start && src[inner_end - 1] == '-') {
                strip_after = true;
                --inner_end;
            }
            texts.push_back(std::move(text));
            text.clear();
            tags.push_back({kind, src.substr(inner_start, inner_end - inner_start), strip_before, strip_after, plus, line});
            for (size_t k = pos; k < end + 2; ++k) line += src[k] == '\n';
            pos = end + 2;
            continue;
        }
        line += src[pos] == '\n';
        text += src[pos++];
    }
    texts.push_back(std::move(text));

    // Whitespace control: '-' strips all white space on its side; a block or
    // comment tag also drops the indentation before it on its line
    // (lstrip_blocks) and the newline right after it (trim_blocks).
    auto is_ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; };
    for (size_t i = 0; i < tags.size(); ++i) {
        const RawTag& t = tags[i];
        std::string& before = texts[i];
        std::string& after = texts[i + 1];
        const bool block = t.kind != Piece::Kind::Output;
        if (t.strip_before) {
            while (!before.empty() && is_ws(before.back())) before.pop_back();
        } else if (block && !t.plus) {
            const size_t nl = before.find_last_of('\n');
            const size_t from = nl == std::string::npos ? 0 : nl + 1;
            if (before.find_first_not_of(" \t", from) == std::string::npos) before.resize(from);
        }
        if (t.strip_after) {
            size_t k = 0;
            while (k < after.size() && is_ws(after[k])) ++k;
            after.erase(0, k);
        } else if (block && !after.empty() && after[0] == '\n') {
            after.erase(0, 1);
        }
    }
    std::vector<Piece> out;
    for (size_t i = 0; i < texts.size(); ++i) {
        if (!texts[i].empty()) out.push_back({Piece::Kind::Text, std::move(texts[i]), 0});
        if (i < tags.size() && tags[i].kind != Piece::Kind::Comment) {
            out.push_back({tags[i].kind, std::move(tags[i].inner), tags[i].line});
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Expression tokens and parser
// ---------------------------------------------------------------------------
struct Token {
    enum class Kind { Name, String, Int, Float, Op, End } kind = Kind::End;
    std::string text;
    Value value;
};

std::vector<Token> tokenize(std::string_view s, size_t line) {
    std::vector<Token> out;
    auto where = [&] { return " at line " + std::to_string(line); };
    size_t i = 0;
    while (i < s.size()) {
        const char c = s[i];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            ++i;
            continue;
        }
        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t j = i;
            while (j < s.size() && (std::isalnum(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
            out.push_back({Token::Kind::Name, std::string(s.substr(i, j - i)), {}});
            i = j;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            size_t j = i;
            bool is_float = false;
            while (j < s.size() && (std::isdigit(static_cast<unsigned char>(s[j])) || s[j] == '_')) ++j;
            if (j + 1 < s.size() && s[j] == '.' && std::isdigit(static_cast<unsigned char>(s[j + 1]))) {
                is_float = true;
                ++j;
                while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
            }
            if (j < s.size() && (s[j] == 'e' || s[j] == 'E')) {
                size_t k = j + 1;
                if (k < s.size() && (s[k] == '+' || s[k] == '-')) ++k;
                if (k < s.size() && std::isdigit(static_cast<unsigned char>(s[k]))) {
                    is_float = true;
                    j = k;
                    while (j < s.size() && std::isdigit(static_cast<unsigned char>(s[j]))) ++j;
                }
            }
            std::string num;
            for (size_t k = i; k < j; ++k) {
                if (s[k] != '_') num += s[k];
            }
            Token t;
            t.text = num;
            if (is_float) {
                t.kind = Token::Kind::Float;
                t.value = Value(std::strtod(num.c_str(), nullptr));
            } else {
                t.kind = Token::Kind::Int;
                t.value = Value(static_cast<int64_t>(std::strtoll(num.c_str(), nullptr, 10)));
            }
            out.push_back(std::move(t));
            i = j;
            continue;
        }
        if (c == '\'' || c == '"') {
            std::string str;
            size_t j = i + 1;
            for (; j < s.size() && s[j] != c; ++j) {
                if (s[j] != '\\' || j + 1 >= s.size()) {
                    str += s[j];
                    continue;
                }
                const char e = s[++j];
                auto hex = [&](size_t digits) {
                    uint32_t v = 0;
                    for (size_t k = 0; k < digits; ++k) {
                        if (j + 1 >= s.size() || !std::isxdigit(static_cast<unsigned char>(s[j + 1]))) {
                            fail("bad escape in string literal" + where());
                        }
                        const char h = s[++j];
                        v = v * 16 + static_cast<uint32_t>(std::isdigit(static_cast<unsigned char>(h)) ? h - '0' : (std::tolower(h) - 'a' + 10));
                    }
                    unicode::append_utf8(str, v);
                };
                switch (e) {
                    case 'n': str += '\n'; break;
                    case 't': str += '\t'; break;
                    case 'r': str += '\r'; break;
                    case 'b': str += '\b'; break;
                    case 'f': str += '\f'; break;
                    case 'v': str += '\v'; break;
                    case 'a': str += '\a'; break;
                    case '0': str += '\0'; break;
                    case '\\': str += '\\'; break;
                    case '\'': str += '\''; break;
                    case '"': str += '"'; break;
                    case '\n': break;  // line continuation
                    case 'x': hex(2); break;
                    case 'u': hex(4); break;
                    case 'U': hex(8); break;
                    default: str += '\\'; str += e; break;  // Python keeps unknown escapes
                }
            }
            if (j >= s.size()) fail("unterminated string literal" + where());
            out.push_back({Token::Kind::String, {}, Value(std::move(str))});
            i = j + 1;
            continue;
        }
        static const char* const kOps[] = {"**", "//", "==", "!=", "<=", ">=", "+", "-", "*", "/", "%", "~", "<", ">",
                                           "=", "(", ")", "[", "]", "{", "}", ",", ".", ":", "|"};
        bool matched = false;
        for (const char* op : kOps) {
            const size_t n = std::char_traits<char>::length(op);
            if (s.substr(i, n) == op) {
                out.push_back({Token::Kind::Op, op, {}});
                i += n;
                matched = true;
                break;
            }
        }
        if (!matched) fail(std::string("unexpected character '") + c + "'" + where());
    }
    out.push_back({Token::Kind::End, "", {}});
    return out;
}

class ExprParser {
public:
    ExprParser(std::vector<Token> tokens, size_t line) : t_(std::move(tokens)), line_(line) {}

    ExprPtr expression() { return ternary(); }
    [[nodiscard]] bool at_end() const { return t_[i_].kind == Token::Kind::End; }
    [[nodiscard]] const Token& peek(size_t ahead = 0) const { return t_[std::min(i_ + ahead, t_.size() - 1)]; }
    bool is_op(std::string_view op, size_t ahead = 0) const {
        return peek(ahead).kind == Token::Kind::Op && peek(ahead).text == op;
    }
    bool is_name(std::string_view n, size_t ahead = 0) const {
        return peek(ahead).kind == Token::Kind::Name && peek(ahead).text == n;
    }
    bool eat_op(std::string_view op) {
        if (!is_op(op)) return false;
        ++i_;
        return true;
    }
    bool eat_name(std::string_view n) {
        if (!is_name(n)) return false;
        ++i_;
        return true;
    }
    void expect_op(std::string_view op) {
        if (!eat_op(op)) error("expected '" + std::string(op) + "'");
    }
    std::string name() {
        if (peek().kind != Token::Kind::Name) error("expected a name");
        return t_[i_++].text;
    }
    [[noreturn]] void error(const std::string& what) const {
        fail(what + " near '" + peek().text + "' at line " + std::to_string(line_));
    }
    void expect_end() {
        if (!at_end()) error("unexpected text");
    }

private:
    std::vector<Token> t_;
    size_t i_ = 0;
    size_t line_;

    static ExprPtr make(Expr::Kind kind) {
        auto e = std::make_unique<Expr>();
        e->kind = kind;
        return e;
    }

    ExprPtr ternary() {
        ExprPtr cond_true = or_expr();
        if (eat_name("if")) {
            auto e = make(Expr::Kind::Ternary);
            e->b = or_expr();  // condition
            e->a = std::move(cond_true);
            if (eat_name("else")) e->c = ternary();
            return e;
        }
        return cond_true;
    }
    ExprPtr or_expr() {
        ExprPtr left = and_expr();
        while (eat_name("or")) {
            auto e = make(Expr::Kind::Or);
            e->a = std::move(left);
            e->b = and_expr();
            left = std::move(e);
        }
        return left;
    }
    ExprPtr and_expr() {
        ExprPtr left = not_expr();
        while (eat_name("and")) {
            auto e = make(Expr::Kind::And);
            e->a = std::move(left);
            e->b = not_expr();
            left = std::move(e);
        }
        return left;
    }
    ExprPtr not_expr() {
        if (eat_name("not")) {
            auto e = make(Expr::Kind::Not);
            e->a = not_expr();
            return e;
        }
        return compare();
    }
    ExprPtr compare() {
        ExprPtr left = math1();
        for (;;) {
            std::string op;
            bool negate = false;
            if (peek().kind == Token::Kind::Op &&
                (is_op("==") || is_op("!=") || is_op("<") || is_op(">") || is_op("<=") || is_op(">="))) {
                op = t_[i_++].text;
            } else if (eat_name("in")) {
                op = "in";
            } else if (is_name("not") && is_name("in", 1)) {
                i_ += 2;
                op = "in";
                negate = true;
            } else {
                return left;
            }
            auto e = make(Expr::Kind::Binary);
            e->name = op;
            e->negate = negate;
            e->a = std::move(left);
            e->b = math1();
            left = std::move(e);
        }
    }
    ExprPtr binary_chain(ExprPtr (ExprParser::*next)(), std::initializer_list<std::string_view> ops) {
        ExprPtr left = (this->*next)();
        for (;;) {
            std::string op;
            for (const std::string_view o : ops) {
                if (is_op(o)) op = std::string(o);
            }
            if (op.empty()) return left;
            ++i_;
            auto e = make(Expr::Kind::Binary);
            e->name = op;
            e->a = std::move(left);
            e->b = (this->*next)();
            left = std::move(e);
        }
    }
    ExprPtr math1() { return binary_chain(&ExprParser::concat, {"+", "-"}); }
    ExprPtr concat() { return binary_chain(&ExprParser::math2, {"~"}); }
    ExprPtr math2() { return binary_chain(&ExprParser::power, {"*", "/", "//", "%"}); }
    ExprPtr power() { return binary_chain(&ExprParser::unary_with_filters, {"**"}); }
    ExprPtr unary_with_filters() { return unary(true); }

    ExprPtr unary(bool with_filter) {
        ExprPtr node;
        if (eat_op("-")) {
            node = make(Expr::Kind::Neg);
            node->a = unary(false);
        } else if (eat_op("+")) {
            node = make(Expr::Kind::Pos);
            node->a = unary(false);
        } else {
            node = primary();
        }
        node = postfix(std::move(node));
        if (with_filter) node = filters(std::move(node));
        return node;
    }

    void call_args(Expr& e) {
        // after '('
        if (eat_op(")")) return;
        for (;;) {
            if (peek().kind == Token::Kind::Name && is_op("=", 1)) {
                std::string key = name();
                ++i_;
                e.kwargs.emplace_back(std::move(key), expression());
            } else {
                e.args.push_back(expression());
            }
            if (eat_op(",")) {
                if (eat_op(")")) return;
                continue;
            }
            expect_op(")");
            return;
        }
    }

    ExprPtr postfix(ExprPtr node) {
        for (;;) {
            if (eat_op(".")) {
                auto e = make(Expr::Kind::Attr);
                if (peek().kind == Token::Kind::Int) {  // x.0 is x[0] in Jinja
                    auto idx = make(Expr::Kind::Literal);
                    idx->literal = t_[i_++].value;
                    e->kind = Expr::Kind::Index;
                    e->b = std::move(idx);
                } else {
                    e->name = name();
                }
                e->a = std::move(node);
                node = std::move(e);
            } else if (eat_op("[")) {
                ExprPtr lo, hi, step;
                bool slice = false;
                if (!is_op(":")) lo = expression();
                if (eat_op(":")) {
                    slice = true;
                    if (!is_op(":") && !is_op("]")) hi = expression();
                    if (eat_op(":") && !is_op("]")) step = expression();
                }
                expect_op("]");
                auto e = make(slice ? Expr::Kind::Slice : Expr::Kind::Index);
                e->a = std::move(node);
                e->b = std::move(lo);
                e->c = std::move(hi);
                e->d = std::move(step);
                node = std::move(e);
            } else if (eat_op("(")) {
                auto e = make(Expr::Kind::Call);
                e->a = std::move(node);
                call_args(*e);
                node = std::move(e);
            } else {
                return node;
            }
        }
    }

    ExprPtr filters(ExprPtr node) {
        for (;;) {
            if (eat_op("|")) {
                auto e = make(Expr::Kind::Filter);
                e->name = name();
                while (eat_op(".")) e->name += "." + name();
                e->a = std::move(node);
                if (eat_op("(")) call_args(*e);
                node = postfix(std::move(e));
            } else if (eat_name("is")) {
                auto e = make(Expr::Kind::Test);
                e->negate = eat_name("not");
                e->name = name();
                e->a = std::move(node);
                if (eat_op("(")) {
                    call_args(*e);
                } else if (starts_primary()) {
                    e->args.push_back(postfix(primary()));
                }
                node = std::move(e);
            } else {
                return node;
            }
        }
    }

    [[nodiscard]] bool starts_primary() const {
        const Token& t = peek();
        if (t.kind == Token::Kind::String || t.kind == Token::Kind::Int || t.kind == Token::Kind::Float) return true;
        if (t.kind == Token::Kind::Name) {
            static const char* const kKeywords[] = {"and", "or", "not", "in", "is", "if", "else"};
            return std::none_of(std::begin(kKeywords), std::end(kKeywords), [&](const char* k) { return t.text == k; });
        }
        return t.kind == Token::Kind::Op && (t.text == "[" || t.text == "{" || t.text == "(");
    }

    ExprPtr primary() {
        const Token& t = peek();
        if (t.kind == Token::Kind::String) {
            auto e = make(Expr::Kind::Literal);
            std::string s = t_[i_++].value.as_string();
            while (peek().kind == Token::Kind::String) s += t_[i_++].value.as_string();  // adjacent literals join
            e->literal = Value(std::move(s));
            return e;
        }
        if (t.kind == Token::Kind::Int || t.kind == Token::Kind::Float) {
            auto e = make(Expr::Kind::Literal);
            e->literal = t_[i_++].value;
            return e;
        }
        if (t.kind == Token::Kind::Name) {
            const std::string n = t_[i_++].text;
            auto e = make(Expr::Kind::Literal);
            if (n == "true" || n == "True") e->literal = Value(true);
            else if (n == "false" || n == "False") e->literal = Value(false);
            else if (n == "none" || n == "None") e->literal = Value::none();
            else {
                e->kind = Expr::Kind::Name;
                e->name = n;
            }
            return e;
        }
        if (eat_op("(")) {
            if (eat_op(")")) return make(Expr::Kind::Tuple);
            ExprPtr first = expression();
            if (!is_op(",")) {
                expect_op(")");
                return first;
            }
            auto e = make(Expr::Kind::Tuple);
            e->args.push_back(std::move(first));
            while (eat_op(",")) {
                if (is_op(")")) break;
                e->args.push_back(expression());
            }
            expect_op(")");
            return e;
        }
        if (eat_op("[")) {
            auto e = make(Expr::Kind::List);
            while (!eat_op("]")) {
                e->args.push_back(expression());
                if (!eat_op(",")) {
                    expect_op("]");
                    break;
                }
            }
            return e;
        }
        if (eat_op("{")) {
            auto e = make(Expr::Kind::Dict);
            while (!eat_op("}")) {
                ExprPtr key = expression();
                expect_op(":");
                e->entries.emplace_back(std::move(key), expression());
                if (!eat_op(",")) {
                    expect_op("}");
                    break;
                }
            }
            return e;
        }
        error("unexpected token");
    }
};

// ---------------------------------------------------------------------------
// Statement parser
// ---------------------------------------------------------------------------
class TemplateParser {
public:
    explicit TemplateParser(std::vector<Piece> pieces) : p_(std::move(pieces)) {}

    std::vector<Node> parse() {
        std::vector<Node> body;
        std::string end = block(body, {});
        if (!end.empty()) fail("unexpected '" + end + "'");
        return body;
    }

private:
    std::vector<Piece> p_;
    size_t i_ = 0;

    // Parses nodes until a statement whose keyword is in `ends`; returns that
    // keyword (its parser positioned after it) or "" at the end of input.
    std::string block(std::vector<Node>& out, std::initializer_list<std::string_view> ends,
                      std::unique_ptr<ExprParser>* end_parser = nullptr) {
        while (i_ < p_.size()) {
            Piece& piece = p_[i_++];
            if (piece.kind == Piece::Kind::Text) {
                Node n;
                n.kind = Node::Kind::Text;
                n.text = std::move(piece.text);
                out.push_back(std::move(n));
                continue;
            }
            auto parser = std::make_unique<ExprParser>(tokenize(piece.text, piece.line), piece.line);
            if (piece.kind == Piece::Kind::Output) {
                Node n;
                n.kind = Node::Kind::Output;
                n.expr = parser->expression();
                parser->expect_end();
                out.push_back(std::move(n));
                continue;
            }
            const std::string keyword = parser->name();
            if (std::find(ends.begin(), ends.end(), keyword) != ends.end()) {
                if (end_parser) *end_parser = std::move(parser);
                else parser->expect_end();
                return keyword;
            }
            out.push_back(statement(keyword, *parser, piece.line));
        }
        if (ends.size() > 0) fail("missing '{% " + std::string(*(ends.end() - 1)) + " %}'");
        return "";
    }

    Node statement(const std::string& keyword, ExprParser& ep, size_t line) {
        Node n;
        if (keyword == "if") {
            n.kind = Node::Kind::If;
            ExprPtr cond = ep.expression();
            ep.expect_end();
            for (;;) {
                std::vector<Node> body;
                std::unique_ptr<ExprParser> end;
                const std::string k = block(body, {"elif", "else", "endif"}, &end);
                n.branches.emplace_back(std::move(cond), std::move(body));
                if (k == "elif") {
                    cond = end->expression();
                    end->expect_end();
                    continue;
                }
                end->expect_end();
                if (k == "else") {
                    const std::string k2 = block(n.else_body, {"endif"});
                    (void)k2;
                }
                break;
            }
            return n;
        }
        if (keyword == "for") {
            n.kind = Node::Kind::For;
            n.targets.push_back(ep.name());
            while (ep.eat_op(",")) n.targets.push_back(ep.name());
            if (!ep.eat_name("in")) ep.error("expected 'in'");
            // "for x in items if cond": the filter parses as a conditional
            // expression without "else", which is what it is in Jinja.
            ExprPtr iter = ep.expression();
            if (iter->kind == Expr::Kind::Ternary && !iter->c) {
                n.filter = std::move(iter->b);
                n.expr = std::move(iter->a);
            } else {
                n.expr = std::move(iter);
            }
            if (ep.eat_name("recursive")) ep.error("recursive loops are not supported");
            ep.expect_end();
            const std::string k = block(n.body, {"else", "endfor"});
            if (k == "else") block(n.else_body, {"endfor"});
            return n;
        }
        if (keyword == "set") {
            n.text = ep.name();
            if (ep.eat_op(".")) n.attr = ep.name();
            if (ep.eat_op("=")) {
                n.kind = Node::Kind::Set;
                n.expr = ep.expression();
                ep.expect_end();
                return n;
            }
            ep.expect_end();
            n.kind = Node::Kind::SetBlock;
            block(n.body, {"endset"});
            return n;
        }
        if (keyword == "macro") {
            n.kind = Node::Kind::Macro;
            n.text = ep.name();
            ep.expect_op("(");
            while (!ep.eat_op(")")) {
                std::string param = ep.name();
                ExprPtr def;
                if (ep.eat_op("=")) def = ep.expression();
                n.params.emplace_back(std::move(param), std::move(def));
                if (!ep.eat_op(",")) {
                    ep.expect_op(")");
                    break;
                }
            }
            ep.expect_end();
            block(n.body, {"endmacro"});
            return n;
        }
        if (keyword == "break" || keyword == "continue") {
            ep.expect_end();
            n.kind = keyword == "break" ? Node::Kind::Break : Node::Kind::Continue;
            return n;
        }
        if (keyword == "generation") {  // transformers' assistant-mask marker: renders its body
            ep.expect_end();
            n.kind = Node::Kind::Block;
            block(n.body, {"endgeneration"});
            return n;
        }
        fail("unsupported statement '{% " + keyword + " %}' at line " + std::to_string(line));
    }

};

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------
// Reverse of a string, by code point (the |reverse filter on strings).
Value eval_slice_reverse(const Value& s) {
    std::vector<std::string> chars;
    const std::string& t = s.as_string();
    for (size_t i = 0; i < t.size();) {
        const size_t at = i;
        (void)unicode::decode_utf8(t, i);
        chars.push_back(t.substr(at, i - at));
    }
    std::string out;
    for (auto it = chars.rbegin(); it != chars.rend(); ++it) out += *it;
    return Value(std::move(out));
}

enum class Flow { Normal, Break, Continue };

class Interpreter {
public:
    using Scope = std::unordered_map<std::string, Value>;

    explicit Interpreter(Scope& globals) { scopes_.push_back(&globals); }

    std::string out;

    Flow exec(const std::vector<Node>& body) {
        for (const Node& n : body) {
            const Flow f = exec(n);
            if (f != Flow::Normal) return f;
        }
        return Flow::Normal;
    }

private:
    std::vector<Scope*> scopes_;
    std::vector<std::unique_ptr<Scope>> owned_;
    int depth_ = 0;

    struct ScopeGuard {
        Interpreter& in;
        explicit ScopeGuard(Interpreter& i) : in(i) {
            in.owned_.push_back(std::make_unique<Scope>());
            in.scopes_.push_back(in.owned_.back().get());
        }
        ~ScopeGuard() {
            in.scopes_.pop_back();
            in.owned_.pop_back();
        }
    };

    Value lookup(const std::string& name) {
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            const auto found = (*it)->find(name);
            if (found != (*it)->end()) return found->second;
        }
        if (const Value g = builtin(name); !g.is_undefined()) return g;
        return Value::undefined(name);
    }

    static Value builtin(const std::string& name);

    Flow exec(const Node& n) {
        switch (n.kind) {
            case Node::Kind::Text: out += n.text; return Flow::Normal;
            case Node::Kind::Output: out += eval(*n.expr).str(); return Flow::Normal;
            case Node::Kind::If:
                for (const auto& [cond, body] : n.branches) {
                    if (eval(*cond).truthy()) return exec(body);
                }
                return exec(n.else_body);
            case Node::Kind::For: return exec_for(n);
            case Node::Kind::Set: {
                Value v = eval(*n.expr);
                assign(n, std::move(v));
                return Flow::Normal;
            }
            case Node::Kind::SetBlock: {
                std::string saved = std::move(out);
                out.clear();
                exec(n.body);
                Value v(std::move(out));
                out = std::move(saved);
                assign(n, std::move(v));
                return Flow::Normal;
            }
            case Node::Kind::Macro: define_macro(n); return Flow::Normal;
            case Node::Kind::Break: return Flow::Break;
            case Node::Kind::Continue: return Flow::Continue;
            case Node::Kind::Block: return exec(n.body);
        }
        return Flow::Normal;
    }

    void assign(const Node& n, Value v) {
        if (n.attr.empty()) {
            (*scopes_.back())[n.text] = std::move(v);
            return;
        }
        const Value target = lookup(n.text);
        if (!target.is_object() || !target.fields().is_namespace) {
            fail("cannot assign attribute '" + n.attr + "' on '" + n.text + "' (not a namespace)");
        }
        target.fields().set(n.attr, std::move(v));
    }

    static Array iterate(const Value& v) {
        switch (v.type()) {
            case Value::Type::Undefined: return {};
            case Value::Type::Array: return v.items();
            case Value::Type::Object: {
                Array keys;
                for (const auto& e : v.fields().entries) keys.emplace_back(e.first);
                return keys;
            }
            case Value::Type::String: {
                Array chars;
                const std::string& s = v.as_string();
                for (size_t i = 0; i < s.size();) {
                    const size_t at = i;
                    (void)unicode::decode_utf8(s, i);
                    chars.emplace_back(s.substr(at, i - at));
                }
                return chars;
            }
            default: fail(std::string("'") + jinja::type_name(v.type()) + "' object is not iterable");
        }
    }

    void bind_targets(const std::vector<std::string>& targets, const Value& item, Scope& scope) {
        if (targets.size() == 1) {
            scope[targets[0]] = item;
            return;
        }
        const Array parts = iterate(item);
        if (parts.size() != targets.size()) fail("cannot unpack " + std::to_string(parts.size()) + " values into " +
                                                 std::to_string(targets.size()));
        for (size_t i = 0; i < targets.size(); ++i) scope[targets[i]] = parts[i];
    }

    Flow exec_for(const Node& n) {
        Array all = iterate(eval(*n.expr));
        Array items;
        if (n.filter) {
            ScopeGuard guard(*this);
            for (const Value& v : all) {
                bind_targets(n.targets, v, *scopes_.back());
                if (eval(*n.filter).truthy()) items.push_back(v);
            }
        } else {
            items = std::move(all);
        }
        if (items.empty()) return exec(n.else_body);
        const auto length = static_cast<int64_t>(items.size());
        for (int64_t i = 0; i < length; ++i) {
            ScopeGuard guard(*this);  // a fresh scope per iteration, as Jinja
            Scope& scope = *scopes_.back();
            bind_targets(n.targets, items[static_cast<size_t>(i)], scope);
            Value loop = Value::object();
            Object& l = loop.fields();
            l.set("index0", Value(i));
            l.set("index", Value(i + 1));
            l.set("revindex", Value(length - i));
            l.set("revindex0", Value(length - i - 1));
            l.set("first", Value(i == 0));
            l.set("last", Value(i == length - 1));
            l.set("length", Value(length));
            l.set("previtem", i > 0 ? items[static_cast<size_t>(i - 1)] : Value::undefined("loop.previtem"));
            l.set("nextitem", i + 1 < length ? items[static_cast<size_t>(i + 1)] : Value::undefined("loop.nextitem"));
            l.set("cycle", Value::callable("loop.cycle", [i](Array& args, Kwargs&) {
                if (args.empty()) fail("loop.cycle needs arguments");
                return args[static_cast<size_t>(i) % args.size()];
            }));
            scope["loop"] = loop;
            const Flow f = exec(n.body);
            if (f == Flow::Break) break;
        }
        return Flow::Normal;
    }

    void define_macro(const Node& n) {
        Scope* root = scopes_.front();
        const Node* node = &n;
        (*scopes_.back())[n.text] = Value::callable(n.text, [this, node, root](Array& args, Kwargs& kwargs) {
            if (args.size() > node->params.size()) fail("macro '" + node->text + "' takes " +
                                                        std::to_string(node->params.size()) + " arguments");
            Interpreter inner(*root);
            inner.depth_ = depth_ + 1;
            if (inner.depth_ > 64) fail("macro recursion too deep");
            ScopeGuard guard(inner);
            Scope& local = *inner.scopes_.back();
            for (size_t p = 0; p < node->params.size(); ++p) {
                const std::string& name = node->params[p].first;
                if (p < args.size()) {
                    local[name] = args[p];
                    continue;
                }
                const auto kw = std::find_if(kwargs.begin(), kwargs.end(), [&](const auto& k) { return k.first == name; });
                if (kw != kwargs.end()) local[name] = kw->second;
                else if (node->params[p].second) local[name] = inner.eval(*node->params[p].second);
                else local[name] = Value::undefined(name);
            }
            inner.exec(node->body);
            return Value(std::move(inner.out));
        });
    }

    // ---- expressions ------------------------------------------------------
    Value eval(const Expr& e);
    Value binary(const std::string& op, const Value& a, const Value& b);
    Value attribute(const Value& target, const std::string& name);
    Value index(const Value& target, const Value& key);
    Value call(const Value& fn, Array& args, Kwargs& kwargs);
    Value filter(const std::string& name, Value input, Array& args, Kwargs& kwargs);
    bool test(const std::string& name, const Value& v, Array& args);
    Value method(const Value& self, const std::string& name);
};

void require_defined(const Value& v) {
    if (v.is_undefined()) fail("'" + (v.as_string().empty() ? std::string("value") : v.as_string()) + "' is undefined");
}

std::string strip_chars(const std::string& s, const Value* chars, bool left, bool right) {
    auto in_set = [&](uint32_t cp) {
        if (chars == nullptr || chars->is_none()) return unicode::is_whitespace(cp);
        const std::string& set = chars->as_string();
        for (size_t i = 0; i < set.size();) {
            if (unicode::decode_utf8(set, i) == cp) return true;
        }
        return false;
    };
    // Code-point boundaries, so multi-byte characters strip whole.
    std::vector<size_t> bounds;
    for (size_t i = 0; i < s.size();) {
        bounds.push_back(i);
        (void)unicode::decode_utf8(s, i);
    }
    bounds.push_back(s.size());
    size_t first = 0, last = bounds.size() - 1;
    auto cp_at = [&](size_t k) {
        size_t i = bounds[k];
        return unicode::decode_utf8(s, i);
    };
    if (left) {
        while (first < last && in_set(cp_at(first))) ++first;
    }
    if (right) {
        while (last > first && in_set(cp_at(last - 1))) --last;
    }
    return s.substr(bounds[first], bounds[last] - bounds[first]);
}

Array split_string(const std::string& s, const Value* sep, int64_t maxsplit, bool from_right) {
    Array out;
    if (sep == nullptr || sep->is_none()) {  // runs of white space, ends ignored
        std::vector<std::string> words;
        std::string word;
        for (size_t i = 0; i < s.size();) {
            const size_t at = i;
            const uint32_t cp = unicode::decode_utf8(s, i);
            if (unicode::is_whitespace(cp)) {
                if (!word.empty()) words.push_back(std::move(word));
                word.clear();
            } else {
                word.append(s, at, i - at);
            }
        }
        if (!word.empty()) words.push_back(std::move(word));
        if (maxsplit >= 0 && static_cast<int64_t>(words.size()) > maxsplit + 1) {
            // Rare in templates; keep Python's result for the left-to-right case.
            std::vector<std::string> joined(words.begin(), words.begin() + maxsplit);
            std::string rest;
            size_t found = 0, pos = 0;
            for (size_t w = 0; w < static_cast<size_t>(maxsplit); ++w) {
                pos = s.find(words[w], pos) + words[w].size();
                found = pos;
            }
            rest = strip_chars(s.substr(found), nullptr, true, false);
            joined.push_back(rest);
            words = std::move(joined);
        }
        for (auto& w : words) out.emplace_back(std::move(w));
        return out;
    }
    const std::string& d = sep->as_string();
    if (d.empty()) fail("empty separator");
    std::vector<std::string> parts;
    if (!from_right) {
        size_t pos = 0;
        int64_t n = 0;
        for (;;) {
            const size_t at = s.find(d, pos);
            if (at == std::string::npos || (maxsplit >= 0 && n >= maxsplit)) break;
            parts.push_back(s.substr(pos, at - pos));
            pos = at + d.size();
            ++n;
        }
        parts.push_back(s.substr(pos));
    } else {
        size_t end = s.size();
        int64_t n = 0;
        for (;;) {
            if (end < d.size()) break;
            const size_t at = s.rfind(d, end - d.size());
            if (at == std::string::npos || (maxsplit >= 0 && n >= maxsplit)) break;
            parts.insert(parts.begin(), s.substr(at + d.size(), end - at - d.size()));
            end = at;
            ++n;
        }
        parts.insert(parts.begin(), s.substr(0, end));
    }
    for (auto& p : parts) out.emplace_back(std::move(p));
    return out;
}

const Value* kwarg(Kwargs& kwargs, std::string_view name) {
    for (auto& [k, v] : kwargs) {
        if (k == name) return &v;
    }
    return nullptr;
}
const Value* arg(Array& args, Kwargs& kwargs, size_t index, std::string_view name) {
    if (index < args.size()) return &args[index];
    return kwarg(kwargs, name);
}

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string to_upper(std::string s) {
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

int64_t py_len(const Value& v) {
    switch (v.type()) {
        case Value::Type::Undefined: return 0;
        case Value::Type::String: {
            int64_t n = 0;
            const std::string& s = v.as_string();
            for (size_t i = 0; i < s.size(); ++n) (void)unicode::decode_utf8(s, i);
            return n;
        }
        case Value::Type::Array: return static_cast<int64_t>(v.items().size());
        case Value::Type::Object: return static_cast<int64_t>(v.fields().entries.size());
        default: fail(std::string("object of type '") + jinja::type_name(v.type()) + "' has no len()");
    }
}

// Python's ordering for <, >: numbers, strings, lists.
int compare_values(const Value& a, const Value& b) {
    if (a.is_number() && b.is_number()) {
        const double x = a.as_float(), y = b.as_float();
        if (a.type() != Value::Type::Float && b.type() != Value::Type::Float) {
            return a.as_int() < b.as_int() ? -1 : a.as_int() > b.as_int() ? 1 : 0;
        }
        return x < y ? -1 : x > y ? 1 : 0;
    }
    if (a.is_string() && b.is_string()) return a.as_string().compare(b.as_string()) < 0 ? -1 : a.as_string() == b.as_string() ? 0 : 1;
    if (a.is_array() && b.is_array()) {
        const size_t n = std::min(a.items().size(), b.items().size());
        for (size_t i = 0; i < n; ++i) {
            if (!(a.items()[i] == b.items()[i])) return compare_values(a.items()[i], b.items()[i]);
        }
        return a.items().size() < b.items().size() ? -1 : a.items().size() > b.items().size() ? 1 : 0;
    }
    fail(std::string("'<' not supported between '") + jinja::type_name(a.type()) + "' and '" + jinja::type_name(b.type()) + "'");
}

Value Interpreter::builtin(const std::string& name) {
    if (name == "raise_exception") {
        return Value::callable(name, [](Array& args, Kwargs&) -> Value {
            fail(args.empty() ? std::string("raise_exception") : args[0].str());
        });
    }
    if (name == "namespace") {
        return Value::callable(name, [](Array& args, Kwargs& kwargs) {
            Value ns = Value::object();
            ns.fields().is_namespace = true;
            for (const Value& a : args) {
                if (a.is_object()) {
                    for (const auto& [k, v] : a.fields().entries) ns.fields().set(k, v);
                }
            }
            for (auto& [k, v] : kwargs) ns.fields().set(k, v);
            return ns;
        });
    }
    if (name == "dict") {
        return Value::callable(name, [](Array& args, Kwargs& kwargs) {
            Value d = Value::object();
            for (const Value& a : args) {
                if (a.is_object()) {
                    for (const auto& [k, v] : a.fields().entries) d.fields().set(k, v);
                }
            }
            for (auto& [k, v] : kwargs) d.fields().set(k, v);
            return d;
        });
    }
    if (name == "range") {
        return Value::callable(name, [](Array& args, Kwargs&) {
            int64_t start = 0, stop = 0, step = 1;
            if (args.size() == 1) stop = args[0].as_int();
            else if (args.size() >= 2) {
                start = args[0].as_int();
                stop = args[1].as_int();
                if (args.size() >= 3) step = args[2].as_int();
            }
            if (step == 0) fail("range() step must not be zero");
            Array out;
            for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) {
                out.emplace_back(i);
                if (out.size() > 1'000'000) fail("range too large");
            }
            return Value::array(std::move(out));
        });
    }
    if (name == "strftime_now") {  // transformers: datetime.now().strftime(format), local time
        return Value::callable(name, [](Array& args, Kwargs&) {
            const std::time_t now = std::time(nullptr);
            std::tm tm{};
            localtime_r(&now, &tm);
            char buf[256];
            const size_t n = std::strftime(buf, sizeof buf, args.empty() ? "%Y-%m-%d" : args[0].str().c_str(), &tm);
            return Value(std::string(buf, n));
        });
    }
    return Value::undefined(name);
}

Value Interpreter::eval(const Expr& e) {
    switch (e.kind) {
        case Expr::Kind::Literal: return e.literal;
        case Expr::Kind::Name: return lookup(e.name);
        case Expr::Kind::List:
        case Expr::Kind::Tuple: {
            Array items;
            for (const auto& a : e.args) items.push_back(eval(*a));
            return Value::array(std::move(items));
        }
        case Expr::Kind::Dict: {
            Value d = Value::object();
            for (const auto& [k, v] : e.entries) {
                const Value key = eval(*k);
                if (!key.is_string()) fail("dict keys must be strings");
                d.fields().set(key.as_string(), eval(*v));
            }
            return d;
        }
        case Expr::Kind::Attr: return attribute(eval(*e.a), e.name);
        case Expr::Kind::Index: return index(eval(*e.a), eval(*e.b));
        case Expr::Kind::Slice: {
            const Value target = eval(*e.a);
            require_defined(target);
            const Value lo = e.b ? eval(*e.b) : Value::none();
            const Value hi = e.c ? eval(*e.c) : Value::none();
            const Value st = e.d ? eval(*e.d) : Value::none();
            Array src = target.is_string() ? iterate(target) : (target.is_array() ? target.items() : Array{});
            if (!target.is_string() && !target.is_array()) fail("object is not subscriptable");
            const auto n = static_cast<int64_t>(src.size());
            const int64_t step = st.is_none() ? 1 : st.as_int();
            if (step == 0) fail("slice step cannot be zero");
            auto norm = [&](const Value& v, int64_t def, bool upper) {
                if (v.is_none()) return def;
                int64_t x = v.as_int();
                if (x < 0) x += n;
                if (step > 0) return std::clamp<int64_t>(x, 0, n);
                return std::clamp<int64_t>(x, -1, n - 1) + (upper ? 0 : 0);
            };
            const int64_t start = norm(lo, step > 0 ? 0 : n - 1, false);
            const int64_t stop = norm(hi, step > 0 ? n : -1, true);
            Array out;
            for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) out.push_back(src[static_cast<size_t>(i)]);
            if (target.is_string()) {
                std::string s;
                for (const Value& c : out) s += c.as_string();
                return Value(std::move(s));
            }
            return Value::array(std::move(out));
        }
        case Expr::Kind::Call: {
            Value fn;
            Array args;
            Kwargs kwargs;
            for (const auto& a : e.args) args.push_back(eval(*a));
            for (const auto& [k, v] : e.kwargs) kwargs.emplace_back(k, eval(*v));
            fn = eval(*e.a);
            return call(fn, args, kwargs);
        }
        case Expr::Kind::Filter: {
            Value input = eval(*e.a);
            Array args;
            Kwargs kwargs;
            for (const auto& a : e.args) args.push_back(eval(*a));
            for (const auto& [k, v] : e.kwargs) kwargs.emplace_back(k, eval(*v));
            return filter(e.name, std::move(input), args, kwargs);
        }
        case Expr::Kind::Test: {
            const Value v = eval(*e.a);
            Array args;
            for (const auto& a : e.args) args.push_back(eval(*a));
            return Value(test(e.name, v, args) != e.negate);
        }
        case Expr::Kind::Not: return Value(!eval(*e.a).truthy());
        case Expr::Kind::Neg: {
            const Value v = eval(*e.a);
            if (v.type() == Value::Type::Float) return Value(-v.as_float());
            if (v.is_number()) return Value(-v.as_int());
            fail("bad operand for unary -");
        }
        case Expr::Kind::Pos: return eval(*e.a);
        case Expr::Kind::And: {
            Value a = eval(*e.a);
            return a.truthy() ? eval(*e.b) : a;
        }
        case Expr::Kind::Or: {
            Value a = eval(*e.a);
            return a.truthy() ? a : eval(*e.b);
        }
        case Expr::Kind::Ternary:
            if (eval(*e.b).truthy()) return eval(*e.a);
            return e.c ? eval(*e.c) : Value::undefined();
        case Expr::Kind::Binary: {
            const Value a = eval(*e.a);
            const Value b = eval(*e.b);
            const Value r = binary(e.name, a, b);
            return e.negate ? Value(!r.truthy()) : r;
        }
    }
    return {};
}

Value Interpreter::binary(const std::string& op, const Value& a, const Value& b) {
    using T = Value::Type;
    if (op == "==") return Value(a == b);
    if (op == "!=") return Value(!(a == b));
    if (op == "<") return Value(compare_values(a, b) < 0);
    if (op == ">") return Value(compare_values(a, b) > 0);
    if (op == "<=") return Value(compare_values(a, b) <= 0);
    if (op == ">=") return Value(compare_values(a, b) >= 0);
    if (op == "in") {
        if (b.is_string()) {
            if (!a.is_string()) fail("'in <string>' requires string as left operand");
            return Value(b.as_string().find(a.as_string()) != std::string::npos);
        }
        if (b.is_array()) return Value(std::any_of(b.items().begin(), b.items().end(), [&](const Value& x) { return x == a; }));
        if (b.is_object()) return Value(a.is_string() && b.fields().find(a.as_string()) != nullptr);
        if (b.is_undefined()) return Value(false);
        fail(std::string("argument of type '") + jinja::type_name(b.type()) + "' is not iterable");
    }
    if (op == "~") return Value(a.str() + b.str());
    require_defined(a);
    require_defined(b);
    const bool floats = a.type() == T::Float || b.type() == T::Float;
    if (op == "+") {
        if (a.is_number() && b.is_number()) return floats ? Value(a.as_float() + b.as_float()) : Value(a.as_int() + b.as_int());
        if (a.is_string() && b.is_string()) return Value(a.as_string() + b.as_string());
        if (a.is_array() && b.is_array()) {
            Array out = a.items();
            out.insert(out.end(), b.items().begin(), b.items().end());
            return Value::array(std::move(out));
        }
    } else if (op == "-") {
        if (a.is_number() && b.is_number()) return floats ? Value(a.as_float() - b.as_float()) : Value(a.as_int() - b.as_int());
    } else if (op == "*") {
        if (a.is_number() && b.is_number()) return floats ? Value(a.as_float() * b.as_float()) : Value(a.as_int() * b.as_int());
        const Value* seq = a.is_number() ? &b : &a;
        const Value* count = a.is_number() ? &a : &b;
        if (count->is_number() && count->type() != T::Float) {
            const int64_t n = std::max<int64_t>(0, count->as_int());
            if (seq->is_string()) {
                std::string out;
                for (int64_t i = 0; i < n; ++i) out += seq->as_string();
                return Value(std::move(out));
            }
            if (seq->is_array()) {
                Array out;
                for (int64_t i = 0; i < n; ++i) out.insert(out.end(), seq->items().begin(), seq->items().end());
                return Value::array(std::move(out));
            }
        }
    } else if (op == "/") {
        if (a.is_number() && b.is_number()) {
            if (b.as_float() == 0.0) fail("division by zero");
            return Value(a.as_float() / b.as_float());
        }
    } else if (op == "//") {
        if (a.is_number() && b.is_number()) {
            if (b.as_float() == 0.0) fail("integer division or modulo by zero");
            if (floats) return Value(std::floor(a.as_float() / b.as_float()));
            const int64_t q = a.as_int() / b.as_int();
            return Value((a.as_int() % b.as_int() != 0 && ((a.as_int() < 0) != (b.as_int() < 0))) ? q - 1 : q);
        }
    } else if (op == "%") {
        if (a.is_number() && b.is_number()) {
            if (b.as_float() == 0.0) fail("integer division or modulo by zero");
            if (floats) {
                const double r = std::fmod(a.as_float(), b.as_float());
                return Value(r != 0 && ((r < 0) != (b.as_float() < 0)) ? r + b.as_float() : r);
            }
            const int64_t r = a.as_int() % b.as_int();
            return Value(r != 0 && ((r < 0) != (b.as_int() < 0)) ? r + b.as_int() : r);
        }
        if (a.is_string()) {  // printf-style formatting: %s %d %r %%
            const Array args = b.is_array() ? b.items() : Array{b};
            std::string out;
            size_t next = 0;
            const std::string& f = a.as_string();
            for (size_t i = 0; i < f.size(); ++i) {
                if (f[i] != '%' || i + 1 >= f.size()) {
                    out += f[i];
                    continue;
                }
                const char spec = f[++i];
                if (spec == '%') {
                    out += '%';
                    continue;
                }
                if (next >= args.size()) fail("not enough arguments for format string");
                const Value& v = args[next++];
                if (spec == 's') out += v.str();
                else if (spec == 'r') out += v.repr();
                else if (spec == 'd' || spec == 'i') out += std::to_string(v.as_int());
                else fail(std::string("unsupported format character '") + spec + "'");
            }
            return Value(std::move(out));
        }
    } else if (op == "**") {
        if (a.is_number() && b.is_number()) {
            if (!floats && b.as_int() >= 0) {
                int64_t r = 1;
                for (int64_t i = 0; i < b.as_int(); ++i) r *= a.as_int();
                return Value(r);
            }
            return Value(std::pow(a.as_float(), b.as_float()));
        }
    }
    fail("unsupported operand type(s) for " + op + ": '" + jinja::type_name(a.type()) + "' and '" +
         jinja::type_name(b.type()) + "'");
}

Value Interpreter::attribute(const Value& target, const std::string& name) {
    require_defined(target);
    if (target.is_object()) {
        const Object& o = target.fields();
        if (!o.is_namespace) {
            static const char* const kDictMethods[] = {"items", "keys", "values", "get", "update", "pop", "setdefault",
                                                        "clear", "copy", "popitem"};
            if (std::any_of(std::begin(kDictMethods), std::end(kDictMethods), [&](const char* m) { return name == m; })) {
                return method(target, name);
            }
        }
        if (const Value* v = o.find(name)) return *v;
        return Value::undefined(name);
    }
    if (target.is_string() || target.is_array()) {
        Value m = method(target, name);
        if (!m.is_undefined()) return m;
    }
    return Value::undefined(name);
}

Value Interpreter::index(const Value& target, const Value& key) {
    require_defined(target);
    if (target.is_object()) {
        if (key.is_string()) {
            if (const Value* v = target.fields().find(key.as_string())) return *v;
            return Value::undefined(key.as_string());
        }
        return Value::undefined();
    }
    if (target.is_array() || target.is_string()) {
        if (!key.is_number()) {
            if (key.is_string()) return attribute(target, key.as_string());
            fail("indices must be integers");
        }
        const Array src = target.is_array() ? target.items() : iterate(target);
        int64_t i = key.as_int();
        if (i < 0) i += static_cast<int64_t>(src.size());
        if (i < 0 || i >= static_cast<int64_t>(src.size())) return Value::undefined("index");
        return src[static_cast<size_t>(i)];
    }
    return Value::undefined();
}

Value Interpreter::call(const Value& fn, Array& args, Kwargs& kwargs) {
    if (fn.type() != Value::Type::Callable) {
        if (fn.is_undefined()) fail("'" + (fn.as_string().empty() ? std::string("function") : fn.as_string()) + "' is undefined");
        fail(std::string("'") + jinja::type_name(fn.type()) + "' object is not callable");
    }
    return fn.function().fn(args, kwargs);
}

Value Interpreter::method(const Value& self, const std::string& name) {
    auto bind = [&](auto body) {
        return Value::callable(name, [self, body](Array& args, Kwargs& kwargs) { return body(self, args, kwargs); });
    };
    if (self.is_string()) {
        if (name == "startswith" || name == "endswith") {
            const bool starts = name == "startswith";
            return bind([starts](const Value& s, Array& args, Kwargs&) {
                if (args.empty()) fail("startswith/endswith need an argument");
                const Array options = args[0].is_array() ? args[0].items() : Array{args[0]};
                for (const Value& o : options) {
                    const std::string& p = o.as_string();
                    if (starts ? s.as_string().starts_with(p) : s.as_string().ends_with(p)) return Value(true);
                }
                return Value(false);
            });
        }
        if (name == "strip" || name == "lstrip" || name == "rstrip") {
            const bool left = name != "rstrip", right = name != "lstrip";
            return bind([left, right](const Value& s, Array& args, Kwargs&) {
                return Value(strip_chars(s.as_string(), args.empty() ? nullptr : &args[0], left, right));
            });
        }
        if (name == "split" || name == "rsplit") {
            const bool right = name == "rsplit";
            return bind([right](const Value& s, Array& args, Kwargs& kwargs) {
                const Value* sep = arg(args, kwargs, 0, "sep");
                const Value* max = arg(args, kwargs, 1, "maxsplit");
                return Value::array(split_string(s.as_string(), sep, max ? max->as_int() : -1, right));
            });
        }
        if (name == "splitlines") {
            return bind([](const Value& s, Array&, Kwargs&) {
                Array out;
                std::string line;
                const std::string& t = s.as_string();
                for (size_t i = 0; i < t.size(); ++i) {
                    if (t[i] == '\n' || t[i] == '\r') {
                        out.emplace_back(line);
                        line.clear();
                        if (t[i] == '\r' && i + 1 < t.size() && t[i + 1] == '\n') ++i;
                    } else {
                        line += t[i];
                    }
                }
                if (!line.empty()) out.emplace_back(line);
                return Value::array(std::move(out));
            });
        }
        if (name == "upper") return bind([](const Value& s, Array&, Kwargs&) { return Value(to_upper(s.as_string())); });
        if (name == "lower") return bind([](const Value& s, Array&, Kwargs&) { return Value(to_lower(s.as_string())); });
        if (name == "capitalize" || name == "title") {
            const bool title = name == "title";
            return bind([title](const Value& s, Array&, Kwargs&) {
                std::string out = to_lower(s.as_string());
                bool start = true;
                for (char& c : out) {
                    if (start && std::isalpha(static_cast<unsigned char>(c))) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                    start = title ? !std::isalpha(static_cast<unsigned char>(c)) : false;
                }
                return Value(std::move(out));
            });
        }
        if (name == "replace") {
            return bind([](const Value& s, Array& args, Kwargs&) {
                if (args.size() < 2) fail("replace needs two arguments");
                const std::string& from = args[0].as_string();
                const std::string& to = args[1].as_string();
                int64_t count = args.size() > 2 ? args[2].as_int() : -1;
                std::string out;
                const std::string& t = s.as_string();
                size_t pos = 0;
                if (from.empty()) return s;
                while (count != 0) {
                    const size_t at = t.find(from, pos);
                    if (at == std::string::npos) break;
                    out.append(t, pos, at - pos).append(to);
                    pos = at + from.size();
                    if (count > 0) --count;
                }
                out.append(t, pos);
                return Value(std::move(out));
            });
        }
        if (name == "find" || name == "rfind" || name == "count") {
            const std::string which = name;
            return bind([which](const Value& s, Array& args, Kwargs&) {
                if (args.empty()) fail(which + " needs an argument");
                const std::string& t = s.as_string();
                const std::string& sub = args[0].as_string();
                if (which == "count") {
                    if (sub.empty()) return Value(py_len(s) + 1);
                    int64_t n = 0;
                    for (size_t p = t.find(sub); p != std::string::npos; p = t.find(sub, p + sub.size())) ++n;
                    return Value(n);
                }
                const size_t at = which == "find" ? t.find(sub) : t.rfind(sub);
                // Python counts code points; the byte offset is converted.
                if (at == std::string::npos) return Value(static_cast<int64_t>(-1));
                return Value(py_len(Value(t.substr(0, at))));
            });
        }
        if (name == "join") {
            return bind([](const Value& s, Array& args, Kwargs&) {
                if (args.empty()) fail("join needs an argument");
                std::string out;
                const Array items = iterate(args[0]);
                for (size_t i = 0; i < items.size(); ++i) out += (i ? s.as_string() : "") + items[i].str();
                return Value(std::move(out));
            });
        }
        if (name == "isdigit" || name == "isspace" || name == "isalpha" || name == "isalnum" || name == "isupper" ||
            name == "islower") {
            const std::string which = name;
            return bind([which](const Value& s, Array&, Kwargs&) {
                const std::string& t = s.as_string();
                if (t.empty()) return Value(false);
                bool cased = false;
                for (size_t i = 0; i < t.size();) {
                    const uint32_t cp = unicode::decode_utf8(t, i);
                    const auto c = unicode::category(cp);
                    const bool letter = c <= unicode::Lo, digit = c == unicode::Nd;
                    if (which == "isdigit" && !digit) return Value(false);
                    if (which == "isspace" && !unicode::is_whitespace(cp)) return Value(false);
                    if (which == "isalpha" && !letter) return Value(false);
                    if (which == "isalnum" && !letter && !digit && c != unicode::Nl && c != unicode::No) return Value(false);
                    if (which == "isupper" && c == unicode::Ll) return Value(false);
                    if (which == "islower" && c == unicode::Lu) return Value(false);
                    cased = cased || c == unicode::Lu || c == unicode::Ll || c == unicode::Lt;
                }
                return Value(which == "isupper" || which == "islower" ? cased : true);
            });
        }
        if (name == "format") {
            return bind([](const Value& s, Array& args, Kwargs& kwargs) {
                std::string out;
                const std::string& f = s.as_string();
                size_t next = 0;
                for (size_t i = 0; i < f.size(); ++i) {
                    if (f[i] == '{' && i + 1 < f.size() && f[i + 1] == '{') {
                        out += '{';
                        ++i;
                    } else if (f[i] == '}' && i + 1 < f.size() && f[i + 1] == '}') {
                        out += '}';
                        ++i;
                    } else if (f[i] == '{') {
                        const size_t close = f.find('}', i);
                        if (close == std::string::npos) fail("bad format string");
                        const std::string field = f.substr(i + 1, close - i - 1);
                        if (field.empty()) {
                            if (next >= args.size()) fail("not enough format arguments");
                            out += args[next++].str();
                        } else if (std::isdigit(static_cast<unsigned char>(field[0]))) {
                            const size_t k = std::stoul(field);
                            if (k >= args.size()) fail("format index out of range");
                            out += args[k].str();
                        } else {
                            const Value* v = kwarg(kwargs, field);
                            if (v == nullptr) fail("missing format argument '" + field + "'");
                            out += v->str();
                        }
                        i = close;
                    } else {
                        out += f[i];
                    }
                }
                return Value(std::move(out));
            });
        }
        return Value::undefined(name);
    }
    if (self.is_array()) {
        if (name == "index" || name == "count") {
            const bool count = name == "count";
            return bind([count](const Value& l, Array& args, Kwargs&) {
                if (args.empty()) fail("needs an argument");
                int64_t n = 0;
                for (size_t i = 0; i < l.items().size(); ++i) {
                    if (l.items()[i] == args[0]) {
                        if (!count) return Value(static_cast<int64_t>(i));
                        ++n;
                    }
                }
                if (!count) fail("value is not in list");
                return Value(n);
            });
        }
        static const char* const kMutating[] = {"append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse"};
        if (std::any_of(std::begin(kMutating), std::end(kMutating), [&](const char* m) { return name == m; })) {
            fail("access to attribute '" + name + "' of 'list' object is unsafe (lists are immutable in chat templates)");
        }
        return Value::undefined(name);
    }
    if (self.is_object()) {
        if (name == "items" || name == "keys" || name == "values") {
            const std::string which = name;
            return bind([which](const Value& d, Array&, Kwargs&) {
                Array out;
                for (const auto& [k, v] : d.fields().entries) {
                    if (which == "items") out.push_back(Value::array({Value(k), v}));
                    else if (which == "keys") out.emplace_back(k);
                    else out.push_back(v);
                }
                return Value::array(std::move(out));
            });
        }
        if (name == "get") {
            return bind([](const Value& d, Array& args, Kwargs&) {
                if (args.empty()) fail("get needs a key");
                if (args[0].is_string()) {
                    if (const Value* v = d.fields().find(args[0].as_string())) return *v;
                }
                return args.size() > 1 ? args[1] : Value::none();
            });
        }
        if (name == "copy") {
            return bind([](const Value& d, Array&, Kwargs&) {
                Value c = Value::object();
                c.fields().entries = d.fields().entries;
                return c;
            });
        }
        fail("access to attribute '" + name + "' of 'dict' object is unsafe (dicts are immutable in chat templates)");
    }
    return Value::undefined(name);
}

bool Interpreter::test(const std::string& name, const Value& v, Array& args) {
    using T = Value::Type;
    if (name == "defined") return !v.is_undefined();
    if (name == "undefined") return v.is_undefined();
    if (name == "none") return v.is_none();
    if (name == "true") return v.type() == T::Bool && v.as_bool();
    if (name == "false") return v.type() == T::Bool && !v.as_bool();
    if (name == "boolean") return v.type() == T::Bool;
    if (name == "string") return v.is_string();
    if (name == "number") return v.is_number();
    if (name == "integer") return v.type() == T::Int;
    if (name == "float") return v.type() == T::Float;
    if (name == "mapping") return v.is_object();
    if (name == "iterable") return v.is_string() || v.is_array() || v.is_object() || v.is_undefined();
    if (name == "sequence") return v.is_string() || v.is_array() || v.is_object();
    if (name == "callable") return v.type() == T::Callable;
    if (name == "odd") return v.as_int() % 2 != 0;
    if (name == "even") return v.as_int() % 2 == 0;
    if (name == "lower") return v.is_string() && to_lower(v.as_string()) == v.as_string();
    if (name == "upper") return v.is_string() && to_upper(v.as_string()) == v.as_string();
    if (args.empty() && (name == "divisibleby" || name == "eq" || name == "equalto" || name == "sameas" || name == "in" ||
                         name == "ne" || name == "lt" || name == "le" || name == "gt" || name == "ge")) {
        fail("test '" + name + "' needs an argument");
    }
    if (name == "divisibleby") return args[0].as_int() != 0 && v.as_int() % args[0].as_int() == 0;
    if (name == "eq" || name == "equalto" || name == "==") return v == args[0];
    if (name == "ne" || name == "!=") return !(v == args[0]);
    if (name == "sameas") return v.type() == args[0].type() && v == args[0];
    if (name == "lt" || name == "lessthan") return compare_values(v, args[0]) < 0;
    if (name == "le") return compare_values(v, args[0]) <= 0;
    if (name == "gt" || name == "greaterthan") return compare_values(v, args[0]) > 0;
    if (name == "ge") return compare_values(v, args[0]) >= 0;
    if (name == "in") return binary("in", v, args[0]).truthy();
    fail("unknown test '" + name + "'");
}

Value Interpreter::filter(const std::string& name, Value input, Array& args, Kwargs& kwargs) {
    auto sequence = [&](const Value& v) { return iterate(v); };
    if (name == "safe" || name == "e" || name == "escape" || name == "forceescape") {
        if (name != "safe") fail("filter '" + name + "' is not supported");
        return input;
    }
    if (name == "trim") {
        require_defined(input);
        const Value* chars = arg(args, kwargs, 0, "chars");
        return Value(strip_chars(input.str(), chars, true, true));
    }
    if (name == "length" || name == "count") return Value(py_len(input));
    if (name == "string") return Value(input.str());
    if (name == "tojson") {
        const Value* indent = arg(args, kwargs, 0, "indent");
        const Value* ensure = kwarg(kwargs, "ensure_ascii");
        const Value* sort = kwarg(kwargs, "sort_keys");
        const Value* seps = kwarg(kwargs, "separators");
        std::string item_sep, key_sep;
        if (seps && seps->is_array() && seps->items().size() == 2) {
            item_sep = seps->items()[0].str();
            key_sep = seps->items()[1].str();
        }
        int ind = -1;
        if (indent && !indent->is_none()) ind = static_cast<int>(indent->as_int());
        return Value(input.to_json(ind, ensure && ensure->truthy(), item_sep, key_sep, sort && sort->truthy()));
    }
    if (name == "items") {
        if (input.is_undefined()) return Value::array();
        if (!input.is_object()) fail("items filter expects a mapping");
        return call(method(input, "items"), args, kwargs);
    }
    if (name == "list") return Value::array(sequence(input));
    if (name == "join") {
        const Value* sep = arg(args, kwargs, 0, "d");
        const Value* attr = arg(args, kwargs, 1, "attribute");
        std::string out;
        const Array items = sequence(input);
        for (size_t i = 0; i < items.size(); ++i) {
            if (i && sep) out += sep->str();
            out += attr ? attribute(items[i], attr->str()).str() : items[i].str();
        }
        return Value(std::move(out));
    }
    if (name == "default" || name == "d") {
        const Value* def = arg(args, kwargs, 0, "default_value");
        const Value* boolean = arg(args, kwargs, 1, "boolean");
        const bool use_default = input.is_undefined() || (boolean && boolean->truthy() && !input.truthy());
        return use_default ? (def ? *def : Value(std::string())) : input;
    }
    if (name == "upper") return Value(to_upper(input.str()));
    if (name == "lower") return Value(to_lower(input.str()));
    if (name == "capitalize" || name == "title") return call(method(Value(input.str()), name), args, kwargs);
    if (name == "replace") return call(method(Value(input.str()), "replace"), args, kwargs);
    if (name == "first" || name == "last") {
        const Array items = sequence(input);
        if (items.empty()) return Value::undefined(name);
        return name == "first" ? items.front() : items.back();
    }
    if (name == "reverse") {
        if (input.is_string()) return eval_slice_reverse(input);
        Array items = sequence(input);
        std::reverse(items.begin(), items.end());
        return Value::array(std::move(items));
    }
    if (name == "int") {
        if (input.is_number()) return Value(input.as_int());
        try {
            return Value(static_cast<int64_t>(std::stoll(input.str())));
        } catch (...) {
            return args.empty() ? Value(static_cast<int64_t>(0)) : args[0];
        }
    }
    if (name == "float") {
        if (input.is_number()) return Value(input.as_float());
        try {
            return Value(std::stod(input.str()));
        } catch (...) {
            return args.empty() ? Value(0.0) : args[0];
        }
    }
    if (name == "abs") {
        if (input.type() == Value::Type::Float) return Value(std::fabs(input.as_float()));
        return Value(input.as_int() < 0 ? -input.as_int() : input.as_int());
    }
    if (name == "round") {
        const int64_t precision = args.empty() ? 0 : args[0].as_int();
        const double scale = std::pow(10.0, static_cast<double>(precision));
        return Value(std::nearbyint(input.as_float() * scale) / scale);
    }
    if (name == "sum") {
        Value total(static_cast<int64_t>(0));
        for (const Value& v : sequence(input)) total = binary("+", total, v);
        return total;
    }
    if (name == "min" || name == "max") {
        const Array items = sequence(input);
        if (items.empty()) return Value::undefined(name);
        Value best = items[0];
        for (const Value& v : items) {
            const int c = compare_values(v, best);
            if (name == "min" ? c < 0 : c > 0) best = v;
        }
        return best;
    }
    if (name == "unique") {
        Array out;
        for (const Value& v : sequence(input)) {
            if (std::none_of(out.begin(), out.end(), [&](const Value& o) { return o == v; })) out.push_back(v);
        }
        return Value::array(std::move(out));
    }
    if (name == "sort" || name == "dictsort") {
        Array items = name == "dictsort" ? call(method(input, "items"), args, kwargs).items() : sequence(input);
        const Value* reverse = kwarg(kwargs, "reverse");
        const Value* attr = kwarg(kwargs, "attribute");
        std::stable_sort(items.begin(), items.end(), [&](const Value& a, const Value& b) {
            const Value ka = attr ? attribute(a, attr->str()) : a;
            const Value kb = attr ? attribute(b, attr->str()) : b;
            return compare_values(ka, kb) < 0;
        });
        if (reverse && reverse->truthy()) std::reverse(items.begin(), items.end());
        return Value::array(std::move(items));
    }
    if (name == "map") {
        Array out;
        const Value* attr = kwarg(kwargs, "attribute");
        for (const Value& v : sequence(input)) {
            if (attr) {
                Value got = attribute(v, attr->str());
                if (got.is_undefined()) {
                    if (const Value* def = kwarg(kwargs, "default")) got = *def;
                }
                out.push_back(got);
            } else {
                if (args.empty()) fail("map needs a filter name or attribute=");
                Array rest(args.begin() + 1, args.end());
                Kwargs none;
                out.push_back(filter(args[0].str(), v, rest, none));
            }
        }
        return Value::array(std::move(out));
    }
    if (name == "select" || name == "reject" || name == "selectattr" || name == "rejectattr") {
        const bool keep = name == "select" || name == "selectattr";
        const bool by_attr = name == "selectattr" || name == "rejectattr";
        Array out;
        for (const Value& v : sequence(input)) {
            size_t first = 0;
            Value subject = v;
            if (by_attr) {
                if (args.empty()) fail(name + " needs an attribute name");
                subject = attribute(v, args[0].str());
                first = 1;
            }
            bool ok;
            if (args.size() <= first) {
                ok = subject.truthy();
            } else {
                Array rest(args.begin() + static_cast<std::ptrdiff_t>(first) + 1, args.end());
                ok = test(args[first].str(), subject, rest);
            }
            if (ok == keep) out.push_back(v);
        }
        return Value::array(std::move(out));
    }
    if (name == "attr") {
        if (args.empty()) fail("attr needs a name");
        return attribute(input, args[0].str());
    }
    if (name == "indent") {
        const int64_t width = args.empty() ? 4 : args[0].as_int();
        const bool first = args.size() > 1 && args[1].truthy();
        const std::string pad(static_cast<size_t>(std::max<int64_t>(0, width)), ' ');
        std::string out;
        const std::string s = input.str();
        bool line_start = first;
        for (const char c : s) {
            if (line_start && c != '\n') out += pad;
            out += c;
            line_start = c == '\n';
        }
        return Value(std::move(out));
    }
    if (name == "wordcount") return Value(static_cast<int64_t>(split_string(input.str(), nullptr, -1, false).size()));
    fail("unknown filter '" + name + "'");
}

}  // namespace

// ---------------------------------------------------------------------------
// ChatTemplate
// ---------------------------------------------------------------------------
Result<ChatTemplate> ChatTemplate::parse(std::string_view source) {
    try {
        ChatTemplate t;
        t.body_ = std::make_shared<const std::vector<Node>>(TemplateParser(split_template(source)).parse());
        return t;
    } catch (const TemplateError& e) {
        return Status(ErrorCode::InvalidArgument, std::string("chat template: ") + e.what());
    }
}

Result<std::string> ChatTemplate::render(const Value& context) const {
    if (!body_) return Status(ErrorCode::InvalidArgument, "chat template: not parsed");
    try {
        Interpreter::Scope globals;
        if (context.is_object()) {
            for (const auto& [k, v] : context.fields().entries) globals[k] = v;
        }
        Interpreter in(globals);
        in.exec(*body_);
        return std::move(in.out);
    } catch (const TemplateError& e) {
        return Status(ErrorCode::InvalidArgument, std::string("chat template: ") + e.what());
    }
}

Result<std::string> ChatTemplate::render_chat(std::string_view request_json, const Value& special) const {
    auto request = jinja::parse_json(request_json);
    if (!request) return request.status();
    if (!request->is_object()) return Status(ErrorCode::InvalidArgument, "chat request: expected a JSON object");
    const Value* messages = request->fields().find("messages");
    if (messages == nullptr || !messages->is_array()) {
        return Status(ErrorCode::InvalidArgument, "chat request: 'messages' must be an array");
    }
    // The variables transformers' apply_chat_template passes: tools and
    // documents (None when absent), add_generation_prompt, the special
    // tokens, then every other key of the request.
    Value context = Value::object();
    Object& c = context.fields();
    c.set("messages", *messages);
    const Value* tools = request->fields().find("tools");
    c.set("tools", tools ? *tools : Value::none());
    const Value* documents = request->fields().find("documents");
    c.set("documents", documents ? *documents : Value::none());
    const Value* agp = request->fields().find("add_generation_prompt");
    c.set("add_generation_prompt", Value(agp != nullptr && agp->truthy()));
    if (special.is_object()) {
        for (const auto& [k, v] : special.fields().entries) c.set(k, v);
    }
    for (const auto& [k, v] : request->fields().entries) {
        if (k == "messages" || k == "tools" || k == "documents" || k == "add_generation_prompt") continue;
        if (special.is_object() && special.fields().find(k) != nullptr) continue;
        c.set(k, v);
    }
    return render(context);
}

namespace {

// One rendered probe conversation; empty on an error (reason set).
struct Prober {
    const ChatTemplate& tmpl;
    Value variables;  // template variables of the probe
    const Value& special;
    std::string reason;

    std::optional<std::string> render(const std::vector<std::pair<std::string, std::string>>& turns, bool generation) {
        Value messages = Value::array();
        for (const auto& [role, content] : turns) {
            Value m = Value::object();
            m.fields().set("role", Value(role));
            m.fields().set("content", Value(content));
            messages.items().push_back(std::move(m));
        }
        Value context = Value::object();
        Object& c = context.fields();
        c.set("messages", messages);
        c.set("tools", Value::none());
        c.set("documents", Value::none());
        c.set("add_generation_prompt", Value(generation));
        for (const auto& [k, v] : special.fields().entries) c.set(k, v);
        if (variables.is_object()) {
            for (const auto& [k, v] : variables.fields().entries) {
                if (k != "reply_prefix") c.set(k, v);
            }
        }
        auto out = tmpl.render(context);
        if (!out) {
            if (reason.empty()) reason = out.status().message();
            return std::nullopt;
        }
        return std::move(out).value();
    }
};

}  // namespace

Result<ChatFormat> ChatTemplate::format(std::string_view probe_json, const Value& special) const {
    Value variables = Value::object();
    if (!probe_json.empty()) {
        auto parsed = jinja::parse_json(probe_json);
        if (!parsed) return parsed.status();
        if (!parsed->is_object()) return Status(ErrorCode::InvalidArgument, "chat format probe: expected a JSON object");
        variables = *parsed;
    }
    std::string reply_prefix;
    if (const Value* rp = variables.fields().find("reply_prefix")) reply_prefix = rp->str();

    ChatFormat f;
    Prober p{*this, variables, special, {}};
    auto unstable = [&](const std::string& why) {
        f.prefix_stable = false;
        f.reason = why;
        return f;
    };
    // Probe texts: plain words a template has no reason to alter.
    const std::string S = "LiyabProbeSystem", U1 = "LiyabProbeUserOne", U2 = "LiyabProbeUserTwo",
                      U3 = "LiyabProbeUserThree", A1 = reply_prefix + "LiyabProbeReplyOne",
                      A2 = reply_prefix + "LiyabProbeReplyTwo";
    using Turns = std::vector<std::pair<std::string, std::string>>;
    const Turns t1 = {{"system", S}, {"user", U1}};
    const Turns t2 = {{"system", S}, {"user", U1}, {"assistant", A1}};
    const Turns t3 = {{"system", S}, {"user", U1}, {"assistant", A1}, {"user", U2}};
    const Turns t5 = {{"system", S}, {"user", U1}, {"assistant", A1}, {"user", U2}, {"assistant", A2}, {"user", U3}};
    // Prompts as a front end sends them (with the generation prompt), and two
    // renders without it to tell a turn's own end from what opens the next.
    const auto w1g = p.render(t1, true), w3g = p.render(t3, true), w5g = p.render(t5, true), w1 = p.render(t1, false),
               w2 = p.render(t2, false), n1g = p.render({{"user", U1}}, true);
    if (!w1g || !w3g || !w5g || !w1 || !w2 || !n1g) return unstable("a probe conversation failed: " + p.reason);
    auto common_prefix = [](const std::string& a, const std::string& b) {
        size_t n = 0;
        while (n < a.size() && n < b.size() && a[n] == b[n]) ++n;
        return a.substr(0, n);
    };

    // What follows a user's text in a prompt: the end of the user turn, then
    // the generation prompt.
    const size_t u1_at = w1g->find(U1);
    if (u1_at == std::string::npos) return unstable("the user's text is altered");
    const std::string after_user = w1g->substr(u1_at + U1.size());
    // n -> n + 1 turns: the reply generated after the first prompt, then the
    // second user turn, must extend that prompt.
    if (!w3g->starts_with(*w1g + A1)) {
        return unstable("the next turn's prompt does not continue the previous prompt and its reply");
    }
    const std::string rest = w3g->substr(w1g->size() + A1.size());
    const size_t u2_at = rest.find(U2);
    if (u2_at == std::string::npos || rest.substr(u2_at + U2.size()) != after_user) {
        return unstable("a later user turn renders differently from the first");
    }
    const std::string between_turns = rest.substr(0, u2_at);  // end of the reply, opening of the user turn
    if (*w5g != *w3g + A2 + between_turns + U3 + after_user) {
        return unstable("the second reply or the third user turn renders differently");
    }
    // Split the two pieces with the renders that end on a user turn / a reply.
    const size_t w1_u1 = w1->find(U1);
    const size_t w2_a1 = w2->find(A1);
    if (w1_u1 == std::string::npos || w2_a1 == std::string::npos) return unstable("a message's text is altered");
    f.user_suffix = common_prefix(w1->substr(w1_u1 + U1.size()), after_user);
    f.generation_prompt = after_user.substr(f.user_suffix.size());
    f.assistant_end = common_prefix(w2->substr(w2_a1 + A1.size()), between_turns);
    f.user_prefix = between_turns.substr(f.assistant_end.size());
    // The first turn: the system block, then what precedes the first user text.
    const std::string head = w1g->substr(0, u1_at);
    const size_t s_at = head.find(S);
    if (s_at == std::string::npos) return unstable("the system text is altered or placed after the user's");
    f.system_prefix = head.substr(0, s_at);
    const std::string between = head.substr(s_at + S.size());
    if (between.ends_with(f.user_prefix)) {
        f.system_suffix = between.substr(0, between.size() - f.user_prefix.size());
        f.first_user_prefix = f.user_prefix;
    } else {
        f.system_suffix = between;  // e.g. Gemma: the system text opens the first user turn
    }
    // Without a system message.
    const size_t n_at = n1g->find(U1);
    if (n_at == std::string::npos || n1g->substr(n_at + U1.size()) != after_user) {
        return unstable("a conversation without a system message ends its first turn differently");
    }
    f.no_system_head = n1g->substr(0, n_at);
    f.prefix_stable = true;
    return f;
}

}  // namespace liyab
