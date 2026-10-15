// JSON Schema → grammar (see liyab/grammar.h for the supported subset).
#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>

#include "core/chat_template.h"
#include "core/grammar.h"

namespace liyab::grammar {
namespace {

using jinja::Value;
using Node = Builder::Node;

// Keywords that only annotate: ignored wherever they appear.
const std::set<std::string_view> kAnnotations = {
    "title",    "description", "default",         "examples",        "$comment", "$schema",
    "$id",      "deprecated",  "readOnly",        "writeOnly",       "$defs",    "definitions",
    "$anchor", "contentMediaType", "contentEncoding", "contentSchema", "strict",  "discriminator",
    "xml",      "externalDocs", "example",
};
// Keywords that constrain values in ways this compiler does not enforce.
const std::set<std::string_view> kUnsupported = {
    "pattern",       "patternProperties", "propertyNames",   "contains",         "minContains",
    "maxContains",   "prefixItems",       "additionalItems", "multipleOf",       "not",
    "if",            "then",              "else",            "dependentRequired", "dependentSchemas",
    "dependencies",  "minProperties",     "maxProperties",   "unevaluatedProperties",
    "unevaluatedItems", "$dynamicRef",    "$recursiveRef",
};

constexpr uint64_t kMaxMagnitude = 9999999999999999999ull;  // 19 digits

std::string pointer_unescape(std::string_view token) {
    std::string out;
    for (size_t i = 0; i < token.size(); ++i) {
        if (token[i] == '~' && i + 1 < token.size() && (token[i + 1] == '0' || token[i + 1] == '1')) {
            out += token[i + 1] == '0' ? '~' : '/';
            ++i;
        } else {
            out += token[i];
        }
    }
    return out;
}

bool is_integral(const Value& v) {
    if (v.type() == Value::Type::Int) return true;
    if (v.type() == Value::Type::Float) {
        const double d = v.as_float();
        return std::isfinite(d) && std::floor(d) == d && std::fabs(d) < 9.2e18;
    }
    return false;
}

bool numeric(const Value& v) { return v.type() == Value::Type::Int || v.type() == Value::Type::Float; }

class SchemaCompiler {
public:
    SchemaCompiler(Builder& b, const JsonFormat& format, Value root) : b_(b), format_(format), root_(std::move(root)) {}

    Result<Node> compile(const Value& s, const std::string& path) {
        if (s.type() == Value::Type::Bool) {
            if (s.as_bool()) return any();
            return fail(ErrorCode::InvalidArgument, path, "the schema `false` matches no value");
        }
        if (!s.is_object()) return fail(ErrorCode::InvalidArgument, path, "a schema must be an object or a boolean");
        const jinja::Object& o = s.fields();
        for (const auto& [key, value] : o.entries) {
            if (kUnsupported.contains(key)) return fail(ErrorCode::Unsupported, path, "keyword '" + key + "' is not supported");
            if (key == "uniqueItems" && value.truthy()) {
                return fail(ErrorCode::Unsupported, path, "keyword 'uniqueItems' is not supported");
            }
            if (key == "items" && value.is_array()) {
                return fail(ErrorCode::Unsupported, path, "tuple 'items' (an array of schemas) is not supported");
            }
        }
        auto node = compile_object(s, o, path);
        if (!node) return node;
        if (const Value* nullable = o.find("nullable"); nullable != nullptr && nullable->truthy()) {
            return b_.alt({*node, b_.literal("null")});
        }
        return node;
    }

    Node any() {
        if (any_rule_) return b_.ref(*any_rule_);
        const uint32_t value = b_.add_rule("json-value");
        any_rule_ = value;
        const Node v = b_.ref(value);
        const Node object = members(b_.seq({string(0, kUnbounded), key_sep(), v}), 0, kUnbounded, '{', '}');
        const Node array = members(v, 0, kUnbounded, '[', ']');
        b_.define(value, b_.alt({object, array, string(0, kUnbounded), number(false, false), b_.literal("true"),
                                 b_.literal("false"), b_.literal("null")}));
        return v;
    }

    // White space around the root value (any_whitespace only).
    Node root(Node value) { return format_.any_whitespace ? b_.seq({ws(), value, ws()}) : value; }

private:
    Status fail(ErrorCode code, const std::string& path, const std::string& what) const {
        return Status(code, "JSON schema at " + path + ": " + what);
    }

    Node ws() {
        if (!ws_) ws_ = format_.any_whitespace ? b_.repeat(b_.chars({{' ', ' '}, {'\t', '\t'}, {'\n', '\n'}, {'\r', '\r'}}), 0, 32)
                                               : b_.empty();
        return *ws_;
    }
    Node key_sep() { return format_.any_whitespace ? b_.seq({ws(), b_.literal(":"), ws()}) : b_.literal(": "); }
    Node item_sep() { return format_.any_whitespace ? b_.seq({b_.literal(","), ws()}) : b_.literal(", "); }
    Node after_item() { return ws(); }

    // open item (sep item){min-1,max-1} close, with the layout's white space.
    Node members(Node item, uint32_t min, uint32_t max, char open, char close) {
        const Node one = b_.seq({item, after_item()});
        const Node more = b_.seq({item_sep(), one});
        const auto lo = min == 0 ? 0 : min - 1;
        const auto hi = max == kUnbounded ? kUnbounded : max - 1;
        Node body = b_.seq({one, b_.repeat(more, lo, hi)});
        if (min == 0) body = b_.optional(body);
        return b_.seq({b_.literal(std::string(1, open)), ws(), body, b_.literal(std::string(1, close))});
    }

    Node digit() { return b_.chars({{'0', '9'}}); }
    Node hex() { return b_.chars({{'0', '9'}, {'a', 'f'}, {'A', 'F'}}); }

    Node string_char() {
        if (!string_char_) {
            const Node plain = b_.chars({{0x20, 0x21}, {0x23, 0x5B}, {0x5D, 0x10FFFF}});
            const Node escape = b_.seq({b_.literal("\\"), b_.alt({b_.chars({{'"', '"'}, {'\\', '\\'}, {'/', '/'}, {'b', 'b'},
                                                                         {'f', 'f'}, {'n', 'n'}, {'r', 'r'}, {'t', 't'}}),
                                                                 b_.seq({b_.literal("u"), b_.repeat(hex(), 4, 4)})})});
            string_char_ = b_.alt({plain, escape});
        }
        return *string_char_;
    }

    Node string(uint32_t min, uint32_t max) {
        const Node body = b_.repeat(string_char(), min, max);
        return b_.seq({b_.literal("\""), body, b_.literal("\"")});
    }

    // RFC 3339 pieces for the formats.
    Node date() {
        return b_.seq({b_.repeat(digit(), 4, 4), b_.literal("-"),
                       b_.alt({b_.seq({b_.literal("0"), b_.chars({{'1', '9'}})}), b_.seq({b_.literal("1"), b_.chars({{'0', '2'}})})}),
                       b_.literal("-"),
                       b_.alt({b_.seq({b_.literal("0"), b_.chars({{'1', '9'}})}), b_.seq({b_.chars({{'1', '2'}}), digit()}),
                               b_.seq({b_.literal("3"), b_.chars({{'0', '1'}})})})});
    }
    Node time() {
        const Node hour = b_.alt({b_.seq({b_.chars({{'0', '1'}}), digit()}), b_.seq({b_.literal("2"), b_.chars({{'0', '3'}})})});
        const Node minute = b_.seq({b_.chars({{'0', '5'}}), digit()});
        const Node second = b_.alt({minute, b_.literal("60")});
        const Node fraction = b_.optional(b_.seq({b_.literal("."), b_.repeat(digit(), 1, 9)}));
        const Node offset = b_.alt({b_.chars({{'Z', 'Z'}, {'z', 'z'}}),
                                    b_.seq({b_.chars({{'+', '+'}, {'-', '-'}}), hour, b_.literal(":"), minute})});
        return b_.seq({hour, b_.literal(":"), minute, b_.literal(":"), second, fraction, offset});
    }

    // -?(0|[1-9][0-9]{0,18})(\.[0-9]{1,16})?([eE][+-]?[0-9]{1,3})?
    Node number(bool non_negative, bool integer) {
        std::vector<Node> parts;
        if (!non_negative) parts.push_back(b_.optional(b_.literal("-")));
        parts.push_back(b_.alt({b_.literal("0"), b_.seq({b_.chars({{'1', '9'}}), b_.repeat(digit(), 0, 18)})}));
        if (!integer) {
            parts.push_back(b_.optional(fraction()));
            parts.push_back(b_.optional(
                b_.seq({b_.chars({{'e', 'e'}, {'E', 'E'}}), b_.optional(b_.chars({{'+', '+'}, {'-', '-'}})), b_.repeat(digit(), 1, 3)})));
        }
        return b_.seq(std::move(parts));
    }
    Node fraction() { return b_.seq({b_.literal("."), b_.repeat(digit(), 1, 16)}); }

    // Decimal numbers from `a` to `b`, digit strings of the same length.
    Node digit_range(const std::string& a, const std::string& b) {
        const size_t n = a.size();
        if (n == 0) return b_.empty();
        if (a == b) return b_.literal(a);
        const std::string at = a.substr(1), bt = b.substr(1);
        if (a[0] == b[0]) return b_.seq({b_.literal(a.substr(0, 1)), digit_range(at, bt)});
        const bool low_all = at.find_first_not_of('0') == std::string::npos;
        const bool high_all = bt.find_first_not_of('9') == std::string::npos;
        std::vector<Node> alts;
        char first = a[0], last = b[0];
        if (!low_all) {
            alts.push_back(b_.seq({b_.literal(a.substr(0, 1)), digit_range(at, std::string(n - 1, '9'))}));
            ++first;
        }
        std::optional<Node> high;
        if (!high_all) {
            high = b_.seq({b_.literal(b.substr(0, 1)), digit_range(std::string(n - 1, '0'), bt)});
            --last;
        }
        if (first <= last) {
            alts.push_back(b_.seq({b_.chars({{static_cast<uint32_t>(first), static_cast<uint32_t>(last)}}),
                                   b_.repeat(digit(), static_cast<uint32_t>(n - 1), static_cast<uint32_t>(n - 1))}));
        }
        if (high) alts.push_back(*high);
        return b_.alt(std::move(alts));
    }

    // Non-negative integers in [lo, hi], without leading zeros.
    Node uint_range(uint64_t lo, uint64_t hi) {
        std::vector<Node> alts;
        const std::string hs = std::to_string(hi);
        uint64_t from = lo;
        for (size_t d = std::to_string(lo).size(); d <= hs.size(); ++d) {
            uint64_t top = 0;
            if (d == hs.size()) {
                top = hi;
            } else {
                top = 1;
                for (size_t i = 0; i < d; ++i) top *= 10;
                --top;
            }
            alts.push_back(digit_range(std::to_string(from), std::to_string(top)));
            from = top + 1;
        }
        return b_.alt(std::move(alts));
    }

    // Integers in [lo, hi] (absent: 19 digits at most).
    Node int_range(std::optional<int64_t> lo, std::optional<int64_t> hi) {
        std::vector<Node> alts;
        auto magnitude = [](int64_t v) { return v < 0 ? uint64_t{0} - static_cast<uint64_t>(v) : static_cast<uint64_t>(v); };
        if (!lo || *lo < 0) {  // negative values: '-' then their magnitude
            const uint64_t mlo = hi && *hi < 0 ? magnitude(*hi) : 1;
            const uint64_t mhi = lo ? magnitude(*lo) : kMaxMagnitude;
            if (mlo <= mhi) alts.push_back(b_.seq({b_.literal("-"), uint_range(mlo, mhi)}));
        }
        if (!hi || *hi >= 0) {
            const uint64_t from = lo && *lo > 0 ? static_cast<uint64_t>(*lo) : 0;
            const uint64_t to = hi ? static_cast<uint64_t>(*hi) : kMaxMagnitude;
            if (from <= to) alts.push_back(uint_range(from, to));
        }
        return b_.alt(std::move(alts));
    }

    // Numbers in [lo, hi] (integral bounds, either may be absent), without
    // an exponent: an integer part in range with any fraction, or the upper
    // bound itself.
    Node real_range(std::optional<int64_t> lo, std::optional<int64_t> hi) {
        auto magnitude = [](int64_t v) { return v < 0 ? uint64_t{0} - static_cast<uint64_t>(v) : static_cast<uint64_t>(v); };
        const Node frac = b_.optional(fraction());
        std::vector<Node> alts;
        if (!hi || *hi >= 0) {
            const uint64_t from = lo && *lo > 0 ? static_cast<uint64_t>(*lo) : 0;
            if (!hi) {
                alts.push_back(b_.seq({uint_range(from, kMaxMagnitude), frac}));
            } else {
                const auto top = static_cast<uint64_t>(*hi);
                if (top > 0 && from <= top - 1) alts.push_back(b_.seq({uint_range(from, top - 1), frac}));
                if (from <= top) alts.push_back(b_.literal(std::to_string(top)));
            }
        }
        if (!lo || *lo < 0) {  // -y with y from max(-hi, 0) up to -lo
            const uint64_t from = hi && *hi < 0 ? magnitude(*hi) : 0;
            std::vector<Node> neg;
            if (!lo) {
                neg.push_back(b_.seq({uint_range(from, kMaxMagnitude), frac}));
            } else {
                const uint64_t top = magnitude(*lo);
                if (from <= top - 1) neg.push_back(b_.seq({uint_range(from, top - 1), frac}));
                neg.push_back(b_.literal(std::to_string(top)));
            }
            alts.push_back(b_.seq({b_.literal("-"), b_.alt(std::move(neg))}));
        }
        return b_.alt(std::move(alts));
    }

    std::string json_literal(const Value& v) const { return v.to_json(-1, false, ", ", ": "); }

    // Whether `v` is an instance of JSON type `t`.
    static bool of_type(const Value& v, std::string_view t) {
        switch (v.type()) {
            case Value::Type::None: return t == "null";
            case Value::Type::Bool: return t == "boolean";
            case Value::Type::Int: return t == "integer" || t == "number";
            case Value::Type::Float: return t == "number" || (t == "integer" && is_integral(v));
            case Value::Type::String: return t == "string";
            case Value::Type::Array: return t == "array";
            case Value::Type::Object: return t == "object";
            default: return false;
        }
    }

    Result<uint32_t> ref_rule(const std::string& ref, const std::string& path) {
        if (const auto it = refs_.find(ref); it != refs_.end()) return it->second;
        if (ref.empty() || ref[0] != '#') return fail(ErrorCode::Unsupported, path, "only local $ref (\"#...\") is supported");
        Value target = root_;
        std::string_view rest = std::string_view(ref).substr(1);
        while (!rest.empty()) {
            if (rest[0] != '/') return fail(ErrorCode::Unsupported, path, "$ref '" + ref + "' is not a JSON pointer");
            rest.remove_prefix(1);
            const size_t slash = rest.find('/');
            const std::string token = pointer_unescape(rest.substr(0, slash));
            rest = slash == std::string_view::npos ? std::string_view() : rest.substr(slash);
            const Value* next = nullptr;
            Value element;
            if (target.is_object()) {
                next = target.fields().find(token);
            } else if (target.is_array() && !token.empty() && token.find_first_not_of("0123456789") == std::string::npos) {
                const size_t i = std::stoul(token);
                if (i < target.items().size()) next = &target.items()[i];
            }
            if (next == nullptr) return fail(ErrorCode::InvalidArgument, path, "$ref '" + ref + "' points nowhere");
            target = *next;
        }
        const uint32_t rule = b_.add_rule(ref);
        refs_.emplace(ref, rule);
        auto node = compile(target, ref);
        if (!node) return node.status();
        b_.define(rule, *node);
        return rule;
    }

    // Constraint keywords next to anyOf / oneOf / allOf / $ref are refused
    // (they would need an intersection of schemas).
    Status no_siblings(const jinja::Object& o, std::string_view keyword, const std::string& path) const {
        for (const auto& [key, value] : o.entries) {
            if (key == keyword || key == "nullable" || kAnnotations.contains(key)) continue;
            if (key == "type" && keyword != "$ref") continue;  // checked per branch below
            return fail(ErrorCode::Unsupported, path, "'" + key + "' next to '" + std::string(keyword) + "' is not supported");
        }
        return Status::ok();
    }

    Result<Node> compile_object(const Value& s, const jinja::Object& o, const std::string& path) {
        if (const Value* ref = o.find("$ref")) {
            LIYAB_RETURN_IF_ERROR(no_siblings(o, "$ref", path));
            if (!ref->is_string()) return fail(ErrorCode::InvalidArgument, path, "$ref must be a string");
            auto rule = ref_rule(ref->as_string(), path);
            if (!rule) return rule.status();
            return b_.ref(*rule);
        }
        for (const char* keyword : {"anyOf", "oneOf"}) {
            const Value* branches = o.find(keyword);
            if (branches == nullptr) continue;
            LIYAB_RETURN_IF_ERROR(no_siblings(o, keyword, path));
            if (!branches->is_array() || branches->items().empty()) {
                return fail(ErrorCode::InvalidArgument, path, std::string(keyword) + " must be a non-empty array");
            }
            if (o.find("type") != nullptr) return fail(ErrorCode::Unsupported, path, "'type' next to '" + std::string(keyword) + "' is not supported");
            std::vector<Node> alts;
            for (size_t i = 0; i < branches->items().size(); ++i) {
                auto n = compile(branches->items()[i], path + "/" + keyword + "/" + std::to_string(i));
                if (!n) return n;
                alts.push_back(*n);
            }
            return b_.alt(std::move(alts));
        }
        if (const Value* all = o.find("allOf")) {
            LIYAB_RETURN_IF_ERROR(no_siblings(o, "allOf", path));
            if (!all->is_array() || all->items().size() != 1 || o.find("type") != nullptr) {
                return fail(ErrorCode::Unsupported, path, "allOf is supported with one schema only");
            }
            return compile(all->items()[0], path + "/allOf/0");
        }

        // The types this schema admits.
        std::vector<std::string> types;
        if (const Value* t = o.find("type")) {
            if (t->is_string()) {
                types.push_back(t->as_string());
            } else if (t->is_array()) {
                for (const Value& v : t->items()) {
                    if (!v.is_string()) return fail(ErrorCode::InvalidArgument, path, "'type' entries must be strings");
                    types.push_back(v.as_string());
                }
            } else {
                return fail(ErrorCode::InvalidArgument, path, "'type' must be a string or an array");
            }
            static const std::set<std::string_view> known = {"object", "array", "string", "number", "integer", "boolean", "null"};
            for (const std::string& t2 : types) {
                if (!known.contains(t2)) return fail(ErrorCode::InvalidArgument, path, "unknown type '" + t2 + "'");
            }
        }
        auto admits = [&](const Value& v) {
            if (types.empty()) return true;
            return std::any_of(types.begin(), types.end(), [&](const std::string& t) { return of_type(v, t); });
        };

        if (const Value* c = o.find("const")) {
            if (!admits(*c)) return fail(ErrorCode::InvalidArgument, path, "'const' does not match 'type'");
            return b_.literal(json_literal(*c));
        }
        if (const Value* e = o.find("enum")) {
            if (!e->is_array()) return fail(ErrorCode::InvalidArgument, path, "'enum' must be an array");
            std::vector<Node> alts;
            std::set<std::string> seen;
            for (const Value& v : e->items()) {
                const std::string text = json_literal(v);
                if (admits(v) && seen.insert(text).second) alts.push_back(b_.literal(text));
            }
            if (alts.empty()) return fail(ErrorCode::InvalidArgument, path, "no 'enum' value matches the schema");
            return b_.alt(std::move(alts));
        }
        if (types.empty()) {  // inferred from the keywords present
            if (o.find("properties") || o.find("additionalProperties") || o.find("required")) types.push_back("object");
            else if (o.find("items") || o.find("minItems") || o.find("maxItems")) types.push_back("array");
            else if (o.find("minLength") || o.find("maxLength") || o.find("format")) types.push_back("string");
            else if (o.find("minimum") || o.find("maximum") || o.find("exclusiveMinimum") || o.find("exclusiveMaximum"))
                types.push_back("number");
            else return any();
        }
        std::vector<Node> alts;
        for (const std::string& t : types) {
            auto n = typed(s, o, t, path);
            if (!n) return n;
            alts.push_back(*n);
        }
        return b_.alt(std::move(alts));
    }

    Result<uint32_t> count(const jinja::Object& o, const char* key, uint32_t fallback, const std::string& path) const {
        const Value* v = o.find(key);
        if (v == nullptr) return fallback;
        if (!is_integral(*v) || v->as_float() < 0) return fail(ErrorCode::InvalidArgument, path, std::string("'") + key + "' must be a non-negative integer");
        if (v->as_float() > kMaxRepeat) {
            return fail(ErrorCode::Unsupported, path, std::string("'") + key + "' above " + std::to_string(kMaxRepeat) + " is not supported");
        }
        return static_cast<uint32_t>(v->as_int());
    }

    Result<Node> typed(const Value& s, const jinja::Object& o, const std::string& t, const std::string& path) {
        (void)s;
        if (t == "null") return b_.literal("null");
        if (t == "boolean") return b_.alt({b_.literal("true"), b_.literal("false")});
        if (t == "string") return string_schema(o, path);
        if (t == "integer" || t == "number") return number_schema(o, t == "integer", path);
        if (t == "array") return array_schema(o, path);
        return object_schema(o, path);
    }

    // `raw`: the string's characters as they are, without quotes or escapes
    // (a value inside markup, such as an XML tool call's parameter).
    Result<Node> string_schema(const jinja::Object& o, const std::string& path, bool raw = false) {
        auto min = count(o, "minLength", 0, path);
        if (!min) return min.status();
        auto max = count(o, "maxLength", kUnbounded, path);
        if (!max) return max.status();
        if (*max != kUnbounded && *min > *max) return fail(ErrorCode::InvalidArgument, path, "minLength exceeds maxLength");
        if (const Value* f = o.find("format"); f != nullptr && f->is_string()) {
            const std::string& name = f->as_string();
            std::optional<Node> body;
            if (name == "date") body = date();
            else if (name == "time") body = time();
            else if (name == "date-time") body = b_.seq({date(), b_.chars({{'T', 'T'}, {'t', 't'}}), time()});
            else if (name == "uuid") {
                const Node h = hex();
                body = b_.seq({b_.repeat(h, 8, 8), b_.literal("-"), b_.repeat(h, 4, 4), b_.literal("-"), b_.repeat(h, 4, 4),
                               b_.literal("-"), b_.repeat(h, 4, 4), b_.literal("-"), b_.repeat(h, 12, 12)});
            }
            if (body) {
                if (o.find("minLength") || o.find("maxLength")) {
                    return fail(ErrorCode::Unsupported, path, "minLength / maxLength with format '" + name + "' is not supported");
                }
                return raw ? *body : b_.seq({b_.literal("\""), *body, b_.literal("\"")});
            }
        }
        return raw ? b_.repeat(b_.any_char(), *min, *max) : string(*min, *max);
    }

public:
    // A string schema's value as raw text (grammar text's %string).
    Result<Node> raw_string(const Value& s, const std::string& path) {
        if (!s.is_object()) return fail(ErrorCode::InvalidArgument, path, "%string needs a schema object");
        const jinja::Object& o = s.fields();
        for (const auto& [key, value] : o.entries) {
            if (kUnsupported.contains(key)) return fail(ErrorCode::Unsupported, path, "keyword '" + key + "' is not supported");
        }
        auto literal_of = [&](const Value& v) -> Result<Node> {
            if (!v.is_string()) return fail(ErrorCode::InvalidArgument, path, "%string values must be strings");
            return b_.literal(v.as_string());
        };
        if (const Value* c = o.find("const")) return literal_of(*c);
        if (const Value* e = o.find("enum")) {
            if (!e->is_array() || e->items().empty()) return fail(ErrorCode::InvalidArgument, path, "'enum' must be a non-empty array");
            std::vector<Node> alts;
            for (const Value& v : e->items()) {
                auto n = literal_of(v);
                if (!n) return n;
                alts.push_back(*n);
            }
            return b_.alt(std::move(alts));
        }
        const Value* t = o.find("type");
        if (t == nullptr || !t->is_string() || t->as_string() != "string") {
            return fail(ErrorCode::InvalidArgument, path, "%string needs a schema of type \"string\"");
        }
        return string_schema(o, path, true);
    }

private:

    Result<Node> number_schema(const jinja::Object& o, bool integer, const std::string& path) {
        std::optional<int64_t> lo, hi;
        bool positive = false;  // number with exclusiveMinimum 0
        auto bound = [&](const char* key, bool upper, bool exclusive) -> Status {
            const Value* v = o.find(key);
            if (v == nullptr) return Status::ok();
            bool excl = exclusive;
            const Value* value = v;
            if (v->type() == Value::Type::Bool) {  // draft 4: exclusiveMinimum: true qualifies minimum
                return Status::ok();
            }
            if (!numeric(*value)) return fail(ErrorCode::InvalidArgument, path, std::string("'") + key + "' must be a number");
            const double d = value->as_float();
            if (!exclusive) {  // draft 4 boolean form
                const Value* flag = o.find(upper ? "exclusiveMaximum" : "exclusiveMinimum");
                excl = flag != nullptr && flag->type() == Value::Type::Bool && flag->as_bool();
            }
            if (std::fabs(d) > 9.2e18) return fail(ErrorCode::Unsupported, path, std::string("'") + key + "' is out of range");
            int64_t b = 0;
            if (integer) {
                // The tightest integer bound: ceil / floor, one past for exclusive integral bounds.
                if (upper) b = static_cast<int64_t>(excl && std::floor(d) == d ? d - 1 : std::floor(d));
                else b = static_cast<int64_t>(excl && std::floor(d) == d ? d + 1 : std::ceil(d));
            } else {
                if (std::floor(d) != d) return fail(ErrorCode::Unsupported, path, std::string("a non-integer '") + key + "' on a number is not supported");
                if (excl) {
                    if (!upper && d == 0) {
                        positive = true;
                        return Status::ok();
                    }
                    return fail(ErrorCode::Unsupported, path, std::string("an exclusive '") + key + "' on a number is not supported (only exclusiveMinimum 0)");
                }
                b = static_cast<int64_t>(d);
            }
            if (upper) hi = hi ? std::min(*hi, b) : b;
            else lo = lo ? std::max(*lo, b) : b;
            return Status::ok();
        };
        LIYAB_RETURN_IF_ERROR(bound("minimum", false, false));
        LIYAB_RETURN_IF_ERROR(bound("maximum", true, false));
        LIYAB_RETURN_IF_ERROR(bound("exclusiveMinimum", false, true));
        LIYAB_RETURN_IF_ERROR(bound("exclusiveMaximum", true, true));
        if (lo && hi && *lo > *hi) return fail(ErrorCode::InvalidArgument, path, "no number lies between the bounds");
        if (integer) {
            if (!lo && !hi) return number(false, true);
            return int_range(lo, hi);
        }
        if (positive) {
            if (hi || (lo && *lo > 0)) {
                return fail(ErrorCode::Unsupported, path, "exclusiveMinimum 0 with other bounds on a number is not supported");
            }
            // > 0: an integer part from 1 (any fraction, exponent), or 0 and a
            // fraction ending in a non-zero digit.
            const Node exp = b_.optional(
                b_.seq({b_.chars({{'e', 'e'}, {'E', 'E'}}), b_.optional(b_.chars({{'+', '+'}, {'-', '-'}})), b_.repeat(digit(), 1, 3)}));
            const Node whole = b_.seq({b_.chars({{'1', '9'}}), b_.repeat(digit(), 0, 18), b_.optional(fraction()), exp});
            const Node small = b_.seq({b_.literal("0."), b_.repeat(digit(), 0, 15), b_.chars({{'1', '9'}}), exp});
            return b_.alt({whole, small});
        }
        if (!lo && !hi) return number(false, false);
        if (lo && *lo == 0 && !hi) return number(true, false);
        return real_range(lo, hi);
    }

    Result<Node> array_schema(const jinja::Object& o, const std::string& path) {
        auto min = count(o, "minItems", 0, path);
        if (!min) return min.status();
        auto max = count(o, "maxItems", kUnbounded, path);
        if (!max) return max.status();
        if (*max != kUnbounded && *min > *max) return fail(ErrorCode::InvalidArgument, path, "minItems exceeds maxItems");
        const Value* items = o.find("items");
        if (items != nullptr && items->type() == Value::Type::Bool && !items->as_bool()) {
            if (*min > 0) return fail(ErrorCode::InvalidArgument, path, "items false with minItems > 0 matches nothing");
            return b_.literal("[]");
        }
        Node item = 0;
        if (items == nullptr) {
            item = any();
        } else {
            auto n = compile(*items, path + "/items");
            if (!n) return n;
            item = *n;
        }
        if (*max == 0) return b_.literal("[]");
        return members(item, *min, *max, '[', ']');
    }

    Result<Node> object_schema(const jinja::Object& o, const std::string& path) {
        const Value* props = o.find("properties");
        const Value* additional = o.find("additionalProperties");
        if (props != nullptr && !props->is_object()) return fail(ErrorCode::InvalidArgument, path, "'properties' must be an object");
        std::vector<std::string> required;
        if (const Value* r = o.find("required")) {
            if (!r->is_array()) return fail(ErrorCode::InvalidArgument, path, "'required' must be an array");
            for (const Value& v : r->items()) {
                if (!v.is_string()) return fail(ErrorCode::InvalidArgument, path, "'required' entries must be strings");
                if (std::find(required.begin(), required.end(), v.as_string()) == required.end()) required.push_back(v.as_string());
            }
        }
        const bool closed = additional != nullptr && additional->type() == Value::Type::Bool && !additional->as_bool();
        // The value schema of names the properties do not declare.
        auto extra_value = [&](const std::string& where) -> Result<Node> {
            if (additional == nullptr || additional->type() == Value::Type::Bool) return any();
            return compile(*additional, where);
        };

        if (props == nullptr || props->fields().entries.empty()) {
            if (required.empty()) {
                if (closed) return b_.literal("{}");
                // A map: string keys, values per additionalProperties.
                auto value = extra_value(path + "/additionalProperties");
                if (!value) return value;
                return members(b_.seq({string(0, kUnbounded), key_sep(), *value}), 0, kUnbounded, '{', '}');
            }
        }
        struct Member {
            std::string name;
            Node value;
            bool required;
        };
        std::vector<Member> list;
        if (props != nullptr) {
            for (const auto& [name, schema] : props->fields().entries) {
                auto n = compile(schema, path + "/properties/" + name);
                if (!n) return n;
                const bool req = std::find(required.begin(), required.end(), name) != required.end();
                list.push_back({name, *n, req});
            }
        }
        for (const std::string& name : required) {
            const bool declared = std::any_of(list.begin(), list.end(), [&](const Member& m) { return m.name == name; });
            if (declared) continue;
            if (closed) return fail(ErrorCode::InvalidArgument, path, "required property '" + name + "' is not allowed (additionalProperties false)");
            auto n = extra_value(path + "/additionalProperties");
            if (!n) return n;
            list.push_back({name, *n, true});
        }
        // Properties in order, optional ones skippable: tail[i][first] is
        // what may follow once members before i are done (`first`: none of
        // them was written, so no separator comes before the next one).
        const size_t n = list.size();
        std::vector<std::array<Node, 2>> tail(n + 1);
        tail[n] = {b_.empty(), b_.empty()};
        for (size_t i = n; i-- > 0;) {
            const Member& m = list[i];
            const Node kv = b_.seq({b_.literal(json_literal(Value(m.name))), key_sep(), m.value, after_item()});
            for (int first = 0; first < 2; ++first) {
                const Node written = b_.seq({first ? b_.empty() : item_sep(), kv, tail[i + 1][0]});
                tail[i][static_cast<size_t>(first)] = m.required ? written : b_.alt({written, tail[i + 1][static_cast<size_t>(first)]});
            }
        }
        return b_.seq({b_.literal("{"), ws(), tail[0][1], b_.literal("}")});
    }

    Builder& b_;
    JsonFormat format_;
    Value root_;
    std::map<std::string, uint32_t> refs_;
    std::optional<uint32_t> any_rule_;
    std::optional<Node> ws_, string_char_;
};

}  // namespace

Result<Builder::Node> json_schema(Builder& b, std::string_view schema_json, const JsonFormat& format) {
    auto parsed = jinja::parse_json(schema_json);
    if (!parsed) return Status(ErrorCode::InvalidArgument, "JSON schema: " + parsed.status().message());
    SchemaCompiler compiler(b, format, *parsed);
    auto node = compiler.compile(*parsed, "#");
    if (!node) return node;
    return compiler.root(*node);
}

Result<Builder::Node> raw_string(Builder& b, std::string_view schema_json) {
    auto parsed = jinja::parse_json(schema_json);
    if (!parsed) return Status(ErrorCode::InvalidArgument, "JSON schema: " + parsed.status().message());
    SchemaCompiler compiler(b, JsonFormat{}, *parsed);
    return compiler.raw_string(*parsed, "#");
}

Builder::Node json_value(Builder& b, const JsonFormat& format) {
    SchemaCompiler compiler(b, format, Value::none());
    return compiler.root(compiler.any());
}

}  // namespace liyab::grammar
