// Grammar-constrained decoding: grammars from JSON Schema and grammar text,
// matching byte by byte (UTF-8 split anywhere), token masks over a
// vocabulary trie, and generation with the engine: every output parses and
// validates, EOS only where the grammar accepts, forced + constrained spans,
// determinism, speculative decoding off while a grammar is active.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <unistd.h>

#include "core/chat_template.h"
#include "core/grammar.h"
#include "core/tokenizer.h"
#include "liyab/engine.h"
#include "liyab/mmap_loader.h"
#include "liyab/grammar.h"
#include "liyab/liyab_c_api.h"
#include "test_model.h"
#include "test_util.h"

using namespace liyab;

namespace {

Grammar schema(const char* json, JsonFormat format = {}) {
    auto g = Grammar::from_json_schema(json, format);
    if (!g) {
        std::printf("  schema refused: %s\n", g.status().to_string().c_str());
        std::abort();
    }
    return *g;
}

Grammar text_grammar(const char* text) {
    auto g = Grammar::parse(text);
    if (!g) {
        std::printf("  grammar refused: %s\n", g.status().to_string().c_str());
        std::abort();
    }
    return *g;
}

}  // namespace

TEST_CASE("grammar: literals, classes, repetition, alternatives") {
    const Grammar g = text_grammar(R"(root ::= "ab" [0-9]+ ("x" | "yz")? tail
tail ::= "!"{2,3}  # a comment
)");
    CHECK(g.matches("ab1!!"));
    CHECK(g.matches("ab123x!!!"));
    CHECK(g.matches("ab9yz!!"));
    CHECK(!g.matches("ab!!"));
    CHECK(!g.matches("ab1!"));
    CHECK(!g.matches("ab1!!!!"));
    CHECK(!g.matches("ab1y!!"));
    CHECK(g.accepts_prefix("ab12"));
    CHECK(g.accepts_prefix(""));
    CHECK(!g.accepts_prefix("b"));
}

TEST_CASE("grammar: UTF-8 classes match whole characters, byte by byte") {
    const Grammar g = text_grammar(R"(root ::= [^a]* "|" [α-ω€😀]+)");
    CHECK(g.matches("xyz|αβ"));
    CHECK(g.matches("é漢字😀|€😀ω"));
    CHECK(!g.matches("a|α"));
    CHECK(!g.matches("x|a"));
    // Partial characters are prefixes, never matches; invalid UTF-8 is refused.
    const std::string euro = "\xE2\x82\xAC";
    CHECK(g.accepts_prefix("|" + euro.substr(0, 1)));
    CHECK(g.accepts_prefix("|" + euro.substr(0, 2)));
    CHECK(!g.matches("|" + euro.substr(0, 2)));
    CHECK(g.matches("|" + euro));
    CHECK(!g.accepts_prefix("\xC0\x80"));            // overlong
    CHECK(!g.accepts_prefix("\xED\xA0\x80"));        // surrogate
    CHECK(!g.accepts_prefix("|\xE2\x82\x41"));       // broken continuation
    CHECK(!g.accepts_prefix("\xF4\x90\x80\x80"));    // above U+10FFFF
    const Grammar any = text_grammar("root ::= .*");
    CHECK(any.matches("\x7F\n\xF0\x9F\x98\x80\xEF\xBF\xBF\xF4\x8F\xBF\xBF"));
}

TEST_CASE("grammar: %until, %json, %schema and errors") {
    const Grammar g = text_grammar(R"(root ::= %until "</think>" "\n" %schema {"type": "integer"})");
    CHECK(g.matches("hmm <the</th</think>\n42"));
    CHECK(g.matches("</think>\n-7"));
    CHECK(!g.matches("a</think>b</think>\n1"));  // ends at the first stop
    CHECK(!g.matches("a</think>\n1.5"));
    const Grammar aa = text_grammar(R"(root ::= %until "aab")");
    CHECK(aa.matches("aaab"));
    CHECK(aa.matches("abaab"));
    CHECK(!aa.matches("aabaab"));
    const Grammar j = text_grammar("root ::= %json");
    CHECK(j.matches(R"({"a": [1, 2.5, -3e+2, true, null, "x\"yé"], "b": {}})"));
    CHECK(!j.matches(R"({"a":1})"));  // the default layout: ": " and ", "
    CHECK(!j.matches("01"));

    // %string: a string schema's raw text (no quotes, no escapes).
    const Grammar raw = text_grammar(R"(root ::= "<p>" %string {"type": "string", "maxLength": 4} "</p>" %string {"enum": ["a b", "\"q\""]} "|" %string {"type": "string", "format": "date"})");
    CHECK(raw.matches("<p>a\"\n<</p>a b|2026-10-15"));
    CHECK(raw.matches("<p></p>\"q\"|2026-01-31"));
    CHECK(!raw.matches("<p>abcde</p>a b|2026-10-15"));
    CHECK(!raw.matches("<p>x</p>\"a b\"|2026-10-15"));
    CHECK(!raw.matches("<p>x</p>a b|\"2026-10-15\""));
    CHECK(!Grammar::parse(R"(root ::= %string {"type": "integer"})"));

    auto undefined = Grammar::parse("root ::= missing");
    CHECK(!undefined && undefined.status().message().find("missing") != std::string::npos);
    auto left = Grammar::parse("root ::= root \"a\" | \"b\"");
    CHECK(!left && left.status().message().find("left recursion") != std::string::npos);
    auto empty_loop = Grammar::parse("root ::= (\"a\"?)*");
    CHECK(!empty_loop && empty_loop.status().message().find("left recursion") != std::string::npos);
    auto twice = Grammar::parse("root ::= \"a\"\nroot ::= \"b\"");
    CHECK(!twice);
    auto no_root = Grammar::parse("start ::= \"a\"");
    CHECK(!no_root);
    auto bad = Grammar::parse("root ::= \"a");
    CHECK(!bad && bad.status().message().find("line 1") != std::string::npos);
    auto unsupported = Grammar::parse("root ::= %schema {\"type\": \"string\", \"pattern\": \"a+\"}");
    CHECK(!unsupported && unsupported.status().code() == ErrorCode::Unsupported);
}

TEST_CASE("json schema: objects, required and optional properties in order") {
    const Grammar g = schema(R"({"type": "object", "properties": {
        "name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
        "required": ["name"], "additionalProperties": false})");
    CHECK(g.matches(R"({"name": "Ann"})"));
    CHECK(g.matches(R"({"name": "Ann", "age": 41})"));
    CHECK(g.matches(R"({"name": "Ann", "tags": ["a", "b"]})"));
    CHECK(g.matches(R"({"name": "Ann", "age": -3, "tags": []})"));
    CHECK(!g.matches(R"({"age": 41})"));                    // required missing
    CHECK(!g.matches(R"({"age": 41, "name": "Ann"})"));     // order
    CHECK(!g.matches(R"({"name": "Ann", "extra": 1})"));
    CHECK(!g.matches(R"({"name": "Ann", "age": 4.5})"));
    CHECK(!g.matches(R"({"name":"Ann"})"));                 // layout
    CHECK(!g.matches(R"({"name": "Ann",})"));
    const Grammar loose = schema(R"({"type": "object", "properties": {"a": {"type": "boolean"}}, "required": ["a"]})",
                                 JsonFormat{.any_whitespace = true});
    CHECK(loose.matches("{\"a\":true}"));
    CHECK(loose.matches(" {\n  \"a\" :  false\n}\n"));
    // Optional only: an empty object, or any subset in order.
    const Grammar opt = schema(R"({"properties": {"a": {"type": "null"}, "b": {"type": "null"}, "c": {"type": "null"}}})");
    for (const char* ok : {"{}", R"({"a": null})", R"({"b": null})", R"({"a": null, "c": null})", R"({"a": null, "b": null, "c": null})"}) {
        CHECK(opt.matches(ok));
    }
    CHECK(!opt.matches(R"({, "b": null})"));
    CHECK(!opt.matches(R"({"b": null, "a": null})"));
    // A map without properties; a required name not declared.
    const Grammar map = schema(R"({"type": "object", "additionalProperties": {"type": "integer"}})");
    CHECK(map.matches(R"({"x": 1, "y": 2})"));
    CHECK(map.matches("{}"));
    CHECK(!map.matches(R"({"x": "1"})"));
    const Grammar req = schema(R"({"type": "object", "required": ["id"]})");
    CHECK(req.matches(R"({"id": [1, {"k": null}]})"));
    CHECK(!req.matches("{}"));
    CHECK(schema(R"({"type": "object", "additionalProperties": false})").matches("{}"));
}

TEST_CASE("json schema: strings, formats, enums, numbers, arrays, refs") {
    const Grammar s = schema(R"({"type": "string", "minLength": 2, "maxLength": 3})");
    CHECK(s.matches(R"("ab")"));
    CHECK(s.matches(R"("a\n")"));   // an escape counts as one character
    CHECK(s.matches(R"("€😀x")"));
    CHECK(!s.matches(R"("a")"));
    CHECK(!s.matches(R"("abcd")"));
    CHECK(!s.matches("\"a\nb\""));  // raw control characters are not JSON
    const Grammar date = schema(R"({"type": "string", "format": "date-time"})");
    CHECK(date.matches(R"("2026-10-15T09:30:00Z")"));
    CHECK(date.matches(R"("2026-02-28T23:59:60.123+02:00")"));
    CHECK(!date.matches(R"("2026-13-01T00:00:00Z")"));
    CHECK(!date.matches(R"("2026-10-15 09:30:00Z")"));
    CHECK(schema(R"({"format": "uuid"})").matches(R"("123e4567-e89b-12d3-a456-426614174000")"));
    const Grammar e = schema(R"({"enum": ["red", "green", 3, null, {"a": [1]}]})");
    CHECK(e.matches(R"("red")"));
    CHECK(e.matches("3"));
    CHECK(e.matches("null"));
    CHECK(e.matches(R"({"a": [1]})"));
    CHECK(!e.matches(R"("blue")"));
    CHECK(schema(R"({"type": "string", "enum": ["a", 1]})").matches(R"("a")"));
    CHECK(!schema(R"({"type": "string", "enum": ["a", 1]})").matches("1"));
    CHECK(schema(R"({"const": "fixed"})").matches(R"("fixed")"));

    const Grammar n = schema(R"({"type": "number"})");
    for (const char* ok : {"0", "-0", "12", "-3.25", "1e5", "6.02E+23", "1.5e-3"}) CHECK(n.matches(ok));
    for (const char* bad : {"01", "+1", ".5", "1.", "1e", "NaN", "--1", "12345678901234567890"}) CHECK(!n.matches(bad));
    const Grammar r = schema(R"({"type": "integer", "minimum": -15, "maximum": 237})");
    for (int v = -40; v <= 300; ++v) CHECK(r.matches(std::to_string(v)) == (v >= -15 && v <= 237));
    CHECK(!r.matches("-0"));
    CHECK(!r.matches("007"));
    const Grammar ex = schema(R"({"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 1000})");
    for (int v = -5; v <= 1005; ++v) CHECK(ex.matches(std::to_string(v)) == (v > 0 && v < 1000));
    const Grammar big = schema(R"({"type": "integer", "minimum": 999999999990})");
    CHECK(big.matches("999999999990"));
    CHECK(big.matches("1000000000000000"));
    CHECK(!big.matches("999999999989"));
    const Grammar real = schema(R"({"type": "number", "minimum": -90, "maximum": 90})");
    for (const char* ok : {"-90", "90", "0", "-0.5", "89.999", "-89.5", "45"}) CHECK(real.matches(ok));
    for (const char* bad : {"90.5", "-90.1", "91", "-100", "1e2"}) CHECK(!real.matches(bad));
    const Grammar pos = schema(R"({"type": "number", "exclusiveMinimum": 0})");
    CHECK(pos.matches("0.5"));
    CHECK(pos.matches("3"));
    CHECK(!pos.matches("0"));
    CHECK(!pos.matches("0.0"));
    CHECK(!pos.matches("-1"));

    const Grammar a = schema(R"({"type": "array", "items": {"type": "boolean"}, "minItems": 1, "maxItems": 2})");
    CHECK(a.matches("[true]"));
    CHECK(a.matches("[true, false]"));
    CHECK(!a.matches("[]"));
    CHECK(!a.matches("[true, false, true]"));
    const Grammar multi = schema(R"({"type": ["string", "null"]})");
    CHECK(multi.matches("null"));
    CHECK(multi.matches(R"("x")"));
    CHECK(!multi.matches("1"));
    const Grammar any_of = schema(R"({"anyOf": [{"type": "integer"}, {"type": "object", "properties": {"k": {"const": 1}}, "required": ["k"]}]})");
    CHECK(any_of.matches("5"));
    CHECK(any_of.matches(R"({"k": 1})"));
    CHECK(!any_of.matches(R"({"k": 2})"));
    // Recursive $ref through $defs.
    const Grammar tree = schema(R"({"$ref": "#/$defs/node", "$defs": {"node": {"type": "object",
        "properties": {"v": {"type": "integer"}, "kids": {"type": "array", "items": {"$ref": "#/$defs/node"}}},
        "required": ["v"]}}})");
    CHECK(tree.matches(R"({"v": 1, "kids": [{"v": 2}, {"v": 3, "kids": [{"v": 4, "kids": []}]}]})"));
    CHECK(!tree.matches(R"({"v": 1, "kids": [{"w": 2}]})"));
    const Grammar nullable = schema(R"({"type": "string", "nullable": true})");
    CHECK(nullable.matches("null"));
    CHECK(schema("true").matches("[1, \"a\"]"));
    CHECK(schema("{}").matches("false"));
}

TEST_CASE("json schema: refused schemas say why") {
    struct Case {
        const char* schema;
        ErrorCode code;
        const char* says;
    };
    const Case cases[] = {
        {R"({"type": "string", "pattern": "^a$"})", ErrorCode::Unsupported, "pattern"},
        {R"({"type": "array", "uniqueItems": true})", ErrorCode::Unsupported, "uniqueItems"},
        {R"({"not": {"type": "string"}})", ErrorCode::Unsupported, "not"},
        {R"({"type": "number", "multipleOf": 2})", ErrorCode::Unsupported, "multipleOf"},
        {R"({"type": "number", "minimum": 0.5})", ErrorCode::Unsupported, "minimum"},
        {R"({"allOf": [{"type": "string"}, {"maxLength": 3}]})", ErrorCode::Unsupported, "allOf"},
        {R"({"type": "object", "properties": {"a": {"pattern": "x"}}})", ErrorCode::Unsupported, "#/properties/a"},
        {R"({"$ref": "https://example.com/s.json"})", ErrorCode::Unsupported, "$ref"},
        {R"({"$ref": "#/$defs/missing"})", ErrorCode::InvalidArgument, "points nowhere"},
        {R"({"type": "strin"})", ErrorCode::InvalidArgument, "unknown type"},
        {R"({"type": "integer", "minimum": 5, "maximum": 1})", ErrorCode::InvalidArgument, "bounds"},
        {R"({"type": "object", "required": ["x"], "additionalProperties": false})", ErrorCode::InvalidArgument, "'x'"},
        {R"({"enum": [1, 2], "type": "string"})", ErrorCode::InvalidArgument, "enum"},
        {"false", ErrorCode::InvalidArgument, "matches no value"},
        {"{\"type\": ", ErrorCode::InvalidArgument, "JSON"},
        {R"({"type": "string", "maxLength": 20000})", ErrorCode::Unsupported, "maxLength"},
    };
    for (const Case& c : cases) {
        auto g = Grammar::from_json_schema(c.schema);
        CHECK(!g);
        if (g) continue;
        CHECK(g.status().code() == c.code);
        if (g.status().message().find(c.says) == std::string::npos) {
            std::printf("  %s -> %s\n", c.schema, g.status().message().c_str());
            CHECK(false);
        }
    }
    // Annotations and unknown keywords are ignored, as JSON Schema does.
    CHECK(Grammar::from_json_schema(R"({"type": "string", "description": "d", "title": "t", "x-custom": 1, "format": "email"})"));
}

// The grammars the app's local API writes for a tool call after
// `<tool_call>` (ToolDialect.callGrammar; app/test/api_test.dart expects
// these exact texts), for Hermes JSON calls and Qwen XML calls.
TEST_CASE("grammar: the app's tool-call grammars accept valid calls only") {
    const Grammar hermes = text_grammar(R"(root ::= "\n" "{\"name\": " (t0 | t1) "}\n</tool_call>"
t0 ::= "\"get_weather\", \"arguments\": " %schema {"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"days":{"type":"integer","minimum":1,"maximum":14}},"required":["city"]}
t1 ::= "\"now\", \"arguments\": " "{}")");
    CHECK(hermes.matches("\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 3}}\n</tool_call>"));
    CHECK(hermes.matches("\n{\"name\": \"now\", \"arguments\": {}}\n</tool_call>"));
    CHECK(!hermes.matches("\n{\"name\": \"get_weather\", \"arguments\": {\"days\": 3}}\n</tool_call>"));
    CHECK(!hermes.matches("\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\", \"days\": 30}}\n</tool_call>"));
    CHECK(!hermes.matches("\n{\"name\": \"get_time\", \"arguments\": {}}\n</tool_call>"));
    grammar::StateTable t(hermes.program());
    CHECK(t.forced_text(t.initial()) == "\n{\"name\": \"");
    const int32_t after_name = t.advance(t.initial(), "\n{\"name\": \"n");
    CHECK(t.forced_text(after_name) == "ow\", \"arguments\": {}}\n</tool_call>");

    const Grammar xml = text_grammar(R"(root ::= "<function=" (t0 | t1) "</function>\n</tool_call>"
p0-0 ::= "<parameter=city>\n" %until "\n</parameter>" "\n"
p0-1 ::= "<parameter=unit>\n" %string {"enum":["celsius","fahrenheit"]} "\n</parameter>\n"
p0-2 ::= "<parameter=days>\n" %schema {"type":"integer","minimum":1,"maximum":14} "\n</parameter>\n"
t0 ::= "get_weather>\n" p0-0 p0-1? p0-2?
t1 ::= "now>\n" "")");
    CHECK(xml.matches("<function=get_weather>\n<parameter=city>\nSão Paulo\nor so\n</parameter>\n<parameter=days>\n7\n</parameter>\n"
                      "</function>\n</tool_call>"));
    CHECK(xml.matches("<function=now>\n</function>\n</tool_call>"));
    CHECK(!xml.matches("<function=get_weather>\n<parameter=days>\n7\n</parameter>\n</function>\n</tool_call>"));
    CHECK(!xml.matches("<function=get_weather>\n<parameter=city>\nRome\n</parameter>\n<parameter=unit>\nkelvin\n</parameter>\n"
                       "</function>\n</tool_call>"));
}

TEST_CASE("grammar: a table past its bound starts over and keeps following the input") {
    // Each position of a bounded string is its own state: 300 positions, a table of at most 40.
    const Grammar g = schema(R"({"type": "string", "maxLength": 300})");
    grammar::StateTable t(g.program(), 40);
    int32_t s = t.initial();
    size_t most = 0;
    std::string text = "\"";
    s = t.advance(s, "\"");
    for (int i = 0; i < 300; ++i) {
        const std::string c = i % 7 == 0 ? "\xC3\xA9" : "x";  // é split byte by byte
        for (const char b : c) {
            s = t.advance(s, std::string_view(&b, 1));
            REQUIRE(s >= 0);
            s = t.compact(s);
            most = std::max(most, t.size());
        }
        text += c;
    }
    CHECK(most <= 41);
    CHECK(t.advance(s, "x") < 0);  // the 301st character is refused
    s = t.advance(s, "\"");
    REQUIRE(s >= 0);
    CHECK(t.accepting(s) && !t.extendable(s));
    CHECK(g.matches(text + "\""));
}

TEST_CASE("grammar: forced text stops at a choice and at partial characters") {
    const Grammar g = schema(R"({"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["°C", "°F"]}},
        "required": ["city", "unit"]})");
    grammar::StateTable t(g.program());
    int32_t s = t.initial();
    CHECK(t.forced_text(s) == "{\"city\": \"");
    s = t.advance(s, "{\"city\": \"Rome\"");
    REQUIRE(s >= 0);
    // "°" is forced whole; the next byte differs (C/F), so forcing stops there.
    CHECK(t.forced_text(s) == ", \"unit\": \"°");
    s = t.advance(s, ", \"unit\": \"\xC2");
    REQUIRE(s >= 0);
    CHECK(t.forced_text(s) == "\xB0");  // the rest of a character the output already started
    s = t.advance(s, "\xB0" "F\"");
    REQUIRE(s >= 0);
    CHECK(t.forced_text(s) == "}");
    s = t.advance(s, "}");
    CHECK(t.accepting(s) && !t.extendable(s));
    // Two literals sharing a first byte of a multi-byte character: nothing is forced.
    const Grammar split = text_grammar(R"(root ::= "é" | "è")");
    grammar::StateTable t2(split.program());
    CHECK(t2.forced_text(t2.initial()).empty());
}

// ---------------------------------------------------------------------------
// With a vocabulary and the engine
// ---------------------------------------------------------------------------

namespace {

std::string temp_path(const std::string& name) {
    return test::temp_dir() + "/liyab_" + std::to_string(getpid()) + "_grammar_" + name;
}

// The tiny model (SentencePiece: byte tokens, letters, pieces of "▁hello"
// and "▁world"), written once per process and removed at exit.
const std::string& tiny_model(bool hybrid = false) {
    struct Written {
        std::string plain, hybrid;
        ~Written() {
            if (!plain.empty()) std::remove(plain.c_str());
            if (!hybrid.empty()) std::remove(hybrid.c_str());
        }
    };
    static Written w;
    std::string& path = hybrid ? w.hybrid : w.plain;
    if (path.empty()) {
        test::TinyModelSpec spec;
        if (hybrid) {
            spec.arch = "qwen35";
            spec.n_layers = 4;
            spec.delta_net_interval = 2;
        }
        path = temp_path(hybrid ? "hybrid.gguf" : "plain.gguf");
        test::write_tiny_model(path, spec);
    }
    return path;
}

EngineConfig config_for(const std::string& path) {
    EngineConfig c;
    c.model_path = path;
    c.n_threads = 4;
    c.thermal_polling = false;
    c.backend = BackendKind::Cpu;
    c.power.profile = PowerProfile::Performance;
    return c;
}

SamplingParams sampling(float temperature, uint64_t seed, int32_t max_tokens) {
    SamplingParams p;
    p.temperature = temperature;
    p.top_k = 0;
    p.top_p = 1.0f;
    p.seed = seed;
    p.max_tokens = max_tokens;
    return p;
}

struct Output {
    std::string text;
    GenerationStats stats;
};

Output generate(Engine& e, const std::string& prompt, const SamplingParams& p, const GenerateOptions& options) {
    Output out;
    auto st = e.generate(prompt, p, [&](std::string_view piece, int32_t) {
        out.text += piece;
        return true;
    }, options);
    if (!st) {
        std::printf("  generate failed: %s\n", st.status().to_string().c_str());
        std::abort();
    }
    out.stats = *st;
    return out;
}

// An independent check of a value against the schema subset the tests use
// (not the grammar: the grammar is what is under test).
bool valid(const jinja::Value& v, const jinja::Value& s) {
    using T = jinja::Value::Type;
    if (!s.is_object()) return true;
    const jinja::Object& o = s.fields();
    if (const jinja::Value* e = o.find("enum")) {
        bool any = false;
        for (const jinja::Value& x : e->items()) any = any || x.to_json() == v.to_json();
        if (!any) return false;
    }
    if (const jinja::Value* c = o.find("const"); c != nullptr && c->to_json() != v.to_json()) return false;
    if (const jinja::Value* any = o.find("anyOf")) {
        bool ok = false;
        for (const jinja::Value& b : any->items()) ok = ok || valid(v, b);
        if (!ok) return false;
    }
    const jinja::Value* type = o.find("type");
    if (type == nullptr) return true;
    const std::string& t = type->as_string();
    if (t == "null") return v.is_none();
    if (t == "boolean") return v.type() == T::Bool;
    if (t == "integer" || t == "number") {
        if (t == "integer" ? v.type() != T::Int : (v.type() != T::Int && v.type() != T::Float)) return false;
        if (const jinja::Value* m = o.find("minimum"); m != nullptr && v.as_float() < m->as_float()) return false;
        if (const jinja::Value* m = o.find("maximum"); m != nullptr && v.as_float() > m->as_float()) return false;
        return true;
    }
    if (t == "string") {
        if (!v.is_string()) return false;
        size_t chars = 0;
        for (const char c : v.as_string()) chars += (static_cast<unsigned char>(c) & 0xC0) != 0x80;
        if (const jinja::Value* m = o.find("maxLength"); m != nullptr && chars > static_cast<size_t>(m->as_int())) return false;
        if (const jinja::Value* m = o.find("minLength"); m != nullptr && chars < static_cast<size_t>(m->as_int())) return false;
        return true;
    }
    if (t == "array") {
        if (!v.is_array()) return false;
        if (const jinja::Value* m = o.find("maxItems"); m != nullptr && v.items().size() > static_cast<size_t>(m->as_int())) return false;
        if (const jinja::Value* m = o.find("minItems"); m != nullptr && v.items().size() < static_cast<size_t>(m->as_int())) return false;
        if (const jinja::Value* items = o.find("items")) {
            for (const jinja::Value& x : v.items()) {
                if (!valid(x, *items)) return false;
            }
        }
        return true;
    }
    if (!v.is_object()) return false;
    const jinja::Value* props = o.find("properties");
    if (const jinja::Value* req = o.find("required")) {
        for (const jinja::Value& name : req->items()) {
            if (v.fields().find(name.as_string()) == nullptr) return false;
        }
    }
    for (const auto& [key, value] : v.fields().entries) {
        const jinja::Value* ps = props != nullptr ? props->fields().find(key) : nullptr;
        if (ps == nullptr) {
            const jinja::Value* add = o.find("additionalProperties");
            if (add != nullptr && add->type() == T::Bool && !add->as_bool()) return false;
            continue;
        }
        if (!valid(value, *ps)) return false;
    }
    return true;
}

const char* const kToolSchema = R"({"type": "object", "properties": {
    "city": {"type": "string", "maxLength": 12},
    "days": {"type": "integer", "minimum": 1, "maximum": 14},
    "unit": {"enum": ["celsius", "fahrenheit"]},
    "hourly": {"type": "boolean"},
    "tags": {"type": "array", "items": {"type": "string", "maxLength": 4}, "maxItems": 3},
    "note": {"anyOf": [{"type": "null"}, {"type": "string", "maxLength": 6}]}},
    "required": ["city", "unit"], "additionalProperties": false})";

}  // namespace

TEST_CASE("masks: the vocabulary trie, split UTF-8, EOS only where the grammar accepts") {
    auto file = MmapLoader::open(tiny_model());
    REQUIRE(file.has_value());
    auto tok = Tokenizer::load(*file.value());
    REQUIRE(tok.has_value());
    auto trie = grammar::VocabTrie::build(*tok);
    const int32_t eos = tok->eos();
    const Grammar g = text_grammar(R"(root ::= "é"+ "!"?)");
    grammar::StateTable t(g.program());
    auto allowed = [&](int32_t state) {
        std::set<std::string> pieces;
        const std::vector<uint64_t>& bits = t.mask(state, *trie);
        for (int32_t id = 0; id < tok->vocab_size(); ++id) {
            if ((bits[static_cast<size_t>(id) >> 6] >> (id & 63)) & 1) pieces.insert(id == eos ? "<eos>" : tok->piece(id));
        }
        return pieces;
    };
    // Only the byte tokens can spell "é" in this vocabulary: first C3, then A9.
    int32_t s = t.initial();
    CHECK(allowed(s) == std::set<std::string>{"\xC3"});
    s = t.advance(s, "\xC3");
    CHECK(allowed(s) == std::set<std::string>{"\xA9"});
    s = t.advance(s, "\xA9");
    CHECK(allowed(s) == (std::set<std::string>{"\xC3", "!", "<eos>"}));
    s = t.advance(s, "!");
    CHECK(allowed(s) == std::set<std::string>{"<eos>"});
    // Longer pieces: every token whose whole piece fits.
    const Grammar w = text_grammar(R"(root ::= " world" | " wo" [a-z] "x")");
    grammar::StateTable tw(w.program());
    const std::vector<uint64_t>& bits = tw.mask(tw.initial(), *trie);
    for (const char* piece : {"\xE2\x96\x81world", "\xE2\x96\x81wor", "\xE2\x96\x81w", "\xE2\x96\x81"}) {
        const int32_t id = test::tiny_token(piece);
        REQUIRE(id >= 0);
        CHECK(((bits[static_cast<size_t>(id) >> 6] >> (id & 63)) & 1) == 1);
    }
    const int32_t hello = test::tiny_token("\xE2\x96\x81hello");
    CHECK(((bits[static_cast<size_t>(hello) >> 6] >> (hello & 63)) & 1) == 0);
    // apply_mask leaves allowed logits as they were and the rest at -inf.
    std::vector<float> logits(static_cast<size_t>(tok->vocab_size()) + 3, 1.5f);
    const size_t n = tw.apply_mask(tw.initial(), *trie, logits);
    size_t finite = 0;
    for (const float l : logits) finite += std::isfinite(l) ? 1 : 0;
    CHECK(n == finite);
    CHECK(std::isinf(logits.back()));  // beyond the vocabulary
}

TEST_CASE("constrained generation: every sampled output parses and validates") {
    auto engine = Engine::create(config_for(tiny_model()));
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    auto schema_value = jinja::parse_json(kToolSchema);
    REQUIRE(schema_value.has_value());
    const Grammar tool = schema(kToolSchema);
    const Grammar loose = schema(kToolSchema, JsonFormat{.any_whitespace = true});
    int complete = 0;
    for (uint64_t seed = 1; seed <= 24; ++seed) {
        const Grammar& g = seed % 3 == 0 ? loose : tool;
        // Hot sampling over random weights: the model alone would write noise.
        const Output out = generate(e, "hello world", sampling(1.5f, seed, 300), GenerateOptions{.grammar = g});
        if (out.stats.generated_tokens >= 300) {  // cut by max_tokens: still a prefix of a match
            CHECK(g.accepts_prefix(out.text));
            continue;
        }
        ++complete;
        CHECK(g.matches(out.text));
        auto parsed = jinja::parse_json(out.text);
        if (!parsed) {
            std::printf("  not JSON: %s\n", out.text.c_str());
            CHECK(false);
            continue;
        }
        if (!valid(*parsed, *schema_value)) {
            std::printf("  invalid: %s\n", out.text.c_str());
            CHECK(false);
        }
        CHECK(out.stats.constrained_tokens > 0);
        CHECK(out.stats.forced_tokens > 0);  // keys and separators
    }
    CHECK(complete >= 20);
    // Free-form JSON and a grammar text, greedy and sampled.
    const Grammar json = Grammar::json();
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        const Output out = generate(e, "hello", sampling(seed == 1 ? 0.0f : 1.2f, seed, 200), GenerateOptions{.grammar = json});
        CHECK(out.stats.generated_tokens >= 200 ? json.accepts_prefix(out.text) : json.matches(out.text));
    }
    const Grammar words = text_grammar(R"(root ::= ("hello" | "world") (" " [a-z]{1,8}){2,4} ".")");
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        const Output out = generate(e, "hello world", sampling(1.0f, seed, 100), GenerateOptions{.grammar = words});
        CHECK(words.matches(out.text));
    }
}

TEST_CASE("constrained generation: the output ends where the grammar does, EOS only when it accepts") {
    auto engine = Engine::create(config_for(tiny_model()));
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    // Fully determined: forced in one pass, nothing sampled, no EOS needed.
    const Output fixed = generate(e, "hello", sampling(0.0f, 1, 50), GenerateOptions{.grammar = text_grammar(R"(root ::= "{\"ok\": true}")")});
    CHECK(fixed.text == R"({"ok": true})");
    CHECK(fixed.stats.generated_tokens == 0);
    CHECK(fixed.stats.forced_tokens > 0);
    // Open-ended: ends with an end-of-generation token only where the grammar
    // accepts, else at max_tokens; never in the middle of a character.
    const Grammar open = text_grammar(R"(root ::= [a-z]+ ("é" [a-z]*)*)");
    int ended = 0;
    for (uint64_t seed = 1; seed <= 16; ++seed) {
        const Output out = generate(e, "hello world", sampling(1.0f, seed, 40), GenerateOptions{.grammar = open});
        if (out.stats.generated_tokens < 40) {
            ++ended;
            CHECK(open.matches(out.text));
        } else {
            CHECK(open.accepts_prefix(out.text));
        }
    }
    std::printf("  %d of 16 ended with EOS before max_tokens\n", ended);
    // Errors: both options at once.
    GenerateOptions both{.grammar = open, .continuation = [](std::string_view) { return Continuation{}; }};
    auto st = e.generate("hello", sampling(0.0f, 1, 4), nullptr, both);
    CHECK(!st && st.status().code() == ErrorCode::InvalidArgument);
}

TEST_CASE("constrained generation: forced text and constrained spans, any decoding, same output") {
    const Grammar args = schema(kToolSchema);
    auto schema_value = jinja::parse_json(kToolSchema);
    REQUIRE(schema_value.has_value());
    // A tool call: "<call>" forced after the first word, the arguments
    // constrained, "</call>" forced when they close; the text after it is free.
    auto callback = [&](int& calls) {
        return [&args, &calls](std::string_view generated) {
            ++calls;
            Continuation c;
            if (generated.ends_with("</call>") || generated.find("<call>") != std::string_view::npos) {
                if (generated.find("</call>") == std::string_view::npos) c.text = "</call>";
                return c;
            }
            if (generated.size() >= 3) {
                c.text = " <call>";
                c.grammar = args;
            }
            return c;
        };
    };
    for (const bool hybrid : {false, true}) {
        const std::string& path = tiny_model(hybrid);
        EngineConfig plain = config_for(path);
        EngineConfig lookup = plain;
        lookup.lookup_drafts = true;
        lookup.adaptive_drafts = false;
        EngineConfig drafted = plain;
        drafted.draft_model_path = path;
        drafted.adaptive_drafts = false;
        std::string reference;
        for (const EngineConfig* config : {&plain, &lookup, &drafted}) {
            auto engine = Engine::create(*config);
            REQUIRE(engine.has_value());
            int calls = 0;
            const Output out = generate(*engine.value(), "hello world hello world", sampling(0.0f, 1, 120),
                                        GenerateOptions{.continuation = callback(calls)});
            const size_t open = out.text.find(" <call>");
            const size_t close = out.text.find("</call>");
            REQUIRE(open != std::string::npos);
            CHECK(close != std::string::npos);  // every value is bounded: the call ends within max_tokens
            if (close == std::string::npos) {
                CHECK(args.accepts_prefix(out.text.substr(open + 7)));
            } else {
                const std::string call = out.text.substr(open + 7, close - open - 7);
                CHECK(args.matches(call));
                auto parsed = jinja::parse_json(call);
                CHECK(parsed.has_value() && valid(*parsed, *schema_value));
            }
            CHECK(out.stats.constrained_tokens > 0);
            CHECK(calls > 0);
            // Greedy: the same text with plain, lookup and draft-model decoding.
            if (config == &plain) {
                reference = out.text;
            } else {
                CHECK(out.text == reference);
                if (out.text != reference) std::printf("  %s\n  %s\n", reference.c_str(), out.text.c_str());
            }
        }
    }
}

TEST_CASE("constrained generation: deterministic, and unconstrained generation unchanged") {
    auto engine = Engine::create(config_for(tiny_model()));
    REQUIRE(engine.has_value());
    Engine& e = *engine.value();
    const Grammar g = schema(kToolSchema);
    for (uint64_t seed : {3u, 9u}) {
        const Output a = generate(e, "hello world", sampling(1.0f, seed, 200), GenerateOptions{.grammar = g});
        const Output b = generate(e, "hello world", sampling(1.0f, seed, 200), GenerateOptions{.grammar = g});
        CHECK(a.text == b.text);
        (void)e.trim_memory();  // drops the trie and the grammar's states: rebuilt, same output
        CHECK(generate(e, "hello world", sampling(1.0f, seed, 200), GenerateOptions{.grammar = g}).text == a.text);
        auto other = Engine::create(config_for(tiny_model()));
        REQUIRE(other.has_value());
        const Output c = generate(*other.value(), "hello world", sampling(1.0f, seed, 200), GenerateOptions{.grammar = g});
        CHECK(a.text == c.text);  // a fresh engine: no cached state changes the output
    }
    // Empty options: the plain path, token for token.
    std::string plain, with_options;
    REQUIRE(e.generate("hello world", sampling(0.9f, 5, 40), [&](std::string_view p, int32_t) {
        plain += p;
        return true;
    }).has_value());
    with_options = generate(e, "hello world", sampling(0.9f, 5, 40), GenerateOptions{}).text;
    CHECK(plain == with_options);
}

TEST_CASE("C API: grammars, a constrained whole output and constrained spans") {
    liyab_grammar* bad = nullptr;
    CHECK(liyab_grammar_from_json_schema(R"({"type": "string", "pattern": "x"})", 0, &bad) == LIYAB_ERR_UNSUPPORTED);
    CHECK(bad == nullptr && std::strstr(liyab_last_error(), "pattern") != nullptr);
    liyab_grammar* args = nullptr;
    REQUIRE(liyab_grammar_from_json_schema(kToolSchema, 0, &args) == LIYAB_OK);
    liyab_grammar* text = nullptr;
    REQUIRE(liyab_grammar_parse("root ::= \"<\" [a-z]{2,5} \">\"", &text) == LIYAB_OK);
    CHECK(liyab_grammar_matches(text, "<abc>", 5) == 1);
    CHECK(liyab_grammar_matches(text, "<a>", 3) == 0);

    liyab_engine_config config;
    liyab_engine_config_default(&config);
    config.model_path = tiny_model().c_str();
    config.n_threads = 4;
    config.thermal_polling = 0;
    liyab_engine* engine = nullptr;
    REQUIRE(liyab_engine_create(&config, &engine) == LIYAB_OK);
    liyab_sampling_params params;
    liyab_sampling_params_default(&params);
    params.temperature = 1.0f;
    params.seed = 3;
    params.max_tokens = 200;
    struct Ctx {
        std::string out;
        liyab_grammar* args;
        int calls = 0;
    } ctx{{}, args};
    auto on_piece = [](const char* piece, size_t len, int32_t, void* user) -> int32_t {
        static_cast<Ctx*>(user)->out.append(piece, len);
        return 1;
    };
    liyab_generation_stats stats;
    liyab_grammar_stats gstats;
    REQUIRE(liyab_engine_generate_structured(engine, "hello", &params, on_piece, args, nullptr, &ctx, &stats, &gstats) ==
            LIYAB_OK);
    CHECK(stats.generated_tokens >= params.max_tokens || liyab_grammar_matches(args, ctx.out.data(), ctx.out.size()) == 1);
    CHECK(gstats.constrained_tokens == stats.generated_tokens);

    // A span after forced text, then forced text again when it completes.
    ctx.out.clear();
    auto continuation = [](const char* generated, size_t len, char* out, size_t cap, const liyab_grammar** grammar,
                           void* user) -> size_t {
        Ctx& c = *static_cast<Ctx*>(user);
        ++c.calls;
        const std::string_view so_far(generated, len);
        const char* text = nullptr;
        if (so_far.empty()) {
            text = "call ";
            *grammar = c.args;
        } else if (so_far.find(" end") == std::string_view::npos) {
            text = " end";
        }
        if (text == nullptr || std::strlen(text) > cap) return 0;
        std::memcpy(out, text, std::strlen(text));
        return std::strlen(text);
    };
    params.temperature = 0.0f;
    REQUIRE(liyab_engine_generate_structured(engine, "hello", &params, on_piece, nullptr, continuation, &ctx, &stats,
                                             nullptr) == LIYAB_OK);
    REQUIRE(ctx.out.starts_with("call "));
    const size_t end = ctx.out.find(" end");
    REQUIRE(end != std::string::npos);
    CHECK(liyab_grammar_matches(args, ctx.out.data() + 5, end - 5) == 1);
    CHECK(ctx.calls >= 2);
    // Both at once: refused.
    CHECK(liyab_engine_generate_structured(engine, "hello", &params, on_piece, args, continuation, &ctx, nullptr,
                                           nullptr) == LIYAB_ERR_INVALID_ARGUMENT);
    liyab_engine_destroy(engine);
    liyab_grammar_destroy(args);
    liyab_grammar_destroy(text);
}

// Real models (LIYAB_GRAMMAR_MODEL=path.gguf[,path2.gguf...]): every output
// of a tool call's arguments validates, and the cost of the masks per token
// against a decode step.
TEST_CASE("real models: tool-call arguments validate; masking cost per token") {
    const char* env = std::getenv("LIYAB_GRAMMAR_MODEL");
    if (env == nullptr) {
        std::printf("  skipped: set LIYAB_GRAMMAR_MODEL to GGUF paths (comma-separated)\n");
        return;
    }
    const char* const schemas[] = {
        kToolSchema,
        R"({"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "since": {"type": "string", "format": "date"}}, "required": ["query"]})",
        R"({"type": "object", "properties": {"title": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "object",
            "properties": {"name": {"type": "string"}, "email": {"type": "string"}}, "required": ["name"]}},
            "start": {"type": "string", "format": "date-time"}, "minutes": {"type": "number", "minimum": 0}},
            "required": ["title", "start"]})",
    };
    std::string paths = env;
    for (size_t at = 0; at < paths.size();) {
        const size_t comma = paths.find(',', at);
        const std::string path = paths.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
        at = comma == std::string::npos ? paths.size() : comma + 1;
        {
            // One mask from scratch (every byte transition computed too), in
            // the states that allow the most tokens: inside a string, in free
            // text, and at a structural choice.
            auto file = MmapLoader::open(path);
            REQUIRE(file.has_value());
            auto tok = Tokenizer::load(*file.value());
            REQUIRE(tok.has_value());
            const auto t0 = std::chrono::steady_clock::now();
            auto trie = grammar::VocabTrie::build(*tok);
            const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            struct Probe {
                const char* what;
                const char* grammar;
                const char* prefix;
            };
            const Probe probes[] = {
                {"inside a JSON string", "root ::= %schema {\"type\": \"string\"}", "\"ab"},
                {"free text (%until)", "root ::= %until \"</think>\"", "hm"},
                {"a choice of JSON values", "root ::= %json", ""},
            };
            for (const Probe& p : probes) {
                const Grammar g = text_grammar(p.grammar);
                double first = 0, again = 0;
                size_t allowed = 0;
                for (int rep = 0; rep < 3; ++rep) {
                    grammar::StateTable t(g.program());
                    const int32_t s = t.advance(t.initial(), p.prefix);
                    const auto a = std::chrono::steady_clock::now();
                    const std::vector<uint64_t>& bits = t.mask(s, *trie);
                    const auto b = std::chrono::steady_clock::now();
                    std::vector<float> logits(static_cast<size_t>(tok->vocab_size()), 0.0f);
                    const auto c = std::chrono::steady_clock::now();
                    allowed = t.apply_mask(s, *trie, logits);
                    const auto d = std::chrono::steady_clock::now();
                    (void)bits;
                    if (rep > 0) {  // the first repetition warms the caches
                        first += std::chrono::duration<double, std::milli>(b - a).count() / 2;
                        again += std::chrono::duration<double, std::milli>(d - c).count() / 2;
                    }
                }
                std::printf("  %s: %s mask %.2f ms from scratch, %.3f ms cached (%zu tokens allowed, trie %zu nodes, built in %.0f ms)\n",
                            path.c_str() + path.rfind('/') + 1, p.what, first, again, allowed, trie->nodes.size(), build_ms);
            }
        }
        EngineConfig config = config_for(path);
        config.n_threads = 0;
        config.backend.reset();
        double trie_ms = 0;
        // Per sampling mode: greedy, top-k sampling as models recommend
        // (checks the likeliest tokens), and top_k 0 (whole masks).
        struct Mode {
            const char* name;
            float temperature;
            int32_t top_k;
            float top_p;
        };
        const Mode modes[] = {{"greedy", 0.0f, 0, 1.0f}, {"top-k 20", 0.7f, 20, 0.8f}, {"top_k 0", 0.7f, 0, 0.8f}};
        int valid_count = 0, runs = 0;
        double decode_ms = 0;
        int32_t sampled = 0;
        for (const Mode& mode : modes) {
            // A new engine per mode, so a schema's first use is a first use.
            auto engine = Engine::create(config);
            REQUIRE(engine.has_value());
            Engine& e = *engine.value();
            const auto t_trie = std::chrono::steady_clock::now();
            // The first constrained token builds the vocabulary trie.
            (void)generate(e, "Hi", sampling(0.0f, 1, 2), GenerateOptions{.grammar = text_grammar("root ::= [a-z]+")});
            trie_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_trie).count();
            double first_ms = 0, again_ms = 0;
            int32_t first_tokens = 0, again_tokens = 0;
            for (const char* text : schemas) {
                auto schema_value = jinja::parse_json(text);
                REQUIRE(schema_value.has_value());
                const Grammar g = schema(text);
                for (const bool again : {false, true}) {  // a grammar's first use, then the same request again
                    const std::string prompt = std::string(
                        "<|im_start|>user\nCall the tool for a trip to Lisbon next week. Arguments as JSON matching ") +
                        text + "<|im_end|>\n<|im_start|>assistant\n";
                    SamplingParams p = sampling(mode.temperature, 7, 160);
                    p.top_k = mode.top_k;
                    p.top_p = mode.top_p;
                    const Output out = generate(e, prompt, p, GenerateOptions{.grammar = g});
                    ++runs;
                    auto parsed = jinja::parse_json(out.text);
                    const bool ok = out.stats.generated_tokens < 160 && parsed.has_value() && valid(*parsed, *schema_value);
                    valid_count += ok ? 1 : 0;
                    if (!ok) std::printf("  invalid: %s\n", out.text.c_str());
                    (again ? again_ms : first_ms) += out.stats.grammar_ms;
                    (again ? again_tokens : first_tokens) += out.stats.constrained_tokens;
                    decode_ms += out.stats.decode_ms - out.stats.forced_ms - out.stats.grammar_ms;
                    sampled += out.stats.generated_tokens;
                }
            }
            std::printf("  %s, %s: grammar %.3f ms/token on a schema's first use, %.3f ms/token again\n",
                        path.c_str() + path.rfind('/') + 1, mode.name, first_ms / std::max(1, first_tokens),
                        again_ms / std::max(1, again_tokens));
        }
        CHECK(valid_count == runs);
        // A tool call as the app's local API makes one: the model's own
        // format, `<tool_call>` forced (tool_choice required), then the call
        // as one span with the grammar the app writes (ToolDialect).
        {
            auto engine = Engine::create(config);
            REQUIRE(engine.has_value());
            const auto tmpl = engine.value()->model_metadata("tokenizer.chat_template");
            const bool xml = tmpl && tmpl->find("<function=") != std::string::npos;
            const bool hermes = !xml && tmpl && tmpl->find("<tool_call>") != std::string::npos;
            if (xml || hermes) {
                const char* tool = R"({"type": "function", "function": {"name": "get_weather", "description": "Weather forecast for a city", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"enum": ["celsius", "fahrenheit"]}, "days": {"type": "integer", "minimum": 1, "maximum": 14}}, "required": ["city"]}}})";
                const std::string system =
                    xml ? std::string("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n") +
                              tool + "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
                        : std::string("<|im_start|>system\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n") +
                              tool + "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
                const std::string prompt = system +
                    "<|im_start|>user\nWhat will the weather be in Lisbon over the next 3 days, in celsius?<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n";
                const Grammar call = text_grammar(
                    xml ? R"(root ::= "\n" "<function=" (t0) "</function>\n</tool_call>"
p0-0 ::= "<parameter=city>\n" %until "\n</parameter>" "\n"
p0-1 ::= "<parameter=unit>\n" %string {"enum":["celsius","fahrenheit"]} "\n</parameter>\n"
p0-2 ::= "<parameter=days>\n" %schema {"type":"integer","minimum":1,"maximum":14} "\n</parameter>\n"
t0 ::= "get_weather>\n" p0-0 p0-1? p0-2?)"
                        : R"(root ::= "\n" "{\"name\": " (t0) "}\n</tool_call>"
t0 ::= "\"get_weather\", \"arguments\": " %schema {"type":"object","properties":{"city":{"type":"string"},"unit":{"enum":["celsius","fahrenheit"]},"days":{"type":"integer","minimum":1,"maximum":14}},"required":["city"]})");
                for (const float temperature : {0.0f, 0.7f}) {
                    SamplingParams p = sampling(temperature, 11, 120);
                    p.top_k = 20;
                    const Output out = generate(*engine.value(), prompt, p, GenerateOptions{.continuation = [&](std::string_view so_far) {
                        return so_far.empty() ? Continuation{"<tool_call>", call} : Continuation{};
                    }});
                    REQUIRE(out.text.starts_with("<tool_call>"));
                    const size_t end = out.text.find("</tool_call>");
                    REQUIRE(end != std::string::npos);
                    const std::string body = out.text.substr(11, end + 12 - 11);
                    CHECK(call.matches(body));
                    std::printf("  %s, %s call: %s\n", path.c_str() + path.rfind('/') + 1, xml ? "XML" : "JSON",
                                out.text.substr(0, end + 12).c_str());
                }
            }
        }
        std::printf("  %s: trie + first token %.0f ms; %d/%d valid; decode step %.2f ms\n",
                    path.c_str() + path.rfind('/') + 1, trie_ms, valid_count, runs, decode_ms / std::max(1, sampled));
    }
}

TEST_MAIN()
