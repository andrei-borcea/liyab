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

#include "core/chat_template.h"
#include "core/grammar.h"
#include "liyab/engine.h"
#include "liyab/grammar.h"
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

TEST_MAIN()
