// Liyab — grammars for constrained decoding (structured output).
//
// A Grammar describes the text a model may write. While one is active,
// Engine::generate masks every token whose bytes the grammar cannot accept at
// that point (see GenerateOptions in liyab/engine.h), so the output always
// matches it: a JSON value valid against a schema, any JSON value, or a
// language written in Liyab's grammar syntax (below).
//
// Grammars work on bytes: a multi-byte UTF-8 character may be split across
// tokens (byte-level BPE, SentencePiece byte fallback) and is matched byte
// by byte, and every character class only admits valid UTF-8, so a matched
// text is always valid UTF-8.
//
// ---- JSON Schema (from_json_schema) ---------------------------------------
// The subset tool parameters and response formats use:
//   * type: "object" | "array" | "string" | "number" | "integer" | "boolean"
//     | "null", or a list of them; without `type`, the type follows from
//     properties / items, else any JSON value; `true` or {} is any value;
//   * enum, const (any JSON values, written as json.dumps writes them);
//   * object: properties, required, additionalProperties. Properties are
//     written in the order the schema lists them; required ones always,
//     optional ones when the model chooses to. Only declared properties are
//     written (also with additionalProperties true or a schema, which the
//     output then trivially satisfies); a `required` name the properties do
//     not declare is written with the additionalProperties schema (any value
//     when absent). An object without `properties` is a map whose values
//     follow additionalProperties (any value when absent or true; `{}` only
//     when false);
//   * array: items (a schema, true, or false), minItems, maxItems (<= 10000);
//   * string: minLength, maxLength (<= 10000, in characters, an escape
//     counting as one), format "date", "time", "date-time" (RFC 3339) and
//     "uuid"; other formats are annotations and are not checked;
//   * integer: minimum, maximum, exclusiveMinimum, exclusiveMaximum (any
//     64-bit bounds); number: minimum / exclusiveMinimum 0 (non-negative /
//     positive numbers), other bounds on non-integer numbers are refused;
//   * anyOf, oneOf (the output matches at least one branch: oneOf branches
//     that overlap are not told apart), allOf with one schema, nullable
//     (OpenAPI), $ref to any JSON pointer in the document ("#",
//     "#/$defs/name", "#/definitions/name", recursion included), $defs.
// Annotations (title, description, default, examples...) are ignored, as
// are unknown keywords, as JSON Schema itself does. Keywords that constrain
// values in ways a grammar cannot express or Liyab does not implement
// (pattern, patternProperties, propertyNames, uniqueItems, contains,
// prefixItems, multipleOf, not, if/then/else, dependentRequired,
// minProperties, maxProperties...) are refused with Unsupported naming the
// keyword and where it is, never silently ignored.
//
// Numbers are written without leading zeros, with at most 19 integer digits,
// 16 fraction digits and a 3-digit exponent, so a model cannot loop on
// digits.
//
// ---- Grammar syntax (parse) ------------------------------------------------
// GBNF-like rules; `root` is where matching starts:
//
//   root   ::= "<tool_call>\n" call "\n</tool_call>"
//   call   ::= "{\"name\": \"get_time\", \"arguments\": " %schema {"type": "object"} "}"
//   digits ::= [0-9]+          # a comment
//
//   "text"        literal (escapes \n \r \t \\ \" \' \xHH \uHHHH \UHHHHHHHH)
//   [a-z_] [^"\\] character class over code points (same escapes, `^` negates)
//   .             any character (newline included)
//   name          another rule (defined before or after its use)
//   ( ... )       grouping;   a | b   alternatives
//   x* x+ x?      repetition; x{n} x{n,} x{n,m} (bounds <= 10000)
//   %json         any JSON value (json.dumps' layout, as JsonFormat{})
//   %schema {..}  a JSON value valid against the JSON Schema that follows
//   %until "s"    any text that does not contain s, then s (e.g. reasoning
//                 up to its closing tag)
//
// Left recursion (a rule that can reach itself before matching a byte) and
// undefined rules are refused with InvalidArgument naming the rule.
//
// A Grammar is immutable and cheap to copy (shared); it can be used by
// several engines and threads at once.
#ifndef LIYAB_GRAMMAR_H
#define LIYAB_GRAMMAR_H

#include <cstdint>
#include <memory>
#include <string_view>

#include "liyab/types.h"

namespace liyab {

namespace grammar {
struct Program;
}

// How JSON is laid out (JSON Schema and %json).
struct JsonFormat {
    // false (default): no white space outside strings except one space after
    // ':' and ',' (json.dumps' default separators, the layout tool calls are
    // trained on). Every character that is then the only one possible (keys,
    // separators, brackets) is forced instead of sampled. true: any JSON
    // white space, at most 32 characters at each place JSON allows it, and
    // before and after the value.
    bool any_whitespace = false;
};

class LIYAB_API Grammar {
public:
    // A JSON value valid against `schema_json` (a JSON Schema, see above).
    // InvalidArgument for malformed JSON or an unsatisfiable schema,
    // Unsupported for a keyword Liyab does not enforce.
    static Result<Grammar> from_json_schema(std::string_view schema_json, const JsonFormat& format = {});
    // Any JSON value.
    static Grammar json(const JsonFormat& format = {});
    // Rules in the grammar syntax above; InvalidArgument names the problem.
    static Result<Grammar> parse(std::string_view text);

    // Whether `text` is a complete match.
    [[nodiscard]] bool matches(std::string_view text) const;
    // Whether `text` can still be continued into a match (a match included).
    [[nodiscard]] bool accepts_prefix(std::string_view text) const;

    // The compiled form (internal: core/grammar.h).
    [[nodiscard]] const std::shared_ptr<const grammar::Program>& program() const noexcept { return program_; }

private:
    explicit Grammar(std::shared_ptr<const grammar::Program> program) : program_(std::move(program)) {}
    std::shared_ptr<const grammar::Program> program_;
};

}  // namespace liyab

#endif  // LIYAB_GRAMMAR_H
