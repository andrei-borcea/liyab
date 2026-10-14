// Liyab — chat templates: a Jinja-subset interpreter for the model's own
// tokenizer.chat_template (internal).
//
// Every GGUF carries the chat template its Hugging Face tokenizer applies
// (`tokenizer.chat_template`, Jinja). This interpreter renders it the way
// transformers' apply_chat_template does: the Jinja environment it uses
// (trim_blocks, lstrip_blocks, loop controls, a trailing newline of the
// source dropped), its `tojson` (json.dumps with ensure_ascii off, keys in
// order), `raise_exception` and `strftime_now`, and the variables it passes
// (`messages`, `tools`, `documents`, `add_generation_prompt`, the special
// tokens, extra keyword arguments such as `enable_thinking`).
//
// Supported: text, {{ }}, {% %}, {# #} with '-' whitespace control; if /
// elif / else, for (tuple targets, `if` filters, else, loop.index / index0 /
// revindex / first / last / length / previtem / nextitem / cycle), break /
// continue, set (names, namespace attributes, blocks), macro with defaults,
// generation blocks; literals, lists, dicts, tuples, attribute / item /
// slice access, calls with keyword arguments, + - * / // % ** ~, comparisons,
// in / not in, and / or / not, conditional expressions; Python string,
// list and dict methods (split, strip, startswith, items, get...); the
// usual filters (trim, length, tojson, items, string, join, default, map,
// select, ...) and tests (defined, none, string, mapping, iterable,
// sequence, number...). Values print as Python prints them. As in the
// sandboxed environment transformers uses, methods that mutate a list or a
// dict (append, update...) are refused. Anything else is an error naming
// it, never a silent difference.
//
// A parsed template is immutable and can render from several threads.
#ifndef LIYAB_CORE_CHAT_TEMPLATE_H
#define LIYAB_CORE_CHAT_TEMPLATE_H

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "liyab/engine.h"
#include "liyab/types.h"

namespace liyab::jinja {

class Value;
struct Object;
struct Callable;
using Array = std::vector<Value>;
using Kwargs = std::vector<std::pair<std::string, Value>>;

// A template value: Python's semantics for the types JSON and Jinja need.
// Lists and dicts are shared by reference, as in Python (a namespace changed
// inside a loop is changed outside it).
class Value {
public:
    enum class Type : uint8_t { Undefined, None, Bool, Int, Float, String, Array, Object, Callable };

    Value() = default;  // undefined
    static Value undefined(std::string name = {});
    static Value none();
    Value(bool b);                 // NOLINT(google-explicit-constructor)
    Value(int64_t i);              // NOLINT(google-explicit-constructor)
    Value(int i) : Value(static_cast<int64_t>(i)) {}  // NOLINT(google-explicit-constructor)
    Value(double f);               // NOLINT(google-explicit-constructor)
    Value(std::string s);          // NOLINT(google-explicit-constructor)
    Value(const char* s) : Value(std::string(s)) {}  // NOLINT(google-explicit-constructor)
    static Value array(Array items = {});
    static Value object();
    static Value callable(std::string name, std::function<Value(Array&, Kwargs&)> fn);

    [[nodiscard]] Type type() const noexcept { return type_; }
    [[nodiscard]] bool is_undefined() const noexcept { return type_ == Type::Undefined; }
    [[nodiscard]] bool is_none() const noexcept { return type_ == Type::None; }
    [[nodiscard]] bool is_string() const noexcept { return type_ == Type::String; }
    [[nodiscard]] bool is_array() const noexcept { return type_ == Type::Array; }
    [[nodiscard]] bool is_object() const noexcept { return type_ == Type::Object; }
    [[nodiscard]] bool is_number() const noexcept { return type_ == Type::Int || type_ == Type::Float || type_ == Type::Bool; }
    [[nodiscard]] bool truthy() const;

    [[nodiscard]] bool as_bool() const noexcept { return b_; }
    [[nodiscard]] int64_t as_int() const noexcept;     // Int / Bool / Float (truncated)
    [[nodiscard]] double as_float() const noexcept;    // any number
    [[nodiscard]] const std::string& as_string() const noexcept { return s_; }  // String; Undefined: its name
    [[nodiscard]] Array& items() const { return *a_; }
    [[nodiscard]] Object& fields() const { return *o_; }
    [[nodiscard]] const Callable& function() const { return *f_; }

    // Python str() and repr().
    [[nodiscard]] std::string str() const;
    [[nodiscard]] std::string repr() const;
    // json.dumps(value, ensure_ascii, indent (< 0: none), separators, sort_keys).
    [[nodiscard]] std::string to_json(int indent = -1, bool ensure_ascii = false, std::string_view item_sep = {},
                                      std::string_view key_sep = {}, bool sort_keys = false) const;

    friend bool operator==(const Value& a, const Value& b);

private:
    Type type_ = Type::Undefined;
    bool b_ = false;
    int64_t i_ = 0;
    double f_val_ = 0.0;
    std::string s_;
    std::shared_ptr<Array> a_;
    std::shared_ptr<Object> o_;
    std::shared_ptr<Callable> f_;
};

struct Object {
    std::vector<std::pair<std::string, Value>> entries;  // insertion order, as Python dicts
    bool is_namespace = false;
    [[nodiscard]] const Value* find(std::string_view key) const;
    void set(const std::string& key, Value value);
};

struct Callable {
    std::string name;
    std::function<Value(Array&, Kwargs&)> fn;
};

// JSON text → value (objects keep their key order; integers stay integers).
Result<Value> parse_json(std::string_view text);

}  // namespace liyab::jinja

namespace liyab {

class ChatTemplate {
public:
    // Parses `source`. InvalidArgument names the problem and its line.
    static Result<ChatTemplate> parse(std::string_view source);

    // Renders with `context` (an object: messages, tools, add_generation_prompt...).
    // A template's raise_exception() and every evaluation error come back as
    // InvalidArgument with the message.
    [[nodiscard]] Result<std::string> render(const jinja::Value& context) const;

    // apply_chat_template: `request_json` is an object with `messages` (the
    // OpenAI / Hugging Face message list), optional `tools`, optional
    // `add_generation_prompt` (default false, as transformers), and any
    // other keys as template variables (`enable_thinking`...). `special`
    // holds the special-token variables (bos_token, eos_token...), which a
    // request may not override.
    [[nodiscard]] Result<std::string> render_chat(std::string_view request_json, const jinja::Value& special) const;

    // How the template lays out a conversation, from renders of probe
    // conversations (see ChatFormat in liyab/engine.h). `probe_json` holds
    // the template variables to render with and an optional `reply_prefix`
    // (what a reply starts with after the generation prompt, e.g. reasoning
    // closed by "</think>\n\n"). A template whose renders are not prefixes of
    // each other comes back with prefix_stable false and the reason.
    [[nodiscard]] Result<ChatFormat> format(std::string_view probe_json, const jinja::Value& special) const;

    struct Node;  // parsed template (chat_template.cpp)

private:
    std::shared_ptr<const std::vector<Node>> body_;
};

}  // namespace liyab

#endif  // LIYAB_CORE_CHAT_TEMPLATE_H
