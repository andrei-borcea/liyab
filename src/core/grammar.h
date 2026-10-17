// Liyab — grammar engine for constrained decoding (internal).
//
// Representation: a grammar is lowered to a byte-level program. Each rule is
// a list of alternatives; an alternative is a sequence of elements, each a
// set of bytes (one input byte) or a reference to a rule, and ends with End.
// Characters outside ASCII are lowered to their UTF-8 byte sequences (a code
// point range becomes a few alternatives of byte ranges), so matching never
// decodes UTF-8 and a token may end inside a character.
//
// Matching is a pushdown automaton: a parse is a stack of program positions
// (return addresses under the current position), and the matcher's state is
// the set of stacks the input so far allows (several when alternatives are
// still open). Every stack is kept "closed": its top is a byte set or the
// stack is empty (the root matched). A reference followed by End is a tail
// call and pushes no return address, so right recursion (repetition) keeps
// stacks bounded.
//
// Constrained decoding needs, per sampled token, the tokens whose bytes the
// state can accept. Two caches make that cheap:
//   * StateTable interns every state set it meets and memoizes the byte
//     transitions between them (a lazily built DFA over the stack sets);
//     inside a string, for example, nearly every byte maps the state to
//     itself, so a transition is one table lookup;
//   * the vocabulary is a trie (VocabTrie) laid out in depth-first order, and
//     a state's token mask is one pass over it that skips each subtree whose
//     prefix the state rejects. Masks are cached per state, so a state seen
//     again (the same position in a string, a repeated key) costs a lookup.
// Both caches are bounded (see StateTable) and rebuilt on demand.
#ifndef LIYAB_CORE_GRAMMAR_H
#define LIYAB_CORE_GRAMMAR_H

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "liyab/grammar.h"
#include "liyab/types.h"

namespace liyab {
class Tokenizer;
}

namespace liyab::grammar {

using ByteSet = std::array<uint64_t, 4>;

inline bool has_byte(const ByteSet& s, uint8_t b) noexcept { return (s[b >> 6] >> (b & 63)) & 1; }
inline void add_byte(ByteSet& s, uint8_t b) noexcept { s[b >> 6] |= uint64_t{1} << (b & 63); }

enum class Op : uint8_t { Bytes, Ref, End };

struct Elem {
    Op op = Op::End;
    uint32_t arg = 0;  // Bytes: index into Program::sets; Ref: rule id
    friend bool operator==(const Elem&, const Elem&) = default;
};

struct Program {
    std::vector<Elem> elems;
    std::vector<ByteSet> sets;
    std::vector<std::vector<uint32_t>> rules;  // per rule: the first position of each alternative
    std::vector<std::string> names;            // per rule, for messages
    uint32_t root = 0;
    uint64_t fingerprint = 0;  // hash of the above (cache key; compared in full on a hit)

    [[nodiscard]] bool same(const Program& other) const {
        return fingerprint == other.fingerprint && root == other.root && elems == other.elems && sets == other.sets &&
               rules == other.rules;
    }
};

// A code point range, inclusive.
struct CpRange {
    uint32_t lo = 0, hi = 0;
};

inline constexpr uint32_t kUnbounded = 0xFFFFFFFFu;
inline constexpr uint32_t kMaxRepeat = 10000;

// Builds a grammar from expression nodes, then lowers it to a Program.
// Nodes are immutable and may be shared (lowering memoizes them).
class Builder {
public:
    using Node = uint32_t;

    Node empty();
    Node literal(std::string_view bytes);
    // Code points in `ranges` (or outside them when `negated`), as UTF-8.
    // Surrogates are never matched.
    Node chars(std::vector<CpRange> ranges, bool negated = false);
    Node any_char() { return chars({}, true); }
    Node seq(std::vector<Node> items);
    Node alt(std::vector<Node> items);
    // `item` between min and max times (max kUnbounded: no limit).
    Node repeat(Node item, uint32_t min, uint32_t max);
    Node optional(Node item) { return repeat(item, 0, 1); }
    Node ref(uint32_t rule);

    // A new rule (names need not be unique; they label messages only).
    uint32_t add_rule(std::string name);
    // The rule named `name`, declared on first use (grammar text).
    uint32_t named_rule(std::string_view name);
    void define(uint32_t rule, Node body);
    [[nodiscard]] bool defined(uint32_t rule) const { return rules_[rule].defined; }
    [[nodiscard]] const std::string& rule_name(uint32_t rule) const { return rules_[rule].name; }

    // Lowers the rules reachable from `root`. InvalidArgument for an
    // undefined rule or left recursion, naming the rule.
    Result<Program> build(uint32_t root) const;

private:
    enum class Kind : uint8_t { Empty, Literal, Chars, Seq, Alt, Repeat, Ref };
    struct NodeData {
        Kind kind = Kind::Empty;
        std::string bytes;            // Literal
        std::vector<CpRange> ranges;  // Chars (normalized, not negated)
        std::vector<Node> items;      // Seq, Alt, Repeat (one item)
        uint32_t min = 0, max = 0;    // Repeat
        uint32_t rule = 0;            // Ref
    };
    struct RuleData {
        std::string name;
        Node body = 0;
        bool defined = false;
    };
    std::vector<NodeData> nodes_;
    std::vector<RuleData> rules_;
    std::unordered_map<std::string, uint32_t> by_name_;
    friend class Lowering;
};

// JSON values into a Builder (core/json_schema.cpp). Each returns the node
// that matches the value; rules for $ref targets and the generic JSON value
// are added to the builder as needed.
Result<Builder::Node> json_schema(Builder& b, std::string_view schema_json, const JsonFormat& format);
Builder::Node json_value(Builder& b, const JsonFormat& format);
// A string schema's value as raw text: no quotes, no escapes (%string).
Result<Builder::Node> raw_string(Builder& b, std::string_view schema_json);
// Grammar text (core/grammar_text.cpp): defines its rules; returns `root`.
Result<uint32_t> parse_text(Builder& b, std::string_view text);

// The vocabulary as a trie in depth-first order, for masks. Tokens with an
// empty piece (control tokens) are not in it; end-of-generation tokens are
// listed apart, allowed exactly when the grammar accepts.
struct VocabTrie {
    struct Node {
        uint8_t byte = 0;
        uint16_t depth = 0;        // 1 for a token's first byte
        uint32_t subtree_end = 0;  // the first node after this one's subtree
        uint32_t tok_begin = 0, tok_end = 0;  // tokens whose piece ends here (into `tokens`)
    };
    std::vector<Node> nodes;
    std::vector<int32_t> tokens;
    std::vector<int32_t> end_of_generation;
    enum class Kind : uint8_t { None, Text, EndOfGeneration };
    std::vector<Kind> kinds;  // per token id
    int32_t vocab_size = 0;
    uint16_t max_depth = 0;

    static std::unique_ptr<VocabTrie> build(const Tokenizer& tokenizer);
};

// Interned matcher states of one program, their byte transitions, forced
// bytes and token masks. Not thread-safe: one per engine, used by the
// generating thread.
class StateTable {
public:
    // `max_states`: how many states compact() lets the table hold (tests
    // lower it; memory is bounded too, at 32 MiB).
    explicit StateTable(std::shared_ptr<const Program> program, size_t max_states = size_t{1} << 16);

    [[nodiscard]] const std::shared_ptr<const Program>& program() const noexcept { return program_; }

    // The state before any input.
    int32_t initial();
    // The state after `byte` (kDead when the grammar rejects it).
    int32_t advance(int32_t state, uint8_t byte);
    int32_t advance(int32_t state, std::string_view bytes);
    static constexpr int32_t kDead = -1;

    [[nodiscard]] bool accepting(int32_t state) const { return states_[static_cast<size_t>(state)].accept; }
    // Whether some byte can follow (false: the match is complete and final).
    [[nodiscard]] bool extendable(int32_t state) const { return states_[static_cast<size_t>(state)].next_count > 0; }
    // The byte that must follow, or -1 when there is a choice (or the state
    // accepts, so ending is possible).
    [[nodiscard]] int forced_byte(int32_t state) const;
    // The longest text every continuation starts with, cut after its last
    // whole UTF-8 character (bytes of an unfinished one stay unforced).
    std::string forced_text(int32_t state, size_t max_bytes = 4096);

    // Tokens the state allows: one bit per token id (end-of-generation tokens
    // when it accepts). Valid until the next mask() or compact().
    const std::vector<uint64_t>& mask(int32_t state, const VocabTrie& trie);
    // Sets the logits of disallowed tokens to -infinity; returns how many
    // tokens stay allowed.
    size_t apply_mask(int32_t state, const VocabTrie& trie, std::span<float> logits);
    // Whether the state allows `token` (its piece, or an end-of-generation
    // token where the state accepts).
    bool allows(int32_t state, int32_t token, const Tokenizer& tokenizer, const VocabTrie& trie);
    // Whether the state allows a token whose piece is `piece` followed by
    // at least one more byte: a token the model could write in place of
    // `piece` and what comes after it (` {"` for a forced ` {`). The engine
    // leaves such a last forced token to the model (token healing, see
    // GenerateOptions in liyab/engine.h). Walks only the trie's subtree under
    // `piece`, skipping what the state rejects; false when `piece` is not a
    // path of the trie or the state rejects it.
    bool extends(int32_t state, std::string_view piece, const VocabTrie& trie);
    // Leaves only the `k` highest logits among allowed tokens (ties: the
    // lower id first) and sets every other logit to -infinity: what masking
    // and then keeping the top k leaves, found by checking the most likely
    // tokens one by one instead of building the state's whole mask. Greedy
    // and top-k sampling then cost a few token checks per step even in a
    // state never seen before (a bounded string has one per position).
    // Returns false, logits untouched, when fewer than k of the 1024 most
    // likely tokens are allowed: the caller masks in full.
    bool keep_top_allowed(int32_t state, const Tokenizer& tokenizer, const VocabTrie& trie, std::span<float> logits,
                          size_t k);

    // Bounds memory: when the table holds more than its limits, forgets every
    // state but `keep` (returned with its new id); call between tokens.
    int32_t compact(int32_t keep);
    [[nodiscard]] size_t size() const noexcept { return states_.size(); }
    [[nodiscard]] size_t memory_bytes() const noexcept { return bytes_; }
    // True once a state set exceeded the depth / width limits (the grammar
    // is too ambiguous); such stacks were dropped.
    [[nodiscard]] bool overflowed() const noexcept { return overflow_; }

private:
    struct KeyHash {
        size_t operator()(const std::vector<uint32_t>& k) const noexcept;
    };
    struct State {
        const std::vector<uint32_t>* key = nullptr;  // owned by index_
        bool accept = false;
        uint16_t next_count = 0;  // distinct bytes that can follow
        ByteSet next{};
        std::unique_ptr<int32_t[]> row;  // 256 transitions, kUnknown until computed
        std::vector<uint64_t> mask;
    };
    static constexpr int32_t kUnknown = -2;

    int32_t intern(std::vector<std::vector<uint32_t>>& stacks);
    int32_t* row(int32_t state);  // the state's transitions, allocated on first use
    void close(std::vector<uint32_t>& stack, std::vector<std::vector<uint32_t>>& out, int depth);
    int32_t compute(int32_t state, uint8_t byte);

    std::shared_ptr<const Program> program_;
    std::vector<State> states_;
    std::unordered_map<std::vector<uint32_t>, int32_t, KeyHash> index_;
    int32_t initial_ = kUnknown;
    size_t masks_ = 0;  // states holding a mask
    size_t bytes_ = 0;  // memory held, approximately (kept as states, rows and masks are added)
    bool overflow_ = false;
    size_t max_states_;
};

}  // namespace liyab::grammar

#endif  // LIYAB_CORE_GRAMMAR_H
