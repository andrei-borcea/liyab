#include "core/grammar.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <map>

#include "core/tokenizer.h"

namespace liyab::grammar {
namespace {

constexpr uint32_t kMaxCodePoint = 0x10FFFF;

// UTF-8 encoding of a code point, `n` bytes long.
void encode_utf8(uint32_t cp, int n, uint8_t out[4]) {
    switch (n) {
        case 1:
            out[0] = static_cast<uint8_t>(cp);
            break;
        case 2:
            out[0] = static_cast<uint8_t>(0xC0 | (cp >> 6));
            out[1] = static_cast<uint8_t>(0x80 | (cp & 0x3F));
            break;
        case 3:
            out[0] = static_cast<uint8_t>(0xE0 | (cp >> 12));
            out[1] = static_cast<uint8_t>(0x80 | ((cp >> 6) & 0x3F));
            out[2] = static_cast<uint8_t>(0x80 | (cp & 0x3F));
            break;
        default:
            out[0] = static_cast<uint8_t>(0xF0 | (cp >> 18));
            out[1] = static_cast<uint8_t>(0x80 | ((cp >> 12) & 0x3F));
            out[2] = static_cast<uint8_t>(0x80 | ((cp >> 6) & 0x3F));
            out[3] = static_cast<uint8_t>(0x80 | (cp & 0x3F));
            break;
    }
}

using ByteRanges = std::vector<std::pair<uint8_t, uint8_t>>;  // one range per byte of a sequence

// Code points [lo, hi], all encoded with `n` bytes, as sequences of byte
// ranges whose cartesian products are exactly those encodings. A range is
// split until, at every continuation byte, either lo and hi agree on the
// bytes before it or they span that byte's whole 0x80-0xBF range; then each
// byte position is independent.
void split_utf8(uint32_t lo, uint32_t hi, int n, std::vector<ByteRanges>& out) {
    for (int i = 1; i < n; ++i) {
        const uint32_t m = (uint32_t{1} << (6 * i)) - 1;
        if ((lo & ~m) != (hi & ~m)) {
            if ((lo & m) != 0) {
                split_utf8(lo, lo | m, n, out);
                split_utf8((lo | m) + 1, hi, n, out);
                return;
            }
            if ((hi & m) != m) {
                split_utf8(lo, (hi & ~m) - 1, n, out);
                split_utf8(hi & ~m, hi, n, out);
                return;
            }
        }
    }
    uint8_t a[4], b[4];
    encode_utf8(lo, n, a);
    encode_utf8(hi, n, b);
    ByteRanges seq;
    for (int k = 0; k < n; ++k) seq.emplace_back(a[k], b[k]);
    out.push_back(std::move(seq));
}

// Every UTF-8 encoding of the code points in `ranges` (surrogates excluded).
std::vector<ByteRanges> utf8_sequences(const std::vector<CpRange>& ranges) {
    static constexpr uint32_t kLengthEnds[] = {0x7F, 0x7FF, 0xFFFF, kMaxCodePoint};
    std::vector<ByteRanges> out;
    for (const CpRange& r : ranges) {
        // Cut out the surrogates, then cut at each encoded-length boundary.
        std::vector<CpRange> parts;
        if (r.hi < 0xD800 || r.lo > 0xDFFF) {
            parts.push_back(r);
        } else {
            if (r.lo < 0xD800) parts.push_back({r.lo, 0xD7FF});
            if (r.hi > 0xDFFF) parts.push_back({0xE000, r.hi});
        }
        for (const CpRange& p : parts) {
            uint32_t lo = p.lo;
            for (int n = 1; n <= 4 && lo <= p.hi; ++n) {
                const uint32_t end = kLengthEnds[n - 1];
                if (lo > end) continue;
                const uint32_t hi = std::min(p.hi, end);
                split_utf8(lo, hi, n, out);
                lo = hi + 1;
            }
        }
    }
    return out;
}

uint64_t fnv(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        h ^= (v >> (8 * i)) & 0xFF;
        h *= 0x100000001B3ull;
    }
    return h;
}

}  // namespace

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

Builder::Node Builder::empty() {
    nodes_.push_back({});
    return static_cast<Node>(nodes_.size() - 1);
}

Builder::Node Builder::literal(std::string_view bytes) {
    NodeData d;
    d.kind = bytes.empty() ? Kind::Empty : Kind::Literal;
    d.bytes.assign(bytes);
    nodes_.push_back(std::move(d));
    return static_cast<Node>(nodes_.size() - 1);
}

Builder::Node Builder::chars(std::vector<CpRange> ranges, bool negated) {
    for (CpRange& r : ranges) {
        r.hi = std::min(r.hi, kMaxCodePoint);
        if (r.lo > r.hi) r = {1, 0};
    }
    std::erase_if(ranges, [](const CpRange& r) { return r.lo > r.hi; });
    std::sort(ranges.begin(), ranges.end(), [](const CpRange& a, const CpRange& b) { return a.lo < b.lo; });
    std::vector<CpRange> merged;
    for (const CpRange& r : ranges) {
        if (!merged.empty() && r.lo <= merged.back().hi + 1) {
            merged.back().hi = std::max(merged.back().hi, r.hi);
        } else {
            merged.push_back(r);
        }
    }
    if (negated) {
        std::vector<CpRange> complement;
        uint32_t next = 0;
        for (const CpRange& r : merged) {
            if (r.lo > next) complement.push_back({next, r.lo - 1});
            next = r.hi + 1;
        }
        if (next <= kMaxCodePoint) complement.push_back({next, kMaxCodePoint});
        merged = std::move(complement);
    }
    NodeData d;
    d.kind = Kind::Chars;
    d.ranges = std::move(merged);
    nodes_.push_back(std::move(d));
    return static_cast<Node>(nodes_.size() - 1);
}

Builder::Node Builder::seq(std::vector<Node> items) {
    if (items.size() == 1) return items[0];
    NodeData d;
    d.kind = items.empty() ? Kind::Empty : Kind::Seq;
    d.items = std::move(items);
    nodes_.push_back(std::move(d));
    return static_cast<Node>(nodes_.size() - 1);
}

Builder::Node Builder::alt(std::vector<Node> items) {
    if (items.size() == 1) return items[0];
    NodeData d;
    d.kind = Kind::Alt;  // no items: matches nothing
    d.items = std::move(items);
    nodes_.push_back(std::move(d));
    return static_cast<Node>(nodes_.size() - 1);
}

Builder::Node Builder::repeat(Node item, uint32_t min, uint32_t max) {
    if (min == 1 && max == 1) return item;
    NodeData d;
    d.kind = max == 0 ? Kind::Empty : Kind::Repeat;
    d.items = {item};
    d.min = min;
    d.max = max;
    nodes_.push_back(std::move(d));
    return static_cast<Node>(nodes_.size() - 1);
}

Builder::Node Builder::ref(uint32_t rule) {
    NodeData d;
    d.kind = Kind::Ref;
    d.rule = rule;
    nodes_.push_back(std::move(d));
    return static_cast<Node>(nodes_.size() - 1);
}

uint32_t Builder::add_rule(std::string name) {
    rules_.push_back({std::move(name), 0, false});
    return static_cast<uint32_t>(rules_.size() - 1);
}

uint32_t Builder::named_rule(std::string_view name) {
    const auto it = by_name_.find(std::string(name));
    if (it != by_name_.end()) return it->second;
    const uint32_t id = add_rule(std::string(name));
    by_name_.emplace(std::string(name), id);
    return id;
}

void Builder::define(uint32_t rule, Node body) {
    rules_[rule].body = body;
    rules_[rule].defined = true;
}

// ---------------------------------------------------------------------------
// Lowering: expression nodes → byte-level program
// ---------------------------------------------------------------------------

class Lowering {
public:
    explicit Lowering(const Builder& b) : b_(b) {}

    Result<Program> run(uint32_t root) {
        p_.root = program_rule(root);
        while (!queue_.empty()) {
            const uint32_t rule = queue_.back();
            queue_.pop_back();
            const Builder::RuleData& r = b_.rules_[rule];
            if (!r.defined) return Status(ErrorCode::InvalidArgument, "grammar: undefined rule '" + r.name + "'");
            alts_[rule_map_.at(rule)] = lower_alts(r.body, r.name);
        }
        // Flatten: each alternative's elements, then End.
        p_.rules.resize(alts_.size());
        for (size_t r = 0; r < alts_.size(); ++r) {
            for (const std::vector<Elem>& alt : alts_[r]) {
                p_.rules[r].push_back(static_cast<uint32_t>(p_.elems.size()));
                p_.elems.insert(p_.elems.end(), alt.begin(), alt.end());
                p_.elems.push_back({Op::End, 0});
            }
        }
        LIYAB_RETURN_IF_ERROR(check_left_recursion());
        uint64_t h = 0xCBF29CE484222325ull;
        h = fnv(h, p_.root);
        for (const Elem& e : p_.elems) h = fnv(h, static_cast<uint64_t>(e.op) << 32 | e.arg);
        for (const ByteSet& s : p_.sets) {
            for (const uint64_t w : s) h = fnv(h, w);
        }
        for (const auto& alts : p_.rules) {
            h = fnv(h, alts.size());
            for (const uint32_t a : alts) h = fnv(h, a);
        }
        p_.fingerprint = h;
        return std::move(p_);
    }

private:
    using Node = Builder::Node;
    using Kind = Builder::Kind;

    uint32_t new_rule(std::string name) {
        alts_.emplace_back();
        p_.names.push_back(std::move(name));
        return static_cast<uint32_t>(alts_.size() - 1);
    }

    uint32_t program_rule(uint32_t rule) {
        const auto it = rule_map_.find(rule);
        if (it != rule_map_.end()) return it->second;
        const uint32_t id = new_rule(b_.rules_[rule].name);
        rule_map_.emplace(rule, id);
        queue_.push_back(rule);
        return id;
    }

    uint32_t set_id(const ByteSet& s) {
        const auto it = set_index_.find(s);
        if (it != set_index_.end()) return it->second;
        p_.sets.push_back(s);
        const auto id = static_cast<uint32_t>(p_.sets.size() - 1);
        set_index_.emplace(s, id);
        return id;
    }

    Elem byte_elem(uint8_t lo, uint8_t hi) {
        ByteSet s{};
        for (unsigned b = lo; b <= hi; ++b) add_byte(s, static_cast<uint8_t>(b));
        return {Op::Bytes, set_id(s)};
    }

    std::vector<std::vector<Elem>> lower_alts(Node n, const std::string& ctx) {
        const Builder::NodeData& d = b_.nodes_[n];
        std::vector<std::vector<Elem>> alts;
        if (d.kind == Kind::Alt) {
            for (const Node item : d.items) {
                std::vector<Elem> seq;
                lower_seq(item, seq, ctx);
                alts.push_back(std::move(seq));
            }
        } else if (d.kind == Kind::Chars) {
            alts = char_alts(d.ranges);
        } else {
            std::vector<Elem> seq;
            lower_seq(n, seq, ctx);
            alts.push_back(std::move(seq));
        }
        return alts;
    }

    // UTF-8 alternatives of a character class. Sequences with the same
    // continuation bytes share one alternative (their first bytes merged),
    // which keeps the number of parallel stacks inside strings small.
    std::vector<std::vector<Elem>> char_alts(const std::vector<CpRange>& ranges) {
        std::vector<ByteRanges> seqs = utf8_sequences(ranges);
        std::map<ByteRanges, ByteSet> by_tail;  // continuation ranges → first bytes
        std::vector<ByteRanges> order;
        for (const ByteRanges& s : seqs) {
            ByteRanges tail(s.begin() + 1, s.end());
            auto [it, inserted] = by_tail.try_emplace(tail, ByteSet{});
            if (inserted) order.push_back(tail);
            for (unsigned b = s[0].first; b <= s[0].second; ++b) add_byte(it->second, static_cast<uint8_t>(b));
        }
        std::vector<std::vector<Elem>> alts;
        for (const ByteRanges& tail : order) {
            std::vector<Elem> alt{{Op::Bytes, set_id(by_tail[tail])}};
            for (const auto& [lo, hi] : tail) alt.push_back(byte_elem(lo, hi));
            alts.push_back(std::move(alt));
        }
        return alts;
    }

    // A rule standing for node `n` (memoized: a shared node is one rule).
    uint32_t helper(Node n, const std::string& ctx) {
        const auto it = helpers_.find(n);
        if (it != helpers_.end()) return it->second;
        const Builder::NodeData& d = b_.nodes_[n];
        const uint32_t rule = new_rule(ctx);
        helpers_.emplace(n, rule);
        if (d.kind != Kind::Repeat) {
            alts_[rule] = lower_alts(n, ctx);
            return rule;
        }
        // x{min,max}: min copies, then x* (S ::= x S | "") or a chain of
        // nested options (O_k ::= x O_k-1 | "").
        const std::vector<Elem> item = item_elems(d.items[0], ctx);
        std::vector<Elem> body;
        for (uint32_t i = 0; i < d.min; ++i) body.insert(body.end(), item.begin(), item.end());
        if (d.max == kUnbounded) {
            const uint32_t star = new_rule(ctx + "*");
            std::vector<Elem> loop = item;
            loop.push_back({Op::Ref, star});
            alts_[star] = {loop, {}};
            body.push_back({Op::Ref, star});
        } else if (d.max > d.min) {
            uint32_t prev = 0;
            for (uint32_t k = 1; k <= d.max - d.min; ++k) {
                const uint32_t opt = new_rule(ctx + "?");
                std::vector<Elem> some = item;
                if (k > 1) some.push_back({Op::Ref, prev});
                alts_[opt] = {some, {}};
                prev = opt;
            }
            body.push_back({Op::Ref, prev});
        }
        alts_[rule] = {std::move(body)};
        return rule;
    }

    // The elements of one repeated item: inline when short, else one
    // reference to a rule for it, so repetitions do not copy large bodies.
    std::vector<Elem> item_elems(Node n, const std::string& ctx) {
        std::vector<Elem> out;
        lower_seq(n, out, ctx);
        if (out.size() <= 2) return out;
        const auto it = helpers_.find(n);
        uint32_t rule;
        if (it != helpers_.end()) {
            rule = it->second;
        } else {
            rule = new_rule(ctx);
            helpers_.emplace(n, rule);
            alts_[rule] = {out};
        }
        return {{Op::Ref, rule}};
    }

    void lower_seq(Node n, std::vector<Elem>& out, const std::string& ctx) {
        const Builder::NodeData& d = b_.nodes_[n];
        switch (d.kind) {
            case Kind::Empty:
                return;
            case Kind::Literal:
                for (const char c : d.bytes) {
                    const auto b = static_cast<uint8_t>(c);
                    out.push_back(byte_elem(b, b));
                }
                return;
            case Kind::Chars:
                if (d.ranges.empty() || d.ranges.back().hi < 0x80) {
                    ByteSet s{};
                    for (const CpRange& r : d.ranges) {
                        for (uint32_t c = r.lo; c <= r.hi; ++c) add_byte(s, static_cast<uint8_t>(c));
                    }
                    out.push_back({Op::Bytes, set_id(s)});  // an empty set matches nothing
                } else {
                    out.push_back({Op::Ref, helper(n, ctx)});
                }
                return;
            case Kind::Seq:
                for (const Node item : d.items) lower_seq(item, out, ctx);
                return;
            case Kind::Alt:
            case Kind::Repeat:
                out.push_back({Op::Ref, helper(n, ctx)});
                return;
            case Kind::Ref:
                out.push_back({Op::Ref, program_rule(d.rule)});
                return;
        }
    }

    // Left recursion would make closing a stack loop forever: refuse it.
    Status check_left_recursion() const {
        const size_t n = p_.rules.size();
        std::vector<bool> nullable(n, false);
        for (bool changed = true; changed;) {
            changed = false;
            for (size_t r = 0; r < n; ++r) {
                if (nullable[r]) continue;
                for (const uint32_t start : p_.rules[r]) {
                    bool all = true;
                    for (uint32_t pos = start; p_.elems[pos].op != Op::End; ++pos) {
                        const Elem& e = p_.elems[pos];
                        if (e.op == Op::Bytes || !nullable[e.arg]) {
                            all = false;
                            break;
                        }
                    }
                    if (all) {
                        nullable[r] = changed = true;
                        break;
                    }
                }
            }
        }
        std::vector<std::vector<uint32_t>> edges(n);
        for (size_t r = 0; r < n; ++r) {
            for (const uint32_t start : p_.rules[r]) {
                for (uint32_t pos = start; p_.elems[pos].op == Op::Ref; ++pos) {
                    edges[r].push_back(p_.elems[pos].arg);
                    if (!nullable[p_.elems[pos].arg]) break;
                }
            }
        }
        std::vector<uint8_t> color(n, 0);  // 0 new, 1 on the path, 2 done
        std::vector<std::pair<uint32_t, size_t>> path;
        for (uint32_t start = 0; start < n; ++start) {
            if (color[start] != 0) continue;
            path.push_back({start, 0});
            color[start] = 1;
            while (!path.empty()) {
                auto& [r, next] = path.back();
                if (next == edges[r].size()) {
                    color[r] = 2;
                    path.pop_back();
                    continue;
                }
                const uint32_t to = edges[r][next++];
                if (color[to] == 1) {
                    return Status(ErrorCode::InvalidArgument,
                                  "grammar: left recursion through rule '" + p_.names[to] +
                                      "' (it can reach itself before matching a character, e.g. a repetition of "
                                      "something that can be empty)");
                }
                if (color[to] == 0) {
                    color[to] = 1;
                    path.push_back({to, 0});
                }
            }
        }
        return Status::ok();
    }

    const Builder& b_;
    Program p_;
    std::vector<std::vector<std::vector<Elem>>> alts_;
    std::unordered_map<uint32_t, uint32_t> rule_map_;
    std::unordered_map<Node, uint32_t> helpers_;
    std::map<ByteSet, uint32_t> set_index_;
    std::vector<uint32_t> queue_;
};

Result<Program> Builder::build(uint32_t root) const {
    if (root >= rules_.size()) return Status(ErrorCode::InvalidArgument, "grammar: no root rule");
    return Lowering(*this).run(root);
}

// ---------------------------------------------------------------------------
// Vocabulary trie
// ---------------------------------------------------------------------------

std::unique_ptr<VocabTrie> VocabTrie::build(const Tokenizer& tokenizer) {
    auto trie = std::make_unique<VocabTrie>();
    trie->vocab_size = tokenizer.vocab_size();
    std::vector<std::pair<std::string_view, int32_t>> pieces;
    pieces.reserve(static_cast<size_t>(trie->vocab_size));
    for (int32_t id = 0; id < trie->vocab_size; ++id) {
        if (tokenizer.is_end_of_generation(id)) {
            trie->end_of_generation.push_back(id);
            continue;
        }
        const std::string& piece = tokenizer.piece(id);
        if (piece.empty() || piece.size() > std::numeric_limits<uint16_t>::max()) continue;
        pieces.emplace_back(piece, id);
    }
    std::sort(pieces.begin(), pieces.end());
    std::vector<uint32_t> path;  // node per depth - 1
    std::string_view prev;
    for (const auto& [piece, id] : pieces) {
        size_t lcp = 0;
        while (lcp < prev.size() && lcp < piece.size() && prev[lcp] == piece[lcp]) ++lcp;
        while (path.size() > lcp) {
            trie->nodes[path.back()].subtree_end = static_cast<uint32_t>(trie->nodes.size());
            path.pop_back();
        }
        for (size_t k = lcp; k < piece.size(); ++k) {
            Node node;
            node.byte = static_cast<uint8_t>(piece[k]);
            node.depth = static_cast<uint16_t>(k + 1);
            path.push_back(static_cast<uint32_t>(trie->nodes.size()));
            trie->nodes.push_back(node);
        }
        Node& end = trie->nodes[path.back()];
        if (end.tok_begin == end.tok_end) end.tok_begin = end.tok_end = static_cast<uint32_t>(trie->tokens.size());
        trie->tokens.push_back(id);
        end.tok_end = static_cast<uint32_t>(trie->tokens.size());
        trie->max_depth = std::max(trie->max_depth, static_cast<uint16_t>(piece.size()));
        prev = piece;
    }
    for (const uint32_t n : path) trie->nodes[n].subtree_end = static_cast<uint32_t>(trie->nodes.size());
    return trie;
}

// ---------------------------------------------------------------------------
// StateTable
// ---------------------------------------------------------------------------

namespace {
// Limits that keep a pathological grammar from exhausting memory: a stack
// this deep (nesting) or this many parallel stacks (ambiguity) is dropped.
constexpr size_t kMaxStackDepth = 2048;
constexpr size_t kMaxStacks = 1024;
constexpr int kMaxCloseDepth = 4096;
// Cache bounds: past these, compact() starts over (keeping the live state).
constexpr size_t kMaxStates = 1 << 16;
constexpr size_t kMaxTableBytes = size_t{48} << 20;
constexpr size_t kMaxMaskBytes = size_t{32} << 20;
}  // namespace

size_t StateTable::KeyHash::operator()(const std::vector<uint32_t>& k) const noexcept {
    uint64_t h = 0xCBF29CE484222325ull;
    for (const uint32_t v : k) h = (h ^ v) * 0x100000001B3ull;
    return static_cast<size_t>(h ^ (h >> 29));
}

StateTable::StateTable(std::shared_ptr<const Program> program) : program_(std::move(program)) {}

void StateTable::close(std::vector<uint32_t>& stack, std::vector<std::vector<uint32_t>>& out, int depth) {
    const Program& p = *program_;
    for (;;) {
        if (stack.empty()) {
            out.push_back(stack);
            return;
        }
        if (stack.size() > kMaxStackDepth || depth > kMaxCloseDepth || out.size() >= kMaxStacks) {
            overflow_ = true;
            return;
        }
        const uint32_t pos = stack.back();
        const Elem& e = p.elems[pos];
        if (e.op == Op::Bytes) {
            out.push_back(stack);
            return;
        }
        stack.pop_back();
        if (e.op == Op::End) continue;  // resume at the return address below
        // Ref: return to the next element unless it ends the alternative
        // (a tail call: returning there would only pop again).
        if (p.elems[pos + 1].op != Op::End) stack.push_back(pos + 1);
        const std::vector<uint32_t>& alts = p.rules[e.arg];
        for (size_t i = 0; i < alts.size(); ++i) {
            std::vector<uint32_t> next;
            if (i + 1 < alts.size()) next = stack;
            else next = std::move(stack);
            next.push_back(alts[i]);
            close(next, out, depth + 1);
        }
        return;
    }
}

int32_t StateTable::intern(std::vector<std::vector<uint32_t>>& stacks) {
    std::sort(stacks.begin(), stacks.end());
    stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());
    std::vector<uint32_t> key;
    for (const auto& s : stacks) {
        key.push_back(static_cast<uint32_t>(s.size()));
        key.insert(key.end(), s.begin(), s.end());
    }
    const auto found = index_.find(key);
    if (found != index_.end()) return found->second;
    State st;
    for (const auto& s : stacks) {
        if (s.empty()) {
            st.accept = true;
            continue;
        }
        const ByteSet& set = program_->sets[program_->elems[s.back()].arg];
        for (int w = 0; w < 4; ++w) st.next[w] |= set[w];
    }
    int count = 0;
    for (const uint64_t w : st.next) count += std::popcount(w);
    st.next_count = static_cast<uint16_t>(count);
    const auto id = static_cast<int32_t>(states_.size());
    const auto [it, inserted] = index_.emplace(std::move(key), id);
    st.key = &it->first;
    states_.push_back(std::move(st));
    return id;
}

int32_t StateTable::initial() {
    if (initial_ != kUnknown) return initial_;
    std::vector<std::vector<uint32_t>> stacks;
    for (const uint32_t alt : program_->rules[program_->root]) {
        std::vector<uint32_t> s{alt};
        close(s, stacks, 0);
    }
    initial_ = intern(stacks);
    return initial_;
}

int32_t StateTable::compute(int32_t state, uint8_t byte) {
    const std::vector<uint32_t> key = *states_[static_cast<size_t>(state)].key;  // copy: interning may rehash
    std::vector<std::vector<uint32_t>> stacks;
    for (size_t i = 0; i < key.size();) {
        const uint32_t len = key[i++];
        if (len > 0) {
            const uint32_t top = key[i + len - 1];
            if (has_byte(program_->sets[program_->elems[top].arg], byte)) {
                std::vector<uint32_t> s(key.begin() + static_cast<ptrdiff_t>(i),
                                        key.begin() + static_cast<ptrdiff_t>(i + len));
                s.back() = top + 1;
                close(s, stacks, 0);
            }
        }
        i += len;
    }
    if (stacks.empty()) return kDead;
    return intern(stacks);
}

int32_t StateTable::advance(int32_t state, uint8_t byte) {
    State& st = states_[static_cast<size_t>(state)];
    if (!has_byte(st.next, byte)) return kDead;
    if (!st.row) {
        st.row = std::make_unique<int32_t[]>(256);
        std::fill_n(st.row.get(), 256, kUnknown);
    }
    int32_t to = st.row[byte];
    if (to != kUnknown) return to;
    int32_t* row = st.row.get();  // stays valid: interning moves State objects, not their rows
    to = compute(state, byte);
    row[byte] = to;
    return to;
}

int32_t StateTable::advance(int32_t state, std::string_view bytes) {
    for (const char c : bytes) {
        if (state < 0) return kDead;
        state = advance(state, static_cast<uint8_t>(c));
    }
    return state;
}

int StateTable::forced_byte(int32_t state) const {
    const State& st = states_[static_cast<size_t>(state)];
    if (st.accept || st.next_count != 1) return -1;
    for (int w = 0; w < 4; ++w) {
        if (st.next[w] != 0) return w * 64 + std::countr_zero(st.next[w]);
    }
    return -1;
}

std::string StateTable::forced_text(int32_t state, size_t max_bytes) {
    std::string out;
    size_t whole = 0;  // bytes up to the last complete character
    int pending = 0;   // continuation bytes the current character still needs
    while (out.size() < max_bytes) {
        const int b = forced_byte(state);
        if (b < 0) break;
        out += static_cast<char>(b);
        state = advance(state, static_cast<uint8_t>(b));
        if (pending > 0) {
            --pending;
        } else if (b >= 0xF0) {
            pending = 3;
        } else if (b >= 0xE0) {
            pending = 2;
        } else if (b >= 0xC0) {
            pending = 1;
        }
        if (pending == 0) whole = out.size();
        if (state < 0) break;
    }
    out.resize(whole);
    return out;
}

const std::vector<uint64_t>& StateTable::mask(int32_t state, const VocabTrie& trie) {
    if (!states_[static_cast<size_t>(state)].mask.empty()) return states_[static_cast<size_t>(state)].mask;
    const size_t words = (static_cast<size_t>(trie.vocab_size) + 63) / 64;
    if ((masks_ + 1) * words * 8 > kMaxMaskBytes) {  // drop every cached mask: they are rebuilt on demand
        for (State& st : states_) std::vector<uint64_t>().swap(st.mask);
        masks_ = 0;
    }
    std::vector<uint64_t> bits(words, 0);
    // One pass over the trie in depth-first order: `at[d]` is the state after
    // the first d bytes of the current node's path; a rejected byte skips its
    // whole subtree.
    std::vector<int32_t> at(static_cast<size_t>(trie.max_depth) + 1);
    at[0] = state;
    const VocabTrie::Node* nodes = trie.nodes.data();
    const auto n = static_cast<uint32_t>(trie.nodes.size());
    for (uint32_t i = 0; i < n;) {
        const VocabTrie::Node& node = nodes[i];
        const int32_t to = advance(at[node.depth - 1], node.byte);
        if (to < 0) {
            i = node.subtree_end;
            continue;
        }
        at[node.depth] = to;
        for (uint32_t t = node.tok_begin; t < node.tok_end; ++t) {
            const auto id = static_cast<uint32_t>(trie.tokens[t]);
            bits[id >> 6] |= uint64_t{1} << (id & 63);
        }
        ++i;
    }
    if (states_[static_cast<size_t>(state)].accept) {
        for (const int32_t id : trie.end_of_generation) {
            bits[static_cast<uint32_t>(id) >> 6] |= uint64_t{1} << (static_cast<uint32_t>(id) & 63);
        }
    }
    ++masks_;
    states_[static_cast<size_t>(state)].mask = std::move(bits);
    return states_[static_cast<size_t>(state)].mask;
}

size_t StateTable::apply_mask(int32_t state, const VocabTrie& trie, std::span<float> logits) {
    const std::vector<uint64_t>& bits = mask(state, trie);
    constexpr float kNegInf = -std::numeric_limits<float>::infinity();
    size_t allowed = 0;
    const size_t n = logits.size();
    for (size_t w = 0; w * 64 < n; ++w) {
        const uint64_t word = w < bits.size() ? bits[w] : 0;
        const size_t end = std::min(n, w * 64 + 64);
        if (word == ~uint64_t{0} && end == w * 64 + 64) {
            allowed += 64;
            continue;
        }
        allowed += static_cast<size_t>(std::popcount(word));
        for (size_t i = w * 64; i < end; ++i) {
            if (((word >> (i & 63)) & 1) == 0) logits[i] = kNegInf;
        }
    }
    return allowed;
}

size_t StateTable::memory_bytes() const noexcept {
    size_t bytes = 0;
    for (const State& st : states_) {
        bytes += sizeof(State) + st.key->size() * 8 + (st.row ? 1024 : 0) + st.mask.size() * 8;
    }
    return bytes;
}

int32_t StateTable::compact(int32_t keep) {
    if (states_.size() <= kMaxStates && memory_bytes() <= kMaxTableBytes) return keep;
    const std::vector<uint32_t> key = *states_[static_cast<size_t>(keep)].key;
    states_.clear();
    index_.clear();
    masks_ = 0;
    initial_ = kUnknown;
    std::vector<std::vector<uint32_t>> stacks;
    for (size_t i = 0; i < key.size();) {
        const uint32_t len = key[i++];
        stacks.emplace_back(key.begin() + static_cast<ptrdiff_t>(i), key.begin() + static_cast<ptrdiff_t>(i + len));
        i += len;
    }
    return intern(stacks);
}

}  // namespace liyab::grammar

// ---------------------------------------------------------------------------
// Public Grammar
// ---------------------------------------------------------------------------

namespace liyab {

Result<Grammar> Grammar::from_json_schema(std::string_view schema_json, const JsonFormat& format) {
    grammar::Builder b;
    auto node = grammar::json_schema(b, schema_json, format);
    if (!node) return node.status();
    const uint32_t root = b.add_rule("root");
    b.define(root, *node);
    auto program = b.build(root);
    if (!program) return program.status();
    return Grammar(std::make_shared<const grammar::Program>(std::move(program).value()));
}

Grammar Grammar::json(const JsonFormat& format) {
    grammar::Builder b;
    const grammar::Builder::Node node = grammar::json_value(b, format);
    const uint32_t root = b.add_rule("root");
    b.define(root, node);
    return Grammar(std::make_shared<const grammar::Program>(std::move(b.build(root)).value()));
}

Result<Grammar> Grammar::parse(std::string_view text) {
    grammar::Builder b;
    auto root = grammar::parse_text(b, text);
    if (!root) return root.status();
    auto program = b.build(*root);
    if (!program) return program.status();
    return Grammar(std::make_shared<const grammar::Program>(std::move(program).value()));
}

bool Grammar::matches(std::string_view text) const {
    grammar::StateTable table(program_);
    const int32_t s = table.advance(table.initial(), text);
    return s >= 0 && table.accepting(s);
}

bool Grammar::accepts_prefix(std::string_view text) const {
    grammar::StateTable table(program_);
    return table.advance(table.initial(), text) >= 0;
}

}  // namespace liyab
