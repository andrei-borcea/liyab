// Liyab — suffix drafter: speculative drafts from the conversation and from
// earlier ones (internal).
//
// What a phone assistant writes repeats what it has read and written: code it
// was asked to change, a text it summarizes, a tool's output it quotes, the
// phrasing of earlier replies. The drafter finds the longest suffix of the
// current token sequence (at most kMaxMatch tokens) that occurred before,
// in the live sequence or in a store of earlier sequences, and proposes how
// those occurrences continue: their continuation tree, each node a token
// some of them continue with. propose() gives its best branch (at each node
// the token most occurrences agree on); propose_tree() the tree's likeliest
// nodes, alternative branches included, which the decoder verifies in one
// pass (Transformer::forward_tree). Every drafted token comes with a
// calibrated probability of being accepted, so the speculative decoder can
// verify just as many drafts as pay off (DraftBudget::choose(survival)).
//
// Structure:
//  * the live sequence (the tokens the target holds, then the one it will
//    read next) is matched by a backward scan, O(length) per step: it changes
//    every step and is cut back by rewinds, and a scan costs microseconds
//    next to a forward pass;
//  * earlier sequences, set aside when the live one stops continuing them (a
//    new conversation, a regenerated reply, an edited turn), are kept with a
//    suffix array each, sorted on their first kMaxMatch + 1 tokens (prefix
//    doubling with radix sorts, O(n log kMaxMatch)), so the occurrences of a
//    pattern are a binary search away. They are bounded: at most
//    Limits::max_stored_tokens tokens (8 bytes each with the suffix array) in
//    at most Limits::max_sequences sequences, least recently used out first.
//    A stored sequence the live one continues (a conversation taken up again)
//    is replaced when the live one is set aside, and is not matched while the
//    live sequence contains it, so text never counts twice.
//
// Confidence: each drafted token falls in a bin by match length, the share
// and number of occurrences that agree on it, whether the live sequence
// supports it and whether it is its parent's likeliest child or an
// alternative (a tree's sibling); per bin, the accepted / verified counts give the probability
// that it is accepted once its parent (the token before it on its branch)
// is. A token's probability is the product along its branch.
// Two levels, so a few verifications already tell: a coarse bin (match length
// and share only, its prior from the two) is the prior of the fine bins in
// it. Counts are halved now and then, so they follow the text and the
// sampling settings. Calibrated from the verification outcomes reported
// through feedback(), on every device and model alike.
//
// Drafts never change the output: the decoder verifies them. Not thread-safe:
// owned by one SpeculativeDecoder.
#ifndef LIYAB_CORE_SUFFIX_DRAFTER_H
#define LIYAB_CORE_SUFFIX_DRAFTER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace liyab {

class SuffixDrafter {
public:
    struct Limits {
        // Tokens kept across all stored sequences (~8 bytes each: the token
        // and its suffix array entry); 262144 tokens are 2 MiB. A longer
        // sequence keeps its last max_stored_tokens tokens.
        int64_t max_stored_tokens = int64_t{1} << 18;
        int32_t max_sequences = 64;
        // Sequences shorter than this are not stored.
        int32_t min_sequence = 32;
    };

    // Matched suffixes are 2..32 tokens long: a single token repeats too
    // often to predict anything, and past 32 tokens a match is already as
    // specific as it gets.
    static constexpr int32_t kMinMatch = 2;
    static constexpr int32_t kMaxMatch = 32;
    // Live sequence: tokens the live sequence loses at once (a new
    // conversation, a regenerated reply) before it is stored; fewer are a
    // rollback or a cut, not text worth keeping twice.
    static constexpr int32_t kMinDropped = 16;

    struct Draft {
        std::vector<int32_t> tokens;
        // parents[i]: the drafted token that token i continues, an earlier
        // one (i - 1 along propose()'s branch), or -1 when it continues the
        // live sequence. Siblings (tokens of one parent) differ.
        std::vector<int32_t> parents;
        // survival[i]: estimated probability that token i is accepted, i.e.
        // its branch up to it: for propose(), tokens[0..i] all accepted. Non-
        // increasing, so the first j tokens are the likeliest j and (a
        // parent before its children) form a tree themselves.
        std::vector<float> survival;
        std::vector<uint16_t> bins;  // calibration bin of each token (for feedback)
        int32_t match = 0;           // length of the matched suffix (0: no draft)
    };

    SuffixDrafter() : SuffixDrafter(Limits{}) {}
    explicit SuffixDrafter(Limits limits);

    // Makes `sequence` the live sequence: usually it extends the previous one
    // by the tokens accepted since. When it does not, the previous one is
    // stored first if at least kMinDropped of its tokens are dropped.
    // O(length) (a comparison of the common prefix).
    void sync(std::span<const int32_t> sequence);
    // Up to `max_tokens` tokens continuing the live sequence, one branch
    // (none when no suffix of at least kMinMatch tokens occurred before).
    // The draft is valid until the next call to propose*().
    const Draft& propose(int32_t max_tokens);
    // Up to `max_tokens` tokens of the continuation tree, the likeliest
    // first (best-first: a node's children become candidates once it is
    // taken), so any first j of them are the j likeliest and form a tree.
    // The siblings' acceptance probabilities (exclusive events: the target
    // picks one token) are scaled down when their estimates sum above 1.
    // With one candidate per node this is propose()'s branch. Valid until the
    // next call to propose*().
    const Draft& propose_tree(int32_t max_tokens);
    // The decoder verified the first `verified` tokens of `draft` and the
    // first `accepted` of them were accepted (so token `accepted`, when
    // verified, was rejected); later ones were never tested. For a branch
    // (propose()).
    void feedback(const Draft& draft, int32_t verified, int32_t accepted);
    // The same for any draft: of its first `verified` tokens, those on
    // `path` (draft indices, ascending: the accepted branch) were accepted.
    // A token was tested when its parent was accepted (or it continues the
    // live sequence): at the parent's row the target picked one token, so
    // its other children were rejected; the rest were never tested.
    void feedback(const Draft& draft, int32_t verified, std::span<const int32_t> path);

    [[nodiscard]] std::span<const int32_t> live() const noexcept { return live_; }
    [[nodiscard]] size_t stored_sequences() const noexcept { return stored_.size(); }
    [[nodiscard]] int64_t stored_tokens() const noexcept { return stored_tokens_; }
    // Memory held by the store and the live sequence, in bytes (capacity).
    [[nodiscard]] size_t memory_bytes() const noexcept;
    // Estimated acceptance probability of a drafted token in a calibration
    // bin (see Draft::bins), for tests and diagnostics.
    [[nodiscard]] float acceptance(uint16_t bin, int32_t match, float share) const;

    // The suffix array of `tokens` sorted on their first `depth` tokens
    // (positions of equal prefixes in any order): sa[i] is the start of the
    // i-th smallest suffix. Exposed for tests.
    static void build_suffix_array(std::span<const int32_t> tokens, int32_t depth, std::vector<int32_t>& sa);

private:
    struct Stored {
        std::vector<int32_t> tokens;
        std::vector<int32_t> sa;
        uint64_t used = 0;        // LRU tick
        size_t common = 0;        // length of the common prefix with the live sequence
        bool prefix_aligned = true;  // false when its start was cut off (never a prefix of the live one)
    };
    // One earlier occurrence of the matched suffix: its continuation starts
    // at tokens[pos] and ends before tokens[end].
    struct Occurrence {
        const int32_t* tokens;
        int32_t pos;
        int32_t end;
        int32_t source;  // -1: live sequence, else index into stored_
    };
    struct StoredMatch {
        int32_t index;   // into stored_
        int32_t length;  // longest match there
        int32_t lo, hi;  // its suffix array range
    };
    struct Tally {
        int32_t token;
        int32_t count;
        bool live;
    };
    // A node of the continuation tree being built: its token at `depth`
    // (0: the first drafted token), the occurrences continuing with its
    // branch (order_[begin, end)), its branch's probability.
    struct Node {
        int32_t token;
        int32_t depth;
        int32_t parent;  // draft index of its parent, -1: the live sequence
        int32_t begin, end;
        float probability;
        uint16_t bin;
        uint32_t sequence;  // creation order: the tie-break among equal probabilities
    };
    struct Bin {
        float accepted = 0.0f;
        float verified = 0.0f;
    };
    // rank x match x share x support x live. Rank: the node's likeliest
    // child or an alternative one (a tree's sibling): an alternative is
    // rejected more often than the likeliest child of the same share (a tie
    // at 50% among them), so it is calibrated apart, and the branch's bins
    // are those they were before trees.
    static constexpr size_t kBins = 2 * 5 * 4 * 3 * 2;
    static constexpr size_t kFinePerCoarse = 3 * 2;    // support x live

    void store(std::span<const int32_t> sequence);
    // propose() and propose_tree(): the tree's likeliest nodes, best first,
    // with every candidate child (`branching`) or only each node's best one.
    const Draft& build(int32_t max_tokens, bool branching);
    // The children of a node of depth `depth` - 1 (the live sequence for
    // depth 0) whose occurrences are order_[begin, end), as candidates
    // (candidates_), its probability `probability`; order_[begin, end) is
    // reordered so each child's occurrences are a run of their own. Only the
    // best child unless `branching`.
    void expand(int32_t parent, int32_t depth, int32_t begin, int32_t end, float probability, bool branching);
    [[nodiscard]] bool shadowed(const Stored& s) const noexcept {
        return s.prefix_aligned && s.common == s.tokens.size();
    }
    // Longest suffix (kMinMatch..kMaxMatch) of the live sequence occurring in
    // `s` with at least one token after it; [lo, hi) its suffix array range.
    int32_t longest_match(const Stored& s, int32_t& lo, int32_t& hi) const;
    void range(const Stored& s, std::span<const int32_t> pattern, int32_t& lo, int32_t& hi) const;
    static uint16_t bin_of(int32_t match, float share, int32_t support, bool live, bool alternative = false);
    static bool heap_less(const Node& a, const Node& b) noexcept;
    static float prior(int32_t match, float share);

    Limits limits_;
    std::vector<int32_t> live_;
    std::vector<Stored> stored_;
    int64_t stored_tokens_ = 0;
    uint64_t tick_ = 0;
    std::array<Bin, kBins> bins_{};
    std::array<Bin, kBins / kFinePerCoarse> coarse_{};  // by rank, match and share only
    Draft draft_;
    std::vector<Occurrence> occurrences_;  // scratch
    std::vector<Tally> tally_;             // scratch
    std::vector<std::pair<int32_t, int32_t>> live_found_;  // scratch: (continuation, match length)
    std::vector<StoredMatch> stored_found_;                // scratch
    std::vector<int32_t> positions_;                       // scratch
    std::vector<int32_t> order_;                           // scratch: occurrences_ indices, grouped by node
    std::vector<int32_t> rank_;                            // scratch: per occurrence, its child's rank
    std::vector<int32_t> children_;                        // scratch: tally_ indices, best child first
    int32_t match_ = 0;                                    // scratch: the matched suffix's length (build)
    std::vector<Node> candidates_;                         // scratch: a max-heap by probability
    uint32_t sequence_ = 0;                                // scratch: Node::sequence counter
    std::vector<uint8_t> on_path_;                         // scratch: feedback()'s accepted tokens
};

}  // namespace liyab

#endif  // LIYAB_CORE_SUFFIX_DRAFTER_H
