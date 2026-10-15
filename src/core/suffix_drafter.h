// Liyab — suffix drafter: speculative drafts from the conversation and from
// earlier ones (internal).
//
// What a phone assistant writes repeats what it has read and written: code it
// was asked to change, a text it summarizes, a tool's output it quotes, the
// phrasing of earlier replies. The drafter finds the longest suffix of the
// current token sequence (at most kMaxMatch tokens) that occurred before,
// in the live sequence or in a store of earlier sequences, and proposes the
// continuation most of those occurrences agree on: the best branch of the
// continuation tree, by frequency. Every drafted token comes with a
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
// and number of occurrences that agree on it and whether the live sequence
// supports it; per bin, the accepted / verified counts (with a prior from the
// share and the match length, halved now and then so it follows the text)
// give the probability. Calibrated from the verification outcomes reported
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
        // survival[i]: estimated probability that tokens[0..i] are all
        // accepted (non-increasing).
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
    // Up to `max_tokens` tokens continuing the live sequence (none when no
    // suffix of at least kMinMatch tokens occurred before). The draft is
    // valid until the next call.
    const Draft& propose(int32_t max_tokens);
    // The decoder verified the first `verified` tokens of `draft` and the
    // first `accepted` of them were accepted (so token `accepted`, when
    // verified, was rejected); later ones were never tested.
    void feedback(const Draft& draft, int32_t verified, int32_t accepted);

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
    struct Bin {
        float accepted = 0.0f;
        float verified = 0.0f;
    };
    static constexpr size_t kBins = 5 * 4 * 3 * 2;

    void store(std::span<const int32_t> sequence);
    [[nodiscard]] bool shadowed(const Stored& s) const noexcept {
        return s.prefix_aligned && s.common == s.tokens.size();
    }
    // Longest suffix (kMinMatch..kMaxMatch) of the live sequence occurring in
    // `s` with at least one token after it; [lo, hi) its suffix array range.
    int32_t longest_match(const Stored& s, int32_t& lo, int32_t& hi) const;
    void range(const Stored& s, std::span<const int32_t> pattern, int32_t& lo, int32_t& hi) const;
    static uint16_t bin_of(int32_t match, float share, int32_t support, bool live);
    static float prior(int32_t match, float share);

    Limits limits_;
    std::vector<int32_t> live_;
    std::vector<Stored> stored_;
    int64_t stored_tokens_ = 0;
    uint64_t tick_ = 0;
    std::array<Bin, kBins> bins_{};
    Draft draft_;
    std::vector<Occurrence> occurrences_;  // scratch
    std::vector<Tally> tally_;             // scratch
    std::vector<std::pair<int32_t, int32_t>> live_found_;  // scratch: (continuation, match length)
    std::vector<StoredMatch> stored_found_;                // scratch
    std::vector<int32_t> positions_;                       // scratch
};

}  // namespace liyab

#endif  // LIYAB_CORE_SUFFIX_DRAFTER_H
