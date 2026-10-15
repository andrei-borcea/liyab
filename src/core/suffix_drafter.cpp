#include "core/suffix_drafter.h"

#include <algorithm>
#include <functional>
#include <numeric>

namespace liyab {

namespace {

// Occurrences the continuation tree is built from: the most recent ones of
// the live sequence first, then per stored sequence its latest ones. A
// matched suffix is usually rare; this only bounds the work for a short,
// common one.
constexpr size_t kMaxOccurrences = 512;
constexpr size_t kMaxStoredOccurrences = 128;
// Weight of the prior in a calibration bin, in observations, and the count
// of verified drafts at which a bin halves its counts (it follows the text
// and the sampling settings instead of averaging all of history).
constexpr float kPriorWeight = 2.0f;
constexpr float kBinHalfLife = 512.0f;

}  // namespace

SuffixDrafter::SuffixDrafter(Limits limits) : limits_(limits) {
    limits_.max_stored_tokens = std::max<int64_t>(0, limits_.max_stored_tokens);
    limits_.max_sequences = std::max(0, limits_.max_sequences);
    limits_.min_sequence = std::max(kMinMatch + 1, limits_.min_sequence);
}

void SuffixDrafter::build_suffix_array(std::span<const int32_t> tokens, int32_t depth, std::vector<int32_t>& sa) {
    const auto n = static_cast<int32_t>(tokens.size());
    sa.resize(static_cast<size_t>(n));
    if (n == 0) return;
    // Dense ranks of the token ids (same order), so every later sort is a
    // counting sort over at most n keys whatever the vocabulary size.
    std::vector<int32_t> ids(tokens.begin(), tokens.end());
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::vector<int32_t> rank(static_cast<size_t>(n)), next(static_cast<size_t>(n)), tmp(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        rank[static_cast<size_t>(i)] = static_cast<int32_t>(
            std::lower_bound(ids.begin(), ids.end(), tokens[static_cast<size_t>(i)]) - ids.begin());
    }
    auto classes = static_cast<int32_t>(ids.size());
    std::vector<int32_t> count(static_cast<size_t>(std::max(classes, n)) + 1);
    auto counting_sort = [&](const std::vector<int32_t>& order) {  // stable, by rank
        std::fill(count.begin(), count.begin() + classes + 1, 0);
        for (int32_t i = 0; i < n; ++i) ++count[static_cast<size_t>(rank[static_cast<size_t>(i)]) + 1];
        for (int32_t c = 0; c < classes; ++c) count[static_cast<size_t>(c) + 1] += count[static_cast<size_t>(c)];
        for (const int32_t i : order) sa[static_cast<size_t>(count[static_cast<size_t>(rank[static_cast<size_t>(i)])]++)] = i;
    };
    std::iota(tmp.begin(), tmp.end(), 0);
    counting_sort(tmp);
    // Prefix doubling: ranks of the first h tokens -> of the first 2h. The
    // order by second key comes for free: suffixes with nothing at i + h
    // first (they are shorter), then i = sa[k] - h in the current order.
    for (int32_t h = 1; h < depth && classes < n; h *= 2) {
        int32_t p = 0;
        for (int32_t i = std::max(0, n - h); i < n; ++i) tmp[static_cast<size_t>(p++)] = i;
        for (int32_t k = 0; k < n; ++k) {
            if (const int32_t j = sa[static_cast<size_t>(k)]; j >= h) tmp[static_cast<size_t>(p++)] = j - h;
        }
        counting_sort(tmp);
        auto second = [&](int32_t i) { return i + h < n ? rank[static_cast<size_t>(i + h)] : -1; };
        next[static_cast<size_t>(sa[0])] = 0;
        for (int32_t k = 1; k < n; ++k) {
            const int32_t a = sa[static_cast<size_t>(k - 1)], b = sa[static_cast<size_t>(k)];
            const bool same = rank[static_cast<size_t>(a)] == rank[static_cast<size_t>(b)] && second(a) == second(b);
            next[static_cast<size_t>(b)] = next[static_cast<size_t>(a)] + (same ? 0 : 1);
        }
        rank.swap(next);
        classes = rank[static_cast<size_t>(sa[static_cast<size_t>(n - 1)])] + 1;
    }
}

void SuffixDrafter::sync(std::span<const int32_t> sequence) {
    const size_t before = live_.size();
    const size_t most = std::min(before, sequence.size());
    const size_t same = static_cast<size_t>(
        std::mismatch(live_.begin(), live_.begin() + static_cast<std::ptrdiff_t>(most), sequence.begin()).first -
        live_.begin());
    if (same < before) {
        if (before - same >= static_cast<size_t>(kMinDropped) && before >= static_cast<size_t>(limits_.min_sequence)) {
            store(live_);
        }
        live_.resize(same);
        for (Stored& s : stored_) s.common = std::min(s.common, same);
    }
    const size_t old = live_.size();
    live_.insert(live_.end(), sequence.begin() + static_cast<std::ptrdiff_t>(old), sequence.end());
    // A stored sequence followed by the live one so far may still be one of
    // its prefixes (then it is shadowed): extend its common prefix.
    for (Stored& s : stored_) {
        if (!s.prefix_aligned || s.common != old) continue;
        while (s.common < s.tokens.size() && s.common < live_.size() && s.tokens[s.common] == live_[s.common]) {
            ++s.common;
        }
    }
}

void SuffixDrafter::store(std::span<const int32_t> sequence) {
    if (limits_.max_sequences == 0 || limits_.max_stored_tokens < limits_.min_sequence) return;
    // A stored sequence this one continues says nothing more: replace it. One
    // that continues this sequence already holds it all: just touch it.
    for (size_t i = 0; i < stored_.size();) {
        Stored& s = stored_[i];
        if (s.prefix_aligned && s.tokens.size() >= sequence.size() &&
            std::equal(sequence.begin(), sequence.end(), s.tokens.begin())) {
            s.used = ++tick_;
            return;
        }
        if (s.prefix_aligned && s.tokens.size() <= sequence.size() &&
            std::equal(s.tokens.begin(), s.tokens.end(), sequence.begin())) {
            stored_tokens_ -= static_cast<int64_t>(s.tokens.size());
            stored_.erase(stored_.begin() + static_cast<std::ptrdiff_t>(i));
            continue;
        }
        ++i;
    }
    Stored entry;
    const bool cut = static_cast<int64_t>(sequence.size()) > limits_.max_stored_tokens;
    if (cut) sequence = sequence.last(static_cast<size_t>(limits_.max_stored_tokens));
    entry.tokens.assign(sequence.begin(), sequence.end());
    entry.prefix_aligned = !cut;
    // Sorted one token past the longest pattern: then the occurrence of a
    // pattern that ends the sequence (no continuation) sorts first among the
    // pattern's occurrences, where range() skips it.
    build_suffix_array(entry.tokens, kMaxMatch + 1, entry.sa);
    entry.used = ++tick_;
    if (entry.prefix_aligned) {
        const size_t most = std::min(entry.tokens.size(), live_.size());
        entry.common = static_cast<size_t>(
            std::mismatch(entry.tokens.begin(), entry.tokens.begin() + static_cast<std::ptrdiff_t>(most), live_.begin())
                .first -
            entry.tokens.begin());
    }
    stored_tokens_ += static_cast<int64_t>(entry.tokens.size());
    stored_.push_back(std::move(entry));
    // Least recently used out until both caps hold (the new one is the most
    // recent, and fits by itself).
    while (stored_tokens_ > limits_.max_stored_tokens || static_cast<int32_t>(stored_.size()) > limits_.max_sequences) {
        auto lru = std::min_element(stored_.begin(), stored_.end(),
                                    [](const Stored& a, const Stored& b) { return a.used < b.used; });
        stored_tokens_ -= static_cast<int64_t>(lru->tokens.size());
        stored_.erase(lru);
    }
}

void SuffixDrafter::range(const Stored& s, std::span<const int32_t> pattern, int32_t& lo, int32_t& hi) const {
    const auto n = static_cast<int32_t>(s.tokens.size());
    const auto m = static_cast<int32_t>(pattern.size());
    // Compares the suffix at `pos` with the pattern on its first m tokens; a
    // suffix that ends first is smaller (as the suffix array orders them).
    auto compare = [&](int32_t pos) {
        for (int32_t k = 0; k < m; ++k) {
            if (pos + k >= n) return -1;
            const int32_t a = s.tokens[static_cast<size_t>(pos + k)], b = pattern[static_cast<size_t>(k)];
            if (a != b) return a < b ? -1 : 1;
        }
        return 0;
    };
    lo = static_cast<int32_t>(std::partition_point(s.sa.begin(), s.sa.end(), [&](int32_t p) { return compare(p) < 0; }) -
                              s.sa.begin());
    hi = static_cast<int32_t>(
        std::partition_point(s.sa.begin() + lo, s.sa.end(), [&](int32_t p) { return compare(p) == 0; }) - s.sa.begin());
    // The occurrence at the very end has no continuation; it sorts first.
    if (lo < hi && s.sa[static_cast<size_t>(lo)] + m == n) ++lo;
}

int32_t SuffixDrafter::longest_match(const Stored& s, int32_t& lo, int32_t& hi) const {
    const auto n = static_cast<int32_t>(live_.size());
    // "Occurs with a continuation" is monotone in the length (a longer
    // match's suffixes occur too, followed by the same token), so the longest
    // length is a binary search over lengths.
    int32_t good = kMinMatch, bad = std::min(kMaxMatch, n) + 1;
    int32_t a = 0, b = 0;
    // The shortest first: most lookups end there.
    range(s, std::span<const int32_t>(live_).last(static_cast<size_t>(kMinMatch)), lo, hi);
    if (lo >= hi) return 0;
    while (bad - good > 1) {
        const int32_t len = (good + bad) / 2;
        range(s, std::span<const int32_t>(live_).last(static_cast<size_t>(len)), a, b);
        if (a < b) {
            good = len;
            lo = a;
            hi = b;
        } else {
            bad = len;
        }
    }
    return good;
}

uint16_t SuffixDrafter::bin_of(int32_t match, float share, int32_t support, bool live) {
    const int32_t m = match <= 2 ? 0 : match <= 4 ? 1 : match <= 8 ? 2 : match <= 16 ? 3 : 4;
    const int32_t f = share >= 1.0f ? 3 : share >= 0.8f ? 2 : share >= 0.5f ? 1 : 0;
    const int32_t c = support <= 1 ? 0 : support <= 3 ? 1 : 2;
    return static_cast<uint16_t>(((m * 4 + f) * 3 + c) * 2 + (live ? 1 : 0));
}

float SuffixDrafter::prior(int32_t match, float share) {
    // Before any verification: as likely as the occurrences agree, less so
    // for a short match (a 2-token match is a coin toss, 32 tokens near sure).
    return share * static_cast<float>(match) / static_cast<float>(match + 2);
}

float SuffixDrafter::acceptance(uint16_t bin, int32_t match, float share) const {
    const Bin& c = coarse_[bin / kFinePerCoarse];
    const float coarse = (c.accepted + kPriorWeight * prior(match, share)) / (c.verified + kPriorWeight);
    const Bin& b = bins_[bin];
    return (b.accepted + kPriorWeight * coarse) / (b.verified + kPriorWeight);
}

const SuffixDrafter::Draft& SuffixDrafter::propose(int32_t max_tokens) {
    Draft& out = draft_;
    out.tokens.clear();
    out.survival.clear();
    out.bins.clear();
    out.match = 0;
    const auto n = static_cast<int32_t>(live_.size());
    if (max_tokens <= 0 || n <= kMinMatch) return out;

    // 1. Live sequence: for every earlier position p whose preceding tokens
    // end like the sequence does, the length of that common suffix
    // (continuation at p). Newest first.
    occurrences_.clear();
    int32_t best = 0;
    const int32_t last = live_[static_cast<size_t>(n - 1)];
    const int32_t* t = live_.data();
    std::vector<std::pair<int32_t, int32_t>>& found = live_found_;
    found.clear();
    for (int32_t p = n - 1; p >= kMinMatch; --p) {
        if (t[p - 1] != last) continue;
        int32_t len = 1;
        while (len < kMaxMatch && p - 1 - len >= 0 && t[p - 1 - len] == t[n - 1 - len]) ++len;
        if (len < kMinMatch || len < best) continue;
        best = len;
        found.emplace_back(p, len);
    }
    // 2. Stored sequences (not those the live one contains).
    stored_found_.clear();
    for (size_t i = 0; i < stored_.size(); ++i) {
        if (shadowed(stored_[i])) continue;
        int32_t lo = 0, hi = 0;
        const int32_t len = longest_match(stored_[i], lo, hi);
        if (len >= kMinMatch) {
            stored_found_.push_back({static_cast<int32_t>(i), len, lo, hi});
            best = std::max(best, len);
        }
    }
    if (best < kMinMatch) return out;
    out.match = best;

    // 3. The occurrences of the longest match, wherever it occurred.
    for (const auto& [p, len] : found) {
        if (len == best && occurrences_.size() < kMaxOccurrences) occurrences_.push_back({t, p, n, -1});
    }
    for (const StoredMatch& m : stored_found_) {
        if (m.length != best) continue;
        const Stored& s = stored_[static_cast<size_t>(m.index)];
        positions_.assign(s.sa.begin() + m.lo, s.sa.begin() + m.hi);
        if (positions_.size() > kMaxStoredOccurrences) {  // keep the latest
            std::nth_element(positions_.begin(), positions_.begin() + kMaxStoredOccurrences, positions_.end(),
                             std::greater<>());
            positions_.resize(kMaxStoredOccurrences);
        }
        std::sort(positions_.begin(), positions_.end(), std::greater<>());
        for (const int32_t start : positions_) {
            if (occurrences_.size() >= kMaxOccurrences) break;
            occurrences_.push_back({s.tokens.data(), start + best, static_cast<int32_t>(s.tokens.size()), m.index});
        }
    }

    // 4. The best branch of the continuation tree: at each depth the token
    // most occurrences continue with (ties: one the live sequence supports,
    // then the newest), keeping only the occurrences that agree.
    float survival = 1.0f;
    for (int32_t depth = 0; depth < max_tokens; ++depth) {
        tally_.clear();
        int32_t active = 0;
        for (const Occurrence& o : occurrences_) {
            if (o.pos + depth >= o.end) continue;
            ++active;
            const int32_t token = o.tokens[o.pos + depth];
            auto it = std::find_if(tally_.begin(), tally_.end(), [&](const Tally& x) { return x.token == token; });
            if (it == tally_.end()) {
                tally_.push_back({token, 1, o.source < 0});
            } else {
                ++it->count;
                it->live = it->live || o.source < 0;
            }
        }
        if (active == 0) break;
        const Tally* choice = &tally_.front();
        for (const Tally& x : tally_) {
            if (x.count > choice->count || (x.count == choice->count && x.live && !choice->live)) choice = &x;
        }
        const int32_t token = choice->token;
        const float share = static_cast<float>(choice->count) / static_cast<float>(active);
        const int32_t match = best + depth;
        const uint16_t bin = bin_of(match, share, choice->count, choice->live);
        survival *= acceptance(bin, match, share);
        out.tokens.push_back(token);
        out.survival.push_back(survival);
        out.bins.push_back(bin);
        // Keep the agreeing occurrences; their sources were useful (LRU).
        size_t kept = 0;
        for (const Occurrence& o : occurrences_) {
            if (o.pos + depth >= o.end || o.tokens[o.pos + depth] != token) continue;
            if (depth == 0 && o.source >= 0) stored_[static_cast<size_t>(o.source)].used = tick_ + 1;
            occurrences_[kept++] = o;
        }
        occurrences_.resize(kept);
        if (depth == 0) ++tick_;
    }
    return out;
}

void SuffixDrafter::feedback(const Draft& draft, int32_t verified, int32_t accepted) {
    verified = std::min(verified, static_cast<int32_t>(draft.bins.size()));
    // Draft i was tested only if every one before it was accepted.
    for (int32_t i = 0; i < verified && i <= accepted; ++i) {
        const uint16_t bin = draft.bins[static_cast<size_t>(i)];
        for (Bin* b : {&bins_[bin], &coarse_[bin / kFinePerCoarse]}) {
            b->verified += 1.0f;
            if (i < accepted) b->accepted += 1.0f;
            if (b->verified >= kBinHalfLife) {
                b->verified *= 0.5f;
                b->accepted *= 0.5f;
            }
        }
    }
}

size_t SuffixDrafter::memory_bytes() const noexcept {
    size_t bytes = live_.capacity() * sizeof(int32_t);
    for (const Stored& s : stored_) bytes += (s.tokens.capacity() + s.sa.capacity()) * sizeof(int32_t);
    return bytes;
}

}  // namespace liyab
