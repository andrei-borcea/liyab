// Liyab — speculative decoding (draft-then-verify).
//
// Up to k tokens are drafted cheaply, then the target model scores all of them
// in ONE batched forward pass, so its weights are streamed once for up to k+1
// tokens. Three draft sources, the same verification:
//  * a draft model (same vocabulary) proposes k tokens autoregressively;
//  * context lookup: the most recent earlier occurrence of the last 4..2
//    tokens in the conversation is found and the tokens that followed it are
//    proposed. No model, no memory, works for any target; strong on text that
//    repeats its context (code edits, quotes, tool output, lists);
//  * suffix drafts (core/suffix_drafter.h): the longest suffix (2..32 tokens)
//    of the conversation that occurred before, in it or in earlier
//    conversations of the same engine (a bounded store, least recently used
//    out first), and how its occurrences continue: the likeliest nodes of
//    their continuation tree (alternative branches where they part), or its
//    best branch only (set_tree(false)), each token with a calibrated
//    probability of being accepted.
// Acceptance follows Leviathan et al. (2023): draft token d is kept with
// probability min(1, p(d)/q(d)) (lookup drafts are deterministic: q(d) = 1);
// on the first rejection a replacement is drawn from norm(max(0, p − q)). The
// output distribution is exactly the target's. With greedy sampling this
// reduces to "keep while the target's argmax agrees", so the output is
// token-for-token identical to plain greedy decoding of the target.
// Suffix drafts (deterministic too) are verified by sampling: every row is
// sampled exactly as plain decoding samples a token (one draw per token) and
// a draft is kept while it equals the sample. It is kept with probability
// p(d), as above, and the tokens are the ones plain decoding draws from the
// same seed (the CPU backend computes a row the same bits in any batch),
// whatever was drafted, however many drafts were verified and whatever
// earlier conversations left in the store. Only after a forced continuation
// cut a step short do they differ (the step drew tokens past the cut).
//
// How many drafts a step verifies (at most draft_tokens) is learned at run
// time when adaptive (the default): see core/draft_budget.h. On a model whose
// verification costs about as much per token as decoding (a MoE streaming its
// experts from flash) it settles on plain decoding; where verification is
// cheap it drafts. Suffix drafts choose per step from their acceptance
// probabilities and the measured cost of a step per draft count
// (DraftBudget::choose(survival)): a confident draft is verified at length,
// a doubtful one not at all. With sampling from a draft model or lookup, the
// tokens drawn depend on that choice (their distribution does not);
// set_adaptive(false) keeps k fixed and the output reproducible for a fixed
// seed. Suffix drafts are reproducible either way (see above).
//
// Rejected tokens are rolled back with Transformer::truncate(): recurrent
// (hybrid) models need a rollback window of at least k + 1
// (Transformer::set_rollback_window), which Engine sets.
//
// Tree verification (suffix drafts): the drafts form a tree, `last` its
// root, laid out depth first with the likeliest child first, and one
// Transformer::forward_tree() pass scores every node as a linear pass over
// its branch would (each node attends to the context and its ancestors;
// hybrid models continue each branch from its parent's recurrent state).
// From the root down, each row is sampled once and the child holding the
// sampled token (siblings differ) is kept, so the tokens are plain
// decoding's, as for a branch; Transformer::keep_path() then moves the kept
// branch into place. The k drafts verified are the tree's likeliest nodes
// (a node's probability: its branch's), so the expected tokens of a step
// are 1 + the sum of their probabilities, which DraftBudget weighs against
// the measured cost of a pass of k + 1 tokens, the same whether they form a
// branch or a tree. A tree that is a single branch runs as a linear pass.
//
// This is an engine component: Engine wires it up when
// EngineConfig::draft_model_path is set, or EngineConfig::lookup_drafts
// (suffix drafts or context lookup, by EngineConfig::suffix_drafts).
#ifndef LIYAB_SPECULATIVE_DECODER_H
#define LIYAB_SPECULATIVE_DECODER_H

#include <chrono>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "liyab/types.h"

namespace liyab {

class DraftBudget;
class SuffixDrafter;
class Transformer;
class Sampler;
class ThreadPool;
struct Route;

class SpeculativeDecoder {
public:
    // Drafts with a model sharing the target's vocabulary.
    SpeculativeDecoder(Transformer& target, Transformer& draft, int32_t draft_tokens);
    // Drafts without a model: by context lookup, or from the suffix drafter.
    enum class TextDrafts { Lookup, Suffix };
    SpeculativeDecoder(Transformer& target, int32_t draft_tokens, TextDrafts drafts = TextDrafts::Lookup);
    ~SpeculativeDecoder();
    SpeculativeDecoder(const SpeculativeDecoder&) = delete;
    SpeculativeDecoder& operator=(const SpeculativeDecoder&) = delete;

    // Learn how many drafts to verify per step (default) or always draft_tokens
    // (suffix drafts: all the drafter offers, up to draft_tokens).
    void set_adaptive(bool adaptive);
    // Suffix drafts: verify the likeliest nodes of the drafter's
    // continuation tree, alternative branches included (default), or only
    // its best branch. No effect on other drafts.
    void set_tree(bool tree) noexcept { tree_ = tree; }

    // Precondition: the models cached the same prefix and `last` (the most
    // recent token) is not yet cached. `context` is that cached prefix's
    // tokens (read by context lookup and suffix drafts). Returns >= 1 new tokens;
    // afterwards the models cached the prefix + `last` + all returned tokens
    // except the final one, which becomes the next `last`.
    Result<std::vector<int32_t>> step(int32_t last, std::span<const int32_t> context, Sampler& sampler,
                                      const Route& target_route, const Route& draft_route, ThreadPool& pool);

    // Context lookup: up to `k` tokens that followed the most recent earlier
    // occurrence of the longest (4..2 token) suffix of `history`; empty when
    // the suffix never occurred before.
    static void lookup(std::span<const int32_t> history, int32_t k, std::vector<int32_t>& out);

    [[nodiscard]] int32_t proposed() const noexcept { return proposed_; }
    [[nodiscard]] int32_t accepted() const noexcept { return accepted_; }
    // Steps verified as a tree (more than one branch), and those of them
    // that kept a branch other than the likeliest one.
    [[nodiscard]] int32_t tree_steps() const noexcept { return tree_steps_; }
    [[nodiscard]] int32_t side_branches() const noexcept { return side_branches_; }
    void reset_stats() noexcept { proposed_ = accepted_ = tree_steps_ = side_branches_ = 0; }
    // The suffix drafter (nullptr unless TextDrafts::Suffix), for diagnostics.
    [[nodiscard]] const SuffixDrafter* suffix_drafter() const noexcept { return suffix_.get(); }

private:
    // step() with suffix drafts; `room`: most drafts the caches and one
    // batch hold.
    Result<std::vector<int32_t>> suffix_step(int32_t last, std::span<const int32_t> context, Sampler& sampler,
                                             int32_t room, const Route& target_route, ThreadPool& pool,
                                             std::chrono::steady_clock::time_point t0);

    Transformer& target_;
    Transformer* draft_ = nullptr;  // nullptr: context lookup
    int32_t draft_tokens_;
    int32_t proposed_ = 0;
    int32_t accepted_ = 0;
    int32_t tree_steps_ = 0;
    int32_t side_branches_ = 0;
    std::vector<std::vector<float>> q_;  // draft distributions, reused
    std::vector<float> p_;
    std::vector<int32_t> history_, drafts_;  // lookup scratch
    std::unique_ptr<DraftBudget> budget_;    // null: fixed draft_tokens_
    std::unique_ptr<SuffixDrafter> suffix_;  // non-null: suffix drafts
    bool tree_ = true;                       // suffix drafts verified as a tree (set_tree)
    // suffix_step() scratch: per draft index + 1 its first child and next
    // sibling and its batch index; per batch token its parent and draft
    // index; the depth-first
    // walk's pending siblings; the kept branch (batch indices) and its
    // draft indices.
    std::vector<int32_t> child_first_, child_next_, batch_of_, batch_parent_, batch_draft_, stack_, path_,
        accepted_path_;
};

}  // namespace liyab

#endif  // LIYAB_SPECULATIVE_DECODER_H
