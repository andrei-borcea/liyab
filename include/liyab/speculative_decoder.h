// Liyab — speculative decoding (draft-then-verify).
//
// A small draft model proposes k tokens autoregressively; the large target
// model scores all of them in ONE batched forward pass, so its weights are
// streamed once for up to k+1 tokens. Acceptance follows Leviathan et al.
// (2023): draft token d is kept with probability min(1, p(d)/q(d)); on the
// first rejection a replacement is drawn from norm(max(0, p − q)). The output
// distribution is exactly the target's. With greedy sampling this reduces to
// "keep while the target's argmax agrees", so the output is token-for-token
// identical to plain greedy decoding of the target.
//
// This is an engine component: Engine wires it up when
// EngineConfig::draft_model_path is set. Draft and target must share a
// vocabulary.
#ifndef LIYAB_SPECULATIVE_DECODER_H
#define LIYAB_SPECULATIVE_DECODER_H

#include <cstdint>
#include <vector>

#include "liyab/types.h"

namespace liyab {

class Transformer;
class Sampler;
class ThreadPool;
struct Route;

class SpeculativeDecoder {
public:
    SpeculativeDecoder(Transformer& target, Transformer& draft, int32_t draft_tokens);

    // Precondition: both models cached the same prefix and `last` (the most
    // recent token) is not yet cached by either. Returns >= 1 new tokens;
    // afterwards both models cached the prefix + `last` + all returned tokens
    // except the final one, which becomes the next `last`.
    Result<std::vector<int32_t>> step(int32_t last, Sampler& sampler, const Route& target_route,
                                      const Route& draft_route, ThreadPool& pool);

    [[nodiscard]] int32_t proposed() const noexcept { return proposed_; }
    [[nodiscard]] int32_t accepted() const noexcept { return accepted_; }
    void reset_stats() noexcept { proposed_ = accepted_ = 0; }

private:
    Transformer& target_;
    Transformer& draft_;
    int32_t draft_tokens_;
    int32_t proposed_ = 0;
    int32_t accepted_ = 0;
    std::vector<std::vector<float>> q_;  // draft distributions, reused
    std::vector<float> p_;
};

}  // namespace liyab

#endif  // LIYAB_SPECULATIVE_DECODER_H
