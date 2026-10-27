// Liyab — group-limited expert selection (internal).
#ifndef LIYAB_CORE_MOE_ROUTING_H
#define LIYAB_CORE_MOE_ROUTING_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace liyab {

// Scores `groups` equal consecutive groups of `n_expert` selection scores
// (router probabilities plus any selection bias), setting `allowed` to one
// byte per group: 1 for the `used` best groups, 0 for the others. A group
// scores its maximum, or the sum of its best two scores with `top2` (one
// score for a one-expert group). Ties choose the lower group. Preconditions:
// 0 < used <= groups <= n_expert, n_expert % groups == 0, sel has n_expert
// finite entries. describe_model() validates the geometry before routing.
// Owns no input; safe on independent outputs from different threads.
void select_expert_groups(const float* sel, size_t n_expert, int32_t groups, int32_t used, bool top2,
                          std::vector<uint8_t>& allowed);

}  // namespace liyab

#endif  // LIYAB_CORE_MOE_ROUTING_H
