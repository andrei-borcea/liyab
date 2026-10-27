#include "core/moe_routing.h"

#include <algorithm>
#include <cmath>

namespace liyab {

// Group-limited expert selection (ModelConfig::n_expert_groups): marks in
// `allowed` (one flag per group) the `used` best of `groups` equal groups of
// the selection scores `sel` (n_expert of them). A group scores the sum of
// its two best scores when `top2` (a router with a selection bias), else its
// best one; ties go to the lower group. The top-k is then taken among the
// allowed groups' experts only.
void select_expert_groups(const float* sel, size_t n_expert, int32_t groups, int32_t used, bool top2,
                   std::vector<uint8_t>& allowed) {
    const size_t per_group = n_expert / static_cast<size_t>(groups);
    thread_local std::vector<float> score;
    thread_local std::vector<int32_t> rank;
    score.resize(static_cast<size_t>(groups));
    rank.resize(static_cast<size_t>(groups));
    for (size_t g = 0; g < static_cast<size_t>(groups); ++g) {
        float best = -INFINITY, second = -INFINITY;
        for (size_t e = g * per_group; e < (g + 1) * per_group; ++e) {
            if (sel[e] > best) {
                second = best;
                best = sel[e];
            } else if (sel[e] > second) {
                second = sel[e];
            }
        }
        score[g] = top2 && per_group > 1 ? best + second : best;
        rank[g] = static_cast<int32_t>(g);
    }
    std::partial_sort(rank.begin(), rank.begin() + used, rank.end(), [&](int32_t a, int32_t b) {
        const float sa = score[static_cast<size_t>(a)], sb = score[static_cast<size_t>(b)];
        return sa > sb || (sa == sb && a < b);
    });
    allowed.assign(static_cast<size_t>(groups), 0);
    for (int32_t i = 0; i < used; ++i) allowed[static_cast<size_t>(rank[static_cast<size_t>(i)])] = 1;
}

}  // namespace liyab
