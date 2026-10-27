// Liyab — resolved rotary parameters for the core (internal).
#ifndef LIYAB_COMPAT_ROTARY_H
#define LIYAB_COMPAT_ROTARY_H

#include "compat/architecture.h"

namespace liyab {

// Logical head stride and per-pair frequencies after format/scaling rules.
struct RopeTable {
    int32_t head_dim = 0;         // logical head stride of this table
    std::vector<float> inv_freq;  // per rotary pair, frequency factors included
    float scale = 1.0f;           // linear position scaling
    float magnitude = 1.0f;       // multiplies the rotated dimensions (ModelConfig::rope_magnitude)
};

struct RotaryPlan {
    std::vector<RopeTable> tables;
    std::vector<int32_t> block_tables;  // table index for each model block
};

// Resolves global/local, LongRoPE, YaRN and compressed-stream tables for a
// validated model and positive context length. Reads only small frequency
// tensors, never rewrites the mapped file, and owns its outputs. Concurrent
// calls require read-only access to the file. Returns InvalidModel for a
// malformed frequency tensor or InvalidArgument for a nonpositive context;
// numerical application remains in the core.
Result<RotaryPlan> make_rotary_plan(const MmapLoader& file, const ModelConfig& config, int32_t context_length);

}  // namespace liyab

#endif  // LIYAB_COMPAT_ROTARY_H
