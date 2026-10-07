// Liyab — Metal entry points for builds without LIYAB_USE_METAL (non-Apple).
#include "backends/probes.h"
#include "liyab/backend.h"

namespace liyab {

Result<std::unique_ptr<Backend>> make_metal_backend() {
    return Status(ErrorCode::Unsupported, "Metal backend not compiled (LIYAB_USE_METAL=OFF)");
}

namespace probes {
AcceleratorProbe probe_metal() { return {false, "not compiled (LIYAB_USE_METAL=OFF)"}; }
}  // namespace probes

}  // namespace liyab
