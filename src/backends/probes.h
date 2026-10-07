// Liyab — runtime accelerator probes (internal).
//
// Each probe answers "can this process reach accelerator X right now?"
// without linking the vendor SDK: runtimes are located with dlopen so the
// library loads on every device and degrades to the next backend in line.
// When the corresponding LIYAB_USE_* flag is off the probe reports so.
#ifndef LIYAB_BACKENDS_PROBES_H
#define LIYAB_BACKENDS_PROBES_H

#include "liyab/device_detect.h"

namespace liyab::probes {

AcceleratorProbe probe_qnn_htp();     // src/backends/qnn/qnn_backend.cpp
AcceleratorProbe probe_neuropilot();  // src/backends/neuropilot/neuropilot_backend.cpp
AcceleratorProbe probe_vulkan();      // src/backends/vulkan/vulkan_backend.cpp
AcceleratorProbe probe_metal();       // src/backends/metal/metal_backend.mm

}  // namespace liyab::probes

#endif  // LIYAB_BACKENDS_PROBES_H
