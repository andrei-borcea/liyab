// Liyab — MediaTek NeuroPilot (APU) runtime discovery.
//
// MediaTek devices expose the Neuron Adapter API (an NNAPI-shaped C API) from
// a vendor library that apps may load. The probe resolves NeuronModel_create
// to confirm the runtime is reachable from this process. Compiling FFN graphs
// for the APU (NeuronModel_* / NeuronCompilation_*) is not implemented yet,
// so the engine routes NeuroPilot-ranked work to the next backend.
#include "backends/probes.h"

#if defined(LIYAB_USE_NEUROPILOT) && defined(__linux__)
#include <dlfcn.h>
#endif

namespace liyab::probes {

AcceleratorProbe probe_neuropilot() {
#if !defined(LIYAB_USE_NEUROPILOT)
    return {false, "not compiled (LIYAB_USE_NEUROPILOT=OFF)"};
#elif !defined(__linux__)
    return {false, "NeuroPilot is only available on Android"};
#else
    static constexpr const char* kCandidates[] = {
        "libneuronusdk_adapter.mtk.so",  // vendor partition, public to apps on Dimensity devices
        "libneuron_adapter.so",          // bundled with the app from the NeuroPilot SDK
    };
    for (const char* name : kCandidates) {
        void* lib = dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) continue;
        const bool has_api = dlsym(lib, "NeuronModel_create") != nullptr;
        dlclose(lib);
        if (has_api) return {true, name};
    }
    return {false, "no Neuron Adapter runtime library found"};
#endif
}

}  // namespace liyab::probes
