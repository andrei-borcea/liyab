// Liyab — Qualcomm QNN (Hexagon HTP) runtime discovery.
//
// Locates the QNN HTP backend library that the app ships in its APK
// (libQnnHtp.so plus the matching libQnnHtpV<arch>Stub.so / Skel) and asks
// it for its interface providers. No QNN SDK headers are needed: only the
// stable C entry point QnnInterface_getProviders is resolved.
//
// Executing the FFN on the HTP additionally requires building and finalizing
// a QNN graph per layer shape (QnnGraph_create / QnnGraph_finalize) with
// the weights registered as shared rpcmem buffers; that compute path is not
// implemented yet, so the engine routes QNN-ranked work to the next backend.
#include "backends/probes.h"

#if defined(LIYAB_USE_QNN) && defined(__linux__)
#include <dlfcn.h>
#endif

namespace liyab::probes {

AcceleratorProbe probe_qnn_htp() {
#if !defined(LIYAB_USE_QNN)
    return {false, "not compiled (LIYAB_USE_QNN=OFF)"};
#elif !defined(__linux__)
    return {false, "QNN HTP is only available on Android"};
#else
    void* lib = dlopen("libQnnHtp.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return {false, "libQnnHtp.so not found in the app's native library path"};

    // Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providers, uint32_t* count)
    using GetProviders = uint64_t (*)(const void***, uint32_t*);
    auto get_providers = reinterpret_cast<GetProviders>(dlsym(lib, "QnnInterface_getProviders"));
    AcceleratorProbe result{false, "libQnnHtp.so lacks QnnInterface_getProviders"};
    if (get_providers != nullptr) {
        const void** providers = nullptr;
        uint32_t count = 0;
        const uint64_t err = get_providers(&providers, &count);
        if (err == 0 && count > 0) {
            result = {true, "libQnnHtp.so, " + std::to_string(count) + " interface provider(s)"};
        } else {
            result = {false, "QnnInterface_getProviders failed (error " + std::to_string(err) + ")"};
        }
    }
    dlclose(lib);
    return result;
#endif
}

}  // namespace liyab::probes
