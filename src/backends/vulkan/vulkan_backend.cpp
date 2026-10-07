// Liyab — Vulkan compute discovery (Adreno / Mali / Xclipse).
//
// Loads the system Vulkan loader at runtime (no link-time dependency), creates
// a throwaway instance and reports the first physical device that exposes a
// compute queue, plus whether it can import host memory
// (VK_EXT_external_memory_host), which is what a zero-copy weight path over
// the mmap'd model needs. Vulkan matmul kernels are not implemented yet, so
// the engine routes Vulkan-ranked work to the next backend.
#include "backends/probes.h"

#if defined(LIYAB_USE_VULKAN) && !defined(__APPLE__)
#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <vulkan/vulkan.h>

#include <cstring>
#include <vector>
#endif

namespace liyab::probes {

AcceleratorProbe probe_vulkan() {
#if !defined(LIYAB_USE_VULKAN)
    return {false, "not compiled (LIYAB_USE_VULKAN=OFF)"};
#elif defined(__APPLE__)
    return {false, "Apple platforms use the Metal backend"};
#else
    void* lib = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib == nullptr) return {false, "Vulkan loader not found"};

    AcceleratorProbe result{false, "no Vulkan device with a compute queue"};
    auto get_instance_proc =
        reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
    auto create_instance = get_instance_proc
        ? reinterpret_cast<PFN_vkCreateInstance>(get_instance_proc(VK_NULL_HANDLE, "vkCreateInstance"))
        : nullptr;
    if (create_instance == nullptr) {
        dlclose(lib);
        return {false, "Vulkan loader is missing vkCreateInstance"};
    }

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "liyab";
    app.pEngineName = "liyab";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    if (create_instance(&create_info, nullptr, &instance) != VK_SUCCESS) {
        dlclose(lib);
        return {false, "vkCreateInstance failed (Vulkan 1.1 required)"};
    }

#define LIYAB_VK_FN(name) auto name = reinterpret_cast<PFN_##name>(get_instance_proc(instance, #name))
    LIYAB_VK_FN(vkDestroyInstance);
    LIYAB_VK_FN(vkEnumeratePhysicalDevices);
    LIYAB_VK_FN(vkGetPhysicalDeviceProperties);
    LIYAB_VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
    LIYAB_VK_FN(vkEnumerateDeviceExtensionProperties);
#undef LIYAB_VK_FN

    if (vkEnumeratePhysicalDevices && vkGetPhysicalDeviceProperties && vkGetPhysicalDeviceQueueFamilyProperties &&
        vkEnumerateDeviceExtensionProperties) {
        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> devices(count);
        if (count > 0 && vkEnumeratePhysicalDevices(instance, &count, devices.data()) >= VK_SUCCESS) {
            for (VkPhysicalDevice device : devices) {
                uint32_t families = 0;
                vkGetPhysicalDeviceQueueFamilyProperties(device, &families, nullptr);
                std::vector<VkQueueFamilyProperties> props(families);
                vkGetPhysicalDeviceQueueFamilyProperties(device, &families, props.data());
                bool compute = false;
                for (const auto& p : props) compute = compute || (p.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0;
                if (!compute) continue;

                uint32_t n_ext = 0;
                vkEnumerateDeviceExtensionProperties(device, nullptr, &n_ext, nullptr);
                std::vector<VkExtensionProperties> exts(n_ext);
                vkEnumerateDeviceExtensionProperties(device, nullptr, &n_ext, exts.data());
                bool host_import = false;
                for (const auto& e : exts) {
                    host_import = host_import || std::strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME) == 0;
                }

                VkPhysicalDeviceProperties dp{};
                vkGetPhysicalDeviceProperties(device, &dp);
                result = {true, std::string(dp.deviceName) + ", Vulkan " +
                                    std::to_string(VK_API_VERSION_MAJOR(dp.apiVersion)) + "." +
                                    std::to_string(VK_API_VERSION_MINOR(dp.apiVersion)) +
                                    (host_import ? ", host-memory import" : ", no host-memory import")};
                break;
            }
        }
    }
    if (vkDestroyInstance) vkDestroyInstance(instance, nullptr);
    dlclose(lib);
    return result;
#endif
}

}  // namespace liyab::probes
