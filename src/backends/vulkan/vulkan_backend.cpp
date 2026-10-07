// Liyab — Vulkan compute (Adreno / Mali / Xclipse): runtime probe and the
// matmul backend.
//
// The system Vulkan loader is opened at runtime (no link-time dependency).
// Mobile drivers rarely support VK_EXT_external_memory_host (Adreno 830 does
// not), so the mmap'd weights cannot be wrapped in place: each tensor is
// copied once into host-visible GPU memory and, during that copy, repacked
// into an aligned layout (float scales + 4-byte-aligned quants) so the shader
// reads whole words instead of 18-byte GGML blocks. On unified-memory SoCs
// this costs RAM, not a PCIe transfer.
#include "backends/probes.h"
#include "liyab/backend.h"

#if defined(LIYAB_USE_VULKAN) && !defined(__APPLE__)
#define VK_NO_PROTOTYPES
#include <dlfcn.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <map>
#include <utility>
#include <vector>

#include "core/log.h"
#include "core/quant.h"
#include "vulkan_shaders.h"  // generated: SPIR-V of shaders/matvec.comp per weight type
#define LIYAB_VULKAN_BACKEND 1
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

namespace liyab {

#if defined(LIYAB_VULKAN_BACKEND)
namespace {

#define LIYAB_VK_INSTANCE_FNS(X)                                                                          \
    X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties)                  \
    X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceMemoryProperties) X(vkCreateDevice) \
    X(vkGetDeviceProcAddr)
#define LIYAB_VK_DEVICE_FNS(X)                                                                            \
    X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer)                           \
    X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory)          \
    X(vkMapMemory) X(vkFlushMappedMemoryRanges) X(vkInvalidateMappedMemoryRanges)                         \
    X(vkCreateShaderModule) X(vkDestroyShaderModule) X(vkCreateDescriptorSetLayout)                       \
    X(vkDestroyDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkDestroyPipelineLayout)                  \
    X(vkCreateComputePipelines) X(vkDestroyPipeline) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) \
    X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool)  \
    X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer) X(vkResetCommandBuffer)     \
    X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch)                \
    X(vkCmdPipelineBarrier) X(vkQueueSubmit) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences)        \
    X(vkResetFences) X(vkGetFenceStatus)

struct VkFns {
#define LIYAB_VK_MEMBER(name) PFN_##name name = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    PFN_vkCreateInstance vkCreateInstance = nullptr;
    LIYAB_VK_INSTANCE_FNS(LIYAB_VK_MEMBER)
    LIYAB_VK_DEVICE_FNS(LIYAB_VK_MEMBER)
#undef LIYAB_VK_MEMBER
};

// Pipeline slot per weight type.
int pipeline_index(DType type) {
    switch (type) {
        case DType::F32: return 0;
        case DType::F16: return 1;
        case DType::Q4_0: return 2;
        case DType::Q4_1: return 3;
        case DType::Q8_0: return 4;
    }
    return 0;
}

struct Params {
    uint32_t rows;
    uint32_t cols;
    uint32_t n;
};

class VulkanBackend final : public Backend {
public:
    static Result<std::unique_ptr<Backend>> create();
    ~VulkanBackend() override;

    [[nodiscard]] BackendKind kind() const noexcept override { return BackendKind::Vulkan; }
    [[nodiscard]] std::string description() const override {
        return "vulkan (" + device_name_ + ", " + std::to_string(uploaded_bytes_ >> 20) + " MiB repacked weights)";
    }
    [[nodiscard]] bool supports(DType) const noexcept override { return true; }
    Status matmul(const TensorView& w, const float* x, float* y, int32_t n) override {
        const TensorView* ws[] = {&w};
        float* ys[] = {y};
        return matmul_group(ws, x, ys, n);
    }
    Status matmul_group(std::span<const TensorView* const> ws, const float* x, std::span<float* const> ys,
                        int32_t n) override;

private:
    struct Buffer {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        void* mapped = nullptr;
        VkDeviceSize size = 0;
    };
    struct Weights {
        Buffer scales;
        Buffer quants;
    };

    VulkanBackend() = default;
    Status init();
    Status create_buffer(VkDeviceSize size, Buffer& out);
    void destroy(Buffer& b) noexcept;
    Status ensure(Buffer& b, VkDeviceSize size);
    void flush(const Buffer& b) const noexcept;
    Result<Weights*> upload(const TensorView& w);

    VkFns f_;
    void* lib_ = nullptr;
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t family_ = 0;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    int32_t memory_type_ = -1;
    bool coherent_ = true;
    std::string device_name_;

    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkPipeline pipelines_[5] = {};
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    static constexpr uint32_t kSlots = 4;  // matmuls per submission
    VkDescriptorSet sets_[kSlots] = {};
    VkCommandPool command_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;

    Buffer x_;
    Buffer y_[kSlots];
    Buffer dummy_;  // binding 0 for F32/F16 weights, which have no scales
    std::map<std::pair<const uint8_t*, size_t>, Weights> weights_;
    size_t uploaded_bytes_ = 0;
};

Status vk_error(const char* what, VkResult r) {
    return {ErrorCode::BackendError, std::string(what) + " failed (VkResult " + std::to_string(r) + ")"};
}

Result<std::unique_ptr<Backend>> VulkanBackend::create() {
    std::unique_ptr<VulkanBackend> backend(new VulkanBackend());
    LIYAB_RETURN_IF_ERROR(backend->init());
    LIYAB_LOG_INFO("vulkan: %s, Vulkan %u.%u, max storage buffer %u MiB", backend->device_name_.c_str(),
                   VK_API_VERSION_MAJOR(backend->props_.apiVersion), VK_API_VERSION_MINOR(backend->props_.apiVersion),
                   backend->props_.limits.maxStorageBufferRange >> 20);
    return std::unique_ptr<Backend>(std::move(backend));
}

Status VulkanBackend::init() {
    lib_ = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (lib_ == nullptr) lib_ = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
    if (lib_ == nullptr) return Status(ErrorCode::Unsupported, "Vulkan loader not found");
    f_.vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib_, "vkGetInstanceProcAddr"));
    if (f_.vkGetInstanceProcAddr == nullptr) return Status(ErrorCode::Unsupported, "vkGetInstanceProcAddr missing");
    f_.vkCreateInstance =
        reinterpret_cast<PFN_vkCreateInstance>(f_.vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    if (f_.vkCreateInstance == nullptr) return Status(ErrorCode::Unsupported, "vkCreateInstance missing");

    VkApplicationInfo app{};

    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "liyab";
    app.pEngineName = "liyab";
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    if (VkResult r = f_.vkCreateInstance(&ici, nullptr, &instance_); r != VK_SUCCESS) return vk_error("vkCreateInstance", r);
#define LIYAB_VK_LOAD_INSTANCE(name)                                                                \
    f_.name = reinterpret_cast<PFN_##name>(f_.vkGetInstanceProcAddr(instance_, #name));            \
    if (f_.name == nullptr) return Status(ErrorCode::Unsupported, "Vulkan function " #name " missing");
    LIYAB_VK_INSTANCE_FNS(LIYAB_VK_LOAD_INSTANCE)
#undef LIYAB_VK_LOAD_INSTANCE

    uint32_t count = 0;
    f_.vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    std::vector<VkPhysicalDevice> devices(count);
    if (count > 0) f_.vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    for (VkPhysicalDevice d : devices) {
        uint32_t families = 0;
        f_.vkGetPhysicalDeviceQueueFamilyProperties(d, &families, nullptr);
        std::vector<VkQueueFamilyProperties> qf(families);
        f_.vkGetPhysicalDeviceQueueFamilyProperties(d, &families, qf.data());
        for (uint32_t i = 0; i < families; ++i) {
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physical_ = d;
                family_ = i;
                break;
            }
        }
        if (physical_ != VK_NULL_HANDLE) break;
    }
    if (physical_ == VK_NULL_HANDLE) return Status(ErrorCode::Unsupported, "no Vulkan device with a compute queue");
    f_.vkGetPhysicalDeviceProperties(physical_, &props_);
    f_.vkGetPhysicalDeviceMemoryProperties(physical_, &memory_);
    device_name_ = props_.deviceName;

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family_;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    if (VkResult r = f_.vkCreateDevice(physical_, &dci, nullptr, &device_); r != VK_SUCCESS) return vk_error("vkCreateDevice", r);
#define LIYAB_VK_LOAD_DEVICE(name)                                                         \
    f_.name = reinterpret_cast<PFN_##name>(f_.vkGetDeviceProcAddr(device_, #name));       \
    if (f_.name == nullptr) return Status(ErrorCode::Unsupported, "Vulkan function " #name " missing");
    LIYAB_VK_DEVICE_FNS(LIYAB_VK_LOAD_DEVICE)
#undef LIYAB_VK_LOAD_DEVICE
    f_.vkGetDeviceQueue(device_, family_, 0, &queue_);

    // Descriptor layout: scales, quants, x, y.
    VkDescriptorSetLayoutBinding bindings[4] = {};
    for (uint32_t i = 0; i < 4; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = 4;
    dslci.pBindings = bindings;
    if (VkResult r = f_.vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &set_layout_); r != VK_SUCCESS) {
        return vk_error("vkCreateDescriptorSetLayout", r);
    }
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Params)};
    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &set_layout_;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &push;
    if (VkResult r = f_.vkCreatePipelineLayout(device_, &plci, nullptr, &pipeline_layout_); r != VK_SUCCESS) {
        return vk_error("vkCreatePipelineLayout", r);
    }

    const std::pair<const unsigned char*, size_t> modules[5] = {
        {kSpv_f32, kSpv_f32_size}, {kSpv_f16, kSpv_f16_size}, {kSpv_q4_0, kSpv_q4_0_size},
        {kSpv_q4_1, kSpv_q4_1_size}, {kSpv_q8_0, kSpv_q8_0_size}};
    for (int i = 0; i < 5; ++i) {
        std::vector<uint32_t> code(modules[i].second / 4);
        std::memcpy(code.data(), modules[i].first, modules[i].second);
        VkShaderModuleCreateInfo smci{};
        smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        smci.codeSize = modules[i].second;
        smci.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        if (VkResult r = f_.vkCreateShaderModule(device_, &smci, nullptr, &module); r != VK_SUCCESS) {
            return vk_error("vkCreateShaderModule", r);
        }
        VkComputePipelineCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cpci.stage.module = module;
        cpci.stage.pName = "main";
        cpci.layout = pipeline_layout_;
        const VkResult r = f_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipelines_[i]);
        f_.vkDestroyShaderModule(device_, module, nullptr);
        if (r != VK_SUCCESS) return vk_error("vkCreateComputePipelines", r);
    }

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * kSlots};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = kSlots;
    dpci.poolSizeCount = 1;
    dpci.pPoolSizes = &pool_size;
    if (VkResult r = f_.vkCreateDescriptorPool(device_, &dpci, nullptr, &descriptor_pool_); r != VK_SUCCESS) {
        return vk_error("vkCreateDescriptorPool", r);
    }
    VkDescriptorSetAllocateInfo dsai{};
    dsai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dsai.descriptorPool = descriptor_pool_;
    const VkDescriptorSetLayout layouts[kSlots] = {set_layout_, set_layout_, set_layout_, set_layout_};
    dsai.descriptorSetCount = kSlots;
    dsai.pSetLayouts = layouts;
    if (VkResult r = f_.vkAllocateDescriptorSets(device_, &dsai, sets_); r != VK_SUCCESS) {
        return vk_error("vkAllocateDescriptorSets", r);
    }

    VkCommandPoolCreateInfo cpi{};

    cpi.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = family_;
    if (VkResult r = f_.vkCreateCommandPool(device_, &cpi, nullptr, &command_pool_); r != VK_SUCCESS) {
        return vk_error("vkCreateCommandPool", r);
    }
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = command_pool_;
    cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (VkResult r = f_.vkAllocateCommandBuffers(device_, &cbai, &cmd_); r != VK_SUCCESS) {
        return vk_error("vkAllocateCommandBuffers", r);
    }
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (VkResult r = f_.vkCreateFence(device_, &fci, nullptr, &fence_); r != VK_SUCCESS) return vk_error("vkCreateFence", r);
    return create_buffer(16, dummy_);
}

VulkanBackend::~VulkanBackend() {
    if (device_ != VK_NULL_HANDLE) {
        for (auto& [key, w] : weights_) {
            destroy(w.scales);
            destroy(w.quants);
        }
        destroy(x_);
        for (Buffer& y : y_) destroy(y);
        destroy(dummy_);
        if (fence_) f_.vkDestroyFence(device_, fence_, nullptr);
        if (command_pool_) f_.vkDestroyCommandPool(device_, command_pool_, nullptr);
        if (descriptor_pool_) f_.vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        for (VkPipeline p : pipelines_) {
            if (p) f_.vkDestroyPipeline(device_, p, nullptr);
        }
        if (pipeline_layout_) f_.vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
        if (set_layout_) f_.vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
        f_.vkDestroyDevice(device_, nullptr);
    }
    if (instance_ != VK_NULL_HANDLE && f_.vkDestroyInstance) f_.vkDestroyInstance(instance_, nullptr);
    if (lib_ != nullptr) dlclose(lib_);
}

Status VulkanBackend::create_buffer(VkDeviceSize size, Buffer& out) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = std::max<VkDeviceSize>(size, 16);
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (VkResult r = f_.vkCreateBuffer(device_, &bci, nullptr, &out.buffer); r != VK_SUCCESS) return vk_error("vkCreateBuffer", r);
    VkMemoryRequirements req{};
    f_.vkGetBufferMemoryRequirements(device_, out.buffer, &req);
    if (memory_type_ < 0) {
        // Prefer device-local + host-visible + coherent (unified memory), then
        // host-visible + coherent, then any host-visible type.
        const VkMemoryPropertyFlags wanted[] = {
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT};
        for (const VkMemoryPropertyFlags flags : wanted) {
            for (uint32_t i = 0; i < memory_.memoryTypeCount && memory_type_ < 0; ++i) {
                if ((req.memoryTypeBits & (1u << i)) && (memory_.memoryTypes[i].propertyFlags & flags) == flags) {
                    memory_type_ = static_cast<int32_t>(i);
                    coherent_ = (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
                }
            }
            if (memory_type_ >= 0) break;
        }
        if (memory_type_ < 0) return Status(ErrorCode::Unsupported, "no host-visible Vulkan memory type");
    }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = static_cast<uint32_t>(memory_type_);
    if (VkResult r = f_.vkAllocateMemory(device_, &mai, nullptr, &out.memory); r != VK_SUCCESS) {
        f_.vkDestroyBuffer(device_, out.buffer, nullptr);
        out.buffer = VK_NULL_HANDLE;
        return r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY
                   ? Status(ErrorCode::OutOfMemory, "Vulkan memory exhausted")
                   : vk_error("vkAllocateMemory", r);
    }
    f_.vkBindBufferMemory(device_, out.buffer, out.memory, 0);
    if (VkResult r = f_.vkMapMemory(device_, out.memory, 0, VK_WHOLE_SIZE, 0, &out.mapped); r != VK_SUCCESS) {
        return vk_error("vkMapMemory", r);
    }
    out.size = bci.size;
    return Status::ok();
}

void VulkanBackend::destroy(Buffer& b) noexcept {
    if (b.buffer) f_.vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) f_.vkFreeMemory(device_, b.memory, nullptr);  // implicitly unmaps
    b = Buffer{};
}

Status VulkanBackend::ensure(Buffer& b, VkDeviceSize size) {
    if (b.buffer != VK_NULL_HANDLE && b.size >= size) return Status::ok();
    destroy(b);
    return create_buffer(std::max<VkDeviceSize>(size, 64 * 1024), b);
}

void VulkanBackend::flush(const Buffer& b) const noexcept {
    if (coherent_) return;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = b.memory;
    range.size = VK_WHOLE_SIZE;
    f_.vkFlushMappedMemoryRanges(device_, 1, &range);
}

Result<VulkanBackend::Weights*> VulkanBackend::upload(const TensorView& w) {
    const auto key = std::make_pair(w.data, w.nbytes);
    if (auto it = weights_.find(key); it != weights_.end()) return &it->second;

    const auto rows = static_cast<size_t>(w.rows());
    const auto cols = static_cast<size_t>(w.cols());
    const size_t blocks = rows * (cols / 32);
    size_t scale_bytes = 16;
    size_t quant_bytes = 0;
    switch (w.type) {
        case DType::F32: quant_bytes = rows * cols * 4; break;
        case DType::F16:
            if (cols % 2 != 0) return Status(ErrorCode::Unsupported, "odd-width F16 tensor");
            quant_bytes = rows * cols * 2;
            break;
        case DType::Q4_0: scale_bytes = blocks * 4; quant_bytes = blocks * 16; break;
        case DType::Q4_1: scale_bytes = blocks * 8; quant_bytes = blocks * 16; break;
        case DType::Q8_0: scale_bytes = blocks * 4; quant_bytes = blocks * 32; break;
    }
    const VkDeviceSize limit = props_.limits.maxStorageBufferRange;
    if (quant_bytes > limit || scale_bytes > limit) {
        return Status(ErrorCode::Unsupported, "tensor '" + std::string(w.name) + "' exceeds maxStorageBufferRange");
    }
    Weights out;
    LIYAB_RETURN_IF_ERROR(create_buffer(scale_bytes, out.scales));
    if (Status s = create_buffer(quant_bytes, out.quants); !s.is_ok()) {
        destroy(out.scales);
        return s;
    }

    auto* scales = static_cast<float*>(out.scales.mapped);
    auto* quants = static_cast<uint8_t*>(out.quants.mapped);
    switch (w.type) {
        case DType::F32:
        case DType::F16:
            std::memcpy(quants, w.data, quant_bytes);
            break;
        case DType::Q4_0: {
            const auto* b = reinterpret_cast<const quant::BlockQ4_0*>(w.data);
            for (size_t i = 0; i < blocks; ++i) {
                scales[i] = quant::fp16_to_fp32(b[i].d);
                std::memcpy(quants + i * 16, b[i].qs, 16);
            }
            break;
        }
        case DType::Q4_1: {
            const auto* b = reinterpret_cast<const quant::BlockQ4_1*>(w.data);
            for (size_t i = 0; i < blocks; ++i) {
                scales[2 * i] = quant::fp16_to_fp32(b[i].d);
                scales[2 * i + 1] = quant::fp16_to_fp32(b[i].m);
                std::memcpy(quants + i * 16, b[i].qs, 16);
            }
            break;
        }
        case DType::Q8_0: {
            const auto* b = reinterpret_cast<const quant::BlockQ8_0*>(w.data);
            for (size_t i = 0; i < blocks; ++i) {
                scales[i] = quant::fp16_to_fp32(b[i].d);
                std::memcpy(quants + i * 32, b[i].qs, 32);
            }
            break;
        }
    }
    flush(out.scales);
    flush(out.quants);
    uploaded_bytes_ += scale_bytes + quant_bytes;
    return &weights_.emplace(key, out).first->second;
}

Status VulkanBackend::matmul_group(std::span<const TensorView* const> ws, const float* x,
                                   std::span<float* const> ys, int32_t n) {
    if (n <= 0 || ws.empty()) return Status::ok();
    if (ws.size() > kSlots) return Backend::matmul_group(ws, x, ys, n);
    const auto cols = static_cast<uint32_t>(ws[0]->cols());
    Weights* weights[kSlots] = {};
    for (size_t i = 0; i < ws.size(); ++i) {
        if (ws[i]->cols() != cols) return Status(ErrorCode::InvalidArgument, "grouped matmuls need equal widths");
        auto uploaded = upload(*ws[i]);
        if (!uploaded) return uploaded.status();
        weights[i] = uploaded.value();
    }
    const VkDeviceSize x_bytes = static_cast<VkDeviceSize>(n) * cols * sizeof(float);
    LIYAB_RETURN_IF_ERROR(ensure(x_, x_bytes));
    std::memcpy(x_.mapped, x, static_cast<size_t>(x_bytes));
    flush(x_);

    f_.vkResetCommandBuffer(cmd_, 0);
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    f_.vkBeginCommandBuffer(cmd_, &begin);
    for (size_t i = 0; i < ws.size(); ++i) {
        const TensorView& w = *ws[i];
        const auto rows = static_cast<uint32_t>(w.rows());
        const VkDeviceSize y_bytes = static_cast<VkDeviceSize>(n) * rows * sizeof(float);
        LIYAB_RETURN_IF_ERROR(ensure(y_[i], y_bytes));

        const bool has_scales = w.type != DType::F32 && w.type != DType::F16;
        const VkDescriptorBufferInfo infos[4] = {{has_scales ? weights[i]->scales.buffer : dummy_.buffer, 0, VK_WHOLE_SIZE},
                                                 {weights[i]->quants.buffer, 0, VK_WHOLE_SIZE},
                                                 {x_.buffer, 0, VK_WHOLE_SIZE},
                                                 {y_[i].buffer, 0, VK_WHOLE_SIZE}};
        VkWriteDescriptorSet writes[4] = {};
        for (uint32_t b = 0; b < 4; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = sets_[i];
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &infos[b];
        }
        f_.vkUpdateDescriptorSets(device_, 4, writes, 0, nullptr);

        // Quantized kernels cover kRowsPerGroup rows per workgroup.
        const uint32_t rows_per_group = has_scales ? 4u : 1u;
        const uint32_t groups = (rows + rows_per_group - 1) / rows_per_group;
        const uint32_t groups_x = std::min(groups, props_.limits.maxComputeWorkGroupCount[0]);
        const uint32_t groups_z = (groups + groups_x - 1) / groups_x;
        if (static_cast<uint32_t>(n) > props_.limits.maxComputeWorkGroupCount[1] ||
            groups_z > props_.limits.maxComputeWorkGroupCount[2]) {
            f_.vkEndCommandBuffer(cmd_);
            return Status(ErrorCode::Unsupported, "matmul shape exceeds Vulkan dispatch limits");
        }
        const Params params{rows, cols, static_cast<uint32_t>(n)};
        f_.vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines_[pipeline_index(w.type)]);
        f_.vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &sets_[i], 0, nullptr);
        f_.vkCmdPushConstants(cmd_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof params, &params);
        f_.vkCmdDispatch(cmd_, groups_x, static_cast<uint32_t>(n), groups_z);
    }
    // Make the shaders' writes visible to the host read after the fence.
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    f_.vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                            nullptr, 0, nullptr);
    f_.vkEndCommandBuffer(cmd_);

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd_;
    if (VkResult r = f_.vkQueueSubmit(queue_, 1, &submit, fence_); r != VK_SUCCESS) return vk_error("vkQueueSubmit", r);
    // Kernels take tens to hundreds of microseconds: poll briefly before
    // falling back to the driver's (sleeping) wait, which adds wake-up latency.
    VkResult wait = VK_NOT_READY;
    const auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(2000);
    while ((wait = f_.vkGetFenceStatus(device_, fence_)) == VK_NOT_READY &&
           std::chrono::steady_clock::now() < spin_until) {
    }
    if (wait == VK_NOT_READY) wait = f_.vkWaitForFences(device_, 1, &fence_, VK_TRUE, 10'000'000'000ULL);
    f_.vkResetFences(device_, 1, &fence_);
    if (wait != VK_SUCCESS) return vk_error("vkWaitForFences", wait);

    for (size_t i = 0; i < ws.size(); ++i) {
        if (!coherent_) {
            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = y_[i].memory;
            range.size = VK_WHOLE_SIZE;
            f_.vkInvalidateMappedMemoryRanges(device_, 1, &range);
        }
        std::memcpy(ys[i], y_[i].mapped, static_cast<size_t>(n) * static_cast<size_t>(ws[i]->rows()) * sizeof(float));
    }
    return Status::ok();
}

}  // namespace

Result<std::unique_ptr<Backend>> make_vulkan_backend() { return VulkanBackend::create(); }

#else

Result<std::unique_ptr<Backend>> make_vulkan_backend() {
    return Status(ErrorCode::Unsupported, "Vulkan backend not compiled (LIYAB_USE_VULKAN=OFF or Apple platform)");
}

#endif

}  // namespace liyab
