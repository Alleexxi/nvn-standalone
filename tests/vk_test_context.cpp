#include "vk_test_context.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace NvnTest {

namespace {
void* OpenVulkanLibrary() {
#ifdef _WIN32
    return reinterpret_cast<void*>(LoadLibraryA("vulkan-1.dll"));
#else
    void* lib = dlopen("libvulkan.so.1", RTLD_NOW);
    return lib ? lib : dlopen("libvulkan.so", RTLD_NOW);
#endif
}

PFN_vkGetInstanceProcAddr LoadGetInstanceProcAddr(void* lib) {
#ifdef _WIN32
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        GetProcAddress(static_cast<HMODULE>(lib), "vkGetInstanceProcAddr"));
#else
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
#endif
}

std::string g_skip_reason;

template <typename T>
T Proc(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, const char* name) {
    return reinterpret_cast<T>(gipa(inst, name));
}
} // namespace

VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                  VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                                  const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    auto* ctx = static_cast<VkTestContext*>(user);
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        ctx->validation_messages_.fetch_add(1);
        ctx->last_message_ = data->pMessage ? data->pMessage : "";
        std::fprintf(stderr, "[validation] %s\n", ctx->last_message_.c_str());
    }
    return VK_FALSE;
}

VkTestContext* VkTestContext::Get() {
    // Intentionally leaked: destroying the device during static destruction (after the driver's
    // own teardown started) crashes NVIDIA's ICD with an access violation at process exit.
    static VkTestContext* instance = [] {
        auto* ctx = new VkTestContext();
        if (!ctx->Init(g_skip_reason)) {
            return static_cast<VkTestContext*>(nullptr);
        }
        return ctx;
    }();
    return instance;
}

const std::string& VkTestContext::SkipReason() {
    return g_skip_reason;
}

bool VkTestContext::Init(std::string& why) {
    library_ = OpenVulkanLibrary();
    if (!library_) {
        why = "Vulkan loader (vulkan-1.dll) not found";
        return false;
    }
    const PFN_vkGetInstanceProcAddr gipa = LoadGetInstanceProcAddr(library_);
    if (!gipa) {
        why = "vkGetInstanceProcAddr missing";
        return false;
    }
    idld_.vkGetInstanceProcAddr = gipa;
    if (!Vulkan::vk::Load(idld_)) {
        why = "failed to load global Vulkan functions";
        return false;
    }

    // Validation layer available?
    const char* no_val = std::getenv("NVN_NO_VALIDATION");
    bool want_validation = !(no_val && no_val[0] == '1');
    std::vector<const char*> layers;
    std::vector<const char*> inst_exts;
    if (want_validation) {
        u32 count = 0;
        idld_.vkEnumerateInstanceLayerProperties(&count, nullptr);
        std::vector<VkLayerProperties> props(count);
        idld_.vkEnumerateInstanceLayerProperties(&count, props.data());
        for (const auto& p : props) {
            if (std::strcmp(p.layerName, "VK_LAYER_KHRONOS_validation") == 0) {
                validation_enabled_ = true;
            }
        }
        if (validation_enabled_) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            inst_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        } else {
            std::fprintf(stderr, "[nvn-test] VK_LAYER_KHRONOS_validation not installed: "
                                 "tests run WITHOUT validation (install the Vulkan SDK)\n");
        }
    }

    const VkApplicationInfo app{.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                .pApplicationName = "nvn-standalone-tests",
                                .apiVersion = VK_API_VERSION_1_3};
    const VkInstanceCreateInfo ici{.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                   .pApplicationInfo = &app,
                                   .enabledLayerCount = static_cast<u32>(layers.size()),
                                   .ppEnabledLayerNames = layers.data(),
                                   .enabledExtensionCount = static_cast<u32>(inst_exts.size()),
                                   .ppEnabledExtensionNames = inst_exts.data()};
    if (idld_.vkCreateInstance(&ici, nullptr, &instance_) != VK_SUCCESS) {
        why = "vkCreateInstance failed (no Vulkan 1.3 driver?)";
        return false;
    }
    if (!Vulkan::vk::Load(instance_, idld_)) {
        why = "failed to load instance functions";
        return false;
    }

    if (validation_enabled_) {
        auto create = Proc<PFN_vkCreateDebugUtilsMessengerEXT>(gipa, instance_,
                                                               "vkCreateDebugUtilsMessengerEXT");
        const VkDebugUtilsMessengerCreateInfoEXT mci{
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT,
            .pfnUserCallback = &DebugCallback,
            .pUserData = this};
        if (create) {
            create(instance_, &mci, nullptr, &messenger_);
        }
    }

    // Pick the first discrete GPU, else the first device with Vulkan 1.3.
    u32 pd_count = 0;
    idld_.vkEnumeratePhysicalDevices(instance_, &pd_count, nullptr);
    std::vector<VkPhysicalDevice> pds(pd_count);
    idld_.vkEnumeratePhysicalDevices(instance_, &pd_count, pds.data());
    VkPhysicalDevice chosen = VK_NULL_HANDLE;
    for (int pass = 0; pass < 2 && chosen == VK_NULL_HANDLE; ++pass) {
        for (VkPhysicalDevice pd : pds) {
            VkPhysicalDeviceProperties p{};
            idld_.vkGetPhysicalDeviceProperties(pd, &p);
            if (p.apiVersion < VK_API_VERSION_1_3) {
                continue;
            }
            if (pass == 1 || p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                chosen = pd;
                timestamp_period_ns_ = p.limits.timestampPeriod;
                std::fprintf(stderr, "[nvn-test] GPU: %s (validation %s)\n", p.deviceName,
                             validation_enabled_ ? "ON" : "OFF");
                break;
            }
        }
    }
    if (chosen == VK_NULL_HANDLE) {
        why = "no Vulkan 1.3 physical device";
        return false;
    }

    try {
        device_ = std::make_unique<Vulkan::Device>(
            instance_, Vulkan::vk::PhysicalDevice(chosen, idld_), VK_NULL_HANDLE, idld_);
    } catch (const std::exception& e) {
        why = std::string("device creation failed: ") + e.what();
        return false;
    }

    const VkCommandPoolCreateInfo pci{.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                      .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                                      .queueFamilyIndex = device_->GetGraphicsFamily()};
    Dld().vkCreateCommandPool(Logical(), &pci, nullptr, &cmd_pool_);
    Dld().vkGetDeviceQueue(Logical(), device_->GetGraphicsFamily(), 0, &queue_);
    const auto gdpa = Dld().vkGetDeviceProcAddr;
    pfn_signal_ = reinterpret_cast<PFN_vkSignalSemaphore>(gdpa(Logical(), "vkSignalSemaphore"));
    pfn_reset_pool_ = reinterpret_cast<PFN_vkResetCommandPool>(gdpa(Logical(), "vkResetCommandPool"));
    pfn_wait_idle_ = reinterpret_cast<PFN_vkQueueWaitIdle>(gdpa(Logical(), "vkQueueWaitIdle"));
    return true;
}

VkTestContext::~VkTestContext() {
    if (device_) {
        WaitIdle();
        if (cmd_pool_) {
            Dld().vkDestroyCommandPool(Logical(), cmd_pool_, nullptr);
        }
    }
    device_.reset();
    if (messenger_) {
        auto destroy = Proc<PFN_vkDestroyDebugUtilsMessengerEXT>(
            idld_.vkGetInstanceProcAddr, instance_, "vkDestroyDebugUtilsMessengerEXT");
        if (destroy) {
            destroy(instance_, messenger_, nullptr);
        }
    }
    if (instance_) {
        idld_.vkDestroyInstance(instance_, nullptr);
    }
}

Vulkan::vk::CommandBuffer VkTestContext::BeginCommands() {
    const VkCommandBufferAllocateInfo ai{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                         .commandPool = cmd_pool_,
                                         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                         .commandBufferCount = 1};
    VkCommandBuffer raw = VK_NULL_HANDLE;
    Dld().vkAllocateCommandBuffers(Logical(), &ai, &raw);
    const VkCommandBufferBeginInfo bi{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    Dld().vkBeginCommandBuffer(raw, &bi);
    return Vulkan::vk::CommandBuffer(raw, Dld());
}

void VkTestContext::Submit(Vulkan::vk::CommandBuffer cmd, std::span<const VkSemaphoreSubmitInfo> waits,
                           std::span<const VkSemaphoreSubmitInfo> signals) {
    VkCommandBuffer raw = *cmd;
    Dld().vkEndCommandBuffer(raw);
    const VkCommandBufferSubmitInfo cbi{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                        .commandBuffer = raw};
    const VkSubmitInfo2 si{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
                           .waitSemaphoreInfoCount = static_cast<u32>(waits.size()),
                           .pWaitSemaphoreInfos = waits.data(),
                           .commandBufferInfoCount = 1,
                           .pCommandBufferInfos = &cbi,
                           .signalSemaphoreInfoCount = static_cast<u32>(signals.size()),
                           .pSignalSemaphoreInfos = signals.data()};
    Dld().vkQueueSubmit2(Queue(), 1, &si, VK_NULL_HANDLE);
}

void VkTestContext::RunAndWait(const std::function<void(Vulkan::vk::CommandBuffer)>& fn) {
    Vulkan::vk::CommandBuffer cmd = BeginCommands();
    fn(cmd);
    Submit(cmd);
    WaitIdle();
}

void VkTestContext::WaitIdle() const {
    pfn_wait_idle_(Queue());
}

void VkTestContext::ResetCommandPool() {
    pfn_reset_pool_(Logical(), cmd_pool_, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
}

void VkTestContext::HostSignal(VkSemaphore sem, u64 value) const {
    const VkSemaphoreSignalInfo info{.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
                                     .semaphore = sem,
                                     .value = value};
    pfn_signal_(Logical(), &info);
}

u64 VkTestContext::CounterValue(VkSemaphore sem) const {
    u64 v = 0;
    Dld().vkGetSemaphoreCounterValue(Logical(), sem, &v);
    return v;
}

} // namespace NvnTest
