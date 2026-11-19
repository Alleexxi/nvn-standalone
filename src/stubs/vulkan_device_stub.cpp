// Standalone Device: creates a REAL logical device with exactly the features NvnClean needs.
//
// The full Eglantine Device (vulkan_device.cpp, ~2,200 lines) pulls in settings, fs, Streamline,
// Aftermath, GPU logging... This file keeps the same class layout (vulkan_device.h is unchanged)
// but implements only what the NvnClean modules and the tests touch:
//   - Vulkan 1.3 device, one graphics+compute queue
//   - timelineSemaphore, bufferDeviceAddress, hostQueryReset, synchronization2, dynamicRendering
//   - VK_EXT_conditional_rendering when the GPU exposes it
// Tests that never construct a Device (layout/math tests) don't touch any of this.
#include <cstring>
#include <vector>

#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan {

namespace {
bool HasExtension(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& e : exts) {
        if (std::strcmp(e.extensionName, name) == 0) {
            return true;
        }
    }
    return false;
}
} // namespace

Device::Device(VkInstance instance_, vk::PhysicalDevice physical_, VkSurfaceKHR /*surface*/,
               const vk::InstanceDispatch& instance_dld)
    : instance{instance_}, physical{physical_} {
    // DeviceDispatch derives from InstanceDispatch: seed it with the instance-level pointers.
    static_cast<vk::InstanceDispatch&>(dld) = instance_dld;

    // ---- Properties ----
    properties.driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    properties2 = VkPhysicalDeviceProperties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &properties.driver,
        .properties = {},
    };
    dld.vkGetPhysicalDeviceProperties2(physical, &properties2);
    properties.properties = properties2.properties;
    instance_version = properties.properties.apiVersion;

    // ---- Queue family: first graphics+compute family ----
    const auto families = physical.GetQueueFamilyProperties();
    for (u32 i = 0; i < families.size(); ++i) {
        constexpr VkQueueFlags want = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((families[i].queueFlags & want) == want) {
            graphics_family = i;
            present_family = i;
            break;
        }
    }

    // ---- Extensions ----
    u32 ext_count = 0;
    dld.vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    dld.vkEnumerateDeviceExtensionProperties(physical, nullptr, &ext_count, exts.data());
    for (const auto& e : exts) {
        supported_extensions.emplace(e.extensionName);
    }
    std::vector<const char*> enabled_exts;
    if (HasExtension(exts, VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME)) {
        enabled_exts.push_back(VK_EXT_CONDITIONAL_RENDERING_EXTENSION_NAME);
        extensions.conditional_rendering = true;
    }
    if (HasExtension(exts, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME)) {
        enabled_exts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        extensions.memory_budget = true;
    }

    // ---- Features: query what is supported, enable the subset NvnClean uses ----
    VkPhysicalDeviceConditionalRenderingFeaturesEXT cond{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT};
    VkPhysicalDeviceVulkan13Features f13{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                         .pNext = extensions.conditional_rendering ? &cond : nullptr};
    VkPhysicalDeviceVulkan12Features f12{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                         .pNext = &f13};
    VkPhysicalDeviceFeatures2 query{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    dld.vkGetPhysicalDeviceFeatures2(physical, &query);
    features.features = query.features;

    VkPhysicalDeviceConditionalRenderingFeaturesEXT en_cond{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CONDITIONAL_RENDERING_FEATURES_EXT,
        .conditionalRendering = cond.conditionalRendering};
    VkPhysicalDeviceVulkan13Features en13{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .pNext = extensions.conditional_rendering ? &en_cond : nullptr,
        .synchronization2 = f13.synchronization2,
        .dynamicRendering = f13.dynamicRendering,
        .maintenance4 = f13.maintenance4};
    VkPhysicalDeviceVulkan12Features en12{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        .pNext = &en13,
        .descriptorIndexing = f12.descriptorIndexing,
        .hostQueryReset = f12.hostQueryReset,
        .timelineSemaphore = f12.timelineSemaphore,
        .bufferDeviceAddress = f12.bufferDeviceAddress};
    VkPhysicalDeviceFeatures base{};
    base.occlusionQueryPrecise = query.features.occlusionQueryPrecise;
    base.shaderInt64 = query.features.shaderInt64;
    base.pipelineStatisticsQuery = query.features.pipelineStatisticsQuery;
    VkPhysicalDeviceFeatures2 enable{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
                                     .pNext = &en12, .features = base};

    features.timeline_semaphore.timelineSemaphore = f12.timelineSemaphore;
    features.host_query_reset.hostQueryReset = f12.hostQueryReset;
    features.buffer_device_address.bufferDeviceAddress = f12.bufferDeviceAddress;
    features.synchronization2.synchronization2 = f13.synchronization2;
    extensions.timeline_semaphore = f12.timelineSemaphore;
    extensions.host_query_reset = f12.hostQueryReset;
    extensions.buffer_device_address = f12.bufferDeviceAddress;
    extensions.synchronization2 = f13.synchronization2;
    extensions.conditional_rendering = extensions.conditional_rendering && cond.conditionalRendering;

    // ---- Logical device ----
    const float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_ci{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = graphics_family,
        .queueCount = 1,
        .pQueuePriorities = &priority,
    };
    logical = vk::Device::Create(physical, vk::Span<VkDeviceQueueCreateInfo>(&queue_ci, 1),
                                 vk::Span<const char*>(enabled_exts.data(),
                                                       static_cast<u32>(enabled_exts.size())),
                                 &enable, dld);
    graphics_queue = logical.GetQueue(graphics_family);
    present_queue = graphics_queue;
    allocator = CreateAllocator(false);
}

Device::~Device() {
    if (allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(allocator);
    }
}

VmaAllocator Device::CreateAllocator(bool externally_synchronized) const {
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = dld.vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr = dld.vkGetDeviceProcAddr;
    VmaAllocatorCreateFlags flags =
        externally_synchronized ? VMA_ALLOCATOR_CREATE_EXTERNALLY_SYNCHRONIZED_BIT : 0;
    if (extensions.memory_budget) {
        flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (extensions.buffer_device_address) {
        flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }
    const VmaAllocatorCreateInfo info{
        .flags = flags,
        .physicalDevice = physical,
        .device = *logical,
        .preferredLargeHeapBlockSize = 16u * 1024u * 1024u,
        .pVulkanFunctions = &functions,
        .instance = instance,
        .vulkanApiVersion = VK_API_VERSION_1_3,
    };
    VmaAllocator out = VK_NULL_HANDLE;
    vmaCreateAllocator(&info, &out);
    return out;
}

VkFormat Device::GetSupportedFormat(VkFormat wanted_format, VkFormatFeatureFlags /*wanted_usage*/,
                                    FormatType /*format_type*/) const {
    return wanted_format;
}

bool Device::IsFormatSupported(VkFormat /*wanted_format*/, VkFormatFeatureFlags /*wanted_usage*/,
                               FormatType /*format_type*/) const {
    return true;
}

void Device::ReportLoss() const {}
void Device::SaveShader(std::span<const u32> /*spirv*/) const {}
std::string Device::GetDriverName() const { return properties.driver.driverName; }
bool Device::ShouldBoostClocks() const { return false; }
bool Device::HasTimelineSemaphore() const {
    return features.timeline_semaphore.timelineSemaphore != VK_FALSE;
}

VmaPool Device::GetSeparatePool(SeparatePool /*kind*/, u32 /*memory_type*/, VmaAllocator /*user*/) const {
    return VK_NULL_HANDLE;
}

VmaAllocator Device::CreateThreadSafeAllocator() const {
    return CreateAllocator(false);
}

bool Device::TryReserveCustomBorderColorSamplers(size_t /*count*/) const { return true; }
void Device::ReleaseCustomBorderColorSamplers(size_t /*count*/) const {}

u64 Device::GetDeviceMemoryUsage() const { return 0; }
u64 Device::GetDeviceMemoryBudget() const { return 0; }
u64 Device::GetDeviceMemorySlack() const { return 0; }

bool Device::MustEmulateBGR565() const { return false; }

void Device::InitializeGPULogging() {}
void Device::ShutdownGPULogging() {}

} // namespace Vulkan
