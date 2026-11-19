// Shared real-GPU context for nvn_vulkan_tests and nvn_bench.
//
// - Loads vulkan-1.dll / libvulkan.so at runtime (no SDK import lib needed).
// - Creates a Vulkan 1.3 instance; enables VK_LAYER_KHRONOS_validation when it is installed
//   (set NVN_NO_VALIDATION=1 to skip it, e.g. for benchmarks).
// - Counts validation ERROR + WARNING messages so tests can fail on any of them.
// - Builds the standalone Vulkan::Device (src/stubs/vulkan_device_stub.cpp).
// - Offers one-shot command buffer recording + vkQueueSubmit2 helpers.
#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "video_core/vulkan_common/vulkan_device.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace NvnTest {

class VkTestContext {
public:
    /// Process-wide context (created once). Returns nullptr when no usable GPU exists;
    /// `SkipReason()` then says why.
    static VkTestContext* Get();
    static const std::string& SkipReason();

    ~VkTestContext();

    Vulkan::Device& Device() { return *device_; }
    VkDevice Logical() const { return *device_->GetLogical(); }
    VkQueue Queue() const { return queue_; }
    const Vulkan::vk::DeviceDispatch& Dld() const { return device_->GetDispatchLoader(); }
    bool ValidationEnabled() const { return validation_enabled_; }

    /// Number of validation-layer errors/warnings seen so far (monotonic).
    u64 ValidationMessageCount() const { return validation_messages_.load(); }
    const std::string& LastValidationMessage() const { return last_message_; }

    /// Allocates a primary command buffer and begins it (ONE_TIME_SUBMIT).
    Vulkan::vk::CommandBuffer BeginCommands();

    /// Ends `cmd` and submits it with vkQueueSubmit2. Does NOT wait.
    void Submit(Vulkan::vk::CommandBuffer cmd, std::span<const VkSemaphoreSubmitInfo> waits = {},
                std::span<const VkSemaphoreSubmitInfo> signals = {});

    /// Records with `fn`, submits, then vkQueueWaitIdle. Convenience for tests.
    void RunAndWait(const std::function<void(Vulkan::vk::CommandBuffer)>& fn);

    void WaitIdle() const;

    /// Frees all command buffers allocated so far (call after WaitIdle).
    void ResetCommandPool();

    /// Host-side signal of a timeline semaphore (vkSignalSemaphore), used to simulate GPU
    /// completion deterministically.
    void HostSignal(VkSemaphore sem, u64 value) const;
    u64 CounterValue(VkSemaphore sem) const;

    double TimestampPeriodNs() const { return timestamp_period_ns_; }

private:
    VkTestContext() = default;
    bool Init(std::string& why);

    void* library_{nullptr};
    Vulkan::vk::InstanceDispatch idld_{};
    VkInstance instance_{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT messenger_{VK_NULL_HANDLE};
    std::unique_ptr<Vulkan::Device> device_;
    VkCommandPool cmd_pool_{VK_NULL_HANDLE};
    VkQueue queue_{VK_NULL_HANDLE};
    // Not in vk::DeviceDispatch: loaded directly.
    PFN_vkSignalSemaphore pfn_signal_{nullptr};
    PFN_vkResetCommandPool pfn_reset_pool_{nullptr};
    PFN_vkQueueWaitIdle pfn_wait_idle_{nullptr};
    bool validation_enabled_{false};
    double timestamp_period_ns_{1.0};

    std::atomic<u64> validation_messages_{0};
    std::string last_message_;

    friend VkBool32 VKAPI_CALL DebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT*, void*);
};

} // namespace NvnTest
