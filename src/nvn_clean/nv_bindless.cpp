// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "nvn_clean/nv_bindless.h"

#include <array>
#include <bit>
#include <mutex>
#include <vector>

#include "common/logging.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::NvnClean {
namespace {

constexpr u32 kSlotFlagResident = 1u << 0;
constexpr u32 kSlotFlagDirty = 1u << 1;
constexpr size_t kInitialScratchCapacity = 256;

void ClearTextureDirtyBit(
    u32 slot, u32& slot_flags,
    std::array<std::atomic<u64>, BindlessDescriptorTable::kTextureBitWords>& dirty_bits,
    std::array<std::atomic<u64>, BindlessDescriptorTable::kTextureSummaryWords>&
        dirty_summary) noexcept {
    slot_flags &= ~kSlotFlagDirty;
    const size_t word_idx = static_cast<size_t>(slot >> 6u);
    const u64 slot_bit = 1ULL << (slot & 63u);
    const u64 prev = dirty_bits[word_idx].fetch_and(~slot_bit, std::memory_order_acq_rel);
    if ((prev & slot_bit) != 0ULL && (prev & ~slot_bit) == 0ULL) {
        const size_t summary_idx = word_idx >> 6u;
        const u64 summary_bit = 1ULL << (word_idx & 63u);
        dirty_summary[summary_idx].fetch_and(~summary_bit, std::memory_order_release);
    }
}

void ClearSamplerDirtyBit(
    u32 slot, u64& slot_flags,
    std::array<std::atomic<u64>, BindlessDescriptorTable::kSamplerBitWords>& dirty_bits,
    std::atomic<u64>& dirty_summary) noexcept {
    slot_flags &= ~u64{kSlotFlagDirty};
    const size_t word_idx = static_cast<size_t>(slot >> 6u);
    const u64 slot_bit = 1ULL << (slot & 63u);
    const u64 prev = dirty_bits[word_idx].fetch_and(~slot_bit, std::memory_order_acq_rel);
    if ((prev & slot_bit) != 0ULL && (prev & ~slot_bit) == 0ULL) {
        const u64 summary_bit = 1ULL << word_idx;
        dirty_summary.fetch_and(~summary_bit, std::memory_order_release);
    }
}

} // namespace

bool BindlessDescriptorTable::IsSupported(const Device& device) noexcept {
    const vk::PhysicalDevice physical = device.GetPhysical();
    if (static_cast<VkPhysicalDevice>(physical) == VK_NULL_HANDLE) {
        return false;
    }

    VkPhysicalDeviceDescriptorIndexingFeatures indexing_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
        .pNext = nullptr,
    };
    VkPhysicalDeviceFeatures2 features2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = &indexing_features,
    };
    physical.GetFeatures2(features2);

    if (!indexing_features.descriptorBindingPartiallyBound ||
        !indexing_features.descriptorBindingSampledImageUpdateAfterBind ||
        !indexing_features.descriptorBindingUpdateUnusedWhilePending ||
        !indexing_features.runtimeDescriptorArray) {
        return false;
    }

    VkPhysicalDeviceDescriptorIndexingProperties indexing_props{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES,
        .pNext = nullptr,
    };
    VkPhysicalDeviceProperties2 props2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &indexing_props,
    };
    physical.GetProperties2(props2);

    return indexing_props.maxDescriptorSetUpdateAfterBindSampledImages >=
               kMaxBindlessSampledImages &&
           indexing_props.maxPerStageDescriptorUpdateAfterBindSampledImages >=
               kMaxBindlessSampledImages &&
           indexing_props.maxDescriptorSetUpdateAfterBindSamplers >= kMaxBindlessSamplers &&
           indexing_props.maxPerStageDescriptorUpdateAfterBindSamplers >= kMaxBindlessSamplers &&
           indexing_props.maxUpdateAfterBindDescriptorsInAllPools >=
               (kMaxBindlessSampledImages + kMaxBindlessSamplers);
}

BindlessDescriptorTable::BindlessDescriptorTable(const Device& device, VmaAllocator allocator,
                                                 const BindlessTableConfig& config)
    : device_{device}, dld_{device.GetDispatchLoader()}, config_{config} {
    if (allocator != VK_NULL_HANDLE) {
        allocator_ = allocator;
    } else {
        owned_allocator_ = device_.CreateThreadSafeAllocator();
        allocator_ = owned_allocator_;
    }

    texture_slots_.resize(kMaxBindlessSampledImages);
    sampler_slots_.resize(kMaxBindlessSamplers);
    scratch_image_infos_.reserve(kInitialScratchCapacity);
    scratch_writes_.reserve(kInitialScratchCapacity);

    try {
        CreateDescriptorTable();
        CreateFallbackResources();
    } catch (...) {
        fallback_sampler_.reset();
        fallback_image_view_.reset();
        fallback_image_.reset();
        descriptor_pool_.reset();
        set_layout_.reset();
        persistent_set_ = VK_NULL_HANDLE;
        if (owned_allocator_ != VK_NULL_HANDLE) {
            vmaDestroyAllocator(owned_allocator_);
            owned_allocator_ = VK_NULL_HANDLE;
            allocator_ = VK_NULL_HANDLE;
        }
        throw;
    }
}

BindlessDescriptorTable::~BindlessDescriptorTable() {
    fallback_sampler_.reset();
    fallback_image_view_.reset();
    fallback_image_.reset();
    descriptor_pool_.reset();
    set_layout_.reset();
    persistent_set_ = VK_NULL_HANDLE;

    if (owned_allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(owned_allocator_);
        owned_allocator_ = VK_NULL_HANDLE;
        allocator_ = VK_NULL_HANDLE;
    }
}

void BindlessDescriptorTable::CreateDescriptorTable() {
    const std::array<VkDescriptorSetLayoutBinding, 2> bindings{{
        VkDescriptorSetLayoutBinding{
            .binding = kBindlessSampledImageBinding,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = kMaxBindlessSampledImages,
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        },
        VkDescriptorSetLayoutBinding{
            .binding = kBindlessSamplerBinding,
            .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = kMaxBindlessSamplers,
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        },
    }};

    constexpr VkDescriptorBindingFlags kBindlessFlags =
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    const std::array<VkDescriptorBindingFlags, 2> binding_flags{kBindlessFlags, kBindlessFlags};

    const VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags_ci{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
        .pNext = nullptr,
        .bindingCount = static_cast<u32>(binding_flags.size()),
        .pBindingFlags = binding_flags.data(),
    };

    set_layout_ = device_.GetLogical().CreateDescriptorSetLayout(VkDescriptorSetLayoutCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = &binding_flags_ci,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    });

    const std::array<VkDescriptorPoolSize, 2> pool_sizes{{
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = kMaxBindlessSampledImages,
        },
        VkDescriptorPoolSize{
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = kMaxBindlessSamplers,
        },
    }};

    descriptor_pool_ = device_.GetLogical().CreateDescriptorPool(VkDescriptorPoolCreateInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
        .maxSets = 1,
        .poolSizeCount = static_cast<u32>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    });
    descriptor_pool_.SetObjectNameEXT("NvnClean::BindlessDescriptorPool");

    const VkDescriptorSetLayout raw_layout = *set_layout_;
    const VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = *descriptor_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &raw_layout,
    };
    vk::Check(dld_.vkAllocateDescriptorSets(*device_.GetLogical(), &alloc_info, &persistent_set_));
}

void BindlessDescriptorTable::CreateFallbackResources() {
    constexpr VkImageUsageFlags kFallbackUsage =
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    const VkImageCreateInfo image_ci{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {1, 1, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = kFallbackUsage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        .requiredFlags = 0,
        .preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 0.0f,
    };

    VkImage raw_image{VK_NULL_HANDLE};
    VmaAllocation image_alloc{VK_NULL_HANDLE};
    VkResult res = vmaCreateImage(allocator_, &image_ci, &alloc_ci, &raw_image, &image_alloc,
                                  nullptr);
    if (res != VK_SUCCESS) {
        alloc_ci.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        res = vmaCreateImage(allocator_, &image_ci, &alloc_ci, &raw_image, &image_alloc, nullptr);
    }
    if (res != VK_SUCCESS) {
        alloc_ci.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_ci.preferredFlags = 0;
        res = vmaCreateImage(allocator_, &image_ci, &alloc_ci, &raw_image, &image_alloc, nullptr);
    }
    vk::Check(res);

    fallback_image_ = vk::Image(raw_image, kFallbackUsage, *device_.GetLogical(), allocator_,
                                image_alloc, dld_);
    fallback_image_.SetObjectNameEXT("NvnClean::BindlessFallbackImage");

    // Transition the 1x1 fallback image to TRANSFER_DST_OPTIMAL, clear to transparent black
    // (0, 0, 0, 0), and transition to SHADER_READ_ONLY_OPTIMAL so unbound handles sample safely.
    {
        vk::CommandPool cmd_pool = device_.GetLogical().CreateCommandPool(VkCommandPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = device_.GetGraphicsFamily(),
        });
        vk::CommandBuffers cmd_bufs = cmd_pool.Allocate(1, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
        vk::CommandBuffer cmd(cmd_bufs[0], dld_);

        cmd.Begin(VkCommandBufferBeginInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = nullptr,
        });

        const VkImageSubresourceRange subresource_range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };

        VkImageMemoryBarrier barrier{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .pNext = nullptr,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = *fallback_image_,
            .subresourceRange = subresource_range,
        };
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            barrier);

        const VkClearColorValue transparent_black{
            .float32 = {0.0f, 0.0f, 0.0f, 0.0f},
        };
        cmd.ClearColorImage(*fallback_image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            transparent_black, subresource_range);

        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        cmd.PipelineBarrier(VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                            barrier);

        cmd.End();

        vk::Fence fence = device_.GetLogical().CreateFence(VkFenceCreateInfo{
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
        });
        const VkCommandBuffer raw_cmd = *cmd;
        const VkSubmitInfo submit_info{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .pWaitDstStageMask = nullptr,
            .commandBufferCount = 1,
            .pCommandBuffers = &raw_cmd,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = nullptr,
        };
        vk::Check(device_.GetGraphicsQueue().Submit(submit_info, *fence));
        vk::Check(fence.Wait());
    }

    fallback_image_view_ = device_.GetLogical().CreateImageView(VkImageViewCreateInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = *fallback_image_,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .components = {
            .r = VK_COMPONENT_SWIZZLE_IDENTITY,
            .g = VK_COMPONENT_SWIZZLE_IDENTITY,
            .b = VK_COMPONENT_SWIZZLE_IDENTITY,
            .a = VK_COMPONENT_SWIZZLE_IDENTITY,
        },
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });
    fallback_image_view_.SetObjectNameEXT("NvnClean::BindlessFallbackImageView");

    fallback_sampler_ = device_.GetLogical().CreateSampler(VkSamplerCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
        .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    });

    if (config_.populate_safe_fallback_slots && persistent_set_ != VK_NULL_HANDLE) {
        const std::vector<VkDescriptorImageInfo> fallback_images(
            kMaxBindlessSampledImages,
            VkDescriptorImageInfo{
                .sampler = VK_NULL_HANDLE,
                .imageView = *fallback_image_view_,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            });
        const std::vector<VkDescriptorImageInfo> fallback_samplers(
            kMaxBindlessSamplers,
            VkDescriptorImageInfo{
                .sampler = *fallback_sampler_,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            });

        const std::array<VkWriteDescriptorSet, 2> writes{{
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = persistent_set_,
                .dstBinding = kBindlessSampledImageBinding,
                .dstArrayElement = 0,
                .descriptorCount = kMaxBindlessSampledImages,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                .pImageInfo = fallback_images.data(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
            VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = persistent_set_,
                .dstBinding = kBindlessSamplerBinding,
                .dstArrayElement = 0,
                .descriptorCount = kMaxBindlessSamplers,
                .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                .pImageInfo = fallback_samplers.data(),
                .pBufferInfo = nullptr,
                .pTexelBufferView = nullptr,
            },
        }};

        dld_.vkUpdateDescriptorSets(*device_.GetLogical(), static_cast<u32>(writes.size()),
                                    writes.data(), 0, nullptr);
    }
}

void BindlessDescriptorTable::WriteImageDescriptorImmediate(u32 slot, VkImageView view,
                                                            VkImageLayout layout) noexcept {
    if (persistent_set_ == VK_NULL_HANDLE || view == VK_NULL_HANDLE) [[unlikely]] {
        return;
    }
    const VkDescriptorImageInfo image_info{
        .sampler = VK_NULL_HANDLE,
        .imageView = view,
        .imageLayout = layout,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = persistent_set_,
        .dstBinding = kBindlessSampledImageBinding,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
        .pImageInfo = &image_info,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    dld_.vkUpdateDescriptorSets(*device_.GetLogical(), 1, &write, 0, nullptr);
}

void BindlessDescriptorTable::WriteSamplerDescriptorImmediate(u32 slot,
                                                              VkSampler sampler) noexcept {
    if (persistent_set_ == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE) [[unlikely]] {
        return;
    }
    const VkDescriptorImageInfo sampler_info{
        .sampler = sampler,
        .imageView = VK_NULL_HANDLE,
        .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    const VkWriteDescriptorSet write{
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = nullptr,
        .dstSet = persistent_set_,
        .dstBinding = kBindlessSamplerBinding,
        .dstArrayElement = slot,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
        .pImageInfo = &sampler_info,
        .pBufferInfo = nullptr,
        .pTexelBufferView = nullptr,
    };
    dld_.vkUpdateDescriptorSets(*device_.GetLogical(), 1, &write, 0, nullptr);
}

void BindlessDescriptorTable::MarkTextureSlotDirtyLocked(u32 slot) noexcept {
    const size_t word_idx = static_cast<size_t>(slot >> 6u);
    const u64 slot_bit = 1ULL << (slot & 63u);
    const size_t summary_idx = word_idx >> 6u;
    const u64 summary_bit = 1ULL << (word_idx & 63u);

    texture_slots_[slot].flags |= kSlotFlagDirty;
    texture_dirty_bits_[word_idx].fetch_or(slot_bit, std::memory_order_release);
    texture_dirty_summary_[summary_idx].fetch_or(summary_bit, std::memory_order_release);
}

void BindlessDescriptorTable::MarkSamplerSlotDirtyLocked(u32 slot) noexcept {
    const size_t word_idx = static_cast<size_t>(slot >> 6u);
    const u64 slot_bit = 1ULL << (slot & 63u);
    const u64 summary_bit = 1ULL << word_idx;

    sampler_slots_[slot].flags |= u64{kSlotFlagDirty};
    sampler_dirty_bits_[word_idx].fetch_or(slot_bit, std::memory_order_release);
    sampler_dirty_summary_.fetch_or(summary_bit, std::memory_order_release);
}

void BindlessDescriptorTable::BumpEpochLocked(bool mark_dirty) noexcept {
    const u64 prev_state = packed_epoch_state_.load(std::memory_order_relaxed);
    const u64 next_epoch = (prev_state >> 1u) + 1ULL;
    const u64 next_dirty = (prev_state & 1ULL) | (mark_dirty ? 1ULL : 0ULL);
    packed_epoch_state_.store((next_epoch << 1u) | next_dirty, std::memory_order_release);
}

bool BindlessDescriptorTable::OnTextureInitialize(u32 texture_id, VkImageView image_view,
                                                  VkImageLayout layout,
                                                  u64 texture_serial) noexcept {
    if (texture_id >= kMaxBindlessSampledImages || image_view == VK_NULL_HANDLE ||
        layout == VK_IMAGE_LAYOUT_UNDEFINED) [[unlikely]] {
        return false;
    }

    std::unique_lock lock{mutex_};
    BindlessTextureSlot& slot = texture_slots_[texture_id];
    const bool was_resident = (slot.flags & kSlotFlagResident) != 0u;
    if (was_resident && slot.image_view == image_view && slot.image_layout == layout &&
        slot.texture_serial == texture_serial) {
        return true;
    }

    slot.image_view = image_view;
    slot.texture_serial = texture_serial;
    slot.image_layout = layout;
    slot.flags |= kSlotFlagResident;

    const size_t word_idx = static_cast<size_t>(texture_id >> 6u);
    const u64 slot_bit = 1ULL << (texture_id & 63u);

    if (config_.immediate_write_on_init) {
        WriteImageDescriptorImmediate(texture_id, image_view, layout);
        ClearTextureDirtyBit(texture_id, slot.flags, texture_dirty_bits_, texture_dirty_summary_);
        if (!was_resident) {
            texture_resident_bits_[word_idx].fetch_or(slot_bit, std::memory_order_release);
            resident_textures_.fetch_add(1u, std::memory_order_relaxed);
        }
        BumpEpochLocked(/*mark_dirty=*/false);
        slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
        immediate_texture_writes_.fetch_add(1ULL, std::memory_order_relaxed);
    } else {
        MarkTextureSlotDirtyLocked(texture_id);
        if (!was_resident) {
            texture_resident_bits_[word_idx].fetch_or(slot_bit, std::memory_order_release);
            resident_textures_.fetch_add(1u, std::memory_order_relaxed);
        }
        BumpEpochLocked(/*mark_dirty=*/true);
        slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
    }

    return true;
}

bool BindlessDescriptorTable::OnSamplerInitialize(u32 sampler_id, VkSampler sampler,
                                                  u64 sampler_serial) noexcept {
    if (sampler_id >= kMaxBindlessSamplers || sampler == VK_NULL_HANDLE) [[unlikely]] {
        return false;
    }

    std::unique_lock lock{mutex_};
    BindlessSamplerSlot& slot = sampler_slots_[sampler_id];
    const bool was_resident = (slot.flags & u64{kSlotFlagResident}) != 0ULL;
    if (was_resident && slot.sampler == sampler && slot.sampler_serial == sampler_serial) {
        return true;
    }

    slot.sampler = sampler;
    slot.sampler_serial = sampler_serial;
    slot.flags |= u64{kSlotFlagResident};

    const size_t word_idx = static_cast<size_t>(sampler_id >> 6u);
    const u64 slot_bit = 1ULL << (sampler_id & 63u);

    if (config_.immediate_write_on_init) {
        WriteSamplerDescriptorImmediate(sampler_id, sampler);
        ClearSamplerDirtyBit(sampler_id, slot.flags, sampler_dirty_bits_, sampler_dirty_summary_);
        if (!was_resident) {
            sampler_resident_bits_[word_idx].fetch_or(slot_bit, std::memory_order_release);
            resident_samplers_.fetch_add(1u, std::memory_order_relaxed);
        }
        BumpEpochLocked(/*mark_dirty=*/false);
        slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
        immediate_sampler_writes_.fetch_add(1ULL, std::memory_order_relaxed);
    } else {
        MarkSamplerSlotDirtyLocked(sampler_id);
        if (!was_resident) {
            sampler_resident_bits_[word_idx].fetch_or(slot_bit, std::memory_order_release);
            resident_samplers_.fetch_add(1u, std::memory_order_relaxed);
        }
        BumpEpochLocked(/*mark_dirty=*/true);
        slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
    }

    return true;
}

bool BindlessDescriptorTable::UpdateTextureLayout(u32 texture_id, VkImageLayout new_layout,
                                                  bool defer_flush) noexcept {
    if (texture_id >= kMaxBindlessSampledImages || new_layout == VK_IMAGE_LAYOUT_UNDEFINED)
        [[unlikely]] {
        return false;
    }

    std::unique_lock lock{mutex_};
    BindlessTextureSlot& slot = texture_slots_[texture_id];
    if ((slot.flags & kSlotFlagResident) == 0u || slot.image_view == VK_NULL_HANDLE) {
        return false;
    }

    if (slot.image_layout == new_layout) {
        if (!defer_flush && (slot.flags & kSlotFlagDirty) != 0u) {
            WriteImageDescriptorImmediate(texture_id, slot.image_view, new_layout);
            ClearTextureDirtyBit(texture_id, slot.flags, texture_dirty_bits_,
                                 texture_dirty_summary_);
            BumpEpochLocked(/*mark_dirty=*/false);
            slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
            layout_rebind_writes_.fetch_add(1ULL, std::memory_order_relaxed);
        }
        return true;
    }

    slot.image_layout = new_layout;
    if (defer_flush) {
        MarkTextureSlotDirtyLocked(texture_id);
        BumpEpochLocked(/*mark_dirty=*/true);
        slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
    } else {
        WriteImageDescriptorImmediate(texture_id, slot.image_view, new_layout);
        ClearTextureDirtyBit(texture_id, slot.flags, texture_dirty_bits_, texture_dirty_summary_);
        BumpEpochLocked(/*mark_dirty=*/false);
        slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
    }

    layout_rebind_writes_.fetch_add(1ULL, std::memory_order_relaxed);
    return true;
}

void BindlessDescriptorTable::OnTextureFinalize(u32 texture_id, u64 expected_serial) noexcept {
    if (texture_id >= kMaxBindlessSampledImages) [[unlikely]] {
        return;
    }

    std::unique_lock lock{mutex_};
    BindlessTextureSlot& slot = texture_slots_[texture_id];
    if ((slot.flags & kSlotFlagResident) == 0u) {
        return;
    }
    if (expected_serial != 0ULL && slot.texture_serial != expected_serial) {
        return;
    }

    const size_t word_idx = static_cast<size_t>(texture_id >> 6u);
    const u64 slot_bit = 1ULL << (texture_id & 63u);
    texture_resident_bits_[word_idx].fetch_and(~slot_bit, std::memory_order_release);
    resident_textures_.fetch_sub(1u, std::memory_order_relaxed);

    ClearTextureDirtyBit(texture_id, slot.flags, texture_dirty_bits_, texture_dirty_summary_);
    slot.image_view = VK_NULL_HANDLE;
    slot.texture_serial = 0ULL;
    slot.image_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    slot.flags = 0u;

    if (fallback_image_view_) {
        WriteImageDescriptorImmediate(texture_id, *fallback_image_view_,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    BumpEpochLocked(/*mark_dirty=*/false);
    slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
}

void BindlessDescriptorTable::OnSamplerFinalize(u32 sampler_id, u64 expected_serial) noexcept {
    if (sampler_id >= kMaxBindlessSamplers) [[unlikely]] {
        return;
    }

    std::unique_lock lock{mutex_};
    BindlessSamplerSlot& slot = sampler_slots_[sampler_id];
    if ((slot.flags & u64{kSlotFlagResident}) == 0ULL) {
        return;
    }
    if (expected_serial != 0ULL && slot.sampler_serial != expected_serial) {
        return;
    }

    const size_t word_idx = static_cast<size_t>(sampler_id >> 6u);
    const u64 slot_bit = 1ULL << (sampler_id & 63u);
    sampler_resident_bits_[word_idx].fetch_and(~slot_bit, std::memory_order_release);
    resident_samplers_.fetch_sub(1u, std::memory_order_relaxed);

    ClearSamplerDirtyBit(sampler_id, slot.flags, sampler_dirty_bits_, sampler_dirty_summary_);
    slot.sampler = VK_NULL_HANDLE;
    slot.sampler_serial = 0ULL;
    slot.flags = 0ULL;

    if (fallback_sampler_) {
        WriteSamplerDescriptorImmediate(sampler_id, *fallback_sampler_);
    }

    BumpEpochLocked(/*mark_dirty=*/false);
    slot.updated_epoch = packed_epoch_state_.load(std::memory_order_relaxed) >> 1u;
}

u32 BindlessDescriptorTable::FlushDirtyDescriptors() noexcept {
    if (!HasDirtyEpochBit()) [[likely]] {
        return 0u;
    }

    std::unique_lock lock{mutex_};
    const u64 state_before = packed_epoch_state_.load(std::memory_order_relaxed);
    if ((state_before & 1ULL) == 0ULL) {
        return 0u;
    }

    scratch_image_infos_.clear();
    scratch_writes_.clear();

    const u64 flush_epoch = state_before >> 1u;
    const VkImageView fallback_view = *fallback_image_view_;
    const VkSampler fallback_smp = *fallback_sampler_;

    // Phase 1: Collect all dirty texture and sampler descriptors via 2-level hierarchical bitmask.
    for (size_t summary_idx = 0; summary_idx < kTextureSummaryWords; ++summary_idx) {
        u64 summary_word = texture_dirty_summary_[summary_idx].load(std::memory_order_relaxed);
        while (summary_word != 0ULL) {
            const u32 word_offset = static_cast<u32>(std::countr_zero(summary_word));
            summary_word &= (summary_word - 1ULL);

            const size_t word_idx = (summary_idx << 6u) + static_cast<size_t>(word_offset);
            u64 dirty_word = texture_dirty_bits_[word_idx].load(std::memory_order_relaxed);
            while (dirty_word != 0ULL) {
                const u32 bit_offset = static_cast<u32>(std::countr_zero(dirty_word));
                dirty_word &= (dirty_word - 1ULL);

                const u32 tex_id = static_cast<u32>((word_idx << 6u) + bit_offset);
                const BindlessTextureSlot& slot = texture_slots_[tex_id];
                const bool use_live =
                    (slot.flags & kSlotFlagResident) != 0u && slot.image_view != VK_NULL_HANDLE;
                const VkImageView view = use_live ? slot.image_view : fallback_view;
                const VkImageLayout layout =
                    use_live ? slot.image_layout : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

                if (view != VK_NULL_HANDLE) [[likely]] {
                    scratch_image_infos_.push_back(VkDescriptorImageInfo{
                        .sampler = VK_NULL_HANDLE,
                        .imageView = view,
                        .imageLayout = layout,
                    });
                    scratch_writes_.push_back(VkWriteDescriptorSet{
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .pNext = nullptr,
                        .dstSet = persistent_set_,
                        .dstBinding = kBindlessSampledImageBinding,
                        .dstArrayElement = tex_id,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        .pImageInfo = nullptr,
                        .pBufferInfo = nullptr,
                        .pTexelBufferView = nullptr,
                    });
                }
            }
        }
    }

    {
        u64 sampler_summary = sampler_dirty_summary_.load(std::memory_order_relaxed);
        while (sampler_summary != 0ULL) {
            const u32 word_idx = static_cast<u32>(std::countr_zero(sampler_summary));
            sampler_summary &= (sampler_summary - 1ULL);

            u64 dirty_word = sampler_dirty_bits_[word_idx].load(std::memory_order_relaxed);
            while (dirty_word != 0ULL) {
                const u32 bit_offset = static_cast<u32>(std::countr_zero(dirty_word));
                dirty_word &= (dirty_word - 1ULL);

                const u32 smp_id = (word_idx << 6u) + bit_offset;
                const BindlessSamplerSlot& slot = sampler_slots_[smp_id];
                const bool use_live = (slot.flags & u64{kSlotFlagResident}) != 0ULL &&
                                      slot.sampler != VK_NULL_HANDLE;
                const VkSampler smp = use_live ? slot.sampler : fallback_smp;

                if (smp != VK_NULL_HANDLE) [[likely]] {
                    scratch_image_infos_.push_back(VkDescriptorImageInfo{
                        .sampler = smp,
                        .imageView = VK_NULL_HANDLE,
                        .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                    });
                    scratch_writes_.push_back(VkWriteDescriptorSet{
                        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        .pNext = nullptr,
                        .dstSet = persistent_set_,
                        .dstBinding = kBindlessSamplerBinding,
                        .dstArrayElement = smp_id,
                        .descriptorCount = 1,
                        .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                        .pImageInfo = nullptr,
                        .pBufferInfo = nullptr,
                        .pTexelBufferView = nullptr,
                    });
                }
            }
        }
    }

    // Phase 2: Issue a single batched vkUpdateDescriptorSets call before publishing clean bits.
    const u32 write_count = static_cast<u32>(scratch_writes_.size());
    if (write_count > 0u && persistent_set_ != VK_NULL_HANDLE) {
        for (size_t i = 0; i < scratch_writes_.size(); ++i) {
            scratch_writes_[i].pImageInfo = &scratch_image_infos_[i];
        }
        dld_.vkUpdateDescriptorSets(*device_.GetLogical(), write_count, scratch_writes_.data(), 0,
                                    nullptr);
        coalesced_flush_calls_.fetch_add(1ULL, std::memory_order_relaxed);
        coalesced_descriptor_writes_.fetch_add(write_count, std::memory_order_relaxed);
    }

    // Phase 3: Clear per-slot dirty flags and hierarchical bitmasks only after GPU Set 0 is updated.
    for (size_t summary_idx = 0; summary_idx < kTextureSummaryWords; ++summary_idx) {
        u64 summary_word =
            texture_dirty_summary_[summary_idx].exchange(0ULL, std::memory_order_acq_rel);
        while (summary_word != 0ULL) {
            const u32 word_offset = static_cast<u32>(std::countr_zero(summary_word));
            summary_word &= (summary_word - 1ULL);

            const size_t word_idx = (summary_idx << 6u) + static_cast<size_t>(word_offset);
            u64 dirty_word =
                texture_dirty_bits_[word_idx].exchange(0ULL, std::memory_order_acq_rel);
            while (dirty_word != 0ULL) {
                const u32 bit_offset = static_cast<u32>(std::countr_zero(dirty_word));
                dirty_word &= (dirty_word - 1ULL);

                const u32 tex_id = static_cast<u32>((word_idx << 6u) + bit_offset);
                BindlessTextureSlot& slot = texture_slots_[tex_id];
                slot.flags &= ~kSlotFlagDirty;
                slot.updated_epoch = flush_epoch;
            }
        }
    }

    {
        u64 sampler_summary = sampler_dirty_summary_.exchange(0ULL, std::memory_order_acq_rel);
        while (sampler_summary != 0ULL) {
            const u32 word_idx = static_cast<u32>(std::countr_zero(sampler_summary));
            sampler_summary &= (sampler_summary - 1ULL);

            u64 dirty_word =
                sampler_dirty_bits_[word_idx].exchange(0ULL, std::memory_order_acq_rel);
            while (dirty_word != 0ULL) {
                const u32 bit_offset = static_cast<u32>(std::countr_zero(dirty_word));
                dirty_word &= (dirty_word - 1ULL);

                const u32 smp_id = (word_idx << 6u) + bit_offset;
                BindlessSamplerSlot& slot = sampler_slots_[smp_id];
                slot.flags &= ~u64{kSlotFlagDirty};
                slot.updated_epoch = flush_epoch;
            }
        }
    }

    // Phase 4: Clear the global epoch dirty bit in O(1).
    packed_epoch_state_.store(state_before & ~1ULL, std::memory_order_release);
    return write_count;
}

void BindlessDescriptorTable::ResetAllSlots() noexcept {
    std::unique_lock lock{mutex_};

    scratch_image_infos_.clear();
    scratch_writes_.clear();

    for (auto& summary : texture_dirty_summary_) {
        summary.store(0ULL, std::memory_order_release);
    }
    sampler_dirty_summary_.store(0ULL, std::memory_order_release);

    const VkImageView fallback_view = *fallback_image_view_;
    const VkSampler fallback_smp = *fallback_sampler_;

    for (size_t word_idx = 0; word_idx < kTextureBitWords; ++word_idx) {
        u64 active_mask =
            texture_resident_bits_[word_idx].exchange(0ULL, std::memory_order_acq_rel) |
            texture_dirty_bits_[word_idx].exchange(0ULL, std::memory_order_acq_rel);
        while (active_mask != 0ULL) {
            const u32 bit_offset = static_cast<u32>(std::countr_zero(active_mask));
            active_mask &= (active_mask - 1ULL);

            const u32 tex_id = static_cast<u32>((word_idx << 6u) + bit_offset);
            texture_slots_[tex_id] = BindlessTextureSlot{};
            if (fallback_view != VK_NULL_HANDLE) {
                scratch_image_infos_.push_back(VkDescriptorImageInfo{
                    .sampler = VK_NULL_HANDLE,
                    .imageView = fallback_view,
                    .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                });
                scratch_writes_.push_back(VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = persistent_set_,
                    .dstBinding = kBindlessSampledImageBinding,
                    .dstArrayElement = tex_id,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    .pImageInfo = nullptr,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                });
            }
        }
    }

    for (size_t word_idx = 0; word_idx < kSamplerBitWords; ++word_idx) {
        u64 active_mask =
            sampler_resident_bits_[word_idx].exchange(0ULL, std::memory_order_acq_rel) |
            sampler_dirty_bits_[word_idx].exchange(0ULL, std::memory_order_acq_rel);
        while (active_mask != 0ULL) {
            const u32 bit_offset = static_cast<u32>(std::countr_zero(active_mask));
            active_mask &= (active_mask - 1ULL);

            const u32 smp_id = static_cast<u32>((word_idx << 6u) + bit_offset);
            sampler_slots_[smp_id] = BindlessSamplerSlot{};
            if (fallback_smp != VK_NULL_HANDLE) {
                scratch_image_infos_.push_back(VkDescriptorImageInfo{
                    .sampler = fallback_smp,
                    .imageView = VK_NULL_HANDLE,
                    .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                });
                scratch_writes_.push_back(VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .pNext = nullptr,
                    .dstSet = persistent_set_,
                    .dstBinding = kBindlessSamplerBinding,
                    .dstArrayElement = smp_id,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER,
                    .pImageInfo = nullptr,
                    .pBufferInfo = nullptr,
                    .pTexelBufferView = nullptr,
                });
            }
        }
    }

    if (!scratch_writes_.empty() && persistent_set_ != VK_NULL_HANDLE) {
        for (size_t i = 0; i < scratch_writes_.size(); ++i) {
            scratch_writes_[i].pImageInfo = &scratch_image_infos_[i];
        }
        dld_.vkUpdateDescriptorSets(*device_.GetLogical(), static_cast<u32>(scratch_writes_.size()),
                                    scratch_writes_.data(), 0, nullptr);
    }

    resident_textures_.store(0u, std::memory_order_relaxed);
    resident_samplers_.store(0u, std::memory_order_relaxed);

    const u64 prev_state = packed_epoch_state_.load(std::memory_order_relaxed);
    const u64 next_epoch = (prev_state >> 1u) + 1ULL;
    packed_epoch_state_.store(next_epoch << 1u, std::memory_order_release);
}

BindlessStats BindlessDescriptorTable::Stats() const noexcept {
    return BindlessStats{
        .current_epoch = CurrentEpoch(),
        .resident_textures = resident_textures_.load(std::memory_order_relaxed),
        .resident_samplers = resident_samplers_.load(std::memory_order_relaxed),
        .immediate_texture_writes = immediate_texture_writes_.load(std::memory_order_relaxed),
        .immediate_sampler_writes = immediate_sampler_writes_.load(std::memory_order_relaxed),
        .coalesced_flush_calls = coalesced_flush_calls_.load(std::memory_order_relaxed),
        .coalesced_descriptor_writes = coalesced_descriptor_writes_.load(std::memory_order_relaxed),
        .layout_rebind_writes = layout_rebind_writes_.load(std::memory_order_relaxed),
        .epoch_fast_hits = epoch_fast_hits_.load(std::memory_order_relaxed),
    };
}

} // namespace Vulkan::NvnClean
