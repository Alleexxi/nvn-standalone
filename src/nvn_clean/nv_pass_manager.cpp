// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "nvn_clean/nv_pass_manager.h"

#include <algorithm>
#include <utility>

#include "common/logging.h"
#include "nvn_clean/nv_bindless.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::NvnClean {
namespace {

constexpr VkAccessFlags2 kWriteAccessMask =
    VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
    VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

constexpr u32 kAllKnownNvnBarrierBits =
    NvnBarrierBits::kOrderPrimitives | NvnBarrierBits::kOrderFragments |
    NvnBarrierBits::kOrderFragToTex | NvnBarrierBits::kOrderCompute |
    NvnBarrierBits::kOrderIndirectData | NvnBarrierBits::kInvalidateTexture |
    NvnBarrierBits::kInvalidateShader | NvnBarrierBits::kInvalidateTextureDescriptor |
    NvnBarrierBits::kInvalidateZcull;

[[nodiscard]] PFN_vkCmdBeginRendering ResolveCmdBeginRendering(
    VkDevice vk_device, const vk::DeviceDispatch& dld) noexcept {
    if (vk_device == VK_NULL_HANDLE || dld.vkGetDeviceProcAddr == nullptr) {
        return nullptr;
    }
    if (auto* fn = reinterpret_cast<PFN_vkCmdBeginRendering>(
            dld.vkGetDeviceProcAddr(vk_device, "vkCmdBeginRendering"))) {
        return fn;
    }
    return reinterpret_cast<PFN_vkCmdBeginRendering>(
        dld.vkGetDeviceProcAddr(vk_device, "vkCmdBeginRenderingKHR"));
}

[[nodiscard]] PFN_vkCmdEndRendering ResolveCmdEndRendering(
    VkDevice vk_device, const vk::DeviceDispatch& dld) noexcept {
    if (vk_device == VK_NULL_HANDLE || dld.vkGetDeviceProcAddr == nullptr) {
        return nullptr;
    }
    if (auto* fn = reinterpret_cast<PFN_vkCmdEndRendering>(
            dld.vkGetDeviceProcAddr(vk_device, "vkCmdEndRendering"))) {
        return fn;
    }
    return reinterpret_cast<PFN_vkCmdEndRendering>(
        dld.vkGetDeviceProcAddr(vk_device, "vkCmdEndRenderingKHR"));
}

[[nodiscard]] PFN_vkCmdPipelineBarrier2 ResolveCmdPipelineBarrier2(
    VkDevice vk_device, const vk::DeviceDispatch& dld) noexcept {
    if (dld.vkCmdPipelineBarrier2 != nullptr) {
        return dld.vkCmdPipelineBarrier2;
    }
    if (vk_device == VK_NULL_HANDLE || dld.vkGetDeviceProcAddr == nullptr) {
        return nullptr;
    }
    if (auto* fn = reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
            dld.vkGetDeviceProcAddr(vk_device, "vkCmdPipelineBarrier2"))) {
        return fn;
    }
    return reinterpret_cast<PFN_vkCmdPipelineBarrier2>(
        dld.vkGetDeviceProcAddr(vk_device, "vkCmdPipelineBarrier2KHR"));
}

struct StageAccessPair {
    VkPipelineStageFlags2 stages;
    VkAccessFlags2 access;
};

[[nodiscard]] constexpr StageAccessPair DefaultStageAccessForLayout(
    VkImageLayout layout) noexcept {
    switch (layout) {
    case VK_IMAGE_LAYOUT_UNDEFINED:
    case VK_IMAGE_LAYOUT_PREINITIALIZED:
        return {VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, VK_ACCESS_2_NONE};
    case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
    case VK_IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    case VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL:
    case VK_IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT};
    case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        return {VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
    case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
        return {VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_ACCESS_2_NONE};
    case VK_IMAGE_LAYOUT_GENERAL:
    default:
        return {VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    }
}

[[nodiscard]] constexpr bool IsShaderSampledDescriptorLayout(VkImageLayout layout) noexcept {
    return layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL ||
           layout == VK_IMAGE_LAYOUT_GENERAL ||
           layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
}

void SyncBindlessSlotLayoutIfNeeded(BindlessDescriptorTable* bindless_table, u32 bindless_tex_id,
                                    VkImageLayout new_layout) noexcept {
    if (bindless_table == nullptr || bindless_tex_id >= kMaxBindlessSampledImages ||
        !IsShaderSampledDescriptorLayout(new_layout)) {
        return;
    }
    if (bindless_table->TextureSlot(bindless_tex_id).image_layout == new_layout) [[likely]] {
        return;
    }
    bindless_table->UpdateTextureLayout(bindless_tex_id, new_layout, /*defer_flush=*/true);
}

[[nodiscard]] constexpr bool CoversEntireRenderArea(const VkRect2D& clear_rect,
                                                    const VkRect2D& render_area) noexcept {
    if (render_area.extent.width == 0 || render_area.extent.height == 0) {
        return false;
    }
    if (clear_rect.offset.x > render_area.offset.x ||
        clear_rect.offset.y > render_area.offset.y) {
        return false;
    }
    const s64 clear_right =
        static_cast<s64>(clear_rect.offset.x) + static_cast<s64>(clear_rect.extent.width);
    const s64 clear_bottom =
        static_cast<s64>(clear_rect.offset.y) + static_cast<s64>(clear_rect.extent.height);
    const s64 area_right =
        static_cast<s64>(render_area.offset.x) + static_cast<s64>(render_area.extent.width);
    const s64 area_bottom =
        static_cast<s64>(render_area.offset.y) + static_cast<s64>(render_area.extent.height);
    return clear_right >= area_right && clear_bottom >= area_bottom;
}

void UpdateAtomicMax(std::atomic<u32>& target, u32 candidate) noexcept {
    u32 prev = target.load(std::memory_order_relaxed);
    while (candidate > prev &&
           !target.compare_exchange_weak(prev, candidate, std::memory_order_relaxed,
                                         std::memory_order_relaxed)) {
    }
}

} // namespace

// ================================================================================================
// Support & Construction
// ================================================================================================

bool PassManager::IsSupported(const Device& device) noexcept {
    const vk::PhysicalDevice physical = device.GetPhysical();
    if (static_cast<VkPhysicalDevice>(physical) == VK_NULL_HANDLE) {
        return false;
    }

    const u32 api_version = device.ApiVersion();
    const bool is_vulkan_1_3 = api_version >= VK_API_VERSION_1_3;
    const auto& available_extensions = device.GetAvailableExtensions();

    const bool can_use_dynamic_rendering =
        is_vulkan_1_3 || available_extensions.contains(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    const bool can_use_sync2 =
        is_vulkan_1_3 || device.HasSynchronization2() ||
        available_extensions.contains(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    if (!can_use_dynamic_rendering || !can_use_sync2) {
        return false;
    }

    bool has_dynamic_rendering = false;
    bool has_synchronization2 = device.HasSynchronization2();

    const auto& dld = device.GetDispatchLoader();
    if (dld.vkGetPhysicalDeviceFeatures2 != nullptr) {
        VkPhysicalDeviceSynchronization2Features sync2_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
            .pNext = nullptr,
            .synchronization2 = VK_FALSE,
        };
        VkPhysicalDeviceDynamicRenderingFeatures dyn_features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
            .pNext = &sync2_features,
            .dynamicRendering = VK_FALSE,
        };
        VkPhysicalDeviceFeatures2 features2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &dyn_features,
            .features = {},
        };
        physical.GetFeatures2(features2);
        has_dynamic_rendering = (dyn_features.dynamicRendering == VK_TRUE);
        has_synchronization2 =
            has_synchronization2 || (sync2_features.synchronization2 == VK_TRUE);
    } else {
        has_dynamic_rendering = is_vulkan_1_3;
    }

    if (!has_dynamic_rendering || !has_synchronization2) {
        return false;
    }

    const VkDevice vk_device = *device.GetLogical();
    if (vk_device != VK_NULL_HANDLE && dld.vkGetDeviceProcAddr != nullptr) {
        if (ResolveCmdBeginRendering(vk_device, dld) == nullptr ||
            ResolveCmdEndRendering(vk_device, dld) == nullptr ||
            ResolveCmdPipelineBarrier2(vk_device, dld) == nullptr) {
            return false;
        }
    }

    return true;
}

PassManager::PassManager(const Device& device, BindlessDescriptorTable* bindless_table) noexcept
    : device_{device}, dld_{device.GetDispatchLoader()}, bindless_table_{bindless_table} {
    const VkDevice vk_device = *device_.GetLogical();
    pfn_cmd_begin_rendering_ = ResolveCmdBeginRendering(vk_device, dld_);
    pfn_cmd_end_rendering_ = ResolveCmdEndRendering(vk_device, dld_);
    pfn_cmd_pipeline_barrier2_ = ResolveCmdPipelineBarrier2(vk_device, dld_);

    try {
        image_states_.reserve(1024);
    } catch (...) {
    }
}

// ================================================================================================
// Frame & Image Registration Lifecycle
// ================================================================================================

void PassManager::OnFrameBegin() noexcept {
    const u32 frame_barriers = barriers_this_frame_;
    barriers_last_frame_.store(frame_barriers, std::memory_order_relaxed);

    const u32 prev_max = max_barriers_in_frame_.load(std::memory_order_relaxed);
    UpdateAtomicMax(max_barriers_in_frame_, frame_barriers);
    if (frame_barriers > kTargetMaxBarriersPerFrame && frame_barriers > prev_max) [[unlikely]] {
        LOG_DEBUG(Render_Vulkan,
                  "NvnClean::PassManager new barrier high-water mark: {} barriers/frame "
                  "(target <= {})",
                  frame_barriers, kTargetMaxBarriersPerFrame);
    }

    barriers_this_frame_ = 0;
}

void PassManager::OnCommandBufferBegin() noexcept {
    is_rendering_active_ = false;
    active_rendering_flags_ = 0;
}

void PassManager::RegisterImage(u64 texture_serial, VkImage image, VkImageAspectFlags aspect_mask,
                                u32 mip_levels, u32 array_layers, u32 bindless_texture_id,
                                VkImageLayout initial_layout) {
    if (texture_serial == 0 || image == VK_NULL_HANDLE) {
        return;
    }

    const StageAccessPair initial_sync = DefaultStageAccessForLayout(initial_layout);
    image_states_[texture_serial] = TrackedImageState{
        .image = image,
        .aspect_mask = aspect_mask != 0 ? aspect_mask : VK_IMAGE_ASPECT_COLOR_BIT,
        .current_layout = initial_layout,
        .last_stage_mask = initial_sync.stages,
        .last_access_mask = initial_sync.access,
        .bindless_texture_id = bindless_texture_id,
        .total_mip_levels = (std::max)(1u, mip_levels),
        .total_array_layers = (std::max)(1u, array_layers),
    };
}

void PassManager::UnregisterImage(u64 texture_serial) noexcept {
    const auto it = image_states_.find(texture_serial);
    if (it == image_states_.end()) {
        return;
    }

    const VkImage dead_image = it->second.image;
    if (dead_image != VK_NULL_HANDLE && pending_barriers_.image_barrier_count > 0) {
        u32 write_idx = 0;
        for (u32 read_idx = 0; read_idx < pending_barriers_.image_barrier_count; ++read_idx) {
            if (pending_barriers_.image_barriers[read_idx].image != dead_image) {
                if (write_idx != read_idx) {
                    pending_barriers_.image_barriers[write_idx] =
                        pending_barriers_.image_barriers[read_idx];
                }
                ++write_idx;
            }
        }
        pending_barriers_.image_barrier_count = write_idx;
    }

    image_states_.erase(it);
}

VkImageLayout PassManager::GetTrackedLayout(u64 texture_serial) const noexcept {
    const auto it = image_states_.find(texture_serial);
    if (it == image_states_.end()) {
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }
    return it->second.current_layout;
}

// ================================================================================================
// Dynamic Rendering Target & Pass Control
// ================================================================================================

void PassManager::SetRenderTargets(vk::CommandBuffer cmd,
                                   const RenderTargetConfig& targets) noexcept {
    RenderTargetConfig sanitized = targets;
    if (sanitized.color_count > kMaxColorAttachments) [[unlikely]] {
        sanitized.color_count = kMaxColorAttachments;
    }

    if (has_targets_ && sanitized.SameTargets(current_targets_)) [[likely]] {
        passes_elided_same_targets_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    if (is_rendering_active_) {
        EndRendering(cmd);
    } else if (has_targets_ && *cmd != VK_NULL_HANDLE) {
        // Flush any folded full-rect clears queued against the previous target set before
        // replacing `current_targets_`, so clear-only render passes are never dropped.
        bool has_unflushed_clear = pending_clears_.clear_depth || pending_clears_.clear_stencil;
        if (!has_unflushed_clear) {
            for (u32 i = 0; i < current_targets_.color_count; ++i) {
                if (pending_clears_.clear_color[i]) {
                    has_unflushed_clear = true;
                    break;
                }
            }
        }
        if (has_unflushed_clear) {
            if (EnsureRenderingActive(cmd, 0)) {
                EndRendering(cmd);
            }
        }
    }

    pending_clears_.Reset();
    current_targets_ = sanitized;
    current_signature_ = current_targets_.MakeSignature();
    has_targets_ = true;
    is_rendering_active_ = false;
    active_rendering_flags_ = 0;
}

bool PassManager::TryFoldClearColor(u32 color_index, const VkClearColorValue& clear_value,
                                    const VkRect2D& clear_rect) noexcept {
    if (is_rendering_active_ || !has_targets_) {
        return false;
    }
    if (color_index >= current_targets_.color_count || !current_targets_.colors[color_index]) {
        return false;
    }
    if (!CoversEntireRenderArea(clear_rect, current_targets_.render_area)) {
        return false;
    }

    pending_clears_.clear_color[color_index] = true;
    pending_clears_.color_values[color_index] = clear_value;
    fast_clears_folded_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool PassManager::TryFoldClearDepthStencil(VkImageAspectFlags aspects,
                                           VkClearDepthStencilValue clear_value,
                                           const VkRect2D& clear_rect) noexcept {
    if (is_rendering_active_ || !has_targets_ || !current_targets_.depth_stencil) {
        return false;
    }

    const VkImageAspectFlags effective_aspects =
        aspects & current_targets_.depth_stencil.aspect_mask &
        (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    if (effective_aspects == 0) {
        return false;
    }
    if (!CoversEntireRenderArea(clear_rect, current_targets_.render_area)) {
        return false;
    }

    if ((effective_aspects & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) {
        pending_clears_.clear_depth = true;
        pending_clears_.depth_stencil_value.depth = clear_value.depth;
    }
    if ((effective_aspects & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) {
        pending_clears_.clear_stencil = true;
        pending_clears_.depth_stencil_value.stencil = clear_value.stencil;
    }
    fast_clears_folded_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool PassManager::EnsureRenderingActive(vk::CommandBuffer cmd,
                                        VkRenderingFlags contents_flags) noexcept {
    if (is_rendering_active_) [[likely]] {
        if (active_rendering_flags_ == contents_flags) [[likely]] {
            return true;
        }
        EndRendering(cmd);
    }

    if (*cmd == VK_NULL_HANDLE || !has_targets_ || pfn_cmd_begin_rendering_ == nullptr)
        [[unlikely]] {
        return false;
    }
    if (current_targets_.render_area.extent.width == 0 ||
        current_targets_.render_area.extent.height == 0) [[unlikely]] {
        return false;
    }

    const auto prepare_attachment = [&](const AttachmentBinding& att,
                                        VkImageLayout optimal_layout,
                                        VkPipelineStageFlags2 att_stages,
                                        VkAccessFlags2 att_access,
                                        VkImageAspectFlags default_aspect) noexcept {
        if (!att || att.texture_serial == 0) {
            return;
        }

        auto it = image_states_.find(att.texture_serial);
        if (it == image_states_.end()) [[unlikely]] {
            try {
                it = image_states_
                         .emplace(att.texture_serial,
                                  TrackedImageState{
                                      .image = att.image,
                                      .aspect_mask =
                                          att.aspect_mask != 0 ? att.aspect_mask : default_aspect,
                                      .current_layout = VK_IMAGE_LAYOUT_UNDEFINED,
                                      .last_stage_mask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                      .last_access_mask = VK_ACCESS_2_NONE,
                                      .bindless_texture_id = att.bindless_texture_id,
                                      .total_mip_levels =
                                          (std::max)(1u, att.base_mip_level + att.level_count),
                                      .total_array_layers =
                                          (std::max)(1u, att.base_array_layer + att.layer_count),
                                  })
                         .first;
            } catch (...) {
                return;
            }
        }

        TrackedImageState& tracked = it->second;
        if (tracked.bindless_texture_id == ~0u && att.bindless_texture_id != ~0u) {
            tracked.bindless_texture_id = att.bindless_texture_id;
        }

        const VkImageLayout required_layout =
            att.feedback_loop ? VK_IMAGE_LAYOUT_GENERAL : optimal_layout;
        VkPipelineStageFlags2 dst_stages = att_stages;
        VkAccessFlags2 dst_access = att_access;
        if (att.feedback_loop) {
            dst_stages |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
            dst_access |= VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        }

        if (tracked.current_layout != required_layout) {
            QueueImageBarrierInternal(tracked, required_layout, dst_stages, dst_access,
                                      /*is_write_access=*/true);
            SyncBindlessSlotLayoutIfNeeded(bindless_table_, tracked.bindless_texture_id,
                                           required_layout);
        } else {
            tracked.last_stage_mask = dst_stages;
            tracked.last_access_mask = dst_access;
            elided_layout_transitions_.fetch_add(1, std::memory_order_relaxed);
        }
    };

    // 1. Check tracked layouts and queue transitions ONLY for mismatched attachments.
    for (u32 i = 0; i < current_targets_.color_count; ++i) {
        prepare_attachment(current_targets_.colors[i], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT);
    }

    if (current_targets_.depth_stencil) {
        prepare_attachment(current_targets_.depth_stencil,
                           VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                               VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    // 2. Flush Set 0 dirty descriptors if layout flips dirtied the bindless table.
    if (bindless_table_ != nullptr && bindless_table_->HasDirtyEpochBit()) {
        bindless_table_->FlushDirtyDescriptors();
    }

    // 3. Emit at most ONE coalesced barrier batch prior to opening Dynamic Rendering.
    FlushCoalescedBarriers(cmd);

    // 4. Assemble VkRenderingInfo and invoke vkCmdBeginRendering.
    std::array<VkRenderingAttachmentInfo, kMaxColorAttachments> color_attachments{};
    for (u32 i = 0; i < current_targets_.color_count; ++i) {
        const AttachmentBinding& color = current_targets_.colors[i];
        if (!color) {
            color_attachments[i] = VkRenderingAttachmentInfo{
                .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                .pNext = nullptr,
                .imageView = VK_NULL_HANDLE,
                .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .resolveMode = VK_RESOLVE_MODE_NONE,
                .resolveImageView = VK_NULL_HANDLE,
                .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
                .clearValue = {},
            };
            continue;
        }

        const bool fold_clear = pending_clears_.clear_color[i];
        VkClearValue clear_val{};
        if (fold_clear) {
            clear_val.color = pending_clears_.color_values[i];
        }

        color_attachments[i] = VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = color.image_view,
            .imageLayout = color.feedback_loop ? VK_IMAGE_LAYOUT_GENERAL
                                               : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = fold_clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clear_val,
        };
    }

    const AttachmentBinding& ds = current_targets_.depth_stencil;
    const bool has_depth = ds && ((ds.aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0);
    const bool has_stencil = ds && ((ds.aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0);
    const VkImageLayout ds_layout =
        ds.feedback_loop ? VK_IMAGE_LAYOUT_GENERAL
                         : VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkClearValue ds_clear_val{};
    ds_clear_val.depthStencil = pending_clears_.depth_stencil_value;

    VkRenderingAttachmentInfo depth_attachment{};
    if (has_depth) {
        depth_attachment = VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = ds.image_view,
            .imageLayout = ds_layout,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = pending_clears_.clear_depth ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                  : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = ds_clear_val,
        };
    }

    VkRenderingAttachmentInfo stencil_attachment{};
    if (has_stencil) {
        stencil_attachment = VkRenderingAttachmentInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .pNext = nullptr,
            .imageView = ds.image_view,
            .imageLayout = ds_layout,
            .resolveMode = VK_RESOLVE_MODE_NONE,
            .resolveImageView = VK_NULL_HANDLE,
            .resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .loadOp = pending_clears_.clear_stencil ? VK_ATTACHMENT_LOAD_OP_CLEAR
                                                    : VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = ds_clear_val,
        };
    }

    const VkRenderingInfo rendering_info{
        .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .pNext = nullptr,
        .flags = contents_flags,
        .renderArea = current_targets_.render_area,
        .layerCount = (std::max)(1u, current_targets_.layer_count),
        .viewMask = current_targets_.view_mask,
        .colorAttachmentCount = current_targets_.color_count,
        .pColorAttachments =
            current_targets_.color_count > 0 ? color_attachments.data() : nullptr,
        .pDepthAttachment = has_depth ? &depth_attachment : nullptr,
        .pStencilAttachment = has_stencil ? &stencil_attachment : nullptr,
    };

    pfn_cmd_begin_rendering_(*cmd, &rendering_info);
    pending_clears_.Reset();
    is_rendering_active_ = true;
    active_rendering_flags_ = contents_flags;
    passes_begun_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void PassManager::EndRendering(vk::CommandBuffer cmd) noexcept {
    if (!is_rendering_active_) {
        return;
    }
    if (*cmd != VK_NULL_HANDLE && pfn_cmd_end_rendering_ != nullptr) {
        pfn_cmd_end_rendering_(*cmd);
    }
    is_rendering_active_ = false;
    active_rendering_flags_ = 0;
}

// ================================================================================================
// Coarse Pass-Boundary Barrier Coalescing (< 50 Barriers / Frame)
// ================================================================================================

void PassManager::AccumulateNvnBarrier(u32 nvn_barrier_bits) noexcept {
    if (nvn_barrier_bits == 0) {
        return;
    }

    VkPipelineStageFlags2 src_stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 src_access = VK_ACCESS_2_NONE;
    VkPipelineStageFlags2 dst_stages = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 dst_access = VK_ACCESS_2_NONE;

    if ((nvn_barrier_bits & NvnBarrierBits::kOrderPrimitives) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                      VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        src_access |= VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT |
                      VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        dst_access |= VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
                      VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kOrderFragments) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        src_access |= VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        dst_access |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_INPUT_ATTACHMENT_READ_BIT |
                      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kOrderFragToTex) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        src_access |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        dst_access |= VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                      VK_ACCESS_2_TRANSFER_READ_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kOrderCompute) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        src_access |= VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_TRANSFER_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        dst_access |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT |
                      VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kOrderIndirectData) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        src_access |= VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_TRANSFER_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT;
        dst_access |= VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
                      VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kInvalidateTexture) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        src_access |= VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_TRANSFER_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        dst_access |= VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kInvalidateShader) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
        src_access |= VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                      VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        dst_access |= VK_ACCESS_2_UNIFORM_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT |
                      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kInvalidateTextureDescriptor) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        src_access |= VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        dst_stages |=
            VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        dst_access |= VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    }

    if ((nvn_barrier_bits & NvnBarrierBits::kInvalidateZcull) != 0) {
        src_stages |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        src_access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                      VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
        dst_stages |= VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
        dst_access |= VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                      VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    if ((nvn_barrier_bits & ~kAllKnownNvnBarrierBits) != 0 &&
        src_stages == VK_PIPELINE_STAGE_2_NONE) {
        src_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        src_access = VK_ACCESS_2_MEMORY_WRITE_BIT;
        dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        dst_access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    }

    AccumulateMemoryHazard(src_stages, src_access, dst_stages, dst_access);
}

void PassManager::AccumulateMemoryHazard(VkPipelineStageFlags2 src_stages,
                                         VkAccessFlags2 src_access,
                                         VkPipelineStageFlags2 dst_stages,
                                         VkAccessFlags2 dst_access) noexcept {
    if (src_stages == VK_PIPELINE_STAGE_2_NONE && dst_stages == VK_PIPELINE_STAGE_2_NONE &&
        src_access == VK_ACCESS_2_NONE && dst_access == VK_ACCESS_2_NONE) {
        return;
    }
    if (src_stages == VK_PIPELINE_STAGE_2_NONE && src_access != VK_ACCESS_2_NONE) {
        src_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }
    if (dst_stages == VK_PIPELINE_STAGE_2_NONE && dst_access != VK_ACCESS_2_NONE) {
        dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    if (pending_barriers_.has_global_memory_barrier) {
        coalesced_hazard_merges_.fetch_add(1, std::memory_order_relaxed);
    }
    pending_barriers_.MergeMemoryHazard(src_stages, src_access, dst_stages, dst_access);
}

void PassManager::RequireImageLayout(u64 texture_serial, VkImageLayout desired_layout,
                                     VkPipelineStageFlags2 dst_stages, VkAccessFlags2 dst_access,
                                     bool is_write_access) noexcept {
    if (texture_serial == 0 || desired_layout == VK_IMAGE_LAYOUT_UNDEFINED ||
        desired_layout == VK_IMAGE_LAYOUT_PREINITIALIZED) {
        return;
    }

    const auto it = image_states_.find(texture_serial);
    if (it == image_states_.end() || it->second.image == VK_NULL_HANDLE) {
        return;
    }

    TrackedImageState& tracked = it->second;
    if (dst_stages == VK_PIPELINE_STAGE_2_NONE && dst_access != VK_ACCESS_2_NONE) {
        dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    if (tracked.current_layout == desired_layout) {
        const bool had_prior_write = (tracked.last_access_mask & kWriteAccessMask) != 0;
        if (!is_write_access && !had_prior_write) [[likely]] {
            tracked.last_stage_mask |= dst_stages;
            tracked.last_access_mask |= dst_access;
            elided_layout_transitions_.fetch_add(1, std::memory_order_relaxed);
            return;
        }

        // Same layout with a RAW / WAW / WAR hazard: fold into existing pending barrier for this
        // image if one is queued in the current batch, otherwise merge into the global barrier.
        for (u32 i = 0; i < pending_barriers_.image_barrier_count; ++i) {
            if (pending_barriers_.image_barriers[i].image == tracked.image) {
                pending_barriers_.image_barriers[i].dstStageMask |= dst_stages;
                pending_barriers_.image_barriers[i].dstAccessMask |= dst_access;
                tracked.last_stage_mask = dst_stages;
                tracked.last_access_mask = dst_access;
                coalesced_hazard_merges_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }

        AccumulateMemoryHazard(tracked.last_stage_mask, tracked.last_access_mask, dst_stages,
                               dst_access);
        tracked.last_stage_mask = dst_stages;
        tracked.last_access_mask = dst_access;
        elided_layout_transitions_.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    QueueImageBarrierInternal(tracked, desired_layout, dst_stages, dst_access, is_write_access);
    SyncBindlessSlotLayoutIfNeeded(bindless_table_, tracked.bindless_texture_id, desired_layout);
}

void PassManager::TransitionForSampling(u64 texture_serial,
                                        VkPipelineStageFlags2 shader_stages) noexcept {
    bool is_feedback_attachment = false;
    if (has_targets_ && texture_serial != 0) {
        for (u32 i = 0; i < current_targets_.color_count; ++i) {
            if (current_targets_.colors[i].texture_serial == texture_serial &&
                current_targets_.colors[i].feedback_loop) {
                is_feedback_attachment = true;
                break;
            }
        }
        if (!is_feedback_attachment &&
            current_targets_.depth_stencil.texture_serial == texture_serial &&
            current_targets_.depth_stencil.feedback_loop) {
            is_feedback_attachment = true;
        }
    }

    const VkImageLayout target_layout =
        is_feedback_attachment ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkPipelineStageFlags2 dst_stages =
        shader_stages != VK_PIPELINE_STAGE_2_NONE
            ? shader_stages
            : (VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    RequireImageLayout(texture_serial, target_layout, dst_stages,
                       VK_ACCESS_2_SHADER_SAMPLED_READ_BIT, /*is_write_access=*/false);
}

void PassManager::TransitionForTransfer(u64 texture_serial, bool is_transfer_dst) noexcept {
    if (is_transfer_dst) {
        RequireImageLayout(texture_serial, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           /*is_write_access=*/true);
    } else {
        RequireImageLayout(texture_serial, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                           /*is_write_access=*/false);
    }
}

void PassManager::TransitionForStorage(u64 texture_serial, bool is_write) noexcept {
    constexpr VkPipelineStageFlags2 kStorageStages =
        VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    const VkAccessFlags2 dst_access =
        is_write ? (VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
                 : VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    RequireImageLayout(texture_serial, VK_IMAGE_LAYOUT_GENERAL, kStorageStages, dst_access,
                       /*is_write_access=*/is_write);
}

bool PassManager::FlushCoalescedBarriers(vk::CommandBuffer cmd) noexcept {
    if (pending_barriers_.Empty()) [[likely]] {
        return false;
    }
    if (*cmd == VK_NULL_HANDLE || pfn_cmd_pipeline_barrier2_ == nullptr) [[unlikely]] {
        return false;
    }

    // Synchronization2 layout transitions and cross-pass memory barriers must execute outside an
    // active Dynamic Rendering instance.
    if (is_rendering_active_) {
        EndRendering(cmd);
    }

    // Ensure Set 0 descriptor layout metadata matches any pending image layout transitions before
    // subsequent compute dispatches or graphics draws consume Set 0.
    if (pending_barriers_.image_barrier_count > 0 && bindless_table_ != nullptr &&
        bindless_table_->HasDirtyEpochBit()) {
        bindless_table_->FlushDirtyDescriptors();
    }

    const VkDependencyInfo dependency_info{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = pending_barriers_.has_global_memory_barrier ? 1u : 0u,
        .pMemoryBarriers = pending_barriers_.has_global_memory_barrier
                               ? &pending_barriers_.global_memory_barrier
                               : nullptr,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = pending_barriers_.image_barrier_count,
        .pImageMemoryBarriers = pending_barriers_.image_barrier_count > 0
                                    ? pending_barriers_.image_barriers.data()
                                    : nullptr,
    };

    pfn_cmd_pipeline_barrier2_(*cmd, &dependency_info);
    pending_barriers_.Clear();

    ++barriers_this_frame_;
    UpdateAtomicMax(max_barriers_in_frame_, barriers_this_frame_);
    barriers_emitted_total_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

PassManagerStats PassManager::Stats() const noexcept {
    return PassManagerStats{
        .passes_begun = passes_begun_.load(std::memory_order_relaxed),
        .passes_elided_same_targets = passes_elided_same_targets_.load(std::memory_order_relaxed),
        .fast_clears_folded = fast_clears_folded_.load(std::memory_order_relaxed),
        .barriers_emitted_total = barriers_emitted_total_.load(std::memory_order_relaxed),
        .barriers_last_frame = barriers_last_frame_.load(std::memory_order_relaxed),
        .max_barriers_in_frame = max_barriers_in_frame_.load(std::memory_order_relaxed),
        .image_layout_transitions = image_layout_transitions_.load(std::memory_order_relaxed),
        .elided_layout_transitions = elided_layout_transitions_.load(std::memory_order_relaxed),
        .coalesced_hazard_merges = coalesced_hazard_merges_.load(std::memory_order_relaxed),
    };
}

// ================================================================================================
// Internal Barrier Helpers
// ================================================================================================

void PassManager::QueueImageBarrierInternal(TrackedImageState& tracked, VkImageLayout new_layout,
                                            VkPipelineStageFlags2 dst_stages,
                                            VkAccessFlags2 dst_access,
                                            bool is_write_access) noexcept {
    (void)is_write_access;
    if (tracked.image == VK_NULL_HANDLE) [[unlikely]] {
        return;
    }

    if (dst_stages == VK_PIPELINE_STAGE_2_NONE && dst_access != VK_ACCESS_2_NONE) {
        dst_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    // Deduplicate against any already-queued transition for the same VkImage in the current batch
    // so a single VkDependencyInfo never contains two conflicting transitions on the same image.
    for (u32 i = 0; i < pending_barriers_.image_barrier_count; ++i) {
        VkImageMemoryBarrier2& existing = pending_barriers_.image_barriers[i];
        if (existing.image == tracked.image) {
            existing.newLayout = new_layout;
            existing.dstStageMask |= dst_stages;
            existing.dstAccessMask |= dst_access;
            tracked.current_layout = new_layout;
            tracked.last_stage_mask = dst_stages;
            tracked.last_access_mask = dst_access;
            coalesced_hazard_merges_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }

    VkPipelineStageFlags2 src_stages = tracked.last_stage_mask;
    VkAccessFlags2 src_access = tracked.last_access_mask;
    if (tracked.current_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
        src_stages = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        src_access = VK_ACCESS_2_NONE;
    } else if (src_stages == VK_PIPELINE_STAGE_2_NONE && src_access != VK_ACCESS_2_NONE) {
        src_stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    if (pending_barriers_.image_barrier_count >= kMaxCoalescedImageBarriers) [[unlikely]] {
        LOG_WARNING(Render_Vulkan,
                    "NvnClean::PassManager exceeded kMaxCoalescedImageBarriers ({}) before flush",
                    kMaxCoalescedImageBarriers);
        AccumulateMemoryHazard(src_stages, src_access, dst_stages, dst_access);
        return;
    }

    pending_barriers_.image_barriers[pending_barriers_.image_barrier_count++] =
        VkImageMemoryBarrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .pNext = nullptr,
            .srcStageMask = src_stages,
            .srcAccessMask = src_access,
            .dstStageMask = dst_stages,
            .dstAccessMask = dst_access,
            .oldLayout = tracked.current_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = tracked.image,
            .subresourceRange =
                VkImageSubresourceRange{
                    .aspectMask = tracked.aspect_mask != 0 ? tracked.aspect_mask
                                                           : VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = tracked.total_mip_levels != 0 ? tracked.total_mip_levels
                                                                : VK_REMAINING_MIP_LEVELS,
                    .baseArrayLayer = 0,
                    .layerCount = tracked.total_array_layers != 0 ? tracked.total_array_layers
                                                                  : VK_REMAINING_ARRAY_LAYERS,
                },
        };

    tracked.current_layout = new_layout;
    tracked.last_stage_mask = dst_stages;
    tracked.last_access_mask = dst_access;
    image_layout_transitions_.fetch_add(1, std::memory_order_relaxed);
}

} // namespace Vulkan::NvnClean
