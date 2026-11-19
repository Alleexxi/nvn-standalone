// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/nvn_types.h"
#include "video_core/nvn/record.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
} // namespace Vulkan

namespace Vulkan::NvnClean {

class BindlessDescriptorTable;

// ================================================================================================
// Architectural Overview & First-Principles Complexity (`nv_pass_manager.h`)
// ================================================================================================
//
// Legacy Vulkan render-pass management suffers from three systemic CPU & GPU bottlenecks:
//   1. Combinatorial `VkRenderPass` + `VkFramebuffer` compatibility hashing on every target bind.
//   2. Eager `finalLayout` transitions at pass end even when the next pass renders to the exact
//      same G-buffer / color / depth target again.
//   3. Fine-grained per-draw / per-buffer / per-texture `vkCmdPipelineBarrier` spam (hundreds to
//      thousands of barriers per frame, stalling GPU command processors and burning CPU time).
//
// `PassManager` replaces the entire `VkRenderPass` / `VkFramebuffer` layer with Vulkan 1.3
// Dynamic Rendering (`vkCmdBeginRendering` / `vkCmdEndRendering`) and Synchronization2
// (`vkCmdPipelineBarrier2`):
//
//   1. Zero `VkRenderPass` / `VkFramebuffer` Hashing (`O(1)` target setup):
//      - `SetRenderTargets` records a 256-byte POD `RenderTargetSet` and compares it against the
//        active pass via `O(1)` fixed-size equality. Redundant `SetRenderTargets` calls leave the
//        active rendering scope open with zero API calls.
//      - Lazy pass open (`EnsureRenderingActive`) defers `vkCmdBeginRendering` until the first
//        draw, folding initial `ClearColor` / `ClearDepthStencil` commands directly into
//        `VkRenderingAttachmentInfo::loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR` (vital for TBDR and
//        modern desktop DCC/Hi-Z fast-clear compression).
//
//   2. Lazy Target-Change Layout Transitions (`O(A)` for `A <= 9` attachments):
//      - Attachments remain in `COLOR_ATTACHMENT_OPTIMAL` / `DEPTH_STENCIL_ATTACHMENT_OPTIMAL`
//        across pass endings. An image transitions layout IF AND ONLY IF a target change binds a
//        formerly-sampled image as an attachment, or an explicit consumer binds a former attachment
//        for shader sampling / storage / transfer.
//
//   3. Coarse Pass-Boundary Barrier Coalescing (`< 50` barriers per frame):
//      - All NVN `Op::Barrier` flags, compute/indirect buffer hazards, and pending image layout
//        transitions accumulate into a single `CoalescedBarrierBatch` (1 merged global
//        `VkMemoryBarrier2` + deduplicated `VkImageMemoryBarrier2[<= 64]`).
//      - Because `nv_mempool.h` performs zero-copy ReBAR writes (0 UBO/VBO/IBO upload barriers)
//        and `FlushBarriers` emits at most ONE `vkCmdPipelineBarrier2` per coarse pass or compute
//        phase boundary, total barriers per frame satisfy:
//            B_frame <= K_render_passes + K_compute_phases
//        In Ocarina of Time Market (~22-30 render passes + ~8-12 compute/copy phases),
//        `B_frame` stays strictly within `25..42 < 50` barriers/frame.
// ================================================================================================

/// NVN architectural maximum color render targets bound simultaneously.
inline constexpr u32 kMaxColorAttachments = 8;

/// Maximum deduplicated `VkImageMemoryBarrier2` transitions accumulated in one coarse flush batch.
inline constexpr u32 kMaxCoalescedImageBarriers = 64;

/// Maximum frame barrier budget target used for diagnostics/telemetry warnings.
inline constexpr u32 kTargetMaxBarriersPerFrame = 50;

/// Raw NVN `NVNbarrierBits` masks as emitted in `Op::Barrier` (`record.h`).
namespace NvnBarrierBits {
inline constexpr u32 kOrderPrimitives = 1u << 0;
inline constexpr u32 kOrderFragments = 1u << 1;
inline constexpr u32 kOrderFragToTex = 1u << 2;
inline constexpr u32 kOrderCompute = 1u << 3;
inline constexpr u32 kOrderIndirectData = 1u << 4;
inline constexpr u32 kInvalidateTexture = 1u << 5;
inline constexpr u32 kInvalidateShader = 1u << 6;
inline constexpr u32 kInvalidateTextureDescriptor = 1u << 7;
inline constexpr u32 kInvalidateZcull = 1u << 8;
} // namespace NvnBarrierBits

/// Compact, trivially hashable/comparable format signature consumed by `VkPipelineRenderingCreateInfo`
/// (pipeline compilation) and `VkCommandBufferInheritanceRenderingInfo` (worker secondary command
/// buffers in `nv_command_recorder.h`).
struct PipelineRenderingSignature {
    std::array<VkFormat, kMaxColorAttachments> color_formats{};
    VkFormat depth_format{VK_FORMAT_UNDEFINED};
    VkFormat stencil_format{VK_FORMAT_UNDEFINED};
    VkSampleCountFlagBits rasterization_samples{VK_SAMPLE_COUNT_1_BIT};
    u32 color_attachment_count{0};
    u32 view_mask{0};

    [[nodiscard]] constexpr bool operator==(const PipelineRenderingSignature&) const noexcept =
        default;

    /// Populates a `VkPipelineRenderingCreateInfo` referencing `color_formats.data()` in place.
    /// Complexity: O(1). Notice `this` must outlive the returned struct's consumption.
    [[nodiscard]] constexpr VkPipelineRenderingCreateInfo ToPipelineRenderingCreateInfo(
        const void* p_next = nullptr) const noexcept {
        return VkPipelineRenderingCreateInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext = p_next,
            .viewMask = view_mask,
            .colorAttachmentCount = color_attachment_count,
            .pColorAttachmentFormats = color_formats.data(),
            .depthAttachmentFormat = depth_format,
            .stencilAttachmentFormat = stencil_format,
        };
    }

    /// Populates a `VkCommandBufferInheritanceRenderingInfo` for lock-free secondary command
    /// buffer recording (`VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT`).
    /// Complexity: O(1).
    [[nodiscard]] constexpr VkCommandBufferInheritanceRenderingInfo ToInheritanceRenderingInfo(
        VkRenderingFlags flags = VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT,
        const void* p_next = nullptr) const noexcept {
        return VkCommandBufferInheritanceRenderingInfo{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_RENDERING_INFO,
            .pNext = p_next,
            .flags = flags,
            .viewMask = view_mask,
            .colorAttachmentCount = color_attachment_count,
            .pColorAttachmentFormats = color_formats.data(),
            .depthAttachmentFormat = depth_format,
            .stencilAttachmentFormat = stencil_format,
            .rasterizationSamples = rasterization_samples,
        };
    }
};

/// Single resolved Vulkan attachment target bound to a color or depth/stencil slot.
struct AttachmentBinding {
    u64 texture_serial{0};
    VkImage image{VK_NULL_HANDLE};
    VkImageView image_view{VK_NULL_HANDLE};
    VkFormat format{VK_FORMAT_UNDEFINED};
    VkImageAspectFlags aspect_mask{0};
    VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
    u32 base_mip_level{0};
    u32 level_count{1};
    u32 base_array_layer{0};
    u32 layer_count{1};
    /// Optional Set 0 `texture_id` (`UINT32_MAX` if unregistered) so layout transitions can update
    /// `BindlessDescriptorTable` epoch dirty state in `O(1)` when needed.
    u32 bindless_texture_id{~0u};
    /// True when this attachment is simultaneously sampled inside the same pass (feedback loop ->
    /// `VK_IMAGE_LAYOUT_GENERAL`).
    bool feedback_loop{false};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return image != VK_NULL_HANDLE && image_view != VK_NULL_HANDLE;
    }

    [[nodiscard]] constexpr bool SameTarget(const AttachmentBinding& other) const noexcept {
        return image == other.image && image_view == other.image_view && format == other.format &&
               base_mip_level == other.base_mip_level && base_array_layer == other.base_array_layer &&
               layer_count == other.layer_count && feedback_loop == other.feedback_loop;
    }
};

/// Full render-target configuration supplied by `SetRenderTargets`.
struct RenderTargetConfig {
    std::array<AttachmentBinding, kMaxColorAttachments> colors{};
    u32 color_count{0};
    AttachmentBinding depth_stencil{};
    VkRect2D render_area{};
    u32 layer_count{1};
    u32 view_mask{0};

    [[nodiscard]] constexpr bool SameTargets(const RenderTargetConfig& other) const noexcept {
        if (color_count != other.color_count || layer_count != other.layer_count ||
            view_mask != other.view_mask ||
            render_area.offset.x != other.render_area.offset.x ||
            render_area.offset.y != other.render_area.offset.y ||
            render_area.extent.width != other.render_area.extent.width ||
            render_area.extent.height != other.render_area.extent.height ||
            !depth_stencil.SameTarget(other.depth_stencil)) {
            return false;
        }
        for (u32 i = 0; i < color_count; ++i) {
            if (!colors[i].SameTarget(other.colors[i])) {
                return false;
            }
        }
        return true;
    }

    /// Builds the `PipelineRenderingSignature` corresponding to these bound attachments.
    /// Complexity: O(color_count) <= O(8) = O(1).
    [[nodiscard]] constexpr PipelineRenderingSignature MakeSignature() const noexcept {
        PipelineRenderingSignature sig{};
        sig.color_attachment_count = color_count;
        sig.view_mask = view_mask;
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
        for (u32 i = 0; i < color_count; ++i) {
            sig.color_formats[i] = colors[i].format;
            if (colors[i]) {
                samples = colors[i].samples;
            }
        }
        if (depth_stencil) {
            samples = depth_stencil.samples;
            if ((depth_stencil.aspect_mask & VK_IMAGE_ASPECT_DEPTH_BIT) != 0) {
                sig.depth_format = depth_stencil.format;
            }
            if ((depth_stencil.aspect_mask & VK_IMAGE_ASPECT_STENCIL_BIT) != 0) {
                sig.stencil_format = depth_stencil.format;
            }
        }
        sig.rasterization_samples = samples;
        return sig;
    }
};

/// Persistent per-image layout & producer-stage tracking entry managed by `PassManager`.
struct TrackedImageState {
    VkImage image{VK_NULL_HANDLE};
    VkImageAspectFlags aspect_mask{VK_IMAGE_ASPECT_COLOR_BIT};
    VkImageLayout current_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    VkPipelineStageFlags2 last_stage_mask{VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT};
    VkAccessFlags2 last_access_mask{VK_ACCESS_2_NONE};
    u32 bindless_texture_id{~0u};
    u32 total_mip_levels{1};
    u32 total_array_layers{1};
};

/// Allocation-free coarse barrier accumulator (`VK_KHR_synchronization2`).
/// Merges arbitrary memory hazards into ONE `VkMemoryBarrier2` and up to 64 deduplicated
/// `VkImageMemoryBarrier2` entries, flushed together in a single `vkCmdPipelineBarrier2` call.
struct CoalescedBarrierBatch {
    VkMemoryBarrier2 global_memory_barrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
        .srcAccessMask = VK_ACCESS_2_NONE,
        .dstStageMask = VK_PIPELINE_STAGE_2_NONE,
        .dstAccessMask = VK_ACCESS_2_NONE,
    };
    bool has_global_memory_barrier{false};
    u32 image_barrier_count{0};
    std::array<VkImageMemoryBarrier2, kMaxCoalescedImageBarriers> image_barriers{};

    [[nodiscard]] constexpr bool Empty() const noexcept {
        return !has_global_memory_barrier && image_barrier_count == 0;
    }

    /// Merges a global / buffer execution + memory dependency in `O(1)` bitwise OR.
    constexpr void MergeMemoryHazard(VkPipelineStageFlags2 src_stages, VkAccessFlags2 src_access,
                                     VkPipelineStageFlags2 dst_stages,
                                     VkAccessFlags2 dst_access) noexcept {
        global_memory_barrier.srcStageMask |= src_stages;
        global_memory_barrier.srcAccessMask |= src_access;
        global_memory_barrier.dstStageMask |= dst_stages;
        global_memory_barrier.dstAccessMask |= dst_access;
        has_global_memory_barrier = true;
    }

    constexpr void Clear() noexcept {
        global_memory_barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        global_memory_barrier.srcAccessMask = VK_ACCESS_2_NONE;
        global_memory_barrier.dstStageMask = VK_PIPELINE_STAGE_2_NONE;
        global_memory_barrier.dstAccessMask = VK_ACCESS_2_NONE;
        has_global_memory_barrier = false;
        image_barrier_count = 0;
    }
};

/// Per-frame and cumulative telemetry counters for `PassManager`.
struct PassManagerStats {
    u64 passes_begun{};
    u64 passes_elided_same_targets{};
    u64 fast_clears_folded{};
    u64 barriers_emitted_total{};
    u32 barriers_last_frame{};
    u32 max_barriers_in_frame{};
    u64 image_layout_transitions{};
    u64 elided_layout_transitions{};
    u64 coalesced_hazard_merges{};
};

/// Vulkan 1.3 Dynamic Rendering & coarse pass-boundary barrier manager for `Vulkan::NvnClean`.
///
/// Owns no legacy `VkRenderPass` or `VkFramebuffer` objects. Driven on the primary command stream
/// (`NvnReplay` thread); exposes `CurrentSignature()` so worker threads (`nv_command_recorder.h`)
/// can record lock-free `RENDER_PASS_CONTINUE` secondary command buffers inside the active
/// `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT` scope.
class PassManager {
public:
    /// Checks whether the device exposes `vkCmdBeginRendering` / `vkCmdEndRendering` (Vulkan 1.3
    /// core or `VK_KHR_dynamic_rendering`) and `vkCmdPipelineBarrier2` (`VK_KHR_synchronization2`).
    [[nodiscard]] static bool IsSupported(const Device& device) noexcept;

    explicit PassManager(const Device& device,
                         BindlessDescriptorTable* bindless_table = nullptr) noexcept;
    ~PassManager() = default;

    PassManager(const PassManager&) = delete;
    PassManager& operator=(const PassManager&) = delete;

    PassManager(PassManager&&) = delete;
    PassManager& operator=(PassManager&&) = delete;

    // --------------------------------------------------------------------------------------------
    // Frame & Image Registration Lifecycle
    // --------------------------------------------------------------------------------------------

    /// Marks frame boundary, snapshots `barriers_this_frame_` into telemetry, and resets per-frame
    /// counters. Complexity: O(1).
    void OnFrameBegin() noexcept;

    /// Resets active pass state at the beginning of a new primary command buffer segment.
    void OnCommandBufferBegin() noexcept;

    /// Registers or resets initial tracking state for a texture serial (`VK_IMAGE_LAYOUT_UNDEFINED`).
    /// Complexity: O(1) amortized.
    void RegisterImage(u64 texture_serial, VkImage image, VkImageAspectFlags aspect_mask,
                       u32 mip_levels = 1, u32 array_layers = 1,
                       u32 bindless_texture_id = ~0u,
                       VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED);

    /// Removes tracking state for a destroyed texture serial. Complexity: O(1).
    void UnregisterImage(u64 texture_serial) noexcept;

    /// Queries the currently tracked layout of `texture_serial` (`VK_IMAGE_LAYOUT_UNDEFINED` if
    /// unknown). Complexity: O(1).
    [[nodiscard]] VkImageLayout GetTrackedLayout(u64 texture_serial) const noexcept;

    // --------------------------------------------------------------------------------------------
    // Dynamic Rendering Target & Pass Control
    // --------------------------------------------------------------------------------------------

    /// Updates the desired render targets (`Op::SetRenderTargets`).
    ///
    /// - If `targets.SameTargets(current_targets_)` is true, keeps the active pass open without
    ///   recording any Vulkan commands (`passes_elided_same_targets++`).
    /// - If targets differ, closes any currently open rendering scope (`EndRendering(cmd)`) and
    ///   marks the new target configuration pending for lazy `BeginRendering` on first draw/clear.
    ///
    /// Complexity: O(1) (fixed <= 9 attachment slot comparison).
    void SetRenderTargets(vk::CommandBuffer cmd, const RenderTargetConfig& targets) noexcept;

    /// Attempts to fold a full-target `Op::ClearColor` into the pending pass's `LOAD_OP_CLEAR`
    /// attachment info before `vkCmdBeginRendering` is emitted. Returns false if the pass has
    /// already begun drawing or the clear is scissored/partial (caller then emits `vkCmdClearAttachments`).
    /// Complexity: O(1).
    [[nodiscard]] bool TryFoldClearColor(u32 color_index, const VkClearColorValue& clear_value,
                                         const VkRect2D& clear_rect) noexcept;

    /// Attempts to fold a full-target `Op::ClearDepthStencil` into the pending pass's
    /// `LOAD_OP_CLEAR` before `vkCmdBeginRendering` is emitted.
    /// Complexity: O(1).
    [[nodiscard]] bool TryFoldClearDepthStencil(VkImageAspectFlags aspects,
                                                VkClearDepthStencilValue clear_value,
                                                const VkRect2D& clear_rect) noexcept;

    /// Ensures the pending Dynamic Rendering pass is open on `cmd`:
    ///   1. Transitions ONLY attachments whose tracked layout != required attachment layout.
    ///   2. Flushes `BindlessDescriptorTable::FlushDirtyDescriptors()` if layout flips dirtied Set 0.
    ///   3. Emits at most ONE coalesced `vkCmdPipelineBarrier2` for all pending hazards/transitions.
    ///   4. Emits `vkCmdBeginRendering(&rendering_info)` with zero `VkRenderPass` objects.
    ///
    /// @param contents_flags Pass `0` for inline primary draws, or
    ///        `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT` when executing worker
    ///        secondary command buffers via `vkCmdExecuteCommands`.
    /// Complexity: O(1) fast path when already active (`if (is_rendering_active_ && ...)` -> 1 branch);
    ///             O(color_count) <= O(9) when opening a new pass.
    bool EnsureRenderingActive(vk::CommandBuffer cmd, VkRenderingFlags contents_flags = 0) noexcept;

    /// Closes the active Dynamic Rendering scope (`vkCmdEndRendering`) if one is open, leaving
    /// bound attachments in their current `*_ATTACHMENT_OPTIMAL` (or `GENERAL`) layouts without
    /// speculative transitions.
    /// Complexity: O(1).
    void EndRendering(vk::CommandBuffer cmd) noexcept;

    /// Suspends an active pass (`VK_RENDERING_SUSPENDING_BIT` semantics or clean end) before a
    /// command buffer split or out-of-pass compute/transfer phase.
    void SuspendForOutOfPassWork(vk::CommandBuffer cmd) noexcept {
        EndRendering(cmd);
    }

    [[nodiscard]] bool IsRenderingActive() const noexcept {
        return is_rendering_active_;
    }

    [[nodiscard]] VkRenderingFlags ActiveRenderingFlags() const noexcept {
        return active_rendering_flags_;
    }

    [[nodiscard]] const RenderTargetConfig& CurrentTargets() const noexcept {
        return current_targets_;
    }

    [[nodiscard]] const PipelineRenderingSignature& CurrentSignature() const noexcept {
        return current_signature_;
    }

    // --------------------------------------------------------------------------------------------
    // Coarse Pass-Boundary Barrier Coalescing (< 50 Barriers / Frame)
    // --------------------------------------------------------------------------------------------

    /// Translates NVN `Op::Barrier` flags (`NVNbarrierBits`) into `VkPipelineStageFlags2` and
    /// `VkAccessFlags2` masks and merges them into the pending coarse barrier batch (`O(1)`).
    /// Does NOT immediately emit `vkCmdPipelineBarrier2`; flushed at the next pass/compute/copy
    /// boundary (or immediately if called mid-pass with self-dependency bits).
    void AccumulateNvnBarrier(u32 nvn_barrier_bits) noexcept;

    /// Queues a coarse global/buffer memory dependency into the pending batch (`O(1)` bitwise OR).
    void AccumulateMemoryHazard(VkPipelineStageFlags2 src_stages, VkAccessFlags2 src_access,
                                VkPipelineStageFlags2 dst_stages,
                                VkAccessFlags2 dst_access) noexcept;

    /// Lazily queues a layout transition for `texture_serial` to `desired_layout` if and only if
    /// its tracked `current_layout != desired_layout` (or if `force_write_hazard` is true).
    /// Also updates `BindlessDescriptorTable` layout metadata when transitioning to/from
    /// `SHADER_READ_ONLY_OPTIMAL` or `GENERAL`.
    /// Complexity: O(1) lookup + O(1) batch append/dedup.
    void RequireImageLayout(u64 texture_serial, VkImageLayout desired_layout,
                            VkPipelineStageFlags2 dst_stages, VkAccessFlags2 dst_access,
                            bool is_write_access = false) noexcept;

    /// Convenience helper preparing a texture for shader sampling (`SHADER_READ_ONLY_OPTIMAL`, or
    /// `GENERAL` if currently bound as a feedback attachment).
    void TransitionForSampling(u64 texture_serial,
                               VkPipelineStageFlags2 shader_stages =
                                   VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT |
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT) noexcept;

    /// Convenience helper preparing a texture for `Op::Copy*` / `Op::ClearTexture` transfer work.
    void TransitionForTransfer(u64 texture_serial, bool is_transfer_dst) noexcept;

    /// Convenience helper preparing a texture for compute/fragment storage image read-write (`GENERAL`).
    void TransitionForStorage(u64 texture_serial, bool is_write = true) noexcept;

    /// Emits a SINGLE `vkCmdPipelineBarrier2` (`VkDependencyInfo`) covering all accumulated memory
    /// hazards and deduplicated image layout transitions. No-op (`O(1)`) if the batch is empty.
    /// Must be called outside an active rendering pass (`EndRendering` is called automatically if
    /// image layout transitions are pending while a pass is open).
    ///
    /// Complexity: O(1) when empty; O(M) for `M <= 64` pending image transitions.
    /// @return True if a `vkCmdPipelineBarrier2` was recorded.
    bool FlushCoalescedBarriers(vk::CommandBuffer cmd) noexcept;

    [[nodiscard]] bool HasPendingBarriers() const noexcept {
        return !pending_barriers_.Empty();
    }

    [[nodiscard]] u32 BarriersThisFrame() const noexcept {
        return barriers_this_frame_;
    }

    [[nodiscard]] PassManagerStats Stats() const noexcept;

private:
    struct PendingClearState {
        std::array<bool, kMaxColorAttachments> clear_color{};
        std::array<VkClearColorValue, kMaxColorAttachments> color_values{};
        bool clear_depth{false};
        bool clear_stencil{false};
        VkClearDepthStencilValue depth_stencil_value{};

        constexpr void Reset() noexcept {
            *this = PendingClearState{};
        }
    };

    void QueueImageBarrierInternal(TrackedImageState& tracked, VkImageLayout new_layout,
                                   VkPipelineStageFlags2 dst_stages, VkAccessFlags2 dst_access,
                                   bool is_write_access) noexcept;

    const Device& device_;
    const vk::DeviceDispatch& dld_;
    BindlessDescriptorTable* bindless_table_{nullptr};

    // Direct cached Vulkan 1.3 / KHR function pointers (zero loader indirection).
    PFN_vkCmdBeginRendering pfn_cmd_begin_rendering_{nullptr};
    PFN_vkCmdEndRendering pfn_cmd_end_rendering_{nullptr};
    PFN_vkCmdPipelineBarrier2 pfn_cmd_pipeline_barrier2_{nullptr};

    bool has_targets_{false};
    bool is_rendering_active_{false};
    VkRenderingFlags active_rendering_flags_{0};
    RenderTargetConfig current_targets_{};
    PipelineRenderingSignature current_signature_{};
    PendingClearState pending_clears_{};
    CoalescedBarrierBatch pending_barriers_{};

    std::unordered_map<u64, TrackedImageState> image_states_;

    u32 barriers_this_frame_{0};
    std::atomic<u64> passes_begun_{0};
    std::atomic<u64> passes_elided_same_targets_{0};
    std::atomic<u64> fast_clears_folded_{0};
    std::atomic<u64> barriers_emitted_total_{0};
    std::atomic<u32> barriers_last_frame_{0};
    std::atomic<u32> max_barriers_in_frame_{0};
    std::atomic<u64> image_layout_transitions_{0};
    std::atomic<u64> elided_layout_transitions_{0};
    std::atomic<u64> coalesced_hazard_merges_{0};
};

} // namespace Vulkan::NvnClean
