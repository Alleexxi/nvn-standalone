// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "nvn_clean/nv_command_recorder.h"

#include <algorithm>
#include <limits>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <immintrin.h>
#endif

#include "common/logging.h"
#include "common/thread.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::NvnClean {
namespace {

/// Maximum slices dispatched in a single atomic 16-bit tagged cursor wave.
/// Larger spans are transparently chunked into consecutive waves.
constexpr u32 kMaxSlicesPerWave = 16384u;

/// Bit 15 of `active_slice_count_` signals that each slice writes to `out[slice.slice_index]`.
constexpr u32 kExplicitSliceIndexFlag = 0x8000u;

/// Mask extracting the 15-bit slice count / cursor index within a wave.
constexpr u32 kWaveIndexMask = 0x7FFFu;

/// Number of secondary command buffers allocated per pool growth step.
constexpr u32 kSecondaryAllocBatchSize = 8u;

/// Spin iterations before falling back to OS futex wait on straggler workers.
constexpr u32 kCompletionSpinIterations = 128u;

[[nodiscard]] constexpr VkIndexType ToVkIndexType(u32 nvn_index_type) noexcept {
    switch (static_cast<VideoCommon::Nvn::NVNindexType>(nvn_index_type)) {
    case VideoCommon::Nvn::NVNindexType::UNSIGNED_BYTE:
        return VK_INDEX_TYPE_UINT8_EXT;
    case VideoCommon::Nvn::NVNindexType::UNSIGNED_SHORT:
        return VK_INDEX_TYPE_UINT16;
    case VideoCommon::Nvn::NVNindexType::UNSIGNED_INT:
    default:
        return VK_INDEX_TYPE_UINT32;
    }
}

[[nodiscard]] PFN_vkResetCommandPool ResolveResetCommandPool(
    VkDevice vk_device, const vk::DeviceDispatch& dld) noexcept {
    if (vk_device == VK_NULL_HANDLE || dld.vkGetDeviceProcAddr == nullptr) [[unlikely]] {
        return nullptr;
    }
    static std::atomic<VkDevice> cached_device{VK_NULL_HANDLE};
    static std::atomic<PFN_vkResetCommandPool> cached_pfn{nullptr};
    if (cached_device.load(std::memory_order_acquire) == vk_device) [[likely]] {
        if (const auto fn = cached_pfn.load(std::memory_order_relaxed)) [[likely]] {
            return fn;
        }
    }
    const auto resolved = reinterpret_cast<PFN_vkResetCommandPool>(
        dld.vkGetDeviceProcAddr(vk_device, "vkResetCommandPool"));
    if (resolved != nullptr) {
        cached_pfn.store(resolved, std::memory_order_relaxed);
        cached_device.store(vk_device, std::memory_order_release);
    }
    return resolved;
}

[[nodiscard]] u32 SelectWorkerLaneCount(u32 configured_workers) noexcept {
    if (configured_workers == 0u) {
        const u32 hw_threads = std::thread::hardware_concurrency();
        const u32 half_cores = hw_threads > 0u ? (hw_threads / 2u) : 2u;
        return std::clamp(half_cores, 2u, 8u);
    }
    return std::clamp(configured_workers, 1u, kMaxRecorderWorkers);
}

[[nodiscard]] bool HasDistinctValidSliceIndices(std::span<const PassCommandSlice> slices,
                                                size_t out_capacity) noexcept {
    if (slices.empty()) {
        return false;
    }
    if (slices.size() == 1u) {
        return slices[0].slice_index < out_capacity;
    }
    // Detect uninitialized default `slice_index == 0` across multi-slice spans in O(1) first.
    if (slices[0].slice_index == slices[1].slice_index) {
        return false;
    }
    for (const PassCommandSlice& s : slices) {
        if (s.slice_index >= out_capacity) {
            return false;
        }
    }
    return true;
}

inline void CpuSpinPause() noexcept {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

} // namespace

// ============================================================================
// DirectCmdDispatch
// ============================================================================

DirectCmdDispatch DirectCmdDispatch::FromDevice(const Device& device) noexcept {
    const vk::DeviceDispatch& dld = device.GetDispatchLoader();
    const VkDevice vk_device = *device.GetLogical();

    const auto load_proc = [&](const char* name) noexcept -> PFN_vkVoidFunction {
        if (vk_device == VK_NULL_HANDLE || dld.vkGetDeviceProcAddr == nullptr) {
            return nullptr;
        }
        return dld.vkGetDeviceProcAddr(vk_device, name);
    };

    DirectCmdDispatch table{
        .vkCmdDraw = dld.vkCmdDraw,
        .vkCmdDrawIndexed = dld.vkCmdDrawIndexed,
        .vkCmdDrawIndirect = dld.vkCmdDrawIndirect,
        .vkCmdDrawIndexedIndirect = dld.vkCmdDrawIndexedIndirect,
        .vkCmdDrawIndirectCount = dld.vkCmdDrawIndirectCount,
        .vkCmdDrawIndexedIndirectCount = dld.vkCmdDrawIndexedIndirectCount,
        .vkCmdDispatch = dld.vkCmdDispatch,
        .vkCmdDispatchIndirect = dld.vkCmdDispatchIndirect,
        .vkCmdBindPipeline = dld.vkCmdBindPipeline,
        .vkCmdBindDescriptorSets = dld.vkCmdBindDescriptorSets,
        .vkCmdBindVertexBuffers = dld.vkCmdBindVertexBuffers,
        .vkCmdBindVertexBuffers2EXT = dld.vkCmdBindVertexBuffers2EXT,
        .vkCmdBindIndexBuffer = dld.vkCmdBindIndexBuffer,
        .vkCmdPushConstants = dld.vkCmdPushConstants,
        .vkCmdSetViewport = dld.vkCmdSetViewport,
        .vkCmdSetScissor = dld.vkCmdSetScissor,
        .vkCmdSetBlendConstants = dld.vkCmdSetBlendConstants,
        .vkCmdSetDepthBias = dld.vkCmdSetDepthBias,
        .vkCmdExecuteCommands =
            reinterpret_cast<PFN_vkCmdExecuteCommands>(load_proc("vkCmdExecuteCommands")),
    };

    if (table.vkCmdDrawIndirectCount == nullptr) {
        table.vkCmdDrawIndirectCount =
            reinterpret_cast<PFN_vkCmdDrawIndirectCount>(load_proc("vkCmdDrawIndirectCount"));
        if (table.vkCmdDrawIndirectCount == nullptr) {
            table.vkCmdDrawIndirectCount = reinterpret_cast<PFN_vkCmdDrawIndirectCount>(
                load_proc("vkCmdDrawIndirectCountKHR"));
        }
    }

    if (table.vkCmdDrawIndexedIndirectCount == nullptr) {
        table.vkCmdDrawIndexedIndirectCount = reinterpret_cast<PFN_vkCmdDrawIndexedIndirectCount>(
            load_proc("vkCmdDrawIndexedIndirectCount"));
        if (table.vkCmdDrawIndexedIndirectCount == nullptr) {
            table.vkCmdDrawIndexedIndirectCount =
                reinterpret_cast<PFN_vkCmdDrawIndexedIndirectCount>(
                    load_proc("vkCmdDrawIndexedIndirectCountKHR"));
        }
    }

    if (table.vkCmdBindVertexBuffers2EXT == nullptr) {
        table.vkCmdBindVertexBuffers2EXT = reinterpret_cast<PFN_vkCmdBindVertexBuffers2EXT>(
            load_proc("vkCmdBindVertexBuffers2"));
        if (table.vkCmdBindVertexBuffers2EXT == nullptr) {
            table.vkCmdBindVertexBuffers2EXT = reinterpret_cast<PFN_vkCmdBindVertexBuffers2EXT>(
                load_proc("vkCmdBindVertexBuffers2EXT"));
        }
    }

    return table;
}

// ============================================================================
// CommandRecorder
// ============================================================================

void CommandRecorder::FlushDirtyGraphicsState() noexcept {
    const u64 mask = dirty_mask_;
    if (mask == 0ULL || cmd_ == VK_NULL_HANDLE) [[unlikely]] {
        dirty_mask_ = 0ULL;
        return;
    }

    // 1. Graphics pipeline bind.
    if ((mask & RecorderDirtyBits::kGraphicsPipeline) != 0ULL) {
        if (graphics_pipeline_ != VK_NULL_HANDLE && dispatch_.vkCmdBindPipeline != nullptr) {
            dispatch_.vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline_);
            ++stats_.pipeline_binds;
        }
    }

    // 2. Persistent Set 0 bindless table (bound once per command buffer).
    if ((mask & RecorderDirtyBits::kBindlessSet0) != 0ULL) {
        VkDescriptorSet target_set0 = desired_bindless_set0_;
        if (target_set0 == VK_NULL_HANDLE && bindless_ != nullptr) {
            target_set0 = bindless_->PersistentSet();
            desired_bindless_set0_ = target_set0;
        }
        if (target_set0 != VK_NULL_HANDLE && graphics_layout_ != VK_NULL_HANDLE &&
            dispatch_.vkCmdBindDescriptorSets != nullptr) {
            dispatch_.vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                              graphics_layout_, kBindlessDescriptorSetIndex, 1u,
                                              &target_set0, 0u, nullptr);
            bound_bindless_set0_ = target_set0;
            ++stats_.descriptor_set_binds;
        }
    }

    // 3. Dynamic UBO descriptor sets (Sets [1 .. 1 + kMaxDynamicUniformSlots)) — coalesced runs.
    if ((mask & RecorderDirtyBits::kDynamicUbos) != 0ULL) {
        if (graphics_layout_ != VK_NULL_HANDLE && dispatch_.vkCmdBindDescriptorSets != nullptr) {
            u32 ubo_mask = static_cast<u32>(dirty_ubo_mask_ & valid_ubo_mask_);
            while (ubo_mask != 0u) {
                const u32 first_slot = static_cast<u32>(std::countr_zero(ubo_mask));
                const u32 run_len = static_cast<u32>(std::countr_one(ubo_mask >> first_slot));
                dispatch_.vkCmdBindDescriptorSets(
                    cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_layout_,
                    kFirstDynamicUboSetIndex + first_slot, run_len,
                    ubo_sets_.data() + first_slot, run_len,
                    ubo_dynamic_offsets_.data() + first_slot);
                ++stats_.descriptor_set_binds;
                const u32 run_bits = ((1u << run_len) - 1u) << first_slot;
                ubo_mask &= ~run_bits;
            }
            dirty_ubo_mask_ = 0;
        }
    }

    // 4. Vertex buffer bindings (`[0 .. 31]`) — coalesced contiguous runs via TZCNT.
    if ((mask & RecorderDirtyBits::kVertexBuffers) != 0ULL) {
        u32 vbo_mask = dirty_vertex_mask_ & valid_vertex_mask_;
        while (vbo_mask != 0u) {
            const u32 first_slot = static_cast<u32>(std::countr_zero(vbo_mask));
            const u32 run_len = static_cast<u32>(std::countr_one(vbo_mask >> first_slot));
            if (extended_vertex_bind_ && dispatch_.vkCmdBindVertexBuffers2EXT != nullptr) {
                dispatch_.vkCmdBindVertexBuffers2EXT(
                    cmd_, first_slot, run_len, vertex_buffers_.data() + first_slot,
                    vertex_offsets_.data() + first_slot, vertex_sizes_.data() + first_slot,
                    vertex_strides_.data() + first_slot);
                ++stats_.vertex_buffer_binds;
            } else if (dispatch_.vkCmdBindVertexBuffers != nullptr) {
                dispatch_.vkCmdBindVertexBuffers(
                    cmd_, first_slot, run_len, vertex_buffers_.data() + first_slot,
                    vertex_offsets_.data() + first_slot);
                ++stats_.vertex_buffer_binds;
            }
            const u32 run_bits =
                (run_len == 32u) ? ~0u : (((1u << run_len) - 1u) << first_slot);
            vbo_mask &= ~run_bits;
        }
        dirty_vertex_mask_ = 0;
    }

    // 5. Index buffer binding.
    if ((mask & RecorderDirtyBits::kIndexBuffer) != 0ULL) {
        if (index_valid_ && index_buffer_ != VK_NULL_HANDLE &&
            dispatch_.vkCmdBindIndexBuffer != nullptr) {
            dispatch_.vkCmdBindIndexBuffer(cmd_, index_buffer_, index_offset_, index_type_);
            ++stats_.index_buffer_binds;
        }
    }

    // 6. Push constants dirty byte window `[push_dirty_begin_, push_dirty_end_)`.
    if ((mask & RecorderDirtyBits::kPushConstants) != 0ULL) {
        if (graphics_layout_ != VK_NULL_HANDLE && push_constants_valid_ &&
            push_dirty_end_ > push_dirty_begin_ && dispatch_.vkCmdPushConstants != nullptr) {
            const u32 aligned_begin = push_dirty_begin_ & ~3u;
            const u32 aligned_end =
                (std::min)(kMaxPushConstantBytes, (push_dirty_end_ + 3u) & ~3u);
            const VkShaderStageFlags stages =
                push_stages_ != 0 ? push_stages_ : VK_SHADER_STAGE_ALL_GRAPHICS;
            dispatch_.vkCmdPushConstants(cmd_, graphics_layout_, stages, aligned_begin,
                                         aligned_end - aligned_begin,
                                         push_constants_.data() + aligned_begin);
            ++stats_.push_constant_updates;
            push_dirty_begin_ = kMaxPushConstantBytes;
            push_dirty_end_ = 0;
        }
    }

    // 7. Dynamic viewports — coalesced contiguous runs.
    if ((mask & RecorderDirtyBits::kViewports) != 0ULL) {
        u32 vp_mask = static_cast<u32>(dirty_viewport_mask_ & valid_viewport_mask_);
        while (vp_mask != 0u) {
            const u32 first_slot = static_cast<u32>(std::countr_zero(vp_mask));
            const u32 run_len = static_cast<u32>(std::countr_one(vp_mask >> first_slot));
            if (dispatch_.vkCmdSetViewport != nullptr) {
                dispatch_.vkCmdSetViewport(cmd_, first_slot, run_len,
                                           viewports_.data() + first_slot);
            }
            const u32 run_bits = ((1u << run_len) - 1u) << first_slot;
            vp_mask &= ~run_bits;
        }
        dirty_viewport_mask_ = 0;
    }

    // 8. Dynamic scissors — coalesced contiguous runs.
    if ((mask & RecorderDirtyBits::kScissors) != 0ULL) {
        u32 sc_mask = static_cast<u32>(dirty_scissor_mask_ & valid_scissor_mask_);
        while (sc_mask != 0u) {
            const u32 first_slot = static_cast<u32>(std::countr_zero(sc_mask));
            const u32 run_len = static_cast<u32>(std::countr_one(sc_mask >> first_slot));
            if (dispatch_.vkCmdSetScissor != nullptr) {
                dispatch_.vkCmdSetScissor(cmd_, first_slot, run_len,
                                          scissors_.data() + first_slot);
            }
            const u32 run_bits = ((1u << run_len) - 1u) << first_slot;
            sc_mask &= ~run_bits;
        }
        dirty_scissor_mask_ = 0;
    }

    dirty_mask_ = 0ULL;
}

void CommandRecorder::RecordPassSlice(const PassCommandSlice& slice) noexcept {
    if (cmd_ == VK_NULL_HANDLE) [[unlikely]] {
        return;
    }

    // 1. Seed persistent Set 0 and validate bindless epoch in O(1).
    VkDescriptorSet set0 = slice.bindless_set0;
    if (bindless_ != nullptr) {
        if (set0 == VK_NULL_HANDLE) {
            set0 = bindless_->PersistentSet();
        }
        if (slice.entry_state.bindless_set_epoch != 0ULL) {
            u64 cached_epoch = slice.entry_state.bindless_set_epoch;
            (void)bindless_->CheckAndAdvanceCleanEpoch(cached_epoch);
        }
    }
    if (set0 != VK_NULL_HANDLE) {
        SetBindlessDescriptorSet(set0);
    }

    // 2. Seed initial viewport & scissor state for secondary command buffer continuity.
    std::array<VkViewport, kMaxViewports> local_viewports{};
    for (VkViewport& vp : local_viewports) {
        vp.minDepth = 0.0f;
        vp.maxDepth = 1.0f;
    }

    const bool flip_viewport_y = (slice.entry_state.primary_viewport.height < 0.0f);
    if (slice.entry_state.primary_viewport.width != 0.0f ||
        slice.entry_state.primary_viewport.height != 0.0f) {
        local_viewports[0] = slice.entry_state.primary_viewport;
        SetViewport(0u, local_viewports[0]);
    } else if (slice.render_area.extent.width > 0u && slice.render_area.extent.height > 0u) {
        local_viewports[0] = VkViewport{
            .x = static_cast<float>(slice.render_area.offset.x),
            .y = static_cast<float>(slice.render_area.offset.y),
            .width = static_cast<float>(slice.render_area.extent.width),
            .height = static_cast<float>(slice.render_area.extent.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        SetViewport(0u, local_viewports[0]);
    }

    if (slice.entry_state.primary_scissor.extent.width > 0u ||
        slice.entry_state.primary_scissor.extent.height > 0u) {
        SetScissor(0u, slice.entry_state.primary_scissor);
    } else if (slice.render_area.extent.width > 0u && slice.render_area.extent.height > 0u) {
        SetScissor(0u, slice.render_area);
    }

    // 3. Seed live vertex buffer streams from `entry_state.active_vertex_stream_mask`.
    std::array<u32, kMaxVertexStreams> stream_strides = slice.entry_state.vertex_strides;
    if (mempool_ != nullptr) {
        u32 vmask = slice.entry_state.active_vertex_stream_mask;
        while (vmask != 0u) {
            const u32 slot = static_cast<u32>(std::countr_zero(vmask));
            vmask &= (vmask - 1u);
            const VideoCommon::Nvn::BufferRange& range = slice.entry_state.vertex_ranges[slot];
            if (range.address != 0ULL) {
                const BufferRangeView view = mempool_->ResolveBufferRange(range.address, 0ULL);
                if (view) [[likely]] {
                    SetVertexBuffer(slot, view.buffer, view.offset);
                }
            }
        }

        // 4. Seed live graphics dynamic UBOs from `entry_state.active_ubo_slot_mask`.
        u32 umask = static_cast<u32>(slice.entry_state.active_ubo_slot_mask);
        while (umask != 0u) {
            const u32 slot = static_cast<u32>(std::countr_zero(umask));
            umask &= (umask - 1u);
            if (slot < kMaxDynamicUniformSlots) {
                const VideoCommon::Nvn::BufferRange& ubo = slice.entry_state.graphics_ubos[slot];
                SetDynamicUniformAddress(slot, ubo.address, ubo.size);
            }
        }

        // 5. Seed live index buffer if active at slice start.
        if (slice.entry_state.has_index_buffer && slice.entry_state.index_address != 0ULL) {
            SetIndexBufferAddress(slice.entry_state.index_address,
                                  ToVkIndexType(slice.entry_state.index_type));
        }
    }

    if (slice.words == nullptr || slice.word_count == 0u) {
        return;
    }

    // Stack-local shadow state for infrequent fixed-function dynamic ops.
    bool blend_valid = false;
    std::array<float, 4> shadow_blend{};
    bool depth_bias_valid = false;
    std::array<float, 3> shadow_depth_bias{};

    // 6. Decode and emit NVN records directly from the immutable u64 stream (zero allocations).
    namespace VC = VideoCommon::Nvn;
    VC::RecordReader reader{slice.words, slice.word_count};

    while (const VC::RecordHeader* const header = reader.Next()) {
        const u32 payload_bytes = VC::RecordPayloadSize(header);
        switch (VC::RecordOp(header)) {
        case VC::Op::SetViewports: {
            if (payload_bytes < sizeof(VC::SetViewportsPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::SetViewportsPayload>(header);
            const u32 avail = static_cast<u32>(
                (payload_bytes - sizeof(VC::SetViewportsPayload)) / sizeof(VC::ViewportRect));
            const u32 count = (std::min)(p->count, avail);
            const auto* const rects = VC::TrailingArray<VC::ViewportRect>(p);
            for (u32 i = 0u; i < count && p->first < kMaxViewports && i < kMaxViewports - p->first;
                 ++i) {
                const u32 slot = p->first + i;
                const VC::ViewportRect& r = rects[i];
                VkViewport& vp = local_viewports[slot];
                vp.x = r.x;
                if (flip_viewport_y && r.height > 0.0f) {
                    vp.y = r.y + r.height;
                    vp.height = -r.height;
                } else {
                    vp.y = r.y;
                    vp.height = r.height;
                }
                vp.width = r.width;
                SetViewport(slot, vp);
            }
            break;
        }
        case VC::Op::SetDepthRanges: {
            if (payload_bytes < sizeof(VC::SetDepthRangesPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::SetDepthRangesPayload>(header);
            const u32 avail = static_cast<u32>(
                (payload_bytes - sizeof(VC::SetDepthRangesPayload)) / sizeof(VC::DepthRange));
            const u32 count = (std::min)(p->count, avail);
            const auto* const ranges = VC::TrailingArray<VC::DepthRange>(p);
            for (u32 i = 0u; i < count && p->first < kMaxViewports && i < kMaxViewports - p->first;
                 ++i) {
                const u32 slot = p->first + i;
                VkViewport& vp = local_viewports[slot];
                vp.minDepth = ranges[i].near_plane;
                vp.maxDepth = ranges[i].far_plane;
                if ((valid_viewport_mask_ & static_cast<u16>(1u << slot)) != 0u) {
                    SetViewport(slot, vp);
                }
            }
            break;
        }
        case VC::Op::SetScissors: {
            if (payload_bytes < sizeof(VC::SetScissorsPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::SetScissorsPayload>(header);
            const u32 avail = static_cast<u32>(
                (payload_bytes - sizeof(VC::SetScissorsPayload)) / sizeof(VC::ScissorRect));
            const u32 count = (std::min)(p->count, avail);
            const auto* const rects = VC::TrailingArray<VC::ScissorRect>(p);
            for (u32 i = 0u; i < count && p->first < kMaxViewports && i < kMaxViewports - p->first;
                 ++i) {
                const u32 slot = p->first + i;
                const VC::ScissorRect& s = rects[i];
                const s32 clamped_x = (std::max)(s.x, 0);
                const s32 clamped_y = (std::max)(s.y, 0);
                const s32 adj_w = s.width - (clamped_x - s.x);
                const s32 adj_h = s.height - (clamped_y - s.y);
                const VkRect2D vk_rect{
                    .offset = {clamped_x, clamped_y},
                    .extent =
                        {
                            static_cast<u32>((std::max)(adj_w, 0)),
                            static_cast<u32>((std::max)(adj_h, 0)),
                        },
                };
                SetScissor(slot, vk_rect);
            }
            break;
        }
        case VC::Op::BindVertexStreams: {
            if (payload_bytes < sizeof(VC::BindVertexStreamsPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::BindVertexStreamsPayload>(header);
            const u32 avail =
                static_cast<u32>((payload_bytes - sizeof(VC::BindVertexStreamsPayload)) /
                                 sizeof(VC::VertexStreamState));
            const u32 count = (std::min)(p->count, avail);
            const auto* const streams = VC::TrailingArray<VC::VertexStreamState>(p);
            for (u32 i = 0u;
                 i < count && p->first < kMaxVertexStreams && i < kMaxVertexStreams - p->first;
                 ++i) {
                stream_strides[p->first + i] = streams[i].stride;
            }
            break;
        }
        case VC::Op::BindVertexBuffers: {
            if (payload_bytes < sizeof(VC::BindVertexBuffersPayload) || mempool_ == nullptr)
                [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::BindVertexBuffersPayload>(header);
            const u32 avail = static_cast<u32>(
                (payload_bytes - sizeof(VC::BindVertexBuffersPayload)) / sizeof(VC::BufferRange));
            const u32 count = (std::min)(p->count, avail);
            const auto* const ranges = VC::TrailingArray<VC::BufferRange>(p);
            for (u32 i = 0u;
                 i < count && p->first < kMaxVertexStreams && i < kMaxVertexStreams - p->first;
                 ++i) {
                const u32 slot = p->first + i;
                if (ranges[i].address == 0ULL) {
                    continue;
                }
                const BufferRangeView view = mempool_->ResolveBufferRange(ranges[i].address, 0ULL);
                if (!view) [[unlikely]] {
                    continue;
                }
                if (extended_vertex_bind_) {
                    const VkDeviceSize clamped_size =
                        ranges[i].size != 0ULL
                            ? (std::min<VkDeviceSize>)(ranges[i].size, view.available_size)
                            : view.available_size;
                    SetVertexBufferExtended(slot, view.buffer, view.offset, clamped_size,
                                            stream_strides[slot]);
                } else {
                    SetVertexBuffer(slot, view.buffer, view.offset);
                }
            }
            break;
        }
        case VC::Op::BindUniformBuffers: {
            if (payload_bytes < sizeof(VC::BindBuffersPayload) || mempool_ == nullptr)
                [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::BindBuffersPayload>(header);
            if (p->stage == static_cast<u32>(VC::NVNshaderStage::COMPUTE)) {
                break;
            }
            if ((header->flags & VC::RECORD_FLAG_DEFERRED) != 0u) {
                if (p->src == 0ULL || p->count == 0u) {
                    break;
                }
                const u64 byte_len = static_cast<u64>(p->count) * sizeof(VC::BufferRange);
                const BufferRangeView deferred_view = mempool_->ResolveBufferRange(p->src, byte_len);
                if (!deferred_view || deferred_view.host_ptr == nullptr) {
                    break;
                }
                for (u32 i = 0u; i < p->count && p->first < kMaxDynamicUniformSlots &&
                                 i < kMaxDynamicUniformSlots - p->first;
                     ++i) {
                    VC::BufferRange range{};
                    std::memcpy(&range,
                                deferred_view.host_ptr + static_cast<size_t>(i) * sizeof(range),
                                sizeof(range));
                    SetDynamicUniformAddress(p->first + i, range.address, range.size);
                }
            } else {
                const u32 avail = static_cast<u32>(
                    (payload_bytes - sizeof(VC::BindBuffersPayload)) / sizeof(VC::BufferRange));
                const u32 count = (std::min)(p->count, avail);
                const auto* const ranges = VC::TrailingArray<VC::BufferRange>(p);
                for (u32 i = 0u; i < count && p->first < kMaxDynamicUniformSlots &&
                                 i < kMaxDynamicUniformSlots - p->first;
                     ++i) {
                    SetDynamicUniformAddress(p->first + i, ranges[i].address, ranges[i].size);
                }
            }
            break;
        }
        case VC::Op::PushConstants: {
            if (payload_bytes < sizeof(VC::PushConstantsPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::PushConstantsPayload>(header);
            if (p->compute != 0u || p->offset >= kMaxPushConstantBytes) {
                break;
            }
            const u32 avail = payload_bytes - static_cast<u32>(sizeof(VC::PushConstantsPayload));
            const u32 clamped_size =
                (std::min)({p->size, avail, kMaxPushConstantBytes - p->offset});
            if (clamped_size > 0u) {
                SetPushConstants(VK_SHADER_STAGE_ALL_GRAPHICS, p->offset, clamped_size,
                                 VC::TrailingArray<u8>(p));
            }
            break;
        }
        case VC::Op::SetBlendColor: {
            if (payload_bytes < sizeof(VC::SetBlendColorPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::SetBlendColorPayload>(header);
            if (blend_valid &&
                std::memcmp(shadow_blend.data(), p->color, sizeof(shadow_blend)) == 0) [[likely]] {
                break;
            }
            std::memcpy(shadow_blend.data(), p->color, sizeof(shadow_blend));
            blend_valid = true;
            if (dispatch_.vkCmdSetBlendConstants != nullptr) {
                dispatch_.vkCmdSetBlendConstants(cmd_, p->color);
            }
            break;
        }
        case VC::Op::SetPolygonOffset: {
            if (payload_bytes < sizeof(VC::SetPolygonOffsetPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::SetPolygonOffsetPayload>(header);
            const std::array<float, 3> next_bias{p->units, p->clamp, p->factor};
            if (depth_bias_valid &&
                std::memcmp(shadow_depth_bias.data(), next_bias.data(),
                            sizeof(shadow_depth_bias)) == 0) [[likely]] {
                break;
            }
            shadow_depth_bias = next_bias;
            depth_bias_valid = true;
            if (dispatch_.vkCmdSetDepthBias != nullptr) {
                dispatch_.vkCmdSetDepthBias(cmd_, p->units, p->clamp, p->factor);
            }
            break;
        }
        case VC::Op::Draw: {
            if (payload_bytes < sizeof(VC::DrawPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::DrawPayload>(header);
            const u32 instances = p->instance_count != 0u ? p->instance_count : 1u;
            if (p->indirect == 0ULL && p->count_address == 0ULL) [[likely]] {
                if (p->count > 0u) [[likely]] {
                    Draw(p->count, instances, static_cast<u32>(p->first), p->base_instance);
                }
            } else if (mempool_ != nullptr && p->indirect != 0ULL) {
                const BufferRangeView args = mempool_->ResolveBufferRange(p->indirect, 0ULL);
                if (!args) [[unlikely]] {
                    break;
                }
                const u32 max_draws = p->max_draws != 0u ? p->max_draws : 1u;
                const u32 stride = p->stride != 0u
                                       ? p->stride
                                       : static_cast<u32>(sizeof(VkDrawIndirectCommand));
                if (p->count_address != 0ULL) {
                    const BufferRangeView cnt =
                        mempool_->ResolveBufferRange(p->count_address, sizeof(u32));
                    if (cnt && dispatch_.vkCmdDrawIndirectCount != nullptr) {
                        if (dirty_mask_ != 0ULL) [[unlikely]] {
                            FlushDirtyGraphicsState();
                        }
                        dispatch_.vkCmdDrawIndirectCount(cmd_, args.buffer, args.offset, cnt.buffer,
                                                         cnt.offset, max_draws, stride);
                        ++stats_.draws_recorded;
                    }
                } else {
                    DrawIndirect(args.buffer, args.offset, max_draws, stride);
                }
            }
            break;
        }
        case VC::Op::DrawIndexed: {
            if (payload_bytes < sizeof(VC::DrawIndexedPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::DrawIndexedPayload>(header);
            const VkIndexType vk_idx_type = ToVkIndexType(p->index_type);
            if (p->index_address != 0ULL) {
                SetIndexBufferAddress(p->index_address, vk_idx_type);
            }
            const u32 instances = p->instance_count != 0u ? p->instance_count : 1u;
            if (p->indirect == 0ULL && p->count_address == 0ULL) [[likely]] {
                if (p->count > 0u) [[likely]] {
                    DrawIndexed(p->count, instances, 0u, p->base_vertex, p->base_instance);
                }
            } else if (mempool_ != nullptr && p->indirect != 0ULL) {
                const BufferRangeView args = mempool_->ResolveBufferRange(p->indirect, 0ULL);
                if (!args) [[unlikely]] {
                    break;
                }
                const u32 max_draws = p->max_draws != 0u ? p->max_draws : 1u;
                const u32 stride = p->stride != 0u
                                       ? p->stride
                                       : static_cast<u32>(sizeof(VkDrawIndexedIndirectCommand));
                if (p->count_address != 0ULL) {
                    const BufferRangeView cnt =
                        mempool_->ResolveBufferRange(p->count_address, sizeof(u32));
                    if (cnt && dispatch_.vkCmdDrawIndexedIndirectCount != nullptr) {
                        if (dirty_mask_ != 0ULL) [[unlikely]] {
                            FlushDirtyGraphicsState();
                        }
                        dispatch_.vkCmdDrawIndexedIndirectCount(
                            cmd_, args.buffer, args.offset, cnt.buffer, cnt.offset, max_draws,
                            stride);
                        ++stats_.draws_recorded;
                    }
                } else {
                    DrawIndexedIndirect(args.buffer, args.offset, max_draws, stride);
                }
            }
            break;
        }
        case VC::Op::Dispatch: {
            if (payload_bytes < sizeof(VC::DispatchPayload)) [[unlikely]] {
                break;
            }
            const auto* const p = VC::RecordPayload<VC::DispatchPayload>(header);
            if ((dirty_mask_ & RecorderDirtyBits::kComputePipeline) != 0ULL) {
                if (compute_pipeline_ != VK_NULL_HANDLE && dispatch_.vkCmdBindPipeline != nullptr) {
                    dispatch_.vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                                compute_pipeline_);
                    ++stats_.pipeline_binds;
                }
                dirty_mask_ &= ~RecorderDirtyBits::kComputePipeline;
            }
            if (p->indirect == 0ULL) {
                if (p->x > 0u && p->y > 0u && p->z > 0u && dispatch_.vkCmdDispatch != nullptr) {
                    dispatch_.vkCmdDispatch(cmd_, p->x, p->y, p->z);
                    ++stats_.dispatches_recorded;
                }
            } else if (mempool_ != nullptr && dispatch_.vkCmdDispatchIndirect != nullptr) {
                const BufferRangeView args =
                    mempool_->ResolveBufferRange(p->indirect, 3u * sizeof(u32));
                if (args) {
                    dispatch_.vkCmdDispatchIndirect(cmd_, args.buffer, args.offset);
                    ++stats_.dispatches_recorded;
                }
            }
            break;
        }
        default:
            break;
        }
    }
}

// ============================================================================
// ParallelCommandPool
// ============================================================================

ParallelCommandPool::ParallelCommandPool(const Device& device, const MemoryPoolManager& mempool,
                                         const BindlessDescriptorTable& bindless,
                                         const ParallelRecorderConfig& config)
    : device_{device},
      dld_{device.GetDispatchLoader()},
      mempool_{mempool},
      bindless_{bindless},
      direct_dispatch_{DirectCmdDispatch::FromDevice(device)},
      config_{config},
      worker_count_{SelectWorkerLaneCount(config.worker_count)},
      active_epoch_index_{0u} {
    config_.worker_count = worker_count_;
    config_.ring_epochs = (std::max)(1u, config_.ring_epochs);

    const VkDevice vk_device = *device_.GetLogical();
    const bool can_create_pools =
        (vk_device != VK_NULL_HANDLE && dld_.vkCreateCommandPool != nullptr);

    lanes_.reserve(worker_count_);
    for (u32 lane_idx = 0u; lane_idx < worker_count_; ++lane_idx) {
        auto lane = std::make_unique<WorkerLane>();
        lane->lane_id = lane_idx;
        lane->recorder = CommandRecorder{direct_dispatch_, &mempool_, &bindless_};
        lane->epochs.resize(config_.ring_epochs);

        if (can_create_pools) {
            const VkCommandPoolCreateInfo pool_ci{
                .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .pNext = nullptr,
                .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
                .queueFamilyIndex = device_.GetGraphicsFamily(),
            };
            for (EpochPoolSlot& slot : lane->epochs) {
                slot.pool = device_.GetLogical().CreateCommandPool(pool_ci);
            }
        }
        lanes_.push_back(std::move(lane));
    }

    if (worker_count_ > 1u) {
        worker_threads_.reserve(worker_count_ - 1u);
        try {
            for (u32 lane_idx = 1u; lane_idx < worker_count_; ++lane_idx) {
                worker_threads_.emplace_back(&ParallelCommandPool::WorkerLoop, this, lane_idx);
            }
        } catch (...) {
            stop_workers_.store(true, std::memory_order_release);
            work_generation_.fetch_add(1ULL, std::memory_order_acq_rel);
            work_generation_.notify_all();
            for (std::thread& worker : worker_threads_) {
                if (worker.joinable()) {
                    worker.join();
                }
            }
            worker_threads_.clear();
            throw;
        }
    }
}

ParallelCommandPool::~ParallelCommandPool() {
    stop_workers_.store(true, std::memory_order_release);
    work_generation_.fetch_add(1ULL, std::memory_order_acq_rel);
    work_generation_.notify_all();

    for (std::thread& worker : worker_threads_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    worker_threads_.clear();
    lanes_.clear();
}

bool ParallelCommandPool::BeginSubmissionEpoch(u64 retire_timeline_value,
                                               u64 completed_timeline_value) {
    if (lanes_.empty() || lanes_[0]->epochs.empty()) [[unlikely]] {
        return false;
    }

    const u32 ring_count = static_cast<u32>(lanes_[0]->epochs.size());
    const EpochPoolSlot& cur_lead = lanes_[0]->epochs[active_epoch_index_];
    const bool cur_fresh =
        (!cur_lead.needs_reset && cur_lead.used_count == 0u && cur_lead.retire_timeline_value == 0ULL);
    const u32 start_idx = cur_fresh ? active_epoch_index_ : ((active_epoch_index_ + 1u) % ring_count);

    u32 selected_epoch = ~0u;
    for (u32 step = 0u; step < ring_count; ++step) {
        const u32 candidate = (start_idx + step) % ring_count;
        bool retired_across_all_lanes = true;
        for (u32 lane_idx = 0u; lane_idx < worker_count_; ++lane_idx) {
            if (lanes_[lane_idx]->epochs[candidate].retire_timeline_value >
                completed_timeline_value) {
                retired_across_all_lanes = false;
                break;
            }
        }
        if (retired_across_all_lanes) {
            selected_epoch = candidate;
            break;
        }
    }

    if (selected_epoch == ~0u) [[unlikely]] {
        return false;
    }

    const VkDevice vk_device = *device_.GetLogical();
    const PFN_vkResetCommandPool pfn_reset_pool = ResolveResetCommandPool(vk_device, dld_);

    for (u32 lane_idx = 0u; lane_idx < worker_count_; ++lane_idx) {
        EpochPoolSlot& slot = lanes_[lane_idx]->epochs[selected_epoch];
        if ((slot.needs_reset || slot.used_count > 0u) && slot.pool && pfn_reset_pool != nullptr) {
            const VkResult res = pfn_reset_pool(vk_device, *slot.pool, 0u);
            if (res != VK_SUCCESS) [[unlikely]] {
                LOG_ERROR(Render_Vulkan,
                          "ParallelCommandPool: vkResetCommandPool failed on lane {} epoch {} ({})",
                          lane_idx, selected_epoch, static_cast<s32>(res));
                return false;
            }
        }
        slot.used_count = 0u;
        slot.needs_reset = false;
        slot.retire_timeline_value = retire_timeline_value;
    }

    active_epoch_index_ = selected_epoch;
    return true;
}

VkCommandBuffer ParallelCommandPool::BeginWorkerSecondary(
    u32 worker_lane, const PipelineRenderingSignature& rendering_sig) {
    if (worker_lane >= lanes_.size()) [[unlikely]] {
        return VK_NULL_HANDLE;
    }

    const VkDevice vk_device = *device_.GetLogical();
    if (vk_device == VK_NULL_HANDLE || dld_.vkAllocateCommandBuffers == nullptr ||
        dld_.vkBeginCommandBuffer == nullptr) [[unlikely]] {
        return VK_NULL_HANDLE;
    }

    WorkerLane& lane = *lanes_[worker_lane];
    EpochPoolSlot& slot = lane.epochs[active_epoch_index_];

    if (!slot.pool) [[unlikely]] {
        if (dld_.vkCreateCommandPool == nullptr) {
            return VK_NULL_HANDLE;
        }
        const VkCommandPoolCreateInfo pool_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            .queueFamilyIndex = device_.GetGraphicsFamily(),
        };
        slot.pool = device_.GetLogical().CreateCommandPool(pool_ci);
    }

    if (slot.used_count >= slot.allocated_buffers.size()) {
        const size_t prev_count = slot.allocated_buffers.size();
        slot.allocated_buffers.resize(prev_count + kSecondaryAllocBatchSize, VK_NULL_HANDLE);
        const VkCommandBufferAllocateInfo alloc_ci{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = nullptr,
            .commandPool = *slot.pool,
            .level = VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            .commandBufferCount = kSecondaryAllocBatchSize,
        };
        const VkResult alloc_res = dld_.vkAllocateCommandBuffers(
            vk_device, &alloc_ci, slot.allocated_buffers.data() + prev_count);
        if (alloc_res != VK_SUCCESS) [[unlikely]] {
            slot.allocated_buffers.resize(prev_count);
            LOG_ERROR(Render_Vulkan,
                      "ParallelCommandPool: vkAllocateCommandBuffers failed on lane {} ({})",
                      worker_lane, static_cast<s32>(alloc_res));
            return VK_NULL_HANDLE;
        }
    }

    const VkCommandBuffer secondary_cmd = slot.allocated_buffers[slot.used_count++];
    slot.needs_reset = true;

    const VkCommandBufferInheritanceRenderingInfo rendering_inheritance =
        rendering_sig.ToInheritanceRenderingInfo(VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT);

    const VkCommandBufferInheritanceInfo inheritance_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
        .pNext = &rendering_inheritance,
        .renderPass = VK_NULL_HANDLE,
        .subpass = 0u,
        .framebuffer = VK_NULL_HANDLE,
        .occlusionQueryEnable = VK_FALSE,
        .queryFlags = 0u,
        .pipelineStatistics = 0u,
    };

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT |
                 VK_COMMAND_BUFFER_USAGE_RENDER_PASS_CONTINUE_BIT,
        .pInheritanceInfo = &inheritance_info,
    };

    const VkResult begin_res = dld_.vkBeginCommandBuffer(secondary_cmd, &begin_info);
    if (begin_res != VK_SUCCESS) [[unlikely]] {
        LOG_ERROR(Render_Vulkan, "ParallelCommandPool: vkBeginCommandBuffer failed on lane {} ({})",
                  worker_lane, static_cast<s32>(begin_res));
        return VK_NULL_HANDLE;
    }

    secondary_cmd_buffers_recorded_.fetch_add(1ULL, std::memory_order_relaxed);
    return secondary_cmd;
}

void ParallelCommandPool::EndWorkerSecondary(VkCommandBuffer secondary_cmd) const noexcept {
    if (secondary_cmd == VK_NULL_HANDLE || dld_.vkEndCommandBuffer == nullptr) [[unlikely]] {
        return;
    }
    const VkResult res = dld_.vkEndCommandBuffer(secondary_cmd);
    if (res != VK_SUCCESS) [[unlikely]] {
        LOG_ERROR(Render_Vulkan, "ParallelCommandPool: vkEndCommandBuffer failed ({})",
                  static_cast<s32>(res));
    }
}

void ParallelCommandPool::ProcessSliceOnLane(u32 lane_id, const PassCommandSlice& slice,
                                             VkCommandBuffer& out_cmd) noexcept {
    const VkCommandBuffer secondary = BeginWorkerSecondary(lane_id, slice.rendering_signature);
    if (secondary == VK_NULL_HANDLE) [[unlikely]] {
        out_cmd = VK_NULL_HANDLE;
        return;
    }

    CommandRecorder& recorder = lanes_[lane_id]->recorder;
    recorder.BeginRecording(secondary);
    recorder.RecordPassSlice(slice);
    EndWorkerSecondary(secondary);

    out_cmd = secondary;
    slices_recorded_.fetch_add(1ULL, std::memory_order_relaxed);
}

void ParallelCommandPool::WorkerLoop(u32 lane_id) noexcept {
    Common::SetCurrentThreadName("NvnCleanRec");

    u64 seen_generation = 0ULL;
    for (;;) {
        if (stop_workers_.load(std::memory_order_acquire)) [[unlikely]] {
            return;
        }

        // Lock-free work-stealing loop over the currently open 16-bit tagged wave.
        for (;;) {
            const u32 packed_cursor = next_slice_cursor_.fetch_add(1u, std::memory_order_acq_rel);
            const u32 packed_count = active_slice_count_.load(std::memory_order_acquire);

            const u32 cursor_tag = packed_cursor >> 16u;
            const u32 count_tag = packed_count >> 16u;
            const u32 cursor = packed_cursor & kWaveIndexMask;
            const u32 count = packed_count & kWaveIndexMask;

            if (cursor_tag == 0u || cursor_tag != count_tag || cursor >= count) {
                break;
            }

            const PassCommandSlice* const slices_ptr =
                active_slices_ptr_.load(std::memory_order_relaxed);
            VkCommandBuffer* const out_ptr =
                active_out_cmds_ptr_.load(std::memory_order_relaxed);

            const PassCommandSlice& slice = slices_ptr[cursor];
            const bool use_explicit_index = (packed_count & kExplicitSliceIndexFlag) != 0u;
            const u32 dst_index = use_explicit_index ? slice.slice_index : cursor;

            ProcessSliceOnLane(lane_id, slice, out_ptr[dst_index]);

            if (pending_slice_completions_.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
                pending_slice_completions_.notify_all();
            }
        }

        const u64 current_gen = work_generation_.load(std::memory_order_acquire);
        if (stop_workers_.load(std::memory_order_acquire)) [[unlikely]] {
            return;
        }
        if (current_gen != seen_generation) {
            seen_generation = current_gen;
            continue;
        }

        work_generation_.wait(seen_generation, std::memory_order_acquire);
        seen_generation = work_generation_.load(std::memory_order_acquire);
    }
}

bool ParallelCommandPool::RecordSlicesParallel(std::span<const PassCommandSlice> slices,
                                               std::span<VkCommandBuffer> out_secondaries) noexcept {
    if (slices.empty()) {
        return true;
    }
    if (out_secondaries.size() < slices.size()) [[unlikely]] {
        return false;
    }

    const bool use_explicit_index = HasDistinctValidSliceIndices(slices, out_secondaries.size());
    std::fill_n(out_secondaries.data(), slices.size(), VK_NULL_HANDLE);

    batches_dispatched_.fetch_add(1ULL, std::memory_order_relaxed);

    // Inline lane-0 fast path when no background workers exist or only a single slice is given.
    if (worker_count_ <= 1u || worker_threads_.empty() || slices.size() == 1u) {
        bool all_ok = true;
        for (size_t i = 0u; i < slices.size(); ++i) {
            const u32 dst_index =
                use_explicit_index ? slices[i].slice_index : static_cast<u32>(i);
            ProcessSliceOnLane(0u, slices[i], out_secondaries[dst_index]);
            all_ok &= (out_secondaries[dst_index] != VK_NULL_HANDLE);
        }
        return all_ok;
    }

    // Dispatch in lock-free 16-bit tagged waves (up to 16,384 slices per wave).
    size_t offset = 0u;
    while (offset < slices.size()) {
        const u32 wave_count =
            static_cast<u32>((std::min<size_t>)(slices.size() - offset, kMaxSlicesPerWave));
        const PassCommandSlice* const wave_slices = slices.data() + offset;
        VkCommandBuffer* const wave_out =
            use_explicit_index ? out_secondaries.data() : (out_secondaries.data() + offset);

        const u64 next_gen = work_generation_.load(std::memory_order_relaxed) + 1ULL;
        u32 wave_tag = static_cast<u32>(next_gen & 0xFFFFULL);
        if (wave_tag == 0u) {
            wave_tag = 1u;
        }

        // Keep cursor gate closed (tag = 0) while publishing wave pointers and slice count.
        next_slice_cursor_.store(0u, std::memory_order_relaxed);
        active_slices_ptr_.store(wave_slices, std::memory_order_relaxed);
        active_out_cmds_ptr_.store(wave_out, std::memory_order_relaxed);
        pending_slice_completions_.store(wave_count, std::memory_order_relaxed);

        const u32 packed_count =
            (wave_tag << 16u) | (use_explicit_index ? kExplicitSliceIndexFlag : 0u) | wave_count;
        active_slice_count_.store(packed_count, std::memory_order_release);
        next_slice_cursor_.store(wave_tag << 16u, std::memory_order_release);

        work_generation_.store(next_gen, std::memory_order_release);
        work_generation_.notify_all();

        // Caller thread participates as lane 0 so the replay thread never sits idle.
        for (;;) {
            const u32 packed_cursor = next_slice_cursor_.fetch_add(1u, std::memory_order_acq_rel);
            const u32 cursor_tag = packed_cursor >> 16u;
            const u32 cursor = packed_cursor & kWaveIndexMask;
            if (cursor_tag != wave_tag || cursor >= wave_count) {
                break;
            }

            const PassCommandSlice& slice = wave_slices[cursor];
            const u32 dst_index = use_explicit_index ? slice.slice_index : cursor;
            ProcessSliceOnLane(0u, slice, wave_out[dst_index]);

            if (pending_slice_completions_.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
                pending_slice_completions_.notify_all();
            }
        }

        // Brief spin-pause before sleeping on the completion atomic.
        u32 remaining = pending_slice_completions_.load(std::memory_order_acquire);
        for (u32 spin = 0u; remaining != 0u && spin < kCompletionSpinIterations; ++spin) {
            CpuSpinPause();
            remaining = pending_slice_completions_.load(std::memory_order_acquire);
        }
        while (remaining != 0u) {
            pending_slice_completions_.wait(remaining, std::memory_order_acquire);
            remaining = pending_slice_completions_.load(std::memory_order_acquire);
        }

        // Close the wave before advancing or returning.
        active_slice_count_.store(0u, std::memory_order_release);
        next_slice_cursor_.store(0u, std::memory_order_release);
        active_slices_ptr_.store(nullptr, std::memory_order_relaxed);
        active_out_cmds_ptr_.store(nullptr, std::memory_order_relaxed);

        offset += wave_count;
    }

    bool all_ok = true;
    if (use_explicit_index) {
        for (const PassCommandSlice& s : slices) {
            all_ok &= (out_secondaries[s.slice_index] != VK_NULL_HANDLE);
        }
    } else {
        for (size_t i = 0u; i < slices.size(); ++i) {
            all_ok &= (out_secondaries[i] != VK_NULL_HANDLE);
        }
    }
    return all_ok;
}

void ParallelCommandPool::ExecuteSecondaryBuffers(
    vk::CommandBuffer primary_cmd, std::span<const VkCommandBuffer> secondaries) const noexcept {
    const VkCommandBuffer raw_primary = *primary_cmd;
    if (raw_primary == VK_NULL_HANDLE || secondaries.empty() ||
        direct_dispatch_.vkCmdExecuteCommands == nullptr) [[unlikely]] {
        return;
    }

    // Fast path: single `vkCmdExecuteCommands` call when all secondary handles are valid.
    bool all_valid = true;
    for (const VkCommandBuffer sec : secondaries) {
        if (sec == VK_NULL_HANDLE) [[unlikely]] {
            all_valid = false;
            break;
        }
    }
    if (all_valid) [[likely]] {
        direct_dispatch_.vkCmdExecuteCommands(
            raw_primary, static_cast<u32>(secondaries.size()), secondaries.data());
        return;
    }

    // Defensive fallback: coalesce contiguous non-null secondary command buffer runs.
    size_t idx = 0u;
    while (idx < secondaries.size()) {
        if (secondaries[idx] == VK_NULL_HANDLE) {
            ++idx;
            continue;
        }
        size_t run_end = idx + 1u;
        while (run_end < secondaries.size() && secondaries[run_end] != VK_NULL_HANDLE) {
            ++run_end;
        }
        direct_dispatch_.vkCmdExecuteCommands(
            raw_primary, static_cast<u32>(run_end - idx), secondaries.data() + idx);
        idx = run_end;
    }
}

ParallelRecorderStats ParallelCommandPool::Stats() const noexcept {
    u64 total_draws = 0ULL;
    u64 total_elided = 0ULL;

    for (const std::unique_ptr<WorkerLane>& lane : lanes_) {
        if (!lane) {
            continue;
        }
        const RecorderStats& s = lane->recorder.Stats();
        total_draws += s.draws_recorded;
        total_elided += s.elided_pipeline_binds + s.elided_descriptor_binds +
                        s.elided_vertex_binds + s.elided_index_binds + s.elided_push_constants;
    }

    return ParallelRecorderStats{
        .worker_count = worker_count_,
        .batches_dispatched = batches_dispatched_.load(std::memory_order_relaxed),
        .slices_recorded = slices_recorded_.load(std::memory_order_relaxed),
        .secondary_cmd_buffers_recorded =
            secondary_cmd_buffers_recorded_.load(std::memory_order_relaxed),
        .total_draws_recorded = total_draws,
        .total_elided_binds = total_elided,
    };
}

} // namespace Vulkan::NvnClean
