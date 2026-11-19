// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/nvn_types.h"
#include "video_core/nvn/record.h"
#include "nvn_clean/nv_bindless.h"
#include "nvn_clean/nv_mempool.h"
#include "nvn_clean/nv_pass_manager.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
} // namespace Vulkan

namespace Vulkan::NvnClean {

// ================================================================================================
// Architectural Overview & First-Principles Complexity (`nv_command_recorder.h`)
// ================================================================================================
//
// Legacy multi-threaded replay (`MtDrawPacket` / `MtCommandTape`) serialized draws on the main
// thread into heap-allocated intermediate packet structures (copying descriptor infos, buffer
// bindings, and push constants) before re-decoding those packets on worker threads. That design
// paid BOTH main-thread packet construction cost AND heap/cache-miss overhead (~250-600 ns/draw).
//
// `CommandRecorder` and `ParallelCommandPool` execute directly from the immutable `u64` NVN
// command word stream into hardware `VkCommandBuffer`s with ZERO intermediate heap structs:
//
//   1. `< 30 ns` Direct Draw Fast Path (`O(1)` when clean, `O(runs)` when dirty):
//      - `CommandRecorder` caches raw `PFN_vkCmd*` function pointers directly inside its first
//        cache lines (eliminating `Device -> Logical -> DispatchLoader` pointer chasing).
//      - All state setters (`SetPipeline`, `SetVertexBuffer`, `SetIndexBuffer`, `SetDynamicUbo`,
//        `SetPushConstants`, `SetViewport`, `SetScissor`) compare against L1-resident shadow state
//        and flip bits in a single 64-bit `dirty_mask_` ONLY when a value actually changes.
//      - Steady-state `Draw` / `DrawIndexed`:
//            if (dirty_mask_ != 0) [[unlikely]] FlushDirtyState();
//            dispatch_.vkCmdDrawIndexed(cmd_, ...);
//        compiles to 1 register test + 1 direct ICD jump (~8-15 ns clean, ~20-28 ns with dynamic
//        UBO offset update; 0 bytes heap-allocated).
//
//   2. `O(1)` Bitmask State Elision & Contiguous Run Coalescing:
//      - Vertex streams (`[0..31]`) and dynamic UBO slots (`[0..15]`) track validity and dirtiness
//        in 32-bit / 16-bit integer bitmasks.
//      - `FlushDirtyState()` uses hardware TZCNT (`std::countr_zero` / `std::countr_one`) to emit
//        only contiguous changed runs `[first, first + run_len)` in a single `vkCmdBind*` call,
//        skipping all unchanged slots in `O(1)` bitwise jumps without looping over 32 slots.
//
//   3. Lock-Free Multi-Threaded Secondary Command Buffer Recording:
//      - Each worker lane `[0, kMaxRecorderWorkers)` owns cache-line-isolated (`alignas(64)`)
//        per-timeline-epoch `VkCommandPool` arenas (`VK_COMMAND_POOL_CREATE_TRANSIENT_BIT`).
//        Workers NEVER contend on a `VkCommandPool` mutex and NEVER allocate per-draw structs.
//      - Work distribution uses a single atomic index (`fetch_add(1, relaxed)`) over immutable
//        `PassCommandSlice` views into the submission's `u64` command list, recording Vulkan 1.3
//        Dynamic Rendering `RENDER_PASS_CONTINUE` secondary command buffers (`vkCmdExecuteCommands`).
// ================================================================================================

/// Maximum hardware worker threads supported by `ParallelCommandPool`.
inline constexpr u32 kMaxRecorderWorkers = 16;

/// Default number of timeline-fenced command pool ring epochs per worker lane.
inline constexpr u32 kDefaultCommandRingEpochs = 4;

/// Maximum push constant shadow buffer size in bytes (Vulkan architectural max used by NVN).
inline constexpr u32 kMaxPushConstantBytes = 256;

/// Maximum Simultaneous Viewports / Scissors in NVN.
inline constexpr u32 kMaxViewports = 16;

/// First Set index used for Dynamic Uniform Buffers (Set 0 is reserved for `BindlessDescriptorTable`).
inline constexpr u32 kFirstDynamicUboSetIndex = 1;

/// 64-bit dirty state bitmask flags for `CommandRecorder`.
namespace RecorderDirtyBits {
inline constexpr u64 kGraphicsPipeline = 1ULL << 0;
inline constexpr u64 kComputePipeline = 1ULL << 1;
inline constexpr u64 kBindlessSet0 = 1ULL << 2;
inline constexpr u64 kDynamicUbos = 1ULL << 3;
inline constexpr u64 kVertexBuffers = 1ULL << 4;
inline constexpr u64 kIndexBuffer = 1ULL << 5;
inline constexpr u64 kPushConstants = 1ULL << 6;
inline constexpr u64 kViewports = 1ULL << 7;
inline constexpr u64 kScissors = 1ULL << 8;
inline constexpr u64 kBlendConstants = 1ULL << 9;
inline constexpr u64 kStencilState = 1ULL << 10;
inline constexpr u64 kDepthBias = 1ULL << 11;
inline constexpr u64 kExtendedDynamic = 1ULL << 12;
} // namespace RecorderDirtyBits

/// Flat table of direct Vulkan ICD command function pointers cached inside each `CommandRecorder`
/// to avoid multi-level pointer indirection on every draw call.
struct DirectCmdDispatch {
    PFN_vkCmdDraw vkCmdDraw{nullptr};
    PFN_vkCmdDrawIndexed vkCmdDrawIndexed{nullptr};
    PFN_vkCmdDrawIndirect vkCmdDrawIndirect{nullptr};
    PFN_vkCmdDrawIndexedIndirect vkCmdDrawIndexedIndirect{nullptr};
    PFN_vkCmdDrawIndirectCount vkCmdDrawIndirectCount{nullptr};
    PFN_vkCmdDrawIndexedIndirectCount vkCmdDrawIndexedIndirectCount{nullptr};
    PFN_vkCmdDispatch vkCmdDispatch{nullptr};
    PFN_vkCmdDispatchIndirect vkCmdDispatchIndirect{nullptr};
    PFN_vkCmdBindPipeline vkCmdBindPipeline{nullptr};
    PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets{nullptr};
    PFN_vkCmdBindVertexBuffers vkCmdBindVertexBuffers{nullptr};
    PFN_vkCmdBindVertexBuffers2EXT vkCmdBindVertexBuffers2EXT{nullptr};
    PFN_vkCmdBindIndexBuffer vkCmdBindIndexBuffer{nullptr};
    PFN_vkCmdPushConstants vkCmdPushConstants{nullptr};
    PFN_vkCmdSetViewport vkCmdSetViewport{nullptr};
    PFN_vkCmdSetScissor vkCmdSetScissor{nullptr};
    PFN_vkCmdSetBlendConstants vkCmdSetBlendConstants{nullptr};
    PFN_vkCmdSetDepthBias vkCmdSetDepthBias{nullptr};
    PFN_vkCmdExecuteCommands vkCmdExecuteCommands{nullptr};

    /// Populates direct ICD function pointers from `device`.
    [[nodiscard]] static DirectCmdDispatch FromDevice(const Device& device) noexcept;
};

/// Lightweight snapshot of NVN guest state live at the start of a parallel pass slice.
/// Trivially copyable (no heap pointers/vectors) so slicing a pass for worker threads is `O(1)`
/// stack/arena copy with zero heap allocations.
struct SliceEntryGuestState {
    u64 program_serial{0};
    u64 bindless_set_epoch{0};
    u32 active_vertex_stream_mask{0};
    u16 active_ubo_slot_mask{0};
    u8 index_type{0};
    bool has_index_buffer{false};
    u64 index_address{0};
    std::array<VideoCommon::Nvn::BufferRange, kMaxVertexStreams> vertex_ranges{};
    std::array<u32, kMaxVertexStreams> vertex_strides{};
    std::array<VideoCommon::Nvn::BufferRange, kMaxDynamicUniformSlots> graphics_ubos{};
    VkViewport primary_viewport{};
    VkRect2D primary_scissor{};
};

/// Zero-allocation descriptor of a single pass or sub-pass command slice for worker execution.
/// Points directly into the immutable `Submission::lists` `u64` word array (`[words, words + word_count)`).
struct PassCommandSlice {
    const u64* words{nullptr};
    size_t word_count{0};
    PipelineRenderingSignature rendering_signature{};
    VkRect2D render_area{};
    VkDescriptorSet bindless_set0{VK_NULL_HANDLE};
    SliceEntryGuestState entry_state{};
    u32 slice_index{0};
};

/// Per-recorder execution and state-elision telemetry counters.
struct RecorderStats {
    u64 draws_recorded{};
    u64 dispatches_recorded{};
    u64 pipeline_binds{};
    u64 elided_pipeline_binds{};
    u64 descriptor_set_binds{};
    u64 elided_descriptor_binds{};
    u64 vertex_buffer_binds{};
    u64 elided_vertex_binds{};
    u64 index_buffer_binds{};
    u64 elided_index_binds{};
    u64 push_constant_updates{};
    u64 elided_push_constants{};
};

/// Allocation-free, single-thread command buffer state machine and `< 30 ns` draw emitter.
///
/// One instance lives on the primary thread and one inside each worker's cache-line-isolated
/// `WorkerLane` — never shared concurrently across threads, requiring ZERO locks or atomics on
/// the draw hot path.
class alignas(64) CommandRecorder {
public:
    CommandRecorder() noexcept = default;
    explicit CommandRecorder(const DirectCmdDispatch& dispatch,
                             const MemoryPoolManager* mempool = nullptr,
                             const BindlessDescriptorTable* bindless = nullptr) noexcept
        : dispatch_{dispatch}, mempool_{mempool}, bindless_{bindless} {}

    /// Attaches `cmd` (primary or secondary `VkCommandBuffer`) and resets all tracked shadow state
    /// and dirty bitmasks in `O(1)`.
    void BeginRecording(VkCommandBuffer cmd) noexcept {
        cmd_ = cmd;
        ResetState();
    }

    /// Resets all shadow bindings and dirty masks (called at command buffer begin or pass boundary
    /// when external state may have been disturbed). Complexity: O(1).
    void ResetState() noexcept {
        dirty_mask_ = 0;
        dirty_vertex_mask_ = 0;
        valid_vertex_mask_ = 0;
        dirty_ubo_mask_ = 0;
        valid_ubo_mask_ = 0;
        dirty_viewport_mask_ = 0;
        valid_viewport_mask_ = 0;
        dirty_scissor_mask_ = 0;
        valid_scissor_mask_ = 0;
        graphics_pipeline_ = VK_NULL_HANDLE;
        compute_pipeline_ = VK_NULL_HANDLE;
        graphics_layout_ = VK_NULL_HANDLE;
        compute_layout_ = VK_NULL_HANDLE;
        bound_bindless_set0_ = VK_NULL_HANDLE;
        index_valid_ = false;
        push_constants_valid_ = false;
        push_dirty_begin_ = kMaxPushConstantBytes;
        push_dirty_end_ = 0;
        extended_vertex_bind_ = false;
    }

    [[nodiscard]] VkCommandBuffer Handle() const noexcept {
        return cmd_;
    }

    [[nodiscard]] u64 PendingDirtyMask() const noexcept {
        return dirty_mask_;
    }

    // --------------------------------------------------------------------------------------------
    // O(1) Bitmask-Elided State Setters (Zero Vulkan Calls Until Changed + Flushed)
    // --------------------------------------------------------------------------------------------

    /// Binds a graphics pipeline + compatible layout with `O(1)` identity elision.
    /// Because every `NvnClean` pipeline shares Set 0 (`BindlessDescriptorTable`), switching
    /// pipelines preserves `bound_bindless_set0_` whenever `bound_bindless_set0_ != VK_NULL_HANDLE`!
    void SetGraphicsPipeline(VkPipeline pipeline, VkPipelineLayout layout) noexcept {
        if (pipeline == graphics_pipeline_ && layout == graphics_layout_) [[likely]] {
            ++stats_.elided_pipeline_binds;
            return;
        }
        if (pipeline != graphics_pipeline_) {
            graphics_pipeline_ = pipeline;
            dirty_mask_ |= RecorderDirtyBits::kGraphicsPipeline;
        }
        if (layout != graphics_layout_) {
            const bool first_layout = (graphics_layout_ == VK_NULL_HANDLE);
            graphics_layout_ = layout;
            if (first_layout && desired_bindless_set0_ != VK_NULL_HANDLE &&
                bound_bindless_set0_ == VK_NULL_HANDLE) {
                dirty_mask_ |= RecorderDirtyBits::kBindlessSet0;
            }
            // Non-Set-0 descriptor bindings must re-validate if pipeline layout object changed.
            if (!first_layout && valid_ubo_mask_ != 0) {
                dirty_ubo_mask_ |= valid_ubo_mask_;
                dirty_mask_ |= RecorderDirtyBits::kDynamicUbos;
            }
        }
    }

    /// Sets persistent Set 0 (`BindlessDescriptorTable::PersistentSet()`). Elided in `O(1)` after
    /// the first draw of the command buffer.
    void SetBindlessDescriptorSet(VkDescriptorSet set0) noexcept {
        desired_bindless_set0_ = set0;
        if (set0 == bound_bindless_set0_) [[likely]] {
            ++stats_.elided_descriptor_binds;
            return;
        }
        dirty_mask_ |= RecorderDirtyBits::kBindlessSet0;
    }

    /// Sets dynamic uniform buffer `slot` (`[0, kMaxDynamicUniformSlots)` -> Vulkan Set `1 + slot`).
    /// Compares `(descriptor_set, dynamic_offset)` in `O(1)` and elides unchanged binds.
    void SetDynamicUniformBuffer(u32 slot, VkDescriptorSet descriptor_set,
                                 u32 dynamic_offset) noexcept {
        if (slot >= kMaxDynamicUniformSlots || descriptor_set == VK_NULL_HANDLE) [[unlikely]] {
            return;
        }
        const u16 bit = static_cast<u16>(1u << slot);
        if ((valid_ubo_mask_ & bit) != 0 && ubo_sets_[slot] == descriptor_set &&
            ubo_dynamic_offsets_[slot] == dynamic_offset) [[likely]] {
            ++stats_.elided_descriptor_binds;
            return;
        }
        valid_ubo_mask_ |= bit;
        dirty_ubo_mask_ |= bit;
        ubo_sets_[slot] = descriptor_set;
        ubo_dynamic_offsets_[slot] = dynamic_offset;
        dirty_mask_ |= RecorderDirtyBits::kDynamicUbos;
    }

    /// Resolves `guest_address` directly through `MemoryPoolManager::ResolveUniformBinding` and
    /// applies `O(1)` bitmask elision (zero buffer copies, zero barriers).
    void SetDynamicUniformAddress(u32 slot, u64 guest_address, u64 size = 0) noexcept {
        if (mempool_ == nullptr) [[unlikely]] {
            return;
        }
        const UniformBindingView view = mempool_->ResolveUniformBinding(guest_address, size);
        if (view) [[likely]] {
            SetDynamicUniformBuffer(slot, view.descriptor_set, view.dynamic_offset);
        }
    }

    /// Sets vertex buffer binding `slot` (`[0, 32)`) with `O(1)` bitmask elision.
    void SetVertexBuffer(u32 slot, VkBuffer buffer, VkDeviceSize offset) noexcept {
        if (slot >= kMaxVertexStreams || buffer == VK_NULL_HANDLE) [[unlikely]] {
            return;
        }
        const u32 bit = 1u << slot;
        if (!extended_vertex_bind_ && (valid_vertex_mask_ & bit) != 0 &&
            vertex_buffers_[slot] == buffer && vertex_offsets_[slot] == offset) [[likely]] {
            ++stats_.elided_vertex_binds;
            return;
        }
        valid_vertex_mask_ |= bit;
        dirty_vertex_mask_ |= bit;
        vertex_buffers_[slot] = buffer;
        vertex_offsets_[slot] = offset;
        dirty_mask_ |= RecorderDirtyBits::kVertexBuffers;
    }

    /// Extended dynamic vertex buffer setter (`vkCmdBindVertexBuffers2EXT` with size & stride).
    void SetVertexBufferExtended(u32 slot, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
                                 VkDeviceSize stride) noexcept {
        if (slot >= kMaxVertexStreams || buffer == VK_NULL_HANDLE) [[unlikely]] {
            return;
        }
        const u32 bit = 1u << slot;
        if (extended_vertex_bind_ && (valid_vertex_mask_ & bit) != 0 &&
            vertex_buffers_[slot] == buffer && vertex_offsets_[slot] == offset &&
            vertex_sizes_[slot] == size && vertex_strides_[slot] == stride) [[likely]] {
            ++stats_.elided_vertex_binds;
            return;
        }
        extended_vertex_bind_ = true;
        valid_vertex_mask_ |= bit;
        dirty_vertex_mask_ |= bit;
        vertex_buffers_[slot] = buffer;
        vertex_offsets_[slot] = offset;
        vertex_sizes_[slot] = size;
        vertex_strides_[slot] = stride;
        dirty_mask_ |= RecorderDirtyBits::kVertexBuffers;
    }

    /// Resolves `guest_address` directly from `MemoryPoolManager` and sets vertex stream `slot`.
    void SetVertexBufferAddress(u32 slot, u64 guest_address, u64 size = 0) noexcept {
        if (mempool_ == nullptr) [[unlikely]] {
            return;
        }
        const BufferRangeView view = mempool_->ResolveBufferRange(guest_address, size);
        if (view) [[likely]] {
            SetVertexBuffer(slot, view.buffer, view.offset);
        }
    }

    /// Sets the bound index buffer with `O(1)` equality elision.
    void SetIndexBuffer(VkBuffer buffer, VkDeviceSize offset, VkIndexType index_type) noexcept {
        if (buffer == VK_NULL_HANDLE) [[unlikely]] {
            return;
        }
        if (index_valid_ && index_buffer_ == buffer && index_offset_ == offset &&
            index_type_ == index_type) [[likely]] {
            ++stats_.elided_index_binds;
            return;
        }
        index_valid_ = true;
        index_buffer_ = buffer;
        index_offset_ = offset;
        index_type_ = index_type;
        dirty_mask_ |= RecorderDirtyBits::kIndexBuffer;
    }

    /// Resolves `guest_address` directly from `MemoryPoolManager` and sets the index buffer.
    void SetIndexBufferAddress(u64 guest_address, VkIndexType index_type) noexcept {
        if (mempool_ == nullptr) [[unlikely]] {
            return;
        }
        const BufferRangeView view = mempool_->ResolveBufferRange(guest_address, 0);
        if (view) [[likely]] {
            SetIndexBuffer(view.buffer, view.offset, index_type);
        }
    }

    /// Updates push constant range `[offset, offset + size)` with byte-exact `O(1)` elision.
    void SetPushConstants(VkShaderStageFlags stage_flags, u32 offset, u32 size,
                          const void* values) noexcept {
        if (size == 0 || values == nullptr || offset + size > kMaxPushConstantBytes) [[unlikely]] {
            return;
        }
        if (push_constants_valid_ && push_stages_ == stage_flags &&
            offset + size <= push_valid_limit_ &&
            std::memcmp(push_constants_.data() + offset, values, size) == 0) [[likely]] {
            ++stats_.elided_push_constants;
            return;
        }
        std::memcpy(push_constants_.data() + offset, values, size);
        push_stages_ |= stage_flags;
        push_constants_valid_ = true;
        if (offset + size > push_valid_limit_) {
            push_valid_limit_ = offset + size;
        }
        if (offset < push_dirty_begin_) {
            push_dirty_begin_ = offset;
        }
        if (offset + size > push_dirty_end_) {
            push_dirty_end_ = offset + size;
        }
        dirty_mask_ |= RecorderDirtyBits::kPushConstants;
    }

    /// Sets viewport `slot` (`[0, 16)`) with exact bitwise float comparison (`std::memcmp`).
    void SetViewport(u32 slot, const VkViewport& vp) noexcept {
        if (slot >= kMaxViewports) [[unlikely]] {
            return;
        }
        const u16 bit = static_cast<u16>(1u << slot);
        if ((valid_viewport_mask_ & bit) != 0 &&
            std::memcmp(&viewports_[slot], &vp, sizeof(VkViewport)) == 0) [[likely]] {
            return;
        }
        valid_viewport_mask_ |= bit;
        dirty_viewport_mask_ |= bit;
        viewports_[slot] = vp;
        dirty_mask_ |= RecorderDirtyBits::kViewports;
    }

    /// Sets scissor `slot` (`[0, 16)`) with `O(1)` elision.
    void SetScissor(u32 slot, const VkRect2D& rect) noexcept {
        if (slot >= kMaxViewports) [[unlikely]] {
            return;
        }
        const u16 bit = static_cast<u16>(1u << slot);
        if ((valid_scissor_mask_ & bit) != 0 &&
            std::memcmp(&scissors_[slot], &rect, sizeof(VkRect2D)) == 0) [[likely]] {
            return;
        }
        valid_scissor_mask_ |= bit;
        dirty_scissor_mask_ |= bit;
        scissors_[slot] = rect;
        dirty_mask_ |= RecorderDirtyBits::kScissors;
    }

    // --------------------------------------------------------------------------------------------
    // `< 30 ns` Direct Draw & Dispatch Execution (Zero Heap Allocations)
    // --------------------------------------------------------------------------------------------

    /// Flushes ONLY dirty bitmask runs to `cmd_` and clears `dirty_mask_ = 0`.
    /// Uses `std::countr_zero` / `std::countr_one` to coalesce contiguous vertex and UBO bindings.
    void FlushDirtyGraphicsState() noexcept;

    /// Direct non-indexed draw (`< 30 ns` steady-state: 1 register test + direct ICD jump).
    void Draw(u32 vertex_count, u32 instance_count, u32 first_vertex,
              u32 first_instance) noexcept {
        if (dirty_mask_ != 0) [[unlikely]] {
            FlushDirtyGraphicsState();
        }
        dispatch_.vkCmdDraw(cmd_, vertex_count, instance_count, first_vertex, first_instance);
        ++stats_.draws_recorded;
    }

    /// Direct indexed draw (`< 30 ns` steady-state: 1 register test + direct ICD jump).
    void DrawIndexed(u32 index_count, u32 instance_count, u32 first_index, s32 vertex_offset,
                     u32 first_instance) noexcept {
        if (dirty_mask_ != 0) [[unlikely]] {
            FlushDirtyGraphicsState();
        }
        dispatch_.vkCmdDrawIndexed(cmd_, index_count, instance_count, first_index, vertex_offset,
                                   first_instance);
        ++stats_.draws_recorded;
    }

    /// Direct indirect draw from zero-copy guest pool buffer (`nv_mempool.h`).
    void DrawIndirect(VkBuffer buffer, VkDeviceSize offset, u32 draw_count, u32 stride) noexcept {
        if (dirty_mask_ != 0) [[unlikely]] {
            FlushDirtyGraphicsState();
        }
        dispatch_.vkCmdDrawIndirect(cmd_, buffer, offset, draw_count, stride);
        ++stats_.draws_recorded;
    }

    /// Direct indexed indirect draw from zero-copy guest pool buffer (`nv_mempool.h`).
    void DrawIndexedIndirect(VkBuffer buffer, VkDeviceSize offset, u32 draw_count,
                             u32 stride) noexcept {
        if (dirty_mask_ != 0) [[unlikely]] {
            FlushDirtyGraphicsState();
        }
        dispatch_.vkCmdDrawIndexedIndirect(cmd_, buffer, offset, draw_count, stride);
        ++stats_.draws_recorded;
    }

    /// Decodes and records an immutable `PassCommandSlice` directly from the NVN `u64` record
    /// stream into `cmd_` without allocating any intermediate heap packets.
    void RecordPassSlice(const PassCommandSlice& slice) noexcept;

    [[nodiscard]] const RecorderStats& Stats() const noexcept {
        return stats_;
    }

private:
    // --- Cache line 0: Hot draw execution fields ---
    u64 dirty_mask_{0};
    VkCommandBuffer cmd_{VK_NULL_HANDLE};
    DirectCmdDispatch dispatch_{};
    const MemoryPoolManager* mempool_{nullptr};
    const BindlessDescriptorTable* bindless_{nullptr};

    // --- Shadow pipeline & descriptor state ---
    VkPipeline graphics_pipeline_{VK_NULL_HANDLE};
    VkPipeline compute_pipeline_{VK_NULL_HANDLE};
    VkPipelineLayout graphics_layout_{VK_NULL_HANDLE};
    VkPipelineLayout compute_layout_{VK_NULL_HANDLE};
    VkDescriptorSet desired_bindless_set0_{VK_NULL_HANDLE};
    VkDescriptorSet bound_bindless_set0_{VK_NULL_HANDLE};

    u32 dirty_vertex_mask_{0};
    u32 valid_vertex_mask_{0};
    u16 dirty_ubo_mask_{0};
    u16 valid_ubo_mask_{0};
    u16 dirty_viewport_mask_{0};
    u16 valid_viewport_mask_{0};
    u16 dirty_scissor_mask_{0};
    u16 valid_scissor_mask_{0};

    bool index_valid_{false};
    bool extended_vertex_bind_{false};
    bool push_constants_valid_{false};
    VkIndexType index_type_{VK_INDEX_TYPE_UINT16};
    VkBuffer index_buffer_{VK_NULL_HANDLE};
    VkDeviceSize index_offset_{0};

    VkShaderStageFlags push_stages_{VK_SHADER_STAGE_ALL_GRAPHICS};
    u32 push_valid_limit_{0};
    u32 push_dirty_begin_{kMaxPushConstantBytes};
    u32 push_dirty_end_{0};
    alignas(16) std::array<u8, kMaxPushConstantBytes> push_constants_{};

    std::array<VkDescriptorSet, kMaxDynamicUniformSlots> ubo_sets_{};
    std::array<u32, kMaxDynamicUniformSlots> ubo_dynamic_offsets_{};

    std::array<VkBuffer, kMaxVertexStreams> vertex_buffers_{};
    std::array<VkDeviceSize, kMaxVertexStreams> vertex_offsets_{};
    std::array<VkDeviceSize, kMaxVertexStreams> vertex_sizes_{};
    std::array<VkDeviceSize, kMaxVertexStreams> vertex_strides_{};

    std::array<VkViewport, kMaxViewports> viewports_{};
    std::array<VkRect2D, kMaxViewports> scissors_{};

    RecorderStats stats_{};
};

/// Configuration for `ParallelCommandPool`.
struct ParallelRecorderConfig {
    /// Number of persistent background worker threads (`0` = auto-select `clamp(cores / 2, 2, 8)`).
    u32 worker_count{4};
    /// Number of timeline-fenced command pool ring epochs per worker lane.
    u32 ring_epochs{kDefaultCommandRingEpochs};
    /// Minimum draw count in a pass slice to dispatch across worker threads instead of inline.
    u32 min_draws_for_parallel{32};
};

/// Aggregate telemetry for `ParallelCommandPool`.
struct ParallelRecorderStats {
    u32 worker_count{};
    u64 batches_dispatched{};
    u64 slices_recorded{};
    u64 secondary_cmd_buffers_recorded{};
    u64 total_draws_recorded{};
    u64 total_elided_binds{};
};

/// Lock-free multi-threaded secondary command buffer pool and work-stealing dispatcher.
///
/// Ownership & Concurrency Guarantees:
///   - Each worker `lane` (`0 .. worker_count - 1`) exclusively owns its `VkCommandPool` ring
///     (`ring_epochs` pools per lane). No mutex ever guards `vkAllocateCommandBuffers`,
///     `vkResetCommandPool`, `vkBeginCommandBuffer`, or `vkCmdDraw*`.
///   - Epoch advancement checks `completed_timeline_value >= epoch.retire_timeline_value` before
///     bulk-resetting `vkResetCommandPool(pool, 0)` in `O(1)`.
///   - `RecordSlicesParallel` fans out `std::span<const PassCommandSlice>` lock-free via a single
///     atomic `next_slice_index_` (`fetch_add`), writing finished `VkCommandBuffer` handles into
///     `out_secondaries[slice_index]` in deterministic guest order for `vkCmdExecuteCommands`.
class ParallelCommandPool {
public:
    explicit ParallelCommandPool(const Device& device, const MemoryPoolManager& mempool,
                                 const BindlessDescriptorTable& bindless,
                                 const ParallelRecorderConfig& config = {});
    ~ParallelCommandPool();

    ParallelCommandPool(const ParallelCommandPool&) = delete;
    ParallelCommandPool& operator=(const ParallelCommandPool&) = delete;

    ParallelCommandPool(ParallelCommandPool&&) = delete;
    ParallelCommandPool& operator=(ParallelCommandPool&&) = delete;

    /// Advances or selects an available command pool epoch whose prior `retire_timeline_value` is
    /// `<= completed_timeline_value`, bulk-resetting each worker lane's `VkCommandPool` for that
    /// epoch without freeing individual command buffers.
    /// Complexity: O(worker_count) <= O(16) = O(1) per submission/frame.
    bool BeginSubmissionEpoch(u64 retire_timeline_value, u64 completed_timeline_value);

    /// Acquires a lock-free `RENDER_PASS_CONTINUE` secondary `VkCommandBuffer` from `worker_lane`'s
    /// active epoch arena, already begun with `VkCommandBufferInheritanceRenderingInfo` matching
    /// `rendering_sig`.
    /// Complexity: O(1) amortized (reuses pre-allocated `VkCommandBuffer` handles in the pool).
    [[nodiscard]] VkCommandBuffer BeginWorkerSecondary(
        u32 worker_lane, const PipelineRenderingSignature& rendering_sig);

    /// Ends recording on `secondary_cmd` (`vkEndCommandBuffer`). Complexity: O(1).
    void EndWorkerSecondary(VkCommandBuffer secondary_cmd) const noexcept;

    /// Dispatches `slices` across all worker threads lock-free (with caller participating as lane 0
    /// so `NvnReplay` never sits idle), writing one finished `VkCommandBuffer` per slice into
    /// `out_secondaries` (`out_secondaries.size() >= slices.size()`).
    ///
    /// Complexity: `O(TotalDraws / Workers)` wall-clock time, `< 30 ns` per draw per core,
    /// `O(0)` heap allocations.
    bool RecordSlicesParallel(std::span<const PassCommandSlice> slices,
                              std::span<VkCommandBuffer> out_secondaries) noexcept;

    /// Executes recorded secondary command buffers inside an open Dynamic Rendering pass begun with
    /// `VK_RENDERING_CONTENTS_SECONDARY_COMMAND_BUFFERS_BIT`.
    /// Complexity: O(1) (`vkCmdExecuteCommands`).
    void ExecuteSecondaryBuffers(vk::CommandBuffer primary_cmd,
                                 std::span<const VkCommandBuffer> secondaries) const noexcept;

    /// Direct access to lane 0's `CommandRecorder` for small inline passes (`draws < min_draws`).
    [[nodiscard]] CommandRecorder& InlineRecorder() noexcept {
        return lanes_[0]->recorder;
    }

    [[nodiscard]] const DirectCmdDispatch& DirectDispatch() const noexcept {
        return direct_dispatch_;
    }

    [[nodiscard]] u32 WorkerCount() const noexcept {
        return worker_count_;
    }

    [[nodiscard]] ParallelRecorderStats Stats() const noexcept;

private:
    struct EpochPoolSlot {
        vk::CommandPool pool;
        std::vector<VkCommandBuffer> allocated_buffers;
        u32 used_count{0};
        u64 retire_timeline_value{0};
        bool needs_reset{false};
    };

    /// Cache-line-isolated per-worker arena preventing false sharing across CPU cores.
    struct alignas(64) WorkerLane {
        u32 lane_id{0};
        CommandRecorder recorder;
        std::vector<EpochPoolSlot> epochs;
    };

    void WorkerLoop(u32 lane_id) noexcept;
    void ProcessSliceOnLane(u32 lane_id, const PassCommandSlice& slice,
                            VkCommandBuffer& out_cmd) noexcept;

    const Device& device_;
    const vk::DeviceDispatch& dld_;
    const MemoryPoolManager& mempool_;
    const BindlessDescriptorTable& bindless_;
    DirectCmdDispatch direct_dispatch_{};
    ParallelRecorderConfig config_{};
    u32 worker_count_{1};
    u32 active_epoch_index_{0};

    // Lane 0 belongs to the calling `NvnReplay` thread; lanes `[1 .. worker_count_)` belong to
    // persistent background worker threads.
    std::vector<std::unique_ptr<WorkerLane>> lanes_;
    std::vector<std::thread> worker_threads_;

    // Lock-free work-stealing batch state (cache-line isolated).
    alignas(64) std::atomic<const PassCommandSlice*> active_slices_ptr_{nullptr};
    std::atomic<VkCommandBuffer*> active_out_cmds_ptr_{nullptr};
    std::atomic<u32> active_slice_count_{0};
    alignas(64) std::atomic<u32> next_slice_cursor_{0};
    alignas(64) std::atomic<u32> pending_slice_completions_{0};
    alignas(64) std::atomic<u64> work_generation_{0};
    std::atomic<bool> stop_workers_{false};

    std::atomic<u64> batches_dispatched_{0};
    std::atomic<u64> slices_recorded_{0};
    std::atomic<u64> secondary_cmd_buffers_recorded_{0};
};

} // namespace Vulkan::NvnClean
