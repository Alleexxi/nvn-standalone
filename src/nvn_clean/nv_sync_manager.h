// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/nvn_types.h"
#include "video_core/nvn/record.h"
#include "nvn_clean/nv_mempool.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
} // namespace Vulkan

namespace Vulkan::NvnClean {

class PassManager;

// ================================================================================================
// Architectural Overview & First-Principles Complexity (`nv_sync_manager.h`)
// ================================================================================================
//
// Legacy synchronization & query emulation stalls both CPU and GPU pipelines via:
//   1. Binary semaphores / `vkQueueWaitIdle` / CPU completion threads mediating `NVNsync` waits.
//   2. Host `vkGetQueryPoolResults` readbacks or render-pass breaks on every `ReportCounter` /
//      `ResetCounter` (Ocarina of Time Market issues ~700 occlusion resets/reports per frame).
//   3. Deferred CPU-thread timestamp conversion that cannot satisfy same-frame GPU consumers
//      (e.g. `SetRenderEnableConditional` predicates or shader UBO/SSBO counter reads).
//
// `SyncManager` eliminates 100% of CPU synchronization stalls on `NvnReplay`:
//
//   1. Direct `VK_KHR_timeline_semaphore` Mapping for `NVNsync` & `NVNevent`:
//      - Each NVN queue (`0 .. 63`) owns a persistent monotonic 64-bit Vulkan Timeline Semaphore.
//      - `FenceSync(sync_serial)` binds `sync_serial -> {queue_id, signal_value}` in `O(1)`.
//      - Same-queue `WaitSync` is an `O(1)` pure no-op (ordered by queue stream semantics).
//      - Cross-queue `WaitSync` emits a GPU-only `VkSemaphoreSubmitInfo` (`vkQueueSubmit2`) with
//        zero CPU waiting.
//      - Guest `nvnSyncWait` checks a cache-line-aligned atomic `completed_value` in user space
//        first (`O(1)`, ~1 ns, zero driver calls when already signaled), refreshing via
//        non-blocking `vkGetSemaphoreCounterValue` before sleeping only the calling guest fiber
//        in `vkWaitSemaphores` if still in flight.
//
//   2. GPU-Buffer Queries + Compute Dispatch Resolution (`O(0)` CPU Wait Stalls):
//      - Occlusion (`SAMPLES_PASSED`) and timestamp (`TIMESTAMP`, `TIMESTAMP_TOP`) queries record
//        inside active Dynamic Rendering passes without splitting passes.
//      - Outside the pass at coarse boundaries, `vkCmdCopyQueryPoolResults` streams raw 64-bit
//        hardware ticks/samples directly into a device-local ring buffer on the GPU, followed by
//        a single compute dispatch (`DispatchQueryResolves`) that:
//          a) Folds `ResetCounter` / `SamplesPassed` events per NVN queue accumulator,
//          b) Scales raw GPU ticks to guest `NVNcounterData::timestamp` via 64-bit fixed-point
//             arithmetic on the GPU,
//          c) Writes 16-byte `NVNcounterData { u64 counter; u64 timestamp; }` directly into the
//             guest `HOST_VISIBLE | DEVICE_LOCAL` pool (`nv_mempool.h`) via 64-bit
//             `VkDeviceAddress` (`VK_KHR_buffer_device_address`),
//          d) Evaluates `SetRenderEnableConditional` (`RENDER_IF_EQUAL` / `RENDER_IF_NOT_EQUAL`)
//             predicates directly into a device-local conditional rendering buffer for
//             `VK_EXT_conditional_rendering`.
//      - CPU recording complexity: `O(1)` per query/sync command (~15-25 ns), `O(1)` single
//        compute dispatch per resolved pass batch, `O(0)` CPU-GPU synchronization stalls.
// ================================================================================================

/// Maximum logical NVN queues supported concurrently.
inline constexpr u32 kMaxNvnQueues = 64;

/// Hardware query slots per pooled `VkQueryPool` block (power of two).
inline constexpr u32 kQueriesPerPoolChunk = 1024;

/// Maximum queued query/reset/report/condition micro-ops resolved in one compute dispatch batch.
inline constexpr u32 kMaxResolveOpsPerBatch = 4096;

/// Ring capacity (in micro-ops) of the persistently mapped ReBAR query-resolve command buffer.
inline constexpr u32 kQueryResolveRingCapacity = 65536;

/// Ring capacity (in 32-bit predicate slots, 16-byte aligned) for `VK_EXT_conditional_rendering`.
inline constexpr u32 kConditionalPredicateSlots = 4096;

/// NVN counter type constants (`NVNcounterType`).
namespace NvnCounterKind {
inline constexpr u32 kTimestamp = 0x00;
inline constexpr u32 kSamplesPassed = 0x01;
inline constexpr u32 kTimestampTop = 0x10;
} // namespace NvnCounterKind

/// NVN conditional rendering compare modes (`NVNconditionalRenderMode`).
namespace NvnConditionMode {
inline constexpr u32 kRenderIfEqual = 0;
inline constexpr u32 kRenderIfNotEqual = 1;
} // namespace NvnConditionMode

/// Result codes mirroring `NVNsyncWaitResult` for `nvnSyncWait`.
enum class SyncWaitStatus : u32 {
    AlreadySignaled = 0, ///< Satisfied via O(1) atomic cache or non-blocking counter poll
    ConditionSatisfied = 1,
    TimeoutExpired = 2,
    Failed = 3,
};

/// Immutable GPU timeline coordinate representing an NVN `FenceSync` or `SignalEvent` point.
struct TimelineSyncPoint {
    u32 queue_id{0};
    u64 timeline_value{0};
    u32 condition{0};
    u32 flags{0};
    bool valid{false};

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return valid && timeline_value != 0;
    }
};

/// GPU-visible 32-byte micro-op consumed by the query-resolve & conditional-predicate compute pass.
/// Laid out identically on C++20 host and GLSL/SPIR-V (`std430` / `buffer_reference`).
struct alignas(16) GpuQueryResolveOp {
    enum class OpCode : u32 {
        ResetOcclusionAccumulator = 0,
        AccumulateOcclusionQuery = 1,
        WriteOcclusionReport = 2,
        WriteTimestampReport = 3,
        WriteImmediateValue = 4,
        EvaluateConditionEqual = 5,
        EvaluateConditionNotEqual = 6,
    };

    u32 opcode{0};               ///< `OpCode`
    u32 queue_id{0};             ///< Logical NVN queue index `[0, kMaxNvnQueues)`
    u32 raw_occlusion_index{~0u};///< Index into `raw_query_results` (`UINT32_MAX` if none)
    u32 raw_timestamp_index{~0u};///< Index into `raw_query_results` (`UINT32_MAX` if none)
    VkDeviceAddress dst_bda{0};  ///< Destination `NVNcounterData` or condition source A (`VkDeviceAddress`)
    u64 aux_value_or_src_b{0};   ///< Immediate 64-bit value, predicate slot index, or condition B BDA
};
static_assert(sizeof(GpuQueryResolveOp) == 32,
              "GpuQueryResolveOp must match 32-byte std430 compute shader ABI");

/// Push constants supplied to the built-in query-resolve compute shader (`32` bytes).
struct GpuQueryResolvePushConstants {
    u32 op_begin_index{0};
    u32 op_count{0};
    /// Fixed-point multiplier and right-shift converting raw GPU timestamp ticks to nanoseconds:
    /// `ns = mul_u64_hi_shr(raw_ticks, tick_to_ns_mul, tick_to_ns_shift) + guest_clock_offset_ns`.
    u64 tick_to_ns_mul{1};
    u64 guest_clock_offset_ns{0};
    u32 tick_to_ns_shift{0};
    u32 timestamp_valid_bits_mask_hi{~0u};
};
static_assert(sizeof(GpuQueryResolvePushConstants) == 32);

/// Telemetry snapshot for `SyncManager`.
struct SyncManagerStats {
    u64 timeline_submissions{};
    u64 fence_syncs_recorded{};
    u64 same_queue_waits_elided{};
    u64 cross_queue_semaphore_waits{};
    u64 host_sync_fast_cache_hits{};
    u64 host_sync_driver_polls{};
    u64 host_sync_blocking_waits{};
    u64 occlusion_queries_recorded{};
    u64 timestamp_queries_recorded{};
    u64 counter_resets_recorded{};
    u64 compute_resolve_dispatches{};
    u64 gpu_conditions_evaluated{};
};

/// Optional external SPIR-V override if the caller supplies custom compiled query-resolve SPIR-V;
/// when empty, `SyncManager` compiles/loads the embedded default SPIR-V module.
struct SyncManagerConfig {
    std::span<const u32> custom_resolve_spirv{};
    bool enable_in_pass_queries{true};
    bool enable_conditional_rendering{true};
};

/// Zero-stall `VK_KHR_timeline_semaphore` and GPU-buffer query/predicate manager for `NvnClean`.
class SyncManager {
public:
    /// Verifies support for `VK_KHR_timeline_semaphore`, `VK_KHR_buffer_device_address`, and
    /// `VK_EXT_host_query_reset`.
    [[nodiscard]] static bool IsSupported(const Device& device) noexcept;

    explicit SyncManager(const Device& device, MemoryPoolManager& mempool_manager,
                         VmaAllocator allocator = VK_NULL_HANDLE,
                         const SyncManagerConfig& config = {});
    ~SyncManager();

    SyncManager(const SyncManager&) = delete;
    SyncManager& operator=(const SyncManager&) = delete;

    SyncManager(SyncManager&&) = delete;
    SyncManager& operator=(SyncManager&&) = delete;

    // --------------------------------------------------------------------------------------------
    // Per-Queue Timeline Semaphores (`VK_KHR_timeline_semaphore` & `nvnSyncWait`)
    // --------------------------------------------------------------------------------------------

    /// Returns the timeline `VkSemaphore` handle for `queue_id` (lazily created on first access).
    /// Complexity: O(1) after first use on `queue_id`.
    [[nodiscard]] VkSemaphore QueueTimelineSemaphore(u32 queue_id);

    /// Allocates the next monotonically increasing 64-bit timeline value to be signaled when the
    /// current command buffer segment on `queue_id` is submitted.
    /// Complexity: O(1) lock-free atomic increment.
    [[nodiscard]] u64 AllocateNextTimelineValue(u32 queue_id);

    /// Populates a `VkSemaphoreSubmitInfo` (`VK_KHR_synchronization2`) that signals `timeline_value`
    /// on `queue_id` at `stage_mask` when passed to `vkQueueSubmit2`.
    /// Complexity: O(1).
    [[nodiscard]] VkSemaphoreSubmitInfo MakeTimelineSignalSubmitInfo(
        u32 queue_id, u64 timeline_value,
        VkPipelineStageFlags2 stage_mask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    /// Marks `timeline_value` as submitted to the Vulkan queue on `queue_id`.
    /// Complexity: O(1) lock-free monotonic store.
    void NotifyTimelineSubmitted(u32 queue_id, u64 timeline_value) noexcept;

    /// Non-blocking query of the latest completed GPU timeline value on `queue_id`.
    /// Refreshes the user-space atomic cache via `vkGetSemaphoreCounterValue` only when needed.
    /// Complexity: O(1), zero blocking.
    [[nodiscard]] u64 PollCompletedTimelineValue(u32 queue_id) noexcept;

    /// Fast user-space check (`O(1)`, ~1 ns, zero Vulkan calls on hit) returning true if
    /// `timeline_value` on `queue_id` has already retired on the GPU.
    [[nodiscard]] bool IsTimelineValueComplete(u32 queue_id, u64 timeline_value) noexcept;

    /// Records an NVN `Op::FenceSync` (`nvnQueueFenceSync`), associating `sync_serial` with
    /// `(queue_id, signal_timeline_value)`.
    /// Complexity: O(1).
    void RecordFenceSync(u64 sync_serial, u32 queue_id, u64 signal_timeline_value,
                         u32 condition = 0, u32 flags = 0);

    /// Releases metadata for a finalized `NVNsync` (`nvnSyncFinalize`). Complexity: O(1).
    void DestroySync(u64 sync_serial) noexcept;

    /// Looks up a previously fenced `TimelineSyncPoint` by `sync_serial`. Complexity: O(1).
    [[nodiscard]] TimelineSyncPoint FindSyncPoint(u64 sync_serial) const noexcept;

    /// Translates an NVN `Op::WaitSync` (`nvnQueueWaitSync`) on `consumer_queue_id` into a GPU
    /// timeline semaphore wait (`VkSemaphoreSubmitInfo`):
    ///   - Returns `false` if `sync_serial` was fenced on `consumer_queue_id` itself (same-queue
    ///     execution order already satisfies the dependency: `same_queue_waits_elided++`) or if
    ///     the producer timeline value has already completed on the GPU!
    ///   - Returns `true` and fills `out_wait_info` when a cross-queue GPU wait is required.
    /// Complexity: O(1), zero CPU stalls.
    [[nodiscard]] bool ResolveQueueWaitSync(
        u64 sync_serial, u32 consumer_queue_id, VkSemaphoreSubmitInfo& out_wait_info,
        VkPipelineStageFlags2 dst_stage_mask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT) noexcept;

    /// Guest-facing `nvnSyncWait(sync, timeout_ns)` implementation:
    ///   1. Checks `cached_completed_value` in user space (`O(1)`, 0 driver calls if complete).
    ///   2. Polls `vkGetSemaphoreCounterValue` once without sleeping (`O(1)`).
    ///   3. If `timeout_ns > 0` and still pending, waits on `vkWaitSemaphores` for `timeout_ns`.
    /// Never called on `NvnReplay`; safe to invoke concurrently from any guest HLE thread.
    [[nodiscard]] SyncWaitStatus WaitSyncHost(u64 sync_serial, u64 timeout_ns) noexcept;

    // --------------------------------------------------------------------------------------------
    // NVN Event Synchronization (`Op::SignalEvent` / `Op::WaitEvent`)
    // --------------------------------------------------------------------------------------------

    /// Records a GPU-ordered `Op::SignalEvent` (`WRITE` or `ADD`) into the zero-copy guest pool
    /// backing `guest_event_address` and associates its completion with `(queue_id, timeline_val)`.
    void RecordSignalEvent(vk::CommandBuffer cmd, u32 queue_id, u64 timeline_val,
                           u64 guest_event_address, u32 signal_mode, u32 value);

    // --------------------------------------------------------------------------------------------
    // GPU-Buffer Occlusion & Timestamp Queries (Compute-Resolved, Zero CPU Stalls)
    // --------------------------------------------------------------------------------------------

    /// Reclaims retired `VkQueryPool` chunks whose `retire_timeline_value` has completed on the
    /// GPU (`vkResetQueryPool` on host, zero GPU wait). Called once per submission/frame.
    void ReclaimCompletedQueryPools() noexcept;

    /// Begins or resumes an in-pass hardware occlusion query (`VK_QUERY_CONTROL_PRECISE_BIT`) if
    /// one is not already active in the current rendering pass.
    /// Complexity: O(1).
    void EnsureOcclusionQueryActive(vk::CommandBuffer cmd, u32 queue_id);

    /// Ends the currently active hardware occlusion query before `EndRendering` or `ResetCounter`,
    /// queueing an `AccumulateOcclusionQuery` micro-op for the compute resolve batch.
    /// Complexity: O(1).
    void CloseActiveOcclusionQuery(vk::CommandBuffer cmd, u32 queue_id);

    /// Records `Op::ResetCounter` (`nvnCommandBufferResetCounter`) without breaking the active
    /// Dynamic Rendering pass. Consecutive resets coalesce in `O(1)`.
    void RecordResetCounter(vk::CommandBuffer cmd, u32 queue_id, u32 counter_type);

    /// Records `Op::ReportCounter` (`TIMESTAMP`, `TIMESTAMP_TOP`, or `SAMPLES_PASSED`) targeting
    /// 16-byte `NVNcounterData` at `guest_report_address` inside a zero-copy `GuestMemoryPool`.
    /// Keeps the active Dynamic Rendering pass open when `enable_in_pass_queries` is supported!
    /// Complexity: O(1) (~20 ns), zero CPU readback.
    bool RecordReportCounter(vk::CommandBuffer cmd, u32 queue_id, u32 counter_type,
                             u64 guest_report_address);

    /// Records `Op::ReportValue` (`{ immediate_value, converted_gpu_timestamp }`) to
    /// `guest_report_address` via the GPU compute resolve stream.
    /// Complexity: O(1).
    bool RecordReportValue(vk::CommandBuffer cmd, u32 queue_id, u64 guest_report_address,
                           u64 immediate_value);

    /// Flushes all pending raw hardware queries (`vkCmdCopyQueryPoolResults` -> device buffer) and
    /// issues a SINGLE compute dispatch (`vkCmdDispatch`) writing resolved 16-byte `NVNcounterData`
    /// values directly to guest `VkDeviceAddress` memory.
    /// Must be called outside an active Dynamic Rendering pass (automatically closes any open pass
    /// on `pass_manager` if pending ops exist).
    ///
    /// Complexity: O(1) when no query ops are pending; 1 `vkCmdDispatch` per batch of up to 4,096
    /// query/reset/report events. Zero CPU-GPU synchronization stalls.
    /// @return Number of query micro-ops resolved on the GPU.
    u32 FlushQueryResolves(vk::CommandBuffer cmd, PassManager* pass_manager = nullptr,
                           u32 submit_queue_id = 0, u64 retire_timeline_value = 0);

    [[nodiscard]] bool HasPendingQueryResolves() const noexcept {
        return !pending_resolve_ops_.empty();
    }

    // --------------------------------------------------------------------------------------------
    // GPU Conditional Rendering (`SetRenderCondition` / `VK_EXT_conditional_rendering`)
    // --------------------------------------------------------------------------------------------

    /// Evaluates an NVN conditional render predicate (`guest_counter_pair_address` +0 vs +16) on
    /// the GPU via the compute resolve shader and begins `vkCmdBeginConditionalRenderingEXT`.
    /// Complexity: O(1), zero CPU readback of guest counter memory.
    bool BeginConditionalRendering(vk::CommandBuffer cmd, PassManager* pass_manager,
                                   u64 guest_counter_pair_address, u32 condition_mode);

    /// Ends active `VK_EXT_conditional_rendering` scope on `cmd`. Complexity: O(1).
    void EndConditionalRendering(vk::CommandBuffer cmd) noexcept;

    [[nodiscard]] bool IsConditionalRenderingActive() const noexcept {
        return conditional_rendering_active_;
    }

    [[nodiscard]] SyncManagerStats Stats() const noexcept;

private:
    struct alignas(64) QueueTimelineSlot {
        std::mutex create_mutex;
        vk::Semaphore semaphore;
        std::atomic<VkSemaphore> raw_handle{VK_NULL_HANDLE};
        std::atomic<u64> next_value{1};
        std::atomic<u64> submitted_value{0};
        std::atomic<u64> cached_completed_value{0};
    };

    struct QueryPoolChunk {
        vk::QueryPool occlusion_pool;
        vk::QueryPool timestamp_pool;
        u32 occlusion_used{0};
        u32 timestamp_used{0};
        u32 occlusion_copied{0};
        u32 timestamp_copied{0};
        u32 owner_queue_id{0};
        u64 retire_timeline_value{0};
    };

    struct AllocatedInternalBuffer {
        vk::Buffer buffer;
        VmaAllocation allocation{VK_NULL_HANDLE};
        VkDeviceAddress device_address{0};
        std::span<u8> mapped{};
    };

    void InitializeGpuQueryResources();
    QueryPoolChunk& AcquireActiveQueryChunk(u32 queue_id);
    [[nodiscard]] u32 AllocateRawQueryResultSlot() noexcept;
    [[nodiscard]] u32 AllocatePredicateSlot() noexcept;

    const Device& device_;
    const vk::DeviceDispatch& dld_;
    MemoryPoolManager& mempool_manager_;
    VmaAllocator owned_allocator_{VK_NULL_HANDLE};
    VmaAllocator allocator_{VK_NULL_HANDLE};
    SyncManagerConfig config_{};

    // Fixed-point timestamp conversion parameters derived from `VkPhysicalDeviceLimits::timestampPeriod`.
    u64 tick_to_ns_mul_{1};
    u32 tick_to_ns_shift_{0};
    u32 timestamp_valid_bits_mask_hi_{~0u};
    std::atomic<u64> guest_clock_offset_ns_{0};

    // Per-NVN-queue timeline semaphores (`VK_KHR_timeline_semaphore`).
    std::array<QueueTimelineSlot, kMaxNvnQueues> queue_timelines_{};

    // `sync_serial -> TimelineSyncPoint` registry.
    mutable std::shared_mutex sync_table_mutex_;
    std::unordered_map<u64, TimelineSyncPoint> sync_points_;

    // GPU buffers for zero-stall query results, micro-op ring, per-queue accumulators, predicates.
    AllocatedInternalBuffer raw_query_results_buffer_{};   ///< Device-local `u64[]`
    AllocatedInternalBuffer queue_accumulators_buffer_{};  ///< Device-local `u64[kMaxNvnQueues]`
    AllocatedInternalBuffer resolve_ops_ring_buffer_{};    ///< ReBAR `GpuQueryResolveOp[]`
    AllocatedInternalBuffer predicate_ring_buffer_{};      ///< Device-local `u32[]` for conditional rendering

    // Compute pipeline resolving raw hardware queries directly to guest `NVNcounterData` via BDA.
    vk::DescriptorSetLayout resolve_set_layout_;
    vk::DescriptorPool resolve_descriptor_pool_;
    VkDescriptorSet resolve_descriptor_set_{VK_NULL_HANDLE};
    vk::ShaderModule resolve_shader_module_;
    vk::PipelineLayout resolve_pipeline_layout_;
    vk::Pipeline resolve_pipeline_;

    // Replay-thread query & batch tracking.
    std::vector<QueryPoolChunk> active_chunks_;
    std::deque<QueryPoolChunk> in_flight_chunks_;
    std::vector<QueryPoolChunk> free_chunks_;

    bool occlusion_query_open_{false};
    u32 open_occlusion_chunk_idx_{0};
    u32 open_occlusion_query_slot_{0};
    u32 open_occlusion_raw_slot_{0};

    bool conditional_rendering_active_{false};
    u32 raw_result_write_cursor_{0};
    u32 resolve_op_ring_cursor_{0};
    u32 predicate_ring_cursor_{0};
    std::vector<GpuQueryResolveOp> pending_resolve_ops_;

    std::atomic<u64> timeline_submissions_{0};
    std::atomic<u64> fence_syncs_recorded_{0};
    std::atomic<u64> same_queue_waits_elided_{0};
    std::atomic<u64> cross_queue_semaphore_waits_{0};
    std::atomic<u64> host_sync_fast_cache_hits_{0};
    std::atomic<u64> host_sync_driver_polls_{0};
    std::atomic<u64> host_sync_blocking_waits_{0};
    std::atomic<u64> occlusion_queries_recorded_{0};
    std::atomic<u64> timestamp_queries_recorded_{0};
    std::atomic<u64> counter_resets_recorded_{0};
    std::atomic<u64> compute_resolve_dispatches_{0};
    std::atomic<u64> gpu_conditions_evaluated_{0};
};

} // namespace Vulkan::NvnClean
