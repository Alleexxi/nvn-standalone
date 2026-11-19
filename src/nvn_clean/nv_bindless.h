// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <shared_mutex>
#include <span>
#include <vector>

#include "common/common_types.h"
#include "video_core/nvn/nvn_types.h"
#include "video_core/nvn/sink.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
} // namespace Vulkan

namespace Vulkan::NvnClean {

// ================================================================================================
// Architectural Overview & First-Principles Complexity (`nv_bindless.h`)
// ================================================================================================
//
// Legacy Maxwell/NVN replay paths resolve every bound guest texture handle on the CPU per draw:
//   guest TIC/TSC handle -> pool snapshot lookup -> hash map lookup -> image view + layout check ->
//   per-draw vkCmdPushDescriptorSetKHR / transient descriptor set allocation.
// In Ocarina of Time's Market scene (~3,500 draws * ~6 textures/draw = ~21,000 handle resolutions
// per frame), CPU handle resolution and descriptor updates consume 1.5 - 3.5 ms per frame.
//
// `BindlessDescriptorTable` eliminates per-draw CPU texture/sampler work completely:
//   1. Single Persistent Set 0 (`VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT`):
//      - Binding 0: `VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE[16384]`
//      - Binding 1: `VK_DESCRIPTOR_TYPE_SAMPLER[4096]`
//      Both configured with `PARTIALLY_BOUND | UPDATE_AFTER_BIND | UPDATE_UNUSED_WHILE_PENDING`
//      across `VK_SHADER_STAGE_ALL`. Every graphics and compute `VkPipelineLayout` shares this
//      exact Set 0 `VkDescriptorSetLayout`, so Set 0 is bound ONCE per command buffer.
//   2. Registration-Time Descriptor Population:
//      - On `nvnTextureInitialize` / `nvnTexturePoolRegisterTexture`, `OnTextureInitialize` writes
//        the `VkImageView` directly into Set 0 slot `texture_id`.
//      - On `nvnSamplerInitialize` / `nvnSamplerPoolRegisterSampler`, `OnSamplerInitialize` writes
//        the `VkSampler` directly into Set 0 slot `sampler_id`.
//   3. Zero-CPU Draw Path (`O(0)` per-draw CPU cost):
//      - Guest shaders read raw 64-bit `NVNtextureHandle` values directly from zero-copy mapped
//        uniform/storage buffers (`nv_mempool.h`) and index Set 0 in SPIR-V:
//            uint tex_id  = uint(handle) & 0xFFFFFu; // bits [19:0]  -> [0, 16384)
//            uint samp_id = uint(handle) >> 20u;     // bits [31:20] -> [0, 4096)
//   4. Fast `O(1)` Residency & Epoch Dirty Check:
//      - Global epoch word packs `(epoch << 1) | dirty_bit` in a single atomic 64-bit integer:
//        verifying table cleanliness at pass/submission start is a single `O(1)` register load
//        and compare (~1 ns).
//      - Per-slot residency and dirty tracking use dense L1-resident bitsets (2 KiB for 16,384
//        textures, 512 B for 4,096 samplers) plus a 256-bit hierarchical summary mask, giving
//        branchless `O(1)` per-handle residency checks (~1-2 CPU cycles) and `O(D)` flush cost
//        proportional strictly to the `D` slots mutated since the last flush (`O(0)` when clean).
// ================================================================================================

/// Persistent descriptor set index reserved for the global bindless table in all pipeline layouts.
inline constexpr u32 kBindlessDescriptorSetIndex = 0;

/// Binding index within Set 0 for the global `texture2D` / sampled image table.
inline constexpr u32 kBindlessSampledImageBinding = 0;

/// Binding index within Set 0 for the global `sampler` table.
inline constexpr u32 kBindlessSamplerBinding = 1;

/// Total sampled image slots in persistent Set 0 (`2^14 = 16,384`).
inline constexpr u32 kMaxBindlessSampledImages = 16384;

/// Total sampler slots in persistent Set 0 (`2^12 = 4,096`).
inline constexpr u32 kMaxBindlessSamplers = 4096;

/// Bit-packing constants matching the NVN hardware/SDK `NVNtextureHandle` ABI and SPIR-V backend.
inline constexpr u32 kBindlessSamplerShift = 20;
inline constexpr u32 kBindlessTextureIdMask = (1u << kBindlessSamplerShift) - 1u; // 0x000FFFFFu
inline constexpr u32 kBindlessSamplerIdMask = (1u << (32u - kBindlessSamplerShift)) - 1u; // 0xFFFu

static_assert(kMaxBindlessSampledImages <= kBindlessTextureIdMask + 1u,
              "Sampled image capacity must fit in the 20-bit texture handle field");
static_assert(kMaxBindlessSamplers <= kBindlessSamplerIdMask + 1u,
              "Sampler capacity must fit in the 12-bit sampler handle field");
static_assert((kMaxBindlessSampledImages % 64u) == 0u,
              "Sampled image capacity must be a multiple of 64 for bitset words");
static_assert((kMaxBindlessSamplers % 64u) == 0u,
              "Sampler capacity must be a multiple of 64 for bitset words");

/// Extracts the 20-bit texture descriptor slot (`handle & 0xFFFFF`) from a 32-bit or 64-bit handle.
/// Complexity: O(1) (1 AND instruction).
[[nodiscard]] inline constexpr u32 UnpackBindlessTextureId(u64 guest_handle) noexcept {
    return static_cast<u32>(guest_handle) & kBindlessTextureIdMask;
}

/// Extracts the 12-bit sampler descriptor slot (`(uint32_t(handle) >> 20)`) from a guest handle.
/// Complexity: O(1) (1 SHR instruction on the 32-bit handle word).
[[nodiscard]] inline constexpr u32 UnpackBindlessSamplerId(u64 guest_handle) noexcept {
    return static_cast<u32>(guest_handle) >> kBindlessSamplerShift;
}

/// Packs `(texture_id, sampler_id)` into the 32-bit shader handle word:
/// `(texture_id & 0xFFFFF) | (sampler_id << 20)`.
/// Complexity: O(1).
[[nodiscard]] inline constexpr u32 PackBindlessHandle32(u32 texture_id, u32 sampler_id) noexcept {
    return (texture_id & kBindlessTextureIdMask) |
           ((sampler_id & kBindlessSamplerIdMask) << kBindlessSamplerShift);
}

/// Packs `(texture_id, sampler_id)` into a 64-bit `NVNtextureHandle`.
/// Complexity: O(1).
[[nodiscard]] inline constexpr VideoCommon::Nvn::NVNtextureHandle PackGuestTextureHandle(
    u32 texture_id, u32 sampler_id) noexcept {
    return static_cast<VideoCommon::Nvn::NVNtextureHandle>(
        PackBindlessHandle32(texture_id, sampler_id));
}

/// Per-slot metadata for a registered sampled image in Set 0 Binding 0.
/// Compact 32-byte POD aligned to half a cache line (2 slots per 64-byte L1 line).
struct alignas(32) BindlessTextureSlot {
    VkImageView image_view{VK_NULL_HANDLE};
    u64 texture_serial{0};
    u64 updated_epoch{0};
    VkImageLayout image_layout{VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    u32 flags{0}; ///< Bit 0: resident, Bit 1: dirty (pending VkWriteDescriptorSet)
};
static_assert(sizeof(BindlessTextureSlot) == 32);

/// Per-slot metadata for a registered sampler in Set 0 Binding 1.
struct alignas(32) BindlessSamplerSlot {
    VkSampler sampler{VK_NULL_HANDLE};
    u64 sampler_serial{0};
    u64 updated_epoch{0};
    u64 flags{0}; ///< Bit 0: resident, Bit 1: dirty (pending VkWriteDescriptorSet)
};
static_assert(sizeof(BindlessSamplerSlot) == 32);

/// Telemetry counters for `BindlessDescriptorTable`.
struct BindlessStats {
    u64 current_epoch{};
    u32 resident_textures{};
    u32 resident_samplers{};
    u64 immediate_texture_writes{};
    u64 immediate_sampler_writes{};
    u64 coalesced_flush_calls{};
    u64 coalesced_descriptor_writes{};
    u64 layout_rebind_writes{};
    u64 epoch_fast_hits{};
};

/// Configuration options for `BindlessDescriptorTable`.
struct BindlessTableConfig {
    /// When true (default), `OnTextureInitialize` / `OnSamplerInitialize` write directly to the
    /// persistent `UPDATE_AFTER_BIND` Set 0 immediately via `vkUpdateDescriptorSets` so the slot is
    /// GPU-visible with zero deferred flush latency. Layout transitions during replay may optionally
    /// batch via `DeferTextureLayoutUpdate` + `FlushDirtyDescriptors`.
    bool immediate_write_on_init{true};

    /// Pre-populates all 16,384 image slots and 4,096 sampler slots at construction time with a
    /// safe 1x1 transparent-black fallback `VkImageView` and linear-clamp fallback `VkSampler` so
    /// null/stale guest handles (`0`) never fault on drivers without `nullDescriptor`.
    bool populate_safe_fallback_slots{true};
};

/// Single persistent Set 0 bindless descriptor table (`UPDATE_AFTER_BIND`) for `Vulkan::NvnClean`.
///
/// Thread-safety:
/// - Fast residency & epoch checks (`IsEpochClean`, `IsTextureResident`, `IsSamplerResident`,
///   `IsHandleResident`, `PersistentSet`, `Layout`) are lock-free `O(1)` operations safe to call
///   concurrently from any worker thread during secondary command buffer recording.
/// - Slot mutations (`OnTextureInitialize`, `OnSamplerInitialize`, `UpdateTextureLayout`,
///   `FlushDirtyDescriptors`) serialize internally via `mutex_` while leaving `UPDATE_AFTER_BIND`
///   command buffer recording completely unblocked.
class BindlessDescriptorTable {
public:
    static constexpr size_t kTextureBitWords = kMaxBindlessSampledImages / 64; // 256 words (2 KiB)
    static constexpr size_t kTextureSummaryWords = kTextureBitWords / 64;      // 4 words (32 B)
    static constexpr size_t kSamplerBitWords = kMaxBindlessSamplers / 64;      // 64 words (512 B)

    /// Queries physical device limits and `VkPhysicalDeviceDescriptorIndexingFeatures` to verify
    /// support for persistent Set 0 with 16,384 sampled images and 4,096 samplers.
    [[nodiscard]] static bool IsSupported(const Device& device) noexcept;

    explicit BindlessDescriptorTable(const Device& device,
                                     VmaAllocator allocator = VK_NULL_HANDLE,
                                     const BindlessTableConfig& config = {});
    ~BindlessDescriptorTable();

    BindlessDescriptorTable(const BindlessDescriptorTable&) = delete;
    BindlessDescriptorTable& operator=(const BindlessDescriptorTable&) = delete;

    BindlessDescriptorTable(BindlessDescriptorTable&&) = delete;
    BindlessDescriptorTable& operator=(BindlessDescriptorTable&&) = delete;

    // --------------------------------------------------------------------------------------------
    // Persistent Vulkan Handles (Set 0)
    // --------------------------------------------------------------------------------------------

    /// Shared Set 0 `VkDescriptorSetLayout` used by every graphics and compute `VkPipelineLayout`.
    [[nodiscard]] VkDescriptorSetLayout DescriptorSetLayout() const noexcept {
        return *set_layout_;
    }

    /// Single persistent Set 0 `VkDescriptorSet` bound once per command buffer.
    [[nodiscard]] VkDescriptorSet PersistentSet() const noexcept {
        return persistent_set_;
    }

    /// Binds the persistent Set 0 table to `cmd` at `kBindlessDescriptorSetIndex` (Set 0).
    /// Complexity: O(1) — invoked once per primary/secondary command buffer or when pipeline layout
    /// root compatibility is first established.
    void BindPersistentSet(vk::CommandBuffer cmd, VkPipelineBindPoint bind_point,
                           VkPipelineLayout pipeline_layout) const noexcept {
        cmd.BindDescriptorSets(bind_point, pipeline_layout, kBindlessDescriptorSetIndex,
                               persistent_set_, {});
    }

    // --------------------------------------------------------------------------------------------
    // Registration & Lifecycle (`nvnTextureInitialize` / `nvnSamplerInitialize`)
    // --------------------------------------------------------------------------------------------

    /// Called on `nvnTextureInitialize` (or `nvnTexturePoolRegisterTexture`): registers `image_view`
    /// at descriptor slot `texture_id` in Set 0 Binding 0.
    ///
    /// - If slot `texture_id` already holds the identical `(image_view, layout, texture_serial)`,
    ///   returns immediately in `O(1)` without issuing a redundant `vkUpdateDescriptorSets`.
    /// - Otherwise updates slot metadata, marks residency, advances the epoch, and writes Set 0
    ///   (immediately when `immediate_write_on_init` is enabled, or marks the epoch dirty bit).
    ///
    /// Complexity: O(1).
    bool OnTextureInitialize(u32 texture_id, VkImageView image_view,
                             VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             u64 texture_serial = 0) noexcept;

    /// Called on `nvnSamplerInitialize` (or `nvnSamplerPoolRegisterSampler`): writes `sampler`
    /// into Set 0 Binding 1 at `sampler_id`.
    /// Complexity: O(1).
    bool OnSamplerInitialize(u32 sampler_id, VkSampler sampler, u64 sampler_serial = 0) noexcept;

    /// Updates the descriptor `imageLayout` for an already-registered `texture_id` when a texture
    /// transitions between `SHADER_READ_ONLY_OPTIMAL` and `GENERAL` (e.g. render-target feedback).
    /// When `defer_flush` is true, sets the per-slot and global epoch dirty bit for batched
    /// `FlushDirtyDescriptors()` before pass begin; otherwise writes immediately.
    /// Complexity: O(1).
    bool UpdateTextureLayout(u32 texture_id, VkImageLayout new_layout,
                             bool defer_flush = true) noexcept;

    /// Unregisters a finalized texture (`nvnTextureFinalize`) if `expected_serial == 0` or matches
    /// the slot's current `texture_serial`, restoring the safe fallback view in Set 0.
    /// Complexity: O(1).
    void OnTextureFinalize(u32 texture_id, u64 expected_serial = 0) noexcept;

    /// Unregisters a finalized sampler (`nvnSamplerFinalize`) if `expected_serial == 0` or matches
    /// the slot's current `sampler_serial`, restoring the safe fallback sampler in Set 0.
    /// Complexity: O(1).
    void OnSamplerFinalize(u32 sampler_id, u64 expected_serial = 0) noexcept;

    /// Flushes all pending dirty texture and sampler slots to `persistent_set_` in a single batched
    /// `vkUpdateDescriptorSets` call using the 2-level hierarchical dirty bitmask, then clears the
    /// global epoch dirty bit.
    ///
    /// Complexity:
    /// - Fast path (epoch dirty bit == 0): `O(1)` (~1 ns, single atomic load, zero Vulkan calls).
    /// - Dirty path: `O(D)` where `D` is the exact number of dirty slots; `O(0)` scan over clean
    ///   64-slot blocks via `std::countr_zero` on `texture_dirty_summary_`.
    /// @return Number of `VkWriteDescriptorSet` operations issued.
    u32 FlushDirtyDescriptors() noexcept;

    /// Resets all non-fallback slots back to the fallback descriptors (e.g. on process shutdown).
    void ResetAllSlots() noexcept;

    // --------------------------------------------------------------------------------------------
    // Fast O(1) Residency & Epoch Dirty-Bit Queries (Lock-Free Hot Path)
    // --------------------------------------------------------------------------------------------

    /// Packed 64-bit epoch state: bit 0 = `dirty_bit`, bits [63:1] = `epoch`.
    [[nodiscard]] u64 RawEpochState() const noexcept {
        return packed_epoch_state_.load(std::memory_order_acquire);
    }

    /// Current monotonic descriptor table epoch (`packed_epoch_state_ >> 1`).
    /// Complexity: O(1).
    [[nodiscard]] u64 CurrentEpoch() const noexcept {
        return packed_epoch_state_.load(std::memory_order_acquire) >> 1u;
    }

    /// True if any descriptor slot has an unflushed mutation pending (`packed_epoch_state_ & 1`).
    /// Complexity: O(1) (single bit test).
    [[nodiscard]] bool HasDirtyEpochBit() const noexcept {
        return (packed_epoch_state_.load(std::memory_order_acquire) & 1ULL) != 0ULL;
    }

    /// Fast `O(1)` observer validation: returns true if the table has NO dirty bit set AND its
    /// epoch equals `inout_cached_epoch`. When clean at a newer epoch, updates `inout_cached_epoch`
    /// after confirming flush completion.
    /// Complexity: O(1) (~1-2 CPU cycles on cache hit).
    [[nodiscard]] bool CheckAndAdvanceCleanEpoch(u64& inout_cached_epoch) const noexcept {
        const u64 state = packed_epoch_state_.load(std::memory_order_acquire);
        if ((state & 1ULL) != 0ULL) [[unlikely]] {
            return false;
        }
        const u64 epoch = state >> 1u;
        if (inout_cached_epoch == epoch) [[likely]] {
            epoch_fast_hits_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        inout_cached_epoch = epoch;
        return true;
    }

    /// Branchless `O(1)` check whether `texture_id` has a live texture bound AND its epoch dirty bit
    /// is clear (i.e. resident and up-to-date in Set 0 on the GPU).
    /// Complexity: O(1) (2 L1-resident bitset word loads, zero locks).
    [[nodiscard]] bool IsTextureResident(u32 texture_id) const noexcept {
        if (texture_id >= kMaxBindlessSampledImages) [[unlikely]] {
            return false;
        }
        const size_t word_idx = texture_id >> 6u;
        const u64 bit = 1ULL << (texture_id & 63u);
        const u64 resident = texture_resident_bits_[word_idx].load(std::memory_order_relaxed);
        const u64 dirty = texture_dirty_bits_[word_idx].load(std::memory_order_relaxed);
        return ((resident & ~dirty) & bit) != 0ULL;
    }

    /// Branchless `O(1)` check whether `sampler_id` has a live sampler bound AND its epoch dirty bit
    /// is clear in Set 0.
    /// Complexity: O(1).
    [[nodiscard]] bool IsSamplerResident(u32 sampler_id) const noexcept {
        if (sampler_id >= kMaxBindlessSamplers) [[unlikely]] {
            return false;
        }
        const size_t word_idx = sampler_id >> 6u;
        const u64 bit = 1ULL << (sampler_id & 63u);
        const u64 resident = sampler_resident_bits_[word_idx].load(std::memory_order_relaxed);
        const u64 dirty = sampler_dirty_bits_[word_idx].load(std::memory_order_relaxed);
        return ((resident & ~dirty) & bit) != 0ULL;
    }

    /// Branchless `O(1)` residency validation for a packed `NVNtextureHandle` (`handle & 0xFFFFF`
    /// and `handle >> 20`). Used only by optional validation/diagnostic probes or residency asserts;
    /// normal draw execution performs ZERO handle resolution on the CPU.
    /// Complexity: O(1).
    [[nodiscard]] bool IsHandleResident(u64 guest_handle) const noexcept {
        const u32 tex_id = UnpackBindlessTextureId(guest_handle);
        const u32 smp_id = UnpackBindlessSamplerId(guest_handle);
        return IsTextureResident(tex_id) && IsSamplerResident(smp_id);
    }

    /// Exact slot serial + layout residency check for attachment-feedback hazard validation.
    /// Complexity: O(1) direct array index.
    [[nodiscard]] bool IsTextureSlotMatching(u32 texture_id, u64 expected_serial,
                                             VkImageLayout expected_layout) const noexcept {
        if (!IsTextureResident(texture_id)) {
            return false;
        }
        const BindlessTextureSlot& slot = texture_slots_[texture_id];
        return slot.texture_serial == expected_serial && slot.image_layout == expected_layout;
    }

    /// Direct read-only inspection of a texture slot's metadata (`texture_id < 16384`).
    [[nodiscard]] const BindlessTextureSlot& TextureSlot(u32 texture_id) const noexcept {
        return texture_slots_[texture_id];
    }

    /// Direct read-only inspection of a sampler slot's metadata (`sampler_id < 4096`).
    [[nodiscard]] const BindlessSamplerSlot& SamplerSlot(u32 sampler_id) const noexcept {
        return sampler_slots_[sampler_id];
    }

    [[nodiscard]] BindlessStats Stats() const noexcept;

private:
    void CreateDescriptorTable();
    void CreateFallbackResources();
    void WriteImageDescriptorImmediate(u32 slot, VkImageView view, VkImageLayout layout) noexcept;
    void WriteSamplerDescriptorImmediate(u32 slot, VkSampler sampler) noexcept;
    void MarkTextureSlotDirtyLocked(u32 slot) noexcept;
    void MarkSamplerSlotDirtyLocked(u32 slot) noexcept;
    void BumpEpochLocked(bool mark_dirty) noexcept;

    const Device& device_;
    const vk::DeviceDispatch& dld_;
    VmaAllocator owned_allocator_{VK_NULL_HANDLE};
    VmaAllocator allocator_{VK_NULL_HANDLE};
    BindlessTableConfig config_{};

    vk::DescriptorSetLayout set_layout_;
    vk::DescriptorPool descriptor_pool_;
    VkDescriptorSet persistent_set_{VK_NULL_HANDLE};

    // Safe 1x1 rgba8 zeroed fallback image + default linear sampler bound to unused slots.
    vk::Image fallback_image_;
    vk::ImageView fallback_image_view_;
    vk::Sampler fallback_sampler_;

    // Hot cache line: packed `(epoch << 1) | dirty_bit` for 1-cycle O(1) epoch cleanliness test.
    alignas(64) std::atomic<u64> packed_epoch_state_{2ULL}; // epoch = 1, dirty_bit = 0

    // Dense lock-free L1-resident residency & dirty bitsets:
    //   256 * 8B = 2,048 B for textures; 64 * 8B = 512 B for samplers.
    alignas(64) std::array<std::atomic<u64>, kTextureBitWords> texture_resident_bits_{};
    alignas(64) std::array<std::atomic<u64>, kTextureBitWords> texture_dirty_bits_{};
    alignas(64) std::array<std::atomic<u64>, kTextureSummaryWords> texture_dirty_summary_{};
    alignas(64) std::array<std::atomic<u64>, kSamplerBitWords> sampler_resident_bits_{};
    alignas(64) std::array<std::atomic<u64>, kSamplerBitWords> sampler_dirty_bits_{};
    std::atomic<u64> sampler_dirty_summary_{0};

    mutable std::shared_mutex mutex_;
    std::vector<BindlessTextureSlot> texture_slots_;
    std::vector<BindlessSamplerSlot> sampler_slots_;

    // Reusable scratch buffers for coalesced `FlushDirtyDescriptors()` (zero steady-state allocs).
    std::vector<VkDescriptorImageInfo> scratch_image_infos_;
    std::vector<VkWriteDescriptorSet> scratch_writes_;

    std::atomic<u32> resident_textures_{0};
    std::atomic<u32> resident_samplers_{0};
    std::atomic<u64> immediate_texture_writes_{0};
    std::atomic<u64> immediate_sampler_writes_{0};
    std::atomic<u64> coalesced_flush_calls_{0};
    std::atomic<u64> coalesced_descriptor_writes_{0};
    std::atomic<u64> layout_rebind_writes_{0};
    mutable std::atomic<u64> epoch_fast_hits_{0};
};

} // namespace Vulkan::NvnClean
