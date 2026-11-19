// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <deque>
#include <optional>
#include <shared_mutex>
#include <span>
#include <unordered_map>
#include <vector>

#include "common/common_types.h"
#ifdef _WIN32
#pragma push_macro("DIFFERENCE")
#undef DIFFERENCE
#endif
#include "video_core/nvn/nvn_types.h"
#ifdef _WIN32
#pragma pop_macro("DIFFERENCE")
#endif
#include "video_core/nvn/record.h"
#include "video_core/nvn/sink.h"
#include "video_core/vulkan_common/vulkan_wrapper.h"

namespace Vulkan {
class Device;
} // namespace Vulkan

namespace Vulkan::NvnClean {

// ================================================================================================
// Architectural Overview & First-Principles Complexity (`nv_mempool.h`)
// ================================================================================================
//
// Legacy Maxwell/NVN buffer caches copy guest UBO, VBO, and IBO ranges through CPU-visible staging
// rings, emit `vkCmdCopyBuffer` commands into device-local shadow buffers, and insert
// `TRANSFER_WRITE -> UNIFORM_READ / VERTEX_ATTRIBUTE_READ` pipeline barriers before every pass.
//
// `GuestMemoryPool` and `MemoryPoolManager` eliminate the staging layer completely:
//
//   1. Direct `HOST_VISIBLE | DEVICE_LOCAL` Mapping (`O(0)` Memory Transfer Overhead):
//      - Physical NVN memory pools are backed by persistently mapped ReBAR / unified-memory
//        `VkBuffer`s (`VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`
//        with `VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`).
//      - Zero staging buffer allocations, zero `vkCmdCopyBuffer` transfers, and zero transfer-to-
//        graphics pipeline barriers: GPU shaders and fixed-function vertex fetchers read directly
//        from the mapped allocation (`O(0)` bytes copied per draw/frame).
//
//   2. Dual Zero-Copy Shader UBO Access Paths:
//      - Path A (`VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC`): Pre-allocated 64 KiB-windowed
//        descriptor sets cover each physical pool across ~4 GiB windows; per-draw UBO updates pass
//        only a 32-bit `dynamic_offset` (`O(1)` window arithmetic, zero `vkUpdateDescriptorSets`).
//      - Path B (`VK_KHR_buffer_device_address` / 64-bit `VkDeviceAddress`): `DeviceAddressAt`
//        (`O(1)`) and `ResolveDeviceAddress` (`O(log P)` over `P` sorted address runs, typically
//        `P <= 32` in L1 cache) yield raw 64-bit GPU pointers for bindless `buffer_reference` UBO
//        and SSBO dereferences in SPIR-V with zero descriptor binds.
// ================================================================================================

/// NVN hardware maximum uniform buffer binding range (64 KiB).
inline constexpr VkDeviceSize kNvnMaxUniformBufferRange = 65536;

/// NVN architectural uniform buffer base alignment (256 bytes).
inline constexpr VkDeviceSize kNvnUniformAlignment = 256;

/// Maximum dynamic uniform buffer slots bound in a single batched call.
inline constexpr u32 kMaxDynamicUniformSlots = 16;

/// Maximum vertex buffer stream bindings supported per draw call.
inline constexpr u32 kMaxVertexStreams = 32;

/// NVNmemoryPoolFlags bit definitions.
namespace PoolFlags {
inline constexpr u32 kCpuNoAccess = 1u << 0;
inline constexpr u32 kCpuUncached = 1u << 1;
inline constexpr u32 kCpuCached = 1u << 2;
inline constexpr u32 kGpuNoAccess = 1u << 3;
inline constexpr u32 kShaderCode = 1u << 6;
inline constexpr u32 kCompressible = 1u << 7;
inline constexpr u32 kPhysical = 1u << 8;
inline constexpr u32 kVirtual = 1u << 9;
} // namespace PoolFlags

/// Parameters for creating a guest memory pool in the zero-copy Vulkan pipeline.
struct PoolCreateInfo {
    u64 serial{};
    u64 guest_address{};
    u64 size{};
    u32 flags{};
    /// Optional initial guest memory snapshot written directly into the mapped BAR allocation.
    const void* initial_data{nullptr};
    /// When true, fails creation if no DEVICE_LOCAL | HOST_VISIBLE memory is available instead of
    /// falling back to pinned HOST_VISIBLE | HOST_COHERENT memory.
    bool strict_device_local{false};
};

/// Descriptor window covering up to ~4 GiB of a physical pool for 32-bit dynamic UBO offsets.
struct UniformDescriptorWindow {
    VkDeviceSize base_offset{};
    VkDescriptorPool owner_pool{VK_NULL_HANDLE};
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
};

/// Cached multi-binding dynamic UBO descriptor set covering a specific pool window.
struct MultiSlotUniformSet {
    u32 slot_count{};
    VkDeviceSize base_offset{};
    VkDescriptorPool owner_pool{VK_NULL_HANDLE};
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
};

/// Zero-copy Vulkan backing for a single NVN guest memory pool.
///
/// Physical pools are backed by a persistently mapped VMA buffer allocated with
/// `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` (ReBAR / unified
/// memory), padded by one uniform buffer window so `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC`
/// descriptors remain valid at every aligned offset up to the end of the guest pool without
/// `vkCmdCopyBuffer` staging or transfer barriers.
class GuestMemoryPool {
public:
    GuestMemoryPool() noexcept = default;
    ~GuestMemoryPool() noexcept = default;

    GuestMemoryPool(const GuestMemoryPool&) = delete;
    GuestMemoryPool& operator=(const GuestMemoryPool&) = delete;

    GuestMemoryPool(GuestMemoryPool&&) noexcept = default;
    GuestMemoryPool& operator=(GuestMemoryPool&&) noexcept = default;

    [[nodiscard]] u64 Serial() const noexcept {
        return serial_;
    }
    [[nodiscard]] u64 GuestAddress() const noexcept {
        return guest_address_;
    }
    [[nodiscard]] u64 Size() const noexcept {
        return logical_size_;
    }
    [[nodiscard]] VkDeviceSize AllocatedSize() const noexcept {
        return allocated_size_;
    }
    [[nodiscard]] u32 Flags() const noexcept {
        return flags_;
    }

    [[nodiscard]] bool IsVirtual() const noexcept {
        return (flags_ & PoolFlags::kVirtual) != 0;
    }
    [[nodiscard]] bool IsDeviceLocal() const noexcept {
        return (memory_flags_ & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    }
    [[nodiscard]] bool IsHostVisible() const noexcept {
        return (memory_flags_ & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }
    [[nodiscard]] bool IsHostCoherent() const noexcept {
        return (memory_flags_ & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    }
    [[nodiscard]] bool IsHostCached() const noexcept {
        return (memory_flags_ & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
    }

    [[nodiscard]] VkBuffer BufferHandle() const noexcept {
        return *buffer_;
    }
    [[nodiscard]] VkDeviceMemory DeviceMemory() const noexcept {
        return buffer_.Location().memory;
    }
    [[nodiscard]] VkDeviceSize MemoryOffset() const noexcept {
        return buffer_.Location().offset;
    }
    [[nodiscard]] VmaAllocation Allocation() const noexcept {
        return allocation_;
    }
    [[nodiscard]] VkMemoryPropertyFlags MemoryPropertyFlags() const noexcept {
        return memory_flags_;
    }
    [[nodiscard]] VkDeviceAddress DeviceAddress() const noexcept {
        return device_address_;
    }

    /// Returns the 64-bit GPU Buffer Device Address (`VkDeviceAddress`) at `offset` within the
    /// logical pool range `[0, Size())`, or `0` if unbacked / out of range.
    /// Complexity: O(1) arithmetic, zero descriptor or staging overhead.
    [[nodiscard]] VkDeviceAddress DeviceAddressAt(u64 offset = 0) const noexcept {
        return (device_address_ != 0 && offset < logical_size_) ? (device_address_ + offset) : 0;
    }

    /// Persistently mapped CPU view over the logical guest pool range [0, Size()).
    [[nodiscard]] std::span<u8> MappedSpan() noexcept {
        return mapped_logical_;
    }
    [[nodiscard]] std::span<const u8> MappedSpan() const noexcept {
        return mapped_logical_;
    }

    /// Returns a direct host pointer to `offset` inside the mapped pool, or nullptr if out of range.
    [[nodiscard]] u8* HostPointer(u64 offset = 0) noexcept {
        return offset < mapped_logical_.size() ? mapped_logical_.data() + static_cast<size_t>(offset)
                                               : nullptr;
    }
    [[nodiscard]] const u8* HostPointer(u64 offset = 0) const noexcept {
        return offset < mapped_logical_.size() ? mapped_logical_.data() + static_cast<size_t>(offset)
                                               : nullptr;
    }

    /// Resolves the pre-bound `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` set and 32-bit dynamic
    /// offset for `aligned_pool_offset`.
    [[nodiscard]] VkDescriptorSet UniformDescriptorSet(u64 aligned_pool_offset = 0,
                                                       u32* out_dynamic_offset = nullptr) const noexcept;

    /// Writes CPU bytes directly into the mapped pool without staging buffers.
    bool Write(u64 offset, const void* src, u64 size) noexcept;

    /// Reads CPU bytes directly from the mapped pool.
    bool Read(u64 offset, void* dst, u64 size) const noexcept;

    /// Flushes host writes when the underlying memory type is not HOST_COHERENT (no-op otherwise).
    void FlushRange(u64 offset = 0, u64 size = VK_WHOLE_SIZE) const noexcept;

    /// Invalidates host caches before CPU readback when not HOST_COHERENT (no-op otherwise).
    void InvalidateRange(u64 offset = 0, u64 size = VK_WHOLE_SIZE) const noexcept;

private:
    friend class MemoryPoolManager;

    struct VirtualRun {
        u64 begin{};
        u64 end{};
        u32 physical_slot{};
        u64 physical_offset{};
    };

    u64 serial_{};
    u64 guest_address_{};
    u64 logical_size_{};
    VkDeviceSize allocated_size_{};
    VkDeviceSize window_stride_{};
    u32 flags_{};
    VmaAllocator allocator_{VK_NULL_HANDLE};
    VmaAllocation allocation_{VK_NULL_HANDLE};
    VkMemoryPropertyFlags memory_flags_{};
    VkDeviceAddress device_address_{};
    vk::Buffer buffer_;
    std::span<u8> mapped_logical_{};
    std::vector<UniformDescriptorWindow> ubo_windows_;
    std::vector<MultiSlotUniformSet> multi_slot_sets_;
    std::vector<VirtualRun> virtual_runs_;
};

/// Resolved zero-copy buffer view inside a mapped guest pool.
struct BufferRangeView {
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceSize offset{};
    VkDeviceSize available_size{};
    VkDeviceAddress device_address{};
    u8* host_ptr{nullptr};
    u64 pool_serial{};

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return buffer != VK_NULL_HANDLE;
    }
};

/// Resolved dynamic uniform buffer binding ready for `vkCmdBindDescriptorSets`.
struct UniformBindingView {
    VkDescriptorSet descriptor_set{VK_NULL_HANDLE};
    u32 dynamic_offset{};
    /// Byte offset inside the bound uniform block when `raw_offset` is not aligned to
    /// `minUniformBufferOffsetAlignment` (0 for all standard 256-byte-aligned NVN bindings).
    u32 sub_block_offset{};
    VkBuffer buffer{VK_NULL_HANDLE};
    VkDeviceSize raw_offset{};
    u64 pool_serial{};

    [[nodiscard]] explicit constexpr operator bool() const noexcept {
        return descriptor_set != VK_NULL_HANDLE;
    }
};

/// Allocation-free per-command-buffer state tracker for eliding redundant buffer and dynamic UBO
/// bind commands across consecutive draw and dispatch calls.
struct DrawBindState {
    struct BoundUniform {
        VkDescriptorSet set{VK_NULL_HANDLE};
        u32 dynamic_offset{};
        bool valid{false};
    };

    struct BoundVertexStream {
        VkBuffer buffer{VK_NULL_HANDLE};
        VkDeviceSize offset{};
        VkDeviceSize size{};
        VkDeviceSize stride{};
        bool has_extended{false};
        bool valid{false};
    };

    struct BoundIndex {
        VkBuffer buffer{VK_NULL_HANDLE};
        VkDeviceSize offset{};
        VkIndexType index_type{VK_INDEX_TYPE_UINT32};
        bool valid{false};
    };

    VkPipelineLayout graphics_layout{VK_NULL_HANDLE};
    VkPipelineLayout compute_layout{VK_NULL_HANDLE};
    std::array<BoundUniform, kMaxDynamicUniformSlots> graphics_uniforms{};
    std::array<BoundUniform, kMaxDynamicUniformSlots> compute_uniforms{};
    std::array<BoundVertexStream, kMaxVertexStreams> vertex_streams{};
    BoundIndex index_buffer{};

    constexpr void Reset() noexcept {
        *this = DrawBindState{};
    }
};

/// Snapshot of zero-copy memory pool and binding telemetry counters.
struct PoolStats {
    u64 active_physical_pools{};
    u64 active_virtual_pools{};
    u64 device_local_bar_bytes{};
    u64 host_fallback_bytes{};
    u64 dynamic_ubo_binds{};
    u64 elided_ubo_binds{};
    u64 vertex_buffer_binds{};
    u64 elided_vertex_binds{};
    u64 index_buffer_binds{};
    u64 elided_index_binds{};
};

/// Foundational zero-copy memory pool and buffer binding manager for `Vulkan::NvnClean`.
///
/// Allocates NVN guest pools directly in `HOST_VISIBLE | DEVICE_LOCAL` memory (ReBAR / unified
/// memory), pre-binds `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` descriptors over each pool, and
/// services uniform, vertex, and index buffer bindings directly from the mapped pool with zero
/// `vkCmdCopyBuffer` commands or transfer pipeline barriers.
class MemoryPoolManager {
public:
    /// Constructs the pool manager. When `external_allocator` is `VK_NULL_HANDLE`, creates and owns
    /// a dedicated thread-safe VMA allocator over `device`.
    explicit MemoryPoolManager(const Device& device,
                               VmaAllocator external_allocator = VK_NULL_HANDLE);
    ~MemoryPoolManager();

    MemoryPoolManager(const MemoryPoolManager&) = delete;
    MemoryPoolManager& operator=(const MemoryPoolManager&) = delete;

    MemoryPoolManager(MemoryPoolManager&&) = delete;
    MemoryPoolManager& operator=(MemoryPoolManager&&) = delete;

    /// Allocates and maps a guest memory pool, pre-populating its dynamic UBO descriptor set(s).
    bool CreatePool(const PoolCreateInfo& info);

    /// Convenience overload accepting `VideoCommon::Nvn::PoolDesc`.
    bool CreatePool(const VideoCommon::Nvn::PoolDesc& desc, const void* initial_data = nullptr);

    /// Releases a guest memory pool and returns its dynamic UBO descriptor set(s).
    bool DestroyPool(u64 serial);

    /// Releases all registered pools and resets tracking tables.
    void Clear();

    /// Updates virtual-to-physical mappings for a virtual NVN memory pool (`PoolFlags::kVirtual`).
    bool MapVirtual(u64 virtual_pool_serial,
                    std::span<const VideoCommon::Nvn::VirtualMapping> mappings);

    /// Looks up a registered pool by its NVN serial.
    [[nodiscard]] GuestMemoryPool* FindPool(u64 serial) noexcept;
    [[nodiscard]] const GuestMemoryPool* FindPool(u64 serial) const noexcept;

    /// Copies bytes directly from `guest_mem` into the mapped BAR pool covering `[guest_address, +size)`.
    bool SyncFromGuest(u64 guest_address, u64 size, VideoCommon::Nvn::GuestMemory& guest_mem);

    /// Writes host bytes directly into the mapped pool covering `[guest_address, +size)`.
    bool WriteGuestBytes(u64 guest_address, const void* src, u64 size);

    /// Flushes or invalidates non-coherent host mappings covering `[guest_address, +size)`.
    void FlushGuestRange(u64 guest_address, u64 size) const noexcept;
    void InvalidateGuestRange(u64 guest_address, u64 size) const noexcept;

    /// Resolves a guest GPU address (`NVNbufferAddress`) into a zero-copy `VkBuffer` + offset view.
    /// Complexity: O(log P) binary search over `P` mapped pool address runs; O(0) copy overhead.
    [[nodiscard]] BufferRangeView ResolveBufferRange(u64 guest_address,
                                                     u64 size = 0) const noexcept;

    /// Resolves a guest GPU address (`NVNbufferAddress`) directly to a 64-bit `VkDeviceAddress`
    /// (`VK_KHR_buffer_device_address`) for zero-descriptor shader UBO/SSBO reads (`0` if unmapped
    /// or smaller than `size`).
    /// Complexity: O(log P) over `P` mapped pool address runs; O(0) transfer/descriptor overhead.
    [[nodiscard]] VkDeviceAddress ResolveDeviceAddress(u64 guest_address,
                                                       u64 size = 0) const noexcept {
        const BufferRangeView view = ResolveBufferRange(guest_address, size);
        return (view && (size == 0 || view.available_size >= size)) ? view.device_address : 0;
    }

    /// Resolves a guest GPU address into a `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` set + 32-bit
    /// dynamic offset. Returns the zero-filled fallback UBO set at offset 0 when `guest_address == 0`.
    /// Complexity: O(log P) address lookup + O(1) dynamic window selection.
    [[nodiscard]] UniformBindingView ResolveUniformBinding(u64 guest_address,
                                                           u64 size = 0) const noexcept;

    /// Shared single-binding `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC` descriptor set layout
    /// (`binding = 0`, `VK_SHADER_STAGE_ALL`).
    [[nodiscard]] VkDescriptorSetLayout UniformDescriptorSetLayout() const noexcept {
        return *single_ubo_layout_;
    }

    /// Zero-filled fallback dynamic UBO descriptor set (safe to bind at dynamic offset 0).
    [[nodiscard]] VkDescriptorSet NullUniformDescriptorSet() const noexcept {
        return null_ubo_set_;
    }

    /// Lazily creates or retrieves a descriptor set layout with `slot_count` dynamic UBO bindings
    /// (`bindings 0 .. slot_count - 1`, `VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC`).
    [[nodiscard]] VkDescriptorSetLayout GetMultiSlotUniformLayout(u32 slot_count);

    /// Lazily creates or retrieves a `VkDescriptorSet` where all `slot_count` dynamic UBO bindings
    /// point to the guest pool identified by `pool_serial` (at `window_base_offset`).
    [[nodiscard]] VkDescriptorSet GetPoolMultiSlotDescriptorSet(u64 pool_serial, u32 slot_count,
                                                                u64 window_base_offset = 0);

    /// Binds a single dynamic uniform buffer by guest address via `vkCmdBindDescriptorSets` with a
    /// 32-bit dynamic offset (zero buffer copies, zero barriers).
    bool BindUniformBuffer(vk::CommandBuffer cmd, VkPipelineBindPoint bind_point,
                           VkPipelineLayout pipeline_layout, u32 set_index, u64 guest_address,
                           DrawBindState* state = nullptr) const noexcept;

    /// Resolves and binds `ranges.size()` dynamic uniform buffers across contiguous descriptor sets
    /// `[first_set, first_set + ranges.size())` in a single `vkCmdBindDescriptorSets` call.
    bool BindUniformBuffers(vk::CommandBuffer cmd, VkPipelineBindPoint bind_point,
                            VkPipelineLayout pipeline_layout, u32 first_set,
                            std::span<const VideoCommon::Nvn::BufferRange> ranges,
                            DrawBindState* state = nullptr) const noexcept;

    /// Binds a multi-slot dynamic uniform descriptor set covering `pool_serial` and supplies the
    /// 32-bit `dynamic_offsets` array directly to `vkCmdBindDescriptorSets`.
    bool BindPoolUniformSlots(vk::CommandBuffer cmd, VkPipelineBindPoint bind_point,
                              VkPipelineLayout pipeline_layout, u32 set_index, u64 pool_serial,
                              std::span<const u32> dynamic_offsets, u64 window_base_offset = 0,
                              DrawBindState* state = nullptr);

    /// Binds a single vertex buffer directly from the mapped guest pool.
    bool BindVertexBuffer(vk::CommandBuffer cmd, u32 binding, u64 guest_address, u64 size = 0,
                          DrawBindState* state = nullptr) const noexcept;

    /// Binds `ranges.size()` vertex buffer streams directly from mapped guest pools in a single
    /// `vkCmdBindVertexBuffers` call. Empty (`address == 0`) slots bind a zeroed sentinel buffer.
    bool BindVertexBuffers(vk::CommandBuffer cmd, u32 first_binding,
                           std::span<const VideoCommon::Nvn::BufferRange> ranges,
                           DrawBindState* state = nullptr) const noexcept;

    /// Binds vertex buffer streams via `vkCmdBindVertexBuffers2EXT` when extended dynamic state is
    /// supported (passing exact clamped `NVNbufferRange` sizes and optional dynamic strides),
    /// falling back to `vkCmdBindVertexBuffers` otherwise.
    bool BindVertexBuffersExtended(vk::CommandBuffer cmd, u32 first_binding,
                                   std::span<const VideoCommon::Nvn::BufferRange> ranges,
                                   std::span<const VkDeviceSize> strides = {},
                                   DrawBindState* state = nullptr) const noexcept;

    /// Binds an index buffer directly from the mapped guest pool using an `NVNindexType`.
    bool BindIndexBuffer(vk::CommandBuffer cmd, u64 guest_address,
                         VideoCommon::Nvn::NVNindexType nvn_index_type,
                         DrawBindState* state = nullptr) const noexcept;

    /// Binds an index buffer directly from the mapped guest pool using a `VkIndexType`.
    bool BindIndexBuffer(vk::CommandBuffer cmd, u64 guest_address, VkIndexType vk_index_type,
                         DrawBindState* state = nullptr) const noexcept;

    [[nodiscard]] VkDeviceSize UniformOffsetAlignment() const noexcept {
        return min_ubo_alignment_;
    }
    [[nodiscard]] VkDeviceSize UniformDescriptorRange() const noexcept {
        return ubo_descriptor_range_;
    }
    [[nodiscard]] u32 MaxDynamicUniformBuffers() const noexcept {
        return max_dynamic_ubos_;
    }
    [[nodiscard]] bool HasDeviceLocalHostVisibleHeap() const noexcept {
        return has_rebar_heap_;
    }
    [[nodiscard]] PoolStats Stats() const noexcept;

private:
    struct AddressRangeEntry {
        u64 begin{};
        u64 end{};
        u32 slot{};
        bool is_virtual{false};
    };

    struct AllocatedBuffer {
        vk::Buffer buffer;
        VmaAllocation allocation{VK_NULL_HANDLE};
        VkMemoryPropertyFlags memory_flags{};
        VkDeviceAddress device_address{};
        VkDeviceSize allocated_size{};
    };

    [[nodiscard]] bool AllocateBarBuffer(VkDeviceSize logical_size, VkDeviceSize padded_size,
                                         u32 nvn_flags, bool strict_device_local,
                                         AllocatedBuffer& out) const;

    [[nodiscard]] bool AllocateUniformSet(VkDescriptorSetLayout layout, u32 descriptor_count,
                                          VkDescriptorPool& out_pool,
                                          VkDescriptorSet& out_set);

    void FreeUniformSet(VkDescriptorPool owner_pool, VkDescriptorSet set) noexcept;
    void RebuildAddressTableLocked();

    [[nodiscard]] const GuestMemoryPool* ResolvePhysicalPoolUnlocked(
        u64 guest_address, u64& out_pool_offset) const noexcept;

    const Device& device_;
    VmaAllocator owned_allocator_{VK_NULL_HANDLE};
    VmaAllocator allocator_{VK_NULL_HANDLE};

    VkDeviceSize min_ubo_alignment_{kNvnUniformAlignment};
    VkDeviceSize non_coherent_atom_size_{64};
    VkDeviceSize ubo_descriptor_range_{kNvnMaxUniformBufferRange};
    VkDeviceSize dynamic_window_stride_{};
    VkBufferUsageFlags pool_buffer_usage_{};
    u32 max_dynamic_ubos_{8};
    bool has_rebar_heap_{false};

    vk::DescriptorSetLayout single_ubo_layout_;
    std::array<vk::DescriptorSetLayout, kMaxDynamicUniformSlots + 1> multi_slot_layouts_{};
    std::vector<vk::DescriptorPool> descriptor_pools_;

    AllocatedBuffer sentinel_buffer_{};
    VkDescriptorPool null_ubo_pool_{VK_NULL_HANDLE};
    VkDescriptorSet null_ubo_set_{VK_NULL_HANDLE};

    mutable std::shared_mutex mutex_;
    std::deque<std::optional<GuestMemoryPool>> slots_;
    std::vector<u32> free_slots_;
    std::unordered_map<u64, u32> serial_to_slot_;
    std::vector<AddressRangeEntry> address_table_;

    std::atomic<u64> active_physical_pools_{0};
    std::atomic<u64> active_virtual_pools_{0};
    std::atomic<u64> device_local_bar_bytes_{0};
    std::atomic<u64> host_fallback_bytes_{0};
    mutable std::atomic<u64> dynamic_ubo_binds_{0};
    mutable std::atomic<u64> elided_ubo_binds_{0};
    mutable std::atomic<u64> vertex_buffer_binds_{0};
    mutable std::atomic<u64> elided_vertex_binds_{0};
    mutable std::atomic<u64> index_buffer_binds_{0};
    mutable std::atomic<u64> elided_index_binds_{0};
};

} // namespace Vulkan::NvnClean
