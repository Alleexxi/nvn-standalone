// SPDX-FileCopyrightText: Copyright 2026 Eglantine Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "nvn_clean/nv_mempool.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <utility>

#include "common/alignment.h"
#include "common/assert.h"
#include "common/logging.h"
#include "video_core/vulkan_common/vma.h"
#include "video_core/vulkan_common/vulkan_device.h"

namespace Vulkan::NvnClean {
namespace {

constexpr u32 kSetsPerDescriptorPoolChunk = 256;
constexpr u32 kDynamicUbosPerPoolChunk = 1024;
constexpr u64 kFourGiBWindowCap = 0xFFFF'0000ULL;

[[nodiscard]] constexpr u32 IndexTypeByteSize(VkIndexType index_type) noexcept {
    switch (index_type) {
    case VK_INDEX_TYPE_UINT8_EXT:
        return 1;
    case VK_INDEX_TYPE_UINT16:
        return 2;
    case VK_INDEX_TYPE_UINT32:
    default:
        return 4;
    }
}

[[nodiscard]] bool DetectRebarHeap(const Device& device) noexcept {
    const auto mem_props = device.GetPhysical().GetMemoryProperties().memoryProperties;
    constexpr VkMemoryPropertyFlags kRebarMask =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    for (u32 i = 0; i < mem_props.memoryTypeCount; ++i) {
        if ((mem_props.memoryTypes[i].propertyFlags & kRebarMask) == kRebarMask) {
            return true;
        }
    }
    return false;
}

} // namespace

// ============================================================================
// GuestMemoryPool
// ============================================================================

VkDescriptorSet GuestMemoryPool::UniformDescriptorSet(u64 aligned_pool_offset,
                                                      u32* out_dynamic_offset) const noexcept {
    if (ubo_windows_.empty() || aligned_pool_offset > logical_size_) {
        if (out_dynamic_offset != nullptr) {
            *out_dynamic_offset = 0;
        }
        return VK_NULL_HANDLE;
    }

    if (ubo_windows_.size() == 1 || window_stride_ == 0) {
        if (out_dynamic_offset != nullptr) {
            *out_dynamic_offset = static_cast<u32>(aligned_pool_offset);
        }
        return ubo_windows_[0].descriptor_set;
    }

    const size_t window_idx =
        std::min<size_t>(static_cast<size_t>(aligned_pool_offset / window_stride_),
                         ubo_windows_.size() - 1);
    const u64 local_offset = aligned_pool_offset - ubo_windows_[window_idx].base_offset;
    if (local_offset > (std::numeric_limits<u32>::max)()) {
        if (out_dynamic_offset != nullptr) {
            *out_dynamic_offset = 0;
        }
        return VK_NULL_HANDLE;
    }
    if (out_dynamic_offset != nullptr) {
        *out_dynamic_offset = static_cast<u32>(local_offset);
    }
    return ubo_windows_[window_idx].descriptor_set;
}

bool GuestMemoryPool::Write(u64 offset, const void* src, u64 size) noexcept {
    if (size == 0) {
        return offset <= logical_size_;
    }
    if (src == nullptr || offset > logical_size_ || size > logical_size_ - offset ||
        mapped_logical_.empty()) {
        return false;
    }
    std::memcpy(mapped_logical_.data() + static_cast<size_t>(offset), src,
                static_cast<size_t>(size));
    FlushRange(offset, size);
    return true;
}

bool GuestMemoryPool::Read(u64 offset, void* dst, u64 size) const noexcept {
    if (size == 0) {
        return offset <= logical_size_;
    }
    if (dst == nullptr || offset > logical_size_ || size > logical_size_ - offset ||
        mapped_logical_.empty()) {
        return false;
    }
    InvalidateRange(offset, size);
    std::memcpy(dst, mapped_logical_.data() + static_cast<size_t>(offset),
                static_cast<size_t>(size));
    return true;
}

void GuestMemoryPool::FlushRange(u64 offset, u64 size) const noexcept {
    if (IsHostCoherent() || allocation_ == VK_NULL_HANDLE || allocator_ == VK_NULL_HANDLE ||
        offset >= allocated_size_) {
        return;
    }
    const VkDeviceSize clamped_size =
        size == VK_WHOLE_SIZE ? VK_WHOLE_SIZE : (std::min)(size, allocated_size_ - offset);
    if (clamped_size == 0) {
        return;
    }
    vmaFlushAllocation(allocator_, allocation_, offset, clamped_size);
}

void GuestMemoryPool::InvalidateRange(u64 offset, u64 size) const noexcept {
    if (IsHostCoherent() || allocation_ == VK_NULL_HANDLE || allocator_ == VK_NULL_HANDLE ||
        offset >= allocated_size_) {
        return;
    }
    const VkDeviceSize clamped_size =
        size == VK_WHOLE_SIZE ? VK_WHOLE_SIZE : (std::min)(size, allocated_size_ - offset);
    if (clamped_size == 0) {
        return;
    }
    vmaInvalidateAllocation(allocator_, allocation_, offset, clamped_size);
}

// ============================================================================
// MemoryPoolManager
// ============================================================================

MemoryPoolManager::MemoryPoolManager(const Device& device, VmaAllocator external_allocator)
    : device_{device} {
    if (external_allocator != VK_NULL_HANDLE) {
        allocator_ = external_allocator;
    } else {
        owned_allocator_ = device_.CreateThreadSafeAllocator();
        allocator_ = owned_allocator_;
    }

    const VkPhysicalDeviceProperties props = device_.GetPhysical().GetProperties();
    min_ubo_alignment_ =
        (std::max)(VkDeviceSize{1}, props.limits.minUniformBufferOffsetAlignment);
    non_coherent_atom_size_ = (std::max)(VkDeviceSize{1}, props.limits.nonCoherentAtomSize);
    ubo_descriptor_range_ =
        (std::min)(kNvnMaxUniformBufferRange,
                   static_cast<VkDeviceSize>(props.limits.maxUniformBufferRange));

    const size_t stride_align =
        static_cast<size_t>((std::max)(min_ubo_alignment_, kNvnUniformAlignment));
    dynamic_window_stride_ = Common::AlignDown(kFourGiBWindowCap, stride_align);
    max_dynamic_ubos_ = props.limits.maxDescriptorSetUniformBuffersDynamic;
    has_rebar_heap_ = DetectRebarHeap(device_);

    pool_buffer_usage_ = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    if (device_.IsBufferDeviceAddressSupported()) {
        pool_buffer_usage_ |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    const VkDescriptorSetLayoutBinding single_binding{
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_ALL,
        .pImmutableSamplers = nullptr,
    };
    single_ubo_layout_ = device_.GetLogical().CreateDescriptorSetLayout(
        VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = 1,
            .pBindings = &single_binding,
        });

    const size_t alloc_align =
        static_cast<size_t>((std::max)(min_ubo_alignment_, non_coherent_atom_size_));
    const VkDeviceSize sentinel_size = Common::AlignUp(ubo_descriptor_range_, alloc_align);
    if (AllocateBarBuffer(0, sentinel_size, PoolFlags::kCpuUncached, false, sentinel_buffer_)) {
        sentinel_buffer_.buffer.SetObjectNameEXT("NvnClean::ZeroSentinelPool");
        if (AllocateUniformSet(*single_ubo_layout_, 1, null_ubo_pool_, null_ubo_set_)) {
            const VkDescriptorBufferInfo buffer_info{
                .buffer = *sentinel_buffer_.buffer,
                .offset = 0,
                .range = ubo_descriptor_range_,
            };
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = null_ubo_set_,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pImageInfo = nullptr,
                .pBufferInfo = &buffer_info,
                .pTexelBufferView = nullptr,
            };
            device_.GetLogical().UpdateDescriptorSets(write, {});
        }
    }
}

MemoryPoolManager::~MemoryPoolManager() {
    Clear();

    if (null_ubo_set_ != VK_NULL_HANDLE && null_ubo_pool_ != VK_NULL_HANDLE) {
        FreeUniformSet(null_ubo_pool_, null_ubo_set_);
        null_ubo_set_ = VK_NULL_HANDLE;
        null_ubo_pool_ = VK_NULL_HANDLE;
    }

    descriptor_pools_.clear();
    for (auto& layout : multi_slot_layouts_) {
        layout.reset();
    }
    single_ubo_layout_.reset();
    sentinel_buffer_.buffer.reset();
    sentinel_buffer_.allocation = VK_NULL_HANDLE;

    if (owned_allocator_ != VK_NULL_HANDLE) {
        vmaDestroyAllocator(owned_allocator_);
        owned_allocator_ = VK_NULL_HANDLE;
    }
}

bool MemoryPoolManager::AllocateBarBuffer(VkDeviceSize logical_size, VkDeviceSize padded_size,
                                          u32 nvn_flags, bool strict_device_local,
                                          AllocatedBuffer& out) const {
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .size = padded_size,
        .usage = pool_buffer_usage_,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
    };

    const VmaAllocationCreateFlags host_access_flag =
        (nvn_flags & PoolFlags::kCpuCached) != 0
            ? VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
            : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

    VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | host_access_flag |
                 VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
        .usage = VMA_MEMORY_USAGE_UNKNOWN,
        .requiredFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        .preferredFlags =
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
        .memoryTypeBits = 0,
        .pool = VK_NULL_HANDLE,
        .pUserData = nullptr,
        .priority = 1.0f,
    };

    VkBuffer handle{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VmaAllocationInfo alloc_info{};

    // Tier 1: Direct ReBAR / unified memory within heap budget.
    VkResult result =
        vmaCreateBuffer(allocator_, &buffer_ci, &alloc_ci, &handle, &allocation, &alloc_info);

    // Tier 2: Direct ReBAR / unified memory allowing driver over-budget paging.
    if (result != VK_SUCCESS) {
        VmaAllocationCreateInfo relaxed_bar = alloc_ci;
        relaxed_bar.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        result = vmaCreateBuffer(allocator_, &buffer_ci, &relaxed_bar, &handle, &allocation,
                                 &alloc_info);
    }

    // Tier 3: Pinned host-visible memory fallback when BAR is exhausted or disabled in BIOS.
    if (result != VK_SUCCESS && !strict_device_local) {
        VmaAllocationCreateInfo host_fallback = alloc_ci;
        host_fallback.flags &= ~VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
        host_fallback.requiredFlags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        host_fallback.preferredFlags =
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        result = vmaCreateBuffer(allocator_, &buffer_ci, &host_fallback, &handle, &allocation,
                                 &alloc_info);
        if (result == VK_SUCCESS) {
            static std::atomic<u32> fallback_log_count{0};
            if (fallback_log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
                LOG_WARNING(Render_Vulkan,
                            "[NvnClean] Guest pool ({} KiB) spilled from DEVICE_LOCAL|HOST_VISIBLE "
                            "BAR heap to host-visible memory",
                            padded_size >> 10);
            }
        }
    }

    if (result != VK_SUCCESS || handle == VK_NULL_HANDLE || alloc_info.pMappedData == nullptr) {
        if (handle != VK_NULL_HANDLE) {
            vmaDestroyBuffer(allocator_, handle, allocation);
        }
        LOG_ERROR(Render_Vulkan, "[NvnClean] Failed to allocate BAR buffer of {} bytes (VkResult={})",
                  padded_size, static_cast<int>(result));
        return false;
    }

    VkMemoryPropertyFlags prop_flags{};
    vmaGetAllocationMemoryProperties(allocator_, allocation, &prop_flags);
    const bool is_coherent = (prop_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;

    u8* const mapped_base = static_cast<u8*>(alloc_info.pMappedData);
    if (padded_size > logical_size) {
        std::memset(mapped_base + static_cast<size_t>(logical_size), 0,
                    static_cast<size_t>(padded_size - logical_size));
        if (!is_coherent) {
            vmaFlushAllocation(allocator_, allocation, logical_size, padded_size - logical_size);
        }
    }

    const vk::MemoryLocation location{
        .memory = alloc_info.deviceMemory,
        .offset = alloc_info.offset,
        .memory_type = alloc_info.memoryType,
    };
    const std::span<u8> full_mapped{mapped_base, static_cast<size_t>(padded_size)};

    VkDeviceAddress bda = 0;
    if (device_.IsBufferDeviceAddressSupported()) {
        bda = device_.GetLogical().GetBufferDeviceAddress(handle);
    }

    out.buffer = vk::Buffer(handle, *device_.GetLogical(), allocator_, allocation, full_mapped,
                            is_coherent, location, device_.GetDispatchLoader());
    out.allocation = allocation;
    out.memory_flags = prop_flags;
    out.device_address = bda;
    out.allocated_size = padded_size;
    return true;
}

bool MemoryPoolManager::AllocateUniformSet(VkDescriptorSetLayout layout, u32 descriptor_count,
                                           VkDescriptorPool& out_pool, VkDescriptorSet& out_set) {
    out_pool = VK_NULL_HANDLE;
    out_set = VK_NULL_HANDLE;
    if (layout == VK_NULL_HANDLE || descriptor_count == 0) {
        return false;
    }

    const auto& dld = device_.GetDispatchLoader();
    const VkDevice logical = *device_.GetLogical();

    for (auto it = descriptor_pools_.rbegin(); it != descriptor_pools_.rend(); ++it) {
        const VkDescriptorPool candidate = **it;
        const VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = candidate,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        VkDescriptorSet allocated{VK_NULL_HANDLE};
        const VkResult res = dld.vkAllocateDescriptorSets(logical, &alloc_info, &allocated);
        if (res == VK_SUCCESS) {
            out_pool = candidate;
            out_set = allocated;
            return true;
        }
        if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) {
            return false;
        }
    }

    const u32 pool_ubos = (std::max)(kDynamicUbosPerPoolChunk, descriptor_count * 16u);
    const VkDescriptorPoolSize pool_size{
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        .descriptorCount = pool_ubos,
    };
    vk::DescriptorPool new_pool = device_.GetLogical().CreateDescriptorPool(
        VkDescriptorPoolCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = nullptr,
            .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = kSetsPerDescriptorPoolChunk,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        });

    const VkDescriptorPool candidate = *new_pool;
    const VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = candidate,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    VkDescriptorSet allocated{VK_NULL_HANDLE};
    if (dld.vkAllocateDescriptorSets(logical, &alloc_info, &allocated) != VK_SUCCESS) {
        return false;
    }

    descriptor_pools_.push_back(std::move(new_pool));
    out_pool = candidate;
    out_set = allocated;
    return true;
}

void MemoryPoolManager::FreeUniformSet(VkDescriptorPool owner_pool,
                                       VkDescriptorSet set) noexcept {
    if (owner_pool != VK_NULL_HANDLE && set != VK_NULL_HANDLE) {
        device_.GetDispatchLoader().vkFreeDescriptorSets(*device_.GetLogical(), owner_pool, 1,
                                                         &set);
    }
}

void MemoryPoolManager::RebuildAddressTableLocked() {
    address_table_.clear();
    address_table_.reserve(serial_to_slot_.size());
    for (const auto& [serial, slot] : serial_to_slot_) {
        static_cast<void>(serial);
        const GuestMemoryPool& pool = *slots_[slot];
        if (pool.GuestAddress() == 0 || pool.Size() == 0) {
            continue;
        }
        address_table_.push_back(AddressRangeEntry{
            .begin = pool.GuestAddress(),
            .end = pool.GuestAddress() + pool.Size(),
            .slot = slot,
            .is_virtual = pool.IsVirtual(),
        });
    }
    std::sort(address_table_.begin(), address_table_.end(),
              [](const AddressRangeEntry& lhs, const AddressRangeEntry& rhs) {
                  if (lhs.begin != rhs.begin) {
                      return lhs.begin < rhs.begin;
                  }
                  return lhs.end < rhs.end;
              });
}

bool MemoryPoolManager::CreatePool(const VideoCommon::Nvn::PoolDesc& desc,
                                   const void* initial_data) {
    return CreatePool(PoolCreateInfo{
        .serial = desc.serial,
        .guest_address = desc.guest_address,
        .size = desc.size,
        .flags = desc.flags,
        .initial_data = initial_data,
        .strict_device_local = false,
    });
}

bool MemoryPoolManager::CreatePool(const PoolCreateInfo& info) {
    if (info.serial == 0 || info.size == 0) {
        return false;
    }
    if (info.guest_address != 0 &&
        info.size > (std::numeric_limits<u64>::max)() - info.guest_address) {
        return false;
    }
    if (info.size >
        (std::numeric_limits<VkDeviceSize>::max)() - ubo_descriptor_range_ - kNvnMaxUniformBufferRange) {
        return false;
    }

    const bool is_virtual = (info.flags & PoolFlags::kVirtual) != 0;

    // Perform VMA buffer allocation outside the exclusive table lock when creating physical pools.
    AllocatedBuffer allocated{};
    if (!is_virtual) {
        const size_t alloc_align =
            static_cast<size_t>((std::max)(min_ubo_alignment_, non_coherent_atom_size_));
        const VkDeviceSize padded_size =
            Common::AlignUp(info.size + ubo_descriptor_range_, alloc_align);
        if (!AllocateBarBuffer(info.size, padded_size, info.flags, info.strict_device_local,
                               allocated)) {
            return false;
        }
    }

    std::unique_lock lk{mutex_};
    if (serial_to_slot_.contains(info.serial)) {
        // Replace an existing pool registered under the same serial cleanly.
        lk.unlock();
        DestroyPool(info.serial);
        lk.lock();
    }

    GuestMemoryPool pool{};
    pool.serial_ = info.serial;
    pool.guest_address_ = info.guest_address;
    pool.logical_size_ = info.size;
    pool.window_stride_ = dynamic_window_stride_;
    pool.flags_ = info.flags;

    if (!is_virtual) {
        pool.allocator_ = allocator_;
        pool.allocation_ = allocated.allocation;
        pool.allocated_size_ = allocated.allocated_size;
        pool.memory_flags_ = allocated.memory_flags;
        pool.device_address_ = allocated.device_address;
        pool.buffer_ = std::move(allocated.buffer);
        pool.mapped_logical_ =
            std::span<u8>{pool.buffer_.Mapped().data(), static_cast<size_t>(info.size)};

        if (info.initial_data != nullptr) {
            std::memcpy(pool.mapped_logical_.data(), info.initial_data,
                        static_cast<size_t>(info.size));
            pool.FlushRange(0, info.size);
        }

        const size_t window_count = (std::max)(
            size_t{1}, static_cast<size_t>(Common::DivideUp(info.size, dynamic_window_stride_)));
        pool.ubo_windows_.reserve(window_count);

        for (size_t w = 0; w < window_count; ++w) {
            VkDescriptorPool owner_pool{VK_NULL_HANDLE};
            VkDescriptorSet ubo_set{VK_NULL_HANDLE};
            if (!AllocateUniformSet(*single_ubo_layout_, 1, owner_pool, ubo_set)) {
                for (const auto& created : pool.ubo_windows_) {
                    FreeUniformSet(created.owner_pool, created.descriptor_set);
                }
                return false;
            }

            const VkDeviceSize window_base = static_cast<VkDeviceSize>(w) * dynamic_window_stride_;
            const VkDescriptorBufferInfo buffer_info{
                .buffer = *pool.buffer_,
                .offset = window_base,
                .range = ubo_descriptor_range_,
            };
            const VkWriteDescriptorSet write{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = nullptr,
                .dstSet = ubo_set,
                .dstBinding = 0,
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
                .pImageInfo = nullptr,
                .pBufferInfo = &buffer_info,
                .pTexelBufferView = nullptr,
            };
            device_.GetLogical().UpdateDescriptorSets(write, {});

            pool.ubo_windows_.push_back(UniformDescriptorWindow{
                .base_offset = window_base,
                .owner_pool = owner_pool,
                .descriptor_set = ubo_set,
            });
        }

        active_physical_pools_.fetch_add(1, std::memory_order_relaxed);
        if (pool.IsDeviceLocal()) {
            device_local_bar_bytes_.fetch_add(pool.allocated_size_, std::memory_order_relaxed);
        } else {
            host_fallback_bytes_.fetch_add(pool.allocated_size_, std::memory_order_relaxed);
        }
    } else {
        active_virtual_pools_.fetch_add(1, std::memory_order_relaxed);
    }

    u32 slot_idx = 0;
    if (!free_slots_.empty()) {
        slot_idx = free_slots_.back();
        free_slots_.pop_back();
        slots_[slot_idx].emplace(std::move(pool));
    } else {
        slot_idx = static_cast<u32>(slots_.size());
        slots_.emplace_back(std::move(pool));
    }

    serial_to_slot_.emplace(info.serial, slot_idx);
    RebuildAddressTableLocked();
    return true;
}

bool MemoryPoolManager::DestroyPool(u64 serial) {
    std::unique_lock lk{mutex_};
    const auto it = serial_to_slot_.find(serial);
    if (it == serial_to_slot_.end()) {
        return false;
    }

    const u32 slot_idx = it->second;
    GuestMemoryPool& pool = *slots_[slot_idx];

    for (const auto& window : pool.ubo_windows_) {
        FreeUniformSet(window.owner_pool, window.descriptor_set);
    }
    for (const auto& multi : pool.multi_slot_sets_) {
        FreeUniformSet(multi.owner_pool, multi.descriptor_set);
    }

    if (pool.IsVirtual()) {
        active_virtual_pools_.fetch_sub(1, std::memory_order_relaxed);
    } else {
        active_physical_pools_.fetch_sub(1, std::memory_order_relaxed);
        if (pool.IsDeviceLocal()) {
            device_local_bar_bytes_.fetch_sub(pool.allocated_size_, std::memory_order_relaxed);
        } else {
            host_fallback_bytes_.fetch_sub(pool.allocated_size_, std::memory_order_relaxed);
        }
        // Scrub virtual pool runs referencing the destroyed physical slot.
        for (auto& candidate_opt : slots_) {
            if (!candidate_opt.has_value() || !candidate_opt->IsVirtual()) {
                continue;
            }
            auto& runs = candidate_opt->virtual_runs_;
            std::erase_if(runs, [slot_idx](const GuestMemoryPool::VirtualRun& run) {
                return run.physical_slot == slot_idx;
            });
        }
    }

    slots_[slot_idx].reset();
    free_slots_.push_back(slot_idx);
    serial_to_slot_.erase(it);
    RebuildAddressTableLocked();
    return true;
}

void MemoryPoolManager::Clear() {
    std::unique_lock lk{mutex_};
    for (auto& slot_opt : slots_) {
        if (!slot_opt.has_value()) {
            continue;
        }
        for (const auto& window : slot_opt->ubo_windows_) {
            FreeUniformSet(window.owner_pool, window.descriptor_set);
        }
        for (const auto& multi : slot_opt->multi_slot_sets_) {
            FreeUniformSet(multi.owner_pool, multi.descriptor_set);
        }
        slot_opt.reset();
    }
    slots_.clear();
    free_slots_.clear();
    serial_to_slot_.clear();
    address_table_.clear();

    active_physical_pools_.store(0, std::memory_order_relaxed);
    active_virtual_pools_.store(0, std::memory_order_relaxed);
    device_local_bar_bytes_.store(0, std::memory_order_relaxed);
    host_fallback_bytes_.store(0, std::memory_order_relaxed);
}

bool MemoryPoolManager::MapVirtual(u64 virtual_pool_serial,
                                   std::span<const VideoCommon::Nvn::VirtualMapping> mappings) {
    std::unique_lock lk{mutex_};
    const auto virt_it = serial_to_slot_.find(virtual_pool_serial);
    if (virt_it == serial_to_slot_.end()) {
        return false;
    }

    GuestMemoryPool& virt_pool = *slots_[virt_it->second];
    if (!virt_pool.IsVirtual()) {
        return false;
    }

    auto& runs = virt_pool.virtual_runs_;
    std::vector<GuestMemoryPool::VirtualRun> updated;
    updated.reserve(runs.size() + 4);

    for (const auto& req : mappings) {
        if (req.size == 0 || req.virtual_offset >= virt_pool.Size()) {
            continue;
        }
        const u64 map_begin = req.virtual_offset;
        const u64 map_end = map_begin + (std::min)(req.size, virt_pool.Size() - map_begin);

        updated.clear();
        for (const auto& existing : runs) {
            if (existing.end <= map_begin || existing.begin >= map_end) {
                updated.push_back(existing);
                continue;
            }
            if (existing.begin < map_begin) {
                updated.push_back(GuestMemoryPool::VirtualRun{
                    .begin = existing.begin,
                    .end = map_begin,
                    .physical_slot = existing.physical_slot,
                    .physical_offset = existing.physical_offset,
                });
            }
            if (existing.end > map_end) {
                updated.push_back(GuestMemoryPool::VirtualRun{
                    .begin = map_end,
                    .end = existing.end,
                    .physical_slot = existing.physical_slot,
                    .physical_offset = existing.physical_offset + (map_end - existing.begin),
                });
            }
        }

        if (req.physical_pool_serial != 0) {
            const auto phys_it = serial_to_slot_.find(req.physical_pool_serial);
            if (phys_it != serial_to_slot_.end()) {
                const GuestMemoryPool& phys_pool = *slots_[phys_it->second];
                if (!phys_pool.IsVirtual() && req.physical_offset < phys_pool.Size()) {
                    const u64 valid_len =
                        (std::min)(map_end - map_begin, phys_pool.Size() - req.physical_offset);
                    if (valid_len > 0) {
                        const GuestMemoryPool::VirtualRun inserted{
                            .begin = map_begin,
                            .end = map_begin + valid_len,
                            .physical_slot = phys_it->second,
                            .physical_offset = req.physical_offset,
                        };
                        const auto pos = std::lower_bound(
                            updated.begin(), updated.end(), inserted.begin,
                            [](const GuestMemoryPool::VirtualRun& run, u64 offset) {
                                return run.begin < offset;
                            });
                        updated.insert(pos, inserted);
                    }
                }
            }
        }

        // Coalesce adjacent contiguous runs targeting the same physical pool.
        runs.clear();
        for (const auto& piece : updated) {
            if (!runs.empty()) {
                auto& prev = runs.back();
                if (prev.end == piece.begin && prev.physical_slot == piece.physical_slot &&
                    prev.physical_offset + (prev.end - prev.begin) == piece.physical_offset) {
                    prev.end = piece.end;
                    continue;
                }
            }
            runs.push_back(piece);
        }
    }
    return true;
}

GuestMemoryPool* MemoryPoolManager::FindPool(u64 serial) noexcept {
    std::shared_lock lk{mutex_};
    const auto it = serial_to_slot_.find(serial);
    if (it == serial_to_slot_.end()) {
        return nullptr;
    }
    return std::addressof(*slots_[it->second]);
}

const GuestMemoryPool* MemoryPoolManager::FindPool(u64 serial) const noexcept {
    std::shared_lock lk{mutex_};
    const auto it = serial_to_slot_.find(serial);
    if (it == serial_to_slot_.end()) {
        return nullptr;
    }
    return std::addressof(*slots_[it->second]);
}

const GuestMemoryPool* MemoryPoolManager::ResolvePhysicalPoolUnlocked(
    u64 guest_address, u64& out_pool_offset) const noexcept {
    out_pool_offset = 0;
    if (guest_address == 0 || address_table_.empty()) {
        return nullptr;
    }

    // Upper-bound search over the flat sorted range array (L1-resident for typical 8..64 pools).
    auto it = std::upper_bound(address_table_.begin(), address_table_.end(), guest_address,
                               [](u64 addr, const AddressRangeEntry& entry) {
                                   return addr < entry.begin;
                               });
    if (it == address_table_.begin()) {
        return nullptr;
    }
    --it;
    if (guest_address < it->begin || guest_address >= it->end) {
        return nullptr;
    }

    const u64 offset_in_entry = guest_address - it->begin;
    const GuestMemoryPool& matched = *slots_[it->slot];
    if (!it->is_virtual) {
        out_pool_offset = offset_in_entry;
        return &matched;
    }

    // Translate virtual pool offset through its flat sorted VirtualRun table.
    const auto& runs = matched.virtual_runs_;
    if (runs.empty()) {
        return nullptr;
    }
    auto run_it = std::upper_bound(runs.begin(), runs.end(), offset_in_entry,
                                   [](u64 virt_off, const GuestMemoryPool::VirtualRun& run) {
                                       return virt_off < run.begin;
                                   });
    if (run_it == runs.begin()) {
        return nullptr;
    }
    --run_it;
    if (offset_in_entry < run_it->begin || offset_in_entry >= run_it->end) {
        return nullptr;
    }

    const auto& phys_opt = slots_[run_it->physical_slot];
    if (!phys_opt.has_value() || phys_opt->IsVirtual()) {
        return nullptr;
    }
    const u64 phys_offset = run_it->physical_offset + (offset_in_entry - run_it->begin);
    if (phys_offset >= phys_opt->Size()) {
        return nullptr;
    }
    out_pool_offset = phys_offset;
    return std::addressof(*phys_opt);
}

bool MemoryPoolManager::SyncFromGuest(u64 guest_address, u64 size,
                                      VideoCommon::Nvn::GuestMemory& guest_mem) {
    if (size == 0) {
        return true;
    }
    std::shared_lock lk{mutex_};
    u64 pool_offset = 0;
    const GuestMemoryPool* const const_pool =
        ResolvePhysicalPoolUnlocked(guest_address, pool_offset);
    if (const_pool == nullptr || size > const_pool->Size() - pool_offset) {
        return false;
    }

    u8* const dst = const_cast<u8*>(const_pool->HostPointer(pool_offset));
    if (dst == nullptr) {
        return false;
    }

    // Fast path when guest virtual range is backed by a single contiguous host mapping.
    const u8* const first_ptr = guest_mem.Pointer(guest_address);
    if (first_ptr != nullptr &&
        (size == 1 || guest_mem.Pointer(guest_address + size - 1) == first_ptr + (size - 1))) {
        std::memcpy(dst, first_ptr, static_cast<size_t>(size));
    } else if (!guest_mem.Read(guest_address, dst, static_cast<size_t>(size))) {
        return false;
    }

    const_pool->FlushRange(pool_offset, size);
    return true;
}

bool MemoryPoolManager::WriteGuestBytes(u64 guest_address, const void* src, u64 size) {
    if (size == 0) {
        return true;
    }
    if (src == nullptr) {
        return false;
    }
    std::shared_lock lk{mutex_};
    u64 pool_offset = 0;
    const GuestMemoryPool* const const_pool =
        ResolvePhysicalPoolUnlocked(guest_address, pool_offset);
    if (const_pool == nullptr || size > const_pool->Size() - pool_offset) {
        return false;
    }
    u8* const dst = const_cast<u8*>(const_pool->HostPointer(pool_offset));
    if (dst == nullptr) {
        return false;
    }
    std::memcpy(dst, src, static_cast<size_t>(size));
    const_pool->FlushRange(pool_offset, size);
    return true;
}

void MemoryPoolManager::FlushGuestRange(u64 guest_address, u64 size) const noexcept {
    if (size == 0) {
        return;
    }
    std::shared_lock lk{mutex_};
    u64 pool_offset = 0;
    const GuestMemoryPool* const pool = ResolvePhysicalPoolUnlocked(guest_address, pool_offset);
    if (pool != nullptr) {
        pool->FlushRange(pool_offset, size);
    }
}

void MemoryPoolManager::InvalidateGuestRange(u64 guest_address, u64 size) const noexcept {
    if (size == 0) {
        return;
    }
    std::shared_lock lk{mutex_};
    u64 pool_offset = 0;
    const GuestMemoryPool* const pool = ResolvePhysicalPoolUnlocked(guest_address, pool_offset);
    if (pool != nullptr) {
        pool->InvalidateRange(pool_offset, size);
    }
}

BufferRangeView MemoryPoolManager::ResolveBufferRange(u64 guest_address, u64 size) const noexcept {
    std::shared_lock lk{mutex_};
    u64 pool_offset = 0;
    const GuestMemoryPool* const pool = ResolvePhysicalPoolUnlocked(guest_address, pool_offset);
    if (pool == nullptr) {
        return BufferRangeView{};
    }

    const u64 remaining = pool->Size() - pool_offset;
    if (size > remaining) {
        return BufferRangeView{};
    }

    return BufferRangeView{
        .buffer = pool->BufferHandle(),
        .offset = pool_offset,
        .available_size = remaining,
        .device_address = pool->DeviceAddress() != 0 ? pool->DeviceAddress() + pool_offset : 0,
        .host_ptr = const_cast<u8*>(pool->HostPointer(pool_offset)),
        .pool_serial = pool->Serial(),
    };
}

UniformBindingView MemoryPoolManager::ResolveUniformBinding(u64 guest_address,
                                                            u64 size) const noexcept {
    if (guest_address == 0) {
        return UniformBindingView{
            .descriptor_set = null_ubo_set_,
            .dynamic_offset = 0,
            .sub_block_offset = 0,
            .buffer = *sentinel_buffer_.buffer,
            .raw_offset = 0,
            .pool_serial = 0,
        };
    }

    std::shared_lock lk{mutex_};
    u64 pool_offset = 0;
    const GuestMemoryPool* const pool = ResolvePhysicalPoolUnlocked(guest_address, pool_offset);
    if (pool == nullptr) {
        return UniformBindingView{};
    }
    if (size > pool->Size() - pool_offset && size > ubo_descriptor_range_) {
        return UniformBindingView{};
    }

    const u64 aligned_offset =
        Common::AlignDown(pool_offset, static_cast<size_t>(min_ubo_alignment_));
    u32 dynamic_offset = 0;
    const VkDescriptorSet set = pool->UniformDescriptorSet(aligned_offset, &dynamic_offset);
    if (set == VK_NULL_HANDLE) {
        return UniformBindingView{};
    }

    return UniformBindingView{
        .descriptor_set = set,
        .dynamic_offset = dynamic_offset,
        .sub_block_offset = static_cast<u32>(pool_offset - aligned_offset),
        .buffer = pool->BufferHandle(),
        .raw_offset = pool_offset,
        .pool_serial = pool->Serial(),
    };
}

VkDescriptorSetLayout MemoryPoolManager::GetMultiSlotUniformLayout(u32 slot_count) {
    if (slot_count == 0 || slot_count > kMaxDynamicUniformSlots || slot_count > max_dynamic_ubos_) {
        return VK_NULL_HANDLE;
    }
    if (slot_count == 1) {
        return *single_ubo_layout_;
    }

    {
        std::shared_lock read_lk{mutex_};
        if (multi_slot_layouts_[slot_count]) {
            return *multi_slot_layouts_[slot_count];
        }
    }

    std::unique_lock write_lk{mutex_};
    if (multi_slot_layouts_[slot_count]) {
        return *multi_slot_layouts_[slot_count];
    }

    std::array<VkDescriptorSetLayoutBinding, kMaxDynamicUniformSlots> bindings{};
    for (u32 i = 0; i < slot_count; ++i) {
        bindings[i] = VkDescriptorSetLayoutBinding{
            .binding = i,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_ALL,
            .pImmutableSamplers = nullptr,
        };
    }
    multi_slot_layouts_[slot_count] = device_.GetLogical().CreateDescriptorSetLayout(
        VkDescriptorSetLayoutCreateInfo{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .bindingCount = slot_count,
            .pBindings = bindings.data(),
        });
    return *multi_slot_layouts_[slot_count];
}

VkDescriptorSet MemoryPoolManager::GetPoolMultiSlotDescriptorSet(u64 pool_serial, u32 slot_count,
                                                                 u64 window_base_offset) {
    if (slot_count == 0 || slot_count > kMaxDynamicUniformSlots || slot_count > max_dynamic_ubos_) {
        return VK_NULL_HANDLE;
    }

    const VkDescriptorSetLayout layout = GetMultiSlotUniformLayout(slot_count);
    if (layout == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }

    {
        std::shared_lock read_lk{mutex_};
        const auto it = serial_to_slot_.find(pool_serial);
        if (it == serial_to_slot_.end()) {
            return VK_NULL_HANDLE;
        }
        const GuestMemoryPool& pool = *slots_[it->second];
        if (pool.IsVirtual() || window_base_offset >= pool.Size()) {
            return VK_NULL_HANDLE;
        }
        if (slot_count == 1 && window_base_offset == 0 && !pool.ubo_windows_.empty()) {
            return pool.ubo_windows_[0].descriptor_set;
        }
        for (const auto& cached : pool.multi_slot_sets_) {
            if (cached.slot_count == slot_count && cached.base_offset == window_base_offset) {
                return cached.descriptor_set;
            }
        }
    }

    std::unique_lock write_lk{mutex_};
    const auto it = serial_to_slot_.find(pool_serial);
    if (it == serial_to_slot_.end()) {
        return VK_NULL_HANDLE;
    }
    GuestMemoryPool& pool = *slots_[it->second];
    if (pool.IsVirtual() || window_base_offset >= pool.Size() ||
        !Common::IsAligned(window_base_offset, static_cast<size_t>(min_ubo_alignment_))) {
        return VK_NULL_HANDLE;
    }
    for (const auto& cached : pool.multi_slot_sets_) {
        if (cached.slot_count == slot_count && cached.base_offset == window_base_offset) {
            return cached.descriptor_set;
        }
    }

    VkDescriptorPool owner_pool{VK_NULL_HANDLE};
    VkDescriptorSet set{VK_NULL_HANDLE};
    if (!AllocateUniformSet(layout, slot_count, owner_pool, set)) {
        return VK_NULL_HANDLE;
    }

    const VkDescriptorBufferInfo buffer_info{
        .buffer = *pool.buffer_,
        .offset = window_base_offset,
        .range = ubo_descriptor_range_,
    };
    std::array<VkWriteDescriptorSet, kMaxDynamicUniformSlots> writes{};
    for (u32 i = 0; i < slot_count; ++i) {
        writes[i] = VkWriteDescriptorSet{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = set,
            .dstBinding = i,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_info,
            .pTexelBufferView = nullptr,
        };
    }
    device_.GetLogical().UpdateDescriptorSets(
        vk::Span<VkWriteDescriptorSet>{writes.data(), slot_count}, {});

    pool.multi_slot_sets_.push_back(MultiSlotUniformSet{
        .slot_count = slot_count,
        .base_offset = window_base_offset,
        .owner_pool = owner_pool,
        .descriptor_set = set,
    });
    return set;
}

bool MemoryPoolManager::BindUniformBuffer(vk::CommandBuffer cmd, VkPipelineBindPoint bind_point,
                                          VkPipelineLayout pipeline_layout, u32 set_index,
                                          u64 guest_address, DrawBindState* state) const noexcept {
    if (pipeline_layout == VK_NULL_HANDLE) {
        return false;
    }

    const UniformBindingView view = ResolveUniformBinding(guest_address);
    if (!view) {
        return false;
    }

    if (state != nullptr && set_index < kMaxDynamicUniformSlots) {
        const bool is_compute = bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
        VkPipelineLayout& tracked_layout =
            is_compute ? state->compute_layout : state->graphics_layout;
        auto& tracked_sets = is_compute ? state->compute_uniforms : state->graphics_uniforms;

        if (tracked_layout != pipeline_layout) {
            tracked_layout = pipeline_layout;
            tracked_sets = {};
        }

        auto& slot = tracked_sets[set_index];
        if (slot.valid && slot.set == view.descriptor_set &&
            slot.dynamic_offset == view.dynamic_offset) {
            elided_ubo_binds_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        slot = DrawBindState::BoundUniform{
            .set = view.descriptor_set,
            .dynamic_offset = view.dynamic_offset,
            .valid = true,
        };
    }

    cmd.BindDescriptorSets(bind_point, pipeline_layout, set_index,
                           vk::Span<VkDescriptorSet>{&view.descriptor_set, 1},
                           vk::Span<u32>{&view.dynamic_offset, 1});
    dynamic_ubo_binds_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool MemoryPoolManager::BindUniformBuffers(
    vk::CommandBuffer cmd, VkPipelineBindPoint bind_point, VkPipelineLayout pipeline_layout,
    u32 first_set, std::span<const VideoCommon::Nvn::BufferRange> ranges,
    DrawBindState* state) const noexcept {
    if (ranges.empty()) {
        return true;
    }
    if (pipeline_layout == VK_NULL_HANDLE || ranges.size() > kMaxDynamicUniformSlots ||
        ranges.size() > max_dynamic_ubos_) {
        return false;
    }

    std::array<VkDescriptorSet, kMaxDynamicUniformSlots> sets{};
    std::array<u32, kMaxDynamicUniformSlots> offsets{};
    const size_t count = ranges.size();

    {
        std::shared_lock lk{mutex_};
        for (size_t i = 0; i < count; ++i) {
            const u64 addr = ranges[i].address;
            if (addr == 0) {
                if (null_ubo_set_ == VK_NULL_HANDLE) {
                    return false;
                }
                sets[i] = null_ubo_set_;
                offsets[i] = 0;
                continue;
            }

            u64 pool_offset = 0;
            const GuestMemoryPool* const pool = ResolvePhysicalPoolUnlocked(addr, pool_offset);
            if (pool == nullptr) {
                return false;
            }
            const u64 aligned_offset =
                Common::AlignDown(pool_offset, static_cast<size_t>(min_ubo_alignment_));
            u32 dyn_offset = 0;
            const VkDescriptorSet ds = pool->UniformDescriptorSet(aligned_offset, &dyn_offset);
            if (ds == VK_NULL_HANDLE) {
                return false;
            }
            sets[i] = ds;
            offsets[i] = dyn_offset;
        }
    }

    if (state != nullptr && first_set + count <= kMaxDynamicUniformSlots) {
        const bool is_compute = bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
        VkPipelineLayout& tracked_layout =
            is_compute ? state->compute_layout : state->graphics_layout;
        auto& tracked_sets = is_compute ? state->compute_uniforms : state->graphics_uniforms;

        if (tracked_layout != pipeline_layout) {
            tracked_layout = pipeline_layout;
            tracked_sets = {};
        }

        bool all_identical = true;
        for (size_t i = 0; i < count; ++i) {
            const auto& slot = tracked_sets[first_set + i];
            if (!slot.valid || slot.set != sets[i] || slot.dynamic_offset != offsets[i]) {
                all_identical = false;
                break;
            }
        }
        if (all_identical) {
            elided_ubo_binds_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        for (size_t i = 0; i < count; ++i) {
            tracked_sets[first_set + i] = DrawBindState::BoundUniform{
                .set = sets[i],
                .dynamic_offset = offsets[i],
                .valid = true,
            };
        }
    }

    cmd.BindDescriptorSets(bind_point, pipeline_layout, first_set,
                           vk::Span<VkDescriptorSet>{sets.data(), count},
                           vk::Span<u32>{offsets.data(), count});
    dynamic_ubo_binds_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool MemoryPoolManager::BindPoolUniformSlots(vk::CommandBuffer cmd, VkPipelineBindPoint bind_point,
                                             VkPipelineLayout pipeline_layout, u32 set_index,
                                             u64 pool_serial,
                                             std::span<const u32> dynamic_offsets,
                                             u64 window_base_offset, DrawBindState* state) {
    if (pipeline_layout == VK_NULL_HANDLE || dynamic_offsets.empty() ||
        dynamic_offsets.size() > kMaxDynamicUniformSlots ||
        dynamic_offsets.size() > max_dynamic_ubos_) {
        return false;
    }

    const u32 slot_count = static_cast<u32>(dynamic_offsets.size());
    u64 pool_logical_size = 0;
    {
        std::shared_lock lk{mutex_};
        const auto it = serial_to_slot_.find(pool_serial);
        if (it == serial_to_slot_.end() || slots_[it->second]->IsVirtual()) {
            return false;
        }
        pool_logical_size = slots_[it->second]->Size();
    }

    for (const u32 offset : dynamic_offsets) {
        if (!Common::IsAligned(offset, static_cast<size_t>(min_ubo_alignment_)) ||
            window_base_offset > pool_logical_size ||
            offset > pool_logical_size - window_base_offset) {
            return false;
        }
    }

    const VkDescriptorSet set =
        GetPoolMultiSlotDescriptorSet(pool_serial, slot_count, window_base_offset);
    if (set == VK_NULL_HANDLE) {
        return false;
    }

    if (state != nullptr && set_index < kMaxDynamicUniformSlots) {
        const bool is_compute = bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
        auto& tracked_sets = is_compute ? state->compute_uniforms : state->graphics_uniforms;
        tracked_sets[set_index] = DrawBindState::BoundUniform{};
    }

    cmd.BindDescriptorSets(bind_point, pipeline_layout, set_index,
                           vk::Span<VkDescriptorSet>{&set, 1},
                           vk::Span<u32>{dynamic_offsets.data(), dynamic_offsets.size()});
    dynamic_ubo_binds_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool MemoryPoolManager::BindVertexBuffer(vk::CommandBuffer cmd, u32 binding, u64 guest_address,
                                         u64 size, DrawBindState* state) const noexcept {
    const VideoCommon::Nvn::BufferRange single{
        .address = guest_address,
        .size = size,
    };
    return BindVertexBuffers(cmd, binding, std::span{&single, 1}, state);
}

bool MemoryPoolManager::BindVertexBuffers(vk::CommandBuffer cmd, u32 first_binding,
                                          std::span<const VideoCommon::Nvn::BufferRange> ranges,
                                          DrawBindState* state) const noexcept {
    if (ranges.empty()) {
        return true;
    }
    if (first_binding >= kMaxVertexStreams || ranges.size() > kMaxVertexStreams - first_binding) {
        return false;
    }

    std::array<VkBuffer, kMaxVertexStreams> buffers{};
    std::array<VkDeviceSize, kMaxVertexStreams> offsets{};
    const size_t count = ranges.size();

    {
        std::shared_lock lk{mutex_};
        for (size_t i = 0; i < count; ++i) {
            const u64 addr = ranges[i].address;
            if (addr == 0) {
                if (!sentinel_buffer_.buffer) {
                    return false;
                }
                buffers[i] = *sentinel_buffer_.buffer;
                offsets[i] = 0;
                continue;
            }

            u64 pool_offset = 0;
            const GuestMemoryPool* const pool = ResolvePhysicalPoolUnlocked(addr, pool_offset);
            if (pool == nullptr || ranges[i].size > pool->Size() - pool_offset) {
                return false;
            }
            buffers[i] = pool->BufferHandle();
            offsets[i] = pool_offset;
        }
    }

    if (state != nullptr) {
        bool all_identical = true;
        for (size_t i = 0; i < count; ++i) {
            const auto& slot = state->vertex_streams[first_binding + i];
            if (!slot.valid || slot.has_extended || slot.buffer != buffers[i] ||
                slot.offset != offsets[i]) {
                all_identical = false;
                break;
            }
        }
        if (all_identical) {
            elided_vertex_binds_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        for (size_t i = 0; i < count; ++i) {
            state->vertex_streams[first_binding + i] = DrawBindState::BoundVertexStream{
                .buffer = buffers[i],
                .offset = offsets[i],
                .size = 0,
                .stride = 0,
                .has_extended = false,
                .valid = true,
            };
        }
    }

    cmd.BindVertexBuffers(first_binding, static_cast<u32>(count), buffers.data(), offsets.data());
    vertex_buffer_binds_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool MemoryPoolManager::BindVertexBuffersExtended(
    vk::CommandBuffer cmd, u32 first_binding,
    std::span<const VideoCommon::Nvn::BufferRange> ranges, std::span<const VkDeviceSize> strides,
    DrawBindState* state) const noexcept {
    if (ranges.empty()) {
        return true;
    }
    if (!device_.IsExtExtendedDynamicStateSupported()) {
        return BindVertexBuffers(cmd, first_binding, ranges, state);
    }
    if (first_binding >= kMaxVertexStreams || ranges.size() > kMaxVertexStreams - first_binding) {
        return false;
    }
    if (!strides.empty() && strides.size() != ranges.size()) {
        return false;
    }

    std::array<VkBuffer, kMaxVertexStreams> buffers{};
    std::array<VkDeviceSize, kMaxVertexStreams> offsets{};
    std::array<VkDeviceSize, kMaxVertexStreams> sizes{};
    const size_t count = ranges.size();

    {
        std::shared_lock lk{mutex_};
        for (size_t i = 0; i < count; ++i) {
            const u64 addr = ranges[i].address;
            if (addr == 0 || ranges[i].size == 0) {
                if (!sentinel_buffer_.buffer) {
                    return false;
                }
                buffers[i] = *sentinel_buffer_.buffer;
                offsets[i] = 0;
                sizes[i] = 0;
                continue;
            }

            u64 pool_offset = 0;
            const GuestMemoryPool* const pool = ResolvePhysicalPoolUnlocked(addr, pool_offset);
            if (pool == nullptr) {
                return false;
            }
            const u64 available = pool->Size() - pool_offset;
            buffers[i] = pool->BufferHandle();
            offsets[i] = pool_offset;
            sizes[i] = (std::min)(ranges[i].size, available);
        }
    }

    const VkDeviceSize* const strides_ptr = strides.empty() ? nullptr : strides.data();

    if (state != nullptr) {
        bool all_identical = true;
        for (size_t i = 0; i < count; ++i) {
            const VkDeviceSize req_stride = strides_ptr != nullptr ? strides_ptr[i] : 0;
            const auto& slot = state->vertex_streams[first_binding + i];
            if (!slot.valid || !slot.has_extended || slot.buffer != buffers[i] ||
                slot.offset != offsets[i] || slot.size != sizes[i] || slot.stride != req_stride) {
                all_identical = false;
                break;
            }
        }
        if (all_identical) {
            elided_vertex_binds_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        for (size_t i = 0; i < count; ++i) {
            state->vertex_streams[first_binding + i] = DrawBindState::BoundVertexStream{
                .buffer = buffers[i],
                .offset = offsets[i],
                .size = sizes[i],
                .stride = strides_ptr != nullptr ? strides_ptr[i] : 0,
                .has_extended = true,
                .valid = true,
            };
        }
    }

    cmd.BindVertexBuffers2EXT(first_binding, static_cast<u32>(count), buffers.data(),
                              offsets.data(), sizes.data(), strides_ptr);
    vertex_buffer_binds_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

bool MemoryPoolManager::BindIndexBuffer(vk::CommandBuffer cmd, u64 guest_address,
                                        VideoCommon::Nvn::NVNindexType nvn_index_type,
                                        DrawBindState* state) const noexcept {
    VkIndexType vk_type = VK_INDEX_TYPE_UINT32;
    switch (nvn_index_type) {
    case VideoCommon::Nvn::NVNindexType::UNSIGNED_BYTE:
        vk_type = VK_INDEX_TYPE_UINT8_EXT;
        break;
    case VideoCommon::Nvn::NVNindexType::UNSIGNED_SHORT:
        vk_type = VK_INDEX_TYPE_UINT16;
        break;
    case VideoCommon::Nvn::NVNindexType::UNSIGNED_INT:
        vk_type = VK_INDEX_TYPE_UINT32;
        break;
    default:
        return false;
    }
    return BindIndexBuffer(cmd, guest_address, vk_type, state);
}

bool MemoryPoolManager::BindIndexBuffer(vk::CommandBuffer cmd, u64 guest_address,
                                        VkIndexType vk_index_type,
                                        DrawBindState* state) const noexcept {
    if (guest_address == 0) {
        return false;
    }
    if (vk_index_type == VK_INDEX_TYPE_UINT8_EXT && !device_.IsExtIndexTypeUint8Supported()) {
        return false;
    }

    const BufferRangeView view = ResolveBufferRange(guest_address);
    if (!view) {
        return false;
    }

    const u32 elem_bytes = IndexTypeByteSize(vk_index_type);
    if (!device_.IsKhrMaintenance5Supported() && (view.offset & (elem_bytes - 1u)) != 0) {
        return false;
    }

    if (state != nullptr) {
        if (state->index_buffer.valid && state->index_buffer.buffer == view.buffer &&
            state->index_buffer.offset == view.offset &&
            state->index_buffer.index_type == vk_index_type) {
            elided_index_binds_.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        state->index_buffer = DrawBindState::BoundIndex{
            .buffer = view.buffer,
            .offset = view.offset,
            .index_type = vk_index_type,
            .valid = true,
        };
    }

    cmd.BindIndexBuffer(view.buffer, view.offset, vk_index_type);
    index_buffer_binds_.fetch_add(1, std::memory_order_relaxed);
    return true;
}

PoolStats MemoryPoolManager::Stats() const noexcept {
    return PoolStats{
        .active_physical_pools = active_physical_pools_.load(std::memory_order_relaxed),
        .active_virtual_pools = active_virtual_pools_.load(std::memory_order_relaxed),
        .device_local_bar_bytes = device_local_bar_bytes_.load(std::memory_order_relaxed),
        .host_fallback_bytes = host_fallback_bytes_.load(std::memory_order_relaxed),
        .dynamic_ubo_binds = dynamic_ubo_binds_.load(std::memory_order_relaxed),
        .elided_ubo_binds = elided_ubo_binds_.load(std::memory_order_relaxed),
        .vertex_buffer_binds = vertex_buffer_binds_.load(std::memory_order_relaxed),
        .elided_vertex_binds = elided_vertex_binds_.load(std::memory_order_relaxed),
        .index_buffer_binds = index_buffer_binds_.load(std::memory_order_relaxed),
        .elided_index_binds = elided_index_binds_.load(std::memory_order_relaxed),
    };
}

} // namespace Vulkan::NvnClean
