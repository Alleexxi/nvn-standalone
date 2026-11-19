// Stub: video_core/vulkan_common/vma.h — VMA include wrapper.
#pragma once

#include "video_core/vulkan_common/vulkan.h"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4189 )
#endif
#include "vk_mem_alloc.h"

#ifdef _MSC_VER
#pragma warning( pop )
#endif
