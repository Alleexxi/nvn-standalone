// Stub: common/settings.h — Minimal settings interface for vulkan_device.h.
#pragma once

namespace Settings {
namespace values {

struct ResolutionScalingInfo {
    bool active{false};
    u32 up_scale{1};
    u32 down_shift{0};
    f32 up_factor{1.0f};
    f32 down_factor{1.0f};
};

inline ResolutionScalingInfo resolution_info{};

} // namespace values
} // namespace Settings
