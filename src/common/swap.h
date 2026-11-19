// Stub: common/swap.h — Byte-swap utilities (simplified for standalone).
#pragma once

#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <bit>
#include "common/common_types.h"

namespace Common {

#if defined(_MSC_VER)
[[nodiscard]] inline u16 swap16(u16 data) noexcept {
    return _byteswap_ushort(data);
}
[[nodiscard]] inline u32 swap32(u32 data) noexcept {
    return _byteswap_ulong(data);
}
[[nodiscard]] inline u64 swap64(u64 data) noexcept {
    return _byteswap_uint64(data);
}
#elif defined(__clang__) || defined(__GNUC__)
[[nodiscard]] inline u16 swap16(u16 data) noexcept {
    return __builtin_bswap16(data);
}
[[nodiscard]] inline u32 swap32(u32 data) noexcept {
    return __builtin_bswap32(data);
}
[[nodiscard]] inline u64 swap64(u64 data) noexcept {
    return __builtin_bswap64(data);
}
#else
[[nodiscard]] inline u16 swap16(u16 data) noexcept {
    return (data >> 8) | (data << 8);
}
[[nodiscard]] inline u32 swap32(u32 data) noexcept {
    return ((data & 0xFF000000U) >> 24) | ((data & 0x00FF0000U) >> 8) |
           ((data & 0x0000FF00U) << 8) | ((data & 0x000000FFU) << 24);
}
[[nodiscard]] inline u64 swap64(u64 data) noexcept {
    return ((u64)swap32((u32)(data >> 32))) | ((u64)swap32((u32)data) << 32);
}
#endif

} // namespace Common

// Minimal SwapStructT for the logging header's formatter
template <typename T, typename S = T>
struct SwapStructT {
    T raw;
    operator T() const { return raw; }
};
