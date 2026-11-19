// Stub: common/container/unordered_map.h — Alias to std::unordered_map.
#pragma once
#include <unordered_map>
// Eglantine uses boost::unordered_flat_map; we alias to std for the standalone build.
namespace Common {
template <typename K, typename V, typename H = std::hash<K>, typename E = std::equal_to<K>>
using UnorderedMap = std::unordered_map<K, V, H, E>;

// Eglantine uses lowercase alias (boost::unordered_flat_map)
template <typename K, typename V, typename H = std::hash<K>, typename E = std::equal_to<K>>
using unordered_map = std::unordered_map<K, V, H, E>;
} // namespace Common
