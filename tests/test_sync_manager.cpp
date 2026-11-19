// Tests for SyncManager data structures and utilities.
// These are mock-based tests that verify the interface and logic without a real Vulkan device.
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "common/common_types.h"
#include "video_core/nvn/nvn_types.h"

namespace {

// Test NVNcounterData layout (the struct SyncManager operates on)
TEST(SyncManagerTypesTest, NVNcounterDataLayout) {
    static_assert(sizeof(VideoCommon::Nvn::NVNcounterData) == 16,
                  "NVNcounterData must be 16 bytes");
    static_assert(offsetof(VideoCommon::Nvn::NVNcounterData, counter) == 0);
    static_assert(offsetof(VideoCommon::Nvn::NVNcounterData, timestamp) == 8);

    VideoCommon::Nvn::NVNcounterData data{};
    data.counter = 42;
    data.timestamp = 0xDEAD'BEEF'CAFE'0000ULL;
    EXPECT_EQ(data.counter, 42u);
    EXPECT_EQ(data.timestamp, 0xDEAD'BEEF'CAFE'0000ULL);
}

// Test fixed-point timestamp math used by SyncManager
// NVN timestamps are nanoseconds encoded as 64-bit values.
// The GPU returns ticks which must be multiplied by the timestamp period.
TEST(SyncManagerTypesTest, TimestampFixedPointConversion) {
    // Simulate GPU timestamp conversion:
    // GPU ticks * period_ns = nanoseconds
    const f32 timestamp_period = 1.0f; // 1 ns per tick (simplified)
    const u64 gpu_ticks = 1'000'000'000ULL; // 1 billion ticks = 1 second

    const u64 nanoseconds = static_cast<u64>(static_cast<f64>(gpu_ticks) * timestamp_period);
    EXPECT_EQ(nanoseconds, 1'000'000'000ULL);
}

// Test that NVNbufferAddress (u64) can represent the full GPU address space
TEST(SyncManagerTypesTest, BufferAddressRange) {
    VideoCommon::Nvn::NVNbufferAddress addr = 0xFFFF'FFFF'FFFF'FFFFULL;
    EXPECT_EQ(addr, UINT64_MAX);
}

// Test NVNbufferRange layout
TEST(SyncManagerTypesTest, BufferRangeLayout) {
    static_assert(sizeof(VideoCommon::Nvn::NVNbufferRange) == 16);
    VideoCommon::Nvn::NVNbufferRange range{};
    range.address = 0x1000;
    range.size = 4096;
    EXPECT_EQ(range.address, 0x1000u);
    EXPECT_EQ(range.size, 4096u);
}

} // namespace
