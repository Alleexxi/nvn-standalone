// Tests for MemoryPoolManager types and address alignment.
#include <gtest/gtest.h>
#include "common/alignment.h"
#include "common/common_types.h"

namespace {

// Memory pool addresses should be 256-byte aligned (NVN requirement)
TEST(MempoolTest, PoolAddressAlignment) {
    constexpr u64 pool_base = 0x8000'0000ULL;
    constexpr size_t alignment = 256;

    EXPECT_TRUE(Common::IsAligned(pool_base, alignment));
    EXPECT_TRUE(Common::IsAligned(Common::AlignUp(pool_base + 1, alignment), alignment));
}

// Test buffer offset alignment within a pool
TEST(MempoolTest, BufferOffsetAlignment) {
    constexpr size_t ssbo_alignment = 256;
    constexpr size_t ubo_alignment = 256;

    u64 offset = 100;
    u64 aligned_ssbo = Common::AlignUp(offset, ssbo_alignment);
    u64 aligned_ubo = Common::AlignUp(offset, ubo_alignment);

    EXPECT_EQ(aligned_ssbo, 256u);
    EXPECT_EQ(aligned_ubo, 256u);
}

// Test page-aligned allocations (64KB pages, NVN standard)
TEST(MempoolTest, PageAlignment) {
    constexpr size_t page_size = 65536; // 64KB
    constexpr u64 addr = 0x12345;
    u64 page_aligned = Common::AlignUp(addr, page_size);
    EXPECT_EQ(page_aligned, 0x20000u);
    EXPECT_TRUE(Common::IsAligned(page_aligned, page_size));
}

// Test DivideUp for block count calculation
TEST(MempoolTest, BlockCountCalculation) {
    constexpr u64 total_size = 1024 * 1024; // 1MB
    constexpr u64 block_size = 65536;       // 64KB

    auto block_count = Common::DivideUp(total_size, block_size);
    EXPECT_EQ(block_count, 16u);

    // Non-even division
    auto block_count2 = Common::DivideUp(total_size + 1, block_size);
    EXPECT_EQ(block_count2, 17u);
}

} // namespace
