// Tests for Common::AlignUp and related alignment utilities.
#include <gtest/gtest.h>
#include "common/alignment.h"

TEST(AlignmentTest, AlignUpBasic) {
    EXPECT_EQ(Common::AlignUp(0u, 16), 0u);
    EXPECT_EQ(Common::AlignUp(1u, 16), 16u);
    EXPECT_EQ(Common::AlignUp(15u, 16), 16u);
    EXPECT_EQ(Common::AlignUp(16u, 16), 16u);
    EXPECT_EQ(Common::AlignUp(17u, 16), 32u);
}

TEST(AlignmentTest, AlignUpPowerOfTwo) {
    EXPECT_EQ(Common::AlignUp(0u, 4096), 0u);
    EXPECT_EQ(Common::AlignUp(1u, 4096), 4096u);
    EXPECT_EQ(Common::AlignUp(4095u, 4096), 4096u);
    EXPECT_EQ(Common::AlignUp(4096u, 4096), 4096u);
}

TEST(AlignmentTest, AlignDown) {
    EXPECT_EQ(Common::AlignDown(0u, 16), 0u);
    EXPECT_EQ(Common::AlignDown(15u, 16), 0u);
    EXPECT_EQ(Common::AlignDown(16u, 16), 16u);
    EXPECT_EQ(Common::AlignDown(31u, 16), 16u);
    EXPECT_EQ(Common::AlignDown(32u, 16), 32u);
}

TEST(AlignmentTest, IsAligned) {
    EXPECT_TRUE(Common::IsAligned(0u, 16));
    EXPECT_TRUE(Common::IsAligned(16u, 16));
    EXPECT_TRUE(Common::IsAligned(256u, 16));
    EXPECT_FALSE(Common::IsAligned(1u, 16));
    EXPECT_FALSE(Common::IsAligned(15u, 16));
}

TEST(AlignmentTest, DivideUp) {
    EXPECT_EQ(Common::DivideUp(0u, 16u), 0u);
    EXPECT_EQ(Common::DivideUp(1u, 16u), 1u);
    EXPECT_EQ(Common::DivideUp(16u, 16u), 1u);
    EXPECT_EQ(Common::DivideUp(17u, 16u), 2u);
}

TEST(AlignmentTest, IsPowerOfTwo) {
    EXPECT_TRUE(Common::IsPowerOfTwo(1));
    EXPECT_TRUE(Common::IsPowerOfTwo(2));
    EXPECT_TRUE(Common::IsPowerOfTwo(4));
    EXPECT_TRUE(Common::IsPowerOfTwo(4096));
    EXPECT_FALSE(Common::IsPowerOfTwo(0));
    EXPECT_FALSE(Common::IsPowerOfTwo(3));
    EXPECT_FALSE(Common::IsPowerOfTwo(6));
}
