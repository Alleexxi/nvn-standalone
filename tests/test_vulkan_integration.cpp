// Real-GPU behaviour tests for SyncManager.
//
// Every test checks an OBSERVABLE fact (semaphore counter values, bytes written into a guest pool,
// GPU ordering) — not just "it didn't crash". Every test also fails if the Khronos validation layer
// reported any warning/error while it ran.
//
// Build: build.cmd -DNVN_ENABLE_VULKAN_TESTS=ON      Run: build\nvn_vulkan_tests.exe
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>

#include <gtest/gtest.h>

#include "nvn_clean/nv_mempool.h"
#include "nvn_clean/nv_sync_manager.h"
#include "vk_test_context.h"

using namespace Vulkan::NvnClean;
using NvnTest::VkTestContext;

namespace {

constexpr u64 kPoolSerial = 1;
constexpr u64 kPoolGuestBase = 0x1'0000'0000ULL;
constexpr u64 kPoolSize = 1ULL << 20; // 1 MiB: room for ~65k NVNcounterData

struct CounterData {
    u64 counter;
    u64 timestamp;
};
static_assert(sizeof(CounterData) == 16);

class SyncManagerGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        ctx = VkTestContext::Get();
        if (ctx == nullptr) {
            GTEST_SKIP() << "no GPU: " << VkTestContext::SkipReason();
        }
        if (!SyncManager::IsSupported(ctx->Device())) {
            GTEST_SKIP() << "GPU lacks timeline/BDA/host-query-reset/sync2";
        }
        validation_before = ctx->ValidationMessageCount();
        mempool = std::make_unique<MemoryPoolManager>(ctx->Device());
        ASSERT_TRUE(mempool->CreatePool(PoolCreateInfo{.serial = kPoolSerial,
                                                       .guest_address = kPoolGuestBase,
                                                       .size = kPoolSize,
                                                       .flags = 0}));
        std::memset(Host(0), 0, kPoolSize);
        mempool->FlushGuestRange(kPoolGuestBase, kPoolSize);
        sync = std::make_unique<SyncManager>(ctx->Device(), *mempool);
    }

    void TearDown() override {
        if (ctx == nullptr || !sync) {
            return;
        }
        ctx->WaitIdle();
        ctx->ResetCommandPool();
        sync.reset();
        mempool.reset();
        EXPECT_EQ(ctx->ValidationMessageCount(), validation_before)
            << "validation layer reported problems; last: " << ctx->LastValidationMessage();
    }

    u8* Host(u64 offset) {
        return mempool->FindPool(kPoolSerial)->HostPointer(offset);
    }
    CounterData ReadCounter(u64 offset) {
        mempool->InvalidateGuestRange(kPoolGuestBase + offset, sizeof(CounterData));
        CounterData d{};
        std::memcpy(&d, Host(offset), sizeof(d));
        return d;
    }
    u32 ReadU32(u64 offset) {
        mempool->InvalidateGuestRange(kPoolGuestBase + offset, sizeof(u32));
        u32 v = 0;
        std::memcpy(&v, Host(offset), sizeof(v));
        return v;
    }
    void WriteU64(u64 offset, u64 v) {
        std::memcpy(Host(offset), &v, sizeof(v));
        mempool->FlushGuestRange(kPoolGuestBase + offset, sizeof(v));
    }
    static u64 Addr(u64 offset) {
        return kPoolGuestBase + offset;
    }

    VkTestContext* ctx{};
    std::unique_ptr<MemoryPoolManager> mempool;
    std::unique_ptr<SyncManager> sync;
    u64 validation_before{};
};

// ---------------------------------------------------------------------------------------------
// Timeline semaphores
// ---------------------------------------------------------------------------------------------

TEST_F(SyncManagerGpuTest, TimelineValuesAreMonotonicPerQueue) {
    EXPECT_EQ(sync->AllocateNextTimelineValue(0), 1u);
    EXPECT_EQ(sync->AllocateNextTimelineValue(0), 2u);
    EXPECT_EQ(sync->AllocateNextTimelineValue(1), 1u) << "queues must have independent timelines";
    EXPECT_EQ(sync->AllocateNextTimelineValue(0), 3u);
    EXPECT_EQ(sync->AllocateNextTimelineValue(kMaxNvnQueues), 0u) << "out-of-range queue";
}

TEST_F(SyncManagerGpuTest, SemaphoreIsPersistentPerQueue) {
    const VkSemaphore a = sync->QueueTimelineSemaphore(3);
    ASSERT_NE(a, VK_NULL_HANDLE);
    EXPECT_EQ(sync->QueueTimelineSemaphore(3), a);
    EXPECT_NE(sync->QueueTimelineSemaphore(4), a);
    EXPECT_EQ(sync->QueueTimelineSemaphore(kMaxNvnQueues), VK_NULL_HANDLE);
}

TEST_F(SyncManagerGpuTest, FenceSyncRegistryRoundTrip) {
    sync->RecordFenceSync(42, 2, 7, 1, 0);
    const TimelineSyncPoint p = sync->FindSyncPoint(42);
    ASSERT_TRUE(p);
    EXPECT_EQ(p.queue_id, 2u);
    EXPECT_EQ(p.timeline_value, 7u);
    EXPECT_EQ(p.condition, 1u);
    sync->DestroySync(42);
    EXPECT_FALSE(sync->FindSyncPoint(42));
    EXPECT_FALSE(sync->FindSyncPoint(9999));
}

TEST_F(SyncManagerGpuTest, CompletionTracksSemaphoreCounter) {
    const u64 v = sync->AllocateNextTimelineValue(0);
    const VkSemaphore sem = sync->QueueTimelineSemaphore(0);
    EXPECT_FALSE(sync->IsTimelineValueComplete(0, v));
    ctx->HostSignal(sem, v);
    EXPECT_TRUE(sync->IsTimelineValueComplete(0, v));
    EXPECT_GE(sync->PollCompletedTimelineValue(0), v);
    EXPECT_TRUE(sync->IsTimelineValueComplete(0, 0)) << "value 0 is always complete";
}

TEST_F(SyncManagerGpuTest, WaitSyncHostStates) {
    EXPECT_EQ(sync->WaitSyncHost(777, 0), SyncWaitStatus::Failed) << "unknown sync";

    const u64 v = sync->AllocateNextTimelineValue(0);
    sync->RecordFenceSync(1, 0, v);
    EXPECT_EQ(sync->WaitSyncHost(1, 0), SyncWaitStatus::TimeoutExpired);
    EXPECT_EQ(sync->WaitSyncHost(1, 1'000'000), SyncWaitStatus::TimeoutExpired) << "1 ms timeout";

    ctx->HostSignal(sync->QueueTimelineSemaphore(0), v);
    EXPECT_EQ(sync->WaitSyncHost(1, 0), SyncWaitStatus::AlreadySignaled);
    const u64 hits_before = sync->Stats().host_sync_fast_cache_hits;
    EXPECT_EQ(sync->WaitSyncHost(1, 0), SyncWaitStatus::AlreadySignaled);
    EXPECT_EQ(sync->Stats().host_sync_fast_cache_hits, hits_before + 1)
        << "second wait on a retired sync must be served from the user-space cache";
}

TEST_F(SyncManagerGpuTest, WaitSyncHostBlocksUntilSignaled) {
    const u64 v = sync->AllocateNextTimelineValue(5);
    sync->RecordFenceSync(10, 5, v);
    const VkSemaphore sem = sync->QueueTimelineSemaphore(5);
    std::thread signaler([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(30));
        ctx->HostSignal(sem, v);
    });
    const auto t0 = std::chrono::steady_clock::now();
    const SyncWaitStatus s = sync->WaitSyncHost(10, 2'000'000'000ULL);
    const auto waited = std::chrono::steady_clock::now() - t0;
    signaler.join();
    EXPECT_EQ(s, SyncWaitStatus::ConditionSatisfied);
    EXPECT_GE(waited, std::chrono::milliseconds(20)) << "returned before the signal happened";
}

TEST_F(SyncManagerGpuTest, GpuSubmitSignalsTimeline) {
    const u64 v = sync->AllocateNextTimelineValue(0);
    sync->RecordFenceSync(5, 0, v);
    const VkSemaphoreSubmitInfo sig = sync->MakeTimelineSignalSubmitInfo(0, v);
    auto cmd = ctx->BeginCommands();
    ctx->Submit(cmd, {}, std::span(&sig, 1));
    sync->NotifyTimelineSubmitted(0, v);
    EXPECT_NE(sync->WaitSyncHost(5, 2'000'000'000ULL), SyncWaitStatus::TimeoutExpired);
    EXPECT_GE(ctx->CounterValue(sync->QueueTimelineSemaphore(0)), v);
}

// ---------------------------------------------------------------------------------------------
// WaitSync (queue-side)
// ---------------------------------------------------------------------------------------------

TEST_F(SyncManagerGpuTest, SameQueueWaitIsElided) {
    sync->RecordFenceSync(1, 0, sync->AllocateNextTimelineValue(0));
    VkSemaphoreSubmitInfo info{};
    EXPECT_FALSE(sync->ResolveQueueWaitSync(1, 0, info));
    EXPECT_EQ(sync->Stats().same_queue_waits_elided, 1u);
    EXPECT_EQ(info.semaphore, VK_NULL_HANDLE);
}

TEST_F(SyncManagerGpuTest, CrossQueueWaitEmitsSemaphoreUntilComplete) {
    const u64 v = sync->AllocateNextTimelineValue(0);
    sync->RecordFenceSync(1, 0, v);
    VkSemaphoreSubmitInfo info{};
    ASSERT_TRUE(sync->ResolveQueueWaitSync(1, 1, info));
    EXPECT_EQ(info.semaphore, sync->QueueTimelineSemaphore(0));
    EXPECT_EQ(info.value, v);
    EXPECT_EQ(info.sType, VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO);

    ctx->HostSignal(sync->QueueTimelineSemaphore(0), v);
    EXPECT_FALSE(sync->ResolveQueueWaitSync(1, 1, info)) << "already-complete producer: no wait";
    EXPECT_EQ(sync->Stats().cross_queue_semaphore_waits, 1u);
}

TEST_F(SyncManagerGpuTest, CrossQueueWaitReallyOrdersGpuWork) {
    // Consumer (NVN queue 1) waits on producer (NVN queue 0) value that nobody has signaled yet.
    const u64 v = sync->AllocateNextTimelineValue(0);
    sync->RecordFenceSync(1, 0, v);
    VkSemaphoreSubmitInfo wait{};
    ASSERT_TRUE(sync->ResolveQueueWaitSync(1, 1, wait));

    const BufferRangeView view = mempool->ResolveBufferRange(Addr(0x100), 4);
    ASSERT_TRUE(view);
    auto cmd = ctx->BeginCommands();
    cmd.FillBuffer(view.buffer, view.offset, 4, 0xC0FFEEu);
    const VkMemoryBarrier2 mb{.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                              .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                              .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                              .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                              .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    const VkDependencyInfo dep{.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                               .memoryBarrierCount = 1,
                               .pMemoryBarriers = &mb};
    ctx->Dld().vkCmdPipelineBarrier2(*cmd, &dep);
    ctx->Submit(cmd, std::span(&wait, 1));

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(ReadU32(0x100), 0u) << "consumer ran before producer signaled: wait is not real";

    ctx->HostSignal(sync->QueueTimelineSemaphore(0), v);
    ctx->WaitIdle();
    EXPECT_EQ(ReadU32(0x100), 0xC0FFEEu);
}

// ---------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------

TEST_F(SyncManagerGpuTest, SignalEventWriteAndAdd) {
    constexpr u32 kWrite = 0; // NVN_EVENT_SIGNAL_MODE_WRITE
    constexpr u32 kAdd = 1;   // NVN_EVENT_SIGNAL_MODE_ADD
    const u64 v = sync->AllocateNextTimelineValue(0);
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        sync->RecordSignalEvent(cmd, 0, v, Addr(0x200), kWrite, 41);
    });
    EXPECT_EQ(ReadU32(0x200), 41u);
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        sync->RecordSignalEvent(cmd, 0, v, Addr(0x200), kAdd, 1);
        sync->RecordSignalEvent(cmd, 0, v, Addr(0x200), kAdd, 2);
    });
    EXPECT_EQ(ReadU32(0x200), 44u);
}

// ---------------------------------------------------------------------------------------------
// Queries / reports (GPU compute resolve into guest memory)
// ---------------------------------------------------------------------------------------------

TEST_F(SyncManagerGpuTest, ReportValueWritesImmediateAndTimestamp) {
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        ASSERT_TRUE(sync->RecordReportValue(cmd, 0, Addr(0x1000), 0xDEADBEEFCAFEULL));
        EXPECT_TRUE(sync->HasPendingQueryResolves());
        EXPECT_EQ(sync->FlushQueryResolves(cmd), 1u);
        EXPECT_FALSE(sync->HasPendingQueryResolves());
    });
    const CounterData d = ReadCounter(0x1000);
    EXPECT_EQ(d.counter, 0xDEADBEEFCAFEULL);
    EXPECT_NE(d.timestamp, 0u);
}

TEST_F(SyncManagerGpuTest, UnmappedReportAddressIsRejected) {
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        EXPECT_FALSE(sync->RecordReportValue(cmd, 0, 0xDEAD0000ULL, 1));
        EXPECT_FALSE(sync->RecordReportCounter(cmd, 0, NvnCounterKind::kTimestamp, 0xDEAD0000ULL));
        EXPECT_FALSE(sync->HasPendingQueryResolves());
    });
}

TEST_F(SyncManagerGpuTest, TimestampsAreMonotonic) {
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        for (u64 i = 0; i < 8; ++i) {
            ASSERT_TRUE(sync->RecordReportCounter(cmd, 0, NvnCounterKind::kTimestamp,
                                                  Addr(0x2000 + i * 16)));
        }
        EXPECT_EQ(sync->FlushQueryResolves(cmd), 8u);
    });
    u64 prev = 0;
    for (u64 i = 0; i < 8; ++i) {
        const CounterData d = ReadCounter(0x2000 + i * 16);
        EXPECT_NE(d.timestamp, 0u) << "report " << i;
        EXPECT_GE(d.timestamp, prev) << "report " << i << " went backwards";
        prev = d.timestamp;
    }
}

TEST_F(SyncManagerGpuTest, TimestampsAdvanceAcrossSubmits) {
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        (void)sync->RecordReportCounter(cmd, 0, NvnCounterKind::kTimestamp, Addr(0x3000));
        (void)sync->FlushQueryResolves(cmd);
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        (void)sync->RecordReportCounter(cmd, 0, NvnCounterKind::kTimestamp, Addr(0x3010));
        (void)sync->FlushQueryResolves(cmd);
    });
    const u64 a = ReadCounter(0x3000).timestamp;
    const u64 b = ReadCounter(0x3010).timestamp;
    EXPECT_GT(b, a);
    // NVN timestamps are nanoseconds: 5 ms sleep must show up as >= ~4 ms.
    EXPECT_GE(b - a, 4'000'000u) << "tick->ns conversion looks wrong (delta " << (b - a) << ")";
    EXPECT_LT(b - a, 5'000'000'000u);
}

TEST_F(SyncManagerGpuTest, SamplesPassedWithoutDrawsIsZero) {
    WriteU64(0x4000, 0x1234); // garbage that must be overwritten
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        sync->RecordResetCounter(cmd, 0, NvnCounterKind::kSamplesPassed);
        sync->EnsureOcclusionQueryActive(cmd, 0);
        sync->CloseActiveOcclusionQuery(cmd, 0);
        ASSERT_TRUE(sync->RecordReportCounter(cmd, 0, NvnCounterKind::kSamplesPassed, Addr(0x4000)));
        (void)sync->FlushQueryResolves(cmd);
    });
    const CounterData d = ReadCounter(0x4000);
    EXPECT_EQ(d.counter, 0u);
    EXPECT_NE(d.timestamp, 0u);
}

TEST_F(SyncManagerGpuTest, ConsecutiveResetsCoalesce) {
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        for (int i = 0; i < 10; ++i) {
            sync->RecordResetCounter(cmd, 0, NvnCounterKind::kSamplesPassed);
        }
        EXPECT_EQ(sync->FlushQueryResolves(cmd), 1u) << "10 back-to-back resets = 1 micro-op";
    });
    EXPECT_EQ(sync->Stats().counter_resets_recorded, 10u);
}

TEST_F(SyncManagerGpuTest, LargeBatchSplitsAcrossDispatches) {
    constexpr u32 kOps = kMaxResolveOpsPerBatch + 904; // 5,000 > one batch
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        for (u32 i = 0; i < kOps; ++i) {
            ASSERT_TRUE(sync->RecordReportValue(cmd, 0, Addr(0x10000 + u64{i} * 16), 1000 + i));
        }
        EXPECT_EQ(sync->FlushQueryResolves(cmd), kOps);
    });
    EXPECT_GE(sync->Stats().compute_resolve_dispatches, 2u);
    for (u32 i = 0; i < kOps; i += 97) {
        EXPECT_EQ(ReadCounter(0x10000 + u64{i} * 16).counter, 1000u + i) << "op " << i;
    }
    EXPECT_EQ(ReadCounter(0x10000 + u64{kOps - 1} * 16).counter, 1000u + kOps - 1);
}

TEST_F(SyncManagerGpuTest, QueryPoolsRecycleOverManyFrames) {
    // 3 "frames" of OoT-like volume (~2,000 timestamps/frame) with retire + reclaim in between.
    for (int frame = 0; frame < 3; ++frame) {
        const u64 v = sync->AllocateNextTimelineValue(0);
        const VkSemaphoreSubmitInfo sig = sync->MakeTimelineSignalSubmitInfo(0, v);
        auto cmd = ctx->BeginCommands();
        for (u32 i = 0; i < 2000; ++i) {
            ASSERT_TRUE(sync->RecordReportCounter(cmd, 0, NvnCounterKind::kTimestamp,
                                                  Addr(0x40000 + u64{i} * 16)));
        }
        (void)sync->FlushQueryResolves(cmd, nullptr, 0, v);
        ctx->Submit(cmd, {}, std::span(&sig, 1));
        sync->NotifyTimelineSubmitted(0, v);
        ctx->WaitIdle();
        sync->ReclaimCompletedQueryPools();
        EXPECT_NE(ReadCounter(0x40000 + 1999 * 16).timestamp, 0u) << "frame " << frame;
    }
}

// ---------------------------------------------------------------------------------------------
// Conditional rendering. Proof of skip/execute: a compute dispatch (event ADD) is recorded inside
// the conditional scope; VK_EXT_conditional_rendering discards dispatches when predicate == 0.
// ---------------------------------------------------------------------------------------------

class ConditionalTest : public SyncManagerGpuTest,
                        public ::testing::WithParamInterface<std::tuple<u64, u64, u32, bool>> {};

TEST_P(ConditionalTest, PredicateGatesGpuWork) {
    if (!ctx->Device().IsExtConditionalRendering()) {
        GTEST_SKIP() << "VK_EXT_conditional_rendering unsupported";
    }
    const auto [a, b, mode, should_run] = GetParam();
    WriteU64(0x8000, a);
    WriteU64(0x8010, b);
    const u64 v = sync->AllocateNextTimelineValue(0);
    ctx->RunAndWait([&](Vulkan::vk::CommandBuffer cmd) {
        ASSERT_TRUE(sync->BeginConditionalRendering(cmd, nullptr, Addr(0x8000), mode));
        EXPECT_TRUE(sync->IsConditionalRenderingActive());
        sync->RecordSignalEvent(cmd, 0, v, Addr(0x8100), /*ADD*/ 1, 1);
        sync->EndConditionalRendering(cmd);
        EXPECT_FALSE(sync->IsConditionalRenderingActive());
    });
    EXPECT_EQ(ReadU32(0x8100), should_run ? 1u : 0u);
}

INSTANTIATE_TEST_SUITE_P(
    Modes, ConditionalTest,
    ::testing::Values(std::make_tuple(5ULL, 5ULL, NvnConditionMode::kRenderIfEqual, true),
                      std::make_tuple(5ULL, 6ULL, NvnConditionMode::kRenderIfEqual, false),
                      std::make_tuple(5ULL, 6ULL, NvnConditionMode::kRenderIfNotEqual, true),
                      std::make_tuple(7ULL, 7ULL, NvnConditionMode::kRenderIfNotEqual, false),
                      // high 32 bits differ only: a 32-bit compare would get this wrong
                      std::make_tuple(0x1'0000'0001ULL, 0x2'0000'0001ULL,
                                      NvnConditionMode::kRenderIfEqual, false)));

} // namespace
