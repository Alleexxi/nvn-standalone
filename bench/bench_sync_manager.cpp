// SyncManager benchmarks (Google Benchmark) on a real GPU.
//
//   build.cmd -DNVN_ENABLE_BENCHMARKS=ON
//   set NVN_NO_VALIDATION=1
//   build\nvn_bench.exe --benchmark_repetitions=10 --benchmark_report_aggregates_only=true ^
//                       --benchmark_out=bench\results\current.json --benchmark_out_format=json
//   python bench\compare.py bench\baseline.json bench\results\current.json
//
// Three tiers:
//   1. Micro: CPU ns per call of each hot entry point (what the replay thread pays).
//   2. Batch: CPU cost of recording + flushing N query ops; GPU us of the resolve dispatch.
//   3. Frame: a synthetic frame with the call mix MEASURED in OoT market (see README "Call
//      volumes"). `cpu_ms` is replay-thread time, `gpu_us` is GPU time of SyncManager's own work.
// ALWAYS run with NVN_NO_VALIDATION=1 — the validation layer is ~10-50x slower.
#include <cstring>
#include <memory>
#include <vector>

#include <benchmark/benchmark.h>

#include "nvn_clean/nv_mempool.h"
#include "nvn_clean/nv_sync_manager.h"
#include "vk_test_context.h"

using namespace Vulkan::NvnClean;
using NvnTest::VkTestContext;

namespace {

constexpr u64 kPoolSerial = 1;
constexpr u64 kPoolBase = 0x1'0000'0000ULL;
constexpr u64 kPoolSize = 4ULL << 20;

/// Fresh MemoryPoolManager + SyncManager per benchmark function.
struct Rig {
    VkTestContext* ctx{};
    std::unique_ptr<MemoryPoolManager> mempool;
    std::unique_ptr<SyncManager> sync;

    bool Init(benchmark::State& state) {
        ctx = VkTestContext::Get();
        if (ctx == nullptr) {
            state.SkipWithError(("no GPU: " + VkTestContext::SkipReason()).c_str());
            return false;
        }
        if (!SyncManager::IsSupported(ctx->Device())) {
            state.SkipWithError("SyncManager unsupported on this GPU");
            return false;
        }
        mempool = std::make_unique<MemoryPoolManager>(ctx->Device());
        mempool->CreatePool(PoolCreateInfo{.serial = kPoolSerial, .guest_address = kPoolBase,
                                           .size = kPoolSize, .flags = 0});
        sync = std::make_unique<SyncManager>(ctx->Device(), *mempool);
        return true;
    }
    ~Rig() {
        if (ctx) {
            ctx->WaitIdle();
            ctx->ResetCommandPool();
        }
    }
};

/// GPU timer: two timestamps around a region, read back after the submit retires.
struct GpuTimer {
    VkTestContext* ctx;
    VkQueryPool pool{VK_NULL_HANDLE};
    explicit GpuTimer(VkTestContext* c) : ctx{c} {
        const VkQueryPoolCreateInfo ci{.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                       .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                       .queryCount = 2};
        ctx->Dld().vkCreateQueryPool(ctx->Logical(), &ci, nullptr, &pool);
    }
    ~GpuTimer() {
        ctx->Dld().vkDestroyQueryPool(ctx->Logical(), pool, nullptr);
    }
    void Begin(Vulkan::vk::CommandBuffer cmd) {
        ctx->Dld().vkCmdResetQueryPool(*cmd, pool, 0, 2);
        ctx->Dld().vkCmdWriteTimestamp(*cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, pool, 0);
    }
    void End(Vulkan::vk::CommandBuffer cmd) {
        ctx->Dld().vkCmdWriteTimestamp(*cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, pool, 1);
    }
    double ReadMicroseconds() {
        u64 t[2]{};
        ctx->Dld().vkGetQueryPoolResults(ctx->Logical(), pool, 0, 2, sizeof(t), t, sizeof(u64),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        return static_cast<double>(t[1] - t[0]) * ctx->TimestampPeriodNs() / 1000.0;
    }
};

u64 ReportAddr(u32 i) {
    return kPoolBase + (u64{i} % (kPoolSize / 16 - 1024)) * 16;
}

// =============================================================================================
// Tier 1: micro (CPU ns per call)
// =============================================================================================

void BM_AllocateNextTimelineValue(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->AllocateNextTimelineValue(0));
    }
}
BENCHMARK(BM_AllocateNextTimelineValue);

void BM_RecordFenceSync(benchmark::State& state) {
    // Steady state: ~`live` syncs alive (guest keeps a ring of NVNsync objects).
    Rig rig;
    if (!rig.Init(state)) return;
    const u64 live = static_cast<u64>(state.range(0));
    u64 serial = 1;
    for (auto _ : state) {
        rig.sync->RecordFenceSync(serial, 0, serial);
        if (serial > live) {
            rig.sync->DestroySync(serial - live);
        }
        ++serial;
    }
}
BENCHMARK(BM_RecordFenceSync)->Arg(64)->Arg(4096);

void BM_FindSyncPoint(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    for (u64 s = 1; s <= 4096; ++s) rig.sync->RecordFenceSync(s, 0, s);
    u64 s = 0;
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->FindSyncPoint((s++ & 4095) + 1));
    }
}
BENCHMARK(BM_FindSyncPoint);

void BM_ResolveWait_SameQueue(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    rig.sync->RecordFenceSync(1, 0, rig.sync->AllocateNextTimelineValue(0));
    VkSemaphoreSubmitInfo info{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->ResolveQueueWaitSync(1, 0, info));
    }
}
BENCHMARK(BM_ResolveWait_SameQueue);

void BM_ResolveWait_CrossQueuePending(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    rig.sync->RecordFenceSync(1, 0, rig.sync->AllocateNextTimelineValue(0)); // never signaled
    VkSemaphoreSubmitInfo info{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->ResolveQueueWaitSync(1, 1, info));
    }
}
BENCHMARK(BM_ResolveWait_CrossQueuePending);

void BM_ResolveWait_CrossQueueComplete(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    const u64 v = rig.sync->AllocateNextTimelineValue(0);
    rig.sync->RecordFenceSync(1, 0, v);
    rig.ctx->HostSignal(rig.sync->QueueTimelineSemaphore(0), v);
    VkSemaphoreSubmitInfo info{};
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->ResolveQueueWaitSync(1, 1, info));
    }
}
BENCHMARK(BM_ResolveWait_CrossQueueComplete);

void BM_IsTimelineValueComplete_CacheHit(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    const u64 v = rig.sync->AllocateNextTimelineValue(0);
    rig.ctx->HostSignal(rig.sync->QueueTimelineSemaphore(0), v);
    (void)rig.sync->IsTimelineValueComplete(0, v); // warm cache
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->IsTimelineValueComplete(0, v));
    }
}
BENCHMARK(BM_IsTimelineValueComplete_CacheHit);

void BM_IsTimelineValueComplete_Pending(benchmark::State& state) {
    // Pending value: every call pays a vkGetSemaphoreCounterValue driver call.
    Rig rig;
    if (!rig.Init(state)) return;
    const u64 v = rig.sync->AllocateNextTimelineValue(0);
    (void)rig.sync->QueueTimelineSemaphore(0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->IsTimelineValueComplete(0, v));
    }
}
BENCHMARK(BM_IsTimelineValueComplete_Pending);

void BM_WaitSyncHost_CacheHit(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    const u64 v = rig.sync->AllocateNextTimelineValue(0);
    rig.sync->RecordFenceSync(1, 0, v);
    rig.ctx->HostSignal(rig.sync->QueueTimelineSemaphore(0), v);
    (void)rig.sync->WaitSyncHost(1, 0);
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->WaitSyncHost(1, 0));
    }
}
BENCHMARK(BM_WaitSyncHost_CacheHit);

void BM_WaitSyncHost_PendingPoll(benchmark::State& state) {
    Rig rig;
    if (!rig.Init(state)) return;
    rig.sync->RecordFenceSync(1, 0, rig.sync->AllocateNextTimelineValue(0));
    for (auto _ : state) {
        benchmark::DoNotOptimize(rig.sync->WaitSyncHost(1, 0));
    }
}
BENCHMARK(BM_WaitSyncHost_PendingPoll);

// =============================================================================================
// Tier 2: batches (record N ops + flush). CPU time per op, GPU us for the whole batch.
// =============================================================================================

enum class OpKind { Timestamp, ReportValue, OcclusionCycle };

void RunBatch(benchmark::State& state, OpKind kind) {
    Rig rig;
    if (!rig.Init(state)) return;
    const u32 n = static_cast<u32>(state.range(0));
    GpuTimer timer{rig.ctx};
    double gpu_us_total = 0;
    u64 iters = 0;
    for (auto _ : state) {
        state.PauseTiming();
        auto cmd = rig.ctx->BeginCommands();
        timer.Begin(cmd);
        state.ResumeTiming();

        for (u32 i = 0; i < n; ++i) {
            switch (kind) {
            case OpKind::Timestamp:
                (void)rig.sync->RecordReportCounter(cmd, 0, NvnCounterKind::kTimestamp, ReportAddr(i));
                break;
            case OpKind::ReportValue:
                (void)rig.sync->RecordReportValue(cmd, 0, ReportAddr(i), i);
                break;
            case OpKind::OcclusionCycle:
                rig.sync->RecordResetCounter(cmd, 0, NvnCounterKind::kSamplesPassed);
                rig.sync->EnsureOcclusionQueryActive(cmd, 0);
                rig.sync->CloseActiveOcclusionQuery(cmd, 0);
                (void)rig.sync->RecordReportCounter(cmd, 0, NvnCounterKind::kSamplesPassed,
                                                    ReportAddr(i));
                break;
            }
        }
        benchmark::DoNotOptimize(rig.sync->FlushQueryResolves(cmd));

        state.PauseTiming();
        timer.End(cmd);
        rig.ctx->Submit(cmd);
        rig.ctx->WaitIdle();
        gpu_us_total += timer.ReadMicroseconds();
        ++iters;
        rig.ctx->ResetCommandPool();
        rig.sync->ReclaimCompletedQueryPools();
        state.ResumeTiming();
    }
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * n);
    state.counters["gpu_us"] = iters ? gpu_us_total / static_cast<double>(iters) : 0.0;
    state.counters["cpu_ns_per_op"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * n,
        benchmark::Counter::kIsRate | benchmark::Counter::kInvert);
}

void BM_Batch_Timestamps(benchmark::State& s) { RunBatch(s, OpKind::Timestamp); }
void BM_Batch_ReportValues(benchmark::State& s) { RunBatch(s, OpKind::ReportValue); }
void BM_Batch_OcclusionCycles(benchmark::State& s) { RunBatch(s, OpKind::OcclusionCycle); }
BENCHMARK(BM_Batch_Timestamps)->Arg(64)->Arg(1024)->Arg(4096)->UseRealTime()->Iterations(100);
BENCHMARK(BM_Batch_ReportValues)->Arg(64)->Arg(1024)->Arg(4096)->UseRealTime()->Iterations(100);
BENCHMARK(BM_Batch_OcclusionCycles)->Arg(64)->Arg(1024)->Arg(2048)->UseRealTime()->Iterations(100);

// =============================================================================================
// Tier 3: synthetic OoT-market frame (call mix measured from the legacy path, README).
// =============================================================================================

struct FrameMix {
    u32 submits;          // command-buffer segments per frame
    u32 occlusion_cycles; // Reset + Begin/End occlusion + SAMPLES_PASSED report
    u32 timestamps;       // TIMESTAMP reports
    u32 conditions;       // SetRenderEnableConditional begin/end
    u32 cross_waits;      // cross-queue WaitSync
};

void RunFrame(benchmark::State& state, const FrameMix& mix) {
    Rig rig;
    if (!rig.Init(state)) return;
    GpuTimer timer{rig.ctx};
    double gpu_us_total = 0;
    u64 iters = 0;
    u64 serial = 1;

    for (auto _ : state) {
        auto first = rig.ctx->BeginCommands();
        timer.Begin(first);
        Vulkan::vk::CommandBuffer cmd = first;

        u32 occ = 0, ts = 0, cond = 0, waits = 0;
        for (u32 seg = 0; seg < mix.submits; ++seg) {
            // Spread work evenly across segments.
            const u32 occ_end = mix.occlusion_cycles * (seg + 1) / mix.submits;
            const u32 ts_end = mix.timestamps * (seg + 1) / mix.submits;
            const u32 cond_end = mix.conditions * (seg + 1) / mix.submits;
            for (; occ < occ_end; ++occ) {
                rig.sync->RecordResetCounter(cmd, 0, NvnCounterKind::kSamplesPassed);
                rig.sync->EnsureOcclusionQueryActive(cmd, 0);
                rig.sync->CloseActiveOcclusionQuery(cmd, 0);
                (void)rig.sync->RecordReportCounter(cmd, 0, NvnCounterKind::kSamplesPassed,
                                                    ReportAddr(occ));
                if (cond < cond_end) {
                    (void)rig.sync->BeginConditionalRendering(cmd, nullptr, ReportAddr(occ), 1);
                    rig.sync->EndConditionalRendering(cmd);
                    ++cond;
                }
            }
            for (; ts < ts_end; ++ts) {
                (void)rig.sync->RecordReportCounter(cmd, 0, NvnCounterKind::kTimestamp,
                                                    ReportAddr(4000 + ts));
            }
            const u64 v = rig.sync->AllocateNextTimelineValue(0);
            (void)rig.sync->FlushQueryResolves(cmd, nullptr, 0, v);
            rig.sync->RecordFenceSync(serial, 0, v);
            if (waits < mix.cross_waits) {
                VkSemaphoreSubmitInfo w{};
                benchmark::DoNotOptimize(rig.sync->ResolveQueueWaitSync(serial, 1, w));
                ++waits;
            }
            if (serial > 256) rig.sync->DestroySync(serial - 256);
            ++serial;

            const VkSemaphoreSubmitInfo sig = rig.sync->MakeTimelineSignalSubmitInfo(0, v);
            const bool last = seg + 1 == mix.submits;
            if (last) timer.End(cmd);
            rig.ctx->Submit(cmd, {}, std::span(&sig, 1));
            rig.sync->NotifyTimelineSubmitted(0, v);
            rig.sync->ReclaimCompletedQueryPools();
            if (!last) cmd = rig.ctx->BeginCommands();
        }

        state.PauseTiming();
        rig.ctx->WaitIdle();
        gpu_us_total += timer.ReadMicroseconds();
        ++iters;
        rig.ctx->ResetCommandPool();
        rig.sync->ReclaimCompletedQueryPools();
        state.ResumeTiming();
    }
    state.counters["gpu_us"] = iters ? gpu_us_total / static_cast<double>(iters) : 0.0;
    state.counters["ops"] = mix.occlusion_cycles * 4.0 + mix.timestamps + mix.conditions;
}

void BM_Frame_OoTMarket(benchmark::State& s) {
    // Measured (legacy path, OoT market, per frame): 24 submits, ~1,800 occlusion begin/end,
    // ~1,950 SAMPLES_PASSED reports, ~2,000 timestamp writes (incl. the ones per report),
    // ~1,200 conditions. Here: 1,800 occlusion cycles (each = reset+begin/end+report),
    // 150 plain timestamps, 1,200 conditions.
    RunFrame(s, FrameMix{.submits = 24, .occlusion_cycles = 1800, .timestamps = 150,
                         .conditions = 1200, .cross_waits = 24});
}
void BM_Frame_Light(benchmark::State& s) {
    // Light scene (menus / 2D): few queries, sync-dominated.
    RunFrame(s, FrameMix{.submits = 8, .occlusion_cycles = 32, .timestamps = 16,
                         .conditions = 0, .cross_waits = 8});
}
// Breakdown of BM_Frame_OoTMarket: same 24 submits, one op family at a time.
// Their gpu_us shows which family dominates the full frame.
void BM_FramePart_Occlusion(benchmark::State& s) {
    RunFrame(s, FrameMix{.submits = 24, .occlusion_cycles = 1800, .timestamps = 0,
                         .conditions = 0, .cross_waits = 0});
}
void BM_FramePart_Timestamps(benchmark::State& s) {
    RunFrame(s, FrameMix{.submits = 24, .occlusion_cycles = 0, .timestamps = 150,
                         .conditions = 0, .cross_waits = 0});
}
void BM_FramePart_Conditions(benchmark::State& s) {
    // Conditions need an occlusion report to compare against, so this is occlusion + conditions;
    // subtract BM_FramePart_Occlusion to get the condition cost.
    RunFrame(s, FrameMix{.submits = 24, .occlusion_cycles = 1800, .timestamps = 0,
                         .conditions = 1200, .cross_waits = 0});
}
void BM_FramePart_SyncOnly(benchmark::State& s) {
    RunFrame(s, FrameMix{.submits = 24, .occlusion_cycles = 0, .timestamps = 0,
                         .conditions = 0, .cross_waits = 24});
}
BENCHMARK(BM_FramePart_Occlusion)->Unit(benchmark::kMillisecond)->UseRealTime()->Iterations(100);
BENCHMARK(BM_FramePart_Timestamps)->Unit(benchmark::kMillisecond)->UseRealTime()->Iterations(100);
BENCHMARK(BM_FramePart_Conditions)->Unit(benchmark::kMillisecond)->UseRealTime()->Iterations(100);
BENCHMARK(BM_FramePart_SyncOnly)->Unit(benchmark::kMillisecond)->UseRealTime()->Iterations(100);
BENCHMARK(BM_Frame_OoTMarket)->Unit(benchmark::kMillisecond)->UseRealTime()->Iterations(100);
BENCHMARK(BM_Frame_Light)->Unit(benchmark::kMillisecond)->UseRealTime()->Iterations(100);

} // namespace
