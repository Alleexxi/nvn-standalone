# NVN Standalone — Eglantine NvnClean Vulkan Backend (Isolated)

A standalone CMake project that extracts the NvnClean Vulkan backend from the Eglantine emulator
for independent development, testing, and iterative improvement.

## Components

| Module | Files | Description |
|--------|-------|-------------|
| **nv_sync_manager** | `.h` + `.cpp` (131KB) | GPU synchronization, timeline semaphores, query resolution, conditional rendering |
| **nv_mempool** | `.h` + `.cpp` (82KB) | NVN memory pool management, GPU address translation, VMA integration |
| **nv_pass_manager** | `.h` + `.cpp` (74KB) | Render pass management, dynamic rendering, attachment state |
| **nv_command_recorder** | `.h` + `.cpp` (84KB) | NVN → Vulkan command translation, draw recording |
| **nv_bindless** | `.h` + `.cpp` (64KB) | Bindless resource management, descriptor indexing |

## Quick Start

```powershell
# From a VS Developer Command Prompt (or set MSVC env vars):
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
.\build\nvn_tests.exe
```

## Dependencies (auto-fetched via CPM)

- **Vulkan Headers** v1.4.345
- **VulkanMemoryAllocator (VMA)** v3.3.0
- **fmt** v12.1.0
- **GoogleTest** v1.17.0

## Project Structure

```
nvn-standalone/
├── CMakeLists.txt              # Root build configuration
├── src/
│   ├── common/                 # Stub replacements for Eglantine's common/ headers
│   │   ├── alignment.h         # AlignUp, AlignDown, IsAligned, etc.
│   │   ├── assert.h            # ASSERT, UNREACHABLE macros
│   │   ├── common_types.h      # u8/u16/u32/u64/s8-s64/f32/f64 typedefs
│   │   ├── logging.h           # LOG_ERROR/LOG_WARNING → stderr
│   │   ├── swap.h              # Byte swap utilities
│   │   ├── thread.h            # Event, Barrier, thread priority stubs
│   │   ├── settings.h          # Minimal settings interface
│   │   └── container/
│   │       └── unordered_map.h # std::unordered_map alias
│   ├── video_core/
│   │   ├── vulkan_common/      # Vulkan infrastructure (mix of real + stubs)
│   │   │   ├── vulkan.h        # Platform defines + vulkan.h include
│   │   │   ├── vulkan_wrapper.h # Real vk:: RAII wrappers (from Eglantine)
│   │   │   ├── vulkan_device.h # Real Device class (from Eglantine, NsightAftermath stubbed)
│   │   │   └── vma.h           # VMA include wrapper
│   │   └── nvn/                # NVN guest types (from Eglantine)
│   │       ├── nvn_types.h     # NVN enums, structs, type aliases
│   │       ├── record.h        # NVN command record format
│   │       ├── sink.h          # NVN replay sink interface
│   │       └── guest_memory.h  # Guest memory abstractions
│   ├── nvn_clean/              # THE CODE UNDER TEST (from Eglantine)
│   │   ├── nv_sync_manager.{h,cpp}
│   │   ├── nv_mempool.{h,cpp}
│   │   ├── nv_pass_manager.{h,cpp}
│   │   ├── nv_command_recorder.{h,cpp}
│   │   └── nv_bindless.{h,cpp}
│   └── stubs/                  # Stub .cpp files for non-inline functions
│       ├── common_stubs.cpp    # Logging + assert implementations
│       ├── vulkan_device_stub.cpp
│       ├── vulkan_wrapper_stub.cpp
│       ├── nvn_types_stub.cpp
│       └── sink_stub.cpp
└── tests/
    ├── test_main.cpp           # GoogleTest entry point
    ├── test_alignment.cpp      # Alignment utility tests
    ├── test_mempool.cpp        # Memory pool logic tests
    └── test_sync_manager.cpp   # SyncManager type/layout tests
```

## For Iterative Improvement Agents

### The loop

```
1. pick ONE hypothesis (see "Known hotspots")
2. edit src/nvn_clean/*
3. check.cmd quick        -> correctness only (~30 s). Must print ALL CHECKS PASSED.
4. check.cmd              -> correctness + benchmarks + compare vs bench/baseline.json
5. kept?  improved metric, nothing regressed > 5 %, all tests green
          -> copy bench/results/current.json over bench/baseline.json, log it in bench/LOG.md
   not kept? revert the change, log why in bench/LOG.md
6. repeat
```

**Done means:** `check.cmd` exits 0. That covers a clean build, all `nvn_tests`, all `nvn_vulkan_tests`
with zero validation messages, and no benchmark slower than the baseline beyond its noise limit. Paste the
`compare.py` table into your report.

Rules:
- Never delete or weaken a test so that a change passes. If a test is wrong, say why in the report.
- Every new behaviour gets a GPU test that checks an observable result (bytes in guest memory,
  semaphore values, GPU ordering). "Doesn't crash" is not a test.
- Benchmarks always run with `NVN_NO_VALIDATION=1` (`check.cmd` sets it).
- `compare.py` is noise-aware: a benchmark fails when it is slower by more than max(5 %, 3 x its
  run-to-run cv). Small batches (`/64`) and `RecordFenceSync` vary 12-15 %, so they only catch big
  regressions. Judge wins on the `BM_Frame*` / `BM_FramePart*` medians, which are stable.
- Install the Vulkan SDK so the validation layer is active. Without it, tests print
  `validation OFF` and API misuse goes unnoticed.

### Commands

| What | Command |
|---|---|
| Build everything | `build.cmd -DNVN_ENABLE_VULKAN_TESTS=ON -DNVN_ENABLE_BENCHMARKS=ON` (the options stick) |
| CPU unit tests | `build\nvn_tests.exe` |
| GPU behaviour tests | `build\nvn_vulkan_tests.exe` (filter: `--gtest_filter=*Conditional*`) |
| All benchmarks | `set NVN_NO_VALIDATION=1` then `build\nvn_bench.exe` |
| One benchmark | `build\nvn_bench.exe --benchmark_filter=Frame` |
| Compare | `python bench\compare.py bench\baseline.json bench\results\current.json` |
| Full gate | `check.cmd` / `check.cmd quick` |

### Benchmarks (`bench/bench_sync_manager.cpp`)

| Tier | Benchmarks | Metric that matters |
|---|---|---|
| Micro | `BM_AllocateNextTimelineValue`, `BM_RecordFenceSync`, `BM_FindSyncPoint`, `BM_ResolveWait_*`, `BM_IsTimelineValueComplete_*`, `BM_WaitSyncHost_*` | ns per call (replay-thread CPU) |
| Batch | `BM_Batch_{Timestamps,ReportValues,OcclusionCycles}/N` | `cpu_ns_per_op`, `gpu_us` for N ops |
| Frame | `BM_Frame_OoTMarket`, `BM_Frame_Light` | `Time` = replay CPU ms/frame, `gpu_us` = GPU us/frame |
| Breakdown | `BM_FramePart_{Occlusion,Timestamps,Conditions,SyncOnly}` | which op family costs the GPU time |

`gpu_us` measures SyncManager's own GPU work (no draws), timed with timestamps around the frame.
It includes idle gaps between submits. `BM_FramePart_SyncOnly` (~0.4 ms) is roughly that floor.

### Call volumes (what to optimise for)

Measured on the **legacy** VTX-HLE path, OoT market, RTX 5080, per frame
(`flash/data/report-nvnreplay.txt`, `flash/data/vtx-deep-brief.md`). The NvnClean path has not been
profiled in-game yet, so treat these as the expected load.

| Operation | Per frame | SyncManager entry point |
|---|---|---|
| Draws | ~2,970 | (context: ~320 replay calls per draw) |
| Command-buffer submits | ~24 | `AllocateNextTimelineValue` + `MakeTimelineSignalSubmitInfo` + `NotifyTimelineSubmitted` + `FlushQueryResolves` |
| Occlusion begin/end | ~1,800 | `EnsureOcclusionQueryActive` / `CloseActiveOcclusionQuery` |
| SAMPLES_PASSED reports | ~1,950 | `RecordReportCounter(kSamplesPassed)` |
| Timestamp writes (all) | ~2,000 | `RecordReportCounter(kTimestamp*)`, `RecordReportValue` |
| Render conditions | ~1,200 | `BeginConditionalRendering` / `EndConditionalRendering` |
| Counter resets | ~1,800 (most coalesce) | `RecordResetCounter` |
| Resolve batches (legacy) | ~19-22 | -> `FlushQueryResolves` |
| FenceSync / WaitSync | not measured; ~1 per submit assumed | `RecordFenceSync`, `ResolveQueueWaitSync` |
| Host sync waits (`nvnSyncWait`) | not measured; few per frame | `WaitSyncHost` |

Budget: 60 FPS = 16.7 ms per frame for everything (guest, replay, GPU). SyncManager should cost
well under 0.5 ms of replay CPU and under 0.5 ms of GPU per frame.

### Baseline numbers (RTX 5080, Ryzen, 2026-11-19)

| Benchmark | CPU | GPU |
|---|---|---|
| `BM_Frame_OoTMarket` | ~2.8 ms | **~14-16 ms** |
| `BM_FramePart_Conditions` (incl. occlusion) | ~2.7 ms | **~15.1 ms** |
| `BM_FramePart_Occlusion` | ~0.9 ms | ~1.9 ms |
| `BM_FramePart_Timestamps` | ~0.6 ms | ~0.57 ms |
| `BM_Batch_Timestamps/4096` | 57 ns/op | 1.68 ms |
| Micro sync calls | 4-80 ns | - |

`bench/baseline.json` holds the exact medians. `bench/baseline.txt` holds the readable table.

### Known hotspots (start here)

1. **Conditions: about 13 ms of GPU per frame.** Each `BeginConditionalRendering` pushes one op and
   calls `FlushQueryResolves`, which gives one copy, barriers and one dispatch per condition, so
   ~1,200 tiny serial dispatches. Ideas: evaluate all predicates of a segment in one dispatch ahead
   of time (guest counters are known when the recorder sees the condition), skip barriers when
   nothing is pending, and reuse predicate slots for the same address.
2. **The resolve shader is single-threaded.** The embedded SPIR-V (`kDefaultQueryResolveSpirv`)
   declares `LocalSize 1 1 1` and is dispatched as `Dispatch(1,1,1)`, so one GPU thread walks every op
   (~0.4 us each). Ops that touch different destinations are independent. Only reset -> accumulate ->
   report on the same queue must keep its order (a per-queue prefix sum works). There is **no GLSL
   source**: write `bench/shaders/query_resolve.comp`, compile it with
   `C:\dev\glslang\bin\glslangValidator -V --target-env vulkan1.2`, and either replace the array or
   pass it through `SyncManagerConfig::custom_resolve_spirv`. The push-constant and `GpuQueryResolveOp`
   layouts are fixed by `static_assert`s.
3. **Occlusion cycles cost ~116 ns of CPU per op** against ~55 ns for plain reports. Look at
   `AcquireActiveQueryChunk` and the per-op `pending_resolve_ops_.push_back`.
4. **`RecordFenceSync` costs 68-80 ns** (an `unordered_map` behind a `shared_mutex`). Serials are
   mostly sequential, so a ring/flat table could reach ~5 ns.

### What the tests don't cover yet

- Occlusion with real draws (needs a graphics pipeline). Only the "no draws = 0 samples" case is tested.
- `PassManager` interaction (`FlushQueryResolves` / `BeginConditionalRendering` ending an open pass).
- `nv_mempool`, `nv_pass_manager`, `nv_command_recorder` and `nv_bindless` have no GPU tests or
  benchmarks. Add them in the same style before optimising those modules.
- Multi-threaded `WaitSyncHost` against concurrent `RecordFenceSync`.

### Reference: another NVN implementation

`reference/nxnative-nvn/` holds nxnative's NVN-on-Metal layer (MIT, not built). Use it to check NVN
behaviour, for example sync/event/counter semantics or the prototype list. Read its `SOURCE.md` first.

### Porting back

A kept change is only real once it is ported to `C:\dev\eglantine-vtxhle` and the game agrees: run the
OoT scenario (`tools\vtx_ab.ps1`, see `flash/data/RESULTS.md`), then compare FPS avg / 1 % low and
screenshots against the previous build.

## Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `NVN_ENABLE_VULKAN_TESTS` | `OFF` | Build `nvn_vulkan_tests` (needs a Vulkan 1.3 GPU) |
| `NVN_ENABLE_BENCHMARKS` | `OFF` | Build `nvn_bench` (Google Benchmark 1.9.4) |
| env `NVN_NO_VALIDATION=1` | unset | Disable the validation layer (always for benchmarks) |
