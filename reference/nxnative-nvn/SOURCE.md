# nxnative `dm/gfx/nvn` (reference only)

- Upstream: https://github.com/seedds/nxnative/tree/main/dm/gfx/nvn
- Commit: 130ef2936771cbeb411d3ea09b418549fba25878 (copied 2026-11-19)
- License: MIT, see `LICENSE` (Copyright (c) 2026 seedds). Some files carry their own notices:
  code by h1k421 (MIT) and an algorithm ported from deko3d `dk_image.cpp` (zlib, fincs).
  Upstream also credits d2r-mobile (MIT) and MacNativeGames (MIT, `nvn_mtl.m`).
- Contents copied unmodified. Only this folder was taken, nothing else from the repo.

## What it is

An independent NVN implementation on **Metal** (Apple silicon): queues, syncs, events, counters,
timestamps, recording, textures, pipelines. It is **not built** here (Objective-C/Metal/ARM asm).

## How agents should use it

- Use it to answer "what does NVN actually do here?" questions: sync/fence vs queue completion
  (`nvn_queue.*`), counters and reports (`nvn_count.h`, `nvn_timestamp.h`), recording (`nvn_rec.h`),
  and the API prototype list (`nvn_protos.h`) for cross-checking `src/video_core/nvn/*.inc`.
- Port **behaviour**, never structure: Metal design choices do not map to Vulkan.
- Never copy code from here into `src/`. If nxnative and our tests disagree, first write a GPU test
  that pins down the behaviour, then fix whichever side is wrong.
