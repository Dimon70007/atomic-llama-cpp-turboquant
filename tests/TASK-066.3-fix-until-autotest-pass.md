# TASK-066.3 — Fix until autotests PASS

**Status:** ✅ (2026-11-06) — Gap A empty-ne filter aligned with pr28164  
**Depends on:** [TASK-066.2](TASK-066.2-compose-autotests.md) tests composed (I3 was RED)  
**Next:** [TASK-066.4-short-model-s8-s1024.md](TASK-066.4-short-model-s8-s1024.md)  
**Evidence:** `logs/task066-0663-after-empty-filter` — `SUITE=cb-fusion` **PASS**

## Goal

Surgical code fixes so the **066.2** autotests are green. One gap at a time.

## DoD

- [x] `SUITE=cb-fusion` **PASS** on Metal (`BUILD_DIR=build-task066-cb-ws`).
- [x] Discrete change logged HELP (empty-ne filter).
- [x] Did not start model ladder while tests were RED.

## Change (HELP)

**File:** `ggml/src/ggml-metal/ggml-metal-fusion.cpp` — `ggml_metal_fusion_max`

- Treat `ggml_is_empty(node)` like empty op: return 1 at start.
- When collecting idxs, skip `ggml_is_empty` tensors (match `pr28164.diff`).

**Before:** `empty_ne_start` fusion_max=2 (ADD_CHAIN over-pack).  
**After:** fusion_max=1; full cb-fusion suite PASS.

## Checklist

- [x] Rebuild Metal test target.
- [x] Fix Gap A from 066.1 / I3.
- [x] Did not disable multi-CB.
- [x] TQ / other TASK-066 pins untouched.
- [x] Re-run 066.2 suite to PASS.

## Commands

```bash
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
```
