# TASK-066.1 — Diff-audit PR #31688 (+ #28164 fusion_max)

**Status:** ✅ **PARTIAL** (2026-11-06) — notes: [TASK-066.1-audit-notes.md](TASK-066.1-audit-notes.md)  
**Depends on:** —  
**Next:** [TASK-066.2-compose-autotests.md](TASK-066.2-compose-autotests.md)  
**Upstream:** [#31688](https://github.com/ggml-org/llama.cpp/pull/31688) (open), [#28164](https://github.com/ggml-org/llama.cpp/pull/28164) (merged fuse table)  
**Artifacts:** `logs/upstream-prs/pr31688.diff`, `upstream-ggml-metal-context.m`, `pr28164.diff`

## Goal

Prove or refute that this fork’s CB-align + `ggml_metal_fusion_max` match upstream behaviorally — not merely that files exist.

## DoD

- [x] Write `tests/TASK-066.1-audit-notes.md` with verdict **MATCH / PARTIAL / MISSING** and file:line evidence.
- [x] Checklist below completed.
- [x] No model ladder in this subtask.

## Checklist

- [x] Diff `ggml/src/ggml-metal/ggml-metal-context.m` CB block vs `logs/upstream-prs/pr31688.diff` / `upstream-ggml-metal-context.m`.
- [x] Verify: `cb_start` sizing, `n_rest`/`n_per_cb` math, `i += ggml_metal_fusion_max`, encode ranges `[0,cb_start[0])` / `[cb_start[i], cb_start[i+1])`, old `n_nodes_1` path gone.
- [x] Diff `ggml_metal_fusion_max` + `ggml-metal-fusion.{h,cpp}` vs #28164 artifacts.
- [x] Call out gaps: STRUCTURAL vs compute-phase patterns; encoder fuse in `ggml-metal-ops.cpp` not sharing rules with `fusion_max`; stubs under-reporting fuse length → mid-group CB splits.
- [x] Note whether draft criticism of ADD-only heuristic at `58d1940` is stale vs HEAD (`16044ea6a` table).

## Summary

- **#31688 CB walk / encode:** MATCH (not the missing piece for S8 vs S1024 by itself).
- **Fusion table present:** yes; ADD-only draft criticism **STALE**.
- **PARTIAL:** (A) `fusion_max` empty-tensor filter vs `pr28164.diff`; (B) MMA stubs → FULL never fuses MUL_MAT+ADD; (C) ops still hand-roll fuse vs `fusion_next`.
- **066.2 must** harden autotests for A/C (and honest MUL_MAT+ADD expectations) before claiming KEEP.
