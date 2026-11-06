# TASK-066.5 — Full fusion table parity (draft Step 1)

**Status:** 🔓 **active** (unlocked 2026-11-06 after ladder confirm DIFF@182)  
**Depends on:** [TASK-066.4](TASK-066.4-short-model-s8-s1024.md) + [066.7 KEEP](TASK-066.7-mul-mat-residual.md); confirm stamp `logs/task066-ubatch-ladder-20261106T212703Z`  
**Next:** [TASK-066.6-cb-and-invariance-gates.md](TASK-066.6-cb-and-invariance-gates.md)  
**Source:** [task066-hist/draft_mtp_research_task-066.md](task066-hist/draft_mtp_research_task-066.md) Step 1  
**Hypothesis (post-066.7):** remaining S8 vs S1024 text DIFF is **not** mv↔mm (residual GREEN under `MUL_MM_MIN=7`); suspect CB/`fusion_max` vs encoder fuse parity on large vs small ubatch graphs.

### TDD entry (before any further fuse/ops edit)

1. ~~Prove product-class RED: C2~~ → **C2 GREEN** after FA always-tile (2026-11-07); primary open gate is **C1** greedy S8 vs S1024 (HELP `first_token_diff` 0→8).
2. ~~Localize: n=16 GREEN / n=32 RED~~ → root cause was FA **vec vs tile** on `ne01<20`; closed in ops, not fusion table.
3. Next fuse work: encoder≠`fusion_max` only if C1 still RED after tile + evidence points at CB/fuse (FIRST_DIFF_TENSOR on C1).
4. Re-check every 066 patch against **C2 GREEN** + C1 (do not regress C2).

## Goal

`ggml_metal_fusion_max` must reflect **all** fuse patterns encoders actually use (`MUL_MAT+ADD`, norms, …), not a partial STRUCTURAL / ADD-only subset.

## Evidence hierarchy (066.0 lock 7)

1. **New/extended autotest per gap** (RED on current encoder≠table mismatch) **in the same change set** as any ops/fusion edit.
2. Fix → same autotest GREEN → only then short model `{8,1024}` confirm.

## DoD

- [ ] Autotest coverage for the patterns claimed fused by Metal encoders (**new cases if not already RED on the gap**).
- [ ] Intentional fork deltas (TurboQuant / TASK-066 pins) documented in this file or 066.1 notes.
- [ ] Short model confirm only after autotest PASS; log HELP/NO_EFFECT/REGRESS vs prior 066.4 stamp.
- [ ] No KEEP without the accompanying autotest diff.

## Checklist

- [ ] Compare fuse table / encoder loops (`ggml-metal-ops.cpp`) with upstream #28164 rules; list mismatches as named test cases first.
- [ ] For each mismatch: write RED corner → surgical port → GREEN (one gap at a time).
- [ ] Do not blind-apply huge diffs over TQ.
- [ ] Short ladder subset only after suite PASS.
