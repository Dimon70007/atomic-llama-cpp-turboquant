# TASK-066.5 — Full fusion table parity (draft Step 1)

**Status:** ⬜  
**Depends on:** [TASK-066.4](TASK-066.4-short-model-s8-s1024.md) (or skip-ahead only if 066.4 showed A alone insufficient — still require 066.3 PASS)  
**Next:** [TASK-066.6-cb-and-invariance-gates.md](TASK-066.6-cb-and-invariance-gates.md)  
**Source:** [task066-hist/draft_mtp_research_task-066.md](task066-hist/draft_mtp_research_task-066.md) Step 1

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
