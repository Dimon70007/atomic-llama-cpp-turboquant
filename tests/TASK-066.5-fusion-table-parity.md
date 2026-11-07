# TASK-066.5 — Full fusion table parity (draft Step 1)

**Status:** 🔓 **active** — **primary next lever** after 066.9 localize (H9faOFF RED)  
**Depends on:** [066.7 KEEP](TASK-066.7-mul-mat-residual.md); FA always-tile C2 GREEN; [066.9](TASK-066.9-fa-batch-invariance.md) Phase A localize  
**Next:** C3-n1344 RED→GREEN via fuse/CB (then C1 / ladder); MTP still 066.9 Phase B  
**Source:** [task066-hist/draft_mtp_research_task-066.md](task066-hist/draft_mtp_research_task-066.md) Step 1  
**Hypothesis (post-066.9 localize):** long-prefill multi-step residual for **n≥1344** survives **FA-off** → suspect CB/`fusion_max` vs encoder fuse (or other non-FA ops), not FA kernel shape.

### TDD entry (before any further fuse/ops edit)

1. ~~C2~~ GREEN (always-tile). **C3** n=1280 GREEN / **n=1344 RED** — primary fuse TDD gate (`SUITE=ubatch-s8-s1024`).
2. Localize: FA-off still RED at n=1536 → fuse/ops, not FA-only.
3. FIRST_DIFF_TENSOR / fuse-table mismatch on H10 n=1344 graphs (ub=8 vs one-shot) **before** blind encoder ports.
4. Re-check C2 + C3-n1280 after every edit; no ε-PASS C3.

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
