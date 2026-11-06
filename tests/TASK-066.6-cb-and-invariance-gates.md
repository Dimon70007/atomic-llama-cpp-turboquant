# TASK-066.6 — CB integration + invariance autotest gates (draft Steps 2–3)

**Status:** ⬜ (partial progress: `SUITE=mul-mat-residual` now fails closed on N∈{256,512,1024} vs N=8)  
**Depends on:** [TASK-066.5](TASK-066.5-fusion-table-parity.md)  
**Next:** [TASK-066.7-mul-mat-residual.md](TASK-066.7-mul-mat-residual.md) only if S8 vs S1024 still large  
**Source:** [task066-hist/draft_mtp_research_task-066.md](task066-hist/draft_mtp_research_task-066.md) Steps 2–3  
**Autotest:** `test-mul-mat-batch-invariance --residual` — golden N=8; widths 9/16/32/256/512/1024 (RED on large as of 2026-11-06)

## Goal

1. Every CB boundary uses the corrected `fusion_max` (autotests break if boundaries ignore fusion).  
2. Strengthen batch/ubatch (or graph-size) **invariance autotests** as the primary FAIL/PASS for this track.  
3. Short model remains confirmation after green tests — not a substitute.  
4. Never disable multi-CB as a “fix”. No premature WAIVE.

## Evidence hierarchy (066.0 lock 7)

1. **Each** CB/invariance change set includes new/extended autotests (RED→GREEN on that gap).
2. Suite PASS → short model `{8,1024}` (+ `{256,512}`).

## DoD

- [ ] Autotests prove CB boundaries cannot split fused groups for covered patterns.
- [ ] Stronger invariance suite named and wired (`SUITE=…`); documented invariants (**new cases for new claims**).
- [ ] Short model stamp after PASS; S8 vs S1024 metric recorded.
- [ ] If still large residual → unlock 066.7; else mark track progressive DoD and keep MTP gated on product judgment in overview.
- [ ] No KEEP without accompanying autotest diff.

## Checklist

- [ ] CB walk still matches #31688 intent after 066.5 table changes.
- [ ] Add/extend invariance corners (graph-size / ubatch-class) beyond structural fuse-only checks **before** claiming KEEP.
- [ ] `SUITE=cb-fusion` + new invariance suite PASS.
- [ ] Short ladder confirm; compare to 066.4 / baseline `170944Z`.
