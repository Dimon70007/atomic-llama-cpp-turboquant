# TASK-066.6 — CB integration + invariance autotest gates (draft Steps 2–3)

**Status:** 🔓 **active** (partial: residual suite GREEN after 066.7; product ladder still DIFF; `SUITE=fullgraph` RED N=1 vs N=2)  
**Depends on:** [TASK-066.5](TASK-066.5-fusion-table-parity.md) (or parallel invariance corner if RED already exists)  
**Next:** short ladder re-confirm after GREEN gates; MTP still blocked  
**Source:** [task066-hist/draft_mtp_research_task-066.md](task066-hist/draft_mtp_research_task-066.md) Steps 2–3  
**Autotest:** `SUITE=mul-mat-residual` GREEN (066.7); **primary product gate now** `SUITE=ubatch-s8-s1024` (greedy token-stream S8 vs S1024; TDD RED expected on HEAD after confirm).  
**Confirm ref:** `logs/task066-ubatch-ladder-20261106T212703Z` S1024 vs S8 DIFF @byte 182  

### Patch / fixture audit (2026-11-06)

| Claim | Evidence | Verdict |
|-------|----------|---------|
| Prior `SUITE=fullgraph` “RED” in status matrix | Log `failed to find a memory slot for batch of size 2048` with `CTX=2048` | **FIXTURE_FAIL**, not N=1 vs N=2 logit proof. Runner now classifies this. |
| `mul-mat-residual` KEEP | residual suite PASS under `MUL_MM_MIN=7` | Still valid; does **not** imply ladder identical |
| Ladder DIFF under product min=7 | stamp `212703Z` | Unlocks this suite; do not treat upstream #31688 / 066.3 alone as Done |

### Localize matrix (stamp `logs/task066-ubatch-localize-20261106T214747Z`)

Product KV `q8_0`/`turbo4` (FA force-on with turbo). `MUL_MM_MIN` default 7.

| Row | Compare | exact | Notes |
|-----|---------|-------|-------|
| H1 | n=512, b=512, **ub=8 vs ub=512** | **0** | Pure ubatch; mad≈20.7 |
| H2 | n=512, product-shaped b/ub | **0** | Same as H1 |
| H3 | full prefill 1696, product pair | **0** | Matches C1 `first_token_diff=0` |
| H4 | H1 with FA=off request | **0** | Invalid ablation — turbo re-enables FA |
| H5 | n=8, ub=8 vs ub=8 | **1** | Sanity |
| H6 | n=16, ub=8 (2 steps) vs ub=16 | **1** | Still bit-exact |
| H7 | n=32, ub=8 (4 steps) vs ub=32 | **0** | **Minimal RED** (mad≈3.36) |

**Conclusion:** residual class is **multi-step ubatch vs one-shot** of the same token count; threshold between 16 (GREEN) and 32 (RED). Not explained by mul-mat residual alone. Next: FIRST_DIFF_TENSOR / op-class on H7; do not disable multi-CB as a “fix”.

**Fast gate:** C2 in `test-ubatch-s8-s1024-row0-parity` = H7 (runs before product C1).

### Lever: `GGML_METAL_MUL_MM_MIN` on C2 (stamp `logs/task066-ubatch-c2-levers-20261106T215032Z`)

| min | C2 exact | mad | argmax |
|-----|----------|-----|--------|
| 1 | 0 | ≈3.36 | 569 vs 236761 |
| 7 (default) | 0 | ≈3.36 | same |
| 32 | 0 | ≈5.36 | **same argmax** |

**NO_EFFECT** on C2 decision — 066.7 mm KEEP is orthogonal to this residual class.

### FA / KV ablation on C2 (stamp `logs/task066-ubatch-fa-ablation-20261106T215517Z`)

`TASK066_DIAG=fa-ablation` — **localization only**. Product path stays **FA-on + ctv=turbo4**.

| Row | ctk / ctv / FA | C2 exact | mad | Notes |
|-----|----------------|----------|-----|-------|
| A0 | q8_0 / turbo4 / on | **0** | ≈3.36 | product RED (argmax 569 vs 236761) |
| A1 | q8_0 / **f16** / on | **1** | 0 | turbo4→f16 alone clears C2 under FA-on |
| A2 | q8_0 / f16 / **off** | **1** | 0 | true FA-off (non-turbo V) |
| A3 | f16 / f16 / off | **1** | 0 | |
| A4 | f16 / f16 / on | **0** | ≈0.017 | soft drift; **same argmax** |

**Read (agreed):** base residual is **inside FA under multi-step ubatch** (A4: f16/f16+FA-on still `exact=0`, soft mad, same argmax). **KV quantization (turbo4) amplifies** that FA error into decision-level DIFF (A0). A1 (FA-on+f16) bit-exact on C2 does **not** mean “fix = drop turbo”; product stays FA-on+turbo4 — fix the FA multi-step invariance, then re-check turbo amplification. FA-off GREEN is localization only.

Fail-closed: `TASK066_FA=off` without non-turbo `TASK066_CTV` → exit 2 (llama would force FA back on).

### Web research (FA-on fix patterns)

SSOT dump: [`task066-hist/TASK-066-fa-batch-invariance-web-research.md`](task066-hist/TASK-066-fa-batch-invariance-web-research.md).

Shortlist: Thinking Machines fixed split-KV / consistent KV layout before attn; #25618 Metal **nsg pin** (fork already pins FA-vec `nsg=1` — C2 still RED ⇒ look beyond vec-only / prefill tile path); Vulkan #26358 no GQA pack for `neq1>1`; avoid shape-keyed FA tuning that changes reduction by `ne01`.

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
