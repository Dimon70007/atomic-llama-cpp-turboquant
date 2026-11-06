# TASK-066.7 — MUL_MAT residual vs golden N=8 (**KEEP — primary gate GREEN**)

**Status:** 🟢 **KEEP** (2026-11-06) — `SUITE=mul-mat-residual` PASS  
**Was:** REOPENED; `{256,512,1024}` RED vs N=8 (mad ~1e-4)  
**Depends on:** 066.1–066.4 done; residual suite extended (vs N=8).  
**Inventory:** [TASK-066-mul-mat-batch-inventory.md](TASK-066-mul-mat-batch-inventory.md)  
**Hist:** [task066-hist/TASK-066-p22-27441-audit.md](task066-hist/TASK-066-p22-27441-audit.md), [TASK-066-p22-30100-recheck.md](task066-hist/TASK-066-p22-30100-recheck.md)  
**Suite:** `SUITE=mul-mat-residual` → `test-mul-mat-batch-invariance --residual`

## Goal

Make Metal `MUL_MAT` / `MUL_MAT_ID` **row0 bit-exact vs golden N=8** for  
`ne11` / `ne21` ∈ `{9,16,32,256,512,1024}` (Q8_0 + IQ4_XS; MoE Q8_0).

## Root cause (evidence)

| Diag | Setting | Residual |
|------|---------|----------|
| A | `MUL_MM_MIN=100000` (all `mul_mv`) | PASS — mv is batch-invariant |
| B | `MUL_MM_MIN=1` (all `mul_mm`) | PASS — **stock mm is batch-invariant** |
| C | default min=32 | FAIL — golden N=8 on **mv**, large N on **mm** (cross-kernel) |

Identical mad across `{256,512,1024}` = same mm row0, not width-dependent mm noise.  
No `MUL_MM_MIN` can satisfy both residual-vs-N=8 **and** vs-N=1 with width 8 while mm≠mv.

## Levers

| # | Change | Residual | `SUITE=mul-mat` | Verdict |
|---|--------|----------|-----------------|---------|
| 1 | Float-tile `mul_mm` / `mul_mm_id` for `q8_0` + `iq4_xs` (size-aware sb; smem 12288) | mad ~1e-4 → **~1 ULP**; still RED at min=32 | GREEN | **HELP** |
| 2 | Product default `GGML_METAL_MUL_MM_MIN` **32 → 7** (`ne11>=8` → mm); residual gate 7; vs-N=1 widths `{2,3,4,5,7}` | **GREEN** | **GREEN** | **KEEP** |

`#27441` skinny not re-applied (prior REGRESS). Raising min to hide mm still forbidden.

## Primary gate (lock)

**Do not burn e2e / ubatch ladder / fullgraph / MTP while iterating 066.7.**

| Gate | Role |
|------|------|
| `SUITE=mul-mat-residual` | **Primary** — GREEN after KEEP |
| `SUITE=mul-mat` / `SUITE=cb-fusion` | Regression smoke after KEEP |
| `run-task066-ubatch-ladder.sh` S8 vs S1024 | **Optional confirm** after residual GREEN |
| `fullgraph` / `mtp-*` | Out of loop (MTP blocked; fullgraph separate) |

## Forbidden

- Raising `MUL_MM_MIN` (e.g. to 256) to hide mm as a “fix”.
- Blind re-apply of `#27441` without a corrected hypothesis (prior REGRESS stamp `175708Z`).
- Model-only / e2e-only KEEP without residual GREEN.
- Skipping autotest RED→GREEN (066.0 lock 7).

## DoD

- [x] `SUITE=mul-mat-residual` **PASS** on Metal (all listed widths vs N=8).
- [x] Inventory table updated with KEEP / residual notes.
- [x] HELP/NO_EFFECT/REGRESS log per lever.
- [ ] Optional: short ladder S8 vs S1024 after GREEN (confirm, not substitute).
- [x] Update [TASK-066.0-overview.md](TASK-066.0-overview.md) + [README-TASK066.md](README-TASK066.md).
- [x] MTP remains blocked until progressive DoD met.

## Commands

```bash
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
# After GREEN only:
TASK066_UBATCH_LIST="8 1024" TASK066_REPEAT_LARGE="" \
  BUILD_DIR=build-task066-cb-ws ./tests/run-task066-ubatch-ladder.sh
```

Stamps: `logs/task066-0667-lever1-float-tile-residual.log` (HELP),  
`logs/task066-metal-corners-mul-mat-residual-20261106T210344Z` (KEEP GREEN).
