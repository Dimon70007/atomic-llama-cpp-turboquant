# TASK-066.7 — MUL_MAT residual vs golden N=8 (**REOPENED — primary**)

**Status:** 🔴 **REOPENED** (2026-11-06) — active engineering track  
**Was:** “only if needed after 066.6”; **now:** primary gate for S8 vs large-ubatch class  
**Depends on:** 066.1–066.4 done; residual suite extended (vs N=8). **Does not** wait for 066.5/066.6 e2e.  
**Inventory:** [TASK-066-mul-mat-batch-inventory.md](TASK-066-mul-mat-batch-inventory.md)  
**Hist:** [task066-hist/TASK-066-p22-27441-audit.md](task066-hist/TASK-066-p22-27441-audit.md), [TASK-066-p22-30100-recheck.md](task066-hist/TASK-066-p22-30100-recheck.md)  
**Suite:** `SUITE=mul-mat-residual` → `test-mul-mat-batch-invariance --residual`

## Goal

Make Metal `MUL_MAT` / `MUL_MAT_ID` **row0 bit-exact vs golden N=8** for  
`ne11` / `ne21` ∈ `{9,16,32,256,512,1024}` (Q8_0 + IQ4_XS; MoE Q8_0).

Current baseline (M4, default `MUL_MM_MIN=32`): `{9,16,32}` **PASS**; `{256,512,1024}` **FAIL** (mad ~1e-4).

## Primary gate (lock)

**Do not burn e2e / ubatch ladder / fullgraph / MTP while iterating 066.7.**

| Gate | Role |
|------|------|
| `SUITE=mul-mat-residual` | **Primary** — iterate until GREEN |
| `SUITE=cb-fusion` / `SUITE=all` | Regression smoke after KEEP (optional mid-loop) |
| `run-task066-ubatch-ladder.sh` S8 vs S1024 | **Confirm only after residual GREEN** — not the daily loop |
| `fullgraph` / `mtp-*` | Out of loop (MTP blocked; fullgraph separate) |

## Forbidden

- Raising `MUL_MM_MIN` (e.g. to 256) to hide mm as a “fix”.
- Blind re-apply of `#27441` without a corrected hypothesis (prior REGRESS stamp `175708Z`).
- Model-only / e2e-only KEEP without residual GREEN.
- Skipping autotest RED→GREEN (066.0 lock 7).

## Evidence hierarchy (066.0 locks 7–8)

1. Hypothesis → (extend residual corner if needed) → **RED** on current HEAD for that gap.  
2. Minimal Metal/kernel fix → same suite **GREEN** (or HELP with logged residual cells).  
3. Only then optional short ladder confirm.  
4. HELP / NO_EFFECT / REGRESS; REGRESS → revert before next lever.

## DoD

- [ ] `SUITE=mul-mat-residual` **PASS** on Metal (all listed widths vs N=8).
- [ ] Inventory table updated with KEEP / residual notes.
- [ ] HELP/NO_EFFECT/REGRESS log per lever.
- [ ] Optional: short ladder S8 vs S1024 after GREEN (confirm, not substitute).
- [ ] Update [TASK-066.0-overview.md](TASK-066.0-overview.md) + [README-TASK066.md](README-TASK066.md).
- [ ] MTP remains blocked until progressive DoD met.

## Checklist

- [ ] Localize: mm path / reduction / split-K vs pin boundary (one lever).
- [ ] Re-evaluate `#27441` only with corrected port + residual still the gate.
- [ ] `#30100` only if encode path appears.
- [ ] Any `MUL_MM_MIN` product default change updates residual suite policy + this file.
- [ ] Do **not** require e2e between levers.

## Commands

```bash
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
# After GREEN only:
TASK066_UBATCH_LIST="8 1024" TASK066_REPEAT_LARGE="" \
  BUILD_DIR=build-task066-cb-ws ./tests/run-task066-ubatch-ladder.sh
```
