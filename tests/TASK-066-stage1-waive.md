# TASK-066 Stage 1 — WAIVE (large ubatch vs S8)

**Status:** ACCEPTED (2026-11-06); extended to **S1024/S2048** same day  
**Authority:** [draft_mtp_research_task-066_answers.md](draft_mtp_research_task-066_answers.md) LOCK + [TASK-066-execution-plan.md](TASK-066-execution-plan.md) P2.3/P2.4  
**Fixture:** [TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md) (Gemma4 UD-IQ4_XS, ~2k Pierre/Dolokhov, FA on, KV `q8_0`/`turbo4`, temp=0)

## Decision

**WAIVE** bit-exact match of owned large-ubatch band **`{256, 512, 1024, 2048}`** to lab golden **S8** (`ubatch<=8`) on this fixture.

Stage 1 DoD = noise owned + explicit WAIVE, **not** bit-exact to S8. Ready upstream Metal mm patches for this path are exhausted. Do **not** invent break-even knobs (e.g. `MUL_MM_MIN=256`) as a substitute fix.

Owned large band means: same-ubatch clean, residual vs S8 documented, no further Stage 1 mm ports required for these cells.

## What is waived

| Compare | Scope | Notes |
|---------|-------|-------|
| S256 vs S8 | product floor vs lab golden | stock `mul_mm`; first DIFF @byte **80** |
| S512 vs S8 | product target vs lab golden | same opening DIFF @byte **80** |
| S1024 vs S8 | large owned cell vs golden | DIFF @byte **80** (`170944Z`) |
| S2048 vs S8 | large owned cell vs golden | DIFF @byte **80**; **S2048 == S1024** |
| S256 vs S512 | within owned band | first DIFF @byte **314**; not a same-ubatch flake |
| S512 vs S1024 | within owned band | first DIFF @byte **134** (`170944Z`) |

**Noise already low at 1024/2048 (no extra Stage 1 work):** same-ubatch run1=run2 identical; **S1024 == S2048** (quietest large pair). Residual vs S8 is the same opening fork class as S256/S512.

**Not waived as product blockers (diagnostic only):** S16/S33 under debug pin. Under product default `GGML_METAL_MUL_MM_MIN=32`, S16==S8 (`mul_mv`); S33 DIFF@253 is mid-ladder diagnostic, not the prod gate.

## Evidence (script stamps; do not rebuild matrix by hand)

| Stamp | Pin | Role |
|-------|-----|------|
| `logs/task066-ubatch-ladder-20261106T170944Z` | `MUL_MM_MIN=8` | Full ladder. S1=S8; same-ubatch OK (S1024/S2048); S256/S512/S1024/S2048 vs S8 DIFF@80; **S256 vs S512 DIFF@314**; S512 vs S1024 DIFF@134; **S1024=S2048** |
| `logs/task066-ubatch-ladder-20261106T172821Z` | default 32 | A/B. S16=S8; S33 DIFF@253; S256 DIFF@80. S256 text == `170944Z` S256 |
| `logs/task066-ubatch-ladder-20261106T181846Z` | default 32 | Post-#27441-revert reconfirm == `172821Z` on S8/S33/S256 |
| `logs/task066-ubatch-ladder-20261106T184055Z` | default 32 | Prod-band completeness `{8,256,512}`: S256/S512 vs S8 DIFF@80; S256 vs S512 DIFF@314; S8/S256/S512 == prior baselines |
| `logs/task066-ubatch-ladder-20261106T175708Z` | min=8 + #27441 | **Not baseline** — regression evidence only (CLOSED) |

Cross-check: S256 identical across `170944Z` / `172821Z` / `181846Z` / `184055Z`; S512 identical `170944Z` == `184055Z` (large `ne11` already on `mul_mm`; pin 8 vs 32 does not change the large-band residual).

## Why WAIVE (not localize further for Stage 1)

1. **Same-ubatch clean** on large cells including 1024/2048 — residual is cross-ubatch / kernel-path, not #19034 flake.
2. **Upstream mm list exhausted for Stage 1:** `#28164`/`#31688` in tree; `#27441` tried + regressed + **CLOSED** ([TASK-066-p22-27441-audit.md](TASK-066-p22-27441-audit.md)); `#30100` **N/A** ([TASK-066-p22-30100-recheck.md](TASK-066-p22-30100-recheck.md)).
3. Metal `MUL_MAT` is **not** batch-invariant by design (kernel choice depends on `ne11`); full invariance is out of scope (answers.md A3).
4. Operator LOCK: at target **512** (and large cells), text drift vs smaller ubatch can be **quality-favorable**; DoD does not force-match S8.
5. Further FA/MoE localization is optional only if a future product need demands a quieter band; it is **not** required to close Stage 1 or unlock Stage 2.

## Product defaults after WAIVE

| Knob | Value |
|------|-------|
| `GGML_METAL_MUL_MM_MIN` | default **32** (B5; S16 diagnostic-quiet) |
| Production ubatch | floor **256**, target **512**; owned large also **1024 / 2048** |
| Lab golden | `ubatch<=8` (S8 preferred) |
| Unit residual | `SUITE=mul-mat-residual` stays GREEN at default 32 |

## Unit gates (Stage 1 close)

Post-revert stamps (`BUILD_DIR=build-task066-cb`):

- `SUITE=all` **PASS** -> `logs/task066-post-revert-suite-all`
- followup: cb-fusion / B1 / mul-mat-residual **PASS**; `mtp-acc-trace` FAIL without log (expected); `mtp-e2e` SKIP

`followup` overall VERDICT=FAIL is only the expected ACC_TRACE gap; Stage 1 mm/CB unit surface is green.

## Optional operator stamp — DONE

Stamp `logs/task066-ubatch-ladder-20261106T184055Z` (`MUL_MM_MIN` unset / default 32, `{8,256,512}`):

- S256 vs S8 / S512 vs S8: DIFF@80 (same opening as prior)
- S256 vs S512: DIFF@314 (matches `170944Z`)
- S8 == `172821Z`; S256 == `181846Z`; S512 == `170944Z`

S1024/S2048 ownership rests on `170944Z` (2x same-ubatch + S1024==S2048). No extra pin-32 re-stamp required for Stage 1 close.

## Stage 2 unlock

Stage 1 closed under this WAIVE. **P3 MTP vs nomtp** may proceed on product band:

- primary: ubatch **512** (floor 256); large owned cells **1024 / 2048** OK for stress
- valid engagement evidence (`-lv 3`); prior stamp `...173712Z` INVALID
- runner: `tests/run-task066-mtp-vs-nomtp-ladder.sh`
- Rule unchanged: do not chase mtp-vs-nomtp fixes until large-ubatch vs S8 nomtp residual is owned — **owned via this WAIVE**.
