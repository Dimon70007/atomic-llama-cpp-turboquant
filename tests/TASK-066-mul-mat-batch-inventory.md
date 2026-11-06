# TASK-066 follow-up B5 — Metal MUL_MAT batch-path inventory

Status: residual **vs N=8** extended (2026-11-06): `{9,16,32}` GREEN; `{256,512,1024}` **RED** (mad ~1e-4). Pin `GGML_METAL_MUL_MM_MIN` default 32; env-gated.

Related: `#25618`, `test-mul-mat-batch-invariance`, `SUITE=mul-mat` / `SUITE=mul-mat-residual`,
[task066-hist/draft_mtp_research_task-066_testing.md](task066-hist/draft_mtp_research_task-066_testing.md) (product `batch`/`ubatch` sensitivity),
[TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md),
active track [TASK-066.7-mul-mat-residual.md](TASK-066.7-mul-mat-residual.md)

## Already pinned (do not re-litigate)

| Path | Shape | Gate | Status |
|------|-------|------|--------|
| `mul_mv_ext` disabled for `ne11` in 2..8 | Q8_0 mat-vec small batch | `SUITE=mul-mat` C1/C4 | KEEP |
| `mul_mat_id` verify widths `ne21` < 32 | MoE | `SUITE=mul-mat` C5 | KEEP |
| FA `nsg` pin | flash attn | `SUITE=fa` | KEEP (FA, not matmul) |
| `mul_mv` / `mul_mm` for `ne11` in {9,16,32} | was vs N=1; now also in vs-N=8 set | `SUITE=mul-mat-residual` | part of vs-N=8 gate |
| `mul_mv_id` / `mul_mm_id` at `ne21=32` | MoE | `SUITE=mul-mat-residual` C5 | part of vs-N=8 gate |
| IQ4_XS residual | same path switch | residual suite | part of vs-N=8 gate |
| **row0 vs golden N=8** for `ne11`∈{9,16,32,**256,512,1024**} | product S8 vs S1024 class | `SUITE=mul-mat-residual` | **extended 2026-11-06** (expect RED on large until mm HELP) |
| Fused `MUL_MAT+ADD` bias | residual add after proj | `test-metal-cb-fusion` | KEEP structural; `#30100` N/A (no encode) |

## Product note (beyond unit residual)

See [README-TASK066.md](README-TASK066.md) / [TASK-066.0-overview.md](TASK-066.0-overview.md) (hist: [task066-hist/TASK-066-execution-plan.md](task066-hist/TASK-066-execution-plan.md)). Debug ladder with env-gated `ne11_mm_min=8`; product default may stay 32.

| Path | Why | Status |
|------|-----|--------|
| env-gate `GGML_METAL_MUL_MM_MIN` | A/B 8 vs 32 without rebuild; MoE `ne21` shares non-TQ | DONE P0.1 |
| residual suite gate | SKIP(77) if env set and != 32 | DONE P0.1 |
| ladder fixture + runner | Stage 0 product stamp | DONE P0.2/P0.3 |
| `mul_mm` / split-K / same-ubatch | #19034; stamp 170944Z run1=run2 | DONE (same-ubatch OK) |
| Skinny-batch #27441 | 8..32 with min=8 | **CLOSED** — port regressed (`175708Z`), reverted; see task066-hist/TASK-066-p22-27441-audit.md; may reopen under 066.7 |
| `#30100` MMA+ADD residual pick | few-row fused store | **N/A** — no encode path; task066-hist/TASK-066-p22-30100-recheck.md |
| Product min=32 A/B (S16 on mul_mv) | stamp 172821Z S16==S8 | DONE; keep default 32 |
| S33 (diag only) | stamps 172821Z / 181846Z | not product gate; DIFF@253 vs S8 |
| **`{256,512,1024,2048}`** large band | 170944Z / 181846Z / 184055Z | Hist **WAIVE** (not Done for active track) — vs S8 DIFF@80; see task066-hist/TASK-066-stage1-waive.md; reopen via 066.1–066.7 |
| FA / MoE at large ubatch | only if quieter band required later | optional; not Stage 1 |

Goal: `SUITE=mul-mat-residual` GREEN vs N=8 (066.7 REOPENED / primary). Hist WAIVE not Done. E2e ladder only after residual PASS. MTP blocked.

## How to close a row

1. Add or name a failing-closed corner that reproduces the drift, **or**
2. Mark SKIP with one-line reason (path unreachable under product knobs), **or**
3. Pin kernel choice in Metal and turn the corner GREEN.

Update this table when status changes; `SUITE=mul-mat-residual` must stay green for every KEEP residual row.
