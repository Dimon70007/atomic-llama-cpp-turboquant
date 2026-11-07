# TASK-066 follow-up B5 — Metal MUL_MAT batch-path inventory

Status: **066.7 KEEP** (2026-11-06). Residual vs N=8 GREEN for `{9,16,32,256,512,1024}`.  
Product default `GGML_METAL_MUL_MM_MIN=1` (ne11≥2 → float-tile `mul_mm` for q8_0/iq4_xs; 066.9 A2). Env-gated.

Related: `#25618`, `test-mul-mat-batch-invariance`, `SUITE=mul-mat` / `SUITE=mul-mat-residual`,
[task066-hist/draft_mtp_research_task-066_testing.md](task066-hist/draft_mtp_research_task-066_testing.md) (product `batch`/`ubatch` sensitivity),
[TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md),
active track [TASK-066.7-mul-mat-residual.md](TASK-066.7-mul-mat-residual.md)

## Already pinned (do not re-litigate)

| Path | Shape | Gate | Status |
|------|-------|------|--------|
| `mul_mv_ext` disabled for `ne11` in 2..8 | Q8_0 mat-vec small batch | `SUITE=mul-mat` C1/C4 | KEEP |
| `mul_mat_id` verify widths `ne21` ≤ 7 | MoE (product min=7) | `SUITE=mul-mat` C5 | KEEP (widths `{2,3,4,5,7}`) |
| FA `nsg` pin | flash attn | `SUITE=fa` | KEEP (FA, not matmul) |
| `mul_mv` for `ne11` ≤ 7; `mul_mm` for `ne11` ≥ 8 | product S8+ on mm | `SUITE=mul-mat-residual` | **066.7 KEEP** |
| `mul_mv_id` / `mul_mm_id` same break-even | MoE | residual C5 | **066.7 KEEP** |
| IQ4_XS residual | float-tile mm | residual suite | **066.7 KEEP** |
| **row0 vs golden N=8** for `ne11`∈{9,16,32,**256,512,1024**} | product S8 vs S1024 class | `SUITE=mul-mat-residual` | **GREEN** (stamp `…T210344Z`) |
| Fused `MUL_MAT+ADD` bias | residual add after proj | `test-metal-cb-fusion` | KEEP structural; `#30100` N/A (no encode) |
| Float-tile `mul_mm`/`mul_mm_id` for q8_0 + iq4_xs | half MMA → ~1e-4 vs mv; float → ~1 ULP | residual + ops | **HELP** then paired with min=7 |

## Product note (beyond unit residual)

See [README-TASK066.md](README-TASK066.md) / [TASK-066.0-overview.md](TASK-066.0-overview.md).

| Path | Why | Status |
|------|-----|--------|
| env-gate `GGML_METAL_MUL_MM_MIN` | A/B without rebuild; MoE `ne21` shares non-TQ | DONE; **default 1** (was 7 / 32) |
| residual suite gate | SKIP(77) if env set and != **7** | DONE 066.7 |
| ladder fixture + runner | Stage 0 product stamp | DONE P0.2/P0.3 |
| `mul_mm` / split-K / same-ubatch | #19034; stamp 170944Z run1=run2 | DONE (same-ubatch OK) |
| Skinny-batch #27441 | 8..32 with min=8 | **CLOSED** — port regressed (`175708Z`), reverted |
| `#30100` MMA+ADD residual pick | few-row fused store | **N/A** — no encode path |
| Product min=32 A/B (S16 on mul_mv) | stamp 172821Z S16==S8 | superseded by 066.7 min=7 |
| **`{256,512,1024,2048}`** large band | hist WAIVE | reopen via 066.7 residual GREEN + optional ladder confirm |
| FA / MoE at large ubatch | only if quieter band required later | optional |

Goal: `SUITE=mul-mat-residual` GREEN vs N=8 — **met**. Hist WAIVE not Done for product ladder. MTP blocked.

## How to close a row

1. Add or name a failing-closed corner that reproduces the drift, **or**
2. Mark SKIP with one-line reason (path unreachable under product knobs), **or**
3. Pin kernel choice in Metal and turn the corner GREEN.

Update this table when status changes; `SUITE=mul-mat-residual` must stay green for every KEEP residual row.
