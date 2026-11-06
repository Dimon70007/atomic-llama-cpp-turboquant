# TASK-066 follow-up B5 — Metal MUL_MAT batch-path inventory

Status: B5 GREEN (2026-11-06 Metal M4) after `ne11_mm_min=32` + `ne21 > 32` pins; **env-gated** via `GGML_METAL_MUL_MM_MIN` (default 32).

Related: `#25618`, `test-mul-mat-batch-invariance`, `SUITE=mul-mat` / `SUITE=mul-mat-residual`,
[draft_mtp_research_task-066_testing.md](draft_mtp_research_task-066_testing.md) (product `batch`/`ubatch` sensitivity),
[TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md)

## Already pinned (do not re-litigate)

| Path | Shape | Gate | Status |
|------|-------|------|--------|
| `mul_mv_ext` disabled for `ne11` in 2..8 | Q8_0 mat-vec small batch | `SUITE=mul-mat` C1/C4 | KEEP |
| `mul_mat_id` verify widths `ne21` < 32 | MoE | `SUITE=mul-mat` C5 | KEEP |
| FA `nsg` pin | flash attn | `SUITE=fa` | KEEP (FA, not matmul) |
| `mul_mv` / `mul_mm` for `ne11` in {9,16,32} | N=1=`mul_mv`, was ne11>8=`mul_mm` | `SUITE=mul-mat-residual` | KEEP (default `GGML_METAL_MUL_MM_MIN=32`) |
| `mul_mv_id` / `mul_mm_id` at `ne21=32` | was `ne21>=32`=`mm_id` | `SUITE=mul-mat-residual` C5 | KEEP (same env; MoE mirrors non-TQ) |
| IQ4_XS residual | same path switch | residual suite | KEEP (same pin) |
| Fused `MUL_MAT+ADD` bias | residual add after proj | `test-metal-cb-fusion` | KEEP structural; `#30100` N/A (no encode) |

## Product note (beyond unit residual)

See [TASK-066-execution-plan.md](TASK-066-execution-plan.md). Debug ladder with env-gated `ne11_mm_min=8`; product default may stay 32.

| Path | Why | Status |
|------|-----|--------|
| env-gate `GGML_METAL_MUL_MM_MIN` | A/B 8 vs 32 without rebuild; MoE `ne21` shares non-TQ | DONE P0.1 |
| residual suite gate | SKIP(77) if env set and != 32 | DONE P0.1 |
| ladder fixture + runner | Stage 0 product stamp | DONE P0.2/P0.3 |
| `mul_mm` / split-K / same-ubatch | #19034; stamp 170944Z run1=run2 | DONE (same-ubatch OK) |
| Skinny-batch #27441 | 8..32 with min=8 | **CLOSED** — port regressed (`175708Z`), reverted; see TASK-066-p22-27441-audit.md |
| `#30100` MMA+ADD residual pick | few-row fused store | **N/A** — no encode path; TASK-066-p22-30100-recheck.md |
| Product min=32 A/B (S16 on mul_mv) | stamp 172821Z S16==S8 | DONE; keep default 32 |
| S33 (diag only) | stamps 172821Z / 181846Z | not product gate; DIFF@253 vs S8 |
| **`{256,512,1024,2048}`** owned large | 170944Z / 181846Z / 184055Z | **WAIVE** — vs S8 DIFF@80; S256 vs S512 @314; S512 vs S1024 @134; **S1024==S2048** + same-ubatch OK; floor 256 / target 512; [TASK-066-stage1-waive.md](TASK-066-stage1-waive.md) |
| FA / MoE at large ubatch | only if quieter band required later | optional; not Stage 1 |

Goal: noise low enough + WAIVE — **met** (Stage 1 closed). MTP = Stage 2 / P3.

## How to close a row

1. Add or name a failing-closed corner that reproduces the drift, **or**
2. Mark SKIP with one-line reason (path unreachable under product knobs), **or**
3. Pin kernel choice in Metal and turn the corner GREEN.

Update this table when status changes; `SUITE=mul-mat-residual` must stay green for every KEEP residual row.
