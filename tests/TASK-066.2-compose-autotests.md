# TASK-066.2 — Compose correct autotests (primary gate)

**Status:** ✅ (2026-11-06) — suite correctly RED on Gap A; green cases documented  
**Depends on:** [TASK-066.1](TASK-066.1-audit-31688.md) audit notes exist  
**Next:** [TASK-066.3-fix-until-autotest-pass.md](TASK-066.3-fix-until-autotest-pass.md)  
**Evidence:** `logs/task066-0662-empty-chain` — `empty_ne_start.fusion_max` FAIL (got=2 want=1)

## Goal

Autotests that **actually** prove #31688 / fusion_max behavior. Existing `SUITE=cb-fusion` may be too weak (e.g. ADD-only / STRUCTURAL-only green while product still drifts).

## DoD

- [x] Document invariants I1–I4 (below + test file header).
- [x] Wrong/incomplete CB or fusion_max → **RED**; correct port → **GREEN** (TDD: red before → green after when fixing).
- [x] Prefer ggml/Metal unit fixtures in `tests/` — no full model required here.
- [x] No short-model ladder in this subtask.

## Invariants (per case)

| ID | Case(s) | Asserts |
|----|---------|---------|
| I1 | `norm_*`, `snake`, `fusion_next_norm_*`, ADD cases | `fusion_max` spans encoder-class fuse groups |
| I2 | `*.align_*` | CB walk never lands mid-group |
| I3 | `empty_ne_start` | `ggml_is_empty` start ⇒ `fusion_max==1` (upstream pr28164) — **RED on HEAD** |
| I4 | `mul_mat_add.*` | STRUCTURAL pack 2 + `may_use_mma`; `use_mma==false` (no false “encoder fuse”) |

## Checklist

- [x] Review `tests/test-metal-cb-fusion.cpp` + `run-task066-cb-fusion-corners.sh`.
- [x] Suite fails when CB boundary would split a fusable group (align loops).
- [x] Coverage beyond ADD: NORM, SNAKE, MUL_MAT+ADD (honest I4), `fusion_next` id.
- [x] Gaps from 066.1: I3 empty filter + I4 stubs documented in tests.
- [x] Wired in existing `SUITE=cb-fusion` (same cmake target).
- [x] Acceptance: not ADD-only green.

## Commands

```bash
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
# Expect FAIL on empty_ne_start until 066.3
```
