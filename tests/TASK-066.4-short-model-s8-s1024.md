# TASK-066.4 — Short model confirm (S8 vs S1024)

**Status:** ✅ done (2026-11-06) — **NO_EFFECT** on product drift vs baseline  
**Depends on:** [TASK-066.3](TASK-066.3-fix-until-autotest-pass.md) autotests **PASS**  
**Next:** [TASK-066.5-fusion-table-parity.md](TASK-066.5-fusion-table-parity.md)  
**Fixture:** [TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md)  
**Stamp:** `logs/task066-ubatch-ladder-0664-post-gapA` (UTC `20261106T203056Z`)

## Goal

**Confirmation only** — not primary proof. After green autotests, run nomtp greedy on a **small text chunk** and compare ubatch cells, especially **S8 vs S1024**.

## DoD

- [x] Ladder stamp with `matrix.md`.
- [x] Cells `{8,1024}` (`n_predict=256`, default pin `MUL_MM_MIN=32`).
- [x] Compare to baseline `170944Z`: still **DIFF @byte 80** (same opening class).
- [x] Verdict: Gap A HELP on units → **NO_EFFECT** on product S8 vs S1024 → proceed 066.5+ (do not WAIVE as Done).
- [x] MTP still blocked.

## Result

| Compare | Result |
|---------|--------|
| S1024 vs S8 | DIFF @byte **80** `a='данном отрывке опис'` / `b='центре повествован…'` |
| vs `170944Z` | Same opening DIFF class — empty-ne filter alone does not quiet product band |

Build: `BUILD_DIR=build-task066-cb-ws` (includes 066.3 fusion_max fix).

## Checklist

- [x] 066.3 suite PASS before ladder.
- [x] Ladder ran (no other IQ4_XS conflict observed).
- [x] Verdict recorded: NO_EFFECT for product; unlock 066.5.

## Commands (repro)

```bash
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
TASK066_UBATCH_LIST="8 1024" TASK066_REPEAT_LARGE="" TASK066_N_PREDICT=256 \
  BUILD_DIR=build-task066-cb-ws ./tests/run-task066-ubatch-ladder.sh
```
