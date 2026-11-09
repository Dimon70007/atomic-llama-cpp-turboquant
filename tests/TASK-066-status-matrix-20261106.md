# TASK-066 autotest status matrix

**When:** 2026-11-06 (UTC ~20:42–20:45)  
**Host:** Apple M4 Metal  
**Build:** `BUILD_DIR=build-task066-cb-ws` (includes 066.3 `ggml_is_empty` fusion_max HELP)  
**Index:** [README-TASK066.md](README-TASK066.md)

## Summary

| Bucket | GREEN | RED | SKIP / blocked |
|--------|-------|-----|----------------|
| `SUITE=all` (fa…moe-router) | 5/5 | 0 | — |
| Follow-up Track A/B units | 3 | 1 (expected) | 1 |
| Extra suites | 2 | 2 | — |

Product short ladder (066.4): S8 vs S1024 still **DIFF@80** — not an autotest suite; recorded in [TASK-066.4](TASK-066.4-short-model-s8-s1024.md).

---

## `SUITE=all` — GREEN

Stamp dir pattern: `logs/task066-status-matrix-all-*`

| SUITE | Verdict | Notes |
|-------|---------|--------|
| `fa` | **GREEN** | C1/C2/C4 + C6 |
| `mul-mat` | **GREEN** | C1/C4/C5 + C6 |
| `longctx` | **GREEN** | FA long-KV row0 |
| `moe` | **GREEN** | mul_mv_id row0 |
| `moe-router` | **GREEN** | router→moe |
| **umbrella** | **PASS** | |

---

## Follow-up (`run-task066-followup-all.sh`)

Stamp: `logs/task066-status-matrix-20261106T204235Z/` · `results.tsv`

| SUITE | Exit | Verdict | Why |
|-------|------|---------|-----|
| `cb-fusion` | 0 | **GREEN** | I1–I4 incl. empty_ne after 066.3 |
| `b1-shared-kv` | 0 | **GREEN** | `#27781` shared cells |
| `mul-mat-residual` | 0 | **GREEN** (066.7 KEEP) | After float-tile mm + `MUL_MM_MIN` default **7**: all widths vs N=8 PASS (stamp `logs/task066-metal-corners-mul-mat-residual-20261106T210344Z`) |
| `mtp-acc-trace` | 1 | **RED** | Expected: no `TASK066_ACC_TRACE_LOG` attached (MTP track blocked) |
| `mtp-e2e` | 0 | **SKIP** | Follow-up path skips without engagement / MODEL gate |

Follow-up overall `VERDICT=FAIL` only because of expected `mtp-acc-trace` RED.

---

## Extra suites (not in `all` / followup defaults)

| SUITE | Verdict | Why |
|-------|---------|-----|
| `moe-out` | **GREEN** | ffn_moe_out unified reduce row0 |
| `fullgraph` | **RED** | With `TASK066_MODEL=…IQ4_XS` `TASK066_CTX=2048`: row0 N=1 vs N=2 not bit-exact (`VERDICT=RED`; also prefix decode rc=1 noise in log — treat as RED gate) |
| `mtp-acc-trace` (solo) | **RED** | Missing `TASK066_ACC_TRACE_LOG` |
| `mtp-e2e` (solo, no MODEL) | **RED**/fail-closed | Missing `TASK066_MODEL` (fail-closed; follow-up may SKIP) |

---

## Legend for operators

- **GREEN / PASS** — suite gate OK for current KEEP surface.  
- **RED** — failing closed; either expected (MTP not unlocked) or real residual (`fullgraph`).  
- **SKIP** — intentionally not run / missing optional input.

## Next

- **066.7 residual GREEN** — optional short ladder S8 vs S1024 confirm; then 066.5/066.6 only if product drift remains.  
- Do not treat `mtp-*` RED as Stage 1 blockers while MTP is blocked.  
- `fullgraph` RED is useful invariant for later MTP/row0 work — not a waiver to skip confirm ladder.
