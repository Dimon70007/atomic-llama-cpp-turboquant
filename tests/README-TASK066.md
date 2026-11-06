# TASK-066 — Metal batch-invariance / nomtp ubatch (fork SSOT)

Self-contained in **this** llama.cpp / ggml fork. Other developers: start here.

**Branch:** `task066-metal-batch-invariance`  
**Active track:** **[066.5](TASK-066.5-fusion-table-parity.md)** / **[066.6](TASK-066.6-cb-and-invariance-gates.md)** — FA multi-step ubatch (C2); turbo4 amplifies.  
**PR fidelity:** **[TASK-066-pr-audit-registry.md](TASK-066-pr-audit-registry.md)** (re-check after every Metal edit + suites).  
**Done behind:** **[066.7 KEEP](TASK-066.7-mul-mat-residual.md)** + **[066.8 R&D DELIVERED](TASK-066.8-fp-comparison-metrics-rd.md)**.  
**Ladder confirm:** `212703Z` HELP 80→182 DIFF@text. **C2** n=32 ub8 vs ub32 RED. MTP blocked. Hist Stage 1 WAIVE ≠ Done.

## Evidence hierarchy (hard)

1. **Every code change ships with autotests** — same change set: new/extended corner that is **RED** on pre-fix → fix → **GREEN**. “Existing suite already covers it” only if you show RED on *this* gap before the fix. No kernel/ops/fusion edits without that.
2. **066.7 primary gate:** `SUITE=mul-mat-residual` (not e2e / ladder / fullgraph).
3. **Short model / ladder** — optional confirm **only after** residual PASS.
4. Never KEEP from a lone model stamp; never burn e2e between mm levers.

## Ordered subtasks

| # | File | Depends on | Status |
|---|------|------------|--------|
| **066.1** | [TASK-066.1-audit-31688.md](TASK-066.1-audit-31688.md) ([notes](TASK-066.1-audit-notes.md)) | — | ✅ PARTIAL |
| **066.2** | [TASK-066.2-compose-autotests.md](TASK-066.2-compose-autotests.md) | 066.1 | ✅ |
| **066.3** | [TASK-066.3-fix-until-autotest-pass.md](TASK-066.3-fix-until-autotest-pass.md) | 066.2 | ✅ Gap A HELP |
| **066.4** | [TASK-066.4-short-model-s8-s1024.md](TASK-066.4-short-model-s8-s1024.md) | 066.3 PASS | ✅ NO_EFFECT |
| **066.5** | [TASK-066.5-fusion-table-parity.md](TASK-066.5-fusion-table-parity.md) | residual GREEN + ladder still DIFF | 🔓 **active** |
| **066.6** | [TASK-066.6-cb-and-invariance-gates.md](TASK-066.6-cb-and-invariance-gates.md) | after 066.5 / parallel invariance | 🔓 **active** (residual suite already extended; `fullgraph` still RED) |
| **066.7** | [TASK-066.7-mul-mat-residual.md](TASK-066.7-mul-mat-residual.md) | 066.4 done; residual was RED | 🟢 **KEEP** |
| **066.8** | [TASK-066.8-fp-comparison-metrics-rd.md](TASK-066.8-fp-comparison-metrics-rd.md) ([lit](task066-hist/TASK-066.8-lit-notes.md)) | 066.7 KEEP | 🟢 **R&D DELIVERED** |

Overview locks / DoD / out-of-scope: [TASK-066.0-overview.md](TASK-066.0-overview.md).  
Latest suite GREEN/RED snapshot: [TASK-066-status-matrix-20261106.md](TASK-066-status-matrix-20261106.md).

## Living ops docs (not numbered)

| Doc | Role |
|-----|------|
| [TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md) | Frozen nomtp ladder knobs / cli map |
| [TASK-066-mul-mat-batch-inventory.md](TASK-066-mul-mat-batch-inventory.md) | Metal MUL_MAT path inventory (066.7) |

## Historical / research (read-only)

Under [`task066-hist/`](task066-hist/): drafts, old execution-plan, stage1-waive, upstream port notes, `#27441` / `#30100` audits. Do not treat waive as Done for the active track.

## Unit + ladder commands

```bash
# Autotests (primary)
BUILD_DIR=build-task066-cb SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
# umbrella / followup:
SUITE=all ./tests/run-task066-metal-corners.sh
./tests/run-task066-followup-all.sh

# 066.7 primary loop (not e2e):
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh

# Product nomtp ladder — ONLY after mul-mat-residual PASS
TASK066_UBATCH_LIST="8 256 512 1024" TASK066_REPEAT_LARGE="" \
  BUILD_DIR=build-task066-cb-ws ./tests/run-task066-ubatch-ladder.sh
```

Configure + build (Metal): see below. Parent product repo may call `make minillm-task066-metal-corners` — that is optional sugar; this fork must stay runnable alone.

### Configure + build (Metal)

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build --parallel \
  --target test-flash-attn-batch-invariance \
  --target test-mul-mat-batch-invariance \
  --target test-longctx-row0-parity \
  --target test-moe-row0-parity \
  --target test-moe-router-row0-parity \
  --target test-moe-out-row0-parity \
  --target test-fullgraph-decode-row0-parity \
  --target test-metal-cb-fusion \
  --target test-backend-ops
```

`BUILD_DIR` defaults to `<repo>/build`. Override with `BUILD_DIR=/path/to/build`.

### Suite table

| SUITE | Runner | What |
|-------|--------|------|
| `fa` | `run-task066-fa-nsg-corners.sh` | FA nsg / row0 |
| `mul-mat` | `run-task066-mul-mat-corners.sh` | mul_mat / mul_mat_id |
| `longctx` | `run-task066-longctx-corners.sh` | long-KV FA row0 |
| `moe` | `run-task066-moe-corners.sh` | MoE mul_mv_id |
| `moe-router` | `run-task066-moe-router-corners.sh` | router→moe |
| `moe-out` | `run-task066-moe-out-corners.sh` | ffn_moe_out topologies |
| `fullgraph` | `run-task066-fullgraph-corners.sh` | decode N=1 vs N=2 |
| `cb-fusion` | `run-task066-cb-fusion-corners.sh` | fusion_max / CB align |
| `b1-shared-kv` | `run-task066-b1-shared-kv-audit.sh` | shared-KV audit |
| `mtp-acc-trace` | `run-task066-mtp-acc-trace-checklist.sh` | ACC_TRACE log attach |
| `mtp-e2e` | `run-task066-mtp-e2e-corners.sh` | MTP e2e (blocked for now) |
| `mul-mat-residual` | `run-task066-mul-mat-residual-corners.sh` | ne11 residual |
| `ubatch-s8-s1024` | `run-task066-ubatch-s8-s1024-corners.sh` | greedy S8 vs S1024 (product ladder class; needs MODEL+CTX) |
| `followup` | `run-task066-followup-all.sh` | Track A+B one-shot |
| `all` | umbrella | fa…moe-router |

Host without Metal: binaries exit `77` (SKIP).

## Extending coverage

Add `tests/test-*.cpp` + `tests/run-task066-*.sh`, register cmake target, wire suite into `run-task066-metal-corners.sh`. Keep fixtures inside this repository.
