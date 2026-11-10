# TASK-066 — Metal batch-invariance / nomtp ubatch (fork SSOT)

Self-contained in **this** llama.cpp / ggml fork. Other developers: start here.

**Branch:** `task066-metal-batch-invariance`  
**Active track:** **[066.9](TASK-066.9-fa-batch-invariance.md)** — **A1+A2+B+C Done** (Phase C `043032Z`; **operator KEEP** `(2,0.95)`).  
**PR fidelity:** **[TASK-066-pr-audit-registry.md](TASK-066-pr-audit-registry.md)** (re-check after every Metal edit + suites).  
**Done:** **[066.7 KEEP](TASK-066.7-mul-mat-residual.md)** + **[066.8](TASK-066.8-fp-comparison-metrics-rd.md)** + FA always-tile + **A2** + **B** + **C** knobs matrix.  
**Open (ideas reopen):** **[066.5](TASK-066.5-fusion-table-parity.md)** + **[066.10](TASK-066.10-kv-ctx-sizing.md)** + **[066.11](TASK-066.11-small-ctx-q8-turbo3-cross-ub-rd.md)** — TODO (pad128 / KLD / estimator / turbo3 small-CTX cross-ub R&D).  
**Open (perf closeout):** **[066.15](TASK-066.15-metal-perf-knobs-rd.md)** — DEFAULT=1 probe stamped; triage **066.16 DONE** → commit/orchestrator still open.  
**Done (triage):** **[066.16](TASK-066.16-min1-red-triage.md)** — R1–R9 all **P**; **KEEP DEFAULT=1**; stamp `…062918Z` (R1–R7) + R8/R9.  
**Priority:** commit tip docs/066.17–19 → lock SHA → **066.19** KEEP-stack speed gates → 066.15 closeout → 066.5/066.10 → TASK-067.

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
| **066.5** | [TASK-066.5-fusion-table-parity.md](TASK-066.5-fusion-table-parity.md) | 066.9 localize (H9faOFF) | ⬜ **TODO** (ideas pad128; C3 + KLD open) |
| **066.6** | [TASK-066.6-cb-and-invariance-gates.md](TASK-066.6-cb-and-invariance-gates.md) | after 066.5 / parallel | 🟢 C2/C7 (C3 historically via prior 066.5 KEEP) |
| **066.10** | [TASK-066.10-kv-ctx-sizing.md](TASK-066.10-kv-ctx-sizing.md) | 066.5 ideas | ⬜ **TODO** (estimator; not closed) |
| **066.11** | [TASK-066.11-small-ctx-q8-turbo3-cross-ub-rd.md](TASK-066.11-small-ctx-q8-turbo3-cross-ub-rd.md) | 066.5 tooling | ⬜ **TODO** (R&D: small-CTX q8+turbo3 cross-ub) |
| **066.7** | [TASK-066.7-mul-mat-residual.md](TASK-066.7-mul-mat-residual.md) | 066.4 done; residual was RED | 🟢 **KEEP** |
| **066.8** | [TASK-066.8-fp-comparison-metrics-rd.md](TASK-066.8-fp-comparison-metrics-rd.md) ([lit](task066-hist/TASK-066.8-lit-notes.md)) | 066.7 KEEP | 🟢 **R&D DELIVERED** |
| **066.9** | [TASK-066.9-fa-batch-invariance.md](TASK-066.9-fa-batch-invariance.md) ([draft](task066-hist/draft_mtp_research_task-066_fa.md)) | C2 GREEN + 066.7 | 🟢 A1+A2+B+C (`043032Z`; KEEP `(2,0.95)`) |
| **066.13** | [TASK-066.13-size-swa-ub-invariant-rd.md](TASK-066.13-size-swa-ub-invariant-rd.md) | 066.11 | 🟢 C2 KEEP (headroom source superseded by 066.14) |
| **066.14** | [TASK-066.14-c1-s8-s1024-localize-rd.md](TASK-066.14-c1-s8-s1024-localize-rd.md) | 066.13 | 🟢 H6 + product pivot `n_ubatch`+pad128 |
| **066.15** | [TASK-066.15-metal-perf-knobs-rd.md](TASK-066.15-metal-perf-knobs-rd.md) | 066.9 + 066.14 evidence | 🟡 DEFAULT=1 probe; env+quality stamps |
| **066.16** | [TASK-066.16-min1-red-triage.md](TASK-066.16-min1-red-triage.md) | 066.15 full-suite RED | 🟢 R1–R9 **P**; **KEEP DEFAULT=1** (`…062918Z`) |
| **066.17** | [TASK-066.17-product-ubatch-256-512-1024.md](TASK-066.17-product-ubatch-256-512-1024.md) | 066.14 | 🟢 DONE product C2p/C1p/C1jp (`…225212Z`); stress C1 RED |
| **066.18** | [TASK-066.18-src-lever-ablation.md](TASK-066.18-src-lever-ablation.md) · [Mode B influence table](TASK-066.18-modeB-influence-table.md) | squash tip vs `0fe16d12d` | 🟢 Mode A+B DONE; size_swa `PAD128(n_batch)` alone-ON `…001626Z` |
| **066.19** | [TASK-066.19-keep-stack-speed-gates.md](TASK-066.19-keep-stack-speed-gates.md) | 066.17 + 066.18 §5 | ⬜ TODO (orchestrator landed; Terminal stamps pending) NEGATIVE hard-stop |

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

# 066.9 Phase C — MTP knobs quality/speed (n_max × p_min; stamp+recommend; no YAML auto-promote)
unset GGML_METAL_MUL_MM_MIN
BUILD_DIR=build-task066-cb-ws ./tests/run-task066-mtp-knob-matrix.sh

# 066.18 — src lever ablation (tip−1 / master+1); one lever per cell
TASK066_MODEL=/path/to/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf TASK066_CTX=3072 \
  ./tests/run-task066-src-lever-ablation.sh --dry-run
# Parent: TASK066_MODEL=… TASK066_CTX=3072 make minillm-task066-src-lever-ablation
# ABLATION_ARGS='--mode A --levers L_size_swa_ub --rows S_ub_c2pf16'

# 066.19 — cumulative KEEP-stack speed (nomtp≥23 / MTP≥30); NEGATIVE hard-stop
# Prefer fork cwd; each cell: rebuild → tests/run-task066-decode-speed-probe.sh
cd atomic-llama-cpp-turboquant
MODEL_NOMTP=gemma4-26b-a4b-mtp-nomtp MODEL_MTP=gemma4-26b-a4b-mtp \
  TASK066_MODEL=/path/to/UD-IQ4_XS.gguf TASK066_MODEL_DRAFT=/path/to/mtp-Q8_0.gguf \
  ./tests/run-task066-keep-stack-speed.sh --dry-run
# Parent sugar: same env + make minillm-task066-keep-stack-speed
# KEEP_STACK_ARGS='--from-cell C2_fa_mv' | '--phase-f' (C_final_tip after C0–C4)

# 066.19 Phase F — after C0–C4 GREEN (Metal Terminal):
STAMP_DIR=logs/task066-19-keep-stack-<UTC> BUILD_DIR=build-task066-cb-ws \
  MODEL_NOMTP=gemma4-26b-a4b-mtp-nomtp MODEL_MTP=gemma4-26b-a4b-mtp \
  TASK066_MODEL=/path/to/UD-IQ4_XS.gguf TASK066_MODEL_DRAFT=/path/to/mtp-Q8_0.gguf \
  PHASE_F_MODE=all ./tests/run-task066-phase-f-battery.sh
# PHASE_F_MODE=speed|suites|all ; parent: make minillm-task066-phase-f-battery
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
| `mul-mat-residual` | `run-task066-mul-mat-residual-corners.sh` | mul_mm policy unit (CPU) + ne11 residual |
| `ubatch-s8-s1024` | `run-task066-ubatch-s8-s1024-corners.sh` | C2/C3 + C2pf16 + C2pq8 + C2p/C1p/C1jp (256↔512, 256↔1024) + stress 16↔1024; DIAG=c1p / c1p-q8 / c1p-f16 / c1jp |
| `followup` | `run-task066-followup-all.sh` | Track A+B one-shot |
| `all` | umbrella | fa…moe-router |

Host without Metal: binaries exit `77` (SKIP).

## Extending coverage

Add `tests/test-*.cpp` + `tests/run-task066-*.sh`, register cmake target, wire suite into `run-task066-metal-corners.sh`. Keep fixtures inside this repository.
