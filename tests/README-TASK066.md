# TASK-066 — Metal batch-invariance corner tests

Self-contained in **this** llama.cpp / ggml fork. No external product repos required.

**Product execution (locked):** [TASK-066-execution-plan.md](TASK-066-execution-plan.md) ← use this.  
Decisions: [draft_mtp_research_task-066_answers.md](draft_mtp_research_task-066_answers.md).  
Status pointer: [TASK-066-cb-fusion-followup.md](TASK-066-cb-fusion-followup.md).  
Upstream port history (done): [TASK-066-upstream-patches-plan.md](TASK-066-upstream-patches-plan.md).

One-command **unit** follow-up gates (not the product ladder):

```bash
./tests/run-task066-followup-all.sh
# or: SUITE=followup ./tests/run-task066-metal-corners.sh
```

These unit tests gate Metal patches for speculative decode row0 class (`mul_mv` / FA `nsg` / MoE). Product nomtp ubatch ladder is separate (execution plan P0-P2); MTP is a later follow-up.

**Product ubatch ladder** (Metal Terminal; not in `SUITE=followup`):

```bash
# Fixture: tests/TASK-066-ladder-fixture.md
# Model: .scratch/gemma-4-26b-a4b/...UD-IQ4_XS.gguf (same canon as scripts/run-gemma4-*.sh)
./scripts/setup-gemma4-26b-a4b-models.sh   # once; links from $HOME/models
GGML_METAL_MUL_MM_MIN=8 BUILD_DIR=build-task066-cb ./tests/run-task066-ubatch-ladder.sh

# P1.2 A/B — product default pin (S16 stays on mul_mv)
unset GGML_METAL_MUL_MM_MIN
TASK066_UBATCH_LIST="8 16 33 256" TASK066_REPEAT_LARGE="" \
  BUILD_DIR=build-task066-cb ./tests/run-task066-ubatch-ladder.sh
```

`GGML_METAL_MUL_MM_MIN` env-gates Metal `mul_mm` break-even (default **32** = B5; **8** = upstream-like debug). `SUITE=mul-mat-residual` SKIP if the env is set to anything other than 32.

**Upstream mm patches:** `#27441` **CLOSED** (tried, regressed on `175708Z`, reverted — [audit](TASK-066-p22-27441-audit.md)). `#30100` **N/A** ([recheck](TASK-066-p22-30100-recheck.md)). Do not invent higher `MUL_MM_MIN`.

**Stage 1 CLOSED:** `{256,512,1024,2048}` vs S8 **WAIVEd** — [TASK-066-stage1-waive.md](TASK-066-stage1-waive.md). **Product LOCK:** floor **256**, target **512**; owned large also **1024/2048** (S16 = diagnostic baseline under pin 32). Ready mm PRs exhausted; residual owned (DIFF@80 vs S8; S1024==S2048 + same-ubatch OK on `170944Z`).

**P3 MTP UNLOCKED** at ubatch **512** (need `-lv 3`; stamp `...173712Z` INVALID).

```bash
# Product-band completeness (done: stamp 184055Z; default pin 32)
unset GGML_METAL_MUL_MM_MIN
TASK066_UBATCH_LIST="8 256 512" TASK066_REPEAT_LARGE="" \
  BUILD_DIR=build-task066-cb ./tests/run-task066-ubatch-ladder.sh
```

## Configure + build (Metal)

From this repository root (see also [docs/build.md](../docs/build.md) § Metal):

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

`test-metal-cb-fusion` requires `-DGGML_METAL=ON` and static backends (`GGML_BACKEND_DL` off). No GPU; CPU-only unit for CB/fusion align.

`BUILD_DIR` defaults to `<repo>/build`. Override with `BUILD_DIR=/path/to/build` if needed.

## Run

Required selector (fail-closed; no silent default):

```bash
SUITE=all ./tests/run-task066-metal-corners.sh
# or one suite:
SUITE=fa ./tests/run-task066-metal-corners.sh
SUITE=mul-mat ./tests/run-task066-metal-corners.sh
SUITE=longctx ./tests/run-task066-metal-corners.sh
SUITE=moe ./tests/run-task066-metal-corners.sh
SUITE=moe-router ./tests/run-task066-metal-corners.sh
# ST-3.2f-ter-loc MoE out dual-topology (expected RED on A+B; not in `all`):
SUITE=moe-out ./tests/run-task066-metal-corners.sh
# ST-3.2f patch-C TDD fullgraph (expected RED on A+B; not included in `all`):
# Product knobs pinned to gemma4_26b_a4b_mtp_metal.yaml; only TASK066_CTX may shrink.
TASK066_MODEL=/path/to/UD-IQ4_XS.gguf TASK066_CTX=16384 \
  SUITE=fullgraph ./tests/run-task066-metal-corners.sh
# CB split vs fusion unit (Track A RED until follow-up A1-A4):
SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
# Track B follow-up gates (not in `all`; see TASK-066-cb-fusion-followup.md):
SUITE=b1-shared-kv ./tests/run-task066-metal-corners.sh
TASK066_ACC_TRACE_LOG=/path/to/mtp.ndjson SUITE=mtp-acc-trace ./tests/run-task066-metal-corners.sh
TASK066_MODEL=/path/to.gguf TASK066_CTX=2048 SUITE=mtp-e2e ./tests/run-task066-metal-corners.sh
SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
```

Optional: `REBUILD=1` rebuilds suite binaries before run; pass an output directory as argv1 (default: `logs/task066-metal-corners-<suite>-<stamp>/` under this repo).

| SUITE | Runner | What |
|-------|--------|------|
| `fa` | `run-task066-fa-nsg-corners.sh` | FA nsg / row0 (C1/C2/C4 + C6 smoke) |
| `mul-mat` | `run-task066-mul-mat-corners.sh` | mul_mat / mul_mat_id (C1/C4/C5 + C6) |
| `longctx` | `run-task066-longctx-corners.sh` | long-KV FA row0 localizer |
| `moe` | `run-task066-moe-corners.sh` | MoE mul_mv_id IQ4_XS / hetero ids |
| `moe-router` | `run-task066-moe-router-corners.sh` | router→soft_max→top_k→moe |
| `moe-out` | `run-task066-moe-out-corners.sh` | ffn_moe_out permute+sum_rows vs view+add (ST-3.2f-ter-loc RED) |
| `fullgraph` | `run-task066-fullgraph-corners.sh` | llama decode N=1 vs N=2 row0 logits (ST-3.2f RED) |
| `cb-fusion` | `run-task066-cb-fusion-corners.sh` | Track A: `fusion_max` / CB align (ADD/NORM/SNAKE/MUL_MAT+ADD) |
| `b1-shared-kv` | `run-task066-b1-shared-kv-audit.sh` | Track B1: `#27781` shared-KV audit (RED until API) |
| `mtp-acc-trace` | `run-task066-mtp-acc-trace-checklist.sh` | Track B2: require ACC_TRACE log attachment |
| `mtp-e2e` | `run-task066-mtp-e2e-corners.sh` | Track B4: delegates to fullgraph N=1 vs N=2 |
| `mul-mat-residual` | `run-task066-mul-mat-residual-corners.sh` | Track B5: ne11 in {9,16,32} residual |
| `followup` | `run-task066-followup-all.sh` | **one-shot** Track A+B (cb-fusion, B1–B5); continues on FAIL |
| `all` | umbrella | fa…moe-router (**not** follow-up RED suites) |

Host without Metal: binaries exit `77` (ctest SKIP). Treat as host-skip, not a patch KEEP.

## Extending coverage

Add `tests/test-*.cpp` + `tests/run-task066-*.sh`, register the cmake target in `tests/CMakeLists.txt`, and wire the suite name into `run-task066-metal-corners.sh`. Keep fixtures and runners inside this repository.
