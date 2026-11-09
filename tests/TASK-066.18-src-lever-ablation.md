# TASK-066.18 — Src-delta lever ablation (tip−1 / master+1)

**Status:** 🟡 **IMPL IN PROGRESS** (2026-11-09) — levers + orchestrator landed; matrix stamps open  
**Depends on:** squash tip `999c737a6` on `0fe16d12d` (single commit vs upstream release); 066.17 locked root (`size_swa∝ub`); suite map in [`README-TASK066.md`](README-TASK066.md)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)  
**Branch note:** operator push uses `task066-metal-batch-invariance-rebased` (no force-push to old tip).  
**Why not “all off”:** disabling every tip change at once ≡ rebuild of `0fe16d12d` — useless. Ablation is **one lever at a time**.

---

## Goal

For each logical src change vs master, measure **which suites flip** when that change is alone toggled:

| Mode | Baseline | Action | Answers |
|------|----------|--------|---------|
| **A — tip−1** | tip (all levers ON) | turn **OFF** lever `i` only | “What does this change still buy on tip?” |
| **B — master+1** | `0fe16d12d` (all OFF) | turn **ON** lever `i` only | “What does this change buy alone?” |

Collect two result tables (rows = suites/gates, columns = levers) for **ON** and **OFF** (or tip / tip−i / master / master+i — see DoD).

**Policy:** RED is never KEEP. Ablation explains contribution; it does not waive 066.17.

---

## Lever registry (must match src delta)

IDs are stable selectors for cmake/env/orchestrator. One run changes **exactly one** lever from the mode baseline.

| Lever id | Src cluster | Default tip | Toggle mechanism (impl) | Primary suites |
|----------|-------------|-------------|-------------------------|----------------|
| `L_fa_tile` | FA always-tile / nsg pin (`ggml-metal-ops`) | ON | compile `#ifndef TASK066_DISABLE_FA_TILE` (or equiv.) | `fa` |
| `L_no_mv_ext` | disable `mul_mv_ext` for bad ne11 | ON | compile flag | `mul-mat` |
| `L_float_tile` | q8_0/iq4_xs float tiles + smem | ON | compile flag / metal template | `mul-mat-residual`, `mul-mat` |
| `L_mm_min` | `MUL_MM_MIN` policy default | ON (=8) | **runtime** `GGML_METAL_MUL_MM_MIN` + optional DEFAULT rebuild | `mul-mat-residual`, `test-metal-mul-mm-policy`, ubatch C1j |
| `L_fusion_cb` | `ggml-metal-fusion` + CB pack | ON | compile flag (skip fusion_max → upstream CB) | `cb-fusion` |
| `L_moe_view_add` | always view+ADD MoE out | ON | compile flag restore N=1 permute path | `moe-out` |
| `L_size_swa_ub` | headroom `PAD128(max(ub,b/4))` | ON | compile/runtime: tip formula vs **batch-only** / **fixed** (066.13/14 class) | `ubatch-s8-s1024` (C2/C3/C2p/C2pq8/C2pf16), `test-iswa-size-swa` |
| `L_iswa_pads` | n_swa pad128 + outer pad256 | ON | compile flag | `test-iswa-size-swa`, ubatch C3 |
| `L_shared_kv` | `get_has_shared_cells` + speculative | ON | compile flag | `b1-shared-kv` |
| `L_swa_hwm_log` | occupancy HWM / ring-full warn | ON | compile flag (telemetry only) | log presence; optional skip in matrix |

Optional later (not blocking DoD): longctx / moe-router / fullgraph as **extra** suite rows if a lever touches them.

**Anti-pattern:** one cmake switch that disables “all TASK-066” — forbidden as primary matrix (≡ master).

---

## Suites / gates (rows of the result table)

Minimum rows (fail-closed names):

| Row id | Command / gate |
|--------|----------------|
| `S_fa` | `SUITE=fa` |
| `S_mul_mat` | `SUITE=mul-mat` |
| `S_mul_mat_res` | `SUITE=mul-mat-residual` |
| `S_cb_fusion` | `SUITE=cb-fusion` |
| `S_moe_out` | `SUITE=moe-out` |
| `S_iswa_unit` | `test-iswa-size-swa` (CPU) |
| `S_ub_c2` | ubatch C2 |
| `S_ub_c3` | ubatch C3 (n1280 + n1536) |
| `S_ub_c2pf16` | `TASK066_DIAG=c1p-f16` (C2pf16 only is enough for first matrix) |
| `S_ub_c2pq8` | `TASK066_DIAG=c1p-q8` |
| `S_ub_c2p` | `TASK066_DIAG=c1p` (C2p+C1p product KV; stop at first RED OK) |
| `S_b1_shared` | `SUITE=b1-shared-kv` |

Env for product ubatch rows (fail-closed):  
`TASK066_MODEL=…/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` `TASK066_CTX=3072`  
`BUILD_DIR=…/build-task066-cb-ws` · GPU: `required_permissions: all`.

Cell values: `GREEN` / `RED` / `SKIP` / `MISUSE` + optional `mad=` / `first_token_diff=` for ubatch rows.

---

## Automation (§5.8)

**Single entry:** Makefile target → thin shell → Python orchestrator.

Proposed names (impl may adjust, keep one Make verb):

- Orchestrator: `tests/task066_src_lever_ablation.py`
- Runner: `tests/run-task066-src-lever-ablation.sh`
- Make: `make task066-src-lever-ablation` (or document under fork README-TASK066 only until parent wires it)

**Config:** `tests/task066_src_lever_ablation.json` — lever ids, suite rows, mode A/B, `--dry-run`, `--from-lever`, resume, stamp dir.

**Per cell:**

1. Select mode + lever (exactly one delta from baseline).  
2. Configure/build if compile lever changed (reuse build dir per lever id).  
3. Run suite row; capture exit + key log lines.  
4. Append to stamp JSON/CSV + human markdown table.

**Stamp root:** `logs/task066-18-ablation-<UTC>/`

Must support `--dry-run` and resume (`--from-lever` / `--from-row`).

---

## Result tables (DoD artifacts)

### Table A — tip−1 (tip ON except listed OFF)

Rows = suite ids; columns = `tip` (control) + each `L_*` (that lever OFF).

### Table B — master+1 (master OFF except listed ON)

Rows = suite ids; columns = `master` (control) + each `L_*` (that lever ON).

Controls first (no lever change): stamp tip once, stamp `0fe16d12d` once (or rebuild with all levers OFF).

Interpret: if `S_ub_c2pf16` stays RED when `L_size_swa_ub` OFF on tip (tip−1) **or** goes GREEN when ON alone on master — confirms / refines 066.17 lock. Other levers should mostly affect their primary suites only.

---

## Implementation order (when «start implementation»)

1. Introduce **compile/runtime levers** for each `L_*` (fail-closed unknown id; no silent “all off”).  
2. JSON + orchestrator + shell + Make; dry-run prints matrix.  
3. Run controls (tip + master).  
4. Run mode A then mode B (or A-only first if master+1 build cost high — document choice in stamp README).  
5. Write tables into stamp + update this file DoD checkboxes.  
6. CHANGELOG + README-TASK066 blurb **same changeset as code** (§5.0). Parent pointer MITL.

**Out of scope:** fixing 066.17 GREEN inside this task (follow-up after ablation evidence); soft-ε; waiving RED.

---

## Definition of Done

1. [x] Every lever in the registry has a documented ON/OFF switch (compile and/or runtime).  
2. [x] Orchestrator + Make entry; `--dry-run` + resume (`--from-lever` / `--from-row`).  
3. [ ] Control stamps: tip all-ON + master all-OFF.  
4. [ ] Mode A table complete (tip−1 per lever × suite rows).  
5. [ ] Mode B table complete **or** explicit deferral recorded with reason + Make flag.  
6. [ ] Stamp dir under `logs/task066-18-ablation-…/` with machine-readable + markdown tables (live GPU).  
7. [ ] This file + parent pointer updated; no KEEP-with-RED language.

### Impl notes (2026-11-09)

| Piece | Path |
|-------|------|
| cmake levers | `cmake/task066-ablation-levers.cmake` → `-DTASK066_ABLATION_DISABLE=SIZE_SWA_UB;…` |
| tip ON | no `TASK066_DISABLE_*` |
| tip−1 | one `TASK066_DISABLE_<ID>` |
| master all-OFF | all `TASK066_DISABLE_*` |
| master+1 | all DISABLE except the lever under test |
| orchestrator | `tests/task066_src_lever_ablation.py` |
| shell | `tests/run-task066-src-lever-ablation.sh` |
| Make | `make minillm-task066-src-lever-ablation` (parent) |

Dry-run evidence: `logs/task066-18-ablation-20261109T184013Z/` (matrix print only).

---

## Non-goals / anti-patterns

- Disabling all levers in one shot as the only experiment.  
- Manual “run suite X by hand for each lever” as primary path (§5.8).  
- Silent default lever list / silent MODEL/CTX.  
- Claiming 066.17 fixed because ablation “looks better” without GREEN gates.
