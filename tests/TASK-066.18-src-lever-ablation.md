# TASK-066.18 — Src-delta lever ablation (tip−1 / master+1)

**Status:** 🟢 **Mode A DONE (corrected)** · **Mode B DONE** (`…211533Z-fullB`, 2026-11-09)  
**Depends on:** squash tip `999c737a6` on `0fe16d12d`; 066.17 locked root (`size_swa∝ub`); suite map in [`README-TASK066.md`](README-TASK066.md)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)  
**Branch note:** `task066-metal-batch-invariance-rebased` (no force-push to old tip).  
**Policy:** RED is never KEEP. Ablation attributes contribution; it does not waive 066.17.  
**Why not “all off” alone:** all-off ≡ master rebuild — useless as the *only* experiment. Use it as Mode B **baseline**, then enable **one** lever.  
**Human-readable Mode B influence table (metrics + plain language):** [`TASK-066.18-modeB-influence-table.md`](TASK-066.18-modeB-influence-table.md).

---

## Goal

| Mode | Baseline | Action | Answers |
|------|----------|--------|---------|
| **A — tip−1** | tip (all levers ON) | turn **OFF** lever `i` only | “What does this tip change still buy?” |
| **B — master+1** | all levers OFF (master-like) | turn **ON** lever `i` only | “What does this change buy alone?” |

---

## Lever registry

| Lever id | Src cluster | Toggle | Primary suites |
|----------|-------------|--------|----------------|
| `L_fa_tile` | FA always-tile / nsg pin | `TASK066_DISABLE_FA_TILE` | `fa` |
| `L_no_mv_ext` | dead `mul_mv_ext` for ne11 2..8 | `TASK066_DISABLE_NO_MV_EXT` | `mul-mat` |
| `L_float_tile` | q8_0/iq4_xs float tiles + smem | `TASK066_DISABLE_FLOAT_TILE` | `mul-mat-residual`, ubatch |
| `L_mm_min` | MoE id min 8 vs master 32 + env policy | `TASK066_DISABLE_MM_MIN` | `mul-mat-residual` |
| `L_fusion_cb` | fusion_max CB snap | `TASK066_DISABLE_FUSION_CB` | `cb-fusion` |
| `L_size_swa_ub` | tip `PAD128(max(ub,b/4))` vs master `n_ubatch` | `TASK066_DISABLE_SIZE_SWA_UB` | ubatch, `test-iswa-size-swa` |
| `L_iswa_pads` | n_swa pad128 | `TASK066_DISABLE_ISWA_PADS` | `test-iswa-size-swa` |
| `L_shared_kv` | `memory_has_shared_cells` | `TASK066_DISABLE_SHARED_KV` | `b1-shared-kv` |
| `L_swa_hwm_log` | SWA HWM telemetry | `TASK066_DISABLE_SWA_HWM_LOG` | logs only |

cmake: `-DTASK066_ABLATION_DISABLE=FA_TILE;…` (`cmake/task066-ablation-levers.cmake`).

---

## Automation

- Orchestrator: `tests/task066_src_lever_ablation.py`
- Shell: `tests/run-task066-src-lever-ablation.sh`
- Make: `TASK066_MODEL=… TASK066_CTX=3072 make minillm-task066-src-lever-ablation`
- Config: `tests/task066_src_lever_ablation.json`
- Flags: `--mode A|B|both`, `--dry-run`, `--from-lever`, `--from-row`, `--reuse-builds-from`, `--controls-only`
- Artifacts: `tables.md`, `keep_drop.md`, `results.json` under `logs/task066-18-ablation-<UTC>/`

Env (fail-closed for ubatch):  
`TASK066_MODEL=…/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` `TASK066_CTX=3072`  
`BUILD_DIR=…/build-task066-cb-ws` (tip_control reuse).

---

## Mode A — DONE (corrected 2026-11-09)

**Stamp:** [`logs/task066-18-ablation-20261109T193709Z-fullA/`](../logs/task066-18-ablation-20261109T193709Z-fullA/)  
**Note:** first write mis-labeled `S_cb_fusion` / `S_moe_out` as RED (classifier matched prose `"RED gate"` / `"PRE-PATCH RED"`). Reclassified from `run.log` with fixed orchestrator; tables below are **corrected**.

### Tip control (all ON)

| Suite | Verdict | Numeric |
|-------|---------|---------|
| `S_fa` | GREEN | mad=0 |
| `S_mul_mat` | GREEN | abs=0 ulp=0 |
| `S_mul_mat_res` | **RED** | residual vs N=8: abs≈**1.19e-6** ulp=**8** (q8); MoE ne21 abs≈9.5e-7 ulp=13; `fails=18`. MTP verify N=2..4 **PASS** exact |
| `S_cb_fusion` | GREEN | — |
| `S_moe_out` | GREEN | exact=1 abs=0 |
| `S_iswa_unit` | GREEN | tip headroom formula |
| `S_ub_c2pf16` | **RED** | mad≈**0.255** (256↔512, f16/f16) |
| `S_ub_c2pq8` | **RED** | mad≈**3.878** |
| `S_ub_c2p` | **RED** | mad≈**7.635** |
| `S_b1_shared` | GREEN | API present |

### Table A — tip−1 (lever OFF). Cell = verdict (+ numeric when RED / meaningful)

|-------|-----|---------------|-----------------|------------------|----------|-------------|---------------|--------------|-------------|-------------|--------------|
| S_fa | GREEN | **RED** mad=0 | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| S_mul_mat | GREEN | GREEN | **RED** abs≈2.4e-7 ulp=5 | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| S_mul_mat_res | RED abs≈1.19e-6 ulp=8 | RED same | RED same | **RED worse** abs≈**1.1e-4** ulp=**10729** (iq4≈3.5e-4/3224) | RED same | RED same | RED same | RED same | RED same | RED same | RED same |
| S_cb_fusion | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| S_moe_out | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| S_iswa_unit | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | **RED** | **RED** | GREEN | GREEN |
| S_ub_c2pf16 | RED 0.255 | RED 0.255 (=) | RED 0.255 (=) | **RED worse 4.703** (Δ+4.45) | RED 0.255 (=) | RED 0.255 (=) | RED 0.255 (=) | RED 0.255 (=) | RED 0.255 (=) | RED 0.255 (=) | RED 0.255 (=) |
| S_ub_c2pq8 | RED 3.878 | RED 3.878 (=) | RED 3.878 (=) | **RED worse 4.779** (Δ+0.90) | RED 3.878 (=) | RED 3.878 (=) | RED 3.878 (=) | RED 3.878 (=) | RED 3.878 (=) | RED 3.878 (=) | RED 3.878 (=) |
| S_ub_c2p | RED 7.635 | RED 7.635 (=) | RED 7.635 (=) | **RED worse 8.094** (Δ+0.46) | RED 7.635 (=) | RED 7.635 (=) | RED 7.635 (=) | RED 7.635 (=) | RED 7.635 (=) | RED 7.635 (=) | RED 7.635 (=) |
| S_b1_shared | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |

`(=)` = mad identical to tip within stamp precision.

### Keep / drop from Mode A (tip−1 flips + severity)

| Lever | KEEP candidate | DROP candidate | Severity / notes |
|-------|----------------|----------------|------------------|
| `L_fa_tile` | **yes** — OFF → `S_fa` GREEN→RED | no | required for FA suite |
| `L_no_mv_ext` | **yes** — OFF → `S_mul_mat` GREEN→RED (abs≈2.4e-7 ulp=5) | no | required for mul-mat |
| `L_float_tile` | **yes (soft)** — no GREEN↔RED flip, but OFF **worsens** residual ulp 8→10729 and ubatch mad | no | do not drop without Mode B |
| `L_mm_min` | undecided on tip−1 | no | no flip; residual/ubatch still RED same |
| `L_fusion_cb` | undecided | no | `S_cb_fusion` stays GREEN with OFF (this host) |
| `L_size_swa_ub` | unit only (`S_iswa_unit`) | no | ubatch mad **unchanged** (master `n_ubatch` still ∝ub for 256↔512@b=512) |
| `L_iswa_pads` | unit only | no | same as size_swa for ubatch |
| `L_shared_kv` | undecided | no | audit gate always sees API in tree |
| `L_swa_hwm_log` | telemetry | no | no suite flip |

**No DROP from Mode A** (nothing went RED→GREEN when a tip lever was turned OFF).

### Earlier partial stamps (still valid)

- C2pf16-only tip−1: [`…184917Z`](../logs/task066-18-ablation-20261109T184917Z/)  
- master_control C2pf16: [`…190351Z-master-c2pf16`](../logs/task066-18-ablation-20261109T190351Z-master-c2pf16/) — RED mad≈**4.703**  
- master+1 `L_float_tile` C2pf16: [`…190548Z-mplus-float`](../logs/task066-18-ablation-20261109T190548Z-mplus-float/) — mad **4.703→0.255** (still RED; float-tile helps)

---

## Subtask 066.18.B — Mode B full matrix (DONE)

**Status:** 🟢 **DONE** — operator Terminal stamp [`…211533Z-fullB`](../logs/task066-18-ablation-20261109T211533Z-fullB/) (`ABLATION_EXIT=0`, Metal smoke OK, 0× `command queue` fails). Operator log: [`logs/run-modeB-full.log`](../logs/run-modeB-full.log).

### Master control (all tip levers OFF)

| Suite | Verdict | Numeric |
|-------|---------|---------|
| `S_fa` | **RED** | suite RED (orchestrator mad=0 on partial exact rows) |
| `S_mul_mat` | **RED** | abs≈**2.38e-7** ulp=**5** |
| `S_mul_mat_res` | **RED** | abs≈**1.11e-4** ulp=**10735** |
| `S_cb_fusion` | GREEN | — |
| `S_moe_out` | GREEN | exact |
| `S_iswa_unit` | **RED** | fails=4 (master `n_ubatch` headroom) |
| `S_ub_c2pf16` | **RED** | mad≈**4.703** |
| `S_ub_c2pq8` | **RED** | mad≈**4.779** |
| `S_ub_c2p` | **RED** | mad≈**8.094** |
| `S_b1_shared` | **RED** | audit MISSING (no shared-cells API in all-off tree) |

### Table B — master+1 (lever ON alone). Cell = verdict (+ numeric / Δ vs master when meaningful)

|-------|--------|-----------|-------------|--------------|----------|-------------|----------------|--------------|-------------|-------------|--------------|
| S_fa | RED | **GREEN** | RED | RED | RED | RED | RED | RED | RED | RED | RED |
| S_mul_mat | RED abs≈2.4e-7 ulp=5 | RED same | **GREEN** abs=0 | RED same | RED same | RED same | RED same | RED same | RED same | RED same | RED same |
| S_mul_mat_res | RED ulp=10735 | RED same | RED ≈ulp 10729 | **RED better** ulp=**8** (abs≈1.19e-6) | RED same | RED same | RED same | RED same | RED same | RED same | RED same |
| S_cb_fusion | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| S_moe_out | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN | GREEN |
| S_iswa_unit | RED fails=4 | RED | RED | RED | RED | RED | RED | RED fails=1 | RED | RED | RED |
| S_ub_c2pf16 | RED 4.703 | RED 4.703 (=) | RED 4.703 (=) | **RED better 0.255** (Δ−4.45) | RED 4.703 (=) | RED 4.703 (=) | RED 4.703 (=) | RED 4.703 (=) | RED 4.703 (=) | RED 4.703 (=) | RED 4.703 (=) |
| S_ub_c2pq8 | RED 4.779 | RED 4.779 (=) | RED 4.779 (=) | **RED better 3.878** (Δ−0.90) | RED 4.779 (=) | RED 4.779 (=) | RED 4.779 (=) | RED 4.779 (=) | RED 4.779 (=) | RED 4.779 (=) | RED 4.779 (=) |
| S_ub_c2p | RED 8.094 | RED 8.094 (=) | RED 8.094 (=) | **RED better 7.635** (Δ−0.46) | RED 8.094 (=) | RED 8.094 (=) | RED 8.094 (=) | RED 8.094 (=) | RED 8.094 (=) | RED 8.094 (=) | RED 8.094 (=) |
| S_b1_shared | RED | RED | RED | RED | RED | RED | RED | RED | RED | RED | RED |

`(=)` = mad identical to master within stamp precision. **Better ≠ GREEN.**

### Mode B keep / drop (alone-ON flips + severity)

| Lever | Alone helps (RED→GREEN) | Severity / notes | KEEP? |
|-------|-------------------------|------------------|-------|
| `L_fa_tile` | **S_fa** | matches Mode A tip−1 | **KEEP** |
| `L_no_mv_ext` | **S_mul_mat** | abs 2.4e-7→0 | **KEEP** |
| `L_float_tile` | — (still RED) | residual ulp 10735→**8**; c2pf16 4.70→**0.255**; c2pq8/c2p also better | **KEEP (soft)** — severity only |
| `L_mm_min` | — | no flip / no mad Δ | undecided (no alone value here) |
| `L_fusion_cb` | — | cb already GREEN on master | no alone need on this host |
| `L_size_swa_ub` | — | iswa fails 4→1 still RED; **ubatch mad unchanged** | not sufficient alone for 066.17 |
| `L_iswa_pads` | — | no alone GREEN; ubatch (=) | not sufficient alone |
| `L_shared_kv` | — | b1 still MISSING alone | needs tip combo / other deps |
| `L_swa_hwm_log` | — | telemetry | no suite value |

**No alone-harms** (no GREEN→RED). **No KEEP-with-claiming-GREEN from better mad.**

### Invalid agent stamps (do not cite)

`…200140Z-fullB`, `…200925Z-fullB` — Cursor agent sandbox Metal `device=null` (`INVALID.txt`).

### Definition of Done (066.18.B)

1. [x] `master_control` all default rows.  
2. [x] Every `L_*` master+1 column.  
3. [x] Numerics + Δ vs master on RED ubatch/residual.  
4. [x] `keep_drop.md` Mode B alone-helps.  
5. [x] This file Table B + CHANGELOG; no KEEP-with-RED-as-GREEN.  
6. [x] Valid stamp; moe GREEN (not HOST_SKIP).

---

## Combined verdict (A + B)

| Lever | Mode A tip−1 | Mode B master+1 | Decision |
|-------|--------------|-----------------|----------|
| `L_fa_tile` | OFF → fa RED | ON → fa GREEN | **KEEP** |
| `L_no_mv_ext` | OFF → mul_mat RED | ON → mul_mat GREEN | **KEEP** |
| `L_float_tile` | OFF worsens residual/ubatch | ON improves residual/ubatch (still RED) | **KEEP (soft)** |
| `L_size_swa_ub` | tip−1 OFF @ new tip: ubatch GREEN→RED (`…232638Z`) | **alone-ON `PAD128(n_batch)`:** ubatch RED→GREEN mad=0 (`…001626Z`) | **KEEP product** ⊥ub headroom |
| `L_iswa_pads` | unit flips only; ubatch mad (=) under old tip | alone cannot clear ubatch | secondary to pads / not 066.17 alone |
| others | no tip−1 flip | no alone GREEN | low / host-undecided |

### Post-066.17 causality (tip @ `PAD128(n_batch)`, Terminal)

Stamps: tip_control [`…232338Z`](../logs/task066-17-tip-control-20261109T232338Z/) · tip−1 OFF [`…232638Z`](../logs/task066-17-size-swa-off-20261109T232638Z/) · log [`run-06617-tip-control-restamp.log`](../logs/run-06617-tip-control-restamp.log)

| Cell | `S_iswa_unit` | `S_ub_c2pf16` | `S_ub_c2pq8` | `S_ub_c2p` |
|------|---------------|---------------|--------------|------------|
| tip_control | GREEN | GREEN mad=0 | GREEN mad=0 | GREEN mad=0 |
| tip−1 `L_size_swa_ub` OFF | **RED** | **RED mad=0.255** | **RED mad=3.878** | **RED mad=7.635** |

**KEEP product lever** = tip `PAD128(n_batch)` (not the old `max(ub,b/4)` ablation id alone). Mode A/B tables above stay historical for pre-066.17 tip.

### Mode B alone-ON @ `PAD128(n_batch)` — DONE

Stamp: [`…001626Z-modeB-size-swa`](../logs/task066-18-ablation-20261110T001626Z-modeB-size-swa/) · log [`run-06618-modeB-size-swa-alone.log`](../logs/run-06618-modeB-size-swa-alone.log)

| suite | master | `L_size_swa_ub` ON alone | Δ |
|-------|--------|--------------------------|---|
| `S_iswa_unit` | RED | RED | still RED (unit expects tip pads/formula set) |
| `S_ub_c2pf16` | RED mad=4.703 | **GREEN mad=0** | alone helps |
| `S_ub_c2pq8` | RED mad=4.779 | **GREEN mad=0** | alone helps |
| `S_ub_c2p` | RED mad=8.094 | **GREEN mad=0** | alone helps |
| `S_mul_mat_res` | RED ulp=10735 | RED ulp=10735 | (=) — size_swa does not touch residual |

**KEEP product lever:** `PAD128(n_batch)` alone clears product ubatch (no FA/float_tile required).  
Historical Mode B on old `max(ub,b/4)` (ubatch mad (=) master) — superseded for this lever.  
Other levers: still covered by `…211533Z-fullB`.

---

## Definition of Done (066.18 overall)

1. [x] Every lever has ON/OFF switch.  
2. [x] Orchestrator + Make; `--dry-run` + resume.  
3. [x] Mode A tip control + tip−1 full matrix (corrected).  
4. [x] Mode B full matrix (subtask **066.18.B**, `…211533Z-fullB`).  
5. [x] Stamp dirs with tables + keep_drop + results.  
6. [x] Parent pointer sync.  
7. [x] Classifier fixed (no bare `\bRED\b` prose match).

---

## Non-goals

- All-off as the *only* experiment (allowed as Mode B baseline only).  
- Manual per-suite leash as primary path.  
- Waiving 066.17 / soft-ε.  
- Claiming GREEN from “mad got better”.
