# TASK-066.19 — KEEP-stack cumulative enable + nomtp/MTP decode floors + final suite report

**Status:** ⬜ **TODO** (spec only — no implementation until user says start)  
**Depends on:** 066.17 DONE (`PAD128(n_batch)`); 066.18 Mode A+B DONE; influence table [`TASK-066.18-modeB-influence-table.md`](TASK-066.18-modeB-influence-table.md) §5  
**Branch:** `task066-metal-batch-invariance-rebased` (no force-push)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)  
**Policy:** RED != KEEP. Speed floor miss = RED cell (not waive). Better mad != GREEN.

## Why

Mode B named four tip pillars that alone (or soft) buy invariance. Product also needs **decode speed** under the same nomtp/MTP configs operators run. Enable those pillars **cumulatively** (one at a time), measure decode after each add, then run a **final** full TASK-066 + e2e battery and write a human report.

## KEEP stack order (from influence table §5) — cumulative

Start from **master-like** (all `TASK066_DISABLE_*=ON` / `ABLATION_DISABLE=all`).  
Each cell **keeps** prior pillars ON and adds the next:

| Step | Cell id | Enabled levers (ON) | Still OFF (ablation) |
|------|---------|---------------------|----------------------|
| 0 | `C0_master` | (none) | all tip levers |
| 1 | `C1_fa` | `L_fa_tile` | rest |
| 2 | `C2_fa_mv` | + `L_no_mv_ext` | rest |
| 3 | `C3_fa_mv_float` | + `L_float_tile` | rest |
| 4 | `C4_keep4` | + `L_size_swa_ub` (`PAD128(n_batch)` in tip src) | rest (`L_mm_min`, fusion, moe_view, pads, shared_kv, hwm, …) |

**Not in §5 cumulative (explicit):** `L_mm_min` / product `MUL_MM_MIN_DEFAULT` — pin **documented** for all speed cells (fail-closed env or tip DEFAULT); do not silently flip between cells. Recommend record both: (a) pin used, (b) tip DEFAULT value at HEAD.

After `C4_keep4` passes floors → **Phase F** final tip_control (product tip: all tip levers ON as shipped) for full suites + e2e + report.

## Product configs (lock — no silent alias)

| Role | Alias | Profile YAML |
|------|-------|--------------|
| nomtp | `gemma4-26b-a4b-mtp-nomtp` | `configs/minillm/gemma4_26b_a4b_mtp_metal_nomtp.yaml` |
| MTP | `gemma4-26b-a4b-mtp` | `configs/minillm/gemma4_26b_a4b_mtp_metal.yaml` |

Shared host knobs (must match YAML): ctx **140000**, batch **2048**, ubatch **512**, `q8_0` / `turbo4`, FA on, threads 4, Unsloth sampling 1.0/0.95/64. MTP only adds draft + `spec_type=draft-mtp` `n_max=2` `p_min=0.95`.

Fail-closed: `MODEL=` / alias required on every Make entry; refuse silent default.

## Decode speed gates (hard floors)

| Workload | Metric | Floor | How measured |
|----------|--------|-------|--------------|
| nomtp | decode **tg** (tok/s) | **≥ 23** | stress harness below |
| MTP | decode **tg** (tok/s) | **≥ 30** | same harness, MTP alias |

- Report **median** of ≥3 timed decode windows (same prompt budget / `-n` as harness lock).  
- Prefill (pp) logged as secondary (no floor unless added later).  
- Cell **FAIL** if either floor missed → stop cumulative (no skip to next pillar) unless operator passes `--continue-on-speed-red` **and** stamp marks RED (never silent KEEP).

### Stress harness (DoD shape — implement under §5.8)

Prefer one orchestrator entry (Darwin):

```text
make minillm-task066-keep-stack-speed
  MODEL_NOMTP=gemma4-26b-a4b-mtp-nomtp
  MODEL_MTP=gemma4-26b-a4b-mtp
  CELL=C1_fa|…|C4_keep4|all
  BUILD_DIR=…/build-task066-cb-ws   # tip rebuild / ablation cell build
```

Thin shell → Python orchestrator:

| Piece | Path (planned) |
|-------|----------------|
| Make | `minillm-task066-keep-stack-speed` |
| Shell | `scripts/macos/minillm/run-task066-keep-stack-speed.sh` |
| Orchestrator | `atomic-llama-cpp-turboquant/tests/task066_keep_stack_speed.py` |
| Config | `tests/task066_keep_stack_speed.json` (cells, floors, bench argv) |
| Flags | `--dry-run`, `--from-cell`, `--skip-build`, `--continue-on-speed-red` |

Measurement options (pick **one** primary in impl, document in stamp):

1. **Preferred product path:** overlay cell build → `make switch-model MODEL=…` → timed decode via existing minillm probe/bench used for MTP/nomtp A/B (reuse TASK-064/066 decode timing if present; else `llama-bench` from install prefix with YAML-equivalent flags).  
2. **Fork-only fallback:** `llama-bench` from ablation `BUILD_DIR` with explicit `-m -b 2048 -ub 512 -ctk q8_0 -ctv turbo4 -fa on -ngl 99` (+ draft flags for MTP). Fail-closed if draft GGUF path missing.

Metal GPU: **macOS Terminal only** (Cursor agent sandbox → `device=null`).

Exclusive GPU: orchestrator/`minillm-stop` before benches (same as 066.18).

## Phase F — final full battery + report

Run **after** `C4_keep4` speed GREEN (or documented RED with continue).

### F1 — tip product binary

Rebuild/install tip_control (all tip levers ON) into documented minillm turboquant prefix (`MINILLM_SYNC_ARGS` / `make minillm-sync-build` as README). Pin SHA in stamp.

### F2 — suites (minimum set)

| Class | Entry | Notes |
|-------|-------|-------|
| TASK-066 Metal corners | `SUITE=all` `make minillm-task066-metal-corners` (or fork `run-task066-metal-corners.sh`) | Include every suite name containing `task066` / TASK-066 runners under `tests/run-task066-*.sh` registered in README |
| Unit | `test-iswa-size-swa` | CPU |
| Ubatch product | `SUITE=ubatch-s8-s1024` (full; expect stress C1 may RED — record, no waive) | |
| e2e / product | `make verify-minillm-macos MODEL=gemma4-26b-a4b-mtp` + `MODEL=gemma4-26b-a4b-mtp-nomtp` | |
| MTP quality smoke | existing mtp-vs-nomtp / duel ladder entry used in 066.9/066.16 | stamp PASS/DIFF telemetry |

First unexpected RED (not in known-residual allowlist written in this file before run) → report RED section; do not mark 066.19 Done.

**Known residual allowlist (pre-declared):**

- Stress C1 16↔1024 RED (066.5/066.6)  
- `S_mul_mat_res` / residual vs N=8 class under tip float-tile (066.7) if still RED with numbers  
- `S_b1_shared` MISSING if API still absent  

Anything else RED = new finding.

### F3 — report artifact

Write stamp dir:

```text
logs/task066-19-keep-stack-<UTC>/
  README.md
  cells/C0_…/ speed json + raw
  cells/C1_…/ …
  cells/C4_…/ …
  final/ suite exits + summaries
  REPORT.md          # human report (required)
  results.json
```

**`REPORT.md` must include:**

1. HEAD SHA + cmake ablation maps per cell  
2. Speed table: cell × (nomtp tg, mtp tg) vs floors 23 / 30 — pass/fail  
3. Which pillar first breached a floor (if any) + Δ tok/s vs previous cell  
4. Final suite scoreboard (GREEN/RED + mad/ulp when present)  
5. e2e verify outcomes  
6. Plain-language summary (what got slower, what stayed invariant)  
7. Links to raw logs  

## Automation DoD (§5.8)

- [ ] JSON cell matrix + floors (no silent MODEL/ctx)  
- [ ] Orchestrator builds cumulative ablation dirs (reuse 066.18 cmake lever flags)  
- [ ] Make target Darwin-only  
- [ ] `--dry-run` / `--from-cell` / resume  
- [ ] dry-run must **not** clobber real `results.json` (lesson from 066.18)

## Definition of Done

1. [ ] Spec linked from 066.18 influence table §5 + README-TASK066 + parent pointer  
2. [ ] Orchestrator + Make + shell landed  
3. [ ] Terminal stamp: C0–C4 speed cells with nomtp≥23 and mtp≥30 **or** first RED cell named with numbers  
4. [ ] Phase F final suites + e2e stamped  
5. [ ] `REPORT.md` complete  
6. [ ] CHANGELOG (fork + product if parent scripts)  
7. [ ] No force-push; tip commit only on explicit ask  

## Non-goals

- Re-running full Mode B alone-ON matrix  
- Changing product YAML ubatch/batch without separate TASK  
- Waiving stress C1 or residual RED as KEEP  
- Claiming floors from agent-sandbox Metal runs  

## Open impl choices (resolve at coding start — ask if unclear)

1. Primary tg source: install-prefix server timed decode vs `llama-bench` from cell `BUILD_DIR`.  
2. Whether Phase F tip_control also re-checks speed floors (recommend **yes**, one row `C_final_tip`).  
3. Whether `L_iswa_pads` should join cumulative after C4 (secondary in Mode B — default **no** unless unit RED blocks product).
