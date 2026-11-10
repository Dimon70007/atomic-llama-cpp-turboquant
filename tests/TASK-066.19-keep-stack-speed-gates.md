# TASK-066.19 — KEEP-stack cumulative enable + nomtp/MTP decode floors + final suite report

**Status:** ⬜ **TODO** (spec only — no implementation until user says start)  
**Depends on:** 066.17 DONE (`PAD128(n_batch)`); 066.18 Mode A+B DONE; influence table [`TASK-066.18-modeB-influence-table.md`](TASK-066.18-modeB-influence-table.md) §5  
**Branch:** `task066-metal-batch-invariance-rebased` (no force-push)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)  
**Policy:** RED != KEEP. **NEGATIVE != KEEP.** Speed floor miss = **NEGATIVE** cell + **hard stop** (no waive, no `--continue-on-speed-red`). Better mad != GREEN. Do **not** ship changes that degrade quality **or** decode speed; large speed loss relative to quality gain = **NEGATIVE** (same stop).

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

After `C4_keep4` passes floors (all C0–C4 speed GREEN) → **Phase F** final tip_control (product tip: all tip levers ON as shipped) for full suites + e2e + report.  
If any cell is **NEGATIVE** → stop; Phase F on that failing stack is **forbidden**.

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
- **Hard stop:** if either floor is missed → cell `verdict=NEGATIVE` for the **newly added** pillar; **stop** cumulative enable (no next pillar); do **not** run Phase F on this stack; do **not** mark the pillar KEEP / product-shippable. No continue / waive flag.  
- Same rule if quality RED (suite invariance) on a cell that would otherwise be “faster”: quality degradation = not KEEP (existing RED != KEEP). Speed floors are the numeric bar for this subtask; quality vs speed tradeoff that burns too much tg for the quality gain = **NEGATIVE**.

### Stress harness (DoD shape — implement under §5.8)

Prefer one orchestrator entry (Darwin):

```text
make minillm-task066-keep-stack-speed
  MODEL_NOMTP=gemma4-26b-a4b-mtp-nomtp
  MODEL_MTP=gemma4-26b-a4b-mtp
  TASK066_MODEL=…/UD-IQ4_XS.gguf
  TASK066_MODEL_DRAFT=…/mtp-…Q8_0.gguf
  KEEP_STACK_ARGS='--cell all|--from-cell C2_fa_mv|--phase-f|--dry-run'
  BUILD_DIR=…/build-task066-cb-ws   # tip reuse for C_final_tip when set
```

Thin shell → Python orchestrator:

| Piece | Path |
|-------|------|
| Make | `minillm-task066-keep-stack-speed` |
| Shell | `scripts/macos/minillm/run-task066-keep-stack-speed.sh` |
| Orchestrator | `atomic-llama-cpp-turboquant/tests/task066_keep_stack_speed.py` |
| Config | `tests/task066_keep_stack_speed.json` (cells, floors, bench argv) |
| Flags | `--dry-run`, `--from-cell`, `--skip-build`, `--phase-f`, `--cell` |

**tg measurement (locked):** after each cell rebuild, orchestrator calls fork script
`tests/run-task066-decode-speed-probe.sh` with `BUILD_DIR=<cell build>`
(cwd = `atomic-llama-cpp-turboquant`). Probe runs nomtp `llama-bench` / MTP
`llama-cli -md` + `draft-mtp` from that `BUILD_DIR/bin`. MTP **must** pass
small `-c` (default 2048) + `-st --simple-io` — `llama-cli -c 0` uses
`n_ctx_train` and hangs. Fail-closed if draft GGUF path missing; per-repeat
timeout `MTP_TIMEOUT_SEC` (default 900).

Metal GPU: **macOS Terminal only** (Cursor agent sandbox → `device=null`).

Exclusive GPU: orchestrator/`minillm-stop` before benches (same as 066.18).

## Phase F — final full battery + report

Run **only after** C0–C4 are all speed GREEN (no NEGATIVE cell). First NEGATIVE → skip Phase F for that stack; stamp + REPORT only through the failing cell.

### F1 — tip product binary

Rebuild/install tip_control (all tip levers ON) into documented minillm turboquant prefix (`MINILLM_SYNC_ARGS` / `make minillm-sync-build` as README). Pin SHA in stamp.

### F2 — suites (minimum set)

**Orchestrator (preferred):** after C0–C4 GREEN,

```text
STAMP_DIR=logs/task066-19-keep-stack-<UTC>
BUILD_DIR=…/build-task066-cb-ws
PHASE_F_MODE=all|speed|suites
  ./tests/run-task066-phase-f-battery.sh
# parent: make minillm-task066-phase-f-battery
```

Writes `stamp/final/` scoreboard + appends Phase F section to `REPORT.md`.
`PHASE_F_MODE=speed` = F1 only; `suites` = F2 only (needs `C_final_tip` GREEN).

| Class | Entry | Notes |
|-------|-------|-------|
| TASK-066 Metal corners | `SUITE=all` via battery / `run-task066-metal-corners.sh` | Include every suite name containing `task066` / TASK-066 runners under `tests/run-task066-*.sh` registered in README |
| Unit | `test-iswa-size-swa` | CPU |
| Ubatch product | `SUITE=ubatch-s8-s1024` (full; expect stress C1 may RED — record, no waive) | |
| e2e / product | `make verify-minillm-macos MODEL=gemma4-26b-a4b-mtp` + `MODEL=gemma4-26b-a4b-mtp-nomtp` | operator after tip sync; stamp under `final/verify-minillm-macos/` |
| MTP quality smoke | `run-task066-mtp-vs-nomtp-ladder.sh` (battery) | stamp PASS/DIFF telemetry |

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
2. Speed table: cell × (nomtp tg, mtp tg) vs floors 23 / 30 — pass / **NEGATIVE**  
3. First **NEGATIVE** pillar (if any) + Δ tok/s vs previous cell; product KEEP list = **only** pillars that stayed above floors (prior cells may remain candidates; the breaching add is NEGATIVE)  
4. Final suite scoreboard (GREEN/RED + mad/ulp when present) — only if Phase F ran  
5. e2e verify outcomes — only if Phase F ran  
6. Plain-language summary (what got slower, what stayed invariant, what was rejected as NEGATIVE)  
7. Links to raw logs  

## Automation DoD (§5.8)

- [x] JSON cell matrix + floors (no silent MODEL/ctx) — `tests/task066_keep_stack_speed.json`
- [x] Orchestrator builds cumulative ablation dirs (reuse 066.18 cmake lever flags) — `tests/task066_keep_stack_speed.py`
- [x] Make target Darwin-only — `make minillm-task066-keep-stack-speed`
- [x] `--dry-run` / `--from-cell` / resume
- [x] dry-run must **not** clobber real `results.json` (writes `stamp/dry-run/`)
- [x] Floor miss → exit hard-stop path with `verdict=NEGATIVE` (exit 3; no continue flag)

## Definition of Done

1. [x] Spec linked from 066.18 influence table §5 + README-TASK066 + parent pointer
2. [x] Orchestrator + Make + shell landed
3. [ ] Terminal stamp: C0–C4 speed cells with nomtp≥23 and mtp≥30 **or** first **NEGATIVE** cell named with numbers (hard stop; no further pillars / no Phase F on that stack)
4. [ ] Phase F final suites + e2e stamped **only if** C0–C4 all speed GREEN
5. [ ] `REPORT.md` complete (generated per stamp; full after Terminal run)
6. [x] CHANGELOG (fork + product)
7. [ ] No force-push; tip commit only on explicit ask

## Non-goals

- Re-running full Mode B alone-ON matrix
- Changing product YAML ubatch/batch without separate TASK
- Waiving stress C1 or residual RED as KEEP
- Waiving a speed floor because invariance mad went GREEN
- Shipping a slower stack because quality “looked better”
- Claiming floors from agent-sandbox Metal runs
- Any `--continue-on-speed-red` / silent KEEP after floor miss

## Locked impl choices (coding start)

1. **tg source:** fork `tests/run-task066-decode-speed-probe.sh` after each cell rebuild (`BUILD_DIR=stamp/builds/<cell>`). Probe owns llama-bench / llama-cli argv. Short windows (066.15 class), not filled 140k ring.
2. **Phase F speed re-check:** **yes** — `--phase-f` adds `C_final_tip` (tip_all_on); floor miss there = NEGATIVE tip ship. Full suite battery remains operator Terminal after C0–C4 GREEN.
3. **`L_iswa_pads` after C4:** **no** (Mode B secondary).
