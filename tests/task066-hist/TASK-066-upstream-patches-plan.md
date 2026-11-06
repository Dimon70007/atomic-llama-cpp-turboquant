# TASK-066 — Plan: apply / debug upstream Metal MTP patches

> **Superseded for product work:** use [TASK-066-execution-plan.md](TASK-066-execution-plan.md) (answers locked). This file remains the **history** of Phases 0-3 upstream ports.

> **For agentic workers (unit regression):** `./tests/run-task066-followup-all.sh`.

**Goal (original):** Port Metal fuse table + CB align. **Product DoD now:** noise-low-enough nomtp ladder (see execution plan); MTP is a later follow-up.

**Architecture:** Port dependency order from [draft_mtp_research_task-066_upstream_patches.md](draft_mtp_research_task-066_upstream_patches.md): first the shared fuse table (**#28164**, already merged upstream), then CB split align (**#31688**, open PR — apply via diff, not cherry-pick from master), then verify / port **#30100** (`MUL_MAT+ADD` residual fix). Debug with existing Track A/B suites in [TASK-066-cb-fusion-followup.md](TASK-066-cb-fusion-followup.md).

**Tech Stack:** ggml Metal (`ggml/src/ggml-metal/`), fork branch `task066-metal-batch-invariance`, upstream `ggml-org/llama.cpp` PRs, `./tests/run-task066-followup-all.sh`.

**Sources:**

| PR | Upstream status (per research draft) | Role |
|----|--------------------------------------|------|
| [#28164](https://github.com/ggml-org/llama.cpp/pull/28164) | Merged | `ggml-metal-fuse.cpp` + real `ggml_metal_fusion_max()` |
| [#31688](https://github.com/ggml-org/llama.cpp/pull/31688) | **Open** (not in master) | CB `cb_start` / boundary walk using fusion_max |
| [#30100](https://github.com/ggml-org/llama.cpp/pull/30100) | Merged / in releases (e.g. b11475) | Correct `MUL_MAT+ADD` when residual is itself `MUL_MAT` |
| [#27781](https://github.com/ggml-org/llama.cpp/pull/27781) | Separate MTP KV (Track B1) | Not Metal fuse; keep as parallel audit |

## Global Constraints

- Remotes: this fork is **not** a clean ggml-org tree; expect conflicts with TurboQuant / TASK-066 Metal pins (`#25618` mul_mv_ext / FA nsg). Prefer surgical port over blind `git apply` of the whole PR when files diverge.
- Do **not** cherry-pick #31688 from `master` (not merged). Fetch PR diff / PR head.
- Do not weaken RED tests to go green; fix code or document SKIP/WAIVE per follow-up rules.
- ASCII only in code comments (`-`, `->`, `...`); no emdash / unicode arrows.
- Baseline before each port: `./tests/run-task066-followup-all.sh` (or `SUITE=followup`) and save `logs/.../summary.txt`.
- Staged fork adaptation (`58d1940` + uncommitted follow-up tests) already wires `cb_start[]` + ADD-only `fusion_max`. Treat that as a **partial #31688**; replacing the helper body is the main work after #28164.

## Current fork state

- Done (2026-11-06): surgical port of upstream `ggml-metal-fusion.{h,cpp}` (#28164); `SUITE=cb-fusion` GREEN.
- Done: `cb_start[]` walk already matched #31688 (blind apply conflicted because already present).
- #30100: re-checked 2026-11-06 — still N/A (no few-row MMA+ADD encode; STRUCTURAL stubs only). See [TASK-066-p22-30100-recheck.md](TASK-066-p22-30100-recheck.md).
- Done (2026-11-06): port `#27781` `llama_memory_has_shared_cells`; `SUITE=b1-shared-kv` PASS.
- Done (2026-11-06): B5 residual pin; `SUITE=mul-mat-residual` PASS on M4.
- Done (2026-11-06): **Stage 0 ubatch ladder** stamps (see Task 4.2).
- Done (2026-11-06): **#27441 CLOSED** — port tried, stamp `175708Z` regressed, reverted. Do **not** invent break-even knobs. Stage 1 **WAIVEd** `{256,512,1024,2048}` vs S8 ([TASK-066-stage1-waive.md](TASK-066-stage1-waive.md); S1024==S2048). Stage 2 MTP unlocked at ubatch 512.

## File map (expected touches)

| File / area | Responsibility |
|-------------|----------------|
| `ggml/src/ggml-metal/ggml-metal-fuse.{cpp,h}` (new or port) | Shared fuse table + `ggml_metal_fusion_max` source of truth |
| `ggml/src/ggml-metal/CMakeLists.txt` | Build fuse sources into `ggml-metal` |
| `ggml/src/ggml-metal/ggml-metal-common.{cpp,h}` | Remove or thin-wrap heuristic; keep API used by context |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp` | Encoder fuse paths must call same table (A2/A4); #30100 MUL_MAT+ADD |
| `ggml/src/ggml-metal/ggml-metal-context.m` | Keep/adjust `cb_start` walk to match upstream #31688 after table lands |
| `tests/test-metal-cb-fusion.cpp` | Track A RED -> GREEN |
| `tests/run-task066-followup-all.sh` | One-command gate |

---

## Phase 0 — Baseline and fetch (no product change)

### Task 0.1: Capture RED baseline

- [x] Ensure Metal build exists (`-DGGML_METAL=ON`, static backends / `GGML_BACKEND_DL=OFF` for `test-metal-cb-fusion`).
- [x] Run:
  ```bash
  BUILD_DIR=build-task066-cb ./tests/run-task066-followup-all.sh \
    logs/task066-baseline-before-upstream-$(date -u +%Y%m%dT%H%M%SZ)
  ```
- [ ] Save `summary.txt` / `results.tsv`. Expect FAIL on `cb-fusion` (A1/A3/A4), `b1-shared-kv`, `mtp-acc-trace` (unless log set).
- [ ] Optional product baseline (if model available):
  ```bash
  TASK066_MODEL=... TASK066_CTX=2048 TASK066_FOLLOWUP_REQUIRE_E2E=1 \
    ./tests/run-task066-followup-all.sh logs/task066-baseline-e2e-...
  ```

### Task 0.2: Fetch upstream PR artifacts into a scratch dir

- [x] Create `logs/upstream-prs/` (gitignored via `logs/` if already ignored).
- [x] Fetch diffs (needs network / `gh` auth to ggml-org):
  ```bash
  mkdir -p logs/upstream-prs
  gh pr diff 28164 --repo ggml-org/llama.cpp > logs/upstream-prs/pr28164.diff
  gh pr diff 31688 --repo ggml-org/llama.cpp > logs/upstream-prs/pr31688.diff
  gh pr diff 30100 --repo ggml-org/llama.cpp > logs/upstream-prs/pr30100.diff
  # fallback:
  # curl -L https://github.com/ggml-org/llama.cpp/pull/NNNNN.diff -o logs/upstream-prs/prNNNNN.diff
  ```
- [ ] Also save PR heads as refs if useful:
  ```bash
  git fetch https://github.com/ggml-org/llama.cpp pull/28164/head:refs/pr/28164
  git fetch https://github.com/ggml-org/llama.cpp pull/31688/head:refs/pr/31688
  git fetch https://github.com/ggml-org/llama.cpp pull/30100/head:refs/pr/30100
  ```
- [ ] Dry-check against working tree (expect conflicts on Metal files):
  ```bash
  git apply --check logs/upstream-prs/pr28164.diff || true
  git apply --check logs/upstream-prs/pr31688.diff || true
  git apply --check logs/upstream-prs/pr30100.diff || true
  ```
- [ ] Write a short note in the out dir: which files conflict and why (TurboQuant / staged heuristic).

---

## Phase 1 — Port #28164 (fuse table)  [dependency of #31688]

### Task 1.1: Inventory upstream fuse API vs fork

- [x] Upstream symbols: `ggml_metal_fusion_max` + table in `ggml-metal-fusion.{h,cpp}` (not `ggml-metal-fuse`).
- [x] Blind `git apply` of full PR fails; surgical port of fusion sources + CMake + includes.
- [x] Keep C API used by `context.m`; remove ADD-only heuristic from `ggml-metal-common.cpp`.

### Task 1.2: TDD — keep Track A RED, then implement table

- [x] Port `ggml-metal-fusion.{h,cpp}`; wire CMakeLists; stubs for `ggml_metal_op_mul_mat_may_use_mma` / `use_mma`.
- [x] Align `test-metal-cb-fusion` expectations with table (ADD max 7, over-count OK for VIEW).
- [x] `SUITE=cb-fusion` GREEN after port (`logs/task066-after-upstream-28164`).

### Task 1.3: Encoder uses the same table (A2)

- [x] Table is shared source of truth for `fusion_max`; encoder still has its own fuse loops (same rules as upstream #28164).
- [ ] Optional: further unify encoder loops onto helper (not required for Track A green).

**Exit criteria Phase 1:** met — fuse table present; Track A GREEN.

---

## Phase 2 — Align / finish #31688 (CB boundaries)

### Task 2.1: Compare fork `cb_start` walk to PR #31688

- [x] Fork already had `cb_start[]` walk; blind apply of #31688 conflicts because present.
- [x] Keep fork walk; Phase 1 replaced fusion_max backend only.
- [x] `SUITE=cb-fusion` GREEN after Phase 1 (table + walk).

### Task 2.2: Apply remaining #31688 hunks surgically

- [x] No further #31688 hunks required for Track A.
- [ ] Product ACC_TRACE (Task 2.3) still open — needs model run.

### Task 2.3: Product debug loop (short 2k)

- [ ] Unset `LLAMA_MTP_SKIP_STREAK_THRESHOLD`.
- [ ] Run greedy vs MTP on the 2k fixture (`temperature=0`).
- [ ] Attach ACC_TRACE + `SUITE=mtp-acc-trace`.
- [ ] If still diverging: Phase 4 / Track B.

**Exit criteria Phase 2 (code):** met. Product bit-exact: open (needs B2/B4).

---

## Phase 3 — Audit / port #30100 (`MUL_MAT+ADD`)

### Task 3.1: Presence check

- [x] Fork has no few-row MMA + ADD encode path that #30100 fixes.
- [x] Structural `mul_mat_add` in `test-metal-cb-fusion` GREEN via fusion table; encode port N/A.
- [x] Inventory updated: KEEP structural; #30100 N/A (re-check 2026-11-06).

### Task 3.2: Gate

- [x] Structural gate OK.
- [x] `SUITE=mul-mat-residual` PASS on M4 after B5 pin.

**Exit criteria Phase 3:** met as deferred-with-reason for #30100 encode; residual unit green.

---

## Phase 4 — Product debug (after Phases 1–3 + B5 pin)

Unit gates green. Focus per [draft_mtp_research_task-066_testing.md](draft_mtp_research_task-066_testing.md):
**Stage 1 = nomtp high-ubatch (fast)**; **Stage 2 = MTP vs nomtp** only after Stage 1 is quiet enough.
Shared root with #19034 / CB fuse / MUL_MAT batch paths — not two unrelated bugs.

### Task 4.1: B1 + B5 (done)

- [x] `#27781` + `SUITE=b1-shared-kv` PASS.
- [x] B5 residual pin; `SUITE=mul-mat-residual` PASS on M4.

### Task 4.2: Stage 0 — nomtp ubatch ladder snapshots (do first)

Same 2k prompt, `temperature=0`, **MTP off**. Out: `logs/task066-ubatch-ladder-<stamp>/`.

- [x] Fixture + runner: `tests/run-task066-ubatch-ladder.sh`, `TASK-066-ladder-fixture.md` (KV locked `q8_0`/`turbo4`, floor batch).
- [x] Primary stamp min=8: `logs/task066-ubatch-ladder-20261106T170944Z` — S1=S8; same-ubatch OK; cross DIFF from S16; S1024=S2048.
- [x] Product pin A/B min=32: `logs/task066-ubatch-ladder-20261106T172821Z` — S16=S8; S33/S256 DIFF.
- [x] Env-gate `GGML_METAL_MUL_MM_MIN` (default 32) already in tree (B5).

**Stage 0 exit: met.** Do not invent further product knobs before adapting remaining upstream patches.

### Task 4.3: Stage 1 — adapt upstream patches for nomtp mm drift (PRIMARY)

Baseline = Stage 0 stamps. **Adapt ready PRs; do not invent break-even workarounds** (e.g. raising `MUL_MM_MIN` to 256 is not a substitute for #27441).

- [x] Same-ubatch clean on stamp 170944Z — proceed to cross-ubatch / mm patches.
- [x] Audit **#27441** vs fork: [TASK-066-p22-27441-audit.md](TASK-066-p22-27441-audit.md) (open draft; blind apply fails; IQ4_XS covered; no TQ).
- [x] **Surgical adapt #27441** tried; stamp `175708Z` regressed vs `170944Z` (S16/S33 first DIFF earlier).
- [x] **Reverted** #27441 port (Metal files restored to HEAD). Keep B5 / CB / TQ. Evidence stamp retained. **CLOSED.**
- [x] Re-check **#30100**: still N/A ([TASK-066-p22-30100-recheck.md](TASK-066-p22-30100-recheck.md)).
- [x] P2.3 cheap reconfirm: stamp `181846Z` == `172821Z` for S8/S33/S256; residual S33@253 / S256@80 vs S8.
- [x] Milestone: **WAIVE** `{256,512,1024,2048}` vs S8 — [TASK-066-stage1-waive.md](TASK-066-stage1-waive.md). No further knobs; FA/MoE localize not required for Stage 1.
- [x] Do **not** burn full S1 every iteration.

### Task 4.4: Stage 2 — MTP vs nomtp (UNLOCKED)

- [ ] **M1** at ubatch **512** (floor 256) with `-lv 3` engagement evidence (stamp `...173712Z` INVALID).
- [ ] **M2** ACC_TRACE if M1 red; **M3** draft knobs if logic-shaped.
- [ ] **M4**: MTP at ubatch=1 only as finish arbiter.
- [ ] `SUITE=mtp-e2e` when MODEL/CTX set.

---

## Phase 5 — Cleanup and acceptance

- [ ] Collapse duplicate TASK-066 comment banners (A5).
- [ ] Fix obsolete heuristic comments if any remain.
- [x] Follow-up doc refocused on Stage 1 high-ubatch nomtp.
- [x] Stage 1 product notes / WAIVE.
- [ ] Final Stage 2 product notes after M1.
- [ ] Acceptance = unit green + ladder quiet vs S1 (or owned) + MTP≡nomtp at high ubatch (or owned). Stage 1 half met via WAIVE.

---

## Debug playbook

1. **Stage 0 first:** ladder + **S33** + same-ubatch 2x + matrix.
2. **Stage 1:** #27441 CLOSED; `{256,512,1024,2048}` vs S8 **WAIVEd**; S1 only at milestones.
3. Goal: noise low enough — not necessarily full MUL_MAT invariance or eternal S1 bit-exact.
4. **Stage 2:** MTP vs nomtp at ubatch **512** (floor 256) with engagement evidence; S1 MTP finish only.
5. Never disable multi-CB as a "fix".

## Risk register

| Risk | Mitigation |
|------|------------|
| Blind `git apply` of #31688 over fork Metal | Phase 0 dry-check; surgical merge; keep TurboQuant kernels |
| #28164 large / moves fuse out of ops.cpp | Port table first; keep `fusion_max` symbol stable |
| Open #31688 changes while we port | Pin fetched diff SHA/date in `logs/upstream-prs/README.txt` |
| Tests green, product still red | Stage 0 ladder; do not loosen unit gates |
| B5 pin hides skinny (#27441) path | pin-off compare stamp; env-gate if needed |
| Same-ubatch nondeterminism (#19034) | 2x repeats before cross-ubatch debugging |
| #30100 needs shader changes | Build Metal + `SUITE=mul-mat` before claiming done |

## Suggested commit slices (when asked to commit)

1. `ggml : port Metal fuse table (#28164)`
2. `ggml : Metal CB split uses shared fusion_max (#31688)`
3. `ggml : fix Metal MUL_MAT+ADD residual fuse (#30100)` (if needed)
4. `tests : TASK-066 follow-up gates green / notes`

Do not commit until the user asks.
