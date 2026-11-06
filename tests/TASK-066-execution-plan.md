# TASK-066 — Coherent execution plan (post answers)

> **For agentic workers:** implement task-by-task; checkboxes track progress. Decisions locked from [draft_mtp_research_task-066_answers.md](draft_mtp_research_task-066_answers.md). Do not re-open resolved forks without updating that file.

**Goal:** Lower Metal nomtp ubatch noise enough that the fixed fixture stops "floating" at `temperature=0`, with residual noise documented (WAIVE). MTP vs nomtp is a **separate follow-up** after that stand is usable.

**Architecture:** Snapshot a short ubatch ladder (same-ubatch repeats first), env-gate `ne11_mm_min` (32 vs upstream 8), keep landed CB/fuse (#28164/#31688), then audit/port skinny-mm / split-K (#27441) as needed. Unit `followup-all` stays code regression; product ladder is a separate gate.

**Tech Stack:** Metal ggml, `llama-cli` (fast) / `llama-server`+yaml (canon), Gemma4 UD-IQ4_XS, ~2k prompt.

**Sources / decisions:** answers.md; testing.md; follow-up; inventory; upstream-patches-plan (Phases 1-3 done).

## Locked decisions (no contradictions)

| Topic | Decision |
|-------|----------|
| Stage 1 Done | Noise low enough on fixture + explicit WAIVE for residual; **not** required bit-exact to `ubatch=1` |
| Stage 2 MTP | **Separate follow-up** after nomtp ladder closed / waived |
| Full MUL_MAT invariance | **Out of scope** as a hard goal |
| Golden | Lab/diagnostic: **`ubatch<=8`**; max purity optional: `ubatch=1` |
| **Product ubatch** | **LOCK:** floor **256**, target **512**; owned large also **`{1024, 2048}`**. S16 = near-baseline under pin 32, not prod gate |
| `ne11_mm_min` | **env-gate** (default keep current 32 for product if desired; debug with `8`) |
| Ladder (time-box) | `{1, 8, 16, 33, 256, 512, 1024, 2048}` — product focus **`{256, 512}`** + owned large **`{1024, 2048}`** (+ S8 golden) |
| Same-ubatch | **2x** on `{1024, 2048}` (run1/run2 drift check); owned large band must stay same-ubatch clean |
| batch vs ubatch | Ladder default: **`-b=max(N,512), -ub=N`** (floor); avoid fixed `-b 2048` (Metal OOM on M4) |
| Fixture | `unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_XS` (local `.scratch`), ~2k prompt, `n_predict` 256-512, `temperature=0`, thinking off |
| FA / cache | FA **on**; KV **locked** `q8_0`/`turbo4` (product) for all TASK-066 stamps |
| Binary | Canon: **llama-server + yaml**; fast loop: **llama-cli** same knobs |
| Fix order | **same-ubatch** then **cross-ubatch**; ready mm PRs exhausted (#27441 CLOSED, #30100 N/A); FA/MoE only after "not matmul" localization |
| Unit vs product | `SUITE=followup` = unit only; ladder = product gate |

## Kernel map while debugging

With env `GGML_METAL_MUL_MM_MIN` (name TBD in impl):

| ubatch | min=8 (upstream-like) | min=32 (current B5 default) |
|--------|----------------------|-----------------------------|
| 1, 8 | `mul_mv` | `mul_mv` |
| 16 | early `mul_mm` / skinny | still `mul_mv` |
| 33+ | `mul_mm` | `mul_mm` |

**Debug default for Stage 0/1 ladder:** run with **min=8** so S16 exercises skinny/mm. Keep min=32 as optional A/B stamp and as possible product default after WAIVE.

---

## Phase P0 — Fixture + env-gate (code)

### Task P0.1: Env-gate `ne11_mm_min`

- [x] In `ggml-metal-ops.cpp`, replace hard `const int ne11_mm_min = 32` with once-init from env (e.g. `GGML_METAL_MUL_MM_MIN`), default **32** (preserve current B5 behavior).
- [x] Mirror for MoE if needed (`ne21 > 32` stays unless env says otherwise); document in inventory.
- [x] Rebuild Metal; `SUITE=mul-mat-residual` still PASS at default 32; with `GGML_METAL_MUL_MM_MIN=8` residual expectations may RED — document or gate residual suite on default only.

### Task P0.2: Fixture README for ladder

- [x] Add `tests/TASK-066-ladder-fixture.md` (or section in follow-up) with exact: model path template, prompt path, `n_predict`, FA, cache types, `batch=ubatch`, server yaml key names, cli flag equivalents.
- [x] Local GGUF via `.scratch/gemma-4-26b-a4b/...UD-IQ4_XS.gguf` (`setup-gemma4-26b-a4b-models.sh`; same canon as `run-gemma4-*.sh`).

### Task P0.3: Ladder runner (minimal)

- [x] Add `tests/run-task066-ubatch-ladder.sh`: loops ubatch in `{1,8,16,33,256,2048}`, optional 2x for large, writes `logs/task066-ubatch-ladder-<stamp>/`, emits stub `matrix.md` (token-file diff / first disagree helper).
- [x] Wire mention in README-TASK066; **not** into `SUITE=followup`.

---

## Phase P1 — Stage 0 baseline (product, Metal Terminal)

### Task P1.1: Primary stamp (`batch=ubatch`, `GGML_METAL_MUL_MM_MIN=8`)

- [x] Capture cells `{1,8,16,33,256,512,1024,2048}` nomtp, FA on, KV `q8_0`/`turbo4`, temp=0.
- [x] 2x repeats for `{1024,2048}`.
- [x] Fill `matrix.md`: vs S8 (and vs S1 if taken); S16 vs S8; S33 vs S16; S512 vs S1024; S1024 vs S2048; same-ubatch run1 vs run2.
- **Stamp:** `logs/task066-ubatch-ladder-20261106T170944Z` — S1=S8; same-ubatch OK; cross DIFF from S16; S1024=S2048.

### Task P1.2: Optional stamps (do not block P2) — **now required before #27441 port**

- [x] Same ladder with default pin **32** (A/B). Stamp `logs/task066-ubatch-ladder-20261106T172821Z`: **S16==S8**; S33/S256 DIFF.
- [ ] One `batch=max, ubatch=N` pass for prod-shaped graph (subset OK: 256 + 2048) — optional, not blocking Stage 1.

**Exit P1:** DONE (min=8 + min=32 A/B).

---

## Phase P2 — Stage 1 fix nomtp noise

### Task P2.1: Same-ubatch first

- [x] If S256/S2048 run1≠run2: prioritize determinism — **N/A on stamp 170944Z** (S1024/S2048 run1=run2 identical).
- [x] Re-run only those cells until stable or WAIVE with evidence.

### Task P2.2: Cross-ubatch — adapt upstream patches (not invent knobs)

Authority: [TASK-066-upstream-patches-plan.md](TASK-066-upstream-patches-plan.md) Phase 4.3.
Stage 0 ladder **done** (stamps 170944Z / 172821Z). Phases 1–3 fuse/CB (#28164/#31688) already in tree.

- [x] Confirm CB fuse still live; same-ubatch clean.
- [x] Audit **#27441** — [TASK-066-p22-27441-audit.md](TASK-066-p22-27441-audit.md).
- [x] Surgical port of **#27441** tried; stamp `175708Z` showed **regression** (S16/S33 first DIFF @80 vs baseline @314/@401).
- [x] **Reverted** #27441 Metal port (back to HEAD); kept B5 env-gate / CB / TQ pins. See audit. **CLOSED.**
- [x] Do **not** treat raising `MUL_MM_MIN` (e.g. to 256) as the Stage 1 fix — that hides mm instead of adapting the patch.
- [x] Re-check **#30100**: N/A (no MMA+ADD encode). [TASK-066-p22-30100-recheck.md](TASK-066-p22-30100-recheck.md).

### Task P2.3: Owned large band `{256,512,1024,2048}`

Ready upstream mm patches exhausted (#27441 closed, #30100 N/A). **Product lens (LOCK):** floor ubatch **256**, target **512**; also own **1024/2048** noise; S16 is diagnostic baseline under pin 32, not DoD.

Evidence (stock `mul_mm`):
- [x] `{8,33,256}` min=32 stamp `181846Z` == `172821Z`. S256 vs S8 DIFF @**80**.
- [x] Full ladder min=8 stamp `170944Z`: S256/S512/S1024/S2048 vs S8 DIFF @80; **S256 vs S512 DIFF @314**; S512 vs S1024 DIFF @134; **S1024==S2048**; same-ubatch OK on 1024/2048.
- [x] **WAIVE** `{256,512,1024,2048}` vs S8 (Metal large-mm; 512+ drift may be quality-favorable) — [TASK-066-stage1-waive.md](TASK-066-stage1-waive.md).
- [x] Do not chase S16/S33 as product blockers.
- [x] Optional completeness: stamp `184055Z` `{8,256,512}` default pin 32 — S512==`170944Z`; S256/S512 vs S8 DIFF@80; S256 vs S512 DIFF@314.

### Task P2.4: Close Stage 1

- [x] #27441 owned closed (tried+reverted); #30100 N/A.
- [x] Explicit WAIVE for **`{256,512,1024,2048}`** vs golden on fixture — [TASK-066-stage1-waive.md](TASK-066-stage1-waive.md).
- [x] Unit: mm/CB surface green at default 32 (`SUITE=all` PASS; followup cb-fusion/B1/residual PASS; `mtp-acc-trace` expected FAIL without log).
- [x] Product default `GGML_METAL_MUL_MM_MIN=32` kept (S16 diagnostic-quiet).

**Exit P2:** DONE — large-band residual WAIVEd -> Stage 2 MTP unlocked at ubatch **512** (floor 256; 1024/2048 owned).

---

## Phase P3 — MTP follow-up (UNLOCKED)

Stage 1 closed via WAIVE. Run MTP vs nomtp on product band only.

- [x] M1 runner exists (`run-task066-mtp-vs-nomtp-ladder.sh`); stamp `...173712Z` INVALID (no engagement evidence / need `-lv 3`).
- [ ] M1 at ubatch **512** (floor 256) with valid engagement (`-lv 3`); then M2–M4 as needed.

---

## Phase P4 — Cleanup

- [x] Sync inventory / follow-up status to "Stage 1 closed" (WAIVE doc).
- [ ] A5 comment banners (optional cleanup).
- [ ] Commit slices only when user asks (fuse/B5/env-gate/ladder script/docs/WAIVE).

---

## Debug playbook (short)

1. Env `GGML_METAL_MUL_MM_MIN=8` for ladder diagnosis.
2. Same-ubatch 2x before trusting cross-ubatch diffs.
3. CB in tree; #27441 CLOSED; #30100 N/A; Stage 1 `{256,512,1024,2048}` vs S8 **WAIVEd**.
4. FA/MoE only if a future quieter band is required (not Stage 1).
5. Never disable multi-CB as a "fix"; never require eternal bit-exact to ubatch=1.

## Risk register

| Risk | Mitigation |
|------|------------|
| Env-gate breaks residual suite at min=8 | Gate residual on default 32; document |
| #27441 large / conflicts TurboQuant | Surgical port; pin-off stamps |
| Same-ubatch noise masks progress | Fix or WAIVE before cross-ubatch |
| Server vs cli drift | Same knobs documented in fixture file |
| Agent cannot Metal | Operator runs P1/P2 product; agent does P0 code + docs |

## Suggested commit slices (when asked)

1. `ggml : env-gate Metal mul_mm batch threshold`
2. `tests : TASK-066 ubatch ladder runner + fixture`
3. `ggml : Metal skinny-batch / split-K (#27441)` (if needed)
4. `docs : Stage 1 WAIVE / ladder results`
