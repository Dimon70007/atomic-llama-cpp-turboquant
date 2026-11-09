# TASK-066.16 — Triage RED suites after `MUL_MM_MIN_DEFAULT=1`

**Status:** 🟢 **DONE** — R1–R9 all **final P**; **no Q**; recommend **KEEP `MUL_MM_MIN_DEFAULT=1`**  
**Depends on:** 066.15 env + min=1 quality probe; stamp `logs/task066-15-full-suite-20261109T042939Z/`  
**Branch:** `task066-metal-batch-invariance`  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)

## Stamps (this subtask)

| Stamp | Role |
|-------|------|
| [`logs/task066-16-r8-ubatch-c3-20261109T044702Z/`](../../logs/task066-16-r8-ubatch-c3-20261109T044702Z/) | R8 control: C3_n1280 RED @ min=0 **and** min=1 (pad128 + headroom=`n_ubatch`) |
| [`logs/task066-16-pad256-ubatch-c3-20261109T052740Z/`](../../logs/task066-16-pad256-ubatch-c3-20261109T052740Z/) | outer pad 256 only: C3_n1280 GREEN; C3_n1536 RED |
| [`logs/task066-16-headroom-pad128max-ub-b4-20261109T060048Z/`](../../logs/task066-16-headroom-pad128max-ub-b4-20261109T060048Z/) | product formula: C2+C3_n1280+C3_n1536 GREEN; C1 RED (unequal-b) |
| [`logs/task066-16-r9-equalb-c1-20261109T061835Z/`](../../logs/task066-16-r9-equalb-c1-20261109T061835Z/) | R9 equal-b C1 RED both mins (size_swa∝ub) |
| [`logs/task066-16-r1r5-control-20261109T062918Z/`](../../logs/task066-16-r1r5-control-20261109T062918Z/) | **R1–R7** control A/B same binary; SUMMARY + exits |

## Why

After pinning `GGML_METAL_MUL_MM_MIN_DEFAULT=1` (decode speed; A2 rem still closed; MTP matmul hazard re-opened), the full TASK-066 Metal suite mixed:

- **expected** bit-exact REDs from the policy change,
- **fixture / harness** failures that do not prove product regression,
- **possible real** quality / cross-ub / MoE-out regressions that need isolation.

This subtask **classifies every RED** before any “fix” — so we do not confuse broken tests, intentional policy, and actual degradations.

## Evidence lock (066.15 baseline)

| Artifact | Role |
|----------|------|
| [`logs/task066-15-full-suite-20261109T042939Z/`](../../logs/task066-15-full-suite-20261109T042939Z/) | full suite @ DEFAULT=1 |
| [`…/SUMMARY.md`](../../logs/task066-15-full-suite-20261109T042939Z/SUMMARY.md) | scoreboard |
| [`logs/task066-15-min1-mtp-quality-20261109T041653Z/`](../../logs/task066-15-min1-mtp-quality-20261109T041653Z/) | mtp-vs-nomtp min=0 vs min=1 quality |
| [`logs/task066-15-perf-knobs-20261109T025448Z/`](../../logs/task066-15-perf-knobs-20261109T025448Z/) | decode tax numbers (tg ~10 vs ~25) |

## Classification rubric (mandatory per RED)

Assign **exactly one primary** label; secondary allowed in notes.

| Code | Name | Meaning | Typical action |
|------|------|---------|----------------|
| **T** | Test / harness | Gate asserts wrong contract for current product policy, or runner/fixture broken | Update suite gate / SKIP / fixture; do **not** change kernels to “please” the old assert |
| **P** | Policy / business logic | RED is the **intended** consequence of KEEP knobs (here: `min=1` → N=1 `mul_mv`, N≥2 `mul_mm`) | Document as accepted tax **or** pursue A+rem / min=0 with named tradeoff |
| **Q** | Quality degradation | Wrong tokens / facts / acceptance vs prior KEEP under same product knobs | Root-cause; may block product DEFAULT=1 |
| **S** | Speed degradation | tok/s regress vs stamped baseline without explaining invariance win | Perf fix or waive with numbers |
| **F** | Fixture / env | `FIXTURE_FAIL`, bad CTX/prompt, missing MODEL, wrong BUILD_DIR | Fix runner args; re-run; do not count as product RED |

**Control method (every item):** re-run the same suite with `GGML_METAL_MUL_MM_MIN=0` (Phase B) **and** unset/`=1` on the **same** binary/build.  

| Outcome | Classification hint |
|---------|---------------------|
| RED only at min=1, GREEN at min=0, matches N=1 vs N≥2 matmul story | **P** (unless product forbids that hazard) |
| RED at **both** 0 and 1 | **T** / pre-existing / **Q** unrelated to min pin — dig further |
| GREEN at both | was flaky / fixed harness |
| FIXTURE_FAIL either way | **F** |

## GREEN lock (do not re-litigate without new stamp)

From `…042939Z` @ DEFAULT=1:

- fa, longctx, cb-fusion, mul-mat-residual (vs N=8 + policy unit)
- mtp-vs-nomtp product duel_hits 2/2, acceptance ~0.893

From `…060048Z` (size_swa formula below) @ min=0 and min=1:

- ubatch **C2**, **C3_n1280**, **C3_n1536** bit-exact logits PASS

## Product `size_swa` formula (landed in tip; 066.16 follow-up)

```text
headroom = PAD128(max(n_ubatch, n_batch / 4))
size_swa = PAD256(n_swa_eff + headroom)
```

| Intent | Effect |
|--------|--------|
| Product YAML `b=2048 ub=512` | `max(512,512)=512` — unchanged vs pure `n_ubatch` |
| Anti-footgun C3 (`b=1536 ub=8`) | `PAD128(max(8,384))=384` → `size_swa=1536` — C3_n1536 GREEN |
| Upstream canon | physical write headroom is **ubatch** (PR #13833); logical `n_batch` need not fit in ring (PR #13194 historically used `n_batch`, then shrunk) |

Helpers: `llama_kv_cache_iswa_calc_swa_headroom` / `llama_kv_cache_iswa_calc_size_swa` in `src/llama-kv-cache-iswa.{h,cpp}`; product path `llama_swa_headroom_product` in `src/llama-model.cpp`. Unit: `test-iswa-size-swa`.

## Session results scoreboard (ubatch track)

### R8 baseline (pad128 + headroom=`n_ubatch` only) — `…044702Z`

| Cell | C2 | C3_n1280 | Notes |
|------|----|----------|-------|
| min=0 | PASS | **REGRESS** mad≈14.35 | `SWA ring full size=1152` |
| min=1 | PASS | **REGRESS** mad≈14.35 | same → **not** min=1 |

**Final R8: P** (066.14 `size_swa∝n_ubatch`; does not block DEFAULT=1).

### pad256-only — `…052740Z`

| Cell | C2 | C3_n1280 | C3_n1536 |
|------|----|----------|----------|
| min=0/1 | PASS | **PASS** | RED mad≈7.63 (multi 1280 vs oneshot 2560) |

### headroom `PAD128(max(ub,b/4))` + pad256 — `…060048Z` (**current**)

| Cell | C2 | C3_n1280 | C3_n1536 | C1 greedy S8 vs S1024 |
|------|----|----------|----------|------------------------|
| min=0 | **PASS** | **PASS** | **PASS** | **RED** `first_token_diff=3` (tok 110549 vs 1246) |
| min=1 | **PASS** | **PASS** | **PASS** | **RED** `first_token_diff=8` (tok 107 vs 236771) |

Suite exit=1 **only** from C1 (C1j not reached). Prompt 1696 tok; `n_predict=64`; pair `A b=512 ub=8` / `B b=1024 ub=1024`.

## RED checklist (one row = one verification)

Provisional label = first-pass guess from `…042939Z`; **final** only after control A/B + notes.

### R1 — `SUITE=mul-mat` C1/C4/C5 (N=1 vs {2,3,4,5,7})

- [x] **Verify** min=0 **PASS** exact / min=1 **RED** max_ulp≈8 (ne21≈13)  
- [x] **Classify:** **final P** — classic MTP cross-kernel; do not “fix” by DEFAULT→0  
- [x] Final: **P** / stamp [`…062918Z`](../../logs/task066-16-r1r5-control-20261109T062918Z/)

### R2 — residual `--mtp-verify` (telemetry under residual PASS)

- [x] residual vs N=8 **GREEN** @ min=0 and min=1  
- [x] `--mtp-verify` **GREEN** @ min=0; **RED** @ min=1 (suite still PASS — telemetry)  
- [x] **Classify:** **final P**  
- [x] Final: **P** / stamp [`…062918Z`](../../logs/task066-16-r1r5-control-20261109T062918Z/)

### R3 — `SUITE=moe` (mul_mv_id vs mul_mm_id widths)

- [x] Control A/B: min=0 **PASS** / min=1 **RED**  
- [x] **Classify:** **final P**  
- [x] Final: **P** / stamp [`…062918Z`](../../logs/task066-16-r1r5-control-20261109T062918Z/)

### R4 — `SUITE=moe-router` (mad_logits ~1e-6; ids match)

- [x] Control A/B: min=0 **PASS** / min=1 **RED** mad_logits~1e-6  
- [x] **ids=1** on all failing rows — logits-only, not routing flip → not **Q**  
- [x] **Classify:** **final P**  
- [x] Final: **P** / stamp [`…062918Z`](../../logs/task066-16-r1r5-control-20261109T062918Z/)

### R5 — `SUITE=moe-out` (ffn_moe_out row0)

- [x] Control A/B: min=0 **PASS** / min=1 **RED** fails=2  
- [x] **Classify:** **final P** (not pre-existing under min=0; not product **Q**)  
- [x] Final: **P** / stamp [`…062918Z`](../../logs/task066-16-r1r5-control-20261109T062918Z/)

### R6 — `SUITE=fullgraph`

- [x] Prior `…042939Z` empty prompt → **F**; re-run with `TASK066_CTX=3072` + pierre prompt  
- [x] min=0 **GREEN** exact; min=1 **RED** mad≈0.83 (argmax still match)  
- [x] **Classify:** **final P** (fixture fixed; residual is N=1 vs N=2 policy)  
- [x] Final: **P** / stamp [`…062918Z`](../../logs/task066-16-r1r5-control-20261109T062918Z/)

### R7 — followup `mtp-e2e` (fullgraph delegate)

- [x] Same fixture as R6: min=0 **PASS** / min=1 **RED**  
- [x] **Classify:** **final P** (cascade of R6 policy)  
- [x] Final: **P** / stamp [`…062918Z`](../../logs/task066-16-r1r5-control-20261109T062918Z/)

### R8 — `SUITE=ubatch-s8-s1024` — **C3 n=1280 REGRESS** (at DEFAULT=1 full suite)

- [x] Control A/B min=0 vs 1 **and** note `size_swa` / `-lv 4` calc for ub=8 vs ub=32  
- [x] C2 was PASS @ min=1 — isolate C3 only  
- [x] **Classify:** **final P** (066.14 `size_swa∝n_ubatch`; **not** min=1)  
- [x] Follow-ups closed C3 gates under new formula (see scoreboard)  
- [x] Final: **P** / stamp [`…044702Z`](../../logs/task066-16-r8-ubatch-c3-20261109T044702Z/) (cause); mitigated in tip by pad256 + `PAD128(max(ub,b/4))` [`…060048Z`](../../logs/task066-16-headroom-pad128max-ub-b4-20261109T060048Z/)

### R9 — ubatch greedy C1 S8 vs S1024 (product ladder pair)

- [x] After C3 GREEN, C1 now runs — stamp [`…060048Z`](../../logs/task066-16-headroom-pad128max-ub-b4-20261109T060048Z/)  
- [x] Evidence (unequal-b product pair): RED @ **both** min=0 and min=1; `first_token_diff` 3 vs 8  
- [x] Equal-`-b` control: `TASK066_DIAG=c1 TASK066_C1_BATCH=1024` — stamp [`…061835Z`](../../logs/task066-16-r9-equalb-c1-20261109T061835Z/)  
  - min=0: `first_token_diff=3` (110549 vs 1246) — **same** as product pair  
  - min=1: `first_token_diff=8` (107 vs 236771) — **same** as product pair  
  - A still `size_swa=1280` (ring-full), B `size_swa=2048` — equal-b does **not** equalize ring under tip formula  
- [x] **Classify:** **final P** — residual is `size_swa∝ub` via `max(ub,b/4)` (066.14 pivot); chunking/`-b` ruled out; min path modulates which token diverges but both RED → not min=1 alone. Historical H1 GREEN was under `size_swa∝n_batch` (equal-b ⇒ equal size_swa).  
- [x] Final: **P** / stamp [`…061835Z`](../../logs/task066-16-r9-equalb-c1-20261109T061835Z/)

**What C1 checks:** two fresh contexts; same raw prompt (1696 tok); greedy argmax `n_predict=64`; require identical token ids for  
`A: -b 512 -ub 8` vs `B: -b 1024 -ub 1024` (product; or equal-b via `TASK066_C1_BATCH`). Not logits-only. C1j (jinja) runs only after raw C1 GREEN.

## Product recommendation

**KEEP `GGML_METAL_MUL_MM_MIN_DEFAULT=1`.**

| Option | Verdict |
|--------|---------|
| KEEP DEFAULT=1 | **Chosen** — decode speed (066.15); no **Q**; bit-exact suite REDs are **P** |
| Revert DEFAULT→0 | Rejected — pays ~2.45× tg tax for autotest cosmetics already waived as policy |
| A+rem (min↑ + rem-rebalance) | Optional follow-up to recover bit-exact MTP/MoE/fullgraph gates without decode tax |

R8/R9 (`size_swa∝ub`) stay accepted C1 tax under 066.14 pivot — orthogonal to min pin.

## Out of scope for 066.16

- Implementing A+rem (min=8 + rem-rebalance) — follow-up after classifications freeze  
- YAML `n_max`/`p_min` flips  
- Treating bit-exact **P** gates as product **Q**

## Definition of Done

- [x] Every R1–R9 has **final** rubric code + one-line rationale + control stamp path  
- [x] No **Q** left unexplained (none found; R4 ids match; R5 GREEN @ min=0)  
- [x] No **T**-only gates requiring suite rewrite this turn (R2 already treats mtp-verify as telemetry)  
- [x] Pointer + `README-TASK066.md` + CHANGELOG updated  
- [x] Explicit recommendation: **KEEP DEFAULT=1** (A+rem optional)

## Non-goals

- Silent undo of FA always-tile / float-tile  
- Treating FIXTURE_FAIL as product quality fail  
- “All suites GREEN” as DoD while DEFAULT=1 without policy-aware gates
