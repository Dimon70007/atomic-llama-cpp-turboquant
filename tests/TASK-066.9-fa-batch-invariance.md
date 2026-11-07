# TASK-066.9 — FA batch-invariance (nomtp-first, MTP second)

**Status:** 🟢 Phase A1 **DONE**; 🟢 **Phase A2 DONE** (jinja-ladder / rem cross-kernel); Phase B MTP gated (explicit start)  
**Depends on:** [066.7 KEEP](TASK-066.7-mul-mat-residual.md); C2 GREEN (FA always-tile); research below  
**Next:** Phase B MTP only after explicit start  
**Research:** [task066-hist/draft_mtp_research_task-066_fa.md](task066-hist/draft_mtp_research_task-066_fa.md), [task066-hist/TASK-066-fa-batch-invariance-web-research.md](task066-hist/TASK-066-fa-batch-invariance-web-research.md)  
**Index:** [README-TASK066.md](README-TASK066.md) · [PR audit](TASK-066-pr-audit-registry.md)

## Priority (hard)

```text
Phase A1 (nomtp C2/C3/C1 + ISWA)  →  Phase A2 (jinja-ladder residual)  →  Phase B (MTP / #28108)
```

A2 found during Phase A confirm; closed 2026-11-07. MTP still needs **explicit start**.

---

## Frozen results — this work package (2026-11-07)

### A. FA always-tile (C2 closed)

| Item | Detail |
|------|--------|
| **Root cause** | `use_vec` iff `ne01 < 20` → `ub=8` **vec** vs `ub≥20` **tile** (H6 n=16 both-vec GREEN; H7/C2 mixed RED) |
| **Code** | `ggml_metal_op_flash_attn_ext_use_vec` — default **always tile**; legacy shape-vec only `GGML_METAL_FA_VEC=1`; log `GGML_METAL_FA_DISPATCH=1` |
| **Autotest** | FA **C7** multi-step causal (n=16/32/…/256): legacy vec RED@32 → default tile GREEN (incl. q8/turbo4) |
| **C2** | **GREEN** bit-exact (`logs/task066-c2-after-tile-*`) |
| **A0–A4 @C2** | all **exact=1** after tile (`logs/task066-regression-after-tile-20261106T231654Z`) |
| **Regression** | `cb-fusion` / `mul-mat-residual` / `fa` **GREEN** |
| **C1** | still RED; HELP `first_token_diff` **0→8** |
| **Ladder** | `231804Z` S1024 vs S8 DIFF@**329** (HELP vs `212703Z`@182) |
| **Product** | unchanged: FA-on + `ctv=turbo4` |

### B. Phase A localize (long prefill — non-FA class named)

Stamps: `logs/task066-localize-threshold-20261106T233840Z`, `…-bisect-20261106T234258Z`, `…-faoff-20261106T235017Z`, `logs/task066-c3-gate-20261106T235908Z`.

| Row | Result | Read |
|-----|--------|------|
| H1/H2 n=512 | exact | Short prefill OK after tile |
| H8… n≤1280 (product pair) | exact | |
| H8 / H10 n=1344 | **RED** | First miss; soft then decision-level as n grows |
| H10 n=1280 pure ub=8 vs ub=1280 | **GREEN** | Regress lock |
| H9 n=1536 pure = product pair | **RED** | Pure **multi-step**, not `-b` shape |
| H9f16 FA-on | **RED** | Not turbo-only |
| **H9faOFF** q8/f16 FA-off | **RED** (mad≈4.8, argmax DIFF) | **Not FA-only** |
| FA C7 op ≤256 | GREEN | Isolated FA multi-step ≠ C3 residual |

**Named residual class (pre-fix):** long-prefill **multi-step ubatch** for **`n≥1344`** (lock **`n=1280` GREEN**). Survives **FA-off** → was **non-FA**. Root after bisect/thresh: **ISWA `size_swa` ∝ `n_ubatch`** (Gemma4 n_swa=1024 → ub=8 capacity 1280). Fixed in [066.5](TASK-066.5-fusion-table-parity.md). FA-only KEEP still forbidden.

### C. Autotest gates (`SUITE=ubatch-s8-s1024`)

| Gate | Meaning | Status |
|------|---------|--------|
| **C2** | n=32 ub=8 vs ub=32 | **GREEN** (must not regress) |
| **C3** | n=1280 GREEN lock; **n=1344** bit-exact after ISWA fix | **GREEN** (`066.5`) |
| **C1** | greedy S8 vs S1024 | **GREEN** (n_predict=64) |

### D. Spec / wiring delivered

- This file + [066.0](TASK-066.0-overview.md) + [README-TASK066](README-TASK066.md) + [PR audit](TASK-066-pr-audit-registry.md)
- Draft moved to [task066-hist/draft_mtp_research_task-066_fa.md](task066-hist/draft_mtp_research_task-066_fa.md); stub at `tests/draft_mtp_research_task-066_fa.md`
- Product pointer: `linux_to_mac/tasks/TASK-066-nomtp-ubatch-s8-s1024.md`
- [066.5](TASK-066.5-fusion-table-parity.md) retargeted: C3-n1344 + H9faOFF

---

## Goal

1. **Phase A1:** Close nomtp cross-ubatch drift at autotest C2/C3/C1 (ISWA KEEP) — **done**.  
2. **Phase A2:** Close product **jinja-ladder** S8 vs S1024 residual — **done** (`023433Z`; `MUL_MM_MIN` default 1 + C1j).  
3. **Phase B:** Metal unify FA geometry for MTP verify widths `{1..4}` (#28108 spirit); then turbo4 × speculative — without dropping turbo.

## Evidence hierarchy (066.0 lock 7)

Every Metal/ops/fusion edit: autotest RED→GREEN same change set; re-run `cb-fusion`, `mul-mat-residual`, `fa`, **C2**, **C3-n1280**. No ε-PASS C3/C1/C2.

---

## Phase A1 — nomtp autotest (DONE)

### Done

- [x] Always-tile + C2 GREEN + C7  
- [x] Localize; **non-FA** class named (H9faOFF)  
- [x] C3 gate wired → C3-n1344 GREEN → C1 GREEN (066.5 ISWA)  
- [x] Autotest diff per lever (C3 RED→GREEN)  
- [x] Ladder stamp vs `231804Z` — **HELP** surfaced A2 (`012625Z`)

### Phase A1 DoD

- [x] C2 GREEN; residual class named (**non-FA** / ISWA)  
- [x] C3-n1344 GREEN → C1 GREEN  
- [x] Confirm stamp logged (HELP → opens A2, does not unlock MTP)

---

## Phase A2 — jinja-ladder residual (DONE 2026-11-07)

**Why A2:** found while finishing Phase A confirm. Product ladder `--jinja --conversation`; raw C1 GREEN ≠ ladder Done.

| Item | Detail |
|------|--------|
| Pre-fix stamp | `logs/task066-ubatch-ladder-20261107T012625Z` — DIFF@22 |
| Post-fix stamp | `logs/task066-ubatch-ladder-20261107T023433Z` — S8≡S256≡S512≡S1024 |
| Fixture | [TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md) |

### Root cause (named)

Jinja chat prompt = **1708** tokens (raw C1 = **1696**). With `-ub 8`: `1708 % 8 = 4` → last ubatch `ne11=4`. Under 066.7 default `MUL_MM_MIN=7`, rem∈{2..7} stays on **mul_mv** while full chunks use **mul_mm** → cross-kernel first-token drift (`summary` vs `резюме`). `ub≥16` rem lands on mm → S16..S1024 identical. Ablation: `MIN=1` → S8≡S16; `MIN=33` (all mv) → also ≡. Raw C1 GREEN because `1696 % 8 = 0` (no rem).

### Fix

Product default `GGML_METAL_MUL_MM_MIN` **7 → 1** (`ne11≥2` → float-tile mm; N=1 decode still mv). Residual suite gate updated (unset or 1).

### Autotest

**C1j** in `test-ubatch-s8-s1024-row0-parity`: apply model jinja chat template (`enable_thinking=false`) then greedy S8 vs S1024. `TASK066_DIAG=c1j` for fast loop. RED@min7 → GREEN@min1 same change set.

### Phase A2 DoD

- [x] Named root cause (last-ubatch rem cross-kernel mv/mm; not jinja CLI bug per se)  
- [x] Autotest C1j RED→GREEN  
- [x] Ladder `023433Z` S8≡S1024  
- [x] Regress C2 / C3-n1280 / C3-n1344 / C1 / C1j + `cb-fusion` / `mul-mat-residual` / `fa` GREEN  

### Anti-patterns (still)

- FA-off / `ctv=f16` as product fix; ε-PASS; reopen `#27441`  
- Treating raw C1 GREEN as ladder Done (need C1j)  
- Assuming MTP widths {2..7} still share mv with N=1 after A2 (they now use mm)  

---

## Phase B — MTP (after Phase A2)

**Unlock:** **Phase A2 DoD** ✅ **and** explicit start.  

1. Autotest RED: verify width N=1 vs `{2,3,4}` FA row0/logits  
2. Port #28108 spirit on Metal (not CUDA paste)  
3. Turbo4 × speculative after FA widths; do not drop turbo  
4. Optional batch-invariant FA pin — YAGNI until RED remains  

---

## Relationship to other 066.*

| Task | Relation |
|------|----------|
| **066.5** | **KEEP** — ISWA `size_swa`; C3+C1 GREEN |
| 066.6 | C2/C7 + C3 via 066.5 |
| 066.7 / 066.8 | KEEP / R&D behind |
| Hist Stage 1 WAIVE | ≠ Done |

## Commands

```bash
cd atomic-llama-cpp-turboquant
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=fa ./tests/run-task066-metal-corners.sh
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  SUITE=ubatch-s8-s1024 ./tests/run-task066-metal-corners.sh
# Expect A1+A2: C2/C3/C1/C1j GREEN
# Confirm: ./tests/run-task066-ubatch-ladder.sh  (jinja; S8≡S1024)
# Diag: TASK066_DIAG=c1j | localize | fa-ablation | c3-bisect | c3-thresh
```
