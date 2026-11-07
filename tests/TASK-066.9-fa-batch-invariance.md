# TASK-066.9 — FA batch-invariance (nomtp-first, MTP second)

**Status:** 🔓 **Phase A localize DONE** → next lever **[066.5](TASK-066.5-fusion-table-parity.md)** (C3-n1344); Phase B MTP still gated  
**Depends on:** [066.7 KEEP](TASK-066.7-mul-mat-residual.md); C2 GREEN (FA always-tile); research below  
**Next code:** 066.5 fuse/CB on C3-n1344 — **not** more FA shape pins; Phase B only after Phase A DoD + explicit start  
**Research:** [task066-hist/draft_mtp_research_task-066_fa.md](task066-hist/draft_mtp_research_task-066_fa.md), [task066-hist/TASK-066-fa-batch-invariance-web-research.md](task066-hist/TASK-066-fa-batch-invariance-web-research.md)  
**Index:** [README-TASK066.md](README-TASK066.md) · [PR audit](TASK-066-pr-audit-registry.md)

## Priority (hard)

```text
Phase A (nomtp)  →  Phase B (MTP verify widths / #28108 spirit)
```

Product MTP e2e **blocked** until Phase A DoD (or explicit unlock in [066.0](TASK-066.0-overview.md)).

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

**Named residual class:** long-prefill **multi-step ubatch** for **`n≥1344`** (lock **`n=1280` GREEN**). Survives **FA-off** → next levers **non-FA** ([066.5](TASK-066.5-fusion-table-parity.md) fusion/CB / FIRST_DIFF_TENSOR / other ops). FA may still add drift when on; **FA-only KEEP forbidden**.

### C. Autotest gates (`SUITE=ubatch-s8-s1024`)

| Gate | Meaning | Status |
|------|---------|--------|
| **C2** | n=32 ub=8 vs ub=32 | **GREEN** (must not regress) |
| **C3** | n=1280 GREEN lock; **n=1344 RED** until fix | **RED** primary TDD (~60s) |
| **C1** | greedy S8 vs S1024 | deferred until C3 GREEN |

### D. Spec / wiring delivered

- This file + [066.0](TASK-066.0-overview.md) + [README-TASK066](README-TASK066.md) + [PR audit](TASK-066-pr-audit-registry.md)
- Draft moved to [task066-hist/draft_mtp_research_task-066_fa.md](task066-hist/draft_mtp_research_task-066_fa.md); stub at `tests/draft_mtp_research_task-066_fa.md`
- Product pointer: `linux_to_mac/tasks/TASK-066-nomtp-ubatch-s8-s1024.md`
- [066.5](TASK-066.5-fusion-table-parity.md) retargeted: C3-n1344 + H9faOFF

---

## Goal (unchanged)

1. **Phase A:** Close nomtp cross-ubatch drift (or name non-FA class — **done**) and drive **C3→C1→ladder** GREEN.  
2. **Phase B:** Metal unify FA geometry for MTP verify widths `{1..4}` (#28108 spirit); then turbo4 × speculative — without dropping turbo.

## Evidence hierarchy (066.0 lock 7)

Every Metal/ops/fusion edit: autotest RED→GREEN same change set; re-run `cb-fusion`, `mul-mat-residual`, `fa`, **C2**, **C3-n1280**. No ε-PASS C3/C1/C2.

---

## Phase A — nomtp (PRIORITY) — remaining work

### Done

- [x] Always-tile + C2 GREEN + C7  
- [x] Localize; **non-FA** class named (H9faOFF)  
- [x] C3 gate wired  

### Open

1. **066.5 / CB:** FIRST_DIFF_TENSOR / fuse-table on n=1344 ub=8 vs one-shot → C3-n1344 RED→GREEN.  
2. Other non-FA ops only with new RED corner.  
3. FA levers **only if** new corner RED under FA-on and GREEN under FA-off.  
4. Confirm: C3 GREEN → C1 → ladder vs `231804Z`.

### Anti-patterns

- FA-off / `ctv=f16` as product fix; FA shape pins without FA-off contrast  
- ε-PASS C3/C1; reopen `#27441`; MTP / `#28108` in Phase A; blind `VLLM_BATCH_INVARIANT`

### Phase A DoD

- [x] C2 GREEN; residual class named (**non-FA** long-prefill multi-step)  
- [ ] C3-n1344 GREEN → C1 GREEN  
- [ ] Ladder confirm vs `231804Z`  
- [ ] Autotest diff per lever  

---

## Phase B — MTP (after Phase A)

**Unlock:** Phase A DoD **and** explicit start.  

1. Autotest RED: verify width N=1 vs `{2,3,4}` FA row0/logits  
2. Port #28108 spirit on Metal (not CUDA paste)  
3. Turbo4 × speculative after FA widths; do not drop turbo  
4. Optional batch-invariant FA pin — YAGNI until RED remains  

---

## Relationship to other 066.*

| Task | Relation |
|------|----------|
| **066.5** | **Next implementation** — C3-n1344 |
| 066.6 | C2/C7 landed; long residual → 066.9/066.5 |
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
# Expect: C2 GREEN, C3-n1280 GREEN, C3-n1344 RED until 066.5 fix
# Diag: TASK066_DIAG=localize | fa-ablation
```
