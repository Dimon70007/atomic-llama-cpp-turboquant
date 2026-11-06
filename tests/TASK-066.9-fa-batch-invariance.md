# TASK-066.9 — FA batch-invariance (nomtp-first, MTP second)

**Status:** 🔓 **active — Phase A (nomtp)**  
**Depends on:** [066.7 KEEP](TASK-066.7-mul-mat-residual.md); C2 GREEN (FA always-tile evidence in [066.6](TASK-066.6-cb-and-invariance-gates.md) + [PR audit](TASK-066-pr-audit-registry.md)); research below  
**Next:** Phase B (MTP) only after Phase A DoD **and** explicit start  
**Research:** [task066-hist/draft_mtp_research_task-066_fa.md](task066-hist/draft_mtp_research_task-066_fa.md), [task066-hist/TASK-066-fa-batch-invariance-web-research.md](task066-hist/TASK-066-fa-batch-invariance-web-research.md)  
**Does not replace:** [066.5](TASK-066.5-fusion-table-parity.md) fusion table — fuse work only if C1 localize points at CB/fuse, not FA

## Priority (hard)

```text
Phase A (nomtp C1 / ladder)  →  Phase B (MTP verify widths / #28108 spirit)
```

Product MTP e2e stays **blocked** until Phase A DoD (or explicit unlock in [066.0](TASK-066.0-overview.md)). Do not mix Phase A and Phase B evidence in one change set.

## Frozen context (entry)

| Item | State |
|------|--------|
| FA always-tile | Default tile; legacy vec only `GGML_METAL_FA_VEC=1`; dispatch log `GGML_METAL_FA_DISPATCH=1` |
| C2 (`n=32` ub=8 vs ub=32) | **GREEN** bit-exact |
| FA C7 multi-step (`SUITE=fa`) | **GREEN** (legacy vec RED) |
| A0–A4 on C2 after tile | all **exact=1** (product turbo4 OK on C2) |
| C1 greedy S8 vs S1024 | **RED**; HELP `first_token_diff` 0→8 |
| Ladder | `231804Z` S1024 vs S8 DIFF@329 (HELP vs `212703Z`@182) |
| Product path | FA-on + `ctv=turbo4` — never “fix” by FA-off / f16 |

## Goal

1. **Phase A:** Close remaining **nomtp** cross-ubatch drift attributable to FlashAttention (or name a non-FA residual class) so C1 / ladder match or HELP is documented outside FA.
2. **Phase B:** Unify Metal FA kernel geometry for MTP verify widths `{1..4}` (#28108 spirit from draft); then re-check turbo4 × speculative — without dropping turbo on product MTP profiles.

## Evidence hierarchy (066.0 lock 7)

Every FA / Metal / ops edit:

1. New or extended corner **RED** on pre-fix HEAD for *this* gap.
2. Minimal fix → same corner **GREEN**.
3. Same change set: re-run `SUITE=cb-fusion`, `mul-mat-residual`, `fa`, and ubatch **C2** (must stay GREEN).
4. Log HELP / NO_EFFECT / REGRESS. No ε-PASS on C1 / C2 / residual.

---

## Phase A — nomtp (PRIORITY)

### Primary gates (order)

1. **Regression lock:** C2 stays GREEN — `SUITE=ubatch-s8-s1024` (C2) + `SUITE=fa` (incl. C7).
2. **Localize C1** beyond always-tile: FIRST_DIFF / op-class on greedy S8 vs S1024; optionally extend FA-op multi-step (C7-style) to longer `n_total` if that corner is RED.
3. **One FA lever per change set** — only if evidence supports (do not shotgun):
   - Remaining shape-dependent FA knobs (tile `nsg`, Metal packing / Vulkan [#26358](https://github.com/ggml-org/llama.cpp/pull/26358) analogue).
   - Fixed split-KV / consistent KV layout before attn (Thinking Machines / vLLM spirit — Metal-local; not a silent product env that drops turbo).
   - Metal [#29340](https://github.com/ggml-org/llama.cpp/pull/29340)-class threadgroup issues **only if** reproduced RED on this fork.
4. **Confirm:** C1 bit-exact **or** overview names residual as **non-FA**; ladder stamp vs `231804Z` (HELP/MATCH/REGRESS).

### Phase A anti-patterns

- FA-off or `ctv=f16` as product “fix”.
- ε / ULP-PASS on C1.
- Reopen `#27441` without new RED hypothesis.
- MTP e2e, ACC_TRACE product stamps, or `#28108` port **during** Phase A.
- Blind full port of `VLLM_BATCH_INVARIANT`.

### Phase A DoD

- [ ] C2 remains GREEN after every FA edit.
- [ ] C1 GREEN **or** residual class explicitly named outside FA in 066.0 / this file.
- [ ] Ladder confirm logged vs `231804Z`.
- [ ] Autotest diff(s) for each lever in the same change set as the fix.

---

## Phase B — MTP (after Phase A)

### Unlock

Phase A DoD met **and** operator / overview says start MTP. Until then: MTP blocked.

### Scope

1. **Autotest RED first:** MTP verify batch width — FA row0 / logits class for `N=1` vs `N∈{2,3,4}` (fork corner; no product e2e as primary gate).
2. **Port [#28108](https://github.com/ggml-org/llama.cpp/pull/28108) spirit on Metal:** same FA kernel geometry for verify widths in `{1..4}` (draft). Not a CUDA Blackwell / SM120 paste.
3. **Turbo4 × speculative:** isolate **after** FA widths unified; product MTP profile stays turbo4 — fix path, do not drop turbo.
4. **Optional** fork “batch-invariant FA pin” (explicit selector, fail-closed) — YAGNI until RED remains after (2)–(3). Inspired by vLLM’s documented mode; not a copy of their env surface.

### Phase B gates

- New/extended MTP FA autotest RED→GREEN same change set.
- Regression pack + C2 still GREEN.
- Then product first-diff / `LLAMA_MTP_ACC_TRACE` confirm — never WAIVE-as-Done.

### Phase B DoD

- [ ] MTP FA width autotest GREEN.
- [ ] Documented Metal dispatch rule for widths `{1..4}` (and intentional deltas).
- [ ] Turbo4 re-check after FA unify; HELP/NO_EFFECT/REGRESS logged.
- [ ] No product README fiction ahead of code (§5.0).

---

## Relationship to other 066.* tasks

| Task | Relation |
|------|----------|
| 066.5 fusion | Parallel only if C1 localize → CB/fuse, not FA |
| 066.6 CB + invariance | C2/C7 landed; C1 open → owned here as Phase A |
| 066.7 / 066.8 | KEEP / R&D behind; orthogonal mm gates stay GREEN |
| Hist Stage 1 WAIVE | ≠ Done for this track |

## Out of scope

- CUDA cluster FA work.
- Disabling multi-CB as a “fix”.
- Treating hist waive or FA-off diag GREEN as product Done.

## Commands (Phase A loop)

```bash
cd atomic-llama-cpp-turboquant
BUILD_DIR=build-task066-cb-ws SUITE=cb-fusion ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
BUILD_DIR=build-task066-cb-ws SUITE=fa ./tests/run-task066-metal-corners.sh
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  SUITE=ubatch-s8-s1024 ./tests/run-task066-metal-corners.sh
# C2 must stay GREEN; C1 is the Phase A TDD gate until GREEN or non-FA named
```
