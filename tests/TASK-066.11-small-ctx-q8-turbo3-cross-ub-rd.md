# TASK-066.11 — R&D: small-CTX logit divergence with q8_0 + turbo3 (cross-ubatch)

**Status:** ⬜ **TODO** (not started / not Done) — investigation only; no fix KEEP until named root + GREEN stamps  
**Depends on:** 066.5 tooling locked (C3-n1536, `run-task066-iswa-kld-ubatch.sh`, ladder); ideas windowed `size_swa`  
**KV lock for this subtask:** `-ctk q8_0 -ctv turbo3` (not turbo4; not layer-bracket `*.8`)  
**FA:** on (product / ideas; refuse FA-off as primary path)

## Why

066.5 host stamps already show **cross-ubatch** breaks at modest context under turbo KV:

- C3 logits RED (n1536 ub8 vs ub1536)
- Ladder S512 vs S1024 generation DIFF
- KLD-UB (turbo4, CTX=4096, ub 256/512/1024): cross-ub KLD ~0.55 vs ref ub1024

Product also cares about **turbo3** (more aggressive V). This subtask isolates **q8_0 + turbo3** at **small CTX** and names the root before any size_swa / Metal KEEP.

## Goal

Explain **why** next-token logits diverge when only `-ub` changes, at small fixed `CTX`, with **uniform** `q8_0` K + `turbo3` V. Deliver a frozen root-cause note + stamp matrix; propose (do not silently ship) the minimal lever.

## Scope locks

1. **CTX small:** primary matrix `CTX ∈ {2048, 4096}` (fail-closed explicit). Optional smoke `3072` to match C3 fixture. No 16K/64K required for this R&D close.
2. **UBATCH_LIST:** `256 512 1024` only (`%128==0`). Refuse non-128 unless `ALLOW_NON128_UB=1` diag.
3. **Cand KV:** `q8_0` / `turbo3` only for PASS of this subtask. turbo4 / f16 are **controls**, not the DoD target.
4. **Metrics (ordered):**
   1. C3-class logits / FIRST_DIFF (reuse `SUITE=ubatch-s8-s1024` + `TASK066_DIAG=c3-bisect` with CTV=turbo3)
   2. `run-task066-iswa-kld-ubatch.sh` with `CAND_LIST=turbo3` (vs-f16 + cross-ub)
   3. Optional short ladder cell DIFF with `TASK066_CTV` unlocked only if runner is extended — else skip ladder if locked to turbo4
5. **No ε-PASS.** Cross-ub KLD want **0** for same cand; C3 want bit-exact. Document RED honestly.
6. **No product YAML change** and no README ahead of a fix (§5.0). Spec/task docs only until a follow-up implementation TASK.
7. **Not Done** while root unnamed or while turbo3 cross-ub still unexplained.

## Hypotheses to prove/refute (one lever at a time)

| ID | Hypothesis | How to test |
|----|------------|-------------|
| H1 | ISWA `size_swa ∝ n_ubatch` capacity asymmetry (same class as C3) | size_swa log lines + C3-n1280 GREEN vs n1536 RED under turbo3 |
| H2 | Turbo3-specific (not general windowed SWA) | Same matrix with `ctv=f16` / `q8_0` / `turbo4` controls |
| H3 | First differing tensor is SWA attn / kqv (not embed) | `c3-bisect` FIRST_DIFF under turbo3 |
| H4 | Small-CTX only (vanishes if CTX << size_swa both sides) | CTX=512/1024 smoke if prompt allows; else note fixture limit |
| H5 | Numerics of turbo3 dequant/WHT vs multi-ub write path | Only after H1–H3; Metal/ops — needs RED corner first |

## Operator entry (fork root)

```bash
# A) KLD-UB turbo3 @ CTX=4096, ub 256/512/1024
MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
BUILD_DIR=build-task066-cb-ws CTX=4096 N_BATCH=2048 FA_MODE=on \
UBATCH_LIST="256 512 1024" REF_UBATCH=1024 CAND_LIST="turbo3" \
  ./tests/run-task066-iswa-kld-ubatch.sh logs/task066-11-kld-ub-turbo3-<UTC>

# B) C3 + bisect with product-like turbo3 (set CTV if runner supports; else CLI env for test binary)
TASK066_MODEL=… TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  ./tests/run-task066-ubatch-s8-s1024-corners.sh logs/task066-11-c3-turbo3-<UTC>
TASK066_DIAG=c3-bisect …   # FIRST_DIFF under turbo3
```

Stamp dir required: `logs/task066-11-*` with argv, CTK/CTV, CTX, UBATCH_LIST, tables, verdict.

## Definition of Done — not met

- [ ] Baseline stamp: KLD-UB `CAND_LIST=turbo3` @ CTX=4096, ub 256/512/1024 (vs-f16 + cross-ub).
- [ ] Control stamps: at least one of f16 / q8_0 / turbo4 cross-ub at same CTX (prove turbo3-specific or shared class).
- [ ] C3-n1280 / C3-n1536 (or FIRST_DIFF) under **q8_0+turbo3** documented.
- [ ] Frozen root note: which of H1–H5 KEEP; named tensor / formula line if H1/H3.
- [ ] Explicit **next fix TASK** proposed (or “no code; accept as known”) — 066.11 itself stays R&D; **not** marked Done as a product fix.
- [ ] Pointer + CHANGELOG note under Unreleased; 066.11 status remains TODO until DoD above checked.

## Non-goals

- Claiming 066.5 Done.
- Shipping size_swa / Metal KEEP inside 066.11 without a dedicated implementation subtask.
- Long-CTX (16K/64K) drift curve (already under 066.5-KLD).
- FP16-SWA fallback; turbo*.8 brackets.
- Weakening bit-exact / KLD==0 gates to ε.
