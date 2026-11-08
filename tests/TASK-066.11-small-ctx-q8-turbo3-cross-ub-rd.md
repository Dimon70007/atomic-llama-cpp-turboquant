# TASK-066.11 — R&D: small-CTX logit divergence with q8_0 + turbo3 (cross-ubatch)

**Status:** 🟨 **root named** (2026-11-08) — H1 KEEP; **not Done** as product fix (no size_swa KEEP shipped here)  
**Depends on:** 066.5 tooling locked (C3-n1536, `run-task066-iswa-kld-ubatch.sh`, ladder); ideas windowed `size_swa`  
**KV lock for this subtask:** `-ctk q8_0 -ctv turbo3` (not turbo4; not layer-bracket `*.8`)  
**FA:** on (product / ideas; refuse FA-off as primary path)  
**Stamp:** `logs/task066-11-localize-20261108T212103Z/`

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

| ID | Hypothesis | Verdict (2026-11-08 stamp) |
|----|------------|----------------------------|
| H1 | ISWA `size_swa ∝ n_ubatch` capacity asymmetry (same class as C3) | **KEEP** — see frozen root |
| H2 | Turbo3-specific (not general windowed SWA) | **REFUTE** — `ctv=f16` also C3-n1536 RED; turbo3/turbo4 same class |
| H3 | First differing tensor is SWA attn / kqv (not embed) | **KEEP** — turbo3/turbo4 `FIRST_DIFF=kqv_out-0` @ n1536; f16 shows `cache_v_l0 (view)` shape split |
| H4 | Small-CTX only (vanishes if CTX << size_swa both sides) | **partial** — logits exact while `n ≤ size_swa(ub_small)` (n1280 @ ub8→1280); breaks when `n > size_swa(ub_small)` |
| H5 | Numerics of turbo3 dequant/WHT vs multi-ub write path | **REFUTE as primary** — equal-`size_swa` ub8 vs ub256 KLD cross-ub **0** @ CTX=4096 |

## Frozen root (KEEP)

```text
size_swa = PAD(min(size_base, n_swa_eff*(unified?n_seq_max:1) + n_ubatch), 256)
```

Changing **only** `-ub` changes SWA cell count:

| ub | size_swa (Gemma4 n_swa=1024) |
|----|------------------------------|
| 8 | 1280 |
| 256 | 1280 |
| 512 | 1536 |
| 1024 | 2048 |
| 1536 | 2560 |

**Decisive control (D):** `UBATCH_LIST="8 256"` (both → 1280), `ALLOW_NON128_UB=1`, CTX=4096, turbo4 → cross-ub **KLD=0 PASS**.  
**Broken matrix (E):** ub 256/512/1024 (1280/1536/2048) turbo3 → cross-ub KLD ≈ 1.03 / 0.98 FAIL (same class as turbo4 ~0.55).

So divergence is **not** “multi-step ubatch stepping” per se; it is **different SWA ring capacity** once the sequence outgrows the smaller cache. Attn path surfaces as `kqv_out-0` (turbo) / asymmetric `cache_v_*` views (f16).

### Proposed next fix TASK (do not ship inside 066.11)

Make `size_swa` **invariant to operator `-ub`** while keeping enough headroom for the largest ubatch that will run, e.g.:

- `size_swa = PAD(n_swa_eff*(…) + n_ubatch_cap, 256)` with `n_ubatch_cap` = product max / `-b` / explicit fail-closed knob — **not** the live `-ub`, **or**
- fixed headroom constant ≥ product max ub (512/1024/2048).

Out of scope here: size_base KEEP (RAM), FP16-SWA fallback. Next R&D → [`TASK-066.13-size-swa-ub-invariant-rd.md`](TASK-066.13-size-swa-ub-invariant-rd.md) (incl. **N_CTX=100000**); implementation KEEP only after 066.13.

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

## Definition of Done

- [x] Baseline stamp: KLD-UB `CAND_LIST=turbo3` @ CTX=4096, ub 256/512/1024 — FAIL (cross-ub ≈1.03/0.98).
- [x] Controls: f16 C3-bisect RED @ n1536 (shared class); equal-`size_swa` ub8↔ub256 turbo4 KLD **PASS**; prior turbo4 KLD-UB FAIL.
- [x] C3-n1280 GREEN / C3-n1536 RED under **q8_0+turbo3**; `FIRST_DIFF=kqv_out-0`.
- [x] Frozen root: **H1+H3 KEEP**; H2/H5 refuted as primary; formula line + table above.
- [x] Next fix proposed: decouple `size_swa` from live `-ub` → **066.13** R&D (+ 100k verify) — **no code in 066.11**.
- [ ] Pointer + CHANGELOG note; mark 066.11 R&D **closed (root named)** — still not a product fix Done for 066.5 gates.

## Non-goals

- Claiming 066.5 Done.
- Shipping size_swa / Metal KEEP inside 066.11 without a dedicated implementation subtask.
- Long-CTX (16K/64K) drift curve (already under 066.5-KLD).
- FP16-SWA fallback; turbo*.8 brackets.
- Weakening bit-exact / KLD==0 gates to ε.
