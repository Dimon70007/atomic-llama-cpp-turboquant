# TASK-066.5 — ISWA size_swa / long-ctx TurboQuant drift

**Status:** ⬜ **TODO** (not done) — ideas pad128 landed; **T-066.5-C3 RED** stamped; **T-066.5-KLD FAIL** (full 4K/16K/64K; turbo4 4K->16K walks up); DoD open on C3/product  

**Depends on:** [066.7 KEEP](TASK-066.7-mul-mat-residual.md); FA always-tile C2 GREEN; [066.9](TASK-066.9-fa-batch-invariance.md) Phase A localize  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md) (Ideas recommendation)  
**Ideas note:** [`ideas/llamacpp_swa_size_calculate.md`](../../../ideas/llamacpp_swa_size_calculate.md)  
**Sizing SSOT:** [TASK-066.10-kv-ctx-sizing.md](TASK-066.10-kv-ctx-sizing.md)

**Current lever:** ISWA `size_swa` = windowed + TurboQuant KV block align (pad128)  
**Prior KEEP (superseded for RAM):** `size_base` headroom made `size_swa == n_ctx` when `ctx >= n_swa` (C3 GREEN, long-ctx RAM side effect)  
**Source hypothesis (superseded):** fuse/`fusion_max` — bisect showed attn/`kqv_out`; root was SWA capacity

### Problem (ideas)

TurboQuant KV (`turbo3` / `turbo4`) uses **blocks of 128**. If metadata `n_swa` is not a multiple of 128, quantized blocks misalign with SWA rotation → error accumulates → **drift on long contexts**.  
Weight TQ3_1S/TQ4_1S (block 32) is a different path — do not conflate.

Uniform CTV only on this track: `f16`, `q8_0`, `turbo3`, `turbo4`.

### Root cause (evidence, historical — C3 batch-invariance)

Gemma4 `n_swa=1024`. Pre-KEEP ISWA size:

```text
size_swa = PAD(min(ctx, n_swa + n_ubatch), 256)
  ub=8     -> PAD(1032,256)=1280
  ub=1344  -> PAD(2368,256)=2560
```

Multi-ubatch contexts slide the SWA cache after **1280** tokens; one-shot / large-ub contexts still hold a longer window -> logits diverge. Matches:

| Probe | Result |
|-------|--------|
| `c3-thresh` | n=1280 GREEN; **n=1288 first RED** |
| `c3-bisect` n=1280 | logits exact; `FIRST_DIFF=<none>` |
| `c3-bisect` n=1344 | `FIRST_DIFF=kqv_out-0` (FA); FA-off -> `kq-0` |
| H9faOFF | RED -> not FA-only (SWA storage class) |

### Prior KEEP (2026-11-07, superseded for RAM)

[`src/llama-kv-cache-iswa.cpp`](../src/llama-kv-cache-iswa.cpp): drop `n_ubatch`; headroom = `size_base` => `size_swa == size_base` when `ctx >= n_swa`. C3+C1 GREEN, but long product `n_ctx` pays full-length SWA cells.

### Preferred fix (ideas — adopted in runtime)

```text
TURBO_KV_BLOCK_SIZE = 128
n_swa_eff = ceil(n_swa / 128) * 128
size_swa  = PAD(min(size_base, n_swa_eff * (unified ? n_seq_max : 1) + n_ubatch), 256)
```

Helper: `llama_kv_cache_iswa_calc_size_swa` + CPU unit `test-iswa-size-swa`.  
Gemma4 (`n_swa=1024`): pad is no-op; `ub=8` -> 1280; `ub=512` -> 1536; `ub=1344` -> 2560.  
**Note:** capacity depends on `-ub` again — C3 cross-ub must be re-stamped (no silent waive).

### Fallback (ideas alt — OUT OF SCOPE)

Ideas mentions forcing FP16 on SWA layers when `n_swa % 128 != 0`. **We will not use this path.** Only pad128 + windowed `size_swa` is in scope.

### Test specification — long-ctx drift gate (from ideas §verify)

Product pointer (same text): [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md) § Test specification.  
Ideas source: [`ideas/llamacpp_swa_size_calculate.md`](../../../ideas/llamacpp_swa_size_calculate.md) (KLD vs FP16; multi-ctx curve).

#### T-066.5-KLD — purpose

Prove that after ideas `size_swa`, TurboQuant KV **does not accumulate** long-context drift relative to an FP16 KV reference. **PPL alone is not a PASS gate** (Gemma-class PPL can mislead).

#### T-066.5-KLD — metric

| Item | Requirement |
|------|-------------|
| Metric | **KL divergence (KLD)** between next-token logit distributions |
| Reference | `-ctk f16 -ctv f16` (same model, same prompts, same sampling/decode knobs) |
| Candidate A | `-ctk q8_0 -ctv turbo3` under ideas `size_swa` |
| Candidate B | `-ctk q8_0 -ctv turbo4` under ideas `size_swa` |
| Aggregation | Per-context-length scalar KLD (document exact reduction: mean over positions / tokens); stamp raw series |

#### T-066.5-KLD — matrix (fail-closed; no silent defaults)

| Knob | Required values | Notes |
|------|-----------------|-------|
| `MODEL` | Explicit GGUF path (Gemma4-26B-A4B product scratch) | No silent model substitute |
| `CTX` | **4096, 16384, 65536** (4K / 16K / 64K) | All three required per candidate |
| `CTK`/`CTV` | ref `f16/f16`; cand `q8_0/turbo3`; cand `q8_0/turbo4` | Uniform only |
| `N_UBATCH` | Explicit (product default **512** unless stamp says otherwise) | Affects `size_swa` |
| `N_BATCH` | Explicit (product gemma4 **2048**) | Long-ctx default |
| `BATCH_EQ_CTX` | `0` (default) or `1` | `-b=CTX` **only** if every CTX **<= 2000**; refuse above |
| `FA` / `-fa` | **`on`** (fail-closed; required for CTX >= 8000) | ideas: FA-off = V-transpose + NxN attn RAM; product `flash_attn: on` |
| Prompts / decode | Identical across ref and cand at each CTX | Fixed seed / greedy or documented policy |
| Tooling | `tests/run-task066-iswa-kld.sh` (+ `llama-perplexity --kl-divergence`) | Fail-closed `MODEL`/`N_UBATCH`/`N_BATCH`/`FA_MODE` |

#### T-066.5-KLD — pass / fail

```text
PASS: for each candidate (turbo3, turbo4), the series
      KLD(4096), KLD(16384), KLD(65536)
      does NOT "walk up" with context
      (no monotonic increase; document tolerance / flat-or-down rule in stamp).

FAIL: KLD rises with CTX (drift still present), or any CTX/candidate missing,
      or PPL-only evidence offered without KLD series.
```

Stamp artifact (required): `logs/task066-iswa-kld-<UTC>/` with per-CTX KLD table, argv, model path, `size_swa` / `-ub`, and PASS/FAIL verdict. No ε-PASS.

#### T-066.5-C3 — companion (batch-invariance; not a KLD substitute)

Re-stamp `SUITE=ubatch-s8-s1024` C3 after ideas formula (capacity depends on `-ub` again). Document GREEN or RED; no silent waive. **C3 GREEN alone does not close long-ctx drift** — **T-066.5-KLD** is still required.

Diag:

```bash
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  TASK066_DIAG=c3-bisect ./tests/run-task066-ubatch-s8-s1024-corners.sh
TASK066_DIAG=c3-thresh …   # pad/SWA threshold matrix
```

#### T-066.5 — non-goals for this test spec

- Using perplexity as the sole acceptance metric.
- Implementing / testing FP16-SWA-layer fallback.
- Layer-bracket CTV (`*.8`) on this track.
- Claiming drift fixed from a single CTX point.

### TDD / stamps (summary)

1. CPU unit: `test-iswa-size-swa` (pad128 + ubatch cases) — **PASS** Mac `build-task066-cb-ws` 2026-11-08.
2. **T-066.5-C3:** `SUITE=ubatch-s8-s1024` — C2 + C3-n1280 + **C3-n1344** + C1 after formula change.
3. **T-066.5-KLD:** FP16 ref vs `q8_0/turbo3` and `q8_0/turbo4` @ 4096 / 16384 / 65536; stamp under `logs/task066-iswa-kld-<UTC>/`.
4. Historical KEEP stamps remain evidence of the size_base path (not current runtime).

### Host findings (2026-11-08) — ideas formula stamps

**T-066.5-C3 — RED (documented; no waive)**  
Stamp: `logs/task066-metal-corners-ubatch-s8-s1024-20261108T185227Z/` (+ `T-066.5-C3-VERDICT.txt`).  
Bisect: `logs/task066-c3-bisect-20261108T185548Z/`.

| Probe | Result |
|-------|--------|
| C2 n=32 | GREEN (exact) |
| C3-n1280 | GREEN (exact) |
| C3-n1344 | **RED** (`mad≈2.35`, `first_diff_i=0`) |
| Bisect n1280 | `FIRST_DIFF=<none>` |
| Bisect n1344 | `FIRST_DIFF_TENSOR_SUMMARY=kqv_out-0` |
| C1 | skipped (C3 RED) |

Class: windowed `+n_ubatch` restores cross-ub capacity asymmetry (ub8 → `size_swa=1280`; ub1344 → `2560`). Same class as pre-KEEP root; size_base KEEP had made C3 GREEN at RAM cost.

**T-066.5-KLD — FAIL (full curve; honest)**  
Merged stamp: `logs/task066-iswa-kld-20261108T194452Z-merged/` (4K/16K from `…190211Z` N_BATCH=512; 64K from `…194452Z-c64k` N_BATCH=2048). Runner: `tests/run-task066-iswa-kld.sh`.  
Argv lock: **`-fa on`**; explicit `N_UBATCH` / `N_BATCH`; **`BATCH_EQ_CTX=1` only for CTX <= 2000**.  
Stream fix: `tools/perplexity/perplexity.cpp` processes logit rows per ubatch (no `n_ctx*n_vocab` f32 reserve; `log_probs` sized to `n_batch`).

| cand | KLD@4096 | KLD@16384 | KLD@65536 |
|------|----------|-----------|-----------|
| turbo3 | 2.334418 | 2.284085 (down) | 2.192946 (down) |
| turbo4 | 1.009278 | 1.284858 (up) | 1.282610 (flat) |

- f16 vs f16 self-KLD = 0 (stream smoke). Absolute PPL on IT IQ4+wikitext — ignore PPL.
- 64K HOST_OOM cleared by stream path (base logits file ~16GiB on disk; host RSS no longer ~66GiB).
- Honest FAIL: turbo4 **4K->16K walks up**; 16K->64K does not continue. turbo3 monotone down. C3 remains RED.

Product pointer Findings mirror: [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md).

## Goal (original fuse-table — deferred)

`ggml_metal_fusion_max` vs encoder fuse parity remains a valid hygiene item if a new RED corner names it. **Not** the ISWA / long-ctx drift lever.

## Evidence hierarchy (066.0 lock 7)

1. Prior KEEP: C3 RED on pre-fix HEAD -> ISWA size_base edit -> C3/C1 GREEN.
2. Ideas path: windowed + pad128 for long-ctx RAM + TurboQuant block align.
3. Long-ctx quality: **T-066.5-KLD** (PPL alone insufficient).
4. No ε-PASS; product stays FA-on + turbo4; no FP16-SWA fallback.

## DoD (ideas reopen)

- [x] Prior KEEP: autotest RED->GREEN on C3-n1344 (historical only; superseded).
- [x] Lever documented for ideas path (ISWA `size_swa` / pad128 + T-066.5-KLD; not FA / not fuse-table).
- [x] Ideas formula landed in `llama_kv_cache_iswa_calc_size_swa` (draft; C3/KLD gates open).
- [x] CPU unit `test-iswa-size-swa` locks pad128 + ubatch cases (**PASS** Mac 2026-11-08).
- [x] **T-066.5-C3:** C3 re-stamp after ideas formula — **RED documented** (C1 skipped; no waive) `20261108T185227Z`.
- [x] **T-066.5-KLD:** full 4K/16K/64K curve stamped — **FAIL honest** (`…194452Z-merged`; turbo4 4K->16K up; stream path unblocked 64K).
- [x] FP16-SWA fallback explicitly left unimplemented (out of scope).
- [x] Handed to 066.9 Phase A2 historically — **DONE** (`MUL_MM_MIN` default 1; ladder `023433Z`).

## Checklist (ideas reopen)

- [x] FIRST_DIFF / thresh before Metal/ops guess (historical + 2026-11-08 bisect).
- [x] One non-FA lever confirmed under ideas formula (ISWA capacity / block align; C3 RED class).
- [x] No blind encoder fuse ports.
- [x] Ladder confirm stamped (HELP) under prior KEEP (historical).
- [x] **T-066.5-C3** stamp under ideas formula (**RED**).
- [ ] **T-066.5-KLD** stamp complete under ideas formula (partial FAIL; 64K open).
