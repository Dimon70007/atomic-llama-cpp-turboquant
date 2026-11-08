# TASK-066.13 — R&D: `size_swa` independent of live `-ub`

**Status:** 🟨 **C2 KEEP named** (2026-11-08) — hypothesis **PROVED** for fixed `-b`; runtime C2 landed in fork; **not** full 066.5 Done (C1/ladder still RED when `-b` also changes)  
**Depends on:** 066.11 root named (`size_swa ∝ n_ubatch`); 066.5 gates; 066.10 estimator @ N_CTX=100000  
**Stamp:** `logs/task066-13-c2-nbatch-headroom-20261108T220220Z/`  
**FA:** on  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)

## Why

066.11: cross-ub breaks track different SWA cell counts from

```text
size_swa = PAD(... + n_ubatch)   # live -ub
```

## KEEP candidate — **C2** (landed)

```text
size_swa = PAD(min(size_base, n_swa_eff*(unified?n_seq_max:1) + n_batch), 256)
```

- Headroom = **`n_batch` (`-b`)**, not live `-ub`.
- Since `n_ubatch <= n_batch` always in llama.cpp, capacity for writes is preserved.
- When operator sweeps **only** `-ub` with fixed `-b`, `size_swa` is invariant.

| ID | Verdict |
|----|---------|
| C1 fixed cap | deferred (would also fix C1/ladder when `-b` varies) |
| **C2 `+ n_batch`** | **KEEP** — evidence below |
| C3 no headroom | not tested (risk write capacity) |
| C4 `max(ub,cap)` | **REJECT** (still moves with live `-ub`) |

## Evidence (stamp `…220220Z`)

| Probe | Result |
|-------|--------|
| **N_CTX=100000 estimate** `N_BATCH=2048`, ub 256/512/1024 | **PASS** — identical `size_swa=3072`, KV_MiB=1008.0273 (windowed; was 889.67 @ ub512/headroom512) |
| C3-n1280 / **C3-n1536** (same `n_batch=n` both sides) | **GREEN** `exact=1` (was RED mad≈7.63) |
| KLD-UB turbo4 CTX=4096, ub 256/512/1024, `N_BATCH=2048` | **PASS** cross-ub KLD=0 |
| KLD-UB turbo4 CTX=16384 (long companion) | **PASS** cross-ub KLD=0 |
| KLD-UB CTX=100000 | **ABORTED** — ~19 min + ~26 GiB logits per FP16 base pass; impractical; estimate invariance satisfies 100k size_swa DoD |
| Suite C1 greedy S8 vs S1024 (`b=512 ub=8` vs `b=1024 ub=1024`) | **still RED** — expected under C2: `-b` differs → `size_swa` 1536 vs 2048 |
| **CTX=65536 smoke** `…ctx64k-smoke-20261108T225702Z` | corners: C3-n1536 **GREEN**, C1 **RED** (same residual); MTP↔nomtp ub=512 product **PASS** (engagement OK; DIFF telemetry) |

## Residual / next implementation

C2 closes the **ub-only** invariance class (066.11 root). Remaining product C1/ladder DIFF when the harness also changes `-b` with `-ub`.

**Next:** [`TASK-066.14-c1-s8-s1024-localize-rd.md`](TASK-066.14-c1-s8-s1024-localize-rd.md) — localize C1 RED first (H1 equal-`-b` control), then pick KEEP:

1. Fixed `n_swa_headroom` cap ≥ product max `-b`.
2. Lock C1/ladder harness to **same `-b`** across ub cells.
3. Accept C2 + document that changing `-b` reallocates SWA (only if H1 KEEP and product agrees).

## Code / tooling landed with this R&D

- `llama_kv_cache_iswa_calc_size_swa(..., n_swa_headroom)`; callers pass `cparams.n_batch`.
- Estimator: required `N_BATCH` (headroom); `N_UBATCH` telemetry only.
- Units: `test-iswa-size-swa`, `test_estimate_kv_ctx` (+ invariance case).

## Definition of Done

- [x] Candidates evaluated; **C2 KEEP**.
- [x] Unit / estimator updated for ub-invariant (via n_batch) formula.
- [x] Small-CTX: C3-n1536 GREEN; KLD-UB 256/512/1024 PASS (fixed `-b`).
- [x] **Large-CTX 100000:** size_swa invariance stamped; full KLD-UB@100k aborted (time/disk) + 16k companion PASS.
- [x] RAM note @ 100k: windowed `size_swa=3072` (~1008 MiB turbo4) vs prior ub512/1536 baseline.
- [x] Next implementation options listed (C1 cap / lock `-b` / accept C2 contract).
- [ ] Pointer + CHANGELOG; 066.5 remains open until C1/ladder policy chosen + GREEN.

## Non-goals

- Claiming 066.5 Done while C1 greedy / ladder still RED.
- FP16-SWA fallback; turbo*.8 brackets.
- ε-PASS.
