# TASK-066.17 — Product ubatch parity gates (256↔512, 256↔1024)

**Status:** 🔴 **OPEN / RED** (2026-11-09) — gates implemented; root class **locked** (below); fix until GREEN  
**Policy:** red tests are **never** KEEP / waive / “accepted tax”.  
**Depends on:** 066.5/066.6 product gate binary `test-ubatch-s8-s1024-row0-parity`; 066.14 size_swa pivot; tip `MUL_MM_MIN_DEFAULT=8` (upstream-like)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)  
**Binary / runner:** `tests/test-ubatch-s8-s1024-row0-parity.cpp` + `tests/run-task066-ubatch-s8-s1024-corners.sh`  
**Umbrella:** `SUITE=ubatch-s8-s1024` (no new suite name)  
**Stamps:** product [`…173359Z`](../logs/task066-17-product-ub-20261109T173359Z/) · q8 [`…174850Z`](../logs/task066-17-c1p-q8-20261109T174850Z/) · f16 [`…175136Z`](../logs/task066-17-c1p-f16-20261109T175136Z/)

---

## Locked conclusion (2026-11-09) — do not re-litigate without new stamps

**Finding (locked):** operator-pair prefill **256↔512** under equal `-b=512` is **not bit-exact**. Cause is **not** KV quant quality.

| Isolate | KV | Stamp | mad (256↔512) | argmax | bit-exact |
|---------|-----|-------|---------------|--------|-----------|
| C2p (product) | q8_0 / turbo4 | `…173359Z` | ≈7.635 | **differ** | RED |
| C2pq8 | q8_0 / q8_0 | `…174850Z` | ≈3.878 | same | RED |
| C2pf16 | f16 / f16 | `…175136Z` | ≈0.255 | same | RED |

**Excluded:** “turbo4 (or q8) alone explains the split” — mad shrinks with better KV, bit-exact stays RED even at **f16/f16**.

**Root class (locked for fix):** structural — under equal outer `-b`, different `-ub` still changes run physics via tip  
`headroom = PAD128(max(n_ubatch, n_batch/4))` → **`size_swa ∝ ub`**. Stress C2/C3 can stay GREEN; product cross-ub does not.

**Required fix direction:** restore bit-exact on C2pf16 → C2pq8 → C2p (then C1p/C1jp) by making SWA sizing **invariant to differing ub under equal `-b`** (or equivalent invariant). **No soft-ε, no KEEP-with-RED, no “accept the tax”.**

**Not Done until:** C2pf16 + C2pq8 + C2p + C1p + C1jp all GREEN on a new stamp.

---

## Why

Product Darwin/Linux profiles use real ubatch **256 / 512 / 1024** (e.g. minillm `ubatch: "512"`, some `256`/`1024`).  
Stress pair **16↔1024** (post-`ub≥9` contract; historical A-side was 8) catches rem/size_swa edge classes but is **not** the operator path.  
Need hard gates on the ubatch values operators actually run.

Prior evidence: triple/KLD/ladder under 066.11–066.14 showed cross-ub divergence when `size_swa∝ub`. 066.17 **locks** product-pair bit-exact as a hard gate. A RED stamp is a **bug / open task**, not a product KEEP.

---

## Design (approved 2026-11-09)

### Pairs (operator)

| Gate id | Pair | Notes |
|---------|------|--------|
| **C2p** | logits: `ub=256` vs `ub=512`; `ub=256` vs `ub=1024` | Prefill exact; equal `-b` per pair |
| **C1p** | greedy raw: same two pairs | Token-stream bit-exact |
| **C1jp** | greedy jinja: same two pairs | After C1p GREEN (A2 class) |

**Not in scope for 066.17:** pair `512↔1024` (explicitly excluded).

### Batch / ubatch contract

- Every product ubatch: **`ub ≥ 9`** and **`ub % 8 == 0`** (256/512/1024 OK). Fail-closed in gate helpers.
- Per pair: **equal `-b` both sides**, `n_batch = max(ub_a, ub_b)` (fail-closed if `n_batch < max(ub)`).  
  Rationale: unequal `-b` confounds `size_swa` via `PAD128(max(ub, b/4))` (066.14/066.16); product pairs must isolate ubatch, not outer batch.
- Stress gates **C2 / C3 / C1 / C1j** (16↔32, 16↔n, 16↔1024) **remain**; C*p* are **additional**, not replacements.

### Order in full suite

```text
Preferred: C2 → C3 → C2pf16(f16/f16) → C2pq8(q8/q8) → C2p(turbo4) → C1p → C1jp → C1(stress) → C1j(stress)
```

First RED stops the suite (no waive).

### Fast loop

- `TASK066_DIAG=c1p` — C2p + C1p only (product KV `q8_0`/`turbo4`).  
- `TASK066_DIAG=c1p-q8` — **C2pq8 + C1pq8** under `ctk=q8_0`/`ctv=q8_0`.  
- `TASK066_DIAG=c1p-f16` — **C2pf16 + C1pf16** under `ctk=f16`/`ctv=f16`.  
- Fail-closed if `TASK066_CTK`/`CTV` already set to another type.  
- `TASK066_DIAG=c1jp` — C1jp only.  
- Unset diag = full suite: after C3 runs **C2pf16 → C2pq8 → C2p**, then C1p/C1jp/stress.

### Fixture / env

- Same product lock as today: MODEL + CTX required; duel prompt fixture; `n_predict` via `TASK066_N_PREDICT` (default 64).  
- KV/FA: product `q8_0` / `turbo4` / FA-on (existing `product_lock`).

### Out of scope

- Changing product YAML ubatch defaults.  
- Waiving C1 stress RED or residual-vs-N=8 under `MUL_MM_MIN=8`.  
- Reviving deleted `run-task066-ubatch-ladder.sh` as primary path (§5.8: extend existing binary/runner).  
- KLD-UB / long-ctx 066.5 closeout.

---

## Definition of Done

1. [x] C2p + C1p + C1jp + C2pq8/C2pf16 isolates implemented; fail-closed ub + KV locks.  
2. [x] `TASK066_DIAG=c1p` / `c1p-q8` / `c1p-f16` / `c1jp` wired; runner + CHANGELOG + README-TASK066.  
3. [x] RED evidence + **locked conclusion** recorded (above).  
4. [ ] **C2pf16 + C2pq8 + C2p + C1p + C1jp all GREEN** — no KEEP / waive / soft-ε.  
5. [ ] This file + parent pointer → Done only after that GREEN stamp.

---

## Stamp / current RED (2026-11-09) — task OPEN

**Stamp:** `logs/task066-17-product-ub-20261109T173359Z`  
**Build:** `build-task066-cb-ws` / tip dirty + `MUL_MM_MIN_DEFAULT=8`  
**Env:** `TASK066_MODEL=…/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` `TASK066_CTX=3072` (diag unset = full suite)

| Gate | Result | Notes |
|------|--------|--------|
| C2 | GREEN | ub16 vs ub32 n=32 |
| C3 | GREEN | n=1280 + n=1536 |
| **C2p 256↔512** | **RED** | `exact=0` mad≈7.635 `first_diff_i=0` argmax `9494` vs `1314`; n_prefix=1536, equal `-b=512` |
| C2p 256↔1024 | not run | suite stops on first RED |
| C1p / C1jp | not run | after C2p |
| C1 / C1j stress | not run | after C*p* |

See **Locked conclusion** above (stamps + mad ladder). SWA ring-full warnings at n=1536 on all three isolates (same class as C3 path).

---

## Non-goals / anti-patterns

- Silent fallback to other ubatch if MODEL/CTX missing.  
- Soft ε on logits/tokens — bit-exact only.  
- Replacing 16↔1024 stress with product pairs.
