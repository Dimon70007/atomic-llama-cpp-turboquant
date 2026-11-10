# TASK-066.17 — Product ubatch parity gates (256↔512, 256↔1024)

**Status:** 🟢 **DONE** (2026-11-10) — product C2pf16/C2pq8/C2p/C1p/C1jp **GREEN** after `headroom=PAD128(n_batch)`; full-suite stress C1 16↔1024 still RED (066.5/066.6, out of 066.17 DoD)  
**Policy:** red tests are **never** KEEP / waive / “accepted tax”.  
**Depends on:** 066.5/066.6 product gate binary `test-ubatch-s8-s1024-row0-parity`; 066.18 Mode B; tip `MUL_MM_MIN_DEFAULT=8`  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)  
**Binary / runner:** `tests/test-ubatch-s8-s1024-row0-parity.cpp` + `tests/run-task066-ubatch-s8-s1024-corners.sh`  
**Umbrella:** `SUITE=ubatch-s8-s1024` (no new suite name)  
**Merged operator log:** [`logs/run-06617-after-nbatch-headroom.log`](../logs/run-06617-after-nbatch-headroom.log)  
**GREEN stamps:** f16 [`…215457Z-c1p-f16`](../logs/task066-17-product-ub-20261109T215457Z-c1p-f16/) · q8 [`…225117Z-c1p-q8`](../logs/task066-17-product-ub-20261109T225117Z-c1p-q8/) · full product path [`…225212Z`](../logs/task066-17-product-ub-20261109T225212Z/) (RED only at stress C1)  
**Pre-fix RED:** [`…173359Z`](../logs/task066-17-product-ub-20261109T173359Z/) / [`…174850Z`](../logs/task066-17-c1p-q8-20261109T174850Z/) / [`…175136Z`](../logs/task066-17-c1p-f16-20261109T175136Z/)

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

### Fix landed in tip (2026-11-10) — restore 066.13 KEEP

```text
headroom = PAD128(n_batch)     # was PAD128(max(n_ubatch, n_batch/4)) — ∝ub
size_swa = PAD256(n_swa_eff + headroom)
```

- Code: `llama_kv_cache_iswa_calc_swa_headroom` in `src/llama-kv-cache-iswa.cpp`.  
- Unit: `test-iswa-size-swa` expects equal headroom for `(b=512,ub=256)` and `(b=512,ub=512)`.  
- Memory: product `b=2048 ub=512` headroom **512→2048** (same tax 066.13 accepted for C2 KEEP).  
- 066.18 Mode B: enabling tip `max(ub,b/4)` alone left ubatch mad≈4.70 — confirms need ⊥ub policy, not that lever.

### GPU evidence (2026-11-09/10) — after `PAD128(n_batch)` — **066.17 DoD met**

| Gate | Stamp | Result |
|------|-------|--------|
| **C2pf16** + **C1pf16** | [`…215457Z-c1p-f16`](../logs/task066-17-product-ub-20261109T215457Z-c1p-f16/) | **GREEN** `mad=0` / `first_token_diff=-1` |
| **C2pq8** + **C1pq8** | [`…225117Z-c1p-q8`](../logs/task066-17-product-ub-20261109T225117Z-c1p-q8/) | **GREEN** `mad=0` / bit-exact greedy |
| **C2** / **C3** / **C2pf16** / **C2pq8** / **C2p** / **C1p** / **C1jp** | [`…225212Z`](../logs/task066-17-product-ub-20261109T225212Z/) full | **all GREEN** (`C2p` `exact=1 mad=0` both pairs) |
| Stress **C1** 16↔1024 | same full stamp | **RED** `first_token_diff=1` — **066.5/066.6**; explicitly out of 066.17 DoD |

Note: SWA ring-full warnings at `used_max_p1=size_swa=1536` under equal `-b=512` still appear; product bit-exact holds.

**066.17 closed.** Residual stress C1 stays open under parent 066.5/066.6 (no waive).

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
4. [x] Fix `headroom=PAD128(n_batch)` + unit PASS.  
5. [x] **C2pf16 + C1pf16 GREEN** (`…215457Z-c1p-f16`).  
6. [x] **C2pq8 + C2p + C1p + C1jp GREEN** (`…225117Z-c1p-q8` + full `…225212Z`).  
7. [x] This file + parent pointer + CHANGELOG synced.

---

## Stamp / post-fix (2026-11-10) — product GREEN

**Full stamp:** [`…225212Z`](../logs/task066-17-product-ub-20261109T225212Z/) · log [`run-06617-after-nbatch-headroom.log`](../logs/run-06617-after-nbatch-headroom.log)

| Gate | Result | Notes |
|------|--------|--------|
| C2 / C3 | GREEN | stress ub16 paths |
| C2pf16 / C2pq8 / **C2p** | **GREEN** | product 256↔512 + 256↔1024 `mad=0` |
| **C1p** / **C1jp** | **GREEN** | both pairs `first_token_diff=-1` |
| Stress C1 16↔1024 | **RED** | `first_token_diff=1` — 066.5/066.6 residual (not 066.17 DoD) |

Pre-fix RED table preserved under **Locked conclusion** above.

### Post-`e46b779` tip_control + causality (2026-11-09 Terminal)

Operator log: [`run-06617-tip-control-restamp.log`](../logs/run-06617-tip-control-restamp.log)  
Harness: [`run-06617-tip-control-restamp.sh`](../logs/run-06617-tip-control-restamp.sh)

| Stamp | What | Result |
|-------|------|--------|
| [`…232338Z`](../logs/task066-17-tip-control-20261109T232338Z/) | tip_control @ `PAD128(n_batch)` | **S_ub_c2pf16/c2pq8/c2p GREEN mad=0**; fa/mul_mat/cb/moe/iswa GREEN; `S_mul_mat_res` RED ulp=8 (066.7); `S_b1_shared` RED=`MISSING` API audit (not headroom) |
| [`…232638Z`](../logs/task066-17-size-swa-off-20261109T232638Z/) | tip−1 `L_size_swa_ub` OFF (= master `n_ubatch`) | **causality GREEN→RED**: iswa RED; c2pf16 mad **0→0.255**; c2pq8 **0→3.878**; c2p **0→7.635** (pre-fix class) |

**Locked:** product ⊥ub headroom is necessary for tip ubatch GREEN; disabling it alone restores pre-fix mad.

---

## Non-goals / anti-patterns

- Silent fallback to other ubatch if MODEL/CTX missing.  
- Soft ε on logits/tokens — bit-exact only.  
- Replacing 16↔1024 stress with product pairs.
