# TASK-066.17 — Product ubatch parity gates (256↔512, 256↔1024)

**Status:** 🟡 **SPEC / design locked** (2026-11-09) — implementation not started  
**Depends on:** 066.5/066.6 product gate binary `test-ubatch-s8-s1024-row0-parity`; 066.14 size_swa pivot; tip `MUL_MM_MIN_DEFAULT=8` (upstream-like)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)  
**Binary / runner:** `tests/test-ubatch-s8-s1024-row0-parity.cpp` + `tests/run-task066-ubatch-s8-s1024-corners.sh`  
**Umbrella:** `SUITE=ubatch-s8-s1024` (no new suite name)

---

## Why

Product Darwin/Linux profiles use real ubatch **256 / 512 / 1024** (e.g. minillm `ubatch: "512"`, some `256`/`1024`).  
Stress pair **16↔1024** (post-`ub≥9` contract; historical A-side was 8) catches rem/size_swa edge classes but is **not** the operator path.  
Need hard gates on the ubatch values operators actually run.

Prior evidence (not waived): triple/KLD/ladder stamps under 066.11–066.14 showed cross-ub divergence when `size_swa∝ub`; product pivot **accepted** that tax for decode speed. 066.17 does **not** waive — it **locks** product-pair bit-exact as an explicit gate so regressions are visible.

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
… existing C2 → C3 → C1(stress) → C1j(stress) …
  OR (preferred wiring): after C3 GREEN, run C2p → C1p → C1jp, then stress C1/C1j
```

Implementation choice (either OK if DoD lists both):

1. **Preferred:** `C2 → C3 → C2p → C1p → C1jp → C1(stress 16↔1024) → C1j(stress)`  
2. **Alt:** keep current stress C1/C1j after C3; append C2p/C1p/C1jp at end.

First RED stops the suite (no waive).

### Fast loop

- `TASK066_DIAG=c1p` — only product pairs (C2p + C1p; optional C1jp via `c1jp`).  
- Unset diag = full suite including stress + product pairs.

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

1. [ ] C2p + C1p + C1jp implemented in `test-ubatch-s8-s1024-row0-parity` with fail-closed ub contract.  
2. [ ] `TASK066_DIAG=c1p` (and `c1jp` if split) wired; runner comment + fork CHANGELOG.  
3. [ ] Stamp under `logs/task066-17-product-ub-…/` with C2p/C1p/C1jp pass/fail recorded (no waive).  
4. [ ] This file + parent pointer row updated to Done only after GREEN **or** explicit KEEP-with-RED decision recorded (numbers + root class).  
5. [ ] Parent `README-TASK066.md` suite blurb updated **in the same change set as code** (§5.0).

---

## Non-goals / anti-patterns

- Silent fallback to other ubatch if MODEL/CTX missing.  
- Soft ε on logits/tokens — bit-exact only.  
- Replacing 16↔1024 stress with product pairs.
