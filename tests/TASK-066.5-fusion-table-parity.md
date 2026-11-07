# TASK-066.5 — C3-n1344 (ISWA size_swa invariance)

**Status:** 🟢 **KEEP** — C3-n1344 + C1 GREEN (2026-11-07)  
**Depends on:** [066.7 KEEP](TASK-066.7-mul-mat-residual.md); FA always-tile C2 GREEN; [066.9](TASK-066.9-fa-batch-invariance.md) Phase A localize  
**Delivered lever:** ISWA `size_swa` independent of context `n_ubatch` (not fuse-table)  
**MTP:** still [066.9](TASK-066.9-fa-batch-invariance.md) Phase B gated  
**Source hypothesis (superseded):** fuse/`fusion_max` — bisect showed attn/`kqv_out`; root was SWA capacity

### Root cause (evidence)

Gemma4 `n_swa=1024`. Old ISWA size:

```text
size_swa = PAD(min(ctx, n_swa + n_ubatch), 256)
  ub=8     → PAD(1032,256)=1280
  ub=1344  → PAD(2368,256)=2432
```

Multi-ubatch contexts slide the SWA cache after **1280** tokens; one-shot / large-ub contexts still hold a longer window → logits diverge. Matches:

| Probe | Result |
|-------|--------|
| `c3-thresh` | n=1280 GREEN; **n=1288 first RED** |
| `c3-bisect` n=1280 | logits exact; `FIRST_DIFF=<none>` |
| `c3-bisect` n=1344 | `FIRST_DIFF=kqv_out-0` (FA); FA-off → `kq-0` |
| H9faOFF | RED → not FA-only (SWA storage class) |

### Fix

[`src/llama-kv-cache-iswa.cpp`](../src/llama-kv-cache-iswa.cpp): drop `n_ubatch` from `size_swa`; headroom = `size_base` so every `-ub` shares capacity (and oneshot ub≤ctx still fits). When `ctx ≥ n_swa`, `size_swa == size_base` (full-length SWA **cells**; window mask via `n_swa` / `swa_type` unchanged).

### TDD / stamps

1. Autotest gate: `SUITE=ubatch-s8-s1024` — C2 + C3-n1280 + **C3-n1344** + C1 (**GREEN** after fix; was RED on C3).
2. Diag: `TASK066_DIAG=c3-bisect|c3-thresh` (evidence; exit 0).
3. Regression same change set: `cb-fusion` / `mul-mat-residual` / `fa` **GREEN** (`logs/task066-regress-iswa-*`).
4. Ladder confirm `logs/task066-ubatch-ladder-20261107T012625Z` vs `231804Z`: **HELP** — S512≡S1024 (was DIFF); S8 vs S1024 still DIFF@22 under jinja/`--conversation` CLI (C1 raw greedy GREEN for n_predict=64).

### Bisect / thresh commands

```bash
TASK066_MODEL=.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
TASK066_CTX=3072 BUILD_DIR=build-task066-cb-ws \
  TASK066_DIAG=c3-bisect ./tests/run-task066-ubatch-s8-s1024-corners.sh
TASK066_DIAG=c3-thresh …   # pad/SWA threshold matrix
```

## Goal (original fuse-table — deferred)

`ggml_metal_fusion_max` vs encoder fuse parity remains a valid hygiene item if a new RED corner names it. **Not** the C3-n1344 KEEP.

## Evidence hierarchy (066.0 lock 7)

1. C3 was RED on pre-fix HEAD → ISWA edit → C3/C1 GREEN in the same change set.
2. No ε-PASS; product stays FA-on + turbo4.

## DoD

- [x] Autotest RED→GREEN on C3-n1344 (`SUITE=ubatch-s8-s1024`).
- [x] Lever documented (ISWA `size_swa`, not FA / not fuse-table).
- [x] C1 GREEN; ladder logged HELP vs `231804Z`.
- [x] Regressions `cb-fusion` / `mul-mat-residual` / `fa` GREEN.
- [ ] Optional follow-up: jinja CLI S8 vs S1024 DIFF@22 (outside C1 gate).

## Checklist

- [x] FIRST_DIFF / thresh before Metal/ops guess.
- [x] One non-FA lever (ISWA capacity).
- [x] No blind encoder fuse ports.
- [x] Ladder confirm stamped (HELP).
