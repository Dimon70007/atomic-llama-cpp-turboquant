# TASK-066.14 — C1 equal-`-b` vs unequal-`-b` / fixed `size_swa` KEEP

**Status:** ✅ **H6 KEEP landed** — product `size_swa` headroom ⊥ live `-b`/`-ub`  
**Depends on:** 066.13 C2 (ub-invariant headroom idea); Exp A stamp  
**Ideas:** [`ideas/size_swa_rnd.md`](../../../ideas/size_swa_rnd.md)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)

## Root (named)

Same model/quant/ctx: C1 RED when harness used unequal `-b` because **066.13** set  
`n_swa_headroom = n_batch` → unequal `size_swa` (1536 vs 2048).  
Equal `-b` (H1) or fixed headroom (Exp A) → GREEN even with unequal decode chunks / unequal `-ub`.

**Not causal here:** Metal mul_mm; decode chunk borders alone (Exp A).

## KEEP (product)

```text
LLAMA_SWA_HEADROOM_MAX_UBATCH = 2048   # llama-kv-cache-iswa.h
n_swa_headroom = that constant         # llama-model.cpp create_memory
fail-closed if n_ubatch > constant
```

R&D override (optional): `TASK066_ALLOW_SIZE_SWA_HEADROOM=1` + `TASK066_SIZE_SWA_HEADROOM=<n>`.

## Stamps

| Stamp | What | Verdict |
|-------|------|---------|
| `…231714Z` baseline c1-only | product unequal `-b` pre-KEEP | RED |
| `…231534Z` H1 equal-b | `TASK066_C1_BATCH=1024` | GREEN |
| `…003249Z` Exp A | `SIZE_SWA_HEADROOM=2048` + unequal `-b` | GREEN |
| `…004302Z` KEEP c1-only | product path, no DIAG env | GREEN |
| **`logs/task066-14-c1-keep-full-20261109T004421Z/`** | full C2+C3+C1+C1j | **GREEN** |
| `logs/task066-14-ubatch-ladder-keep-20261109T005138Z/` | ubatch ladder floor, ub 1…2048 | **all identical vs S8** |
| `logs/task066-14-mtp-vs-nomtp-keep-20261109T010049Z/` | MTP↔nomtp ub512 product | **PASS** (DIFF telemetry OK) |

## Definition of Done

- [x] H1 / Exp A / H6 named  
- [x] Product KEEP (`MAX_UBATCH` headroom)  
- [x] Unit `test-iswa-size-swa` PASS (env clear)  
- [x] `pytest tests/test_estimate_kv_ctx.py` PASS  
- [x] Full corners GREEN without DIAG  
- [x] Ubatch ladder all≡S8 (`…005138Z`); MTP↔nomtp PASS (`…010049Z`)  
- [x] CHANGELOG + pointer  
- [x] Metal install-src overlay + sync-build `--skip-checkout`; `switch-model MODEL=gemma4-26b-a4b-mtp` (ctx=140000, health OK) 

## Non-goals

- Long-ctx TurboQuant drift / `TURBO_LAYER_ADAPTIVE` (066.5 / 066.10)  
- ε-PASS  
- Claiming all of 066.5 Done without remaining long-ctx KLD gates  
