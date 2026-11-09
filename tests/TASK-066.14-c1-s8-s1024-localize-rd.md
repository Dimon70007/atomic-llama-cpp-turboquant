# TASK-066.14 — C1 equal-`-b` vs unequal-`-b` / fixed `size_swa` KEEP

**Status:** ✅ **H6 R&D done** → **product pivot (speed):** headroom back to live `-ub` + **PAD128**  
**Depends on:** 066.13 C2; Exp A stamp; post-KEEP decode regression (size_swa×2 ≈ tok/s÷2)  
**Ideas:** [`ideas/size_swa_rnd.md`](../../../ideas/size_swa_rnd.md)  
**Product pointer:** [`tasks/TASK-066-nomtp-ubatch-s8-s1024.md`](../../../tasks/TASK-066-nomtp-ubatch-s8-s1024.md)

## Root (named)

Same model/quant/ctx: C1 RED when harness used unequal `-b` because **066.13** set  
`n_swa_headroom = n_batch` → unequal `size_swa` (1536 vs 2048).  
Equal `-b` (H1) or fixed headroom (Exp A) → GREEN even with unequal decode chunks / unequal `-ub`.

**Not causal here:** Metal mul_mm; decode chunk borders alone (Exp A).

**Speed lesson:** fixed headroom `2048` with product `-ub=512` doubles `size_swa` (1536→3072). When the SWA ring fills, `used_max_p1→size_swa` so FA `n_kv` grows with allocation — keep **minimal sufficient** (`n_swa + n_ubatch`), pad **128** (TurboQuant), not 256 (256 is FA *graph* pad in `get_n_kv`, not allocation). Stamp `…022827Z`: filled-ring nomtp ~+6.5% vs fixed-2048; short bench flat; earlier ÷2 claim not reproduced.

## Product (post-pivot)

```text
n_swa_headroom = cparams.n_ubatch
size_swa = PAD128(min(size_base, n_swa_eff + n_swa_headroom))
```

Cross-ub text divergence accepted (already present). R&D override still:  
`TASK066_ALLOW_SIZE_SWA_HEADROOM=1` + `TASK066_SIZE_SWA_HEADROOM=<n>`.

Telemetry: create log `size_swa calc:…`; apply_ubatch `SWA occupancy hwm: used_max_p1=… size_swa=… slack=…`.  
**Visibility:** `LLAMA_LOG_INFO` maps to verbosity TRACE (**4**); product default `-lv 3` hides calc/hwm — use `-lv 4` / `LLAMA_ARG_LOG_VERBOSITY=4`.

## What `llama_kv_cache_iswa_calc_size_swa` is (and is not)

**One job:** return how many **cells** to allocate for the SWA half of iSWA
(`kv_swa`). It does **not** size the full-attn cache (`kv_base` ≈ `n_ctx`).

```text
llama_kv_cache_iswa
├── kv_base  ← size_base ≈ PAD(n_ctx)     // non-SWA / full layers
└── kv_swa   ← size_swa = calc_size_swa(…) // SWA layers only
```

Ctor wires that number straight into `llama_kv_cache(…, size_swa, …)`
(`src/llama-kv-cache-iswa.cpp`). Downstream: SWA buffer bytes, `get_size()`,
and the ceiling for FA `n_kv` / KQ mask on SWA layers. That is why oversized
`size_swa` slows decode (~∝ SWA FA work).

### Windowed vs `--swa-full` (memory saving)

| Mode | `size_swa` | Meaning |
|------|------------|---------|
| default (`swa_full=false`) | `≈ n_swa + headroom` (pad128) | sliding SWA ring + prune |
| `--swa-full` | `= size_base` | full-ctx SWA; no window prune |

Upstream iSWA ([PR #13194](https://github.com/ggml-org/llama.cpp/pull/13194)):
original window formula `PAD(n_swa·n_seq_max + n_batch)`; later product path
uses **`n_ubatch` headroom** (minimal room for one ubatch write while keeping
the last `n_swa` tokens). `--swa-full` is the fallback when you must keep all
SWA history (context edit / no prune) — more RAM, not “more FA”.

`calc_size_swa` branches on **`swa_full` only**. It does **not** read
`flash_attn`.

### FA vs this helper (common confusion)

- **Windowed allocation** (not allocating `size_base` for SWA) comes from
  **iSWA + `swa_full=false`**, not from enabling FA.
- **FA** is still required on our product path for **quantized V**
  (`turbo3`/`turbo4` — fail-closed / auto-enable in `llama-context`). FA is a
  *consumer* of `size_swa` via `n_kv`; it does not decide the ring length.
- Without FA, windowed iSWA can still exist; without FA we simply cannot run
  turbo-V.

### Why pad **128** (not 256)

- **128** = TurboQuant **storage/quant block** (`QK_TURBO3` / `QK_TURBO4` =
  128 in `ggml-common.h`; one block per WHT rotation group). Aligning
  `size_swa` to 128 avoids mid-block rotation / overwrite errors.
- **256** = FA **graph** pad in `get_n_kv` (reuse / ggml#17037) over *used*
  cells — orthogonal to how large we **allocate** the SWA ring. Padding
  allocation to 256 was an upstream habit that oversizes the ring and costs
  decode for no TurboQuant reason.

### Minimal sufficient (product)

```text
need  ≈ n_swa_eff + n_ubatch     // window + one ubatch write
alloc = PAD128(min(size_base, need))
```

Product YAML `-ub 512`, Gemma `n_swa=1024` → **`size_swa=1536`** (raw need
exactly; pad slack 0). Fixed-2048 KEEP forced **3072** (filled-ring hwm→3072).

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
| **`logs/task066-14-pivot-evidence-20261109T022827Z/`** | pivot live calc/hwm + nomtp A/B vs fixed-2048 | **pivot OK**; filled-ring decode **8.61 vs 8.08** tok/s (~+6.5%); short `llama-bench` flat; **÷2 not reproduced** |

## Definition of Done

- [x] H1 / Exp A / H6 named  
- [x] Product KEEP (`MAX_UBATCH` headroom) — **superseded** by ubatch+pad128 pivot (speed)  
- [x] Unit `test-iswa-size-swa` PASS (env clear)  
- [x] `pytest tests/test_estimate_kv_ctx.py` PASS  
- [x] Full corners GREEN without DIAG  
- [x] Ubatch ladder all≡S8 (`…005138Z`); MTP↔nomtp PASS (`…010049Z`)  
- [x] CHANGELOG + pointer  
- [x] Metal install-src overlay + sync-build `--skip-checkout`; `switch-model MODEL=gemma4-26b-a4b-mtp` (ctx=140000, health OK)  
- [x] **Pivot code:** product headroom=`n_ubatch`, outer pad=128; occupancy hwm log; Metal overlay rebuild + MTP switch  
- [x] **Pivot evidence:** live `size_swa calc` / occupancy hwm at `-lv 4` (`…022827Z`); clean nomtp A/B vs fixed-2048 (mechanism confirmed; ÷2 not reproduced in harness)

**Follow-up (perf, not size_swa):** dual pp+tg regression vs `d60d01875^` → **[066.15](TASK-066.15-metal-perf-knobs-rd.md)**.

## Non-goals

- Long-ctx TurboQuant drift / `TURBO_LAYER_ADAPTIVE` (066.5 / 066.10)  
- ε-PASS  
- Claiming all of 066.5 Done without remaining long-ctx KLD gates  
- Metal FA/mm default retune (066.15)  
