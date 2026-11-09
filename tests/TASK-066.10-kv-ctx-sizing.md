# TASK-066.10 KV / n_ctx sizing (SSOT)

**Status:** ⬜ **TODO** (not done) — estimator stamps @100k/ub512 locked; closeout still gated on 066.5 C3 RED + KLD FAIL  
**Depends on:** 066.5 ideas `size_swa` (pad128 + `n_ubatch`; C3 RED + KLD FAIL honest stamped)  
**Scope first:** uniform KV types `f16`, `q8_0`, `turbo3`, `turbo4` only (no layer-bracket aliases).  
**Operator entry (product repo root):**

```bash
make estimate-kv-ctx \
  MODEL=atomic-llama-cpp-turboquant/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
  N_CTX=3072 CTK=q8_0 CTV=turbo4 N_UBATCH=512 \
  N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false
```

ASCII only. Scope: llama.cpp / this TurboQuant fork (`-ctk` / `-ctv`). Do **not** mix vLLM block/fp8 math into product estimates.

## Closed form (full-attention cells)

Canonical shape matches ggml-org/llama.cpp#10068 and `llama-kv-cache.cpp` tensors:

```text
KV_bytes = n_cells x Sum_il (
    row_size(type_k(il), n_embd_k_gqa_eff(il))
  + row_size(type_v(il), n_embd_v_gqa_eff(il))
) x n_stream
```

`row_size(type, ne) = type_size(type) * ne / blck_size(type)` (`ggml_row_size`).  
`n_stream = 1` if unified KV, else `n_seq_max`.

| Symbol | Source |
|--------|--------|
| `n_layer` | GGUF `%s.block_count` |
| `n_head_kv` | `%s.attention.head_count_kv` (scalar or per-layer array; not `head_count`) |
| `d_k` / `d_v` | `key_length` / `value_length`, else `embd / n_head` |
| SWA head dims | `key_length_swa` / `value_length_swa` when present |
| `n_embd_k_gqa` | `n_head_kv * d_k` (same for V) |
| `type_k` / `type_v` | **`-ctk` / `-ctv`**, never weight GGUF quant |
| `n_cells` | `size_base` (non-SWA) or `size_swa` (SWA layers) |

## Bytes per element (`ggml-common.h`)

Primary types for this subtask:

| type | type_size / blck | bytes/elem |
|------|------------------|------------|
| f16 | 2 / 1 | 2 |
| q8_0 | 34 / 32 | **1.0625** (not 1.0) |
| turbo3 | sizeof(block_turbo3_0) / QK_TURBO3 | 50/128 = 0.390625 |
| turbo4 | sizeof(block_turbo4_0) / QK_TURBO4 (`TURBO4_USE_4BIT`) | 66/128 = 0.515625 |

q8_0 vs f16: size ratio is **17/32** (~0.531), not `32/17`. Using integer "1 byte" underestimates KV by ~6.25%.

## Turbo WHT pad

If type is turbo* and `head_dim % 128 != 0`:

```text
padded_head = ceil(head_dim / 128) * 128
n_embd_*_gqa_eff = n_head_kv * padded_head
```

Same as `llama-kv-cache.cpp` (turbo zero-pad). MLA / no-V layouts are out of scope for this estimator (fail closed if needed later).

## ISWA cell counts (066.5 ideas formula)

Runtime helper: `llama_kv_cache_iswa_calc_size_swa` in `src/llama-kv-cache-iswa.cpp`.  
Adopted from product [`ideas/llamacpp_swa_size_calculate.md`](../../../ideas/llamacpp_swa_size_calculate.md).

```text
size_base = PAD(n_ctx, 256)   # llama_context pads n_ctx before memory create
n_swa_eff = ceil(n_swa / 128) * 128
size_swa  = PAD(min(size_base, n_swa_eff * (unified ? n_seq_max : 1) + n_ubatch), 256)
```

Locks:

- `N_UBATCH` is **required** for estimates (fail-closed; no silent default).
- Long product `n_ctx` keeps **windowed** SWA cells (RAM win vs prior size_base KEEP).
- Capacity depends on `-ub` again; C3 cross-ub must be re-stamped under 066.5 (no silent waive).
- Prior size_base KEEP (`size_swa == n_ctx` when ctx >= n_swa) is **superseded** for product long-ctx.

## Inverse n_ctx

```text
bytes_per_token = KV_bytes / size_base   # coarse; SWA portion is windowed
n_ctx_max = floor(KV_BUDGET_BYTES / bytes_per_token)
```

Only when `KV_BUDGET_MIB=` is set. Weights + compute/graph have **no** closed form here; dry-run / init log (`KV buffer size = ... MiB`) is still required for host fit.

## Rejected conflations

| Wrong | Why |
|-------|-----|
| Weight GGUF quant as KV bytes/elem | KV types are `-ctk`/`-ctv` only |
| vLLM `fp8` = 1 B/elem for llama `q8_0` | ggml q8_0 is 1.0625 B/elem |
| size_base KEEP as long-ctx product truth | ideas formula restores windowed SWA |
| Max ctx from KV formula alone | Ignores weights + compute |

## Tooling

- Python: `tests/estimate_kv_ctx.py`
- Shell: `tests/run-estimate-kv-ctx.sh`
- Units: `tests/test_estimate_kv_ctx.py`
- Product Make: `make estimate-kv-ctx MODEL=... N_CTX=... CTK=... CTV=... N_UBATCH=... N_SEQ_MAX=... UNIFIED=... SWA_FULL=...`

## Gemma4 26B-A4B stamp

Model: `.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`  
Arch metadata: `n_layer=30`, per-layer `n_head_kv` (SWA=8 / full=2), full `d=512`, SWA `d=256`, `n_swa=1024`, pattern 25 SWA / 5 full (period-6 False every 6th). Synthetic helper: `gemma4_26b_a4b_meta()` in `estimate_kv_ctx.py`.

Estimator commands (fail-closed args). **Unit / doc lock: `N_CTX=100000`, `N_UBATCH=512`.**

```bash
# from product repo root
make estimate-kv-ctx MODEL=atomic-llama-cpp-turboquant/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
  N_CTX=100000 CTK=f16 CTV=f16 N_UBATCH=512 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false
make estimate-kv-ctx MODEL=... N_CTX=100000 CTK=q8_0 CTV=q8_0 N_UBATCH=512 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false
make estimate-kv-ctx MODEL=... N_CTX=100000 CTK=q8_0 CTV=turbo3 N_UBATCH=512 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false
make estimate-kv-ctx MODEL=... N_CTX=100000 CTK=q8_0 CTV=turbo4 N_UBATCH=512 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false
```

**Stamp id:** `066.10-estimate-ideas-20261108` (unit lock @100000; PAD->100096; size_swa=1536)  
**Host stamp dir:** `logs/task066-10-kv-ctx-100k-ub512-20261108T201019Z/` (DoD matrix `dod-*.log`; pytest 12 passed).

| knobs | N_CTX | N_UBATCH | size_base | size_swa | KV_MiB | notes |
|-------|-------|----------|-----------|----------|--------|-------|
| f16 / f16 | 100000 | 512 | 100096 | 1536 | 2255.0000 | baseline |
| q8_0 / q8_0 | 100000 | 512 | 100096 | 1536 | 1197.9688 | |
| q8_0 / turbo3 | 100000 | 512 | 100096 | 1536 | 819.1992 | windowed |
| q8_0 / turbo4 | 100000 | 512 | 100096 | 1536 | 889.6680 | |

Host fit still needs weights + compute from a dry-run.

## Definition of Done — not met

- [x] Doc SSOT aligned with ideas formula + uniform CTV only.
- [x] Tooling: `make estimate-kv-ctx` fail-closed (`N_UBATCH` required); units green (`test_estimate_kv_ctx.py` 12 passed).
- [x] Frozen stamps @ N_CTX=100000 / N_UBATCH=512 for `f16/f16`, `q8_0/q8_0`, `q8_0/turbo3`, `q8_0/turbo4` (`…201019Z`).
- [ ] Cross-check vs init-log KV MiB for at least one uniform turbo config.
- [ ] CHANGELOG / pointer row closed only after 066.5 C3/KLD product decision (C3 RED + KLD FAIL honest remain).

## Non-goals

- Claiming C3 GREEN without a new stamp after ideas formula.
- Treating weight GGUF quant as KV bytes/elem.
- Claiming max context from KV alone without compute/weights budget.
- FP16 force for SWA layers (ideas alt; out of scope here).
- Layer-bracket CTV schedules (deferred until uniform types are locked).
