# TASK-066.10 KV / n_ctx sizing (SSOT)

**Status:** DONE (tooling + docs; runtime ISWA KEEP unchanged)  
**Depends on:** 066.5 KEEP (`size_swa` ub-independent; C3 GREEN)  
**Operator entry (product repo root):**

```bash
make estimate-kv-ctx \
  MODEL=atomic-llama-cpp-turboquant/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
  N_CTX=3072 CTK=q8_0 CTV=turbo4 \
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

| type | type_size / blck | bytes/elem |
|------|------------------|------------|
| f16 / bf16 | 2 / 1 | 2 |
| f32 | 4 / 1 | 4 |
| q8_0 | 34 / 32 | **1.0625** (not 1.0) |
| q4_0 | 18 / 32 | 0.5625 |
| turbo2 | sizeof(block_turbo2_0) / QK_TURBO2 | 34/128 = 0.265625 |
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

## ISWA cell counts (066.5 KEEP)

Runtime helper: `llama_kv_cache_iswa_calc_size_swa` in `src/llama-kv-cache-iswa.cpp`.

```text
size_base = PAD(n_ctx, 256)   # llama_context pads n_ctx before memory create
# KEEP (n_ubatch ignored):
size_swa  = PAD(min(size_base, n_swa * (unified ? n_seq_max : 1) + size_base), 256)
# when ctx >= n_swa: size_swa == size_base
```

Locks:

- `n_ubatch` must **not** affect `size_swa` (old `n_swa + n_ubatch` caused C3-n1344 drift).
- Pure windowed `PAD(n_swa)` is **RED** on C3 (`find_slot 1280 > 1024`). Do not ship it without a new GREEN stamp.
- Long product `n_ctx` pays **full-length SWA cells** as a side effect of KEEP (not windowed RAM savings).

`ideas/llamacpp_swa_size_calculate.md` in the product repo described textbook windowed + `+ n_ubatch`. That note is **superseded** by this SSOT for fork/runtime math.

## q8 brackets (`turbo3.8` / `turbo4.8`)

Named `-ctv` schedules: head + tail layers use V=`q8_0`, mid uses `turbo3` / `turbo4`. Estimator requires explicit `--head` / `--tail` when CTV is a bracket alias (fail-closed; no silent HEAD/TAIL).

## Inverse n_ctx

```text
bytes_per_token = Sum over layers of (row_k + row_v) / n_cells_of_that_layer
                # practical print: KV_bytes(n_ctx) / n_ctx for linear KEEP case
n_ctx_max = floor(KV_BUDGET_BYTES / bytes_per_token)
```

Only when `KV_BUDGET_MIB=` is set. Weights + compute/graph have **no** closed form here; dry-run / init log (`KV buffer size = ... MiB`) is still required for host fit.

## Rejected conflations

| Wrong | Why |
|-------|-----|
| Weight GGUF quant as KV bytes/elem | KV types are `-ctk`/`-ctv` only |
| vLLM `fp8` = 1 B/elem for llama `q8_0` | ggml q8_0 is 1.0625 B/elem |
| Windowed `size_swa` independent of `n_ctx` as product truth | KEEP makes `size_swa == n_ctx` when ctx >= n_swa |
| Max ctx from KV formula alone | Ignores weights + compute |

## Tooling

- Python: `tests/estimate_kv_ctx.py`
- Shell: `tests/run-estimate-kv-ctx.sh`
- Units: `tests/test_estimate_kv_ctx.py`
- Product Make: `make estimate-kv-ctx MODEL=... N_CTX=... CTK=... CTV=... N_SEQ_MAX=... UNIFIED=... SWA_FULL=...`

## Gemma4 26B-A4B stamp

Model: `.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf`  
Arch metadata: `n_layer=30`, per-layer `n_head_kv` (SWA=8 / full=2), full `d=512`, SWA `d=256`, `n_swa=1024`, pattern 25 SWA / 5 full (period-6 False every 6th). Synthetic helper: `gemma4_26b_a4b_meta()` in `estimate_kv_ctx.py`. Unit locks: `test_q8_turbo3_8_kv_size_stamp` / `test_q8_turbo4_8_kv_size_stamp`.

Estimator commands (fail-closed args). **Unit tests lock `N_CTX=100000`.**

```bash
# from product repo root
make estimate-kv-ctx MODEL=atomic-llama-cpp-turboquant/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf \
  N_CTX=100000 CTK=q8_0 CTV=turbo3 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false
make estimate-kv-ctx MODEL=... N_CTX=100000 CTK=q8_0 CTV=turbo4 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false
make estimate-kv-ctx MODEL=... N_CTX=100000 CTK=q8_0 CTV=turbo3.8 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false HEAD=2 TAIL=8
make estimate-kv-ctx MODEL=... N_CTX=100000 CTK=q8_0 CTV=turbo4.8 N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false HEAD=2 TAIL=8
```

**Stamp id:** `066.10-estimate-20261108` (unit lock @100000; PAD→100096)

| knobs | N_CTX | size_base | size_swa | bytes/tok | est KV_MiB | notes |
|-------|-------|-----------|----------|-----------|------------|-------|
| q8_0 / turbo3 | 100000 | 100096 | 100096 | 81840 | 7812.3633 | KEEP |
| q8_0 / turbo4 | 100000 | 100096 | 100096 | 88880 | 8484.3945 | |
| q8_0 / turbo3.8 H2 T8 | 100000 | 100096 | 100096 | 94224 | 8994.5273 | brackets > turbo3 |
| q8_0 / turbo4.8 H2 T8 | 100000 | 100096 | 100096 | 98960 | 9446.6211 | brackets > turbo4 |

**Cross-check (log):** `logs/kvtype-verify-lv5/turbo3.err` @ `n_ctx=8192`, `-ctk q8_0 -ctv turbo3`:

```text
non-SWA:  58.12 MiB (5 layers)
SWA:     581.25 MiB (25 layers)
sum:     639.37 MiB
estimate_kv_ctx N_CTX=8192: KV_MiB=639.3750  (delta ~0%)
```

Tensor KV closed form matches init `llama_kv_cache: size = ... MiB` lines. Host fit still needs weights + compute from a dry-run.

## Non-goals

- Replacing 066.5 KEEP with silent windowed SWA.
- Treating weight GGUF quant as KV bytes/elem.
- Claiming max context from KV alone without compute/weights budget.
