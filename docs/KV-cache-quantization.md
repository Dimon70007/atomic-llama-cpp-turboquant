# KV Cache Quantization with TurboQuant

TurboQuant adds three runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization:

| Type               | Enum                       | Size            | Compression vs f16 |
|--------------------|----------------------------|-----------------|--------------------|
| `turbo2`           | `GGML_TYPE_TURBO2_0` (43)  | 2 bits/value    | 6.4x               |
| `turbo3`           | `GGML_TYPE_TURBO3_0` (44)  | 3.25 bits/value | 4.9x               |
| `turbo4`           | `GGML_TYPE_TURBO4_0` (47)  | 4.25 bits/value | 3.8x               |

These are KV-cache-only types: they are never stored in model files. The
corresponding model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S`
(46) - 3/4-bit WHT-rotated Lloyd-Max quantization, block size 32, exposed in
`llama-quantize` as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

Any combination of `f16`, `q8_0`, `turbo2`, `turbo3`, `turbo4` for K and V is
supported; mixing quantized V with unquantized K is the common configuration.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed). A quantized V cache with flash attention explicitly disabled is an
error, matching upstream behavior for all quantized V types.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## Model-specific quality

Models with attention sinks can be unusually sensitive to K-cache quantization. GPT-OSS is a known case: even `q8_0` K changes the output distribution substantially, and lower-bit K types degrade it further despite normal codec and kernel accuracy. Use `f16` K for GPT-OSS and other sink-heavy models. Validate a quantized V cache separately against an `f16` K/V baseline before deploying it.

Short output samples are not a sufficient quality check for this class of model because the text can remain fluent while token probabilities move significantly. Use `llama-perplexity --kl-divergence` or an equivalent logit comparison when selecting cache types.

## Rotation

K and V vectors are rotated by a fixed 128x128 orthonormal Walsh-Hadamard
matrix before quantization and inverse-rotated after dequantization. Head
dimensions that are not multiples of 128 are zero-padded to the next multiple
of 128 for turbo types. MLA models have no separate V cache (V is a view of
K), so V rotation and padding are skipped for them.

## Environment knobs

| Variable                        | Default | Effect                                                          |
|---------------------------------|---------|-----------------------------------------------------------------|
| `TURBO_LAYER_ADAPTIVE`          | `0`     | Layer-adaptive KV precision on plain `turbo*`; `7` = Boundary V (first/last layers in `q8_0`, middle in turbo2). **Not** supported with `-ctv turbo3.8\|turbo4.8` (abort) |
| `TURBO_LAYER_HEAD`              | `2`     | With q8 brackets: # of first layers with V=`q8_0` (optional)   |
| `TURBO_LAYER_TAIL`              | `8`     | With q8 brackets: # of last layers with V=`q8_0` (optional)    |
| `TURBO_LAYER_Q8_BRACKETS`       | unset   | Set by `-ctv turbo3.8\|turbo4.8` (do not set by hand)           |
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                | `1`     | Sparse-V dequant skip in flash attention (`0` disables)        |
| `LLAMA_ATTN_ROT_K_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for K                |
| `LLAMA_ATTN_ROT_V_OVERRIDE`     | off     | Enable upstream #21038 attention rotation for V                |
| `LLAMA_ATTN_ROT_DISABLE`        | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

### q8 head/tail brackets (`turbo3.8` / `turbo4.8`)

Named `-ctv` / `quant_v` aliases enable a fixed **layer-adaptive V schedule**
(K stays as requested, usually `q8_0`). They are first-class cache-type
names, not shorthand for uniform `turbo3`/`turbo4`.

#### Why (MTP + long context)

Uniform aggressive V (`turbo3` / `turbo4`) is fine on short prompts (~2k
tokens). Under **MTP speculative decoding** with a **long prompt
(roughly >= 70k tokens)**, quantization error accumulates in the KV cache
and shows up as **factual LM errors** (wrong actors / outcomes in long
summaries), even when short fixtures still look correct.

`turbo3.8` / `turbo4.8` keep the **head** (user question / instructions)
and the **tail** layers at `q8_0`, and compress only the **middle** with
`turbo3` or `turbo4`. That cuts mid-depth KV traffic while protecting the
layers that matter most for instruction fidelity and late-context readout.
Turning MTP off is **not** the intended product fix for this class of drift.

#### Layout

| Alias       | Mid V    | Layout                                      |
|-------------|----------|---------------------------------------------|
| `turbo4.8`  | `turbo4` | front q8 + mid **turbo4** + tail q8         |
| `turbo3.8`  | `turbo3` | front q8 + mid **turbo3** + tail q8         |

```text
[ front M_eff x q8_0 ][ mid x turbo4 ][ tail N x q8_0 ]   # -ctv turbo4.8
[ front M_eff x q8_0 ][ mid x turbo3 ][ tail N x q8_0 ]   # -ctv turbo3.8
```

Widths: `N = min(TAIL, n_layer-HEAD-1)`, `M_eff = min(HEAD, n_layer-N-1)`.
At least one mid layer must remain. Unset HEAD/TAIL -> defaults **2/8**
with a loud log. Plain `turbo3`/`turbo4` stay **uniform** (brackets do
**not** auto-enable). If `TURBO_LAYER_ADAPTIVE` is set with a bracket
alias, the server **aborts** (legacy modes `0-7` stay on plain `turbo*`
only). Block KV streaming refuses mixed per-layer types (same as
`TURBO_LAYER_ADAPTIVE`). When `n_layer == 0` (MTP draft stub), the KV
**ctor** skips brackets so the draft context can still start; calling
`llama_kv_cache_q8_brackets_type_v` with `n_layer == 0` **throws**.

```bash
llama-server -m model.gguf -c 8192 -ngl 99 -ctk q8_0 -ctv turbo4.8
# optional width override (no rebuild):
# TURBO_LAYER_HEAD=1 TURBO_LAYER_TAIL=8
# FORBIDDEN with aliases: TURBO_LAYER_ADAPTIVE=...
```

#### Regression gates (TASK-068)

**CI / PR (`ctest -L main`):** only the CPU schedule unit is in the default suite.

```bash
ctest --test-dir build -L main -R test-q8-brackets-schedule --output-on-failure
```

`test-q8-brackets-schedule` locks HEAD/TAIL defaults (incl. HEAD=0/1, min 1+1),
mid turbo3|turbo4, ADAPTIVE abort, `n_layer==0` throw on `type_v` (ctor still
skips MTP draft stubs), width overflow, and bad env ints.

**Manual e2e (not in ctest):** long-ctx KL ladder for reviewers who need to
reproduce the drift class and see why brackets help.

| | Uniform same mid (`turbo3` / `turbo4`) | Brackets (`turbo3.8` / `turbo4.8`) |
|--|----------------------------------------|-------------------------------------|
| Short prompt (~2k) | Often looks fine | Same |
| Long prompt + MTP (~>=70k) | KV V error accumulates -> factual LM mistakes | q8 head/tail protect instructions + late layers; mid stays compressed |
| Fork measurable proxy | Higher Mean KLD vs `-ctv q8_0` | **Lower** Mean KLD vs same q8_V **and same mid** |

Factual MTP KEEP (narrative duel / Voyna-class) is product evidence, not this
script. The fork script only proves distribution drift on long prefill:

```bash
# NOT part of ctest -L main. Requires GGUF + free GPU. No silent defaults.
cmake --build build --target llama-perplexity
export TASK068_MODEL=/path/to/model.gguf
export TASK068_CTX=8192    # Darwin: prefer 8192; CTX>8192 needs TASK068_ALLOW_HIGH_CTX_MEM=1
# optional: TASK068_PROMPT=/path/to/prompt.txt  # must tokenize to >= 2*CTX tokens
# optional: TASK068_CHUNKS=2 (default)  # cap logits file
# Darwin ST-8: default TASK068_BATCH=CTX (num_batches=1; no host logits.reserve).
# Product-like -b 2048 with CTX=8192 needs TASK068_ALLOW_MULTI_BATCH_PPL=1 (swap risk).
# optional: TASK068_UBATCH=512 TASK068_PARALLEL=1
# Exclusive Metal/CUDA: stop other llama-server / minillm first.
./tests/run-task068-q8-brackets-kl.sh ./build/bin/llama-perplexity
# Pass (same-mid): KL(turbo3.8)<KL(turbo3) AND KL(turbo4.8)<KL(turbo4) vs q8_V.
# Cross-mid turbo3.8 vs turbo4 is NOT a gate (mid bitwidth confound).
# Opt-out turbo4 arms: TASK068_KL_SKIP_TURBO4_GATE=1 (5 PPL runs -> 3; Darwin RAM).
# Missing env or GPU busy: exit 77 (skip).
# Darwin RAM: --chunks 2, CTX<=8192 unless ALLOW_HIGH_CTX_MEM; BATCH=CTX default;
#   BATCH < CTX refused unless ALLOW_MULTI_BATCH_PPL; logits cleaned after parse.
# ST-8 mem probe (host Terminal): TASK068_MEM_PHASE=product|base|kl_one|kl_full|batch_eq_ctx \
#   ./tests/run-task068-mem-st8-probe.sh  -> logs/task068-mem-st8-<stamp>/
```

**Regression entry points (split unit vs e2e):**

```bash
# Everyday UNIT (no model, no KL): Metal smoke + test-q8-brackets-schedule / ctest -L main
BUILD_DIR=./build-st7-post-rebase ./tests/run-task068-unit-verify.sh
# Optional: TASK068_SKIP_CTEST=1; TASK068_ALLOW_UNRELATED_CTEST_FAILS=1; REBUILD=1
# Logs: logs/task068-unit-<stamp>/

# Holiday / major TURBOQUANT_VERSION bump: UNIT + KL e2e (explicit flag, no silent e2e)
# Stop minillm / llama-server first (exclusive GPU).
export TASK068_MODEL=/path/to/model.gguf
export TASK068_CTX=8192   # Darwin; >8192 needs TASK068_ALLOW_HIGH_CTX_MEM=1
TASK068_RUN_KL_E2E=1 BUILD_DIR=./build-st7-post-rebase \
  ./tests/run-task068-post-rebase-verify.sh
# Or KL alone: ./tests/run-task068-q8-brackets-kl.sh ./build-st7-post-rebase/bin/llama-perplexity
# KL defaults: -b = CTX (ST-8); both same-mid gates; SKIP turbo4: TASK068_KL_SKIP_TURBO4_GATE=1
# Logs: logs/task068-kl-post-rebase-<stamp>/summary.txt
```

Upstream attention rotation is off by default: TurboQuant manages rotation
itself (the WHT applied at cache write is equivalent and interacts with the
cache types). `LLAMA_ATTN_ROT_*` only affects the optional upstream rotation
path for models that benefit from it.

## Model-weight quantization (TQ3_1S / TQ4_1S)

```bash
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

`TQ3_1S` and `TQ4_1S` are first-class weight types with CUDA/HIP (warp
cooperative mmvq), Metal, and Vulkan kernels. MoE models disable CUDA graphs
for TQ `MUL_MAT_ID` automatically.
