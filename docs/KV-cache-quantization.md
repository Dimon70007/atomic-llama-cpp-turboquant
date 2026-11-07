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
`TURBO_LAYER_ADAPTIVE`). When `n_layer == 0` (MTP draft stub), brackets
are skipped so the draft context can still start.

```bash
llama-server -m model.gguf -c 8192 -ngl 99 -ctk q8_0 -ctv turbo4.8
# optional width override (no rebuild):
# TURBO_LAYER_HEAD=1 TURBO_LAYER_TAIL=8
# FORBIDDEN with aliases: TURBO_LAYER_ADAPTIVE=...
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
