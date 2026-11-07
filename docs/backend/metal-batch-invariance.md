# Metal batch-invariance (Apple GPU)

Apple Metal decode / prefill used to change numeric results when the same tokens
were processed under different micro-batch (`n_ubatch`) widths, or when
speculative verify (`MTP`) compared width `N=1` against draft widths `{2,3,4}`.
That matches the failure class tracked upstream as
[ggml-org/llama.cpp#25618](https://github.com/ggml-org/llama.cpp/issues/25618).

This document is the public before/after for the Metal KEEP stack in this fork:
flash-attn always-tile, ISWA `size_swa` independent of `n_ubatch`, and
`GGML_METAL_MUL_MM_MIN` product default `0` with a CPU policy unit.

ASCII only in this file (project AGENTS.md).

## Symptom (before)

| Surface | Failure |
|---------|---------|
| FA op | `ub=8` took the **vec** kernel (`ne01 < 20`) while `ub>=20` took **tile** -> bit drift on shared queries |
| Long prefill (incl. FA-off) | Multi-step ubatch for `n >= 1344` drifted vs a single wide ubatch; root was ISWA `size_swa` scaling with `n_ubatch` |
| Prefill rem + MTP | Last ubatch remainder stayed on `mul_mv` while full chunks / verify widths used `mul_mm` (`MUL_MM_MIN` default was 7, then briefly 1) |
| Product MTP e2e | Speculative path engaged, but matmul path split decode vs verify until `MUL_MM_MIN=0` |

## Fixes (after)

| Fix | Code / knob | After |
|-----|-------------|-------|
| FA always-tile | `ggml_metal_op_flash_attn_ext_use_vec` default always tile; legacy shape-vec only with `GGML_METAL_FA_VEC=1` | FA C2 / multi-step C7 bit-exact on Metal |
| ISWA capacity | `size_swa` perpendicular to `n_ubatch` (sliding window sized from model `n_swa`, not ubatch) | Long-prefill C3 (`n=1344`) + product S8 vs S1024 greedy stream GREEN |
| Float-tile mm for all `ne11` | Product `GGML_METAL_MUL_MM_MIN` default **0**; policy header `ggml/src/ggml-metal/ggml-metal-mul-mm-policy.h` | Prefill rem + MTP N=1 share `mul_mm` with verify widths; residual + `--mtp-verify` GREEN |

Do not raise `MUL_MM_MIN` to "hide" mm. Opt into legacy FA vec only for A/B (`GGML_METAL_FA_VEC=1`).

## Regression gates (ctest)

Build with Metal and tests:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DLLAMA_BUILD_TESTS=ON

cmake --build build --parallel \
  --target test-flash-attn-batch-invariance \
  --target test-mul-mat-batch-invariance \
  --target test-longctx-row0-parity \
  --target test-moe-row0-parity \
  --target test-moe-router-row0-parity \
  --target test-moe-out-row0-parity \
  --target test-metal-mul-mm-policy \
  --target test-metal-cb-fusion \
  --target test-ubatch-s8-s1024-row0-parity
```

Default suite (same as CI): tests carry `LABELS main`, so they run with the
project-wide gate documented in `.github/workflows/build-cpu.yml` /
`build-vulkan.yml`:

```bash
unset GGML_METAL_MUL_MM_MIN
cd build
ctest -L main --output-on-failure --timeout 900
```

Metal hosts run the GPU corners; hosts without Metal or without a model SKIP
with exit code 77 (still green for the suite). Narrow filter if needed:

```bash
ctest --test-dir build -L main -R 'test-(flash-attn-batch-invariance|mul-mat-batch-invariance|metal-mul-mm-policy|metal-cb-fusion|longctx-row0|moe-row0|moe-router-row0|moe-out-row0|ubatch-s8-s1024)' --output-on-failure
```

`test-ubatch-s8-s1024-row0-parity` needs an explicit model + ctx (no silent
substitution). Set both or the test SKIPs:

```bash
export TASK066_MODEL=/path/to/model.gguf
export TASK066_CTX=3072
# optional: TASK066_PROMPT=/path/to/prompt.txt
ctest --test-dir build -R test-ubatch-s8-s1024-row0-parity --output-on-failure
```

Optional shell umbrella for the same corner binaries (not required for ctest):

```bash
BUILD_DIR=build SUITE=mul-mat-residual ./tests/run-task066-metal-corners.sh
BUILD_DIR=build SUITE=fa              ./tests/run-task066-metal-corners.sh
BUILD_DIR=build SUITE=cb-fusion       ./tests/run-task066-metal-corners.sh
```

## Product e2e / knobs (examples only — not ctest)

Operator scripts (not registered in `ctest`):

| Path | Role |
|------|------|
| [`examples/metal-batch-invariance/`](../../examples/metal-batch-invariance/) | runners |
| [`benchmarks/metal-batch-invariance/`](../../benchmarks/metal-batch-invariance/) | default duel prompt + model notes |

### How to run

```bash
# 1) build llama-cli (Metal)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --parallel --target llama-cli

# 2) models under .scratch/ (or set TASK066_MODEL / TASK066_DRAFT)
./scripts/setup-gemma4-26b-a4b-models.sh

# 3) stop any other llama-server holding Metal VRAM (exclusive preflight)

# 4) MTP vs nomtp product duel (default prompt from benchmarks/)
unset GGML_METAL_MUL_MM_MIN
unset TASK066_PROMPT   # use benchmarks/metal-batch-invariance/pierre_dolokhov_duel_2k.prompt.txt
BUILD_DIR=build \
  TASK066_SAMPLE_MODE=product \
  TASK066_UBATCH_LIST=512 \
  TASK066_CELL_MODES=both \
  ./examples/metal-batch-invariance/run-mtp-vs-nomtp-ladder.sh

# optional: knobs matrix / nomtp ubatch ladder / dry-run
BUILD_DIR=build ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh
BUILD_DIR=build ./examples/metal-batch-invariance/run-ubatch-ladder.sh
DRY_RUN=1 ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh
```

Stamps: `logs/metal-mtp-vs-nomtp-<UTC>/`, `logs/metal-mtp-knob-matrix-<UTC>/`,
`logs/metal-ubatch-ladder-<UTC>/`.

More detail: [`examples/metal-batch-invariance/README.md`](../../examples/metal-batch-invariance/README.md).

Gates (product sample mode):

- HARD: MTP engagement evidence + non-empty generation (no Metal OOM/assert)
- MTP vs nomtp text DIFF under product sampling: **telemetry**, not FAIL
- Product sample is intentionally not bit-exact vs nomtp

### MTP vs nomtp (stamp `20261107T041336Z`)

HARD OK, duel 2/2, acceptance ~0.88; tg ~12 (MTP) vs ~9 (nomtp).
Post-fixtures-move reconfirm: `logs/metal-mtp-vs-nomtp-20261107T063420Z` PASS.

### Knobs matrix (stamp `20261107T043032Z`)

All 12 cells HARD OK + duel 2/2. Speed-Pareto among duel=2/2 was `n4-p085`
(tg ~13.05, acceptance ~0.72). Operator KEEP for product knobs stays
`(n_max=2, p_min=0.95)` (acceptance ~0.88, tg ~12.29) — acceptance preferred
over the speed-Pareto cell. Product YAML is not auto-promoted by these scripts.

Acceptance-sorted summary:

| id | n_max | p_min | acceptance | tg (tok/s) |
|----|------:|------:|-----------:|-----------:|
| n1-p095 | 1 | 0.95 | 0.915 | 11.62 |
| n2-p095 | 2 | 0.95 | 0.884 | 12.29 |
| n1-p090 | 1 | 0.90 | 0.871 | 11.74 |
| n1-p085 | 1 | 0.85 | 0.849 | 11.84 |
| n3-p095 | 3 | 0.95 | 0.838 | 12.40 |
| n4-p095 | 4 | 0.95 | 0.821 | 12.39 |
| n2-p090 | 2 | 0.90 | 0.812 | 12.52 |
| n3-p090 | 3 | 0.90 | 0.777 | 12.79 |
| n2-p085 | 2 | 0.85 | 0.777 | 12.61 |
| n4-p090 | 4 | 0.90 | 0.744 | 12.81 |
| n3-p085 | 3 | 0.85 | 0.743 | 13.02 |
| n4-p085 | 4 | 0.85 | 0.718 | 13.05 |

## Environment knobs

| Variable | Default | Effect |
|----------|---------|--------|
| `GGML_METAL_MUL_MM_MIN` | `0` (product) | Min `ne11` for float-tile `mul_mm`; `0` => all widths on mm |
| `GGML_METAL_FA_VEC` | unset | `1` restores legacy FA shape-based vec (`ne01 < 20`) |
| `GGML_METAL_FA_DISPATCH` | unset | `1` logs FA vec vs tile dispatch |
| `TASK066_MODEL` / `TASK066_DRAFT` | `.scratch/...` if unset | Main / MTP draft GGUF for e2e |
| `TASK066_PROMPT` | `benchmarks/metal-batch-invariance/pierre_dolokhov_duel_2k.prompt.txt` | Override duel prompt |
| `TASK066_CTX` | e2e default `3072`; ubatch ctest required | Context; ubatch ctest SKIP 77 if unset |

## Related

- Issue class: ggml-org/llama.cpp#25618
- Speculative decoding overview: [docs/speculative.md](../speculative.md)
- Build Metal: [docs/build.md](../build.md#metal-build)
