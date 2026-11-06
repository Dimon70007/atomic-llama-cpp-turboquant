# TASK-066 — Metal batch-invariance corner tests

Self-contained in **this** llama.cpp / ggml fork. No external product repos required.

These tests gate Metal patches that keep speculative decode row0 bit-exact vs serial decode (see ggml-org/llama.cpp#25618 class: `mul_mv` / FA `nsg` / MoE paths).

## Configure + build (Metal)

From this repository root (see also [docs/build.md](../docs/build.md) § Metal):

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
  --target test-fullgraph-decode-row0-parity \
  --target test-backend-ops
```

`BUILD_DIR` defaults to `<repo>/build`. Override with `BUILD_DIR=/path/to/build` if needed.

## Run

Required selector (fail-closed; no silent default):

```bash
SUITE=all ./tests/run-task066-metal-corners.sh
# or one suite:
SUITE=fa ./tests/run-task066-metal-corners.sh
SUITE=mul-mat ./tests/run-task066-metal-corners.sh
SUITE=longctx ./tests/run-task066-metal-corners.sh
SUITE=moe ./tests/run-task066-metal-corners.sh
SUITE=moe-router ./tests/run-task066-metal-corners.sh
# ST-3.2f-ter-loc MoE out dual-topology (expected RED on A+B; not in `all`):
SUITE=moe-out ./tests/run-task066-metal-corners.sh
# ST-3.2f patch-C TDD fullgraph (expected RED on A+B; not included in `all`):
# Product knobs pinned to gemma4_26b_a4b_mtp_metal.yaml; only TASK066_CTX may shrink.
TASK066_MODEL=/path/to/UD-IQ4_XS.gguf TASK066_CTX=16384 \
  SUITE=fullgraph ./tests/run-task066-metal-corners.sh
```

Optional: `REBUILD=1` rebuilds suite binaries before run; pass an output directory as argv1 (default: `logs/task066-metal-corners-<suite>-<stamp>/` under this repo).

| SUITE | Runner | What |
|-------|--------|------|
| `fa` | `run-task066-fa-nsg-corners.sh` | FA nsg / row0 (C1/C2/C4 + C6 smoke) |
| `mul-mat` | `run-task066-mul-mat-corners.sh` | mul_mat / mul_mat_id (C1/C4/C5 + C6) |
| `longctx` | `run-task066-longctx-corners.sh` | long-KV FA row0 localizer |
| `moe` | `run-task066-moe-corners.sh` | MoE mul_mv_id IQ4_XS / hetero ids |
| `moe-router` | `run-task066-moe-router-corners.sh` | router→soft_max→top_k→moe |
| `moe-out` | `run-task066-moe-out-corners.sh` | ffn_moe_out permute+sum_rows vs view+add (ST-3.2f-ter-loc RED) |
| `fullgraph` | `run-task066-fullgraph-corners.sh` | llama decode N=1 vs N=2 row0 logits (ST-3.2f RED) |
| `all` | umbrella | fa…moe-router (**not** moe-out/fullgraph) |

Host without Metal: binaries exit `77` (ctest SKIP). Treat as host-skip, not a patch KEEP.

## Extending coverage

Add `tests/test-*.cpp` + `tests/run-task066-*.sh`, register the cmake target in `tests/CMakeLists.txt`, and wire the suite name into `run-task066-metal-corners.sh`. Keep fixtures and runners inside this repository.
