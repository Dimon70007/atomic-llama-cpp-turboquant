# Metal batch-invariance - product e2e (not ctest)

Operator scripts for long MTP vs nomtp duels and the `n_max x p_min` knobs
matrix. Autotest corners stay under `tests/` and default `ctest -L main`.

| Piece | Path |
|-------|------|
| Public before/after | [docs/backend/metal-batch-invariance.md](../../docs/backend/metal-batch-invariance.md) |
| Runners (this dir) | `examples/metal-batch-invariance/` |
| Default prompt / model notes | [benchmarks/metal-batch-invariance/](../../benchmarks/metal-batch-invariance/) |

## Prerequisites

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --parallel --target llama-cli
./scripts/setup-gemma4-26b-a4b-models.sh   # or export TASK066_MODEL / TASK066_DRAFT
unset GGML_METAL_MUL_MM_MIN
```

Stop any other `llama-server` that holds Metal VRAM (scripts refuse shared GPU
unless `TASK066_ALLOW_SHARED_METAL=1`).

Optional overrides (otherwise `.scratch/` + benchmarks prompt):

```bash
export TASK066_MODEL=/path/to/main.gguf
export TASK066_DRAFT=/path/to/mtp-draft.gguf   # MTP scripts only
export TASK066_PROMPT=/path/to/prompt.txt      # default: benchmarks/.../pierre_dolokhov_duel_2k.prompt.txt
export TASK066_CTX=3072
```

## Scripts

| Script | Role |
|--------|------|
| `run-mtp-vs-nomtp-ladder.sh` | HARD engagement + duel; DIFF under product sample = telemetry |
| `run-mtp-knob-matrix.sh` | Thin entry -> `mtp_knob_matrix.py` (`n_max x p_min`) |
| `run-ubatch-ladder.sh` | Nomtp ubatch ladder (product confirm, not ctest) |
| `run-mtp-e2e-corners.sh` | Delegates to fullgraph N=1 vs N=2 row0 localizer |

```bash
# primary e2e (product sample, ubatch 512)
BUILD_DIR=build \
  TASK066_SAMPLE_MODE=product \
  TASK066_UBATCH_LIST=512 \
  TASK066_CELL_MODES=both \
  ./examples/metal-batch-invariance/run-mtp-vs-nomtp-ladder.sh

BUILD_DIR=build ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh
BUILD_DIR=build ./examples/metal-batch-invariance/run-ubatch-ladder.sh
DRY_RUN=1 ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh
```

Stamps: `logs/metal-mtp-vs-nomtp-<UTC>/`, `logs/metal-mtp-knob-matrix-<UTC>/`,
`logs/metal-ubatch-ladder-<UTC>/`.

## Gates (fixed)

- HARD: non-empty generation + MTP engagement evidence; no Metal OOM/assert
- Product sample: MTP vs nomtp text DIFF is telemetry (not FAIL)
- Product sample is not bit-exact by design
- Do not auto-rewrite product YAML from matrix recommend

## ctest (corners only)

```bash
cd build && ctest -L main --output-on-failure --timeout 900
```

See [docs/backend/metal-batch-invariance.md](../../docs/backend/metal-batch-invariance.md).
