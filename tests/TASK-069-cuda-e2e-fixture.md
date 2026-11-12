# TASK-069 — CUDA e2e fixture (fork SSOT)

In-repo knobs for **CUDA** architecture e2e under `tests/` (ladders, fullgraph/mtp-e2e, ubatch-s8-s1024, mtp-vs-nomtp).  
Metal TASK-066 fixture stays: [TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md).

**Reference YAMLs (this fork):**

| Role | Arm | File | Profile |
|------|-----|------|---------|
| **Primary (ST-069.4 Metal-parity)** | MTP | [fixtures/task069/cuda-mtp.reference.yaml](fixtures/task069/cuda-mtp.reference.yaml) | `…_mtp_cuda_e2e` |
| **Primary** | nomtp | [fixtures/task069/cuda-nomtp.reference.yaml](fixtures/task069/cuda-nomtp.reference.yaml) | `…_nomtp_e2e` |
| Archive (V100 product knobs) | MTP | [fixtures/task069/cuda-mtp-v100-product.reference.yaml](fixtures/task069/cuda-mtp-v100-product.reference.yaml) | `…_v100_product_e2e` |
| Archive | nomtp | [fixtures/task069/cuda-nomtp-v100-product.reference.yaml](fixtures/task069/cuda-nomtp-v100-product.reference.yaml) | `…_nomtp_v100_product_e2e` |

**Upstream product (LiteLLM, parent repo):** `configs/gemma4_26b_a4b_mtp_cuda.yaml` / `…_nomtp.yaml` — **unchanged** by ST-069.4 (`8192/2048`, `turbo3`).  
Primary e2e fixtures align with Metal TASK-066 host/KV for apples-to-apples; product ship-check uses archived v100-product YAML via `TASK069_SUITE_CONFIG=…`.

## Model

Same as TASK-066 ladder ([fixtures/task066/](fixtures/task066/)):

| Field | Value |
|-------|-------|
| HF | `unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_XS` |
| Draft | `mtp-gemma-4-26B-A4B-it-Q8_0.gguf` |
| Setup | `./scripts/setup-gemma4-26b-a4b-models.sh` |
| Override | `TASK066_MODEL=` / `TASK066_MODEL_DRAFT=` (fail-closed; no silent default) |

## Prompt

| Field | Value |
|-------|-------|
| In-repo | `tests/fixtures/task069/task066_pierre_dolokhov_duel_2k.prompt.txt` → `../task066/…` |
| Override | `TASK066_PROMPT=` |

## Host / graph — primary Metal-parity (default)

| Knob | Value | Notes |
|------|-------|-------|
| `batch` / `ubatch` | **2048 / 512** | Metal ladder / product parity |
| `quant_k` / `quant_v` | **q8_0 / turbo4** | Metal parity (was CUDA product turbo3) |
| `context_size` (e2e) | **3072** | Metal ladder; ~2k duel + predict |
| `cache_ram` | **4096** | Metal product (ladder CLI often `0`) |
| `ctx_checkpoints` | **32** | Metal product |
| `threads` / `threads_batch` | **4 / 4** | Metal parity |
| `tensor_split` | `1.05,0.95` | dual V100 (CUDA-only) |
| `main_gpu` | `1` | CUDA-only |
| `flash_attn` | `on` | |
| sampling | `1.0` / `0.95` / `64` | CUDA product sampling (not Metal ladder greedy) |
| MTP | `draft-mtp` `n_max=2` `p_min=0.95` | forbidden: MTP-off / `p_min=1` |

## Host / graph — archive V100 product

| Knob | Value |
|------|-------|
| `batch` / `ubatch` | **8192 / 2048** |
| `quant_v` | **turbo3** |
| `context_size` | **8192** (product `230000` reduced for speed) |
| `cache_ram` / `ctx_checkpoints` | **16384** / **128** |
| `threads` | **16 / 16** |
| `tensor_split` / `main_gpu` | same as primary |

Select via:

```bash
TASK069_SUITE_CONFIG=tests/fixtures/task069/cuda-mtp-v100-product.reference.yaml \
  ./tests/run-task069-cuda-e2e.sh suite
```

Loader fail-closed (`tests/task069_suite_knobs.py`): family `metal_parity` requires turbo4+2048/512; `v100_product` requires turbo3+8192/2048.

## Honest expectation

Knob align helps fullgraph / ubatch / ladders compare Metal vs CUDA. Unit micros (`fa` / `longctx` / `moe` / …) may stay RED on CUDA even after parity if Metal-only FA/mm ports remain inert.

## Runner (preferred)

One entry — sets suite YAML knobs + resolves model/draft/prompt (fail-closed):

```bash
# GPU Terminal (agent sandbox often hides /dev/nvidia* → CPU trash).
cd atomic-llama-cpp-turboquant
./scripts/setup-gemma4-26b-a4b-models.sh   # once (links $HOME/models → .scratch)

./tests/run-task069-cuda-e2e.sh help
./tests/run-task069-cuda-e2e.sh env              # print resolved knobs
./tests/run-task069-cuda-e2e.sh mtp-e2e
./tests/run-task069-cuda-e2e.sh ubatch-s8
./tests/run-task069-cuda-e2e.sh ubatch-ladder
./tests/run-task069-cuda-e2e.sh mtp-vs-nomtp
./tests/run-task069-cuda-e2e.sh all-e2e          # four e2e runners in order
./tests/run-task069-cuda-e2e.sh suite            # OFF→ON full suite compare

# optional overrides:
BUILD_DIR=build-task069-cuda-off ./tests/run-task069-cuda-e2e.sh mtp-e2e
TASK066_MODEL=/path/to.gguf TASK066_MODEL_DRAFT=/path/to-mtp.gguf \
  ./tests/run-task069-cuda-e2e.sh suite
```

## cli map (from reference yaml)

| yaml key | cli |
|----------|-----|
| `batch` / `ubatch` | `-b` / `-ub` |
| `quant_k` / `quant_v` | `-ctk` / `-ctv` |
| `tensor_split` | `-ts` |
| `main_gpu` | `-mg` |
| `cache_ram` | `-cram` |
| `ctx_checkpoints` | `-ctxcp` |
| `gpu_layers` | `-ngl` |
| `threads` / `threads_batch` | `-t` / `-tb` |
| `model_draft` + `spec_*` | `--mtp-head` / `--spec-type` / `--spec-draft-n-max` / `--spec-draft-p-min` |
