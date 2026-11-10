# TASK-069 — CUDA e2e fixture (fork SSOT)

In-repo knobs for **CUDA** architecture e2e under `tests/` (ladders, fullgraph/mtp-e2e, ubatch-s8-s1024, mtp-vs-nomtp).  
Metal TASK-066 fixture stays: [TASK-066-ladder-fixture.md](TASK-066-ladder-fixture.md).

**Reference YAMLs (this fork):**

| Arm | File |
|-----|------|
| MTP | [fixtures/task069/cuda-mtp.reference.yaml](fixtures/task069/cuda-mtp.reference.yaml) |
| nomtp | [fixtures/task069/cuda-nomtp.reference.yaml](fixtures/task069/cuda-nomtp.reference.yaml) |

**Upstream product (LiteLLM, parent repo):** `configs/gemma4_26b_a4b_mtp_cuda.yaml` / `…_nomtp.yaml`.  
E2E YAMLs copy those knobs; only `context_size` is reduced for test speed (`230000` → `8192`).

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

## Host / graph (CUDA product lock)

| Knob | Value | Notes |
|------|-------|-------|
| `batch` / `ubatch` | **8192 / 2048** | CUDA SSOT (Metal was 2048/512) |
| `quant_k` / `quant_v` | **q8_0 / turbo3** | CUDA SSOT (Metal turbo4) |
| `tensor_split` | `1.05,0.95` | dual V100 |
| `main_gpu` | `1` | |
| `cache_ram` | `16384` | product; do not silent-default to Metal `0`/`4096` |
| `ctx_checkpoints` | `128` | |
| `threads` / `threads_batch` | `16` / `16` | |
| `flash_attn` | `on` | |
| `context_size` (e2e) | `8192` | product `230000`; raise if FIXTURE_FAIL |
| MTP | `draft-mtp` `n_max=2` `p_min=0.95` | forbidden: MTP-off / `p_min=1` |

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
