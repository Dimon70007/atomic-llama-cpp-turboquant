# TASK-066 — nomtp ubatch ladder fixture

Product ladder knobs for active track [README-TASK066.md](README-TASK066.md) / [TASK-066.4-short-model-s8-s1024.md](TASK-066.4-short-model-s8-s1024.md)  
(hist plan: [task066-hist/TASK-066-execution-plan.md](task066-hist/TASK-066-execution-plan.md)).  
Unit suites stay separate (`SUITE=followup`); this fixture is for Metal Terminal / operator runs.

In-repo files: [fixtures/task066/](fixtures/task066/).

## Model

Same layout as `scripts/run-gemma4-mtp-server.sh`. Details: [fixtures/task066/models.md](fixtures/task066/models.md).

| Field | Value |
|-------|-------|
| HF | `unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_XS` |
| Default path | `.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` |
| Setup | `./scripts/setup-gemma4-26b-a4b-models.sh` (links from `$HOME/models`) |
| Override | `MAIN_GGUF=...` or `TASK066_MODEL=...` |

## Prompt (~2k)

| Field | Value |
|-------|-------|
| In-repo SSOT | `tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt` |
| Override | `TASK066_PROMPT=/path/to/prompt.txt` |

Short Pierre/Dolokhov duel excerpt + RU summary instruction. Ladder runner defaults to the in-repo path.

## Generation / sampling (ladder)

| Knob | Value | Notes |
|------|-------|-------|
| `n_predict` | `256` (default; up to 512 OK) | `TASK066_N_PREDICT` |
| `temperature` | `0` | greedy |
| `top_k` | `1` | pin greedy |
| `seed` | `42` | |
| thinking | off | cli: `--reasoning off` |
| MTP | Stage 1: **off**; Stage 2/P3: on via `run-task066-mtp-vs-nomtp-ladder.sh` | draft: `DRAFT_GGUF` / `.scratch/.../mtp-gemma-4-26B-A4B-it-Q8_0.gguf` |

## Host / graph

| Knob | Ladder default | Notes |
|------|----------------|-------|
| `flash_attn` | `on` | |
| `cache-type-k` | `q8_0` | **locked** (product) |
| `cache-type-v` | `turbo4` | **locked** (product); no other KV types for TASK-066 stamps |
| `ngl` | `99` | |
| `threads` / `threads_batch` | `4` / `4` | |
| `ctx` | `3072` | ~2k prompt + 256 predict + template |
| `cache-ram` | `0` (**required**) | CLI default is **8192 MiB** and OOMs 24 GB with this model; product yaml uses 4096 |
| `ctx-checkpoints` | `32` | product |
| `batch` / `ubatch` | **default:** `-b = max(N, 512)`, `-ub N` (`TASK066_BATCH_MODE=floor`) | `-b 1` breaks ~2k prefill |
| **Product ubatch** | floor **256**, target **512**; large **1024/2048** | Prefill+decode on float-tile mm under default `MUL_MM_MIN=0` (066.9 B; A2 rem + MTP verify) |
| equal / max stamps | optional A/B | |
| cli turn | `--conversation --single-turn --simple-io --no-warmup` + stdin `/dev/null` | `--simple-io` required so generation lands in `cli.log` / `generation.txt` |

Reference yaml (key names only): [fixtures/task066/ladder-nomtp.reference.yaml](fixtures/task066/ladder-nomtp.reference.yaml).

## Env (Metal)

| Variable | Ladder Stage 0/1 debug | Product / unit residual |
|----------|------------------------|-------------------------|
| `GGML_METAL_MUL_MM_MIN` | **`7`** / **`8`** / **`1`** (legacy; rem or MTP holes) | unset or **`0`** (066.9 B product: N≥1 on float-tile mm) |

`SUITE=mul-mat-residual` SKIP(77) if `GGML_METAL_MUL_MM_MIN` is set to anything other than `0`.

## Binary

| Mode | How |
|------|-----|
| Fast loop | `llama-cli` via `tests/run-task066-ubatch-ladder.sh` |
| Canon | `llama-server` with the same knobs as the reference yaml + ephemeral `batch`/`ubatch` per cell |

### llama-cli flag map

```bash
PROMPT="${TASK066_PROMPT:-tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt}"
MODEL="${MAIN_GGUF:-.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf}"
B=$(( N > 512 ? N : 512 ))
llama-cli \
  -m "$MODEL" \
  -f "$PROMPT" \
  -c 3072 -ngl 99 -t 4 -tb 4 \
  -b "$B" -ub "$N" \
  -fa on -ctk q8_0 -ctv turbo4 \
  -cram 0 -ctxcp 32 \
  -n 256 --temp 0 --top-k 1 --seed 42 \
  --reasoning off --jinja \
  --conversation --single-turn --simple-io --no-display-prompt --no-warmup \
  < /dev/null
```

### llama-server yaml keys (canon)

See [fixtures/task066/ladder-nomtp.reference.yaml](fixtures/task066/ladder-nomtp.reference.yaml).

| yaml key | cli / meaning |
|----------|----------------|
| `hf_model` / resolved GGUF | `-m` |
| `batch` | `-b` / `--batch-size` |
| `ubatch` | `-ub` / `--ubatch-size` |
| `flash_attn` | `-fa` |
| `quant_k` / `quant_v` | `-ctk` / `-ctv` (**locked** `q8_0`/`turbo4`) |
| `gpu_layers` | `-ngl` |
| `threads` / `threads_batch` | `-t` / `-tb` |
| `context_size` | `-c` |
| `chat_template_kwargs` | `--reasoning off` |
| `temperature` / `top_p` / `top_k` | `0` / `1` / `1` for greedy stamp |
| (no `model_draft` / `spec_*`) | MTP off |

## Ladder cells

`{1, 8, 16, 33, 256, 512, 1024, 2048}` with **2x** same-ubatch repeats on `{1024, 2048}` (run1/run2 — not a duplicate ladder step).

Out dir: `logs/task066-ubatch-ladder-<stamp>/` + stub `matrix.md`.

## Runner

```bash
# Metal Terminal (GPU). Agent sandbox has no Metal.
# Stop product llama-server first — a second UD-IQ4_XS load OOMs ~24 GB Metal.
# (minillm often listens on 127.0.0.1:8180 with the same GGUF.)
./scripts/setup-gemma4-26b-a4b-models.sh   # once
unset TASK066_PROMPT TASK066_MODEL
export GGML_METAL_MUL_MM_MIN=8
BUILD_DIR=build-task066-cb ./tests/run-task066-ubatch-ladder.sh
```
