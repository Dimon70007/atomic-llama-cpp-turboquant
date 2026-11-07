# Metal batch-invariance fixtures

In-repo prompts and notes for the Metal batch-invariance e2e / corner runners.
Lives under `benchmarks/` so it is not swallowed by `tests/.gitignore` (`*` +
directory un-ignore rules).

| File | Role |
|------|------|
| `pierre_dolokhov_duel_2k.prompt.txt` | ~2k duel prompt (nomtp ladder / MTP e2e) |
| `ladder-nomtp.reference.yaml` | Key map for server/canon docs; ladder uses `llama-cli` |
| `models.md` | Local GGUF registry (`.scratch/gemma-4-26b-a4b/`) |

Models follow repo canon (`MAIN_GGUF`, `.scratch/...`), same as `scripts/run-gemma4-*.sh`.
Product KV for these fixtures: `-ctk q8_0 -ctv turbo4`.

```bash
./scripts/setup-gemma4-26b-a4b-models.sh
unset GGML_METAL_MUL_MM_MIN
unset TASK066_PROMPT   # default: this directory / pierre_dolokhov_duel_2k.prompt.txt
BUILD_DIR=build \
  TASK066_SAMPLE_MODE=product \
  TASK066_UBATCH_LIST=512 \
  ./examples/metal-batch-invariance/run-mtp-vs-nomtp-ladder.sh
```

How to run (full): [examples/metal-batch-invariance/README.md](../../examples/metal-batch-invariance/README.md)  
Before/after: [docs/backend/metal-batch-invariance.md](../../docs/backend/metal-batch-invariance.md).
