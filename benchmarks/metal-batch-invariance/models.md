# Gemma 4 26B-A4B — local GGUF layout (Metal batch-invariance)

Follow the same convention as `scripts/run-gemma4-mtp-server.sh`:

```text
.scratch/gemma-4-26b-a4b/<name>.gguf
```

`.scratch/` is gitignored. Populate once:

```bash
./scripts/setup-gemma4-26b-a4b-models.sh
# searches $HOME/models by default; override with MODELS_DIR=...
```

## Registry

| Role | File under `.scratch/` | Override |
|------|------------------------|----------|
| MTP / turbo default (scripts) | `gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` | `MAIN_GGUF` |
| Nomtp / MTP ladder target | `gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf` | `MAIN_GGUF` / `TASK066_MODEL` |
| Optional QAT sibling | `gemma-4-26b-a4b/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf` | `MAIN_GGUF` |
| MTP draft head | `gemma-4-26b-a4b/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` | `DRAFT_GGUF` / `TASK066_DRAFT` |
| Alias for `run-gemma4-mtp-server.sh` | `gemma-assistant-mtp.gguf` (repo `.scratch/` root) | `DRAFT_GGUF` |

HF / local sources:

- Target IQ4_XS: `unsloth/gemma-4-26B-A4B-it-GGUF:UD-IQ4_XS`
- Draft: `$HOME/models/mtp-gemma-4-26B-A4B-it-Q8_0.gguf` (F16 sibling optional)
