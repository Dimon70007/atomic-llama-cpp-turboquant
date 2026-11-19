#!/usr/bin/env bash
# TASK-073.1: find the shortest prompt that still drifts under checkpoint restore.
#
# Runs one llama-server per size, each repeating one greedy request, and reports
# how many checkpoints were created/restored. A size whose cell never restored a
# checkpoint is NOT-EXERCISED, not a pass.
#
# On Metal the product service must be stopped first (the GPU is exclusive
# there); on CUDA/Vulkan a running server is only a warning.
#
# Usage:
#   make minillm-task073.1-prompt-ladder
#   make minillm-task073.1-prompt-ladder ARGS='--words 24,18,12'
#
# Env (optional):
#   TASK0731_MODEL   SWA GGUF (default ~/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf)
#   TASK0731_SERVER  llama-server binary (default: resolved from BUILD_DIR / build/ / build-dev/)
#   BUILD_DIR        CMake build directory to take llama-server from
#
# Exit: 0 no exercised cell drifted, 1 a cell drifted, 2 bad input.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/lib/task0731_prompt_ladder.py"
# shellcheck source=tests/lib/llama_server_bin.sh
source "${REPO_ROOT}/tests/lib/llama_server_bin.sh"

MODEL="${TASK0731_MODEL:-${HOME}/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf}"
OUT="${TASK0731_OUT:-${REPO_ROOT}/logs/task073.1-prompt-ladder}"

[[ -f "${MODEL}" ]] || { echo "SKIP: model not found: ${MODEL}" >&2; exit 77; }

SERVER="$(llama_server_resolve "${REPO_ROOT}" "${TASK0731_SERVER:-}")" || exit 2

llama_server_busy_guard || exit 2

# shellcheck disable=SC2086 # ARGS is a deliberate flag list
exec python3 "${PY}" --model "${MODEL}" --bin "${SERVER}" --out-dir "${OUT}" ${ARGS:-}
