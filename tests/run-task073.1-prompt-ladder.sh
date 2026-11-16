#!/usr/bin/env bash
# TASK-073.1: find the shortest prompt that still drifts under checkpoint restore.
#
# Runs one llama-server per size, each repeating one greedy request, and reports
# how many checkpoints were created/restored. A size whose cell never restored a
# checkpoint is NOT-EXERCISED, not a pass.
#
# Metal is exclusive: stop the product service first.
#
# Usage:
#   make minillm-task073.1-prompt-ladder
#   make minillm-task073.1-prompt-ladder ARGS='--words 24,18,12'
#
# Exit: 0 no exercised cell drifted, 1 a cell drifted, 2 bad input.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/lib/task0731_prompt_ladder.py"

MODEL="${TASK0731_MODEL:-${HOME}/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf}"
SERVER="${TASK0731_SERVER:-${REPO_ROOT}/build-dev/bin/llama-server}"
OUT="${TASK0731_OUT:-${REPO_ROOT}/logs/task073.1-prompt-ladder}"

[[ -f "${MODEL}" ]] || { echo "SKIP: model not found: ${MODEL}" >&2; exit 77; }
[[ -x "${SERVER}" ]] || { echo "FAIL: server not executable: ${SERVER}" >&2; exit 2; }

if pgrep -f 'llama-server' >/dev/null 2>&1; then
  echo "WARN: a llama-server is running - Metal is exclusive, stop it first:" >&2
  echo "      make minillm-stop" >&2
fi

# shellcheck disable=SC2086 # ARGS is a deliberate flag list
exec python3 "${PY}" --model "${MODEL}" --bin "${SERVER}" --out-dir "${OUT}" ${ARGS:-}
