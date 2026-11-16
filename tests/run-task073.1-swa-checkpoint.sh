#!/usr/bin/env bash
# TASK-073.1 ST-1: checkpoint restore must reproduce one emission on an SWA model.
#
# Same contract as tests/run-task073-mtp-paths.sh (one greedy request repeated N
# times in one server emits one text), but the matrix runs an SWA model with MTP
# off. A drift there can only come from the checkpoint blob, not the drafter.
#
# The product service must be stopped first: Metal is exclusive.
#
# Usage:
#   make minillm-task073.1-swa-checkpoints
#   ARGS='--dry-run' make minillm-task073.1-swa-checkpoints
#   ARGS='--only swa-checkpoints-off' make minillm-task073.1-swa-checkpoints
#
# Env (optional):
#   TASK073_JSON     matrix file (default <repo>/tests/task073.1-swa-checkpoint.json)
#   TASK073_SERVER   llama-server binary (default <repo>/build-dev/bin/llama-server)
#   TASK073_OUT      output root (default <repo>/logs/task073.1-swa-checkpoints)
#   TASK0731_MODEL   override the SWA model; ctest sets it, otherwise the matrix
#                    default (~/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf) is used
#   ARGS             extra flags for the orchestrator (--dry-run, --only x,y)
#
# Exit: 0 all runnable cells deterministic, 1 a cell drifts or failed,
#       2 bad input, 77 the SWA model is absent (ctest skip).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/lib/task073_mtp_paths.py"

JSON="${TASK073_JSON:-${REPO_ROOT}/tests/task073.1-swa-checkpoint.json}"
SERVER="${TASK073_SERVER:-${REPO_ROOT}/build-dev/bin/llama-server}"
OUT="${TASK073_OUT:-${REPO_ROOT}/logs/task073.1-swa-checkpoints}"

[[ -f "${PY}" ]] || { echo "FAIL: missing ${PY}" >&2; exit 2; }
[[ -f "${JSON}" ]] || { echo "FAIL: missing matrix ${JSON}" >&2; exit 2; }

# ctest runs this without a model on the host: report the skip, do not fail.
if [[ -n "${TASK0731_MODEL:-}" && ! -f "${TASK0731_MODEL}" ]]; then
  echo "SKIP: TASK0731_MODEL not found: ${TASK0731_MODEL}" >&2
  exit 77
fi
if [[ -z "${TASK0731_MODEL:-}" && ! -f "${HOME}/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf" ]]; then
  echo "SKIP: SWA model absent (~/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf); set TASK0731_MODEL" >&2
  exit 77
fi

if pgrep -f 'llama-server' >/dev/null 2>&1; then
  echo "WARN: a llama-server is running - Metal is exclusive, stop it first:" >&2
  echo "      make minillm-stop" >&2
  pgrep -fl 'llama-server' >&2 || true
fi

# shellcheck disable=SC2086 # ARGS is a deliberate flag list
exec python3 "${PY}" --json "${JSON}" --server "${SERVER}" --out-dir "${OUT}" \
  ${TASK0731_MODEL:+--model-override "${TASK0731_MODEL}"} ${ARGS:-}
