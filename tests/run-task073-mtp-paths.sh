#!/usr/bin/env bash
# TASK-073 ST-4: run every MTP drafting path against the checkpoint-restore
# contract and report which ones still drift.
#
# Operator entry point. Stops nothing and installs nothing: the product service
# must be stopped first (Metal is exclusive).
#
# Usage:
#   make minillm-task073-mtp-paths            # plan + run, ~10 min per Gemma cell
#   make minillm-task073-mtp-paths ARGS='--dry-run'
#
# Or directly:
#   TASK073_JSON=tests/task073-mtp-paths.json ./tests/run-task073-mtp-paths.sh
#
# Env (optional):
#   TASK073_JSON    matrix file (default <repo>/tests/task073-mtp-paths.json)
#   TASK073_SERVER  llama-server binary (default <repo>/build-dev/bin/llama-server)
#   TASK073_OUT     output root (default <repo>/logs/task073-st4-paths)
#   ARGS            extra flags for the orchestrator (--dry-run, --only x,y)
#
# Exit: 0 all runnable cells deterministic, 1 a cell drifts or failed, 2 bad input.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/lib/task073_mtp_paths.py"

JSON="${TASK073_JSON:-${REPO_ROOT}/tests/task073-mtp-paths.json}"
SERVER="${TASK073_SERVER:-${REPO_ROOT}/build-dev/bin/llama-server}"
OUT="${TASK073_OUT:-${REPO_ROOT}/logs/task073-st4-paths}"

[[ -f "${PY}" ]] || { echo "FAIL: missing ${PY}" >&2; exit 2; }
[[ -f "${JSON}" ]] || { echo "FAIL: missing matrix ${JSON}" >&2; exit 2; }

# The orchestrator refuses a busy host itself; warn early so the operator sees it
# before a long cell starts.
if pgrep -f 'llama-server' >/dev/null 2>&1; then
  echo "WARN: a llama-server is running - Metal is exclusive, stop it first:" >&2
  echo "      make minillm-stop" >&2
  pgrep -fl 'llama-server' >&2 || true
fi

# shellcheck disable=SC2086 # ARGS is a deliberate flag list
exec python3 "${PY}" --json "${JSON}" --server "${SERVER}" --out-dir "${OUT}" ${ARGS:-}
