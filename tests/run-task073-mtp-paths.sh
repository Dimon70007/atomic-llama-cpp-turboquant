#!/usr/bin/env bash
# TASK-073 ST-4: run every MTP drafting path against the checkpoint-restore
# contract and report which ones still drift.
#
# Operator entry point. Stops nothing and installs nothing. On Metal the product
# service must be stopped first (the GPU is exclusive there); on CUDA/Vulkan a
# running server is only a warning.
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
#   TASK073_SERVER  llama-server binary (default: resolved from BUILD_DIR / build/ / build-dev/)
#   TASK073_OUT     output root (default <repo>/logs/task073-st4-paths)
#   BUILD_DIR       CMake build directory to take llama-server from
#   ARGS            extra flags for the orchestrator (--dry-run, --only x,y)
#
# Exit: 0 all runnable cells deterministic, 1 a cell drifts or failed, 2 bad input.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/lib/task073_mtp_paths.py"
# shellcheck source=tests/lib/llama_server_bin.sh
source "${REPO_ROOT}/tests/lib/llama_server_bin.sh"

JSON="${TASK073_JSON:-${REPO_ROOT}/tests/task073-mtp-paths.json}"
OUT="${TASK073_OUT:-${REPO_ROOT}/logs/task073-st4-paths}"

[[ -f "${PY}" ]] || { echo "FAIL: missing ${PY}" >&2; exit 2; }
[[ -f "${JSON}" ]] || { echo "FAIL: missing matrix ${JSON}" >&2; exit 2; }

# Resolving here (not in a default assignment) keeps the failure message useful:
# it lists every path tried instead of naming one Darwin-only directory.
SERVER="$(llama_server_resolve "${REPO_ROOT}" "${TASK073_SERVER:-}")" || exit 2

# Warn early so the operator sees a busy GPU before a long cell starts.
llama_server_busy_guard || exit 2

# shellcheck disable=SC2086 # ARGS is a deliberate flag list
exec python3 "${PY}" --json "${JSON}" --server "${SERVER}" --out-dir "${OUT}" ${ARGS:-}
