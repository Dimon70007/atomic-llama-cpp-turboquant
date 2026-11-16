#!/usr/bin/env bash
# TASK-072 - minimal `spec_draft_p_min` sweep on Groot-19B-A3B (in-GGUF MTP).
#
# Every cell starts its own llama-server on a free port, runs one greedy request
# and exits, so the product service must be stopped first (Metal is exclusive).
#
# Usage:
#   ./tests/run-task072-groot-pmin-sweep.sh
#   ONLY=n2-p090,n2-p095 ./tests/run-task072-groot-pmin-sweep.sh
#
# Env (optional):
#   TASK072_MODEL      GGUF (default ~/models/Groot-19B-A3B-261023-Q4_K_M.gguf)
#   TASK072_SERVER     default <repo>/build-dev/bin/llama-server
#   TASK072_OUT        default <repo>/logs/task072-pmin-sweep
#   TASK072_MAX_TOKENS default 4096 (thinking needs room to close)
#   TASK072_CTX        default 16384
#   ONLY               comma-separated cell ids
#   TASK072_REPEATS    requests per cell on one server load (default 1; median is reported)
#   TASK072_SERVER_ARGS extra llama-server args, space separated (e.g. "--ctx-checkpoints 0")
#
# Exit: 0 all cells measured, 1 a cell failed, 2 bad input.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/lib/task072_groot_pmin_sweep.py"

MODEL="${TASK072_MODEL:-${HOME}/models/Groot-19B-A3B-261023-Q4_K_M.gguf}"
SERVER="${TASK072_SERVER:-${REPO_ROOT}/build-dev/bin/llama-server}"
OUT="${TASK072_OUT:-${REPO_ROOT}/logs/task072-pmin-sweep}"
MAX_TOKENS="${TASK072_MAX_TOKENS:-4096}"
CTX="${TASK072_CTX:-16384}"
REPEATS="${TASK072_REPEATS:-1}"

[[ -f "${PY}" ]] || { echo "FAIL: missing ${PY}" >&2; exit 2; }
[[ -x "${SERVER}" ]] || { echo "FAIL: missing server binary ${SERVER}" >&2; exit 2; }
[[ -f "${MODEL}" ]] || { echo "FAIL: missing model ${MODEL}" >&2; exit 2; }

# A second Metal process cannot start while the product service holds the GPU.
if pgrep -f 'llama-server' >/dev/null 2>&1; then
  echo "FAIL: a llama-server is already running; stop it first (make minillm-stop)" >&2
  pgrep -fl 'llama-server' >&2 || true
  exit 2
fi

ARGS=(--model "${MODEL}" --bin "${SERVER}" --out-dir "${OUT}"
      --max-tokens "${MAX_TOKENS}" --ctx "${CTX}" --repeats "${REPEATS}")
if [[ -n "${ONLY:-}" ]]; then
  ARGS+=(--only "${ONLY}")
fi
if [[ -n "${TASK072_SERVER_ARGS:-}" ]]; then
  read -r -a _extra <<< "${TASK072_SERVER_ARGS}"
  for _arg in "${_extra[@]}"; do
    ARGS+=(--server-arg="${_arg}")
  done
fi

exec python3 "${PY}" "${ARGS[@]}"
