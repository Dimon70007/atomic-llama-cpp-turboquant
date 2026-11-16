#!/usr/bin/env bash
# TASK-073 ST-1: checkpoint restore must reproduce one emission.
#
# Sends the same greedy request N times to one llama-server with context
# checkpoints enabled. From the second request on the server restores a
# checkpoint instead of re-running prefill, so the draft driver has to be
# restored with it. Pre-fix the cell cycles between texts (RED).
#
# Delegates the measurement to tests/run-task072-groot-pmin-sweep.sh and only
# adds the verdict (tests/lib/task073_checkpoint_repeat_verdict.py).
#
# Usage:
#   TASK073_MODEL=... TASK073_SERVER=... ./tests/run-task073-mtp-checkpoint-repeat.sh
#
# Env (all required):
#   TASK073_MODEL     hybrid MTP GGUF (e.g. Groot-19B-A3B)
#   TASK073_SERVER    llama-server binary (dev build)
# Env (optional):
#   TASK073_REPEATS   requests in the cell, default 4
#   TASK073_CTX       default 16384
#   TASK073_MAX_TOKENS default 4096
#   TASK073_OUT       default <repo>/logs/task073-checkpoint-repeat
#   TASK073_SERVER_ARGS extra llama-server args, space separated
#                     (e.g. "--ctx-checkpoints 0" for the ST-3 control arm)
#
# Exit: 0 one emission, 1 more than one, 2 bad input or a failed cell,
#       77 the model is absent (ctest skip).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SWEEP="${REPO_ROOT}/tests/run-task072-groot-pmin-sweep.sh"
VERDICT="${REPO_ROOT}/tests/lib/task073_checkpoint_repeat_verdict.py"
CELL=n2-p090

MODEL="${TASK073_MODEL:-}"
SERVER="${TASK073_SERVER:-}"
[[ -n "${MODEL}" ]] || { echo "SKIP: TASK073_MODEL is not set" >&2; exit 77; }
[[ -f "${MODEL}" ]] || { echo "SKIP: TASK073_MODEL not found: ${MODEL}" >&2; exit 77; }
[[ -n "${SERVER}" ]] || { echo "FAIL: TASK073_SERVER is not set" >&2; exit 2; }
[[ -x "${SERVER}" ]] || { echo "FAIL: TASK073_SERVER not executable: ${SERVER}" >&2; exit 2; }

REPEATS="${TASK073_REPEATS:-4}"
CTX="${TASK073_CTX:-16384}"
MAX_TOKENS="${TASK073_MAX_TOKENS:-4096}"
OUT="${TASK073_OUT:-${REPO_ROOT}/logs/task073-checkpoint-repeat}"
SERVER_ARGS=()
if [[ -n "${TASK073_SERVER_ARGS:-}" ]]; then
  read -r -a SERVER_ARGS <<< "${TASK073_SERVER_ARGS}"
fi

[[ -f "${SWEEP}" ]] || { echo "FAIL: missing ${SWEEP}" >&2; exit 2; }
[[ -f "${VERDICT}" ]] || { echo "FAIL: missing ${VERDICT}" >&2; exit 2; }

mkdir -p "${OUT}"
TASK072_MODEL="${MODEL}" \
TASK072_SERVER="${SERVER}" \
TASK072_OUT="${OUT}" \
TASK072_CTX="${CTX}" \
TASK072_MAX_TOKENS="${MAX_TOKENS}" \
TASK072_REPEATS="${REPEATS}" \
TASK072_SERVER_ARGS="${TASK073_SERVER_ARGS:-}" \
ONLY="${CELL}" \
  bash "${SWEEP}"

STAMP_DIR="$(ls -1d "${OUT}"/*/ | sort | tail -1)"
STAMP_DIR="${STAMP_DIR%/}"
echo "stamp: ${STAMP_DIR}"

python3 "${VERDICT}" --cell-dir "${STAMP_DIR}/${CELL}" --min-repeats 3
