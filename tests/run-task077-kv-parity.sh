#!/usr/bin/env bash
# TASK-077 ST-14: KV parity matrix inside the fork (no parent project needed).
#
# Serves one llama-server per cell with product flags, sends the same greedy
# request REPEATS times and requires the answers to be identical bit for bit.
# Cells: the five shipping V types at ctx_checkpoints 32, plus turbo3.8 and
# turbo4.8 at ctx_checkpoints 0 so the scheduler effect is separated from the
# codec (operator protocol 2026-11-18).
#
# Everything resolves inside this repository: the prompt is a local fixture and
# the server is taken from the tree ctest runs in. Nothing is read from
# ../linux_to_mac.
#
# Usage:
#   ./tests/run-task077-kv-parity.sh
#   TASK077_MODEL=/path/model.gguf ./tests/run-task077-kv-parity.sh
#   ARGS='--dry-run' ./tests/run-task077-kv-parity.sh
#
# Env (optional):
#   TASK077_MODEL   GGUF to serve (default ~/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf)
#   TASK077_SERVER  llama-server binary (default: resolved from the build tree)
#   TASK077_OUT     output root (default <repo>/logs/task077-kv-parity)
#   TASK077_CORPUS  prompt file (default <repo>/tests/fixtures/task066/...)
#   TASK077_V_TYPES comma list, default q8_0,turbo3,turbo3.8,turbo4,turbo4.8
#   ARGS            extra flags for the orchestrator (--dry-run, --v-types)
#
# Exit: 0 all runnable cells deterministic, 1 a cell drifted or crashed,
#       2 bad input, 77 model or server absent (ctest skip).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/lib/task077_kv_parity.py"
OUT="${TASK077_OUT:-${REPO_ROOT}/logs/task077-kv-parity}"

[[ -f "${PY}" ]] || { echo "FAIL: missing ${PY}" >&2; exit 2; }

# Server resolution: an explicit selector wins, then the tree ctest runs from,
# then the common local build dirs. Same ladder as run-task073.1-swa-checkpoint.sh.
_build_candidates=()
[[ -n "${TASK077_SERVER:-}" ]]         && _build_candidates+=("${TASK077_SERVER}")
[[ -n "${CTEST_BINARY_DIRECTORY:-}" ]] && _build_candidates+=("${CTEST_BINARY_DIRECTORY}/bin/llama-server")
[[ -n "${BUILD_DIR:-}" ]]              && _build_candidates+=("${BUILD_DIR}/bin/llama-server")
_build_candidates+=(
  "${REPO_ROOT}/build/bin/llama-server"
  "${REPO_ROOT}/build-cuda/bin/llama-server"
  "${REPO_ROOT}/build-dev/bin/llama-server"
)
# Any build-* tree that actually contains a server, newest first. Hosts here use
# task-specific names (build-task069-cuda-on), so a fixed list is not enough.
while IFS= read -r _cand; do
  _build_candidates+=("${_cand}")
done < <(ls -1dt "${REPO_ROOT}"/build-*/bin/llama-server 2>/dev/null || true)
SERVER=""
for _cand in "${_build_candidates[@]}"; do
  if [[ -x "${_cand}" ]]; then
    SERVER="${_cand}"
    break
  fi
done

MODEL="${TASK077_MODEL:-${HOME}/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf}"

# ctest runs this on hosts without the model: skip, do not fail.
if [[ ! -f "${MODEL}" ]]; then
  echo "SKIP: model not found: ${MODEL} (set TASK077_MODEL)" >&2
  exit 77
fi
if [[ -z "${SERVER}" ]]; then
  # The Python side reports the skip for a missing binary, but only when it is
  # given a path. Report it here so ctest sees 77 rather than 2.
  echo "SKIP: llama-server not found; tried: ${_build_candidates[*]}" >&2
  echo "      build it: cmake --build <dir> --target llama-server" >&2
  exit 77
fi

# Only a real llama-server process holds the GPU. A `pgrep -f` match also hits
# unrelated processes that mention the string (TASK-077 F32).
if pgrep -x 'llama-server' >/dev/null 2>&1; then
  echo "WARN: a llama-server is already running; it may hold the GPU:" >&2
  pgrep -xl 'llama-server' >&2 || true
fi

# shellcheck disable=SC2086 # ARGS is a deliberate flag list
exec python3 "${PY}" \
  --server "${SERVER}" \
  --model "${MODEL}" \
  --out-dir "${OUT}" \
  ${TASK077_CORPUS:+--corpus "${TASK077_CORPUS}"} \
  ${TASK077_V_TYPES:+--v-types "${TASK077_V_TYPES}"} \
  ${ARGS:-}
