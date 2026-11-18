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
OUT="${TASK073_OUT:-${REPO_ROOT}/logs/task073.1-swa-checkpoints}"

# Server resolution: explicit selector wins, then the build tree ctest runs from
# (so a CUDA build works without a Metal dev tree), then `build-dev` as the last
# fallback for historic Darwin invocations. The old hardcoded `build-dev` default
# made this gate unrunnable on any host without one (TASK-077 F6/F10, ST-6.4).
#
# ctest sets CTEST_BINARY_DIRECTORY for tests it starts; use it when present so the
# runner finds the server next to the binaries that are actually under test.
_build_candidates=()
[[ -n "${TASK073_SERVER:-}" ]]           && _build_candidates+=("${TASK073_SERVER}")
[[ -n "${CTEST_BINARY_DIRECTORY:-}" ]]   && _build_candidates+=("${CTEST_BINARY_DIRECTORY}/bin/llama-server")
[[ -n "${BUILD_DIR:-}" ]]                && _build_candidates+=("${BUILD_DIR}/bin/llama-server")
_build_candidates+=(
  "${REPO_ROOT}/build/bin/llama-server"
  "${REPO_ROOT}/build-dev/bin/llama-server"
)
SERVER=""
for _cand in "${_build_candidates[@]}"; do
  if [[ -x "${_cand}" ]]; then
    SERVER="${_cand}"
    break
  fi
done
if [[ -z "${SERVER}" ]]; then
  echo "FAIL: server binary not found; tried: ${_build_candidates[*]}" >&2
  echo "      build it (cmake --build <dir> --target llama-server) or set TASK073_SERVER" >&2
  exit 2
fi

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

# Only a process whose executable IS llama-server counts. A bare `pgrep -f
# llama-server` matches anything that merely mentions the string in its command
# line (the agent runtime does), which produced a false "Metal is exclusive"
# warning that failed this gate without starting a server (TASK-077 F32). On CUDA
# the guard is informational anyway: GPU exclusivity is per device, not global.
if pgrep -x 'llama-server' >/dev/null 2>&1; then
  echo "WARN: a llama-server is already running; it may hold the GPU:" >&2
  echo "      stop it before running this gate (on Darwin: make minillm-stop)" >&2
  pgrep -xl 'llama-server' >&2 || true
fi

# shellcheck disable=SC2086 # ARGS is a deliberate flag list
exec python3 "${PY}" --json "${JSON}" --server "${SERVER}" --out-dir "${OUT}" \
  ${TASK0731_MODEL:+--model-override "${TASK0731_MODEL}"} ${ARGS:-}
