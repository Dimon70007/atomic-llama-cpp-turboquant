#!/usr/bin/env bash
# TASK-066 ST-3.2f — full-graph decode row0 localizer (patch-C TDD RED gate).
# Lives in the turboquant / llama.cpp tree (not linux_to_mac).
#
# Usage:
#   TASK066_MODEL=/path/to/model.gguf ./tests/run-task066-fullgraph-corners.sh [OUT_DIR]
#   TASK066_MODEL=… TASK066_PROMPT=/path/to/prompt.txt ./tests/run-task066-fullgraph-corners.sh
# Env:
#   TASK066_MODEL   — required GGUF path (fail-closed; no silent substitute)
#   TASK066_PROMPT  — optional prompt file (first-diff / Voyna text)
#   TASK066_N_TOKENS / TASK066_PROMPT_CHARS — optional truncates (forwarded)
#   TASK066_REQUIRE_GPU=1 — exit 77 if no GPU (default 1 on Darwin)
#   BUILD_DIR — cmake build dir (default: <repo>/build)
#   REBUILD=1 — rebuild binary before run
#   EXTRA_ARGS — extra argv for the test (e.g. '-c 16384 -fa on')

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-fullgraph-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-fullgraph-decode-row0-parity"
MODEL="${TASK066_MODEL:-}"

if [[ -z "${MODEL}" ]]; then
  echo "FAIL: missing required TASK066_MODEL=/path/to.gguf (refusing silent model substitution)" >&2
  exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "FAIL: TASK066_MODEL not a file: ${MODEL}" >&2
  exit 2
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-fullgraph-decode-row0-parity"
  cmake --build "${BUILD_DIR}" --target test-fullgraph-decode-row0-parity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

ARGS=(-m "${MODEL}" -ngl 999)
if [[ -n "${TASK066_PROMPT:-}" ]]; then
  if [[ ! -f "${TASK066_PROMPT}" ]]; then
    echo "FAIL: TASK066_PROMPT not a file: ${TASK066_PROMPT}" >&2
    exit 2
  fi
  ARGS+=(-f "${TASK066_PROMPT}")
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )
ARGS+=("${EXTRA[@]}")

if [[ -z "${TASK066_REQUIRE_GPU:-}" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
  export TASK066_REQUIRE_GPU=1
fi

echo "==> FULLGRAPH-C1 (test-fullgraph-decode-row0-parity)"
echo "    model=${MODEL}"
echo "    prompt=${TASK066_PROMPT:-<synthesized>}"
set +e
"${TEST_BIN}" "${ARGS[@]}" 2>&1 | tee "${OUT_DIR}/fullgraph-row0.log"
RC=${PIPESTATUS[0]}
set -e
echo "fullgraph_row0_exit=${RC}" | tee "${OUT_DIR}/fullgraph-row0.exit"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "model=${MODEL}"
  echo "prompt=${TASK066_PROMPT:-}"
  echo "fullgraph_row0_exit=${RC}"
  if [[ "${RC}" -eq 0 ]]; then
    echo "VERDICT=GREEN"
    echo "note=GREEN on this prompt; live first-diff/P5 may still MISMATCH — try longer prompt-file"
  elif [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=HOST_SKIP"
  else
    echo "VERDICT=RED"
    echo "note=full-graph row0 logits not bit-exact under N=2 verify width (patch-C TDD RED)"
  fi
} | tee "${OUT_DIR}/summary.txt"

exit "${RC}"
