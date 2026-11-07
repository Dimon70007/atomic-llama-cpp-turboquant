#!/usr/bin/env bash
# TASK-066 - long-KV FA row0 localizer (patch-C TDD gate for FA hypothesis).
# Lives in the turboquant / llama.cpp tree (not linux_to_mac).
#
# Usage:
#   ./tests/run-task066-longctx-corners.sh [OUT_DIR]
# Env:
#   BUILD_DIR - cmake build dir (default: <repo>/build)
#   REBUILD=1 - rebuild test-longctx-row0-parity before run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-longctx-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-longctx-row0-parity"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  echo "Configure/build this fork first (cmake -B build ...), then re-run." >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-longctx-row0-parity"
  cmake --build "${BUILD_DIR}" --target test-longctx-row0-parity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

echo "==> LONG-C1 + LONG-C2 (test-longctx-row0-parity) product q8_0/turbo4"
set +e
"${TEST_BIN}" 2>&1 | tee "${OUT_DIR}/longctx-fa.log"
RC=${PIPESTATUS[0]}
set -e
echo "longctx_fa_exit=${RC}" | tee "${OUT_DIR}/longctx-fa.exit"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "longctx_fa_exit=${RC}"
  if [[ "${RC}" -eq 0 ]]; then
    echo "VERDICT=GREEN"
    echo "note=GREEN on this host without new FA patch; CUDA re-run still required"
  elif [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=HOST_SKIP"
  else
    echo "VERDICT=RED"
    echo "note=FA long-KV patch-C candidate on this backend (TDD red gate)"
  fi
} | tee "${OUT_DIR}/summary.txt"

exit "${RC}"
