#!/usr/bin/env bash
# TASK-066 ST-3.2f-ter-loc — MoE ffn_moe_out dual-topology localizer (patch-C TDD).
# Lives in the turboquant / llama.cpp tree (not linux_to_mac).
#
# Usage:
#   ./tests/run-task066-moe-out-corners.sh [OUT_DIR]
# Env:
#   BUILD_DIR — cmake build dir (default: <repo>/build)
#   REBUILD=1 — rebuild test-moe-out-row0-parity before run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-moe-out-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-moe-out-row0-parity"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  echo "Configure/build this fork first (cmake -B build …), then re-run." >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-moe-out-row0-parity"
  cmake --build "${BUILD_DIR}" --target test-moe-out-row0-parity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

echo "==> MOE-OUT C1..C5 (test-moe-out-row0-parity) permute+sum_rows vs view+add"
set +e
"${TEST_BIN}" 2>&1 | tee "${OUT_DIR}/moe-out-row0.log"
RC=${PIPESTATUS[0]}
set -e
echo "moe_out_row0_exit=${RC}" | tee "${OUT_DIR}/moe-out-row0.exit"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "moe_out_row0_exit=${RC}"
  if [[ "${RC}" -eq 0 ]]; then
    echo "VERDICT=GREEN"
    echo "note=GREEN unified view+ADD MoE-out row0 (patch C KEEP candidate on this host)"
  elif [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=HOST_SKIP"
  else
    echo "VERDICT=RED"
    echo "note=MoE ffn_moe_out row0 still diverges under unified path"
  fi
} | tee "${OUT_DIR}/summary.txt"

exit "${RC}"
