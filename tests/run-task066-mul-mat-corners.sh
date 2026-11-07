#!/usr/bin/env bash
# TASK-066 — Metal mul_mat / mul_mat_id corner suite (C1/C4/C5; C6 smoke).
# Lives in the turboquant / llama.cpp tree (not linux_to_mac).
#
# Usage:
#   ./tests/run-task066-mul-mat-corners.sh [OUT_DIR]
# Env:
#   BUILD_DIR, REBUILD=1, BACKEND_FILTER (default MTL0)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-mul-mat-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-mul-mat-batch-invariance"
OPS_BIN="${BUILD_DIR}/bin/test-backend-ops"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-mul-mat-batch-invariance"
  cmake --build "${BUILD_DIR}" --target test-mul-mat-batch-invariance --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild-corners.log"
fi

if [[ ! -x "${OPS_BIN}" ]]; then
  echo "==> rebuild test-backend-ops (C6)"
  cmake --build "${BUILD_DIR}" --target test-backend-ops --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee -a "${OUT_DIR}/rebuild-corners.log"
fi

echo "==> C1+C4+C5 (test-mul-mat-batch-invariance)"
set +e
"${TEST_BIN}" 2>&1 | tee "${OUT_DIR}/c1c4c5.log"
RC_CORNERS=${PIPESTATUS[0]}
set -e
echo "corners_exit=${RC_CORNERS}" | tee "${OUT_DIR}/c1c4c5.exit"

BACKEND_FILTER="${BACKEND_FILTER:-MTL0}"

echo "==> C6 smoke: MUL_MAT q8_0 subset -b ${BACKEND_FILTER}"
set +e
"${OPS_BIN}" -b "${BACKEND_FILTER}" -o MUL_MAT -p 'type_a=q8_0' 2>&1 | tee "${OUT_DIR}/c6-mul-mat-q8.log"
RC_MM=${PIPESTATUS[0]}
set -e
echo "c6_mul_mat_exit=${RC_MM}" | tee "${OUT_DIR}/c6-mul-mat-q8.exit"
if ! grep -q "tests passed" "${OUT_DIR}/c6-mul-mat-q8.log"; then
  echo "FAIL: C6 MUL_MAT q8_0 produced no 'tests passed'" >&2
  RC_MM=1
fi

echo "==> C6 smoke: unrelated RMS_NORM -b ${BACKEND_FILTER}"
set +e
"${OPS_BIN}" -b "${BACKEND_FILTER}" -o RMS_NORM 2>&1 | tee "${OUT_DIR}/c6-rms-norm.log"
RC_NORM=${PIPESTATUS[0]}
set -e
echo "c6_rms_norm_exit=${RC_NORM}" | tee "${OUT_DIR}/c6-rms-norm.exit"
if ! grep -q "tests passed" "${OUT_DIR}/c6-rms-norm.log"; then
  echo "FAIL: C6 RMS_NORM produced no 'tests passed'" >&2
  RC_NORM=1
fi

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "c1c4c5_exit=${RC_CORNERS}"
  echo "c6_mul_mat_q8_exit=${RC_MM}"
  echo "c6_rms_norm_exit=${RC_NORM}"
  if [[ "${RC_CORNERS}" -eq 0 && "${RC_MM}" -eq 0 && "${RC_NORM}" -eq 0 ]]; then
    echo "VERDICT=PASS"
  else
    echo "VERDICT=FAIL"
  fi
} | tee "${OUT_DIR}/summary.txt"

if [[ "${RC_CORNERS}" -ne 0 || "${RC_MM}" -ne 0 || "${RC_NORM}" -ne 0 ]]; then
  exit 1
fi
echo "OK: TASK-066 mul_mat corners C1/C4/C5 + C6 green"
