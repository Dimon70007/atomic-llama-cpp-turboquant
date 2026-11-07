#!/usr/bin/env bash
# TASK-066 — Metal FA nsg corner suite (C1/C2/C4 in-binary; C5 N/A; C6 smoke).
# Lives in the turboquant / llama.cpp tree (not linux_to_mac).
#
# Usage (from this repo root or any cwd):
#   ./tests/run-task066-fa-nsg-corners.sh [OUT_DIR]
# Env:
#   BUILD_DIR   — cmake build dir (default: <repo>/build)
#   REBUILD=1   — rebuild test-flash-attn-batch-invariance before run
#   BACKEND_FILTER — test-backend-ops -b (default: MTL0 on Darwin Metal)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-fa-nsg-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-flash-attn-batch-invariance"
OPS_BIN="${BUILD_DIR}/bin/test-backend-ops"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  echo "Configure/build this fork first (cmake -B build …), then re-run." >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-flash-attn-batch-invariance"
  cmake --build "${BUILD_DIR}" --target test-flash-attn-batch-invariance --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild-corners.log"
fi

if [[ ! -x "${OPS_BIN}" ]]; then
  echo "==> rebuild test-backend-ops (C6)"
  cmake --build "${BUILD_DIR}" --target test-backend-ops --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee -a "${OUT_DIR}/rebuild-corners.log"
fi

echo "==> C1+C2+C4 (test-flash-attn-batch-invariance); C5=N/A in-binary"
set +e
"${TEST_BIN}" 2>&1 | tee "${OUT_DIR}/c1c2c4.log"
RC_CORNERS=${PIPESTATUS[0]}
set -e
echo "corners_exit=${RC_CORNERS}" | tee "${OUT_DIR}/c1c2c4.exit"

# Backend id on Darwin Metal is MTL0 (not the string "Metal").
BACKEND_FILTER="${BACKEND_FILTER:-MTL0}"

echo "==> C6 smoke: FLASH_ATTN_EXT product turbo4_vec cases (related regression) -b ${BACKEND_FILTER}"
set +e
"${OPS_BIN}" -b "${BACKEND_FILTER}" -o FLASH_ATTN_EXT -p 'turbo4_vec' 2>&1 | tee "${OUT_DIR}/c6-flash-attn-turbo4.log"
RC_FA=${PIPESTATUS[0]}
set -e
echo "c6_fa_exit=${RC_FA}" | tee "${OUT_DIR}/c6-flash-attn-turbo4.exit"
if ! grep -q "tests passed" "${OUT_DIR}/c6-flash-attn-turbo4.log"; then
  echo "FAIL: C6 FLASH_ATTN turbo4_vec produced no 'tests passed' (likely wrong -b filter / all skipped)" >&2
  RC_FA=1
fi

echo "==> C6 smoke: unrelated op RMS_NORM (must stay green after FA pin) -b ${BACKEND_FILTER}"
set +e
"${OPS_BIN}" -b "${BACKEND_FILTER}" -o RMS_NORM 2>&1 | tee "${OUT_DIR}/c6-rms-norm.log"
RC_NORM=${PIPESTATUS[0]}
set -e
echo "c6_rms_norm_exit=${RC_NORM}" | tee "${OUT_DIR}/c6-rms-norm.exit"
if ! grep -q "tests passed" "${OUT_DIR}/c6-rms-norm.log"; then
  echo "FAIL: C6 RMS_NORM produced no 'tests passed' (likely wrong -b filter / all skipped)" >&2
  RC_NORM=1
fi

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "c1c2c4_exit=${RC_CORNERS}"
  echo "c5=N/A (FA nsg pin does not touch mul_mat_id; ST-2.1b)"
  echo "c6_flash_attn_turbo4_exit=${RC_FA}"
  echo "c6_rms_norm_exit=${RC_NORM}"
  if [[ "${RC_CORNERS}" -eq 0 && "${RC_FA}" -eq 0 && "${RC_NORM}" -eq 0 ]]; then
    echo "VERDICT=PASS"
  else
    echo "VERDICT=FAIL"
  fi
} | tee "${OUT_DIR}/summary.txt"

if [[ "${RC_CORNERS}" -ne 0 || "${RC_FA}" -ne 0 || "${RC_NORM}" -ne 0 ]]; then
  exit 1
fi
echo "OK: TASK-066 FA nsg corners C1/C2/C4 + C6 green (C5 N/A)"
