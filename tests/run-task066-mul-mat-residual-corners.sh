#!/usr/bin/env bash
# TASK-066 — Metal MUL_MAT residual: ne11 in {9,16,32,256,512,1024} row0 vs golden N=8.
# See ../../tasks/TASK-066-mul-mat-batch-inventory.md / TASK-066.6.
#
# Usage:
#   ./tests/run-task066-mul-mat-residual-corners.sh [OUT_DIR]
# Env: BUILD_DIR, REBUILD=1
# Gate: upstream mul_mm break-even is now unconditional (ne11 > 8). The former
# GGML_METAL_MUL_MM_MIN env knob and its policy unit were deleted in TASK-071.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-mul-mat-residual-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-mul-mat-batch-invariance"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  exit 1
fi

# Darwin has no nproc; never fall through to a missing binary (exit 127).
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)"

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-mul-mat-batch-invariance"
  cmake --build "${BUILD_DIR}" --target test-mul-mat-batch-invariance --parallel "${NCPU}" \
    | tee "${OUT_DIR}/rebuild.log"
fi

if [[ ! -x "${TEST_BIN}" ]]; then
  echo "FAIL: missing ${TEST_BIN}" >&2
  exit 1
fi

echo "==> residual vs N=8 (test-mul-mat-batch-invariance --residual)"
set +e
"${TEST_BIN}" --residual 2>&1 | tee "${OUT_DIR}/residual.log"
RC=${PIPESTATUS[0]}
set -e
echo "residual_exit=${RC}" | tee "${OUT_DIR}/residual.exit"

echo "==> Phase B MTP verify N=1 vs {2,3,4} (test-mul-mat-batch-invariance --mtp-verify)"
set +e
"${TEST_BIN}" --mtp-verify 2>&1 | tee "${OUT_DIR}/mtp-verify.log"
MTP_RC=${PIPESTATUS[0]}
set -e
echo "mtp_verify_exit=${MTP_RC}" | tee "${OUT_DIR}/mtp-verify.exit"

{
  echo "track=B5+ladder-band+upstream-min8"
  echo "inventory=../../tasks/TASK-066-mul-mat-batch-inventory.md"
  echo "golden_n=8"
  echo "widths=9,16,32,256,512,1024"
  echo "mtp_verify_widths=2,3,4"
  echo "residual_exit=${RC}"
  echo "mtp_verify_exit=${MTP_RC}"
  # Upstream break-even (ne11 > 8): residual + mtp-verify are hard gates (no expected RED).
  if [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=SKIP (no Metal)"
  elif [[ "${RC}" -ne 0 ]]; then
    echo "VERDICT=FAIL (residual vs N=8 RED)"
  elif [[ "${MTP_RC}" -ne 0 && "${MTP_RC}" -ne 77 ]]; then
    echo "VERDICT=FAIL (mtp-verify RED — unexpected)"
  else
    echo "VERDICT=PASS"
  fi
} | tee "${OUT_DIR}/summary.txt"

if [[ "${RC}" -eq 77 ]]; then
  exit 77
fi
if [[ "${RC}" -ne 0 ]]; then
  exit "${RC}"
fi
if [[ "${MTP_RC}" -ne 0 && "${MTP_RC}" -ne 77 ]]; then
  exit "${MTP_RC}"
fi
echo "OK: TASK-066 mul-mat residual vs N=8 + mtp-verify (exit=${MTP_RC})"
exit 0
