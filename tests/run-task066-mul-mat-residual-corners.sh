#!/usr/bin/env bash
# TASK-066 — Metal MUL_MAT residual: ne11 in {9,16,32,256,512,1024} row0 vs golden N=8.
# See tests/TASK-066-mul-mat-batch-inventory.md / TASK-066.6.
#
# Usage:
#   ./tests/run-task066-mul-mat-residual-corners.sh [OUT_DIR]
# Env: BUILD_DIR, REBUILD=1
# Gate: product default GGML_METAL_MUL_MM_MIN=1 (066.9 A2; unset or 1). Other values SKIP (77).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-mul-mat-residual-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-mul-mat-batch-invariance"
POLICY_BIN="${BUILD_DIR}/bin/test-metal-mul-mm-policy"

# 066.7 residual vs N=8; 066.9 A2 product default min=1 (ne11>=2 → float-tile mul_mm).
if [[ -n "${GGML_METAL_MUL_MM_MIN:-}" && "${GGML_METAL_MUL_MM_MIN}" != "1" ]]; then
  {
    echo "track=B5+066.7+066.9A2"
    echo "inventory=tests/TASK-066-mul-mat-batch-inventory.md"
    echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN}"
    echo "VERDICT=SKIP (residual suite gates on default MUL_MM_MIN=1; unset env or set 1)"
  } | tee "${OUT_DIR}/summary.txt"
  echo "SKIP: SUITE=mul-mat-residual requires GGML_METAL_MUL_MM_MIN unset or 1 (got ${GGML_METAL_MUL_MM_MIN})" >&2
  exit 77
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  exit 1
fi

NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"

# 066.9 A2 — CPU policy unit first (no Metal). Catches DEFAULT=7 / rem hazard on merge.
if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${POLICY_BIN}" ]]; then
  echo "==> rebuild test-metal-mul-mm-policy"
  cmake --build "${BUILD_DIR}" --target test-metal-mul-mm-policy --parallel "${NCPU}" \
    | tee "${OUT_DIR}/rebuild-policy.log"
fi
if [[ ! -x "${POLICY_BIN}" ]]; then
  echo "FAIL: missing ${POLICY_BIN}" >&2
  exit 1
fi
echo "==> mul_mm policy unit (test-metal-mul-mm-policy)"
set +e
"${POLICY_BIN}" 2>&1 | tee "${OUT_DIR}/policy.log"
POLICY_RC=${PIPESTATUS[0]}
set -e
echo "policy_exit=${POLICY_RC}" | tee "${OUT_DIR}/policy.exit"
if [[ "${POLICY_RC}" -ne 0 ]]; then
  {
    echo "track=B5+066.7+066.9A2"
    echo "policy_exit=${POLICY_RC}"
    echo "VERDICT=FAIL (mul_mm policy unit RED — check GGML_METAL_MUL_MM_MIN_DEFAULT / rem hazard)"
  } | tee "${OUT_DIR}/summary.txt"
  echo "FAIL: mul_mm policy unit RED (see ${OUT_DIR}/policy.log)" >&2
  exit "${POLICY_RC}"
fi

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

{
  echo "track=B5+ladder-band+066.9A2-policy"
  echo "inventory=tests/TASK-066-mul-mat-batch-inventory.md"
  echo "golden_n=8"
  echo "widths=9,16,32,256,512,1024"
  echo "policy_exit=${POLICY_RC}"
  echo "residual_exit=${RC}"
  if [[ "${RC}" -eq 0 ]]; then
    echo "VERDICT=PASS"
  elif [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=SKIP (no Metal)"
  else
    echo "VERDICT=FAIL (residual vs N=8 RED)"
  fi
} | tee "${OUT_DIR}/summary.txt"

if [[ "${RC}" -eq 77 ]]; then
  exit 77
fi
if [[ "${RC}" -ne 0 ]]; then
  exit "${RC}"
fi
echo "OK: TASK-066 mul-mat residual vs N=8 (+ mul_mm policy unit)"
exit 0
