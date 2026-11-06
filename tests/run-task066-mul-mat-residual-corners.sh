#!/usr/bin/env bash
# TASK-066 follow-up B5 — Metal MUL_MAT residual batch widths (ne11 in 9/16/32).
# See tests/TASK-066-mul-mat-batch-inventory.md.
#
# Usage:
#   ./tests/run-task066-mul-mat-residual-corners.sh [OUT_DIR]
# Env: BUILD_DIR, REBUILD=1
# Gate: requires default GGML_METAL_MUL_MM_MIN=32 (unset or 32). Other values SKIP (77).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-mul-mat-residual-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-mul-mat-batch-invariance"

# B5 residual expects the product default pin. Debug ladder uses MUL_MM_MIN=8 separately.
if [[ -n "${GGML_METAL_MUL_MM_MIN:-}" && "${GGML_METAL_MUL_MM_MIN}" != "32" ]]; then
  {
    echo "track=B5"
    echo "inventory=tests/TASK-066-mul-mat-batch-inventory.md"
    echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN}"
    echo "VERDICT=SKIP (residual suite gates on default MUL_MM_MIN=32; unset env or set 32)"
  } | tee "${OUT_DIR}/summary.txt"
  echo "SKIP: SUITE=mul-mat-residual requires GGML_METAL_MUL_MM_MIN unset or 32 (got ${GGML_METAL_MUL_MM_MIN})" >&2
  exit 77
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-mul-mat-batch-invariance"
  cmake --build "${BUILD_DIR}" --target test-mul-mat-batch-invariance --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

if [[ ! -x "${TEST_BIN}" ]]; then
  echo "FAIL: missing ${TEST_BIN}" >&2
  exit 1
fi

echo "==> B5 residual (test-mul-mat-batch-invariance --residual)"
set +e
"${TEST_BIN}" --residual 2>&1 | tee "${OUT_DIR}/residual.log"
RC=${PIPESTATUS[0]}
set -e
echo "residual_exit=${RC}" | tee "${OUT_DIR}/residual.exit"

{
  echo "track=B5"
  echo "inventory=tests/TASK-066-mul-mat-batch-inventory.md"
  echo "residual_exit=${RC}"
  if [[ "${RC}" -eq 0 ]]; then
    echo "VERDICT=PASS"
  elif [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=SKIP (no Metal)"
  else
    echo "VERDICT=FAIL (B5 residual RED)"
  fi
} | tee "${OUT_DIR}/summary.txt"

if [[ "${RC}" -eq 77 ]]; then
  exit 77
fi
if [[ "${RC}" -ne 0 ]]; then
  exit "${RC}"
fi
echo "OK: TASK-066 mul-mat residual B5"
exit 0
