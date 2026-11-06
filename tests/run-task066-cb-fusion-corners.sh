#!/usr/bin/env bash
# TASK-066 follow-up — Metal CB fusion align unit suite (TDD RED until follow-up High/Medium).
#
# Usage (from this repo root):
#   ./tests/run-task066-cb-fusion-corners.sh [OUT_DIR]
# Env:
#   BUILD_DIR — cmake build dir (default: <repo>/build)
#   REBUILD=1 — rebuild test-metal-cb-fusion before run

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-cb-fusion-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-metal-cb-fusion"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  echo "Configure with -DGGML_METAL=ON (and without GGML_BACKEND_DL), then re-run." >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-metal-cb-fusion"
  cmake --build "${BUILD_DIR}" --target test-metal-cb-fusion --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

if [[ ! -x "${TEST_BIN}" ]]; then
  echo "FAIL: missing ${TEST_BIN} (need GGML_METAL=ON and NOT GGML_BACKEND_DL)" >&2
  exit 1
fi

echo "==> test-metal-cb-fusion (TASK-066-cb-fusion-followup RED gate)"
set +e
"${TEST_BIN}" 2>&1 | tee "${OUT_DIR}/cb-fusion.log"
RC=${PIPESTATUS[0]}
set -e
echo "cb_fusion_exit=${RC}" | tee "${OUT_DIR}/cb-fusion.exit"

if [[ "${RC}" -ne 0 ]]; then
  echo "VERDICT=FAIL (expected RED until follow-up High/Medium/Low)" | tee -a "${OUT_DIR}/summary.txt"
  exit "${RC}"
fi

echo "VERDICT=PASS" | tee "${OUT_DIR}/summary.txt"
echo "OK: TASK-066 cb-fusion corners"
exit 0
