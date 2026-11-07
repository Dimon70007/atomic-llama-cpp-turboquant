#!/usr/bin/env bash
# TASK-066 ST-3.2d - bisect router localizer crash / RED without sudo.
# Self-contained in this llama.cpp fork. Run from a normal terminal (Metal needs GPU access).
#
# Usage (from fork root, after cmake -B build && cmake --build build --target test-moe-router-row0-parity):
#   ./tests/run-task066-moe-router-bisect.sh
#   BUILD_DIR=/path/to/build ./tests/run-task066-moe-router-bisect.sh [OUT_DIR]
#
# Stages: logits -> probs -> ids -> moe. Stops at first non-zero exit (crash->139, FAIL->1, skip->77).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-moe-router-bisect-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-moe-router-row0-parity"

if [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-moe-router-row0-parity"
  cmake --build "${BUILD_DIR}" --target test-moe-router-row0-parity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

SUMMARY="${OUT_DIR}/bisect-summary.txt"
: > "${SUMMARY}"
{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "test_bin=${TEST_BIN}"
} | tee -a "${SUMMARY}"

FINAL_RC=0
for stage in logits probs ids moe; do
  echo "==> STAGE=${stage}"
  set +e
  TASK066_ROUTER_STAGE="${stage}" "${TEST_BIN}" >"${OUT_DIR}/stage-${stage}.log" 2>&1
  RC=$?
  set -e
  echo "stage_${stage}_exit=${RC}" | tee -a "${SUMMARY}"
  tail -20 "${OUT_DIR}/stage-${stage}.log" || true
  if [[ "${RC}" -ne 0 ]]; then
    FINAL_RC="${RC}"
    echo "FIRST_FAIL_STAGE=${stage}" | tee -a "${SUMMARY}"
    if [[ "${RC}" -eq 139 ]] || [[ "${RC}" -eq 134 ]]; then
      echo "VERDICT=CRASH" | tee -a "${SUMMARY}"
    elif [[ "${RC}" -eq 77 ]]; then
      echo "VERDICT=HOST_SKIP" | tee -a "${SUMMARY}"
    else
      echo "VERDICT=RED" | tee -a "${SUMMARY}"
    fi
    exit "${RC}"
  fi
done

echo "VERDICT=GREEN" | tee -a "${SUMMARY}"
echo "note=all stages PASS; router->top_k->moe row0 bit-exact on this host"
exit 0
