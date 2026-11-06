#!/usr/bin/env bash
# TASK-066 — umbrella Metal corner / localizer suite (self-contained in this fork).
#
# Usage (from this repo root):
#   SUITE=all ./tests/run-task066-metal-corners.sh [OUT_DIR]
#   SUITE=moe ./tests/run-task066-metal-corners.sh
# Env:
#   SUITE      — required (no silent default):
#                all|fa|mul-mat|longctx|moe|moe-router|moe-out|fullgraph|
#                cb-fusion|b1-shared-kv|mtp-acc-trace|mtp-e2e|mul-mat-residual|
#                ubatch-s8-s1024|followup
#                RED / follow-up suites are NOT part of `all`
#                followup = one-shot Track A+B (see run-task066-followup-all.sh)
#   BUILD_DIR  — cmake build dir (default: <repo>/build)
#   REBUILD=1  — rebuild suite binaries before run (forwarded to child runners)
#
# Third parties: configure Metal per docs/build.md, build, then run this script.
# This tree must not depend on any external product repo.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
SUITE="${SUITE:-}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-metal-corners-${SUITE:-unset}-${STAMP}}"

ALLOWED="all|fa|mul-mat|longctx|moe|moe-router|moe-out|fullgraph|cb-fusion|b1-shared-kv|mtp-acc-trace|mtp-e2e|mul-mat-residual|ubatch-s8-s1024|followup"

if [[ -z "${SUITE}" ]]; then
  echo "FAIL: missing required SUITE=${ALLOWED}" >&2
  echo "(refusing silent suite substitution)" >&2
  exit 1
fi

case "${SUITE}" in
  all|fa|mul-mat|longctx|moe|moe-router|moe-out|fullgraph|cb-fusion|b1-shared-kv|mtp-acc-trace|mtp-e2e|mul-mat-residual|ubatch-s8-s1024|followup) ;;
  *)
    echo "FAIL: unknown SUITE=${SUITE} (allowed: ${ALLOWED})" >&2
    exit 1
    ;;
esac

mkdir -p "${OUT_DIR}"
export BUILD_DIR
export REBUILD="${REBUILD:-0}"

run_one() {
  local name="$1"
  local script="$2"
  local sub="${OUT_DIR}/${name}"
  mkdir -p "${sub}"
  echo "==> SUITE part: ${name}"
  set +e
  bash "${script}" "${sub}"
  local rc=$?
  set -e
  echo "${name}_exit=${rc}" | tee "${OUT_DIR}/${name}.exit"
  if [[ "${rc}" -ne 0 ]]; then
    echo "FAIL: ${name} exit=${rc}" >&2
    echo "FIRST_FAIL=${name}" | tee -a "${OUT_DIR}/summary.txt"
    echo "VERDICT=FAIL" | tee -a "${OUT_DIR}/summary.txt"
    exit "${rc}"
  fi
}

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "suite=${SUITE}"
} | tee "${OUT_DIR}/summary.txt"

case "${SUITE}" in
  fa)
    run_one fa "${REPO_ROOT}/tests/run-task066-fa-nsg-corners.sh"
    ;;
  mul-mat)
    run_one mul-mat "${REPO_ROOT}/tests/run-task066-mul-mat-corners.sh"
    ;;
  longctx)
    run_one longctx "${REPO_ROOT}/tests/run-task066-longctx-corners.sh"
    ;;
  moe)
    run_one moe "${REPO_ROOT}/tests/run-task066-moe-corners.sh"
    ;;
  moe-router)
    run_one moe-router "${REPO_ROOT}/tests/run-task066-moe-router-corners.sh"
    ;;
  moe-out)
    run_one moe-out "${REPO_ROOT}/tests/run-task066-moe-out-corners.sh"
    ;;
  fullgraph)
    run_one fullgraph "${REPO_ROOT}/tests/run-task066-fullgraph-corners.sh"
    ;;
  cb-fusion)
    run_one cb-fusion "${REPO_ROOT}/tests/run-task066-cb-fusion-corners.sh"
    ;;
  b1-shared-kv)
    run_one b1-shared-kv "${REPO_ROOT}/tests/run-task066-b1-shared-kv-audit.sh"
    ;;
  mtp-acc-trace)
    run_one mtp-acc-trace "${REPO_ROOT}/tests/run-task066-mtp-acc-trace-checklist.sh"
    ;;
  mtp-e2e)
    run_one mtp-e2e "${REPO_ROOT}/tests/run-task066-mtp-e2e-corners.sh"
    ;;
  mul-mat-residual)
    run_one mul-mat-residual "${REPO_ROOT}/tests/run-task066-mul-mat-residual-corners.sh"
    ;;
  ubatch-s8-s1024)
    run_one ubatch-s8-s1024 "${REPO_ROOT}/tests/run-task066-ubatch-s8-s1024-corners.sh"
    ;;
  followup)
    # One-shot Track A+B; continues through failures and prints a TSV summary.
    exec bash "${REPO_ROOT}/tests/run-task066-followup-all.sh" "${OUT_DIR}"
    ;;
  all)
    run_one fa "${REPO_ROOT}/tests/run-task066-fa-nsg-corners.sh"
    run_one mul-mat "${REPO_ROOT}/tests/run-task066-mul-mat-corners.sh"
    run_one longctx "${REPO_ROOT}/tests/run-task066-longctx-corners.sh"
    run_one moe "${REPO_ROOT}/tests/run-task066-moe-corners.sh"
    run_one moe-router "${REPO_ROOT}/tests/run-task066-moe-router-corners.sh"
    # follow-up / expected-RED suites omitted from `all`
    ;;
esac

echo "VERDICT=PASS" | tee -a "${OUT_DIR}/summary.txt"
echo "OK: TASK-066 Metal corners SUITE=${SUITE}"
exit 0
