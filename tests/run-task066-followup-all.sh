#!/usr/bin/env bash
# TASK-066 follow-up — one-command runner for Track A + Track B gates.
#
# Usage (from this repo root):
#   ./tests/run-task066-followup-all.sh [OUT_DIR]
#   # same via umbrella:
#   SUITE=followup ./tests/run-task066-metal-corners.sh
#
# Env (forwarded to children):
#   BUILD_DIR, REBUILD=1
#   TASK066_B1_WAIVE=1              — accept B1 PARTIAL (fork is_mem_shared)
#   TASK066_ACC_TRACE_LOG=...       — required for B2 PASS (else B2 stays FAIL)
#   TASK066_ACC_TRACE_NOTE=...      — optional first-diverge note for B2
#   TASK066_MODEL + TASK066_CTX     — required to run B4 mtp-e2e; else B4 is SKIP
#   TASK066_FOLLOWUP_REQUIRE_E2E=1  — fail closed if MODEL/CTX missing (no B4 SKIP)
#   TASK066_FOLLOWUP_SKIP=csv       — skip named parts (e.g. mtp-acc-trace,mul-mat-residual)
#
# Runs every gate even if earlier ones fail; final exit is non-zero if any FAIL
# (host-skip exit 77 counts as SKIP, not FAIL).
#
# See: tests/TASK-066-cb-fusion-followup.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-followup-all-${STAMP}}"
mkdir -p "${OUT_DIR}"

export BUILD_DIR
export REBUILD="${REBUILD:-0}"

SUMMARY="${OUT_DIR}/summary.txt"
RESULTS="${OUT_DIR}/results.tsv"
: > "${RESULTS}"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "suite=followup"
  echo "ref=tests/TASK-066-cb-fusion-followup.md"
} | tee "${SUMMARY}"

should_skip() {
  local name="$1"
  local csv="${TASK066_FOLLOWUP_SKIP:-}"
  [[ -z "${csv}" ]] && return 1
  case ",${csv}," in
    *",${name},"*) return 0 ;;
    *) return 1 ;;
  esac
}

record() {
  local name="$1"
  local rc="$2"
  local verdict="$3"
  printf '%s\t%s\t%s\n' "${name}" "${rc}" "${verdict}" >> "${RESULTS}"
  echo "${name}_exit=${rc} verdict=${verdict}" | tee -a "${SUMMARY}"
}

run_part() {
  local name="$1"
  local script="$2"
  local sub="${OUT_DIR}/${name}"
  mkdir -p "${sub}"

  if should_skip "${name}"; then
    echo "==> SKIP ${name} (TASK066_FOLLOWUP_SKIP)"
    record "${name}" 0 "SKIP"
    return 0
  fi

  echo "==> RUN ${name}"
  set +e
  bash "${script}" "${sub}"
  local rc=$?
  set -e

  local verdict="FAIL"
  if [[ "${rc}" -eq 0 ]]; then
    verdict="PASS"
  elif [[ "${rc}" -eq 77 ]]; then
    verdict="SKIP"
  fi
  record "${name}" "${rc}" "${verdict}"
  return 0
}

# Track A
run_part cb-fusion "${REPO_ROOT}/tests/run-task066-cb-fusion-corners.sh"

# Track B
run_part b1-shared-kv "${REPO_ROOT}/tests/run-task066-b1-shared-kv-audit.sh"
run_part mtp-acc-trace "${REPO_ROOT}/tests/run-task066-mtp-acc-trace-checklist.sh"

if [[ -n "${TASK066_MODEL:-}" && -n "${TASK066_CTX:-}" ]]; then
  run_part mtp-e2e "${REPO_ROOT}/examples/metal-batch-invariance/run-mtp-e2e-corners.sh"
elif [[ "${TASK066_FOLLOWUP_REQUIRE_E2E:-0}" == "1" ]]; then
  echo "==> FAIL mtp-e2e (TASK066_FOLLOWUP_REQUIRE_E2E=1 but MODEL/CTX unset)"
  record "mtp-e2e" 2 "FAIL"
else
  echo "==> SKIP mtp-e2e (set TASK066_MODEL + TASK066_CTX; script under examples/metal-batch-invariance/)"
  record "mtp-e2e" 0 "SKIP"
fi

run_part mul-mat-residual "${REPO_ROOT}/tests/run-task066-mul-mat-residual-corners.sh"

echo "----" | tee -a "${SUMMARY}"
echo "results_tsv=${RESULTS}" | tee -a "${SUMMARY}"
if [[ -f "${RESULTS}" ]]; then
  column -t -s $'\t' "${RESULTS}" 2>/dev/null | tee -a "${SUMMARY}" || cat "${RESULTS}" | tee -a "${SUMMARY}"
fi

n_fail=0
n_pass=0
n_skip=0
while IFS=$'\t' read -r _name _rc verdict; do
  case "${verdict}" in
    PASS) n_pass=$((n_pass + 1)) ;;
    SKIP) n_skip=$((n_skip + 1)) ;;
    FAIL) n_fail=$((n_fail + 1)) ;;
  esac
done < "${RESULTS}"

{
  echo "n_pass=${n_pass}"
  echo "n_skip=${n_skip}"
  echo "n_fail=${n_fail}"
} | tee -a "${SUMMARY}"

if [[ "${n_fail}" -ne 0 ]]; then
  echo "VERDICT=FAIL (follow-up gates still RED; see TASK-066-cb-fusion-followup.md)" | tee -a "${SUMMARY}"
  exit 1
fi

echo "VERDICT=PASS" | tee -a "${SUMMARY}"
echo "OK: TASK-066 follow-up all gates green"
exit 0
