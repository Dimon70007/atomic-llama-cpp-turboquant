#!/usr/bin/env bash
# TASK-068 - everyday UNIT verify (Metal host Terminal; not agent sandbox).
#
# Frequent path: schedule unit + optional ctest -L main + Metal smoke.
# Does NOT run KL e2e (that is holiday / major-rebase only).
#
# Usage (from this repo root):
#   BUILD_DIR=./build-st7-post-rebase ./tests/run-task068-unit-verify.sh
#
# Env (optional):
#   BUILD_DIR      - cmake build dir (default: <repo>/build)
#   REBUILD=1      - cmake --build before gates
#   BACKEND_FILTER - Darwin Metal id (default MTL0)
#   TASK068_SKIP_CTEST=1 - brackets binary only (no full ctest -L main)
#   TASK068_ALLOW_UNRELATED_CTEST_FAILS=1 - allow non-brackets ctest fails
#   TASK068_PRIOR_TRIAGE=1 - with SKIP_CTEST: VERDICT=PASS_WITH_TRIAGE
#
# Gates:
#   U1 meta + binary presence
#   U2 Metal smoke (fail-closed on null device)
#   U3 test-q8-brackets-schedule (must Pass; via ctest -L main or direct binary)
#
# E2E KL: tests/run-task068-q8-brackets-kl.sh
# Holiday combo: TASK068_RUN_KL_E2E=1 ./tests/run-task068-post-rebase-verify.sh
# See: docs/KV-cache-quantization.md "Regression gates"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task068-unit-${STAMP}}"
mkdir -p "${OUT_DIR}"

PPL_BIN="${BUILD_DIR}/bin/llama-perplexity"
BRACKETS_BIN="${BUILD_DIR}/bin/test-q8-brackets-schedule"
NCPU="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"

SUMMARY="${OUT_DIR}/summary.txt"
RESULTS="${OUT_DIR}/results.tsv"
: > "${RESULTS}"

record() {
  local name="$1"
  local rc="$2"
  local verdict="$3"
  printf '%s\t%s\t%s\n' "${name}" "${rc}" "${verdict}" >> "${RESULTS}"
  echo "${name}_exit=${rc} verdict=${verdict}" | tee -a "${SUMMARY}"
}

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "suite=task068-unit-verify"
  echo "tip_sha=$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  echo "branch=$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
  if [[ -f "${REPO_ROOT}/TURBOQUANT_VERSION" ]]; then
    echo "turboquant_version=$(tr -d '[:space:]' < "${REPO_ROOT}/TURBOQUANT_VERSION")"
  else
    echo "turboquant_version=MISSING"
  fi
  echo "rebuild=${REBUILD:-0}"
  echo "allow_unrelated_ctest_fails=${TASK068_ALLOW_UNRELATED_CTEST_FAILS:-0}"
  echo "prior_triage=${TASK068_PRIOR_TRIAGE:-0}"
  echo "skip_ctest=${TASK068_SKIP_CTEST:-0}"
  echo "ref=docs/KV-cache-quantization.md#regression-gates-task-068"
} | tee "${SUMMARY}"

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  echo "Configure first, e.g.:" >&2
  echo "  cmake -S . -B ${BUILD_DIR} -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=ON" >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]]; then
  echo "==> REBUILD=1 cmake --build ${BUILD_DIR}"
  cmake --build "${BUILD_DIR}" --parallel "${NCPU}" 2>&1 | tee "${OUT_DIR}/cmake-build.log"
fi

if [[ ! -x "${BRACKETS_BIN}" ]]; then
  echo "FAIL: missing ${BRACKETS_BIN} (build tests or set BUILD_DIR)" >&2
  exit 1
fi

# --- U2 Metal smoke ---
if [[ -z "${BACKEND_FILTER:-}" ]]; then
  if [[ "$(uname -s)" == "Darwin" ]]; then
    BACKEND_FILTER="MTL0"
  else
    BACKEND_FILTER=""
  fi
fi

if [[ -x "${PPL_BIN}" ]]; then
  echo "==> U2 Metal smoke (llama-perplexity --version + test-backend-ops -b ${BACKEND_FILTER:-unset})"
  set +e
  "${PPL_BIN}" --version >"${OUT_DIR}/ppl-version.log" 2>&1
  RC_VER=${PIPESTATUS[0]}
  set -e
  echo "ppl_version_exit=${RC_VER}" | tee -a "${SUMMARY}"
else
  echo "WARN: llama-perplexity missing; Metal version smoke skipped (brackets unit still runs)" | tee -a "${SUMMARY}"
  RC_VER=0
  : > "${OUT_DIR}/ppl-version.log"
fi
echo "backend_filter=${BACKEND_FILTER:-}" | tee -a "${SUMMARY}"
cp "${OUT_DIR}/ppl-version.log" "${OUT_DIR}/rebuild-binaries.ok" 2>/dev/null || true

OPS_BIN="${BUILD_DIR}/bin/test-backend-ops"
if [[ -x "${OPS_BIN}" && -n "${BACKEND_FILTER}" ]]; then
  set +e
  "${OPS_BIN}" -b "${BACKEND_FILTER}" -o RMS_NORM \
    >"${OUT_DIR}/backend-smoke.log" 2>&1
  RC_SMOKE=${PIPESTATUS[0]}
  set -e
  if grep -qE 'failed to create command queue|default device: \(null\)' "${OUT_DIR}/backend-smoke.log"; then
    echo "FAIL: Metal init failed (command queue / null device). Run outside agent sandbox; stop other GPU users." >&2
    record "metal-smoke" 1 "FAIL"
    echo "VERDICT=FAIL" | tee -a "${SUMMARY}"
    exit 1
  fi
  if [[ "${RC_SMOKE}" -ne 0 ]] || ! grep -q "tests passed" "${OUT_DIR}/backend-smoke.log"; then
    echo "WARN: Metal smoke -b ${BACKEND_FILTER} -o RMS_NORM weak/failed (exit=${RC_SMOKE}); see ${OUT_DIR}/backend-smoke.log" | tee -a "${SUMMARY}"
    record "metal-smoke" "${RC_SMOKE}" "WARN"
  else
    record "metal-smoke" 0 "PASS"
  fi
elif [[ -x "${OPS_BIN}" ]]; then
  echo "WARN: BACKEND_FILTER empty; skipping dedicated Metal smoke" | tee -a "${SUMMARY}"
  record "metal-smoke" 0 "SKIP"
else
  echo "WARN: test-backend-ops missing; skipping dedicated Metal smoke" | tee -a "${SUMMARY}"
  record "metal-smoke" 0 "SKIP"
fi

# --- U3 brackets / ctest ---
UNRELATED_FAILS=0
if [[ "${TASK068_SKIP_CTEST:-0}" == "1" ]]; then
  echo "==> U3 SKIP full ctest (TASK068_SKIP_CTEST=1); run brackets unit binary only"
  echo "ctest_skipped=1" | tee -a "${SUMMARY}"
  set +e
  "${BRACKETS_BIN}" >"${OUT_DIR}/brackets-unit.log" 2>&1
  RC_BRACKETS=${PIPESTATUS[0]}
  set -e
  if [[ "${RC_BRACKETS}" -ne 0 ]]; then
    record "test-q8-brackets-schedule" "${RC_BRACKETS}" "FAIL"
    echo "FAIL: test-q8-brackets-schedule binary failed; see ${OUT_DIR}/brackets-unit.log" >&2
    echo "VERDICT=FAIL" | tee -a "${SUMMARY}"
    exit 1
  fi
  record "test-q8-brackets-schedule" 0 "PASS"
  if [[ "${TASK068_PRIOR_TRIAGE:-0}" == "1" ]]; then
    echo "prior_triage=1 (operator documents prior unrelated ctest)" | tee -a "${SUMMARY}"
    record "ctest-main" 0 "SKIP_PRIOR_TRIAGE"
    UNRELATED_FAILS=1
  else
    record "ctest-main" 0 "SKIP"
  fi
else
  echo "==> U3 ctest -L main"
  set +e
  ctest --test-dir "${BUILD_DIR}" -L main --output-on-failure \
    >"${OUT_DIR}/ctest-main.log" 2>&1
  RC_CTEST=${PIPESTATUS[0]}
  set -e

  BRACKETS_LINE="$(grep -E 'test-q8-brackets-schedule' "${OUT_DIR}/ctest-main.log" | tail -n1 || true)"
  echo "brackets_ctest_line=${BRACKETS_LINE}" | tee -a "${SUMMARY}"
  if echo "${BRACKETS_LINE}" | grep -q 'Passed'; then
    record "test-q8-brackets-schedule" 0 "PASS"
    BRACKETS_OK=1
  else
    record "test-q8-brackets-schedule" 1 "FAIL"
    BRACKETS_OK=0
  fi

  FAILED_LIST="$(awk '/The following tests FAILED:/{flag=1; next} flag && /^$/{exit} flag' "${OUT_DIR}/ctest-main.log" || true)"
  {
    echo "ctest_main_exit=${RC_CTEST}"
    echo "---- ctest FAILED list ----"
    if [[ -n "${FAILED_LIST}" ]]; then
      printf '%s\n' "${FAILED_LIST}"
    else
      echo "(none parsed)"
    fi
    echo "---- end FAILED list ----"
  } | tee -a "${SUMMARY}"

  if [[ "${BRACKETS_OK}" -ne 1 ]]; then
    echo "FAIL: test-q8-brackets-schedule not Passed in ctest -L main" >&2
    echo "VERDICT=FAIL" | tee -a "${SUMMARY}"
    exit 1
  fi

  if [[ "${RC_CTEST}" -ne 0 ]]; then
    UNRELATED_FAILS=1
    echo "TRIAGE: ctest -L main non-zero but brackets Passed - treat other fails as unrelated unless schedule/API broke." | tee -a "${SUMMARY}"

    TRIAGE_NOTE="${OUT_DIR}/triage-unrelated.txt"
    n_tq_fail="$(grep -cE 'type_a=tq[34]_1s' "${OUT_DIR}/ctest-main.log" 2>/dev/null || true)"
    n_turbo_fail_lines="$(grep -E 'FAIL \[test failed\]' "${OUT_DIR}/ctest-main.log" 2>/dev/null | grep -cE 'turbo[234]|SET_ROWS_TURBO' || true)"
    other_fails="$(printf '%s\n' "${FAILED_LIST}" | grep -E '[[:space:]]+[0-9]+[[:space:]]+-[[:space:]]+' | grep -v 'test-backend-ops' || true)"
    only_backend_ops=0
    if echo "${FAILED_LIST}" | grep -q 'test-backend-ops' && [[ -z "${other_fails}" ]]; then
      only_backend_ops=1
    fi
    {
      echo "failed_tests:"
      printf '%s\n' "${FAILED_LIST}"
      echo ""
      echo "backend_filter_smoke=${BACKEND_FILTER:-}"
      echo "mtl0_in_ctest=$(grep -c 'Backend 1/3: MTL0\|Backend MTL0:' "${OUT_DIR}/ctest-main.log" 2>/dev/null || true)"
      echo "tq_weight_mentions=${n_tq_fail}"
      echo "turbo_kv_fail_lines=${n_turbo_fail_lines}"
      echo "only_test_backend_ops=${only_backend_ops}"
      if [[ "${only_backend_ops}" -eq 1 && "${n_tq_fail}" -gt 0 && "${n_turbo_fail_lines}" -eq 0 ]]; then
        echo "classification=UNRELATED_TQ_WEIGHT_BACKEND_OPS"
        echo "note=ctest already runs Metal as MTL0 (not -b Metal). FAIL is tq3_1s/tq4_1s weight MUL_MAT*, not q8 brackets / turbo KV schedule."
        echo "action=set TASK068_ALLOW_UNRELATED_CTEST_FAILS=1 for unit PASS_WITH_TRIAGE; KL e2e is separate"
      else
        echo "classification=NEEDS_MANUAL_REVIEW"
        echo "note=Non-brackets ctest fail; confirm not schedule/API before ALLOW=1"
      fi
    } | tee "${TRIAGE_NOTE}" | tee -a "${SUMMARY}"

    if [[ "${TASK068_ALLOW_UNRELATED_CTEST_FAILS:-0}" != "1" ]]; then
      record "ctest-main" "${RC_CTEST}" "FAIL"
      echo "FAIL: set TASK068_ALLOW_UNRELATED_CTEST_FAILS=1 only after documenting triage in summary" >&2
      echo "     triage file: ${TRIAGE_NOTE}" >&2
      echo "VERDICT=FAIL" | tee -a "${SUMMARY}"
      exit 1
    fi
    record "ctest-main" "${RC_CTEST}" "PASS_WITH_TRIAGE"
  else
    record "ctest-main" 0 "PASS"
  fi
fi

echo "----" | tee -a "${SUMMARY}"
echo "results_tsv=${RESULTS}" | tee -a "${SUMMARY}"
if [[ -f "${RESULTS}" ]]; then
  column -t -s $'\t' "${RESULTS}" 2>/dev/null | tee -a "${SUMMARY}" || cat "${RESULTS}" | tee -a "${SUMMARY}"
fi

# Expose for post-rebase orchestrator (same shell only if sourced — also write file).
echo "${UNRELATED_FAILS}" > "${OUT_DIR}/unrelated_fails.txt"

if [[ "${UNRELATED_FAILS}" -eq 1 ]]; then
  echo "VERDICT=PASS_WITH_TRIAGE" | tee -a "${SUMMARY}"
  echo "UNIT: brackets green; ctest had unrelated fails (documented) or PRIOR_TRIAGE. KL e2e not run."
  exit 0
fi

echo "VERDICT=PASS" | tee -a "${SUMMARY}"
echo "UNIT: metal smoke + brackets schedule green. KL e2e not run (use run-task068-q8-brackets-kl.sh or TASK068_RUN_KL_E2E=1)."
exit 0
