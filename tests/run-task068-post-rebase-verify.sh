#!/usr/bin/env bash
# TASK-068 ST-7 - post-dev-rebase orchestrator (Metal host, outside agent sandbox).
#
# Default = UNIT only (everyday-safe). KL e2e is opt-in for major rebase / holidays.
#
# Everyday:
#   BUILD_DIR=./build-st7-post-rebase ./tests/run-task068-unit-verify.sh
#
# Holiday / major TURBOQUANT_VERSION bump (unit + KL):
#   export TASK068_MODEL=/path/to/model.gguf
#   export TASK068_CTX=8192
#   TASK068_RUN_KL_E2E=1 BUILD_DIR=./build-st7-post-rebase \
#     ./tests/run-task068-post-rebase-verify.sh
#
# Env:
#   TASK068_RUN_KL_E2E=1  - required to run KL (no silent e2e)
#   + all knobs from run-task068-unit-verify.sh and run-task068-q8-brackets-kl.sh
#
# See: docs/KV-cache-quantization.md "Regression gates"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task068-kl-post-rebase-${STAMP}}"
mkdir -p "${OUT_DIR}"

UNIT_SCRIPT="${REPO_ROOT}/tests/run-task068-unit-verify.sh"
KL_SCRIPT="${REPO_ROOT}/tests/run-task068-q8-brackets-kl.sh"
PPL_BIN="${BUILD_DIR}/bin/llama-perplexity"
SUMMARY="${OUT_DIR}/summary.txt"
RESULTS="${OUT_DIR}/results.tsv"

if [[ ! -f "${UNIT_SCRIPT}" ]]; then
  echo "FAIL: missing ${UNIT_SCRIPT}" >&2
  exit 1
fi

# --- Unit (always); owns initial summary.txt ---
echo "==> UNIT via run-task068-unit-verify.sh (TASK068_RUN_KL_E2E=${TASK068_RUN_KL_E2E:-0})"
set +e
bash "${UNIT_SCRIPT}" "${OUT_DIR}"
RC_UNIT=$?
set -e
{
  echo "orchestrator=task068-post-rebase-verify"
  echo "run_kl_e2e=${TASK068_RUN_KL_E2E:-0}"
  echo "note=default_is_unit_only_set_TASK068_RUN_KL_E2E=1_for_holiday_KL"
} | tee -a "${SUMMARY}"
if [[ "${RC_UNIT}" -ne 0 ]]; then
  echo "FAIL: unit verify failed (exit=${RC_UNIT}); see ${OUT_DIR}/summary.txt" >&2
  exit "${RC_UNIT}"
fi

UNRELATED_FAILS=0
if [[ -f "${OUT_DIR}/unrelated_fails.txt" ]]; then
  UNRELATED_FAILS="$(tr -d '[:space:]' < "${OUT_DIR}/unrelated_fails.txt" || echo 0)"
fi

# --- KL e2e (holiday / major rebase only) ---
if [[ "${TASK068_RUN_KL_E2E:-0}" != "1" ]]; then
  printf 'kl-e2e\t0\tSKIP_NOT_REQUESTED\n' >> "${RESULTS}"
  {
    echo "kl-e2e_exit=0 verdict=SKIP_NOT_REQUESTED"
    echo "----"
    echo "results_tsv=${RESULTS}"
  } | tee -a "${SUMMARY}"
  if [[ -f "${RESULTS}" ]]; then
    column -t -s $'\t' "${RESULTS}" 2>/dev/null | tee -a "${SUMMARY}" || cat "${RESULTS}" | tee -a "${SUMMARY}"
  fi
  if [[ "${UNRELATED_FAILS}" -eq 1 ]]; then
    echo "VERDICT=PASS_WITH_TRIAGE" | tee -a "${SUMMARY}"
    echo "ST-7 unit green with triage; KL skipped (set TASK068_RUN_KL_E2E=1 for holiday e2e)."
    exit 0
  fi
  echo "VERDICT=PASS" | tee -a "${SUMMARY}"
  echo "ST-7 unit green; KL e2e skipped (everyday default). Holiday: TASK068_RUN_KL_E2E=1."
  exit 0
fi

MODEL="${TASK068_MODEL:-}"
CTX="${TASK068_CTX:-}"
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
  echo "FAIL: TASK068_RUN_KL_E2E=1 requires TASK068_MODEL=/path/to.gguf" >&2
  exit 1
fi
if [[ -z "${CTX}" ]]; then
  echo "FAIL: TASK068_RUN_KL_E2E=1 requires TASK068_CTX=<n>" >&2
  exit 1
fi
if [[ ! -x "${PPL_BIN}" ]]; then
  echo "FAIL: missing ${PPL_BIN} for KL e2e" >&2
  exit 1
fi

echo "==> KL e2e (TASK068_RUN_KL_E2E=1, exclusive GPU)" | tee -a "${SUMMARY}"
echo "model=${MODEL}" | tee -a "${SUMMARY}"
echo "ctx=${CTX}" | tee -a "${SUMMARY}"
echo "task068_batch=${TASK068_BATCH:-}" | tee -a "${SUMMARY}"
echo "force_batch_eq_ctx=${TASK068_FORCE_BATCH_EQ_CTX:-0}" | tee -a "${SUMMARY}"
echo "kl_skip_turbo4_gate=${TASK068_KL_SKIP_TURBO4_GATE:-0}" | tee -a "${SUMMARY}"

KL_WORKDIR="${OUT_DIR}/kl"
mkdir -p "${KL_WORKDIR}"
export TASK068_MODEL="${MODEL}"
export TASK068_CTX="${CTX}"
export TASK068_WORKDIR="${KL_WORKDIR}"
if [[ -n "${TASK068_PROMPT:-}" ]]; then export TASK068_PROMPT; fi
if [[ -n "${TASK068_BATCH:-}" ]]; then export TASK068_BATCH; fi
if [[ -n "${TASK068_UBATCH:-}" ]]; then export TASK068_UBATCH; fi
if [[ -n "${TASK068_PARALLEL:-}" ]]; then export TASK068_PARALLEL; fi
if [[ -n "${TASK068_CHUNKS:-}" ]]; then export TASK068_CHUNKS; fi
if [[ -n "${TASK068_FORCE_BATCH_EQ_CTX:-}" ]]; then export TASK068_FORCE_BATCH_EQ_CTX; fi
if [[ -n "${TASK068_ALLOW_HIGH_CTX_MEM:-}" ]]; then export TASK068_ALLOW_HIGH_CTX_MEM; fi
if [[ -n "${TASK068_KL_SKIP_TURBO4_GATE:-}" ]]; then export TASK068_KL_SKIP_TURBO4_GATE; fi

set +e
bash "${KL_SCRIPT}" "${PPL_BIN}" >"${OUT_DIR}/kl-e2e.log" 2>&1
RC_KL=$?
set -e
tail -n 50 "${OUT_DIR}/kl-e2e.log" | tee -a "${SUMMARY}" || true

if [[ "${RC_KL}" -eq 77 ]]; then
  printf 'kl-e2e\t77\tSKIP\n' >> "${RESULTS}"
  echo "kl-e2e_exit=77 verdict=SKIP" | tee -a "${SUMMARY}"
  echo "FAIL: KL host-skip (77) - free GPU / check Metal; see ${OUT_DIR}/kl-e2e.log" >&2
  echo "VERDICT=FAIL" | tee -a "${SUMMARY}"
  exit 1
elif [[ "${RC_KL}" -ne 0 ]]; then
  printf 'kl-e2e\t%s\tFAIL\n' "${RC_KL}" >> "${RESULTS}"
  echo "kl-e2e_exit=${RC_KL} verdict=FAIL" | tee -a "${SUMMARY}"
  echo "VERDICT=FAIL" | tee -a "${SUMMARY}"
  exit 1
fi
printf 'kl-e2e\t0\tPASS\n' >> "${RESULTS}"
echo "kl-e2e_exit=0 verdict=PASS" | tee -a "${SUMMARY}"

echo "----" | tee -a "${SUMMARY}"
echo "results_tsv=${RESULTS}" | tee -a "${SUMMARY}"
if [[ -f "${RESULTS}" ]]; then
  column -t -s $'\t' "${RESULTS}" 2>/dev/null | tee -a "${SUMMARY}" || cat "${RESULTS}" | tee -a "${SUMMARY}"
fi

if [[ "${UNRELATED_FAILS}" -eq 1 ]]; then
  echo "VERDICT=PASS_WITH_TRIAGE" | tee -a "${SUMMARY}"
  echo "ST-7: unit + KL green; ctest had unrelated fails (documented above)."
  exit 0
fi

echo "VERDICT=PASS" | tee -a "${SUMMARY}"
echo "ST-7: unit + KL e2e green (holiday full verify)."
exit 0
