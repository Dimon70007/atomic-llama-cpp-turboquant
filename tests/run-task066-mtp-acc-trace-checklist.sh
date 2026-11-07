#!/usr/bin/env bash
# TASK-066 follow-up B2 — ACC_TRACE checklist on the short (~2k) MTP fixture.
#
# Not a code change: fail-closed until an operator attaches an NDJSON / stderr
# capture from LLAMA_MTP_ACC_TRACE for MTP vs non-MTP on the same prompt.
#
# Stage 2 gate (after Stage 1 nomtp high-ubatch is quiet — see follow-up U1).
# Prefer ACC_TRACE at the same high ubatch used for U1; optional batch=1 insurance.
# Note regime in TASK066_ACC_TRACE_NOTE (e.g. ubatch=1024 nomtp_ok=1 first_diverge=...).
#
# Usage:
#   TASK066_ACC_TRACE_LOG=/path/to/mtp.ndjson \
#     ./tests/run-task066-mtp-acc-trace-checklist.sh [OUT_DIR]
# Env:
#   TASK066_ACC_TRACE_LOG — required existing file (fail-closed)
#   TASK066_ACC_TRACE_NOTE — optional one-line first-diverge note copied into OUT_DIR

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-mtp-acc-trace-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

SUMMARY="${OUT_DIR}/summary.txt"
LOG="${TASK066_ACC_TRACE_LOG:-}"

{
  echo "track=B2"
  echo "contract=LLAMA_MTP_ACC_TRACE run on ~2k fixture; skip-streak unset; first diverge recorded"
  echo "ref=MTP.md / TASK-066-cb-fusion-followup.md B2"
} | tee "${SUMMARY}"

if [[ -z "${LOG}" ]]; then
  echo "FAIL: missing TASK066_ACC_TRACE_LOG=/path/to.ndjson (or captured stderr file)" >&2
  echo "  Run greedy + MTP with LLAMA_MTP_ACC_TRACE unset-skip-streak, then re-run this checklist." >&2
  echo "VERDICT=FAIL (no log)" | tee -a "${SUMMARY}"
  exit 1
fi
if [[ ! -f "${LOG}" ]]; then
  echo "FAIL: TASK066_ACC_TRACE_LOG not a file: ${LOG}" >&2
  echo "VERDICT=FAIL (missing file)" | tee -a "${SUMMARY}"
  exit 1
fi
if [[ ! -s "${LOG}" ]]; then
  echo "FAIL: TASK066_ACC_TRACE_LOG is empty: ${LOG}" >&2
  echo "VERDICT=FAIL (empty log)" | tee -a "${SUMMARY}"
  exit 1
fi

cp -f "${LOG}" "${OUT_DIR}/acc-trace.capture"
BYTES=$(wc -c < "${OUT_DIR}/acc-trace.capture" | tr -d ' ')
echo "acc_trace_bytes=${BYTES}" | tee -a "${SUMMARY}"

if [[ -n "${TASK066_ACC_TRACE_NOTE:-}" ]]; then
  printf '%s\n' "${TASK066_ACC_TRACE_NOTE}" | tee "${OUT_DIR}/first-diverge-note.txt" | tee -a "${SUMMARY}"
else
  echo "NOTE: set TASK066_ACC_TRACE_NOTE='step=N draft_wrong|target_reject ...' for a complete B2 trail" | tee -a "${SUMMARY}"
fi

echo "VERDICT=PASS (log attached; review first-diverge note)" | tee -a "${SUMMARY}"
exit 0
