#!/usr/bin/env bash
# TASK-068 ST-8 - instrumented Darwin RAM/swap probe (Metal host Terminal only).
#
# Phases (required TASK068_MEM_PHASE, no silent default):
#   product      - llama-cli product-like load @ CTX_PRODUCT (default 160000) idle hold
#   base         - single llama-perplexity base logits save (KL recipe batch rules)
#   kl_one       - base + one KL arm (turbo3.8)
#   kl_full      - full same-mid KL via run-task068-q8-brackets-kl.sh
#   batch_eq_ctx - full KL with TASK068_BATCH forced to CTX (num_batches=1 path)
#
# Required:
#   TASK068_MODEL
#   TASK068_CTX          - KL ctx (prefer 8192)
#   TASK068_MEM_PHASE
# Env (optional):
#   BUILD_DIR / LLAMA_PERPLEXITY / TASK068_PROMPT / TASK068_CHUNKS / TASK068_BATCH
#   TASK068_UBATCH / TASK068_PARALLEL / TASK068_ALLOW_MULTI_BATCH_PPL
#   TASK068_ALLOW_HIGH_CTX_MEM / TASK068_CTX_PRODUCT / TASK068_DRAFT
#   TASK068_PRODUCT_HOLD_SEC (default 20; product phase idle hold)
#   TASK068_CTV_PRODUCT (default turbo3.8)
#
# Logs: logs/task068-mem-st8-<stamp>/
# Agent sandboxes cannot init Metal - run in a real Terminal; stop minillm first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLER="${REPO_ROOT}/tests/task068_mem_sample.py"
KL_SCRIPT="${REPO_ROOT}/tests/run-task068-q8-brackets-kl.sh"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task068-mem-st8-${STAMP}}"
mkdir -p "${OUT_DIR}"

MODEL="${TASK068_MODEL:-}"
CTX="${TASK068_CTX:-}"
PHASE="${TASK068_MEM_PHASE:-}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
PPL_BIN="${LLAMA_PERPLEXITY:-${BUILD_DIR}/bin/llama-perplexity}"
CLI_BIN="${LLAMA_CLI:-${BUILD_DIR}/bin/llama-cli}"
SAMPLE_TSV="${OUT_DIR}/samples.tsv"
SUMMARY="${OUT_DIR}/summary.txt"

sample() {
  local label="$1"
  local pid="${2:-0}"
  local hdr=()
  if [[ ! -f "${SAMPLE_TSV}" ]]; then
    hdr=(--header)
  fi
  python3 "${SAMPLER}" --label "${label}" --pid "${pid}" --append "${SAMPLE_TSV}" "${hdr[@]}" \
    | tee -a "${SUMMARY}"
}

fail_closed() {
  echo "FAIL: $*" >&2
  exit 1
}

{
  echo "stamp_dir=${OUT_DIR}"
  echo "suite=task068-mem-st8-probe"
  echo "tip_sha=$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "phase=${PHASE}"
  echo "model=${MODEL}"
  echo "ctx=${CTX}"
  echo "build_dir=${BUILD_DIR}"
  echo "ppl_bin=${PPL_BIN}"
} | tee "${SUMMARY}"

[[ -f "${SAMPLER}" ]] || fail_closed "missing sampler ${SAMPLER}"
[[ -n "${PHASE}" ]] || fail_closed "set TASK068_MEM_PHASE=product|base|kl_one|kl_full|batch_eq_ctx (required; no silent default)"
[[ -n "${MODEL}" && -f "${MODEL}" ]] || fail_closed "set TASK068_MODEL=/path/to.gguf (required)"
[[ -n "${CTX}" ]] || fail_closed "set TASK068_CTX=<n> (required; prefer 8192)"
[[ "${CTX}" =~ ^[0-9]+$ && "${CTX}" -ge 1 ]] || fail_closed "TASK068_CTX must be a positive integer (got '${CTX}')"
[[ -x "${PPL_BIN}" ]] || fail_closed "missing llama-perplexity: ${PPL_BIN}"

case "${PHASE}" in
  product|base|kl_one|kl_full|batch_eq_ctx) ;;
  *) fail_closed "unknown TASK068_MEM_PHASE=${PHASE}" ;;
esac

sample "host_idle_before" 0

run_monitored() {
  local label="$1"
  shift
  local log="${OUT_DIR}/${label}.log"
  set +e
  "$@" >"${log}" 2>&1 &
  local pid=$!
  set -e
  sample "${label}_start" "${pid}"
  local rc=0
  while kill -0 "${pid}" 2>/dev/null; do
    sample "${label}_running" "${pid}"
    sleep 5
  done
  set +e
  wait "${pid}"
  rc=$?
  set -e
  sample "${label}_after" 0
  echo "${label}_exit=${rc}" | tee -a "${SUMMARY}"
  return "${rc}"
}

if [[ "${PHASE}" == "product" ]]; then
  [[ -x "${CLI_BIN}" ]] || fail_closed "missing llama-cli for product phase: ${CLI_BIN}"
  CTX_PRODUCT="${TASK068_CTX_PRODUCT:-160000}"
  CTV_PRODUCT="${TASK068_CTV_PRODUCT:-turbo3.8}"
  HOLD_SEC="${TASK068_PRODUCT_HOLD_SEC:-20}"
  if ! [[ "${CTX_PRODUCT}" =~ ^[0-9]+$ ]] || [[ "${CTX_PRODUCT}" -lt 1 ]]; then
    fail_closed "TASK068_CTX_PRODUCT must be a positive integer"
  fi
  if ! [[ "${HOLD_SEC}" =~ ^[0-9]+$ ]] || [[ "${HOLD_SEC}" -lt 1 ]]; then
    fail_closed "TASK068_PRODUCT_HOLD_SEC must be a positive integer"
  fi
  echo "product_ctx=${CTX_PRODUCT} ctv=${CTV_PRODUCT} hold_sec=${HOLD_SEC}" | tee -a "${SUMMARY}"
  PROMPT_FILE="${OUT_DIR}/product-prompt.txt"
  printf 'ST-8 product idle probe.\n' > "${PROMPT_FILE}"
  CLI_ARGS=(
    -m "${MODEL}"
    -f "${PROMPT_FILE}"
    -c "${CTX_PRODUCT}"
    -b 2048
    -ub 512
    -np 1
    -ngl 99
    -fa on
    -fit off
    -ctk q8_0
    -ctv "${CTV_PRODUCT}"
    -n 1
    --cache-ram 4096
  )
  if [[ -n "${TASK068_DRAFT:-}" ]]; then
    [[ -f "${TASK068_DRAFT}" ]] || fail_closed "TASK068_DRAFT not a file: ${TASK068_DRAFT}"
    CLI_ARGS+=(--model-draft "${TASK068_DRAFT}" --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.95)
  fi
  set +e
  "${CLI_BIN}" "${CLI_ARGS[@]}" >"${OUT_DIR}/product.log" 2>&1 &
  PID=$!
  set -e
  sample "product_start" "${PID}"
  SECS=0
  while kill -0 "${PID}" 2>/dev/null && [[ "${SECS}" -lt "${HOLD_SEC}" ]]; do
    sample "product_hold" "${PID}"
    sleep 5
    SECS=$((SECS + 5))
  done
  if kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" 2>/dev/null || true
    wait "${PID}" 2>/dev/null || true
  fi
  sample "product_after" 0
  echo "phase_product=done" | tee -a "${SUMMARY}"
  sample "host_idle_after" 0
  echo "VERDICT=PROBE_COMPLETE phase=${PHASE} samples=${SAMPLE_TSV}" | tee -a "${SUMMARY}"
  exit 0
fi

# KL-family phases share KL script env; fail-closed batch rules live in KL script.
export TASK068_MODEL="${MODEL}"
export TASK068_CTX="${CTX}"
export TASK068_WORKDIR="${OUT_DIR}/kl"
mkdir -p "${TASK068_WORKDIR}"
if [[ -n "${TASK068_PROMPT:-}" ]]; then
  export TASK068_PROMPT
fi

if [[ "${PHASE}" == "batch_eq_ctx" ]]; then
  export TASK068_BATCH="${CTX}"
  unset TASK068_ALLOW_MULTI_BATCH_PPL || true
  echo "forced TASK068_BATCH=${CTX} (batch_eq_ctx)" | tee -a "${SUMMARY}"
fi

if [[ "${PHASE}" == "base" || "${PHASE}" == "kl_one" ]]; then
  # Thin wrappers: reuse KL script only for full; for base/kl_one invoke perplexity directly
  # with the same fail-closed batch policy as the KL script (source checks by calling it
  # with a dry path is overkill - duplicate the guard via env and call PPL).
  CHUNKS="${TASK068_CHUNKS:-2}"
  if [[ -z "${TASK068_BATCH:-}" ]]; then
    BATCH="${CTX}"
  else
    BATCH="${TASK068_BATCH}"
  fi
  UBATCH="${TASK068_UBATCH:-512}"
  PARALLEL="${TASK068_PARALLEL:-1}"
  if [[ "${BATCH}" -lt "${CTX}" && "${TASK068_ALLOW_MULTI_BATCH_PPL:-0}" != "1" ]]; then
    fail_closed "TASK068_BATCH=${BATCH} < CTX=${CTX} needs TASK068_ALLOW_MULTI_BATCH_PPL=1 (logits.reserve path)"
  fi
  TARGET_BYTES=$((CTX * CHUNKS * 8))
  PROMPT_FILE="${TASK068_WORKDIR}/prompt.txt"
  if [[ -n "${TASK068_PROMPT:-}" ]]; then
    cp "${TASK068_PROMPT}" "${PROMPT_FILE}"
  else
    SEED="Pierre Belukhov and Dolokhov duel at dawn. The shot wounds Dolokhov badly. "
    : > "${PROMPT_FILE}"
    while [[ "$(wc -c < "${PROMPT_FILE}" | tr -d ' ')" -lt "${TARGET_BYTES}" ]]; do
      printf '%s' "${SEED}" >> "${PROMPT_FILE}"
    done
  fi
  PPL_COMMON=(
    -m "${MODEL}"
    -f "${PROMPT_FILE}"
    -c "${CTX}"
    -b "${BATCH}"
    -ub "${UBATCH}"
    -np "${PARALLEL}"
    --chunks "${CHUNKS}"
    -ngl 99
    -fa on
    -fit off
    -ctk q8_0
  )
  BASE_LOGITS="${TASK068_WORKDIR}/base-q8.logits"
  echo "batch=${BATCH} ubatch=${UBATCH} chunks=${CHUNKS} parallel=${PARALLEL}" | tee -a "${SUMMARY}"
  unset TURBO_LAYER_ADAPTIVE TURBO_LAYER_HEAD TURBO_LAYER_TAIL TURBO_LAYER_Q8_BRACKETS || true
  run_monitored "base_logits" "${PPL_BIN}" "${PPL_COMMON[@]}" -ctv q8_0 --save-all-logits "${BASE_LOGITS}" \
    || fail_closed "base logits failed; see ${OUT_DIR}/base_logits.log"
  if [[ "${PHASE}" == "kl_one" ]]; then
    run_monitored "kl_turbo38" "${PPL_BIN}" "${PPL_COMMON[@]}" -ctv turbo3.8 \
      --kl-divergence-base "${BASE_LOGITS}" --kl-divergence \
      || fail_closed "kl_one failed; see ${OUT_DIR}/kl_turbo38.log"
  fi
  rm -f "${BASE_LOGITS}"
  sample "host_idle_after" 0
  echo "VERDICT=PROBE_COMPLETE phase=${PHASE} samples=${SAMPLE_TSV}" | tee -a "${SUMMARY}"
  exit 0
fi

# kl_full | batch_eq_ctx -> canon KL script under sample loop via background is hard
# (script runs 3 processes). Sample host before/after and stream KL log.
sample "kl_full_before" 0
set +e
bash "${KL_SCRIPT}" "${PPL_BIN}" >"${OUT_DIR}/kl-e2e.log" 2>&1 &
KL_PID=$!
set -e
while kill -0 "${KL_PID}" 2>/dev/null; do
  # Track llama-perplexity child if present
  CHILD="$(pgrep -P "${KL_PID}" -f llama-perplexity 2>/dev/null | head -n1 || true)"
  if [[ -n "${CHILD}" ]]; then
    sample "kl_full_child" "${CHILD}"
  else
    sample "kl_full_host" 0
  fi
  sleep 5
done
set +e
wait "${KL_PID}"
RC_KL=$?
set -e
tail -n 30 "${OUT_DIR}/kl-e2e.log" | tee -a "${SUMMARY}" || true
sample "kl_full_after" 0
echo "kl_full_exit=${RC_KL}" | tee -a "${SUMMARY}"
sample "host_idle_after" 0
if [[ "${RC_KL}" -ne 0 ]]; then
  echo "VERDICT=PROBE_FAIL phase=${PHASE} rc=${RC_KL}" | tee -a "${SUMMARY}"
  exit "${RC_KL}"
fi
echo "VERDICT=PROBE_COMPLETE phase=${PHASE} samples=${SAMPLE_TSV}" | tee -a "${SUMMARY}"
exit 0
