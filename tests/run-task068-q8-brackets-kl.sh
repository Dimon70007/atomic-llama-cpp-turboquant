#!/usr/bin/env bash
# TASK-068 ST-6.2 / ST-7 - MANUAL long-ctx KL ladder (NOT in ctest -L main).
#
# For reviewers: show that q8 brackets are closer to q8_V than uniform of the
# SAME mid type. Prefill proxy only; factual MTP KEEP stays product Voyna.
# See docs/KV-cache-quantization.md "Regression gates".
#
# SKIP 77 when required env / binary missing or GPU busy - no silent defaults.
#
# Env (required):
#   TASK068_MODEL  - GGUF path
#   TASK068_CTX    - context size. Prefer 8192 on Darwin unified memory.
#                    CTX>8192 needs TASK068_ALLOW_HIGH_CTX_MEM=1 (half-window
#                    logits peak ~ (CTX/2)*n_vocab*4B plus multi-GiB logits file).
# Env (optional):
#   TASK068_PROMPT - prompt file (else tile a short built-in seed)
#   TASK068_WORKDIR - scratch dir for logits / padded prompt
#   LLAMA_PERPLEXITY - override binary (else arg1)
#   TASK068_CHUNKS - max chunks (default 2)
#   TASK068_BATCH  - llama -b  (default = min(2048,CTX); product-like; perplexity
#                    streams logits when -b < -c and -np 1 — no full-window reserve)
#   TASK068_UBATCH - llama -ub (default 512)
#   TASK068_PARALLEL - llama -np (default 1; product-like, one sequence)
#   TASK068_FORCE_BATCH_EQ_CTX=1 - force BATCH=CTX (legacy single-batch; higher peak)
#   TASK068_ALLOW_HIGH_CTX_MEM=1 - permit CTX>8192 despite host RAM risk
#   TASK068_KL_SKIP_TURBO4_GATE=1 - skip turbo4/turbo4.8 arms (Darwin RAM; not default)
#
# Pass (same-mid, both mids):
#   Mean KLD(turbo3.8) < Mean KLD(turbo3)  AND
#   Mean KLD(turbo4.8) < Mean KLD(turbo4)  vs -ctv q8_0 base.
# Cross-mid KL(turbo3.8)<KL(turbo4) is NOT a gate (mid bitwidth confound).

set -euo pipefail

PPL_BIN="${1:-${LLAMA_PERPLEXITY:-}}"
if [[ -z "${PPL_BIN}" || ! -x "${PPL_BIN}" ]]; then
  for cand in \
      "${LLAMA_PERPLEXITY:-}" \
      "./bin/llama-perplexity" \
      "../bin/llama-perplexity"; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
      PPL_BIN="${cand}"
      break
    fi
  done
fi
if [[ -z "${PPL_BIN}" || ! -x "${PPL_BIN}" ]]; then
  echo "host-skip: missing llama-perplexity (build target + pass path, or LLAMA_PERPLEXITY)" >&2
  exit 77
fi

MODEL="${TASK068_MODEL:-}"
CTX="${TASK068_CTX:-}"
if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
  echo "host-skip: set TASK068_MODEL=/path/to.gguf" >&2
  exit 77
fi
if [[ -z "${CTX}" ]]; then
  echo "host-skip: set TASK068_CTX=<n> (prefer 8192 on Darwin; >8192 needs TASK068_ALLOW_HIGH_CTX_MEM=1)" >&2
  exit 77
fi
if ! [[ "${CTX}" =~ ^[0-9]+$ ]] || [[ "${CTX}" -lt 1 ]]; then
  echo "fail-closed: TASK068_CTX must be a positive integer (got '${CTX}')" >&2
  exit 1
fi

# Half-window decode logits ~ (CTX/2)*n_vocab*4B; @16384 + 26B model -> swap on unified memory.
if [[ "${CTX}" -gt 8192 && "${TASK068_ALLOW_HIGH_CTX_MEM:-0}" != "1" ]]; then
  echo "fail-closed: TASK068_CTX=${CTX} > 8192 risks host swap (half-window logits + logits file)." >&2
  echo "  Prefer: export TASK068_CTX=8192" >&2
  echo "  Or explicit: export TASK068_ALLOW_HIGH_CTX_MEM=1" >&2
  exit 1
fi

CHUNKS="${TASK068_CHUNKS:-2}"
if ! [[ "${CHUNKS}" =~ ^[0-9]+$ ]] || [[ "${CHUNKS}" -lt 2 ]]; then
  echo "fail-closed: TASK068_CHUNKS must be an integer >= 2 (got '${CHUNKS}')" >&2
  exit 1
fi

# ST-8: default product-like BATCH=min(2048,CTX). perplexity.cpp streams when -b < -c
# and -np 1 (no logits.reserve(n_ctx*n_vocab)). FORCE_BATCH_EQ_CTX keeps legacy path.
if [[ "${TASK068_FORCE_BATCH_EQ_CTX:-0}" == "1" ]]; then
  BATCH="${CTX}"
  BATCH_SOURCE="force_eq_ctx"
elif [[ -z "${TASK068_BATCH:-}" ]]; then
  if [[ "${CTX}" -lt 2048 ]]; then
    BATCH="${CTX}"
  else
    BATCH=2048
  fi
  BATCH_SOURCE="default_product_2048"
else
  BATCH="${TASK068_BATCH}"
  BATCH_SOURCE="env"
fi
UBATCH="${TASK068_UBATCH:-512}"
PARALLEL="${TASK068_PARALLEL:-1}"
if ! [[ "${BATCH}" =~ ^[0-9]+$ ]] || [[ "${BATCH}" -lt 1 ]]; then
  echo "fail-closed: TASK068_BATCH must be a positive integer (got '${BATCH}')" >&2
  exit 1
fi
if ! [[ "${UBATCH}" =~ ^[0-9]+$ ]] || [[ "${UBATCH}" -lt 1 ]]; then
  echo "fail-closed: TASK068_UBATCH must be a positive integer (got '${UBATCH}')" >&2
  exit 1
fi
if [[ "${UBATCH}" -gt "${BATCH}" ]]; then
  echo "fail-closed: TASK068_UBATCH=${UBATCH} > TASK068_BATCH=${BATCH}" >&2
  exit 1
fi
if ! [[ "${PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${PARALLEL}" -lt 1 ]]; then
  echo "fail-closed: TASK068_PARALLEL must be an integer >= 1 (got '${PARALLEL}')" >&2
  exit 1
fi
# perplexity.cpp: if n_batch >= n_ctx then n_batch must be a multiple of n_ctx.
if [[ "${BATCH}" -ge "${CTX}" ]] && (( BATCH % CTX != 0 )); then
  echo "fail-closed: TASK068_BATCH=${BATCH} >= CTX=${CTX} must be a multiple of CTX (perplexity assert)" >&2
  exit 1
fi
# Streaming path requires -np 1; multi-seq still accumulates (legacy reserve risk).
if [[ "${BATCH}" -lt "${CTX}" && "${PARALLEL}" -ne 1 ]]; then
  echo "fail-closed: TASK068_BATCH=${BATCH} < CTX=${CTX} with TASK068_PARALLEL=${PARALLEL} != 1" >&2
  echo "  Streaming avoid-reserve path needs -np 1. Use PARALLEL=1 or FORCE_BATCH_EQ_CTX=1." >&2
  exit 1
fi
STREAM_EXPECTED=0
if [[ "${BATCH}" -lt "${CTX}" && "${PARALLEL}" -eq 1 ]]; then
  STREAM_EXPECTED=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${TASK068_WORKDIR:-${REPO_ROOT}/logs/task068-kl-$$}"
mkdir -p "${WORKDIR}"

PROMPT_SRC="${TASK068_PROMPT:-}"
PROMPT_FILE="${WORKDIR}/prompt.txt"
MIN_TOKENS=$((CTX * CHUNKS))
if [[ "${MIN_TOKENS}" -lt $((CTX * 2)) ]]; then
  MIN_TOKENS=$((CTX * 2))
fi
TARGET_BYTES=$((MIN_TOKENS * 8))
if [[ -n "${PROMPT_SRC}" ]]; then
  if [[ ! -f "${PROMPT_SRC}" ]]; then
    echo "fail-closed: TASK068_PROMPT not a file: ${PROMPT_SRC}" >&2
    exit 1
  fi
  cp "${PROMPT_SRC}" "${PROMPT_FILE}"
  PROMPT_BYTES="$(wc -c < "${PROMPT_FILE}" | tr -d ' ')"
  if [[ "${PROMPT_BYTES}" -lt "${TARGET_BYTES}" ]]; then
    echo "fail-closed: TASK068_PROMPT is ${PROMPT_BYTES} bytes; need >= ${TARGET_BYTES} (~${MIN_TOKENS} tokens)" >&2
    exit 1
  fi
else
  SEED="Pierre Belukhov and Dolokhov duel at dawn. The shot wounds Dolokhov badly. "
  : > "${PROMPT_FILE}"
  while [[ "$(wc -c < "${PROMPT_FILE}" | tr -d ' ')" -lt "${TARGET_BYTES}" ]]; do
    printf '%s' "${SEED}" >> "${PROMPT_FILE}"
  done
fi
EST_LOGITS_GIB="$(awk -v c="${CHUNKS}" -v ctx="${CTX}" 'BEGIN { printf "%.1f", c * (ctx/16384.0) * 4.0 }')"
EST_HALF_LOGITS_GIB="$(awk -v ctx="${CTX}" 'BEGIN { printf "%.1f", (ctx/2.0) * 262144.0 * 4.0 / (1024.0*1024.0*1024.0) }')"
EST_STREAM_BATCH_GIB="$(awk -v b="${BATCH}" 'BEGIN { printf "%.1f", b * 262144.0 * 4.0 / (1024.0*1024.0*1024.0) }')"
echo "prompt_bytes=$(wc -c < "${PROMPT_FILE}" | tr -d ' ') min_tokens_required=${MIN_TOKENS} target_bytes=${TARGET_BYTES}"
echo "mem_guard: --chunks ${CHUNKS} -b ${BATCH} -ub ${UBATCH} -np ${PARALLEL} CTX=${CTX}; batch_source=${BATCH_SOURCE}; stream_expected=${STREAM_EXPECTED}; est_logits_file~${EST_LOGITS_GIB}GiB; est_halfwin_if_batch_eq_ctx~${EST_HALF_LOGITS_GIB}GiB; est_stream_batch_logits~${EST_STREAM_BATCH_GIB}GiB (vocab~262k)"
if [[ "${STREAM_EXPECTED}" -eq 1 ]]; then
  echo "mem_note: -b < -c && -np 1 => perplexity stream_batches=1 (no host logits.reserve)"
elif [[ "${BATCH}" -ge "${CTX}" ]]; then
  echo "mem_note: BATCH>=CTX => single-batch; half-window float logits stay resident (~est_halfwin)"
fi

BASE_LOGITS="${WORKDIR}/base-q8.logits"
LOG_BASE="${WORKDIR}/run-base.log"
LOG_UNI="${WORKDIR}/run-turbo3.log"
LOG_BR="${WORKDIR}/run-turbo38.log"
LOG_T4="${WORKDIR}/run-turbo4.log"
LOG_T48="${WORKDIR}/run-turbo48.log"
SKIP_T4_GATE="${TASK068_KL_SKIP_TURBO4_GATE:-0}"

unset TURBO_LAYER_ADAPTIVE || true
unset TURBO_LAYER_HEAD || true
unset TURBO_LAYER_TAIL || true
unset TURBO_LAYER_Q8_BRACKETS || true

echo "TASK-068 KL drift (same-mid gates): model=${MODEL} ctx=${CTX} chunks=${CHUNKS} batch=${BATCH} ubatch=${UBATCH} parallel=${PARALLEL} skip_turbo4_gate=${SKIP_T4_GATE} workdir=${WORKDIR}"

# -np 1: explicit single sequence (product parallel:1). Without it, perplexity may set
# n_parallel = max(1, n_batch/n_ctx) and multiply allocated ctx.
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

cleanup_logits() {
  if [[ -f "${BASE_LOGITS}" ]]; then
    rm -f "${BASE_LOGITS}"
    echo "cleaned: removed ${BASE_LOGITS}"
  fi
}

echo "== base: -ctk q8_0 -ctv q8_0 (save logits) =="
set +e
"${PPL_BIN}" \
  "${PPL_COMMON[@]}" \
  -ctv q8_0 \
  --save-all-logits "${BASE_LOGITS}" \
  >"${LOG_BASE}" 2>&1
RC_BASE=$?
set -e
if grep -qE 'calculating perplexity over|computing over' "${LOG_BASE}"; then
  grep -E 'calculating perplexity over|computing over' "${LOG_BASE}" | tail -n1 || true
fi
if grep -qE 'failed to create command queue|failed to initialize.*(backend|context)|Metal' "${LOG_BASE}" 2>/dev/null; then
  echo "host-skip: Metal/backend init failed (free GPU / stop other llama-server); see ${LOG_BASE}" >&2
  exit 77
fi
if grep -qE 'you need at least .* tokens to evaluate perplexity' "${LOG_BASE}"; then
  echo "fail-closed: base prompt too short for -c ${CTX} (need >= ${MIN_TOKENS} tokens); see ${LOG_BASE}" >&2
  exit 1
fi
if [[ "${RC_BASE}" -ne 0 ]]; then
  echo "fail: base logits run failed (exit=${RC_BASE}); see ${LOG_BASE}" >&2
  exit 1
fi
if [[ "${STREAM_EXPECTED}" -eq 1 ]]; then
  if ! grep -qE 'stream_batches=1' "${LOG_BASE}"; then
    echo "fail-closed: expected stream_batches=1 in base log (rebuild llama-perplexity with ST-8 stream); see ${LOG_BASE}" >&2
    exit 1
  fi
  echo "stream_batches=1 confirmed in base log"
fi
BASE_SZ="$(wc -c < "${BASE_LOGITS}" | tr -d ' ')"
if [[ ! -f "${BASE_LOGITS}" || "${BASE_SZ}" -lt 1024 ]]; then
  echo "fail-closed: base logits too small (${BASE_SZ} bytes); see ${LOG_BASE}" >&2
  exit 1
fi
echo "base_logits_bytes=${BASE_SZ}"

run_kl() {
  local label="$1"
  local ctv="$2"
  local log="$3"
  echo "== KL: -ctv ${ctv} (${label}) =="
  set +e
  "${PPL_BIN}" \
    "${PPL_COMMON[@]}" \
    -ctv "${ctv}" \
    --kl-divergence-base "${BASE_LOGITS}" \
    --kl-divergence \
    >"${log}" 2>&1
  local rc=$?
  set -e
  if grep -qE 'calculating perplexity over|computing over' "${log}"; then
    grep -E 'calculating perplexity over|computing over' "${log}" | tail -n1 || true
  fi
  if grep -qE 'failed to create command queue|failed to initialize.*(backend|context)|Metal' "${log}" 2>/dev/null; then
    echo "host-skip: Metal/backend init failed during ${label}; see ${log}" >&2
    exit 77
  fi
  if grep -qE 'you need at least .* tokens to evaluate perplexity|failed reading n_vocab' "${log}"; then
    echo "fail: KL run ${label} incomplete; see ${log}" >&2
    exit 1
  fi
  if [[ "${rc}" -ne 0 ]]; then
    echo "fail: KL run ${label} failed (exit=${rc}); see ${log}" >&2
    exit 1
  fi
}

# Gate arms: same-mid turbo3, then same-mid turbo4 (unless explicitly skipped).
run_kl "uniform_turbo3" "turbo3" "${LOG_UNI}"
run_kl "brackets_turbo3.8" "turbo3.8" "${LOG_BR}"
if [[ "${SKIP_T4_GATE}" != "1" ]]; then
  run_kl "uniform_turbo4" "turbo4" "${LOG_T4}"
  run_kl "brackets_turbo4.8" "turbo4.8" "${LOG_T48}"
else
  echo "note: TASK068_KL_SKIP_TURBO4_GATE=1 - turbo4/turbo4.8 same-mid gate skipped"
fi

parse_mean_kld() {
  local log="$1"
  local line
  line="$(grep -E 'Mean[[:space:]]+KLD:' "${log}" | tail -n1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return 0
  fi
  echo "${line}" | sed -E 's/.*Mean[[:space:]]+KLD:[[:space:]]*//' | awk '{print $1}'
}

KLD_UNI="$(parse_mean_kld "${LOG_UNI}")"
KLD_BR="$(parse_mean_kld "${LOG_BR}")"
KLD_T4=""
KLD_T48=""
if [[ "${SKIP_T4_GATE}" != "1" ]]; then
  KLD_T4="$(parse_mean_kld "${LOG_T4}")"
  KLD_T48="$(parse_mean_kld "${LOG_T48}")"
fi

# Free multi-GiB file before exit either way (Mean KLD already in logs).
cleanup_logits

if [[ -z "${KLD_UNI}" || -z "${KLD_BR}" ]]; then
  echo "fail: could not parse Mean KLD from turbo3 logs" >&2
  echo "  uni=${LOG_UNI} br=${LOG_BR}" >&2
  exit 1
fi
if [[ "${SKIP_T4_GATE}" != "1" ]] && [[ -z "${KLD_T4}" || -z "${KLD_T48}" ]]; then
  echo "fail: could not parse Mean KLD from turbo4 logs" >&2
  echo "  uni=${LOG_T4} br=${LOG_T48}" >&2
  exit 1
fi

echo "Mean KLD turbo3   = ${KLD_UNI}  (uniform same-mid)"
echo "Mean KLD turbo3.8 = ${KLD_BR}  (q8 brackets)"
if [[ "${SKIP_T4_GATE}" != "1" ]]; then
  echo "Mean KLD turbo4   = ${KLD_T4}  (uniform same-mid)"
  echo "Mean KLD turbo4.8 = ${KLD_T48}  (q8 brackets)"
fi

PASS3="$(awk -v a="${KLD_BR}" -v b="${KLD_UNI}" 'BEGIN { print (a+0 < b+0) ? "yes" : "no" }')"
if [[ "${PASS3}" != "yes" ]]; then
  echo "FAIL: expected Mean KLD(turbo3.8) < Mean KLD(turbo3) (got ${KLD_BR} vs ${KLD_UNI})" >&2
  exit 1
fi
echo "PASS: KL(turbo3.8) < KL(turbo3) vs q8_V base (same-mid)"

if [[ "${SKIP_T4_GATE}" != "1" ]]; then
  PASS4="$(awk -v a="${KLD_T48}" -v b="${KLD_T4}" 'BEGIN { print (a+0 < b+0) ? "yes" : "no" }')"
  if [[ "${PASS4}" != "yes" ]]; then
    echo "FAIL: expected Mean KLD(turbo4.8) < Mean KLD(turbo4) (got ${KLD_T48} vs ${KLD_T4})" >&2
    exit 1
  fi
  echo "PASS: KL(turbo4.8) < KL(turbo4) vs q8_V base (same-mid)"
fi

if [[ "${SKIP_T4_GATE}" == "1" ]]; then
  echo "PASS: same-mid brackets gate vs q8_V base (turbo3.8 only; turbo4 gate skipped)"
else
  echo "PASS: same-mid brackets gates vs q8_V base (turbo3.8 and turbo4.8)"
fi
exit 0
