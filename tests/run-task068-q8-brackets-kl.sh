#!/usr/bin/env bash
# TASK-068 ST-6.2 - MANUAL long-ctx KL ladder (NOT in ctest -L main / PR suite).
#
# For reviewers: reproduce KV distribution drift under long prefill and show
# that -ctv turbo3.8 is closer to q8_V than uniform -ctv turbo4.
# Prefill proxy only; factual MTP KEEP stays product Voyna evidence.
# See docs/KV-cache-quantization.md "Regression gates".
#
# SKIP 77 when required env / binary missing or GPU busy - no silent defaults.
#
# Env (required):
#   TASK068_MODEL  - GGUF path
#   TASK068_CTX    - context size (recommend >=16384)
# Env (optional):
#   TASK068_PROMPT - prompt file (else tile a short built-in seed)
#   TASK068_WORKDIR - scratch dir for logits / padded prompt
#   LLAMA_PERPLEXITY - override binary (else arg1)
#
# Pass: Mean KLD(-ctv turbo3.8) < Mean KLD(-ctv turbo4) vs -ctv q8_0 base.

set -euo pipefail

PPL_BIN="${1:-${LLAMA_PERPLEXITY:-}}"
if [[ -z "${PPL_BIN}" || ! -x "${PPL_BIN}" ]]; then
  # Fallback: common build layouts when ctest cwd is the build tree.
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
  echo "host-skip: set TASK068_CTX=<n> (recommend >=16384)" >&2
  exit 77
fi
if ! [[ "${CTX}" =~ ^[0-9]+$ ]] || [[ "${CTX}" -lt 1 ]]; then
  echo "fail-closed: TASK068_CTX must be a positive integer (got '${CTX}')" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKDIR="${TASK068_WORKDIR:-${REPO_ROOT}/logs/task068-kl-$$}"
mkdir -p "${WORKDIR}"

PROMPT_SRC="${TASK068_PROMPT:-}"
PROMPT_FILE="${WORKDIR}/prompt.txt"
if [[ -n "${PROMPT_SRC}" ]]; then
  if [[ ! -f "${PROMPT_SRC}" ]]; then
    echo "fail-closed: TASK068_PROMPT not a file: ${PROMPT_SRC}" >&2
    exit 1
  fi
  cp "${PROMPT_SRC}" "${PROMPT_FILE}"
else
  # Tile ASCII seed until roughly CTX*4 bytes (approx token budget without tokenizer).
  SEED="Pierre Belukhov and Dolokhov duel at dawn. The shot wounds Dolokhov badly. "
  : > "${PROMPT_FILE}"
  TARGET_BYTES=$((CTX * 4))
  while [[ "$(wc -c < "${PROMPT_FILE}" | tr -d ' ')" -lt "${TARGET_BYTES}" ]]; do
    printf '%s' "${SEED}" >> "${PROMPT_FILE}"
  done
fi

BASE_LOGITS="${WORKDIR}/base-q8.logits"
LOG_BASE="${WORKDIR}/run-base.log"
LOG_UNI="${WORKDIR}/run-turbo4.log"
LOG_BR="${WORKDIR}/run-turbo38.log"

unset TURBO_LAYER_ADAPTIVE || true
unset TURBO_LAYER_HEAD || true
unset TURBO_LAYER_TAIL || true
unset TURBO_LAYER_Q8_BRACKETS || true

echo "TASK-068 KL drift: model=${MODEL} ctx=${CTX} workdir=${WORKDIR}"

# Common argv: -fit off avoids hard abort when VRAM fit probe fails on Metal hosts.
PPL_COMMON=(
  -m "${MODEL}"
  -f "${PROMPT_FILE}"
  -c "${CTX}"
  -ngl 99
  -fa on
  -fit off
  -ctk q8_0
)

echo "== base: -ctk q8_0 -ctv q8_0 (save logits) =="
"${PPL_BIN}" \
  "${PPL_COMMON[@]}" \
  -ctv q8_0 \
  --save-all-logits "${BASE_LOGITS}" \
  >"${LOG_BASE}" 2>&1 || {
    if grep -qE 'failed to create command queue|failed to initialize.*(backend|context)|Metal' "${LOG_BASE}" 2>/dev/null; then
      echo "host-skip: Metal/backend init failed (free GPU / stop other llama-server); see ${LOG_BASE}" >&2
      exit 77
    fi
    echo "fail: base logits run failed; see ${LOG_BASE}" >&2
    exit 1
  }

run_kl() {
  local label="$1"
  local ctv="$2"
  local log="$3"
  echo "== KL: -ctv ${ctv} (${label}) =="
  "${PPL_BIN}" \
    "${PPL_COMMON[@]}" \
    -ctv "${ctv}" \
    --kl-divergence-base "${BASE_LOGITS}" \
    --kl-divergence \
    >"${log}" 2>&1 || {
      if grep -qE 'failed to create command queue|failed to initialize.*(backend|context)|Metal' "${log}" 2>/dev/null; then
        echo "host-skip: Metal/backend init failed during ${label}; see ${log}" >&2
        exit 77
      fi
      echo "fail: KL run ${label} failed; see ${log}" >&2
      exit 1
    }
}

run_kl "uniform_turbo4" "turbo4" "${LOG_UNI}"
run_kl "brackets_turbo3.8" "turbo3.8" "${LOG_BR}"

parse_mean_kld() {
  local log="$1"
  # Example: "Mean    KLD:   0.012345 +/-   0.000001"
  local line
  line="$(grep -E '^Mean[[:space:]]+KLD:' "${log}" | tail -n1 || true)"
  if [[ -z "${line}" ]]; then
    echo ""
    return 0
  fi
  # Field after KLD:
  echo "${line}" | awk -F: '{print $2}' | awk '{print $1}'
}

KLD_UNI="$(parse_mean_kld "${LOG_UNI}")"
KLD_BR="$(parse_mean_kld "${LOG_BR}")"

if [[ -z "${KLD_UNI}" || -z "${KLD_BR}" ]]; then
  echo "fail: could not parse Mean KLD from logs" >&2
  echo "  uni=${LOG_UNI} br=${LOG_BR}" >&2
  exit 1
fi

echo "Mean KLD turbo4   = ${KLD_UNI}"
echo "Mean KLD turbo3.8 = ${KLD_BR}"

# awk compare: pass if brackets < uniform
PASS="$(awk -v a="${KLD_BR}" -v b="${KLD_UNI}" 'BEGIN { print (a+0 < b+0) ? "yes" : "no" }')"
if [[ "${PASS}" != "yes" ]]; then
  echo "FAIL: expected Mean KLD(turbo3.8) < Mean KLD(turbo4) (got ${KLD_BR} vs ${KLD_UNI})" >&2
  exit 1
fi

echo "PASS: KL(turbo3.8) < KL(turbo4) vs q8_V base"
exit 0
