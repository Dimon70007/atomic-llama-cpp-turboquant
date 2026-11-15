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
#   TASK068_PROMPT - prompt file (else take a real-prose Voyna i Mir slice)
#   TASK068_CORPUS - corpus for the default prompt (assets/literature/voyna_i_mir_ru.txt)
#   TASK068_TOKENIZE_BIN - llama-tokenize next to llama-perplexity by default
#   TASK068_PROMPT_MIN_TOKENS - prompt size target (default 2 * CTX)
#   TASK068_WORKDIR - scratch dir for logits / padded prompt
#   TASK068_PROTECT_DENSE - 0 (default) = positional head/tail brackets;
#                           1 = q8 on the layers AFTER each dense layer, with the
#                           tail spread outwards from the last dense layer
#   TASK068_HEAD_PERCENT - head width as % of KV-bearing layers (default: library)
#   TASK068_TAIL_PERCENT - tail width as % of KV-bearing layers (default: library)
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

# Cost preflight: --save-all-logits writes ~0.5 MiB per token at n_vocab=262144,
# so CTX=16384 already needs ~8.6 GB. Refuse a point that cannot fit the disk.
PLAN_PY="${REPO_ROOT}/tests/lib/kld_ladder_plan.py"
if [[ ! -f "${PLAN_PY}" ]]; then
  echo "fail-closed: missing ${PLAN_PY}" >&2
  exit 1
fi
MAX_GB="${TASK068_PAYLOAD_MAX_GB:-40}"
CHUNKS="${TASK068_CHUNKS:-}"
PLAN_ARGS=(--model "${MODEL}" --ctx-list "${CTX}" --out-dir "${WORKDIR}" --max-gb "${MAX_GB}" --allow-single-ctx)
if [[ -n "${CHUNKS}" ]]; then
  # Pin the chunk count exactly: budget = CTX * CHUNKS.
  PLAN_ARGS+=(--token-budget "$(( CTX * CHUNKS ))" --chunks-floor 1 --allow-few-chunks)
else
  # Whole prompt: one chunk covering CTX.
  PLAN_ARGS+=(--token-budget "${CTX}" --chunks-floor 1 --allow-few-chunks)
fi
if ! PLAN_OUT="$(python3 "${PLAN_PY}" plan "${PLAN_ARGS[@]}" 2>&1)"; then
  echo "${PLAN_OUT}" >&2
  echo "fail-closed: KLD payload refused (see above); nothing was run" >&2
  exit 1
fi
echo "${PLAN_OUT}" | tee "${WORKDIR}/plan.txt"
# No --chunks flag means the whole prompt is one chunk; keep it explicit below.
PLAN_CHUNKS="$(echo "${PLAN_OUT}" | awk -F'\t' 'NR > 2 { print $2; exit }')"
if [[ -z "${PLAN_CHUNKS}" ]]; then
  echo "fail-closed: could not read chunk count from the plan" >&2
  exit 1
fi

PROMPT_SRC="${TASK068_PROMPT:-}"
PROMPT_FILE="${WORKDIR}/prompt.txt"
if [[ -n "${PROMPT_SRC}" ]]; then
  if [[ ! -f "${PROMPT_SRC}" ]]; then
    echo "fail-closed: TASK068_PROMPT not a file: ${PROMPT_SRC}" >&2
    exit 1
  fi
  cp "${PROMPT_SRC}" "${PROMPT_FILE}"
else
  # Real prose, never tiled filler: on a repeated sentence the model is nearly
  # certain of the next token, every arm lands at KLD ~1e-6 and the ranking is
  # noise. Take a contiguous Voyna i Mir slice instead (TASK-068 10.28 F).
  PROMPT_PY="${REPO_ROOT}/tests/lib/voyna_kl_prompt.py"
  # Default prompt is exactly 2 * CTX worth of prose (perplexity needs that much).
  PROMPT_TOKENS="${TASK068_PROMPT_MIN_TOKENS:-$((CTX * 2))}"
  TOKENIZE_BIN="${TASK068_TOKENIZE_BIN:-$(dirname "${PPL_BIN}")/llama-tokenize}"
  if [[ ! -f "${PROMPT_PY}" ]]; then
    echo "fail-closed: missing ${PROMPT_PY}" >&2
    exit 1
  fi
  if [[ ! -x "${TOKENIZE_BIN}" ]]; then
    echo "host-skip: missing llama-tokenize for real-prose prompt (TASK068_TOKENIZE_BIN)" >&2
    exit 77
  fi
  if ! PROMPT_OUT="$(
        python3 "${PROMPT_PY}" \
          --model "${MODEL}" \
          --tokenize-bin "${TOKENIZE_BIN}" \
          --min-tokens "${PROMPT_TOKENS}" \
          --out "${PROMPT_FILE}" \
          --corpus "${TASK068_CORPUS:-${REPO_ROOT}/../assets/literature/voyna_i_mir_ru.txt}"
      )"; then
    echo "${PROMPT_OUT}" >&2
    echo "fail-closed: could not build a real-prose prompt (no tiled fallback)" >&2
    exit 1
  fi
  echo "${PROMPT_OUT}"
fi

BASE_LOGITS="${WORKDIR}/base-q8.logits"
LOG_BASE="${WORKDIR}/run-base.log"
LOG_T3="${WORKDIR}/run-turbo3.log"
LOG_T4="${WORKDIR}/run-turbo4.log"
LOG_T38="${WORKDIR}/run-turbo38.log"
LOG_T48="${WORKDIR}/run-turbo48.log"

unset TURBO_LAYER_ADAPTIVE || true
unset TURBO_LAYER_HEAD || true
unset TURBO_LAYER_TAIL || true
unset TURBO_LAYER_Q8_BRACKETS || true

# Dense mode is opt-in: an unset/0 value must leave the positional schedule in
# place, and only "1" switches the plan to the dense-neighbour spread.
PROTECT_DENSE="${TASK068_PROTECT_DENSE:-0}"
case "${PROTECT_DENSE}" in
  0|1) ;;
  *)
    echo "fail-closed: TASK068_PROTECT_DENSE must be 0 or 1 (got '${PROTECT_DENSE}')" >&2
    exit 1
    ;;
esac
if [[ "${PROTECT_DENSE}" == "1" ]]; then
  export TURBO_LAYER_PROTECT_DENSE=1
else
  unset TURBO_LAYER_PROTECT_DENSE || true
fi

# Widths are optional: unset means "library default" (6/25). Setting them here is
# what makes a 6/35 sweep comparable across arms, so an explicit value must reach
# every bracket arm.
for pair in "TASK068_HEAD_PERCENT:TURBO_LAYER_HEAD_PERCENT" \
            "TASK068_TAIL_PERCENT:TURBO_LAYER_TAIL_PERCENT"; do
  src="${pair%%:*}"; dst="${pair##*:}"
  val="${!src:-}"
  if [[ -n "${val}" ]]; then
    if ! [[ "${val}" =~ ^[0-9]+$ ]] || (( val > 100 )); then
      echo "fail-closed: ${src} must be an integer 0..100 (got '${val}')" >&2
      exit 1
    fi
    export "${dst}=${val}"
  fi
done

echo "TASK-068 KL drift: model=${MODEL} ctx=${CTX} workdir=${WORKDIR} protect_dense=${PROTECT_DENSE} head_pct=${TASK068_HEAD_PERCENT:-default} tail_pct=${TASK068_TAIL_PERCENT:-default}"

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

# perplexity can exit 0 after refusing the input ("you need at least ... tokens"),
# leaving a stub header and no logits. Fail here instead of running four arms
# that all report "failed reading n_vocab, n_chunk".
BASE_BYTES="$(wc -c < "${BASE_LOGITS}" | tr -d ' ')"
if [[ "${BASE_BYTES}" -lt 1024 ]]; then
  echo "fail: base logits is only ${BASE_BYTES} bytes; perplexity refused the prompt" >&2
  grep -E 'you need at least|tokenizes to only' "${LOG_BASE}" >&2 || true
  echo "fail: raise TASK068_CTX or the prompt size (needs >= 2 * CTX tokens)" >&2
  exit 1
fi

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

run_kl "uniform_turbo3"   "turbo3"   "${LOG_T3}"
run_kl "uniform_turbo4"   "turbo4"   "${LOG_T4}"
run_kl "brackets_turbo3.8" "turbo3.8" "${LOG_T38}"
run_kl "brackets_turbo4.8" "turbo4.8" "${LOG_T48}"

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

KLD_T3="$(parse_mean_kld "${LOG_T3}")"
KLD_T4="$(parse_mean_kld "${LOG_T4}")"
KLD_T38="$(parse_mean_kld "${LOG_T38}")"
KLD_T48="$(parse_mean_kld "${LOG_T48}")"

if [[ -z "${KLD_T3}" || -z "${KLD_T4}" || -z "${KLD_T38}" || -z "${KLD_T48}" ]]; then
  echo "fail: could not parse Mean KLD from logs" >&2
  echo "  t3=${LOG_T3} t4=${LOG_T4} t38=${LOG_T38} t48=${LOG_T48}" >&2
  exit 1
fi

echo "Mean KLD turbo3    = ${KLD_T3}"
echo "Mean KLD turbo4    = ${KLD_T4}"
echo "Mean KLD turbo3.8  = ${KLD_T38}"
echo "Mean KLD turbo4.8  = ${KLD_T48}"

# Same-mid gates (TASK-068 DoD 8): brackets must beat uniform of the same mid,
# i.e. KL(turbo3.8) < KL(turbo3) and KL(turbo4.8) < KL(turbo4) vs the q8_V base.
RC=0
cmp_lt() {
  local label="$1"
  local a="$2"
  local b="$3"
  if [[ "$(awk -v x="${a}" -v y="${b}" 'BEGIN { print (x+0 < y+0) ? "yes" : "no" }')" == "yes" ]]; then
    echo "PASS: ${label} (${a} < ${b})"
  else
    echo "FAIL: ${label} (got ${a} vs ${b})" >&2
    RC=1
  fi
}

cmp_lt "KL(turbo3.8) < KL(turbo3)" "${KLD_T38}" "${KLD_T3}"
cmp_lt "KL(turbo4.8) < KL(turbo4)" "${KLD_T48}" "${KLD_T4}"

if [[ "${RC}" -eq 0 ]]; then
  echo "PASS: brackets beat uniform mid vs q8_V base (ctx=${CTX} chunks=${PLAN_CHUNKS})"
fi
exit "${RC}"
