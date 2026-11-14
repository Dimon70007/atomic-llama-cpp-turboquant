#!/usr/bin/env bash
# T-066.5-KLD — FP16 KV vs q8_0/turbo{3,4} KLD curve @ CTX 4096/16384/65536.
# Fail-closed: MODEL, N_UBATCH, BUILD_DIR required. No silent defaults.
#
# Usage (from fork root):
#   MODEL=path.gguf N_UBATCH=512 BUILD_DIR=build-task066-cb-ws \
#     ./tests/run-task066-iswa-kld.sh [OUT_DIR]
#
# Optional:
#   TEXT=wikitext-2-raw/wiki.test.raw
#   CTX_LIST="4096 16384 65536"
#   CAND_LIST="turbo3 turbo4"   # each implies -ctk q8_0 -ctv <cand>
#   NGL=99  N_BATCH=2048 (product gemma4 batch; required unless BATCH_EQ_CTX=1)
#   CHUNKS=6                    # wikitext chunks per CTX. Minimum 6: with 1 chunk the
#                               # +-unc of Mean PPL(Q)/PPL(base) is wider than the effect
#                               # between turbo3 and turbo4, so ppl_ratio becomes noise.
#                               # Raise for tighter intervals; lower only for smoke runs.
#   KEEP_LOGITS=0               # delete per-CTX base logits after candidates
#   FA_MODE=on                  # required; for CTX>=8000 refuse anything else
#   BATCH_EQ_CTX=0|1            # -b=CTX; ONLY allowed when every CTX <= 2000 (diag/smoke)
#
# ideas/llamacpp_swa_size_calculate.md: FA must stay ON with quantized V / long ctx
# (else V-transpose + N×N attn materialization blow RAM). Product YAML: flash_attn: on.
#
# Stamp: logs/task066-iswa-kld-<UTC>/ with per-CTX table + PASS/FAIL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-}"
MODEL="${MODEL:-}"
N_UBATCH="${N_UBATCH:-}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-iswa-kld-$(date -u +%Y%m%dT%H%M%SZ)}"
TEXT="${TEXT:-${REPO_ROOT}/wikitext-2-raw/wiki.test.raw}"
CTX_LIST="${CTX_LIST:-4096 16384 65536}"
CAND_LIST="${CAND_LIST:-turbo3 turbo4}"
NGL="${NGL:-99}"
CHUNKS="${CHUNKS:-6}"
KEEP_LOGITS="${KEEP_LOGITS:-0}"
# KLD rise threshold for walks_up: a step must exceed the noise floor to count as a
# rise, where floor = max(KLD_EPS_ABS, KLD_EPS_REL * a, unc_a, unc_b).
# Defaults target llama-perplexity KLD on wiki text: the measurement noise dominates
# at small CTX (unc ~1e-3..1e-2), so a bare "b > a" test flipped PASS on a +29% jump.
KLD_EPS_ABS="${KLD_EPS_ABS:-0.001}"
KLD_EPS_REL="${KLD_EPS_REL:-0.02}"
# Product gemma4_*_metal.yaml: batch 2048 / ubatch 512.
# BATCH_EQ_CTX=1 (-b=CTX) only for small CTX (<=2000); larger blows ~n_batch*n_vocab logits.
N_BATCH="${N_BATCH:-}"
BATCH_EQ_CTX="${BATCH_EQ_CTX:-0}"
BATCH_EQ_CTX_MAX=2000
FA_MODE="${FA_MODE:-on}"
FA_REQUIRED_CTX_MIN=8000

die() { echo "FAIL: $*" >&2; exit 2; }

if [[ "${FA_MODE}" != "on" ]]; then
  die "FA_MODE=${FA_MODE}: refuse (ideas + product require flash_attn on; quantized V / long-ctx >= ${FA_REQUIRED_CTX_MIN})"
fi

if [[ "${BATCH_EQ_CTX}" != "0" && "${BATCH_EQ_CTX}" != "1" ]]; then
  die "BATCH_EQ_CTX must be 0|1 (got ${BATCH_EQ_CTX})"
fi

# CHUNKS>=6 is the measurement floor: at 1 chunk the +-unc of Mean PPL(Q)/PPL(base)
# exceeds the turbo3-vs-turbo4 effect, so ppl_ratio stops resolving the difference
# (TASK-071 ST-7.6). ALLOW_FEW_CHUNKS=1 is the explicit opt-out for smoke runs only.
if ! [[ "${CHUNKS}" =~ ^[0-9]+$ ]]; then
  die "CHUNKS must be a positive integer (got '${CHUNKS}')"
fi
if [[ "${CHUNKS}" -lt 6 ]]; then
  if [[ "${ALLOW_FEW_CHUNKS:-0}" == "1" ]]; then
    echo "WARN: CHUNKS=${CHUNKS} < 6 (ALLOW_FEW_CHUNKS=1) - ppl_ratio will not resolve" >&2
    echo "WARN: a turbo3-vs-turbo4 comparison from this stamp is not valid" >&2
  else
    die "CHUNKS=${CHUNKS} < 6: one chunk makes ppl_ratio noise (TASK-071 ST-7.6); set ALLOW_FEW_CHUNKS=1 for a smoke run only"
  fi
fi

[[ -n "${BUILD_DIR}" ]] || die "missing required BUILD_DIR (refusing silent build dir)"
[[ -n "${MODEL}" ]] || die "missing required MODEL=/path/to.gguf (refusing silent model substitution)"
[[ -n "${N_UBATCH}" ]] || die "missing required N_UBATCH (product default is 512; must be explicit)"
if [[ "${BATCH_EQ_CTX}" != "1" ]]; then
  [[ -n "${N_BATCH}" ]] || die "missing required N_BATCH (product gemma4 uses 2048; or set BATCH_EQ_CTX=1 for CTX<=${BATCH_EQ_CTX_MAX} only)"
fi
[[ -f "${MODEL}" ]] || die "MODEL not a file: ${MODEL}"
[[ -f "${TEXT}" ]] || die "TEXT not a file: ${TEXT} (run scripts/get-wikitext-2.sh)"
[[ -d "${BUILD_DIR}" ]] || die "BUILD_DIR missing: ${BUILD_DIR}"

if [[ "${BATCH_EQ_CTX}" == "1" ]]; then
  for _ctx in ${CTX_LIST}; do
    if [[ "${_ctx}" -gt "${BATCH_EQ_CTX_MAX}" ]]; then
      die "BATCH_EQ_CTX=1 refused: CTX=${_ctx} > ${BATCH_EQ_CTX_MAX} (only small-ctx smoke; use N_BATCH=2048 for long KLD)"
    fi
  done
fi

PPL_BIN="${BUILD_DIR}/bin/llama-perplexity"
if [[ ! -x "${PPL_BIN}" ]]; then
  echo "==> rebuild llama-perplexity"
  cmake --build "${BUILD_DIR}" --target llama-perplexity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
fi
[[ -x "${PPL_BIN}" ]] || die "missing ${PPL_BIN}"

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.txt"
TABLE="${OUT_DIR}/kld_table.tsv"
VERDICT_FILE="${OUT_DIR}/verdict.txt"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "model=${MODEL}"
  echo "text=${TEXT}"
  echo "n_ubatch=${N_UBATCH}"
  echo "n_batch=${N_BATCH:-<BATCH_EQ_CTX>}"
  echo "batch_eq_ctx=${BATCH_EQ_CTX}"
  echo "batch_eq_ctx_max=${BATCH_EQ_CTX_MAX}"
  echo "ngl=${NGL}"
  echo "chunks=${CHUNKS}"
  echo "fa=${FA_MODE}"
  echo "fa_required_ctx_min=${FA_REQUIRED_CTX_MIN}"
  echo "ctx_list=${CTX_LIST}"
  echo "cand_list=${CAND_LIST}"
  echo "size_swa_formula=PAD(min(size_base, pad128(n_swa)*(unified?n_seq_max:1)+n_ubatch),512)"
  echo "ref=ctk=f16 ctv=f16 fa=${FA_MODE}"
  echo "note=ideas: FA-off @ long ctx = V-transpose + NxN attn RAM blow; product flash_attn:on"
  echo "note=BATCH_EQ_CTX=1 only when every CTX<=${BATCH_EQ_CTX_MAX}; long KLD uses N_BATCH (product 2048)"
} | tee "${SUMMARY}"

printf 'cand\tctx\tkld_mean\tkld_unc\tppl_ratio\tlog\n' > "${TABLE}"

parse_kld_mean() {
  # llama-perplexity: "Mean    KLD: %10.6lf ± %10.6lf"
  local logf="$1"
  local line
  line="$(grep -E 'Mean[[:space:]]+KLD:' "${logf}" | tail -1 || true)"
  if [[ -z "${line}" ]]; then
    echo "NA NA"
    return
  fi
  echo "${line}" | awk '
    {
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+\.[0-9]+$/ ) {
          mean = $i
          unc = "NA"
          if ($(i+1) == "±" && $(i+2) ~ /^[0-9]+\.[0-9]+$/) unc = $(i+2)
          print mean, unc
          exit
        }
      }
      print "NA NA"
    }'
}

run_ppl() {
  local tag="$1"; shift
  local logf="${OUT_DIR}/${tag}.log"
  echo "==> ${tag}"
  set +e
  "${PPL_BIN}" "$@" 2>&1 | tee "${logf}"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "${tag}_exit=${rc}" | tee -a "${SUMMARY}"
  return "${rc}"
}

ANY_FAIL=0
declare -a SERIES_TURBO3=()
declare -a SERIES_TURBO4=()
declare -a UNC_TURBO3=()
declare -a UNC_TURBO4=()

for CTX in ${CTX_LIST}; do
  if [[ "${BATCH_EQ_CTX}" == "1" ]]; then
    RUN_BATCH="${CTX}"
  else
    RUN_BATCH="${N_BATCH}"
  fi
  # Belt-and-braces: refuse -b=CTX for large windows even if N_BATCH was set equal to CTX.
  if [[ "${RUN_BATCH}" -eq "${CTX}" && "${CTX}" -gt "${BATCH_EQ_CTX_MAX}" ]]; then
    die "refusing -b=${RUN_BATCH} (==CTX) for CTX=${CTX} > ${BATCH_EQ_CTX_MAX} (logits footprint); use N_BATCH=2048"
  fi
  BASE_LOGITS="${OUT_DIR}/base-f16-c${CTX}.kld"
  BASE_TAG="base-f16-c${CTX}"
  echo "==> CTX=${CTX} n_batch=${RUN_BATCH} n_ubatch=${N_UBATCH} fa=${FA_MODE} save FP16 logits -> ${BASE_LOGITS}"
  if ! run_ppl "${BASE_TAG}" \
      -m "${MODEL}" -f "${TEXT}" -c "${CTX}" -ngl "${NGL}" \
      -b "${RUN_BATCH}" -ub "${N_UBATCH}" \
      -ctk f16 -ctv f16 -fa "${FA_MODE}" \
      --chunks "${CHUNKS}" \
      --save-all-logits "${BASE_LOGITS}"; then
    echo "FAIL: base logits save CTX=${CTX}" | tee -a "${SUMMARY}"
    ANY_FAIL=1
    continue
  fi
  [[ -f "${BASE_LOGITS}" ]] || { echo "FAIL: missing logits ${BASE_LOGITS}" | tee -a "${SUMMARY}"; ANY_FAIL=1; continue; }

  for CAND in ${CAND_LIST}; do
    TAG="cand-q8_${CAND}-c${CTX}"
    echo "==> CTX=${CTX} n_batch=${RUN_BATCH} fa=${FA_MODE} cand=q8_0/${CAND}"
    if ! run_ppl "${TAG}" \
        -m "${MODEL}" -f "${TEXT}" -c "${CTX}" -ngl "${NGL}" \
        -b "${RUN_BATCH}" -ub "${N_UBATCH}" \
        -ctk q8_0 -ctv "${CAND}" -fa "${FA_MODE}" \
        --chunks "${CHUNKS}" \
        --kl-divergence-base "${BASE_LOGITS}" \
        --kl-divergence; then
      echo "FAIL: KLD run ${TAG}" | tee -a "${SUMMARY}"
      ANY_FAIL=1
      printf '%s\t%s\tNA\tNA\tNA\t%s\n' "${CAND}" "${CTX}" "${OUT_DIR}/${TAG}.log" >> "${TABLE}"
      continue
    fi
    read -r KLD_MEAN KLD_UNC <<<"$(parse_kld_mean "${OUT_DIR}/${TAG}.log")"
    PPL_RATIO="$(grep -E 'Mean PPL\(Q\)/PPL\(base\)' "${OUT_DIR}/${TAG}.log" | tail -1 | awk '{print $(NF-2)}' || echo NA)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${CAND}" "${CTX}" "${KLD_MEAN}" "${KLD_UNC}" "${PPL_RATIO}" "${OUT_DIR}/${TAG}.log" >> "${TABLE}"
    echo "KLD ${CAND}@${CTX} mean=${KLD_MEAN} unc=${KLD_UNC}" | tee -a "${SUMMARY}"
    if [[ "${CAND}" == "turbo3" ]]; then
      SERIES_TURBO3+=("${KLD_MEAN}")
      UNC_TURBO3+=("${KLD_UNC}")
    elif [[ "${CAND}" == "turbo4" ]]; then
      SERIES_TURBO4+=("${KLD_MEAN}")
      UNC_TURBO4+=("${KLD_UNC}")
    fi
  done

  if [[ "${KEEP_LOGITS}" != "1" ]]; then
    rm -f "${BASE_LOGITS}"
  fi
done

# Return 0 if the KLD series walks up with context, i.e. any step rises beyond the
# noise floor. Return 1 if it is flat-or-down.
#
# The old rule demanded strict monotonic increase, so a series that dipped and then
# jumped (0.169537 0.168799 0.218236, +29% at the last step) was reported as
# "flat-or-down" and PASSED. A step counts as a rise when
#   b - a > max(KLD_EPS_ABS, KLD_EPS_REL * a, unc_a, unc_b)
# so a rise inside the reported +-unc stays within noise (flat), while any rise
# larger than the noise fails.
#
# argv: means... -- uncs... (uncs optional; missing entries treated as 0)
walks_up() {
  local -a vals=()
  local -a uncs=()
  local seen_sep=0
  local arg
  for arg in "$@"; do
    if [[ "${arg}" == "--" ]]; then
      seen_sep=1
      continue
    fi
    if [[ "${seen_sep}" -eq 0 ]]; then
      vals+=("${arg}")
    else
      uncs+=("${arg}")
    fi
  done
  local i
  if [[ "${#vals[@]}" -lt 2 ]]; then
    return 0   # cannot judge -> caller marks FAIL (unchanged behaviour)
  fi
  for ((i = 1; i < ${#vals[@]}; i++)); do
    local a="${vals[i-1]}"
    local b="${vals[i]}"
    if [[ "${a}" == "NA" || "${b}" == "NA" ]]; then
      return 0
    fi
    local ua="${uncs[i-1]:-0}"
    local ub="${uncs[i]:-0}"
    # awk: does b exceed a by more than the noise floor?
    if awk -v a="${a}" -v b="${b}" -v ua="${ua}" -v ub="${ub}" \
            -v eabs="${KLD_EPS_ABS}" -v erel="${KLD_EPS_REL}" '
        BEGIN {
          noise = eabs + erel * a;
          if (ua + 0 > noise) noise = ua;
          if (ub + 0 > noise) noise = ub;
          exit !(b - a > noise);
        }'; then
      return 0
    fi
  done
  return 1
}

PASS_TURBO3=0
PASS_TURBO4=0
NOTE=""

if [[ "${#SERIES_TURBO3[@]}" -eq 3 ]] && [[ "${SERIES_TURBO3[*]}" != *"NA"* ]]; then
  if walks_up "${SERIES_TURBO3[@]}" -- "${UNC_TURBO3[@]}"; then
    NOTE+="turbo3 KLD walks up (${SERIES_TURBO3[*]}); "
  else
    PASS_TURBO3=1
    NOTE+="turbo3 KLD flat-or-down (${SERIES_TURBO3[*]}); "
  fi
else
  NOTE+="turbo3 series incomplete; "
fi

if [[ "${#SERIES_TURBO4[@]}" -eq 3 ]] && [[ "${SERIES_TURBO4[*]}" != *"NA"* ]]; then
  if walks_up "${SERIES_TURBO4[@]}" -- "${UNC_TURBO4[@]}"; then
    NOTE+="turbo4 KLD walks up (${SERIES_TURBO4[*]}); "
  else
    PASS_TURBO4=1
    NOTE+="turbo4 KLD flat-or-down (${SERIES_TURBO4[*]}); "
  fi
else
  NOTE+="turbo4 series incomplete; "
fi

if [[ "${ANY_FAIL}" -eq 0 && "${PASS_TURBO3}" -eq 1 && "${PASS_TURBO4}" -eq 1 ]]; then
  VERDICT=PASS
  RC=0
else
  VERDICT=FAIL
  RC=1
fi

{
  echo "VERDICT=${VERDICT}"
  echo "note=${NOTE}"
  echo "pass_rule=for each cand, KLD(4096),KLD(16384),KLD(65536) must not strictly increase"
  echo "table=${TABLE}"
} | tee -a "${SUMMARY}" | tee "${VERDICT_FILE}"

echo "T-066.5-KLD ${VERDICT} stamp=${OUT_DIR}"
exit "${RC}"
