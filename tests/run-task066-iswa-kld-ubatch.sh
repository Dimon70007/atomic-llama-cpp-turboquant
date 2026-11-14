#!/usr/bin/env bash
# T-066.5-KLD-UB — same CTX, sweep -ub (KLD analogue of C3 + ubatch ladder).
# Fail-closed: MODEL, BUILD_DIR, CTX, N_BATCH, FA_MODE, UBATCH_LIST. No silent defaults.
#
# Usage (from fork root):
#   MODEL=path.gguf BUILD_DIR=build-task066-cb-ws CTX=4096 N_BATCH=2048 \
#   FA_MODE=on UBATCH_LIST="256 512 1024" \
#     ./tests/run-task066-iswa-kld-ubatch.sh [OUT_DIR]
#
# Tables:
#   1) vs-f16:   for each ub, KLD(q8_0/cand @ ub  vs  f16/f16 @ same ub)
#   2) cross-ub: REF_UBATCH saves logits with cand KV; other ubs KLD against that base
#                (C3-class: only -ub changes; bit-exact => KLD~0)
#
# Optional:
#   TEXT=wikitext-2-raw/wiki.test.raw
#   CAND_LIST="turbo3 turbo4"
#   REF_UBATCH=<max of UBATCH_LIST>   # oneshot / large-ub reference for cross-ub
#   NGL=99  KEEP_LOGITS=0
#   CHUNKS=6                          # wikitext chunks per run. Minimum 6: below that the
#                                     # +-unc dominates and both ppl_ratio and the KLD point
#                                     # estimate stop resolving (TASK-071 ST-7.6). The sweep
#                                     # cost scales linearly with CHUNKS.
#   ALLOW_NON128_UB=0|1               # default 0: refuse ub % 128 != 0 (turbo block)
#
# Stamp: logs/task066-iswa-kld-ubatch-<UTC>/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-}"
MODEL="${MODEL:-}"
CTX="${CTX:-}"
N_BATCH="${N_BATCH:-}"
FA_MODE="${FA_MODE:-}"
UBATCH_LIST="${UBATCH_LIST:-}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-iswa-kld-ubatch-$(date -u +%Y%m%dT%H%M%SZ)}"
TEXT="${TEXT:-${REPO_ROOT}/wikitext-2-raw/wiki.test.raw}"
CAND_LIST="${CAND_LIST:-turbo3 turbo4}"
REF_UBATCH="${REF_UBATCH:-}"
NGL="${NGL:-99}"
CHUNKS="${CHUNKS:-6}"
KEEP_LOGITS="${KEEP_LOGITS:-0}"
ALLOW_NON128_UB="${ALLOW_NON128_UB:-0}"
FA_REQUIRED_CTX_MIN=8000
BATCH_EQ_CTX_MAX=2000

die() { echo "FAIL: $*" >&2; exit 2; }

[[ -n "${BUILD_DIR}" ]] || die "missing required BUILD_DIR"
[[ -n "${MODEL}" ]] || die "missing required MODEL=/path/to.gguf"
[[ -n "${CTX}" ]] || die "missing required CTX (same context for all ub)"
[[ -n "${N_BATCH}" ]] || die "missing required N_BATCH (product gemma4 uses 2048)"
[[ -n "${FA_MODE}" ]] || die "missing required FA_MODE (must be on)"
[[ -n "${UBATCH_LIST}" ]] || die "missing required UBATCH_LIST e.g. \"256 512 1024\""
[[ "${FA_MODE}" == "on" ]] || die "FA_MODE=${FA_MODE}: refuse (ideas + product require flash_attn on)"
[[ -f "${MODEL}" ]] || die "MODEL not a file: ${MODEL}"
[[ -f "${TEXT}" ]] || die "TEXT not a file: ${TEXT}"
[[ -d "${BUILD_DIR}" ]] || die "BUILD_DIR missing: ${BUILD_DIR}"
[[ "${CTX}" =~ ^[0-9]+$ ]] && [[ "${CTX}" -ge 4 ]] || die "CTX must be int >= 4 (got ${CTX})"
[[ "${N_BATCH}" =~ ^[0-9]+$ ]] && [[ "${N_BATCH}" -ge 1 ]] || die "N_BATCH must be int >= 1"

if [[ "${N_BATCH}" -eq "${CTX}" && "${CTX}" -gt "${BATCH_EQ_CTX_MAX}" ]]; then
  die "refusing -b=CTX for CTX=${CTX} > ${BATCH_EQ_CTX_MAX}; use N_BATCH=2048"
fi
if [[ "${CTX}" -ge "${FA_REQUIRED_CTX_MIN}" && "${FA_MODE}" != "on" ]]; then
  die "CTX=${CTX} >= ${FA_REQUIRED_CTX_MIN} requires FA_MODE=on"
fi

# CHUNKS>=6 is the measurement floor (TASK-071 ST-7.6): fewer chunks and the +-unc
# exceeds the effect being measured. ALLOW_FEW_CHUNKS=1 is the explicit smoke opt-out.
if ! [[ "${CHUNKS}" =~ ^[0-9]+$ ]]; then
  die "CHUNKS must be a positive integer (got '${CHUNKS}')"
fi
if [[ "${CHUNKS}" -lt 6 ]]; then
  if [[ "${ALLOW_FEW_CHUNKS:-0}" == "1" ]]; then
    echo "WARN: CHUNKS=${CHUNKS} < 6 (ALLOW_FEW_CHUNKS=1) - measurement will not resolve" >&2
  else
    die "CHUNKS=${CHUNKS} < 6: too few chunks to resolve a difference (TASK-071 ST-7.6); set ALLOW_FEW_CHUNKS=1 for a smoke run only"
  fi
fi

declare -a UBS=()
for ub in ${UBATCH_LIST}; do
  [[ "${ub}" =~ ^[0-9]+$ ]] && [[ "${ub}" -ge 1 ]] || die "bad ubatch in UBATCH_LIST: ${ub}"
  if [[ "${ALLOW_NON128_UB}" != "1" ]] && (( ub % 128 != 0 )); then
    die "ubatch=${ub} not multiple of 128 (turbo KV block); set ALLOW_NON128_UB=1 only for diag"
  fi
  UBS+=("${ub}")
done
[[ "${#UBS[@]}" -ge 2 ]] || die "UBATCH_LIST needs >= 2 values (got ${#UBS[@]})"

if [[ -z "${REF_UBATCH}" ]]; then
  REF_UBATCH="${UBS[0]}"
  for ub in "${UBS[@]}"; do
    if [[ "${ub}" -gt "${REF_UBATCH}" ]]; then
      REF_UBATCH="${ub}"
    fi
  done
fi
[[ "${REF_UBATCH}" =~ ^[0-9]+$ ]] || die "REF_UBATCH must be int (got ${REF_UBATCH})"
REF_IN_LIST=0
for ub in "${UBS[@]}"; do
  if [[ "${ub}" -eq "${REF_UBATCH}" ]]; then
    REF_IN_LIST=1
    break
  fi
done
[[ "${REF_IN_LIST}" -eq 1 ]] || die "REF_UBATCH=${REF_UBATCH} not in UBATCH_LIST"

PPL_BIN="${BUILD_DIR}/bin/llama-perplexity"
if [[ ! -x "${PPL_BIN}" ]]; then
  echo "==> rebuild llama-perplexity"
  cmake --build "${BUILD_DIR}" --target llama-perplexity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
fi
[[ -x "${PPL_BIN}" ]] || die "missing ${PPL_BIN}"

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.txt"
TABLE_F16="${OUT_DIR}/kld_vs_f16.tsv"
TABLE_XUB="${OUT_DIR}/kld_cross_ub.tsv"
VERDICT_FILE="${OUT_DIR}/verdict.txt"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "model=${MODEL}"
  echo "text=${TEXT}"
  echo "ctx=${CTX}"
  echo "n_batch=${N_BATCH}"
  echo "fa=${FA_MODE}"
  echo "ubatch_list=${UBATCH_LIST}"
  echo "ref_ubatch=${REF_UBATCH}"
  echo "cand_list=${CAND_LIST}"
  echo "ngl=${NGL}"
  echo "chunks=${CHUNKS}"
  echo "allow_non128_ub=${ALLOW_NON128_UB}"
  echo "size_swa_formula=PAD(min(size_base, pad128(n_swa)*(unified?n_seq_max:1)+n_ubatch),512)"
  echo "note=vs-f16: cand@ub vs f16@same-ub; cross-ub: cand@ub vs cand@REF_UBATCH (C3-class)"
} | tee "${SUMMARY}"

printf 'cand\tub\tkld_mean\tkld_unc\tppl_ratio\tlog\n' > "${TABLE_F16}"
printf 'cand\tref_ub\tub\tkld_mean\tkld_unc\tppl_ratio\tlog\n' > "${TABLE_XUB}"

parse_kld_mean() {
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

# Return 0 if mean is exactly 0 (C3-class bit-exact via KLD tooling).
kld_is_zero() {
  local mean="$1"
  [[ "${mean}" != "NA" ]] || return 1
  awk -v m="${mean}" 'BEGIN { exit !(m == 0) }'
}

ANY_FAIL=0
CROSS_FAIL=0
CROSS_NOTE=""

# --- Table 1: vs-f16 per ub -------------------------------------------------
echo "==> phase vs-f16 (same CTX=${CTX}, sweep ub)" | tee -a "${SUMMARY}"
for ub in "${UBS[@]}"; do
  BASE_LOGITS="${OUT_DIR}/base-f16-c${CTX}-ub${ub}.kld"
  BASE_TAG="base-f16-c${CTX}-ub${ub}"
  echo "==> CTX=${CTX} ub=${ub} save FP16 logits -> ${BASE_LOGITS}"
  if ! run_ppl "${BASE_TAG}" \
      -m "${MODEL}" -f "${TEXT}" -c "${CTX}" -ngl "${NGL}" \
      -b "${N_BATCH}" -ub "${ub}" \
      -ctk f16 -ctv f16 -fa "${FA_MODE}" \
      --chunks "${CHUNKS}" \
      --save-all-logits "${BASE_LOGITS}"; then
    echo "FAIL: base logits save CTX=${CTX} ub=${ub}" | tee -a "${SUMMARY}"
    ANY_FAIL=1
    continue
  fi
  [[ -f "${BASE_LOGITS}" ]] || { echo "FAIL: missing ${BASE_LOGITS}" | tee -a "${SUMMARY}"; ANY_FAIL=1; continue; }

  for CAND in ${CAND_LIST}; do
    TAG="vsf16-q8_${CAND}-c${CTX}-ub${ub}"
    if ! run_ppl "${TAG}" \
        -m "${MODEL}" -f "${TEXT}" -c "${CTX}" -ngl "${NGL}" \
        -b "${N_BATCH}" -ub "${ub}" \
        -ctk q8_0 -ctv "${CAND}" -fa "${FA_MODE}" \
        --chunks "${CHUNKS}" \
        --kl-divergence-base "${BASE_LOGITS}" \
        --kl-divergence; then
      echo "FAIL: vs-f16 KLD ${TAG}" | tee -a "${SUMMARY}"
      ANY_FAIL=1
      printf '%s\t%s\tNA\tNA\tNA\t%s\n' "${CAND}" "${ub}" "${OUT_DIR}/${TAG}.log" >> "${TABLE_F16}"
      continue
    fi
    read -r KLD_MEAN KLD_UNC <<<"$(parse_kld_mean "${OUT_DIR}/${TAG}.log")"
    PPL_RATIO="$(grep -E 'Mean PPL\(Q\)/PPL\(base\)' "${OUT_DIR}/${TAG}.log" | tail -1 | awk '{print $(NF-2)}' || echo NA)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${CAND}" "${ub}" "${KLD_MEAN}" "${KLD_UNC}" "${PPL_RATIO}" "${OUT_DIR}/${TAG}.log" >> "${TABLE_F16}"
    echo "KLD vs-f16 ${CAND}@ub${ub} mean=${KLD_MEAN}" | tee -a "${SUMMARY}"
  done

  if [[ "${KEEP_LOGITS}" != "1" ]]; then
    rm -f "${BASE_LOGITS}"
  fi
done

# --- Table 2: cross-ub (C3-class) -------------------------------------------
echo "==> phase cross-ub (ref_ub=${REF_UBATCH}, CTX=${CTX})" | tee -a "${SUMMARY}"
for CAND in ${CAND_LIST}; do
  REF_LOGITS="${OUT_DIR}/cross-q8_${CAND}-c${CTX}-ub${REF_UBATCH}.kld"
  REF_TAG="cross-base-q8_${CAND}-c${CTX}-ub${REF_UBATCH}"
  echo "==> save cand logits ref ub=${REF_UBATCH} ctv=${CAND}"
  if ! run_ppl "${REF_TAG}" \
      -m "${MODEL}" -f "${TEXT}" -c "${CTX}" -ngl "${NGL}" \
      -b "${N_BATCH}" -ub "${REF_UBATCH}" \
      -ctk q8_0 -ctv "${CAND}" -fa "${FA_MODE}" \
      --chunks "${CHUNKS}" \
      --save-all-logits "${REF_LOGITS}"; then
    echo "FAIL: cross-ub base save cand=${CAND} ub=${REF_UBATCH}" | tee -a "${SUMMARY}"
    ANY_FAIL=1
    CROSS_FAIL=1
    CROSS_NOTE+="${CAND}: ref save failed; "
    continue
  fi
  [[ -f "${REF_LOGITS}" ]] || { ANY_FAIL=1; CROSS_FAIL=1; continue; }

  for ub in "${UBS[@]}"; do
    if [[ "${ub}" -eq "${REF_UBATCH}" ]]; then
      # Self-check: same ub vs own logits => KLD must be 0.
      TAG="cross-q8_${CAND}-c${CTX}-ub${ub}-vs-ub${REF_UBATCH}-self"
    else
      TAG="cross-q8_${CAND}-c${CTX}-ub${ub}-vs-ub${REF_UBATCH}"
    fi
    if ! run_ppl "${TAG}" \
        -m "${MODEL}" -f "${TEXT}" -c "${CTX}" -ngl "${NGL}" \
        -b "${N_BATCH}" -ub "${ub}" \
        -ctk q8_0 -ctv "${CAND}" -fa "${FA_MODE}" \
        --chunks "${CHUNKS}" \
        --kl-divergence-base "${REF_LOGITS}" \
        --kl-divergence; then
      echo "FAIL: cross-ub KLD ${TAG}" | tee -a "${SUMMARY}"
      ANY_FAIL=1
      CROSS_FAIL=1
      printf '%s\t%s\t%s\tNA\tNA\tNA\t%s\n' \
        "${CAND}" "${REF_UBATCH}" "${ub}" "${OUT_DIR}/${TAG}.log" >> "${TABLE_XUB}"
      continue
    fi
    read -r KLD_MEAN KLD_UNC <<<"$(parse_kld_mean "${OUT_DIR}/${TAG}.log")"
    PPL_RATIO="$(grep -E 'Mean PPL\(Q\)/PPL\(base\)' "${OUT_DIR}/${TAG}.log" | tail -1 | awk '{print $(NF-2)}' || echo NA)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${CAND}" "${REF_UBATCH}" "${ub}" "${KLD_MEAN}" "${KLD_UNC}" "${PPL_RATIO}" \
      "${OUT_DIR}/${TAG}.log" >> "${TABLE_XUB}"
    echo "KLD cross-ub ${CAND} ub${ub} vs ub${REF_UBATCH} mean=${KLD_MEAN}" | tee -a "${SUMMARY}"

    if ! kld_is_zero "${KLD_MEAN}"; then
      CROSS_FAIL=1
      CROSS_NOTE+="${CAND}: ub${ub} vs ub${REF_UBATCH} KLD=${KLD_MEAN} (want 0); "
    fi
  done

  if [[ "${KEEP_LOGITS}" != "1" ]]; then
    rm -f "${REF_LOGITS}"
  fi
done

if [[ "${ANY_FAIL}" -eq 0 && "${CROSS_FAIL}" -eq 0 ]]; then
  VERDICT=PASS
  RC=0
else
  VERDICT=FAIL
  RC=1
fi

{
  echo "VERDICT=${VERDICT}"
  echo "note=${CROSS_NOTE:-cross-ub KLD all zero vs REF_UBATCH}"
  echo "pass_rule=cross-ub: for each cand and each ub in UBATCH_LIST, KLD(cand@ub vs cand@REF_UBATCH) == 0"
  echo "table_vs_f16=${TABLE_F16}"
  echo "table_cross_ub=${TABLE_XUB}"
} | tee -a "${SUMMARY}" | tee "${VERDICT_FILE}"

echo "T-066.5-KLD-UB ${VERDICT} stamp=${OUT_DIR}"
exit "${RC}"
