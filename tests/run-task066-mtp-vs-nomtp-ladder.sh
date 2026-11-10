#!/usr/bin/env bash
# TASK-066 P3 / Phase B e2e — MTP vs nomtp on the Stage 1 fixture (Metal Terminal).
# Not in SUITE=followup. Same host knobs as run-task066-ubatch-ladder.sh + draft head.
#
# Gates (066.9 Phase B — fixed):
#   HARD: non-empty gen; MTP engagement evidence; no Metal OOM/assert.
#   PRODUCT sample mode: text DIFF vs nomtp is TELEMETRY (not FAIL) — p_min/n_max/sampling
#     exist precisely because MTP≠nomtp in general. Quality glance on duel prompt.
#   GREEDY sample mode: optional lossless A/B (temp=0); DIFF = HELP, still not auto-FAIL
#     unless TASK066_REQUIRE_IDENTICAL=1.
#
# Usage:
#   unset GGML_METAL_MUL_MM_MIN   # product default 0 (ne11>=1 on float-tile mul_mm)
#   BUILD_DIR=build-task066-cb-ws TASK066_SAMPLE_MODE=product TASK066_UBATCH_LIST=512 \
#     ./tests/run-task066-mtp-vs-nomtp-ladder.sh
#
# Env:
#   MAIN_GGUF / TASK066_MODEL — target GGUF (default .scratch ...UD-IQ4_XS)
#   DRAFT_GGUF / TASK066_DRAFT — MTP head (default .scratch ...mtp-...Q8_0 or $HOME/models)
#   TASK066_UBATCH_LIST — default "512" (product ubatch); M1 hist used "16"
#   TASK066_SPEC_DRAFT_N_MAX — default 2 (product; verify width N=3)
#   TASK066_SPEC_DRAFT_P_MIN — default 0.95 (product quality KEEP; greedy mode ignores)
#   TASK066_SAMPLE_MODE — product|greedy (default product)
#   TASK066_CELL_MODES — both|nomtp|mtp (default both). Phase C matrix uses nomtp once + mtp per knob.
#   TASK066_REQUIRE_IDENTICAL=1 — hard-fail on MTP vs nomtp DIFF (greedy lossless only)
#   TASK066_N_PREDICT, TASK066_CTX, TASK066_CRAM, TASK066_CTXCP, BATCH_* — same as ubatch ladder
#   DRY_RUN=1, REBUILD=1, TASK066_ALLOW_SHARED_METAL=1, TASK066_STOP_ON_FAIL=1
#
# Writes: logs/task066-mtp-vs-nomtp-<stamp>/{cells/,matrix.md,README.txt}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-mtp-vs-nomtp-${STAMP}}"
mkdir -p "${OUT_DIR}/cells"

CLI_BIN="${LLAMA_CLI:-${BUILD_DIR}/bin/llama-cli}"
SCRATCH_DIR="${REPO_ROOT}/.scratch/gemma-4-26b-a4b"
SCRATCH_MODEL="${SCRATCH_DIR}/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
MODEL="${MAIN_GGUF:-${TASK066_MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
  MODEL="${SCRATCH_MODEL}"
fi

resolve_draft() {
  local d="${DRAFT_GGUF:-${TASK066_DRAFT:-}}"
  if [[ -n "${d}" ]]; then
    echo "${d}"
    return
  fi
  local cand
  for cand in \
    "${SCRATCH_DIR}/mtp-gemma-4-26B-A4B-it-Q8_0.gguf" \
    "${SCRATCH_DIR}/gemma-assistant-mtp.gguf" \
    "${REPO_ROOT}/.scratch/gemma-assistant-mtp.gguf" \
    "${HOME}/models/mtp-gemma-4-26B-A4B-it-Q8_0.gguf" \
    "${HOME}/models/mtp-gemma-4-26B-A4B-it-F16.gguf"
  do
    if [[ -f "${cand}" ]]; then
      echo "${cand}"
      return
    fi
  done
  echo ""
}

DRAFT="$(resolve_draft)"
DEFAULT_PROMPT="${REPO_ROOT}/tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt"
PROMPT="${TASK066_PROMPT:-${DEFAULT_PROMPT}}"
N_PREDICT="${TASK066_N_PREDICT:-256}"
CTX="${TASK066_CTX:-3072}"
# Product default ubatch=512 (gemma4_26b_a4b_mtp_metal.yaml). Hist M1 used "16".
UBATCH_LIST="${TASK066_UBATCH_LIST:-512}"
BATCH_MODE="${TASK066_BATCH_MODE:-floor}"
BATCH_FLOOR="${TASK066_BATCH_FLOOR:-512}"
BATCH_MAX="${TASK066_BATCH_MAX:-2048}"
CTK="q8_0"
CTV="turbo4"
CRAM="${TASK066_CRAM:-0}"
CTXCP="${TASK066_CTXCP:-32}"
SPEC_N_MAX="${TASK066_SPEC_DRAFT_N_MAX:-2}"
SPEC_P_MIN="${TASK066_SPEC_DRAFT_P_MIN:-0.95}"
SAMPLE_MODE="${TASK066_SAMPLE_MODE:-product}"
CELL_MODES="${TASK066_CELL_MODES:-both}"
REQUIRE_IDENTICAL="${TASK066_REQUIRE_IDENTICAL:-0}"
STOP_ON_FAIL="${TASK066_STOP_ON_FAIL:-1}"
NGL="${TASK066_NGL:-99}"
THREADS="${TASK066_THREADS:-4}"
THREADS_BATCH="${TASK066_THREADS_BATCH:-4}"
TENSOR_SPLIT="${TASK069_TENSOR_SPLIT:-}"
MAIN_GPU="${TASK069_MAIN_GPU:-}"

fail() { echo "FAIL: $*" >&2; exit 2; }

if [[ "${TASK069_SUITE_PROFILE:-}" == "cuda" ]]; then
  CTK="${TASK069_CTK:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_CTK}"
  CTV="${TASK069_CTV:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_CTV}"
  CRAM="${TASK069_CRAM:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_CRAM}"
  CTXCP="${TASK069_CTXCP:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_CTXCP}"
  NGL="${TASK069_NGL:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_NGL}"
  THREADS="${TASK069_THREADS:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_THREADS}"
  THREADS_BATCH="${TASK069_THREADS_BATCH:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_THREADS_BATCH}"
  TENSOR_SPLIT="${TASK069_TENSOR_SPLIT:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_TENSOR_SPLIT}"
  MAIN_GPU="${TASK069_MAIN_GPU:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_MAIN_GPU}"
  BATCH_MAX="${TASK069_BATCH:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_BATCH}"
  BATCH_FLOOR="${TASK069_UBATCH:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_UBATCH}"
  SPEC_N_MAX="${TASK069_SPEC_DRAFT_N_MAX:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_SPEC_DRAFT_N_MAX}"
  SPEC_P_MIN="${TASK069_SPEC_DRAFT_P_MIN:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_SPEC_DRAFT_P_MIN}"
  echo "NOTE: TASK069 CUDA suite knobs: ctk=${CTK} ctv=${CTV} b_max=${BATCH_MAX} floor=${BATCH_FLOOR} ts=${TENSOR_SPLIT} mg=${MAIN_GPU}" >&2
fi

case "${SAMPLE_MODE}" in
  product|greedy) ;;
  *) fail "TASK066_SAMPLE_MODE must be product|greedy (got ${SAMPLE_MODE})" ;;
esac

case "${CELL_MODES}" in
  both|nomtp|mtp) ;;
  *) fail "TASK066_CELL_MODES must be both|nomtp|mtp (got ${CELL_MODES})" ;;
esac

[[ -f "${MODEL}" ]] || fail "model not found: ${MODEL}
hint: ./scripts/setup-gemma4-26b-a4b-models.sh"
[[ -f "${DRAFT}" ]] || fail "MTP draft not found (set DRAFT_GGUF / TASK066_DRAFT)
hint: ./scripts/setup-gemma4-26b-a4b-models.sh
hint: expected e.g. ${HOME}/models/mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
[[ -f "${PROMPT}" ]] || fail "prompt not a file: ${PROMPT}"
[[ -d "${BUILD_DIR}" ]] || fail "BUILD_DIR missing: ${BUILD_DIR}"

case "${BATCH_MODE}" in
  floor|max|equal) ;;
  *) fail "TASK066_BATCH_MODE must be floor|max|equal (got ${BATCH_MODE})" ;;
esac

preflight_exclusive_metal() {
  if [[ "${TASK066_ALLOW_SHARED_METAL:-0}" == "1" ]]; then
    echo "NOTE: TASK066_ALLOW_SHARED_METAL=1 — skipping exclusive-Metal preflight" >&2
    return 0
  fi
  local hits=""
  if command -v pgrep >/dev/null 2>&1; then
    hits="$(pgrep -fl 'llama-server|llama-cli' 2>/dev/null | grep -v " ${$}" || true)"
  fi
  if [[ -z "${hits}" ]]; then
    return 0
  fi
  if echo "${hits}" | grep -q 'llama-server'; then
    echo "FAIL: another llama-server is already running (holds Metal VRAM):" >&2
    echo "${hits}" | LC_ALL=C sed 's/^/  /' >&2
    echo "hint:   pkill -f 'llama-server.*gemma-4-26B'   # then re-run" >&2
    exit 2
  fi
}
preflight_exclusive_metal

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${CLI_BIN}" ]]; then
  echo "==> rebuild llama-cli"
  cmake --build "${BUILD_DIR}" --target llama-cli --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi
[[ -x "${CLI_BIN}" ]] || fail "missing ${CLI_BIN}"

batch_for() {
  local ub="$1"
  case "${BATCH_MODE}" in
    equal) echo "${ub}" ;;
    max) echo "${BATCH_MAX}" ;;
    floor)
      if [[ "${ub}" -gt "${BATCH_FLOOR}" ]]; then echo "${ub}"; else echo "${BATCH_FLOOR}"; fi
      ;;
  esac
}

cell_dir() {
  # mode=nomtp|mtp
  local ub="$1" mode="$2"
  printf '%s/cells/ub%04d-%s' "${OUT_DIR}" "${ub}" "${mode}"
}

# Same capture logic as run-task066-ubatch-ladder.sh (keep in sync).
# Strip interleaved -lv 3 server timing lines that otherwise pollute generation.txt
# (and make MTP↔nomtp DIFF look like timing noise).
extract_generation() {
  local logf="$1"
  local cleaned filtered
  cleaned="$(mktemp "${TMPDIR:-/tmp}/task066-gen.XXXXXX")"
  filtered="$(mktemp "${TMPDIR:-/tmp}/task066-genf.XXXXXX")"
  LC_ALL=C sed $'s/\x1b\\[[0-9;]*[A-Za-z]//g' "${logf}" > "${cleaned}"
  # Drop pure log lines; also strip mid-line "0.NN.NNN.NNN I slot print_timing:…" junk.
  # Progress lines from -lv 3 often splice mid-token into stdout under --simple-io.
  LC_ALL=C sed -E \
    -e '/^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ [IWE] /d' \
    -e 's/[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ [IWE] [^[:cntrl:]]*//g' \
    -e 's/n_decoded[[:space:]]*=[[:space:]]*[0-9]+,[[:space:]]*tg[[:space:]]*=[[:space:]]*[0-9.]+ t\/s,[[:space:]]*tg_3s[[:space:]]*=[[:space:]]*[0-9.]+ t\/s//g' \
    -e 's/prompt eval time[[:space:]]*=[[:space:]]*[0-9.]+ ms[^[:cntrl:]]*//g' \
    -e 's/eval time[[:space:]]*=[[:space:]]*[0-9.]+ ms[^[:cntrl:]]*//g' \
    -e 's/total time[[:space:]]*=[[:space:]]*[0-9.]+ ms[^[:cntrl:]]*//g' \
    -e 's/draft acceptance[[:space:]]*=[[:space:]]*[0-9.]+[^[:cntrl:]]*//g' \
    -e 's/graphs reused[[:space:]]*=[[:space:]]*[0-9]+//g' \
    -e 's/ns \([[:space:]]*[0-9.]+ ms per token[^)]*\)//g' \
    -e '/^ggml_/d' \
    -e '/^llama_/d' \
    "${cleaned}" > "${filtered}"

  if LC_ALL=C grep -E -q '^(Assistant:|assistant:)' "${filtered}"; then
    LC_ALL=C awk '/^(Assistant:|assistant:)/{p=1} p{print} /^\[ Prompt:/{exit}' "${filtered}"
    rm -f "${cleaned}" "${filtered}"
    return 0
  fi
  if LC_ALL=C grep -F -q '(truncated)' "${filtered}"; then
    LC_ALL=C awk '
      /\(truncated\)/ {p=1; next}
      /^\[ Prompt:/ {exit}
      /^Exiting\.\.\./ {exit}
      p && NF {print}
    ' "${filtered}"
    rm -f "${cleaned}" "${filtered}"
    return 0
  fi
  if LC_ALL=C grep -E -q '^> ' "${filtered}"; then
    LC_ALL=C awk '
      /^> / {p=1; next}
      /^\[ Prompt:/ {exit}
      /^Exiting\.\.\./ {exit}
      p && NF {print}
    ' "${filtered}"
    rm -f "${cleaned}" "${filtered}"
    return 0
  fi
  LC_ALL=C awk '
    /^\[ Prompt:/ {exit}
    /^Exiting\.\.\./ {exit}
    NF {print}
  ' "${filtered}"
  rm -f "${cleaned}" "${filtered}"
}

first_disagree() {
  local a="$1" b="$2"
  if [[ ! -f "${a}" || ! -f "${b}" ]]; then
    echo "missing"
    return
  fi
  if [[ ! -s "${a}" || ! -s "${b}" ]]; then
    echo "EMPTY_CAPTURE"
    return
  fi
  if cmp -s "${a}" "${b}"; then
    echo "identical"
    return
  fi
  python3 - "${a}" "${b}" <<'PY'
import sys
a, b = open(sys.argv[1], "rb").read(), open(sys.argv[2], "rb").read()
n = min(len(a), len(b))
i = 0
while i < n and a[i] == b[i]:
    i += 1
def snip(x, i, w=36):
    s = x[i:i+w].decode("utf-8", "replace").replace("\n", "\\n")
    return repr(s)
print(f"DIFF @byte {i} a={snip(a,i)} b={snip(b,i)}")
PY
}

run_cell() {
  local ub="$1" mode="$2"
  local b dir log gen
  b="$(batch_for "${ub}")"
  dir="$(cell_dir "${ub}" "${mode}")"
  mkdir -p "${dir}"
  log="${dir}/cli.log"
  gen="${dir}/generation.txt"

  local args=(
    -m "${MODEL}"
    -f "${PROMPT}"
    -c "${CTX}"
    -ngl "${NGL}"
    -t "${THREADS}"
    -tb "${THREADS_BATCH}"
    -b "${b}"
    -ub "${ub}"
    -fa on
    -ctk "${CTK}"
    -ctv "${CTV}"
    -cram "${CRAM}"
    -ctxcp "${CTXCP}"
    -n "${N_PREDICT}"
    --seed 42
    --reasoning off
    --jinja
    --chat-template-kwargs '{"enable_thinking":false}'
    --conversation
    --single-turn
    --simple-io
    --no-display-prompt
    --no-warmup
    # llama-cli defaults to ERROR (-lv 1); INFO needed to prove draft/MTP engaged.
    -lv 3
  )
  if [[ -n "${TENSOR_SPLIT}" ]]; then
    args+=(-ts "${TENSOR_SPLIT}")
  fi
  if [[ -n "${MAIN_GPU}" ]]; then
    args+=(-mg "${MAIN_GPU}")
  fi

  if [[ "${SAMPLE_MODE}" == "product" ]]; then
    # gemma4_26b_a4b_mtp_metal.yaml sampling (Unsloth 1.0/0.95/64).
    args+=(--temp 1.0 --top-p 0.95 --top-k 64)
  else
    args+=(--temp 0 --top-k 1 --top-p 1.0)
  fi

  if [[ "${mode}" == "mtp" ]]; then
    args+=(
      --mtp-head "${DRAFT}"
      --spec-type mtp
      --spec-draft-n-max "${SPEC_N_MAX}"
      --spec-draft-ngl "${NGL}"
      -ctkd "${CTK}"
      -ctvd "${CTV}"
    )
    if [[ "${SAMPLE_MODE}" == "product" ]]; then
      args+=(--spec-draft-p-min "${SPEC_P_MIN}")
    fi
  fi

  {
    echo "ubatch=${ub}"
    echo "mode=${mode}"
    echo "batch=${b}"
    echo "batch_mode=${BATCH_MODE}"
    echo "sample_mode=${SAMPLE_MODE}"
    echo "ctk=${CTK} ctv=${CTV}"
    echo "cram=${CRAM} ctxcp=${CTXCP} ctx=${CTX}"
    echo "draft=${DRAFT}"
    echo "spec_draft_n_max=${SPEC_N_MAX}"
    echo "spec_draft_p_min=${SPEC_P_MIN}"
    echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN:-<unset:default 0>}"
    echo "cmd=${CLI_BIN} ${args[*]}"
  } > "${dir}/meta.txt"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN skip ub=${ub} mode=${mode}" | tee "${dir}/dry_run.txt"
    printf '(dry-run stub ubatch=%s mode=%s)\n' "${ub}" "${mode}" > "${gen}"
    echo "exit=0" > "${dir}/exit.txt"
    if [[ "${mode}" == "mtp" ]]; then
      echo "draft acceptance = 0.00000 (dry-run stub)" > "${dir}/mtp-evidence.txt"
    fi
    return 0
  fi

  echo "==> ubatch=${ub} mode=${mode} batch=${b} ctk=${CTK} ctv=${CTV}"
  set +e
  "${CLI_BIN}" "${args[@]}" < /dev/null > "${log}" 2>&1
  local rc=$?
  set -e

  if grep -E -q 'Insufficient Memory|Compute error\.|GGML_ASSERT' "${log}" 2>/dev/null; then
    if [[ "${rc}" -eq 0 ]]; then
      rc=75
    fi
  fi

  extract_generation "${log}" > "${gen}"
  if [[ ! -s "${gen}" ]]; then
    echo "FAIL: empty generation.txt for ub=${ub} mode=${mode} (see ${log})" >&2
    if [[ "${rc}" -eq 0 ]]; then
      rc=76
    fi
  fi

  # Fail-closed: MTP cell must show speculative engagement (hidden at default -lv 1).
  if [[ "${mode}" == "mtp" && "${rc}" -eq 0 ]]; then
    if ! LC_ALL=C grep -E -q 'draft acceptance|\[spec\]|draft-mtp|adding speculative|spec_model|n_draft' "${log}"; then
      echo "FAIL: MTP cell has no speculative evidence in cli.log (flags may have been ignored)" >&2
      echo "hint: expect 'draft acceptance =' or '[spec]' with -lv 3; see ${log}" >&2
      rc=77
    else
      # Save a short proof line for the matrix.
      LC_ALL=C grep -E 'draft acceptance|\[spec\]|draft-mtp|adding speculative' "${log}" \
        | head -n 5 > "${dir}/mtp-evidence.txt" || true
    fi
  fi
  echo "exit=${rc}" > "${dir}/exit.txt"

  if [[ "${rc}" -ne 0 ]]; then
    echo "FAIL: llama-cli exit ${rc} for ub=${ub} mode=${mode} (see ${log})" >&2
    if [[ "${STOP_ON_FAIL}" == "1" ]]; then
      exit "${rc}"
    fi
  fi
  return 0
}

{
  echo "TASK-066 MTP vs nomtp ladder (P3 / Phase B e2e)"
  echo "stamp=${STAMP}"
  echo "fixture=tests/TASK-066-ladder-fixture.md"
  echo "gate=quality+evidence (product); DIFF vs nomtp is telemetry unless REQUIRE_IDENTICAL=1"
  echo "model=${MODEL}"
  echo "draft=${DRAFT}"
  echo "prompt=${PROMPT}"
  echo "n_predict=${N_PREDICT}"
  echo "ctx=${CTX}"
  echo "ctk=${CTK} ctv=${CTV}"
  echo "cram=${CRAM} ctxcp=${CTXCP}"
  echo "batch_mode=${BATCH_MODE} batch_floor=${BATCH_FLOOR}"
  echo "ubatch_list=${UBATCH_LIST}"
  echo "sample_mode=${SAMPLE_MODE}"
  echo "cell_modes=${CELL_MODES}"
  echo "spec_draft_n_max=${SPEC_N_MAX}"
  echo "spec_draft_p_min=${SPEC_P_MIN}"
  echo "require_identical=${REQUIRE_IDENTICAL}"
  echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN:-<unset:default 0>}"
  echo "cli=${CLI_BIN}"
  echo "build_dir=${BUILD_DIR}"
} | tee "${OUT_DIR}/README.txt"

env | grep -E '^(TASK066_|DRAFT_GGUF|MAIN_GGUF|GGML_METAL_MUL_MM_MIN|BUILD_DIR)=' | sort > "${OUT_DIR}/env.txt" || true

if [[ -z "${GGML_METAL_MUL_MM_MIN:-}" ]]; then
  echo "NOTE: product default MUL_MM_MIN=0 (ne11>=1 on float-tile mul_mm). Legacy A2=1 / 7 / 32." >&2
fi

for ub in ${UBATCH_LIST}; do
  if [[ "${CELL_MODES}" == "both" || "${CELL_MODES}" == "nomtp" ]]; then
    run_cell "${ub}" nomtp
  fi
  if [[ "${CELL_MODES}" == "both" || "${CELL_MODES}" == "mtp" ]]; then
    run_cell "${ub}" mtp
  fi
done

MATRIX="${OUT_DIR}/matrix.md"
DIFF_N=0
IDENTICAL_N=0
{
  echo "# TASK-066 MTP vs nomtp matrix"
  echo
  echo "stamp: ${STAMP}"
  echo "sample_mode: ${SAMPLE_MODE}"
  echo "MUL_MM_MIN: ${GGML_METAL_MUL_MM_MIN:-<unset:default 0>}"
  echo "draft: ${DRAFT}"
  echo "spec_draft_n_max: ${SPEC_N_MAX}"
  echo "spec_draft_p_min: ${SPEC_P_MIN}"
  echo "gate: hard=engagement+non-empty; MTP↔nomtp DIFF is telemetry in product mode"
  echo
  echo "## Cells"
  echo
  echo "| cell | batch | gen |"
  echo "|------|-------|-----|"
  for ub in ${UBATCH_LIST}; do
    b="$(batch_for "${ub}")"
    if [[ "${CELL_MODES}" == "both" || "${CELL_MODES}" == "nomtp" ]]; then
      echo "| S${ub} nomtp | ${b} | \`cells/ub$(printf '%04d' "${ub}")-nomtp/generation.txt\` |"
    fi
    if [[ "${CELL_MODES}" == "both" || "${CELL_MODES}" == "mtp" ]]; then
      echo "| S${ub} mtp | ${b} | \`cells/ub$(printf '%04d' "${ub}")-mtp/generation.txt\` |"
    fi
  done
  echo
  echo "## Auto first-disagree (MTP vs nomtp) — telemetry"
  echo
  echo "| compare | result |"
  echo "|---------|--------|"
  if [[ "${CELL_MODES}" == "both" ]]; then
    for ub in ${UBATCH_LIST}; do
      g_n="$(cell_dir "${ub}" nomtp)/generation.txt"
      g_m="$(cell_dir "${ub}" mtp)/generation.txt"
      fd="$(first_disagree "${g_m}" "${g_n}")"
      echo "| S${ub} mtp vs nomtp | ${fd} |"
      if [[ "${fd}" == "identical" ]]; then
        IDENTICAL_N=$((IDENTICAL_N + 1))
      elif [[ "${fd}" == DIFF* ]]; then
        DIFF_N=$((DIFF_N + 1))
      fi
    done
    # Cross-ubatch nomtp reference if both present
    if echo " ${UBATCH_LIST} " | grep -q ' 8 ' && echo " ${UBATCH_LIST} " | grep -q ' 16 '; then
      echo "| S16 nomtp vs S8 nomtp | $(first_disagree "$(cell_dir 16 nomtp)/generation.txt" "$(cell_dir 8 nomtp)/generation.txt") |"
    fi
  else
    echo "| (skipped) | CELL_MODES=${CELL_MODES} — cross-mode DIFF deferred to Phase C orchestrator |"
  fi
  echo
  echo "## Capture check"
  echo
  empty_n=0
  for ub in ${UBATCH_LIST}; do
    for mode in nomtp mtp; do
      if [[ "${CELL_MODES}" == "nomtp" && "${mode}" != "nomtp" ]]; then
        continue
      fi
      if [[ "${CELL_MODES}" == "mtp" && "${mode}" != "mtp" ]]; then
        continue
      fi
      g="$(cell_dir "${ub}" "${mode}")/generation.txt"
      if [[ ! -s "${g}" ]]; then
        echo "- EMPTY S${ub} ${mode}"
        empty_n=$((empty_n + 1))
      fi
    done
  done
  if [[ "${empty_n}" -eq 0 ]]; then
    echo "- all generation.txt non-empty"
  else
    echo "- FAIL: ${empty_n} empty generation file(s)"
  fi
  echo
  echo "## MTP engagement"
  echo
  miss_ev=0
  if [[ "${CELL_MODES}" == "both" || "${CELL_MODES}" == "mtp" ]]; then
    for ub in ${UBATCH_LIST}; do
      ev="$(cell_dir "${ub}" mtp)/mtp-evidence.txt"
      if [[ -s "${ev}" ]]; then
        echo "- S${ub} mtp: OK (\`cells/ub$(printf '%04d' "${ub}")-mtp/mtp-evidence.txt\`)"
      else
        echo "- S${ub} mtp: MISSING evidence"
        miss_ev=$((miss_ev + 1))
      fi
    done
  else
    echo "- (skipped) CELL_MODES=nomtp"
  fi
  echo
  echo "## Quality glance (duel KEEP phrases; not bit-exact)"
  echo
  # ST-3.2l-2q KEEP: Pierre hits Dolokhov + Dolokhov misses. Soft signal (not bit-exact phrases).
  for ub in ${UBATCH_LIST}; do
    for mode in nomtp mtp; do
      if [[ "${CELL_MODES}" == "nomtp" && "${mode}" != "nomtp" ]]; then
        continue
      fi
      if [[ "${CELL_MODES}" == "mtp" && "${mode}" != "mtp" ]]; then
        continue
      fi
      g="$(cell_dir "${ub}" "${mode}")/generation.txt"
      hits=0
      # UTF-8 Cyrillic — do not force LC_ALL=C (breaks casefold / matches).
      if grep -Eqi 'левый бок|попадает в [Дд]олохов|ранен(ый|ого) [Дд]олохов' "${g}" 2>/dev/null; then
        hits=$((hits + 1))
      fi
      if grep -Eqi 'стреляет мимо|промахивается' "${g}" 2>/dev/null; then
        hits=$((hits + 1))
      fi
      echo "- S${ub} ${mode}: duel_hits=${hits}/2 (hit Dolokhov / Dolokhov misses)"
    done
  done
  echo
  echo "## Operator notes"
  echo
  echo "- sample_mode=${SAMPLE_MODE}; DIFF_n=${DIFF_N}; identical_n=${IDENTICAL_N}"
  echo "- product: engagement OK + non-empty gen => PASS; DIFF vs nomtp expected/OK."
  echo "- greedy+REQUIRE_IDENTICAL=1: DIFF => FAIL (lossless A/B only)."
  echo "- engagement missing => INVALID stamp."
  echo "- Product ubatch pin: TASK066_UBATCH_LIST=512 (or ladder 8 256 512 1024)."
} > "${MATRIX}"

echo "OK: MTP vs nomtp written to ${OUT_DIR}"
echo "    matrix: ${MATRIX}"
if grep -q 'EMPTY S\|EMPTY_CAPTURE\|FAIL: .*empty' "${MATRIX}" 2>/dev/null; then
  echo "FAIL: empty generation capture" >&2
  exit 76
fi
if grep -q 'MISSING evidence' "${MATRIX}" 2>/dev/null; then
  echo "FAIL: MTP engagement evidence missing" >&2
  exit 77
fi
if [[ "${REQUIRE_IDENTICAL}" == "1" && "${DIFF_N}" -gt 0 ]]; then
  echo "FAIL: TASK066_REQUIRE_IDENTICAL=1 and MTP vs nomtp DIFF count=${DIFF_N}" >&2
  exit 1
fi
echo "VERDICT=PASS (engagement+capture; sample_mode=${SAMPLE_MODE}; DIFF telemetry=${DIFF_N})"
exit 0
