#!/usr/bin/env bash
# TASK-066 P3 / M1 — product MTP vs nomtp on the Stage 1 fixture (Metal Terminal).
# Not in SUITE=followup. Same knobs as run-task066-ubatch-ladder.sh + draft head.
#
# Usage:
#   unset GGML_METAL_MUL_MM_MIN   # product default 0 (ne11>=1 on float-tile mul_mm)
#   BUILD_DIR=build-task066-cb ./tests/run-task066-mtp-vs-nomtp-ladder.sh
#
# Env:
#   MAIN_GGUF / TASK066_MODEL — target GGUF (default .scratch ...UD-IQ4_XS)
#   DRAFT_GGUF / TASK066_DRAFT — MTP head (default .scratch ...mtp-...Q8_0 or $HOME/models)
#   TASK066_UBATCH_LIST — default "16" (M1 fast); use "8 16" for M4 arbiter later
#   TASK066_SPEC_DRAFT_N_MAX — default 2 (= block size 3 - 1; same as run-gemma4-mtp-server)
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
# M1 default: ubatch=16 (product pin 32 => mul_mv). Override for M4: "8" or "8 16".
UBATCH_LIST="${TASK066_UBATCH_LIST:-16}"
BATCH_MODE="${TASK066_BATCH_MODE:-floor}"
BATCH_FLOOR="${TASK066_BATCH_FLOOR:-512}"
BATCH_MAX="${TASK066_BATCH_MAX:-2048}"
CTK="q8_0"
CTV="turbo4"
CRAM="${TASK066_CRAM:-0}"
CTXCP="${TASK066_CTXCP:-32}"
SPEC_N_MAX="${TASK066_SPEC_DRAFT_N_MAX:-2}"
STOP_ON_FAIL="${TASK066_STOP_ON_FAIL:-1}"

fail() { echo "FAIL: $*" >&2; exit 2; }

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
extract_generation() {
  local logf="$1"
  local cleaned
  cleaned="$(mktemp "${TMPDIR:-/tmp}/task066-gen.XXXXXX")"
  LC_ALL=C sed $'s/\x1b\\[[0-9;]*[A-Za-z]//g' "${logf}" > "${cleaned}"

  if LC_ALL=C grep -E -q '^(Assistant:|assistant:)' "${cleaned}"; then
    LC_ALL=C awk '/^(Assistant:|assistant:)/{p=1} p{print} /^\[ Prompt:/{exit}' "${cleaned}"
    rm -f "${cleaned}"
    return 0
  fi
  if LC_ALL=C grep -F -q '(truncated)' "${cleaned}"; then
    LC_ALL=C awk '
      /\(truncated\)/ {p=1; next}
      /^\[ Prompt:/ {exit}
      /^Exiting\.\.\./ {exit}
      p && NF {print}
    ' "${cleaned}"
    rm -f "${cleaned}"
    return 0
  fi
  if LC_ALL=C grep -E -q '^> ' "${cleaned}"; then
    LC_ALL=C awk '
      /^> / {p=1; next}
      /^\[ Prompt:/ {exit}
      /^Exiting\.\.\./ {exit}
      p && NF {print}
    ' "${cleaned}"
    rm -f "${cleaned}"
    return 0
  fi
  LC_ALL=C awk '
    /^\[ Prompt:/ {exit}
    /^Exiting\.\.\./ {exit}
    NF {print}
  ' "${cleaned}"
  rm -f "${cleaned}"
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
    -ngl 99
    -t 4
    -tb 4
    -b "${b}"
    -ub "${ub}"
    -fa on
    -ctk "${CTK}"
    -ctv "${CTV}"
    -cram "${CRAM}"
    -ctxcp "${CTXCP}"
    -n "${N_PREDICT}"
    --temp 0
    --top-k 1
    --seed 42
    --reasoning off
    --jinja
    --conversation
    --single-turn
    --simple-io
    --no-display-prompt
    --no-warmup
    # llama-cli defaults to ERROR (-lv 1); INFO needed to prove draft/MTP engaged.
    -lv 3
  )

  if [[ "${mode}" == "mtp" ]]; then
    args+=(
      --mtp-head "${DRAFT}"
      --spec-type mtp
      --spec-draft-n-max "${SPEC_N_MAX}"
      --spec-draft-ngl 99
      -ctkd "${CTK}"
      -ctvd "${CTV}"
    )
  fi

  {
    echo "ubatch=${ub}"
    echo "mode=${mode}"
    echo "batch=${b}"
    echo "batch_mode=${BATCH_MODE}"
    echo "ctk=${CTK} ctv=${CTV}"
    echo "cram=${CRAM} ctxcp=${CTXCP} ctx=${CTX}"
    echo "draft=${DRAFT}"
    echo "spec_draft_n_max=${SPEC_N_MAX}"
    echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN:-<unset:default 0>}"
    echo "cmd=${CLI_BIN} ${args[*]}"
  } > "${dir}/meta.txt"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN skip ub=${ub} mode=${mode}" | tee "${dir}/dry_run.txt"
    printf '(dry-run stub ubatch=%s mode=%s)\n' "${ub}" "${mode}" > "${gen}"
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
  echo "TASK-066 MTP vs nomtp ladder (P3/M1)"
  echo "stamp=${STAMP}"
  echo "fixture=tests/TASK-066-ladder-fixture.md"
  echo "model=${MODEL}"
  echo "draft=${DRAFT}"
  echo "prompt=${PROMPT}"
  echo "n_predict=${N_PREDICT}"
  echo "ctx=${CTX}"
  echo "ctk=${CTK} ctv=${CTV}"
  echo "cram=${CRAM} ctxcp=${CTXCP}"
  echo "batch_mode=${BATCH_MODE} batch_floor=${BATCH_FLOOR}"
  echo "ubatch_list=${UBATCH_LIST}"
  echo "spec_draft_n_max=${SPEC_N_MAX}"
  echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN:-<unset:default 0>}"
  echo "cli=${CLI_BIN}"
  echo "build_dir=${BUILD_DIR}"
} | tee "${OUT_DIR}/README.txt"

env | grep -E '^(TASK066_|DRAFT_GGUF|MAIN_GGUF|GGML_METAL_MUL_MM_MIN|BUILD_DIR)=' | sort > "${OUT_DIR}/env.txt" || true

if [[ -z "${GGML_METAL_MUL_MM_MIN:-}" ]]; then
  echo "NOTE: product default MUL_MM_MIN=7 (ne11>=8 on float-tile mul_mm). Legacy =32; N=8 on mv debug =8." >&2
fi

for ub in ${UBATCH_LIST}; do
  run_cell "${ub}" nomtp
  run_cell "${ub}" mtp
done

MATRIX="${OUT_DIR}/matrix.md"
{
  echo "# TASK-066 MTP vs nomtp matrix"
  echo
  echo "stamp: ${STAMP}"
  echo "MUL_MM_MIN: ${GGML_METAL_MUL_MM_MIN:-<unset:default 0>}"
  echo "draft: ${DRAFT}"
  echo "spec_draft_n_max: ${SPEC_N_MAX}"
  echo
  echo "## Cells"
  echo
  echo "| cell | batch | gen |"
  echo "|------|-------|-----|"
  for ub in ${UBATCH_LIST}; do
    b="$(batch_for "${ub}")"
    echo "| S${ub} nomtp | ${b} | \`cells/ub$(printf '%04d' "${ub}")-nomtp/generation.txt\` |"
    echo "| S${ub} mtp | ${b} | \`cells/ub$(printf '%04d' "${ub}")-mtp/generation.txt\` |"
  done
  echo
  echo "## Auto first-disagree (MTP vs nomtp)"
  echo
  echo "| compare | result |"
  echo "|---------|--------|"
  for ub in ${UBATCH_LIST}; do
    g_n="$(cell_dir "${ub}" nomtp)/generation.txt"
    g_m="$(cell_dir "${ub}" mtp)/generation.txt"
    echo "| S${ub} mtp vs nomtp | $(first_disagree "${g_m}" "${g_n}") |"
  done
  # Cross-ubatch nomtp reference if both present
  if echo " ${UBATCH_LIST} " | grep -q ' 8 ' && echo " ${UBATCH_LIST} " | grep -q ' 16 '; then
    echo "| S16 nomtp vs S8 nomtp | $(first_disagree "$(cell_dir 16 nomtp)/generation.txt" "$(cell_dir 8 nomtp)/generation.txt") |"
  fi
  echo
  echo "## Capture check"
  echo
  empty_n=0
  for ub in ${UBATCH_LIST}; do
    for mode in nomtp mtp; do
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
  for ub in ${UBATCH_LIST}; do
    ev="$(cell_dir "${ub}" mtp)/mtp-evidence.txt"
    if [[ -s "${ev}" ]]; then
      echo "- S${ub} mtp: OK (\`cells/ub$(printf '%04d' "${ub}")-mtp/mtp-evidence.txt\`)"
    else
      echo "- S${ub} mtp: MISSING evidence (stamp invalid for M1)"
    fi
  done
  echo
  echo "## Operator notes"
  echo
  echo "- identical + engagement OK => M1 green at this ubatch."
  echo "- identical without engagement => INVALID (re-run with this script; -lv 3 required)."
  echo "- DIFF => attach cli.log; run ACC_TRACE / M2 if logic-shaped (not mul_mm noise)."
  echo "- Finish arbiter (M4): TASK066_UBATCH_LIST=\"8\" (or \"8 16\")."
} > "${MATRIX}"

echo "OK: MTP vs nomtp written to ${OUT_DIR}"
echo "    matrix: ${MATRIX}"
if grep -q 'EMPTY S\|EMPTY_CAPTURE\|FAIL: .*empty' "${MATRIX}" 2>/dev/null; then
  echo "FAIL: empty generation capture" >&2
  exit 76
fi
exit 0
