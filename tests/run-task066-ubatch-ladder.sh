#!/usr/bin/env bash
# TASK-066 Stage 0 — nomtp ubatch ladder (product gate; NOT in SUITE=followup).
# See tests/TASK-066-ladder-fixture.md and tests/README-TASK066.md (TASK-066.4).
#
# Model resolution matches scripts/run-gemma4-mtp-server.sh (MAIN_GGUF / .scratch):
#   1) MAIN_GGUF or TASK066_MODEL
#   2) ${ROOT}/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf
# Setup: ./scripts/setup-gemma4-26b-a4b-models.sh
#
# Usage (Metal Terminal):
#   unset TASK066_PROMPT TASK066_MODEL   # drop stale external paths if any
#   GGML_METAL_MUL_MM_MIN=8 \
#   BUILD_DIR=build-task066-cb ./tests/run-task066-ubatch-ladder.sh [OUT_DIR]
#
# Env:
#   MAIN_GGUF / TASK066_MODEL — optional GGUF override
#   TASK066_PROMPT      — optional; default tests/fixtures/task066/...prompt.txt
#   TASK066_N_PREDICT   — default 256
#   TASK066_CTX         — default 4096
#   TASK066_UBATCH_LIST — default "1 8 16 33 256 512 1024 2048"
#   TASK066_REPEAT_LARGE — default "1024 2048" (2x same-ubatch on large cells)
#   TASK066_BATCH_MODE  — "floor" (default): -b = max(ubatch, TASK066_BATCH_FLOOR)
#                         "equal": -b = ubatch (small ub breaks ~2k prefill)
#                         "max":   -b = TASK066_BATCH_MAX always (often Metal OOM on M4)
#   TASK066_BATCH_FLOOR — default 512 (enough for ~2k prefill chunks; avoids -b 2048 OOM)
#   TASK066_BATCH_MAX   — default 2048 (only for mode=max)
#   TASK066_CTK/CTV     — locked q8_0 / turbo4 (product KV; other values rejected)
#   TASK066_CRAM        — default 0 (disable prompt cache). CLI default is 8192 MiB and
#                         OOMs this 26B MoE on 24 GB Metal; product yaml uses 4096.
#   TASK066_CTXCP       — default 32 (product ctx_checkpoints)
#   TASK066_STOP_ON_FAIL — default 1 (abort on non-zero OR Metal OOM / Compute error in log)
#   TASK066_ALLOW_SHARED_METAL — default 0; set 1 to skip "other llama-* holding GPU" preflight
#   GGML_METAL_MUL_MM_MIN — recommended 8 for Stage 0/1 debug
#   LLAMA_CLI, BUILD_DIR, REBUILD=1, DRY_RUN=1
#
# Writes: logs/task066-ubatch-ladder-<stamp>/{cells/,matrix.md,README.txt,env.txt}

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-ubatch-ladder-${STAMP}}"
mkdir -p "${OUT_DIR}/cells"

CLI_BIN="${LLAMA_CLI:-${BUILD_DIR}/bin/llama-cli}"
SCRATCH_MODEL="${REPO_ROOT}/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
MODEL="${MAIN_GGUF:-${TASK066_MODEL:-}}"
if [[ -z "${MODEL}" ]]; then
  MODEL="${SCRATCH_MODEL}"
fi
DEFAULT_PROMPT="${REPO_ROOT}/tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt"
PROMPT="${TASK066_PROMPT:-${DEFAULT_PROMPT}}"
N_PREDICT="${TASK066_N_PREDICT:-256}"
# ~2k prompt + 256 predict + chat template headroom
CTX="${TASK066_CTX:-3072}"
UBATCH_LIST="${TASK066_UBATCH_LIST:-1 8 16 33 256 512 1024 2048}"
# Use ${VAR-default} (not :-) so TASK066_REPEAT_LARGE="" disables 2x repeats.
REPEAT_LARGE="${TASK066_REPEAT_LARGE-1024 2048}"
# Default floor: -b = max(ubatch, 512). KV locked q8_0/turbo4.
BATCH_MODE="${TASK066_BATCH_MODE:-floor}"
BATCH_FLOOR="${TASK066_BATCH_FLOOR:-512}"
BATCH_MAX="${TASK066_BATCH_MAX:-2048}"
CTK="q8_0"
CTV="turbo4"
# Critical: llama-cli default -cram 8192 reserves another 8 GiB and OOMs 24 GB boxes.
CRAM="${TASK066_CRAM:-0}"
CTXCP="${TASK066_CTXCP:-32}"
STOP_ON_FAIL="${TASK066_STOP_ON_FAIL:-1}"

fail() { echo "FAIL: $*" >&2; exit 2; }

if [[ -n "${TASK066_CTK:-}" && "${TASK066_CTK}" != "q8_0" ]]; then
  fail "TASK066_CTK locked to q8_0 (got ${TASK066_CTK})"
fi
if [[ -n "${TASK066_CTV:-}" && "${TASK066_CTV}" != "turbo4" ]]; then
  fail "TASK066_CTV locked to turbo4 (got ${TASK066_CTV})"
fi

if [[ ! -f "${MODEL}" ]]; then
  fail "model not found: ${MODEL}
hint: ./scripts/setup-gemma4-26b-a4b-models.sh
hint: or set MAIN_GGUF=/path/to/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"
fi
[[ -f "${PROMPT}" ]] || fail "prompt not a file: ${PROMPT} (set TASK066_PROMPT or add fixtures/task066/...)"
[[ -d "${BUILD_DIR}" ]] || fail "BUILD_DIR missing: ${BUILD_DIR}"

case "${BATCH_MODE}" in
  floor|max|equal) ;;
  *) fail "TASK066_BATCH_MODE must be floor|max|equal (got ${BATCH_MODE})" ;;
esac
if [[ "${BATCH_MODE}" == "equal" ]]; then
  echo "NOTE: batch_mode=equal with small ubatch can abort on ~2k prefill; prefer floor." >&2
fi
if [[ "${BATCH_MODE}" == "max" ]]; then
  echo "NOTE: batch_mode=max forces -b=${BATCH_MAX}; often Metal OOM with q8 KV on M4." >&2
fi
# Stale exports from older docs point outside the repo.
if [[ "${PROMPT}" == *"/linux_to_mac/"* ]]; then
  echo "WARN: TASK066_PROMPT points outside this repo (${PROMPT}); unset it to use tests/fixtures/task066/" >&2
fi

# Second full GPU load of this 26B MoE OOMs 24 GB Metal when minillm llama-server
# already holds the model (typical: localhost:8180).
preflight_exclusive_metal() {
  if [[ "${TASK066_ALLOW_SHARED_METAL:-0}" == "1" ]]; then
    echo "NOTE: TASK066_ALLOW_SHARED_METAL=1 — skipping exclusive-Metal preflight" >&2
    return 0
  fi
  local hits=""
  # pgrep -fl is enough; ignore our own $$ tree.
  if command -v pgrep >/dev/null 2>&1; then
    hits="$(pgrep -fl 'llama-server|llama-cli' 2>/dev/null | grep -v " ${$}" || true)"
  fi
  if [[ -z "${hits}" ]]; then
    return 0
  fi
  # If the only hits are unrelated short-lived, still warn — any llama-server is fatal.
  if echo "${hits}" | grep -q 'llama-server'; then
    echo "FAIL: another llama-server is already running (holds Metal VRAM):" >&2
    echo "${hits}" | LC_ALL=C sed 's/^/  /' >&2
    echo "hint: stop minillm first, e.g. make switch / stop on the product stack, or:" >&2
    echo "hint:   pkill -f 'llama-server.*gemma-4-26B'   # then re-run ladder" >&2
    echo "hint: ladder spawns its own llama-cli load; two copies of UD-IQ4_XS OOMs ~24 GB." >&2
    echo "hint: override only if you know what you are doing: TASK066_ALLOW_SHARED_METAL=1" >&2
    exit 2
  fi
  if echo "${hits}" | grep -q 'llama-cli'; then
    echo "WARN: other llama-cli process(es) still alive:" >&2
    echo "${hits}" | LC_ALL=C sed 's/^/  /' >&2
  fi
}
preflight_exclusive_metal

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${CLI_BIN}" ]]; then
  echo "==> rebuild llama-cli"
  cmake --build "${BUILD_DIR}" --target llama-cli --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi
[[ -x "${CLI_BIN}" ]] || fail "missing ${CLI_BIN}"

is_repeat_cell() {
  local n="$1" x
  for x in ${REPEAT_LARGE}; do
    [[ "${x}" == "${n}" ]] && return 0
  done
  return 1
}

batch_for() {
  local ub="$1"
  case "${BATCH_MODE}" in
    equal)
      echo "${ub}"
      ;;
    max)
      echo "${BATCH_MAX}"
      ;;
    *)
      # floor: max(ubatch, BATCH_FLOOR)
      if [[ "${ub}" -gt "${BATCH_FLOOR}" ]]; then
        echo "${ub}"
      else
        echo "${BATCH_FLOOR}"
      fi
      ;;
  esac
}

cell_dir() {
  local ub="$1" run="$2"
  printf '%s/cells/ub%04d-run%d' "${OUT_DIR}" "${ub}" "${run}"
}

run_cell() {
  local ub="$1" run="$2"
  local b dir log gen
  b="$(batch_for "${ub}")"
  dir="$(cell_dir "${ub}" "${run}")"
  mkdir -p "${dir}"
  log="${dir}/cli.log"
  gen="${dir}/generation.txt"

  # --simple-io keeps console on stdout (not /dev/tty) so redirect captures text.
  # --single-turn exits after -f. stdin=/dev/null avoids REPL.
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
  )

  {
    echo "ubatch=${ub}"
    echo "batch=${b}"
    echo "run=${run}"
    echo "batch_mode=${BATCH_MODE}"
    echo "ctk=${CTK} ctv=${CTV}"
    echo "cram=${CRAM} ctxcp=${CTXCP} ctx=${CTX}"
    echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN:-<unset:default 7>}"
    echo "cmd=${CLI_BIN} ${args[*]}"
  } > "${dir}/meta.txt"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN skip ub=${ub} run=${run}" | tee "${dir}/dry_run.txt"
    printf '(dry-run stub for ubatch=%s run=%d)\n' "${ub}" "${run}" > "${gen}"
    return 0
  fi

  echo "==> ubatch=${ub} batch=${b} ctk=${CTK} ctv=${CTV} cram=${CRAM} run=${run}"
  set +e
  "${CLI_BIN}" "${args[@]}" < /dev/null > "${log}" 2>&1
  local rc=$?
  set -e

  # llama-cli can exit 0 after server-path "Compute error" / Metal OOM.
  if grep -E -q 'Insufficient Memory|Compute error\.|GGML_ASSERT' "${log}" 2>/dev/null; then
    if [[ "${rc}" -eq 0 ]]; then
      rc=75
    fi
  fi

  # Full transcript in cli.log; strip ANSI and keep assistant body for matrix diffs.
  extract_generation "${log}" > "${gen}"
  if [[ ! -s "${gen}" ]]; then
    echo "FAIL: empty generation.txt for ub=${ub} run=${run} (see ${log})" >&2
    echo "hint: need --simple-io so console goes to stdout; do not rebuild matrix by hand." >&2
    if [[ "${rc}" -eq 0 ]]; then
      rc=76
    fi
  fi
  echo "exit=${rc}" > "${dir}/exit.txt"

  if [[ "${rc}" -ne 0 ]]; then
    echo "FAIL: llama-cli exit ${rc} for ub=${ub} run=${run} (see ${log})" >&2
    if grep -E -q 'Insufficient Memory' "${log}" 2>/dev/null; then
      echo "hint: Metal OOM — ensure -cram 0 (default here; CLI default 8192 MiB OOMs 24 GB)." >&2
      echo "hint: kill leftover llama-cli/llama-server; BATCH_MODE=floor; CTX~3072; KV q8_0/turbo4." >&2
    fi
    if [[ "${STOP_ON_FAIL}" == "1" ]]; then
      exit "${rc}"
    fi
  fi
  return 0
}

# Pull assistant text out of a captured cli.log for automatic matrix diffs.
# macOS sed under UTF-8 locale dies on non-UTF8 bytes in Metal/CLI logs
# ("RE error: illegal byte sequence"); force C locale for strip + match.
#
# Conversation --simple-io layout (llama-cli):
#   > <prompt...>
#   ... (truncated)          # often mid-line / next line, not alone
#   <assistant body>
#   [ Prompt: ... | Generation: ... ]
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

  # Prefer body after "(truncated)" (prompt echo), stop at timing / exit banner.
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

  # No truncated marker: take lines after the first "> " prompt until timing.
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

  # Last resort: keep everything up to timing (noisy; still better than empty).
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
  # Compact: first differing byte + short snippets (no path noise).
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

{
  echo "TASK-066 ubatch ladder"
  echo "stamp=${STAMP}"
  echo "fixture=tests/TASK-066-ladder-fixture.md"
  echo "model=${MODEL}"
  echo "prompt=${PROMPT}"
  echo "n_predict=${N_PREDICT}"
  echo "ctx=${CTX}"
  echo "ctk=${CTK} ctv=${CTV}"
  echo "cram=${CRAM} ctxcp=${CTXCP}"
  echo "batch_mode=${BATCH_MODE} batch_floor=${BATCH_FLOOR} batch_max=${BATCH_MAX}"
  echo "ubatch_list=${UBATCH_LIST}"
  echo "repeat_large=${REPEAT_LARGE}"
  echo "GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN:-<unset:default 7>}"
  echo "cli=${CLI_BIN}"
  echo "build_dir=${BUILD_DIR}"
} | tee "${OUT_DIR}/README.txt"

env | grep -E '^(TASK066_|GGML_METAL_MUL_MM_MIN|BUILD_DIR)=' | sort > "${OUT_DIR}/env.txt" || true

for ub in ${UBATCH_LIST}; do
  run_cell "${ub}" 1
  if is_repeat_cell "${ub}"; then
    run_cell "${ub}" 2
  fi
done

# Stub matrix for operator fill / auto helpers.
MATRIX="${OUT_DIR}/matrix.md"
{
  echo "# TASK-066 ubatch ladder matrix"
  echo
  echo "stamp: ${STAMP}"
  echo "MUL_MM_MIN: ${GGML_METAL_MUL_MM_MIN:-<unset:default 7>}"
  echo "batch_mode: ${BATCH_MODE}"
  echo "golden: ubatch<=8 (prefer S8); S1 optional purity"
  echo
  echo "## Cells"
  echo
  echo "| cell | batch | gen | notes |"
  echo "|------|-------|-----|-------|"
  for ub in ${UBATCH_LIST}; do
    b="$(batch_for "${ub}")"
    echo "| S${ub} run1 | ${b} | \`cells/ub$(printf '%04d' "${ub}")-run1/generation.txt\` | |"
    if is_repeat_cell "${ub}"; then
      echo "| S${ub} run2 | ${b} | \`cells/ub$(printf '%04d' "${ub}")-run2/generation.txt\` | same-ubatch |"
    fi
  done
  echo
  echo "## Auto first-disagree (generation.txt)"
  echo
  echo "| compare | result |"
  echo "|---------|--------|"
  s8="$(cell_dir 8 1)/generation.txt"
  s1="$(cell_dir 1 1)/generation.txt"
  # Only emit rows for cells that exist in this stamp (subset ladders OK).
  cell_gen() { echo "$(cell_dir "$1" "${2:-1}")/generation.txt"; }
  has_cell() { [[ -s "$(cell_gen "$1" "${2:-1}")" ]]; }
  if has_cell 1 && has_cell 8; then
    echo "| S1 vs S8 | $(first_disagree "$(cell_gen 1)" "$(cell_gen 8)") |"
  fi
  if has_cell 8; then
    for ub in ${UBATCH_LIST}; do
      [[ "${ub}" == "1" || "${ub}" == "8" ]] && continue
      if has_cell "${ub}"; then
        echo "| S${ub} vs S8 | $(first_disagree "$(cell_gen "${ub}")" "$(cell_gen 8)") |"
      fi
    done
  fi
  if has_cell 33 && has_cell 16; then
    echo "| S33 vs S16 | $(first_disagree "$(cell_gen 33)" "$(cell_gen 16)") |"
  fi
  if has_cell 512 && has_cell 1024; then
    echo "| S512 vs S1024 | $(first_disagree "$(cell_gen 512)" "$(cell_gen 1024)") |"
  fi
  if has_cell 1024 && has_cell 2048; then
    echo "| S1024 vs S2048 | $(first_disagree "$(cell_gen 1024)" "$(cell_gen 2048)") |"
  fi
  for ub in ${REPEAT_LARGE}; do
    if has_cell "${ub}" 1 && has_cell "${ub}" 2; then
      echo "| S${ub} run1 vs run2 | $(first_disagree "$(cell_gen "${ub}" 1)" "$(cell_gen "${ub}" 2)") |"
    fi
  done
  echo
  echo "## Capture check"
  echo
  empty_n=0
  for ub in ${UBATCH_LIST}; do
    g="$(cell_dir "${ub}" 1)/generation.txt"
    if [[ ! -s "${g}" ]]; then
      echo "- EMPTY S${ub} run1 generation.txt"
      empty_n=$((empty_n + 1))
    fi
    if is_repeat_cell "${ub}"; then
      g2="$(cell_dir "${ub}" 2)/generation.txt"
      if [[ ! -s "${g2}" ]]; then
        echo "- EMPTY S${ub} run2 generation.txt"
        empty_n=$((empty_n + 1))
      fi
    fi
  done
  if [[ "${empty_n}" -eq 0 ]]; then
    echo "- all generation.txt non-empty"
  else
    echo "- FAIL: ${empty_n} empty generation file(s); matrix diffs are not trustworthy"
  fi
  echo
  echo "## Operator notes"
  echo
  echo "- Same-ubatch DIFF before cross-ubatch attribution (#19034)."
  echo "- Re-run with \`TASK066_BATCH_MODE=max\` for prod-shaped secondary stamp."
} > "${MATRIX}"

echo "OK: ladder written to ${OUT_DIR}"
echo "    matrix: ${MATRIX}"
if [[ -z "${GGML_METAL_MUL_MM_MIN:-}" ]]; then
  echo "NOTE: GGML_METAL_MUL_MM_MIN unset (product default 7; ne11>=8 uses float-tile mul_mm). Legacy pin =32; N=8 on mv debug =8." >&2
fi
# Fail the whole ladder if any capture was empty (even if llama-cli exited 0).
if grep -q 'EMPTY generation.txt\|EMPTY_CAPTURE\|FAIL: .*empty generation' "${MATRIX}" 2>/dev/null; then
  echo "FAIL: empty generation capture; re-run after fixing --simple-io redirect" >&2
  exit 76
fi
if grep -q 'EMPTY S' "${MATRIX}" 2>/dev/null; then
  echo "FAIL: empty generation capture in matrix" >&2
  exit 76
fi
exit 0
