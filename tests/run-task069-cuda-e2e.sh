#!/usr/bin/env bash
# TASK-069 — one-shot CUDA e2e entry (fork-local).
#
# Loads tests/fixtures/task069/cuda-mtp.reference.yaml knobs and runs a named
# runner. Fail-closed on model/draft/config (no silent model substitution).
#
# Usage (from fork root or any cwd):
#   ./tests/run-task069-cuda-e2e.sh suite
#   ./tests/run-task069-cuda-e2e.sh mtp-e2e
#   ./tests/run-task069-cuda-e2e.sh ubatch-s8
#   ./tests/run-task069-cuda-e2e.sh ubatch-ladder
#   ./tests/run-task069-cuda-e2e.sh mtp-vs-nomtp
#   ./tests/run-task069-cuda-e2e.sh all-e2e
#   ./tests/run-task069-cuda-e2e.sh env
#   ./tests/run-task069-cuda-e2e.sh help
#
# Optional overrides (still fail-closed if set to a missing path):
#   TASK066_MODEL / TASK066_MODEL_DRAFT / TASK069_SUITE_CONFIG / BUILD_DIR
#   OFF_BUILD / ON_BUILD / STAMP_DIR / ONLY / CELL / SKIP_HEAVY
#   TASK066_UBATCH_LIST / TASK066_REPEAT_LARGE / TASK066_CTX_OVERRIDE
#
# Docs: tests/TASK-069-cuda-e2e-fixture.md

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KNOBS_PY="${REPO_ROOT}/tests/task069_suite_knobs.py"
DEFAULT_SUITE_CONFIG="${REPO_ROOT}/tests/fixtures/task069/cuda-mtp.reference.yaml"
DEFAULT_PROMPT="${REPO_ROOT}/tests/fixtures/task069/task066_pierre_dolokhov_duel_2k.prompt.txt"
SCRATCH_DIR="${REPO_ROOT}/.scratch/gemma-4-26b-a4b"
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"

CMD="${1:-}"
if [[ -n "${CMD}" ]]; then
  shift
fi

fail() { echo "FAIL: $*" >&2; exit 2; }

usage() {
  cat <<EOF
Usage: ./tests/run-task069-cuda-e2e.sh <command> [runner-args...]

Commands:
  suite           OFF→ON full fork suite compare (run-task069-cuda-suite-compare.sh)
  mtp-e2e         run-task066-mtp-e2e-corners.sh
  ubatch-s8       run-task066-ubatch-s8-s1024-corners.sh
  ubatch-ladder   run-task066-ubatch-ladder.sh
  mtp-vs-nomtp    run-task066-mtp-vs-nomtp-ladder.sh
  all-e2e         mtp-e2e → ubatch-s8 → ubatch-ladder → mtp-vs-nomtp (same env)
  env             load knobs and print exports (no test run)
  help            this text

Default suite config:
  ${DEFAULT_SUITE_CONFIG}
EOF
}

resolve_file() {
  local label="$1"
  shift
  local cand
  for cand in "$@"; do
    if [[ -n "${cand}" && -f "${cand}" ]]; then
      echo "${cand}"
      return 0
    fi
  done
  fail "missing ${label}; tried: $*"
}

stamp_utc() {
  date -u +%Y%m%dT%H%M%SZ
}

load_env() {
  local suite_config model draft prompt build_dir

  suite_config="$(resolve_file TASK069_SUITE_CONFIG \
    "${TASK069_SUITE_CONFIG:-}" \
    "${DEFAULT_SUITE_CONFIG}")"

  model="$(resolve_file TASK066_MODEL \
    "${TASK066_MODEL:-}" \
    "${SCRATCH_DIR}/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf" \
    "${MODELS_DIR}/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf")"

  draft="$(resolve_file TASK066_MODEL_DRAFT \
    "${TASK066_MODEL_DRAFT:-}" \
    "${SCRATCH_DIR}/mtp-gemma-4-26B-A4B-it-Q8_0.gguf" \
    "${MODELS_DIR}/mtp-gemma-4-26B-A4B-it-Q8_0.gguf")"

  prompt="$(resolve_file TASK066_PROMPT \
    "${TASK066_PROMPT:-}" \
    "${DEFAULT_PROMPT}" \
    "${REPO_ROOT}/tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt")"

  export TASK069_SUITE_CONFIG="${suite_config}"
  export TASK066_MODEL="${model}"
  export MAIN_GGUF="${model}"
  export TASK066_MODEL_DRAFT="${draft}"
  export DRAFT_GGUF="${draft}"
  export TASK066_PROMPT="${prompt}"

  # shellcheck disable=SC1090
  eval "$(python3 "${KNOBS_PY}" --config "${suite_config}" --export-shell)"

  if [[ -n "${TASK066_CTX_OVERRIDE:-}" ]]; then
    export TASK066_CTX="${TASK066_CTX_OVERRIDE}"
    export TASK069_CTX="${TASK066_CTX_OVERRIDE}"
  fi
  [[ -n "${TASK066_CTX:-}" ]] || fail "TASK066_CTX empty after suite config load"

  build_dir="${BUILD_DIR:-${REPO_ROOT}/build-task069-cuda-on}"
  [[ -d "${build_dir}" ]] || fail "BUILD_DIR missing: ${build_dir}"
  export BUILD_DIR="${build_dir}"
  export OFF_BUILD="${OFF_BUILD:-${REPO_ROOT}/build-task069-cuda-off}"
  export ON_BUILD="${ON_BUILD:-${REPO_ROOT}/build-task069-cuda-on}"

  # Ladder / e2e defaults (operator may override before calling).
  export TASK066_UBATCH_LIST="${TASK066_UBATCH_LIST:-512 1024 2048}"
  export TASK066_REPEAT_LARGE="${TASK066_REPEAT_LARGE-}"
  export TASK066_STOP_ON_FAIL="${TASK066_STOP_ON_FAIL:-0}"
  export TASK066_ALLOW_SHARED_METAL="${TASK066_ALLOW_SHARED_METAL:-1}"
  export TASK066_REQUIRE_GPU="${TASK066_REQUIRE_GPU:-1}"
  export SKIP_HEAVY="${SKIP_HEAVY:-1}"
  export REBUILD="${REBUILD:-0}"

  echo "task069-cuda-e2e: profile=${TASK069_SUITE_PROFILE:-?} arm=${TASK069_SUITE_ARM:-?}"
  echo "  config=${TASK069_SUITE_CONFIG}"
  echo "  model=${TASK066_MODEL}"
  echo "  draft=${TASK066_MODEL_DRAFT}"
  echo "  prompt=${TASK066_PROMPT}"
  echo "  build=${BUILD_DIR}"
  echo "  ctx=${TASK066_CTX} batch=${TASK069_BATCH} ubatch=${TASK069_UBATCH} ctv=${TASK069_CTV} ts=${TASK069_TENSOR_SPLIT} mg=${TASK069_MAIN_GPU}"
}

run_one() {
  local name="$1"
  local script="$2"
  shift 2
  local out="${REPO_ROOT}/logs/task069-cuda-e2e-${name}-$(stamp_utc)"
  mkdir -p "${out}"
  echo "==> ${name} OUT=${out}"
  bash "${script}" "${out}" "$@"
  echo "OK: ${name} → ${out}"
}

case "${CMD}" in
  ""|help|-h|--help)
    usage
    exit 0
    ;;
  env)
    load_env
    echo "---- exports ----"
    env | grep -E '^(TASK069_|TASK066_|BUILD_DIR|OFF_BUILD|ON_BUILD|MAIN_GGUF|DRAFT_GGUF|SKIP_HEAVY)=' | sort
    ;;
  suite)
    load_env
    exec bash "${REPO_ROOT}/tests/run-task069-cuda-suite-compare.sh" "$@"
    ;;
  mtp-e2e)
    load_env
    run_one mtp-e2e "${REPO_ROOT}/tests/run-task066-mtp-e2e-corners.sh" "$@"
    ;;
  ubatch-s8)
    load_env
    run_one ubatch-s8 "${REPO_ROOT}/tests/run-task066-ubatch-s8-s1024-corners.sh" "$@"
    ;;
  ubatch-ladder)
    load_env
    run_one ubatch-ladder "${REPO_ROOT}/tests/run-task066-ubatch-ladder.sh" "$@"
    ;;
  mtp-vs-nomtp)
    load_env
    run_one mtp-vs-nomtp "${REPO_ROOT}/tests/run-task066-mtp-vs-nomtp-ladder.sh" "$@"
    ;;
  all-e2e)
    load_env
    run_one mtp-e2e "${REPO_ROOT}/tests/run-task066-mtp-e2e-corners.sh"
    run_one ubatch-s8 "${REPO_ROOT}/tests/run-task066-ubatch-s8-s1024-corners.sh"
    run_one ubatch-ladder "${REPO_ROOT}/tests/run-task066-ubatch-ladder.sh"
    run_one mtp-vs-nomtp "${REPO_ROOT}/tests/run-task066-mtp-vs-nomtp-ladder.sh"
    echo "OK: all-e2e finished"
    ;;
  *)
    fail "unknown command '${CMD}' (try: help)"
    ;;
esac
