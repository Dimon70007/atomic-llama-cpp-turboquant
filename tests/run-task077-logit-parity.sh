#!/usr/bin/env bash
# TASK-077: build and run the numeric KV parity test.
#
# Why a script and not ad-hoc commands: the build flags here are not the
# library defaults (this host uses a task-specific build tree with CUDA on), and
# repeating them by hand is how a measurement ends up describing a different
# binary than the one under test.
#
# What it runs: tests/test-task077-kv-logit-parity, which compares LOGITS for
# one prompt across cache types, ubatch sizes and the prefilled-vs-cached path.
# Numbers, not text: text comparison proved useless (TASK-077 F41).
#
# Usage:
#   ./tests/run-task077-logit-parity.sh --ctv turbo4.8 --ub-a 512 --ub-b 256
#   ./tests/run-task077-logit-parity.sh --build-only
#   SUITE=1 ./tests/run-task077-logit-parity.sh          # every ctv x ubatch pair
#
# Env (optional):
#   TASK077_MODEL   GGUF (default ~/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf)
#   TASK077_BUILD   build tree (default: newest build-*/ with a CMakeCache.txt)
#   TASK077_PROMPT  prompt file (default tests/fixtures/task066/...)
#   CMAKE_FLAGS     extra configure flags; only used when the tree is created
#   JOBS            parallel build jobs (default 24)
#
# Exit: 0 every required comparison matched, 1 a required comparison failed,
#       2 bad input, 77 model or build tree absent (ctest skip).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${TASK077_MODEL:-${HOME}/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf}"
PROMPT="${TASK077_PROMPT:-${REPO_ROOT}/tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt}"
JOBS="${JOBS:-24}"
TARGET="test-task077-kv-logit-parity"

# ubatch values the operator allows. Below UBATCH_FLOOR a different code path is
# taken, so a run there would describe a configuration nobody ships.
UBATCH_FLOOR=16

# ---- build tree ------------------------------------------------------------
# Prefer an explicit selector, then the tree ctest runs from, then the newest
# build-* that already has a cache. This host uses task-specific tree names, so
# a hardcoded list is not enough.
BUILD="${TASK077_BUILD:-}"
if [[ -z "${BUILD}" ]]; then
  if [[ -n "${CTEST_BINARY_DIRECTORY:-}" && -f "${CTEST_BINARY_DIRECTORY}/CMakeCache.txt" ]]; then
    BUILD="${CTEST_BINARY_DIRECTORY}"
  elif [[ -n "${BUILD_DIR:-}" && -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    BUILD="${BUILD_DIR}"
  else
    BUILD="$(ls -1dt "${REPO_ROOT}"/build-*/ 2>/dev/null | head -1 || true)"
  fi
fi

if [[ -z "${BUILD}" || ! -d "${BUILD}" ]]; then
  echo "SKIP: no build tree found; set TASK077_BUILD or create one with cmake" >&2
  exit 77
fi
if [[ ! -f "${BUILD}/CMakeCache.txt" ]]; then
  echo "FAIL: ${BUILD} has no CMakeCache.txt (not a configured tree)" >&2
  exit 2
fi

echo "build tree: ${BUILD}"

# Configure only when the test target is unknown to this tree. Reconfiguring a
# working tree costs seconds but hides nothing; skipping it on a stale tree is
# how a gate ends up testing an old binary (TASK-077 F10).
if ! cmake --build "${BUILD}" --target help 2>/dev/null | grep -qx "${TARGET}"; then
  echo "configuring: cmake -S ${REPO_ROOT} -B ${BUILD} ${CMAKE_FLAGS:-}"
  # shellcheck disable=SC2086 # CMAKE_FLAGS is a deliberate flag list
  cmake -S "${REPO_ROOT}" -B "${BUILD}" ${CMAKE_FLAGS:-} > /dev/null
fi

echo "building: ${TARGET}"
cmake --build "${BUILD}" --target "${TARGET}" --parallel "${JOBS}"

BIN="${BUILD}/bin/${TARGET}"
if [[ ! -x "${BIN}" ]]; then
  echo "FAIL: ${BIN} not built" >&2
  exit 2
fi

if [[ -n "${TASK077_BUILD_ONLY:-}" ]]; then
  echo "built: ${BIN}"
  exit 0
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "SKIP: model not found: ${MODEL} (set TASK077_MODEL)" >&2
  exit 77
fi

# ---- run -------------------------------------------------------------------
# Parse the same --ub-a/--ub-b/--ctv the test understands, so the guard can be
# applied before a single GPU instruction runs. --build-only is consumed here.
ARGS=()
BUILD_ONLY=0
set -- "$@"  # keep "$@" intact for the shift loop
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)
      BUILD_ONLY=1
      shift
      ;;
    --ub-a|--ub-b)
      val="${2:-}"
      if [[ -n "${val}" && "${val}" =~ ^[0-9]+$ && "${val}" -lt "${UBATCH_FLOOR}" ]]; then
        echo "FAIL: ubatch ${val} is below ${UBATCH_FLOOR}; that path is not used in" >&2
        echo "      production and is excluded by operator instruction" >&2
        exit 2
      fi
      ARGS+=("$1" "${2:-}")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${BUILD_ONLY}" -eq 1 ]]; then
  echo "built: ${BIN}"
  exit 0
fi

if [[ -n "${SUITE:-}" ]]; then
  rc=0
  for ctv in q8_0 turbo3 turbo4; do
    for ub in 512 256 16; do
      echo
      echo "=== ctv=${ctv} ub-a=${ub} ub-b=256 ==="
      "${BIN}" -m "${MODEL}" -f "${PROMPT}" --ctv "${ctv}" --ub-a "${ub}" --ub-b 256 || rc=1
    done
  done
  exit "${rc}"
fi

# SPLIT_SWEEP=1: at a FIXED ubatch, vary where the prompt is cut in two. This
# isolates the trigger: is any split enough, or only one that disagrees with the
# ubatch boundary? The earlier "it vanishes at ubatch 256" reading was an
# artefact of the default split being exactly half, i.e. the two paths were the
# same two decodes. A sweep makes that impossible to repeat.
if [[ -n "${SPLIT_SWEEP:-}" ]]; then
  rc=0
  ub="${SWEEP_UB:-512}"
  ctv="${SWEEP_CTV:-q8_0}"
  for sp in ${SWEEP_POINTS:-128 256 384 448}; do
    printf 'split@%-6s ' "${sp}"
    "${BIN}" -m "${MODEL}" -f "${PROMPT}" --ctv "${ctv}" \
             --ub-a "${ub}" --ub-b "${ub}" --split-at "${sp}" 2>/dev/null \
      | awk -v rc_file="${rc_file:-}" '/^prefill one batch vs split/ {printf "max=%-10s p50=%-10s p99=%-10s KL=%s\n", $6, $7, $8, $11}' || true
  done
  exit "${rc}"
fi

"${BIN}" -m "${MODEL}" -f "${PROMPT}" "${ARGS[@]+"${ARGS[@]}"}"
