#!/usr/bin/env bash
# TASK-066.5/066.6 — S8 vs S1024 greedy token-stream parity (TDD RED gate).
# Product-class autotest for ladder confirm DIFF@text (stamp 20261106T212703Z).
#
# Usage (from this repo root):
#   TASK066_MODEL=… TASK066_CTX=3072 \
#     ./tests/run-task066-ubatch-s8-s1024-corners.sh [OUT_DIR]
# Env:
#   TASK066_MODEL     — required GGUF (fail-closed)
#   TASK066_CTX       — required (>= prompt + n_predict); ladder uses 3072
#   TASK066_PROMPT    — optional; default benchmarks/metal-batch-invariance/...prompt.txt
#   TASK066_N_PREDICT — optional; default 64 (binary); raise if GREEN but ladder DIFF
#   TASK066_DIAG=localize|fa-ablation|c3-bisect|c3-thresh — evidence-only (exit 0)
#   TASK066_BISECT_FILTER / TASK066_BISECT_HASH_ONLY — c3-bisect knobs
#   TASK066_FA=off + TASK066_CTV=f16 — true FA-off diag (refuse turbo KV); NOT product waiver
#   BUILD_DIR, REBUILD=1, TASK066_REQUIRE_GPU (default 1 on Darwin)
#
# Not in SUITE=all. Umbrella: SUITE=ubatch-s8-s1024

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-ubatch-s8-s1024-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-ubatch-s8-s1024-row0-parity"
MODEL="${TASK066_MODEL:-}"
CTX="${TASK066_CTX:-}"
DEFAULT_PROMPT="${REPO_ROOT}/benchmarks/metal-batch-invariance/pierre_dolokhov_duel_2k.prompt.txt"
PROMPT="${TASK066_PROMPT:-${DEFAULT_PROMPT}}"

if [[ -z "${MODEL}" ]]; then
  # Match ladder default scratch path when unset would fail-closed at binary —
  # still require explicit TASK066_MODEL (no silent product substitution).
  echo "FAIL: missing required TASK066_MODEL=/path/to.gguf (refusing silent model substitution)" >&2
  echo "hint: TASK066_MODEL=${REPO_ROOT}/.scratch/gemma-4-26b-a4b/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf" >&2
  exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "FAIL: TASK066_MODEL not a file: ${MODEL}" >&2
  exit 2
fi
if [[ -z "${CTX}" ]]; then
  echo "FAIL: missing required TASK066_CTX=<n> (ladder uses 3072; refusing silent ctx)" >&2
  exit 2
fi
if ! [[ "${CTX}" =~ ^[0-9]+$ ]] || [[ "${CTX}" -lt 4 ]]; then
  echo "FAIL: TASK066_CTX must be an integer >= 4 (got ${CTX})" >&2
  exit 2
fi
if [[ ! -f "${PROMPT}" ]]; then
  echo "FAIL: prompt not a file: ${PROMPT}" >&2
  exit 2
fi
if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  exit 1
fi

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-ubatch-s8-s1024-row0-parity"
  cmake --build "${BUILD_DIR}" --target test-ubatch-s8-s1024-row0-parity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

if [[ ! -x "${TEST_BIN}" ]]; then
  echo "FAIL: missing ${TEST_BIN}" >&2
  exit 1
fi

if [[ -z "${TASK066_REQUIRE_GPU:-}" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
  export TASK066_REQUIRE_GPU=1
fi

ARGS=(
  -m "${MODEL}"
  -f "${PROMPT}"
  -c "${CTX}"
  -ngl 999
)

echo "==> UBATCH-S8-S1024-C1 (test-ubatch-s8-s1024-row0-parity)"
echo "    model=${MODEL}"
echo "    ctx=${CTX} prompt=${PROMPT}"
echo "    n_predict=${TASK066_N_PREDICT:-64}"
echo "    GGML_METAL_MUL_MM_MIN=${GGML_METAL_MUL_MM_MIN-<unset:default 0>}"
set +e
"${TEST_BIN}" "${ARGS[@]}" 2>&1 | tee "${OUT_DIR}/ubatch-s8-s1024.log"
RC=${PIPESTATUS[0]}
set -e
echo "ubatch_s8_s1024_exit=${RC}" | tee "${OUT_DIR}/ubatch-s8-s1024.exit"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "model=${MODEL}"
  echo "ctx=${CTX}"
  echo "prompt=${PROMPT}"
  echo "n_predict=${TASK066_N_PREDICT:-64}"
  echo "ubatch_s8_s1024_exit=${RC}"
  if [[ "${RC}" -eq 0 ]]; then
    echo "VERDICT=GREEN"
    echo "note=GREEN on this n_predict; if ladder still DIFF raise TASK066_N_PREDICT or check fixture"
  elif [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=HOST_SKIP"
  elif [[ "${RC}" -eq 2 ]]; then
    echo "VERDICT=MISUSE"
  else
    echo "VERDICT=RED"
    echo "note=greedy S8 vs S1024 token stream not bit-exact (066.5/066.6 TDD gate)"
  fi
} | tee "${OUT_DIR}/summary.txt"

exit "${RC}"
