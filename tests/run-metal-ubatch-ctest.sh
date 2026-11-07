#!/usr/bin/env bash
# ctest wrapper for test-ubatch-s8-s1024-row0-parity.
# SKIP 77 when model/ctx/Metal are missing; no silent substitution.
#
# Env:
#   TASK066_MODEL or METAL_BI_MODEL  - required GGUF
#   TASK066_CTX   or METAL_BI_CTX    - required context
#   TASK066_PROMPT or METAL_BI_PROMPT - optional prompt file
#   TASK066_REQUIRE_GPU - default 1 on Darwin

set -euo pipefail

BIN="${1:-}"
if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
  echo "host-skip: missing test binary (arg1)" >&2
  exit 77
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${TASK066_MODEL:-${METAL_BI_MODEL:-}}"
CTX="${TASK066_CTX:-${METAL_BI_CTX:-}}"
DEFAULT_PROMPT="${REPO_ROOT}/benchmarks/metal-batch-invariance/pierre_dolokhov_duel_2k.prompt.txt"
PROMPT="${TASK066_PROMPT:-${METAL_BI_PROMPT:-${DEFAULT_PROMPT}}}"

if [[ -z "${MODEL}" || ! -f "${MODEL}" ]]; then
  echo "host-skip: set TASK066_MODEL=/path/to.gguf (or METAL_BI_MODEL) for ubatch ctest" >&2
  exit 77
fi
if [[ -z "${CTX}" ]]; then
  echo "host-skip: set TASK066_CTX=<n> (or METAL_BI_CTX) for ubatch ctest" >&2
  exit 77
fi
if [[ ! -f "${PROMPT}" ]]; then
  echo "host-skip: prompt file missing: ${PROMPT}" >&2
  exit 77
fi

if [[ -z "${TASK066_REQUIRE_GPU:-}" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
  export TASK066_REQUIRE_GPU=1
fi

exec "${BIN}" \
  -m "${MODEL}" \
  -f "${PROMPT}" \
  -c "${CTX}" \
  -ngl 999
