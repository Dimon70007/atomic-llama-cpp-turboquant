#!/usr/bin/env bash
# TASK-066.19 — thin entry for KEEP-stack decode speed gates.
#
# Usage (from this repo root):
#   MODEL_NOMTP=gemma4-26b-a4b-mtp-nomtp MODEL_MTP=gemma4-26b-a4b-mtp \
#     TASK066_MODEL=…/UD-IQ4_XS.gguf TASK066_MODEL_DRAFT=…/mtp-…Q8_0.gguf \
#     ./tests/run-task066-keep-stack-speed.sh --dry-run
#   … ./tests/run-task066-keep-stack-speed.sh --from-cell C2_fa_mv
#   … ./tests/run-task066-keep-stack-speed.sh --phase-f
#
# Forwards args to tests/task066_keep_stack_speed.py.
# Fail-closed: aliases + GGUF paths required (python checks).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/task066_keep_stack_speed.py"

if [[ ! -f "${PY}" ]]; then
  echo "FAIL: missing ${PY}" >&2
  exit 2
fi

export BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-task066-cb-ws}"
export MODEL_NOMTP MODEL_MTP TASK066_MODEL TASK066_MODEL_DRAFT
exec python3 "${PY}" "$@"
