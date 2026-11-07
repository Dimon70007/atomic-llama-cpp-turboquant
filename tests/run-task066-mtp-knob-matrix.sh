#!/usr/bin/env bash
# TASK-066.9 Phase C — thin entry for MTP knobs quality/speed matrix.
#
# Usage:
#   unset GGML_METAL_MUL_MM_MIN
#   BUILD_DIR=build-task066-cb-ws ./tests/run-task066-mtp-knob-matrix.sh
#   DRY_RUN=1 ./tests/run-task066-mtp-knob-matrix.sh
#   ./tests/run-task066-mtp-knob-matrix.sh --from-run n2-p095
#   ./tests/run-task066-mtp-knob-matrix.sh --stamp-dir logs/task066-mtp-knob-matrix-STAMP
#
# Forwards all args to tests/task066_mtp_knob_matrix.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/task066_mtp_knob_matrix.py"

if [[ ! -f "${PY}" ]]; then
  echo "FAIL: missing ${PY}" >&2
  exit 2
fi

if [[ -z "${GGML_METAL_MUL_MM_MIN:-}" ]]; then
  echo "NOTE: MUL_MM_MIN unset → product default 0" >&2
fi

export BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
exec python3 "${PY}" "$@"
