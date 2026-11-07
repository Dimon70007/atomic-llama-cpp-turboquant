#!/usr/bin/env bash
# Metal MTP knobs quality/speed matrix (examples only — not ctest).
#
# Usage:
#   unset GGML_METAL_MUL_MM_MIN
#   BUILD_DIR=build ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh
#   DRY_RUN=1 ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh
#   ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh --from-run n2-p095
#   ./examples/metal-batch-invariance/run-mtp-knob-matrix.sh --stamp-dir logs/metal-mtp-knob-matrix-STAMP
#
# Forwards all args to mtp_knob_matrix.py in this directory.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/mtp_knob_matrix.py"

if [[ ! -f "${PY}" ]]; then
  echo "FAIL: missing ${PY}" >&2
  exit 2
fi

if [[ -z "${GGML_METAL_MUL_MM_MIN:-}" ]]; then
  echo "NOTE: MUL_MM_MIN unset → product default 0" >&2
fi

export BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
exec python3 "${PY}" "$@"
