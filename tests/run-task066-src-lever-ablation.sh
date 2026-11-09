#!/usr/bin/env bash
# TASK-066.18 — thin entry for src-delta lever ablation (tip−1 / master+1).
#
# Usage (from this repo root):
#   TASK066_MODEL=…/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf TASK066_CTX=3072 \
#     ./tests/run-task066-src-lever-ablation.sh --dry-run
#   TASK066_MODEL=… TASK066_CTX=3072 ./tests/run-task066-src-lever-ablation.sh
#   ./tests/run-task066-src-lever-ablation.sh --mode A --from-lever L_size_swa_ub
#   ./tests/run-task066-src-lever-ablation.sh --controls-only
#
# Forwards all args to tests/task066_src_lever_ablation.py.
# Fail-closed: ubatch rows need TASK066_MODEL + TASK066_CTX (python checks).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/task066_src_lever_ablation.py"

if [[ ! -f "${PY}" ]]; then
  echo "FAIL: missing ${PY}" >&2
  exit 2
fi

export BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
exec python3 "${PY}" "$@"
