#!/usr/bin/env bash
# Thin wrapper: TASK-066.10 KV / n_ctx estimator (fail-closed; uniform CTV only).
# Prefer: make estimate-kv-ctx MODEL=... N_CTX=... CTK=... CTV=... N_BATCH=... N_UBATCH=...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FORK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PRODUCT_ROOT="$(cd "${FORK_ROOT}/.." && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/estimate_kv_ctx.py"
GGUF_PY="${FORK_ROOT}/gguf-py"

usage() {
  cat <<'EOF'
Usage (env or flags via Make):
  MODEL=path.gguf N_CTX=3072 CTK=q8_0 CTV=turbo4 \
  N_SEQ_MAX=1 N_BATCH=2048 N_UBATCH=512 UNIFIED=true SWA_FULL=false \
  [KV_BUDGET_MIB=...] \
  ./tests/run-estimate-kv-ctx.sh

Required: MODEL N_CTX CTK CTV N_SEQ_MAX N_BATCH N_UBATCH UNIFIED SWA_FULL
Optional: N_SWA_HEADROOM (default = N_UBATCH; product headroom = live -ub, pad128)
N_BATCH = telemetry; N_UBATCH = product size_swa headroom (unless N_SWA_HEADROOM set).
CTV is uniform only (f16, q8_0, turbo3, turbo4, ... — no layer-bracket aliases).
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

missing=0
for var in MODEL N_CTX CTK CTV N_SEQ_MAX N_BATCH N_UBATCH UNIFIED SWA_FULL; do
  if [[ -z "${!var:-}" ]]; then
    echo "run-estimate-kv-ctx: missing required ${var}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${MODEL}" ]]; then
  echo "run-estimate-kv-ctx: MODEL not found: ${MODEL}" >&2
  exit 1
fi

if [[ -x "${PRODUCT_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${PRODUCT_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "run-estimate-kv-ctx: no python3 found" >&2
  exit 1
fi

export PYTHONPATH="${GGUF_PY}${PYTHONPATH:+:${PYTHONPATH}}"

args=(
  "${PY_SCRIPT}"
  --model "${MODEL}"
  --n-ctx "${N_CTX}"
  --ctk "${CTK}"
  --ctv "${CTV}"
  --n-seq-max "${N_SEQ_MAX}"
  --n-batch "${N_BATCH}"
  --n-ubatch "${N_UBATCH}"
  --unified "${UNIFIED}"
  --swa-full "${SWA_FULL}"
)

if [[ -n "${N_SWA_HEADROOM:-}" ]]; then
  args+=(--n-swa-headroom "${N_SWA_HEADROOM}")
fi

if [[ -n "${KV_BUDGET_MIB:-}" ]]; then
  args+=(--kv-budget-mib "${KV_BUDGET_MIB}")
fi

exec "${PYTHON}" "${args[@]}"
