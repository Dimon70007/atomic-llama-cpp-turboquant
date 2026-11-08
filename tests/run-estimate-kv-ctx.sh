#!/usr/bin/env bash
# Thin wrapper: TASK-066.10 KV / n_ctx estimator (fail-closed).
# Prefer: make estimate-kv-ctx MODEL=... N_CTX=... CTK=... CTV=... ...
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
  N_SEQ_MAX=1 UNIFIED=true SWA_FULL=false \
  [KV_BUDGET_MIB=...] [HEAD=2 TAIL=8] \
  ./tests/run-estimate-kv-ctx.sh

Required: MODEL N_CTX CTK CTV N_SEQ_MAX UNIFIED SWA_FULL
HEAD+TAIL required when CTV is turbo3.8 or turbo4.8
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

missing=0
for var in MODEL N_CTX CTK CTV N_SEQ_MAX UNIFIED SWA_FULL; do
  if [[ -z "${!var:-}" ]]; then
    echo "run-estimate-kv-ctx: missing required ${var}" >&2
    missing=1
  fi
done
if [[ "${missing}" -ne 0 ]]; then
  usage >&2
  exit 1
fi

ctv_lc="$(printf '%s' "${CTV}" | tr '[:upper:]' '[:lower:]')"
if [[ "${ctv_lc}" == "turbo3.8" || "${ctv_lc}" == "turbo4.8" ]]; then
  if [[ -z "${HEAD:-}" || -z "${TAIL:-}" ]]; then
    echo "run-estimate-kv-ctx: HEAD and TAIL required when CTV=${CTV}" >&2
    exit 1
  fi
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
  --unified "${UNIFIED}"
  --swa-full "${SWA_FULL}"
)

if [[ -n "${KV_BUDGET_MIB:-}" ]]; then
  args+=(--kv-budget-mib "${KV_BUDGET_MIB}")
fi
if [[ -n "${HEAD:-}" ]]; then
  args+=(--head "${HEAD}")
fi
if [[ -n "${TAIL:-}" ]]; then
  args+=(--tail "${TAIL}")
fi

exec "${PYTHON}" "${args[@]}"
