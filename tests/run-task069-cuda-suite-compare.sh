#!/usr/bin/env bash
# TASK-069 Phase 2.0d — thin entry for CUDA suite OFF→ON compare.
# Fail-closed: MODEL required. Suite YAML defaults to fork fixtures/task069/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${REPO_ROOT}/tests/run_task069_cuda_suite_compare.py"
KNOBS_PY="${REPO_ROOT}/tests/task069_suite_knobs.py"
DEFAULT_SUITE_CONFIG="${REPO_ROOT}/tests/fixtures/task069/cuda-mtp.reference.yaml"

if [[ -z "${TASK066_MODEL:-}" ]]; then
  echo "FAIL: missing required TASK066_MODEL=/path/to.gguf (refusing silent model substitution)" >&2
  echo "hint: ./scripts/setup-gemma4-26b-a4b-models.sh" >&2
  exit 2
fi

SUITE_CONFIG="${TASK069_SUITE_CONFIG:-${DEFAULT_SUITE_CONFIG}}"
if [[ ! -f "${SUITE_CONFIG}" ]]; then
  echo "FAIL: TASK069_SUITE_CONFIG not a file: ${SUITE_CONFIG}" >&2
  echo "hint: tests/fixtures/task069/cuda-mtp.reference.yaml (see tests/TASK-069-cuda-e2e-fixture.md)" >&2
  exit 2
fi

# Export CUDA product knobs into this process (and children).
# shellcheck disable=SC1090
eval "$(python3 "${KNOBS_PY}" --config "${SUITE_CONFIG}" --export-shell)"

if [[ -n "${TASK066_CTX_OVERRIDE:-}" ]]; then
  export TASK066_CTX="${TASK066_CTX_OVERRIDE}"
  export TASK069_CTX="${TASK066_CTX_OVERRIDE}"
fi
if [[ -z "${TASK066_CTX:-}" ]]; then
  echo "FAIL: TASK066_CTX empty after suite config load" >&2
  exit 2
fi

OFF_BUILD="${OFF_BUILD:-${REPO_ROOT}/build-task069-cuda-off}"
ON_BUILD="${ON_BUILD:-${REPO_ROOT}/build-task069-cuda-on}"
DRAFT="${TASK066_MODEL_DRAFT:-}"
S_VAL="${TASK066_S:-FA_TILE;NO_MV_EXT;FLOAT_TILE;MM_MIN;FUSION_CB;MOE_VIEW_ADD;ISWA_PADS;SHARED_KV;SWA_HWM_LOG}"

ARGS=(
  --off-build "${OFF_BUILD}"
  --on-build "${ON_BUILD}"
  --model "${TASK066_MODEL}"
  --ctx "${TASK066_CTX}"
  --S "${S_VAL}"
  --suite-config "${SUITE_CONFIG}"
)
if [[ -n "${DRAFT}" ]]; then
  ARGS+=(--draft "${DRAFT}")
fi
if [[ -n "${TASK066_PROMPT:-}" ]]; then
  ARGS+=(--prompt "${TASK066_PROMPT}")
fi
if [[ -n "${STAMP_DIR:-}" ]]; then
  ARGS+=(--stamp-dir "${STAMP_DIR}")
fi
if [[ "${SKIP_HEAVY:-1}" == "1" ]]; then
  ARGS+=(--skip-heavy)
fi
if [[ -n "${CELL:-}" ]]; then
  ARGS+=(--cell "${CELL}")
fi
if [[ -n "${ONLY:-}" ]]; then
  ARGS+=(--only "${ONLY}")
fi

exec python3 "${PY}" "${ARGS[@]}" "$@"
