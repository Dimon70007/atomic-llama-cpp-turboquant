#!/usr/bin/env bash
# TASK-066 follow-up B1 — shared KV / catch-up decode audit (#27781).
#
# Fail-closed RED until llama_memory_has_shared_cells (or equivalent #27781 port)
# is present. After the #27781 port, this gate should PASS.
#
# Usage:
#   ./tests/run-task066-b1-shared-kv-audit.sh [OUT_DIR]
# Env:
#   TASK066_B1_WAIVE=1 — record PARTIAL as accepted (exit 0); default refuse
#
# Note (TASK-069.1 ST-069.1.1): do not require Cursor-only `rg` on PATH. Suite hosts
# without ripgrep previously reported false VERDICT=MISSING while the API was present.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-b1-shared-kv-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

SUMMARY="${OUT_DIR}/summary.txt"
{
  echo "track=B1"
  echo "contract=ggml-org/llama.cpp#27781 llama_memory_has_shared_cells (or documented equivalent)"
} | tee "${SUMMARY}"

# Portable source search: prefer rg, else grep -R. Never treat "tool missing" as API missing.
search_src() {
  local pattern="$1"
  shift
  local paths=("$@")
  if command -v rg >/dev/null 2>&1; then
    rg -n --glob '*.{c,cpp,h,hpp}' "${pattern}" "${paths[@]}" >/dev/null 2>&1
    return $?
  fi
  if command -v grep >/dev/null 2>&1; then
    grep -Rsn --include='*.c' --include='*.cpp' --include='*.h' --include='*.hpp' \
      -E "${pattern}" "${paths[@]}" >/dev/null 2>&1
    return $?
  fi
  echo "TOOL_MISSING: need rg or grep to audit shared-KV symbols" | tee -a "${SUMMARY}"
  echo "VERDICT=TOOL_MISSING" | tee -a "${SUMMARY}"
  exit 2
}

search_speculative() {
  local pattern="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -n --glob 'speculative.cpp' "${pattern}" "${REPO_ROOT}/common" >/dev/null 2>&1
    return $?
  fi
  if command -v grep >/dev/null 2>&1; then
    grep -n -E "${pattern}" "${REPO_ROOT}/common/speculative.cpp" >/dev/null 2>&1
    return $?
  fi
  echo "TOOL_MISSING: need rg or grep to audit shared-KV symbols" | tee -a "${SUMMARY}"
  echo "VERDICT=TOOL_MISSING" | tee -a "${SUMMARY}"
  exit 2
}

HAS_API=0
HAS_FORK_EQ=0
if search_src 'llama_memory_has_shared_cells' \
    "${REPO_ROOT}/include" "${REPO_ROOT}/src" "${REPO_ROOT}/common"; then
  HAS_API=1
fi
# Ablation-OFF path still keeps the old ctx_other equality (see TASK066_DISABLE_SHARED_KV).
if search_speculative 'is_mem_shared[[:space:]]*=[[:space:]]*llama_get_ctx_other'; then
  HAS_FORK_EQ=1
fi

{
  echo "search_tool=$(command -v rg >/dev/null 2>&1 && echo rg || echo grep)"
  echo "has_llama_memory_has_shared_cells=${HAS_API}"
  echo "has_fork_is_mem_shared_via_ctx_other=${HAS_FORK_EQ}"
} | tee -a "${SUMMARY}"

if [[ "${HAS_API}" -eq 1 ]]; then
  echo "VERDICT=PASS (upstream #27781 API present)" | tee -a "${SUMMARY}"
  exit 0
fi

if [[ "${HAS_FORK_EQ}" -eq 1 ]]; then
  echo "VERDICT=PARTIAL (fork uses llama_get_ctx_other==ctx_tgt; #27781 API still missing)" | tee -a "${SUMMARY}"
  if [[ "${TASK066_B1_WAIVE:-0}" == "1" ]]; then
    echo "WAIVE=1 accepted PARTIAL" | tee -a "${SUMMARY}"
    exit 0
  fi
  echo "FAIL: B1 RED until #27781 API lands (or TASK066_B1_WAIVE=1 after explicit audit note)" >&2
  exit 1
fi

echo "VERDICT=MISSING (no #27781 API and no fork is_mem_shared path found)" | tee -a "${SUMMARY}"
exit 1
