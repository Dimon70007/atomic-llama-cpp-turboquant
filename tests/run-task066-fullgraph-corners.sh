#!/usr/bin/env bash
# TASK-066 ST-3.2f — full-graph decode row0 localizer (patch-C TDD RED gate).
# Lives in the turboquant / llama.cpp tree (not linux_to_mac).
#
# Product-knob lock (operator, 2026-11-06): argv must match
#   linux_to_mac configs/minillm/gemma4_26b_a4b_mtp_metal.yaml
# except ONLY these may differ for test speed / MTP A/B:
#   - context_size  → TASK066_CTX / -c
#   - spec_draft_n_max / spec_draft_p_min (N/A here: localizer has no draft GGUF;
#     residual class is N=1 vs N=2 decode width, not draft acceptance)
# Everything else (batch/ubatch, KV quants, FA, ngl, threads, cache_ram, parallel)
# is pinned below. EXTRA_ARGS that override product knobs are rejected.
#
# Usage:
#   TASK066_MODEL=/path/to/model.gguf TASK066_CTX=16384 \
#     ./tests/run-task066-fullgraph-corners.sh [OUT_DIR]
#   TASK066_MODEL=… TASK066_CTX=16384 TASK066_PROMPT=/path/to/prompt.txt ./tests/...
# Env:
#   TASK066_MODEL   — required GGUF path (fail-closed)
#   TASK066_CTX     — required context size (allowed reduction from YAML 140000)
#   TASK066_PROMPT  — optional prompt file (first-diff / Voyna text)
#   TASK066_N_TOKENS / TASK066_PROMPT_CHARS — optional truncates (forwarded)
#   TASK066_REQUIRE_GPU=1 — exit 77 if no GPU (default 1 on Darwin)
#   TASK066_BISECT=1 — ST-3.2f-ter: cb_eval capture + FIRST_DIFF_TENSOR report
#   TASK066_BISECT_FILTER — optional ECMA regex on tensor names
#   TASK066_BISECT_HASH_ONLY=1 — skip storing flat values (shape-diff inconclusive)
#   BUILD_DIR — cmake build dir (default: <repo>/build)
#   REBUILD=1 — rebuild binary before run
#
# Operator entry from linux_to_mac (preferred):
#   make minillm-task066-fullgraph-bisect LLAMACPP_FORK=… TASK066_MODEL=… TASK066_CTX=… TASK066_N_TOKENS=…

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
OUT_DIR="${1:-${REPO_ROOT}/logs/task066-fullgraph-corners-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT_DIR}"

TEST_BIN="${BUILD_DIR}/bin/test-fullgraph-decode-row0-parity"
MODEL="${TASK066_MODEL:-}"
CTX="${TASK066_CTX:-}"

if [[ -z "${MODEL}" ]]; then
  echo "FAIL: missing required TASK066_MODEL=/path/to.gguf (refusing silent model substitution)" >&2
  exit 2
fi
if [[ ! -f "${MODEL}" ]]; then
  echo "FAIL: TASK066_MODEL not a file: ${MODEL}" >&2
  exit 2
fi
if [[ -z "${CTX}" ]]; then
  echo "FAIL: missing required TASK066_CTX=<n> (allowed test-speed reduction of YAML context_size; refusing silent ctx)" >&2
  exit 2
fi
if ! [[ "${CTX}" =~ ^[0-9]+$ ]] || [[ "${CTX}" -lt 4 ]]; then
  echo "FAIL: TASK066_CTX must be an integer >= 4 (got ${CTX})" >&2
  exit 2
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
  echo "FAIL: BUILD_DIR missing: ${BUILD_DIR}" >&2
  exit 1
fi

# Reject EXTRA_ARGS that would violate the product-knob lock.
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  case " ${EXTRA_ARGS} " in
    *" -b "*|*" --batch"*|*" -ub "*|*" --ubatch"*|*" -ctk "*|*" --cache-type-k"*|*" -ctv "*|*" --cache-type-v"*|*" -ngl "*|*" --n-gpu-layers"*|*" -fa "*|*" --flash-attn"*|*" -cram "*|*" --cache-ram"*|*" -t "*|*" --threads "*|*" -tb "*|*" --threads-batch"*|*" -np "*|*" --parallel"*)
      echo "FAIL: EXTRA_ARGS overrides product knobs (only TASK066_CTX / draft n_max|p_min allowed vs mtp_metal.yaml)" >&2
      echo "  EXTRA_ARGS=${EXTRA_ARGS}" >&2
      exit 2
      ;;
  esac
fi
# Binary also forces product knobs after parse; EXIT=2 = misuse, not RED.

if [[ "${REBUILD:-0}" == "1" ]] || [[ ! -x "${TEST_BIN}" ]]; then
  echo "==> rebuild test-fullgraph-decode-row0-parity"
  cmake --build "${BUILD_DIR}" --target test-fullgraph-decode-row0-parity --parallel "$(sysctl -n hw.ncpu 2>/dev/null || nproc)" \
    | tee "${OUT_DIR}/rebuild.log"
fi

# CLI pins what LLAMA_EXAMPLE_COMMON accepts; binary forces product lock
# (Metal: cache_ram_mib=4096; CUDA TASK-069: TASK069_SUITE_PROFILE=cuda + TASK069_*).
if [[ "${TASK069_SUITE_PROFILE:-}" == "cuda" ]]; then
  : "${TASK069_BATCH:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_BATCH}"
  : "${TASK069_UBATCH:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_UBATCH}"
  : "${TASK069_CTK:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_CTK}"
  : "${TASK069_CTV:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_CTV}"
  : "${TASK069_THREADS:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_THREADS}"
  : "${TASK069_THREADS_BATCH:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_THREADS_BATCH}"
  : "${TASK069_NGL:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_NGL}"
  ARGS=(
    -m "${MODEL}"
    -ngl "${TASK069_NGL}"
    -c "${CTX}"
    -b "${TASK069_BATCH}"
    -ub "${TASK069_UBATCH}"
    -ctk "${TASK069_CTK}"
    -ctv "${TASK069_CTV}"
    -fa on
    -t "${TASK069_THREADS}"
    -tb "${TASK069_THREADS_BATCH}"
    -np "${TASK069_PARALLEL:-1}"
    -ts "${TASK069_TENSOR_SPLIT:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_TENSOR_SPLIT}"
    -mg "${TASK069_MAIN_GPU:?FAIL: TASK069_SUITE_PROFILE=cuda requires TASK069_MAIN_GPU}"
  )
else
  ARGS=(
    -m "${MODEL}"
    -ngl 999
    -c "${CTX}"
    -b 2048
    -ub 512
    -ctk q8_0
    -ctv turbo4
    -fa on
    -t 4
    -tb 4
    -np 1
  )
fi
if [[ -n "${TASK066_PROMPT:-}" ]]; then
  if [[ ! -f "${TASK066_PROMPT}" ]]; then
    echo "FAIL: TASK066_PROMPT not a file: ${TASK066_PROMPT}" >&2
    exit 2
  fi
  ARGS+=(-f "${TASK066_PROMPT}")
fi

# shellcheck disable=SC2206
EXTRA=( ${EXTRA_ARGS:-} )
ARGS+=("${EXTRA[@]}")

if [[ -z "${TASK066_REQUIRE_GPU:-}" ]] && [[ "$(uname -s)" == "Darwin" ]]; then
  export TASK066_REQUIRE_GPU=1
fi

echo "==> FULLGRAPH-C1 (test-fullgraph-decode-row0-parity) [product-knob lock]"
echo "    model=${MODEL}"
echo "    ctx=${CTX} (YAML context_size may be reduced for speed)"
if [[ "${TASK069_SUITE_PROFILE:-}" == "cuda" ]]; then
  echo "    pinned(CUDA suite): -b ${TASK069_BATCH} -ub ${TASK069_UBATCH} -ctk ${TASK069_CTK} -ctv ${TASK069_CTV} -fa on -ngl ${TASK069_NGL} -t ${TASK069_THREADS} -tb ${TASK069_THREADS_BATCH} -ts ${TASK069_TENSOR_SPLIT} -mg ${TASK069_MAIN_GPU}"
else
  echo "    pinned(Metal): -b 2048 -ub 512 -ctk q8_0 -ctv turbo4 -fa on -ngl 999 -t 4 -tb 4 -np 1 (+ binary cram=4096)"
fi
echo "    prompt=${TASK066_PROMPT:-<synthesized>}"
set +e
"${TEST_BIN}" "${ARGS[@]}" 2>&1 | tee "${OUT_DIR}/fullgraph-row0.log"
RC=${PIPESTATUS[0]}
set -e
echo "fullgraph_row0_exit=${RC}" | tee "${OUT_DIR}/fullgraph-row0.exit"

{
  echo "stamp_dir=${OUT_DIR}"
  echo "repo_root=${REPO_ROOT}"
  echo "build_dir=${BUILD_DIR}"
  echo "model=${MODEL}"
  echo "ctx=${CTX}"
  if [[ "${TASK069_SUITE_PROFILE:-}" == "cuda" ]]; then
    echo "product_knobs=cuda_suite:b${TASK069_BATCH},ub${TASK069_UBATCH},ctk=${TASK069_CTK},ctv=${TASK069_CTV},fa=on,ngl=${TASK069_NGL},ts=${TASK069_TENSOR_SPLIT},mg=${TASK069_MAIN_GPU}"
  else
    echo "product_knobs=b2048,ub512,ctk=q8_0,ctv=turbo4,fa=on,ngl=999,cram=4096(binary),t=4,tb=4,np=1"
  fi
  echo "prompt=${TASK066_PROMPT:-}"
  echo "fullgraph_row0_exit=${RC}"
  if [[ "${RC}" -eq 0 ]]; then
    echo "VERDICT=GREEN"
    echo "note=GREEN on this prompt under product knobs; live first-diff/P5 may still MISMATCH — try longer prompt / larger TASK066_CTX"
  elif [[ "${RC}" -eq 77 ]]; then
    echo "VERDICT=HOST_SKIP"
  elif [[ "${RC}" -eq 2 ]]; then
    echo "VERDICT=MISUSE"
    echo "note=argv/env misuse (not a patch-C RED)"
  else
    # Audit: stamp 20261106T204326Z was labeled RED but log was prefix-decode
    # "failed to find a memory slot for batch of size 2048" with CTX=2048 — fixture
    # misuse, not logit N=1 vs N=2 proof. Classify before claiming patch-C RED.
    if grep -q "failed to find a memory slot\|FAIL: prefix decode" "${OUT_DIR}/fullgraph-row0.log" 2>/dev/null; then
      echo "VERDICT=FIXTURE_FAIL"
      echo "note=prefix decode / KV slot failure (often CTX<=batch); NOT proof of row0 N=1 vs N=2 divergence — raise TASK066_CTX (e.g. 3072+) or shorten prompt"
    else
      echo "VERDICT=RED"
      echo "note=full-graph row0 logits not bit-exact under N=2 verify width (patch-C TDD RED; product knobs)"
    fi
  fi
} | tee "${OUT_DIR}/summary.txt"

exit "${RC}"
