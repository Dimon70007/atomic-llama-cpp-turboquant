#!/usr/bin/env bash
# TASK-066.19 Phase F — C_final_tip speed + final suites into stamp/final/.
#
# Metal GPU: macOS Terminal only (agent sandbox → device=null).
# Run only after C0–C4 are all GREEN in STAMP_DIR/results.json.
#
# Usage (fork root):
#   MODEL_NOMTP=gemma4-26b-a4b-mtp-nomtp MODEL_MTP=gemma4-26b-a4b-mtp \
#     TASK066_MODEL=…/UD-IQ4_XS.gguf TASK066_MODEL_DRAFT=…/mtp-…Q8_0.gguf \
#     BUILD_DIR=…/build-task066-cb-ws \
#     STAMP_DIR=logs/task066-19-keep-stack-20261110T021337Z \
#     ./tests/run-task066-phase-f-battery.sh
#
#   PHASE_F_MODE=speed   # F1 only (C_final_tip); skip suites
#   PHASE_F_MODE=suites  # F2 only (requires C_final_tip GREEN already)
#   PHASE_F_MODE=all     # F1 then F2 (default)
#
# SSOT: tests/TASK-066.19-keep-stack-speed-gates.md § Phase F

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PHASE_F_MODE="${PHASE_F_MODE:-all}"
STAMP_DIR="${STAMP_DIR:-}"
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-task066-cb-ws}"
KEEP_STACK="${REPO_ROOT}/tests/run-task066-keep-stack-speed.sh"
CORNERS="${REPO_ROOT}/tests/run-task066-metal-corners.sh"
MTP_LADDER="${REPO_ROOT}/tests/run-task066-mtp-vs-nomtp-ladder.sh"
UNIT_BIN="${BUILD_DIR}/bin/test-iswa-size-swa"

if [[ -z "${STAMP_DIR}" ]]; then
  echo "FAIL: missing required STAMP_DIR (refusing silent stamp)" >&2
  exit 1
fi
if [[ "${STAMP_DIR}" != /* ]]; then
  STAMP_DIR="${REPO_ROOT}/${STAMP_DIR}"
fi
STAMP_DIR="$(cd "${STAMP_DIR}" && pwd)"

if [[ -z "${MODEL_NOMTP:-}" || -z "${MODEL_MTP:-}" ]]; then
  echo "FAIL: MODEL_NOMTP and MODEL_MTP required" >&2
  exit 1
fi
if [[ -z "${TASK066_MODEL:-}" || -z "${TASK066_MODEL_DRAFT:-}" ]]; then
  echo "FAIL: TASK066_MODEL and TASK066_MODEL_DRAFT required" >&2
  exit 1
fi

case "${PHASE_F_MODE}" in
  all|speed|suites) ;;
  *)
    echo "FAIL: PHASE_F_MODE=${PHASE_F_MODE} (allowed: all|speed|suites)" >&2
    exit 1
    ;;
esac

RESULTS="${STAMP_DIR}/results.json"
if [[ ! -f "${RESULTS}" ]]; then
  echo "FAIL: missing ${RESULTS}" >&2
  exit 2
fi

python3 - "${RESULTS}" <<'PY'
import json, sys
path = sys.argv[1]
rows = json.loads(open(path, encoding="utf-8").read())
by = {r.get("cell_id"): r for r in rows if isinstance(r, dict)}
need = ["C0_master", "C1_fa", "C2_fa_mv", "C3_fa_mv_float", "C4_keep4"]
missing = [c for c in need if c not in by]
if missing:
    raise SystemExit(f"FAIL: stamp missing cells {missing}")
bad = [c for c in need if by[c].get("verdict") != "GREEN"]
if bad:
    raise SystemExit(
        f"FAIL: Phase F forbidden — non-GREEN among C0–C4: {bad}"
    )
print("PRECHECK C0–C4 all GREEN")
PY

FINAL_DIR="${STAMP_DIR}/final"
mkdir -p "${FINAL_DIR}"
OPLOG="${REPO_ROOT}/logs/run-task066-phase-f-$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "${REPO_ROOT}/logs"

log() {
  echo "$@" | tee -a "${OPLOG}"
}

append_report_section() {
  local title="$1"
  local body_file="$2"
  {
    echo ""
    echo "## ${title}"
    echo ""
    cat "${body_file}"
  } >> "${STAMP_DIR}/REPORT.md"
}

run_speed() {
  log "==> F1 C_final_tip speed (--phase-f --skip-build)"
  log "    STAMP_DIR=${STAMP_DIR}"
  log "    BUILD_DIR=${BUILD_DIR}"
  export MODEL_NOMTP MODEL_MTP TASK066_MODEL TASK066_MODEL_DRAFT BUILD_DIR
  set +e
  bash "${KEEP_STACK}" \
    --stamp-dir "${STAMP_DIR}" \
    --phase-f \
    --skip-build \
    2>&1 | tee -a "${OPLOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    log "FAIL: C_final_tip speed rc=${rc} (floor miss = NEGATIVE; suites forbidden)"
    echo "phase_f_speed_exit=${rc}" > "${FINAL_DIR}/speed.exit"
    exit "${rc}"
  fi
  echo "phase_f_speed_exit=0" > "${FINAL_DIR}/speed.exit"
  python3 - "${RESULTS}" <<'PY'
import json, sys
rows = json.loads(open(sys.argv[1], encoding="utf-8").read())
tip = next((r for r in rows if r.get("cell_id") == "C_final_tip"), None)
if tip is None:
    raise SystemExit("FAIL: C_final_tip missing from results.json after --phase-f")
if tip.get("verdict") != "GREEN":
    raise SystemExit(f"FAIL: C_final_tip verdict={tip.get('verdict')} (suites forbidden)")
print(
    f"C_final_tip GREEN nomtp={tip.get('nomtp_tg')} mtp={tip.get('mtp_tg')}"
)
PY
}

# Classify ubatch exit: product path GREEN + only stress C1 RED = allowlisted (066.5/066.6).
classify_ubatch_verdict() {
  local logf="$1"
  local rc="$2"
  if [[ "${rc}" -eq 0 ]]; then
    echo "GREEN"
    return
  fi
  if [[ ! -f "${logf}" ]]; then
    echo "RED"
    return
  fi
  if grep -q 'C2p PASS — both product pairs bit-exact' "${logf}" \
    && grep -q 'RESULT PASS (GREEN) — C1p both product pairs' "${logf}" \
    && grep -q 'RESULT PASS (GREEN) — C1jp both product pairs' "${logf}" \
    && grep -q 'RESULT FAIL (RED) — greedy token stream not bit-exact S16 vs S1024' "${logf}" \
    && ! grep -q 'RESULT FAIL (RED) — C2p\|RESULT FAIL (RED) — C1p\|RESULT FAIL (RED) — C1jp\|RESULT FAIL (REGRESS)' "${logf}"; then
    echo "ALLOWLISTED_STRESS_C1"
    return
  fi
  echo "RED"
}

run_one_suite() {
  local name="$1"
  shift
  local out="${FINAL_DIR}/${name}"
  mkdir -p "${out}"
  log "==> F2 suite: ${name}"
  set +e
  "$@" 2>&1 | tee "${out}/run.log" | tee -a "${OPLOG}"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "exit=${rc}" | tee "${out}/exit.txt"
  local verd="GREEN"
  if [[ "${rc}" -ne 0 ]]; then
    verd="RED"
    if [[ "${name}" == "ubatch-s8-s1024" ]]; then
      verd="$(classify_ubatch_verdict "${out}/run.log" "${rc}")"
    fi
  fi
  echo "${name}|${verd}|exit=${rc}" >> "${FINAL_DIR}/scoreboard.txt"
  return 0
}

run_suites() {
  log "==> F2 suites into ${FINAL_DIR}"
  : > "${FINAL_DIR}/scoreboard.txt"
  export BUILD_DIR
  export TASK066_MODEL
  export TASK066_MODEL_DRAFT
  export MAIN_GGUF="${TASK066_MODEL}"
  export DRAFT_GGUF="${TASK066_MODEL_DRAFT}"

  if [[ -x "${UNIT_BIN}" ]]; then
    run_one_suite test-iswa-size-swa "${UNIT_BIN}"
  else
    log "WARN: missing ${UNIT_BIN} — record MISSING"
    echo "test-iswa-size-swa|MISSING|no-bin" >> "${FINAL_DIR}/scoreboard.txt"
  fi

  # Known residual: stress C1 may RED — record, do not waive as KEEP.
  # Ubatch runner fail-closes without TASK066_CTX. Phase F locks ladder ctx=3072
  # (066.17 stamps); not a silent product-140k substitute.
  log "    ubatch TASK066_CTX=3072 (Phase F ladder lock)"
  run_one_suite ubatch-s8-s1024 \
    env SUITE=ubatch-s8-s1024 TASK066_CTX=3072 \
    TASK066_MODEL="${TASK066_MODEL}" \
    bash "${CORNERS}" "${FINAL_DIR}/ubatch-s8-s1024"

  run_one_suite metal-corners-all \
    env SUITE=all bash "${CORNERS}" "${FINAL_DIR}/metal-corners-all"

  run_one_suite mtp-vs-nomtp \
    env TASK066_SAMPLE_MODE=product TASK066_UBATCH_LIST=512 \
    bash "${MTP_LADDER}" "${FINAL_DIR}/mtp-vs-nomtp"

  {
    echo "stamp=${STAMP_DIR}"
    echo "build_dir=${BUILD_DIR}"
    echo "scoreboard:"
    cat "${FINAL_DIR}/scoreboard.txt"
    echo ""
    echo "Known residual allowlist (pre-declared in SSOT):"
    echo "  - Stress C1 16↔1024 RED (066.5/066.6)"
    echo "  - S_mul_mat_res / residual vs N=8 under tip float-tile (066.7) if still RED"
    echo "  - S_b1_shared MISSING if API still absent"
    echo "Anything else RED = new finding (do not mark 066.19 Done)."
    echo ""
    echo "e2e verify-minillm-macos: run from product repo after tip sync:"
    echo "  make verify-minillm-macos MODEL=gemma4-26b-a4b-mtp"
    echo "  make verify-minillm-macos MODEL=gemma4-26b-a4b-mtp-nomtp"
    echo "Stamp exits under final/verify-minillm-macos/ when operator runs them."
  } | tee "${FINAL_DIR}/SUMMARY.md"

  append_report_section "Phase F suite scoreboard" "${FINAL_DIR}/SUMMARY.md"
  log "WROTE ${FINAL_DIR}/SUMMARY.md (appended to REPORT.md)"
}

log "OPLOG=${OPLOG}"
log "PHASE_F_MODE=${PHASE_F_MODE}"

case "${PHASE_F_MODE}" in
  speed)
    run_speed
    ;;
  suites)
    python3 - "${RESULTS}" <<'PY'
import json, sys
rows = json.loads(open(sys.argv[1], encoding="utf-8").read())
tip = next((r for r in rows if r.get("cell_id") == "C_final_tip"), None)
if tip is None or tip.get("verdict") != "GREEN":
    raise SystemExit("FAIL: suites mode needs C_final_tip GREEN in results.json")
print("PRECHECK C_final_tip GREEN")
PY
    run_suites
    ;;
  all)
    run_speed
    run_suites
    ;;
esac

log "DONE Phase F mode=${PHASE_F_MODE}"
log "See ${STAMP_DIR}/REPORT.md and ${FINAL_DIR}/"
