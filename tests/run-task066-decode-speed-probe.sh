#!/usr/bin/env bash
# TASK-066.19 — decode speed probe from a cell BUILD_DIR (fork-local).
#
# Called by tests/task066_keep_stack_speed.py after each cell cmake/build.
# Does NOT rebuild — caller owns BUILD_DIR.
#
# Usage (from atomic-llama-cpp-turboquant root or any cwd; script finds repo):
#   BUILD_DIR=…/builds/C1_fa TASK066_MODEL=…/model.gguf \
#     OUT_DIR=…/cells/C1_fa/nomtp MODE=nomtp \
#     ./tests/run-task066-decode-speed-probe.sh
#   BUILD_DIR=… TASK066_MODEL=… TASK066_MODEL_DRAFT=…/draft.gguf \
#     OUT_DIR=…/cells/C1_fa/mtp MODE=mtp \
#     ./tests/run-task066-decode-speed-probe.sh
#
# Fail-closed: BUILD_DIR, TASK066_MODEL, MODE, OUT_DIR required.
# MTP also requires TASK066_MODEL_DRAFT.
# Writes: $OUT_DIR/run.log, $OUT_DIR/summary.json (tg_median + samples).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BUILD_DIR="${BUILD_DIR:-}"
TASK066_MODEL="${TASK066_MODEL:-}"
TASK066_MODEL_DRAFT="${TASK066_MODEL_DRAFT:-}"
MODE="${MODE:-}"
OUT_DIR="${OUT_DIR:-}"
REPEATS="${REPEATS:-3}"
PROMPT="${PROMPT:-Count from 1 to 80, one number per line.}"
# llama-cli default -c 0 = model n_ctx_train (huge) → appears hung after llama-bench.
CTX="${CTX:-2048}"
# Per-repeat wall timeout (seconds). Empty = no timeout.
MTP_TIMEOUT_SEC="${MTP_TIMEOUT_SEC:-900}"

if [[ -z "${BUILD_DIR}" ]]; then
  echo "missing required BUILD_DIR (refusing silent build dir)" >&2
  exit 1
fi
if [[ -z "${TASK066_MODEL}" ]]; then
  echo "missing required TASK066_MODEL (refusing silent model)" >&2
  exit 1
fi
if [[ -z "${MODE}" ]]; then
  echo "missing required MODE=nomtp|mtp" >&2
  exit 1
fi
if [[ -z "${OUT_DIR}" ]]; then
  echo "missing required OUT_DIR" >&2
  exit 1
fi
if [[ ! -d "${BUILD_DIR}/bin" ]]; then
  echo "FAIL: BUILD_DIR has no bin/: ${BUILD_DIR}" >&2
  exit 2
fi
if [[ ! -f "${TASK066_MODEL}" ]]; then
  echo "FAIL: model GGUF not found: ${TASK066_MODEL}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.json"
LOG="${OUT_DIR}/run.log"
: >"${LOG}"

parse_and_write_summary() {
  local workload="$1"
  local rc="$2"
  local log_path="$3"
  # nomtp: llama-bench JSON rows with n_gen>0; mtp: tok/s lines from llama-cli.
  python3 - "$workload" "$rc" "$log_path" "$SUMMARY" <<'PY'
import json, math, re, statistics, sys
from pathlib import Path

workload, rc_s, log_path, summary_path = sys.argv[1:5]
rc = int(rc_s)
text = Path(log_path).read_text(encoding="utf-8", errors="replace")
samples: list[float] = []

if workload == "nomtp":
    # Prefer structured llama-bench JSON: only decode rows (n_gen > 0).
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            rows = json.loads(text[start : end + 1])
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    if int(row.get("n_gen") or 0) <= 0:
                        continue
                    if "avg_ts" in row:
                        samples.append(float(row["avg_ts"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            samples = []

if not samples:
    for m in re.finditer(
        r'"avg_ts"\s*:\s*([0-9.eE+-]+)|"decode_tok_per_sec"\s*:\s*([0-9.eE+-]+)',
        text,
    ):
        raw = m.group(1) or m.group(2)
        if raw:
            samples.append(float(raw))
if not samples:
    for m in re.finditer(
        r"(?:tg|decode)[^\d]*([0-9]+(?:\.[0-9]+)?)\s*(?:t/?s|tok(?:ens)?/s)",
        text,
        re.I,
    ):
        samples.append(float(m.group(1)))
if not samples:
    for m in re.finditer(
        r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens per second|t/s|tok/s)",
        text,
        re.I,
    ):
        samples.append(float(m.group(1)))

tg_vals = samples
tg_median = float(statistics.median(tg_vals)) if tg_vals else None
payload = {
    "workload": workload,
    "rc": rc,
    "samples": tg_vals,
    "tg_median": tg_median,
}
Path(summary_path).write_text(
    json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
)
if tg_median is None or (isinstance(tg_median, float) and math.isnan(tg_median)):
    print(f"TG_MEDIAN=n/a rc={rc}", flush=True)
    raise SystemExit(3 if rc == 0 else rc)
print(f"TG_MEDIAN={tg_median:.6f} rc={rc}", flush=True)
PY
}

run_with_timeout() {
  # Usage: run_with_timeout SEC command args...
  local sec="$1"
  shift
  if [[ -z "${sec}" || "${sec}" == "0" ]]; then
    "$@"
    return $?
  fi
  if command -v timeout >/dev/null 2>&1; then
    timeout --signal=KILL "${sec}" "$@"
    return $?
  fi
  # macOS: no GNU timeout — python waiter.
  python3 - "$sec" "$@" <<'PY'
import os, signal, subprocess, sys
sec = int(sys.argv[1])
cmd = sys.argv[2:]
p = subprocess.Popen(cmd)
try:
    rc = p.wait(timeout=sec)
except subprocess.TimeoutExpired:
    p.kill()
    try:
        p.wait(timeout=5)
    except Exception:
        os.kill(p.pid, signal.SIGKILL)
    print(f"FAIL: timed out after {sec}s: {' '.join(cmd)}", file=sys.stderr, flush=True)
    raise SystemExit(124)
raise SystemExit(rc)
PY
}

case "${MODE}" in
  nomtp)
    BIN="${BUILD_DIR}/bin/llama-bench"
    if [[ ! -x "${BIN}" ]]; then
      echo "FAIL: missing ${BIN}" >&2
      exit 2
    fi
    set +e
    "${BIN}" -m "${TASK066_MODEL}" \
      -ngl 99 -fa on -ctk q8_0 -ctv turbo4 \
      -b 2048 -ub 512 -t 4 \
      -p 512 -n 128 -r "${REPEATS}" -o json \
      >>"${LOG}" 2>&1
    rc=$?
    set -e
    parse_and_write_summary nomtp "${rc}" "${LOG}"
    exit $?
    ;;
  mtp)
    if [[ -z "${TASK066_MODEL_DRAFT}" ]]; then
      echo "missing required TASK066_MODEL_DRAFT for MODE=mtp" >&2
      exit 1
    fi
    if [[ ! -f "${TASK066_MODEL_DRAFT}" ]]; then
      echo "FAIL: draft GGUF not found: ${TASK066_MODEL_DRAFT}" >&2
      exit 2
    fi
    BIN="${BUILD_DIR}/bin/llama-cli"
    if [[ ! -x "${BIN}" ]]; then
      echo "FAIL: missing ${BIN}" >&2
      exit 2
    fi
    last_rc=0
    i=1
    while [[ "${i}" -le "${REPEATS}" ]]; do
      echo "===== REPEAT ${i}/${REPEATS} =====" >>"${LOG}"
      echo "NOTE: CTX=${CTX} -st --simple-io (avoid model n_ctx_train hang)" >>"${LOG}"
      set +e
      # Critical: -c must be small; default -c 0 = n_ctx_train and looks hung.
      # -st --no-conversation: single non-interactive turn; --simple-io for subprocess logs.
      run_with_timeout "${MTP_TIMEOUT_SEC}" \
        "${BIN}" -m "${TASK066_MODEL}" -md "${TASK066_MODEL_DRAFT}" \
        -ngl 99 -fa on -ctk q8_0 -ctv turbo4 \
        -c "${CTX}" -b 2048 -ub 512 -t 4 -n 128 \
        -no-cnv -st --simple-io \
        --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.95 \
        -p "${PROMPT}" \
        >>"${LOG}" 2>&1
      last_rc=$?
      set -e
      echo "REPEAT_EXIT=${last_rc}" >>"${LOG}"
      if [[ "${last_rc}" -eq 124 ]]; then
        echo "FAIL: MTP llama-cli timed out (CTX=${CTX} TIMEOUT=${MTP_TIMEOUT_SEC}s)" >>"${LOG}"
        echo "FAIL: MTP llama-cli timed out" >&2
        parse_and_write_summary mtp "${last_rc}" "${LOG}" || true
        exit 124
      fi
      if [[ "${last_rc}" -ne 0 ]]; then
        echo "FAIL: MTP llama-cli rc=${last_rc} on repeat ${i}" >>"${LOG}"
        parse_and_write_summary mtp "${last_rc}" "${LOG}" || true
        exit "${last_rc}"
      fi
      i=$((i + 1))
    done
    parse_and_write_summary mtp "${last_rc}" "${LOG}"
    exit $?
    ;;
  *)
    echo "MODE must be nomtp|mtp (got: ${MODE})" >&2
    exit 1
    ;;
esac
