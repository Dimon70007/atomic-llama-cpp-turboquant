#!/usr/bin/env python3
"""TASK-069 Phase 2.0d — CUDA fork suite OFF→ON compare (measure-first).

Runs the locked inventory against C_off then C_on, writes suite-summary +
red-metrics, then classified compare-report. Metal-hardcoded harnesses are
tagged N/A_METAL (not scored as FIXED/REGRESS).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]

S_DEFAULT = (
    "FA_TILE;NO_MV_EXT;FLOAT_TILE;MM_MIN;FUSION_CB;"
    "MOE_VIEW_ADD;ISWA_PADS;SHARED_KV;SWA_HWM_LOG"
)

BUILD_TARGETS = [
    "test-iswa-size-swa",
    "test-flash-attn-batch-invariance",
    "test-backend-ops",
    "test-mul-mat-batch-invariance",
    "test-longctx-row0-parity",
    "test-moe-row0-parity",
    "test-moe-router-row0-parity",
    "test-moe-out-row0-parity",
    "test-fullgraph-decode-row0-parity",
    "test-metal-mul-mm-policy",
    "test-ubatch-s8-s1024-row0-parity",
    "llama-cli",
]

# metal_hardcoded: binary/script looks for Metal device name → N/A_METAL on CUDA.
SUITE_INVENTORY: list[dict[str, Any]] = [
    {
        "id": "iswa-size-swa",
        "suite": "iswa",
        "kind": "bin",
        "bin": "test-iswa-size-swa",
        "metal_hardcoded": False,
        "needs_model": False,
    },
    {
        "id": "fa",
        "suite": "fa",
        "kind": "script",
        "script": "run-task066-fa-nsg-corners.sh",
        "metal_hardcoded": False,  # tip prefers Metal but falls back to CUDA GPU
        "needs_model": False,
    },
    {
        "id": "mul-mat",
        "suite": "mul-mat",
        "kind": "script",
        "script": "run-task066-mul-mat-corners.sh",
        "metal_hardcoded": False,
        "needs_model": False,
    },
    {
        "id": "longctx",
        "suite": "longctx",
        "kind": "script",
        "script": "run-task066-longctx-corners.sh",
        "metal_hardcoded": False,
        "needs_model": False,
    },
    {
        "id": "moe",
        "suite": "moe",
        "kind": "script",
        "script": "run-task066-moe-corners.sh",
        "metal_hardcoded": False,
        "needs_model": False,
    },
    {
        "id": "moe-router",
        "suite": "moe-router",
        "kind": "script",
        "script": "run-task066-moe-router-corners.sh",
        "metal_hardcoded": False,
        "needs_model": False,
    },
    {
        "id": "moe-out",
        "suite": "moe-out",
        "kind": "script",
        "script": "run-task066-moe-out-corners.sh",
        "metal_hardcoded": False,
        "needs_model": False,
    },
    {
        "id": "cb-fusion",
        "suite": "cb-fusion",
        "kind": "script",
        "script": "run-task066-cb-fusion-corners.sh",
        "metal_hardcoded": True,  # target only with GGML_METAL=ON
        "needs_model": False,
    },
    {
        "id": "mul-mat-residual",
        "suite": "mul-mat-residual",
        "kind": "script",
        "script": "run-task066-mul-mat-residual-corners.sh",
        "metal_hardcoded": True,  # residual GPU part host-skips without Metal name
        "needs_model": False,
        "partial_cpu": True,
    },
    {
        "id": "b1-shared-kv",
        "suite": "b1-shared-kv",
        "kind": "script",
        "script": "run-task066-b1-shared-kv-audit.sh",
        "metal_hardcoded": False,
        "needs_model": False,
    },
    {
        "id": "mtp-acc-trace",
        "suite": "mtp-acc-trace",
        "kind": "script",
        "script": "run-task066-mtp-acc-trace-checklist.sh",
        "metal_hardcoded": False,
        "needs_model": False,
        "blocked_without_trace": True,
    },
    {
        "id": "mtp-e2e",
        "suite": "mtp-e2e",
        "kind": "script",
        "script": "run-task066-mtp-e2e-corners.sh",
        "metal_hardcoded": False,
        "needs_model": True,
    },
    {
        "id": "ubatch-s8-s1024",
        "suite": "ubatch-s8-s1024",
        "kind": "script",
        "script": "run-task066-ubatch-s8-s1024-corners.sh",
        "metal_hardcoded": False,
        "needs_model": True,
    },
    {
        "id": "ubatch-ladder",
        "suite": "ubatch-ladder",
        "kind": "script",
        "script": "run-task066-ubatch-ladder.sh",
        "metal_hardcoded": False,
        "needs_model": True,
    },
    {
        "id": "mtp-vs-nomtp-ladder",
        "suite": "mtp-vs-nomtp-ladder",
        "kind": "script",
        "script": "run-task066-mtp-vs-nomtp-ladder.sh",
        "metal_hardcoded": False,
        "needs_model": True,
        "needs_draft": True,
    },
    {
        "id": "mtp-knob-matrix",
        "suite": "mtp-knob-matrix",
        "kind": "script",
        "script": "run-task066-mtp-knob-matrix.sh",
        "metal_hardcoded": False,
        "needs_model": True,
        "needs_draft": True,
        "optional_heavy": True,
    },
    {
        "id": "followup",
        "suite": "followup",
        "kind": "script",
        "script": "run-task066-followup-all.sh",
        "metal_hardcoded": False,  # mixed; child Metal parts → SKIP inside
        "needs_model": True,
    },
    {
        "id": "phase-f-battery",
        "suite": "phase-f",
        "kind": "skip",
        "skip_reason": "optional battery; skipped for 2.0d wall-clock (log reason)",
        "metal_hardcoded": False,
        "needs_model": True,
    },
]

ISWA_FAIL_RE = re.compile(
    r"^FAIL\s+(?P<tid>\S+):\s+got\s+(?P<got>-?\d+)\s+want\s+(?P<want>-?\d+)\s*$"
)
GENERIC_GOT_WANT_RE = re.compile(
    r"(?P<tid>\S+).*\bgot\b[=\s:]*(?P<got>-?\d+(?:\.\d+)?).*\bwant\b[=\s:]*(?P<want>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Corner logs: "  N=3 exact=0 max_abs_diff=7.37e-06 FAIL"
ABS_DIFF_FAIL_RE = re.compile(
    r"^\s*(?:ne21=)?N=(?P<n>\d+)\s+.*?max_abs_diff=(?P<diff>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s+FAIL\s*$",
    re.IGNORECASE,
)
# "LONG-C2 ... N=3 ... max_abs_diff=1.39e-06 FAIL" (single-line cases)
ABS_DIFF_FAIL_INLINE_RE = re.compile(
    r"(?P<label>\S.*?)\s+N=(?P<n>\d+)\s+.*?max_abs_diff=(?P<diff>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s+FAIL\s*$",
    re.IGNORECASE,
)
# "MOE-OUT-W [c5-...] ... exact=0 max_abs_diff=1.52e-05 FAIL"
CASE_ABS_DIFF_FAIL_RE = re.compile(
    r"^(?P<label>\S.*?)\s+exact=0\s+max_abs_diff=(?P<diff>[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s+FAIL\s*$",
    re.IGNORECASE,
)
METAL_SKIP_MARKERS = (
    "no Metal backend",
    "host-skip: no Metal",
    "HOST_SKIP",
    "need GGML_METAL=ON",
    "VERDICT=SKIP (no Metal)",
    "VERDICT=HOST_SKIP",
    "unknown target 'test-metal-cb-fusion'",
    "Configure with -DGGML_METAL=ON",
)
NA_BLOCKED_MARKERS = (
    "missing TASK066_ACC_TRACE_LOG",
    "prompt not a file",
    "VERDICT=FIXTURE_FAIL",
)


@dataclass
class RunResult:
    test_id: str
    suite: str
    verdict: str
    rc: int
    log_ref: str
    reason: str = ""
    duration_s: float = 0.0


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(2)


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int | None = None,
) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            return proc.returncode, log_path.read_text(encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            fh.write("\nFAIL: timeout\n")
            return 124, log_path.read_text(encoding="utf-8", errors="replace")


def ensure_builds(build_dir: Path, log_path: Path, env: dict[str, str]) -> None:
    if not build_dir.is_dir():
        fail(f"BUILD_DIR missing: {build_dir}")
    nproc = str(os.cpu_count() or 4)
    cmd = ["cmake", "--build", str(build_dir), "--parallel", nproc, "--target", *BUILD_TARGETS]
    rc, _ = run_cmd(cmd, cwd=REPO_ROOT, env=env, log_path=log_path)
    if rc != 0:
        # Some targets (test-metal-cb-fusion) may be absent without METAL — retry core set.
        core = [t for t in BUILD_TARGETS if t != "test-backend-ops"]
        cmd2 = ["cmake", "--build", str(build_dir), "--parallel", nproc, "--target", *core]
        rc2, _ = run_cmd(cmd2, cwd=REPO_ROOT, env=env, log_path=log_path.with_suffix(".core.log"))
        if rc2 != 0:
            fail(f"cmake --build failed for {build_dir} (see {log_path})")


def classify_verdict(
    *,
    entry: dict[str, Any],
    rc: int,
    text: str,
) -> tuple[str, str]:
    if entry.get("kind") == "skip":
        return "N/A", str(entry.get("skip_reason") or "skipped")

    metal = bool(entry.get("metal_hardcoded"))
    if any(m in text for m in METAL_SKIP_MARKERS) or (
        metal and (rc == 77 or "VERDICT=SKIP (no Metal)" in text)
    ):
        return "N/A", "N/A_METAL"

    if rc == 77:
        return "SKIP", "host-skip rc=77"

    if any(m in text for m in NA_BLOCKED_MARKERS):
        if "FIXTURE_FAIL" in text:
            return "N/A", "FIXTURE_FAIL (raise CTX / shorten prompt)"
        if "ACC_TRACE" in text:
            return "N/A", "blocked: missing TASK066_ACC_TRACE_LOG"
        if "prompt not a file" in text:
            return "N/A", "blocked: missing TASK066_PROMPT fixture"
        return "N/A", "blocked"

    if rc == 0:
        if "VERDICT=PASS" in text or "VERDICT=GREEN" in text or "RESULT PASS" in text:
            return "PASS", ""
        if "VERDICT=FAIL" in text or "RESULT FAIL" in text:
            return "FAIL", "rc=0 but FAIL marker in log"
        return "PASS", "rc=0"

    if rc == 2 and ("missing required" in text or "MISUSE" in text or "refusing silent" in text):
        return "N/A", "misuse/env"

    return "FAIL", f"rc={rc}"


def _abs_diff_row(
    *,
    cell: str,
    suite: str,
    test_id: str,
    case_id: str,
    diff: float,
    log_ref: str,
) -> dict[str, Any]:
    return {
        "cell": cell,
        "suite": suite,
        "test_id": f"{test_id}:{case_id}",
        "verdict": "FAIL",
        "expected": 0.0,
        "received": diff,
        "delta": abs(diff),
        "metric_name": "max_abs_diff",
        "log_ref": log_ref,
        "metrics_parse": "ok",
    }


def parse_red_metrics(
    *,
    cell: str,
    suite: str,
    test_id: str,
    text: str,
    log_ref: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = ISWA_FAIL_RE.match(line.strip())
        if m:
            got = int(m.group("got"))
            want = int(m.group("want"))
            rows.append(
                {
                    "cell": cell,
                    "suite": suite,
                    "test_id": f"{test_id}:{m.group('tid')}",
                    "verdict": "FAIL",
                    "expected": want,
                    "received": got,
                    "delta": abs(got - want),
                    "metric_name": "headroom_or_size",
                    "log_ref": log_ref,
                    "metrics_parse": "ok",
                }
            )
    if rows:
        return rows

    # Prefer child corner logs (c1c2c4.log etc.) when runner.log only has summary.
    # Caller may pass concatenated text; parse max_abs_diff FAIL lines.
    section = ""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if "row0 parity" in s or s.startswith("MOE ") or s.startswith("LONG-") or s.startswith("MOE-OUT"):
            section = s
        m = ABS_DIFF_FAIL_RE.match(line)
        if m:
            case_id = f"{section}|N={m.group('n')}" if section else f"N={m.group('n')}"
            rows.append(
                _abs_diff_row(
                    cell=cell,
                    suite=suite,
                    test_id=test_id,
                    case_id=case_id[:180],
                    diff=float(m.group("diff")),
                    log_ref=log_ref,
                )
            )
            continue
        m2 = CASE_ABS_DIFF_FAIL_RE.match(s)
        if m2 and "max_abs_diff=" in s:
            rows.append(
                _abs_diff_row(
                    cell=cell,
                    suite=suite,
                    test_id=test_id,
                    case_id=m2.group("label")[:180],
                    diff=float(m2.group("diff")),
                    log_ref=log_ref,
                )
            )
            continue
        m3 = ABS_DIFF_FAIL_INLINE_RE.search(s)
        if m3 and "exact=0" in s:
            rows.append(
                _abs_diff_row(
                    cell=cell,
                    suite=suite,
                    test_id=test_id,
                    case_id=f"{m3.group('label')[:120]}|N={m3.group('n')}",
                    diff=float(m3.group("diff")),
                    log_ref=log_ref,
                )
            )
        if len(rows) >= 64:
            break
    if rows:
        return rows

    for line in text.splitlines():
        if "FAIL" not in line and "fail" not in line.lower():
            continue
        m = GENERIC_GOT_WANT_RE.search(line)
        if not m:
            continue
        try:
            got_f = float(m.group("got"))
            want_f = float(m.group("want"))
        except ValueError:
            continue
        tid = m.group("tid").rstrip(":")
        rows.append(
            {
                "cell": cell,
                "suite": suite,
                "test_id": f"{test_id}:{tid}",
                "verdict": "FAIL",
                "expected": want_f,
                "received": got_f,
                "delta": abs(got_f - want_f),
                "metric_name": "got_want",
                "log_ref": log_ref,
                "metrics_parse": "ok",
            }
        )
        if len(rows) >= 32:
            break
    if rows:
        return rows

    excerpt = "\n".join(text.splitlines()[-40:])
    return [
        {
            "cell": cell,
            "suite": suite,
            "test_id": test_id,
            "verdict": "FAIL",
            "expected": None,
            "received": None,
            "delta": None,
            "metric_name": "unparsed",
            "log_ref": log_ref,
            "metrics_parse": "fail",
            "raw_excerpt": excerpt[-2000:],
        }
    ]


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def _gather_parse_text(log_path: Path, out_sub: Path) -> str:
    chunks: list[str] = []
    if log_path.is_file():
        chunks.append(log_path.read_text(encoding="utf-8", errors="replace"))
    # Prefer child corner logs for numeric RED (often richer than summary).
    for child in sorted(out_sub.glob("*.log")):
        chunks.append(child.read_text(encoding="utf-8", errors="replace"))
    for child in sorted(out_sub.rglob("*.log")):
        if child.parent == out_sub:
            continue
        chunks.append(child.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks)


def run_cell(
    *,
    cell: str,
    build_dir: Path,
    stamp_dir: Path,
    model: str,
    draft: str,
    ctx: str,
    prompt: str,
    env_base: dict[str, str],
    skip_heavy: bool,
    only: set[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cell_dir = stamp_dir / cell
    raw_dir = cell_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    env = dict(env_base)
    env["BUILD_DIR"] = str(build_dir)
    env["REBUILD"] = "0"
    env["TASK066_MODEL"] = model
    env["TASK066_CTX"] = ctx
    env["MAIN_GGUF"] = model
    env["TASK066_MODEL_DRAFT"] = draft
    env["DRAFT_GGUF"] = draft
    env["TASK066_STOP_ON_FAIL"] = "0"
    env["TASK066_ALLOW_SHARED_METAL"] = "1"
    env["TASK066_REQUIRE_GPU"] = "1"
    # Avoid Metal-default filter on CUDA smoke (still N/A for Metal units).
    env["BACKEND_FILTER"] = env.get("BACKEND_FILTER", "CUDA0")
    if prompt:
        env["TASK066_PROMPT"] = prompt
    # Followup: skip Metal-only / blocked checklist parts (still recorded as N/A elsewhere).
    env.setdefault(
        "TASK066_FOLLOWUP_SKIP",
        "cb-fusion,mtp-acc-trace,mul-mat-residual",
    )
    # Product-ish ladder (README) — still multi-ubatch; shorter than full 1..2048.
    env.setdefault("TASK066_UBATCH_LIST", "8 256 512 1024")
    env.setdefault("TASK066_REPEAT_LARGE", "")

    ensure_builds(build_dir, raw_dir / "cmake-build.log", env)

    summaries: list[dict[str, Any]] = []
    red_rows: list[dict[str, Any]] = []

    for entry in SUITE_INVENTORY:
        tid = str(entry["id"])
        if only is not None and tid not in only:
            continue
        suite = str(entry["suite"])
        out_sub = raw_dir / tid
        out_sub.mkdir(parents=True, exist_ok=True)
        log_rel = f"raw/{tid}/runner.log"
        log_path = cell_dir / log_rel

        if entry.get("kind") == "skip" or (skip_heavy and entry.get("optional_heavy")):
            reason = str(entry.get("skip_reason") or "optional_heavy skipped")
            if skip_heavy and entry.get("optional_heavy"):
                reason = "optional_heavy skipped (--skip-heavy)"
            summaries.append(
                {
                    "id": tid,
                    "suite": suite,
                    "verdict": "N/A",
                    "rc": 0,
                    "log_path": log_rel,
                    "reason": reason,
                    "duration_s": 0.0,
                }
            )
            log_path.write_text(f"SKIP: {reason}\n", encoding="utf-8")
            print(f"[{cell}] N/A {tid}: {reason}")
            continue

        if entry.get("needs_draft") and not draft:
            summaries.append(
                {
                    "id": tid,
                    "suite": suite,
                    "verdict": "N/A",
                    "rc": 2,
                    "log_path": log_rel,
                    "reason": "missing TASK066_MODEL_DRAFT",
                    "duration_s": 0.0,
                }
            )
            log_path.write_text("SKIP: missing draft GGUF\n", encoding="utf-8")
            print(f"[{cell}] N/A {tid}: missing draft")
            continue

        t0 = time.time()
        if entry["kind"] == "bin":
            bin_path = build_dir / "bin" / str(entry["bin"])
            if not bin_path.is_file():
                summaries.append(
                    {
                        "id": tid,
                        "suite": suite,
                        "verdict": "N/A",
                        "rc": 2,
                        "log_path": log_rel,
                        "reason": f"missing binary {bin_path}",
                        "duration_s": 0.0,
                    }
                )
                print(f"[{cell}] N/A {tid}: missing binary")
                continue
            rc, text = run_cmd([str(bin_path)], cwd=REPO_ROOT, env=env, log_path=log_path)
        else:
            script = REPO_ROOT / "tests" / str(entry["script"])
            if not script.is_file():
                summaries.append(
                    {
                        "id": tid,
                        "suite": suite,
                        "verdict": "N/A",
                        "rc": 2,
                        "log_path": log_rel,
                        "reason": f"missing script {script}",
                        "duration_s": 0.0,
                    }
                )
                print(f"[{cell}] N/A {tid}: missing script")
                continue
            # Child runners take OUT_DIR as $1.
            rc, text = run_cmd(
                ["bash", str(script), str(out_sub)],
                cwd=REPO_ROOT,
                env=env,
                log_path=log_path,
            )
            # Also copy child summary if present.
            child_sum = out_sub / "summary.txt"
            if child_sum.is_file():
                (out_sub / "summary.copied.txt").write_text(
                    child_sum.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8",
                )

        dur = time.time() - t0
        parse_text = _gather_parse_text(log_path, out_sub)
        verdict, reason = classify_verdict(entry=entry, rc=rc, text=parse_text)
        if verdict not in ("N/A", "SKIP") and (
            "VERDICT=RED" in parse_text or ("VERDICT=FAIL" in parse_text and "FIXTURE_FAIL" not in parse_text)
        ):
            if verdict == "PASS":
                verdict = "FAIL"

        summaries.append(
            {
                "id": tid,
                "suite": suite,
                "verdict": verdict,
                "rc": rc,
                "log_path": log_rel,
                "reason": reason,
                "duration_s": round(dur, 3),
                "metal_hardcoded": bool(entry.get("metal_hardcoded")),
            }
        )
        print(f"[{cell}] {verdict} {tid} rc={rc} {dur:.1f}s {reason}", flush=True)

        if verdict == "FAIL":
            red_rows.extend(
                parse_red_metrics(
                    cell=cell,
                    suite=suite,
                    test_id=tid,
                    text=parse_text,
                    log_ref=log_rel,
                )
            )

    # Merge with prior summary when --only re-runs a subset.
    summary_path = cell_dir / "suite-summary.json"
    if only is not None and summary_path.is_file():
        prev = json.loads(summary_path.read_text(encoding="utf-8")).get("tests", [])
        by_id = {t["id"]: t for t in prev}
        for t in summaries:
            by_id[t["id"]] = t
        summaries = [by_id[k] for k in sorted(by_id)]
    if only is not None and (cell_dir / "red-metrics.jsonl").is_file():
        prev_red = [
            json.loads(line)
            for line in (cell_dir / "red-metrics.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        kept = [r for r in prev_red if str(r.get("test_id", "")).split(":", 1)[0] not in only]
        red_rows = kept + red_rows

    write_json(cell_dir / "suite-summary.json", {"cell": cell, "tests": summaries})
    append_jsonl(cell_dir / "red-metrics.jsonl", red_rows)
    return summaries, red_rows


def _num(v: Any) -> float | None:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    return None


def compare(
    *,
    stamp_dir: Path,
    off_sum: list[dict[str, Any]],
    on_sum: list[dict[str, Any]],
    off_red: list[dict[str, Any]],
    on_red: list[dict[str, Any]],
) -> dict[str, Any]:
    off_by_id = {t["id"]: t for t in off_sum}
    on_by_id = {t["id"]: t for t in on_sum}
    ids = sorted(set(off_by_id) | set(on_by_id))

    buckets: dict[str, list[dict[str, Any]]] = {
        "FIXED": [],
        "REGRESS": [],
        "PARTIAL_HELP": [],
        "STILL_RED": [],
        "STILL_GREEN": [],
        "N/A": [],
    }

    off_red_by_tid = {r["test_id"]: r for r in off_red if r.get("metrics_parse") == "ok"}
    on_red_by_tid = {r["test_id"]: r for r in on_red if r.get("metrics_parse") == "ok"}

    for tid in ids:
        a = off_by_id.get(tid, {})
        b = on_by_id.get(tid, {})
        va = str(a.get("verdict", "N/A"))
        vb = str(b.get("verdict", "N/A"))
        row = {
            "test_id": tid,
            "off": va,
            "on": vb,
            "off_rc": a.get("rc"),
            "on_rc": b.get("rc"),
            "off_reason": a.get("reason", ""),
            "on_reason": b.get("reason", ""),
        }
        if va == "N/A" or vb == "N/A" or va == "SKIP" or vb == "SKIP":
            row["class"] = "N/A"
            if "N/A_METAL" in str(a.get("reason", "")) or "N/A_METAL" in str(b.get("reason", "")):
                row["na_tag"] = "N/A_METAL"
            buckets["N/A"].append(row)
            continue
        if va == "FAIL" and vb == "PASS":
            row["class"] = "FIXED"
            buckets["FIXED"].append(row)
        elif va == "PASS" and vb == "FAIL":
            row["class"] = "REGRESS"
            buckets["REGRESS"].append(row)
        elif va == "PASS" and vb == "PASS":
            row["class"] = "STILL_GREEN"
            buckets["STILL_GREEN"].append(row)
        elif va == "FAIL" and vb == "FAIL":
            # PARTIAL_HELP only with matching numeric metrics.
            improved = False
            for key, orow in off_red_by_tid.items():
                if not key.startswith(tid):
                    continue
                nrow = on_red_by_tid.get(key)
                if not nrow:
                    continue
                if orow.get("metric_name") != nrow.get("metric_name"):
                    continue
                d0 = _num(orow.get("delta"))
                d1 = _num(nrow.get("delta"))
                if d0 is None or d1 is None:
                    continue
                if d1 < d0:
                    improved = True
                    row["partial_metric"] = key
                    row["delta_off"] = d0
                    row["delta_on"] = d1
                    break
            if improved:
                row["class"] = "PARTIAL_HELP"
                buckets["PARTIAL_HELP"].append(row)
            else:
                row["class"] = "STILL_RED"
                buckets["STILL_RED"].append(row)
        else:
            row["class"] = "N/A"
            buckets["N/A"].append(row)

    counts = {k: len(v) for k, v in buckets.items()}
    summary = {"counts": counts, "buckets": buckets}
    write_json(stamp_dir / "compare-summary.json", summary)

    lines = [
        "# TASK-069 Phase 2.0d — CUDA suite OFF→ON compare",
        "",
        f"**Stamp dir:** `{stamp_dir}`",
        "",
        "## Counts",
        "",
        "| Class | N |",
        "|-------|---|",
    ]
    for k in ("FIXED", "REGRESS", "PARTIAL_HELP", "STILL_RED", "STILL_GREEN", "N/A"):
        lines.append(f"| **{k}** | {counts.get(k, 0)} |")
    lines.append("")

    def section(title: str, key: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        rows = buckets.get(key) or []
        if not rows:
            lines.append("_none_")
            lines.append("")
            return
        lines.append("| test_id | off | on | notes |")
        lines.append("|---------|-----|----|-------|")
        for r in rows:
            note = r.get("na_tag") or r.get("partial_metric") or r.get("on_reason") or r.get("off_reason") or ""
            if key == "PARTIAL_HELP":
                note = f"delta {r.get('delta_off')} -> {r.get('delta_on')} ({r.get('partial_metric')})"
            lines.append(
                f"| `{r['test_id']}` | {r['off']} | {r['on']} | {note} |"
            )
        lines.append("")

    section("FIXED (перестали падать)", "FIXED")
    section("REGRESS (не падали, начали падать)", "REGRESS")
    section("PARTIAL_HELP (ещё RED, но closer)", "PARTIAL_HELP")
    section("STILL_RED", "STILL_RED")
    section("STILL_GREEN (count detail)", "STILL_GREEN")
    section("N/A (incl. N/A_METAL)", "N/A")

    lines.extend(
        [
            "## Hard rules observed",
            "",
            "- Metal-hardcoded suites → `N/A_METAL` (not CUDA HELP).",
            "- PARTIAL_HELP only when same `test_id`+`metric_name` numeric in both red-metrics.",
            "- Product LiteLLM `@63` is **not** in this stamp (Phase 2.1).",
            "",
        ]
    )
    (stamp_dir / "compare-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TASK-069 2.0d CUDA suite OFF→ON compare")
    p.add_argument("--stamp-dir", default="", help="output root; default logs/task069-cuda-suite-<UTC>")
    p.add_argument("--off-build", required=True)
    p.add_argument("--on-build", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--draft", default="")
    p.add_argument("--ctx", required=True)
    p.add_argument("--S", dest="S", default=S_DEFAULT)
    p.add_argument("--head", default="")
    p.add_argument("--cell", choices=("off", "on", "both", "compare-only"), default="both")
    p.add_argument("--skip-heavy", action="store_true", help="skip mtp-knob-matrix")
    p.add_argument("--skip-phase-f", action="store_true", default=True)
    p.add_argument("--prompt", default="", help="TASK066_PROMPT path (fail-closed if e2e needs it)")
    p.add_argument(
        "--suite-config",
        default="",
        help="path to gemma4_*_cuda_task069_suite.yaml (exports TASK069_* CUDA knobs)",
    )
    p.add_argument(
        "--only",
        default="",
        help="comma-separated test ids to run (merge into existing stamp summaries)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    model = str(args.model).strip()
    ctx = str(args.ctx).strip()
    draft = str(args.draft).strip()
    if not model:
        fail("missing required --model (refusing silent model substitution)")
    if not Path(model).is_file():
        fail(f"--model not a file: {model}")
    if not ctx:
        fail("missing required --ctx (refusing silent ctx)")
    if not ctx.isdigit() or int(ctx) < 4:
        fail(f"--ctx must be integer >= 4 (got {ctx})")

    off_build = Path(args.off_build).resolve()
    on_build = Path(args.on_build).resolve()
    if not off_build.is_dir():
        fail(f"--off-build missing: {off_build}")
    if not on_build.is_dir():
        fail(f"--on-build missing: {on_build}")

    stamp = args.stamp_dir.strip() or str(REPO_ROOT / "logs" / f"task069-cuda-suite-{utc_stamp()}")
    stamp_dir = Path(stamp).resolve()
    stamp_dir.mkdir(parents=True, exist_ok=True)

    head = args.head.strip()
    if not head:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()

    prompt = str(args.prompt).strip()
    if not prompt:
        prompt = str(Path(REPO_ROOT) / "tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt")
    if not Path(prompt).is_file():
        fail(f"--prompt not a file: {prompt}")

    only_set: set[str] | None = None
    if str(args.only).strip():
        only_set = {x.strip() for x in str(args.only).split(",") if x.strip()}
        if not only_set:
            fail("--only parsed empty (refusing silent full-suite substitution)")

    suite_config = str(args.suite_config).strip() or str(os.environ.get("TASK069_SUITE_CONFIG") or "").strip()
    if not suite_config:
        fail("missing required --suite-config / TASK069_SUITE_CONFIG (CUDA suite YAML)")
    # Load knobs into this process so child runners inherit CUDA product lock.
    knobs_mod = REPO_ROOT / "tests" / "task069_suite_knobs.py"
    if not knobs_mod.is_file():
        fail(f"missing knobs loader: {knobs_mod}")
    sys.path.insert(0, str(REPO_ROOT / "tests"))
    from task069_suite_knobs import load_suite_config  # noqa: WPS433

    suite_knobs = load_suite_config(Path(suite_config))
    os.environ.update(suite_knobs)
    # Keep explicit --ctx as operator override of suite context_size.
    if ctx:
        os.environ["TASK066_CTX"] = ctx
        os.environ["TASK069_CTX"] = ctx

    meta = {
        "task": "TASK-069",
        "phase": "2.0d",
        "stamp_utc": stamp_dir.name.replace("task069-cuda-suite-", ""),
        "head_sha": head,
        "S": args.S,
        "S_rule": "C_on restores only ids in S; Metal members inert on CUDA",
        "off_build_dir": str(off_build),
        "on_build_dir": str(on_build),
        "model": model,
        "draft": draft,
        "ctx": int(ctx),
        "prompt": prompt,
        "suite_config": str(Path(suite_config).resolve()),
        "suite_profile": suite_knobs.get("TASK069_SUITE_PROFILE"),
        "suite_knobs": {
            k: suite_knobs[k]
            for k in (
                "TASK069_BATCH",
                "TASK069_UBATCH",
                "TASK069_CTK",
                "TASK069_CTV",
                "TASK069_CRAM",
                "TASK069_TENSOR_SPLIT",
                "TASK069_MAIN_GPU",
                "TASK069_THREADS",
            )
            if k in suite_knobs
        },
        "only": sorted(only_set) if only_set else None,
        "host": os.uname().nodename if hasattr(os, "uname") else "",
        "metrics_convention": "delta=abs(received-expected); PARTIAL_HELP requires same test_id+metric_name numeric both cells",
        "inventory": [e["id"] for e in SUITE_INVENTORY],
        "nvidia_visible": Path("/dev/nvidia0").exists(),
        "ubatch_list": os.environ.get("TASK066_UBATCH_LIST", "8 256 512 1024"),
    }
    write_json(stamp_dir / "meta.json", meta)
    print(f"stamp={stamp_dir}", flush=True)
    print(f"nvidia0={meta['nvidia_visible']}", flush=True)
    if not meta["nvidia_visible"]:
        print(
            "WARN: /dev/nvidia0 missing — GPU e2e will SKIP/CPU-fallback; use non-sandbox TTY",
            file=sys.stderr,
        )

    env_base = os.environ.copy()
    # Ensure child sees GPUs when present.
    env_base.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

    off_sum: list[dict[str, Any]] = []
    on_sum: list[dict[str, Any]] = []
    off_red: list[dict[str, Any]] = []
    on_red: list[dict[str, Any]] = []

    if args.cell in ("off", "both"):
        off_sum, off_red = run_cell(
            cell="off",
            build_dir=off_build,
            stamp_dir=stamp_dir,
            model=model,
            draft=draft,
            ctx=ctx,
            prompt=prompt,
            env_base=env_base,
            skip_heavy=bool(args.skip_heavy),
            only=only_set,
        )
    else:
        off_sum = json.loads((stamp_dir / "off" / "suite-summary.json").read_text())["tests"]
        off_red = [
            json.loads(line)
            for line in (stamp_dir / "off" / "red-metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]

    if args.cell in ("on", "both"):
        on_sum, on_red = run_cell(
            cell="on",
            build_dir=on_build,
            stamp_dir=stamp_dir,
            model=model,
            draft=draft,
            ctx=ctx,
            prompt=prompt,
            env_base=env_base,
            skip_heavy=bool(args.skip_heavy),
            only=only_set,
        )
    elif args.cell == "compare-only":
        on_sum = json.loads((stamp_dir / "on" / "suite-summary.json").read_text())["tests"]
        on_red = [
            json.loads(line)
            for line in (stamp_dir / "on" / "red-metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]

    if off_sum and on_sum:
        summary = compare(
            stamp_dir=stamp_dir,
            off_sum=off_sum,
            on_sum=on_sum,
            off_red=off_red,
            on_red=on_red,
        )
        print("compare counts:", json.dumps(summary["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
