#!/usr/bin/env python3
"""TASK-066.18 — src-delta lever ablation orchestrator (tip−1 / master+1).

One lever per cell. All-off alone ≡ master is not the primary experiment.
Writes stamp under ``logs/task066-18-ablation-<UTC>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

_TESTS_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _TESTS_DIR.parent
_DEFAULT_CFG: Final[Path] = _TESTS_DIR / "task066_src_lever_ablation.json"
_CORNERS: Final[Path] = _TESTS_DIR / "run-task066-metal-corners.sh"
_UBATCH: Final[Path] = _TESTS_DIR / "run-task066-ubatch-s8-s1024-corners.sh"

_RE_MAD: Final[re.Pattern[str]] = re.compile(r"\bmad[=:\s]+([0-9.eE+-]+)")
_RE_RESULT_FAIL: Final[re.Pattern[str]] = re.compile(r"RESULT FAIL|\bRED\b|exact=0")
_RE_RESULT_PASS: Final[re.Pattern[str]] = re.compile(r"RESULT PASS|exact=1|\bGREEN\b")


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _fail(msg: str, code: int = 2) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _load_cfg(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"config JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        _fail(f"config root must be object: {path}")
    levers = data.get("levers")
    suites = data.get("suites")
    if not isinstance(levers, list) or not levers:
        _fail("config.levers must be a non-empty list")
    if not isinstance(suites, list) or not suites:
        _fail("config.suites must be a non-empty list")
    seen_l: set[str] = set()
    for lev in levers:
        if not isinstance(lev, dict):
            _fail("each levers[] entry must be an object")
        lid = str(lev.get("id") or "").strip()
        cmake = str(lev.get("cmake") or "").strip()
        if not lid or not cmake:
            _fail("levers[].id and levers[].cmake required")
        if lid in seen_l:
            _fail(f"duplicate lever id: {lid}")
        seen_l.add(lid)
    seen_s: set[str] = set()
    for row in suites:
        if not isinstance(row, dict):
            _fail("each suites[] entry must be an object")
        sid = str(row.get("id") or "").strip()
        if not sid:
            _fail("suites[].id required")
        if sid in seen_s:
            _fail(f"duplicate suite id: {sid}")
        seen_s.add(sid)
        kind = str(row.get("kind") or "").strip()
        if kind not in ("suite", "bin", "ubatch"):
            _fail(f"suite {sid}: kind must be suite|bin|ubatch")
    return data


def _lever_by_id(cfg: dict[str, Any], lid: str) -> dict[str, Any]:
    for lev in cfg["levers"]:
        if lev["id"] == lid:
            return lev
    _fail(f"unknown lever id: {lid}")
    raise AssertionError  # unreachable


def _disabled_cmake_ids(
    cfg: dict[str, Any],
    *,
    mode: str,
    lever_id: str | None,
) -> list[str]:
    """Return cmake short ids to DISABLE for this cell.

    tip control: []
    tip−1 L_x: [cmake of L_x]
    master control: all cmake ids
    master+1 L_x: all except cmake of L_x
    """
    all_ids = [str(lev["cmake"]) for lev in cfg["levers"]]
    if mode == "tip_control":
        return []
    if mode == "tip_minus":
        if not lever_id:
            _fail("tip_minus requires lever_id")
        return [str(_lever_by_id(cfg, lever_id)["cmake"])]
    if mode == "master_control":
        return list(all_ids)
    if mode == "master_plus":
        if not lever_id:
            _fail("master_plus requires lever_id")
        keep = str(_lever_by_id(cfg, lever_id)["cmake"])
        return [x for x in all_ids if x != keep]
    _fail(f"unknown mode: {mode}")
    raise AssertionError


def _cell_key(mode: str, lever_id: str | None, suite_id: str) -> str:
    lev = lever_id or "CONTROL"
    return f"{mode}/{lev}/{suite_id}"


def _build_dir_for(ablation_root: Path, mode: str, lever_id: str | None) -> Path:
    lev = lever_id or "CONTROL"
    return ablation_root / "builds" / mode / lev


def _configure_and_build(
    *,
    repo: Path,
    build_dir: Path,
    disabled: list[str],
    cmake_base: list[str],
    targets: list[str],
    dry_run: bool,
) -> int:
    build_dir.mkdir(parents=True, exist_ok=True)
    disable_csv = ";".join(disabled)
    cmake_cmd = [
        "cmake",
        "-S",
        str(repo),
        "-B",
        str(build_dir),
        *cmake_base,
        f"-DTASK066_ABLATION_DISABLE={disable_csv}",
    ]
    # Clear prior FORCE-cached DISABLE_* so a tip_control rebuild is clean.
    for short in (
        "FA_TILE",
        "NO_MV_EXT",
        "FLOAT_TILE",
        "MM_MIN",
        "FUSION_CB",
        "MOE_VIEW_ADD",
        "SIZE_SWA_UB",
        "ISWA_PADS",
        "SHARED_KV",
        "SWA_HWM_LOG",
    ):
        if short not in disabled:
            cmake_cmd.append(f"-DTASK066_DISABLE_{short}=OFF")
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
    if targets:
        for t in targets:
            build_cmd.extend(["--target", t])
    print(f"CONFIGURE disabled=[{disable_csv}] build_dir={build_dir}")
    if dry_run:
        print("DRY_RUN:", " ".join(cmake_cmd))
        print("DRY_RUN:", " ".join(build_cmd))
        return 0
    r1 = subprocess.run(cmake_cmd, cwd=repo)
    if r1.returncode != 0:
        return r1.returncode
    r2 = subprocess.run(build_cmd, cwd=repo)
    return r2.returncode


def _classify_log(text: str, rc: int) -> tuple[str, str | None]:
    if rc == 77:
        return "SKIP", None
    mad_m = _RE_MAD.search(text)
    mad = mad_m.group(1) if mad_m else None
    if rc != 0:
        return "RED", mad
    if _RE_RESULT_FAIL.search(text):
        return "RED", mad
    if _RE_RESULT_PASS.search(text) or rc == 0:
        return "GREEN", mad
    return "GREEN", mad


def _run_suite_row(
    *,
    repo: Path,
    build_dir: Path,
    row: dict[str, Any],
    cell_out: Path,
    dry_run: bool,
) -> dict[str, Any]:
    sid = str(row["id"])
    kind = str(row["kind"])
    cell_out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BUILD_DIR"] = str(build_dir)
    env["REBUILD"] = "0"
    log_path = cell_out / "run.log"
    cmd: list[str]
    if kind == "suite":
        suite = str(row.get("suite") or "").strip()
        if not suite:
            _fail(f"{sid}: suite name required")
        env["SUITE"] = suite
        cmd = ["bash", str(_CORNERS), str(cell_out / "corners")]
    elif kind == "bin":
        binary = str(row.get("bin") or "").strip()
        if not binary:
            _fail(f"{sid}: bin name required")
        bin_path = build_dir / "bin" / binary
        cmd = [str(bin_path)]
    elif kind == "ubatch":
        model = env.get("TASK066_MODEL", "").strip()
        ctx = env.get("TASK066_CTX", "").strip()
        if not model:
            _fail("TASK066_MODEL required for ubatch rows (refusing silent model)")
        if not ctx:
            _fail("TASK066_CTX required for ubatch rows (refusing silent ctx)")
        diag = str(row.get("diag") or "").strip()
        if diag:
            env["TASK066_DIAG"] = diag
        else:
            env.pop("TASK066_DIAG", None)
        cmd = ["bash", str(_UBATCH), str(cell_out / "ubatch")]
    else:
        _fail(f"{sid}: bad kind {kind}")
        raise AssertionError

    print(f"RUN {sid}: {' '.join(cmd)}")
    if dry_run:
        return {
            "suite_id": sid,
            "verdict": "DRY",
            "exit": None,
            "mad": None,
            "cmd": cmd,
        }

    proc = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    log_path.write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    verdict, mad = _classify_log(proc.stdout + "\n" + proc.stderr, proc.returncode)
    return {
        "suite_id": sid,
        "verdict": verdict,
        "exit": proc.returncode,
        "mad": mad,
        "cmd": cmd,
        "log": str(log_path),
    }


def _write_tables(
    stamp: Path,
    results: list[dict[str, Any]],
    levers: list[str],
    suites: list[str],
) -> None:
    # Mode A: tip_control + tip_minus per lever
    # Mode B: master_control + master_plus per lever
    def lookup(mode: str, lev: str | None, suite: str) -> str:
        for r in results:
            if r.get("mode") == mode and r.get("lever") == lev and r.get("suite_id") == suite:
                cell = r.get("verdict", "?")
                mad = r.get("mad")
                if mad:
                    return f"{cell} mad={mad}"
                return str(cell)
        return ""

    lines: list[str] = ["# TASK-066.18 ablation tables", ""]
    lines.append("## Table A — tip−1 (columns = CONTROL tip + lever OFF)")
    lines.append("")
    cols_a = ["suite", "tip"] + levers
    lines.append("| " + " | ".join(cols_a) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols_a)) + " |")
    for s in suites:
        row = [s, lookup("tip_control", None, s)]
        for lev in levers:
            row.append(lookup("tip_minus", lev, s))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Table B — master+1 (columns = CONTROL master + lever ON alone)")
    lines.append("")
    cols_b = ["suite", "master"] + levers
    lines.append("| " + " | ".join(cols_b) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols_b)) + " |")
    for s in suites:
        row = [s, lookup("master_control", None, s)]
        for lev in levers:
            row.append(lookup("master_plus", lev, s))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    (stamp / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (stamp / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="TASK-066.18 src lever ablation")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CFG)
    ap.add_argument("--stamp-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--mode",
        choices=["A", "B", "both"],
        default="both",
        help="A=tip−1, B=master+1, both=default",
    )
    ap.add_argument("--from-lever", default="", help="resume: first lever id")
    ap.add_argument("--from-row", default="", help="resume: first suite id")
    ap.add_argument(
        "--levers",
        default="",
        help="comma-separated lever ids (default: all from config)",
    )
    ap.add_argument(
        "--rows",
        default="",
        help="comma-separated suite ids (default: config minus skip_suite_ids)",
    )
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse existing ablation build dirs (no cmake)",
    )
    ap.add_argument(
        "--controls-only",
        action="store_true",
        help="run tip_control + master_control only",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    stamp = args.stamp_dir or (_REPO_ROOT / "logs" / f"task066-18-ablation-{_utc_stamp()}")
    stamp = stamp.resolve()
    stamp.mkdir(parents=True, exist_ok=True)
    (stamp / "README.md").write_text(
        "\n".join(
            [
                "# TASK-066.18 ablation stamp",
                "",
                f"- stamp: `{stamp.name}`",
                f"- mode arg: `{args.mode}`",
                f"- dry_run: `{args.dry_run}`",
                f"- config: `{args.config}`",
                "",
                "See `tables.md` and `results.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    all_levers = [str(x["id"]) for x in cfg["levers"]]
    if args.levers.strip():
        want = [x.strip() for x in args.levers.split(",") if x.strip()]
        for w in want:
            _lever_by_id(cfg, w)
        levers = want
    else:
        levers = all_levers

    skip = set(cfg.get("default_matrix", {}).get("skip_suite_ids") or [])
    all_rows = [str(x["id"]) for x in cfg["suites"] if str(x["id"]) not in skip]
    if args.rows.strip():
        want_r = [x.strip() for x in args.rows.split(",") if x.strip()]
        known = {str(x["id"]) for x in cfg["suites"]}
        for w in want_r:
            if w not in known:
                _fail(f"unknown suite id: {w}")
        rows = want_r
    else:
        rows = all_rows
    row_objs = {str(x["id"]): x for x in cfg["suites"]}

    # Resume gates
    skip_until_lever = bool(args.from_lever.strip())
    skip_until_row = bool(args.from_row.strip())
    from_lever = args.from_lever.strip()
    from_row = args.from_row.strip()
    if from_lever and from_lever not in all_levers:
        _fail(f"--from-lever unknown: {from_lever}")
    if from_row and from_row not in {str(x["id"]) for x in cfg["suites"]}:
        _fail(f"--from-row unknown: {from_row}")

    cells: list[tuple[str, str | None]] = []
    run_a = args.mode in ("A", "both")
    run_b = args.mode in ("B", "both")
    if run_a:
        cells.append(("tip_control", None))
        if not args.controls_only:
            for lid in levers:
                cells.append(("tip_minus", lid))
    if run_b:
        cells.append(("master_control", None))
        if not args.controls_only:
            for lid in levers:
                cells.append(("master_plus", lid))

    results: list[dict[str, Any]] = []
    cmake_base = [str(x) for x in cfg.get("cmake_base") or []]
    targets = [str(x) for x in cfg.get("build_targets") or []]

    print(f"STAMP={stamp}")
    print(f"MATRIX levers={levers}")
    print(f"MATRIX rows={rows}")
    if args.dry_run:
        print("DRY_RUN matrix cells:")
        for mode, lid in cells:
            disabled = _disabled_cmake_ids(cfg, mode=mode, lever_id=lid)
            print(f"  {mode} lever={lid or 'CONTROL'} disable={disabled}")
            for sid in rows:
                print(f"    row={sid} key={_cell_key(mode, lid, sid)}")

    for mode, lid in cells:
        if skip_until_lever and lid is not None:
            if lid != from_lever:
                continue
            skip_until_lever = False
        elif skip_until_lever and lid is None:
            # controls always allowed before from-lever
            pass

        disabled = _disabled_cmake_ids(cfg, mode=mode, lever_id=lid)
        bdir = _build_dir_for(stamp, mode, lid)
        if not args.skip_build:
            brc = _configure_and_build(
                repo=_REPO_ROOT,
                build_dir=bdir,
                disabled=disabled,
                cmake_base=cmake_base,
                targets=targets,
                dry_run=args.dry_run,
            )
            if brc != 0 and not args.dry_run:
                _fail(f"build failed mode={mode} lever={lid} rc={brc}")

        row_skip = skip_until_row
        for sid in rows:
            if row_skip:
                if sid != from_row:
                    continue
                row_skip = False
                skip_until_row = False
            cell_out = stamp / "cells" / mode / (lid or "CONTROL") / sid
            meta = _run_suite_row(
                repo=_REPO_ROOT,
                build_dir=bdir if not args.dry_run else bdir,
                row=row_objs[sid],
                cell_out=cell_out,
                dry_run=args.dry_run,
            )
            meta["mode"] = mode
            meta["lever"] = lid
            meta["disabled_cmake"] = disabled
            meta["build_dir"] = str(bdir)
            results.append(meta)
            print(
                f"CELL {_cell_key(mode, lid, sid)} -> {meta['verdict']}"
                + (f" mad={meta['mad']}" if meta.get("mad") else "")
            )

    _write_tables(stamp, results, levers, rows)
    print(f"WROTE {stamp / 'tables.md'}")
    print(f"WROTE {stamp / 'results.json'}")


if __name__ == "__main__":
    main()
