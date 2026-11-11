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

_RE_MAD: Final[re.Pattern[str]] = re.compile(
    r"\bmad[=:\s]+([0-9.eE+-]+)|max_abs_diff=([0-9.eE+-]+).*max_ulp=([0-9]+)",
    re.IGNORECASE,
)
# Do NOT match bare "RED" / "GREEN" in prose ("RED gate", "PRE-PATCH RED").
_RE_RESULT_FAIL: Final[re.Pattern[str]] = re.compile(
    r"RESULT FAIL|VERDICT=FAIL|VERDICT=RED\b|exact=0\b.*\bFAIL\b|\bfails=\d+"
)
_RE_RESULT_PASS: Final[re.Pattern[str]] = re.compile(
    r"RESULT PASS|VERDICT=PASS|VERDICT=GREEN\b|exact=1\b"
)


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
    mad: str | None = None
    mad_m = _RE_MAD.search(text)
    if mad_m:
        if mad_m.group(1):
            mad = mad_m.group(1)
        elif mad_m.group(2) and mad_m.group(3):
            mad = f"abs={mad_m.group(2)} ulp={mad_m.group(3)}"
    # Prefer explicit RESULT/VERDICT lines over raw exit code (some runners exit 0
    # after RESULT FAIL; some banner text said "RED gate" and fooled old regex).
    if _RE_RESULT_FAIL.search(text):
        return "RED", mad
    if _RE_RESULT_PASS.search(text):
        return "GREEN", mad
    if rc != 0:
        return "RED", mad
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

    # Metal / llama logs can emit non-UTF8 (e.g. 0xa0); never crash the matrix.
    proc = subprocess.run(
        cmd,
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8", errors="replace")
    verdict, mad = _classify_log(combined, proc.returncode)
    return {
        "suite_id": sid,
        "verdict": verdict,
        "exit": proc.returncode,
        "mad": mad,
        "cmd": cmd,
        "log": str(log_path),
    }


def _verdict_of(
    results: list[dict[str, Any]], mode: str, lev: str | None, suite: str
) -> str | None:
    for r in results:
        if r.get("mode") == mode and r.get("lever") == lev and r.get("suite_id") == suite:
            return str(r.get("verdict") or "")
    return None


def _cell_label(
    results: list[dict[str, Any]], mode: str, lev: str | None, suite: str
) -> str:
    for r in results:
        if r.get("mode") == mode and r.get("lever") == lev and r.get("suite_id") == suite:
            cell = r.get("verdict", "?")
            mad = r.get("mad")
            if mad:
                return f"{cell} mad={mad}"
            return str(cell)
    return ""


def _mad_of(
    results: list[dict[str, Any]], mode: str, lev: str | None, suite: str
) -> str | None:
    for r in results:
        if r.get("mode") == mode and r.get("lever") == lev and r.get("suite_id") == suite:
            mad = r.get("mad")
            return str(mad) if mad else None
    return None


def _primary_mad_float(mad: str | None) -> float | None:
    """Parse leading numeric from mad=… or abs=… (for Δ when still RED)."""
    if not mad:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)", mad)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _mad_delta_note(master_mad: str | None, cur_mad: str | None) -> str:
    """Human Δ note: better/worse/same when both cells still RED."""
    a = _primary_mad_float(master_mad)
    b = _primary_mad_float(cur_mad)
    if a is None or b is None:
        if master_mad or cur_mad:
            return f"mad {master_mad or '?'}->{cur_mad or '?'}"
        return ""
    if abs(a - b) <= max(1e-12, abs(a) * 1e-9):
        return f"mad={cur_mad} (=)"
    delta = b - a
    tag = "better" if abs(b) < abs(a) else "worse"
    return f"mad {master_mad}->{cur_mad} (Δ{delta:+.6g}, {tag})"


def _write_keep_drop(
    stamp: Path,
    results: list[dict[str, Any]],
    levers: list[str],
    suites: list[str],
) -> None:
    """Per-lever keep/drop aid: which suites fail when tip change is OFF (Mode A)."""
    lines: list[str] = [
        "# TASK-066.18 — per-lever keep/drop (Mode A tip−1)",
        "",
        "Baseline = tip all-ON. Each section = that tip change **OFF** (master-like at that site).",
        "",
        "- **harm (DROP candidate):** tip ON was RED, OFF → GREEN (change caused the fail)",
        "- **help (KEEP candidate):** tip ON was GREEN, OFF → RED (change was fixing the suite)",
        "- **still RED:** fail with or without the change (not decided by this lever alone)",
        "- **still GREEN:** pass with or without the change",
        "",
    ]
    tip = {s: _verdict_of(results, "tip_control", None, s) for s in suites}
    if not any(tip.values()):
        lines.append("_No tip_control rows yet._")
        lines.append("")
    for lev in levers:
        harm: list[str] = []
        help_: list[str] = []
        still_red: list[str] = []
        still_green: list[str] = []
        missing: list[str] = []
        for s in suites:
            base = tip.get(s)
            cur = _verdict_of(results, "tip_minus", lev, s)
            if base is None or cur is None:
                missing.append(s)
                continue
            if base == "RED" and cur == "GREEN":
                harm.append(s)
            elif base == "GREEN" and cur == "RED":
                help_.append(s)
            elif base == "RED" and cur == "RED":
                still_red.append(s)
            elif base == "GREEN" and cur == "GREEN":
                still_green.append(s)
            else:
                still_red.append(f"{s} ({base}->{cur})")
        lines.append(f"## `{lev}` OFF")
        lines.append("")
        lines.append(f"- harm (DROP?): {', '.join(harm) if harm else '—'}")
        lines.append(f"- help (KEEP?): {', '.join(help_) if help_ else '—'}")
        lines.append(f"- still RED: {', '.join(still_red) if still_red else '—'}")
        lines.append(f"- still GREEN: {', '.join(still_green) if still_green else '—'}")
        if missing:
            lines.append(f"- pending: {', '.join(missing)}")
        lines.append("")

    # Mode B brief if present
    master = {s: _verdict_of(results, "master_control", None, s) for s in suites}
    if any(master.values()):
        lines.append("# Mode B — master+1 (change ON alone vs master all-OFF)")
        lines.append("")
        lines.append(
            "- **alone helps:** master RED, master+1 GREEN "
            "(this change alone fixes the suite)"
        )
        lines.append(
            "- **still RED but improved/worsened:** both RED — cite mad Δ "
            "(Better ≠ GREEN; RED ≠ KEEP)"
        )
        lines.append(
            "- **alone harms:** master GREEN→RED"
        )
        lines.append("")
        for lev in levers:
            alone_helps: list[str] = []
            alone_harms: list[str] = []
            still_red_notes: list[str] = []
            for s in suites:
                m = master.get(s)
                cur = _verdict_of(results, "master_plus", lev, s)
                if m is None or cur is None:
                    continue
                m_mad = _mad_of(results, "master_control", None, s)
                c_mad = _mad_of(results, "master_plus", lev, s)
                if m == "RED" and cur == "GREEN":
                    note = s
                    if c_mad or m_mad:
                        note = f"{s} ({m_mad or 'RED'}->{c_mad or 'GREEN'})"
                    alone_helps.append(note)
                elif m == "GREEN" and cur == "RED":
                    alone_harms.append(s)
                elif cur == "RED":
                    delta = _mad_delta_note(m_mad, c_mad)
                    still_red_notes.append(f"{s}: {delta}" if delta else s)
            if alone_helps or alone_harms or still_red_notes:
                lines.append(f"## `{lev}` ON alone")
                lines.append("")
                lines.append(
                    f"- alone helps: {', '.join(alone_helps) if alone_helps else '—'}"
                )
                lines.append(
                    f"- alone harms: {', '.join(alone_harms) if alone_harms else '—'}"
                )
                lines.append(
                    "- still RED (+ mad Δ): "
                    + (", ".join(still_red_notes) if still_red_notes else "—")
                )
                lines.append("")

    (stamp / "keep_drop.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_tables(
    stamp: Path,
    results: list[dict[str, Any]],
    levers: list[str],
    suites: list[str],
) -> None:
    # Mode A: tip_control + tip_minus per lever
    # Mode B: master_control + master_plus per lever
    lines: list[str] = ["# TASK-066.18 ablation tables", ""]
    lines.append("## Table A — tip−1 (columns = CONTROL tip + lever OFF)")
    lines.append("")
    cols_a = ["suite", "tip"] + levers
    lines.append("| " + " | ".join(cols_a) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols_a)) + " |")
    for s in suites:
        row = [s, _cell_label(results, "tip_control", None, s)]
        for lev in levers:
            row.append(_cell_label(results, "tip_minus", lev, s))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Table B — master+1 (columns = CONTROL master + lever ON alone)")
    lines.append("")
    cols_b = ["suite", "master"] + levers
    lines.append("| " + " | ".join(cols_b) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols_b)) + " |")
    for s in suites:
        row = [s, _cell_label(results, "master_control", None, s)]
        for lev in levers:
            row.append(_cell_label(results, "master_plus", lev, s))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("See also `keep_drop.md` (per-lever FAIL lists for keep/remove).")
    lines.append("")
    (stamp / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (stamp / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_keep_drop(stamp, results, levers, suites)


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
        "--reuse-builds-from",
        type=Path,
        default=None,
        help="symlink tip_minus/master_* builds from a prior stamp (skip cmake when link ok)",
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
    prior_results = stamp / "results.json"
    if prior_results.is_file() and not args.dry_run:
        try:
            loaded = json.loads(prior_results.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                results = loaded
                print(f"RESUME loaded {len(results)} cells from {prior_results}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: could not load prior results.json ({exc}); starting empty")

    def _already_done(mode: str, lever_id: str | None, suite_id: str) -> bool:
        for r in results:
            if (
                r.get("mode") == mode
                and r.get("lever") == lever_id
                and r.get("suite_id") == suite_id
            ):
                # DRY / empty never counts as done (dry-run must not block resume).
                ver = str(r.get("verdict") or "")
                if ver in ("", "DRY"):
                    return False
                return True
        return False

    cmake_base = [str(x) for x in cfg.get("cmake_base") or []]
    targets = [str(x) for x in cfg.get("build_targets") or []]
    tip_reuse = Path(os.environ.get("BUILD_DIR", "")).expanduser()
    if tip_reuse and not tip_reuse.is_absolute():
        tip_reuse = (_REPO_ROOT / tip_reuse).resolve()
    elif tip_reuse:
        tip_reuse = tip_reuse.resolve()

    print(f"STAMP={stamp}")
    print(f"MATRIX levers={levers}")
    print(f"MATRIX rows={rows}")
    if tip_reuse.is_dir():
        print(f"TIP_REUSE_BUILD_DIR={tip_reuse} (tip_control only when disable=[])")
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
            # Resume mid-lever: skip controls when prior results already cover them.
            if any(r.get("mode") in ("tip_control", "master_control") for r in results):
                print(f"SKIP {mode}/CONTROL (resume; prior control rows present)")
                continue
            # Fresh resume from lever: still run controls first.
            pass

        disabled = _disabled_cmake_ids(cfg, mode=mode, lever_id=lid)
        bdir = _build_dir_for(stamp, mode, lid)
        reuse_tip = (
            mode == "tip_control"
            and not disabled
            and tip_reuse.is_dir()
            and (tip_reuse / "bin").is_dir()
        )
        reused_prior = False
        if (
            not reuse_tip
            and args.reuse_builds_from is not None
            and mode != "tip_control"
        ):
            prior_root = args.reuse_builds_from.expanduser().resolve()
            prior = prior_root / "builds" / mode / (lid or "CONTROL")
            if prior.is_dir() and (prior / "bin").is_dir():
                bdir.parent.mkdir(parents=True, exist_ok=True)
                if bdir.exists() or bdir.is_symlink():
                    if bdir.is_symlink() or bdir.is_file():
                        bdir.unlink()
                    else:
                        # keep existing real dir
                        pass
                if not bdir.exists():
                    bdir.symlink_to(prior, target_is_directory=True)
                reused_prior = True
                print(f"REUSE prior build {prior} -> {bdir}")
        if reuse_tip:
            bdir = tip_reuse
            print(f"REUSE tip build_dir={bdir} (no cmake)")
        elif reused_prior or args.skip_build:
            if not (bdir / "bin").is_dir() and not args.dry_run:
                _fail(f"skip/reuse build missing bin/: {bdir}")
        elif not args.skip_build:
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
            if _already_done(mode, lid, sid):
                print(f"SKIP {_cell_key(mode, lid, sid)} (already in results.json)")
                continue
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
            # Replace any prior DRY placeholder for this cell.
            results = [
                r
                for r in results
                if not (
                    r.get("mode") == mode
                    and r.get("lever") == lid
                    and r.get("suite_id") == sid
                )
            ]
            results.append(meta)
            print(
                f"CELL {_cell_key(mode, lid, sid)} -> {meta['verdict']}"
                + (f" mad={meta['mad']}" if meta.get("mad") else "")
            )
            # Incremental stamp so a long GPU sweep can be resumed / inspected.
            # Never let --dry-run clobber a real stamp's results.json.
            if not args.dry_run:
                _write_tables(stamp, results, levers, rows)

    if args.dry_run:
        dry_stamp = stamp / "dry-run"
        dry_stamp.mkdir(parents=True, exist_ok=True)
        _write_tables(dry_stamp, results, levers, rows)
        print(f"WROTE {dry_stamp / 'tables.md'} (dry-run; real stamp untouched)")
        print(f"WROTE {dry_stamp / 'results.json'} (dry-run; real stamp untouched)")
    else:
        _write_tables(stamp, results, levers, rows)
        print(f"WROTE {stamp / 'tables.md'}")
        print(f"WROTE {stamp / 'results.json'}")


if __name__ == "__main__":
    main()
