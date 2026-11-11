#!/usr/bin/env python3
"""TASK-066.19 — cumulative KEEP-stack decode speed gates.

Builds ablation cells C0..C4 (then optional C_final_tip), measures nomtp/MTP
tg from cell BUILD_DIR bins, hard-stops on floor miss with verdict=NEGATIVE.
Dry-run never clobbers a real stamp results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

_TESTS_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _TESTS_DIR.parent
_DEFAULT_CFG: Final[Path] = _TESTS_DIR / "task066_keep_stack_speed.json"
_PROBE_SH: Final[Path] = _TESTS_DIR / "run-task066-decode-speed-probe.sh"

_ALL_CMAKE_SHORT: Final[tuple[str, ...]] = (
    "FA_TILE",
    "NO_MV_EXT",
    "FLOAT_TILE",
    "MM_MIN",
    "FUSION_CB",
        "SIZE_SWA_UB",
    "ISWA_PADS",
    "SHARED_KV",
    "SWA_HWM_LOG",
)

_RE_TG: Final[re.Pattern[str]] = re.compile(
    r"(?:tg|decode)[^\d]*([0-9]+(?:\.[0-9]+)?)\s*(?:t/?s|tok(?:ens)?/s)",
    re.IGNORECASE,
)
_RE_BENCH_JSON_TG: Final[re.Pattern[str]] = re.compile(
    r'"avg_ts"\s*:\s*([0-9.eE+-]+)|"decode_tok_per_sec"\s*:\s*([0-9.eE+-]+)'
)
_RE_CLI_SPEED: Final[re.Pattern[str]] = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:tokens per second|t/s|tok/s)",
    re.IGNORECASE,
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
    cells = data.get("cells")
    floors = data.get("floors")
    levers = data.get("levers")
    aliases = data.get("aliases")
    if not isinstance(cells, list) or not cells:
        _fail("config.cells must be a non-empty list")
    if not isinstance(floors, dict):
        _fail("config.floors must be an object")
    if not isinstance(levers, list) or not levers:
        _fail("config.levers must be a non-empty list")
    if not isinstance(aliases, dict):
        _fail("config.aliases must be an object")
    for key in ("nomtp", "mtp"):
        if not str(aliases.get(key) or "").strip():
            _fail(f"config.aliases.{key} required")
    for key in ("nomtp_tg_min", "mtp_tg_min", "median_runs"):
        if key not in floors:
            _fail(f"config.floors.{key} required")
    seen: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            _fail("each cells[] entry must be an object")
        cid = str(cell.get("id") or "").strip()
        if not cid:
            _fail("cells[].id required")
        if cid in seen:
            _fail(f"duplicate cell id: {cid}")
        seen.add(cid)
    return data


def _lever_cmake_map(cfg: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for lev in cfg["levers"]:
        lid = str(lev.get("id") or "").strip()
        cmake = str(lev.get("cmake") or "").strip()
        if not lid or not cmake:
            _fail("levers[].id and levers[].cmake required")
        out[lid] = cmake
    return out


def _disabled_for_cell(cfg: dict[str, Any], cell: dict[str, Any]) -> list[str]:
    """Return cmake short ids to DISABLE (master-like) for this cell."""
    cmake_map = _lever_cmake_map(cfg)
    all_ids = list(cmake_map.values())
    enabled = cell.get("enabled_levers")
    if enabled == "tip_all_on":
        return []
    if not isinstance(enabled, list):
        _fail(f"cell {cell.get('id')}: enabled_levers must be list or tip_all_on")
    keep: set[str] = set()
    for lid in enabled:
        lid_s = str(lid).strip()
        if lid_s not in cmake_map:
            _fail(f"cell {cell.get('id')}: unknown lever {lid_s}")
        keep.add(cmake_map[lid_s])
    return [x for x in all_ids if x not in keep]


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
    for short in _ALL_CMAKE_SHORT:
        if short not in disabled:
            cmake_cmd.append(f"-DTASK066_DISABLE_{short}=OFF")
    build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
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


def _median(values: list[float]) -> float:
    if not values:
        _fail("median() called with empty list")
    return float(statistics.median(values))


def _parse_tg_samples(text: str) -> list[float]:
    found: list[float] = []
    for m in _RE_BENCH_JSON_TG.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw:
            found.append(float(raw))
    if found:
        return found
    for m in _RE_TG.finditer(text):
        found.append(float(m.group(1)))
    for m in _RE_CLI_SPEED.finditer(text):
        found.append(float(m.group(1)))
    return found


def _run_fork_probe(
    *,
    mode: str,
    build_dir: Path,
    model: Path,
    draft: Path | None,
    out_dir: Path,
    repeats: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Invoke fork ``run-task066-decode-speed-probe.sh`` (cwd = fork root)."""
    if not _PROBE_SH.is_file():
        _fail(f"missing fork probe script: {_PROBE_SH}")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BUILD_DIR"] = str(build_dir)
    env["TASK066_MODEL"] = str(model)
    env["MODE"] = mode
    env["OUT_DIR"] = str(out_dir)
    env["REPEATS"] = str(repeats)
    if draft is not None:
        env["TASK066_MODEL_DRAFT"] = str(draft)
    elif "TASK066_MODEL_DRAFT" in env and mode == "nomtp":
        # Avoid leaking draft into nomtp probe.
        env.pop("TASK066_MODEL_DRAFT", None)
    cmd = ["bash", str(_PROBE_SH)]
    print("+", f"MODE={mode}", f"BUILD_DIR={build_dir}", " ".join(cmd))
    if dry_run:
        return {
            "workload": mode,
            "rc": 0,
            "samples": [],
            "tg_median": None,
            "cmd": cmd,
            "probe": str(_PROBE_SH),
            "out_dir": str(out_dir),
        }
    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (out_dir / "probe-stdout.log").write_text(text, encoding="utf-8")
    summary_path = out_dir / "summary.json"
    tg_median: float | None = None
    samples: list[float] = []
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw_med = payload.get("tg_median")
                if isinstance(raw_med, (int, float)):
                    tg_median = float(raw_med)
                raw_s = payload.get("samples")
                if isinstance(raw_s, list):
                    samples = [float(x) for x in raw_s if isinstance(x, (int, float))]
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            # Fall back to log scrape if summary missing/corrupt.
            scraped = _parse_tg_samples(text)
            samples = scraped
            tg_median = _median(scraped) if scraped else None
    elif text.strip():
        scraped = _parse_tg_samples(text)
        samples = scraped
        tg_median = _median(scraped) if scraped else None
    return {
        "workload": mode,
        "rc": proc.returncode,
        "samples": samples,
        "tg_median": tg_median,
        "cmd": cmd,
        "probe": str(_PROBE_SH),
        "out_dir": str(out_dir),
    }


def _bench_nomtp(
    *,
    build_dir: Path,
    model: Path,
    cfg: dict[str, Any],
    cell_out: Path,
    dry_run: bool,
) -> dict[str, Any]:
    repeats = int(cfg["floors"]["median_runs"])
    return _run_fork_probe(
        mode="nomtp",
        build_dir=build_dir,
        model=model,
        draft=None,
        out_dir=cell_out / "nomtp",
        repeats=repeats,
        dry_run=dry_run,
    )


def _bench_mtp(
    *,
    build_dir: Path,
    model: Path,
    draft: Path,
    cfg: dict[str, Any],
    cell_out: Path,
    dry_run: bool,
) -> dict[str, Any]:
    repeats = int(
        (cfg.get("bench_mtp") or {}).get("repeats") or cfg["floors"]["median_runs"]
    )
    return _run_fork_probe(
        mode="mtp",
        build_dir=build_dir,
        model=model,
        draft=draft,
        out_dir=cell_out / "mtp",
        repeats=repeats,
        dry_run=dry_run,
    )


def _verdict_for_cell(
    *,
    nomtp_tg: float | None,
    mtp_tg: float | None,
    floors: dict[str, Any],
) -> str:
    n_floor = float(floors["nomtp_tg_min"])
    m_floor = float(floors["mtp_tg_min"])
    if nomtp_tg is None or mtp_tg is None:
        return "NEGATIVE"
    if nomtp_tg < n_floor or mtp_tg < m_floor:
        return "NEGATIVE"
    return "GREEN"


def _write_report(
    stamp: Path,
    *,
    cfg: dict[str, Any],
    results: list[dict[str, Any]],
    head_sha: str,
    hard_stopped: bool,
    negative_cell: str | None,
) -> None:
    floors = cfg["floors"]
    lines = [
        "# TASK-066.19 KEEP-stack speed REPORT",
        "",
        f"- HEAD: `{head_sha}`",
        f"- hard_stopped: `{hard_stopped}`",
        f"- first_NEGATIVE: `{negative_cell or 'none'}`",
        f"- floors: nomtp≥{floors['nomtp_tg_min']} MTP≥{floors['mtp_tg_min']}",
        f"- tg_source: `{cfg.get('tg_source')}` — {cfg.get('tg_source_note', '')}",
        "",
        "## Speed table",
        "",
        "| cell | nomtp tg | mtp tg | verdict | disabled_cmake |",
        "| --- | --- | --- | --- | --- |",
    ]
    keep_list: list[str] = []
    for row in results:
        n = row.get("nomtp_tg")
        m = row.get("mtp_tg")
        ver = row.get("verdict")
        ns = f"{n:.3f}" if isinstance(n, (int, float)) else "n/a"
        ms = f"{m:.3f}" if isinstance(m, (int, float)) else "n/a"
        disabled = ";".join(row.get("disabled_cmake") or [])
        lines.append(
            f"| {row.get('cell_id')} | {ns} | {ms} | {ver} | `{disabled}` |"
        )
        if ver == "GREEN" and row.get("cell_id") not in ("C_final_tip",):
            keep_list.append(str(row.get("cell_id")))
    lines.extend(
        [
            "",
            "## Product KEEP candidates (speed floors only)",
            "",
            ", ".join(keep_list) if keep_list else "(none — stack NEGATIVE or empty)",
            "",
            "## Notes",
            "",
            "- NEGATIVE != KEEP. Hard stop: no further pillars after first floor miss.",
            "- Phase F suites only when C0–C4 all GREEN (see final/ if present).",
            "",
        ]
    )
    (stamp / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_head(repo: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode == 0:
            return (r.stdout or "").strip() or "unknown"
    except OSError:
        pass
    return "unknown"


def _select_cells(
    cfg: dict[str, Any],
    *,
    cell_arg: str,
    phase_f: bool,
) -> list[dict[str, Any]]:
    by_id = {str(c["id"]): c for c in cfg["cells"]}
    stack = ["C0_master", "C1_fa", "C2_fa_mv", "C3_fa_mv_float", "C4_keep4"]
    if cell_arg.strip() and cell_arg.strip() != "all":
        want = cell_arg.strip()
        if want not in by_id:
            _fail(f"unknown CELL={want}")
        selected = [by_id[want]]
    else:
        selected = [by_id[cid] for cid in stack if cid in by_id]
    if phase_f:
        if "C_final_tip" not in by_id:
            _fail("phase-f requested but C_final_tip missing from config.cells")
        # Only append if running full stack or explicitly final.
        if cell_arg.strip() in ("", "all", "C_final_tip"):
            if cell_arg.strip() == "C_final_tip":
                selected = [by_id["C_final_tip"]]
            elif by_id["C_final_tip"] not in selected:
                selected.append(by_id["C_final_tip"])
    else:
        selected = [c for c in selected if str(c["id"]) != "C_final_tip"]
    return selected


def main() -> None:
    ap = argparse.ArgumentParser(description="TASK-066.19 KEEP-stack speed gates")
    ap.add_argument("--config", type=Path, default=_DEFAULT_CFG)
    ap.add_argument("--stamp-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from-cell", default="", help="resume from this cell id")
    ap.add_argument(
        "--skip-build",
        action="store_true",
        help="reuse stamp/builds/<cell>/bin when present; still cmake+build if missing",
    )
    ap.add_argument(
        "--cell",
        default="all",
        help="C0_master|C1_fa|…|C4_keep4|C_final_tip|all",
    )
    ap.add_argument(
        "--phase-f",
        action="store_true",
        help="include C_final_tip speed re-check after C0-C4 (still hard-stops)",
    )
    ap.add_argument(
        "--model-gguf",
        default="",
        help="target GGUF path (or env TASK066_MODEL)",
    )
    ap.add_argument(
        "--draft-gguf",
        default="",
        help="MTP draft GGUF path (or env TASK066_MODEL_DRAFT)",
    )
    ap.add_argument(
        "--alias-nomtp",
        default="",
        help="must match config.aliases.nomtp (or env MODEL_NOMTP)",
    )
    ap.add_argument(
        "--alias-mtp",
        default="",
        help="must match config.aliases.mtp (or env MODEL_MTP)",
    )
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    want_nomtp = str(
        args.alias_nomtp or os.environ.get("MODEL_NOMTP", "")
    ).strip()
    want_mtp = str(args.alias_mtp or os.environ.get("MODEL_MTP", "")).strip()
    if not want_nomtp:
        _fail("MODEL_NOMTP / --alias-nomtp required (refusing silent alias)")
    if not want_mtp:
        _fail("MODEL_MTP / --alias-mtp required (refusing silent alias)")
    if want_nomtp != cfg["aliases"]["nomtp"]:
        _fail(
            f"alias-nomtp={want_nomtp!r} != config.aliases.nomtp="
            f"{cfg['aliases']['nomtp']!r}"
        )
    if want_mtp != cfg["aliases"]["mtp"]:
        _fail(
            f"alias-mtp={want_mtp!r} != config.aliases.mtp="
            f"{cfg['aliases']['mtp']!r}"
        )

    model_s = str(args.model_gguf or os.environ.get("TASK066_MODEL", "")).strip()
    draft_s = str(
        args.draft_gguf or os.environ.get("TASK066_MODEL_DRAFT", "")
    ).strip()
    if not model_s:
        _fail("TASK066_MODEL / --model-gguf required (refusing silent model)")
    if not draft_s:
        _fail(
            "TASK066_MODEL_DRAFT / --draft-gguf required for MTP floor "
            "(refusing silent draft)"
        )
    model = Path(model_s).expanduser().resolve()
    draft = Path(draft_s).expanduser().resolve()
    if not args.dry_run:
        if not model.is_file():
            _fail(f"model GGUF not found: {model}")
        if not draft.is_file():
            _fail(f"draft GGUF not found: {draft}")

    stamp = args.stamp_dir or (
        _REPO_ROOT / "logs" / f"task066-19-keep-stack-{_utc_stamp()}"
    )
    stamp = stamp.resolve()
    stamp.mkdir(parents=True, exist_ok=True)
    head = _git_head(_REPO_ROOT)
    (stamp / "README.md").write_text(
        "\n".join(
            [
                "# TASK-066.19 KEEP-stack speed stamp",
                "",
                f"- stamp: `{stamp.name}`",
                f"- HEAD: `{head}`",
                f"- dry_run: `{args.dry_run}`",
                f"- aliases: nomtp=`{want_nomtp}` mtp=`{want_mtp}`",
                f"- model: `{model}`",
                f"- draft: `{draft}`",
                f"- cell: `{args.cell}` phase_f=`{args.phase_f}`",
                "",
                "See `REPORT.md` and `results.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    cells = _select_cells(cfg, cell_arg=args.cell, phase_f=args.phase_f)
    from_cell = args.from_cell.strip()
    if from_cell and from_cell not in {str(c["id"]) for c in cfg["cells"]}:
        _fail(f"--from-cell unknown: {from_cell}")

    tip_reuse = Path(os.environ.get("BUILD_DIR", "")).expanduser()
    if tip_reuse and not tip_reuse.is_absolute():
        tip_reuse = (_REPO_ROOT / tip_reuse).resolve()
    elif tip_reuse:
        tip_reuse = tip_reuse.resolve()

    results: list[dict[str, Any]] = []
    prior = stamp / "results.json"
    if prior.is_file() and not args.dry_run:
        try:
            loaded = json.loads(prior.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                results = loaded
                print(f"RESUME loaded {len(results)} cells from {prior}")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"WARN: could not load prior results.json ({exc})")

    def _done(cid: str) -> bool:
        for r in results:
            if r.get("cell_id") == cid:
                ver = str(r.get("verdict") or "")
                if ver in ("", "DRY"):
                    return False
                return True
        return False

    cmake_base = [str(x) for x in cfg.get("cmake_base") or []]
    targets = [str(x) for x in cfg.get("build_targets") or []]
    floors = cfg["floors"]
    skip_until = bool(from_cell)
    hard_stopped = False
    negative_cell: str | None = None

    print(f"STAMP={stamp}")
    print(f"CELLS={[c['id'] for c in cells]}")
    if args.dry_run:
        for cell in cells:
            disabled = _disabled_for_cell(cfg, cell)
            print(f"  {cell['id']} disable={disabled}")

    for cell in cells:
        cid = str(cell["id"])
        if skip_until:
            if cid != from_cell:
                continue
            skip_until = False
        if hard_stopped:
            print(f"SKIP {cid} (hard-stop after NEGATIVE)")
            break
        if _done(cid):
            print(f"SKIP {cid} (already in results.json)")
            # If prior was NEGATIVE, still hard-stop further cells.
            for r in results:
                if r.get("cell_id") == cid and r.get("verdict") == "NEGATIVE":
                    hard_stopped = True
                    negative_cell = cid
                    break
            continue

        disabled = _disabled_for_cell(cfg, cell)
        bdir = stamp / "builds" / cid
        reuse_tip = (
            cid == "C_final_tip"
            and not disabled
            and tip_reuse.is_dir()
            and (tip_reuse / "bin").is_dir()
        )
        if reuse_tip:
            bdir = tip_reuse
            print(f"REUSE tip build_dir={bdir}")
        elif args.skip_build and (bdir / "bin").is_dir():
            # Resume UX: reuse cells that already have a build; build the rest.
            print(f"REUSE existing build_dir={bdir} (--skip-build)")
        else:
            if args.skip_build and not args.dry_run:
                print(
                    f"NOTE: --skip-build but missing bin/ for {cid}; building {bdir}"
                )
            brc = _configure_and_build(
                repo=_REPO_ROOT,
                build_dir=bdir,
                disabled=disabled,
                cmake_base=cmake_base,
                targets=targets,
                dry_run=args.dry_run,
            )
            if brc != 0 and not args.dry_run:
                _fail(f"build failed cell={cid} rc={brc}")

        cell_out = stamp / "cells" / cid
        cell_out.mkdir(parents=True, exist_ok=True)
        (cell_out / "cmake_disable.txt").write_text(
            ";".join(disabled) + "\n", encoding="utf-8"
        )

        nomtp = _bench_nomtp(
            build_dir=bdir,
            model=model,
            cfg=cfg,
            cell_out=cell_out,
            dry_run=args.dry_run,
        )
        mtp = _bench_mtp(
            build_dir=bdir,
            model=model,
            draft=draft,
            cfg=cfg,
            cell_out=cell_out,
            dry_run=args.dry_run,
        )

        if args.dry_run:
            verdict = "DRY"
            nomtp_tg: float | None = None
            mtp_tg: float | None = None
        else:
            nomtp_tg = nomtp.get("tg_median")  # type: ignore[assignment]
            mtp_tg = mtp.get("tg_median")  # type: ignore[assignment]
            if not isinstance(nomtp_tg, (int, float)):
                nomtp_tg = None
            if not isinstance(mtp_tg, (int, float)):
                mtp_tg = None
            verdict = _verdict_for_cell(
                nomtp_tg=nomtp_tg, mtp_tg=mtp_tg, floors=floors
            )

        row: dict[str, Any] = {
            "cell_id": cid,
            "enabled_levers": cell.get("enabled_levers"),
            "disabled_cmake": disabled,
            "build_dir": str(bdir),
            "nomtp_tg": nomtp_tg,
            "mtp_tg": mtp_tg,
            "nomtp": nomtp,
            "mtp": mtp,
            "verdict": verdict,
            "aliases": {"nomtp": want_nomtp, "mtp": want_mtp},
        }
        results = [r for r in results if r.get("cell_id") != cid]
        results.append(row)
        print(
            f"CELL {cid} -> {verdict}"
            f" nomtp={nomtp_tg} mtp={mtp_tg}"
        )

        if not args.dry_run:
            (stamp / "results.json").write_text(
                json.dumps(results, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            _write_report(
                stamp,
                cfg=cfg,
                results=results,
                head_sha=head,
                hard_stopped=verdict == "NEGATIVE",
                negative_cell=cid if verdict == "NEGATIVE" else negative_cell,
            )

        if verdict == "NEGATIVE":
            hard_stopped = True
            negative_cell = cid
            print(
                f"HARD_STOP NEGATIVE cell={cid} "
                f"(floors nomtp>={floors['nomtp_tg_min']} "
                f"mtp>={floors['mtp_tg_min']})"
            )
            break

    if args.dry_run:
        dry_stamp = stamp / "dry-run"
        dry_stamp.mkdir(parents=True, exist_ok=True)
        (dry_stamp / "results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_report(
            dry_stamp,
            cfg=cfg,
            results=results,
            head_sha=head,
            hard_stopped=False,
            negative_cell=None,
        )
        print(f"WROTE {dry_stamp / 'results.json'} (dry-run; real stamp untouched)")
        print(f"WROTE {dry_stamp / 'REPORT.md'} (dry-run; real stamp untouched)")
    else:
        _write_report(
            stamp,
            cfg=cfg,
            results=results,
            head_sha=head,
            hard_stopped=hard_stopped,
            negative_cell=negative_cell,
        )
        print(f"WROTE {stamp / 'results.json'}")
        print(f"WROTE {stamp / 'REPORT.md'}")

    if hard_stopped:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
