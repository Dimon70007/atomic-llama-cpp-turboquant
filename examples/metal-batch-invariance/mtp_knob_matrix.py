#!/usr/bin/env python3
"""Metal MTP knobs quality/speed matrix orchestrator (examples only, not ctest).

Composes ``run-mtp-vs-nomtp-ladder.sh`` once for a shared nomtp baseline and once
per ``(n_max, p_min)`` MTP cell. Fail-closed on missing run ids / knobs.
Writes stamp under ``logs/metal-mtp-knob-matrix-<stamp>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

_EXAMPLES_DIR: Final[Path] = Path(__file__).resolve().parent
_REPO_ROOT: Final[Path] = _EXAMPLES_DIR.parent.parent
_DEFAULT_MATRIX: Final[Path] = _EXAMPLES_DIR / "mtp_knob_matrix.json"
_LADDER: Final[Path] = _EXAMPLES_DIR / "run-mtp-vs-nomtp-ladder.sh"

_RE_TG: Final[re.Pattern[str]] = re.compile(
    r"eval time\s*=\s*[0-9.]+\s*ms\s*/\s*[0-9]+\s*tokens.*?([0-9.]+)\s*tokens per second",
    re.IGNORECASE,
)
_RE_TG_SLOT: Final[re.Pattern[str]] = re.compile(
    r"\btg\s*=\s*([0-9.]+)\s*t/s",
    re.IGNORECASE,
)
_RE_ACCEPT: Final[re.Pattern[str]] = re.compile(
    r"draft acceptance\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)
_RE_HIT_DOLOKHOV: Final[re.Pattern[str]] = re.compile(
    r"левый бок|попадает в [Дд]олохов|ранен(?:ый|ого) [Дд]олохов",
    re.IGNORECASE,
)
_RE_MISS: Final[re.Pattern[str]] = re.compile(
    r"стреляет мимо|промахивается",
    re.IGNORECASE,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _fail(msg: str, code: int = 2) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _load_matrix(path: Path) -> dict[str, Any]:
    if not path.is_file():
        _fail(f"matrix JSON not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        _fail(f"matrix root must be object: {path}")
    baseline = data.get("baseline")
    runs = data.get("runs")
    locks = data.get("locks")
    if not isinstance(baseline, dict) or not str(baseline.get("id") or "").strip():
        _fail("matrix.baseline.id required")
    if not isinstance(runs, list) or not runs:
        _fail("matrix.runs must be a non-empty list")
    if not isinstance(locks, dict):
        _fail("matrix.locks required")
    product_keep = locks.get("product_keep")
    if not isinstance(product_keep, dict):
        _fail("matrix.locks.product_keep required")
    keep_id = str(product_keep.get("run_id") or "").strip()
    if not keep_id:
        _fail("matrix.locks.product_keep.run_id required (no silent KEEP id)")
    seen: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            _fail("each matrix.runs[] entry must be an object")
        rid = str(run.get("id") or "").strip()
        if not rid:
            _fail("matrix.runs[].id required")
        if rid in seen:
            _fail(f"duplicate run id: {rid}")
        seen.add(rid)
        if "n_max" not in run or "p_min" not in run:
            _fail(f"run {rid}: n_max and p_min required")
        try:
            n_max = int(run["n_max"])
            p_min = float(run["p_min"])
        except (TypeError, ValueError) as exc:
            _fail(f"run {rid}: invalid n_max/p_min ({exc})")
        if n_max < 1:
            _fail(f"run {rid}: n_max must be >= 1")
        if not (0.0 <= p_min <= 1.0):
            _fail(f"run {rid}: p_min must be in [0,1]")
        run["n_max"] = n_max
        run["p_min"] = p_min
    if keep_id not in seen:
        _fail(f"product_keep.run_id={keep_id!r} missing from matrix.runs")
    return data


def _duel_hits(text: str) -> int:
    hits = 0
    if _RE_HIT_DOLOKHOV.search(text):
        hits += 1
    if _RE_MISS.search(text):
        hits += 1
    return hits


def _parse_tg(log_text: str) -> float | None:
    matches = _RE_TG.findall(log_text)
    if matches:
        return float(matches[-1])
    slot = _RE_TG_SLOT.findall(log_text)
    if slot:
        return float(slot[-1])
    return None


def _parse_acceptance(log_text: str) -> float | None:
    matches = _RE_ACCEPT.findall(log_text)
    if not matches:
        return None
    return float(matches[-1])


def _first_disagree(a: Path, b: Path) -> str:
    if not a.is_file() or not b.is_file():
        return "missing"
    raw_a = a.read_bytes()
    raw_b = b.read_bytes()
    if not raw_a or not raw_b:
        return "EMPTY_CAPTURE"
    if raw_a == raw_b:
        return "identical"
    n = min(len(raw_a), len(raw_b))
    i = 0
    while i < n and raw_a[i] == raw_b[i]:
        i += 1

    def snip(buf: bytes, idx: int, width: int = 36) -> str:
        return repr(buf[idx : idx + width].decode("utf-8", "replace").replace("\n", "\\n"))

    return f"DIFF @byte {i} a={snip(raw_a, i)} b={snip(raw_b, i)}"


@dataclass(frozen=True)
class CellMetrics:
    """Parsed HARD + soft metrics for one matrix cell."""

    run_id: str
    mode: str
    n_max: int | None
    p_min: float | None
    hard_ok: bool
    duel_hits: int
    tg: float | None
    acceptance: float | None
    first_disagree: str
    ladder_dir: str
    gen_path: str
    error: str


def _cell_paths(ladder_out: Path, ubatch: int, mode: str) -> tuple[Path, Path, Path]:
    cell = ladder_out / "cells" / f"ub{ubatch:04d}-{mode}"
    return cell / "generation.txt", cell / "cli.log", cell / "mtp-evidence.txt"


def _scrape_mtp_cell(
    *,
    run_id: str,
    n_max: int,
    p_min: float,
    ladder_out: Path,
    ubatch: int,
    nomtp_gen: Path | None,
) -> CellMetrics:
    gen, log, evidence = _cell_paths(ladder_out, ubatch, "mtp")
    err = ""
    hard_ok = True
    if not gen.is_file() or gen.stat().st_size == 0:
        hard_ok = False
        err = "empty_or_missing_generation"
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    if not evidence.is_file() or evidence.stat().st_size == 0:
        # engagement may still be in cli.log
        if not re.search(
            r"draft acceptance|\[spec\]|draft-mtp|adding speculative|spec_model|n_draft",
            log_text,
        ):
            hard_ok = False
            err = err or "missing_mtp_engagement"
    if re.search(r"Insufficient Memory|Compute error\.|GGML_ASSERT", log_text):
        hard_ok = False
        err = err or "oom_or_assert"
    gen_text = gen.read_text(encoding="utf-8", errors="replace") if gen.is_file() else ""
    disagree = _first_disagree(gen, nomtp_gen) if nomtp_gen is not None else "no_baseline"
    return CellMetrics(
        run_id=run_id,
        mode="mtp",
        n_max=n_max,
        p_min=p_min,
        hard_ok=hard_ok,
        duel_hits=_duel_hits(gen_text),
        tg=_parse_tg(log_text),
        acceptance=_parse_acceptance(log_text),
        first_disagree=disagree,
        ladder_dir=str(ladder_out),
        gen_path=str(gen),
        error=err,
    )


def _scrape_nomtp_cell(*, ladder_out: Path, ubatch: int) -> CellMetrics:
    gen, log, _ev = _cell_paths(ladder_out, ubatch, "nomtp")
    err = ""
    hard_ok = True
    if not gen.is_file() or gen.stat().st_size == 0:
        hard_ok = False
        err = "empty_or_missing_generation"
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    if re.search(r"Insufficient Memory|Compute error\.|GGML_ASSERT", log_text):
        hard_ok = False
        err = err or "oom_or_assert"
    gen_text = gen.read_text(encoding="utf-8", errors="replace") if gen.is_file() else ""
    return CellMetrics(
        run_id="nomtp-baseline",
        mode="nomtp",
        n_max=None,
        p_min=None,
        hard_ok=hard_ok,
        duel_hits=_duel_hits(gen_text),
        tg=_parse_tg(log_text),
        acceptance=None,
        first_disagree="n/a",
        ladder_dir=str(ladder_out),
        gen_path=str(gen),
        error=err,
    )


def _metrics_to_dict(m: CellMetrics) -> dict[str, Any]:
    return {
        "run_id": m.run_id,
        "mode": m.mode,
        "n_max": m.n_max,
        "p_min": m.p_min,
        "hard_ok": m.hard_ok,
        "duel_hits": m.duel_hits,
        "tg": m.tg,
        "acceptance": m.acceptance,
        "first_disagree": m.first_disagree,
        "ladder_dir": m.ladder_dir,
        "gen_path": m.gen_path,
        "error": m.error,
    }


def _rank_key(m: CellMetrics) -> tuple[Any, ...]:
    # Prefer hard_ok, then duel_hits, then tg, then acceptance.
    return (
        1 if m.hard_ok else 0,
        m.duel_hits,
        m.tg if m.tg is not None else -1.0,
        m.acceptance if m.acceptance is not None else -1.0,
    )


def _run_ladder(
    *,
    out_dir: Path,
    cell_modes: str,
    n_max: int | None,
    p_min: float | None,
    ubatch: int,
    dry_run: bool,
    env_extra: dict[str, str],
) -> int:
    if not _LADDER.is_file():
        _fail(f"ladder missing: {_LADDER}")
    env = os.environ.copy()
    env.update(env_extra)
    env["TASK066_CELL_MODES"] = cell_modes
    env["TASK066_UBATCH_LIST"] = str(ubatch)
    env["TASK066_SAMPLE_MODE"] = env.get("TASK066_SAMPLE_MODE", "product")
    if n_max is not None:
        env["TASK066_SPEC_DRAFT_N_MAX"] = str(n_max)
    if p_min is not None:
        env["TASK066_SPEC_DRAFT_P_MIN"] = f"{p_min:g}"
    if dry_run:
        env["DRY_RUN"] = "1"
    # Child ladder owns exclusive-Metal preflight; allow shared only if operator set it.
    cmd = ["bash", str(_LADDER), str(out_dir)]
    print(f"==> ladder modes={cell_modes} n_max={n_max} p_min={p_min} out={out_dir}")
    proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, check=False)
    return int(proc.returncode)


def _write_reports(
    stamp_dir: Path,
    *,
    matrix: dict[str, Any],
    baseline: CellMetrics | None,
    cells: list[CellMetrics],
    keep_id: str,
) -> None:
    product_keep = matrix["locks"]["product_keep"]
    rows = sorted(cells, key=_rank_key, reverse=True)
    winner = rows[0] if rows else None
    keep_row = next((c for c in cells if c.run_id == keep_id), None)

    recommend = "INCOMPLETE"
    promote = False
    if winner is not None and winner.hard_ok:
        if winner.run_id == keep_id:
            recommend = f"KEEP product knobs ({keep_id}: n_max={product_keep['n_max']}, p_min={product_keep['p_min']})"
        else:
            promote = True
            recommend = (
                f"PROMOTE candidate {winner.run_id} "
                f"(n_max={winner.n_max}, p_min={winner.p_min}) "
                f"vs KEEP {keep_id} — YAML change requires operator ack"
            )

    matrix_md = stamp_dir / "matrix.md"
    lines: list[str] = [
        "# TASK-066.9 Phase C — MTP knobs matrix",
        "",
        f"stamp_dir: `{stamp_dir}`",
        f"ubatch: {matrix['locks'].get('ubatch')}",
        f"product_keep: {keep_id}",
        "",
        "## Results",
        "",
        "| id | n_max | p_min | hard | duel | tg | acceptance | vs nomtp |",
        "|----|------:|------:|:----:|-----:|---:|-----------:|----------|",
    ]
    if baseline is not None:
        lines.append(
            f"| {baseline.run_id} | — | — | {'OK' if baseline.hard_ok else 'FAIL'} | "
            f"{baseline.duel_hits}/2 | {baseline.tg if baseline.tg is not None else '—'} | "
            f"— | n/a |"
        )
    for c in cells:
        lines.append(
            f"| {c.run_id} | {c.n_max} | {c.p_min} | {'OK' if c.hard_ok else 'FAIL'} | "
            f"{c.duel_hits}/2 | {c.tg if c.tg is not None else '—'} | "
            f"{c.acceptance if c.acceptance is not None else '—'} | `{c.first_disagree}` |"
        )
    lines.extend(
        [
            "",
            "## Ranking (hard_ok → duel_hits → tg → acceptance)",
            "",
        ]
    )
    for i, c in enumerate(rows, start=1):
        lines.append(
            f"{i}. `{c.run_id}` hard={c.hard_ok} duel={c.duel_hits}/2 "
            f"tg={c.tg} acceptance={c.acceptance}"
        )
    matrix_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = stamp_dir / "summary.md"
    keep_cmp = "n/a"
    if keep_row is not None and winner is not None:
        keep_cmp = (
            f"KEEP `{keep_id}`: duel={keep_row.duel_hits}/2 tg={keep_row.tg} "
            f"acc={keep_row.acceptance}; winner `{winner.run_id}`: "
            f"duel={winner.duel_hits}/2 tg={winner.tg} acc={winner.acceptance}"
        )
    summary.write_text(
        "\n".join(
            [
                "# TASK-066.9 Phase C summary",
                "",
                f"- recommend: **{recommend}**",
                f"- promote_candidate: {promote}",
                f"- compare: {keep_cmp}",
                "- DoD: stamp + recommend only; do **not** auto-edit product YAML.",
                f"- matrix: `{matrix_md.name}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    payload = {
        "baseline": _metrics_to_dict(baseline) if baseline else None,
        "cells": [_metrics_to_dict(c) for c in cells],
        "winner_id": winner.run_id if winner else None,
        "recommend": recommend,
        "promote_candidate": promote,
        "product_keep_id": keep_id,
    }
    (stamp_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cell_done(cell_dir: Path) -> bool:
    return (cell_dir / "metrics.json").is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TASK-066.9 Phase C MTP knobs matrix")
    parser.add_argument(
        "--matrix",
        type=Path,
        default=_DEFAULT_MATRIX,
        help="Path to mtp_knob_matrix.json",
    )
    parser.add_argument(
        "--stamp-dir",
        type=Path,
        default=None,
        help="Existing or new stamp directory (default: logs/metal-mtp-knob-matrix-<UTC>)",
    )
    parser.add_argument(
        "--from-run",
        default="",
        help="Resume from this run id (skip prior MTP runs; baseline still required)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass DRY_RUN=1 to ladder cells",
    )
    args = parser.parse_args(argv)

    matrix = _load_matrix(args.matrix)
    ubatch = int(matrix["locks"]["ubatch"])
    keep_id = str(matrix["locks"]["product_keep"]["run_id"]).strip()
    baseline_id = str(matrix["baseline"]["id"]).strip()

    stamp_dir = args.stamp_dir
    if stamp_dir is None:
        stamp_dir = _REPO_ROOT / "logs" / f"metal-mtp-knob-matrix-{_utc_stamp()}"
    stamp_dir = stamp_dir.resolve()
    cells_root = stamp_dir / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)

    readme = stamp_dir / "README.txt"
    if not readme.is_file():
        readme.write_text(
            "\n".join(
                [
                    "TASK-066.9 Phase C — MTP knobs quality/speed matrix",
                    f"stamp={stamp_dir.name}",
                    f"matrix={args.matrix}",
                    f"ubatch={ubatch}",
                    f"product_keep={keep_id}",
                    f"BUILD_DIR={os.environ.get('BUILD_DIR', '')}",
                    f"dry_run={args.dry_run or os.environ.get('DRY_RUN', '0')}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    shutil.copy2(args.matrix, stamp_dir / "mtp_knob_matrix.json")

    dry = bool(args.dry_run or os.environ.get("DRY_RUN", "0") == "1")
    env_extra = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("TASK066_")
        or k in {"BUILD_DIR", "MAIN_GGUF", "DRAFT_GGUF", "LLAMA_CLI", "GGML_METAL_MUL_MM_MIN"}
    }
    # Orchestrator owns knobs; strip single-cell ladder overrides that would clash.
    for drop in (
        "TASK066_SPEC_DRAFT_N_MAX",
        "TASK066_SPEC_DRAFT_P_MIN",
        "TASK066_CELL_MODES",
        "TASK066_UBATCH_LIST",
    ):
        env_extra.pop(drop, None)

    # --- baseline ---
    baseline_dir = cells_root / baseline_id
    baseline_metrics_path = baseline_dir / "metrics.json"
    if baseline_metrics_path.is_file():
        print(f"==> skip baseline (resume): {baseline_id}")
        baseline = CellMetrics(**json.loads(baseline_metrics_path.read_text(encoding="utf-8")))
    else:
        ladder_out = baseline_dir / "ladder"
        rc = _run_ladder(
            out_dir=ladder_out,
            cell_modes="nomtp",
            n_max=None,
            p_min=None,
            ubatch=ubatch,
            dry_run=dry,
            env_extra=env_extra,
        )
        if rc != 0 and not dry:
            _fail(f"nomtp baseline ladder exit {rc}", code=rc)
        baseline = _scrape_nomtp_cell(ladder_out=ladder_out, ubatch=ubatch)
        if dry and not baseline.hard_ok:
            # dry stub generation is non-empty from ladder; tolerate scrape gaps
            baseline = CellMetrics(
                run_id=baseline.run_id,
                mode=baseline.mode,
                n_max=None,
                p_min=None,
                hard_ok=True,
                duel_hits=baseline.duel_hits,
                tg=baseline.tg,
                acceptance=None,
                first_disagree="n/a",
                ladder_dir=baseline.ladder_dir,
                gen_path=baseline.gen_path,
                error="dry_run",
            )
        baseline_dir.mkdir(parents=True, exist_ok=True)
        baseline_metrics_path.write_text(
            json.dumps(_metrics_to_dict(baseline), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not baseline.hard_ok:
            _fail(f"nomtp baseline HARD fail: {baseline.error}")

    nomtp_gen = Path(baseline.gen_path)

    # --- MTP runs ---
    from_run = str(args.from_run or "").strip()
    skip_until = bool(from_run)
    results: list[CellMetrics] = []
    for run in matrix["runs"]:
        rid = str(run["id"])
        cell_dir = cells_root / rid
        if skip_until:
            if rid != from_run:
                if _cell_done(cell_dir):
                    results.append(
                        CellMetrics(**json.loads((cell_dir / "metrics.json").read_text(encoding="utf-8")))
                    )
                    print(f"==> skip (before --from-run): {rid}")
                    continue
                _fail(
                    f"--from-run={from_run}: prior run {rid} missing metrics.json "
                    "(cannot skip incomplete prefix)"
                )
            skip_until = False

        if _cell_done(cell_dir) and not from_run:
            print(f"==> skip complete: {rid}")
            results.append(
                CellMetrics(**json.loads((cell_dir / "metrics.json").read_text(encoding="utf-8")))
            )
            continue
        if _cell_done(cell_dir) and from_run and rid != from_run:
            results.append(
                CellMetrics(**json.loads((cell_dir / "metrics.json").read_text(encoding="utf-8")))
            )
            continue

        n_max = int(run["n_max"])
        p_min = float(run["p_min"])
        ladder_out = cell_dir / "ladder"
        if ladder_out.exists() and from_run == rid:
            shutil.rmtree(ladder_out)
        rc = _run_ladder(
            out_dir=ladder_out,
            cell_modes="mtp",
            n_max=n_max,
            p_min=p_min,
            ubatch=ubatch,
            dry_run=dry,
            env_extra=env_extra,
        )
        if rc != 0 and not dry:
            # Still scrape + record for resume visibility, then fail closed.
            metrics = _scrape_mtp_cell(
                run_id=rid,
                n_max=n_max,
                p_min=p_min,
                ladder_out=ladder_out,
                ubatch=ubatch,
                nomtp_gen=nomtp_gen,
            )
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "metrics.json").write_text(
                json.dumps(_metrics_to_dict(metrics), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            results.append(metrics)
            _write_reports(
                stamp_dir,
                matrix=matrix,
                baseline=baseline,
                cells=results,
                keep_id=keep_id,
            )
            _fail(f"MTP cell {rid} ladder exit {rc}", code=rc)

        metrics = _scrape_mtp_cell(
            run_id=rid,
            n_max=n_max,
            p_min=p_min,
            ladder_out=ladder_out,
            ubatch=ubatch,
            nomtp_gen=nomtp_gen,
        )
        if dry and not metrics.hard_ok:
            metrics = CellMetrics(
                run_id=rid,
                mode="mtp",
                n_max=n_max,
                p_min=p_min,
                hard_ok=True,
                duel_hits=metrics.duel_hits,
                tg=metrics.tg,
                acceptance=metrics.acceptance,
                first_disagree=metrics.first_disagree,
                ladder_dir=metrics.ladder_dir,
                gen_path=metrics.gen_path,
                error="dry_run",
            )
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "metrics.json").write_text(
            json.dumps(_metrics_to_dict(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(metrics)
        if not metrics.hard_ok and not dry:
            _write_reports(
                stamp_dir,
                matrix=matrix,
                baseline=baseline,
                cells=results,
                keep_id=keep_id,
            )
            _fail(f"MTP cell {rid} HARD fail: {metrics.error}")

    _write_reports(
        stamp_dir,
        matrix=matrix,
        baseline=baseline,
        cells=results,
        keep_id=keep_id,
    )
    print(f"OK: Phase C stamp {stamp_dir}")
    print(f"    matrix: {stamp_dir / 'matrix.md'}")
    print(f"    summary: {stamp_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
