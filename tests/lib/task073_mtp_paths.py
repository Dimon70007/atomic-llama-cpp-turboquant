#!/usr/bin/env python3
"""TASK-073 ST-4: does the checkpoint-restore defect reach other MTP paths?

ST-1/ST-2 fixed the single-head own-KV path (Groot, `qwen35moe`). The driver has
two more shapes, and neither may be assumed clean:

  is_mem_shared    (gemma4-assistant drafter: shares the target KV, and with
                    same_position_draft runs every draft row at one position)
  chain_heads      (step35: one trained head per draft step, draft KV rebuilt
                    per head)

This orchestrator runs the same contract for every path: the same greedy request
N times in ONE server with checkpoints enabled must emit ONE text. It composes
the existing measurement (task072_groot_pmin_sweep) and verdict
(task073_checkpoint_repeat_verdict) instead of re-implementing either.

Usage:
    python3 tests/lib/task073_mtp_paths.py --json tests/task073-mtp-paths.json
    python3 tests/lib/task073_mtp_paths.py --json ... --dry-run
    python3 tests/lib/task073_mtp_paths.py --json ... --only gemma-shared-kv

Exit: 0 every runnable cell passed (skips are not failures), 1 a cell failed,
      2 bad input.

A cell whose model or drafter is absent is reported SKIP, never silently passed:
"the path was not tested" and "the path is clean" are different results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import task072_groot_pmin_sweep as sweep
import task073_checkpoint_repeat_verdict as verdict

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
MODELS_DIR: Final[Path] = Path.home() / "models"
DEFAULT_SERVER: Final[Path] = REPO_ROOT / "build-dev/bin/llama-server"


@dataclass(frozen=True)
class PathCell:
    """One MTP drafting path under test."""

    id: str
    note: str
    sweep_cell: str
    ctx: int
    max_tokens: int
    repeats: int
    server_args: tuple[str, ...]
    model: Path | None
    draft: Path | None
    ctk: str
    ctv: str
    skip_reason: str

    def missing(self) -> str:
        if self.skip_reason:
            return self.skip_reason
        if self.model is None:
            return "cell has no model"
        if not self.model.is_file():
            return f"model not found: {self.model}"
        if self.draft is not None and not self.draft.is_file():
            return f"drafter not found: {self.draft}"
        return ""

    def sweep_cell_def(self) -> sweep.Cell:
        for cell in sweep.DEFAULT_CELLS:
            if cell.id == self.sweep_cell:
                return cell
        raise SystemExit(
            f"unknown sweep cell {self.sweep_cell!r} in {self.id}; "
            f"known={[c.id for c in sweep.DEFAULT_CELLS]}"
        )


def _resolve(value: Any, label: str, cell_id: str) -> Path | None:
    """A bare filename resolves against ~/models; a path is used as given."""
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute() and path.parent == Path("."):
        path = MODELS_DIR / path
    if not path.is_absolute():
        raise SystemExit(f"{cell_id}: {label} must be absolute or a bare filename: {raw}")
    return path


def load_cells(json_path: Path) -> list[PathCell]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    raw_cells = data.get("cells")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise SystemExit(f"{json_path}: 'cells' must be a non-empty list")

    cells: list[PathCell] = []
    for raw in raw_cells:
        cell_id = str(raw.get("id") or "").strip()
        if not cell_id:
            raise SystemExit(f"{json_path}: a cell has no id")
        cells.append(
            PathCell(
                id=cell_id,
                note=str(raw.get("note") or ""),
                sweep_cell=str(raw.get("sweep_cell") or "").strip() or "n2-p090",
                ctx=int(raw.get("ctx", 16384)),
                max_tokens=int(raw.get("max_tokens", 4096)),
                # three repeats is the floor: a two-way cycle is invisible below it
                repeats=int(raw.get("repeats", 4)),
                server_args=tuple(str(a) for a in raw.get("server_args") or []),
                model=_resolve(raw.get("model"), "model", cell_id),
                draft=_resolve(raw.get("draft"), "draft", cell_id),
                ctk=str(raw.get("ctk") or "q8_0"),
                ctv=str(raw.get("ctv") or "turbo4.8"),
                skip_reason=str(raw.get("skip_reason") or "").strip(),
            )
        )
    return cells


def server_busy() -> bool:
    """Metal is exclusive: a second llama-server cannot start beside the product one."""
    found = shutil.which("pgrep")
    if found is None:
        return False
    res = subprocess.run([found, "-f", "llama-server"], capture_output=True, text=True)
    return res.returncode == 0


def run_cell(
    cell: PathCell,
    server: Path,
    out_root: Path,
    prompt: str,
    dry_run: bool,
) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cell_root = out_root / cell.id / stamp

    extra: list[str] = list(cell.server_args)
    if cell.draft is not None:
        # the drafter is a separate GGUF, so the MTP context is built from it and
        # is_mem_shared/same_position_draft follow the drafter's arch
        extra += ["-md", str(cell.draft)]

    row: dict[str, Any] = {
        "id": cell.id,
        "note": cell.note,
        "model": str(cell.model),
        "draft": str(cell.draft) if cell.draft else None,
        "sweep_cell": cell.sweep_cell,
        "ctx": cell.ctx,
        "repeats": cell.repeats,
        "server_args": extra,
        "out": str(cell_root),
    }

    if dry_run:
        row["status"] = "DRY-RUN"
        return row

    measured = sweep.run_cell(
        cell=cell.sweep_cell_def(),
        bin_path=server,
        model=cell.model,
        out_dir=cell_root,
        prompt=prompt,
        ctx=cell.ctx,
        ngl=999,
        batch=2048,
        ubatch=512,
        ctk=cell.ctk,
        ctv=cell.ctv,
        max_tokens=cell.max_tokens,
        load_timeout=900.0,
        repeats=cell.repeats,
        extra_server_args=extra,
    )
    cell_dir = cell_root / cell.sweep_cell
    row["measured"] = {
        k: measured.get(k)
        for k in ("decode_tok_per_s", "acceptance_log", "mean_len_log", "completion_tokens", "error")
    }

    if measured.get("error"):
        row["status"] = "ERROR"
        row["error"] = measured["error"]
        return row

    runs = verdict.emissions(cell_dir)
    if len(runs) < 3:
        row["status"] = "ERROR"
        row["error"] = f"only {len(runs)} repeat(s) recorded in {cell_dir}"
        return row

    hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest()[:10] for _, text in runs]
    row["hashes"] = hashes
    row["distinct"] = len(set(hashes))
    row["status"] = "PASS" if len(set(hashes)) == 1 else "FAIL"
    return row


def write_summary(out_root: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    lines = [
        "# TASK-073 ST-4 - MTP drafting paths vs checkpoint restore",
        "",
        f"stamp: {meta['stamp']}",
        f"server: `{meta['server']}`",
        f"contract: the same greedy request repeated N times in one server must emit one text",
        "",
        "| path | status | distinct / repeats | draft | why |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        if row["status"] == "SKIP":
            counts = "-"
        elif row["status"] == "DRY-RUN":
            counts = f"- / {row['repeats']}"
        elif "distinct" in row:
            counts = f"{row['distinct']} / {len(row['hashes'])}"
        else:
            counts = f"- / {row['repeats']}"

        why = row.get("skip_reason") or row.get("error") or row.get("note") or ""
        draft = Path(row["draft"]).name if row.get("draft") else "-"
        lines.append(f"| `{row['id']}` | {row['status']} | {counts} | {draft} | {why} |")

    failed = [r["id"] for r in rows if r["status"] == "FAIL"]
    errored = [r["id"] for r in rows if r["status"] == "ERROR"]
    skipped = [r["id"] for r in rows if r["status"] == "SKIP"]
    passed = [r["id"] for r in rows if r["status"] == "PASS"]

    lines += ["", "## Reading", ""]
    lines.append(
        "Deterministic under checkpoint restore: "
        + (", ".join(f"`{c}`" for c in passed) if passed else "none")
        + "."
    )
    if failed:
        lines.append(
            "**Still drifting: " + ", ".join(f"`{c}`" for c in failed)
            + ".** The restore drops this path's draft state as well; these cells need "
            "the same treatment as the Groot path before the defect can be called closed."
        )
    if errored:
        lines.append("Could not measure: " + ", ".join(f"`{c}`" for c in errored) + ".")
    if skipped:
        lines.append(
            "Not tested (state this, do not assume clean): "
            + ", ".join(f"`{c}`" for c in skipped) + "."
        )

    path = out_root / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="TASK-073 ST-4 MTP path matrix")
    ap.add_argument("--json", required=True, type=Path, help="matrix file")
    ap.add_argument("--server", type=Path, default=DEFAULT_SERVER)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "logs/task073-st4-paths")
    ap.add_argument("--prompt", default=sweep._DEFAULT_PROMPT)
    ap.add_argument("--only", default="", help="comma-separated cell ids")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, start no server")
    args = ap.parse_args(argv)

    if not args.json.is_file():
        print(f"FAIL: matrix not found: {args.json}", file=sys.stderr)
        return 2

    cells = load_cells(args.json)
    if args.only:
        wanted = [c.strip() for c in args.only.split(",") if c.strip()]
        known = {c.id: c for c in cells}
        missing = [c for c in wanted if c not in known]
        if missing:
            raise SystemExit(f"unknown --only cells {missing}; known={sorted(known)}")
        cells = [known[c] for c in wanted]

    if not args.dry_run:
        if not args.server.is_file():
            print(f"FAIL: server binary not found: {args.server}", file=sys.stderr)
            return 2
        if server_busy():
            print(
                "FAIL: a llama-server is already running; Metal is exclusive.\n"
                "      stop the product service first: make minillm-stop",
                file=sys.stderr,
            )
            return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.out_dir / stamp
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {"stamp": stamp, "server": str(args.server), "prompt_chars": len(args.prompt)}
    (out_root / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    for i, cell in enumerate(cells, start=1):
        why = cell.missing()
        print(f"[{i}/{len(cells)}] {cell.id}", flush=True)
        if why:
            print(f"  SKIP: {why}", flush=True)
            rows.append({"id": cell.id, "status": "SKIP", "note": cell.note,
                         "skip_reason": why, "draft": str(cell.draft) if cell.draft else None,
                         "repeats": cell.repeats})
            continue

        row = run_cell(cell, args.server, out_root, args.prompt, args.dry_run)
        rows.append(row)
        (out_root / "rows.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        if row["status"] == "DRY-RUN":
            print("  DRY-RUN:", " ".join(row["server_args"]), flush=True)
        elif row["status"] == "ERROR":
            print(f"  ERROR: {row['error']}", file=sys.stderr, flush=True)
        else:
            print(f"  {row['status']}: distinct {row['distinct']} / {len(row['hashes'])}",
                  flush=True)

    summary = write_summary(out_root, rows, meta)
    print(f"==> {summary}")

    if any(r["status"] in ("FAIL", "ERROR") for r in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
