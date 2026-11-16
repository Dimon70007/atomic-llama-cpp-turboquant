#!/usr/bin/env python3
"""TASK-073.1: which prompt lengths still drift under checkpoint restore?

The short-prompt drift is real (TASK-073.1 ST-1) but the mechanism is not the one
the task file assumed: a 60-token prompt cannot overflow a 1024-token SWA ring.
This ladder finds the smallest prompt that still drifts, which bounds the search.

A cell is only evidence if the checkpoint path actually ran. Each cell reports the
number of checkpoints created and restored, read from the server log, so
"one text" means "reproducible", not "the restore never happened".

Usage:
    python3 tests/lib/task0731_prompt_ladder.py --model ~/models/gemma.gguf \\
        --bin build-dev/bin/llama-server --out-dir logs/task073.1-ladder

Exit: 0 every exercised cell emitted one text, 1 a cell drifted, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import task072_groot_pmin_sweep as sweep
import task073_checkpoint_repeat_verdict as verdict

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

_SOURCE: Final[str] = (
    "The train left the station at dawn and crossed the open plain before the rain began, "
    "carrying letters, tools and a tired crew toward the northern harbour where a small "
    "steamer waited.")

DEFAULT_WORDS: Final[tuple[int, ...]] = (24, 18, 14, 10, 6, 3)


def build_prompt(words: int) -> str:
    """A prompt of roughly `words` words; the chat template adds the rest."""
    vocab = _SOURCE.split()
    return " ".join(vocab[i % len(vocab)] for i in range(words)) + " Repeat this back."


def counts(log_path: Path) -> dict[str, Any]:
    """Checkpoint activity and prompt size, read back from the server log."""
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    sizes = [int(m) for m in re.findall(r"task\.n_tokens = (\d+)", text)]
    return {
        "created": text.count("created context checkpoint"),
        "restored": text.count("restored context checkpoint"),
        "forcing_full": text.count("forcing full prompt re-processing"),
        "prompt_tokens": max(sizes) if sizes else None,
    }


def run_cell(
    words: int,
    model: Path,
    server: Path,
    out_dir: Path,
    max_tokens: int,
    repeats: int,
    ctk: str,
    ctv: str,
) -> dict[str, Any]:
    cell = sweep.Cell("nomtp", None, None)
    cell_dir = out_dir / cell.id
    row: dict[str, Any] = {"words": words, "prompt": build_prompt(words)}

    measured = sweep.run_cell(
        cell=cell,
        bin_path=server,
        model=model,
        out_dir=out_dir,
        prompt=row["prompt"],
        ctx=16384,
        ngl=999,
        batch=2048,
        ubatch=512,
        ctk=ctk,
        ctv=ctv,
        max_tokens=max_tokens,
        load_timeout=900.0,
        repeats=repeats,
        extra_server_args=("-lv", "4"),
    )
    if measured.get("error"):
        row["status"] = "ERROR"
        row["error"] = measured["error"]
        return row

    info = counts(cell_dir / "server.log")
    row.update(info)

    runs = verdict.emissions(cell_dir)
    if len(runs) < repeats:
        row["status"] = "ERROR"
        row["error"] = f"only {len(runs)} of {repeats} repeat(s) recorded"
        return row

    hashes = [hashlib.sha256(t.encode("utf-8")).hexdigest()[:10] for _, t in runs]
    row["hashes"] = hashes
    row["distinct"] = len(set(hashes))

    # a cell that never restored a checkpoint did not exercise the defect
    if info["restored"] == 0:
        row["status"] = "NOT-EXERCISED"
    else:
        row["status"] = "PASS" if row["distinct"] == 1 else "FAIL"
    return row


def write_summary(out_dir: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    lines = [
        "# TASK-073.1 - shortest prompt that still drifts under checkpoint restore",
        "",
        f"stamp: {meta['stamp']}",
        f"model: `{Path(meta['model']).name}`, ctx {meta['ctx']}, "
        f"KV {meta['ctk']}/{meta['ctv']}, greedy, max_tokens {meta['max_tokens']}, "
        f"{meta['repeats']} repeats",
        "",
        "| words | prompt tokens | created | restored | distinct | status |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            "| {w} | {t} | {c} | {rs} | {d} | {s} |".format(
                w=r["words"],
                t=r.get("prompt_tokens") or "-",
                c=r.get("created", "-"),
                rs=r.get("restored", "-"),
                d=r.get("distinct", "-") if r["status"] != "ERROR" else "-",
                s=r["status"],
            )
        )

    exercised = [r for r in rows if r["status"] in ("PASS", "FAIL")]
    drifted = [r for r in exercised if r["status"] == "FAIL"]
    short = [r["words"] for r in exercised if r["status"] == "PASS"]

    lines += ["", "## Reading", ""]
    if drifted:
        lines.append(
            "Drifting: "
            + ", ".join(f"{r['words']} words ({r.get('prompt_tokens')} tokens)" for r in drifted)
            + "."
        )
    if short:
        lines.append(
            "Reproducible with the restore exercised: "
            + ", ".join(f"{w} words" for w in short)
            + "."
        )
    if any(r["status"] == "NOT-EXERCISED" for r in rows):
        lines.append(
            "Not tested (no checkpoint was restored, so one text proves nothing): "
            + ", ".join(str(r["words"]) for r in rows if r["status"] == "NOT-EXERCISED")
            + "."
        )
    if any(r["status"] == "ERROR" for r in rows):
        lines.append(
            "Could not measure: "
            + ", ".join(str(r["words"]) for r in rows if r["status"] == "ERROR")
            + "."
        )

    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="TASK-073.1 prompt-length ladder")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--bin", required=True, type=Path, dest="bin_path")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--words", default=",".join(str(w) for w in DEFAULT_WORDS))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--ctk", default="q8_0")
    ap.add_argument("--ctv", default="q8_0")
    args = ap.parse_args(argv)

    for path, name in ((args.model, "--model"), (args.bin_path, "--bin")):
        if not path.is_file():
            print(f"FAIL: {name} not found: {path}", file=sys.stderr)
            return 2

    try:
        sizes = [int(w) for w in args.words.split(",") if w.strip()]
    except ValueError:
        print(f"FAIL: --words must be integers: {args.words}", file=sys.stderr)
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = args.out_dir / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    meta = {
        "stamp": stamp,
        "model": str(args.model),
        "ctx": 16384,
        "ctk": args.ctk,
        "ctv": args.ctv,
        "max_tokens": args.max_tokens,
        "repeats": args.repeats,
    }
    (out_root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for i, words in enumerate(sizes, start=1):
        print(f"[{i}/{len(sizes)}] {words} words", flush=True)
        row = run_cell(words, args.model, args.bin_path, out_root / f"w{words:02d}",
                       args.max_tokens, args.repeats, args.ctk, args.ctv)
        rows.append(row)
        (out_root / "rows.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(
            f"  {row['status']}: tokens {row.get('prompt_tokens')}, "
            f"created {row.get('created')}, restored {row.get('restored')}, "
            f"distinct {row.get('distinct')}",
            flush=True,
        )

    print(f"==> {write_summary(out_root, rows, meta)}")
    return 1 if any(r["status"] == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
