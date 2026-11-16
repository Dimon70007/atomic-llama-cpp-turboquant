#!/usr/bin/env python3
"""TASK-073 ST-1: distinct-emission verdict for a checkpoint repeatability cell.

One greedy request repeated N times against one llama-server must emit one text.
The repeats share the process, so `--ctx-checkpoints` restore is exercised from
the second request on. Before the MTP-state fix the restored checkpoint leaves
the draft driver on the previous request and the cell cycles between texts.

Reads the cell directory written by task072_groot_pmin_sweep.py and only adds a
pass/fail verdict; it does not run the server.

Usage:
    python3 tests/lib/task073_checkpoint_repeat_verdict.py --cell-dir <out>/<stamp>/n2-p090

Exit: 0 one distinct emission, 1 more than one, 2 bad input.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_HASH_CHARS = 10


def emissions(cell_dir: Path) -> list[tuple[str, str]]:
    """(run label, reasoning + answer) for every repeat, in run order.

    Thinking mode can spend the whole budget inside the trace, so the answer
    alone is not a sufficient fingerprint; the runner compares the same pair.
    """
    single = cell_dir / "reasoning.txt"
    if single.is_file():
        pairs = [(single, cell_dir / "answer.txt")]
    else:
        pairs = [
            (reasoning, reasoning.with_name(reasoning.name.replace("reasoning", "answer")))
            for reasoning in sorted(cell_dir.glob("reasoning[0-9]*.txt"))
        ]
    return [
        (
            reasoning.stem,
            reasoning.read_text(encoding="utf-8")
            + "\n<<<ANSWER>>>\n"
            + answer.read_text(encoding="utf-8"),
        )
        for reasoning, answer in pairs
        if reasoning.is_file() and answer.is_file()
    ]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="TASK-073 checkpoint repeatability verdict")
    ap.add_argument("--cell-dir", required=True, type=Path)
    ap.add_argument("--min-repeats", type=int, default=3)
    args = ap.parse_args(argv)

    if not args.cell_dir.is_dir():
        print(f"FAIL: cell dir not found: {args.cell_dir}", file=sys.stderr)
        return 2

    runs = emissions(args.cell_dir)
    if len(runs) < args.min_repeats:
        print(
            f"FAIL: {len(runs)} repeat(s) in {args.cell_dir}, need at least {args.min_repeats};"
            " a cycle cannot be seen below that",
            file=sys.stderr,
        )
        return 2

    digests = [hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_CHARS] for _, text in runs]
    print("run  hash        chars")
    for (label, text), digest in zip(runs, digests):
        print(f"{label:>3}  {digest}  {len(text):6d}")

    unique = sorted(set(digests))
    order = " ".join(digests)
    print(f"distinct={len(unique)} of {len(digests)}: {', '.join(unique)}")
    print(f"sequence={order}")
    if len(unique) == 1:
        print(f"VERDICT=PASS: one emission over {len(digests)} repeats in one server")
        return 0

    print(
        "VERDICT=FAIL: one server emitted "
        f"{len(unique)} texts over {len(digests)} repeats of the same greedy request"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
