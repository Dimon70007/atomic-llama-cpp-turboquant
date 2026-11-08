#!/usr/bin/env python3
"""TASK-068 ST-8: one-shot Darwin host / process memory sample (no inline bash Python).

Writes a single TSV line to stdout (or --append file). Fail-closed on bad args.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str], timeout: float = 10.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def _parse_swap_used_mib(raw: str) -> str:
    # vm.swapusage: total = N.00M  used = M.00M  free = F.00M  (encrypted)
    match = re.search(r"used\s*=\s*([0-9.]+)M", raw)
    if not match:
        return ""
    return match.group(1)


def _rss_kib(pid: int) -> str:
    rc, out = _run(["/bin/ps", "-o", "rss=", "-p", str(pid)])
    if rc != 0:
        return ""
    return out.strip().splitlines()[0].strip() if out.strip() else ""


def _pressure_free_pct(raw: str) -> str:
    match = re.search(r"free percentage:\s*([0-9]+)%", raw, re.IGNORECASE)
    if not match:
        return ""
    return match.group(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="sample label (required)")
    parser.add_argument("--pid", type=int, default=0, help="optional process pid for RSS")
    parser.add_argument(
        "--append",
        type=Path,
        default=None,
        help="append TSV line to this file (also print to stdout)",
    )
    parser.add_argument(
        "--header",
        action="store_true",
        help="write TSV header before the sample line",
    )
    args = parser.parse_args()

    label = str(args.label).strip()
    if not label:
        print("fail-closed: --label must be non-empty", file=sys.stderr)
        return 1

    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _, swap_raw = _run(["/usr/sbin/sysctl", "vm.swapusage"])
    swap_used = _parse_swap_used_mib(swap_raw)
    _, mp_raw = _run(["/usr/bin/memory_pressure", "-Q"])
    free_pct = _pressure_free_pct(mp_raw)
    rss = _rss_kib(args.pid) if args.pid > 0 else ""

    header = "utc\tlabel\tpid\trss_kib\tswap_used_mib\tmem_free_pct\tswap_raw"
    line = (
        f"{utc}\t{label}\t{args.pid or ''}\t{rss}\t{swap_used}\t{free_pct}\t"
        f"{swap_raw.replace(chr(9), ' ').replace(chr(10), ' | ')}"
    )

    chunks: list[str] = []
    if args.header:
        chunks.append(header)
    chunks.append(line)
    text = "\n".join(chunks) + "\n"
    sys.stdout.write(text)
    if args.append is not None:
        args.append.parent.mkdir(parents=True, exist_ok=True)
        with args.append.open("a", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
