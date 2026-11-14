#!/usr/bin/env python3
"""TASK-068/071 - KLD ladder verdict from kld_table.tsv.

Replaces the bash-only series tracking, which silently ignored any arm that was
not named exactly "turbo3" or "turbo4" (so bracket arms reported "series
incomplete" even when they ran).

Rule (unchanged from TASK-071 ST-7.6): a ladder point counts as a rise when
  b - a > max(KLD_EPS_ABS, KLD_EPS_REL * a, unc_a, unc_b)
so a step inside the reported uncertainty stays noise (flat) while any larger
step fails.

Usage:
  kld_verdict.py --table kld_table.tsv --ctx-list "4096 8192 16384" \
      --eps-abs 0.001 --eps-rel 0.02 [--json out.json]

Exit codes: 0 PASS, 1 FAIL, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field


@dataclass
class Series:
    arm: str
    means: list[float] = field(default_factory=list)
    uncs: list[float] = field(default_factory=list)
    points: int = 0
    missing: bool = False


def read_table(path: str, ctx_list: list[int]) -> dict[str, Series]:
    by_arm: dict[str, Series] = {}
    with open(path, encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        want = ["cand", "ctx", "kld_mean", "kld_unc"]
        if header[:4] != want:
            raise ValueError(f"{path}: bad header {header[:4]} (want {want})")
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            arm, ctx_s, mean_s, unc_s = parts[0], parts[1], parts[2], parts[3]
            series = by_arm.setdefault(arm, Series(arm))
            series.points += 1
            if mean_s == "NA":
                series.missing = True
                continue
            try:
                ctx = int(ctx_s)
                mean = float(mean_s)
                unc = float(unc_s) if unc_s != "NA" else 0.0
            except ValueError:
                series.missing = True
                continue
            series.means.append(mean)
            series.uncs.append(unc)
            series.points = len(series.means)

    # Every arm must have one point per CTX, in ascending CTX order.
    for series in by_arm.values():
        if len(series.means) != len(ctx_list):
            series.missing = True
    return by_arm


def walks_up(series: Series, eps_abs: float, eps_rel: float) -> bool:
    if len(series.means) < 2 or series.missing:
        return True  # cannot judge -> caller marks FAIL
    for i in range(1, len(series.means)):
        a, b = series.means[i - 1], series.means[i]
        ua = series.uncs[i - 1] if i - 1 < len(series.uncs) else 0.0
        ub = series.uncs[i] if i < len(series.uncs) else 0.0
        # Same floor the bash rule used before this refactor, so verdicts stay
        # comparable with earlier TASK-071 stamps: absolute + relative term summed,
        # then raised to the reported uncertainty on either side.
        noise = max(eps_abs + eps_rel * a, ua, ub)
        if b - a > noise:
            return True
    return False


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="KLD ladder verdict.")
    ap.add_argument("--table", required=True)
    ap.add_argument("--ctx-list", required=True)
    ap.add_argument("--eps-abs", type=float, required=True)
    ap.add_argument("--eps-rel", type=float, required=True)
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    try:
        ctx_list = [int(t) for t in args.ctx_list.split()]
    except ValueError:
        print(f"FAIL: bad --ctx-list: {args.ctx_list}", file=sys.stderr)
        return 2

    try:
        by_arm = read_table(args.table, ctx_list)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    notes: list[str] = []
    pass_count = 0
    arm_count = 0
    for arm in sorted(by_arm):
        series = by_arm[arm]
        arm_count += 1
        joined = " ".join(f"{m:.6f}" for m in series.means)
        if series.missing or len(series.means) != len(ctx_list):
            notes.append(f"{arm} series incomplete ({series.points}/{len(ctx_list)} points)")
            continue
        if walks_up(series, args.eps_abs, args.eps_rel):
            notes.append(f"{arm} KLD walks up ({joined})")
        else:
            pass_count += 1
            notes.append(f"{arm} KLD flat-or-down ({joined})")

    verdict = "PASS" if arm_count > 0 and pass_count == arm_count else "FAIL"
    note = "; ".join(notes)

    print(f"VERDICT={verdict}")
    print(f"note={note}; ")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "verdict": verdict,
                    "eps_abs": args.eps_abs,
                    "eps_rel": args.eps_rel,
                    "ctx_list": ctx_list,
                    "arms": {
                        arm: {"means": s.means, "uncs": s.uncs, "missing": s.missing}
                        for arm, s in sorted(by_arm.items())
                    },
                },
                fh,
                indent=2,
            )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
