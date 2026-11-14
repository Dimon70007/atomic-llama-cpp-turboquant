#!/usr/bin/env python3
"""TASK-068: numeric e2e scorer for KV arms (turbo3 vs turbo3.8, turbo4 vs turbo4.8).

Why this exists
---------------
KLD measures distribution drift on teacher-forced tokens. The operator's question is
different: does the *answer* get facts wrong, and how - softened or inverted? This
scorer turns generated summaries into per-arm counters so arms can be compared.

Distortion taxonomy (per fact):
    correct   - the fact is stated and not contradicted
    softened  - the fact is hedged, modalised or made vague enough to be wrong
                ("возможно ранен", "вероятно стрелял", "по-видимому")
    inverted  - the fact is contradicted (wrong actor, wrong outcome)
    missing   - the fact is absent and not contradicted

Counters per arm: facts_correct / facts_softened / facts_inverted / facts_missing,
plus a weighted score and a hard-fail flag for any inversion.

Usage:
    python3 task068_distortion_rubric.py --json-out out.json summary1.txt [dir2 ...]
    python3 task068_distortion_rubric.py --compare a.json b.json --label-a turbo3 --label-b turbo3.8

Exit codes: 0 scored, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final


@dataclass(frozen=True)
class Fact:
    """One checkable fact derived from the fixture text."""

    id: str
    weight: float
    label: str
    patterns: tuple[str, ...]
    # Patterns that invert this fact (wrong actor / opposite outcome).
    inversions: tuple[str, ...] = ()


# Derived from the fixture itself: assets/literature/fixtures/task066_pierre_dolokhov_duel_2k.txt
# (5194 chars, ~2k tokens). Only facts whose evidence is IN this excerpt are listed;
# a fact the text never states cannot be "missing" and would inflate every arm equally.
#   "Мимо! -- крикнул Долохов"                       -> Dolokhov's shot misses
#   "Ростов с Денисовым повезли раненого Долохова"    -> wound + transport
#   "Несвицкий остановил его и повез домой"           -> Nesvitsky takes Pierre home
#   "Пьер схватился за голову ... Глупо... глупо!"    -> Pierre's reaction
#   "Мы в Москве"                                     -> they arrive in Moscow
#   "восторженно-нежное выражение лица Долохова"      -> tenderness beat
#   "Она не перенесет этого"                          -> Dolokhov fears for her
_FACTS: Final[tuple[Fact, ...]] = (
    Fact(
        "pierre_shoots",
        3.0,
        "Pierre fires at Dolokhov",
        (r"пьер.*выстрел", r"выстрел\w*\s+пьера", r"пьер\w*\s+стреля"),
        (r"долохов\w*\s+(?:смертельно\s+)?ранил\w*\s+пьер", r"долохов\w*\s+стреля\w*\s+в\s+пьер"),
    ),
    Fact(
        "dolokhov_misses",
        2.0,
        "Dolokhov fires and misses",
        (r"мимо", r"промах\w*", r"не\s+попал"),
    ),
    Fact(
        "dolokhov_wounded",
        3.0,
        "Dolokhov is wounded",
        (r"ранен\w*\s+долох", r"долохов\w*\s+(?:был\s+)?ранен", r"ранил\w*\s+долох", r"ран(?:я|ит|ил)\s+его", r"раненого\s+долохова"),
        (r"никто\s+не\s+(?:был\s+)?ранен",),
    ),
    Fact(
        "second_nestvitsky",
        1.0,
        "Nesvitsky is present and takes Pierre home",
        (r"несвицк",),
    ),
    Fact(
        "second_denisov_rostov",
        1.0,
        "Denisov and Rostov carry the wounded Dolokhov",
        (r"денисов", r"ростов"),
    ),
    Fact(
        "arrive_moscow",
        2.0,
        "They arrive in Moscow after the duel",
        (r"москв",),
    ),
    Fact(
        "dolokhov_tenderness",
        2.0,
        "Dolokhov's changed, tender expression",
        (r"нежн", r"изменивш", r"любящ", r"восторженн"),
    ),
    Fact(
        "dolokhov_fears_for_her",
        2.0,
        "Dolokhov says she will not survive it",
        (r"не\s+перенес", r"она\s+не\s+перенес", r"убил\s+ее", r"убил\s+её"),
    ),
)

# Hedges that turn a statement into a softened, non-committal claim.
# Applied per sentence, only when the fact pattern also matches there.
_HEDGES: Final[tuple[str, ...]] = (
    r"возможно",
    r"вероятно",
    r"по-видимому",
    r"похоже",
    r"кажется",
    r"якобы",
    r"будто",
    r"як\s*бы",
    r"неясно",
    r"неизвестно",
    r"предположительно",
    r"может\s+быть",
    r"словно",
    r"как\s+будто",
)

_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?…])\s+|\n+")

# Always-wrong claims, independent of whether the excerpt states the positive fact.
# "Pierre died" is false because the text shows him walking away after Dolokhov misses.
_GLOBAL_INVERSIONS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    (
        "pierre_died",
        "Claims Pierre was killed / died",
        (r"пьер\w*\s+(?:был\s+)?убит", r"смерть\s+пьера", r"пьер\w*\s+погиб"),
    ),
    (
        "nobody_wounded",
        "Claims nobody was wounded",
        (r"никто\s+не\s+(?:был\s+)?ранен",),
    ),
)


@dataclass
class TextScore:
    """Scoring result for one generated text."""

    target: str
    per_fact: list[dict[str, Any]] = field(default_factory=list)
    counters: dict[str, int] = field(default_factory=dict)
    score: float = 0.0
    hard_fail: bool = False
    chars: int = 0
    sentences: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "per_fact": self.per_fact,
            "counters": self.counters,
            "score": self.score,
            "hard_fail": self.hard_fail,
            "chars": self.chars,
            "sentences": self.sentences,
        }


def _sentences(text: str) -> list[str]:
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    return parts


def _has_hedge(sentence: str) -> bool:
    low = sentence.casefold()
    return any(re.search(h, low) for h in _HEDGES)


def classify_fact(fact: Fact, sentences: list[str]) -> tuple[str, str | None]:
    """Return (verdict, matched_text). Verdict: correct|softened|inverted|missing."""
    for s in sentences:
        low = s.casefold()
        if any(re.search(p, low) for p in fact.inversions):
            return "inverted", s
    hits = [s for s in sentences if any(re.search(p, s.casefold()) for p in fact.patterns)]
    if not hits:
        return "missing", None
    # Any unhedged statement wins; a hedged one is softened.
    for s in hits:
        if not _has_hedge(s):
            return "correct", s
    return "softened", hits[0]


def score_text(target: str, text: str) -> TextScore:
    """Score one summary against the frozen fact list."""
    sents = _sentences(text or "")
    result = TextScore(target=target, chars=len(text or ""), sentences=len(sents))
    earned = 0.0
    possible = 0.0
    counters = {"correct": 0, "softened": 0, "inverted": 0, "missing": 0}

    for fact in _FACTS:
        possible += fact.weight
        verdict, evidence = classify_fact(fact, sents)
        counters[verdict] += 1
        if verdict == "correct":
            earned += fact.weight
        elif verdict == "softened":
            earned += fact.weight * 0.5  # half credit: present but unreliable
        result.per_fact.append(
            {
                "id": fact.id,
                "weight": fact.weight,
                "label": fact.label,
                "verdict": verdict,
                "evidence": (evidence or "")[:200],
            }
        )

    result.counters = counters
    result.score = round(earned / possible, 4) if possible else 0.0

    # Global inversions: wrong regardless of which per-fact patterns matched.
    for gid, glabel, patterns in _GLOBAL_INVERSIONS:
        for s in sents:
            if any(re.search(p, s.casefold()) for p in patterns):
                result.per_fact.append(
                    {
                        "id": gid,
                        "weight": 0.0,
                        "label": glabel,
                        "verdict": "inverted",
                        "evidence": s[:200],
                    }
                )
                counters["inverted"] += 1
                break

    result.hard_fail = counters["inverted"] > 0
    return result


def _iter_targets(paths: list[Path]) -> list[tuple[str, str]]:
    """Expand files/dirs into (label, text) pairs."""
    out: list[tuple[str, str]] = []
    for p in paths:
        if p.is_dir():
            for f in sorted(p.rglob("*.txt")):
                out.append((str(f.relative_to(p)), f.read_text(encoding="utf-8", errors="replace")))
        elif p.is_file():
            out.append((p.name, p.read_text(encoding="utf-8", errors="replace")))
        else:
            raise SystemExit(f"missing path: {p}")
    if not out:
        raise SystemExit("no .txt files found")
    return out


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    tot = {"correct": 0, "softened": 0, "inverted": 0, "missing": 0}
    for r in rows:
        for k in tot:
            tot[k] += r["counters"][k]
    facts_total = sum(tot.values()) or 1
    return {
        "runs": n,
        "facts": tot,
        "facts_total": sum(tot.values()),
        "rate_correct": round(tot["correct"] / facts_total, 4),
        "rate_softened": round(tot["softened"] / facts_total, 4),
        "rate_inverted": round(tot["inverted"] / facts_total, 4),
        "rate_missing": round(tot["missing"] / facts_total, 4),
        "mean_score": round(sum(r["score"] for r in rows) / n, 4),
        "runs_with_inversion": sum(1 for r in rows if r["hard_fail"]),
    }


def _compare(a: dict[str, Any], b: dict[str, Any], la: str, lb: str) -> int:
    agg_a = a.get("aggregate", {})
    agg_b = b.get("aggregate", {})
    if not agg_a or not agg_b:
        print("FAIL: both inputs need an aggregate block", file=sys.stderr)
        return 2
    print(f"{'metric':<22} {la:>12} {lb:>12} {'delta':>12}")
    print("-" * 62)
    for key in ("mean_score", "rate_correct", "rate_softened", "rate_inverted", "rate_missing"):
        va = agg_a.get(key, 0.0)
        vb = agg_b.get(key, 0.0)
        print(f"{key:<22} {va:>12.4f} {vb:>12.4f} {vb - va:>+12.4f}")
    inv_a = agg_a.get("runs_with_inversion", 0)
    inv_b = agg_b.get("runs_with_inversion", 0)
    print(f"{'runs_with_inversion':<22} {inv_a:>12} {inv_b:>12} {inv_b - inv_a:>+12}")
    better = agg_b.get("mean_score", 0.0) > agg_a.get("mean_score", 0.0)
    print(f"\nVERDICT: {lb} {'BETTER' if better else 'not better'} than {la} (mean_score)")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="TASK-068 e2e distortion rubric")
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A_JSON", "B_JSON"))
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument(
        "--verify-against",
        type=Path,
        help="fixture text: assert every fact pattern actually occurs (guards dead facts)",
    )
    args = ap.parse_args(argv)

    if args.verify_against:
        try:
            fixture = args.verify_against.read_text(encoding="utf-8").casefold()
        except OSError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        dead = [
            f.id
            for f in _FACTS
            if not any(re.search(p, fixture) for p in f.patterns)
        ]
        if dead:
            print(f"FAIL: facts with no textual support in {args.verify_against}: {dead}", file=sys.stderr)
            print("FAIL: a fact absent from the fixture scores 'missing' for every arm", file=sys.stderr)
            return 2
        print(f"OK: all {len(_FACTS)} facts are supported by {args.verify_against.name}")
        return 0

    if args.compare:
        try:
            a = json.loads(args.compare[0].read_text(encoding="utf-8"))
            b = json.loads(args.compare[1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        return _compare(a, b, args.label_a, args.label_b)

    if not args.paths:
        print("FAIL: give summary paths or --compare", file=sys.stderr)
        return 2

    rows = [score_text(label, text).as_dict() for label, text in _iter_targets(args.paths)]
    for r in rows:
        c = r["counters"]
        flag = "HARDFAIL" if r["hard_fail"] else "ok"
        print(
            f"{flag:9} score={r['score']:.2f}  correct={c['correct']} softened={c['softened']} "
            f"inverted={c['inverted']} missing={c['missing']}  {r['target']}"
        )
        for f in r["per_fact"]:
            if f["verdict"] in ("softened", "inverted"):
                print(f"          {f['verdict']:8} {f['id']}: {f['evidence'][:110]}")

    payload = {"targets": rows, "aggregate": _aggregate(rows)}
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"==> {args.json_out}")
    else:
        print(json.dumps(payload["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
