#!/usr/bin/env python3
"""TASK-073.1.5: are repeated greedy requests bit-identical, measured on logprobs.

Why not text: a ~4% top-1 change relocates a 60-token answer (TASK-077 F41), so two
different texts can come from a numerically identical distribution and one text can
hide a real difference. Why not test-task077-kv-logit-parity: it drives
llama_context directly, has no speculative path and compares two *different*
configurations, so it cannot express "same request, repeated" with MTP on
(TASK-073.1.5 F1).

What this does: one llama-server, one prompt, N identical greedy requests with
`logprobs` requested, then compares the per-token distributions of every repeat
against repeat 0. It reports the first token index whose top-1 or distribution
differs, and the numeric distance there. That separates the two hypotheses the
task is split on:

  * identical logprobs, different chosen token -> the sampler/verify step made a
    different choice from the same numbers;
  * different logprobs at token k -> the kernels produced different numbers, which
    is a batch/shape problem (TASK-069 / TASK-077 class).

The server is started by the caller; this script only talks HTTP. That keeps the
MTP flags, ctv and ubatch under the caller's control, since those are the knobs
under test.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Final

_DEFAULT_PROMPT: Final[str] = (
    "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
    "Calculate the total distance and explain the arithmetic in two sentences."
)

# A top-1 probability difference below this is rounding, not a different outcome.
_EPS_TOP1: Final[float] = 1e-6


def _post(base_url: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    """POST one chat completion and return the parsed JSON."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _token_logprobs(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-token logprob records in generation order."""
    choices = response.get("choices") or []
    if not choices:
        return []
    content = choices[0].get("logprobs") or {}
    return list(content.get("content") or [])


def _top_candidates(record: dict[str, Any], limit: int) -> dict[str, float]:
    """{token: logprob} for the most likely candidates of one step."""
    rows = list(record.get("top_logprobs") or [])
    rows.sort(key=lambda r: float(r.get("logprob") or -math.inf), reverse=True)
    out: dict[str, float] = {}
    for row in rows[:limit]:
        token = str(row.get("token") or "")
        out[token] = float(row.get("logprob") or -math.inf)
    return out


def _max_abs_delta(a: dict[str, float], b: dict[str, float]) -> tuple[float, int]:
    """Largest |logprob| difference over the candidates both runs list.

    Returns (delta, membership_changes). Candidates present in only one run are
    counted separately instead of collapsing the result to infinity: with a top-N
    cut, a token at rank 6 in one run and rank 5 in the other changes membership
    for reasons that are not a numeric difference in the distribution. Reporting
    infinity there would have made the first version of this probe claim a huge
    divergence on every comparison.
    """
    keys_common = set(a) & set(b)
    membership = len(set(a) ^ set(b))
    worst = 0.0
    for key in keys_common:
        worst = max(worst, abs(a[key] - b[key]))
    return worst, membership


def compare(run0: list[dict[str, Any]], other: list[dict[str, Any]], top_n: int) -> dict[str, Any]:
    """Compare one repeat against repeat 0, step by step."""
    n = min(len(run0), len(other))
    first_divergence: int | None = None
    first_top1_change: int | None = None
    worst_delta = 0.0
    worst_at = -1
    membership_changes = 0

    for i in range(n):
        a = _top_candidates(run0[i], top_n)
        b = _top_candidates(other[i], top_n)
        delta, membership = _max_abs_delta(a, b)
        membership_changes += membership
        if delta > _EPS_TOP1 and first_divergence is None:
            first_divergence = i
        if delta > worst_delta:
            worst_delta, worst_at = delta, i

        tok_a = str(run0[i].get("token") or "")
        tok_b = str(other[i].get("token") or "")
        if tok_a != tok_b and first_top1_change is None:
            first_top1_change = i

    return {
        "compared_tokens": n,
        "first_distribution_divergence": first_divergence,
        "first_top1_change": first_top1_change,
        "max_logprob_delta": worst_delta,
        "max_delta_at": worst_at,
        "membership_changes": membership_changes,
    }


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="TASK-073.1.5 repeated-greedy logprobs probe")
    ap.add_argument("--base-url", required=True, help="llama-server base URL, e.g. http://127.0.0.1:8080")
    ap.add_argument("--repeats", type=int, default=4, help="identical requests to send (default 4)")
    ap.add_argument("--max-tokens", type=int, default=256, help="tokens per request (default 256)")
    ap.add_argument("--top-logprobs", type=int, default=5, help="candidates per step (default 5)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--prompt", default=_DEFAULT_PROMPT, help="prompt text, or use --prompt-file")
    ap.add_argument("--prompt-file", type=Path, default=None)
    ap.add_argument("--think", action="store_true", help="leave thinking enabled")
    ap.add_argument("--timeout-sec", type=float, default=1800.0)
    ap.add_argument("--out-dir", type=Path, default=None, help="write raw responses here")
    args = ap.parse_args(argv)

    if args.repeats < 2:
        print("FAIL: --repeats must be >= 2; a single run has nothing to compare", file=sys.stderr)
        return 2

    prompt = args.prompt
    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            print(f"FAIL: prompt file not found: {args.prompt_file}", file=sys.stderr)
            return 2
        prompt = args.prompt_file.read_text(encoding="utf-8")

    body: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_k": 1,
        "top_p": 1.0,
        "seed": args.seed,
        "logprobs": True,
        "top_logprobs": args.top_logprobs,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": bool(args.think)},
    }

    responses: list[dict[str, Any]] = []
    for i in range(args.repeats):
        try:
            responses.append(_post(args.base_url, body, args.timeout_sec))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            print(f"FAIL: request {i} -> HTTP {exc.code}: {detail}", file=sys.stderr)
            return 2
        except (urllib.error.URLError, OSError) as exc:
            print(f"FAIL: request {i} -> {exc}", file=sys.stderr)
            return 2

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for i, resp in enumerate(responses):
            (args.out_dir / f"response{i:02d}.json").write_text(
                json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    series = [_token_logprobs(r) for r in responses]
    if not series[0]:
        print(
            "FAIL: the server returned no logprobs; MTP may bypass them entirely. "
            "That is itself a finding - record it instead of reporting a pass.",
            file=sys.stderr,
        )
        return 2

    print(f"repeats: {args.repeats}, tokens compared: {len(series[0])}")
    print()
    print(f"{'repeat':>6}  {'tokens':>6}  {'first_div':>9}  {'first_top1':>10}  {'max_dlogp':>10}  {'rank_chg':>8}")
    worst: dict[str, Any] | None = None
    for i in range(1, len(series)):
        result = compare(series[0], series[i], args.top_logprobs)
        div = "-" if result["first_distribution_divergence"] is None else str(result["first_distribution_divergence"])
        t1 = "-" if result["first_top1_change"] is None else str(result["first_top1_change"])
        print(
            f"{i:>6}  {result['compared_tokens']:>6}  {div:>9}  {t1:>10}  "
            f"{result['max_logprob_delta']:>10.6g}  {result['membership_changes']:>8}"
        )
        if worst is None or result["max_logprob_delta"] > worst["max_logprob_delta"]:
            worst = result

    print()
    assert worst is not None
    if worst["max_logprob_delta"] <= _EPS_TOP1:
        print("VERDICT=IDENTICAL: logprobs match across repeats.")
        print("Any text difference is therefore not a numeric difference in this distribution -")
        print("look at the sampler/verify step, not the kernels.")
        return 0

    print(f"VERDICT=DIVERGED: max |dlogprob| = {worst['max_logprob_delta']:.6g} "
          f"at token {worst['max_delta_at']}")
    if worst["first_top1_change"] is None:
        print("Top-1 never changed, so the same token was chosen from different numbers.")
    print("Different numbers for the same prompt means a kernel/batch-shape effect:")
    print("that is TASK-069 / TASK-077 territory, not a sampler question.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
