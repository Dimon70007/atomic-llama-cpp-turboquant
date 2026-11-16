#!/usr/bin/env python3
"""TASK-072: minimal p_min sweep on Groot-19B-A3B (in-GGUF MTP head).

Answers one question: does `--spec-draft-p-min` change the *text*, or only the
draft block length (speed / acceptance)?

p_min is a `break` in the draft loop (common/speculative.cpp): it truncates the
drafted block when the top-1 probability drops below the threshold. The target
verifies every token, so under greedy sampling the text should not depend on it.
If the text does move, that is a batch-shape leak (TASK-066 territory) and a
much larger finding than the knob itself.

Every cell runs greedy (temp 0, seed 42) in thinking mode - the operator
measurement mode - against one fixed prompt. Cells differ only in
`n_max` / `p_min` / MTP-on.

Usage:
    python3 tests/lib/task072_groot_pmin_sweep.py \
        --model ~/models/Groot-19B-A3B-261023-Q4_K_M.gguf \
        --bin build-dev/bin/llama-server --out-dir logs/task072-pmin

Exit: 0 every cell produced text, 1 a cell failed, 2 bad input.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

_RE_ACCEPT: Final[re.Pattern[str]] = re.compile(r"draft acceptance\s*=\s*([0-9.]+)")
_RE_MEANLEN: Final[re.Pattern[str]] = re.compile(r"mean len\s*=\s*([0-9.]+)")

# One English prompt: Groot is English-only (Russian PPL 392.8), and the answer
# carries a checkable number so a text change also reads as a fact change.
_DEFAULT_PROMPT: Final[str] = (
    "A train travels at 60 km/h for 2.5 hours, then at 80 km/h for 1.5 hours. "
    "Calculate the total distance and explain the arithmetic in two sentences."
)


@dataclass(frozen=True)
class Cell:
    """One measurement cell. `p_min` None means MTP off."""

    id: str
    n_max: int | None
    p_min: str | None

    @property
    def mtp(self) -> bool:
        return self.p_min is not None


# Minimal grid: the product cap (n_max 2) across the threshold, a no-speculation
# control at p_min 1.0 (draft always truncated), and two cap probes.
DEFAULT_CELLS: Final[tuple[Cell, ...]] = (
    Cell("nomtp", None, None),
    Cell("n2-p000", 2, "0.0"),
    Cell("n2-p071", 2, "0.71"),
    Cell("n2-p080", 2, "0.80"),
    Cell("n2-p090", 2, "0.90"),
    Cell("n2-p095", 2, "0.95"),
    Cell("n2-p100", 2, "1.0"),
    Cell("n1-p090", 1, "0.90"),
    Cell("n3-p071", 3, "0.71"),
    Cell("n3-p080", 3, "0.80"),
    Cell("n3-p090", 3, "0.90"),
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(port: int, proc: subprocess.Popen[bytes], timeout: float) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.25)
    return False


def _server_argv(
    bin_path: Path,
    model: Path,
    port: int,
    cell: Cell,
    ctx: int,
    ngl: int,
    batch: int,
    ubatch: int,
    ctk: str,
    ctv: str,
    extra_server_args: Sequence[str] = (),
) -> list[str]:
    argv = [
        str(bin_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        "-m", str(model),
        "-c", str(ctx),
        "-ngl", str(ngl),
        "-fit", "off",
        "-np", "1",
        "-b", str(batch),
        "-ub", str(ubatch),
        "-ctk", ctk,
        "-ctv", ctv,
        "-fa", "on",
        "-lv", "3",
        "--jinja",
        # Product mode: thinking on; sampling fixed greedy for reproducibility.
        "--chat-template-kwargs", '{"enable_thinking":true}',
        "--temp", "0",
        "--top-k", "1",
        "--top-p", "1.0",
        "--seed", "42",
    ]
    if cell.mtp:
        argv += [
            "--spec-type", "draft-mtp",
            "--spec-draft-n-max", str(cell.n_max),
            "--spec-draft-p-min", str(cell.p_min),
            # Draft KV types are product knobs: F16 default costs 1024 MiB at 262k.
            "-ctkd", ctk,
            "-ctvd", "turbo4",
        ]
    argv += extra_server_args
    return argv


def _request(port: int, prompt: str, max_tokens: int) -> dict[str, Any]:
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": 42,
            "stream": False,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read().decode("utf-8"))


class PromptTooLong(RuntimeError):
    """Server rejected the request because prompt + max_tokens exceeds its ctx."""


def parse_facts(raw: Any, *, source: str) -> list[tuple[str, str]]:
    """Validate ``[{id, regex}]`` fact rows; abort on anything unusable.

    Args:
        raw: Parsed JSON payload, expected to carry a ``facts`` list.
        source: Path label for error messages.

    Returns:
        ``[(id, regex)]`` in file order.

    Raises:
        SystemExit: The payload is not a mapping, ``facts`` is missing/empty, a
            row lacks ``id``/``regex``, or a regex does not compile.
    """
    if not isinstance(raw, dict):
        raise SystemExit(f"facts file must be a JSON object: {source}")
    rows = raw.get("facts")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{source}: missing non-empty 'facts' list")
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise SystemExit(f"{source}: fact row must be an object, got {row!r}")
        fact_id = str(row.get("id") or "").strip()
        regex = str(row.get("regex") or "").strip()
        if not fact_id or not regex:
            raise SystemExit(f"{source}: fact row needs id and regex, got {row!r}")
        if fact_id in seen:
            raise SystemExit(f"{source}: duplicate fact id {fact_id!r}")
        seen.add(fact_id)
        try:
            re.compile(regex)
        except re.error as exc:
            raise SystemExit(f"{source}: fact {fact_id!r} regex does not compile: {exc}")
        out.append((fact_id, regex))
    return out


def score_facts(emission: str, facts: Sequence[tuple[str, str]]) -> dict[str, bool]:
    """Return ``{fact_id: present}`` over the full emission.

    A fact is present when its regex matches anywhere in the emission
    (``reasoning_content`` + ``content``): thinking mode can put the arithmetic
    in the trace and never restate it in the answer.
    """
    return {fact_id: re.search(regex, emission) is not None for fact_id, regex in facts}


def check_facts_prompt(raw: Any, prompt: str, *, source: str) -> None:
    """Abort when the facts file declares a prompt the sweep is not sending.

    A facts file is only meaningful for the question it was written for. Scored
    against a different prompt it silently reports misses that are not model
    failures, so a mismatch is an error rather than a warning.

    Args:
        raw: Parsed facts JSON.
        prompt: The prompt the sweep will actually send.
        source: Path label for the error message.

    Raises:
        SystemExit: A declared prompt differs from the one in use.
    """
    declared = str(raw.get("prompt") or "").strip() if isinstance(raw, dict) else ""
    if not declared:
        raise SystemExit(
            f"{source}: missing 'prompt' (a facts file must name the question it scores)"
        )
    if declared != prompt.strip():
        raise SystemExit(
            f"{source}: declared prompt does not match the prompt being sent "
            "(scoring a different question would report false misses)"
        )


def run_cell(
    cell: Cell,
    bin_path: Path,
    model: Path,
    out_dir: Path,
    prompt: str,
    ctx: int,
    ngl: int,
    batch: int,
    ubatch: int,
    ctk: str,
    ctv: str,
    max_tokens: int,
    load_timeout: float,
    repeats: int = 1,
    extra_server_args: Sequence[str] = (),
    facts: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Start a server for one cell, run `repeats` greedy requests, tear it down.

    Repeats share one server process: model load dominates a cell (~11 s for
    Groot) while one 512-token request is a couple of seconds, so reloading per
    repeat would measure loading, not decoding.
    """
    cell_dir = out_dir / cell.id
    cell_dir.mkdir(parents=True, exist_ok=True)
    port = _free_port()
    argv = _server_argv(
        bin_path, model, port, cell, ctx, ngl, batch, ubatch, ctk, ctv, extra_server_args
    )
    (cell_dir / "argv.txt").write_text(" ".join(argv) + "\n", encoding="utf-8")

    log_path = cell_dir / "server.log"
    row: dict[str, Any] = {"id": cell.id, "n_max": cell.n_max, "p_min": cell.p_min}
    runs: list[dict[str, Any]] = []
    emissions: list[str] = []

    t0 = time.time()
    try:
        with log_path.open("wb") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                if not _wait_health(port, proc, load_timeout):
                    row["error"] = "server did not become healthy"
                    return row
                row["load_sec"] = round(time.time() - t0, 1)

                for r in range(repeats):
                    t1 = time.time()
                    try:
                        resp = _request(port, prompt, max_tokens)
                    except urllib.error.HTTPError as exc:
                        detail = exc.read().decode("utf-8", errors="replace")[:400]
                        if exc.code == 400 and "exceed" in detail.lower():
                            raise PromptTooLong(
                                f"prompt + max_tokens does not fit ctx: {detail}"
                            ) from exc
                        row["error"] = f"HTTP {exc.code}: {detail}"
                        return row
                    wall = round(time.time() - t1, 1)

                    choice = (resp.get("choices") or [{}])[0]
                    message = choice.get("message") or {}
                    text = str(message.get("content") or "")
                    reasoning = str(message.get("reasoning_content") or "")
                    emissions.append(reasoning + "\n<<<ANSWER>>>\n" + text)
                    if repeats == 1:
                        (cell_dir / "answer.txt").write_text(text, encoding="utf-8")
                        (cell_dir / "reasoning.txt").write_text(reasoning, encoding="utf-8")
                    else:
                        (cell_dir / f"answer{r:02d}.txt").write_text(text, encoding="utf-8")
                        (cell_dir / f"reasoning{r:02d}.txt").write_text(reasoning, encoding="utf-8")
                    (cell_dir / f"response{r:02d}.json").write_text(
                        json.dumps(resp, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )

                    timings = resp.get("timings") or {}
                    runs.append(
                        {
                            "wall_sec": wall,
                            "finish_reason": choice.get("finish_reason"),
                            "completion_tokens": timings.get("predicted_n"),
                            "decode_tok_per_s": timings.get("predicted_per_second"),
                            "draft_n": timings.get("draft_n"),
                            "draft_n_accepted": timings.get("draft_n_accepted"),
                            "answer_chars": len(text),
                        }
                    )
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=30)
    except PromptTooLong:
        raise

    row["runs"] = runs
    ok = [r for r in runs if r.get("decode_tok_per_s") is not None]
    if not ok:
        row["error"] = "no usable run"
        return row
    speeds = sorted(float(r["decode_tok_per_s"]) for r in ok)
    row["decode_tok_per_s"] = statistics.median(speeds)
    row["decode_min"] = speeds[0]
    row["decode_max"] = speeds[-1]
    for key in ("completion_tokens", "answer_chars", "finish_reason"):
        row[key] = ok[-1].get(key)

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    acc = _RE_ACCEPT.findall(log_text)
    mean_len = _RE_MEANLEN.findall(log_text)
    row["acceptance_log"] = float(acc[-1]) if acc else None
    row["mean_len_log"] = float(mean_len[-1]) if mean_len else None
    row["mtp_draft_context"] = "creating MTP draft context against the target model" in log_text
    # Fact scoring runs on the last emission: with repeats > 1 the earlier runs
    # are the ones affected by cross-request state (TASK-073), so the reported
    # accuracy would otherwise mix a clean run with a stale-state one.
    if facts and emissions:
        facts_last = score_facts(emissions[-1], facts)
        row["facts"] = facts_last
        row["facts_passed"] = sum(1 for ok_fact in facts_last.values() if ok_fact)
        row["facts_total"] = len(facts)
        row["facts_all_pass"] = all(facts_last.values())
    return row


def _read_emission(cell_dir: Path) -> str:
    """Full emission for one cell: reasoning trace + answer, all runs joined.

    Thinking mode puts the trace in reasoning_content and the answer in content,
    and a cell can finish inside the trace with an empty answer; comparing only
    the answer would make two empty answers look identical.
    """
    if (cell_dir / "reasoning.txt").is_file():
        pairs = [(cell_dir / "reasoning.txt", cell_dir / "answer.txt")]
    else:
        pairs = sorted(
            (p, p.with_name(p.name.replace("reasoning", "answer")))
            for p in cell_dir.glob("reasoning*.txt")
        )
    return "\n<<<RUN>>>\n".join(
        r.read_text(encoding="utf-8") + "\n<<<ANSWER>>>\n" + a.read_text(encoding="utf-8")
        for r, a in pairs
        if r.is_file() and a.is_file()
    )


def _write_summary(out_dir: Path, rows: list[dict[str, Any]], meta: dict[str, Any]) -> Path:
    texts = {
        r["id"]: _read_emission(out_dir / r["id"])
        for r in rows
        if (out_dir / r["id"]).is_dir()
    }
    texts = {k: v for k, v in texts.items() if v}
    # The reference is MTP-off: it is the sequential-decode truth. Any difference
    # between an MTP cell and it is drift introduced by speculative verification.
    ref_id = "nomtp" if "nomtp" in texts else (rows[0]["id"] if rows else "")
    ref_text = texts.get(ref_id, "")
    dist = {r["id"]: (texts[r["id"]] == ref_text) for r in rows if r["id"] in texts}

    lines = [
        "# TASK-072 p_min sweep (Groot-19B-A3B)",
        "",
        f"stamp: {meta['stamp']}",
        f"model: `{meta['model']}`",
        f"ctx {meta['ctx']}, greedy (temp 0, top_k 1, seed 42), thinking on, "
        f"max_tokens {meta['max_tokens']}",
        f"reference for text equality: `{ref_id}` (sequential decode, no draft)",
        "",
        "Text equality compares the full emission (`reasoning_content` + `content`),"
        " because thinking mode can consume the whole budget before any answer starts.",
        "",
    ]
    has_facts = any("facts_total" in r for r in rows)
    header = (
        "| cell | n_max | p_min | decode t/s (median) | min..max | acceptance | "
        "mean len | tokens | finish | answer chars | text == ref |"
    )
    separator = "|---|---|---|---|---|---|---|---|---|---|---|"
    if has_facts:
        header += " facts |"
        separator += "---|"
        lines += [f"scored facts: {meta.get('facts_passed_rule', '')}", ""]
    lines += [header, separator]
    for r in rows:
        spread = "-"
        if r.get("decode_min") is not None and r.get("decode_max") is not None:
            if r["decode_min"] != r["decode_max"]:
                spread = f"{r['decode_min']:.2f}..{r['decode_max']:.2f}"
        line = (
            "| {id} | {n} | {p} | {d} | {s} | {a} | {m} | {t} | {f} | {c} | {eq} |".format(
                id=r["id"],
                n=r.get("n_max") if r.get("n_max") is not None else "-",
                p=r.get("p_min") if r.get("p_min") is not None else "-",
                d=f"{r['decode_tok_per_s']:.2f}" if r.get("decode_tok_per_s") else "-",
                s=spread,
                a=(
                    f"{r['acceptance_log']:.4f}"
                    if r.get("acceptance_log") is not None
                    else "-"
                ),
                m=f"{r['mean_len_log']:.2f}" if r.get("mean_len_log") is not None else "-",
                t=r.get("completion_tokens") or "-",
                f=r.get("finish_reason") or r.get("error") or "-",
                c=r.get("answer_chars", "-"),
                eq="yes" if dist.get(r["id"]) else "**NO**",
            )
        )
        if has_facts:
            if r.get("facts_total") is None:
                line += " - |"
            else:
                line += f" {r['facts_passed']}/{r['facts_total']} |"
        lines.append(line)
    moved = [cid for cid, same in dist.items() if not same]
    truncated = [r["id"] for r in rows if r.get("finish_reason") == "length"]
    mtp_cells = [r["id"] for r in rows if r.get("p_min") is not None]
    exact = [c for c in mtp_cells if dist.get(c)]
    reference_is_nomtp = ref_id == "nomtp"
    lines += [
        "",
        "## Reading",
        "",
        f"MTP cells identical to `{ref_id}`: "
        + (", ".join(f"`{c}`" for c in exact) if exact else "none")
        + ".",
        "",
        (
            "Every MTP cell reproduced the sequential-decode trajectory, so `p_min` "
            "does not change the emitted text: it is a speed/acceptance knob."
            if not moved
            else "Cells that diverge from `" + ref_id + "`: "
            + ", ".join(f"`{c}`" for c in moved)
            + (
                " -> speculative verification changes the trajectory (TASK-066 class: "
                "the verify batch is not equivalent to sequential decode). `p_min`/`n_max` "
                "decide how far that divergence is allowed to run, they do not cause it."
                if reference_is_nomtp
                else " -> no `nomtp` cell in this run, so equality is only shown inside "
                "the MTP grid. Run with `nomtp` to compare against sequential decode."
            )
        ),
    ]
    if truncated:
        lines += [
            "",
            "Cells that hit `finish_reason=length`: "
            + ", ".join(f"`{c}`" for c in truncated)
            + ". A truncated cell compares only the prefix, so raise `--max-tokens`"
            " before quoting these cells as equality evidence.",
        ]
    if has_facts:
        scored = [r for r in rows if r.get("facts_total") is not None]
        perfect = [r["id"] for r in scored if r.get("facts_all_pass")]
        broken = [r["id"] for r in scored if not r.get("facts_all_pass")]
        lines += [
            "",
            "### Scored facts",
            "",
            f"Source: `{meta.get('facts_source', '')}` over the full emission.",
            "",
            "Cells with every fact present: "
            + (", ".join(f"`{c}`" for c in perfect) if perfect else "none")
            + ".",
        ]
        if broken:
            lines += [
                "",
                "Cells missing at least one fact: " + ", ".join(f"`{c}`" for c in broken) + ".",
            ]
            for r in scored:
                if r.get("facts_all_pass"):
                    continue
                missing = sorted(k for k, ok_fact in (r.get("facts") or {}).items() if not ok_fact)
                lines.append(f"- `{r['id']}`: missing {', '.join(missing)}")
        lines += [
            "",
            "A fact miss is a quality finding, not a knob effect, unless the speed"
            " column also separates: divergence between MTP and sequential decode is"
            " the TASK-066 class effect and reproduces at any `p_min`.",
        ]
    path = out_dir / "summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="TASK-072 Groot p_min sweep")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--bin", required=True, type=Path, dest="bin_path")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--prompt", default=_DEFAULT_PROMPT)
    ap.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="read the prompt from a file (overrides --prompt); use for long-context cells",
    )
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--ngl", type=int, default=999)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--ctk", default="q8_0")
    ap.add_argument("--ctv", default="turbo4.8")
    ap.add_argument("--load-timeout", type=float, default=600.0)
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="requests per cell on one server load (median is reported when >1)",
    )
    ap.add_argument(
        "--server-arg",
        action="append",
        default=[],
        dest="server_args",
        metavar="ARG",
        help="extra llama-server argument, repeatable (e.g. --server-arg=--cache-ram --server-arg=0)",
    )
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated cell ids (default: the built-in minimal grid)",
    )
    ap.add_argument(
        "--facts",
        type=Path,
        default=None,
        help="JSON facts file ({facts:[{id,regex}]}) scored against each emission",
    )
    args = ap.parse_args(argv)

    for path, name in ((args.model, "--model"), (args.bin_path, "--bin")):
        if not path.is_file():
            print(f"FAIL: {name} not found: {path}", file=sys.stderr)
            return 2

    cells = list(DEFAULT_CELLS)
    if args.only:
        wanted = [c.strip() for c in args.only.split(",") if c.strip()]
        known = {c.id: c for c in cells}
        missing = [c for c in wanted if c not in known]
        if missing:
            raise SystemExit(f"unknown --only cells {missing}; known={sorted(known)}")
        cells = [known[c] for c in wanted]

    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            print(f"FAIL: --prompt-file not found: {args.prompt_file}", file=sys.stderr)
            return 2
        prompt = args.prompt_file.read_text(encoding="utf-8")
        prompt_source = str(args.prompt_file)
    else:
        prompt = args.prompt
        prompt_source = "<--prompt>"

    facts: list[tuple[str, str]] = []
    if args.facts is not None:
        if not args.facts.is_file():
            print(f"FAIL: --facts not found: {args.facts}", file=sys.stderr)
            return 2
        try:
            facts_raw = json.loads(args.facts.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"FAIL: --facts is not valid JSON: {exc}", file=sys.stderr)
            return 2
        facts = parse_facts(facts_raw, source=str(args.facts))
        check_facts_prompt(facts_raw, prompt, source=str(args.facts))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "stamp": stamp,
        "model": str(args.model),
        "bin": str(args.bin_path),
        "ctx": args.ctx,
        "max_tokens": args.max_tokens,
        "batch": args.batch,
        "ubatch": args.ubatch,
        "ctk": args.ctk,
        "ctv": args.ctv,
        # Long prompts are not echoed into the stamp; record the source and size.
        "prompt_source": prompt_source,
        "prompt_chars": len(prompt),
        "prompt": prompt if len(prompt) <= 2000 else prompt[:2000] + "...[truncated]",
        "cells": [c.id for c in cells],
        "facts_source": str(args.facts) if args.facts is not None else "",
        "facts_passed_rule": (
            "regex must match the full emission (reasoning_content + content); "
            "last run only when repeats > 1"
        ),
        "facts": [{"id": fid, "regex": rx} for fid, rx in facts],
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, Any]] = []
    failed = False
    for i, cell in enumerate(cells, start=1):
        print(f"[{i}/{len(cells)}] {cell.id}", flush=True)
        try:
            row = run_cell(
                cell, args.bin_path, args.model, out_dir, prompt,
                args.ctx, args.ngl, args.batch, args.ubatch, args.ctk, args.ctv,
                args.max_tokens, args.load_timeout, args.repeats, args.server_args,
                facts,
            )
        except PromptTooLong as exc:
            print(
                f"FAIL: {exc}\n"
                f"  prompt is {len(prompt)} chars; raise --ctx "
                f"(currently {args.ctx}) or lower --max-tokens ({args.max_tokens})",
                file=sys.stderr,
            )
            return 2
        rows.append(row)
        (out_dir / "rows.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if row.get("error") or row.get("decode_tok_per_s") is None:
            failed = True
            print(f"  FAILED: {row.get('error') or 'no timings'}", file=sys.stderr)
            continue
        facts_note = ""
        if row.get("facts_total") is not None:
            facts_note = f", facts {row['facts_passed']}/{row['facts_total']}"
        print(
            "  {:.2f} t/s, acc {}, mean len {}, {} tok, finish {}{}".format(
                row["decode_tok_per_s"],
                row.get("acceptance_log"),
                row.get("mean_len_log"),
                row.get("completion_tokens"),
                row.get("finish_reason"),
                facts_note,
            ),
            flush=True,
        )

    summary = _write_summary(out_dir, rows, meta)
    print(f"==> {summary}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
