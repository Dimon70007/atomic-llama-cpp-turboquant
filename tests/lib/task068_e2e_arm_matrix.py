#!/usr/bin/env python3
"""TASK-068: e2e MTP quality comparison across KV arms (numeric, per-arm).

Answers the operator question: for each KV type, how often does the model get facts
wrong, and how (softened vs inverted)? Runs the same prompt N times per arm and
scores every generation, so the output is a rate per arm rather than one anecdote.

Arms are compared on the SAME mid, one pair at a time:
    turbo3   vs turbo3.8
    turbo4   vs turbo4.8

Every arm runs with the same everything else (model, ctx, MTP on/off, sampling),
so the only difference is the V schedule.

Usage (from the fork root):
    python3 tests/lib/task068_e2e_arm_matrix.py \
        --model /path/to/target.gguf --bin /path/to/llama-cli \
        --prompt /path/to/prompt.txt --runs 4 --n-predict 400 \
        --pairs turbo3:turbo3.8,turbo4:turbo4.8 \
        --out-dir /tmp/e2e-arms

MTP arms add --spec-type mtp --model-draft. Pass --mtp to enable.

Exit codes: 0 all arms scored, 1 a generation failed, 2 bad input.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Final

_SCRIPT_DIR: Final[Path] = Path(__file__).resolve().parent
_RUBRIC: Final[Path] = _SCRIPT_DIR / "task068_distortion_rubric.py"


def _load_rubric():
    spec = importlib.util.spec_from_file_location("task068_rubric", _RUBRIC)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {_RUBRIC}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _parse_pairs(spec: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" not in chunk:
            raise SystemExit(f"bad --pairs entry '{chunk}' (want uniform:bracket)")
        a, b = chunk.split(":", 1)
        pairs.append((a.strip(), b.strip()))
    if not pairs:
        raise SystemExit("--pairs is empty")
    return pairs


def run_arm(
    bin_path: Path,
    model: Path,
    prompt_file: Path,
    arm: str,
    runs: int,
    n_predict: int,
    ctx: int,
    ngl: int,
    batch: int,
    ubatch: int,
    ctk: str,
    mtp: bool,
    model_draft: Path | None,
    seed: int,
    out_dir: Path,
) -> list[Path]:
    """Generate `runs` summaries for one arm; return the text files."""
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    prompt = prompt_file.read_text(encoding="utf-8")
    produced: list[Path] = []

    for i in range(runs):
        out_txt = arm_dir / f"run{i:02d}.txt"
        if out_txt.exists() and out_txt.stat().st_size > 0:
            print(f"  [{arm} {i}] cached")
            produced.append(out_txt)
            continue

        cmd = [
            str(bin_path),
            "-m", str(model),
            "-c", str(ctx),
            "-ngl", str(ngl),
            "-b", str(batch),
            "-ub", str(ubatch),
            "-ctk", ctk,
            "-ctv", arm,
            "-n", str(n_predict),
            "-no-cnv",
            "--seed", str(seed + i),
            "-fit", "off",
            # Thinking is forbidden in tests: it consumes the whole budget and the
            # answer never closes. Same as the product path, which does not request it.
            "--reasoning", "off",
            "-p", prompt,
        ]
        if mtp:
            if model_draft is None:
                raise SystemExit("--mtp needs --model-draft")
            cmd += ["-md", str(model_draft), "--spec-type", "mtp"]

        log = arm_dir / f"run{i:02d}.log"
        t0 = time.time()
        # Bytes, not text: llama-cli can cut a multi-byte char at the n_predict
        # boundary, and text=True would raise mid-pipeline.
        proc = subprocess.run(cmd, capture_output=True, timeout=7200)
        dt = time.time() - t0
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        log.write_text(stdout + stderr, encoding="utf-8", errors="replace")

        if proc.returncode != 0:
            print(f"  [{arm} {i}] FAILED rc={proc.returncode} see {log}", file=sys.stderr)
            continue

        text = _extract_generation(stdout, prompt[:80])
        out_txt.write_text(text, encoding="utf-8")
        print(f"  [{arm} {i}] {dt:.1f}s  {len(text)} chars -> {out_txt.name}")
        produced.append(out_txt)

    return produced


def _extract_generation(stdout: str, prompt_marker: str = "") -> str:
    """Strip llama-cli chrome and the echoed prompt; keep the generated body.

    With reasoning disabled the body follows the echoed prompt directly, so the
    echo must be removed or the scorer would match the prompt's own words.
    """
    lines = stdout.splitlines()
    body: list[str] = []
    started = False
    for line in lines:
        stripped = line.strip()
        if not started:
            # The echoed prompt starts with "> "; the answer follows it.
            if stripped.startswith(">"):
                started = True
                continue
            continue
        if stripped.startswith("[ Prompt:") or stripped.startswith("Exiting"):
            break
        # Drop the echoed prompt itself (single long line) and the CLI's elision.
        if prompt_marker and prompt_marker[:40] in line:
            continue
        if stripped.endswith("(truncated)") and len(stripped) > 100:
            continue
        body.append(line)
    out = "\n".join(body).strip()
    return out if out else stdout.strip()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="TASK-068 e2e arm matrix")
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--bin", required=True, type=Path, dest="bin_path")
    ap.add_argument("--prompt", required=True, type=Path, help="prompt file (fixture .prompt.txt)")
    ap.add_argument("--pairs", required=True, help='e.g. "turbo3:turbo3.8,turbo4:turbo4.8"')
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--n-predict", type=int, default=400)
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--ctk", default="q8_0")
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--model-draft", type=Path)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)

    for p, name in ((args.model, "--model"), (args.bin_path, "--bin"), (args.prompt, "--prompt")):
        if not p.exists():
            print(f"FAIL: {name} not found: {p}", file=sys.stderr)
            return 2
    if args.mtp and (args.model_draft is None or not args.model_draft.exists()):
        print("FAIL: --mtp needs an existing --model-draft", file=sys.stderr)
        return 2
    if args.runs < 2:
        print("FAIL: --runs must be >= 2 (one sample cannot estimate a rate)", file=sys.stderr)
        return 2

    rubric = _load_rubric()
    pairs = _parse_pairs(args.pairs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "model": str(args.model),
        "prompt": str(args.prompt),
        "mtp": args.mtp,
        "runs": args.runs,
        "ctx": args.ctx,
        "n_predict": args.n_predict,
        "pairs": [],
    }

    failed = False
    for uniform, bracket in pairs:
        print(f"\n=== pair: {uniform} vs {bracket} ===")
        pair_block: dict[str, Any] = {"uniform": uniform, "bracket": bracket, "arms": {}}

        for arm in (uniform, bracket):
            print(f" arm {arm}")
            files = run_arm(
                args.bin_path, args.model, args.prompt, arm, args.runs, args.n_predict,
                args.ctx, args.ngl, args.batch, args.ubatch, args.ctk,
                args.mtp, args.model_draft, args.seed, args.out_dir,
            )
            if len(files) < args.runs:
                print(f" WARN: {arm} produced {len(files)}/{args.runs} runs", file=sys.stderr)
                failed = True
            rows = [
                rubric.score_text(f"{arm}/{f.name}", f.read_text(encoding="utf-8")).as_dict()
                for f in files
            ]
            agg = rubric._aggregate(rows)
            pair_block["arms"][arm] = {"runs": rows, "aggregate": agg}
            print(
                f"  -> {arm}: mean_score={agg.get('mean_score')} "
                f"correct={agg.get('rate_correct')} softened={agg.get('rate_softened')} "
                f"inverted={agg.get('rate_inverted')} missing={agg.get('rate_missing')}"
            )

        u = pair_block["arms"][uniform]["aggregate"]
        b = pair_block["arms"][bracket]["aggregate"]
        pair_block["delta_mean_score"] = round(b.get("mean_score", 0) - u.get("mean_score", 0), 4)
        pair_block["delta_rate_inverted"] = round(
            b.get("rate_inverted", 0) - u.get("rate_inverted", 0), 4
        )
        summary["pairs"].append(pair_block)

    out_json = args.out_dir / "e2e_arm_matrix.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== summary ===")
    for pb in summary["pairs"]:
        print(
            f"{pb['uniform']} vs {pb['bracket']}: "
            f"d_score={pb['delta_mean_score']:+.4f} d_inverted={pb['delta_rate_inverted']:+.4f}"
        )
    print(f"==> {out_json}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
