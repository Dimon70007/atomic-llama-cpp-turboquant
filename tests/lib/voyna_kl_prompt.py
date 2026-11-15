#!/usr/bin/env python3
"""Build a real-prose KL prompt from the Voyna i Mir corpus (TASK-068 10.28 F/E2).

Why this exists: the old prompt was a tiled ASCII sentence (the same 76 bytes
repeated until the file was big enough). On repeated filler the model is almost
certain of the next token, so every KV arm lands at KLD ~1e-6 and the numbers
rank the arms but say nothing about real drift. Perplexity only means something
on natural text.

This helper cuts one contiguous, paragraph-aligned slice from the corpus, then
tokenizes it with llama-tokenize and widens or trims it until the token count
clears the perplexity floor (2 * n_ctx). No repetition, no synthetic text.

Usage:
  voyna_kl_prompt.py --model m.gguf --tokenize-bin ./bin/llama-tokenize \
      --min-tokens 8192 --out prompt.txt [--corpus path] [--anchor 0.227]

Writes the prompt and a ``.meta.json`` sidecar. Exit 2 when the corpus, the
tokenizer binary or the model is missing (caller maps that to skip 77).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

# 22.7% of the corpus is where the Pierre/Dolokhov duel starts, which is the
# same episode the TASK-068 distortion rubric scores. A slice anchored there
# keeps the KL run and the fact rubric on the same text.
_DEFAULT_ANCHOR: Final[float] = 0.227


def default_corpus(repo_root: Path) -> Path:
    """Corpus path when this helper lives in the fork submodule."""
    return repo_root.parent / "assets" / "literature" / "voyna_i_mir_ru.txt"


def token_count(tokenize_bin: Path, model: Path, text: str) -> int:
    """Run llama-tokenize on ``text`` and return the token count."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as fh:
        fh.write(text)
        tmp = Path(fh.name)
    try:
        proc = subprocess.run(
            [str(tokenize_bin), "-m", str(model), "-f", str(tmp), "--ids"],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise SystemExit(
            f"fail-closed: llama-tokenize exited {proc.returncode}: "
            f"{proc.stderr.strip().splitlines()[-1:] or 'no stderr'}"
        )
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("[") and line.endswith("]"):
            ids = ast.literal_eval(line)
            return len(ids)
    raise SystemExit("fail-closed: llama-tokenize printed no id list")


def paragraph_slice(text: str, anchor: float, chars: int) -> str:
    """Return ~``chars`` characters from ``anchor``, starting and ending on a blank line."""
    start = int(len(text) * anchor)
    nl = text.rfind("\n\n", 0, start)
    if nl >= 0:
        start = nl + 2
    end = min(len(text), start + chars)
    nl = text.find("\n\n", end)
    if nl >= 0:
        end = nl
    return text[start:end]


def fit_slice(text: str, anchor: float, target_tokens: int, counter) -> tuple[str, int]:
    """Return the smallest paragraph-aligned slice from ``anchor`` holding ``target_tokens``.

    The requirement is a floor (perplexity refuses a prompt under 2 * n_ctx), so
    the search takes the first length that clears it. Overshoot is fine; falling
    short is not. Binary search beats a chars-per-token estimate because Russian
    prose varies enough that one guess lands 20% off.
    """
    cache: dict[int, tuple[str, int]] = {}

    def count(chars: int) -> tuple[str, int]:
        if chars not in cache:
            candidate = paragraph_slice(text, anchor, chars)
            cache[chars] = (candidate, counter(candidate))
        return cache[chars]

    lo = 1024
    hi = max(2048, int(target_tokens * 4.0))  # generous upper bound, 4 chars/token
    best: tuple[str, int] | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate, n = count(mid)
        if n >= target_tokens:
            if best is None or n < best[1]:
                best = (candidate, n)
            hi = mid - 1
        else:
            lo = mid + 1
    if best is None:
        raise SystemExit(
            f"fail-closed: no slice of the corpus reaches {target_tokens} tokens"
        )
    return best


def main(argv: list[str] | None = None) -> int:
    """CLI entry: build the prompt file and its sidecar."""
    parser = argparse.ArgumentParser(description="Real-prose Voyna KL prompt builder.")
    parser.add_argument("--model", type=Path, required=True, help="GGUF for tokenization")
    parser.add_argument("--tokenize-bin", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--anchor", type=float, default=_DEFAULT_ANCHOR)
    args = parser.parse_args(argv)

    if args.min_tokens < 1:
        print("fail-closed: --min-tokens must be positive", file=sys.stderr)
        return 2
    if not args.model.is_file():
        print(f"host-skip: missing model {args.model}", file=sys.stderr)
        return 2
    if not args.tokenize_bin.is_file():
        print(f"host-skip: missing tokenizer {args.tokenize_bin}", file=sys.stderr)
        return 2

    corpus = args.corpus or default_corpus(Path(__file__).resolve().parents[2])
    if not corpus.is_file():
        print(f"host-skip: missing corpus {corpus}", file=sys.stderr)
        return 2
    text = corpus.read_text(encoding="utf-8")
    if not text.strip():
        print(f"fail-closed: corpus is empty: {corpus}", file=sys.stderr)
        return 2

    target = int(args.min_tokens)
    slice_text, tokens = fit_slice(
        text,
        args.anchor,
        target,
        lambda t: token_count(args.tokenize_bin, args.model, t),
    )
    # Slices end on a paragraph break, so the count lands at or just above the
    # floor; report the real number rather than pretending it is exact.
    if tokens < args.min_tokens:
        print(
            f"fail-closed: prompt has {tokens} tokens, need >= {args.min_tokens}",
            file=sys.stderr,
        )
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(slice_text, encoding="utf-8")
    meta = {
        "corpus": str(corpus),
        "corpus_chars": len(text),
        "anchor": args.anchor,
        "chars": len(slice_text),
        "tokens": tokens,
        "min_tokens": args.min_tokens,
        "sha256": hashlib.sha256(slice_text.encode("utf-8")).hexdigest(),
        "first_line": slice_text.splitlines()[0][:120] if slice_text else "",
    }
    meta_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"prompt: {args.out} tokens={tokens} chars={len(slice_text)} "
        f"anchor={args.anchor} corpus={corpus.name}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
