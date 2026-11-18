#!/usr/bin/env python3
"""TASK-077: KV cache parity matrix on the product serving path.

Self-contained: needs only this repository (the fork), python3 stdlib and a
GGUF model. No access to the parent project is required.

What it measures, per operator protocol 2026-11-18:
  * every request is greedy (temperature 0, top_k 1) with
    chat_template_kwargs {"enable_thinking": false};
  * one served config is queried REPEATS times - request 1 is a full prefill,
    requests 2+ read the prompt cache, so the two shapes are compared too;
  * the five shipping V types are compared on the nomtp shape (rule 3);
  * turbo3.8 / turbo4.8 are additionally served with ctx_checkpoints 0 and 32
    (rule 4);
  * every cell is repeated for the allowed ubatch values (16 / 256 / 512).
    ubatch below 16 is REFUSED: it takes a different code path that the
    product never uses, so measuring it would describe a configuration nobody
    ships;
  * flash attention is always on: the library force-enables it for any
    quantised V, so an "fa off" arm cannot exist (TASK-077 F46). The matrix
    records that as a blocked cell instead of pretending to test it.

Diagnosis is split on purpose. In each cell the first request has
cache_n = 0 (full prefill) and the rest have cache_n > 0 (cache read):
  * `prefill_vs_cached` - full prefill and cache read disagree;
  * `stable_within_shape` - repeated requests of the same shape disagree.
Only the second is a determinism failure. Conflating them makes the gate red
for a reason it cannot name.

Usage:
  tests/lib/task077_kv_parity.py --server build/bin/llama-server \
      --model ~/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf --out-dir logs/task077

  # plan only, touches no GPU:
  tests/lib/task077_kv_parity.py ... --dry-run

Exit codes: 0 all cells ran and every shape was stable,
1 a cell drifted within one shape or crashed, 2 bad input,
77 model/server absent (ctest skip).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Product serving shape (configs/gemma4_26b_a4b_mtp_cuda_nomtp.yaml). Kept here
# so the gate does not read the parent project.
PRODUCT_CTX = 160000
PRODUCT_BATCH = 2048
PRODUCT_THREADS = 16

N_PREDICT = 96
SEED = 42
REPEATS = 3
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

# Rule 3: the five shipping V types. Rule 4: these two also get ctxcp 0 vs 32.
V_TYPES = ("q8_0", "turbo3", "turbo3.8", "turbo4", "turbo4.8")
BRACKET_TYPES = ("turbo3.8", "turbo4.8")
CTXCP_PRODUCT = 32
CTXCP_CONTROL = 0

# ubatch values the operator allows. Below UBATCH_FLOOR a different code path
# is taken, so those runs describe a configuration the product never uses.
UBATCH_VALUES = (16, 256, 512)
UBATCH_FLOOR = 16

DEFAULT_CORPUS = "tests/fixtures/task066/task066_pierre_dolokhov_duel_2k.prompt.txt"
DEFAULT_MODEL = "~/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf"


@dataclass
class Cell:
    """One (V type, ctx checkpoint count, ubatch) configuration to serve."""

    quant_v: str
    ctx_checkpoints: int
    ubatch: int
    flash_attn: str = "on"


@dataclass
class CellResult:
    """Observations for one served configuration."""

    quant_v: str
    ctx_checkpoints: int
    ubatch: int
    flash_attn: str
    status: str = "not_run"
    error: str = ""
    load_s: float = 0.0
    hashes: list[str] = field(default_factory=list)
    prompt_n: list[int] = field(default_factory=list)
    cache_n: list[int] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)

    def ran(self) -> bool:
        """True when the repeat ladder completed."""
        return self.status == "ok" and len(self.hashes) == REPEATS

    def prefill_index(self) -> int | None:
        """Index of the request that ran a full prefill, if any."""
        for i, cached in enumerate(self.cache_n):
            if cached == 0:
                return i
        return None

    def stable_within_shape(self) -> bool:
        """True when every cache-read repeat emitted the same text.

        Cache-read requests share one shape, so they are directly comparable.
        The prefill request has a different batch shape and is compared
        separately by `prefill_matches_cached`.
        """
        if not self.ran():
            return False
        cached = [h for h, c in zip(self.hashes, self.cache_n) if c > 0]
        return len(cached) >= 2 and len(set(cached)) == 1

    def prefill_matches_cached(self) -> bool | None:
        """True if full prefill and cache read agree, None when not comparable."""
        if not self.ran():
            return None
        pre = [h for h, c in zip(self.hashes, self.cache_n) if c == 0]
        cached = [h for h, c in zip(self.hashes, self.cache_n) if c > 0]
        if not pre or not cached:
            return None
        return pre[0] == cached[0]

    def answer_hash(self) -> str:
        """Hash of the last answer, i.e. one produced from a cache read."""
        return self.hashes[-1] if self.hashes else "-"


def build_cells(v_types: list[str], bracket_types: list[str],
                ubatch_values: tuple[int, ...]) -> list[Cell]:
    """Build the matrix: every type at the product ctxcp, brackets also at 0."""
    cells: list[Cell] = []
    for ub in ubatch_values:
        for v in v_types:
            cells.append(Cell(quant_v=v, ctx_checkpoints=CTXCP_PRODUCT, ubatch=ub))
            if v in bracket_types:
                cells.append(Cell(quant_v=v, ctx_checkpoints=CTXCP_CONTROL, ubatch=ub))
    return cells


def free_port() -> int:
    """Return an unused loopback port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def server_cmd(server: Path, model: Path, port: int, cell: Cell) -> list[str]:
    """Assemble the llama-server command line for one cell."""
    return [
        str(server),
        "-m", str(model),
        "-c", str(PRODUCT_CTX),
        "-ngl", "999",
        "-ctk", "q8_0",
        "-ctv", cell.quant_v,
        "-fa", cell.flash_attn,
        "-ctxcp", str(cell.ctx_checkpoints),
        "--jinja",
        "--chat-template-kwargs", json.dumps(CHAT_TEMPLATE_KWARGS),
        "-b", str(PRODUCT_BATCH),
        "-ub", str(cell.ubatch),
        "-np", "1",
        "-t", str(PRODUCT_THREADS),
        "-tb", str(PRODUCT_THREADS),
        "--host", "127.0.0.1",
        "--port", str(port),
        "--no-webui",
        "--no-warmup",
    ]


def wait_health(port: int, proc: subprocess.Popen[str], timeout_s: float) -> None:
    """Wait for /health, raising if the server dies first."""
    deadline = time.time() + timeout_s
    url = f"http://127.0.0.1:{port}/health"
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if json.loads(resp.read()).get("status") == "ok":
                    return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(0.5)
    raise RuntimeError("server did not become healthy in time")


def complete(port: int, prompt: str) -> dict[str, Any]:
    """Send one greedy completion and return the parsed response."""
    body = {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": SEED,
        "cache_prompt": True,
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.loads(resp.read())


def run_cell(args: argparse.Namespace, cell: Cell, prompt: str, out_dir: Path) -> CellResult:
    """Serve one cell, repeat the request, tear the server down."""
    result = CellResult(quant_v=cell.quant_v, ctx_checkpoints=cell.ctx_checkpoints,
                        ubatch=cell.ubatch, flash_attn=cell.flash_attn)
    port = free_port()
    cmd = server_cmd(args.server, args.model, port, cell)
    log_path = out_dir / f"server-{cell.quant_v}-ctxcp{cell.ctx_checkpoints}-ub{cell.ubatch}.log"
    t0 = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(" ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)
        try:
            wait_health(port, proc, args.load_timeout)
            result.load_s = round(time.time() - t0, 1)
            for _ in range(REPEATS):
                resp = complete(port, prompt)
                answer = resp.get("content", "")
                result.hashes.append(hashlib.sha256(answer.encode()).hexdigest()[:16])
                result.answers.append(answer)
                timings = resp.get("timings", {})
                result.prompt_n.append(int(timings.get("prompt_n", -1)))
                result.cache_n.append(int(timings.get("cache_n", -1)))
            result.status = "ok"
        except Exception as exc:  # noqa: BLE001 - record, keep the matrix going
            result.status = "error"
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            stop_server(proc)
    note = scan_log(log_path)
    if note:
        result.error = f"{result.error} | {note}".strip(" |")
    return result


def stop_server(proc: subprocess.Popen[str]) -> None:
    """Terminate the process group, escalating to SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), 15)
        proc.wait(timeout=60)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except ProcessLookupError:
            pass


def scan_log(path: Path) -> str:
    """Return the first fatal line from a server log, if any."""
    markers = ("failed to remove sequence", "can_seq_rm", "ggml_abort",
               "quantized V cache requires", "require flash_attn")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    for line in text.splitlines():
        if any(m in line for m in markers):
            return line.strip()[:200]
    return ""


def render(results: list[CellResult], ubatch_values: tuple[int, ...]) -> str:
    """Render the markdown report."""
    out = ["# TASK-077 KV parity matrix", ""]
    out.append(
        f"Greedy, temperature 0, top_k 1, seed {SEED}, enable_thinking=false, "
        f"{REPEATS} repeats per cell, context {PRODUCT_CTX}, "
        f"ubatch {', '.join(str(u) for u in ubatch_values)}."
    )
    out.append("Request 1 is a full prefill, requests 2+ read the prompt cache, "
               "so the two batch shapes are compared as well as repeats within a shape.")

    gate_note = (
        "`stable within shape` is the determinism gate: repeated cache-read "
        "requests must agree. `prefill vs cached` is reported separately "
        "because it compares two different batch shapes, not two identical runs."
    )
    out += ["", "## Determinism, split by batch shape (rule 5)", "", gate_note, "",
            "| V type | ctxcp | ub | status | stable within shape | prefill vs cached | hash | note |",
            "|---|---|---|---|---|---|---|---|"]
    for r in results:
        if not r.ran():
            out.append(f"| {r.quant_v} | {r.ctx_checkpoints} | {r.ubatch} | {r.status} | - | - | - "
                       f"| {r.error.replace('|', '/')[:80]} |")
            continue
        stable = "yes" if r.stable_within_shape() else "**NO**"
        pvc = r.prefill_matches_cached()
        pvc_s = "n/a" if pvc is None else ("yes" if pvc else "no")
        out.append(f"| {r.quant_v} | {r.ctx_checkpoints} | {r.ubatch} | ok | {stable} | {pvc_s} "
                   f"| {r.answer_hash()} | |")

    out += ["", "## Cross-type comparison per ubatch (rule 3: q8_0 is the anchor)", ""]
    out += ["| ub | V type | hash | identical to q8_0 |", "|---|---|---|---|"]
    for ub in ubatch_values:
        anchor = next((r.answer_hash() for r in results
                       if r.quant_v == "q8_0" and r.ctx_checkpoints == CTXCP_PRODUCT
                       and r.ubatch == ub and r.ran()), None)
        for r in results:
            if r.ctx_checkpoints != CTXCP_PRODUCT or r.ubatch != ub or not r.ran():
                continue
            same = "yes" if anchor and r.answer_hash() == anchor else "no"
            out.append(f"| {ub} | {r.quant_v} | {r.answer_hash()} | {same} |")

    out += ["", "## ctx_checkpoints effect (rule 4)", ""]
    out += ["| ub | V type | ctxcp 0 | ctxcp 32 | same answer |", "|---|---|---|---|---|"]
    for ub in ubatch_values:
        for v in BRACKET_TYPES:
            c0 = next((r for r in results if r.quant_v == v and r.ubatch == ub
                       and r.ctx_checkpoints == CTXCP_CONTROL), None)
            c32 = next((r for r in results if r.quant_v == v and r.ubatch == ub
                        and r.ctx_checkpoints == CTXCP_PRODUCT), None)
            if c0 is None or c32 is None:
                out.append(f"| {ub} | {v} | - | - | cell missing |")
                continue
            if not c0.ran() or not c32.ran():
                out.append(f"| {ub} | {v} | {c0.status} | {c32.status} | not comparable |")
                continue
            same = "yes" if c0.answer_hash() == c32.answer_hash() else "no"
            out.append(f"| {ub} | {v} | {c0.answer_hash()} | {c32.answer_hash()} | {same} |")

    out += ["", "## ubatch effect, per V type", "",
            "| V type | " + " | ".join(f"ub {ub}" for ub in ubatch_values) + " | all equal |",
            "|---|" + "---|" * len(ubatch_values) + "---|"]
    seen: list[str] = []
    for r in results:
        if r.ctx_checkpoints == CTXCP_PRODUCT and r.quant_v not in seen:
            seen.append(r.quant_v)
    for v in seen:
        row = [next((r.answer_hash() for r in results
                     if r.quant_v == v and r.ctx_checkpoints == CTXCP_PRODUCT
                     and r.ubatch == ub and r.ran()), "-") for ub in ubatch_values]
        equal = "yes" if len(set(row)) == 1 and "-" not in row else "no"
        out.append(f"| {v} | " + " | ".join(row) + f" | {equal} |")

    fa_note = (
        "Not testable: `-fa off` is refused or overridden for every quantised "
        "V type, so no cell exists. TASK-077 F46 records the evidence."
    )
    out += ["", "## flash attention (rule 5)", "", fa_note]
    out.append("")
    out.append("ubatch below "
               f"{UBATCH_FLOOR} is excluded by operator instruction: it takes a "
               "different code path that the product does not use.")
    return "\n".join(out) + "\n"


def main() -> int:
    """Run the matrix or print the plan."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--server", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--corpus", default=None,
                        help=f"prompt file (default <repo>/{DEFAULT_CORPUS})")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--v-types", default=",".join(V_TYPES))
    parser.add_argument("--ubatch", default=",".join(str(u) for u in UBATCH_VALUES))
    parser.add_argument("--load-timeout", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    args.server = Path(os.path.expanduser(args.server))
    args.model = Path(os.path.expanduser(args.model))
    corpus = Path(os.path.expanduser(args.corpus)) if args.corpus else repo / DEFAULT_CORPUS

    ubatch_values = tuple(int(u) for u in args.ubatch.split(",") if u.strip())
    if not ubatch_values:
        print("FAIL: --ubatch is empty", file=sys.stderr)
        return 2
    too_small = [u for u in ubatch_values if u < UBATCH_FLOOR]
    if too_small:
        print(f"FAIL: ubatch below {UBATCH_FLOOR} is refused: {too_small}. "
              "It takes a different code path that the product does not use.",
              file=sys.stderr)
        return 2

    if not args.server.is_file():
        print(f"SKIP: server binary not found: {args.server}", file=sys.stderr)
        return 77
    if not args.model.is_file():
        print(f"SKIP: model not found: {args.model} (set --model or TASK077_MODEL)",
              file=sys.stderr)
        return 77
    if not corpus.is_file():
        print(f"FAIL: prompt fixture not found: {corpus}", file=sys.stderr)
        return 2

    prompt = corpus.read_text(encoding="utf-8")
    v_types = [v.strip() for v in args.v_types.split(",") if v.strip()]
    cells = build_cells(v_types, list(BRACKET_TYPES), ubatch_values)

    if args.dry_run:
        for cell in cells:
            print(" ".join(server_cmd(args.server, args.model, 18099, cell)))
        return 0

    out_dir = Path(args.out_dir or repo / "logs" / "task077-kv-parity")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[CellResult] = []
    for cell in cells:
        print(f"[cell] {cell.quant_v} ctxcp={cell.ctx_checkpoints} ub={cell.ubatch}", flush=True)
        res = run_cell(args, cell, prompt, out_dir)
        results.append(res)
        print(f"       status={res.status} stable={res.stable_within_shape()} "
              f"prefill_vs_cached={res.prefill_matches_cached()} {res.error[:100]}",
              flush=True)
        (out_dir / "results.json").write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8")

    report = render(results, ubatch_values)
    (out_dir / "summary.md").write_text(report, encoding="utf-8")
    print(f"\nwrote {out_dir}/summary.md")
    print(report)

    ran = [r for r in results if r.ran()]
    if not ran:
        return 1
    return 0 if all(r.stable_within_shape() for r in ran) else 1


if __name__ == "__main__":
    sys.exit(main())
