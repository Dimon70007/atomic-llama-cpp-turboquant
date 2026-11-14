#!/usr/bin/env python3
"""TASK-068/071 - plan and validate KLD ladder cost before a run starts.

Why this exists: llama-perplexity --save-all-logits writes one uint16 row per
(token, vocab) entry, so the base logits file grows with CTX * CHUNKS only. A
CTX=32768 point with CHUNKS=6 needs ~100 GB, which cannot fit a laptop disk and
takes far longer than the measurement is worth. The ladder therefore plans
chunk counts from a fixed token budget: equal cost per point, equal uncertainty
scale, and the chunk count is stamped for every CTX.

n_vocab is read from the GGUF header (no gguf-py / numpy dependency).

Usage:
  kld_ladder_plan.py plan --model m.gguf --ctx-list "4096 16384 32768" \
      --token-budget 49152 --chunks-floor 6 --out-dir /tmp/stamp [--max-gb 40]

Exit codes: 0 plan fits, 2 plan refused (payload or disk), 1 bad usage.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import sys

GGUF_MAGIC = b"GGUF"
# gguf metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12
SCALAR_FMT = {
    T_UINT8: "<B", T_INT8: "<b", T_UINT16: "<H", T_INT16: "<h",
    T_UINT32: "<I", T_INT32: "<i", T_FLOAT32: "<f", T_BOOL: "<?",
    T_UINT64: "<Q", T_INT64: "<q", T_FLOAT64: "<d",
}


class Reader:
    def __init__(self, fh) -> None:
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise ValueError("truncated GGUF header")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.raw(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.raw(8))[0]

    def string(self) -> str:
        return self.raw(self.u64()).decode("utf-8", "replace")

    def scalar(self, vtype: int):
        return struct.unpack(SCALAR_FMT[vtype], self.raw(struct.calcsize(SCALAR_FMT[vtype])))[0]

    def skip_value(self, vtype: int) -> None:
        """Skip one metadata value; returns nothing. For arrays, returns count via side channel."""
        if vtype in SCALAR_FMT:
            self.raw(struct.calcsize(SCALAR_FMT[vtype]))
        elif vtype == T_STRING:
            self.raw(self.u64())
        elif vtype == T_ARRAY:
            self.skip_array()
        else:
            raise ValueError(f"unknown GGUF value type {vtype}")

    def skip_array(self) -> tuple[int, int]:
        etype = self.u32()
        count = self.u64()
        if etype == T_STRING:
            for _ in range(count):
                self.raw(self.u64())
        elif etype in SCALAR_FMT:
            self.raw(count * struct.calcsize(SCALAR_FMT[etype]))
        else:
            for _ in range(count):
                self.skip_value(etype)
        return etype, count


def read_vocab_size(path: str) -> int:
    """Read <arch>.vocab_size, else the length of tokenizer.ggml.tokens."""
    with open(path, "rb") as fh:
        r = Reader(fh)
        if r.raw(4) != GGUF_MAGIC:
            raise ValueError(f"{path} is not a GGUF file")
        version = r.u32()
        if version < 2:
            raise ValueError(f"unsupported GGUF version {version} (need >= 2)")
        r.u64()  # tensor count
        kv_count = r.u64()

        vocab_scalar = None
        tokens_count = None
        for _ in range(kv_count):
            key = r.string()
            vtype = r.u32()
            if key == "tokenizer.ggml.tokens" and vtype == T_ARRAY:
                _, tokens_count = r.skip_array()
            elif key.endswith(".vocab_size") and vtype in SCALAR_FMT:
                vocab_scalar = int(r.scalar(vtype))
            else:
                r.skip_value(vtype)

    if vocab_scalar:
        return vocab_scalar
    if tokens_count:
        return tokens_count
    raise ValueError(f"{path}: no vocab size metadata found")


# llama-perplexity log_softmax writer: nv = 2*((n_vocab + 1)/2) + 4, uint16 rows.
def bytes_per_token(n_vocab: int) -> int:
    return (2 * ((n_vocab + 1) // 2) + 4) * 2


def chunks_for_ctx(ctx: int, budget: int, floor: int, allow_few: bool) -> int:
    n = max(1, budget // ctx)
    if n < floor and not allow_few:
        n = floor
    return n

def gb(nbytes: int) -> float:
    return nbytes / 1_000_000_000


def cmd_plan(args: argparse.Namespace) -> int:
    try:
        vocab = read_vocab_size(args.model)
    except (OSError, ValueError) as exc:
        print(f"FAIL: cannot read vocab size: {exc}", file=sys.stderr)
        return 1

    bpt = bytes_per_token(vocab)
    free = shutil.disk_usage(args.out_dir).free
    usable = int(free * 0.7)

    print(f"vocab={vocab} bytes_per_token={bpt} free_gb={gb(free):.1f}")
    print("ctx\tchunks\ttokens\tlogits_gb")

    rc = 0
    total = 0
    for ctx in args.ctx_list:
        chunks = chunks_for_ctx(ctx, args.token_budget, args.chunks_floor, args.allow_few_chunks)
        nbytes = ctx * chunks * bpt
        total += nbytes
        print(f"{ctx}\t{chunks}\t{ctx * chunks}\t{gb(nbytes):.1f}")
        if gb(nbytes) > args.max_gb:
            print(f"FAIL: CTX={ctx} chunks={chunks} base logits ~{gb(nbytes):.1f} GB exceeds cap {args.max_gb} GB", file=sys.stderr)
            print("FAIL: lower --token-budget, drop high CTX, or raise --max-gb deliberately", file=sys.stderr)
            rc = 2
        elif nbytes > usable:
            print(f"FAIL: CTX={ctx} chunks={chunks} needs ~{gb(nbytes):.1f} GB, usable={gb(usable):.1f} GB (70% of free)", file=sys.stderr)
            rc = 2

    if rc == 0 and len(args.ctx_list) < 2 and not args.allow_single_ctx:
        print(f"FAIL: ladder needs >= 2 CTX points (got {len(args.ctx_list)})", file=sys.stderr)
        print("FAIL: pass --allow-single-ctx for a single-point KLD check", file=sys.stderr)
        rc = 2
    if rc == 0:
        print(f"total={gb(total):.1f} GB (sequential; one base file at a time)", file=sys.stderr)
    return rc


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Plan KLD ladder cost from a token budget.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="print and validate the per-CTX chunk plan")
    p.add_argument("--model", required=True, help="GGUF path (vocab size source)")
    p.add_argument("--ctx-list", required=True, help='space separated, e.g. "4096 16384 32768"')
    p.add_argument("--token-budget", type=int, required=True)
    p.add_argument("--chunks-floor", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-gb", type=float, default=40.0)
    p.add_argument("--allow-few-chunks", action="store_true")
    p.add_argument(
        "--allow-single-ctx",
        action="store_true",
        help="permit a one-CTX plan (single-point KLD check, not a ladder)",
    )
    p.set_defaults(func=cmd_plan)

    args = ap.parse_args(argv)
    try:
        args.ctx_list = [int(t) for t in args.ctx_list.split()]
    except ValueError:
        print(f"FAIL: bad --ctx-list: {args.ctx_list}", file=sys.stderr)
        return 1
    if not args.ctx_list or any(c <= 0 for c in args.ctx_list):
        print(f"FAIL: --ctx-list must be positive integers: {args.ctx_list}", file=sys.stderr)
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
