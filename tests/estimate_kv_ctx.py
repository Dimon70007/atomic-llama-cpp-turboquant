#!/usr/bin/env python3
"""Fail-closed KV / n_ctx estimator for this TurboQuant fork (TASK-066.10).

Mirrors llama.cpp allocation math under ISWA KEEP (066.5): size_swa does not
depend on n_ubatch; when ctx >= n_swa, size_swa == size_base.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# type_size / blck_size from ggml-common.h + ggml.c type_traits (TURBO4_USE_4BIT=1)
_TYPE_LAYOUT: dict[str, tuple[int, int]] = {
    "f16": (2, 1),
    "bf16": (2, 1),
    "f32": (4, 1),
    "q8_0": (34, 32),
    "q4_0": (18, 32),
    "turbo2": (34, 128),
    "turbo3": (50, 128),
    "turbo4": (66, 128),
}

_CTK_TYPES = frozenset(_TYPE_LAYOUT.keys())
_CTV_TYPES = _CTK_TYPES | frozenset({"turbo3.8", "turbo4.8"})
_TURBO_TYPES = frozenset({"turbo2", "turbo3", "turbo4"})


def die(msg: str, code: int = 2) -> None:
    print(f"estimate_kv_ctx: {msg}", file=sys.stderr)
    raise SystemExit(code)


def parse_bool(raw: str, name: str) -> bool:
    v = raw.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    die(f"{name} must be true|false (got {raw!r})")


def ggml_pad(n: int, multiple: int = 256) -> int:
    if multiple <= 0:
        die(f"invalid pad multiple {multiple}")
    return ((n + multiple - 1) // multiple) * multiple


def bytes_per_elem(type_name: str) -> float:
    key = type_name.strip().lower()
    if key not in _TYPE_LAYOUT:
        die(f"unsupported type {type_name!r}")
    type_size, blck = _TYPE_LAYOUT[key]
    return type_size / blck


def row_size_bytes(type_name: str, n_embd: int) -> int:
    key = type_name.strip().lower()
    if key not in _TYPE_LAYOUT:
        die(f"unsupported type {type_name!r}")
    type_size, blck = _TYPE_LAYOUT[key]
    if n_embd % blck != 0:
        die(f"n_embd={n_embd} not divisible by blck_size={blck} for {key}")
    return type_size * n_embd // blck


def calc_size_swa(
    size_base: int,
    n_swa: int,
    n_seq_max: int,
    unified: bool,
    swa_full: bool,
    n_ubatch: int = 0,
) -> int:
    """Python port of llama_kv_cache_iswa_calc_size_swa (KEEP).

    n_ubatch is accepted for API parity and must not affect the result.
    """
    _ = n_ubatch
    if swa_full:
        return size_base
    swa_window = n_swa * (n_seq_max if unified else 1)
    return ggml_pad(min(size_base, swa_window + size_base), 256)


def pad128(n: int) -> int:
    return ((n + 127) // 128) * 128


def embd_gqa_eff(n_head_kv: int, head_dim: int, type_name: str) -> int:
    key = type_name.strip().lower()
    if key in _TURBO_TYPES and head_dim % 128 != 0:
        return n_head_kv * pad128(head_dim)
    return n_head_kv * head_dim


def resolve_v_type(ctv: str, layer_idx: int, n_layer: int, head: int | None, tail: int | None) -> str:
    key = ctv.strip().lower()
    if key == "turbo3.8":
        if head is None or tail is None:
            die("--head and --tail required when --ctv is turbo3.8")
        if layer_idx < head or layer_idx >= n_layer - tail:
            return "q8_0"
        return "turbo3"
    if key == "turbo4.8":
        if head is None or tail is None:
            die("--head and --tail required when --ctv is turbo4.8")
        if layer_idx < head or layer_idx >= n_layer - tail:
            return "q8_0"
        return "turbo4"
    if key not in _TYPE_LAYOUT:
        die(f"unsupported --ctv {ctv!r}")
    return key


def _field_parts_values(field: Any) -> list[Any]:
    out: list[Any] = []
    for idx in field.data:
        part = field.parts[idx]
        if hasattr(part, "tolist"):
            v = part.tolist()
            if isinstance(v, list):
                if len(v) == 1:
                    out.append(v[0])
                else:
                    out.extend(v)
            else:
                out.append(v)
        else:
            out.append(part)
    return out


def _decode_string(values: Sequence[Any]) -> str:
    if len(values) == 1 and isinstance(values[0], (bytes, bytearray)):
        return values[0].decode("utf-8")
    if values and all(isinstance(x, int) for x in values):
        return bytes(int(x) for x in values).decode("utf-8")
    if len(values) == 1 and isinstance(values[0], str):
        return values[0]
    die(f"cannot decode string field from {values!r}")


def _is_string_field(field: Any) -> bool:
    if not field.types:
        return False
    t0 = field.types[0]
    name = getattr(t0, "name", str(t0))
    return str(name).endswith("STRING")


def read_gguf_scalar(fields: Mapping[str, Any], key: str, required: bool = True) -> Any | None:
    field = fields.get(key)
    if field is None:
        if required:
            die(f"missing GGUF key {key!r}")
        return None
    vals = _field_parts_values(field)
    if not vals:
        if required:
            die(f"empty GGUF key {key!r}")
        return None
    if _is_string_field(field):
        return _decode_string(vals)
    # ARRAY of BOOL / numbers: return list if multi
    if len(vals) == 1:
        return vals[0]
    return vals


def _as_per_layer_int(raw: Any, n_layer: int, name: str) -> list[int]:
    if isinstance(raw, list):
        if len(raw) != n_layer:
            die(f"{name} length {len(raw)} != block_count {n_layer}")
        return [int(x) for x in raw]
    if isinstance(raw, (int, float)):
        return [int(raw)] * n_layer
    die(f"{name}: expected int or list of length n_layer, got {type(raw)!r}")


def load_model_meta(model_path: Path) -> dict[str, Any]:
    try:
        from gguf import GGUFReader
    except ImportError as exc:
        die(
            "gguf package required (set PYTHONPATH to fork gguf-py and use a venv with numpy). "
            f"ImportError: {exc}"
        )

    reader = GGUFReader(str(model_path))
    fields = reader.fields
    arch = read_gguf_scalar(fields, "general.architecture")
    if not isinstance(arch, str) or not arch.strip():
        die("general.architecture missing or empty")
    prefix = arch.strip()

    def req(suffix: str) -> Any:
        return read_gguf_scalar(fields, f"{prefix}.{suffix}", required=True)

    def opt(suffix: str) -> Any:
        return read_gguf_scalar(fields, f"{prefix}.{suffix}", required=False)

    n_layer = int(req("block_count"))
    n_head_arr = _as_per_layer_int(req("attention.head_count"), n_layer, "attention.head_count")
    n_head_kv_arr = _as_per_layer_int(
        req("attention.head_count_kv"), n_layer, "attention.head_count_kv"
    )
    n_embd = int(req("embedding_length"))

    key_len = opt("attention.key_length")
    val_len = opt("attention.value_length")
    if key_len is None:
        d_k_arr = [n_embd // max(h, 1) for h in n_head_arr]
    else:
        d_k_arr = _as_per_layer_int(key_len, n_layer, "attention.key_length")
    if val_len is None:
        d_v_arr = list(d_k_arr)
    else:
        d_v_arr = _as_per_layer_int(val_len, n_layer, "attention.value_length")

    key_len_swa = opt("attention.key_length_swa")
    val_len_swa = opt("attention.value_length_swa")
    d_k_swa = int(key_len_swa) if key_len_swa is not None else None
    d_v_swa = int(val_len_swa) if val_len_swa is not None else None

    n_swa_raw = opt("attention.sliding_window")
    n_swa = int(n_swa_raw) if n_swa_raw is not None else 0

    pattern_raw = opt("attention.sliding_window_pattern")
    is_swa: list[bool]
    if pattern_raw is None:
        is_swa = [False] * n_layer
    elif isinstance(pattern_raw, list):
        if len(pattern_raw) != n_layer:
            die(
                f"sliding_window_pattern length {len(pattern_raw)} != block_count {n_layer}"
            )
        is_swa = [bool(x) for x in pattern_raw]
    elif isinstance(pattern_raw, (bool, int)):
        # scalar period style is not Gemma4; fail closed for ambiguous scalar
        die(
            "sliding_window_pattern is scalar; expected bool array of length n_layer "
            "(Gemma4). Refusing silent period expansion."
        )
    else:
        die(f"unsupported sliding_window_pattern type {type(pattern_raw)!r}")

    return {
        "arch": prefix,
        "n_layer": n_layer,
        "n_head_arr": n_head_arr,
        "n_head_kv_arr": n_head_kv_arr,
        "n_embd": n_embd,
        "d_k_arr": d_k_arr,
        "d_v_arr": d_v_arr,
        "d_k_swa": d_k_swa,
        "d_v_swa": d_v_swa,
        "n_swa": n_swa,
        "is_swa": is_swa,
        "n_swa_layers": sum(1 for x in is_swa if x),
        "n_full_layers": sum(1 for x in is_swa if not x),
    }


@dataclass(frozen=True)
class EstimateResult:
    model: str
    n_ctx: int
    size_base: int
    size_swa: int
    ctk: str
    ctv: str
    n_seq_max: int
    unified: bool
    swa_full: bool
    n_stream: int
    n_layer: int
    n_swa: int
    n_swa_layers: int
    n_full_layers: int
    bytes_per_token: float
    kv_bytes: int
    kv_mib: float
    n_ctx_max: int | None
    head: int | None
    tail: int | None


def gemma4_26b_a4b_meta() -> dict[str, Any]:
    """Synthetic Gemma-4-26B-A4B geometry (matches UD-IQ4_XS GGUF stamp).

    Period-6 SWA pattern: five SWA then one full; SWA n_head_kv=8 / d=256,
    full n_head_kv=2 / d=512; n_swa=1024; n_layer=30.
    """
    n_layer = 30
    is_swa = [((il % 6) != 5) for il in range(n_layer)]
    n_head_kv_arr = [8 if swa else 2 for swa in is_swa]
    return {
        "arch": "gemma4",
        "n_layer": n_layer,
        "n_head_arr": [16] * n_layer,
        "n_head_kv_arr": n_head_kv_arr,
        "n_embd": 2816,
        "d_k_arr": [512] * n_layer,
        "d_v_arr": [512] * n_layer,
        "d_k_swa": 256,
        "d_v_swa": 256,
        "n_swa": 1024,
        "is_swa": is_swa,
        "n_swa_layers": sum(1 for x in is_swa if x),
        "n_full_layers": sum(1 for x in is_swa if not x),
    }


def compute_kv_from_meta(
    meta: Mapping[str, Any],
    *,
    n_ctx: int,
    ctk: str,
    ctv: str,
    n_seq_max: int,
    unified: bool,
    swa_full: bool,
    head: int | None = None,
    tail: int | None = None,
    kv_budget_mib: float | None = None,
    model_label: str = "<meta>",
) -> EstimateResult:
    """Compute KV estimate from an already-loaded / synthetic meta dict."""
    if n_ctx <= 0:
        die("--n-ctx must be > 0")
    if n_seq_max <= 0:
        die("--n-seq-max must be > 0")
    ctk_l = ctk.strip().lower()
    ctv_l = ctv.strip().lower()
    if ctk_l not in _CTK_TYPES:
        die(f"--ctk must be one of {sorted(_CTK_TYPES)}")
    if ctv_l not in _CTV_TYPES:
        die(f"--ctv must be one of {sorted(_CTV_TYPES)}")
    if ctv_l in ("turbo3.8", "turbo4.8") and (head is None or tail is None):
        die("--head and --tail are required for bracket --ctv")
    if ctv_l not in ("turbo3.8", "turbo4.8") and (head is not None or tail is not None):
        die("--head/--tail only valid with --ctv turbo3.8|turbo4.8")

    size_base = ggml_pad(n_ctx, 256)
    size_swa = calc_size_swa(
        size_base,
        int(meta["n_swa"]),
        n_seq_max,
        unified,
        swa_full,
        n_ubatch=0,
    )
    n_stream = 1 if unified else n_seq_max
    n_layer = int(meta["n_layer"])
    n_head_kv_arr: list[int] = list(meta["n_head_kv_arr"])
    d_k_arr: list[int] = list(meta["d_k_arr"])
    d_v_arr: list[int] = list(meta["d_v_arr"])
    d_k_swa = meta["d_k_swa"]
    d_v_swa = meta["d_v_swa"]

    total = 0
    for il in range(n_layer):
        is_swa = bool(meta["is_swa"][il])
        n_cells = size_swa if is_swa else size_base
        n_head_kv = int(n_head_kv_arr[il])
        if is_swa and d_k_swa is not None:
            d_k = int(d_k_swa)
        else:
            d_k = int(d_k_arr[il])
        if is_swa and d_v_swa is not None:
            d_v = int(d_v_swa)
        else:
            d_v = int(d_v_arr[il])
        type_k = ctk_l
        type_v = resolve_v_type(ctv_l, il, n_layer, head, tail)
        n_k = embd_gqa_eff(n_head_kv, d_k, type_k)
        n_v = embd_gqa_eff(n_head_kv, d_v, type_v)
        layer_bytes = (
            row_size_bytes(type_k, n_k) + row_size_bytes(type_v, n_v)
        ) * n_cells * n_stream
        total += layer_bytes

    # Under KEEP with ctx>=n_swa, size_swa==size_base: bytes scale linearly with n_ctx.
    bytes_per_token = total / float(size_base) if size_base else float("nan")
    kv_mib = total / (1024.0 * 1024.0)
    n_ctx_max: int | None = None
    if kv_budget_mib is not None:
        if kv_budget_mib <= 0:
            die("--kv-budget-mib must be > 0 when set")
        budget = kv_budget_mib * 1024.0 * 1024.0
        n_ctx_max = int(math.floor(budget / bytes_per_token)) if bytes_per_token > 0 else 0

    return EstimateResult(
        model=model_label,
        n_ctx=n_ctx,
        size_base=size_base,
        size_swa=size_swa,
        ctk=ctk_l,
        ctv=ctv_l,
        n_seq_max=n_seq_max,
        unified=unified,
        swa_full=swa_full,
        n_stream=n_stream,
        n_layer=n_layer,
        n_swa=int(meta["n_swa"]),
        n_swa_layers=int(meta["n_swa_layers"]),
        n_full_layers=int(meta["n_full_layers"]),
        bytes_per_token=bytes_per_token,
        kv_bytes=total,
        kv_mib=kv_mib,
        n_ctx_max=n_ctx_max,
        head=head,
        tail=tail,
    )


def estimate(
    *,
    model: Path,
    n_ctx: int,
    ctk: str,
    ctv: str,
    n_seq_max: int,
    unified: bool,
    swa_full: bool,
    kv_budget_mib: float | None,
    head: int | None,
    tail: int | None,
) -> EstimateResult:
    meta = load_model_meta(model)
    return compute_kv_from_meta(
        meta,
        n_ctx=n_ctx,
        ctk=ctk,
        ctv=ctv,
        n_seq_max=n_seq_max,
        unified=unified,
        swa_full=swa_full,
        head=head,
        tail=tail,
        kv_budget_mib=kv_budget_mib,
        model_label=str(model),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="TASK-066.10 fail-closed KV / n_ctx estimator (llama.cpp KEEP math)"
    )
    p.add_argument("--model", required=True, help="Path to GGUF (required)")
    p.add_argument("--n-ctx", required=True, type=int, help="Context length (required)")
    p.add_argument("--ctk", required=True, help="Cache type K (required)")
    p.add_argument("--ctv", required=True, help="Cache type V (required)")
    p.add_argument("--n-seq-max", required=True, type=int, help="--parallel / n_seq_max")
    p.add_argument("--unified", required=True, help="true|false (KV unified)")
    p.add_argument("--swa-full", required=True, help="true|false (--swa-full)")
    p.add_argument("--kv-budget-mib", type=float, default=None, help="Optional KV budget")
    p.add_argument("--head", type=int, default=None, help="q8 bracket HEAD (turbo*.8)")
    p.add_argument("--tail", type=int, default=None, help="q8 bracket TAIL (turbo*.8)")
    p.add_argument("--json", action="store_true", help="Print JSON only")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    model = Path(args.model)
    if not model.is_file():
        die(f"model not found: {model}")

    result = estimate(
        model=model,
        n_ctx=int(args.n_ctx),
        ctk=str(args.ctk),
        ctv=str(args.ctv),
        n_seq_max=int(args.n_seq_max),
        unified=parse_bool(str(args.unified), "--unified"),
        swa_full=parse_bool(str(args.swa_full), "--swa-full"),
        kv_budget_mib=args.kv_budget_mib,
        head=args.head,
        tail=args.tail,
    )
    payload = asdict(result)
    print("### estimate_kv_ctx JSON")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json:
        return 0
    print("### summary")
    print(f"size_base={result.size_base} size_swa={result.size_swa} n_stream={result.n_stream}")
    print(
        f"layers={result.n_layer} swa_layers={result.n_swa_layers} "
        f"full_layers={result.n_full_layers} n_swa={result.n_swa}"
    )
    print(f"bytes_per_token={result.bytes_per_token:.6f}")
    print(f"KV_bytes={result.kv_bytes} KV_MiB={result.kv_mib:.4f}")
    if result.n_ctx_max is not None:
        print(f"n_ctx_max={result.n_ctx_max} (KV budget only; weights+compute not included)")
    else:
        print("n_ctx_max=(unset; pass --kv-budget-mib)")
    print(
        "NOTE: estimate is tensor KV only under 066.5 KEEP; "
        "init log still required for host fit."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
