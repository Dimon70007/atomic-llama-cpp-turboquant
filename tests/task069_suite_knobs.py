#!/usr/bin/env python3
"""Load TASK-069 CUDA suite YAML and export fail-closed TASK069_* env knobs.

Usage:
  eval "$(python3 tests/task069_suite_knobs.py --config /path/to.yaml --export-shell)"
  python3 tests/task069_suite_knobs.py --config ... --print-json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

import yaml

# Shared host/KV keys (MTP + nomtp).
HOST_KEYS = (
    "batch",
    "ubatch",
    "quant_k",
    "quant_v",
    "cache_ram",
    "ctx_checkpoints",
    "gpu_layers",
    "parallel",
    "threads",
    "threads_batch",
    "tensor_split",
    "main_gpu",
    "context_size",
    "flash_attn",
    "temperature",
    "top_p",
    "top_k",
)

MTP_KEYS = (
    "spec_draft_n_max",
    "spec_draft_p_min",
)

# profile -> (arm, knob_family)
# metal_parity: ST-069.4 primary fixtures (align with Metal TASK-066 host/KV).
# v100_product: archived LiteLLM dual-V100 product knobs (explicit TASK069_SUITE_CONFIG).
ALLOWED_PROFILES: Final[dict[str, tuple[str, str]]] = {
    "gemma4_26b_a4b_mtp_cuda_e2e": ("mtp", "metal_parity"),
    "gemma4_26b_a4b_mtp_cuda_nomtp_e2e": ("nomtp", "metal_parity"),
    "gemma4_26b_a4b_mtp_cuda_v100_product_e2e": ("mtp", "v100_product"),
    "gemma4_26b_a4b_mtp_cuda_nomtp_v100_product_e2e": ("nomtp", "v100_product"),
    # legacy alias from parent-repo mis-place (still accepted as metal_parity)
    "gemma4_26b_a4b_mtp_cuda_task069_suite": ("mtp", "metal_parity"),
}

# Fail-closed locks per family: quant_v, batch, ubatch.
FAMILY_LOCKS: Final[dict[str, tuple[str, str, str]]] = {
    "metal_parity": ("turbo4", "2048", "512"),
    "v100_product": ("turbo3", "8192", "2048"),
}

DEFAULT_SUITE_CONFIG = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "task069"
    / "cuda-mtp.reference.yaml"
)


def _req(data: dict[str, Any], key: str) -> str:
    if key not in data or data[key] is None or str(data[key]).strip() == "":
        raise SystemExit(f"FAIL: suite config missing required key {key!r} (refusing silent default)")
    return str(data[key]).strip()


def load_suite_config(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise SystemExit(f"FAIL: TASK069_SUITE_CONFIG not a file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"FAIL: suite config must be a mapping: {path}")
    profile = str(raw.get("profile") or "").strip()
    meta = ALLOWED_PROFILES.get(profile)
    if meta is None:
        raise SystemExit(
            "FAIL: unexpected profile "
            f"{profile!r} (want one of {sorted(ALLOWED_PROFILES)}; refusing silent)"
        )
    arm, family = meta
    for key in HOST_KEYS:
        _req(raw, key)
    fa = _req(raw, "flash_attn").lower()
    if fa not in ("on", "1", "true"):
        raise SystemExit(f"FAIL: flash_attn must be on for CUDA suite (got {fa!r})")

    want_qv, want_b, want_u = FAMILY_LOCKS[family]
    qv = _req(raw, "quant_v")
    if qv != want_qv:
        raise SystemExit(
            f"FAIL: quant_v must be {want_qv} for profile family {family!r} "
            f"(profile={profile!r}, got {qv!r})"
        )
    qb = _req(raw, "batch")
    qu = _req(raw, "ubatch")
    if qb != want_b or qu != want_u:
        raise SystemExit(
            f"FAIL: batch/ubatch must be {want_b}/{want_u} for profile family "
            f"{family!r} (profile={profile!r}, got {qb}/{qu})"
        )

    if arm == "mtp":
        for key in MTP_KEYS:
            _req(raw, key)
        spec_n = _req(raw, "spec_draft_n_max")
        spec_p = _req(raw, "spec_draft_p_min")
    else:
        # nomtp arm: still export MTP knobs empty-forbidden for mixed runners that need them
        # from a paired mtp yaml — require explicit TASK066_MODEL_DRAFT at runner level.
        spec_n = str(raw.get("spec_draft_n_max") or "2").strip()
        spec_p = str(raw.get("spec_draft_p_min") or "0.95").strip()

    prompt = str(raw.get("prompt_file") or "").strip()
    out: dict[str, str] = {
        "TASK069_SUITE_PROFILE": "cuda",
        "TASK069_SUITE_CONFIG": str(path.resolve()),
        "TASK069_SUITE_ARM": arm,
        "TASK069_SUITE_FAMILY": family,
        "TASK069_BATCH": qb,
        "TASK069_UBATCH": qu,
        "TASK069_CTK": _req(raw, "quant_k"),
        "TASK069_CTV": qv,
        "TASK069_CRAM": _req(raw, "cache_ram"),
        "TASK069_CTXCP": _req(raw, "ctx_checkpoints"),
        "TASK069_NGL": _req(raw, "gpu_layers"),
        "TASK069_PARALLEL": _req(raw, "parallel"),
        "TASK069_THREADS": _req(raw, "threads"),
        "TASK069_THREADS_BATCH": _req(raw, "threads_batch"),
        "TASK069_TENSOR_SPLIT": _req(raw, "tensor_split"),
        "TASK069_MAIN_GPU": _req(raw, "main_gpu"),
        "TASK069_CTX": _req(raw, "context_size"),
        "TASK069_SPEC_DRAFT_N_MAX": spec_n,
        "TASK069_SPEC_DRAFT_P_MIN": spec_p,
        "TASK069_TEMPERATURE": _req(raw, "temperature"),
        "TASK069_TOP_P": _req(raw, "top_p"),
        "TASK069_TOP_K": _req(raw, "top_k"),
        # Ladder / e2e helpers
        "TASK066_CTX": _req(raw, "context_size"),
        "TASK066_CTK": _req(raw, "quant_k"),
        "TASK066_CTV": qv,
        "TASK066_CRAM": _req(raw, "cache_ram"),
        "TASK066_CTXCP": _req(raw, "ctx_checkpoints"),
        "TASK066_BATCH_MAX": qb,
        "TASK066_BATCH_FLOOR": qu,
        "TASK066_BATCH_MODE": "floor",
        "TASK066_SPEC_DRAFT_N_MAX": spec_n,
        "TASK066_SPEC_DRAFT_P_MIN": spec_p,
    }
    if prompt:
        # Resolve relative to fork root (parent of tests/).
        fork_root = Path(__file__).resolve().parents[1]
        prompt_path = Path(prompt)
        if not prompt_path.is_absolute():
            prompt_path = (fork_root / prompt).resolve()
            if not prompt_path.is_file():
                # also try relative to tests/
                prompt_path = (Path(__file__).resolve().parent / prompt).resolve()
        if not prompt_path.is_file():
            raise SystemExit(f"FAIL: prompt_file not a file: {prompt_path}")
        out["TASK066_PROMPT"] = str(prompt_path)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="TASK-069 CUDA suite knobs loader")
    p.add_argument(
        "--config",
        default=str(DEFAULT_SUITE_CONFIG),
        help="path to tests/fixtures/task069/cuda-mtp.reference.yaml (or nomtp / v100-product)",
    )
    p.add_argument("--export-shell", action="store_true", help="print export KEY=VAL lines")
    p.add_argument("--print-json", action="store_true", help="print knobs JSON")
    p.add_argument("--apply-environ", action="store_true", help="os.environ.update then exec leftover")
    args, rest = p.parse_known_args()
    knobs = load_suite_config(Path(args.config))
    if args.print_json:
        print(json.dumps(knobs, indent=2, sort_keys=True))
        return 0
    if args.export_shell:
        for k, v in sorted(knobs.items()):
            # shell-safe single quotes
            esc = v.replace("'", "'\"'\"'")
            print(f"export {k}='{esc}'")
        return 0
    if args.apply_environ:
        os.environ.update(knobs)
        if not rest:
            raise SystemExit("FAIL: --apply-environ needs a command")
        os.execvp(rest[0], rest)
    print("FAIL: pass --export-shell, --print-json, or --apply-environ CMD", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
