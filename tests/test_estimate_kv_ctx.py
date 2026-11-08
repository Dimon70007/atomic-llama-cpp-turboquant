#!/usr/bin/env python3
"""Unit tests for TASK-066.10 estimate_kv_ctx (no GPU / no heavy model required)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
FORK_ROOT = TESTS_DIR.parent
sys.path.insert(0, str(TESTS_DIR))

from estimate_kv_ctx import (  # noqa: E402
    bytes_per_elem,
    calc_size_swa,
    compute_kv_from_meta,
    gemma4_26b_a4b_meta,
    ggml_pad,
    resolve_v_type,
)

# Product/test ctx lock: all sizing cases use N_CTX=100000 (PAD -> 100096).
_STAMP_N_CTX = 100000
_STAMP_SIZE_BASE = ggml_pad(_STAMP_N_CTX)  # 100096
# bytes/tok independent of n_ctx under KEEP (size_swa == size_base).
_STAMP_T3_BYTES_PER_TOKEN = 81840.0
_STAMP_T4_BYTES_PER_TOKEN = 88880.0
_STAMP_T38_BYTES_PER_TOKEN = 94224.0
_STAMP_T48_BYTES_PER_TOKEN = 98960.0
# Frozen totals at N_CTX=100000 / size_base=100096 (HEAD=2 TAIL=8 for *.8).
_STAMP_T38_KV_BYTES = 9431445504
_STAMP_T38_KV_MIB = 8994.52734375
_STAMP_T48_KV_BYTES = 9905500160
_STAMP_T48_KV_MIB = 9446.62109375
_STAMP_T3_KV_BYTES = 8191856640
_STAMP_T4_KV_BYTES = 8896532480


def test_bytes_per_elem_q8_and_f16() -> None:
    assert bytes_per_elem("f16") == 2.0
    assert bytes_per_elem("q8_0") == 1.0625
    assert bytes_per_elem("q8_0") != 1.0
    # vs f16: 17/32, not 32/17
    assert abs(bytes_per_elem("q8_0") / bytes_per_elem("f16") - 17 / 32) < 1e-12


def test_calc_size_swa_ubatch_invariant_keep() -> None:
    size_base = _STAMP_SIZE_BASE
    n_swa = 1024
    for ub in (8, 1024, 2048):
        got = calc_size_swa(
            size_base, n_swa, n_seq_max=1, unified=True, swa_full=False, n_ubatch=ub
        )
        assert got == size_base


def test_calc_size_swa_long_ctx_side_effect() -> None:
    size_base = _STAMP_SIZE_BASE
    got = calc_size_swa(
        size_base, 1024, n_seq_max=1, unified=True, swa_full=False, n_ubatch=512
    )
    assert got == size_base


def test_calc_size_swa_swa_full_and_clamp() -> None:
    assert (
        calc_size_swa(
            _STAMP_SIZE_BASE, 1024, 1, unified=True, swa_full=True, n_ubatch=8
        )
        == _STAMP_SIZE_BASE
    )
    # Edge: size_base < n_swa still clamps to size_base (then pad).
    assert calc_size_swa(512, 1024, 1, unified=True, swa_full=False, n_ubatch=8) == 512


def test_bracket_schedule_turbo4_8() -> None:
    n_layer = 30
    head = 2
    tail = 8
    types = [resolve_v_type("turbo4.8", il, n_layer, head, tail) for il in range(n_layer)]
    assert types[:2] == ["q8_0", "q8_0"]
    assert types[-8:] == ["q8_0"] * 8
    assert all(t == "turbo4" for t in types[2 : n_layer - 8])


def test_bracket_schedule_turbo3_8() -> None:
    n_layer = 30
    types = [
        resolve_v_type("turbo3.8", il, n_layer, head=2, tail=8) for il in range(n_layer)
    ]
    assert types[:2] == ["q8_0", "q8_0"]
    assert types[-8:] == ["q8_0"] * 8
    assert all(t == "turbo3" for t in types[2 : n_layer - 8])


def test_gemma4_meta_iswa_pattern() -> None:
    meta = gemma4_26b_a4b_meta()
    assert meta["n_layer"] == 30
    assert meta["n_swa"] == 1024
    assert meta["n_swa_layers"] == 25
    assert meta["n_full_layers"] == 5
    assert meta["is_swa"][5] is False
    assert meta["is_swa"][0] is True
    assert meta["n_head_kv_arr"][0] == 8
    assert meta["n_head_kv_arr"][5] == 2


def test_q8_turbo3_8_kv_size_stamp() -> None:
    """q8_0 + turbo3.8 (HEAD=2 TAIL=8) at N_CTX=100000."""
    meta = gemma4_26b_a4b_meta()
    got = compute_kv_from_meta(
        meta,
        n_ctx=_STAMP_N_CTX,
        ctk="q8_0",
        ctv="turbo3.8",
        n_seq_max=1,
        unified=True,
        swa_full=False,
        head=2,
        tail=8,
    )
    assert got.size_base == _STAMP_SIZE_BASE
    assert got.size_swa == _STAMP_SIZE_BASE  # KEEP
    assert got.bytes_per_token == _STAMP_T38_BYTES_PER_TOKEN
    assert got.kv_bytes == _STAMP_T38_KV_BYTES
    assert abs(got.kv_mib - _STAMP_T38_KV_MIB) < 1e-9
    uniform = compute_kv_from_meta(
        meta,
        n_ctx=_STAMP_N_CTX,
        ctk="q8_0",
        ctv="turbo3",
        n_seq_max=1,
        unified=True,
        swa_full=False,
    )
    assert uniform.bytes_per_token == _STAMP_T3_BYTES_PER_TOKEN
    assert uniform.kv_bytes == _STAMP_T3_KV_BYTES
    assert got.kv_bytes > uniform.kv_bytes


def test_q8_turbo4_8_kv_size_stamp() -> None:
    """q8_0 + turbo4.8 (HEAD=2 TAIL=8) at N_CTX=100000."""
    meta = gemma4_26b_a4b_meta()
    got = compute_kv_from_meta(
        meta,
        n_ctx=_STAMP_N_CTX,
        ctk="q8_0",
        ctv="turbo4.8",
        n_seq_max=1,
        unified=True,
        swa_full=False,
        head=2,
        tail=8,
    )
    assert got.size_base == _STAMP_SIZE_BASE
    assert got.size_swa == _STAMP_SIZE_BASE
    assert got.bytes_per_token == _STAMP_T48_BYTES_PER_TOKEN
    assert got.kv_bytes == _STAMP_T48_KV_BYTES
    assert abs(got.kv_mib - _STAMP_T48_KV_MIB) < 1e-9
    uniform = compute_kv_from_meta(
        meta,
        n_ctx=_STAMP_N_CTX,
        ctk="q8_0",
        ctv="turbo4",
        n_seq_max=1,
        unified=True,
        swa_full=False,
    )
    assert uniform.bytes_per_token == _STAMP_T4_BYTES_PER_TOKEN
    assert uniform.kv_bytes == _STAMP_T4_KV_BYTES
    assert got.kv_bytes > uniform.kv_bytes


def test_bracket_ctv_requires_head_tail() -> None:
    meta = gemma4_26b_a4b_meta()
    try:
        compute_kv_from_meta(
            meta,
            n_ctx=_STAMP_N_CTX,
            ctk="q8_0",
            ctv="turbo3.8",
            n_seq_max=1,
            unified=True,
            swa_full=False,
        )
        raise AssertionError("expected SystemExit for missing HEAD/TAIL")
    except SystemExit as exc:
        assert exc.code != 0


def test_cli_missing_ctk_fails() -> None:
    script = TESTS_DIR / "estimate_kv_ctx.py"
    gguf_py = FORK_ROOT / "gguf-py"
    env = {"PYTHONPATH": str(gguf_py)}
    # Use product .venv if present for numpy
    venv_py = FORK_ROOT.parent / ".venv" / "bin" / "python"
    py = str(venv_py) if venv_py.is_file() else sys.executable
    proc = subprocess.run(
        [
            py,
            str(script),
            "--model",
            "/nonexistent.gguf",
            "--n-ctx",
            str(_STAMP_N_CTX),
            "--ctv",
            "turbo4",
            "--n-seq-max",
            "1",
            "--unified",
            "true",
            "--swa-full",
            "false",
        ],
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), **env},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    # argparse complains about missing --ctk
    assert "ctk" in (proc.stderr + proc.stdout).lower()


def test_shell_missing_model_fails() -> None:
    script = TESTS_DIR / "run-estimate-kv-ctx.sh"
    proc = subprocess.run(
        ["bash", str(script)],
        env={
            **{k: v for k, v in __import__("os").environ.items()},
            # leave MODEL unset; set others so we hit MODEL check path clearly
            "N_CTX": str(_STAMP_N_CTX),
            "CTK": "q8_0",
            "CTV": "turbo4",
            "N_SEQ_MAX": "1",
            "UNIFIED": "true",
            "SWA_FULL": "false",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "MODEL" in (proc.stderr + proc.stdout)
