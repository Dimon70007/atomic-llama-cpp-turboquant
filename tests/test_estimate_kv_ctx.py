#!/usr/bin/env python3
"""Unit tests for TASK-066.10 estimate_kv_ctx (n_ubatch headroom + pad128)."""

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

# Product: headroom = n_ubatch=512 → size_swa = PAD128(1024+512)=1536.
# N_CTX=100000 (PAD -> 100096).
_STAMP_N_CTX = 100000
_STAMP_SIZE_BASE = ggml_pad(_STAMP_N_CTX)  # 100096
_STAMP_N_BATCH = 2048  # telemetry only
_STAMP_N_UBATCH = 512
_STAMP_SIZE_SWA = 1536

_STAMP_F16_KV_BYTES = 2364538880
_STAMP_F16_KV_MIB = 2255.0
_STAMP_Q8_KV_BYTES = 1256161280
_STAMP_Q8_KV_MIB = 1197.96875
_STAMP_T3_KV_BYTES = 858992640
_STAMP_T3_KV_MIB = 819.19921875
_STAMP_T4_KV_BYTES = 932884480
_STAMP_T4_KV_MIB = 889.66796875


def test_bytes_per_elem_q8_and_f16() -> None:
    assert bytes_per_elem("f16") == 2.0
    assert bytes_per_elem("q8_0") == 1.0625
    assert bytes_per_elem("q8_0") != 1.0
    assert abs(bytes_per_elem("q8_0") / bytes_per_elem("f16") - 17 / 32) < 1e-12


def test_calc_size_swa_ideas_headroom_windowed() -> None:
    size_base = _STAMP_SIZE_BASE
    n_swa = 1024
    # Product outer_pad=256 (matches C++ test-iswa-size-swa).
    assert calc_size_swa(size_base, n_swa, 1, True, False, n_swa_headroom=8) == 1280
    assert calc_size_swa(size_base, n_swa, 1, True, False, n_swa_headroom=512) == 1536
    assert calc_size_swa(size_base, n_swa, 1, True, False, n_swa_headroom=1344) == 2560
    assert calc_size_swa(size_base, n_swa, 1, True, False, n_swa_headroom=2048) == 3072


def test_calc_size_swa_long_ctx_stays_windowed() -> None:
    got = calc_size_swa(
        _STAMP_SIZE_BASE,
        1024,
        n_seq_max=1,
        unified=True,
        swa_full=False,
        n_swa_headroom=_STAMP_N_UBATCH,
    )
    assert got == 1536
    assert got < _STAMP_SIZE_BASE


def test_calc_size_swa_pad256_and_swa_full() -> None:
    assert (
        calc_size_swa(
            _STAMP_SIZE_BASE, 1024, 1, unified=True, swa_full=True, n_swa_headroom=8
        )
        == _STAMP_SIZE_BASE
    )
    assert calc_size_swa(512, 1024, 1, unified=True, swa_full=False, n_swa_headroom=8) == 512
    assert calc_size_swa(3072, 1000, 1, unified=True, swa_full=False, n_swa_headroom=8) == 1280


def test_calc_block_pad_h_swa1() -> None:
    from estimate_kv_ctx import calc_block_pad

    assert calc_block_pad(512) == 512
    assert calc_block_pad(2048) == 2048
    assert calc_block_pad(1000) == 1024


def test_size_swa_scales_with_n_ubatch() -> None:
    """Product headroom = n_ubatch → different -ub ⇒ different size_swa."""
    meta = gemma4_26b_a4b_meta()
    sizes = []
    for ub in (8, 256, 512, 1024):
        got = compute_kv_from_meta(
            meta,
            n_ctx=_STAMP_N_CTX,
            ctk="q8_0",
            ctv="turbo4",
            n_seq_max=1,
            unified=True,
            swa_full=False,
            n_batch=_STAMP_N_BATCH,
            n_ubatch=ub,
        )
        sizes.append(got.size_swa)
        assert got.n_swa_headroom == ub
        assert got.n_ubatch == ub
    assert sizes == [1280, 1280, 1536, 2048]


def test_resolve_v_type_uniform_only() -> None:
    assert resolve_v_type("turbo3") == "turbo3"
    assert resolve_v_type("turbo4") == "turbo4"
    assert resolve_v_type("q8_0") == "q8_0"
    assert resolve_v_type("f16") == "f16"
    try:
        resolve_v_type("not-a-kv-type")
        raise AssertionError("expected SystemExit for unknown CTV")
    except SystemExit as exc:
        assert exc.code != 0


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


def _stamp(
    ctk: str, ctv: str
) -> object:
    return compute_kv_from_meta(
        gemma4_26b_a4b_meta(),
        n_ctx=_STAMP_N_CTX,
        ctk=ctk,
        ctv=ctv,
        n_seq_max=1,
        unified=True,
        swa_full=False,
        n_batch=_STAMP_N_BATCH,
        n_ubatch=_STAMP_N_UBATCH,
    )


def test_f16_f16_kv_size_stamp() -> None:
    got = _stamp("f16", "f16")
    assert got.size_base == _STAMP_SIZE_BASE
    assert got.size_swa == _STAMP_SIZE_SWA
    assert got.kv_bytes == _STAMP_F16_KV_BYTES
    assert abs(got.kv_mib - _STAMP_F16_KV_MIB) < 1e-9


def test_q8_q8_kv_size_stamp() -> None:
    got = _stamp("q8_0", "q8_0")
    assert got.size_swa == _STAMP_SIZE_SWA
    assert got.kv_bytes == _STAMP_Q8_KV_BYTES
    assert abs(got.kv_mib - _STAMP_Q8_KV_MIB) < 1e-9


def test_q8_turbo3_kv_size_stamp() -> None:
    got = _stamp("q8_0", "turbo3")
    assert got.size_swa == _STAMP_SIZE_SWA
    assert got.kv_bytes == _STAMP_T3_KV_BYTES
    assert abs(got.kv_mib - _STAMP_T3_KV_MIB) < 1e-9


def test_q8_turbo4_kv_size_stamp() -> None:
    got = _stamp("q8_0", "turbo4")
    assert got.size_swa == _STAMP_SIZE_SWA
    assert got.kv_bytes == _STAMP_T4_KV_BYTES
    assert abs(got.kv_mib - _STAMP_T4_KV_MIB) < 1e-9
    assert got.kv_bytes > _stamp("q8_0", "turbo3").kv_bytes


def test_cli_missing_ctk_fails() -> None:
    script = TESTS_DIR / "estimate_kv_ctx.py"
    gguf_py = FORK_ROOT / "gguf-py"
    env = {"PYTHONPATH": str(gguf_py)}
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
            "--n-batch",
            str(_STAMP_N_BATCH),
            "--n-ubatch",
            str(_STAMP_N_UBATCH),
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
    assert "ctk" in (proc.stderr + proc.stdout).lower()


def test_shell_missing_model_fails() -> None:
    script = TESTS_DIR / "run-estimate-kv-ctx.sh"
    proc = subprocess.run(
        ["bash", str(script)],
        env={
            **{k: v for k, v in __import__("os").environ.items()},
            "N_CTX": str(_STAMP_N_CTX),
            "CTK": "q8_0",
            "CTV": "turbo4",
            "N_SEQ_MAX": "1",
            "N_BATCH": str(_STAMP_N_BATCH),
            "N_UBATCH": str(_STAMP_N_UBATCH),
            "UNIFIED": "true",
            "SWA_FULL": "false",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "model" in (proc.stderr + proc.stdout).lower()
