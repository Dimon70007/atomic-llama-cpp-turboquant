"""Unit tests for TASK-069 CUDA suite knobs loader (ST-069.4 profile families)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from task069_suite_knobs import FAMILY_LOCKS, load_suite_config

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "task069"
_KNOBS_PY = Path(__file__).resolve().parent / "task069_suite_knobs.py"


def test_primary_mtp_is_metal_parity() -> None:
    knobs = load_suite_config(_FIXTURES / "cuda-mtp.reference.yaml")
    assert knobs["TASK069_SUITE_ARM"] == "mtp"
    assert knobs["TASK069_SUITE_FAMILY"] == "metal_parity"
    assert knobs["TASK069_CTV"] == "turbo4"
    assert knobs["TASK069_BATCH"] == "2048"
    assert knobs["TASK069_UBATCH"] == "512"
    assert knobs["TASK069_CTX"] == "3072"
    assert knobs["TASK069_CRAM"] == "4096"
    assert knobs["TASK069_CTXCP"] == "32"
    assert knobs["TASK069_THREADS"] == "4"
    assert knobs["TASK069_THREADS_BATCH"] == "4"
    assert knobs["TASK069_TENSOR_SPLIT"] == "1.05,0.95"
    assert knobs["TASK069_MAIN_GPU"] == "1"


def test_primary_nomtp_shares_host_with_mtp() -> None:
    mtp = load_suite_config(_FIXTURES / "cuda-mtp.reference.yaml")
    nomtp = load_suite_config(_FIXTURES / "cuda-nomtp.reference.yaml")
    assert nomtp["TASK069_SUITE_ARM"] == "nomtp"
    assert nomtp["TASK069_SUITE_FAMILY"] == "metal_parity"
    for key in (
        "TASK069_BATCH",
        "TASK069_UBATCH",
        "TASK069_CTV",
        "TASK069_CTX",
        "TASK069_CRAM",
        "TASK069_CTXCP",
        "TASK069_THREADS",
        "TASK069_TENSOR_SPLIT",
        "TASK069_MAIN_GPU",
    ):
        assert nomtp[key] == mtp[key], key


def test_v100_product_archive_locks() -> None:
    knobs = load_suite_config(_FIXTURES / "cuda-mtp-v100-product.reference.yaml")
    assert knobs["TASK069_SUITE_FAMILY"] == "v100_product"
    want_qv, want_b, want_u = FAMILY_LOCKS["v100_product"]
    assert knobs["TASK069_CTV"] == want_qv
    assert knobs["TASK069_BATCH"] == want_b
    assert knobs["TASK069_UBATCH"] == want_u
    assert knobs["TASK069_CTX"] == "8192"


def test_wrong_quant_for_family_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    src = (_FIXTURES / "cuda-mtp.reference.yaml").read_text(encoding="utf-8")
    bad.write_text(src.replace("quant_v: turbo4", "quant_v: turbo3"), encoding="utf-8")
    with pytest.raises(SystemExit, match="quant_v must be turbo4"):
        load_suite_config(bad)


def test_cli_print_json_primary() -> None:
    proc = subprocess.run(
        [sys.executable, str(_KNOBS_PY), "--config", str(_FIXTURES / "cuda-mtp.reference.yaml"), "--print-json"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert '"TASK069_CTV": "turbo4"' in proc.stdout
    assert '"TASK069_SUITE_FAMILY": "metal_parity"' in proc.stdout
