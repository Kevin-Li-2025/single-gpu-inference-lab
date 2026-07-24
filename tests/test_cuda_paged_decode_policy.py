import importlib.util
from pathlib import Path

import pytest


def test_cuda_paged_decode_gate_matches_measured_regimes():
    pytest.importorskip("torch")
    path = Path("scripts/benchmark_cuda_paged_decode.py")
    spec = importlib.util.spec_from_file_location("cuda_bench", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.should_use_l20_cuda_paged_decode(1, 2304)
    assert not module.should_use_l20_cuda_paged_decode(1, 4096)
    assert module.should_use_l20_cuda_paged_decode(4, 640)
    assert not module.should_use_l20_cuda_paged_decode(4, 2048)
    assert not module.should_use_l20_cuda_paged_decode(8, 512)
