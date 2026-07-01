import importlib.util
import sys
from pathlib import Path


def load_probe():
    path = Path("scripts/probe_vllm_sampling_semantics.py")
    spec = importlib.util.spec_from_file_location("probe_vllm_sampling_semantics", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_cases_cover_next_epilogue_boundaries():
    probe = load_probe()
    cases = {case.name: case for case in probe.build_probe_cases()}

    assert "greedy_no_penalty" in cases
    assert "greedy_default_repetition" in cases
    assert "sample_topk_topp" in cases
    assert "sample_topk_topp_penalty" in cases
    assert "greedy_token_logprobs" in cases

    assert cases["greedy_no_penalty"].sampling["temperature"] == 0.0
    assert cases["greedy_default_repetition"].sampling["repetition_penalty"] > 1.0
    assert cases["sample_topk_topp"].sampling["temperature"] > 0.0
    assert cases["sample_topk_topp"].sampling["top_k"] == 50
    assert cases["sample_topk_topp_penalty"].sampling["presence_penalty"] > 0.0
    assert cases["greedy_token_logprobs"].sampling["logprobs"] == 5


def test_probe_summarize_reports_distribution():
    probe = load_probe()

    result = probe.summarize([3.0, 1.0, 2.0])

    assert result["min"] == 1.0
    assert result["median"] == 2.0
    assert result["mean"] == 2.0
    assert result["max"] == 3.0
