import importlib.util
from pathlib import Path


def load_candidate():
    path = Path("integrations/vllm/l20_flashsampling_candidate.py")
    spec = importlib.util.spec_from_file_location("l20_flashsampling_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flashsampling_candidate_defaults_to_batch_one(monkeypatch):
    candidate = load_candidate()
    monkeypatch.delenv(candidate.MAX_BATCH_ENV, raising=False)

    assert candidate._candidate_max_batch() == 1
    assert candidate._candidate_batch_reasons({"batch_size": 1}) == []
    assert candidate._candidate_batch_reasons({"batch_size": 4}) == [
        "candidate_batch_gt_1"
    ]


def test_flashsampling_candidate_max_batch_override(monkeypatch):
    candidate = load_candidate()
    monkeypatch.setenv(candidate.MAX_BATCH_ENV, "4")

    assert candidate._candidate_max_batch() == 4
    assert candidate._candidate_batch_reasons({"batch_size": 4}) == []
    assert candidate._candidate_batch_reasons({"batch_size": 5}) == [
        "candidate_batch_gt_4"
    ]


def test_flashsampling_candidate_max_batch_is_conservative(monkeypatch):
    candidate = load_candidate()

    monkeypatch.setenv(candidate.MAX_BATCH_ENV, "0")
    assert candidate._candidate_max_batch() == 1

    monkeypatch.setenv(candidate.MAX_BATCH_ENV, "not-an-int")
    assert candidate._candidate_max_batch() == 1


def test_flashsampling_campaign_forwards_candidate_max_batch():
    source = Path("scripts/run_vllm_l20_flashsampling_trace_campaign.sh").read_text()

    assert "VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH" in source
    assert 'candidate_env+=(VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH="' in source
