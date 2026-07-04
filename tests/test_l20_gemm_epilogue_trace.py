import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_helper():
    path = Path("integrations/vllm/l20_gemm_epilogue_trace.py")
    spec = importlib.util.spec_from_file_location("l20_gemm_epilogue_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Device:
    type = "cuda"


class Tensor:
    shape = (1, 1024)
    dtype = "torch.float16"
    device = Device()


class WeightTensor:
    shape = (151_936, 1024)
    dtype = "torch.float16"
    device = Device()


class InputBatch:
    num_reqs = 1
    num_draft_tokens = 0
    has_structured_output_reqs = False
    temperature_cpu = np.array([0.8], dtype=np.float32)
    top_k_cpu = np.array([20], dtype=np.int32)
    top_p_cpu = np.array([0.95], dtype=np.float32)
    min_p_cpu = np.array([0.0], dtype=np.float32)
    frequency_penalties_cpu = np.array([0.0], dtype=np.float32)
    presence_penalties_cpu = np.array([0.0], dtype=np.float32)
    repetition_penalties_cpu = np.array([1.0], dtype=np.float32)
    logits_processing_needs_token_ids = np.array([False])
    num_logprobs = {}
    logprob_token_ids = {}
    has_allowed_token_ids = set()
    bad_words_token_ids = {}
    generators = {}


class SparsePenaltyInputBatch(InputBatch):
    temperature_cpu = np.array([0.8], dtype=np.float32)
    top_k_cpu = np.array([50], dtype=np.int32)
    top_p_cpu = np.array([0.9], dtype=np.float32)
    frequency_penalties_cpu = np.array([0.1], dtype=np.float32)
    presence_penalties_cpu = np.array([0.2], dtype=np.float32)
    repetition_penalties_cpu = np.array([1.1], dtype=np.float32)
    token_ids_cpu = np.arange(128, dtype=np.int64).reshape(1, 128)
    num_tokens_no_spec = np.array([128], dtype=np.int32)


class MissingHistorySparsePenaltyInputBatch(SparsePenaltyInputBatch):
    token_ids_cpu = None
    num_tokens_no_spec = None


class GreedyInputBatch(InputBatch):
    all_greedy = True
    no_top_p = True
    no_top_k = True
    temperature_cpu = np.array([-1.0], dtype=np.float32)
    top_k_cpu = np.array([32_000], dtype=np.int32)
    top_p_cpu = np.array([1.0], dtype=np.float32)


class SchedulerOutput:
    num_scheduled_tokens = {"req0": 1}
    total_num_scheduled_tokens = 1


class ParallelConfig:
    tensor_parallel_size = 1


class LogitsProcessor:
    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def try_sample_from_lm_head(
        self,
        lm_head,
        hidden_states,
        sampling_metadata,
        embedding_bias=None,
    ):
        self.calls.append((lm_head, hidden_states, sampling_metadata, embedding_bias))
        return self.output


class LmHead:
    bias = "bias"


class WeightedLmHead(LmHead):
    weight = WeightTensor()


class Model:
    def __init__(self, logits_processor):
        self.logits_processor = logits_processor
        self.lm_head = LmHead()


class Runner:
    parallel_config = ParallelConfig()

    def __init__(self, logits_processor):
        self.model = Model(logits_processor)


def read_event(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gemm_epilogue_trace_calls_fallback_first_api(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    logits_processor = LogitsProcessor()
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(logits_processor),
        InputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    assert len(logits_processor.calls) == 1
    event = read_event(trace)
    assert event["event"] == "l20_gemm_epilogue_boundary"
    assert event["eligible"] is True
    assert event["reasons"] == []
    assert event["metadata"]["phase"] == "fallback_first_api_trace"
    assert event["metadata"]["api"]["try_api_found"] is True
    assert event["metadata"]["api"]["api_called"] is True
    assert event["metadata"]["api"]["output_enabled"] is False
    assert event["metadata"]["api"]["fallback_to_compute_logits"] is True
    assert event["metadata"]["mutates_outputs"] is False


def test_gemm_epilogue_trace_rejects_unsupported_semantics(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    batch = InputBatch()
    batch.num_logprobs = {"req0": 1}
    batch.frequency_penalties_cpu = np.array([0.1], dtype=np.float32)
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor()),
        batch,
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    event = read_event(trace)
    assert event["eligible"] is False
    assert "token_logprobs" in event["reasons"]
    assert event["metadata"]["api"]["api_called"] is False
    assert event["metadata"]["semantic_candidate"]["target"] == "unsupported_semantics"
    assert "token_logprobs" in event["metadata"]["semantic_candidate"]["reasons"]
    assert "sparse_penalties" in event["metadata"]["semantic_candidate"]["features"]


def test_gemm_epilogue_trace_marks_sparse_penalty_p0_candidate(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    logits_processor = LogitsProcessor()
    runner = Runner(logits_processor)
    runner.model.lm_head = WeightedLmHead()
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        runner,
        SparsePenaltyInputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    event = read_event(trace)
    semantic = event["metadata"]["semantic_candidate"]
    assert event["eligible"] is True
    assert event["metadata"]["api"]["api_called"] is True
    assert semantic["target"] == "fused_topk_topp_sparse_penalty_lm_head_epilogue"
    assert semantic["priority"] == "p0"
    assert semantic["eligible"] is True
    assert semantic["features"] == ["sparse_penalties", "topk_topp"]
    assert semantic["estimated_logits_bytes_fp32"] == 1 * 151_936 * 4
    assert round(semantic["estimated_logits_mib_fp32"], 3) == 0.58
    assert semantic["history"]["source"] == "input_batch_token_ids_cpu"


def test_gemm_epilogue_trace_requires_history_for_sparse_penalty_candidate(
    tmp_path, monkeypatch
):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    logits_processor = LogitsProcessor()
    runner = Runner(logits_processor)
    runner.model.lm_head = WeightedLmHead()
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        runner,
        MissingHistorySparsePenaltyInputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    event = read_event(trace)
    semantic = event["metadata"]["semantic_candidate"]
    assert event["eligible"] is True
    assert semantic["target"] == "fused_topk_topp_sparse_penalty_lm_head_epilogue"
    assert semantic["eligible"] is False
    assert semantic["reasons"] == ["missing_sparse_history"]
    assert semantic["history"]["available"] is False


def test_gemm_epilogue_enable_can_surface_non_none_output(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    output = object()
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_ENABLE", "1")
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor(output=output)),
        GreedyInputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is output
    event = read_event(trace)
    assert event["metadata"]["api"]["api_returned_output"] is True
    assert event["metadata"]["api"]["output_enabled"] is True
    assert event["metadata"]["api"]["fallback_to_compute_logits"] is False
    assert event["metadata"]["mutates_outputs"] is True


def test_gemm_epilogue_enable_can_surface_greedy_candidate_output(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    output = object()

    def fake_candidate(*args, **kwargs):
        return output, None, {
            "attempted": True,
            "mode": "greedy_argmax",
            "returned_output": True,
            "fallback_to_compute_logits": False,
        }

    monkeypatch.setattr(module, "_try_lm_head_greedy_sampler_output", fake_candidate)
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_ENABLE", "1")
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor(output=None)),
        GreedyInputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is output
    event = read_event(trace)
    assert event["eligible"] is True
    assert event["metadata"]["api"]["api_called"] is True
    assert event["metadata"]["api"]["api_returned_output"] is False
    assert event["metadata"]["api"]["fallback_to_compute_logits"] is False
    assert event["metadata"]["epilogue"]["attempted"] is True
    assert event["metadata"]["epilogue"]["returned_output"] is True
    assert event["metadata"]["mutates_outputs"] is True


def test_gemm_epilogue_enable_rejects_non_greedy_candidate(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_ENABLE", "1")
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor(output=None)),
        InputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    event = read_event(trace)
    assert event["eligible"] is False
    assert "non_greedy_temperature" in event["reasons"]
    assert "top_k" in event["reasons"]
    assert "top_p" in event["reasons"]
    assert event["metadata"]["api"]["api_called"] is False
    assert event["metadata"]["epilogue"]["attempted"] is False


def test_gemm_epilogue_argmax_correctness_check_matches_torch_baseline():
    torch = pytest.importorskip("torch")
    module = load_helper()
    hidden = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float16)
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, -1.0, 0.0],
            [-1.0, 3.0, 1.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float16,
    )
    sampled = torch.tensor([[1]], dtype=torch.int32)

    details = module._argmax_correctness_check(hidden, weight, sampled, vocab_size=4)

    assert details["checked"] is True
    assert details["matches_baseline_argmax"] is True
    assert details["expected_tokens"] == [1]
    assert details["actual_tokens"] == [1]


def test_gemm_epilogue_argmax_correctness_check_reports_mismatch():
    torch = pytest.importorskip("torch")
    module = load_helper()
    hidden = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float16)
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, -1.0, 0.0],
            [-1.0, 3.0, 1.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=torch.float16,
    )
    sampled = torch.tensor([[3]], dtype=torch.int32)

    details = module._argmax_correctness_check(hidden, weight, sampled, vocab_size=4)

    assert details["checked"] is True
    assert details["matches_baseline_argmax"] is False
    assert details["expected_tokens"] == [1]
    assert details["actual_tokens"] == [3]
