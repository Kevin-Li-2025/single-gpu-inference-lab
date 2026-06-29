import importlib.util
import json

import numpy as np

from test_l20_logits_boundary_trace import (
    ModelRunner,
    Tensor,
    InputBatch,
    V2InputBatch,
    ParallelConfig,
    SchedulerOutput,
)


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_flashsampling_shadow_plan_falls_back_for_current_topk_topp(monkeypatch):
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")
    module = load_module(
        "integrations/vllm/l20_flashsampling_epilogue.py",
        "l20_flashsampling_epilogue_topk",
    )

    plan = module.plan_l20_flashsampling_epilogue(
        ModelRunner(),
        InputBatch(),
        None,
        Tensor((2, 2048)),
        Tensor((2, 151936)),
    )

    assert plan["boundary_eligible"] is True
    assert plan["would_use_epilogue"] is False
    assert "top_k_top_p_unsupported" in plan["fallback_reasons"]
    assert plan["avoidable_logits_materialization_bytes"] == 0


def test_flashsampling_shadow_plan_accepts_full_vocab_gumbel_v2(monkeypatch):
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")
    module = load_module(
        "integrations/vllm/l20_flashsampling_epilogue.py",
        "l20_flashsampling_epilogue_full_vocab",
    )
    batch = V2InputBatch()
    batch.top_k_cpu = np.array([-1, -1, 999], dtype=np.int32)
    batch.top_p_cpu = np.array([1.0, 1.0, 999.0], dtype=np.float32)

    plan = module.plan_l20_flashsampling_epilogue(
        type("V2ModelRunner", (), {"parallel_config": ParallelConfig()})(),
        batch,
        None,
        Tensor((2, 2048)),
        Tensor((2, 151936)),
        SchedulerOutput(),
    )

    assert plan["would_use_epilogue"] is True
    assert plan["fallback_reasons"] == []
    assert plan["policy"]["block_vocab"] == 64
    assert plan["policy"]["block_hidden"] == 128
    assert plan["avoidable_logits_materialization_bytes"] == 2 * 151936 * 2


def test_flashsampling_shadow_trace_writes_jsonl_event(tmp_path, monkeypatch):
    trace = tmp_path / "flashsampling.jsonl"
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")
    monkeypatch.setenv("VLLM_L20_FLASHSAMPLING_TRACE", str(trace))
    module = load_module(
        "integrations/vllm/l20_flashsampling_epilogue.py",
        "l20_flashsampling_epilogue_trace",
    )
    module._TRACE_COUNT = 0
    batch = V2InputBatch()
    batch.top_k_cpu = np.array([-1, -1, 999], dtype=np.int32)
    batch.top_p_cpu = np.array([1.0, 1.0, 999.0], dtype=np.float32)

    module.maybe_trace_l20_flashsampling_epilogue(
        type("V2ModelRunner", (), {"parallel_config": ParallelConfig()})(),
        batch,
        None,
        Tensor((2, 2048)),
        Tensor((2, 151936)),
        SchedulerOutput(),
    )

    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["event"] == "l20_flashsampling_epilogue_gate"
    assert event["eligible"] is True
    plan = event["metadata"]["flashsampling_epilogue"]
    assert plan["would_use_epilogue"] is True
    assert plan["mutates_outputs"] is False
    assert plan["policy"]["block_hidden"] == 128
