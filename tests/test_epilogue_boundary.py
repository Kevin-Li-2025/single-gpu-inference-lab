import json

from l20_stack.epilogue import (
    SamplerConfig,
    build_boundary_impacts,
    load_logits_boundary_budget,
    plan_sampler_optimization,
    sampler_gate_reasons,
)
from l20_stack.epilogue.logits_boundary import budget_from_summary


def test_logits_boundary_budget_reads_campaign_summary():
    budget = load_logits_boundary_budget(
        "benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1"
    )
    assert budget.total_events == 775
    assert budget.eligible_events == 744
    assert budget.eligible_fraction == 0.96
    assert round(budget.eligible_logits_mib, 2) == 339.93
    assert budget.top_shape == "1x151936"


def test_budget_from_summary_accepts_trace_summary_shape_budget():
    summary = {
        "total_events": 3,
        "eligible_events": 2,
        "fallback_events": 1,
        "eligible_fraction": 2 / 3,
        "eligible_logits_mib": 12.5,
        "total_logits_mib": 18.0,
        "logits_unknown_bytes_events": 0,
        "shape_budget": [
            {"shape": "1x10", "eligible_logits_mib": 8.0},
        ],
    }
    budget = budget_from_summary(summary)
    assert budget.total_events == 3
    assert budget.top_shape == "1x10"
    assert budget.top_shape_eligible_logits_mib == 8.0


def test_sampler_epilogue_gate_accepts_simple_decode_and_rejects_complex_cases():
    assert sampler_gate_reasons(SamplerConfig(temperature=0.8, top_k=50, top_p=0.9)) == []
    reasons = sampler_gate_reasons(
        SamplerConfig(
            min_p=0.1,
            num_logprobs=5,
            has_grammar=True,
            has_penalties=True,
            per_request_generators=True,
            prefill=True,
            tensor_parallel_size=2,
        )
    )
    assert "prefill" in reasons
    assert "tensor_parallel_not_1" in reasons
    assert "grammar_or_structured_output" in reasons
    assert "token_logprobs" in reasons
    assert "min_p" in reasons
    assert "penalties" in reasons
    assert "per_request_generators" in reasons


def test_sampler_optimization_plan_marks_greedy_as_control():
    plan = plan_sampler_optimization(
        SamplerConfig(temperature=0.0, top_k=-1, top_p=1.0)
    )

    assert plan.target == "greedy_no_penalty_control"
    assert plan.priority == "control"
    assert plan.eligible_for_next_prototype is False
    assert plan.expected_itl_delta_vs_greedy_pct == 0.0


def test_sampler_optimization_plan_targets_topk_topp_and_logprobs_first():
    topk = plan_sampler_optimization(
        SamplerConfig(temperature=0.8, top_k=50, top_p=0.9)
    )
    logprobs = plan_sampler_optimization(
        SamplerConfig(temperature=0.0, top_k=-1, top_p=1.0, num_logprobs=5)
    )

    assert topk.target == "fused_topk_topp"
    assert topk.priority == "p0"
    assert topk.eligible_for_next_prototype is True
    assert topk.expected_itl_delta_vs_greedy_pct == 42.0
    assert logprobs.target == "fused_token_logprobs"
    assert logprobs.priority == "p0"
    assert logprobs.eligible_for_next_prototype is True
    assert logprobs.expected_itl_delta_vs_greedy_pct == 39.0


def test_sampler_optimization_plan_keeps_penalties_and_unsafe_semantics_separate():
    penalty = plan_sampler_optimization(
        SamplerConfig(temperature=0.0, top_k=-1, top_p=1.0, has_penalties=True)
    )
    unsafe = plan_sampler_optimization(
        SamplerConfig(temperature=0.8, top_k=50, top_p=0.9, has_grammar=True)
    )

    assert penalty.target == "fused_repetition_penalty"
    assert penalty.priority == "p1"
    assert penalty.eligible_for_next_prototype is True
    assert unsafe.target == "unsupported_semantics"
    assert unsafe.priority == "defer"
    assert unsafe.eligible_for_next_prototype is False
    assert "grammar_or_structured_output" in unsafe.reasons


def test_boundary_impacts_include_negative_controls_and_p0_budget():
    rows = build_boundary_impacts(".")
    by_name = {row.boundary: row for row in rows}
    assert by_name["RoPE + paged KV append"].micro_speedup_x > 3.0
    sampler = by_name["Self-written standalone sampler"]
    assert sampler.status == "superseded_semantics"
    assert sampler.serving_impact_pct is None
    assert by_name["Standalone LM-head top-k"].micro_speedup_x < 1.0
    logits = by_name["LM-head/logits epilogue"]
    assert logits.status == "active_p0_budget"
    assert logits.eligible_fraction_pct == 96.0
    assert round(logits.materialization_mib, 2) == 339.93


def test_boundary_impacts_are_json_serializable(tmp_path):
    rows = build_boundary_impacts(".")
    path = tmp_path / "rows.json"
    path.write_text(
        json.dumps([row.to_dict() for row in rows], indent=2),
        encoding="utf-8",
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["boundary"] == rows[0].boundary
