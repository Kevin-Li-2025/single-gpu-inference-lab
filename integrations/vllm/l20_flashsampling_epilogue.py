"""Shadow planner for an L20 FlashSampling-style vLLM epilogue.

This helper is behavior-preserving. It converts the existing logits-boundary
trace metadata into the narrower FlashSampling gate: safe decode, small batch,
full-vocabulary greedy/Gumbel only, and no downstream sampler features that the
first epilogue kernel does not implement.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

try:
    from l20_logits_boundary_trace import l20_logits_boundary_gate
except ImportError:  # pragma: no cover - used by repo-local tests.
    import importlib.util
    from pathlib import Path

    _trace_path = Path(__file__).with_name("l20_logits_boundary_trace.py")
    _trace_spec = importlib.util.spec_from_file_location("l20_logits_boundary_trace", _trace_path)
    _trace_module = importlib.util.module_from_spec(_trace_spec)
    _trace_spec.loader.exec_module(_trace_module)
    l20_logits_boundary_gate = _trace_module.l20_logits_boundary_gate

TRACE_MODE_ENV = "VLLM_L20_FLASHSAMPLING_MODE"
TRACE_ENV = "VLLM_L20_FLASHSAMPLING_TRACE"
TRACE_LIMIT_ENV = "VLLM_L20_FLASHSAMPLING_TRACE_LIMIT"
_TRACE_COUNT = 0


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sampling_scalar(metadata: dict[str, Any], name: str, default: float) -> float:
    sampling = metadata.get("sampling", {}) or {}
    min_value = sampling.get(f"{name}_min")
    max_value = sampling.get(f"{name}_max")
    if min_value is None and max_value is None:
        return default
    if min_value == max_value:
        return float(min_value)
    return float(max_value if max_value is not None else min_value)


def flashsampling_request_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the CPU-safe request dict used by l20_stack.epilogue.flash_sampling."""

    mode = os.environ.get(TRACE_MODE_ENV, "gumbel").lower()
    return {
        "batch_size": _safe_int(metadata.get("num_reqs"), 0),
        "vocab_size": _safe_int(metadata.get("vocab_size"), 0),
        "hidden_size": _safe_int(metadata.get("hidden_dim"), 0),
        "decode_only": True,
        "sampling_mode": mode,
        "top_k": _safe_int(_sampling_scalar(metadata, "top_k", -1.0), -1),
        "top_p": _sampling_scalar(metadata, "top_p", 1.0),
        "num_logprobs": _safe_int(_sampling_scalar(metadata, "num_logprobs", 0.0), 0),
        "has_penalties": "penalties" in metadata.get("boundary_reasons", []),
        "has_bad_words": "bad_words" in metadata.get("boundary_reasons", []),
        "has_structured_output": "grammar_or_structured_output" in metadata.get(
            "boundary_reasons", []
        ),
        "speculative_decode": "spec_decode" in metadata.get("boundary_reasons", []),
    }


def plan_l20_flashsampling_epilogue(
    model_runner: Any,
    input_batch: Any,
    grammar_output: Any,
    sample_hidden_states: Any,
    logits: Any,
    scheduler_output: Any = None,
) -> dict[str, Any]:
    """Return a no-mutation shadow plan for the future vLLM epilogue path."""

    boundary_eligible, boundary_reasons, metadata = l20_logits_boundary_gate(
        model_runner,
        input_batch,
        grammar_output,
        sample_hidden_states,
        logits,
        scheduler_output,
    )
    metadata = dict(metadata)
    metadata["boundary_reasons"] = list(boundary_reasons)
    request_kwargs = flashsampling_request_from_metadata(metadata)
    try:
        from l20_stack.epilogue.flash_sampling import (
            FlashSamplingRequest,
            plan_flash_sampling_epilogue,
        )

        decision = plan_flash_sampling_epilogue(FlashSamplingRequest(**request_kwargs))
        flash_reasons = list(decision.reasons)
        policy = decision.policy.to_dict() if decision.policy is not None else None
    except Exception as exc:  # pragma: no cover - defensive vLLM runtime path.
        flash_reasons = [f"flashsampling_plan_failed:{type(exc).__name__}"]
        policy = None

    combined_reasons = sorted(set(boundary_reasons + flash_reasons))
    eligible = bool(boundary_eligible and not flash_reasons)
    logits_bytes = metadata.get("logits_bytes")
    return {
        "prototype_boundary": "lm_head_flashsampling_epilogue",
        "mode": "shadow_trace_only",
        "would_use_epilogue": eligible,
        "mutates_outputs": False,
        "fallback_reasons": combined_reasons,
        "boundary_eligible": boundary_eligible,
        "flashsampling_request": request_kwargs,
        "policy": policy if eligible else None,
        "logits_materialization_bytes": logits_bytes,
        "avoidable_logits_materialization_bytes": logits_bytes if eligible else 0,
    }


def maybe_trace_l20_flashsampling_epilogue(
    model_runner: Any,
    input_batch: Any,
    grammar_output: Any,
    sample_hidden_states: Any,
    logits: Any,
    scheduler_output: Any = None,
) -> None:
    """Write a behavior-preserving FlashSampling shadow-plan event if enabled."""

    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    global _TRACE_COUNT
    limit = int(os.environ.get(TRACE_LIMIT_ENV, "4096"))
    if _TRACE_COUNT >= limit:
        return
    plan = plan_l20_flashsampling_epilogue(
        model_runner,
        input_batch,
        grammar_output,
        sample_hidden_states,
        logits,
        scheduler_output,
    )
    event = {
        "ts": time.time(),
        "event": "l20_flashsampling_epilogue_gate",
        "eligible": plan["would_use_epilogue"],
        "reasons": plan["fallback_reasons"],
        "metadata": {"flashsampling_epilogue": plan},
    }
    _TRACE_COUNT += 1
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
