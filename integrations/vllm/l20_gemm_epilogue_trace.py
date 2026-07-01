"""Fallback-first LM-head GEMM epilogue trace hook for vLLM.

This helper is intentionally behavior-preserving with the default vLLM patch:
it calls a new ``LogitsProcessor.try_sample_from_lm_head`` API only when tracing
or explicit experimentation is enabled. The default API returns ``None``, so the
runner falls back to vLLM's existing ``compute_logits`` plus sampler path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

TRACE_ENV = "VLLM_L20_GEMM_EPILOGUE_TRACE"
TRACE_LIMIT_ENV = "VLLM_L20_GEMM_EPILOGUE_TRACE_LIMIT"
ENABLE_ENV = "VLLM_L20_GEMM_EPILOGUE_ENABLE"
ALLOW_NON_L20_ENV = "VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20"

_TRACE_COUNT = 0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _trace_enabled() -> bool:
    return bool(os.environ.get(TRACE_ENV))


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _shape(value: Any) -> list[int] | None:
    raw = getattr(value, "shape", None)
    if raw is None:
        return None
    return [int(dim) for dim in raw]


def _dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def _device_reason(tensor: Any) -> str | None:
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) != "cuda":
        return "not_cuda"
    if _env_flag(ALLOW_NON_L20_ENV):
        return None
    try:
        import torch

        capability = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
    except Exception as exc:  # pragma: no cover - defensive runtime path.
        return f"device_query_failed:{type(exc).__name__}"
    if capability != (8, 9):
        return f"not_sm89:{capability[0]}{capability[1]}"
    if "L20" not in name:
        return f"not_l20:{name}"
    return None


def _active_num_reqs(input_batch: Any) -> int:
    return _safe_int(getattr(input_batch, "num_reqs", 0), 0) or 0


def _active_values(input_batch: Any, field: str) -> Any:
    values = getattr(input_batch, field, None)
    num_reqs = _active_num_reqs(input_batch)
    if values is None:
        return None
    try:
        return values[:num_reqs]
    except Exception:
        return values


def _array_any(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(value.any())
    except AttributeError:
        return any(bool(item) for item in value)
    except Exception:
        return bool(value)


def _array_non_default(values: Any, default: float) -> bool:
    if values is None:
        return False
    try:
        return bool((values != default).any())
    except Exception:
        return any(value != default for value in values)


def _non_empty_attr(input_batch: Any, field: str) -> bool:
    value = getattr(input_batch, field, None)
    try:
        return bool(value)
    except ValueError:
        return _array_any(value)


def _scheduled_reasons(scheduler_output: Any, expected_reqs: int) -> list[str]:
    if scheduler_output is None:
        return []
    reasons: list[str] = []
    if getattr(scheduler_output, "grammar_bitmask", None) is not None:
        reasons.append("grammar_output")
    counts = getattr(scheduler_output, "num_scheduled_tokens", None)
    total = _safe_int(
        getattr(scheduler_output, "total_num_scheduled_tokens", None),
        expected_reqs,
    )
    if counts:
        values = [_safe_int(value, 0) or 0 for value in counts.values()]
        if max(values or [0]) != 1:
            reasons.append("not_single_token_decode")
        if sum(values) != expected_reqs:
            reasons.append("scheduled_tokens_mismatch")
    elif total != expected_reqs:
        reasons.append("scheduled_tokens_mismatch")
    return reasons


def _sampling_reasons(input_batch: Any) -> list[str]:
    reasons: list[str] = []
    if _non_empty_attr(input_batch, "num_logprobs"):
        reasons.append("token_logprobs")
    if _non_empty_attr(input_batch, "logprob_token_ids"):
        reasons.append("logprob_token_ids")
    if _non_empty_attr(input_batch, "has_allowed_token_ids"):
        reasons.append("allowed_token_ids")
    if _non_empty_attr(input_batch, "bad_words_token_ids"):
        reasons.append("bad_words")
    if _non_empty_attr(input_batch, "generators"):
        reasons.append("per_request_generators")
    if bool(getattr(input_batch, "has_structured_output_reqs", False)):
        reasons.append("structured_output")
    if _array_any(getattr(input_batch, "logits_processing_needs_token_ids", None)):
        reasons.append("logits_processing_needs_token_ids")
    if _array_non_default(_active_values(input_batch, "frequency_penalties_cpu"), 0.0):
        reasons.append("penalties")
    if _array_non_default(_active_values(input_batch, "presence_penalties_cpu"), 0.0):
        reasons.append("penalties")
    if _array_non_default(_active_values(input_batch, "repetition_penalties_cpu"), 1.0):
        reasons.append("penalties")
    if _array_non_default(_active_values(input_batch, "min_p_cpu"), 0.0):
        reasons.append("min_p")
    return sorted(set(reasons))


def _find_logits_owner(model_runner: Any) -> tuple[Any | None, Any | None, Any | None]:
    model = getattr(model_runner, "model", None)
    owners = [
        model,
        getattr(model, "module", None),
        getattr(model, "model", None),
        getattr(model, "language_model", None),
    ]
    for owner in owners:
        if owner is None:
            continue
        logits_processor = getattr(owner, "logits_processor", None)
        lm_head = getattr(owner, "lm_head", None)
        if logits_processor is not None and lm_head is not None:
            return owner, logits_processor, lm_head
    return None, None, None


def _embedding_bias(lm_head: Any) -> Any | None:
    return getattr(lm_head, "bias", None)


def _write_trace(event: dict[str, Any]) -> None:
    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    global _TRACE_COUNT
    limit = int(os.environ.get(TRACE_LIMIT_ENV, "200000"))
    if _TRACE_COUNT >= limit:
        return
    _TRACE_COUNT += 1
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _base_event(
    input_batch: Any,
    hidden_states: Any,
    scheduler_output: Any,
    reasons: list[str],
    api: dict[str, Any],
) -> dict[str, Any]:
    return {
        "event": "l20_gemm_epilogue_boundary",
        "ts": time.time(),
        "eligible": not reasons,
        "reasons": sorted(set(reasons)),
        "metadata": {
            "phase": "fallback_first_api_trace",
            "num_reqs": _active_num_reqs(input_batch),
            "hidden_shape": _shape(hidden_states),
            "hidden_dtype": _dtype(hidden_states),
            "scheduler_total_num_scheduled_tokens": _safe_int(
                getattr(scheduler_output, "total_num_scheduled_tokens", None)
            ),
            "api": api,
            "mutates_outputs": False,
        },
    }


def maybe_try_l20_gemm_epilogue(
    model_runner: Any,
    input_batch: Any,
    grammar_output: Any,
    sample_hidden_states: Any,
    scheduler_output: Any = None,
    spec_decode_metadata: Any = None,
) -> Any | None:
    """Try the fallback-first LM-head epilogue API, then return ``None``.

    ``None`` means the caller must continue through vLLM's existing
    ``compute_logits`` and sampler path. A non-``None`` return is only surfaced
    when ``VLLM_L20_GEMM_EPILOGUE_ENABLE=1`` is set for an explicit experiment.
    """

    if not _trace_enabled() and not _env_flag(ENABLE_ENV):
        return None

    reasons: list[str] = []
    if grammar_output is not None:
        reasons.append("grammar_output")
    if spec_decode_metadata is not None:
        reasons.append("spec_decode")
    if _safe_int(getattr(input_batch, "num_draft_tokens", 0), 0):
        reasons.append("draft_tokens")
    device_reason = _device_reason(sample_hidden_states)
    if device_reason is not None:
        reasons.append(device_reason)
    expected_reqs = _active_num_reqs(input_batch)
    reasons.extend(_scheduled_reasons(scheduler_output, expected_reqs))
    reasons.extend(_sampling_reasons(input_batch))

    parallel_config = getattr(model_runner, "parallel_config", None)
    tp_size = _safe_int(getattr(parallel_config, "tensor_parallel_size", 1), 1)
    if tp_size != 1:
        reasons.append("tensor_parallel")

    _, logits_processor, lm_head = _find_logits_owner(model_runner)
    try_api = getattr(logits_processor, "try_sample_from_lm_head", None)
    api = {
        "logits_processor_found": logits_processor is not None,
        "lm_head_found": lm_head is not None,
        "try_api_found": callable(try_api),
        "api_called": False,
        "api_returned_output": False,
        "output_enabled": _env_flag(ENABLE_ENV),
        "fallback_to_compute_logits": True,
    }
    if logits_processor is None:
        reasons.append("missing_logits_processor")
    if lm_head is None:
        reasons.append("missing_lm_head")
    if not callable(try_api):
        reasons.append("missing_try_sample_from_lm_head")

    output = None
    if not reasons and callable(try_api):
        api["api_called"] = True
        output = try_api(
            lm_head,
            sample_hidden_states,
            input_batch,
            embedding_bias=_embedding_bias(lm_head),
        )
        api["api_returned_output"] = output is not None
        api["fallback_to_compute_logits"] = output is None or not _env_flag(ENABLE_ENV)

    event = _base_event(input_batch, sample_hidden_states, scheduler_output, reasons, api)
    if output is not None and _env_flag(ENABLE_ENV):
        event["metadata"]["mutates_outputs"] = True
    _write_trace(event)
    if output is not None and _env_flag(ENABLE_ENV):
        return output
    return None


def maybe_take_l20_gemm_epilogue_sampler_output(model_runner: Any) -> Any | None:
    output = getattr(model_runner, "_l20_gemm_epilogue_sampler_output", None)
    setattr(model_runner, "_l20_gemm_epilogue_sampler_output", None)
    return output
