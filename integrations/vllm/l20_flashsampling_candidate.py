"""Experimental vLLM LM-head FlashSampling candidate for L20.

This helper is opt-in and intentionally narrow. It bypasses full-logits
materialization only for decode-only, full-vocabulary greedy/Gumbel requests
that the CPU-safe FlashSampling gate accepts. Everything else returns to the
baseline vLLM path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

ENABLE_ENV = "VLLM_L20_FLASHSAMPLING_CANDIDATE"
TRACE_ENV = "VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE"
TRACE_LIMIT_ENV = "VLLM_L20_FLASHSAMPLING_CANDIDATE_TRACE_LIMIT"
_ALLOW_NON_L20_ENV = "VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20"
_TRACE_COUNT = 0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _active_num_reqs(input_batch: Any) -> int:
    return _safe_int(getattr(input_batch, "num_reqs", 0), 0)


def _active_values(input_batch: Any, field: str) -> Any:
    values = getattr(input_batch, field, None)
    num_reqs = _active_num_reqs(input_batch)
    if values is None:
        return None
    try:
        return values[:num_reqs]
    except Exception:
        return values


def _array_minmax(values: Any, default: float) -> tuple[float, float]:
    if values is None:
        return default, default
    try:
        return float(values.min()), float(values.max())
    except AttributeError:
        data = list(values)
        if not data:
            return default, default
        return float(min(data)), float(max(data))


def _any_active(input_batch: Any, field: str) -> bool:
    value = getattr(input_batch, field, None)
    return bool(value)


def _non_default(input_batch: Any, field: str, default: float) -> bool:
    values = _active_values(input_batch, field)
    if values is None:
        return False
    try:
        return bool((values != default).any())
    except Exception:
        return any(value != default for value in values)


def _scheduled_decode_only(scheduler_output: Any, num_reqs: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    counts = getattr(scheduler_output, "num_scheduled_tokens", None)
    total = _safe_int(getattr(scheduler_output, "total_num_scheduled_tokens", num_reqs), num_reqs)
    if counts:
        values = [_safe_int(value, 0) for value in counts.values()]
        if max(values or [0]) != 1:
            reasons.append("not_single_token_decode")
        if sum(values) != num_reqs:
            reasons.append("scheduled_tokens_mismatch")
    elif total != num_reqs:
        reasons.append("scheduled_tokens_mismatch")
    return not reasons, reasons


def _device_ok(tensor: Any) -> tuple[bool, list[str]]:
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) != "cuda":
        return False, ["not_cuda"]
    if _env_flag(_ALLOW_NON_L20_ENV):
        return True, []
    try:
        import torch

        if torch.cuda.get_device_capability(device) != (8, 9):
            return False, ["not_sm89"]
        if "L20" not in torch.cuda.get_device_name(device):
            return False, ["not_l20"]
    except Exception as exc:  # pragma: no cover - defensive runtime path.
        return False, [f"device_query_failed:{type(exc).__name__}"]
    return True, []


def _lm_head_weight(model_runner: Any) -> Any | None:
    model = getattr(model_runner, "model", None)
    if model is None:
        return None
    for owner in (model, getattr(model, "module", None), getattr(model, "model", None)):
        if owner is None:
            continue
        lm_head = getattr(owner, "lm_head", None)
        weight = getattr(lm_head, "weight", None)
        if weight is not None:
            return weight
        getter = getattr(owner, "get_output_embeddings", None)
        if getter is not None:
            try:
                emb = getter()
                weight = getattr(emb, "weight", None)
                if weight is not None:
                    return weight
            except Exception:
                pass
    return None


def _sampling_request(input_batch: Any, hidden: Any, weight: Any) -> tuple[dict[str, Any], list[str]]:
    reasons: list[str] = []
    num_reqs = _active_num_reqs(input_batch)
    if num_reqs <= 0:
        reasons.append("empty_batch")
    hidden_shape = list(getattr(hidden, "shape", []))
    weight_shape = list(getattr(weight, "shape", []))
    if len(hidden_shape) != 2 or len(weight_shape) != 2:
        reasons.append("bad_shape")
        return {}, reasons
    batch, hidden_size = int(hidden_shape[0]), int(hidden_shape[1])
    vocab_size, weight_hidden = int(weight_shape[0]), int(weight_shape[1])
    if batch != num_reqs:
        reasons.append("batch_mismatch")
    if weight_hidden != hidden_size:
        reasons.append("weight_hidden_mismatch")

    top_k_min, top_k_max = _array_minmax(_active_values(input_batch, "top_k_cpu"), -1.0)
    top_p_min, top_p_max = _array_minmax(_active_values(input_batch, "top_p_cpu"), 1.0)
    temp_min, temp_max = _array_minmax(_active_values(input_batch, "temperature_cpu"), 0.0)
    min_p_min, min_p_max = _array_minmax(_active_values(input_batch, "min_p_cpu"), 0.0)
    if top_k_min != top_k_max:
        reasons.append("mixed_top_k")
    if top_p_min != top_p_max:
        reasons.append("mixed_top_p")
    if temp_min != temp_max:
        reasons.append("mixed_temperature")
    if min_p_min != 0.0 or min_p_max != 0.0:
        reasons.append("min_p")
    if _any_active(input_batch, "num_logprobs"):
        reasons.append("token_logprobs")
    if _any_active(input_batch, "logprob_token_ids"):
        reasons.append("logprob_token_ids")
    if _non_default(input_batch, "frequency_penalties_cpu", 0.0):
        reasons.append("penalties")
    if _non_default(input_batch, "presence_penalties_cpu", 0.0):
        reasons.append("penalties")
    if _non_default(input_batch, "repetition_penalties_cpu", 1.0):
        reasons.append("penalties")
    if _any_active(input_batch, "has_allowed_token_ids"):
        reasons.append("allowed_token_ids")
    if _any_active(input_batch, "bad_words_token_ids"):
        reasons.append("bad_words")
    if _any_active(input_batch, "generators"):
        reasons.append("per_request_generators")
    if bool(getattr(input_batch, "has_structured_output_reqs", False)):
        reasons.append("structured_output")

    top_k = int(top_k_max)
    top_p = float(top_p_max)
    temperature = float(temp_max)
    request = {
        "batch_size": batch,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "sampling_mode": "greedy" if temperature <= 1e-5 else "gumbel",
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
    }
    return request, sorted(set(reasons))


def _trace(event: dict[str, Any]) -> None:
    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    global _TRACE_COUNT
    limit = int(os.environ.get(TRACE_LIMIT_ENV, "200000"))
    if _TRACE_COUNT >= limit:
        return
    _TRACE_COUNT += 1
    event = dict(event)
    event.setdefault("ts", time.time())
    event.setdefault("event", "l20_flashsampling_candidate")
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def maybe_l20_flashsampling_compute_logits_or_sample(
    model_runner: Any,
    input_batch: Any,
    scheduler_output: Any,
    spec_decode_metadata: Any,
    sample_hidden_states: Any,
    compute_logits,
) -> Any:
    """Return logits, or None when the candidate produced a SamplerOutput."""

    setattr(model_runner, "_l20_flashsampling_sampler_output", None)
    if not _env_flag(ENABLE_ENV):
        return compute_logits(sample_hidden_states)
    reasons: list[str] = []
    if spec_decode_metadata is not None:
        reasons.append("spec_decode")
    ok, device_reasons = _device_ok(sample_hidden_states)
    if not ok:
        reasons.extend(device_reasons)
    weight = _lm_head_weight(model_runner)
    if weight is None:
        reasons.append("missing_lm_head_weight")
    request: dict[str, Any] = {}
    if weight is not None:
        request, sampling_reasons = _sampling_request(input_batch, sample_hidden_states, weight)
        reasons.extend(sampling_reasons)
    scheduled_ok, scheduled_reasons = _scheduled_decode_only(
        scheduler_output, _active_num_reqs(input_batch)
    )
    if not scheduled_ok:
        reasons.extend(scheduled_reasons)

    if not reasons and request:
        try:
            from l20_stack.epilogue.flash_sampling import FlashSamplingRequest, plan_flash_sampling_epilogue
            decision = plan_flash_sampling_epilogue(FlashSamplingRequest(**request))
            reasons.extend(decision.reasons)
        except Exception as exc:
            reasons.append(f"plan_failed:{type(exc).__name__}")

    if reasons or not request or weight is None:
        _trace({"eligible": False, "reasons": sorted(set(reasons)), "request": request})
        return compute_logits(sample_hidden_states)

    try:
        import torch
        from vllm.v1.outputs import SamplerOutput
        from l20_stack.ops.triton_lm_head_sampling import lm_head_sample_out, lm_head_sampling_launch_config

        batch = int(request["batch_size"])
        vocab = int(request["vocab_size"])
        hidden = int(request["hidden_size"])
        config = lm_head_sampling_launch_config(batch, vocab, hidden)
        cache = getattr(model_runner, "_l20_flashsampling_workspace", {})
        key = (batch, vocab, hidden, str(sample_hidden_states.dtype), str(sample_hidden_states.device))
        workspace = cache.get(key)
        if workspace is None:
            workspace = {
                "values": torch.empty((batch,), device=sample_hidden_states.device, dtype=torch.float32),
                "tokens": torch.empty((batch,), device=sample_hidden_states.device, dtype=torch.int64),
                "partial_values": torch.empty((batch, config.blocks_per_row), device=sample_hidden_states.device, dtype=torch.float32),
                "partial_tokens": torch.empty((batch, config.blocks_per_row), device=sample_hidden_states.device, dtype=torch.int64),
            }
            cache[key] = workspace
            setattr(model_runner, "_l20_flashsampling_workspace", cache)
        sampling_metadata = getattr(input_batch, "sampling_metadata", None)
        seeds = getattr(sampling_metadata, "l20_seeds", None)
        positions = getattr(sampling_metadata, "l20_positions", None)
        if seeds is None:
            seeds = torch.arange(batch, device=sample_hidden_states.device, dtype=torch.int64)
        else:
            seeds = seeds[:batch]
        if positions is not None:
            positions = positions[:batch]
        lm_head_sample_out(
            sample_hidden_states,
            weight,
            workspace["values"],
            workspace["tokens"],
            partial_values=workspace["partial_values"],
            partial_tokens=workspace["partial_tokens"],
            seeds=seeds,
            positions=positions,
            use_gumbel=request["sampling_mode"] == "gumbel",
            temperature=float(request["temperature"]),
        )
        sampled = workspace["tokens"].to(torch.int32).unsqueeze(-1)
        setattr(
            model_runner,
            "_l20_flashsampling_sampler_output",
            SamplerOutput(sampled_token_ids=sampled, logprobs_tensors=None),
        )
        _trace({"eligible": True, "reasons": [], "request": request, "policy": config.to_dict()})
        return None
    except Exception as exc:  # pragma: no cover - defensive runtime path.
        _trace({"eligible": False, "reasons": [f"candidate_failed:{type(exc).__name__}"], "request": request})
        setattr(model_runner, "_l20_flashsampling_sampler_output", None)
        return compute_logits(sample_hidden_states)


def maybe_take_l20_flashsampling_sampler_output(model_runner: Any, grammar_output: Any) -> Any | None:
    output = getattr(model_runner, "_l20_flashsampling_sampler_output", None)
    setattr(model_runner, "_l20_flashsampling_sampler_output", None)
    if output is None or grammar_output is not None:
        return None
    return output
