"""Fallback-first LM-head GEMM epilogue trace hook for vLLM.

This helper is intentionally behavior-preserving with the default vLLM patch:
it calls a new ``LogitsProcessor.try_sample_from_lm_head`` API only when tracing
or explicit experimentation is enabled. The default API returns ``None``, so the
runner falls back to vLLM's existing ``compute_logits`` plus sampler path.

When ``VLLM_L20_GEMM_EPILOGUE_ENABLE=1`` is set, the helper may take a narrow
output-changing greedy path: batch-1, single-token decode, no penalties,
temperature=0, no logprobs, no structured output, TP=1. Everything else stays on
the baseline path.
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


def _array_minmax(values: Any, default: float) -> tuple[float, float]:
    if values is None:
        return default, default
    try:
        return float(values.min()), float(values.max())
    except Exception:
        data = list(values)
        if not data:
            return default, default
        return float(min(data)), float(max(data))


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
    if _non_empty_attr(input_batch, "num_prompt_logprobs"):
        reasons.append("prompt_logprobs")
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


def _sampling_hard_reasons(input_batch: Any) -> list[str]:
    """Semantic blockers for shadow tracing.

    Sparse penalties are intentionally not hard blockers here. They are the
    current P0 producer-side epilogue target, so the trace should classify them
    rather than hide the opportunity behind a generic fallback reason.
    """

    return [
        reason
        for reason in _sampling_reasons(input_batch)
        if reason != "penalties"
    ]


def _topk_topp_active(input_batch: Any, vocab_size: int) -> bool:
    top_k_min, top_k_max = _array_minmax(_active_values(input_batch, "top_k_cpu"), -1.0)
    top_p_min, top_p_max = _array_minmax(_active_values(input_batch, "top_p_cpu"), 1.0)
    top_k = int(top_k_max)
    top_k_active = (
        top_k_min == top_k_max
        and top_k not in {-1, 0, 1}
        and (vocab_size <= 0 or top_k < vocab_size)
    )
    top_p_active = top_p_min == top_p_max and abs(float(top_p_max) - 1.0) > 1e-6
    return bool(top_k_active or top_p_active)


def _penalties_active(input_batch: Any) -> bool:
    return any(
        (
            _array_non_default(_active_values(input_batch, "frequency_penalties_cpu"), 0.0),
            _array_non_default(_active_values(input_batch, "presence_penalties_cpu"), 0.0),
            _array_non_default(_active_values(input_batch, "repetition_penalties_cpu"), 1.0),
        )
    )


def _history_source_metadata(input_batch: Any) -> dict[str, Any]:
    history_tokens = getattr(input_batch, "l20_history_tokens", None)
    history_lengths = getattr(input_batch, "l20_history_lengths", None)
    if history_tokens is not None and history_lengths is not None:
        return {
            "available": True,
            "source": "l20_history_tensors",
            "tokens_shape": _shape(history_tokens),
            "lengths_shape": _shape(history_lengths),
            "tokens_device": str(getattr(history_tokens, "device", None)),
        }
    token_ids_cpu = getattr(input_batch, "token_ids_cpu", None)
    num_tokens_no_spec = getattr(input_batch, "num_tokens_no_spec", None)
    if token_ids_cpu is not None and num_tokens_no_spec is not None:
        return {
            "available": True,
            "source": "input_batch_token_ids_cpu",
            "tokens_shape": _shape(token_ids_cpu),
            "lengths_shape": _shape(num_tokens_no_spec),
            "tokens_device": "cpu",
        }
    return {
        "available": False,
        "source": None,
        "tokens_shape": None,
        "lengths_shape": None,
        "tokens_device": None,
    }


def _dtype_nbytes(dtype: str | None) -> int:
    if dtype is None:
        return 0
    text = dtype.lower()
    if "float64" in text or "int64" in text:
        return 8
    if "float32" in text or "int32" in text:
        return 4
    if "float16" in text or "bfloat16" in text or "int16" in text:
        return 2
    if "int8" in text or "uint8" in text or "bool" in text:
        return 1
    return 0


def _semantic_candidate(
    input_batch: Any,
    hidden_states: Any,
    lm_head: Any | None,
) -> dict[str, Any]:
    hidden_shape = _shape(hidden_states) or []
    batch = int(hidden_shape[0]) if len(hidden_shape) == 2 else _active_num_reqs(input_batch)
    weight = _lm_head_weight(lm_head) if lm_head is not None else None
    weight_shape = _shape(weight)
    vocab_size = _vocab_size(input_batch, weight) if weight is not None else 0
    hidden_size = int(hidden_shape[1]) if len(hidden_shape) == 2 else 0
    logits_bytes = batch * vocab_size * 4 if batch > 0 and vocab_size > 0 else 0
    hidden_bytes = batch * hidden_size * _dtype_nbytes(_dtype(hidden_states))
    features: list[str] = []
    if _topk_topp_active(input_batch, vocab_size):
        features.append("topk_topp")
    if _penalties_active(input_batch):
        features.append("sparse_penalties")
    temperature_min, temperature_max = _array_minmax(
        _active_values(input_batch, "temperature_cpu"),
        0.0,
    )
    if temperature_min == temperature_max and abs(float(temperature_max)) <= 1e-6:
        features.append("greedy")
    elif "topk_topp" not in features:
        features.append("full_vocab_sampling")

    hard_reasons = _sampling_hard_reasons(input_batch)
    history = _history_source_metadata(input_batch)
    target = "unsupported_semantics"
    priority = "defer"
    candidate_reasons = list(hard_reasons)
    if not candidate_reasons:
        if "topk_topp" in features and "sparse_penalties" in features:
            target = "fused_topk_topp_sparse_penalty_lm_head_epilogue"
            priority = "p0"
            if not history["available"]:
                candidate_reasons.append("missing_sparse_history")
        elif "topk_topp" in features:
            target = "fused_topk_topp_lm_head_epilogue"
            priority = "p1"
        elif "sparse_penalties" in features:
            target = "fused_sparse_penalty_greedy_lm_head_epilogue"
            priority = "p1"
            if not history["available"]:
                candidate_reasons.append("missing_sparse_history")
        elif "full_vocab_sampling" in features:
            target = "gumbel_max_lm_head_epilogue"
            priority = "p2"
        else:
            target = "greedy_argmax_control"
            priority = "control"

    return {
        "target": target,
        "priority": priority,
        "eligible": not candidate_reasons and priority != "control",
        "reasons": sorted(set(candidate_reasons)),
        "features": sorted(set(features)),
        "batch": batch,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "weight_shape": weight_shape,
        "estimated_logits_bytes_fp32": logits_bytes,
        "estimated_logits_mib_fp32": logits_bytes / 1048576 if logits_bytes else 0.0,
        "hidden_bytes": hidden_bytes,
        "history": history,
    }


def _greedy_epilogue_reasons(input_batch: Any, hidden_states: Any) -> list[str]:
    reasons: list[str] = []
    hidden_shape = _shape(hidden_states)
    if hidden_shape is None or len(hidden_shape) != 2:
        reasons.append("bad_hidden_shape")
    else:
        if hidden_shape[0] != 1:
            reasons.append("batch_not_one")

    all_greedy = getattr(input_batch, "all_greedy", None)
    if all_greedy is not None:
        if not bool(all_greedy):
            reasons.append("not_all_greedy")
        if not bool(getattr(input_batch, "no_top_p", True)):
            reasons.append("top_p")
        if not bool(getattr(input_batch, "no_top_k", True)):
            reasons.append("top_k")
        return sorted(set(reasons))

    temperature_min, temperature_max = _array_minmax(
        _active_values(input_batch, "temperature_cpu"),
        0.0,
    )
    top_p_min, top_p_max = _array_minmax(_active_values(input_batch, "top_p_cpu"), 1.0)
    top_k_min, top_k_max = _array_minmax(_active_values(input_batch, "top_k_cpu"), -1.0)
    if temperature_min != temperature_max:
        reasons.append("mixed_temperature")
    if abs(temperature_max) > 1e-6:
        reasons.append("non_greedy_temperature")
    if top_p_min != top_p_max:
        reasons.append("mixed_top_p")
    if abs(top_p_max - 1.0) > 1e-6:
        reasons.append("top_p")
    if top_k_min != top_k_max:
        reasons.append("mixed_top_k")
    if int(top_k_max) not in {-1, 0, 1}:
        reasons.append("top_k")
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


def _lm_head_weight(lm_head: Any) -> Any | None:
    weight = getattr(lm_head, "weight", None)
    if weight is not None:
        return weight
    getter = getattr(lm_head, "get_output_embeddings", None)
    if getter is None:
        return None
    try:
        embeddings = getter()
    except Exception:
        return None
    return getattr(embeddings, "weight", None)


def _vocab_size(input_batch: Any, weight: Any) -> int:
    from_weight = _shape(weight)
    weight_vocab = int(from_weight[0]) if from_weight else 0
    value = _safe_int(getattr(input_batch, "vocab_size", None), weight_vocab)
    if value is None or value <= 0:
        return weight_vocab
    return value


def _make_sampler_output(sampled_token_ids: Any) -> Any:
    from vllm.v1.outputs import SamplerOutput

    return SamplerOutput(
        sampled_token_ids=sampled_token_ids,
        logprobs_tensors=None,
    )


def _argmax_correctness_check(
    sample_hidden_states: Any,
    weight: Any,
    sampled_token_ids: Any,
    vocab_size: int,
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "checked": False,
        "matches_baseline_argmax": False,
    }
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime install.
        details["reason"] = f"torch_unavailable:{type(exc).__name__}"
        return details
    if sample_hidden_states is None or weight is None or sampled_token_ids is None:
        details["reason"] = "missing_tensor"
        return details
    hidden_shape = _shape(sample_hidden_states)
    weight_shape = _shape(weight)
    if hidden_shape is None or weight_shape is None:
        details["reason"] = "missing_shape"
        return details
    if len(hidden_shape) != 2 or len(weight_shape) != 2:
        details["reason"] = "bad_rank"
        return details
    if int(weight_shape[1]) != int(hidden_shape[1]):
        details["reason"] = "hidden_mismatch"
        return details
    if vocab_size <= 0 or vocab_size > int(weight_shape[0]):
        details["reason"] = "bad_vocab_size"
        return details

    with torch.no_grad():
        logits = torch.matmul(
            sample_hidden_states.float(),
            weight[:vocab_size, :].float().transpose(0, 1),
        )
        expected = torch.argmax(logits, dim=-1).to(torch.int64).reshape(-1)
        actual = sampled_token_ids.to(torch.int64).reshape(-1)
        matches = bool(torch.equal(actual.cpu(), expected.cpu()))
    details.update(
        {
            "checked": True,
            "matches_baseline_argmax": matches,
            "expected_tokens": [int(value) for value in expected.cpu().tolist()],
            "actual_tokens": [int(value) for value in actual.cpu().tolist()],
        }
    )
    return details


def _try_lm_head_greedy_sampler_output(
    model_runner: Any,
    input_batch: Any,
    sample_hidden_states: Any,
    lm_head: Any,
    embedding_bias: Any | None,
) -> tuple[Any | None, str | None, dict[str, Any]]:
    details: dict[str, Any] = {
        "mode": "greedy_argmax",
        "returned_output": False,
        "uses_full_logits": False,
    }
    if embedding_bias is not None:
        return None, "embedding_bias", details
    try:
        import torch
        from l20_stack.ops.triton_lm_head_sampling import (
            lm_head_sample_out,
            lm_head_sampling_launch_config,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime install.
        return None, f"candidate_import_failed:{type(exc).__name__}", details

    weight = _lm_head_weight(lm_head)
    if weight is None:
        return None, "missing_lm_head_weight", details
    if not getattr(sample_hidden_states, "is_cuda", False):
        return None, "not_cuda", details
    if not getattr(weight, "is_cuda", False):
        return None, "weight_not_cuda", details
    hidden_shape = _shape(sample_hidden_states)
    weight_shape = _shape(weight)
    if hidden_shape is None or weight_shape is None or len(weight_shape) != 2:
        return None, "bad_weight_shape", details
    batch, hidden_size = int(hidden_shape[0]), int(hidden_shape[1])
    real_vocab = _vocab_size(input_batch, weight)
    if real_vocab <= 0 or real_vocab > int(weight_shape[0]):
        return None, "bad_vocab_size", details
    if int(weight_shape[1]) != hidden_size:
        return None, "weight_hidden_mismatch", details
    try:
        config = lm_head_sampling_launch_config(batch, real_vocab, hidden_size)
    except Exception as exc:
        return None, f"bad_launch_shape:{type(exc).__name__}", details

    cache = getattr(model_runner, "_l20_gemm_epilogue_workspace", {})
    key = (
        batch,
        real_vocab,
        hidden_size,
        str(getattr(sample_hidden_states, "dtype", "")),
        str(getattr(sample_hidden_states, "device", "")),
    )
    workspace = cache.get(key)
    if workspace is None:
        workspace = {
            "values": torch.empty(
                (batch,),
                device=sample_hidden_states.device,
                dtype=torch.float32,
            ),
            "tokens": torch.empty(
                (batch,),
                device=sample_hidden_states.device,
                dtype=torch.int64,
            ),
            "partial_values": torch.empty(
                (batch, config.blocks_per_row),
                device=sample_hidden_states.device,
                dtype=torch.float32,
            ),
            "partial_tokens": torch.empty(
                (batch, config.blocks_per_row),
                device=sample_hidden_states.device,
                dtype=torch.int64,
            ),
        }
        cache[key] = workspace
        setattr(model_runner, "_l20_gemm_epilogue_workspace", cache)

    try:
        if not sample_hidden_states.is_contiguous():
            sample_hidden_states = sample_hidden_states.contiguous()
        sampled_weight = weight[:real_vocab, :]
        dummy_seeds = torch.empty((batch,), device=sample_hidden_states.device, dtype=torch.int32)
        lm_head_sample_out(
            sample_hidden_states,
            sampled_weight,
            workspace["values"],
            workspace["tokens"],
            partial_values=workspace["partial_values"],
            partial_tokens=workspace["partial_tokens"],
            seeds=dummy_seeds,
            use_gumbel=False,
        )
        sampled = workspace["tokens"].to(torch.int32).unsqueeze(-1)
        output = _make_sampler_output(sampled)
        correctness = _argmax_correctness_check(
            sample_hidden_states,
            weight,
            sampled,
            real_vocab,
        )
    except Exception as exc:  # pragma: no cover - runtime kernel path.
        return None, f"candidate_failed:{type(exc).__name__}:{str(exc)[:160]}", details

    details.update(
        {
            "returned_output": True,
            "batch": batch,
            "vocab_size": real_vocab,
            "hidden_size": hidden_size,
            "policy": config.to_dict(),
            "correctness": correctness,
        }
    )
    return output, None, details


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
    epilogue: dict[str, Any],
    semantic_candidate: dict[str, Any],
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
            "epilogue": epilogue,
            "semantic_candidate": semantic_candidate,
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
    if _env_flag(ENABLE_ENV):
        reasons.extend(_sampling_reasons(input_batch))
    else:
        reasons.extend(_sampling_hard_reasons(input_batch))
    if _env_flag(ENABLE_ENV):
        reasons.extend(_greedy_epilogue_reasons(input_batch, sample_hidden_states))

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
    epilogue = {
        "attempted": False,
        "returned_output": False,
        "fallback_to_compute_logits": True,
    }
    semantic_candidate = _semantic_candidate(input_batch, sample_hidden_states, lm_head)
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
        if output is None and _env_flag(ENABLE_ENV):
            epilogue["attempted"] = True
            output, candidate_reason, candidate_details = _try_lm_head_greedy_sampler_output(
                model_runner,
                input_batch,
                sample_hidden_states,
                lm_head,
                _embedding_bias(lm_head),
            )
            epilogue.update(candidate_details)
            epilogue["returned_output"] = output is not None
            epilogue["fallback_to_compute_logits"] = output is None
            if candidate_reason is not None:
                reasons.append(candidate_reason)
                api["fallback_to_compute_logits"] = True
            elif output is not None:
                api["fallback_to_compute_logits"] = False

    event = _base_event(
        input_batch,
        sample_hidden_states,
        scheduler_output,
        reasons,
        api,
        epilogue,
        semantic_candidate,
    )
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
