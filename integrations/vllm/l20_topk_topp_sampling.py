"""Opt-in L20 top-k/top-p sampler hook for vLLM.

This module is intentionally narrow. It only replaces FlashInfer sampling for
the measured L20 win regime and returns ``None`` for every unsupported request
so vLLM can fall back to its native path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from l20_stack.ops.triton_sampling import (
    should_prefer_l20_topk_topp_sampling,
    topk_topp_sample_with_vllm_rng_out,
    topk_topp_sampling_launch_config,
)

ENABLE_ENV = "VLLM_L20_TOPK_TOPP_SAMPLER"
TRACE_ENV = "VLLM_L20_TOPK_TOPP_SAMPLER_TRACE"
ALLOW_NON_L20_ENV = "VLLM_L20_TOPK_TOPP_ALLOW_NON_L20"

_WORKSPACE_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
_TRACE_COUNT = 0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _scalar_from_tensor(value: torch.Tensor | None, *, name: str) -> int | float | None:
    if value is None:
        return None
    if value.numel() == 0:
        return None
    first = value.reshape(-1)[0]
    if not torch.all(value == first):
        raise ValueError(f"mixed_{name}")
    if value.dtype.is_floating_point:
        return float(first.item())
    return int(first.item())


def _device_reason(logits: torch.Tensor) -> str | None:
    if not logits.is_cuda:
        return "not_cuda"
    if _env_flag(ALLOW_NON_L20_ENV):
        return None
    capability = torch.cuda.get_device_capability(logits.device)
    name = torch.cuda.get_device_name(logits.device)
    if capability != (8, 9):
        return f"not_sm89:{capability[0]}{capability[1]}"
    if "L20" not in name:
        return f"not_l20:{name}"
    return None


def _trace(event: dict[str, Any]) -> None:
    global _TRACE_COUNT
    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    _TRACE_COUNT += 1
    event = {
        "schema_version": 1,
        "timestamp_ns": time.time_ns(),
        "sequence": _TRACE_COUNT,
        **event,
    }
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _workspace(
    logits: torch.Tensor,
    *,
    top_k: int,
    block_vocab_override: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, vocab = (int(logits.shape[0]), int(logits.shape[1]))
    config = topk_topp_sampling_launch_config(
        vocab,
        top_k,
        batch=batch,
        block_vocab_override=block_vocab_override,
    )
    shape = (batch, config.blocks_per_row, top_k)
    key = (
        logits.device.type,
        int(logits.device.index or 0),
        str(logits.dtype),
        batch,
        vocab,
        top_k,
        config.block_vocab,
    )
    cached = _WORKSPACE_CACHE.get(key)
    if cached is not None and cached[0].shape == shape:
        return cached
    partial_values = torch.empty(shape, device=logits.device, dtype=torch.float32)
    partial_tokens = torch.empty(shape, device=logits.device, dtype=torch.int64)
    _WORKSPACE_CACHE[key] = (partial_values, partial_tokens)
    return partial_values, partial_tokens


def maybe_l20_topk_topp_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator] | None = None,
    *,
    expanded_idx_mapping: torch.Tensor | None = None,
    seeds: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    top_k_value: int | None = None,
    top_p_value: float | None = None,
) -> torch.Tensor | None:
    """Return sampled token ids or ``None`` when the guarded path is ineligible."""

    reasons: list[str] = []
    metadata: dict[str, Any] = {
        "logits_shape": list(logits.shape) if hasattr(logits, "shape") else None,
        "logits_dtype": str(getattr(logits, "dtype", None)),
    }
    if not _env_flag(ENABLE_ENV):
        reasons.append("disabled")
    if logits.ndim != 2:
        reasons.append("not_2d_logits")
    device_reason = _device_reason(logits)
    if device_reason is not None:
        reasons.append(device_reason)
    if generators:
        reasons.append("per_request_generators")
    if k is None or p is None:
        reasons.append("missing_topk_or_topp")
    if expanded_idx_mapping is None or seeds is None or positions is None:
        reasons.append("missing_vllm_rng_state")

    top_k: int | None = None
    top_p: float | None = None
    if not reasons:
        if top_k_value is not None and top_p_value is not None:
            top_k = int(top_k_value)
            top_p = float(top_p_value)
        else:
            try:
                top_k = int(_scalar_from_tensor(k, name="top_k"))
                top_p = float(_scalar_from_tensor(p, name="top_p"))
            except ValueError as exc:
                reasons.append(str(exc))
    metadata["top_k"] = top_k
    metadata["top_p"] = top_p
    metadata["vllm_rng_state"] = not (
        expanded_idx_mapping is None or seeds is None or positions is None
    )

    if not reasons:
        batch, vocab = int(logits.shape[0]), int(logits.shape[1])
        metadata["batch"] = batch
        metadata["vocab"] = vocab
        if top_k is None or top_p is None:
            reasons.append("missing_topk_or_topp")
        elif not should_prefer_l20_topk_topp_sampling(batch, vocab, top_k, top_p):
            reasons.append("outside_l20_profitability_gate")

    if reasons:
        _trace({"eligible": False, "reasons": reasons, "metadata": metadata})
        return None

    assert top_k is not None and top_p is not None
    output = torch.empty((logits.shape[0],), device=logits.device, dtype=torch.int64)
    partial_values, partial_tokens = _workspace(logits, top_k=top_k)
    assert expanded_idx_mapping is not None
    assert seeds is not None
    assert positions is not None
    topk_topp_sample_with_vllm_rng_out(
        logits,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        expanded_idx_mapping=expanded_idx_mapping,
        seeds=seeds,
        positions=positions,
        top_k=top_k,
        top_p=top_p,
        temperature=1.0,
    )
    _trace({"eligible": True, "reasons": [], "metadata": metadata})
    return output
