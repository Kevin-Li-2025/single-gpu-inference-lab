"""vLLM custom logits processor scaffold for the L20 sparse penalty op.

The module keeps the optimized path opt-in. Requests must pass
``l20_sparse_repetition_penalty=True`` and ``l20_repetition_penalty`` through
vLLM custom logits-processor extra args before logits are modified.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

try:  # pragma: no cover - exercised on the serving host.
    import torch
except Exception:  # pragma: no cover - keeps source tests CPU/package safe.
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - vLLM is optional for local contract tests.
    from vllm.v1.sample.logits_processor.interface import (  # type: ignore
        BatchUpdate,
        LogitsProcessor,
        MoveDirectionality,
    )
except Exception:  # pragma: no cover
    try:
        from vllm.v1.sample.logits_processor import (  # type: ignore
            BatchUpdate,
            LogitsProcessor,
            MoveDirectionality,
        )
    except Exception:
        BatchUpdate = Any  # type: ignore[assignment]

        class LogitsProcessor:  # type: ignore[no-redef]
            pass

        class MoveDirectionality:  # type: ignore[no-redef]
            SWAP = "swap"


GATE_MIN_VOCAB = 65536
GATE_MIN_DENSE_ELEMENTS = 524288
GATE_MAX_UNIQUE_TOKENS = 1024
LIBRARY_ENV = "VLLM_L20_SPARSE_REPETITION_PENALTY_LIBRARY"
TRACE_ENV = "VLLM_L20_SPARSE_REPETITION_PENALTY_TRACE"
FORCE_FALLBACK_ENV = "VLLM_L20_SPARSE_REPETITION_PENALTY_FORCE_TORCH"

_LIBRARY_LOADED = False
_FAKE_REGISTERED = False
_TRACE_COUNT = 0


@dataclass(frozen=True)
class SparsePenaltyDecision:
    provider: str
    reason: str
    batch: int
    vocab: int
    max_unique_tokens: int
    op_available: bool

    @property
    def use_sparse_op(self) -> bool:
        return self.provider == "sparse_op"


@dataclass
class _RequestState:
    row: int
    penalty: float
    include_prompt: bool
    prompt_token_ids: Any
    output_token_ids: Any


def should_use_sparse_repetition_penalty(
    batch: int,
    vocab: int,
    unique_tokens: int,
) -> bool:
    dense_elements = int(batch) * int(vocab)
    return (
        int(vocab) >= GATE_MIN_VOCAB
        and dense_elements >= GATE_MIN_DENSE_ELEMENTS
        and int(unique_tokens) <= GATE_MAX_UNIQUE_TOKENS
    )


def _validated_penalty(value: Any, *, name: str = "repetition_penalty") -> float:
    penalty = float(value)
    if not math.isfinite(penalty) or penalty <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return penalty


def _uniform_penalty(penalties: list[float]) -> float | None:
    if not penalties:
        return None
    first = penalties[0]
    return first if all(value == first for value in penalties[1:]) else None


def select_sparse_repetition_penalty_provider(
    batch: int,
    vocab: int,
    unique_tokens: int,
    *,
    op_available: bool,
) -> SparsePenaltyDecision:
    if not should_use_sparse_repetition_penalty(batch, vocab, unique_tokens):
        return SparsePenaltyDecision(
            provider="torch_fallback",
            reason="outside_sparse_gate",
            batch=int(batch),
            vocab=int(vocab),
            max_unique_tokens=int(unique_tokens),
            op_available=op_available,
        )
    if not op_available:
        return SparsePenaltyDecision(
            provider="torch_fallback",
            reason="op_unavailable",
            batch=int(batch),
            vocab=int(vocab),
            max_unique_tokens=int(unique_tokens),
            op_available=False,
        )
    return SparsePenaltyDecision(
        provider="sparse_op",
        reason="inside_sparse_gate",
        batch=int(batch),
        vocab=int(vocab),
        max_unique_tokens=int(unique_tokens),
        op_available=True,
    )


def load_library(path: Optional[Union[str, Path]] = None) -> bool:
    """Load the dispatcher op when the compiled shared object is available."""

    global _LIBRARY_LOADED
    if torch is None:
        return False
    if _LIBRARY_LOADED:
        return True

    library_path = (
        Path(path)
        if path is not None
        else Path(
            os.environ.get(
                LIBRARY_ENV,
                Path(__file__).with_name("l20_sparse_repetition_penalty_ops.so"),
            )
        )
    )
    if not library_path.exists():
        return False
    torch.ops.load_library(str(library_path))
    _LIBRARY_LOADED = True
    _register_fake()
    return True


def _register_fake() -> None:
    global _FAKE_REGISTERED
    if torch is None or _FAKE_REGISTERED:
        return

    try:
        @torch.library.register_fake("l20_stack::sparse_repetition_penalty_out")
        def _sparse_repetition_penalty_out_fake(
            logits,
            token_ids,
            lengths,
            repetition_penalty,
        ):
            return logits
    except RuntimeError:
        pass
    _FAKE_REGISTERED = True


def _op_available() -> bool:
    if torch is None:
        return False
    if os.environ.get(FORCE_FALLBACK_ENV, "0").lower() in {"1", "true", "yes", "on"}:
        return False
    load_library()
    try:
        torch.ops.l20_stack.sparse_repetition_penalty_out
    except (AttributeError, RuntimeError):
        return False
    return True


def _trace(event: dict[str, Any]) -> None:
    global _TRACE_COUNT
    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    _TRACE_COUNT += 1
    record = {
        "schema_version": 1,
        "timestamp_ns": time.time_ns(),
        "sequence": _TRACE_COUNT,
        **event,
    }
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _apply_torch_fallback(
    logits: Any,
    token_ids: Any,
    lengths: Any,
    repetition_penalty: float,
) -> Any:
    if torch is None:
        raise RuntimeError("torch is required to apply repetition penalty")
    batch = int(logits.shape[0])
    for row in range(batch):
        length = int(lengths[row].item())
        if length <= 0:
            continue
        row_tokens = token_ids[row, :length].to(dtype=torch.long, device=logits.device)
        values = logits[row].index_select(0, row_tokens)
        adjusted = torch.where(
            values > 0,
            values / float(repetition_penalty),
            values * float(repetition_penalty),
        )
        logits[row].index_copy_(0, row_tokens, adjusted)
    return logits


def apply_sparse_repetition_penalty(
    logits: Any,
    token_ids: Any,
    lengths: Any,
    repetition_penalty: float,
) -> Any:
    """Apply repetition penalty in-place and return ``logits``.

    Each active prefix ``token_ids[row, : lengths[row]]`` must contain unique
    token IDs. ``L20SparseRepetitionPenaltyLogitsProcessor`` constructs that
    representation; direct callers are responsible for the same invariant.
    """

    if torch is None:
        raise RuntimeError("torch is required to apply repetition penalty")
    if logits.ndim != 2:
        raise ValueError("logits must be [batch, vocab]")
    if token_ids.ndim != 2:
        raise ValueError("token_ids must be [batch, max_tokens]")
    if lengths.ndim != 1:
        raise ValueError("lengths must be [batch]")
    if token_ids.shape[0] != logits.shape[0] or lengths.shape[0] != logits.shape[0]:
        raise ValueError("history batch must match logits batch")
    penalty = _validated_penalty(repetition_penalty)

    # The padded history width is a conservative upper bound for every row.
    # Using it keeps provider selection free of a device-to-host synchronization
    # in the per-token serving path.
    max_unique = int(token_ids.shape[1])
    compatible_tensors = (
        bool(getattr(logits, "is_cuda", False))
        and bool(getattr(token_ids, "is_cuda", False))
        and bool(getattr(lengths, "is_cuda", False))
        and logits.is_contiguous()
        and token_ids.is_contiguous()
        and lengths.is_contiguous()
        and logits.dtype == torch.float32
        and token_ids.dtype == torch.long
        and lengths.dtype == torch.long
    )
    decision = select_sparse_repetition_penalty_provider(
        int(logits.shape[0]),
        int(logits.shape[1]),
        max_unique,
        op_available=_op_available() and compatible_tensors,
    )
    _trace(
        {
            "event": "apply_sparse_repetition_penalty",
            "provider": decision.provider,
            "reason": decision.reason,
            "batch": decision.batch,
            "vocab": decision.vocab,
            "max_unique_tokens": decision.max_unique_tokens,
        }
    )
    if decision.use_sparse_op:
        return torch.ops.l20_stack.sparse_repetition_penalty_out(
            logits,
            token_ids,
            lengths,
            penalty,
        )
    return _apply_torch_fallback(logits, token_ids, lengths, penalty)


def _as_extra_args(params: Any) -> dict[str, Any]:
    extra = getattr(params, "extra_args", None)
    if extra is None:
        extra = getattr(params, "vllm_xargs", None)
    return dict(extra or {})


def _extra_bool(extra: dict[str, Any], key: str, default: bool = False) -> bool:
    value = extra.get(key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _tokens_from(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(token) for token in value]
    return [int(token) for token in list(value)]


def _is_swap(direction: Any) -> bool:
    if direction == getattr(MoveDirectionality, "SWAP", object()):
        return True
    return str(direction).lower().endswith("swap")


class L20SparseRepetitionPenaltyProcessor(LogitsProcessor):
    """Opt-in vLLM processor that routes eligible rows to the L20 dispatcher op."""

    @classmethod
    def validate_params(cls, params: Any) -> None:
        extra = _as_extra_args(params)
        if not _extra_bool(extra, "l20_sparse_repetition_penalty", False):
            return
        penalty = _validated_penalty(
            extra.get("l20_repetition_penalty", 1.0),
            name="l20_repetition_penalty",
        )
        if penalty == 1.0:
            raise ValueError("l20_repetition_penalty must differ from 1.0")

    def __init__(
        self,
        vllm_config: Any = None,
        device: Any = None,
        is_pin_memory: bool = False,
    ):
        self.states: dict[int, _RequestState] = {}
        self.device = device
        self.is_pin_memory = is_pin_memory
        self.vllm_config = vllm_config

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate) -> None:
        for removed in getattr(batch_update, "removed", []):
            self.states.pop(int(removed), None)

        for added in getattr(batch_update, "added", []):
            index, params, prompt_token_ids, output_token_ids = added
            extra = _as_extra_args(params)
            if not _extra_bool(extra, "l20_sparse_repetition_penalty", False):
                continue
            penalty = _validated_penalty(
                extra.get("l20_repetition_penalty", 1.0),
                name="l20_repetition_penalty",
            )
            if penalty == 1.0:
                continue
            self.states[int(index)] = _RequestState(
                row=int(index),
                penalty=penalty,
                include_prompt=_extra_bool(extra, "l20_penalty_include_prompt", False),
                prompt_token_ids=prompt_token_ids,
                output_token_ids=output_token_ids,
            )

        for moved in getattr(batch_update, "moved", []):
            if len(moved) == 2:
                old, new = moved
                direction = None
            else:
                old, new, direction = moved
            old = int(old)
            new = int(new)
            if _is_swap(direction):
                old_state = self.states.get(old)
                new_state = self.states.get(new)
                if old_state is not None:
                    old_state.row = new
                    self.states[new] = old_state
                else:
                    self.states.pop(new, None)
                if new_state is not None:
                    new_state.row = old
                    self.states[old] = new_state
                else:
                    self.states.pop(old, None)
                continue
            state = self.states.pop(old, None)
            if state is None:
                continue
            state.row = int(new)
            self.states[state.row] = state

    def _build_history_tensors(self, logits: Any) -> tuple[Any, Any, float | None]:
        if torch is None:
            raise RuntimeError("torch is required to apply repetition penalty")
        batch = int(logits.shape[0])
        vocab = int(logits.shape[1])
        rows: list[list[int]] = [[] for _ in range(batch)]
        penalties: list[float] = []
        for state in self.states.values():
            if state.row < 0 or state.row >= batch:
                continue
            seen: set[int] = set()
            source_tokens = []
            if state.include_prompt:
                source_tokens.extend(_tokens_from(state.prompt_token_ids))
            source_tokens.extend(_tokens_from(state.output_token_ids))
            for token in source_tokens:
                if 0 <= token < vocab and token not in seen:
                    seen.add(token)
                    rows[state.row].append(token)
            penalties.append(float(state.penalty))

        max_tokens = max((len(row) for row in rows), default=0)
        token_ids = torch.zeros(
            (batch, max_tokens),
            device=logits.device,
            dtype=torch.long,
        )
        lengths = torch.zeros((batch,), device=logits.device, dtype=torch.long)
        for row, tokens in enumerate(rows):
            if not tokens:
                continue
            lengths[row] = len(tokens)
            token_ids[row, : len(tokens)] = torch.tensor(
                tokens,
                device=logits.device,
                dtype=torch.long,
            )
        uniform_penalty = _uniform_penalty(penalties)
        return token_ids, lengths, uniform_penalty

    def apply(self, logits: Any) -> Any:
        if not self.states:
            return logits
        token_ids, lengths, uniform_penalty = self._build_history_tensors(logits)
        max_unique_tokens = int(token_ids.shape[1])
        if max_unique_tokens == 0:
            return logits
        if uniform_penalty is None:
            _trace(
                {
                    "event": "apply_sparse_repetition_penalty",
                    "provider": "torch_fallback",
                    "reason": "mixed_repetition_penalties",
                    "batch": int(logits.shape[0]),
                    "vocab": int(logits.shape[1]),
                    "max_unique_tokens": max_unique_tokens,
                }
            )
            for state in self.states.values():
                if state.row < 0 or state.row >= int(logits.shape[0]):
                    continue
                row_tokens = token_ids[state.row : state.row + 1]
                row_lengths = lengths[state.row : state.row + 1]
                _apply_torch_fallback(
                    logits[state.row : state.row + 1],
                    row_tokens,
                    row_lengths,
                    state.penalty,
                )
            return logits
        return apply_sparse_repetition_penalty(
            logits,
            token_ids,
            lengths,
            float(uniform_penalty),
        )
