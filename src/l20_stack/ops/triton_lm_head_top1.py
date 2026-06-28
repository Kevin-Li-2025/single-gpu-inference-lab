"""Experimental L20 fused LM-head top-1 candidate.

This is a boundary probe, not a production sampler.  It computes
`hidden @ weight.T` and returns only the top-1 token without materializing the
full `[batch, vocab]` logits tensor.  The intent is to quantify whether fusing
LM-head production with sampling is worth deeper engineering on L20.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


@dataclass(frozen=True)
class LMHeadTop1LaunchConfig:
    block_vocab: int
    block_hidden: int
    blocks_per_row: int
    reduce_block: int
    num_warps: int
    strategy: str

    def to_dict(self):
        return asdict(self)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def lm_head_top1_launch_config(
    vocab_size: int,
    hidden_size: int,
    *,
    block_vocab: Optional[int] = None,
    block_hidden: int = 64,
) -> LMHeadTop1LaunchConfig:
    if vocab_size <= 0 or hidden_size <= 0:
        raise ValueError("vocab_size and hidden_size must be positive")
    if hidden_size % block_hidden != 0:
        raise ValueError("hidden_size must be divisible by block_hidden")
    selected_block_vocab = block_vocab or 32
    if selected_block_vocab not in {16, 32, 64}:
        raise ValueError("block_vocab must be one of 16, 32, or 64")
    if block_hidden not in {32, 64, 128}:
        raise ValueError("block_hidden must be one of 32, 64, or 128")
    blocks_per_row = (vocab_size + selected_block_vocab - 1) // selected_block_vocab
    return LMHeadTop1LaunchConfig(
        block_vocab=selected_block_vocab,
        block_hidden=block_hidden,
        blocks_per_row=blocks_per_row,
        reduce_block=next_power_of_2(blocks_per_row),
        num_warps=4 if selected_block_vocab <= 32 else 8,
        strategy="two_stage_direct_lm_head_top1",
    )


def should_use_l20_lm_head_top1(
    batch: int,
    vocab_size: int,
    hidden_size: int,
    *,
    top_k: int = 1,
) -> bool:
    """Conservative gate for the experimental direct top-1 path."""

    if top_k != 1:
        return False
    if batch <= 0 or vocab_size <= 0 or hidden_size <= 0:
        return False
    if hidden_size % 64 != 0:
        return False
    return batch <= 4 and vocab_size <= 262_144 and hidden_size <= 8192


if triton is not None:  # pragma: no cover - requires CUDA

    @triton.jit
    def _lm_head_top1_partial_kernel(
        hidden,
        weight,
        partial_values,
        partial_tokens,
        HIDDEN: tl.constexpr,
        VOCAB: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
    ):
        row = tl.program_id(0)
        vocab_block = tl.program_id(1)
        vocab_offsets = vocab_block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
        acc = tl.zeros((BLOCK_VOCAB,), dtype=tl.float32)

        for hidden_start in range(0, HIDDEN, BLOCK_HIDDEN):
            h_offsets = hidden_start + hidden_offsets
            h = tl.load(hidden + row * HIDDEN + h_offsets)
            w = tl.load(
                weight + vocab_offsets[:, None] * HIDDEN + h_offsets[None, :],
                mask=vocab_offsets[:, None] < VOCAB,
                other=0.0,
            )
            scores = tl.dot(w, h[:, None], out_dtype=tl.float32)
            acc += tl.reshape(scores, (BLOCK_VOCAB,))

        valid = vocab_offsets < VOCAB
        acc = tl.where(valid, acc, -float("inf"))
        max_value = tl.max(acc, axis=0)
        token_candidates = tl.where(acc == max_value, vocab_offsets, VOCAB)
        token = tl.min(token_candidates, axis=0)
        out_offset = row * BLOCKS_PER_ROW + vocab_block
        tl.store(partial_values + out_offset, max_value)
        tl.store(partial_tokens + out_offset, token)

    @triton.jit
    def _lm_head_top1_reduce_kernel(
        partial_values,
        partial_tokens,
        output_values,
        output_tokens,
        VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        REDUCE_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, REDUCE_BLOCK)
        mask = offsets < BLOCKS_PER_ROW
        values = tl.load(
            partial_values + row * BLOCKS_PER_ROW + offsets,
            mask=mask,
            other=-float("inf"),
        )
        tokens = tl.load(
            partial_tokens + row * BLOCKS_PER_ROW + offsets,
            mask=mask,
            other=VOCAB,
        )
        max_value = tl.max(values, axis=0)
        token_candidates = tl.where(values == max_value, tokens, VOCAB)
        token = tl.min(token_candidates, axis=0)
        tl.store(output_values + row, max_value)
        tl.store(output_tokens + row, token)


def lm_head_top1_out(
    hidden,
    weight,
    output_values,
    output_tokens,
    *,
    partial_values,
    partial_tokens,
    block_vocab: Optional[int] = None,
    block_hidden: int = 64,
):
    """Compute direct LM-head top-1 into caller-owned output tensors."""

    if torch is None or triton is None:
        raise RuntimeError("lm_head_top1_out requires PyTorch and Triton")
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected hidden [batch, hidden] and weight [vocab, hidden]")
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("hidden and weight must be CUDA tensors")
    batch, hidden_size = hidden.shape
    vocab_size, weight_hidden = weight.shape
    if weight_hidden != hidden_size:
        raise ValueError("weight hidden dimension must match hidden")
    if output_values.shape != (batch,) or output_tokens.shape != (batch,):
        raise ValueError("output tensors must have shape [batch]")
    if output_values.dtype != torch.float32 or output_tokens.dtype != torch.int64:
        raise ValueError("output tensors must be float32 and int64")
    if not output_values.is_cuda or not output_tokens.is_cuda:
        raise ValueError("output tensors must be CUDA tensors")
    if not should_use_l20_lm_head_top1(
        int(batch), int(vocab_size), int(hidden_size), top_k=1
    ):
        raise ValueError("shape is outside the L20 LM-head top-1 gate")
    config = lm_head_top1_launch_config(
        int(vocab_size),
        int(hidden_size),
        block_vocab=block_vocab,
        block_hidden=block_hidden,
    )
    expected_workspace = (batch, config.blocks_per_row)
    if partial_values.shape != expected_workspace or partial_tokens.shape != expected_workspace:
        raise ValueError("partial workspaces have the wrong shape")
    if partial_values.dtype != torch.float32 or partial_tokens.dtype != torch.int64:
        raise ValueError("partial workspaces must be float32 and int64")
    if not partial_values.is_cuda or not partial_tokens.is_cuda:
        raise ValueError("partial workspaces must be CUDA tensors")

    _lm_head_top1_partial_kernel[(batch, config.blocks_per_row)](
        hidden,
        weight,
        partial_values,
        partial_tokens,
        HIDDEN=int(hidden_size),
        VOCAB=int(vocab_size),
        BLOCK_VOCAB=config.block_vocab,
        BLOCK_HIDDEN=config.block_hidden,
        BLOCKS_PER_ROW=config.blocks_per_row,
        num_warps=config.num_warps,
        num_stages=3,
    )
    _lm_head_top1_reduce_kernel[(batch,)](
        partial_values,
        partial_tokens,
        output_values,
        output_tokens,
        VOCAB=int(vocab_size),
        BLOCKS_PER_ROW=config.blocks_per_row,
        REDUCE_BLOCK=config.reduce_block,
        num_warps=8,
        num_stages=1,
    )
    return output_values, output_tokens


def lm_head_top1(
    hidden,
    weight,
    *,
    block_vocab: Optional[int] = None,
    block_hidden: int = 64,
):
    """Allocate workspaces and compute direct LM-head top-1."""

    if torch is None:
        raise RuntimeError("lm_head_top1 requires PyTorch")
    batch, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    config = lm_head_top1_launch_config(
        int(vocab_size),
        int(hidden_size),
        block_vocab=block_vocab,
        block_hidden=block_hidden,
    )
    output_values = torch.empty((batch,), device=hidden.device, dtype=torch.float32)
    output_tokens = torch.empty((batch,), device=hidden.device, dtype=torch.int64)
    partial_values = torch.empty(
        (batch, config.blocks_per_row), device=hidden.device, dtype=torch.float32
    )
    partial_tokens = torch.empty(
        (batch, config.blocks_per_row), device=hidden.device, dtype=torch.int64
    )
    return lm_head_top1_out(
        hidden,
        weight,
        output_values,
        output_tokens,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        block_vocab=block_vocab,
        block_hidden=block_hidden,
    )


def full_logits_top1_reference(hidden, weight):
    """Reference top-1 through materialized logits."""

    if torch is None:
        raise RuntimeError("full_logits_top1_reference requires PyTorch")
    logits = hidden.float() @ weight.float().T
    return torch.max(logits, dim=-1)
