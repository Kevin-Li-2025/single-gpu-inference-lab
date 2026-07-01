"""Experimental L20 LM-head epilogue sampling primitive.

This module is the first FlashSampling-style boundary probe in this repo: it
computes LM-head logits tile by tile and keeps only the winning token candidate,
instead of materializing a full ``[batch, vocab]`` logits tensor.  It supports a
greedy mode and an exact Gumbel-max mode for full-vocabulary categorical
sampling.  Top-k, top-p, penalties, logprobs, and structured-output semantics
must still stay on the baseline vLLM path.
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
class LMHeadSamplingLaunchConfig:
    block_vocab: int
    block_hidden: int
    block_batch: int
    blocks_per_row: int
    reduce_block: int
    num_warps: int
    num_stages: int
    strategy: str

    def to_dict(self):
        return asdict(self)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def lm_head_sampling_launch_config(
    batch: int,
    vocab_size: int,
    hidden_size: int,
    *,
    block_vocab: Optional[int] = None,
    block_hidden: Optional[int] = None,
) -> LMHeadSamplingLaunchConfig:
    """Return the launch policy for the LM-head sampling boundary.

    The policy started as an L20 probe, but the batch tile is padded to 16 so
    the same Triton dot path compiles on A100/Triton 3.4 as well.
    """

    if batch <= 0 or vocab_size <= 0 or hidden_size <= 0:
        raise ValueError("batch, vocab_size, and hidden_size must be positive")
    selected_block_vocab = block_vocab if block_vocab is not None else (64 if batch > 1 else 32)
    if block_hidden is not None:
        selected_block_hidden = block_hidden
    elif hidden_size % 256 == 0:
        selected_block_hidden = 256
    elif hidden_size % 128 == 0:
        selected_block_hidden = 128
    else:
        selected_block_hidden = 64
    if selected_block_vocab not in {16, 32, 64, 128, 256}:
        raise ValueError("block_vocab must be one of 16, 32, 64, 128, or 256")
    if selected_block_hidden not in {32, 64, 128, 256}:
        raise ValueError("block_hidden must be one of 32, 64, 128, or 256")
    if hidden_size % selected_block_hidden != 0:
        raise ValueError("hidden_size must be divisible by block_hidden")
    blocks_per_row = (vocab_size + selected_block_vocab - 1) // selected_block_vocab
    selected_num_stages = (
        2 if selected_block_vocab * selected_block_hidden >= 32_768 else 3
    )
    return LMHeadSamplingLaunchConfig(
        block_vocab=selected_block_vocab,
        block_hidden=selected_block_hidden,
        # Triton 3.4 requires tl.dot's N dimension to be at least 16. The
        # kernel masks padded batch lanes, so small decode batches still store
        # only the real rows.
        block_batch=16,
        blocks_per_row=blocks_per_row,
        reduce_block=next_power_of_2(blocks_per_row),
        num_warps=8 if selected_block_vocab >= 64 else 4,
        num_stages=selected_num_stages,
        strategy="two_stage_lm_head_gumbel_max",
    )


def should_use_l20_lm_head_sampling(
    batch: int,
    vocab_size: int,
    hidden_size: int,
    *,
    top_k: int = -1,
    top_p: float = 1.0,
) -> bool:
    """Conservative shape gate for the first L20 FlashSampling-style primitive."""

    if batch <= 0 or vocab_size <= 0 or hidden_size <= 0:
        return False
    if batch > 4 or vocab_size > 262_144 or hidden_size > 8192:
        return False
    if hidden_size % 64 != 0:
        return False
    if top_k not in (-1, 0, 1):
        return False
    return top_p == 1.0


if triton is not None:  # pragma: no cover - requires CUDA

    @triton.jit
    def _lm_head_sampling_partial_kernel(
        hidden,
        weight,
        seeds,
        positions,
        partial_values,
        partial_tokens,
        BATCH: tl.constexpr,
        HIDDEN: tl.constexpr,
        VOCAB: tl.constexpr,
        BLOCK_BATCH: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCK_HIDDEN: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        USE_GUMBEL: tl.constexpr,
        HAS_POSITIONS: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        batch_block = tl.program_id(0)
        vocab_block = tl.program_id(1)
        batch_offsets = batch_block * BLOCK_BATCH + tl.arange(0, BLOCK_BATCH)
        vocab_offsets = vocab_block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        hidden_offsets = tl.arange(0, BLOCK_HIDDEN)
        acc = tl.zeros((BLOCK_VOCAB, BLOCK_BATCH), dtype=tl.float32)

        for hidden_start in range(0, HIDDEN, BLOCK_HIDDEN):
            h_offsets = hidden_start + hidden_offsets
            h = tl.load(
                hidden + h_offsets[:, None] + batch_offsets[None, :] * HIDDEN,
                mask=batch_offsets[None, :] < BATCH,
                other=0.0,
            )
            w = tl.load(
                weight + vocab_offsets[:, None] * HIDDEN + h_offsets[None, :],
                mask=vocab_offsets[:, None] < VOCAB,
                other=0.0,
            )
            acc += tl.dot(w, h, out_dtype=tl.float32)

        valid = (vocab_offsets[:, None] < VOCAB) & (batch_offsets[None, :] < BATCH)
        if TEMPERATURE != 1.0:
            acc = acc / TEMPERATURE
        if USE_GUMBEL:
            seed_values = tl.load(seeds + batch_offsets, mask=batch_offsets < BATCH, other=0)
            seed_values = seed_values.to(tl.uint32)
            if HAS_POSITIONS:
                pos_values = tl.load(positions + batch_offsets, mask=batch_offsets < BATCH, other=0)
                pos_values = pos_values.to(tl.uint32)
            else:
                pos_values = tl.zeros((BLOCK_BATCH,), dtype=tl.uint32)

            x = seed_values[None, :]
            x = x ^ (vocab_offsets[:, None].to(tl.uint32) * 1664525)
            x = x ^ (batch_offsets[None, :].to(tl.uint32) * 1013904223)
            x = x ^ (pos_values[None, :] * 747796405)
            x = x ^ (x << 13)
            x = x ^ (x >> 17)
            x = x ^ (x << 5)
            mantissa = x & 0x00FFFFFF
            uniform = mantissa.to(tl.float32) * 0.000000059604644775390625
            uniform = tl.maximum(uniform, 0.000000059604644775390625)
            uniform = tl.minimum(uniform, 0.9999999403953552)
            gumbel = -tl.log(-tl.log(uniform))
            acc += gumbel

        acc = tl.where(valid, acc, -float("inf"))
        max_values = tl.max(acc, axis=0)
        token_candidates = tl.where(acc == max_values[None, :], vocab_offsets[:, None], VOCAB)
        tokens = tl.min(token_candidates, axis=0)
        out_offsets = batch_offsets * BLOCKS_PER_ROW + vocab_block
        out_mask = batch_offsets < BATCH
        tl.store(partial_values + out_offsets, max_values, mask=out_mask)
        tl.store(partial_tokens + out_offsets, tokens, mask=out_mask)

    @triton.jit
    def _lm_head_sampling_reduce_kernel(
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


def lm_head_sample_out(
    hidden,
    weight,
    output_values,
    output_tokens,
    *,
    partial_values,
    partial_tokens,
    seeds=None,
    positions=None,
    use_gumbel: bool = True,
    temperature: float = 1.0,
    block_vocab: Optional[int] = None,
    block_hidden: Optional[int] = None,
):
    """Compute greedy or exact Gumbel-max LM-head sampling into caller tensors."""

    if torch is None or triton is None:
        raise RuntimeError("lm_head_sample_out requires PyTorch and Triton")
    if hidden.ndim != 2 or weight.ndim != 2:
        raise ValueError("expected hidden [batch, hidden] and weight [vocab, hidden]")
    if not hidden.is_cuda or not weight.is_cuda:
        raise ValueError("hidden and weight must be CUDA tensors")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
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
    if not should_use_l20_lm_head_sampling(int(batch), int(vocab_size), int(hidden_size)):
        raise ValueError("shape is outside the L20 LM-head sampling gate")
    if use_gumbel:
        if seeds is None:
            raise ValueError("seeds are required for Gumbel-max sampling")
        if seeds.shape != (batch,):
            raise ValueError("seeds must have shape [batch]")
        if seeds.dtype not in {torch.int32, torch.int64, torch.uint8, torch.int16}:
            raise ValueError("seeds must use an integer dtype")
        if not seeds.is_cuda:
            raise ValueError("seeds must be a CUDA tensor")
    else:
        if seeds is None:
            seeds = torch.empty((batch,), device=hidden.device, dtype=torch.int32)
    has_positions = positions is not None
    if positions is None:
        positions = seeds
    elif positions.shape != (batch,):
        raise ValueError("positions must have shape [batch]")
    elif positions.dtype not in {torch.int32, torch.int64, torch.uint8, torch.int16}:
        raise ValueError("positions must use an integer dtype")
    elif not positions.is_cuda:
        raise ValueError("positions must be a CUDA tensor")

    config = lm_head_sampling_launch_config(
        int(batch),
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

    grid = (triton.cdiv(int(batch), config.block_batch), config.blocks_per_row)
    _lm_head_sampling_partial_kernel[grid](
        hidden,
        weight,
        seeds,
        positions,
        partial_values,
        partial_tokens,
        BATCH=int(batch),
        HIDDEN=int(hidden_size),
        VOCAB=int(vocab_size),
        BLOCK_BATCH=config.block_batch,
        BLOCK_VOCAB=config.block_vocab,
        BLOCK_HIDDEN=config.block_hidden,
        BLOCKS_PER_ROW=config.blocks_per_row,
        USE_GUMBEL=bool(use_gumbel),
        HAS_POSITIONS=bool(has_positions),
        TEMPERATURE=float(temperature),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    _lm_head_sampling_reduce_kernel[(batch,)](
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


def lm_head_sample(
    hidden,
    weight,
    *,
    seeds=None,
    positions=None,
    use_gumbel: bool = True,
    temperature: float = 1.0,
    block_vocab: Optional[int] = None,
    block_hidden: Optional[int] = None,
):
    """Allocate workspaces and run the experimental LM-head sampling path."""

    if torch is None:
        raise RuntimeError("lm_head_sample requires PyTorch")
    batch, hidden_size = hidden.shape
    vocab_size = weight.shape[0]
    config = lm_head_sampling_launch_config(
        int(batch),
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
    return lm_head_sample_out(
        hidden,
        weight,
        output_values,
        output_tokens,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        seeds=seeds,
        positions=positions,
        use_gumbel=use_gumbel,
        temperature=temperature,
        block_vocab=block_vocab,
        block_hidden=block_hidden,
    )
