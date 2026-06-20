"""Triton RoPE + KV-cache write kernels for L20-style Ada GPUs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

from l20_stack.ops.triton_rmsnorm import next_power_of_2


try:  # pragma: no cover - optional GPU dependency
    import torch
except ImportError:  # pragma: no cover - optional GPU dependency
    torch = None

try:  # pragma: no cover - optional GPU dependency
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional GPU dependency
    triton = None
    tl = None


@dataclass(frozen=True)
class RopeKvLaunchConfig:
    head_dim: int
    rotary_dim: int
    block_size: int
    num_warps: int
    num_stages: int
    sm_target: str
    rationale: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def rope_kv_launch_config(head_dim: int, rotary_dim: int | None = None) -> RopeKvLaunchConfig:
    if head_dim <= 0:
        raise ValueError("head_dim must be positive")
    if rotary_dim is None:
        rotary_dim = head_dim
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError("rotary_dim must be a positive even value <= head_dim")

    block_size = next_power_of_2(head_dim)
    if block_size > 256:
        raise ValueError("single-pass RoPE KV write supports head_dim <= 256")
    num_warps = 4 if block_size >= 128 else 2
    return RopeKvLaunchConfig(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        block_size=block_size,
        num_warps=num_warps,
        num_stages=1,
        sm_target="sm_89",
        rationale=(
            "one Triton program per token/head, half-rotation RoPE, contiguous "
            "KV-cache write, and small blocks for L20 decode occupancy"
        ),
    )


if triton is not None:  # pragma: no cover - requires Triton

    @triton.jit
    def _rope_kv_cache_write_kernel(
        k,
        v,
        cos,
        sin,
        cache_positions,
        k_cache,
        v_cache,
        tokens: tl.constexpr,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        block: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block)
        mask = offsets < head_dim
        half_rotary = rotary_dim // 2
        base = (token * kv_heads + head) * head_dim + offsets
        cache_token = tl.load(cache_positions + token)
        cache_base = (cache_token * kv_heads + head) * head_dim + offsets

        k_row = tl.load(k + base, mask=mask, other=0.0)
        v_row = tl.load(v + base, mask=mask, other=0.0)
        k_out = k_row

        first_half = offsets < half_rotary
        second_half = (offsets >= half_rotary) & (offsets < rotary_dim)
        pair_offsets = offsets % half_rotary
        cos_row = tl.load(cos + token * half_rotary + pair_offsets, mask=offsets < rotary_dim)
        sin_row = tl.load(sin + token * half_rotary + pair_offsets, mask=offsets < rotary_dim)
        k_pair = tl.load(
            k + (token * kv_heads + head) * head_dim + pair_offsets + half_rotary,
            mask=first_half,
            other=0.0,
        )
        k_pair2 = tl.load(
            k + (token * kv_heads + head) * head_dim + pair_offsets,
            mask=second_half,
            other=0.0,
        )
        rotated_first = k_row.to(tl.float32) * cos_row - k_pair.to(tl.float32) * sin_row
        rotated_second = k_row.to(tl.float32) * cos_row + k_pair2.to(tl.float32) * sin_row
        k_out = tl.where(first_half, rotated_first, k_out)
        k_out = tl.where(second_half, rotated_second, k_out)

        tl.store(k_cache + cache_base, k_out, mask=mask)
        tl.store(v_cache + cache_base, v_row, mask=mask)


def rope_kv_reference(k, v, cos, sin, cache_positions, k_cache, v_cache):
    """PyTorch reference for half-rotation RoPE and contiguous KV-cache writes."""

    if torch is None:
        raise RuntimeError("rope_kv_reference requires torch")
    if k.shape != v.shape:
        raise ValueError("k and v must have the same shape")
    tokens, kv_heads, head_dim = k.shape
    half_rotary = cos.shape[1]
    rotary_dim = half_rotary * 2
    if sin.shape != cos.shape or cos.shape[0] != tokens:
        raise ValueError("cos and sin must be [tokens, rotary_dim / 2]")
    if rotary_dim > head_dim:
        raise ValueError("rotary_dim cannot exceed head_dim")

    rotated = k.clone()
    first = k[:, :, :half_rotary].float()
    second = k[:, :, half_rotary:rotary_dim].float()
    cos_view = cos[:, None, :].float()
    sin_view = sin[:, None, :].float()
    rotated[:, :, :half_rotary] = (first * cos_view - second * sin_view).to(k.dtype)
    rotated[:, :, half_rotary:rotary_dim] = (
        second * cos_view + first * sin_view
    ).to(k.dtype)
    k_cache[cache_positions] = rotated
    v_cache[cache_positions] = v
    return k_cache, v_cache


def paged_rope_kv_reference(
    k, v, cos, sin, sequence_ids, positions, block_table, k_cache, v_cache
):
    """PyTorch reference for RoPE writes through a logical-to-physical block table."""

    if torch is None:
        raise RuntimeError("paged_rope_kv_reference requires torch")
    block_size = k_cache.shape[1]
    logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
    physical_blocks = block_table[sequence_ids, logical_blocks]
    block_offsets = positions % block_size

    rotated = k.clone()
    half_rotary = cos.shape[1]
    rotary_dim = half_rotary * 2
    first = k[:, :, :half_rotary].float()
    second = k[:, :, half_rotary:rotary_dim].float()
    cos_view = cos[:, None, :].float()
    sin_view = sin[:, None, :].float()
    rotated[:, :, :half_rotary] = (first * cos_view - second * sin_view).to(k.dtype)
    rotated[:, :, half_rotary:rotary_dim] = (
        second * cos_view + first * sin_view
    ).to(k.dtype)
    k_cache[physical_blocks, block_offsets] = rotated
    v_cache[physical_blocks, block_offsets] = v
    return k_cache, v_cache


if triton is not None:  # pragma: no cover - requires Triton

    @triton.jit
    def _paged_rope_kv_cache_write_kernel(
        k,
        v,
        cos,
        sin,
        sequence_ids,
        positions,
        block_table,
        k_cache,
        v_cache,
        kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        rotary_dim: tl.constexpr,
        max_blocks_per_sequence: tl.constexpr,
        block_size: tl.constexpr,
        block: tl.constexpr,
    ):
        token = tl.program_id(0)
        head = tl.program_id(1)
        offsets = tl.arange(0, block)
        mask = offsets < head_dim
        half_rotary = rotary_dim // 2

        sequence = tl.load(sequence_ids + token)
        position = tl.load(positions + token)
        logical_block = position // block_size
        physical_block = tl.load(
            block_table + sequence * max_blocks_per_sequence + logical_block
        )
        block_offset = position % block_size

        input_base = (token * kv_heads + head) * head_dim + offsets
        cache_base = (
            ((physical_block * block_size + block_offset) * kv_heads + head)
            * head_dim
            + offsets
        )
        k_row = tl.load(k + input_base, mask=mask, other=0.0)
        v_row = tl.load(v + input_base, mask=mask, other=0.0)

        first_half = offsets < half_rotary
        second_half = (offsets >= half_rotary) & (offsets < rotary_dim)
        pair_offsets = offsets % half_rotary
        trig_mask = offsets < rotary_dim
        cos_row = tl.load(
            cos + token * half_rotary + pair_offsets, mask=trig_mask, other=0.0
        )
        sin_row = tl.load(
            sin + token * half_rotary + pair_offsets, mask=trig_mask, other=0.0
        )
        k_second = tl.load(
            k + (token * kv_heads + head) * head_dim + pair_offsets + half_rotary,
            mask=first_half,
            other=0.0,
        )
        k_first = tl.load(
            k + (token * kv_heads + head) * head_dim + pair_offsets,
            mask=second_half,
            other=0.0,
        )
        rotated_first = k_row.to(tl.float32) * cos_row - k_second.to(tl.float32) * sin_row
        rotated_second = k_row.to(tl.float32) * cos_row + k_first.to(tl.float32) * sin_row
        k_out = tl.where(first_half, rotated_first, k_row)
        k_out = tl.where(second_half, rotated_second, k_out)
        tl.store(k_cache + cache_base, k_out, mask=mask)
        tl.store(v_cache + cache_base, v_row, mask=mask)


def rope_kv_cache_write_triton(k, v, cos, sin, cache_positions, k_cache, v_cache):
    """Fuse RoPE on K with contiguous K/V cache writes.

    Shapes:
    - k, v: [tokens, kv_heads, head_dim]
    - cos, sin: [tokens, rotary_dim / 2]
    - cache_positions: [tokens]
    - k_cache, v_cache: [cache_tokens, kv_heads, head_dim]
    """

    if torch is None or triton is None:
        raise RuntimeError("rope_kv_cache_write_triton requires torch and triton")
    if k.dim() != 3 or v.shape != k.shape:
        raise ValueError("k and v must be matching [tokens, kv_heads, head_dim] tensors")
    if cos.dim() != 2 or sin.shape != cos.shape or cos.shape[0] != k.shape[0]:
        raise ValueError("cos and sin must be [tokens, rotary_dim / 2]")
    if cache_positions.dim() != 1 or cache_positions.numel() != k.shape[0]:
        raise ValueError("cache_positions must have one entry per token")
    if cache_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("cache_positions must use int32 or int64 indices")
    if k_cache.dim() != 3 or v_cache.shape != k_cache.shape:
        raise ValueError("k_cache and v_cache must be matching 3D tensors")
    if k_cache.shape[1:] != k.shape[1:]:
        raise ValueError("cache tensors must match kv_heads and head_dim")
    if not all(tensor.is_cuda for tensor in (k, v, cos, sin, cache_positions, k_cache, v_cache)):
        raise ValueError("all tensors must be CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in (k, v, cos, sin, k_cache, v_cache)):
        raise ValueError("k, v, cos, sin, and caches must be contiguous")
    if not cache_positions.is_contiguous():
        raise ValueError("cache_positions must be contiguous")
    if not (k.dtype == v.dtype == k_cache.dtype == v_cache.dtype):
        raise ValueError("k, v, and cache tensors must use the same dtype")
    if cos.dtype != sin.dtype:
        raise ValueError("cos and sin must use the same dtype")

    tokens, kv_heads, head_dim = k.shape
    rotary_dim = int(cos.shape[1]) * 2
    config = rope_kv_launch_config(int(head_dim), rotary_dim)
    _rope_kv_cache_write_kernel[(tokens, kv_heads)](
        k,
        v,
        cos,
        sin,
        cache_positions,
        k_cache,
        v_cache,
        tokens,
        kv_heads,
        head_dim,
        rotary_dim,
        config.block_size,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return k_cache, v_cache


def paged_rope_kv_cache_write_triton(
    k, v, cos, sin, sequence_ids, positions, block_table, k_cache, v_cache
):
    """Fuse K RoPE and NHD paged KV writes using a two-dimensional block table."""

    if torch is None or triton is None:
        raise RuntimeError("paged_rope_kv_cache_write_triton requires torch and triton")
    if k.dim() != 3 or v.shape != k.shape:
        raise ValueError("k and v must be matching [tokens, kv_heads, head_dim] tensors")
    tokens, kv_heads, head_dim = k.shape
    if cos.dim() != 2 or sin.shape != cos.shape or cos.shape[0] != tokens:
        raise ValueError("cos and sin must be [tokens, rotary_dim / 2]")
    if sequence_ids.shape != (tokens,) or positions.shape != (tokens,):
        raise ValueError("sequence_ids and positions must have one entry per token")
    if block_table.dim() != 2:
        raise ValueError("block_table must be [sequences, max_blocks_per_sequence]")
    if k_cache.dim() != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("paged caches must be matching [blocks, block_size, heads, dim]")
    if k_cache.shape[2:] != (kv_heads, head_dim):
        raise ValueError("paged caches must match kv_heads and head_dim")
    tensors = (k, v, cos, sin, sequence_ids, positions, block_table, k_cache, v_cache)
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("all tensors must be CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all tensors must be contiguous")
    if any(index.dtype not in (torch.int32, torch.int64) for index in (sequence_ids, positions, block_table)):
        raise ValueError("sequence_ids, positions, and block_table must use integer indices")
    if not (k.dtype == v.dtype == k_cache.dtype == v_cache.dtype):
        raise ValueError("k, v, and cache tensors must use the same dtype")
    if cos.dtype != sin.dtype:
        raise ValueError("cos and sin must use the same dtype")

    rotary_dim = int(cos.shape[1]) * 2
    config = rope_kv_launch_config(int(head_dim), rotary_dim)
    _paged_rope_kv_cache_write_kernel[(tokens, kv_heads)](
        k,
        v,
        cos,
        sin,
        sequence_ids,
        positions,
        block_table,
        k_cache,
        v_cache,
        kv_heads,
        head_dim,
        rotary_dim,
        int(block_table.shape[1]),
        int(k_cache.shape[1]),
        config.block_size,
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return k_cache, v_cache
