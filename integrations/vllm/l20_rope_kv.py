"""SM89 fused RoPE and paged KV-cache update for vLLM."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _l20_rope_kv_kernel(
    query,
    key,
    value,
    positions,
    cos_sin_cache,
    slot_mapping,
    key_cache,
    value_cache,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    v_stride_t,
    v_stride_h,
    kc_stride_b,
    kc_stride_s,
    kc_stride_h,
    vc_stride_b,
    vc_stride_s,
    vc_stride_h,
    cos_stride_t,
    num_tokens: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    cache_block_size: tl.constexpr,
    is_neox: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)
    dim_mask = offsets < head_dim
    rotary_mask = offsets < rotary_dim
    half = rotary_dim // 2
    position = tl.load(positions + token)

    if is_neox:
        pair_offsets = tl.where(offsets < half, offsets + half, offsets - half)
        trig_offsets = offsets % half
        rotate_sign = tl.where(offsets < half, -1.0, 1.0)
    else:
        pair_offsets = tl.where(offsets % 2 == 0, offsets + 1, offsets - 1)
        trig_offsets = offsets // 2
        rotate_sign = tl.where(offsets % 2 == 0, -1.0, 1.0)

    cos = tl.load(
        cos_sin_cache + position * cos_stride_t + trig_offsets,
        mask=rotary_mask,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + position * cos_stride_t + half + trig_offsets,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)

    q_base = token * q_stride_t + head * q_stride_h
    q = tl.load(query + q_base + offsets, mask=dim_mask, other=0.0)
    q_pair = tl.load(
        query + q_base + pair_offsets,
        mask=rotary_mask,
        other=0.0,
    )
    q_out = q.to(tl.float32) * cos + q_pair.to(tl.float32) * sin * rotate_sign
    tl.store(query + q_base + offsets, tl.where(rotary_mask, q_out, q), mask=dim_mask)

    if head < num_kv_heads:
        k_base = token * k_stride_t + head * k_stride_h
        k = tl.load(key + k_base + offsets, mask=dim_mask, other=0.0)
        k_pair = tl.load(
            key + k_base + pair_offsets,
            mask=rotary_mask,
            other=0.0,
        )
        k_out = k.to(tl.float32) * cos + k_pair.to(tl.float32) * sin * rotate_sign
        k_out = tl.where(rotary_mask, k_out, k)
        tl.store(key + k_base + offsets, k_out, mask=dim_mask)

        slot = tl.load(slot_mapping + token)
        valid_slot = slot >= 0
        safe_slot = tl.where(valid_slot, slot, 0)
        physical_block = safe_slot // cache_block_size
        block_offset = safe_slot % cache_block_size
        k_cache_base = (
            physical_block * kc_stride_b
            + block_offset * kc_stride_s
            + head * kc_stride_h
        )
        v_cache_base = (
            physical_block * vc_stride_b
            + block_offset * vc_stride_s
            + head * vc_stride_h
        )
        v = tl.load(
            value + token * v_stride_t + head * v_stride_h + offsets,
            mask=dim_mask,
            other=0.0,
        )
        tl.store(
            key_cache + k_cache_base + offsets,
            k_out,
            mask=dim_mask & valid_slot,
        )
        tl.store(
            value_cache + v_cache_base + offsets,
            v,
            mask=dim_mask & valid_slot,
        )


def l20_rope_and_cache(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if torch.cuda.get_device_capability(query.device) != (8, 9):
        raise RuntimeError("l20_rope_and_cache requires an SM89 GPU")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("l20_rope_and_cache supports FP16 and BF16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise RuntimeError("Q, K, and V must use the same dtype")
    if query.shape[0] > 64:
        raise RuntimeError("l20_rope_and_cache is restricted to at most 64 tokens")
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise RuntimeError("quantized KV cache is not supported")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise RuntimeError("Q, K, and V must be [tokens, heads, head_dim]")
    if key.shape != value.shape or query.shape[0] != key.shape[0]:
        raise RuntimeError("Q, K, and V shapes are incompatible")
    if key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise RuntimeError("cache must use NHD [blocks, block, heads, dim]")
    if key_cache.shape[2:] != key.shape[1:]:
        raise RuntimeError("cache head shape must match K/V")
    if positions.numel() != query.shape[0] or slot_mapping.numel() != query.shape[0]:
        raise RuntimeError("positions and slot_mapping must match token count")

    num_tokens, num_q_heads, head_dim = query.shape
    num_kv_heads = key.shape[1]
    rotary_dim = cos_sin_cache.shape[1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise RuntimeError("invalid cos_sin_cache rotary dimension")
    block_size = triton.next_power_of_2(head_dim)
    if block_size > 256:
        raise RuntimeError("head_dim above 256 is not supported")

    _l20_rope_kv_kernel[(num_tokens, num_q_heads)](
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        slot_mapping,
        key_cache,
        value_cache,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        cos_sin_cache.stride(0),
        num_tokens,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        key_cache.shape[1],
        is_neox,
        BLOCK_SIZE=block_size,
        num_warps=4 if head_dim >= 128 else 2,
        num_stages=1,
    )
