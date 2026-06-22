"""L20 paged split-KV GQA decode attention."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _paged_split_kv_partial(
    query,
    key_cache,
    value_cache,
    block_table,
    seq_lens,
    partial_output,
    partial_max,
    partial_sum,
    q_stride_b,
    q_stride_h,
    kc_stride_p,
    kc_stride_t,
    kc_stride_h,
    vc_stride_p,
    vc_stride_t,
    vc_stride_h,
    bt_stride_b,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    page_size: tl.constexpr,
    num_splits: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    split = tl.program_id(0)
    head_program = tl.program_id(1)
    batch = head_program // num_q_heads
    q_head = head_program % num_q_heads
    kv_head = q_head // (num_q_heads // num_kv_heads)
    seq_len = tl.load(seq_lens + batch)
    split_start = split * SPLIT_SIZE
    dim = tl.arange(0, head_dim)
    q = tl.load(query + batch * q_stride_b + q_head * q_stride_h + dim).to(
        tl.float32
    )
    scale = 1.0 / tl.sqrt(float(head_dim))
    max_score = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((head_dim,), tl.float32)

    for offset in range(0, SPLIT_SIZE, BLOCK_T):
        token = split_start + offset + tl.arange(0, BLOCK_T)
        token_mask = token < tl.minimum(split_start + SPLIT_SIZE, seq_len)
        logical_page = token // page_size
        page_offset = token % page_size
        safe_page = tl.where(token_mask, logical_page, 0)
        physical_page = tl.load(
            block_table + batch * bt_stride_b + safe_page,
            mask=token_mask,
            other=0,
        )
        keys = tl.load(
            key_cache
            + physical_page[:, None] * kc_stride_p
            + page_offset[:, None] * kc_stride_t
            + kv_head * kc_stride_h
            + dim[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(keys * q[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(max_score, tile_max)
        old_scale = tl.exp(max_score - next_max)
        probabilities = tl.exp(scores - next_max)
        values = tl.load(
            value_cache
            + physical_page[:, None] * vc_stride_p
            + page_offset[:, None] * vc_stride_t
            + kv_head * vc_stride_h
            + dim[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        accumulator = (
            accumulator * old_scale
            + tl.sum(probabilities[:, None] * values, axis=0)
        )
        normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
        max_score = next_max

    partial_index = head_program * num_splits + split
    valid_split = split_start < seq_len
    tl.store(
        partial_output + partial_index * head_dim + dim,
        tl.where(valid_split, accumulator, 0.0),
    )
    tl.store(
        partial_max + partial_index,
        tl.where(valid_split, max_score, -float("inf")),
    )
    tl.store(
        partial_sum + partial_index,
        tl.where(valid_split, normalizer, 0.0),
    )


@triton.jit
def _paged_split_kv_reduce(
    partial_output,
    partial_max,
    partial_sum,
    output,
    out_stride_b,
    out_stride_h,
    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    num_splits: tl.constexpr,
):
    head_program = tl.program_id(0)
    batch = head_program // num_q_heads
    q_head = head_program % num_q_heads
    dim = tl.arange(0, head_dim)
    split = tl.arange(0, num_splits)
    base = head_program * num_splits
    maxima = tl.load(partial_max + base + split)
    global_max = tl.max(maxima, axis=0)
    correction = tl.exp(maxima - global_max)
    denominator = tl.sum(tl.load(partial_sum + base + split) * correction, axis=0)
    partials = tl.load(
        partial_output + (base + split[:, None]) * head_dim + dim[None, :]
    )
    numerator = tl.sum(partials * correction[:, None], axis=0)
    tl.store(
        output + batch * out_stride_b + q_head * out_stride_h + dim,
        numerator / denominator,
    )


def l20_paged_split_kv_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    split_size: int = 512,
) -> torch.Tensor:
    if torch.cuda.get_device_capability(query.device) != (8, 9):
        raise RuntimeError("requires an SM89 GPU")
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise RuntimeError("expected Q=[B,H,D], cache=[pages,page,Hkv,D]")
    batch, num_q_heads, head_dim = query.shape
    _, page_size, num_kv_heads, cache_dim = key_cache.shape
    if (
        head_dim != 128
        or cache_dim != head_dim
        or num_q_heads % num_kv_heads
        or block_table.shape[0] != batch
        or seq_lens.numel() != batch
    ):
        raise RuntimeError("requires compatible head_dim=128 GQA tensors")
    max_seq_len = int(seq_lens.max().item())
    num_splits = triton.cdiv(max_seq_len, split_size)
    if num_splits > 16:
        raise RuntimeError("supports at most 16 split-KV partitions")
    partial_shape = (batch, num_q_heads, num_splits)
    partial_output = torch.empty(
        (*partial_shape, head_dim), device=query.device, dtype=torch.float32
    )
    partial_max = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty_like(query)
    _paged_split_kv_partial[(num_splits, batch * num_q_heads)](
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partial_output,
        partial_max,
        partial_sum,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_table.stride(0),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        num_splits=num_splits,
        SPLIT_SIZE=split_size,
        BLOCK_T=32,
        num_warps=4,
        num_stages=1,
    )
    _paged_split_kv_reduce[(batch * num_q_heads,)](
        partial_output,
        partial_max,
        partial_sum,
        output,
        output.stride(0),
        output.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        num_splits=num_splits,
        num_warps=4,
        num_stages=1,
    )
    return output


def should_use_l20_paged_split_kv(batch: int, max_seq_len: int) -> bool:
    # Correctness is established, but the current Triton implementation does
    # not beat FlashInfer's production paged-decode kernel on L20.
    del batch, max_seq_len
    return False
