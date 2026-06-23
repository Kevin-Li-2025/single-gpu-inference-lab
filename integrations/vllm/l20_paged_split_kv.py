"""L20 paged split-KV GQA decode attention."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def allocate_l20_paged_split_kv_workspace(
    query: torch.Tensor,
    max_seq_len: int,
    *,
    split_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, num_q_heads, head_dim = query.shape
    num_splits = _next_power_of_two(triton.cdiv(max_seq_len, split_size))
    partial_shape = (batch, num_q_heads, num_splits)
    return (
        torch.empty(
            (*partial_shape, head_dim), device=query.device, dtype=query.dtype
        ),
        torch.empty(partial_shape, device=query.device, dtype=torch.float32),
        torch.empty(partial_shape, device=query.device, dtype=torch.float32),
        torch.empty_like(query),
    )


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
def _paged_split_kv_fp8_partial(
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
    k_scale: tl.constexpr,
    v_scale: tl.constexpr,
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
        physical_page = tl.load(
            block_table + batch * bt_stride_b + tl.where(token_mask, logical_page, 0),
            mask=token_mask,
            other=0,
        )
        keys = (
            tl.load(
                key_cache
                + physical_page[:, None] * kc_stride_p
                + page_offset[:, None] * kc_stride_t
                + kv_head * kc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            * k_scale
        )
        scores = tl.sum(keys * q[None, :], axis=1) * scale
        scores = tl.where(token_mask, scores, -float("inf"))
        tile_max = tl.max(scores, axis=0)
        next_max = tl.maximum(max_score, tile_max)
        old_scale = tl.exp(max_score - next_max)
        probabilities = tl.exp(scores - next_max)
        values = (
            tl.load(
                value_cache
                + physical_page[:, None] * vc_stride_p
                + page_offset[:, None] * vc_stride_t
                + kv_head * vc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            * v_scale
        )
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
def _paged_split_kv_partial_page16(
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
    num_splits: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
):
    split = tl.program_id(0)
    head_program = tl.program_id(1)
    batch = head_program // num_q_heads
    q_head = head_program % num_q_heads
    kv_head = q_head // (num_q_heads // num_kv_heads)
    seq_len = tl.load(seq_lens + batch)
    split_start = split * SPLIT_SIZE
    dim = tl.arange(0, head_dim)
    page_offset = tl.arange(0, 16)
    q = tl.load(query + batch * q_stride_b + q_head * q_stride_h + dim).to(
        tl.float32
    )
    scale = 1.0 / tl.sqrt(float(head_dim))
    max_score = -float("inf")
    normalizer = 0.0
    accumulator = tl.zeros((head_dim,), tl.float32)

    for page_start in range(0, SPLIT_SIZE, 16):
        token_start = split_start + page_start
        token = token_start + page_offset
        token_mask = token < tl.minimum(split_start + SPLIT_SIZE, seq_len)
        logical_page = token_start // 16
        physical_page = tl.load(
            block_table + batch * bt_stride_b + logical_page,
            mask=token_start < seq_len,
            other=0,
        )
        keys = tl.load(
            key_cache
            + physical_page * kc_stride_p
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
            + physical_page * vc_stride_p
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
def _paged_split_kv_partial_gqa2(
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
    kv_program = tl.program_id(1)
    batch = kv_program // num_kv_heads
    kv_head = kv_program % num_kv_heads
    q_head0 = kv_head * 2
    q_head1 = q_head0 + 1
    seq_len = tl.load(seq_lens + batch)
    split_start = split * SPLIT_SIZE
    dim = tl.arange(0, head_dim)
    q0 = tl.load(query + batch * q_stride_b + q_head0 * q_stride_h + dim).to(
        tl.float32
    )
    q1 = tl.load(query + batch * q_stride_b + q_head1 * q_stride_h + dim).to(
        tl.float32
    )
    scale = 1.0 / tl.sqrt(float(head_dim))
    max0 = -float("inf")
    max1 = -float("inf")
    sum0 = 0.0
    sum1 = 0.0
    acc0 = tl.zeros((head_dim,), tl.float32)
    acc1 = tl.zeros((head_dim,), tl.float32)

    for offset in range(0, SPLIT_SIZE, BLOCK_T):
        token = split_start + offset + tl.arange(0, BLOCK_T)
        token_mask = token < tl.minimum(split_start + SPLIT_SIZE, seq_len)
        logical_page = token // page_size
        page_offset = token % page_size
        physical_page = tl.load(
            block_table
            + batch * bt_stride_b
            + tl.where(token_mask, logical_page, 0),
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
        score0 = tl.sum(keys * q0[None, :], axis=1) * scale
        score1 = tl.sum(keys * q1[None, :], axis=1) * scale
        score0 = tl.where(token_mask, score0, -float("inf"))
        score1 = tl.where(token_mask, score1, -float("inf"))
        tile_max0 = tl.max(score0, axis=0)
        tile_max1 = tl.max(score1, axis=0)
        next_max0 = tl.maximum(max0, tile_max0)
        next_max1 = tl.maximum(max1, tile_max1)
        old_scale0 = tl.exp(max0 - next_max0)
        old_scale1 = tl.exp(max1 - next_max1)
        probability0 = tl.exp(score0 - next_max0)
        probability1 = tl.exp(score1 - next_max1)
        values = tl.load(
            value_cache
            + physical_page[:, None] * vc_stride_p
            + page_offset[:, None] * vc_stride_t
            + kv_head * vc_stride_h
            + dim[None, :],
            mask=token_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        acc0 = acc0 * old_scale0 + tl.sum(probability0[:, None] * values, axis=0)
        acc1 = acc1 * old_scale1 + tl.sum(probability1[:, None] * values, axis=0)
        sum0 = sum0 * old_scale0 + tl.sum(probability0, axis=0)
        sum1 = sum1 * old_scale1 + tl.sum(probability1, axis=0)
        max0 = next_max0
        max1 = next_max1

    valid_split = split_start < seq_len
    head_program0 = batch * num_q_heads + q_head0
    head_program1 = head_program0 + 1
    index0 = head_program0 * num_splits + split
    index1 = head_program1 * num_splits + split
    tl.store(
        partial_output + index0 * head_dim + dim,
        tl.where(valid_split, acc0, 0.0),
    )
    tl.store(
        partial_output + index1 * head_dim + dim,
        tl.where(valid_split, acc1, 0.0),
    )
    tl.store(partial_max + index0, tl.where(valid_split, max0, -float("inf")))
    tl.store(partial_max + index1, tl.where(valid_split, max1, -float("inf")))
    tl.store(partial_sum + index0, tl.where(valid_split, sum0, 0.0))
    tl.store(partial_sum + index1, tl.where(valid_split, sum1, 0.0))


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
    workspace: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
    page_granular: bool = False,
    grouped_gqa: bool = False,
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
    num_splits = _next_power_of_two(triton.cdiv(max_seq_len, split_size))
    if num_splits > 16:
        raise RuntimeError("supports at most 16 split-KV partitions")
    partial_shape = (batch, num_q_heads, num_splits)
    if workspace is None:
        workspace = allocate_l20_paged_split_kv_workspace(
            query, max_seq_len, split_size=split_size
        )
    partial_output, partial_max, partial_sum, output = workspace
    if (
        partial_output.shape != (*partial_shape, head_dim)
        or partial_max.shape != partial_shape
        or partial_sum.shape != partial_shape
        or output.shape != query.shape
    ):
        raise RuntimeError("paged split-KV workspace shape does not match the request")
    use_grouped_gqa = grouped_gqa and num_q_heads == 2 * num_kv_heads
    if use_grouped_gqa:
        partial_kernel = _paged_split_kv_partial_gqa2
        partial_grid = (num_splits, batch * num_kv_heads)
    elif page_size == 16 and page_granular:
        partial_kernel = _paged_split_kv_partial_page16
        partial_grid = (num_splits, batch * num_q_heads)
    else:
        partial_kernel = _paged_split_kv_partial
        partial_grid = (num_splits, batch * num_q_heads)
    partial_kernel[partial_grid](
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
        num_splits=num_splits,
        SPLIT_SIZE=split_size,
        num_warps=4,
        num_stages=1,
        **(
            {"page_size": page_size, "BLOCK_T": 32}
            if use_grouped_gqa
            else (
                {}
                if page_size == 16 and page_granular
                else {"page_size": page_size, "BLOCK_T": 32}
            )
        ),
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


def l20_paged_split_kv_attention_fp8(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    k_scale: float,
    v_scale: float,
    max_seq_len: int | None = None,
    split_size: int = 512,
    output: torch.Tensor | None = None,
    workspace: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ] | None = None,
) -> torch.Tensor:
    if torch.cuda.get_device_capability(query.device) != (8, 9):
        raise RuntimeError("requires an SM89 GPU")
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.shape != key_cache.shape:
        raise RuntimeError("expected Q=[B,H,D], cache=[pages,page,Hkv,D]")
    if (
        key_cache.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}
        or value_cache.dtype != key_cache.dtype
    ):
        raise RuntimeError("key/value cache must use a torch FP8 dtype")
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
    if max_seq_len is None:
        max_seq_len = int(seq_lens.max().item())
    num_splits = _next_power_of_two(triton.cdiv(max_seq_len, split_size))
    if num_splits > 16:
        raise RuntimeError("supports at most 16 split-KV partitions")
    partial_shape = (batch, num_q_heads, num_splits)
    if workspace is None:
        workspace = allocate_l20_paged_split_kv_workspace(
            query, max_seq_len, split_size=split_size
        )
    partial_output, partial_max, partial_sum, workspace_output = workspace
    if output is None:
        output = workspace_output
    if (
        partial_output.shape != (*partial_shape, head_dim)
        or partial_max.shape != partial_shape
        or partial_sum.shape != partial_shape
        or output.shape != query.shape
    ):
        raise RuntimeError("paged split-KV workspace shape does not match the request")
    _paged_split_kv_fp8_partial[(num_splits, batch * num_q_heads)](
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
        k_scale=float(k_scale),
        v_scale=float(v_scale),
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


def should_use_l20_paged_fp8_split_kv(batch: int, max_seq_len: int) -> bool:
    # The microbenchmark win at batch 8/context 4096 did not survive the first
    # vLLM FP8 KV-cache ITL smoke, so production dispatch remains disabled.
    del batch, max_seq_len
    return False
