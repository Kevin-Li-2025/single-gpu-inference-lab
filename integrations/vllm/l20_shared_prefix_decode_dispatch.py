"""Opt-in dispatch helper for L20 shared-prefix decode attention."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Sequence

import torch
from vllm.v1.attention.ops.l20_decode_attention import (
    shared_paged_prefix_suffix_gqa_decode_attention,
)


def l20_shared_prefix_decode_enabled() -> bool:
    return os.getenv("VLLM_ENABLE_L20_SHARED_PREFIX_DECODE", "0") == "1"


def _to_int_list(values: torch.Tensor | Sequence[int]) -> list[int]:
    if isinstance(values, torch.Tensor):
        return [int(item) for item in values.detach().cpu().tolist()]
    return [int(item) for item in values]


def shared_prefix_block_key(
    block_table_row: torch.Tensor,
    prefix_length: int,
    *,
    page_size: int = 16,
) -> tuple[int, tuple[int, ...]]:
    pages = (int(prefix_length) + page_size - 1) // page_size
    blocks = block_table_row[:pages].detach().cpu().tolist()
    return int(prefix_length), tuple(int(block) for block in blocks)


def find_l20_shared_prefix_groups(
    block_tables: torch.Tensor,
    prefix_lengths: torch.Tensor | Sequence[int],
    *,
    min_batch: int = 8,
    min_prefix_length: int = 4096,
    page_size: int = 16,
) -> list[list[int]]:
    """Group request indices by identical prefix length and block chain."""
    if block_tables.ndim != 2:
        raise ValueError("block_tables must have shape [B, max_pages]")
    lengths = _to_int_list(prefix_lengths)
    if len(lengths) != block_tables.shape[0]:
        raise ValueError("prefix_lengths must have one entry per request")
    groups: dict[tuple[int, tuple[int, ...]], list[int]] = defaultdict(list)
    for index, prefix_length in enumerate(lengths):
        if prefix_length < min_prefix_length:
            continue
        key = shared_prefix_block_key(
            block_tables[index],
            prefix_length,
            page_size=page_size,
        )
        groups[key].append(index)
    return [indices for indices in groups.values() if len(indices) >= min_batch]


def should_dispatch_l20_shared_prefix_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    suffix_key: torch.Tensor,
    block_tables: torch.Tensor,
    prefix_lengths: torch.Tensor | Sequence[int],
    *,
    page_size: int = 16,
    min_batch: int = 8,
    min_prefix_length: int = 4096,
) -> bool:
    if not l20_shared_prefix_decode_enabled():
        return False
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key_cache.dtype != query.dtype or suffix_key.dtype != query.dtype:
        return False
    if query.ndim != 3 or key_cache.ndim != 4 or suffix_key.ndim != 4:
        return False
    if block_tables.ndim != 2 or block_tables.shape[0] != query.shape[0]:
        return False
    if query.shape[0] < min_batch:
        return False
    if query.shape[-1] != 128 or key_cache.shape[-1] != 128:
        return False
    if key_cache.shape[1] != page_size:
        return False
    if suffix_key.shape[0] != query.shape[0] or suffix_key.shape[2] != key_cache.shape[2]:
        return False
    groups = find_l20_shared_prefix_groups(
        block_tables,
        prefix_lengths,
        min_batch=min_batch,
        min_prefix_length=min_prefix_length,
        page_size=page_size,
    )
    return len(groups) == 1 and len(groups[0]) == query.shape[0]


def maybe_l20_shared_prefix_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    prefix_lengths: torch.Tensor | Sequence[int],
    *,
    page_size: int = 16,
    prefix_block_t: int = 128,
    prefix_block_m: int = 8,
    suffix_split_size: int = 512,
    suffix_block_t: int = 128,
):
    """Return L20 shared-prefix output, or None when the conservative gate fails."""
    if not should_dispatch_l20_shared_prefix_decode(
        query,
        key_cache,
        suffix_key,
        block_tables,
        prefix_lengths,
        page_size=page_size,
    ):
        return None
    prefix_length = _to_int_list(prefix_lengths)[0]
    return shared_paged_prefix_suffix_gqa_decode_attention(
        query,
        key_cache,
        value_cache,
        block_tables[0],
        suffix_key,
        suffix_value,
        prefix_length,
        page_size=page_size,
        prefix_block_t=prefix_block_t,
        prefix_block_m=prefix_block_m,
        suffix_split_size=suffix_split_size,
        suffix_block_t=suffix_block_t,
        num_warps=4,
    )
