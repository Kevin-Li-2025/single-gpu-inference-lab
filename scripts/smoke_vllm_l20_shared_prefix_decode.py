#!/usr/bin/env python3
"""Smoke vLLM import-path dispatch for L20 shared-prefix decode attention."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def make_paged_prefix(prefix_key, prefix_value, page_size: int = 16):
    prefix_length, num_kv_heads, head_dim = prefix_key.shape
    if prefix_length % page_size:
        raise ValueError("prefix_length must be divisible by page_size")
    pages = prefix_length // page_size
    block_table = torch.arange(pages, device=prefix_key.device, dtype=torch.int32)
    key_cache = prefix_key.reshape(pages, page_size, num_kv_heads, head_dim).contiguous()
    value_cache = prefix_value.reshape_as(key_cache).contiguous()
    return key_cache, value_cache, block_table


def reference(query, prefix_key, prefix_value, suffix_key, suffix_value):
    batch = query.shape[0]
    key = torch.cat([prefix_key.unsqueeze(0).expand(batch, -1, -1, -1), suffix_key], dim=1)
    value = torch.cat(
        [prefix_value.unsqueeze(0).expand(batch, -1, -1, -1), suffix_value], dim=1
    )
    ratio = query.shape[1] // key.shape[2]
    expanded_key = key.repeat_interleave(ratio, dim=2).transpose(1, 2)
    expanded_value = value.repeat_interleave(ratio, dim=2).transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(
        query.unsqueeze(2),
        expanded_key,
        expanded_value,
    ).squeeze(2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prefix-length", type=int, default=4096)
    parser.add_argument("--suffix-length", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    os.environ["VLLM_ENABLE_L20_SHARED_PREFIX_DECODE"] = "1"
    from vllm.v1.attention.ops.l20_shared_prefix_decode_dispatch import (
        find_l20_shared_prefix_groups,
        maybe_l20_shared_prefix_decode,
        should_dispatch_l20_shared_prefix_decode,
    )

    torch.manual_seed(20260626)
    q_heads = 16
    kv_heads = 8
    head_dim = 128
    query = torch.randn(args.batch, q_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    prefix_key = torch.randn(
        args.prefix_length, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    prefix_value = torch.randn_like(prefix_key)
    suffix_key = torch.randn(
        args.batch,
        args.suffix_length,
        kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    suffix_value = torch.randn_like(suffix_key)
    key_cache, value_cache, shared_blocks = make_paged_prefix(prefix_key, prefix_value)
    block_tables = shared_blocks.unsqueeze(0).expand(args.batch, -1).contiguous()
    prefix_lengths = torch.full((args.batch,), args.prefix_length, device="cuda", dtype=torch.int32)
    groups = find_l20_shared_prefix_groups(block_tables, prefix_lengths)
    should_dispatch = should_dispatch_l20_shared_prefix_decode(
        query,
        key_cache,
        suffix_key,
        block_tables,
        prefix_lengths,
    )
    actual = maybe_l20_shared_prefix_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        suffix_key,
        suffix_value,
        prefix_lengths,
    )
    if actual is None:
        raise RuntimeError("L20 shared-prefix dispatch unexpectedly returned None")
    expected = reference(query, prefix_key, prefix_value, suffix_key, suffix_value)
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "import_path": "vllm.v1.attention.ops.l20_shared_prefix_decode_dispatch",
        "batch": args.batch,
        "prefix_length": args.prefix_length,
        "suffix_length": args.suffix_length,
        "groups": groups,
        "should_dispatch": bool(should_dispatch),
        "correct": bool(torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)),
        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
