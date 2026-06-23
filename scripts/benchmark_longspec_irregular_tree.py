#!/usr/bin/env python3
"""Benchmark LongSpec-style irregular ancestor masks on L20."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from l20_stack.ops.triton_tree_attention import (
    allocate_tree_attention_workspace,
    hybrid_tree_attention,
    hybrid_tree_attention_paged_prefix,
    hybrid_tree_attention_split,
    l20_tree_attention_block_t,
    make_chain_tree_mask,
    torch_tree_attention_reference,
)


def latency_ms(function, warmup=20, iterations=100):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def make_paged_prefix(prefix_key, prefix_value, page_size: int = 16, page_order: str = "random"):
    batch, cached_length, num_kv_heads, head_dim = prefix_key.shape
    if cached_length % page_size:
        raise ValueError("cached_length must be divisible by page_size")
    pages_per_batch = cached_length // page_size
    num_pages = batch * pages_per_batch
    if page_order == "random":
        block_table = torch.randperm(
            num_pages, device=prefix_key.device, dtype=torch.int32
        ).reshape(batch, pages_per_batch)
        page_base = None
    elif page_order == "contiguous":
        block_table = torch.arange(num_pages, device=prefix_key.device, dtype=torch.int32).reshape(
            batch, pages_per_batch
        )
        page_base = block_table[:, 0].contiguous()
    else:
        raise ValueError(f"unknown page order: {page_order}")
    key_cache = torch.empty(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=prefix_key.device,
        dtype=prefix_key.dtype,
    )
    value_cache = torch.empty_like(key_cache)
    key_pages = prefix_key.reshape(batch, pages_per_batch, page_size, num_kv_heads, head_dim)
    value_pages = prefix_value.reshape_as(key_pages)
    for batch_index in range(batch):
        key_cache[block_table[batch_index].long()] = key_pages[batch_index]
        value_cache[block_table[batch_index].long()] = value_pages[batch_index]
    return key_cache, value_cache, block_table, page_base


def make_balanced_tree_mask(draft_length: int, *, branch: int, device):
    parent = torch.full((draft_length,), -1, device=device, dtype=torch.int64)
    for token in range(1, draft_length):
        parent[token] = (token - 1) // branch
    return ancestor_mask_from_parent(parent), parent


def make_random_tree_mask(draft_length: int, *, branch: int, seed: int, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    parent = torch.full((draft_length,), -1, device=device, dtype=torch.int64)
    child_counts = torch.zeros((draft_length,), device=device, dtype=torch.int64)
    frontier = [0]
    for token in range(1, draft_length):
        candidates = torch.tensor(frontier, device=device, dtype=torch.int64)
        pick = int(torch.randint(0, candidates.numel(), (1,), generator=generator, device=device))
        selected = int(candidates[pick].item())
        parent[token] = selected
        child_counts[selected] += 1
        if int(child_counts[selected].item()) >= branch:
            frontier.remove(selected)
        frontier.append(token)
    return ancestor_mask_from_parent(parent), parent


def ancestor_mask_from_parent(parent):
    draft_length = parent.numel()
    mask = torch.eye(draft_length, device=parent.device, dtype=torch.bool)
    for token in range(draft_length):
        current = int(parent[token].item())
        while current >= 0:
            mask[token, current] = True
            current = int(parent[current].item())
    return mask


def mask_stats(mask, parent):
    ancestors = mask.sum(dim=1).float()
    depths = ancestors - 1.0
    return {
        "density": float(mask.float().mean().item()),
        "max_depth": int(depths.max().item()),
        "mean_depth": float(depths.mean().item()),
        "mean_visible_draft_tokens": float(ancestors.mean().item()),
        "parent": [int(x) for x in parent.cpu().tolist()],
    }


def run_case(
    batch: int,
    cached: int,
    draft: int,
    tree: str,
    branch: int,
    page_order: str,
    iterations: int,
):
    torch.manual_seed(123 + batch * 13 + cached + draft + branch)
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    dtype = torch.float16
    query = torch.randn(batch, draft, num_q_heads, head_dim, device="cuda", dtype=dtype)
    prefix_key = torch.randn(batch, cached, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    prefix_value = torch.randn_like(prefix_key)
    suffix_key = torch.randn(batch, draft, num_kv_heads, head_dim, device="cuda", dtype=dtype)
    suffix_value = torch.randn_like(suffix_key)
    key = torch.cat([prefix_key, suffix_key], dim=1)
    value = torch.cat([prefix_value, suffix_value], dim=1)
    if tree == "chain":
        mask = make_chain_tree_mask(draft, device="cuda")
        parent = torch.arange(-1, draft - 1, device="cuda", dtype=torch.int64)
    elif tree == "balanced":
        mask, parent = make_balanced_tree_mask(draft, branch=branch, device="cuda")
    elif tree == "random":
        mask, parent = make_random_tree_mask(
            draft, branch=branch, seed=17 + draft + branch, device="cuda"
        )
    else:
        raise ValueError(f"unknown tree shape: {tree}")
    key_cache, value_cache, block_table, page_base = make_paged_prefix(
        prefix_key, prefix_value, page_order=page_order
    )
    expected = torch_tree_attention_reference(query, key, value, mask, cached)
    workspace = allocate_tree_attention_workspace(query)
    monolithic = hybrid_tree_attention(query, key, value, mask, cached)
    split = hybrid_tree_attention_split(query, key, value, mask, cached, workspace=workspace)
    paged = hybrid_tree_attention_paged_prefix(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        mask,
        cached,
        workspace=workspace,
        contiguous_pages=page_order == "contiguous",
        page_base=page_base,
    )
    monolithic_ms = latency_ms(
        lambda: hybrid_tree_attention(query, key, value, mask, cached),
        iterations=iterations,
    )
    split_ms = latency_ms(
        lambda: hybrid_tree_attention_split(query, key, value, mask, cached, workspace=workspace),
        iterations=iterations,
    )
    paged_ms = latency_ms(
        lambda: hybrid_tree_attention_paged_prefix(
            query,
            key_cache,
            value_cache,
            suffix_key,
            suffix_value,
            block_table,
            mask,
            cached,
            workspace=workspace,
            contiguous_pages=page_order == "contiguous",
            page_base=page_base,
        ),
        iterations=iterations,
    )
    dense_ms = latency_ms(
        lambda: torch_tree_attention_reference(query, key, value, mask, cached),
        iterations=max(10, iterations // 4),
    )
    stats = mask_stats(mask, parent)
    return {
        "batch": batch,
        "block_t": l20_tree_attention_block_t(cached),
        "branch": branch,
        "cached_length": cached,
        "draft_length": draft,
        "tree": tree,
        "mask": stats,
        "page_order": page_order,
        "monolithic_correct": bool(torch.allclose(monolithic, expected, rtol=2e-2, atol=2e-2)),
        "split_correct": bool(torch.allclose(split, expected, rtol=2e-2, atol=2e-2)),
        "paged_correct": bool(torch.allclose(paged, expected, rtol=2e-2, atol=2e-2)),
        "monolithic_max_abs_error": float((monolithic.float() - expected.float()).abs().max()),
        "split_max_abs_error": float((split.float() - expected.float()).abs().max()),
        "paged_max_abs_error": float((paged.float() - expected.float()).abs().max()),
        "torch_dense_ms": dense_ms,
        "monolithic_ms": monolithic_ms,
        "split_ms": split_ms,
        "paged_ms": paged_ms,
        "monolithic_speedup_vs_torch_dense": dense_ms / monolithic_ms,
        "split_speedup_vs_torch_dense": dense_ms / split_ms,
        "paged_speedup_vs_torch_dense": dense_ms / paged_ms,
        "split_vs_monolithic": monolithic_ms / split_ms,
        "paged_vs_monolithic": monolithic_ms / paged_ms,
        "paged_vs_split": split_ms / paged_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="+", default=[1])
    parser.add_argument("--cached", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--draft", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--trees", nargs="+", default=["balanced", "random"])
    parser.add_argument("--branches", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--page-orders", nargs="+", default=["random"])
    args = parser.parse_args()

    reports = []
    for batch in args.batches:
        for cached in args.cached:
            for draft in args.draft:
                for tree in args.trees:
                    for branch in args.branches:
                        for page_order in args.page_orders:
                            reports.append(
                                run_case(
                                    batch,
                                    cached,
                                    draft,
                                    tree,
                                    branch,
                                    page_order,
                                    args.iterations,
                                )
                            )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "operator": "longspec_irregular_tree_attention",
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
