#!/usr/bin/env python3
"""Benchmark L20 paged split-KV attention against FlashInfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import flashinfer
    from vllm.v1.attention.ops.l20_paged_split_kv import (
        allocate_l20_paged_split_kv_workspace,
        l20_paged_split_kv_attention,
    )

    reports = []
    torch.manual_seed(23)
    for batch in (1, 4, 8):
        for context in (2048, 4096):
            page_size = 16
            pages_per_sequence = context // page_size
            num_pages = batch * pages_per_sequence
            block_table = torch.randperm(
                num_pages, device="cuda", dtype=torch.int32
            ).reshape(batch, pages_per_sequence)
            indptr = (
                torch.arange(batch + 1, device="cuda", dtype=torch.int32)
                * pages_per_sequence
            )
            indices = block_table.flatten()
            last_page_len = torch.full(
                (batch,), page_size, device="cuda", dtype=torch.int32
            )
            seq_lens = torch.full(
                (batch,), context, device="cuda", dtype=torch.int32
            )
            query = torch.randn(
                batch, 16, 128, device="cuda", dtype=torch.float16
            )
            cache = (
                torch.randn(
                    num_pages, page_size, 8, 128, device="cuda", dtype=torch.float16
                ),
                torch.randn(
                    num_pages, page_size, 8, 128, device="cuda", dtype=torch.float16
                ),
            )
            workspace = torch.empty(
                128 * 1024 * 1024, device="cuda", dtype=torch.uint8
            )
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace, "NHD"
            )
            wrapper.plan(
                indptr,
                indices,
                last_page_len,
                16,
                8,
                128,
                page_size,
                pos_encoding_mode="NONE",
                q_data_type=query.dtype,
                kv_data_type=query.dtype,
            )
            expected = wrapper.run(query, cache)
            l20_workspace = allocate_l20_paged_split_kv_workspace(query, context)
            actual = l20_paged_split_kv_attention(
                query,
                cache[0],
                cache[1],
                block_table,
                seq_lens,
                workspace=l20_workspace,
            )
            ungrouped = l20_paged_split_kv_attention(
                query,
                cache[0],
                cache[1],
                block_table,
                seq_lens,
                workspace=l20_workspace,
                grouped_gqa=False,
            )
            baseline_ms = latency_ms(lambda: wrapper.run(query, cache))
            fused_ms = latency_ms(
                lambda: l20_paged_split_kv_attention(
                    query,
                    cache[0],
                    cache[1],
                    block_table,
                    seq_lens,
                    workspace=l20_workspace,
                )
            )
            ungrouped_ms = latency_ms(
                lambda: l20_paged_split_kv_attention(
                    query,
                    cache[0],
                    cache[1],
                    block_table,
                    seq_lens,
                    workspace=l20_workspace,
                    grouped_gqa=False,
                )
            )
            reports.append(
                {
                    "batch": batch,
                    "context": context,
                    "correct": bool(
                        torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
                    ),
                    "max_abs_error": float(
                        (actual.float() - expected.float()).abs().max()
                    ),
                    "flashinfer_ms": baseline_ms,
                    "l20_ms": fused_ms,
                    "speedup": baseline_ms / fused_ms,
                    "ungrouped_correct": bool(
                        torch.allclose(ungrouped, expected, rtol=2e-2, atol=2e-2)
                    ),
                    "ungrouped_ms": ungrouped_ms,
                    "grouped_vs_ungrouped": ungrouped_ms / fused_ms,
                }
            )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
