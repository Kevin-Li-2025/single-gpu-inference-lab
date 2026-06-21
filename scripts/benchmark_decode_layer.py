#!/usr/bin/env python3
"""Benchmark paged RoPE/KV append plus FlashInfer decode attention on L20."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from l20_stack.ops.triton_rope_kv import paged_rope_kv_cache_write_triton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--qo-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--workspace-mb", type=int, default=128)
    parser.add_argument("--cache-flush-mb", type=int, default=0)
    parser.add_argument("--require-l20", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def rotate_k(torch, k, cos, sin):
    half = cos.shape[1]
    rotated = torch.empty_like(k)
    first = k[:, :, :half].float()
    second = k[:, :, half : 2 * half].float()
    rotated[:, :, :half] = (first * cos[:, None] - second * sin[:, None]).to(k.dtype)
    rotated[:, :, half : 2 * half] = (
        second * cos[:, None] + first * sin[:, None]
    ).to(k.dtype)
    if 2 * half < k.shape[2]:
        rotated[:, :, 2 * half :] = k[:, :, 2 * half :]
    return rotated


def measure(torch, function, warmup, iterations, cache_flush):
    for _ in range(warmup):
        if cache_flush is not None:
            cache_flush.zero_()
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        if cache_flush is not None:
            cache_flush.zero_()
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def summarize(values):
    ordered = sorted(values)
    return {
        "p50": round(statistics.median(ordered), 5),
        "p95": round(ordered[round((len(ordered) - 1) * 0.95)], 5),
        "mean": round(statistics.mean(ordered), 5),
    }


def main() -> int:
    args = parse_args()
    dimensions = (
        args.batch_size,
        args.context_length,
        args.qo_heads,
        args.kv_heads,
        args.head_dim,
        args.block_size,
        args.iters,
    )
    if min(dimensions) <= 0:
        raise SystemExit("dimensions and iterations must be positive")
    if args.qo_heads % args.kv_heads:
        raise SystemExit("qo-heads must be divisible by kv-heads")

    import flashinfer
    import torch
    import triton

    gpu = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    if args.require_l20 and ("L20" not in gpu.upper() or capability != (8, 9)):
        raise SystemExit(f"expected NVIDIA L20 sm_89, got {gpu} sm_{capability[0]}{capability[1]}")

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    pages_per_sequence = (args.context_length + args.block_size - 1) // args.block_size
    num_pages = args.batch_size * pages_per_sequence
    block_table = torch.randperm(num_pages, device=device, dtype=torch.int32).reshape(
        args.batch_size, pages_per_sequence
    )
    kv_indptr = (
        torch.arange(args.batch_size + 1, device=device, dtype=torch.int32)
        * pages_per_sequence
    )
    kv_indices = block_table.flatten()
    last_page_len = torch.full(
        (args.batch_size,),
        args.context_length % args.block_size or args.block_size,
        device=device,
        dtype=torch.int32,
    )
    sequence_ids = torch.arange(args.batch_size, device=device, dtype=torch.int32)
    positions = torch.full(
        (args.batch_size,), args.context_length - 1, device=device, dtype=torch.int32
    )
    q = torch.randn(
        args.batch_size, args.qo_heads, args.head_dim, device=device, dtype=dtype
    )
    k = torch.randn(
        args.batch_size, args.kv_heads, args.head_dim, device=device, dtype=dtype
    )
    v = torch.randn_like(k)
    angles = torch.randn(args.batch_size, args.head_dim // 2, device=device)
    cos, sin = angles.cos().half(), angles.sin().half()
    cache_shape = (
        num_pages,
        args.block_size,
        args.kv_heads,
        args.head_dim,
    )
    base_cache = (
        torch.randn(cache_shape, device=device, dtype=dtype),
        torch.randn(cache_shape, device=device, dtype=dtype),
    )
    separate_cache = (base_cache[0].clone(), base_cache[1].clone())
    fused_cache = (base_cache[0].clone(), base_cache[1].clone())
    workspace = torch.empty(
        args.workspace_mb * 1024 * 1024, device=device, dtype=torch.uint8
    )
    cache_flush = (
        torch.empty(args.cache_flush_mb * 1024 * 1024, device=device, dtype=torch.uint8)
        if args.cache_flush_mb
        else None
    )

    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.plan(
        kv_indptr,
        kv_indices,
        last_page_len,
        args.qo_heads,
        args.kv_heads,
        args.head_dim,
        args.block_size,
        pos_encoding_mode="NONE",
        q_data_type=dtype,
        kv_data_type=dtype,
    )

    def separate_append():
        rotated = rotate_k(torch, k, cos, sin)
        flashinfer.append_paged_kv_cache(
            rotated,
            v,
            sequence_ids,
            positions,
            separate_cache,
            kv_indices,
            kv_indptr,
            last_page_len,
            "NHD",
        )

    def fused_append():
        paged_rope_kv_cache_write_triton(
            k,
            v,
            cos,
            sin,
            sequence_ids,
            positions,
            block_table,
            *fused_cache,
        )

    def separate_layer():
        separate_append()
        return wrapper.run(q, separate_cache)

    def fused_layer():
        fused_append()
        return wrapper.run(q, fused_cache)

    separate_append()
    fused_append()
    separate_output = wrapper.run(q, separate_cache)
    fused_output = wrapper.run(q, fused_cache)
    torch.cuda.synchronize()
    correctness = {
        "cache_equal": all(torch.equal(a, b) for a, b in zip(separate_cache, fused_cache)),
        "output_close": torch.allclose(
            separate_output, fused_output, rtol=1e-3, atol=1e-3
        ),
        "output_max_abs_error": float(
            (separate_output.float() - fused_output.float()).abs().max().item()
        ),
    }

    functions = {
        "separate_append": separate_append,
        "l20_fused_append": fused_append,
        "attention_only": lambda: wrapper.run(q, fused_cache),
        "separate_layer": separate_layer,
        "l20_fused_layer": fused_layer,
    }
    timings = {
        name: summarize(measure(torch, fn, args.warmup, args.iters, cache_flush))
        for name, fn in functions.items()
    }
    separate_p50 = timings["separate_layer"]["p50"]
    fused_p50 = timings["l20_fused_layer"]["p50"]
    result = {
        "benchmark_version": 1,
        "scope": "one decode attention layer: RoPE + paged KV append + attention",
        "gpu_name": gpu,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "flashinfer": flashinfer.__version__,
        "shape": {
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "qo_heads": args.qo_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "block_size": args.block_size,
        },
        "correctness": correctness,
        "timing_ms": timings,
        "layer_speedup": round(separate_p50 / fused_p50, 4),
        "layer_latency_reduction_pct": round(
            (separate_p50 - fused_p50) / separate_p50 * 100, 3
        ),
        "notes": [
            "FlashInfer plan time is excluded from all measurements.",
            "Both paths use the same FlashInfer paged decode attention implementation.",
            "This is a layer-level serving benchmark, not full-model tokens/s.",
        ],
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if correctness["cache_equal"] and correctness["output_close"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
