#!/usr/bin/env python3
"""Benchmark fused RoPE + block-table paged KV writes on NVIDIA L20."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from l20_stack.ops.triton_rope_kv import (
    paged_rope_kv_cache_write_triton,
    paged_rope_kv_reference,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cache-flush-mb", type=int, default=256)
    parser.add_argument("--require-l20", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def rotate_k(torch, k, cos, sin):
    half = cos.shape[1]
    rotated = k.clone()
    first = k[:, :, :half].float()
    second = k[:, :, half : 2 * half].float()
    rotated[:, :, :half] = (first * cos[:, None] - second * sin[:, None]).to(k.dtype)
    rotated[:, :, half : 2 * half] = (second * cos[:, None] + first * sin[:, None]).to(k.dtype)
    return rotated


def timings(torch, function, reset, warmup, iterations, cache_flush):
    for _ in range(warmup):
        reset()
        if cache_flush is not None:
            cache_flush.zero_()
        function()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        reset()
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
        "p50": round(statistics.median(ordered), 4),
        "p95": round(ordered[round((len(ordered) - 1) * 0.95)], 4),
        "mean": round(statistics.mean(ordered), 4),
    }


def correctness(torch, actual, expected):
    errors = [(a.float() - e.float()).abs().max().item() for a, e in zip(actual, expected)]
    return {"correct": all(torch.equal(a, e) for a, e in zip(actual, expected)), "max_abs_error": max(errors)}


def main() -> int:
    args = parse_args()
    if min(args.tokens, args.sequences, args.kv_heads, args.head_dim, args.block_size, args.iters) <= 0:
        raise SystemExit("dimensions and iterations must be positive")
    import torch
    import triton

    gpu = torch.cuda.get_device_name()
    capability = torch.cuda.get_device_capability()
    if args.require_l20 and ("L20" not in gpu.upper() or capability != (8, 9)):
        raise SystemExit(f"expected NVIDIA L20 sm_89, got {gpu} sm_{capability[0]}{capability[1]}")
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16
    sequence_ids = torch.arange(args.tokens, device=device, dtype=torch.int32) % args.sequences
    positions = torch.div(torch.arange(args.tokens, device=device), args.sequences, rounding_mode="floor").int()
    max_position = int(positions.max().item()) + 1
    blocks_per_sequence = (max_position + args.block_size - 1) // args.block_size
    num_blocks = args.sequences * blocks_per_sequence
    block_table = torch.randperm(num_blocks, device=device, dtype=torch.int32).reshape(args.sequences, blocks_per_sequence)
    k = torch.randn(args.tokens, args.kv_heads, args.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    angles = torch.randn(args.tokens, args.head_dim // 2, device=device)
    cos, sin = angles.cos().half(), angles.sin().half()
    cache_shape = (num_blocks, args.block_size, args.kv_heads, args.head_dim)
    expected_cache = (torch.zeros(cache_shape, device=device, dtype=dtype), torch.zeros(cache_shape, device=device, dtype=dtype))
    expected = paged_rope_kv_reference(k, v, cos, sin, sequence_ids, positions, block_table, *expected_cache)
    cache_flush = torch.empty(args.cache_flush_mb * 1024 * 1024, device=device, dtype=torch.uint8) if args.cache_flush_mb else None

    providers = {}
    torch_cache = (torch.zeros_like(expected[0]), torch.zeros_like(expected[1]))
    providers["torch_separate"] = (
        lambda: paged_rope_kv_reference(k, v, cos, sin, sequence_ids, positions, block_table, *torch_cache),
        lambda: [x.zero_() for x in torch_cache],
    )
    triton_cache = (torch.zeros_like(expected[0]), torch.zeros_like(expected[1]))
    providers["l20_triton_fused"] = (
        lambda: paged_rope_kv_cache_write_triton(k, v, cos, sin, sequence_ids, positions, block_table, *triton_cache),
        lambda: [x.zero_() for x in triton_cache],
    )

    unavailable = {}
    try:
        import flashinfer

        fi_cache = (torch.zeros_like(expected[0]), torch.zeros_like(expected[1]))
        kv_indices = block_table.flatten()
        kv_indptr = torch.arange(args.sequences + 1, device=device, dtype=torch.int32) * blocks_per_sequence
        last_len = torch.full((args.sequences,), max_position % args.block_size or args.block_size, device=device, dtype=torch.int32)

        def flashinfer_provider():
            rotated = rotate_k(torch, k, cos, sin)
            flashinfer.append_paged_kv_cache(rotated, v, sequence_ids, positions, fi_cache, kv_indices, kv_indptr, last_len, "NHD")
            return fi_cache

        providers["flashinfer_separate"] = (flashinfer_provider, lambda: [x.zero_() for x in fi_cache])
    except ImportError as error:
        unavailable["flashinfer_separate"] = str(error)

    try:
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash

        vllm_cache = (torch.zeros_like(expected[0]), torch.zeros_like(expected[1]))
        slots = block_table[sequence_ids.long(), torch.div(positions, args.block_size, rounding_mode="floor").long()] * args.block_size + positions % args.block_size
        scales = torch.ones(1, device=device, dtype=torch.float32)

        def vllm_provider():
            rotated = rotate_k(torch, k, cos, sin)
            triton_reshape_and_cache_flash(rotated, v, *vllm_cache, slots.long(), "auto", scales, scales)
            return vllm_cache

        providers["vllm_separate"] = (vllm_provider, lambda: [x.zero_() for x in vllm_cache])
    except ImportError as error:
        unavailable["vllm_separate"] = str(error)

    reports = {}
    for name, (run, reset) in providers.items():
        reset()
        actual = run()
        torch.cuda.synchronize()
        report = correctness(torch, actual, expected)
        if report["correct"]:
            report["timing_ms"] = summarize(timings(torch, run, reset, args.warmup, args.iters, cache_flush))
        reports[name] = report
    baseline = reports["torch_separate"].get("timing_ms", {}).get("p50")
    for report in reports.values():
        p50 = report.get("timing_ms", {}).get("p50")
        if baseline and p50:
            report["speedup_vs_torch_separate"] = round(baseline / p50, 3)
    result = {
        "benchmark_version": 1,
        "gpu_name": gpu,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shape": vars(args) | {"output": str(args.output) if args.output else None},
        "layout": "NHD [blocks, block_size, kv_heads, head_dim]",
        "physical_block_order": block_table.cpu().tolist(),
        "providers": reports,
        "unavailable_providers": unavailable,
        "all_correct": all(report["correct"] for report in reports.values()),
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    return 0 if result["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
