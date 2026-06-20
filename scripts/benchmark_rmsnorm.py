#!/usr/bin/env python3
"""Benchmark the L20 Triton RMSNorm baseline against a PyTorch reference."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from l20_stack.ops.triton_rmsnorm import rmsnorm_reference, rmsnorm_triton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    return parser.parse_args()


def percentile(values, pct):
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[index]


def main() -> int:
    args = parse_args()

    import torch
    import triton

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    dtype = getattr(torch, args.dtype)
    x = torch.randn(args.rows, args.hidden_size, device="cuda", dtype=dtype)
    weight = torch.randn(args.hidden_size, device="cuda", dtype=dtype)

    ref = rmsnorm_reference(x, weight, args.eps)
    out = rmsnorm_triton(x, weight, args.eps)
    torch.cuda.synchronize()

    max_abs_error = (ref.float() - out.float()).abs().max().item()
    max_rel_error = ((ref.float() - out.float()).abs() / ref.float().abs().clamp_min(1e-6)).max().item()

    for _ in range(args.warmup):
        rmsnorm_triton(x, weight, args.eps)
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(args.iters):
        start = time.perf_counter()
        rmsnorm_triton(x, weight, args.eps)
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter() - start) * 1000)

    bytes_moved = args.rows * args.hidden_size * torch.tensor([], dtype=dtype).element_size() * 3
    median_ms = statistics.median(timings_ms)
    effective_gbps = bytes_moved / (median_ms / 1000) / 1_000_000_000

    report = {
        "operator": "rmsnorm",
        "gpu_name": torch.cuda.get_device_name(),
        "compute_capability": ".".join(str(x) for x in torch.cuda.get_device_capability()),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shape": {
            "rows": args.rows,
            "hidden_size": args.hidden_size,
            "dtype": args.dtype,
        },
        "timing_ms": {
            "p50": round(percentile(timings_ms, 0.50), 4),
            "p95": round(percentile(timings_ms, 0.95), 4),
            "mean": round(statistics.mean(timings_ms), 4),
        },
        "effective_gbps": round(effective_gbps, 2),
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "note": "Do not claim speedup until compared against PyTorch eager and torch.compile.",
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
