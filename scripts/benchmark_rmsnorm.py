#!/usr/bin/env python3
"""Benchmark standalone and fused RMSNorm paths on an NVIDIA L20."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from l20_stack.operators import (
    OperatorShape,
    residual_rmsnorm_minimum_bytes,
    rmsnorm_minimum_bytes,
)
from l20_stack.ops.triton_rmsnorm import (
    residual_rmsnorm_reference,
    residual_rmsnorm_l20,
    residual_rmsnorm_l20_inplace,
    residual_rmsnorm_triton,
    rmsnorm_reference,
    rmsnorm_triton,
    rmsnorm_warp_candidates,
)


MATRIX_HIDDEN_SIZES = (4096, 5120, 6144, 8192)
MATRIX_ROWS = (1, 8, 32, 128, 512, 4096)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        choices=("rmsnorm", "residual-rmsnorm", "both"),
        default="both",
    )
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="benchmark common LLM hidden sizes: 4096, 5120, 6144, and 8192",
    )
    parser.add_argument(
        "--rows-matrix",
        action="store_true",
        help="benchmark decode and prefill rows: 1, 8, 32, 128, 512, and 4096",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument(
        "--cache-flush-mb",
        type=int,
        default=256,
        help="bytes touched before each timing sample to evict L2; use 0 for warm-cache tests",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument(
        "--require-l20",
        action="store_true",
        help="fail unless the active device is an NVIDIA L20 with compute capability 8.9",
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser.parse_args()


def percentile(values, pct):
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[index]


def cuda_event_timings(torch, function, reset, warmup, iterations, cache_flush):
    for _ in range(warmup):
        if reset is not None:
            reset()
        if cache_flush is not None:
            cache_flush.zero_()
        function()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        if reset is not None:
            reset()
        if cache_flush is not None:
            cache_flush.zero_()
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def result_tensors(value):
    return value if isinstance(value, tuple) else (value,)


def correctness(torch, actual, expected, dtype):
    actual_tensors = result_tensors(actual)
    expected_tensors = result_tensors(expected)
    if len(actual_tensors) != len(expected_tensors):
        return {"correct": False, "error": "result arity mismatch"}

    if dtype == torch.float32:
        atol = 1e-5
    elif dtype == torch.bfloat16:
        atol = 1e-2
    else:
        atol = 5e-3
    max_abs = 0.0
    max_rel = 0.0
    correct = True
    for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors):
        difference = (actual_tensor.float() - expected_tensor.float()).abs()
        max_abs = max(max_abs, difference.max().item())
        relative = difference / expected_tensor.float().abs().clamp_min(1e-6)
        max_rel = max(max_rel, relative.max().item())
        correct = correct and torch.allclose(
            actual_tensor.float(), expected_tensor.float(), atol=atol, rtol=1e-3
        )
    return {
        "correct": bool(correct),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "atol": atol,
        "rtol": 1e-3,
    }


def timing_report(timings_ms, minimum_bytes):
    p50 = percentile(timings_ms, 0.50)
    return {
        "timing_ms": {
            "p50": round(p50, 4),
            "p95": round(percentile(timings_ms, 0.95), 4),
            "mean": round(statistics.mean(timings_ms), 4),
        },
        "minimum_effective_gbps": round(minimum_bytes / p50 / 1_000_000, 2),
    }


def benchmark_providers(
    torch, providers, expected, dtype, minimum_bytes, warmup, iterations, cache_flush
):
    reports = {}
    for name, provider in providers.items():
        if isinstance(provider, tuple):
            function, reset = provider
        else:
            function, reset = provider, None
        try:
            if reset is not None:
                reset()
            actual = function()
            torch.cuda.synchronize()
            provider_report = correctness(torch, actual, expected, dtype)
            if provider_report["correct"]:
                timings = cuda_event_timings(
                    torch, function, reset, warmup, iterations, cache_flush
                )
                provider_report.update(timing_report(timings, minimum_bytes))
            reports[name] = provider_report
        except Exception as exc:  # Keep the rest of the benchmark matrix usable.
            reports[name] = {
                "correct": False,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }

    eager_p50 = reports.get("torch_eager", {}).get("timing_ms", {}).get("p50")
    if eager_p50:
        for provider_report in reports.values():
            provider_p50 = provider_report.get("timing_ms", {}).get("p50")
            if provider_p50:
                provider_report["speedup_vs_torch_eager"] = round(eager_p50 / provider_p50, 3)
    return reports


def fastest_provider(reports):
    measured = {
        name: report["timing_ms"]["p50"]
        for name, report in reports.items()
        if "timing_ms" in report
    }
    return min(measured, key=measured.get) if measured else None


def benchmark_shape(torch, args, rows, hidden_size, flashinfer):
    dtype = getattr(torch, args.dtype)
    x = torch.randn(rows, hidden_size, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, device="cuda", dtype=dtype)
    cache_flush = None
    if args.cache_flush_mb:
        cache_flush = torch.empty(
            args.cache_flush_mb * 1024 * 1024, device="cuda", dtype=torch.uint8
        )
    shape = OperatorShape(rows, hidden_size, x.element_size())
    report = {
        "shape": {"rows": rows, "hidden_size": hidden_size, "dtype": args.dtype},
        "operators": {},
    }

    if args.operator in ("rmsnorm", "both"):
        expected = rmsnorm_reference(x, weight, args.eps)

        def torch_rmsnorm(input_tensor, norm_weight):
            return torch.nn.functional.rms_norm(
                input_tensor, (hidden_size,), norm_weight, args.eps
            )

        providers = {"torch_eager": lambda: torch_rmsnorm(x, weight)}
        for num_warps in rmsnorm_warp_candidates(hidden_size):
            providers[f"triton_w{num_warps}"] = (
                lambda warps=num_warps: rmsnorm_triton(x, weight, args.eps, warps)
            )
        if not args.skip_compile:
            compiled_rmsnorm = torch.compile(torch_rmsnorm, fullgraph=True)
            providers["torch_compile"] = lambda: compiled_rmsnorm(x, weight)

        minimum_bytes = rmsnorm_minimum_bytes(shape)
        provider_reports = benchmark_providers(
            torch,
            providers,
            expected,
            dtype,
            minimum_bytes,
            args.warmup,
            args.iters,
            cache_flush,
        )
        report["operators"]["rmsnorm"] = {
            "minimum_bytes": minimum_bytes,
            "fastest_provider": fastest_provider(provider_reports),
            "providers": provider_reports,
        }

    if args.operator in ("residual-rmsnorm", "both"):
        expected = residual_rmsnorm_reference(x, residual, weight, args.eps)

        def torch_residual_rmsnorm(input_tensor, residual_tensor, norm_weight):
            merged = input_tensor + residual_tensor
            normalized = torch.nn.functional.rms_norm(
                merged, (hidden_size,), norm_weight, args.eps
            )
            return normalized, merged

        providers = {"torch_eager": lambda: torch_residual_rmsnorm(x, residual, weight)}
        providers["l20_dispatch"] = lambda: residual_rmsnorm_l20(
            x, residual, weight, args.eps
        )
        dispatch_input = x.clone()
        dispatch_residual = residual.clone()

        def reset_l20_inplace():
            dispatch_input.copy_(x)
            dispatch_residual.copy_(residual)

        def run_l20_inplace():
            residual_rmsnorm_l20_inplace(
                dispatch_input, dispatch_residual, weight, args.eps
            )
            return dispatch_input, dispatch_residual

        providers["l20_inplace"] = (run_l20_inplace, reset_l20_inplace)
        if flashinfer is not None:
            flashinfer_input = x.clone()
            flashinfer_residual = residual.clone()

            def reset_flashinfer():
                flashinfer_input.copy_(x)
                flashinfer_residual.copy_(residual)

            def run_flashinfer():
                flashinfer.norm.fused_add_rmsnorm(
                    flashinfer_input, flashinfer_residual, weight, args.eps
                )
                return flashinfer_input, flashinfer_residual

            providers["flashinfer"] = (run_flashinfer, reset_flashinfer)
        for num_warps in rmsnorm_warp_candidates(hidden_size):
            providers[f"triton_w{num_warps}"] = (
                lambda warps=num_warps: residual_rmsnorm_triton(
                    x, residual, weight, args.eps, warps
                )
            )
        if not args.skip_compile:
            compiled_residual_rmsnorm = torch.compile(torch_residual_rmsnorm, fullgraph=True)
            providers["torch_compile"] = lambda: compiled_residual_rmsnorm(x, residual, weight)

        minimum_bytes = residual_rmsnorm_minimum_bytes(shape, fused=True)
        unfused_bytes = residual_rmsnorm_minimum_bytes(shape, fused=False)
        provider_reports = benchmark_providers(
            torch,
            providers,
            expected,
            dtype,
            minimum_bytes,
            args.warmup,
            args.iters,
            cache_flush,
        )
        report["operators"]["residual_rmsnorm"] = {
            "fused_minimum_bytes": minimum_bytes,
            "unfused_minimum_bytes": unfused_bytes,
            "minimum_traffic_reduction_pct": round(
                100 * (unfused_bytes - minimum_bytes) / unfused_bytes, 2
            ),
            "fastest_provider": fastest_provider(provider_reports),
            "providers": provider_reports,
        }

    return report


def main() -> int:
    args = parse_args()
    if (
        args.rows <= 0
        or args.hidden_size <= 0
        or args.warmup < 0
        or args.iters <= 0
        or args.cache_flush_mb < 0
    ):
        raise SystemExit(
            "rows, hidden-size, and iters must be positive; warmup must be non-negative"
        )

    import torch
    import triton
    try:
        import flashinfer
    except ImportError:
        flashinfer = None

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    gpu_name = torch.cuda.get_device_name()
    compute_capability = torch.cuda.get_device_capability()
    if args.require_l20 and ("L20" not in gpu_name.upper() or compute_capability != (8, 9)):
        actual_target = f"{gpu_name} sm_{compute_capability[0]}{compute_capability[1]}"
        raise SystemExit(f"--require-l20 expected NVIDIA L20 sm_89, got {actual_target}")

    torch.manual_seed(0)
    hidden_sizes = MATRIX_HIDDEN_SIZES if args.matrix else (args.hidden_size,)
    row_sizes = MATRIX_ROWS if args.rows_matrix else (args.rows,)
    shapes = [
        benchmark_shape(torch, args, rows, hidden_size, flashinfer)
        for rows in row_sizes
        for hidden_size in hidden_sizes
    ]
    all_correct = all(
        provider.get("correct", False)
        for shape in shapes
        for operator in shape["operators"].values()
        for provider in operator["providers"].values()
    )
    report = {
        "benchmark_version": 2,
        "gpu_name": gpu_name,
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "flashinfer": getattr(flashinfer, "__version__", None),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iters,
        "cache_flush_mb": args.cache_flush_mb,
        "all_correct": all_correct,
        "shapes": shapes,
        "note": (
            "Effective GB/s uses the semantic minimum traffic. Speedups are valid only for "
            "the GPU and software versions recorded in this report. FlashInfer and "
            "l20_inplace use the production in-place contract; other providers return "
            "new output tensors."
        ),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if all_correct else 1


if __name__ == "__main__":
    raise SystemExit(main())
