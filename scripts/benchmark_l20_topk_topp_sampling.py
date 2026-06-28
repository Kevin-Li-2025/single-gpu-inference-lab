#!/usr/bin/env python3
"""Benchmark the L20 two-stage top-k/top-p sampling prototype."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env
from l20_stack.ops.triton_sampling import (
    topk_topp_sample_from_uniform,
    topk_topp_sample_from_uniform_out,
    topk_topp_sample_from_uniform_reference,
    topk_topp_sampling_launch_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--block-vocab", type=int)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--skip-flashinfer", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.90 * (len(ordered) - 1))],
        "samples_ms": samples,
    }


def time_gpu(fn, warmup, rounds):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def time_cpu(fn, warmup, rounds):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def cpu_roundtrip_reference(logits, uniforms, top_k, top_p, temperature):
    return topk_topp_sample_from_uniform_reference(
        logits.cpu(),
        uniforms.cpu(),
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )


def try_flashinfer(logits, args):
    if args.skip_flashinfer:
        return None
    try:
        flashinfer_cuda_env = configure_flashinfer_cuda13_env(required=True)
        import flashinfer
        import flashinfer.sampling as flashinfer_sampling
        scaled_logits = logits / args.temperature
        seed = torch.full((args.batch,), 12345, device="cuda", dtype=torch.int64)
        offset = torch.zeros((args.batch,), device="cuda", dtype=torch.int64)
        out = flashinfer_sampling.top_k_top_p_sampling_from_logits(
            scaled_logits,
            args.top_k,
            args.top_p,
            filter_apply_order="top_k_first",
            deterministic=True,
            seed=seed,
            offset=offset,
        )
        if out.shape != (args.batch,):
            raise AssertionError("unexpected FlashInfer sampler output shape")
        return {
            "available": True,
            "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
            "flashinfer_cuda_env": flashinfer_cuda_env.to_dict(),
            "flashinfer_topk_topp_from_logits": summarize(
                time_gpu(
                    lambda: flashinfer_sampling.top_k_top_p_sampling_from_logits(
                        scaled_logits,
                        args.top_k,
                        args.top_p,
                        filter_apply_order="top_k_first",
                        deterministic=True,
                        seed=seed,
                        offset=offset,
                    ),
                    args.warmup,
                    args.rounds,
                )
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(73)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=dtype)
    uniforms = torch.rand((args.batch,), device="cuda", dtype=torch.float32)
    config = topk_topp_sampling_launch_config(
        args.vocab,
        args.top_k,
        batch=args.batch,
        block_vocab_override=args.block_vocab,
    )
    partial_shape = (args.batch, config.blocks_per_row, args.top_k)
    partial_values = torch.empty(partial_shape, device="cuda", dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device="cuda", dtype=torch.int64)
    output = torch.empty((args.batch,), device="cuda", dtype=torch.int64)

    expected = topk_topp_sample_from_uniform_reference(
        logits,
        uniforms,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )
    actual = topk_topp_sample_from_uniform(
        logits,
        uniforms,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(
            f"Triton top-k/top-p sampler mismatch: actual={actual.cpu()} "
            f"expected={expected.cpu()}"
        )
    topk_topp_sample_from_uniform_out(
        logits,
        uniforms,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(output.cpu(), expected.cpu()):
        raise AssertionError("preallocated Triton top-k/top-p sampler mismatch")

    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "vocab": args.vocab,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "dtype": args.dtype,
        },
        "launch": config.to_dict(),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "triton_topk_topp_from_uniform": summarize(
            time_gpu(
                lambda: topk_topp_sample_from_uniform(
                    logits,
                    uniforms,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    block_vocab_override=args.block_vocab,
                ),
                args.warmup,
                args.rounds,
            )
        ),
        "triton_topk_topp_from_uniform_preallocated": summarize(
            time_gpu(
                lambda: topk_topp_sample_from_uniform_out(
                    logits,
                    uniforms,
                    output,
                    partial_values=partial_values,
                    partial_tokens=partial_tokens,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    block_vocab_override=args.block_vocab,
                ),
                args.warmup,
                args.rounds,
            )
        ),
        "torch_gpu_topk_topp_from_uniform": summarize(
            time_gpu(
                lambda: topk_topp_sample_from_uniform_reference(
                    logits,
                    uniforms,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                ),
                args.warmup,
                args.rounds,
            )
        ),
        "cpu_roundtrip_topk_topp_from_uniform": summarize(
            time_cpu(
                lambda: cpu_roundtrip_reference(
                    logits,
                    uniforms,
                    args.top_k,
                    args.top_p,
                    args.temperature,
                ),
                args.warmup,
                args.rounds,
            )
        ),
    }
    flashinfer_result = try_flashinfer(logits, args)
    if flashinfer_result:
        result["flashinfer"] = flashinfer_result
    result["ratios"] = {
        "triton_preallocated_vs_torch_gpu": (
            result["torch_gpu_topk_topp_from_uniform"]["median_ms"]
            / result["triton_topk_topp_from_uniform_preallocated"]["median_ms"]
        ),
        "triton_preallocated_vs_cpu_roundtrip": (
            result["cpu_roundtrip_topk_topp_from_uniform"]["median_ms"]
            / result["triton_topk_topp_from_uniform_preallocated"]["median_ms"]
        ),
    }
    if flashinfer_result and flashinfer_result.get("available"):
        result["ratios"]["triton_preallocated_speedup_vs_flashinfer"] = (
            flashinfer_result["flashinfer_topk_topp_from_logits"]["median_ms"]
            / result["triton_topk_topp_from_uniform_preallocated"]["median_ms"]
        )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
