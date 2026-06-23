#!/usr/bin/env python3
"""Benchmark FlashInfer fused sampling on L20."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=40)
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


def torch_gpu_topk_topp_multinomial(logits, top_k, top_p, temperature):
    values, indices = torch.topk(logits / temperature, k=top_k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sorted_probs, sorted_order = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= top_p
    keep[..., 0] = True
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    sample = torch.multinomial(filtered, num_samples=1)
    topk_index = torch.gather(sorted_order, dim=-1, index=sample)
    return torch.gather(indices, dim=-1, index=topk_index).squeeze(-1)


def cpu_roundtrip_topk_topp_multinomial(logits, top_k, top_p, temperature):
    return torch_gpu_topk_topp_multinomial(logits.cpu(), top_k, top_p, temperature)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    flashinfer_cuda_env = configure_flashinfer_cuda13_env(required=True)
    import flashinfer
    import flashinfer.sampling as flashinfer_sampling

    torch.manual_seed(47)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=torch.float16)
    scaled_logits = logits / args.temperature
    seed = torch.full((args.batch,), 12345, device="cuda", dtype=torch.int64)
    offset = torch.zeros((args.batch,), device="cuda", dtype=torch.int64)

    # Smoke calls compile kernels and validate output shape/dtype. Stochastic
    # samplers are distributional, so equality with PyTorch multinomial is not
    # a correctness criterion here.
    flash_out = flashinfer_sampling.top_k_top_p_sampling_from_logits(
        scaled_logits,
        args.top_k,
        args.top_p,
        filter_apply_order="top_k_first",
        deterministic=True,
        seed=seed,
        offset=offset,
    )
    if flash_out.shape != (args.batch,) or flash_out.dtype != torch.int32:
        raise AssertionError("unexpected FlashInfer sampler output shape or dtype")

    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
        "flashinfer_cuda_env": flashinfer_cuda_env.to_dict(),
        "shape": {
            "batch": args.batch,
            "vocab": args.vocab,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "dtype": "float16",
        },
        "torch_gpu_argmax": summarize(
            time_gpu(lambda: torch.argmax(logits, dim=-1), args.warmup, args.rounds)
        ),
        "torch_gpu_topk_topp_multinomial": summarize(
            time_gpu(
                lambda: torch_gpu_topk_topp_multinomial(
                    logits, args.top_k, args.top_p, args.temperature
                ),
                args.warmup,
                args.rounds,
            )
        ),
        "cpu_roundtrip_topk_topp_multinomial": summarize(
            time_cpu(
                lambda: cpu_roundtrip_topk_topp_multinomial(
                    logits, args.top_k, args.top_p, args.temperature
                ),
                args.warmup,
                args.rounds,
            )
        ),
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
    result["ratios"] = {
        "flashinfer_vs_torch_gpu_pipeline": (
            result["torch_gpu_topk_topp_multinomial"]["median_ms"]
            / result["flashinfer_topk_topp_from_logits"]["median_ms"]
        ),
        "flashinfer_vs_cpu_roundtrip_pipeline": (
            result["cpu_roundtrip_topk_topp_multinomial"]["median_ms"]
            / result["flashinfer_topk_topp_from_logits"]["median_ms"]
        ),
        "flashinfer_vs_torch_argmax": (
            result["torch_gpu_argmax"]["median_ms"]
            / result["flashinfer_topk_topp_from_logits"]["median_ms"]
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
