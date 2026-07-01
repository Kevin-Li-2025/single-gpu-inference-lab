#!/usr/bin/env python3
"""Benchmark fused dense-penalty + top-k/top-p sampling.

This is a prototype boundary: token counts are dense ``[batch, vocab]`` tensors
so the fused arithmetic is easy to verify. A production serving path should
replace this with vLLM's sparse token-history state.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from l20_stack.ops.triton_sampling import (
    apply_dense_token_penalties_reference,
    topk_topp_penalty_sample_from_uniform,
    topk_topp_penalty_sample_from_uniform_out,
    topk_topp_penalty_sample_from_uniform_reference,
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
    parser.add_argument("--frequency-penalty", type=float, default=0.1)
    parser.add_argument("--presence-penalty", type=float, default=0.1)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--history-tokens", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-vocab", type=int)
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


def make_dense_counts(batch: int, vocab: int, history_tokens: int):
    counts = torch.zeros((batch, vocab), device="cuda", dtype=torch.int16)
    if history_tokens <= 0:
        return counts
    token_ids = torch.randint(0, vocab, (batch, history_tokens), device="cuda")
    ones = torch.ones_like(token_ids, dtype=counts.dtype)
    counts.scatter_add_(1, token_ids, ones)
    return counts


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(97)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=dtype)
    token_counts = make_dense_counts(args.batch, args.vocab, args.history_tokens)
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
    adjusted_logits = torch.empty_like(logits, dtype=torch.float32)

    expected = topk_topp_penalty_sample_from_uniform_reference(
        logits,
        token_counts,
        uniforms,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
    )
    actual = topk_topp_penalty_sample_from_uniform(
        logits,
        token_counts,
        uniforms,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError(
            f"fused penalty sampler mismatch: actual={actual.cpu()} expected={expected.cpu()}"
        )
    topk_topp_penalty_sample_from_uniform_out(
        logits,
        token_counts,
        uniforms,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(output.cpu(), expected.cpu()):
        raise AssertionError("preallocated fused penalty sampler mismatch")

    def baseline_preallocated():
        adjusted_logits.copy_(
            apply_dense_token_penalties_reference(
                logits,
                token_counts,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
            )
        )
        topk_topp_sample_from_uniform_out(
            adjusted_logits,
            uniforms,
            output,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            block_vocab_override=args.block_vocab,
        )

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
            "history_tokens": args.history_tokens,
        },
        "penalties": {
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
        },
        "launch": config.to_dict(),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "fused_penalty_topk_topp": summarize(
            time_gpu(
                lambda: topk_topp_penalty_sample_from_uniform_out(
                    logits,
                    token_counts,
                    uniforms,
                    output,
                    partial_values=partial_values,
                    partial_tokens=partial_tokens,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    block_vocab_override=args.block_vocab,
                ),
                args.warmup,
                args.rounds,
            )
        ),
        "baseline_apply_penalty_then_topk_topp": summarize(
            time_gpu(baseline_preallocated, args.warmup, args.rounds)
        ),
        "torch_reference": summarize(
            time_gpu(
                lambda: topk_topp_penalty_sample_from_uniform_reference(
                    logits,
                    token_counts,
                    uniforms,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                ),
                args.warmup,
                args.rounds,
            )
        ),
        "cpu_roundtrip_reference": summarize(
            time_cpu(
                lambda: topk_topp_sample_from_uniform_reference(
                    apply_dense_token_penalties_reference(
                        logits.cpu(),
                        token_counts.cpu(),
                        frequency_penalty=args.frequency_penalty,
                        presence_penalty=args.presence_penalty,
                        repetition_penalty=args.repetition_penalty,
                    ),
                    uniforms.cpu(),
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                ),
                max(1, args.warmup // 10),
                max(1, args.rounds // 10),
            )
        ),
    }
    result["ratios"] = {
        "fused_speedup_vs_apply_then_sample": (
            result["baseline_apply_penalty_then_topk_topp"]["median_ms"]
            / result["fused_penalty_topk_topp"]["median_ms"]
        ),
        "fused_speedup_vs_torch_reference": (
            result["torch_reference"]["median_ms"]
            / result["fused_penalty_topk_topp"]["median_ms"]
        ),
        "fused_speedup_vs_cpu_roundtrip": (
            result["cpu_roundtrip_reference"]["median_ms"]
            / result["fused_penalty_topk_topp"]["median_ms"]
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
