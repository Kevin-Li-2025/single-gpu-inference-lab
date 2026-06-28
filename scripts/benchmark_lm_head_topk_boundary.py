#!/usr/bin/env python3
"""Benchmark the LM-head -> top-k boundary on L20.

The key comparison is:

1. materialize full logits with `hidden @ weight.T`, then run `torch.topk`;
2. chunk the vocab and merge per-chunk top-k candidates, avoiding a full logits
   tensor but paying multiple GEMM/top-k launches;
3. for top_k=1 only, run the experimental Triton direct LM-head top-1 kernel.

This keeps the next sampler-fusion step honest: if chunking or direct fusion is
slower than the highly optimized full GEMM path, the right target is a real GEMM
epilogue/upstream integration, not a standalone replacement sampler.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from l20_stack.ops.triton_lm_head_top1 import (
    lm_head_top1_launch_config,
    lm_head_top1_out,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--chunk-vocab", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-vocab", type=int, default=32)
    parser.add_argument("--block-hidden", type=int, default=64)
    parser.add_argument("--include-triton-top1", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 10),
        "p90_ms": percentile(samples, 90),
        "samples_ms": samples,
    }


def time_gpu(fn, warmup: int, rounds: int):
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


def full_logits_topk(hidden, weight, top_k):
    logits = hidden @ weight.T
    return torch.topk(logits, k=top_k, dim=-1)


def chunked_lm_head_topk(hidden, weight, top_k, chunk_vocab):
    value_chunks = []
    index_chunks = []
    vocab = weight.shape[0]
    for start in range(0, vocab, chunk_vocab):
        end = min(vocab, start + chunk_vocab)
        logits = hidden @ weight[start:end].T
        k = min(top_k, end - start)
        values, indices = torch.topk(logits, k=k, dim=-1)
        value_chunks.append(values)
        index_chunks.append(indices + start)
    merged_values = torch.cat(value_chunks, dim=-1)
    merged_indices = torch.cat(index_chunks, dim=-1)
    values, order = torch.topk(merged_values, k=top_k, dim=-1)
    indices = torch.gather(merged_indices, dim=-1, index=order)
    return values, indices


def assert_topk_close(reference, candidate, top_k):
    ref_values, ref_indices = reference
    cand_values, cand_indices = candidate
    if ref_values.shape != cand_values.shape or ref_indices.shape != cand_indices.shape:
        raise AssertionError("top-k output shapes differ")
    atol = 5e-2 if ref_values.dtype in {torch.float16, torch.bfloat16} else 1e-4
    if not torch.allclose(ref_values.float(), cand_values.float(), atol=atol, rtol=1e-3):
        max_err = (ref_values.float() - cand_values.float()).abs().max().item()
        raise AssertionError(f"top-k values differ: max_err={max_err}")
    if top_k <= 16 and not torch.equal(ref_indices.cpu(), cand_indices.cpu()):
        # Equal logits are possible with random low-precision inputs, so only
        # enforce token equality for small K after value equality passed.
        raise AssertionError("top-k indices differ")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.top_k <= 0 or args.top_k > args.vocab:
        raise ValueError("top-k must be in [1, vocab]")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(2026)
    hidden = torch.randn((args.batch, args.hidden), device="cuda", dtype=dtype)
    weight = torch.randn((args.vocab, args.hidden), device="cuda", dtype=dtype)

    reference = full_logits_topk(hidden, weight, args.top_k)
    chunked = chunked_lm_head_topk(hidden, weight, args.top_k, args.chunk_vocab)
    assert_topk_close(reference, chunked, args.top_k)

    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "top_k": args.top_k,
            "chunk_vocab": args.chunk_vocab,
            "dtype": args.dtype,
        },
        "bytes": {
            "materialized_logits_bytes": args.batch
            * args.vocab
            * torch.empty((), dtype=dtype).element_size(),
            "weight_bytes": args.vocab
            * args.hidden
            * torch.empty((), dtype=dtype).element_size(),
            "hidden_bytes": args.batch
            * args.hidden
            * torch.empty((), dtype=dtype).element_size(),
        },
        "full_logits_topk": summarize(
            time_gpu(lambda: full_logits_topk(hidden, weight, args.top_k), args.warmup, args.rounds)
        ),
        "chunked_lm_head_topk": summarize(
            time_gpu(
                lambda: chunked_lm_head_topk(hidden, weight, args.top_k, args.chunk_vocab),
                args.warmup,
                args.rounds,
            )
        ),
    }
    result["ratios"] = {
        "chunked_over_full_logits_topk": (
            result["chunked_lm_head_topk"]["median_ms"]
            / result["full_logits_topk"]["median_ms"]
        ),
        "full_logits_topk_over_chunked": (
            result["full_logits_topk"]["median_ms"]
            / result["chunked_lm_head_topk"]["median_ms"]
        ),
    }

    if args.include_triton_top1:
        if args.top_k != 1:
            raise ValueError("--include-triton-top1 requires --top-k 1")
        config = lm_head_top1_launch_config(
            args.vocab,
            args.hidden,
            block_vocab=args.block_vocab,
            block_hidden=args.block_hidden,
        )
        output_values = torch.empty((args.batch,), device="cuda", dtype=torch.float32)
        output_tokens = torch.empty((args.batch,), device="cuda", dtype=torch.int64)
        partial_values = torch.empty(
            (args.batch, config.blocks_per_row), device="cuda", dtype=torch.float32
        )
        partial_tokens = torch.empty(
            (args.batch, config.blocks_per_row), device="cuda", dtype=torch.int64
        )
        lm_head_top1_out(
            hidden,
            weight,
            output_values,
            output_tokens,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            block_vocab=args.block_vocab,
            block_hidden=args.block_hidden,
        )
        torch.cuda.synchronize()
        ref_values, ref_indices = reference
        if not torch.allclose(output_values, ref_values.squeeze(-1).float(), atol=5e-2, rtol=1e-3):
            max_err = (output_values - ref_values.squeeze(-1).float()).abs().max().item()
            raise AssertionError(f"Triton top-1 values differ: max_err={max_err}")
        if not torch.equal(output_tokens.cpu(), ref_indices.squeeze(-1).cpu()):
            raise AssertionError("Triton top-1 tokens differ")
        result["triton_lm_head_top1_launch"] = config.to_dict()
        result["triton_lm_head_top1"] = summarize(
            time_gpu(
                lambda: lm_head_top1_out(
                    hidden,
                    weight,
                    output_values,
                    output_tokens,
                    partial_values=partial_values,
                    partial_tokens=partial_tokens,
                    block_vocab=args.block_vocab,
                    block_hidden=args.block_hidden,
                ),
                args.warmup,
                args.rounds,
            )
        )
        result["ratios"]["triton_top1_over_full_logits_top1"] = (
            result["triton_lm_head_top1"]["median_ms"]
            / result["full_logits_topk"]["median_ms"]
        )
        result["ratios"]["full_logits_top1_over_triton_top1"] = (
            result["full_logits_topk"]["median_ms"]
            / result["triton_lm_head_top1"]["median_ms"]
        )

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
