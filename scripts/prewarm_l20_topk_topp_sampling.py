#!/usr/bin/env python3
"""Prewarm the L20 top-k/top-p Triton sampler compile cache."""

from __future__ import annotations

import argparse
import json
import traceback

import torch

from l20_stack.ops.triton_sampling import (
    topk_topp_sample_from_uniform,
    topk_topp_sample_with_vllm_rng_out,
    topk_topp_sampling_launch_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=torch.float16)
        uniforms = torch.rand((args.batch,), device="cuda", dtype=torch.float32)
        uniform_output = topk_topp_sample_from_uniform(
            logits,
            uniforms,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        config = topk_topp_sampling_launch_config(
            args.vocab,
            args.top_k,
            batch=args.batch,
        )
        partial_shape = (args.batch, config.blocks_per_row, args.top_k)
        partial_values = torch.empty(partial_shape, device="cuda", dtype=torch.float32)
        partial_tokens = torch.empty(partial_shape, device="cuda", dtype=torch.int64)
        vllm_rng_output = torch.empty((args.batch,), device="cuda", dtype=torch.int64)
        expanded_idx_mapping = torch.arange(args.batch, device="cuda", dtype=torch.int64)
        seeds = torch.full((args.batch,), 12345, device="cuda", dtype=torch.int64)
        positions = torch.arange(args.batch, device="cuda", dtype=torch.int64)
        topk_topp_sample_with_vllm_rng_out(
            logits,
            vllm_rng_output,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            expanded_idx_mapping=expanded_idx_mapping,
            seeds=seeds,
            positions=positions,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        torch.cuda.synchronize()
        result = {
            "schema_version": 1,
            "hardware": torch.cuda.get_device_name(),
            "uniform_output_shape": list(uniform_output.shape),
            "uniform_output_dtype": str(uniform_output.dtype),
            "vllm_rng_output_shape": list(vllm_rng_output.shape),
            "vllm_rng_output_dtype": str(vllm_rng_output.dtype),
            "status": "ok",
        }
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-40:],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
