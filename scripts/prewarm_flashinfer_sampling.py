#!/usr/bin/env python3
"""Prewarm FlashInfer sampling JIT with the L20 CUDA 13 nvcc environment."""

from __future__ import annotations

import argparse
import json

import torch

from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    env = configure_flashinfer_cuda13_env(required=True)
    import flashinfer
    import flashinfer.sampling as sampling

    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=torch.float16)
    seed = torch.full((args.batch,), 12345, device="cuda", dtype=torch.int64)
    offset = torch.zeros((args.batch,), device="cuda", dtype=torch.int64)
    output = sampling.top_k_top_p_sampling_from_logits(
        logits,
        args.top_k,
        args.top_p,
        filter_apply_order="top_k_first",
        deterministic=True,
        seed=seed,
        offset=offset,
    )
    torch.cuda.synchronize()
    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "flashinfer_version": getattr(flashinfer, "__version__", "unknown"),
        "flashinfer_cuda_env": env.to_dict(),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "status": "ok",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
