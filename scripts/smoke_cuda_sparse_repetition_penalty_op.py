#!/usr/bin/env python3
"""Build and smoke-test the L20 sparse repetition-penalty dispatcher op."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def reference(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    lengths: torch.Tensor,
    penalty: float,
):
    expected = logits.clone()
    for row in range(expected.shape[0]):
        length = int(lengths[row])
        if length <= 0:
            continue
        tokens = token_ids[row, :length]
        values = expected[row].index_select(0, tokens)
        adjusted = torch.where(values > 0, values / penalty, values * penalty)
        expected[row].index_copy_(0, tokens, adjusted)
    return expected


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    build = Path("/tmp/l20-sparse-repetition-penalty-op-smoke")
    build.mkdir(parents=True, exist_ok=True)
    extension = load(
        "l20_sparse_repetition_penalty_cuda",
        [
            root / "integrations/vllm/cuda/l20_sparse_repetition_penalty.cpp",
            root / "integrations/vllm/cuda/l20_sparse_repetition_penalty.cu",
        ],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
        build_directory=build,
    )
    torch.ops.load_library(extension.__file__)

    torch.manual_seed(17)
    logits = torch.randn(8, 151936, device="cuda", dtype=torch.float32)
    base = torch.arange(512, device="cuda", dtype=torch.long)
    offsets = torch.arange(8, device="cuda", dtype=torch.long).unsqueeze(1) * 8191
    token_ids = (base.unsqueeze(0) + offsets) % logits.shape[1]
    lengths = torch.full((8,), token_ids.shape[1], device="cuda", dtype=torch.long)
    expected = reference(logits, token_ids, lengths, 1.1)
    actual = logits.clone()
    torch.ops.l20_stack.sparse_repetition_penalty_out(actual, token_ids, lengths, 1.1)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    print(
        {
            "gpu": torch.cuda.get_device_name(),
            "batch": int(actual.shape[0]),
            "vocab": int(actual.shape[1]),
            "unique_history_tokens": int(token_ids.shape[1]),
            "max_abs_diff": float((actual - expected).abs().max()),
        }
    )


if __name__ == "__main__":
    main()
