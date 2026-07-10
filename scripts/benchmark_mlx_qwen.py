#!/usr/bin/env python3
"""Measure persistent-process MLX generation for the M4 same-model comparison."""

from __future__ import annotations

import argparse
import json
import statistics

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler


DEFAULT_PROMPT = (
    "Write a concise Python function named add_numbers that returns the sum "
    "of two integers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    return parser.parse_args()


def generate_once(model, tokenizer, prompt: str, max_tokens: int):
    final = None
    token_checksum = 0
    text = []
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0, top_k=1),
    ):
        final = response
        token_checksum = (token_checksum * 1315423911 + int(response.token)) & 0xFFFFFFFF
        text.append(response.text)
    if final is None:
        raise RuntimeError("MLX generated no tokens")
    return final, token_checksum, "".join(text)


def main() -> int:
    args = parse_args()
    model, tokenizer = load(args.model)
    generate_once(model, tokenizer, args.prompt, args.warmup_tokens)

    rows = []
    outputs = []
    for _ in range(args.iterations):
        response, checksum, text = generate_once(
            model, tokenizer, args.prompt, args.max_tokens
        )
        rows.append(
            {
                "prompt_tokens": response.prompt_tokens,
                "prompt_tokens_per_s": response.prompt_tps,
                "generation_tokens": response.generation_tokens,
                "generation_tokens_per_s": response.generation_tps,
                "peak_memory_gb": response.peak_memory,
                "token_checksum": checksum,
                "finish_reason": response.finish_reason,
            }
        )
        outputs.append(text)

    generation_tps = [row["generation_tokens_per_s"] for row in rows]
    prompt_tps = [row["prompt_tokens_per_s"] for row in rows]
    payload = {
        "schema_version": 1,
        "implementation": "scripts/benchmark_mlx_qwen.py",
        "mode": "persistent_mlx_same_model_4bit",
        "model": args.model,
        "quantization_boundary": "MLX 4-bit; not bitwise-identical to GGUF Q4_K_M",
        "iterations": args.iterations,
        "max_tokens": args.max_tokens,
        "output_stable": len(set(outputs)) == 1,
        "median_generation_tokens_per_s": statistics.median(generation_tps),
        "median_prompt_tokens_per_s": statistics.median(prompt_tps),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
