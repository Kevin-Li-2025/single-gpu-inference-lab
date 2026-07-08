#!/usr/bin/env python3
"""Run a real GGUF small-model CPU decode benchmark with llama.cpp bindings."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any


DEFAULT_REPO_ID = "bartowski/SmolLM2-135M-Instruct-GGUF"
DEFAULT_FILENAME = "SmolLM2-135M-Instruct-Q4_K_M.gguf"
DEFAULT_PROMPT = (
    "Write a concise Python function named add_numbers that returns the sum "
    "of two integers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a real GGUF model on CPU via llama.cpp."
    )
    model = parser.add_argument_group("model")
    model.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    model.add_argument("--filename", default=DEFAULT_FILENAME)
    model.add_argument("--model-path", default=None, help="Use a local GGUF instead of HF.")
    model.add_argument("--cache-dir", default=None)
    model.add_argument("--local-files-only", action="store_true")

    run = parser.add_argument_group("run")
    run.add_argument("--prompt", default=DEFAULT_PROMPT)
    run.add_argument("--decode-tokens", type=positive_int, default=16)
    run.add_argument("--n-ctx", type=positive_int, default=256)
    run.add_argument("--n-batch", type=positive_int, default=128)
    run.add_argument("--threads", type=positive_int, default=4)
    run.add_argument("--threads-batch", type=positive_int, default=None)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--repeat-penalty", type=float, default=1.0)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top-k", type=positive_int, default=1)
    run.add_argument("--top-p", type=float, default=1.0)
    run.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        path = Path(args.model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"GGUF model path does not exist: {path}")
        validate_gguf(path)
        return path

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface_hub is required when --model-path is not provided"
        ) from error

    path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
    )
    validate_gguf(path)
    return path


def validate_gguf(path: Path) -> None:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise ValueError(
            f"not a valid GGUF file: {path} is missing the GGUF magic header"
        )


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def decode_text(llm: Any, tokens: list[int]) -> str:
    return llm.detokenize(tokens).decode("utf-8", errors="replace")


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from llama_cpp import Llama
    except ImportError as error:
        raise RuntimeError("llama_cpp is required for this benchmark") from error

    model_path = resolve_model_path(args)
    model_size = model_path.stat().st_size

    load_start = time.perf_counter()
    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=0,
        n_ctx=args.n_ctx,
        n_batch=args.n_batch,
        n_threads=args.threads,
        n_threads_batch=args.threads_batch or args.threads,
        seed=args.seed,
        logits_all=False,
        verbose=args.verbose,
    )
    load_ms = (time.perf_counter() - load_start) * 1000.0

    prompt_tokens = llm.tokenize(args.prompt.encode("utf-8"), add_bos=True, special=True)
    if len(prompt_tokens) + args.decode_tokens > args.n_ctx:
        raise ValueError(
            f"prompt+decode needs {len(prompt_tokens) + args.decode_tokens} tokens, "
            f"but --n-ctx is {args.n_ctx}"
        )

    llm.reset()
    prefill_start = time.perf_counter()
    llm.eval(prompt_tokens)
    prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

    generated: list[int] = []
    step_ms: list[float] = []
    decode_start = time.perf_counter()
    for _ in range(args.decode_tokens):
        step_start = time.perf_counter()
        token = int(
            llm.sample(
                top_k=args.top_k,
                top_p=args.top_p,
                temp=args.temperature,
                repeat_penalty=args.repeat_penalty,
            )
        )
        generated.append(token)
        llm.eval([token])
        step_ms.append((time.perf_counter() - step_start) * 1000.0)
    decode_ms = (time.perf_counter() - decode_start) * 1000.0

    total_eval_tokens = len(prompt_tokens) + len(generated)
    output_text = decode_text(llm, generated)
    token_checksum = sum((idx + 1) * token for idx, token in enumerate(generated))

    return {
        "schema_version": 1,
        "implementation": "scripts/benchmark_cpu_real_model.py",
        "mode": "real_gguf_cpu_decode",
        "backend": "llama_cpp_python",
        "model_repo_id": args.repo_id if not args.model_path else None,
        "model_filename": args.filename if not args.model_path else model_path.name,
        "model_size_bytes": model_size,
        "n_gpu_layers": 0,
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "threads": args.threads,
        "threads_batch": args.threads_batch or args.threads,
        "seed": args.seed,
        "prompt_text": args.prompt,
        "prompt_tokens": len(prompt_tokens),
        "decode_tokens_requested": args.decode_tokens,
        "decode_tokens": len(generated),
        "generated_text": output_text,
        "generated_token_checksum": token_checksum,
        "load_ms": load_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "total_eval_ms": prefill_ms + decode_ms,
        "median_decode_step_ms": statistics.median(step_ms) if step_ms else 0.0,
        "p90_decode_step_ms": percentile(step_ms, 0.90),
        "decode_tokens_per_s": len(generated) * 1000.0 / max(decode_ms, 1e-9),
        "total_eval_tokens_per_s": total_eval_tokens
        * 1000.0
        / max(prefill_ms + decode_ms, 1e-9),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
    }


def main() -> int:
    args = parse_args()
    result = benchmark(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
