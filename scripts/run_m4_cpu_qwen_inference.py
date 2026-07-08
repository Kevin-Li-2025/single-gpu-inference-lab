#!/usr/bin/env python3
"""Run optimized local M4 CPU inference through llama.cpp's C++ completion path."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


DEFAULT_MODEL_GLOB = (
    ".cache/huggingface/hub/models--Qwen--Qwen2.5-Coder-0.5B-Instruct-GGUF/"
    "snapshots/*/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
)
DEFAULT_PROMPT = (
    "Write a concise Python function named add_numbers that returns the sum "
    "of two integers."
)
EVAL_TIME_RE = re.compile(
    r"(?P<label>prompt eval|eval) time\s*=\s*"
    r"(?P<time_ms>[0-9.]+)\s*ms\s*/\s*"
    r"(?P<count>[0-9]+)\s*(?P<count_unit>tokens|runs)\s*"
    r"\(\s*(?P<ms_per_token>[0-9.]+)\s*ms per token,\s*"
    r"(?P<tokens_per_s>[0-9.]+)\s*tokens per second"
)
TOTAL_TIME_RE = re.compile(
    r"total time\s*=\s*(?P<time_ms>[0-9.]+)\s*ms\s*/\s*"
    r"(?P<tokens>[0-9]+)\s*tokens"
)
GRAPHS_REUSED_RE = re.compile(r"graphs reused\s*=\s*(?P<count>[0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llama-completion-bin",
        default="build/llama.cpp/build-cpu/bin/llama-completion",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--ctx-size", type=positive_int, default=512)
    parser.add_argument("--predict", type=positive_int, default=64)
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--ubatch-size", type=positive_int, default=128)
    parser.add_argument("--threads", type=positive_int, default=6)
    parser.add_argument("--threads-batch", type=positive_int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=positive_int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--no-mlock", action="store_true")
    return parser.parse_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_model_path(value: str | None) -> Path:
    if value:
        path = Path(value).expanduser()
    else:
        matches = sorted((Path.home()).glob(DEFAULT_MODEL_GLOB))
        if not matches:
            raise FileNotFoundError(
                "Qwen2.5-Coder-0.5B Q4_K_M GGUF is not in the Hugging Face cache"
            )
        path = matches[-1]
    validate_gguf(path)
    return path


def validate_gguf(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"GGUF model path does not exist: {path}")
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise ValueError(f"not a valid GGUF file: {path} is missing the GGUF magic header")


def output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    return repo_root() / "benchmarks/results/cpu-real-model/qwen25-coder-0p5b-q4km-m4-inference"


def build_command(args: argparse.Namespace, model_path: Path, log_path: Path) -> list[str]:
    binary = Path(args.llama_completion_bin)
    if not binary.is_absolute():
        binary = repo_root() / binary
    if not binary.exists():
        raise FileNotFoundError(f"missing llama-completion binary: {binary}")

    command = [
        str(binary),
        "-m",
        str(model_path),
        "-p",
        args.prompt,
        "-n",
        str(args.predict),
        "-c",
        str(args.ctx_size),
        "-b",
        str(args.batch_size),
        "-ub",
        str(args.ubatch_size),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads_batch),
        "-ngl",
        "0",
        "--mmap",
        "--temp",
        str(args.temperature),
        "--top-k",
        str(args.top_k),
        "--top-p",
        str(args.top_p),
        "--repeat-penalty",
        str(args.repeat_penalty),
        "--seed",
        str(args.seed),
        "--no-display-prompt",
        "--simple-io",
        "--log-file",
        str(log_path),
        "-no-cnv",
    ]
    if not args.no_mlock:
        command.append("--mlock")
    return command


def parse_common_perf(log_text: str) -> dict[str, object]:
    perf: dict[str, object] = {}
    for line in log_text.splitlines():
        if "prompt eval time" in line or " eval time" in line:
            match = EVAL_TIME_RE.search(line)
            if match:
                key = "prompt_eval" if match.group("label") == "prompt eval" else "decode_eval"
                perf[key] = {
                    "time_ms": float(match.group("time_ms")),
                    "count": int(match.group("count")),
                    "count_unit": match.group("count_unit"),
                    "ms_per_token": float(match.group("ms_per_token")),
                    "tokens_per_s": float(match.group("tokens_per_s")),
                }
            continue
        if "total time" in line:
            match = TOTAL_TIME_RE.search(line)
            if match:
                perf["total"] = {
                    "time_ms": float(match.group("time_ms")),
                    "tokens": int(match.group("tokens")),
                }
            continue
        if "graphs reused" in line:
            match = GRAPHS_REUSED_RE.search(line)
            if match:
                perf["graphs_reused"] = int(match.group("count"))
    return perf


def sanitize_command(command: list[str], model_path: Path, log_path: Path) -> list[str]:
    root = repo_root()
    sanitized: list[str] = []
    for item in command:
        if item == str(model_path):
            sanitized.append(model_path.name)
            continue
        if item == str(log_path):
            sanitized.append("runtime.log")
            continue
        path = Path(item)
        if path.is_absolute():
            try:
                sanitized.append(str(path.resolve().relative_to(root)))
                continue
            except ValueError:
                pass
        sanitized.append(item)
    return sanitized


def main() -> int:
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "runtime.log"
    command = build_command(args, model_path, log_path)

    start = time.perf_counter()
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output_text = completed.stdout
    stderr_text = completed.stderr

    (out_dir / "output.txt").write_text(output_text, encoding="utf-8")
    if stderr_text:
        (out_dir / "stderr.txt").write_text(stderr_text, encoding="utf-8")
    runtime_log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    summary = {
        "schema_version": 1,
        "implementation": "scripts/run_m4_cpu_qwen_inference.py",
        "mode": "m4_cpu_llama_completion_inference",
        "model_filename": model_path.name,
        "model_size_bytes": model_path.stat().st_size,
        "n_gpu_layers": 0,
        "ctx_size": args.ctx_size,
        "predict_tokens": args.predict,
        "batch_size": args.batch_size,
        "ubatch_size": args.ubatch_size,
        "threads": args.threads,
        "threads_batch": args.threads_batch,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "repeat_penalty": args.repeat_penalty,
        "seed": args.seed,
        "mlock": not args.no_mlock,
        "elapsed_ms": elapsed_ms,
        "returncode": completed.returncode,
        "output_chars": len(output_text),
        "stderr_chars": len(stderr_text),
        "runtime_log_chars": len(runtime_log_text),
        "common_perf": parse_common_perf(runtime_log_text),
        "prompt_text": args.prompt,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
        },
        "command": sanitize_command(command, model_path, log_path),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
