#!/usr/bin/env python3
"""Run real-Qwen llama.cpp baseline/custom and optional MLX A/B on Apple M4."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PROMPT = (
    "Write a concise Python function named add_numbers that returns the sum "
    "of two integers."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-bench", required=True)
    parser.add_argument("--llama-completion", required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--threads-batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--completion-runs", type=int, default=4)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--mlx-python")
    parser.add_argument(
        "--mlx-model", default="mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/cpu-m4-q4k-real-model/qwen25-coder-0p5b-v1",
    )
    return parser.parse_args()


def candidate_env(trace: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["GGML_M4_Q4K_CUSTOM"] = "1"
    if trace:
        env["GGML_M4_Q4K_TRACE"] = "1"
    return env


def run_bench(args: argparse.Namespace, candidate: bool) -> dict:
    command = [
        args.llama_bench,
        "-m",
        args.model,
        "-p",
        "0",
        "-n",
        str(args.tokens),
        "-t",
        str(args.threads),
        "-ngl",
        "0",
        "-r",
        str(args.repetitions),
        "-o",
        "json",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=candidate_env() if candidate else None,
    )
    rows = json.loads(completed.stdout)
    generation = next(row for row in rows if row["n_prompt"] == 0)
    return {
        "avg_tokens_per_s": generation["avg_ts"],
        "stddev_tokens_per_s": generation["stddev_ts"],
        "samples_tokens_per_s": generation["samples_ts"],
        "threads": generation["n_threads"],
        "build_commit": generation["build_commit"],
    }


def parse_completion_perf(log_text: str) -> dict[str, float]:
    import re

    pattern = re.compile(
        r"(?P<label>prompt eval|eval) time\s*=\s*(?P<ms>[0-9.]+) ms\s*/\s*"
        r"(?P<count>[0-9]+) (?:tokens|runs)\s*\(\s*(?P<per>[0-9.]+) ms per token,\s*"
        r"(?P<tps>[0-9.]+) tokens per second"
    )
    result = {}
    for match in pattern.finditer(log_text):
        prefix = "prompt" if match.group("label") == "prompt eval" else "decode"
        result[f"{prefix}_ms"] = float(match.group("ms"))
        result[f"{prefix}_tokens_per_s"] = float(match.group("tps"))
        result[f"{prefix}_count"] = int(match.group("count"))
    if "decode_tokens_per_s" not in result:
        raise RuntimeError("llama-completion performance counters were not found")
    return result


def run_completion(args: argparse.Namespace, candidate: bool, directory: Path) -> dict:
    log_path = directory / "runtime.log"
    command = [
        args.llama_completion,
        "-m",
        args.model,
        "-p",
        args.prompt,
        "-n",
        str(args.tokens),
        "-c",
        "512",
        "-b",
        "128",
        "-ub",
        "128",
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads_batch),
        "-ngl",
        "0",
        "--mmap",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--repeat-penalty",
        "1",
        "--seed",
        "7",
        "--no-display-prompt",
        "--simple-io",
        "--log-file",
        str(log_path),
        "-no-cnv",
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        env=candidate_env(trace=candidate) if candidate else None,
    )
    perf = parse_completion_perf(log_path.read_text(encoding="utf-8"))
    perf["output"] = completed.stdout
    perf["custom_trace_hit"] = "kevin_m4_q4k" in completed.stderr
    return perf


def median(rows: list[dict], key: str) -> float:
    return statistics.median(row[key] for row in rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run both orders to reduce first-mode and thermal bias.
    bench_baseline = run_bench(args, candidate=False)
    bench_candidate = run_bench(args, candidate=True)

    baseline_rows = []
    candidate_rows = []
    output_pairs = []
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for iteration in range(args.completion_runs):
            modes = (False, True) if iteration % 2 == 0 else (True, False)
            pair = {}
            for candidate in modes:
                name = "candidate" if candidate else "baseline"
                run_dir = root / f"{iteration}-{name}"
                run_dir.mkdir()
                row = run_completion(args, candidate, run_dir)
                pair[name] = row["output"]
                (candidate_rows if candidate else baseline_rows).append(row)
            output_pairs.append(pair["baseline"] == pair["candidate"])

    mlx = None
    if args.mlx_python:
        command = [
            args.mlx_python,
            str(Path(__file__).with_name("benchmark_mlx_qwen.py")),
            "--model",
            args.mlx_model,
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.tokens),
            "--iterations",
            str(args.repetitions),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        mlx = json.loads(completed.stdout)

    baseline_decode = median(baseline_rows, "decode_tokens_per_s")
    candidate_decode = median(candidate_rows, "decode_tokens_per_s")
    payload = {
        "schema_version": 1,
        "implementation": "scripts/run_m4_q4k_real_model_ab.py",
        "mode": "apple_m4_real_qwen_llama_custom_mlx_ab",
        "model_filename": Path(args.model).name,
        "llama_quantization": "GGUF Q4_K_M",
        "custom_scope": "raw Q4_K tensors; opt-in; repacked Q4_K disabled",
        "threads": args.threads,
        "tokens": args.tokens,
        "bench": {
            "baseline": bench_baseline,
            "candidate": bench_candidate,
            "candidate_speedup": bench_candidate["avg_tokens_per_s"]
            / bench_baseline["avg_tokens_per_s"],
        },
        "completion": {
            "runs_per_mode": args.completion_runs,
            "all_outputs_exact": all(output_pairs),
            "all_candidate_traces_hit": all(
                row["custom_trace_hit"] for row in candidate_rows
            ),
            "baseline_median_prompt_tokens_per_s": median(
                baseline_rows, "prompt_tokens_per_s"
            ),
            "candidate_median_prompt_tokens_per_s": median(
                candidate_rows, "prompt_tokens_per_s"
            ),
            "baseline_median_decode_tokens_per_s": baseline_decode,
            "candidate_median_decode_tokens_per_s": candidate_decode,
            "candidate_decode_speedup": candidate_decode / baseline_decode,
            "baseline_rows": [
                {key: value for key, value in row.items() if key != "output"}
                for row in baseline_rows
            ],
            "candidate_rows": [
                {key: value for key, value in row.items() if key != "output"}
                for row in candidate_rows
            ],
        },
        "mlx": mlx,
        "comparison_boundary": (
            "llama baseline/custom use identical GGUF bytes; MLX uses the same model "
            "architecture with a different 4-bit format and Metal backend"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
