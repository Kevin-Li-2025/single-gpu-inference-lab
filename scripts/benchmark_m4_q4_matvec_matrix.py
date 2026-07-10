#!/usr/bin/env python3
"""Benchmark the custom M4 Q4 x Q8 kernel across Qwen2.5-0.5B layer shapes."""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
from pathlib import Path


QWEN25_05B_SHAPES = (
    ("q_proj", 896, 896),
    ("k_proj", 128, 896),
    ("v_proj", 128, 896),
    ("o_proj", 896, 896),
    ("gate_up_proj", 4864, 896),
    ("down_proj", 896, 4864),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--cache-flush-mib", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/cpu-m4-q4-matvec/qwen25-0p5b-m4",
    )
    return parser.parse_args()


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    source = root / "cpp/m4_q4_matvec.cpp"
    binary = root / "build/cpu/m4_q4_matvec"
    binary.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "clang++",
            "-O3",
            "-std=c++20",
            "-mcpu=apple-m4",
            "-ffast-math",
            "-DNDEBUG",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )

    rows = []
    for name, output_rows, input_cols in QWEN25_05B_SHAPES:
        completed = subprocess.run(
            [
                str(binary),
                "--rows",
                str(output_rows),
                "--cols",
                str(input_cols),
                "--threads",
                str(args.threads),
                "--warmup",
                str(args.warmup),
                "--iterations",
                str(args.iterations),
                "--cache-flush-mib",
                str(args.cache_flush_mib),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        payload["name"] = name
        rows.append(payload)

    same_thread_speedups = [row["speedup_vs_scalar_same_threads"] for row in rows]
    single_thread_speedups = [row["speedup_vs_scalar"] for row in rows]
    summary = {
        "schema_version": 1,
        "implementation": "cpp/m4_q4_matvec.cpp",
        "runner": "scripts/benchmark_m4_q4_matvec_matrix.py",
        "mode": "qwen25_0p5b_model_shaped_q4_q8_matvec_matrix",
        "proof_boundary": "microbenchmark; synthetic packed weights at real layer shapes",
        "host": {
            "machine": platform.machine(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "threads": args.threads,
        "cache_flush_mib": args.cache_flush_mib,
        "shape_count": len(rows),
        "all_correct": all(row["correct"] for row in rows),
        "geomean_speedup_vs_scalar_single_thread": geometric_mean(single_thread_speedups),
        "geomean_speedup_vs_scalar_same_threads": geometric_mean(same_thread_speedups),
        "min_speedup_vs_scalar_same_threads": min(same_thread_speedups),
        "max_speedup_vs_scalar_same_threads": max(same_thread_speedups),
        "rows": rows,
    }

    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
