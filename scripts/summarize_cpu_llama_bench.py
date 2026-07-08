#!/usr/bin/env python3
"""Convert llama-bench JSON output into a compact checked-in summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", help="Path to llama-bench -o json output.")
    parser.add_argument("summary_json", help="Output summary JSON path.")
    return parser.parse_args()


def test_name(row: dict[str, Any]) -> str:
    n_prompt = int(row["n_prompt"])
    n_gen = int(row["n_gen"])
    if n_prompt > 0 and n_gen > 0:
        return f"pp{n_prompt}+tg{n_gen}"
    if n_prompt > 0:
        return f"pp{n_prompt}"
    return f"tg{n_gen}"


def row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_prompt": row["n_prompt"],
        "n_gen": row["n_gen"],
        "n_threads": row["n_threads"],
        "avg_ms": row["avg_ns"] / 1_000_000.0,
        "stddev_ms": row["stddev_ns"] / 1_000_000.0,
        "avg_tokens_per_s": row["avg_ts"],
        "stddev_tokens_per_s": row["stddev_ts"],
        "samples_tokens_per_s": row.get("samples_ts", []),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("llama-bench JSON contains no rows")

    first = rows[0]
    thread_sweep: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        thread_sweep.setdefault(test_name(row), []).append(row_summary(row))

    for entries in thread_sweep.values():
        entries.sort(key=lambda item: item["n_threads"])

    best_tests = {
        name: max(entries, key=lambda item: item["avg_tokens_per_s"])
        for name, entries in sorted(thread_sweep.items())
    }
    generation_tests = [
        entry
        for entry in best_tests.values()
        if int(entry["n_prompt"]) == 0 and int(entry["n_gen"]) > 0
    ]
    recommended = max(generation_tests or best_tests.values(), key=lambda item: item["avg_tokens_per_s"])

    return {
        "schema_version": 1,
        "implementation": "scripts/bench_cpu_llama_bench.sh",
        "summary_tool": "scripts/summarize_cpu_llama_bench.py",
        "mode": "llama_bench_real_gguf_cpu",
        "benchmark_semantics": (
            "llama-bench excludes tokenization and sampling; use the "
            "llama_cpp_python smoke for a Python-call-path measurement."
        ),
        "build_commit": first.get("build_commit"),
        "build_number": first.get("build_number"),
        "cpu_info": first.get("cpu_info"),
        "gpu_info": first.get("gpu_info"),
        "backends": first.get("backends"),
        "model_filename": Path(first["model_filename"]).name,
        "model_type": first.get("model_type"),
        "model_size_bytes": first.get("model_size"),
        "model_n_params": first.get("model_n_params"),
        "n_batch": first.get("n_batch"),
        "n_ubatch": first.get("n_ubatch"),
        "thread_values": sorted({row.get("n_threads") for row in rows}),
        "recommended_generation_threads": recommended["n_threads"],
        "n_gpu_layers": first.get("n_gpu_layers"),
        "repetition_count": len(first.get("samples_ts", [])),
        "tests": best_tests,
        "thread_sweep": thread_sweep,
    }


def main() -> int:
    args = parse_args()
    rows = json.loads(Path(args.raw_json).read_text(encoding="utf-8"))
    summary = summarize(rows)
    output = Path(args.summary_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
