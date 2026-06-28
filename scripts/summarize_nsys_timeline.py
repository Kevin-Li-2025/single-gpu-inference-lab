#!/usr/bin/env python3
"""Summarize Nsight Systems CSV stats for a vLLM serving timeline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-kernels", type=int, default=20)
    return parser.parse_args()


def parse_number(value):
    text = str(value or "").strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "none"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(line for line in handle if not line.startswith("#"))
        ]
    return rows


def find_report_csv(input_dir: Path, report: str) -> Path:
    candidates = [
        input_dir / f"{report}.csv",
        input_dir / f"{report}_{report}.csv",
    ]
    candidates.extend(sorted(input_dir.glob(f"*{report}*.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return input_dir / f"{report}.csv"


def first_key(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    lowered = {key.lower().strip(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return key
    return None


def numeric(row: dict[str, str], names: tuple[str, ...]) -> float:
    key = first_key(row, names)
    if key is None:
        return 0.0
    return parse_number(row.get(key)) or 0.0


def text_value(row: dict[str, str], names: tuple[str, ...]) -> str:
    key = first_key(row, names)
    return row.get(key, "") if key is not None else ""


def normalize_name(row: dict[str, str]) -> str:
    return text_value(row, ("Name", "Kernel Name", "Range", "Operation"))


def rows_with_times(rows: list[dict[str, str]]) -> list[dict]:
    normalized = []
    for row in rows:
        name = normalize_name(row)
        total_ns = numeric(
            row,
            (
                "Total Time (ns)",
                "Total Time",
                "Time (ns)",
                "Duration (ns)",
            ),
        )
        instances = numeric(row, ("Instances", "Calls", "Count", "Num Calls"))
        avg_ns = numeric(
            row,
            (
                "Avg (ns)",
                "Average (ns)",
                "Avg",
                "Average",
                "KAvg (ns)",
                "QAvg (ns)",
                "TAvg (ns)",
            ),
        )
        if not total_ns and avg_ns and instances:
            total_ns = avg_ns * instances
        pct = numeric(row, ("Time (%)", "% Time", "Total Time (%)"))
        normalized.append(
            {
                "name": name,
                "total_time_ns": total_ns,
                "instances": int(instances) if instances else 0,
                "avg_ns": avg_ns,
                "time_pct": pct,
            }
        )
    return normalized


def top_by_time(rows: list[dict], limit: int) -> list[dict]:
    return sorted(rows, key=lambda row: row["total_time_ns"], reverse=True)[:limit]


def count_matching(rows: list[dict], needles: tuple[str, ...]) -> int:
    total = 0
    for row in rows:
        name = row["name"]
        if any(needle in name for needle in needles):
            total += row["instances"]
    return total


def rows_matching(rows: list[dict], needles: tuple[str, ...]) -> list[dict]:
    return [row for row in rows if any(needle in row["name"] for needle in needles)]


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    reports = (
        "cuda_gpu_kern_sum",
        "cuda_kern_exec_sum",
        "cuda_api_sum",
        "nvtx_sum",
        "cuda_gpu_trace",
    )
    files = {report: find_report_csv(input_dir, report) for report in reports}
    raw = {name: read_csv(path) for name, path in files.items()}
    kernels = rows_with_times(raw["cuda_gpu_kern_sum"])
    kernel_exec = rows_with_times(raw["cuda_kern_exec_sum"])
    cuda_api = rows_with_times(raw["cuda_api_sum"])
    nvtx = rows_with_times(raw["nvtx_sum"])
    gpu_trace = rows_with_times(raw["cuda_gpu_trace"])

    qk_needles = ("_l20_qk_norm_rope_kv_kernel", "l20_qk_norm_rope_kv")
    launch_needles = ("cudaLaunchKernel", "cuLaunchKernel", "cudaGraphLaunch")
    graph_needles = ("cudaGraphLaunch",)

    summary = {
        "schema_version": 1,
        "source_dir": str(input_dir),
        "files": {name: str(path) for name, path in files.items() if path.exists()},
        "row_counts": {name: len(rows) for name, rows in raw.items()},
        "cuda_kernel_instance_count": sum(row["instances"] for row in kernels),
        "cuda_kernel_unique_count": len(kernels),
        "cuda_api_call_count": sum(row["instances"] for row in cuda_api),
        "cuda_kernel_launch_api_count": count_matching(cuda_api, launch_needles),
        "cuda_graph_launch_api_count": count_matching(cuda_api, graph_needles),
        "custom_qk_kernel_instance_count": count_matching(kernels, qk_needles),
        "custom_qk_kernel_rows": rows_matching(kernels, qk_needles),
        "top_cuda_kernels_by_time": top_by_time(kernels, args.top_kernels),
        "top_cuda_kernel_exec_by_time": top_by_time(kernel_exec, args.top_kernels),
        "top_cuda_apis_by_time": top_by_time(cuda_api, args.top_kernels),
        "top_nvtx_ranges_by_time": top_by_time(nvtx, args.top_kernels),
        "first_cuda_gpu_trace_rows": gpu_trace[: min(100, len(gpu_trace))],
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    print(serialized)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
