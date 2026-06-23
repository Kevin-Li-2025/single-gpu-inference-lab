#!/usr/bin/env python3
"""Summarize Nsight Compute raw CSV into an L20 roofline dashboard."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

L20_DRAM_BW_GBPS = 864.0
L20_FP16_TFLOPS = 59.8

METRIC_ALIASES = {
    "duration_ns": ["gpu__time_duration.sum"],
    "dram_bytes": ["dram__bytes.sum"],
    "dram_pct": ["dram__throughput.avg.pct_of_peak_sustained_elapsed"],
    "l2_pct": ["lts__throughput.avg.pct_of_peak_sustained_elapsed"],
    "sm_pct": ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    "active_warps_pct": [
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ],
    "l2_read_sectors": ["lts__t_sectors_srcunit_tex_op_read.sum"],
    "l2_write_sectors": ["lts__t_sectors_srcunit_tex_op_write.sum"],
    "l1_global_load_sectors": ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"],
    "l1_global_store_sectors": ["l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum"],
    "long_scoreboard_pct": ["smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"],
    "short_scoreboard_pct": ["smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct"],
    "barrier_pct": ["smsp__warp_issue_stalled_barrier_per_warp_active.pct"],
    "membar_pct": ["smsp__warp_issue_stalled_membar_per_warp_active.pct"],
    "fadd": ["smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"],
    "fmul": ["smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"],
    "ffma": ["smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def parse_number(value: str):
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "inf", "-inf"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_unit(value: float, unit: str, metric_name: str) -> float:
    unit_key = (unit or "").strip().lower()
    if metric_name == "gpu__time_duration.sum":
        if unit_key in {"second", "seconds", "s"}:
            return value * 1e9
        if unit_key in {"msecond", "ms", "millisecond", "milliseconds"}:
            return value * 1e6
        if unit_key in {"usecond", "us", "microsecond", "microseconds"}:
            return value * 1e3
    if unit_key in {"kbyte", "kb"}:
        return value * 1_000
    if unit_key in {"mbyte", "mb"}:
        return value * 1_000_000
    if unit_key in {"gbyte", "gb"}:
        return value * 1_000_000_000
    return value


def row_value(row, names):
    lowered = {key.lower().strip(): value for key, value in row.items() if key is not None}
    for name in names:
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def read_ncu_csv(path: Path):
    rows_by_kernel = defaultdict(dict)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(line for line in handle if not line.startswith("#"))
        for row in reader:
            metric = row_value(row, ["Metric Name", "Metric Name "])
            if not metric:
                continue
            kernel = row_value(row, ["Kernel Name", "Kernel", "Name"]) or "unknown"
            raw_value = row_value(row, ["Metric Value", "Value"])
            value = parse_number(raw_value)
            if value is None:
                continue
            unit = row_value(row, ["Metric Unit", "Unit"]) or ""
            rows_by_kernel[kernel][metric.strip()] = normalize_unit(value, unit, metric.strip())
    return rows_by_kernel


def first_metric(metrics, aliases):
    for name in aliases:
        if name in metrics:
            return metrics[name]
    return None


def summarize_kernel(kernel, metrics):
    extracted = {key: first_metric(metrics, aliases) for key, aliases in METRIC_ALIASES.items()}
    flops = None
    if (
        extracted["ffma"] is not None
        or extracted["fadd"] is not None
        or extracted["fmul"] is not None
    ):
        flops = (
            2.0 * (extracted["ffma"] or 0.0)
            + (extracted["fadd"] or 0.0)
            + (extracted["fmul"] or 0.0)
        )
    duration_ns = extracted["duration_ns"]
    dram_bytes = extracted["dram_bytes"]
    achieved_gbps = None
    arithmetic_intensity = None
    if duration_ns and dram_bytes is not None:
        achieved_gbps = dram_bytes / duration_ns
    if flops is not None and dram_bytes:
        arithmetic_intensity = flops / dram_bytes
    roofline_bound = None
    balance = (L20_FP16_TFLOPS * 1_000) / L20_DRAM_BW_GBPS
    if arithmetic_intensity is not None:
        roofline_bound = "memory_bound" if arithmetic_intensity < balance else "compute_bound"
    l2_sectors = None
    if extracted["l2_read_sectors"] is not None or extracted["l2_write_sectors"] is not None:
        l2_sectors = (extracted["l2_read_sectors"] or 0.0) + (extracted["l2_write_sectors"] or 0.0)
    l1_sectors = None
    if (
        extracted["l1_global_load_sectors"] is not None
        or extracted["l1_global_store_sectors"] is not None
    ):
        l1_sectors = (extracted["l1_global_load_sectors"] or 0.0) + (
            extracted["l1_global_store_sectors"] or 0.0
        )
    sector_excess_ratio = None
    if l1_sectors and l2_sectors is not None:
        sector_excess_ratio = l1_sectors / l2_sectors if l2_sectors else math.inf
    return {
        "kernel_name": kernel,
        "duration_ns": duration_ns,
        "dram_bytes": dram_bytes,
        "estimated_flops": flops,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "roofline_balance_flops_per_byte": balance,
        "roofline_class": roofline_bound,
        "achieved_memory_bandwidth_gbps": achieved_gbps,
        "memory_bandwidth_utilization_pct": extracted["dram_pct"],
        "l2_throughput_utilization_pct": extracted["l2_pct"],
        "sm_throughput_utilization_pct": extracted["sm_pct"],
        "active_warps_pct": extracted["active_warps_pct"],
        "sector_excess_ratio_l1_over_l2": sector_excess_ratio,
        "stall_long_scoreboard_pct": extracted["long_scoreboard_pct"],
        "stall_short_scoreboard_pct": extracted["short_scoreboard_pct"],
        "stall_barrier_pct": extracted["barrier_pct"],
        "stall_membar_pct": extracted["membar_pct"],
        "raw_metrics": {name: metrics[name] for name in sorted(metrics)},
    }


def render_markdown(summary):
    lines = [
        "# Nsight Roofline Summary",
        "",
        "| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | SM % | Active warps % | Long scoreboard % | Sector excess |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for kernel in summary["kernels"]:

        def fmt(value, digits=2):
            if value is None:
                return "n/a"
            if isinstance(value, float) and math.isinf(value):
                return "inf"
            return f"{value:.{digits}f}"

        lines.append(
            "| {name} | {ai} | {roofline} | {gbps} | {dram} | {l2} | {sm} | {warps} | {long} | {sector} |".format(
                name=kernel["kernel_name"],
                ai=fmt(kernel["arithmetic_intensity_flops_per_byte"]),
                roofline=kernel["roofline_class"] or "n/a",
                gbps=fmt(kernel["achieved_memory_bandwidth_gbps"]),
                dram=fmt(kernel["memory_bandwidth_utilization_pct"]),
                l2=fmt(kernel["l2_throughput_utilization_pct"]),
                sm=fmt(kernel["sm_throughput_utilization_pct"]),
                warps=fmt(kernel["active_warps_pct"]),
                long=fmt(kernel["stall_long_scoreboard_pct"]),
                sector=fmt(kernel["sector_excess_ratio_l1_over_l2"]),
            )
        )
    lines.append("")
    lines.append(
        "Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = read_ncu_csv(args.csv)
    summary = {
        "schema_version": 1,
        "source_csv": str(args.csv),
        "hardware": {
            "gpu": "NVIDIA L20",
            "dram_bandwidth_gbps": L20_DRAM_BW_GBPS,
            "fp16_tflops": L20_FP16_TFLOPS,
        },
        "kernels": [summarize_kernel(kernel, metrics) for kernel, metrics in sorted(rows.items())],
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
