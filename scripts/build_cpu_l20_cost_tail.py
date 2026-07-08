#!/usr/bin/env python3
"""Build L20 cost-per-token and tail-latency tables from serving artifacts."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


DEFAULT_ARTIFACT_DIR = Path(
    "benchmarks/results/cpu-l20-break-even/qwen25-coder-0p5b-identical-model-v1"
)
DEFAULT_L20_HOURLY_USD = 0.80
DEFAULT_PRICE_SOURCE = (
    "https://inferencebench.io/gpus/nvidia-l20/ "
    "(illustrative public L20 rental rate; override with --l20-hourly-usd for real billing)"
)
SHAPE_DIR_RE = re.compile(r"^p(?P<prompt>\d+)-o(?P<output>\d+)$")
REPORT_RE = re.compile(r"^c(?P<concurrency>\d+)-i(?P<input>\d+)-r(?P<run>\d+)\.json$")
PERCENTILE_METRICS = (
    "ttft_ms",
    "itl_ms",
    "tpot_ms",
    "e2el_ms",
)
SUMMARY_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--l20-hourly-usd", type=float, default=DEFAULT_L20_HOURLY_USD)
    parser.add_argument("--price-source", default=DEFAULT_PRICE_SOURCE)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def cost_per_1m(hourly_usd: float, units_per_s: float | None) -> float | None:
    if units_per_s is None or units_per_s <= 0.0:
        return None
    return hourly_usd * 1_000_000.0 / (units_per_s * 3600.0)


def shape_tokens(shape_dir: Path) -> tuple[int, int]:
    match = SHAPE_DIR_RE.match(shape_dir.name)
    if not match:
        raise ValueError(f"cannot parse prompt/output tokens from {shape_dir}")
    return int(match.group("prompt")), int(match.group("output"))


def reports_for_shape(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(run_dir.glob("c*-i*-r*.json")):
        match = REPORT_RE.match(path.name)
        if not match:
            continue
        shape = f"c{match.group('concurrency')}-i{match.group('input')}"
        groups.setdefault(shape, []).append(load_json(path))
    return groups


def summarize_reports(
    reports: list[dict[str, Any]],
    *,
    hourly_usd: float,
    output_tokens: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {"runs": len(reports)}
    if not reports:
        return row

    for metric in SUMMARY_METRICS:
        row[metric] = median([float(report[metric]) for report in reports if metric in report])

    # vLLM benchmark stores per-run percentiles, not every request latency in
    # these compact artifacts, so this table reports medians of run percentiles.
    for family in PERCENTILE_METRICS:
        for prefix in ("median", "p95", "p99"):
            key = f"{prefix}_{family}"
            row[key] = median([float(report[key]) for report in reports if key in report])

    if row.get("request_throughput") is None and row.get("output_throughput") is not None:
        row["request_throughput"] = float(row["output_throughput"]) / output_tokens

    row["cost_per_1m_output_tokens_usd"] = cost_per_1m(
        hourly_usd,
        row.get("output_throughput"),
    )
    row["cost_per_1m_total_tokens_usd"] = cost_per_1m(
        hourly_usd,
        row.get("total_token_throughput"),
    )
    row["cost_per_1m_requests_usd"] = cost_per_1m(
        hourly_usd,
        row.get("request_throughput"),
    )
    return row


def discover_mode_dirs(shape_dir: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for path in sorted(p for p in shape_dir.iterdir() if p.is_dir()):
        if "-flashinfer-" in path.name:
            dirs["flashinfer"] = path
        elif "-torch-" in path.name:
            dirs["torch"] = path
    return dirs


def build_summary(
    artifact_dir: Path,
    *,
    l20_hourly_usd: float,
    price_source: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for shape_dir in sorted(p for p in artifact_dir.iterdir() if p.is_dir()):
        if not SHAPE_DIR_RE.match(shape_dir.name):
            continue
        prompt_tokens, output_tokens = shape_tokens(shape_dir)
        for mode, run_dir in discover_mode_dirs(shape_dir).items():
            groups = reports_for_shape(run_dir)
            for shape, reports in sorted(groups.items()):
                match = re.match(r"c(?P<concurrency>\d+)-i(?P<input>\d+)", shape)
                if not match:
                    continue
                row = summarize_reports(
                    reports,
                    hourly_usd=l20_hourly_usd,
                    output_tokens=output_tokens,
                )
                row.update(
                    {
                        "mode": mode,
                        "shape_group": shape_dir.name,
                        "shape": shape,
                        "prompt_tokens": prompt_tokens,
                        "output_tokens": output_tokens,
                        "input_tokens": int(match.group("input")),
                        "concurrency": int(match.group("concurrency")),
                        "source_dir": str(run_dir),
                    }
                )
                rows.append(row)

    flashinfer_rows = [row for row in rows if row["mode"] == "flashinfer"]
    best_by_shape = []
    for shape_group in sorted({row["shape_group"] for row in flashinfer_rows}):
        candidates = [row for row in flashinfer_rows if row["shape_group"] == shape_group]
        if candidates:
            best_by_shape.append(
                max(candidates, key=lambda row: float(row.get("request_throughput") or 0.0))
            )

    return {
        "schema_version": 1,
        "mode": "l20_cost_tail_from_vllm_serving_artifacts",
        "artifact_dir": str(artifact_dir),
        "l20_hourly_usd": l20_hourly_usd,
        "price_source": price_source,
        "claim_boundary": [
            "Cost uses a configurable L20 hourly rental rate and excludes CPU host, storage, network, idle time, and provider discounts.",
            "Tail rows are medians of per-run vLLM benchmark percentiles because the compact checked-in artifacts do not store every request latency.",
            "Use the JSON rows for audit; the Markdown table highlights the FlashInfer serving path and torch/native comparator.",
        ],
        "rows": sorted(
            rows,
            key=lambda row: (
                row["shape_group"],
                row["mode"],
                row["concurrency"],
            ),
        ),
        "best_flashinfer_by_shape": best_by_shape,
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# L20 Cost and Tail Latency",
        "",
        "This derived artifact adds cost-per-1M-token and p95/p99 tail columns to",
        "the same-model Qwen2.5-Coder-0.5B CPU-vs-L20 serving evidence.",
        "",
        f"- L20 hourly rate used: `${summary['l20_hourly_usd']:.3f}/h`",
        f"- Price source: {summary['price_source']}",
        "- Cost formula: `hourly_usd / (throughput_per_s * 3600) * 1e6`.",
        "- Tail values are medians of per-run vLLM benchmark percentiles.",
        "",
        "## Best FlashInfer Rows",
        "",
        "| Shape | Concurrency | Req/s | Output tok/s | Total tok/s | $/1M output tok | $/1M total tok | p95 TTFT | p99 TTFT | p95 ITL | p99 ITL | p95 E2E | p99 E2E |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["best_flashinfer_by_shape"]:
        lines.append(
            f"| `{row['shape_group']}` | {row['concurrency']} | "
            f"{fmt(row.get('request_throughput'))} | "
            f"{fmt(row.get('output_throughput'))} | "
            f"{fmt(row.get('total_token_throughput'))} | "
            f"{fmt(row.get('cost_per_1m_output_tokens_usd'), 4)} | "
            f"{fmt(row.get('cost_per_1m_total_tokens_usd'), 4)} | "
            f"{fmt(row.get('p95_ttft_ms'))} ms | "
            f"{fmt(row.get('p99_ttft_ms'))} ms | "
            f"{fmt(row.get('p95_itl_ms'))} ms | "
            f"{fmt(row.get('p99_itl_ms'))} ms | "
            f"{fmt(row.get('p95_e2el_ms'))} ms | "
            f"{fmt(row.get('p99_e2el_ms'))} ms |"
        )

    lines.extend(
        [
            "",
            "## Full L20 Tail Table",
            "",
            "| Mode | Shape | Concurrency | Runs | Req/s | Output tok/s | $/1M output tok | Median TTFT | p95 TTFT | p99 TTFT | Median ITL | p95 ITL | p99 ITL | Median E2E | p95 E2E | p99 E2E |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| `{row['mode']}` | `{row['shape_group']}/{row['shape']}` | "
            f"{row['concurrency']} | {row['runs']} | "
            f"{fmt(row.get('request_throughput'))} | "
            f"{fmt(row.get('output_throughput'))} | "
            f"{fmt(row.get('cost_per_1m_output_tokens_usd'), 4)} | "
            f"{fmt(row.get('median_ttft_ms'))} ms | "
            f"{fmt(row.get('p95_ttft_ms'))} ms | "
            f"{fmt(row.get('p99_ttft_ms'))} ms | "
            f"{fmt(row.get('median_itl_ms'))} ms | "
            f"{fmt(row.get('p95_itl_ms'))} ms | "
            f"{fmt(row.get('p99_itl_ms'))} ms | "
            f"{fmt(row.get('median_e2el_ms'))} ms | "
            f"{fmt(row.get('p95_e2el_ms'))} ms | "
            f"{fmt(row.get('p99_e2el_ms'))} ms |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    output_json = args.output_json or (args.artifact_dir / "cost-tail-summary.json")
    output_md = args.output_md or (args.artifact_dir / "cost-tail.md")
    summary = build_summary(
        args.artifact_dir,
        l20_hourly_usd=args.l20_hourly_usd,
        price_source=args.price_source,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
