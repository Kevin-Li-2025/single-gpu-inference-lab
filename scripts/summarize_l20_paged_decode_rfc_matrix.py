#!/usr/bin/env python3
"""Summarize L20 paged decode RFC serving matrix outputs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("matrix_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_reports(root: Path) -> dict[tuple[int, int], list[dict]]:
    pattern = re.compile(r"c(\d+)-i(\d+)-r(\d+)\.json")
    groups: dict[tuple[int, int], list[dict]] = {}
    for path in sorted(root.glob("*.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        groups.setdefault((int(match.group(1)), int(match.group(2))), []).append(
            json.loads(path.read_text(encoding="utf-8"))
        )
    return groups


def load_config(root: Path) -> dict:
    path = root / "run-config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def log_flags(root: Path) -> dict:
    log_path = root / "server.log"
    trace_path = root / "l20-paged-decode-trace.txt"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    trace_text = (
        trace_path.read_text(encoding="utf-8", errors="replace")
        if trace_path.exists()
        else ""
    )
    return {
        "server_log_exists": log_path.exists(),
        "flashinfer_backend": "AttentionBackendEnum.FLASHINFER" in log_text,
        "cudagraph_disabled": "Cudagraph is disabled" in log_text,
        "cudagraph_mentioned": "Cudagraph" in log_text or "cudagraph" in log_text,
        "trace_hit_count": sum(1 for line in trace_text.splitlines() if "hit " in line),
    }


def summarize_pair(baseline_dir: Path, l20_dir: Path) -> dict:
    baseline = load_reports(baseline_dir)
    l20 = load_reports(l20_dir)
    shapes = []
    if baseline.keys() != l20.keys():
        return {
            "baseline_dir": str(baseline_dir),
            "l20_dir": str(l20_dir),
            "error": "shape_mismatch",
            "baseline_shapes": sorted(map(list, baseline.keys())),
            "l20_shapes": sorted(map(list, l20.keys())),
        }
    for shape in sorted(baseline):
        metrics = {}
        for name in METRICS:
            base_value = statistics.median(row[name] for row in baseline[shape])
            l20_value = statistics.median(row[name] for row in l20[shape])
            metrics[name] = {
                "baseline": round(base_value, 6),
                "l20": round(l20_value, 6),
                "change_pct": round((l20_value / base_value - 1.0) * 100.0, 3),
            }
        shapes.append(
            {
                "max_concurrency": shape[0],
                "input_tokens": shape[1],
                "runs_per_variant": len(baseline[shape]),
                "metrics": metrics,
            }
        )
    config = load_config(l20_dir) or load_config(baseline_dir)
    return {
        "execution_mode": config.get("execution_mode"),
        "model": config.get("model"),
        "served_name": config.get("served_name"),
        "baseline_dir": str(baseline_dir),
        "l20_dir": str(l20_dir),
        "baseline_flags": log_flags(baseline_dir),
        "l20_flags": log_flags(l20_dir),
        "shapes": shapes,
    }


def main() -> int:
    args = parse_args()
    matrix_dir = args.matrix_dir
    summaries = []
    modes = sorted(
        {
            path.name.rsplit("-", 1)[0]
            for path in matrix_dir.iterdir()
            if path.is_dir() and path.name.endswith(("-baseline", "-l20"))
        }
    )
    for mode in modes:
        baseline_dir = matrix_dir / f"{mode}-baseline"
        l20_dir = matrix_dir / f"{mode}-l20"
        if baseline_dir.exists() and l20_dir.exists():
            summaries.append(summarize_pair(baseline_dir, l20_dir))
    result = {
        "schema_version": 1,
        "matrix_dir": str(matrix_dir),
        "summaries": summaries,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

