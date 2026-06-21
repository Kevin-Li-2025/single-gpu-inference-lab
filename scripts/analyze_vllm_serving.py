#!/usr/bin/env python3
"""Aggregate paired vLLM serving benchmark reports."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "request_throughput",
    "output_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_itl_ms",
    "p95_itl_ms",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--fused", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_reports(root: Path):
    groups = defaultdict(list)
    pattern = re.compile(r"c(\d+)-i(\d+)-r(\d+)\.json")
    for path in sorted(root.glob("*.json")):
        match = pattern.fullmatch(path.name)
        if match:
            groups[(int(match.group(1)), int(match.group(2)))].append(
                json.loads(path.read_text(encoding="utf-8"))
            )
    return groups


def main() -> int:
    args = parse_args()
    baseline = load_reports(args.baseline)
    fused = load_reports(args.fused)
    if baseline.keys() != fused.keys() or not baseline:
        raise SystemExit("baseline and fused report shapes must match")
    shapes = []
    for batch_context in sorted(baseline):
        base_rows = baseline[batch_context]
        fused_rows = fused[batch_context]
        metrics = {}
        for name in METRICS:
            base_value = statistics.median(row[name] for row in base_rows)
            fused_value = statistics.median(row[name] for row in fused_rows)
            metrics[name] = {
                "baseline": round(base_value, 5),
                "fused": round(fused_value, 5),
                "change_pct": round((fused_value / base_value - 1) * 100, 3),
            }
        shapes.append(
            {
                "max_concurrency": batch_context[0],
                "input_tokens": batch_context[1],
                "runs_per_provider": len(base_rows),
                "metrics": metrics,
            }
        )
    result = {
        "schema_version": 1,
        "aggregation": "median across runs; prefix caching disabled",
        "shapes": shapes,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
