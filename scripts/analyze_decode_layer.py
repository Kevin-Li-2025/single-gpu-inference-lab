#!/usr/bin/env python3
"""Aggregate repeated L20 decode-layer benchmark reports."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def median(rows, provider, metric="p50"):
    return statistics.median(row["timing_ms"][provider][metric] for row in rows)


def main() -> int:
    args = parse_args()
    groups = defaultdict(list)
    for path in sorted(args.reports.glob("*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        shape = report["shape"]
        groups[(shape["batch_size"], shape["context_length"])].append(report)
    if not groups:
        raise SystemExit("no JSON reports found")

    shapes = []
    for (batch_size, context_length), rows in sorted(groups.items()):
        separate_append = median(rows, "separate_append")
        fused_append = median(rows, "l20_fused_append")
        separate_layer = median(rows, "separate_layer")
        fused_layer = median(rows, "l20_fused_layer")
        shapes.append(
            {
                "batch_size": batch_size,
                "context_length": context_length,
                "runs": len(rows),
                "all_correct": all(
                    row["correctness"]["cache_equal"]
                    and row["correctness"]["output_close"]
                    for row in rows
                ),
                "median_p50_ms": {
                    "separate_append": round(separate_append, 5),
                    "l20_fused_append": round(fused_append, 5),
                    "attention_only": round(median(rows, "attention_only"), 5),
                    "separate_layer": round(separate_layer, 5),
                    "l20_fused_layer": round(fused_layer, 5),
                },
                "append_speedup": round(separate_append / fused_append, 4),
                "layer_speedup": round(separate_layer / fused_layer, 4),
                "layer_latency_reduction_pct": round(
                    (separate_layer - fused_layer) / separate_layer * 100, 3
                ),
            }
        )
    result = {
        "schema_version": 1,
        "aggregation": "median of per-run p50 CUDA Event timings",
        "shapes": shapes,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if all(shape["all_correct"] for shape in shapes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
