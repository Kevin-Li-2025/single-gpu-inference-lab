#!/usr/bin/env python3
"""Summarize a sparse-penalty triangle serving matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_delta(row: dict[str, Any], variant: str, metric: str) -> dict[str, Any]:
    for item in row["delta_vs_baseline"][variant]:
        if item["metric"] == metric:
            return item
    raise KeyError(f"missing {variant} metric {metric}")


def build_row(path: Path, root: Path) -> dict[str, Any]:
    summary = load_json(path / "summary.json")
    workload = summary["workloads"]["baseline"]
    fused_trace = summary["trace_proof"]["fused"]
    standalone_trace = summary["trace_proof"]["standalone"]
    return {
        "artifact": path.name,
        "path": str(path.relative_to(root)),
        "comparable_workload": bool(summary["comparable_workload"]),
        "input_tokens": workload["input_tokens_requested"],
        "output_tokens": workload["output_tokens_requested"],
        "num_prompts": workload["num_prompts"],
        "max_concurrency": workload["max_concurrency"],
        "standalone_itl": metric_delta(summary, "standalone", "median_itl_ms"),
        "fused_itl": metric_delta(summary, "fused", "median_itl_ms"),
        "standalone_e2e": metric_delta(summary, "standalone", "median_e2el_ms"),
        "fused_e2e": metric_delta(summary, "fused", "median_e2el_ms"),
        "standalone_output_throughput": metric_delta(
            summary, "standalone", "output_throughput"
        ),
        "fused_output_throughput": metric_delta(summary, "fused", "output_throughput"),
        "fused_trace": {
            "eligible_events": fused_trace.get("eligible_events", 0),
            "total_events": fused_trace.get("total_events", 0),
            "eligible_fraction": fused_trace.get("eligible_fraction", 0.0),
        },
        "standalone_trace": standalone_trace,
    }


def build_summary(root: Path) -> dict[str, Any]:
    rows = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "summary.json").exists():
            rows.append(build_row(child, root))
    comparable = [row for row in rows if row["comparable_workload"]]
    return {
        "schema_version": 1,
        "artifact": root.name,
        "row_count": len(rows),
        "comparable_row_count": len(comparable),
        "fused_itl_positive_rows": sum(
            1 for row in comparable if row["fused_itl"]["improvement_pct"] > 0
        ),
        "standalone_itl_positive_rows": sum(
            1 for row in comparable if row["standalone_itl"]["improvement_pct"] > 0
        ),
        "fused_e2e_positive_rows": sum(
            1 for row in comparable if row["fused_e2e"]["improvement_pct"] > 0
        ),
        "rows": rows,
        "claim_boundary": [
            "This is a serving matrix, but each row is still scoped to its model and traffic shape.",
            "Latency rows are no-trace runs; trace sub-runs are path proof only.",
            "Positive rows are evidence for this fused sampler boundary, not a general vLLM claim.",
            "Standalone logits-processor rows remain useful as the architecture-control baseline.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# {summary['artifact']}",
        "",
        "This artifact summarizes a native-vs-standalone-vs-fused repetition-penalty",
        "serving matrix on the L20 vLLM path.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary['row_count']}`",
        f"- Comparable rows: `{summary['comparable_row_count']}`",
        f"- Fused median ITL positives: `{summary['fused_itl_positive_rows']}`",
        f"- Standalone median ITL positives: `{summary['standalone_itl_positive_rows']}`",
        f"- Fused median E2E positives: `{summary['fused_e2e_positive_rows']}`",
        "",
        "## Rows",
        "",
        "| Row | c | input | output | prompts | Standalone ITL | Fused ITL | Fused E2E | Fused trace |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["rows"]:
        trace = row["fused_trace"]
        lines.append(
            f"| `{row['artifact']}` | {row['max_concurrency']} | {row['input_tokens']} | "
            f"{row['output_tokens']} | {row['num_prompts']} | "
            f"{row['standalone_itl']['improvement_pct']:+.3f}% | "
            f"{row['fused_itl']['improvement_pct']:+.3f}% | "
            f"{row['fused_e2e']['improvement_pct']:+.3f}% | "
            f"{trace['eligible_events']}/{trace['total_events']} |"
        )
    lines.extend(["", "## Claim Boundary", ""])
    lines.extend(f"- {item}" for item in summary["claim_boundary"])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = build_summary(args.root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        args.output_md.write_text(render_markdown(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
