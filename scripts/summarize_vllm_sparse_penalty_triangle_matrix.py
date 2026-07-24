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
    for item in row["historical_delta_vs_baseline"][variant]:
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
        "evidence_status": summary["evidence_status"],
        "workload_signature_matches": bool(summary["workload_signature_matches"]),
        "performance_comparable": False,
        "input_tokens": workload["input_tokens_requested"],
        "output_tokens": workload["output_tokens_requested"],
        "num_prompts": workload["num_prompts"],
        "max_concurrency": workload["max_concurrency"],
        "historical_standalone_itl": metric_delta(
            summary, "standalone", "median_itl_ms"
        ),
        "historical_fused_itl": metric_delta(summary, "fused", "median_itl_ms"),
        "historical_standalone_e2e": metric_delta(
            summary, "standalone", "median_e2el_ms"
        ),
        "historical_fused_e2e": metric_delta(summary, "fused", "median_e2el_ms"),
        "historical_standalone_output_throughput": metric_delta(
            summary, "standalone", "output_throughput"
        ),
        "historical_fused_output_throughput": metric_delta(
            summary, "fused", "output_throughput"
        ),
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
    signature_matches = [row for row in rows if row["workload_signature_matches"]]
    evidence_status = (
        "superseded_semantics"
        if rows
        and all(row["evidence_status"] == "superseded_semantics" for row in rows)
        else "requires_semantic_validation"
    )
    return {
        "schema_version": 1,
        "evidence_status": evidence_status,
        "performance_comparable": False,
        "artifact": root.name,
        "row_count": len(rows),
        "workload_signature_match_row_count": len(signature_matches),
        "historical_fused_itl_positive_rows": sum(
            1
            for row in signature_matches
            if row["historical_fused_itl"]["improvement_pct"] > 0
        ),
        "historical_standalone_itl_positive_rows": sum(
            1
            for row in signature_matches
            if row["historical_standalone_itl"]["improvement_pct"] > 0
        ),
        "historical_fused_e2e_positive_rows": sum(
            1
            for row in signature_matches
            if row["historical_fused_e2e"]["improvement_pct"] > 0
        ),
        "rows": rows,
        "claim_boundary": [
            "This is a serving matrix, but each row is still scoped to its model and traffic shape.",
            "Latency rows are no-trace runs; trace sub-runs are path proof only.",
            "Positive and negative rows are not current evidence until native-equivalent semantic parity is independently verified.",
            "Standalone logits-processor rows remain useful as the architecture-control baseline.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    if summary["evidence_status"] == "superseded_semantics":
        evidence_banner = [
            "> **Superseded performance comparison:** all recorded deltas are",
            "> historical; only the trace/path structure remains current evidence.",
        ]
    else:
        evidence_banner = [
            "> **Provisional semantics:** generated latency deltas are not current",
            "> performance evidence until the sampling revalidation gate passes.",
        ]
    lines = [
        f"# {summary['artifact']}",
        "",
        *evidence_banner,
        "",
        "This artifact summarizes a native-vs-standalone-vs-fused repetition-penalty",
        "serving matrix on the L20 vLLM path.",
        "",
        "## Recorded summary",
        "",
        f"- Rows: `{summary['row_count']}`",
        (
            "- Workload-signature matches: "
            f"`{summary['workload_signature_match_row_count']}`"
        ),
        f"- Performance comparable: `{summary['performance_comparable']}`",
        (
            "- Historical fused median ITL positives: "
            f"`{summary['historical_fused_itl_positive_rows']}`"
        ),
        (
            "- Historical standalone median ITL positives: "
            f"`{summary['historical_standalone_itl_positive_rows']}`"
        ),
        (
            "- Historical fused median E2E positives: "
            f"`{summary['historical_fused_e2e_positive_rows']}`"
        ),
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
            f"{row['historical_standalone_itl']['improvement_pct']:+.3f}% | "
            f"{row['historical_fused_itl']['improvement_pct']:+.3f}% | "
            f"{row['historical_fused_e2e']['improvement_pct']:+.3f}% | "
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
