#!/usr/bin/env python3
"""Summarize paired sparse repetition-penalty serving reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRICS = (
    "request_throughput",
    "output_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "median_e2el_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_rows(baseline: dict, candidate: dict) -> list[dict]:
    rows = []
    for metric in METRICS:
        base = float(baseline.get(metric, 0.0))
        cand = float(candidate.get(metric, 0.0))
        change = ((cand / base) - 1.0) * 100.0 if base else 0.0
        rows.append(
            {
                "metric": metric,
                "baseline": round(base, 6),
                "candidate": round(cand, 6),
                "change_pct": round(change, 3),
            }
        )
    return rows


def render_markdown(summary: dict) -> str:
    smoke_note = (
        "This is a runner smoke, not a serving-speed claim: the candidate did "
        "not hit the sparse CUDA op."
        if summary["candidate_sparse_op_hits"] == 0
        else "The candidate trace hit the sparse CUDA op; inspect request shape before claiming speed."
    )
    lines = [
        "# L20 Sparse Repetition-Penalty Serving A/B",
        "",
        smoke_note,
        "",
        "This summary is valid only when both variants report zero failed requests "
        "and candidate trace coverage matches the intended gate.",
        "",
        "| Metric | Baseline | Candidate | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in summary["metrics"]:
        lines.append(
            f"| `{row['metric']}` | {row['baseline']} | "
            f"{row['candidate']} | {row['change_pct']}% |"
        )
    trace = summary["candidate_trace"]
    lines.extend(
        [
            "",
            "## Candidate Trace",
            "",
            f"- Trace exists: `{trace.get('trace_exists')}`",
            f"- Event count: `{trace.get('event_count', 0)}`",
            f"- Provider counts: `{trace.get('provider_counts', {})}`",
            f"- Reason counts: `{trace.get('reason_counts', {})}`",
            f"- Max unique tokens seen: `{trace.get('max_unique_tokens_seen', 0)}`",
            f"- Sparse op hits: `{summary['candidate_sparse_op_hits']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    candidate_trace = candidate.get("trace", {})
    candidate_provider_counts = candidate_trace.get("provider_counts", {})
    sparse_hits = int(candidate_provider_counts.get("sparse_op", 0))
    summary = {
        "schema_version": 1,
        "boundary": (
            "paired serving summary; not a speed claim unless request failures "
            "are zero and candidate_sparse_op_hits is positive"
        ),
        "smoke_only": sparse_hits == 0,
        "candidate_sparse_op_hits": sparse_hits,
        "baseline_failed": int(baseline.get("failed", 0)),
        "candidate_failed": int(candidate.get("failed", 0)),
        "metrics": metric_rows(baseline, candidate),
        "baseline_trace": baseline.get("trace", {}),
        "candidate_trace": candidate_trace,
    }
    serialized = json.dumps(summary, indent=2, sort_keys=True)
    print(serialized)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
