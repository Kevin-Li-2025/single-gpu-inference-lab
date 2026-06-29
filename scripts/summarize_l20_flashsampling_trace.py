#!/usr/bin/env python3
"""Summarize L20 FlashSampling epilogue shadow traces."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, help="Markdown summary output path.")
    parser.add_argument("--output-json", type=Path, help="JSON summary output path.")
    return parser.parse_args()


def read_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _plan(event: dict[str, Any]) -> dict[str, Any]:
    return event.get("metadata", {}).get("flashsampling_epilogue", {}) or {}


def _shape_key(plan: dict[str, Any]) -> str:
    request = plan.get("flashsampling_request", {}) or {}
    batch = request.get("batch_size", "?")
    hidden = request.get("hidden_size", "?")
    vocab = request.get("vocab_size", "?")
    mode = request.get("sampling_mode", "?")
    return f"b{batch}-h{hidden}-v{vocab}-{mode}"


def summarize(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    events = list(events)
    reason_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    eligible_shape_counts: Counter[str] = Counter()
    total_avoidable = 0
    total_logits = 0
    eligible = 0

    for event in events:
        plan = _plan(event)
        is_eligible = bool(plan.get("would_use_epilogue", event.get("eligible", False)))
        if is_eligible:
            eligible += 1
        for reason in plan.get("fallback_reasons", event.get("reasons", [])):
            reason_counts[str(reason)] += 1
        shape = _shape_key(plan)
        shape_counts[shape] += 1
        if is_eligible:
            eligible_shape_counts[shape] += 1
        total_avoidable += int(plan.get("avoidable_logits_materialization_bytes") or 0)
        total_logits += int(plan.get("logits_materialization_bytes") or 0)

    total = len(events)
    return {
        "schema_version": 1,
        "total_events": total,
        "eligible_events": eligible,
        "fallback_events": total - eligible,
        "eligible_fraction": (eligible / total) if total else 0.0,
        "avoidable_logits_bytes": total_avoidable,
        "avoidable_logits_mib": total_avoidable / (1024 * 1024),
        "total_logits_bytes": total_logits,
        "total_logits_mib": total_logits / (1024 * 1024),
        "reason_counts": dict(sorted(reason_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "eligible_shape_counts": dict(sorted(eligible_shape_counts.items())),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# L20 FlashSampling Trace Summary",
        "",
        f"- total events: {summary['total_events']}",
        f"- eligible events: {summary['eligible_events']}",
        f"- eligible fraction: {summary['eligible_fraction']:.2%}",
        f"- avoidable logits: {summary['avoidable_logits_mib']:.2f} MiB",
        "",
        "## Fallback Reasons",
        "",
    ]
    if summary["reason_counts"]:
        for reason, count in summary["reason_counts"].items():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    summary = summarize(read_events(args.trace))
    markdown = render_markdown(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
