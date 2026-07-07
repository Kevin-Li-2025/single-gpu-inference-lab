#!/usr/bin/env python3
"""Summarize baseline vs standalone logits processor vs fused sampler serving runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


METRICS = (
    "request_throughput",
    "output_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_itl_ms",
    "p95_itl_ms",
    "median_e2el_ms",
)
HIGHER_IS_BETTER = {"request_throughput", "output_throughput"}
WORKLOAD_KEYS = (
    "model",
    "input_tokens_requested",
    "output_tokens_requested",
    "num_prompts",
    "max_concurrency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_variant(root: Path, variant: str) -> dict[str, Any]:
    path = root / variant / f"{variant}_summary.json"
    if path.exists():
        return load_json(path)
    if variant == "standalone":
        legacy = root / "candidate" / "candidate_summary.json"
        if legacy.exists():
            payload = load_json(legacy)
            payload["variant"] = "standalone"
            return payload
    raise FileNotFoundError(path)


def summarize_standalone_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"trace_exists": False}
    provider_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    max_unique = 0
    events = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        events += 1
        provider_counts[str(event.get("provider", "unknown"))] += 1
        reason_counts[str(event.get("reason", "unknown"))] += 1
        max_unique = max(max_unique, int(event.get("max_unique_tokens") or 0))
    return {
        "trace_exists": True,
        "event_count": events,
        "provider_counts": dict(sorted(provider_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "max_unique_tokens_seen": max_unique,
    }


def summarize_fused_trace(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"trace_exists": False}
    reasons: Counter[str] = Counter()
    shapes: Counter[str] = Counter()
    total = 0
    eligible = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        total += 1
        if event.get("eligible"):
            eligible += 1
        for reason in event.get("reasons") or []:
            reasons[str(reason)] += 1
        metadata = event.get("metadata") or {}
        shape = metadata.get("logits_shape")
        if isinstance(shape, list) and len(shape) == 2:
            shapes[f"{shape[0]}x{shape[1]}"] += 1
    return {
        "trace_exists": True,
        "total_events": total,
        "eligible_events": eligible,
        "fallback_events": total - eligible,
        "eligible_fraction": eligible / total if total else 0.0,
        "logits_shape_counts": dict(sorted(shapes.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def workload_signature(summary: dict[str, Any]) -> dict[str, Any]:
    return {key: summary.get(key) for key in WORKLOAD_KEYS}


def compare_metric(metric: str, baseline: float, candidate: float) -> dict[str, Any]:
    if metric in HIGHER_IS_BETTER:
        improvement = ((candidate / baseline) - 1.0) * 100.0 if baseline else 0.0
        speedup = candidate / baseline if baseline else 0.0
    else:
        improvement = ((baseline / candidate) - 1.0) * 100.0 if candidate else 0.0
        speedup = baseline / candidate if candidate else 0.0
    return {
        "metric": metric,
        "baseline": round(baseline, 6),
        "candidate": round(candidate, 6),
        "improvement_pct": round(improvement, 3),
        "speedup": round(speedup, 6),
        "higher_is_better": metric in HIGHER_IS_BETTER,
    }


def compare_against_baseline(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for metric in METRICS:
        rows.append(
            compare_metric(
                metric,
                float(baseline.get(metric, 0.0)),
                float(candidate.get(metric, 0.0)),
            )
        )
    return rows


def build_summary(root: Path) -> dict[str, Any]:
    baseline = load_variant(root, "baseline")
    standalone = load_variant(root, "standalone")
    fused = load_variant(root, "fused")
    workloads = {
        "baseline": workload_signature(baseline),
        "standalone": workload_signature(standalone),
        "fused": workload_signature(fused),
    }
    comparable = len({json.dumps(value, sort_keys=True) for value in workloads.values()}) == 1
    standalone_trace_path = root / "standalone-trace" / "sparse-rp-trace.jsonl"
    fused_trace_path = root / "fused-trace" / "l20-topk-topp-trace.jsonl"
    return {
        "schema_version": 1,
        "artifact": root.name,
        "boundary": (
            "three-way serving comparison: native vLLM penalty path vs request-level "
            "standalone logits processor vs fused sampler-boundary sparse penalty"
        ),
        "comparable_workload": comparable,
        "workloads": workloads,
        "variants": {
            "baseline": baseline,
            "standalone": standalone,
            "fused": fused,
        },
        "delta_vs_baseline": {
            "standalone": compare_against_baseline(baseline, standalone),
            "fused": compare_against_baseline(baseline, fused),
        },
        "trace_proof": {
            "standalone": summarize_standalone_trace(standalone_trace_path),
            "fused": summarize_fused_trace(fused_trace_path),
        },
        "claim_boundary": [
            "Only compare rows when comparable_workload is true.",
            "Latency variants should run without trace enabled; trace variants are path proof only.",
            "This comparison isolates repetition penalty so standalone and fused routes share one semantic target.",
            "Treat small deltas as directional until request count and traffic shape are expanded.",
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    standalone_trace = summary["trace_proof"]["standalone"]
    fused_trace = summary["trace_proof"]["fused"]
    lines = [
        f"# {summary['artifact']}",
        "",
        "This artifact compares three real vLLM HTTP serving paths for repetition penalty:",
        "native vLLM baseline, request-level standalone logits processor, and fused sampler boundary.",
        "",
        f"- Comparable workload: `{summary['comparable_workload']}`",
        "",
        "## Delta vs Baseline",
        "",
        "| Variant | Metric | Baseline | Candidate | Improvement | Speedup |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("standalone", "fused"):
        for row in summary["delta_vs_baseline"][variant]:
            lines.append(
                f"| `{variant}` | `{row['metric']}` | {row['baseline']} | "
                f"{row['candidate']} | {row['improvement_pct']}% | {row['speedup']}x |"
            )
    trace = summary["trace_proof"]
    lines.extend(
        [
            "",
            "## Trace Proof",
            "",
            f"- Standalone events: `{standalone_trace.get('event_count', 0)}`",
            f"- Standalone provider counts: `{standalone_trace.get('provider_counts', {})}`",
            f"- Standalone max unique tokens: `{standalone_trace.get('max_unique_tokens_seen', 0)}`",
            f"- Fused events: `{fused_trace.get('total_events', 0)}`",
            f"- Fused eligible events: `{fused_trace.get('eligible_events', 0)}`",
            f"- Fused eligible fraction: `{100.0 * fused_trace.get('eligible_fraction', 0.0):.2f}%`",
            "",
            "## Claim Boundary",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in summary["claim_boundary"])
    lines.append("")
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
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
