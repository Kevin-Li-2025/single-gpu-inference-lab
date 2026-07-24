#!/usr/bin/env python3
"""Summarize paired vLLM sparse-sampler serving A/B artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


METRICS = ("itl_ms", "ms_per_output_token", "total_ms", "ttft_ms")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_case_summary(root: Path, mode: str) -> dict[str, Any]:
    payload = load_json(root / mode / "probe" / "sampling_semantics_summary.json")
    cases = payload.get("cases") or []
    if not cases:
        raise RuntimeError(f"no cases in {mode} summary")
    if len(cases) != 1:
        raise RuntimeError(f"expected one selected case in {mode}, got {len(cases)}")
    return cases[0]


def median_metric(case: dict[str, Any], metric: str) -> float:
    values = case.get(metric) or {}
    if "median" not in values:
        raise RuntimeError(f"missing median for {metric}")
    return float(values["median"])


def pct_delta(candidate: float, baseline: float) -> float:
    return 0.0 if baseline == 0 else 100.0 * (candidate - baseline) / baseline


def summarize_trace(trace: Path) -> dict[str, Any] | None:
    if not trace.exists():
        return None
    total = 0
    eligible = 0
    shapes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        total += 1
        if event.get("eligible"):
            eligible += 1
        metadata = event.get("metadata") or {}
        shape = metadata.get("logits_shape")
        if isinstance(shape, list) and len(shape) == 2:
            shapes[f"{shape[0]}x{shape[1]}"] += 1
        for reason in event.get("reasons") or []:
            reasons[str(reason)] += 1
    return {
        "schema_version": 1,
        "trace": str(trace),
        "total_events": total,
        "eligible_events": eligible,
        "fallback_events": total - eligible,
        "eligible_fraction": eligible / total if total else 0.0,
        "logits_shape_counts": dict(sorted(shapes.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def collect_raw_values(root: Path, mode: str, metric: str) -> list[float]:
    raw_path = root / mode / "probe" / "sampling_semantics_raw.jsonl"
    values = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("warmup") or row.get("status") != "ok":
            continue
        values.append(float(row[metric]))
    return values


def build_summary(root: Path) -> dict[str, Any]:
    config_path = root / "run-config.json"
    baseline = load_case_summary(root, "baseline-flashinfer")
    candidate = load_case_summary(root, "candidate-sparse")
    deltas = {}
    for metric in METRICS:
        base = median_metric(baseline, metric)
        cand = median_metric(candidate, metric)
        deltas[metric] = {
            "baseline_median": base,
            "candidate_median": cand,
            "delta_percent": pct_delta(cand, base),
            "speedup": base / cand if cand else 0.0,
        }
    trace_summary = summarize_trace(root / "candidate-trace" / "l20-topk-topp-trace.jsonl")
    prewarm_path = root / "baseline-flashinfer" / "flashinfer-prewarm.json"
    if not prewarm_path.exists():
        prewarm_path = root / "flashinfer-prewarm.json"
    baseline_path = root / "baseline-flashinfer" / "sampling-path.json"
    candidate_path = root / "candidate-sparse" / "sampling-path.json"
    result = {
        "schema_version": 1,
        "artifact": root.name,
        "evidence_status": "requires_semantic_validation",
        "performance_comparable": False,
        "config": load_json(config_path) if config_path.exists() else {},
        "case": {
            "name": baseline.get("case"),
            "description": baseline.get("description"),
            "sampling": baseline.get("sampling"),
        },
        "baseline": {
            "mode": "vllm_flashinfer_topk_topp_penalty",
            "ok_runs": baseline.get("ok_runs"),
            "summary": {metric: baseline.get(metric) for metric in METRICS},
            "path_match_counts": (
                load_json(baseline_path).get("match_counts")
                if baseline_path.exists()
                else {}
            ),
        },
        "candidate": {
            "mode": "opt_in_sparse_token_history_sampler_no_trace",
            "ok_runs": candidate.get("ok_runs"),
            "summary": {metric: candidate.get(metric) for metric in METRICS},
            "path_match_counts": (
                load_json(candidate_path).get("match_counts")
                if candidate_path.exists()
                else {}
            ),
        },
        "historical_delta": deltas,
        "trace_proof": trace_summary,
        "flashinfer_prewarm": load_json(prewarm_path) if prewarm_path.exists() else {},
        "claim_boundary": [
            "These deltas are not current performance evidence.",
            "The custom sampler must pass the corrected top-p semantic revalidation gate before comparison.",
            "This was collected through a real vLLM HTTP path, not a standalone microbenchmark.",
            "The baseline uses vLLM's FlashInfer top-k/top-p sampler path.",
            "The no-trace candidate is compared against the FlashInfer-enabled baseline.",
            "The separate trace run proves custom hook coverage but is not used for latency.",
        ],
        "stability": {
            metric: {
                "baseline_measured_median": statistics.median(
                    collect_raw_values(root, "baseline-flashinfer", metric)
                ),
                "candidate_measured_median": statistics.median(
                    collect_raw_values(root, "candidate-sparse", metric)
                ),
            }
            for metric in METRICS
        },
    }
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    config = summary.get("config") or {}
    delta = summary["historical_delta"]
    trace = summary.get("trace_proof") or {}
    prewarm = summary.get("flashinfer_prewarm") or {}
    case = summary.get("case") or {}
    lines = [
        f"# {summary['artifact']}",
        "",
        "> **Provisional semantics:** the deltas below are historical and are",
        "> not current performance evidence until the corrected top-p sampler",
        "> passes native-equivalent semantic revalidation.",
        "",
        "This artifact compares vLLM's FlashInfer top-k/top-p sampler with the",
        "opt-in sparse token-history penalty sampler on a real OpenAI-compatible",
        "vLLM serving path.",
        "",
        "## Setup",
        "",
        f"- GPU: `{config.get('gpu', 'unknown')}`",
        f"- Model: `{config.get('model', 'unknown')}`",
        f"- vLLM: `{config.get('vllm_version', 'unknown')}`",
        f"- Torch: `{config.get('torch_version', 'unknown')}`",
        f"- FlashInfer: `{prewarm.get('flashinfer_version', 'unknown')}`",
        f"- Output length: {config.get('max_tokens', 'unknown')} tokens",
        f"- Probe: {config.get('warmup', 'unknown')} warmup, "
        f"{config.get('runs', 'unknown')} measured requests",
        f"- Case: `{case.get('name', config.get('probe_case', 'unknown'))}`",
        "",
        "## Historical result (not current evidence)",
        "",
        "| Metric | FlashInfer median | Sparse sampler median | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, label in [
        ("itl_ms", "ITL"),
        ("ms_per_output_token", "ms/output token"),
        ("total_ms", "Total request time"),
        ("ttft_ms", "TTFT"),
    ]:
        row = delta[metric]
        lines.append(
            f"| {label} | {row['baseline_median']:.3f} ms | "
            f"{row['candidate_median']:.3f} ms | "
            f"{row['delta_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Path Proof",
            "",
            "| Trace metric | Value |",
            "| --- | ---: |",
            f"| Total sampler events | {trace.get('total_events', 0)} |",
            f"| Eligible custom events | {trace.get('eligible_events', 0)} |",
            f"| Eligible fraction | {100.0 * trace.get('eligible_fraction', 0.0):.2f}% |",
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
