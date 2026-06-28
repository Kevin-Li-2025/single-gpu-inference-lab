#!/usr/bin/env python3
"""Summarize paired vLLM stochastic sampling winner runs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


RUN_RE = re.compile(
    r"^(?P<model>.+)-(?P<mode>torch|flashinfer)-(?P<suffix>c.+)$"
)
REPORT_RE = re.compile(r"^c(?P<concurrency>\d+)-i(?P<input>\d+)-r(?P<run>\d+)\.json$")

METRICS = (
    "median_itl_ms",
    "mean_itl_ms",
    "output_throughput",
    "median_ttft_ms",
    "p99_itl_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_reports(run_dir: Path) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for path in sorted(run_dir.glob("c*-i*-r*.json")):
        match = REPORT_RE.match(path.name)
        if not match:
            continue
        shape = f"c{match.group('concurrency')}-i{match.group('input')}"
        report = json.loads(path.read_text(encoding="utf-8"))
        groups.setdefault(shape, []).append(report)
    return groups


def summarize_reports(rows: list[dict]) -> dict:
    if not rows:
        return {"runs": 0}
    result = {"runs": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if metric in row]
        if values:
            result[metric] = statistics.median(values)
    return result


def pct_delta(candidate: float, baseline: float) -> float:
    if baseline == 0:
        return 0.0
    return 100.0 * (candidate - baseline) / baseline


def compare_pair(torch_summary: dict, flashinfer_summary: dict) -> dict:
    deltas = {}
    for metric in METRICS:
        if metric in torch_summary and metric in flashinfer_summary:
            deltas[f"{metric}_pct"] = pct_delta(
                flashinfer_summary[metric],
                torch_summary[metric],
            )
    median_itl_delta = deltas.get("median_itl_ms_pct", 0.0)
    throughput_delta = deltas.get("output_throughput_pct", 0.0)
    return {
        "torch": torch_summary,
        "flashinfer": flashinfer_summary,
        "deltas": deltas,
        "median_itl_win": median_itl_delta < 0.0,
        "throughput_win": throughput_delta > 0.0,
        "strict_win": median_itl_delta < 0.0 and throughput_delta > 0.0,
    }


def discover_runs(input_dir: Path) -> dict[str, dict[str, Path]]:
    runs: dict[str, dict[str, Path]] = {}
    for run_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        key = f"{match.group('model')}-{match.group('suffix')}"
        runs.setdefault(key, {})[match.group("mode")] = run_dir
    return runs


def build_summary(input_dir: Path) -> dict:
    pairs = []
    strict_wins = 0
    total_pairs = 0
    for key, modes in discover_runs(input_dir).items():
        if "torch" not in modes or "flashinfer" not in modes:
            continue
        torch_groups = load_reports(modes["torch"])
        flashinfer_groups = load_reports(modes["flashinfer"])
        shapes = sorted(set(torch_groups) & set(flashinfer_groups))
        shape_results = {}
        for shape in shapes:
            total_pairs += 1
            result = compare_pair(
                summarize_reports(torch_groups[shape]),
                summarize_reports(flashinfer_groups[shape]),
            )
            strict_wins += int(result["strict_win"])
            shape_results[shape] = result
        pairs.append(
            {
                "key": key,
                "torch_dir": str(modes["torch"]),
                "flashinfer_dir": str(modes["flashinfer"]),
                "shapes": shape_results,
            }
        )
    return {
        "schema_version": 1,
        "input_dir": str(input_dir),
        "winner_gate": {
            "definition": "FlashInfer must reduce median ITL and increase output throughput versus the paired torch/native sampler.",
            "strict_win_pairs": strict_wins,
            "total_pairs": total_pairs,
            "strict_win_fraction": strict_wins / total_pairs if total_pairs else 0.0,
        },
        "pairs": pairs,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# L20 vLLM Sampling Winner Summary",
        "",
        "This report compares paired stochastic serving runs on one NVIDIA L20.",
        "A strict win requires both lower median ITL and higher output throughput.",
        "",
        "## Gate",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            f"| Strict wins | {summary['winner_gate']['strict_win_pairs']} / "
            f"{summary['winner_gate']['total_pairs']} |"
        ),
        f"| Strict win fraction | {summary['winner_gate']['strict_win_fraction']:.2%} |",
        "",
        "## Results",
        "",
        "| Model/run | Shape | Torch ITL | FlashInfer ITL | ITL delta | "
        "Torch tok/s | FlashInfer tok/s | Throughput delta | Strict win |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in summary["pairs"]:
        for shape, result in pair["shapes"].items():
            torch = result["torch"]
            flashinfer = result["flashinfer"]
            deltas = result["deltas"]
            lines.append(
                f"| `{pair['key']}` | `{shape}` | "
                f"{torch.get('median_itl_ms', 0.0):.3f} | "
                f"{flashinfer.get('median_itl_ms', 0.0):.3f} | "
                f"{deltas.get('median_itl_ms_pct', 0.0):+.2f}% | "
                f"{torch.get('output_throughput', 0.0):.1f} | "
                f"{flashinfer.get('output_throughput', 0.0):.1f} | "
                f"{deltas.get('output_throughput_pct', 0.0):+.2f}% | "
                f"{'yes' if result['strict_win'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The measured positive path is vLLM's FlashInfer top-k/top-p sampler",
            "with CUDA 13 JIT prewarm and explicit fallback checks. The custom",
            "standalone L20 sampler remains disabled because it regresses serving",
            "ITL despite kernel-level wins.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = build_summary(args.input_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
