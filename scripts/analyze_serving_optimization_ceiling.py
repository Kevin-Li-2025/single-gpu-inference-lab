#!/usr/bin/env python3
"""Estimate serving optimization ceilings from NSYS family summaries."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GPU_BOUNDARIES = {
    "gemm_or_gemv": ("cutlass_or_cublas_gemm", "cublas_gemv"),
    "metadata_fill": ("pytorch_fill",),
    "attention": ("flashinfer_attention",),
    "standalone_sampling": (
        "flashinfer_sampling",
        "sampler_other",
        "pytorch_softmax",
    ),
    "custom_l20_current": ("custom_l20",),
}

API_BOUNDARIES = {
    "launch_sync_transfer": ("launch", "sync", "memcpy", "graph"),
    "allocation_and_loading": ("alloc_free", "library_load", "memory_info"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-summary", action="append", type=Path, required=True)
    parser.add_argument("--lm-head-result", action="append", type=Path, default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def family_pct(summary: dict, section: str, names: tuple[str, ...]) -> float:
    families = summary[section]["families"]
    return sum(float(families.get(name, {}).get("time_pct", 0.0)) for name in names)


def family_time_ns(summary: dict, section: str, names: tuple[str, ...]) -> float:
    families = summary[section]["families"]
    return sum(float(families.get(name, {}).get("total_time_ns", 0.0)) for name in names)


def amdahl_speedup(time_pct: float, factor: float | None) -> float:
    fraction = max(0.0, min(1.0, time_pct / 100.0))
    if fraction <= 0.0:
        return 1.0
    if factor is None or math.isinf(factor):
        if fraction >= 1.0:
            return math.inf
        return 1.0 / (1.0 - fraction)
    if factor <= 0:
        raise ValueError("factor must be positive")
    return 1.0 / ((1.0 - fraction) + (fraction / factor))


def boundary_summary(summary: dict, section: str, boundaries: dict[str, tuple[str, ...]]) -> dict:
    result = {}
    for name, families in boundaries.items():
        pct = family_pct(summary, section, families)
        result[name] = {
            "families": list(families),
            "time_pct": pct,
            "total_time_ms": family_time_ns(summary, section, families) / 1e6,
            "speedup_if_2x": amdahl_speedup(pct, 2.0),
            "speedup_if_5x": amdahl_speedup(pct, 5.0),
            "speedup_if_eliminated": amdahl_speedup(pct, math.inf),
        }
    return result


def summarize_lm_head(paths: list[Path]) -> dict:
    candidates = []
    for path in paths:
        result = load_json(path)
        ratios = result.get("ratios", {})
        shape = result.get("shape", {})
        for key, value in ratios.items():
            if key.endswith("_over_full_logits_topk") or key.endswith("_over_full_logits_top1"):
                candidates.append(
                    {
                        "path": str(path),
                        "ratio_name": key,
                        "ratio": float(value),
                        "shape": shape,
                    }
                )
    best = min(candidates, key=lambda row: row["ratio"]) if candidates else None
    return {
        "candidate_count": len(candidates),
        "best_candidate": best,
        "all_candidates": sorted(candidates, key=lambda row: row["ratio"]),
    }


def run_label(path: Path, summary: dict) -> str:
    source = str(summary.get("source_dir") or "")
    if source:
        parts = Path(source).parts
        if len(parts) >= 2:
            return "/".join(parts[-3:-1]) if parts[-1] == "stats" else "/".join(parts[-2:])
    return path.parent.name


def build_recommendations(runs: list[dict], lm_head: dict) -> list[dict]:
    recommendations = []
    max_gemm = max(
        (run["gpu_boundaries"]["gemm_or_gemv"]["time_pct"] for run in runs),
        default=0.0,
    )
    max_sampling = max(
        (run["gpu_boundaries"]["standalone_sampling"]["time_pct"] for run in runs),
        default=0.0,
    )
    max_custom = max(
        (run["gpu_boundaries"]["custom_l20_current"]["time_pct"] for run in runs),
        default=0.0,
    )
    max_fill = max(
        (run["gpu_boundaries"]["metadata_fill"]["time_pct"] for run in runs),
        default=0.0,
    )
    max_host = max(
        (run["api_boundaries"]["launch_sync_transfer"]["time_pct"] for run in runs),
        default=0.0,
    )
    best_lm_ratio = None
    if lm_head.get("best_candidate"):
        best_lm_ratio = lm_head["best_candidate"]["ratio"]

    if max_gemm >= 25.0:
        recommendations.append(
            {
                "priority": "P0",
                "target": "production GEMM/GEMV epilogue or upstream logits boundary",
                "reason": (
                    f"GEMM/GEMV reaches {max_gemm:.2f}% of GPU kernel time; "
                    "this is the only measured boundary with a large compute-side ceiling."
                ),
            }
        )
    if best_lm_ratio is not None and best_lm_ratio >= 1.0:
        recommendations.append(
            {
                "priority": "P0",
                "target": "avoid standalone LM-head replacement",
                "reason": (
                    f"The best standalone LM-head/top-k candidate is {best_lm_ratio:.3f}x "
                    "of full logits, so it does not beat the optimized GEMM path."
                ),
            }
        )
    if max_host >= 30.0:
        recommendations.append(
            {
                "priority": "P1",
                "target": "CUDA graph, launch, memcpy, and synchronization reduction",
                "reason": (
                    f"Launch/sync/transfer reaches {max_host:.2f}% of CUDA API time. "
                    "Treat this as a host-side ceiling, not additive with GPU kernel time."
                ),
            }
        )
    if max_fill >= 10.0:
        recommendations.append(
            {
                "priority": "P1",
                "target": "metadata and fill/bookkeeping kernels",
                "reason": (
                    f"Fill/bookkeeping reaches {max_fill:.2f}% of GPU kernel time; "
                    "this is a real vLLM serving overhead to isolate."
                ),
            }
        )
    if max_sampling < 5.0:
        recommendations.append(
            {
                "priority": "Stop",
                "target": "standalone sampling kernels",
                "reason": (
                    f"Sampling/logits-processor kernels peak at {max_sampling:.2f}% "
                    "of GPU time, so sampler-only work has a low Amdahl ceiling."
                ),
            }
        )
    if max_custom < 5.0:
        recommendations.append(
            {
                "priority": "Stop",
                "target": "micro-optimizing the existing Q/K/RoPE/KV kernel alone",
                "reason": (
                    f"The current custom L20 kernel peaks at {max_custom:.2f}% "
                    "of GPU time; further work must remove adjacent kernels or launches."
                ),
            }
        )
    return recommendations


def analyze(family_paths: list[Path], lm_head_paths: list[Path]) -> dict:
    runs = []
    for path in family_paths:
        summary = load_json(path)
        runs.append(
            {
                "label": run_label(path, summary),
                "path": str(path),
                "source_dir": summary.get("source_dir", ""),
                "gpu_boundaries": boundary_summary(summary, "gpu", GPU_BOUNDARIES),
                "api_boundaries": boundary_summary(summary, "api", API_BOUNDARIES),
            }
        )
    lm_head = summarize_lm_head(lm_head_paths)
    return {
        "schema_version": 1,
        "runs": runs,
        "lm_head_boundary": lm_head,
        "recommendations": build_recommendations(runs, lm_head),
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# L20 Serving Optimization Ceiling",
        "",
        "This report converts NSYS family summaries into Amdahl-style ceilings. "
        "GPU-family and CUDA-API percentages are separate denominators.",
        "",
        "## GPU Boundaries",
        "",
        "| Run | Boundary | Time share | 2x speedup ceiling | Eliminate ceiling |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        for name, row in run["gpu_boundaries"].items():
            lines.append(
                f"| `{run['label']}` | `{name}` | {row['time_pct']:.2f}% | "
                f"{row['speedup_if_2x']:.3f}x | {row['speedup_if_eliminated']:.3f}x |"
            )
    lines.extend(
        [
            "",
            "## CUDA API Boundaries",
            "",
            "| Run | Boundary | Time share | 2x speedup ceiling | Eliminate ceiling |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for run in result["runs"]:
        for name, row in run["api_boundaries"].items():
            lines.append(
                f"| `{run['label']}` | `{name}` | {row['time_pct']:.2f}% | "
                f"{row['speedup_if_2x']:.3f}x | {row['speedup_if_eliminated']:.3f}x |"
            )
    lm_head = result["lm_head_boundary"]
    if lm_head.get("best_candidate"):
        best = lm_head["best_candidate"]
        lines.extend(
            [
                "",
                "## LM-Head Boundary",
                "",
                f"Best standalone candidate: `{best['ratio_name']}` = "
                f"{best['ratio']:.3f}x of full-logits baseline from `{best['path']}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Recommendations",
            "",
            "| Priority | Target | Reason |",
            "| --- | --- | --- |",
        ]
    )
    for item in result["recommendations"]:
        lines.append(
            f"| `{item['priority']}` | {item['target']} | {item['reason']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result = analyze(args.family_summary, args.lm_head_result)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(serialized + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
