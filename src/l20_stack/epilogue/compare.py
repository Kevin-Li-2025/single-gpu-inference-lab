"""Build the L20 boundary-impact table used by the paper-style summary."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from l20_stack.epilogue.logits_boundary import load_logits_boundary_budget


@dataclass(frozen=True)
class BoundaryImpact:
    """One optimization boundary and its measured system impact."""

    boundary: str
    status: str
    micro_speedup_x: float | None
    serving_impact_pct: float | None
    serving_metric: str
    gpu_time_pct: float | None
    eligible_fraction_pct: float | None
    materialization_mib: float | None
    decision: str
    evidence: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary": self.boundary,
            "status": self.status,
            "micro_speedup_x": self.micro_speedup_x,
            "serving_impact_pct": self.serving_impact_pct,
            "serving_metric": self.serving_metric,
            "gpu_time_pct": self.gpu_time_pct,
            "eligible_fraction_pct": self.eligible_fraction_pct,
            "materialization_mib": self.materialization_mib,
            "decision": self.decision,
            "evidence": self.evidence,
            "note": self.note,
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _max_paged_rope_micro_speedup(root: Path) -> float | None:
    values: list[float] = []
    for path in (root / "benchmarks/results/l20-paged-rope-policy-v3").glob("*.json"):
        summary = _load_json(path)
        if not summary:
            continue
        provider = summary.get("providers", {}).get("l20_triton_fused", {})
        if "speedup_vs_torch_separate" in provider:
            values.append(float(provider["speedup_vs_torch_separate"]))
    return max(values) if values else None


def _max_rope_kv_e2e_output_gain(root: Path) -> float | None:
    summary = _load_json(
        root / "benchmarks/results/l20-vllm-e2e/qwen-policy-v3-safe64-summary.json"
    )
    if not summary:
        return None
    values = []
    for row in summary.get("shapes", []):
        metric = row.get("metrics", {}).get("output_throughput", {})
        if "change_pct" in metric:
            values.append(float(metric["change_pct"]))
    return max(values) if values else None


def _ceiling_summary(root: Path) -> dict[str, Any]:
    return (
        _load_json(root / "benchmarks/results/l20-serving-optimization-ceiling/summary.json")
        or {}
    )


def _max_gpu_boundary_pct(ceiling: dict[str, Any], name: str) -> float | None:
    values = [
        float(run.get("gpu_boundaries", {}).get(name, {}).get("time_pct", 0.0))
        for run in ceiling.get("runs", [])
    ]
    return max(values) if values else None


def _best_lm_head_speedup(ceiling: dict[str, Any]) -> float | None:
    candidate = ceiling.get("lm_head_boundary", {}).get("best_candidate")
    if not candidate:
        return None
    ratio = float(candidate.get("ratio", 0.0))
    if ratio <= 0.0:
        return None
    return 1.0 / ratio


def _max_flashinfer_sampling_itl_win(root: Path) -> float | None:
    base = root / "benchmarks/results/l20-vllm-sampling-winner-v2"
    wins: list[float] = []
    for path in base.glob("*/summary.json"):
        summary = _load_json(path)
        if not summary:
            continue
        for pair in summary.get("pairs", []):
            for shape in pair.get("shapes", {}).values():
                if shape.get("strict_win"):
                    delta = float(shape.get("deltas", {}).get("median_itl_ms_pct", 0.0))
                    wins.append(-delta)
    return max(wins) if wins else None


def _best_batched_lm_head_top1_speedup(root: Path) -> float | None:
    values: list[float] = []
    base = root / "benchmarks/results/l20-lm-head-topk-boundary"
    for path in base.glob("qwen25-b4-h1536-v151936-k1-batched*.json"):
        summary = _load_json(path)
        if not summary:
            continue
        ratio = summary.get("ratios", {}).get("triton_top1_over_full_logits_top1")
        if ratio:
            ratio = float(ratio)
            if ratio > 0.0:
                values.append(1.0 / ratio)
    return max(values) if values else None


def _custom_sampler_regression(root: Path) -> float | None:
    summary = _load_json(
        root / "benchmarks/results/l20-vllm-sampling-itl/qwen25-coder-1p5b-summary.json"
    )
    if not summary:
        return None
    values = [
        float(row.get("median_itl_pct", 0.0))
        for row in summary.get("deltas", {}).get("l20_notrace", {}).values()
    ]
    return max(values) if values else None


def build_boundary_impacts(root: str | Path = ".") -> list[BoundaryImpact]:
    """Build rows from the checked-in benchmark artifacts."""

    repo = Path(root)
    ceiling = _ceiling_summary(repo)
    logits_budget = load_logits_boundary_budget(
        repo / "benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1"
    )
    append_speedup = _max_paged_rope_micro_speedup(repo)
    rope_e2e_gain = _max_rope_kv_e2e_output_gain(repo)
    flashinfer_sampling_win = _max_flashinfer_sampling_itl_win(repo)
    custom_sampler_regression = _custom_sampler_regression(repo)
    current_custom_gpu_pct = _max_gpu_boundary_pct(ceiling, "custom_l20_current")
    sampling_gpu_pct = _max_gpu_boundary_pct(ceiling, "standalone_sampling")
    gemm_gpu_pct = _max_gpu_boundary_pct(ceiling, "gemm_or_gemv")
    lm_head_speedup = _best_lm_head_speedup(ceiling)
    batched_top1_speedup = _best_batched_lm_head_top1_speedup(repo)

    return [
        BoundaryImpact(
            boundary="RoPE + paged KV append",
            status="confirmed_kernel_amdahl_limited",
            micro_speedup_x=append_speedup,
            serving_impact_pct=rope_e2e_gain,
            serving_metric="best safe64 vLLM output-throughput gain across measured shapes",
            gpu_time_pct=current_custom_gpu_pct,
            eligible_fraction_pct=None,
            materialization_mib=None,
            decision="case_study_evidence_not_next_target",
            evidence=(
                "benchmarks/results/l20-paged-rope-policy-v3/ and "
                "benchmarks/results/l20-vllm-e2e/qwen-policy-v3-safe64-summary.json"
            ),
            note="Large append speedup collapses as attention/model/runtime dominate.",
        ),
        BoundaryImpact(
            boundary="Q/K norm + Q/K RoPE + KV write",
            status="o2_path_proven_small_fraction",
            micro_speedup_x=1.47,
            serving_impact_pct=4.52,
            serving_metric="paired serving median ITL improvement from path-proof matrix",
            gpu_time_pct=current_custom_gpu_pct,
            eligible_fraction_pct=None,
            materialization_mib=None,
            decision="do_not_micro_optimize_alone",
            evidence="benchmarks/results/nsys/qk-norm-rope-kv/",
            note="Path is live under O2 but current custom kernel is a small GPU-time share.",
        ),
        BoundaryImpact(
            boundary="FlashInfer sampling route",
            status="production_route_confirmed",
            micro_speedup_x=None,
            serving_impact_pct=flashinfer_sampling_win,
            serving_metric="best strict-win median ITL reduction in paired v2 matrix",
            gpu_time_pct=sampling_gpu_pct,
            eligible_fraction_pct=None,
            materialization_mib=None,
            decision="harden_existing_route",
            evidence="benchmarks/results/l20-vllm-sampling-winner-v2/",
            note="Production route beats torch/native on the useful serving shapes.",
        ),
        BoundaryImpact(
            boundary="Self-written standalone sampler",
            status="negative_serving_result",
            micro_speedup_x=None,
            serving_impact_pct=(
                -custom_sampler_regression if custom_sampler_regression is not None else None
            ),
            serving_metric="median ITL regression versus clean FlashInfer",
            gpu_time_pct=sampling_gpu_pct,
            eligible_fraction_pct=None,
            materialization_mib=None,
            decision="keep_disabled",
            evidence="benchmarks/results/l20-vllm-sampling-itl/qwen25-coder-1p5b-summary.json",
            note="The hook reaches the path but loses to FlashInfer once integrated.",
        ),
        BoundaryImpact(
            boundary="Standalone LM-head top-k",
            status="negative_micro_result",
            micro_speedup_x=lm_head_speedup,
            serving_impact_pct=None,
            serving_metric="not run in serving because micro path is slower",
            gpu_time_pct=gemm_gpu_pct,
            eligible_fraction_pct=None,
            materialization_mib=None,
            decision="avoid_standalone_replacement",
            evidence="benchmarks/results/l20-lm-head-topk-boundary/",
            note="Best standalone candidate is slower than full logits plus optimized top-k.",
        ),
        BoundaryImpact(
            boundary="Batched LM-head greedy top-1",
            status="positive_greedy_micro_only",
            micro_speedup_x=batched_top1_speedup,
            serving_impact_pct=None,
            serving_metric="not run in serving; greedy top-1 only",
            gpu_time_pct=gemm_gpu_pct,
            eligible_fraction_pct=None,
            materialization_mib=None,
            decision="epilogue_prototype_only",
            evidence="benchmarks/results/l20-lm-head-topk-boundary/",
            note="Batch-4 batched partial kernel beats full logits top-1 but lacks production sampler semantics.",
        ),
        BoundaryImpact(
            boundary="LM-head/logits epilogue",
            status="active_p0_budget",
            micro_speedup_x=None,
            serving_impact_pct=None,
            serving_metric="implementation pending",
            gpu_time_pct=gemm_gpu_pct,
            eligible_fraction_pct=logits_budget.eligible_fraction * 100.0,
            materialization_mib=logits_budget.eligible_logits_mib,
            decision="next_core_module",
            evidence="benchmarks/results/l20-vllm-logits-boundary-trace-p1/qwen3-0p6b-o2-v1/",
            note=(
                f"{logits_budget.eligible_events}/{logits_budget.total_events} events are safe "
                "under the current trace gate."
            ),
        ),
    ]


def write_json(rows: Iterable[BoundaryImpact], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([row.to_dict() for row in rows], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(rows: Iterable[BoundaryImpact], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].to_dict()),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_dict())


def render_markdown(rows: Iterable[BoundaryImpact]) -> str:
    lines = [
        "# L20 Boundary Impact",
        "",
        "Positive serving impact means latency reduction or throughput improvement. "
        "Negative values are regressions. Empty cells are unimplemented or not measured.",
        "",
        "| Boundary | Status | Micro speedup | Serving impact | GPU time | Budget | Decision |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        micro = "" if row.micro_speedup_x is None else f"{row.micro_speedup_x:.3f}x"
        serving = (
            ""
            if row.serving_impact_pct is None
            else f"{row.serving_impact_pct:+.2f}%"
        )
        gpu = "" if row.gpu_time_pct is None else f"{row.gpu_time_pct:.2f}%"
        budget = (
            ""
            if row.materialization_mib is None
            else f"{row.eligible_fraction_pct:.2f}% / {row.materialization_mib:.2f} MiB"
        )
        lines.append(
            f"| {row.boundary} | `{row.status}` | {micro} | {serving} | "
            f"{gpu} | {budget} | `{row.decision}` |"
        )
    lines.extend(
        [
            "",
            "## Reading The Table",
            "",
            "- RoPE/KV and Q/K fusion rows show why micro wins are not enough.",
            "- Standalone sampler and standalone LM-head top-k rows are negative controls.",
            "- Batched greedy top-1 is a positive micro signal, not a serving claim.",
            "- The logits epilogue row is not a speed claim; it is the measured "
            "budget for the next implementation.",
        ]
    )
    return "\n".join(lines) + "\n"
