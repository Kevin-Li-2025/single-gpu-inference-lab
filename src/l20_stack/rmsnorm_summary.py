"""Summaries for RMSNorm benchmark JSON reports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OperatorSummary:
    shape_count: int
    fastest_counts: dict[str, int]
    incorrect_results: list[dict[str, object]]
    best_speedup_min: float | None
    best_speedup_max: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RmsNormSummary:
    artifact: str
    gpu_name: str | None
    compute_capability: str | None
    cuda: str | None
    torch: str | None
    triton: str | None
    flashinfer: str | None
    dtype: str | None
    cache_flush_mb: int | None
    warmup_iterations: int | None
    measured_iterations: int | None
    shape_count: int
    all_correct: bool
    operators: dict[str, OperatorSummary]
    large_prefill_rows_4096: list[dict[str, object]]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["operators"] = {
            name: summary.to_dict() for name, summary in self.operators.items()
        }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def summarize_rmsnorm_report(report_path: str | Path) -> RmsNormSummary:
    path = Path(report_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    shapes = payload.get("shapes", [])

    return RmsNormSummary(
        artifact=path.name,
        gpu_name=payload.get("gpu_name"),
        compute_capability=payload.get("compute_capability"),
        cuda=payload.get("cuda"),
        torch=payload.get("torch"),
        triton=payload.get("triton"),
        flashinfer=payload.get("flashinfer"),
        dtype=_first_dtype(shapes),
        cache_flush_mb=payload.get("cache_flush_mb"),
        warmup_iterations=payload.get("warmup_iterations"),
        measured_iterations=payload.get("measured_iterations"),
        shape_count=len(shapes),
        all_correct=bool(payload.get("all_correct")),
        operators={
            name: _summarize_operator(shapes, name)
            for name in ("residual_rmsnorm", "rmsnorm")
            if _has_operator(shapes, name)
        },
        large_prefill_rows_4096=_large_prefill_rows(shapes),
    )


def _has_operator(shapes: list[dict[str, Any]], operator: str) -> bool:
    return any(operator in shape.get("operators", {}) for shape in shapes)


def _first_dtype(shapes: list[dict[str, Any]]) -> str | None:
    for shape in shapes:
        dtype = shape.get("shape", {}).get("dtype")
        if dtype is not None:
            return str(dtype)
    return None


def _summarize_operator(shapes: list[dict[str, Any]], operator: str) -> OperatorSummary:
    shape_count = 0
    fastest_counts: dict[str, int] = {}
    incorrect_results: list[dict[str, object]] = []
    best_speedups: list[float] = []

    for shape in shapes:
        operator_payload = shape.get("operators", {}).get(operator)
        if not operator_payload:
            continue
        shape_count += 1
        fastest = operator_payload.get("fastest_provider")
        if fastest:
            fastest_counts[str(fastest)] = fastest_counts.get(str(fastest), 0) + 1
            best = operator_payload.get("providers", {}).get(fastest, {})
            speedup = best.get("speedup_vs_torch_eager")
            if speedup is not None:
                best_speedups.append(float(speedup))

        for provider, result in operator_payload.get("providers", {}).items():
            if not result.get("correct"):
                incorrect_results.append(
                    {"shape": shape.get("shape"), "provider": provider}
                )

    return OperatorSummary(
        shape_count=shape_count,
        fastest_counts=fastest_counts,
        incorrect_results=incorrect_results,
        best_speedup_min=min(best_speedups) if best_speedups else None,
        best_speedup_max=max(best_speedups) if best_speedups else None,
    )


def _large_prefill_rows(shapes: list[dict[str, Any]]) -> list[dict[str, object]]:
    rows = []
    for shape in shapes:
        shape_payload = shape.get("shape", {})
        if shape_payload.get("rows") != 4096:
            continue
        row = {"shape": shape_payload, "operators": {}}
        for operator_name, operator_payload in shape.get("operators", {}).items():
            fastest = operator_payload.get("fastest_provider")
            if not fastest:
                continue
            providers = operator_payload.get("providers", {})
            best = providers.get(fastest, {})
            torch_eager = providers.get("torch_eager", {})
            row["operators"][operator_name] = {
                "fastest_provider": fastest,
                "best_mean_ms": best.get("timing_ms", {}).get("mean"),
                "torch_eager_mean_ms": torch_eager.get("timing_ms", {}).get("mean"),
                "speedup_vs_torch_eager": best.get("speedup_vs_torch_eager"),
            }
        rows.append(row)
    return rows
