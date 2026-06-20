"""Build measured L20 RMSNorm dispatch policies from benchmark reports."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

from l20_stack.ops.triton_rmsnorm import residual_rmsnorm_backend


PRODUCTION_PROVIDERS = ("l20_inplace", "flashinfer", "torch_eager")


@dataclass(frozen=True)
class ShapePolicy:
    rows: int
    hidden_size: int
    dtype: str
    fastest_provider: str
    recommended_backend: str
    current_backend: str
    median_p50_ms: float
    median_speedup_vs_torch_eager: float
    margin_vs_next_provider_pct: float
    stable: bool
    source_runs: int

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["median_p50_ms"] = round(self.median_p50_ms, 4)
        payload["median_speedup_vs_torch_eager"] = round(
            self.median_speedup_vs_torch_eager, 3
        )
        payload["margin_vs_next_provider_pct"] = round(
            self.margin_vs_next_provider_pct, 2
        )
        return payload


def _shape_key(shape_report: Mapping[str, object]) -> Tuple[int, int, str]:
    shape = shape_report["shape"]
    return int(shape["rows"]), int(shape["hidden_size"]), str(shape["dtype"])


def _provider_timings(
    shape_reports: Iterable[Mapping[str, object]],
) -> Dict[str, List[float]]:
    timings: Dict[str, List[float]] = {}
    for shape_report in shape_reports:
        operator = shape_report["operators"]["residual_rmsnorm"]
        for provider, provider_report in operator["providers"].items():
            timing = provider_report.get("timing_ms")
            if provider_report.get("correct") and timing is not None:
                timings.setdefault(provider, []).append(float(timing["p50"]))
    return timings


def _recommended_backend(provider: str, rows: int, hidden_size: int) -> str:
    if provider in ("triton_w4", "triton_w8", "l20_dispatch"):
        return "triton"
    if provider == "l20_inplace":
        return residual_rmsnorm_backend(rows, hidden_size, flashinfer_available=True)
    if provider == "flashinfer":
        return "flashinfer"
    if provider == "torch_eager":
        return "torch_eager"
    return provider


def build_residual_rmsnorm_policy(
    reports: Iterable[Mapping[str, object]],
    *,
    minimum_margin_pct: float = 2.0,
) -> List[ShapePolicy]:
    """Aggregate repeated L20 benchmark reports into a conservative policy."""

    report_list = list(reports)
    if not report_list:
        raise ValueError("at least one benchmark report is required")

    grouped: Dict[Tuple[int, int, str], List[Mapping[str, object]]] = {}
    for report in report_list:
        for shape_report in report["shapes"]:
            if "residual_rmsnorm" in shape_report["operators"]:
                grouped.setdefault(_shape_key(shape_report), []).append(shape_report)

    policies = []
    for (rows, hidden_size, dtype), shape_reports in sorted(grouped.items()):
        timings = _provider_timings(shape_reports)
        complete = {
            provider: values
            for provider, values in timings.items()
            if len(values) == len(shape_reports)
        }
        production = {
            provider: statistics.median(values)
            for provider, values in complete.items()
            if provider in PRODUCTION_PROVIDERS
        }
        if not production:
            continue

        ranked = sorted(production.items(), key=lambda item: item[1])
        fastest_provider, fastest_ms = ranked[0]
        next_ms = ranked[1][1] if len(ranked) > 1 else fastest_ms
        margin_pct = 0.0
        if fastest_ms > 0 and next_ms > fastest_ms:
            margin_pct = 100.0 * (next_ms - fastest_ms) / fastest_ms

        eager_ms = production.get("torch_eager")
        speedup = eager_ms / fastest_ms if eager_ms else 1.0
        stable = margin_pct >= minimum_margin_pct or fastest_provider == "torch_eager"
        current_backend = residual_rmsnorm_backend(
            rows, hidden_size, flashinfer_available=True
        )
        policies.append(
            ShapePolicy(
                rows=rows,
                hidden_size=hidden_size,
                dtype=dtype,
                fastest_provider=fastest_provider,
                recommended_backend=_recommended_backend(
                    fastest_provider, rows, hidden_size
                ),
                current_backend=current_backend,
                median_p50_ms=fastest_ms,
                median_speedup_vs_torch_eager=speedup,
                margin_vs_next_provider_pct=margin_pct,
                stable=stable,
                source_runs=len(shape_reports),
            )
        )
    return policies


def load_reports(paths: Iterable[Path]) -> List[Mapping[str, object]]:
    reports = []
    for path in paths:
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    return reports


def policy_payload(policies: Iterable[ShapePolicy]) -> Dict[str, object]:
    return {
        "operator": "residual_rmsnorm",
        "target": "NVIDIA L20 sm_89",
        "policy": [policy.to_dict() for policy in policies],
    }
