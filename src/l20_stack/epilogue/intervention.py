"""Summarize logits-boundary A/B intervention artifacts."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any


REPORT_RE = re.compile(r"^c(?P<concurrency>\d+)-i(?P<input>\d+)-r(?P<run>\d+)\.json$")

METRICS = (
    "median_itl_ms",
    "output_throughput",
    "request_throughput",
    "median_ttft_ms",
    "p95_itl_ms",
    "p95_ttft_ms",
)
REQUIRED_COMPARISON_METRICS = ("median_itl_ms", "output_throughput")

CONTINUE_EPILOGUE_PROTOTYPE = "continue_epilogue_prototype"
NEEDS_MORE_RUNS = "needs_more_runs"
DO_NOT_CLAIM_WIN = "do_not_claim_win"
NOT_COMPARABLE = "not_comparable"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_delta(candidate: float, baseline: float) -> float | None:
    if baseline == 0.0:
        return None
    return 100.0 * (candidate - baseline) / baseline


def _shape_key(concurrency: int, input_tokens: int) -> str:
    return f"c{concurrency}-i{input_tokens}"


def _shape_sort_key(shape: str) -> tuple[int, int, str]:
    match = re.match(r"^c(?P<concurrency>\d+)-i(?P<input>\d+)$", shape)
    if not match:
        return (0, 0, shape)
    return (int(match.group("concurrency")), int(match.group("input")), shape)


def _has_trace_summary(path: Path) -> bool:
    if (path / "logits-boundary-summary.json").exists():
        return True
    campaign = path / "campaign-summary.json"
    if not campaign.exists():
        return False
    try:
        return "trace_summary" in _load_json(campaign)
    except json.JSONDecodeError:
        return False


def _has_serving_reports(path: Path) -> bool:
    if any(path.glob("c*-i*-r*.json")):
        return True
    campaign = path / "campaign-summary.json"
    if not campaign.exists():
        return False
    try:
        return bool(_load_json(campaign).get("shapes"))
    except json.JSONDecodeError:
        return False


def _score_baseline_dir(path: Path) -> int:
    name = path.name.lower().replace("_", "-")
    score = 0
    if any(token in name for token in ("baseline", "control", "reference")):
        score += 8
    if "trace" in name:
        score += 4
    if _has_trace_summary(path):
        score += 2
    if any(token in name for token in ("candidate", "intervention", "epilogue")):
        score -= 6
    return score


def _score_candidate_dir(path: Path) -> int:
    name = path.name.lower().replace("_", "-")
    score = 0
    if any(token in name for token in ("candidate", "intervention", "epilogue", "prototype")):
        score += 8
    if "serving" in name:
        score += 3
    if _has_serving_reports(path):
        score += 2
    if any(token in name for token in ("baseline", "control", "reference")):
        score -= 6
    return score


def _best_scored_dir(children: list[Path], scorer) -> Path | None:
    scored = [(scorer(path), path) for path in children]
    scored = [(score, path) for score, path in scored if score > 0]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], item[1].name))
    return scored[0][1]


def _resolve_run_dir(root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def discover_ab_dirs(
    input_dir: str | Path,
    baseline_dir: str | Path | None = None,
    candidate_dir: str | Path | None = None,
) -> dict[str, str | None]:
    """Discover baseline and candidate subdirectories under an A/B artifact root."""

    root = Path(input_dir)
    baseline = _resolve_run_dir(root, baseline_dir)
    candidate = _resolve_run_dir(root, candidate_dir)
    children = sorted(path for path in root.iterdir() if path.is_dir())

    if baseline is None:
        baseline = _best_scored_dir(children, _score_baseline_dir)
    if candidate is None:
        candidates = [path for path in children if path != baseline]
        candidate = _best_scored_dir(candidates, _score_candidate_dir)

    if baseline is not None and candidate is None and len(children) == 2:
        other = [path for path in children if path != baseline]
        candidate = other[0] if other else None
    if candidate is not None and baseline is None and len(children) == 2:
        other = [path for path in children if path != candidate]
        baseline = other[0] if other else None
    if baseline is None and candidate is None and len(children) == 2:
        first, second = children
        if _has_trace_summary(first) and not _has_trace_summary(second):
            baseline, candidate = first, second
        elif _has_trace_summary(second) and not _has_trace_summary(first):
            baseline, candidate = second, first

    return {
        "baseline_dir": str(baseline) if baseline is not None else None,
        "candidate_dir": str(candidate) if candidate is not None else None,
    }


def _median_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        if values:
            metrics[metric] = round(statistics.median(values), 5)
    return metrics


def _summarize_raw_reports(run_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str], int]:
    groups: dict[str, list[dict[str, Any]]] = {}
    warnings = []
    report_count = 0
    for path in sorted(run_dir.glob("c*-i*-r*.json")):
        match = REPORT_RE.match(path.name)
        if not match:
            continue
        report = _load_json(path)
        report_count += 1
        if report.get("failed") not in (0, None):
            warnings.append(f"skipped_failed_report:{path.name}")
            continue
        key = _shape_key(int(match.group("concurrency")), int(match.group("input")))
        groups.setdefault(key, []).append(report)

    shapes = {}
    for key, rows in groups.items():
        concurrency, input_tokens, _ = _shape_sort_key(key)
        shapes[key] = {
            "shape": key,
            "max_concurrency": concurrency,
            "input_tokens": input_tokens,
            "runs": len(rows),
            "metrics": _median_metrics(rows),
            "source": "raw_reports",
        }
    return shapes, warnings, report_count


def _summarize_campaign_shapes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    shapes = {}
    for row in payload.get("shapes", []):
        concurrency = int(row["max_concurrency"])
        input_tokens = int(row["input_tokens"])
        key = _shape_key(concurrency, input_tokens)
        shapes[key] = {
            "shape": key,
            "max_concurrency": concurrency,
            "input_tokens": input_tokens,
            "runs": int(row.get("runs", 0)),
            "metrics": {
                name: float(value)
                for name, value in row.get("metrics", {}).items()
                if name in METRICS and value is not None
            },
            "source": "campaign-summary.json",
        }
    return shapes


def _trace_summary(run_dir: Path, campaign: dict[str, Any] | None) -> dict[str, Any]:
    path = run_dir / "logits-boundary-summary.json"
    if path.exists():
        payload = _load_json(path)
        payload.setdefault("trace_source", "logits_boundary")
        return payload
    sampler_path = run_dir / "l20-topk-topp-summary.json"
    if sampler_path.exists():
        payload = _load_json(sampler_path)
        payload.setdefault("trace_source", "l20_topk_topp_sampler")
        return payload
    if campaign is not None:
        payload = dict(campaign.get("trace_summary", {}))
        if payload:
            payload.setdefault("trace_source", "campaign_trace_summary")
        return payload
    return {}


def _trace_eligibility(trace: dict[str, Any]) -> dict[str, Any]:
    if not trace:
        return {"present": False, "shadow_present": False}
    shadow_present = "shadow_events" in trace
    return {
        "present": True,
        "source": trace.get("trace_source", "unknown"),
        "total_events": int(trace.get("total_events", 0)),
        "eligible_events": int(trace.get("eligible_events", 0)),
        "fallback_events": int(trace.get("fallback_events", 0)),
        "eligible_fraction": float(trace.get("eligible_fraction", 0.0)),
        "eligible_logits_mib": float(trace.get("eligible_logits_mib", 0.0)),
        "total_logits_mib": float(trace.get("total_logits_mib", 0.0)),
        "shadow_present": shadow_present,
        "shadow_events": int(trace.get("shadow_events", 0)),
        "shadow_eligible_events": int(trace.get("shadow_eligible_events", 0)),
        "shadow_fallback_events": int(trace.get("shadow_fallback_events", 0)),
        "shadow_eligible_fraction": float(trace.get("shadow_eligible_fraction", 0.0)),
        "shadow_avoidable_logits_mib": float(
            trace.get("shadow_avoidable_logits_mib", 0.0)
        ),
    }


def summarize_run_dir(run_dir: str | Path | None, role: str) -> dict[str, Any]:
    """Summarize one baseline or candidate artifact directory."""

    if run_dir is None:
        return {
            "role": role,
            "dir": None,
            "exists": False,
            "serving_report_count": 0,
            "shapes": [],
            "trace_eligibility": {"present": False, "shadow_present": False},
            "warnings": [f"missing_{role}_dir"],
        }

    path = Path(run_dir)
    if not path.exists() or not path.is_dir():
        return {
            "role": role,
            "dir": str(path),
            "exists": False,
            "serving_report_count": 0,
            "shapes": [],
            "trace_eligibility": {"present": False, "shadow_present": False},
            "warnings": [f"missing_{role}_dir"],
        }

    campaign_path = path / "campaign-summary.json"
    campaign = _load_json(campaign_path) if campaign_path.exists() else None
    shapes, warnings, report_count = _summarize_raw_reports(path)
    if not shapes and campaign is not None:
        shapes = _summarize_campaign_shapes(campaign)
        report_count = int(campaign.get("serving_report_count", report_count))
    trace = _trace_summary(path, campaign)
    return {
        "role": role,
        "dir": str(path),
        "exists": True,
        "serving_report_count": report_count,
        "shapes": [shapes[key] for key in sorted(shapes, key=_shape_sort_key)],
        "trace_eligibility": _trace_eligibility(trace),
        "warnings": warnings,
    }


def _shape_map(run_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["shape"]: row for row in run_summary.get("shapes", [])}


def _metric_payload(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    metrics = row.get("metrics", {})
    return {
        "runs": row.get("runs", 0),
        "median_itl_ms": metrics.get("median_itl_ms"),
        "output_throughput": metrics.get("output_throughput"),
    }


def _missing_metrics(label: str, row: dict[str, Any] | None) -> list[str]:
    if row is None:
        return [f"missing_{label}_shape"]
    metrics = row.get("metrics", {})
    return [
        f"missing_{label}_{metric}"
        for metric in REQUIRED_COMPARISON_METRICS
        if metric not in metrics
    ]


def _compare_shape(
    shape: str,
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    min_runs_per_shape: int,
) -> dict[str, Any]:
    reasons = []
    reasons.extend(_missing_metrics("baseline", baseline))
    reasons.extend(_missing_metrics("candidate", candidate))

    if baseline is not None and baseline.get("runs", 0) < min_runs_per_shape:
        reasons.append("insufficient_baseline_runs")
    if candidate is not None and candidate.get("runs", 0) < min_runs_per_shape:
        reasons.append("insufficient_candidate_runs")

    baseline_metrics = baseline.get("metrics", {}) if baseline else {}
    candidate_metrics = candidate.get("metrics", {}) if candidate else {}
    deltas = {}
    for metric in REQUIRED_COMPARISON_METRICS:
        if metric in baseline_metrics and metric in candidate_metrics:
            deltas[f"{metric}_pct"] = _pct_delta(
                float(candidate_metrics[metric]),
                float(baseline_metrics[metric]),
            )

    median_itl_delta = deltas.get("median_itl_ms_pct")
    throughput_delta = deltas.get("output_throughput_pct")
    metric_complete = median_itl_delta is not None and throughput_delta is not None
    strict_win = (
        bool(metric_complete)
        and float(median_itl_delta) < 0.0
        and float(throughput_delta) > 0.0
    )

    concurrency, input_tokens, _ = _shape_sort_key(shape)
    return {
        "shape": shape,
        "max_concurrency": concurrency,
        "input_tokens": input_tokens,
        "baseline": _metric_payload(baseline),
        "candidate": _metric_payload(candidate),
        "deltas": deltas,
        "median_itl_win": median_itl_delta is not None and median_itl_delta < 0.0,
        "throughput_win": throughput_delta is not None and throughput_delta > 0.0,
        "strict_win": strict_win,
        "incomplete": bool(reasons),
        "complete": not reasons,
        "incomplete_reasons": reasons,
    }


def _verdict(rows: list[dict[str, Any]], incomplete_reasons: list[str]) -> str:
    compared_rows = [
        row
        for row in rows
        if row["baseline"] is not None and row["candidate"] is not None
    ]
    complete_rows = [row for row in compared_rows if not row["incomplete_reasons"]]
    if any(row["strict_win"] is False for row in complete_rows):
        return DO_NOT_CLAIM_WIN
    if incomplete_reasons or not compared_rows:
        return NEEDS_MORE_RUNS
    if any(not row["strict_win"] for row in compared_rows):
        return DO_NOT_CLAIM_WIN
    return CONTINUE_EPILOGUE_PROTOTYPE


def summarize_logits_boundary_ab(
    input_dir: str | Path,
    baseline_dir: str | Path | None = None,
    candidate_dir: str | Path | None = None,
    min_runs_per_shape: int = 2,
) -> dict[str, Any]:
    """Build a conservative A/B verdict from logits-boundary artifact directories."""

    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"input root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"input root is not a directory: {root}")

    discovered = discover_ab_dirs(root, baseline_dir, candidate_dir)
    baseline = summarize_run_dir(discovered["baseline_dir"], "baseline")
    candidate = summarize_run_dir(discovered["candidate_dir"], "candidate")
    baseline_shapes = _shape_map(baseline)
    candidate_shapes = _shape_map(candidate)
    shape_keys = sorted(set(baseline_shapes) | set(candidate_shapes), key=_shape_sort_key)
    rows = [
        _compare_shape(
            shape,
            baseline_shapes.get(shape),
            candidate_shapes.get(shape),
            min_runs_per_shape,
        )
        for shape in shape_keys
    ]

    incomplete_reasons = []
    if not baseline["exists"]:
        incomplete_reasons.append("missing_baseline_dir")
    if not candidate["exists"]:
        incomplete_reasons.append("missing_candidate_dir")
    if baseline["exists"] and not baseline_shapes:
        incomplete_reasons.append("missing_baseline_serving_reports")
    if candidate["exists"] and not candidate_shapes:
        incomplete_reasons.append("missing_candidate_serving_reports")
    for row in rows:
        for reason in row["incomplete_reasons"]:
            incomplete_reasons.append(f"{row['shape']}:{reason}")

    historical_verdict = _verdict(rows, incomplete_reasons)
    compared_shape_count = sum(
        1 for row in rows if row["baseline"] is not None and row["candidate"] is not None
    )
    strict_win_count = sum(1 for row in rows if row["strict_win"])
    return {
        "schema_version": 1,
        "evidence_status": "requires_semantic_validation",
        "performance_comparable": False,
        "input_dir": str(root),
        "baseline_dir": discovered["baseline_dir"],
        "candidate_dir": discovered["candidate_dir"],
        "status": NOT_COMPARABLE,
        "collection_status": "incomplete" if incomplete_reasons else "complete",
        "incomplete": bool(incomplete_reasons),
        "verdict": NOT_COMPARABLE,
        "historical_verdict": historical_verdict,
        "incomplete_reasons": incomplete_reasons,
        "min_runs_per_shape": min_runs_per_shape,
        "gate": {
            "definition": (
                "A strict win requires lower candidate median ITL and higher "
                "candidate output throughput versus baseline for every compared shape."
            ),
            "compared_shapes": compared_shape_count,
            "strict_win_shapes": strict_win_count,
            "total_shapes": len(rows),
            "strict_win_fraction": strict_win_count / compared_shape_count
            if compared_shape_count
            else 0.0,
        },
        "baseline": baseline,
        "candidate": candidate,
        "shapes": rows,
    }


def _format_float(value: Any, places: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{places}f}"


def _format_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):+.2f}%"


def _eligibility_lines(label: str, summary: dict[str, Any]) -> list[str]:
    eligibility = summary.get("trace_eligibility", {})
    if not eligibility.get("present"):
        return [f"| {label} | no | n/a | n/a | n/a | n/a |"]
    return [
        (
            f"| {label} | yes | {eligibility.get('eligible_events', 0)} / "
            f"{eligibility.get('total_events', 0)} | "
            f"{float(eligibility.get('eligible_fraction', 0.0)):.2%} | "
            f"{'yes' if eligibility.get('shadow_present') else 'no'} | "
            f"{float(eligibility.get('shadow_eligible_fraction', 0.0)):.2%} |"
        )
    ]


def render_logits_boundary_ab_markdown(summary: dict[str, Any]) -> str:
    """Render a Markdown report for a logits-boundary A/B summary."""

    lines = [
        "# L20 Logits Boundary A/B Verdict",
        "",
        "> **Provisional semantics:** the recorded deltas are historical and",
        "> not current performance evidence until the corrected sampler passes",
        "> native-equivalent semantic revalidation.",
        "",
        f"- Input: `{summary['input_dir']}`",
        f"- Baseline: `{summary.get('baseline_dir')}`",
        f"- Candidate: `{summary.get('candidate_dir')}`",
        f"- Evidence status: `{summary['evidence_status']}`",
        f"- Performance comparable: `{summary['performance_comparable']}`",
        f"- Collection status: `{summary['collection_status']}`",
        f"- Current verdict: `{summary['verdict']}`",
        f"- Historical verdict: `{summary['historical_verdict']}`",
        "",
        "## Historical gate",
        "",
        summary["gate"]["definition"],
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Compared shapes | {summary['gate']['compared_shapes']} |",
        f"| Strict-win shapes | {summary['gate']['strict_win_shapes']} |",
        f"| Total shapes | {summary['gate']['total_shapes']} |",
        f"| Minimum runs per shape | {summary['min_runs_per_shape']} |",
        "",
        "## Trace Eligibility",
        "",
        "| Run | Present | Eligible events | Eligible fraction | "
        "Shadow present | Shadow eligible |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    lines.extend(_eligibility_lines("baseline", summary["baseline"]))
    lines.extend(_eligibility_lines("candidate", summary["candidate"]))
    lines.extend(
        [
            "",
            "## Historical recorded shape results",
            "",
            "| Shape | Baseline runs | Candidate runs | Baseline ITL ms | Candidate ITL ms | "
            "ITL delta | Baseline tok/s | Candidate tok/s | Throughput delta | Strict win |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    if summary["shapes"]:
        for row in summary["shapes"]:
            baseline = row.get("baseline") or {}
            candidate = row.get("candidate") or {}
            deltas = row.get("deltas", {})
            lines.append(
                f"| `{row['shape']}` | "
                f"{baseline.get('runs', 0)} | "
                f"{candidate.get('runs', 0)} | "
                f"{_format_float(baseline.get('median_itl_ms'))} | "
                f"{_format_float(candidate.get('median_itl_ms'))} | "
                f"{_format_pct(deltas.get('median_itl_ms_pct'))} | "
                f"{_format_float(baseline.get('output_throughput'), 1)} | "
                f"{_format_float(candidate.get('output_throughput'), 1)} | "
                f"{_format_pct(deltas.get('output_throughput_pct'))} | "
                f"{'yes' if row['strict_win'] else 'no'} |"
            )
    else:
        lines.append("| n/a | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | no |")

    if summary.get("incomplete_reasons"):
        lines.extend(["", "## Incomplete Reasons", "", "| Reason |", "| --- |"])
        for reason in summary["incomplete_reasons"]:
            lines.append(f"| `{reason}` |")
    lines.append("")
    return "\n".join(lines)
