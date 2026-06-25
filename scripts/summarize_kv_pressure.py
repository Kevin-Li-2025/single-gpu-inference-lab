#!/usr/bin/env python3
"""Summarize L20 multi-turn KV-pressure benchmark directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None


def summarize_success(path: Path, payload: dict, metadata: dict) -> dict:
    reports = payload.get("reports", [])
    ttft = [row["ttft_ms"] for row in reports if row.get("ttft_ms") is not None]
    e2e = [row["e2e_ms"] for row in reports if row.get("e2e_ms") is not None]
    first_ttft = ttft[0] if ttft else None
    last_ttft = ttft[-1] if ttft else None
    late_over_first = None
    if first_ttft not in (None, 0) and last_ttft is not None:
        late_over_first = last_ttft / first_ttft
    return {
        "path": str(path),
        "status": "ok",
        "metadata": metadata,
        "turns_completed": len(reports),
        "first_turn_ttft_ms": first_ttft,
        "last_turn_ttft_ms": last_ttft,
        "late_over_first_ttft": late_over_first,
        "median_ttft_ms": median_or_none(ttft),
        "median_e2e_ms": median_or_none(e2e),
        "max_ttft_ms": max(ttft) if ttft else None,
    }


def summarize_run_dir(run_dir: Path) -> dict:
    metadata_path = run_dir / "kv-pressure-run.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    failure_path = run_dir / "kv-pressure-failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        return {
            "path": str(run_dir),
            "status": failure.get("status", "failed"),
            "reason": failure.get("reason"),
            "oom_suspected": failure.get("oom_suspected"),
            "flashinfer_observed": failure.get("flashinfer_observed"),
            "metadata": failure.get("metadata", metadata),
        }

    candidates = sorted(run_dir.glob("kv-pressure-prefix-cache-*.json"))
    if not candidates:
        return {
            "path": str(run_dir),
            "status": "missing_result",
            "metadata": metadata,
        }
    return summarize_success(run_dir, load_json(candidates[0]), metadata)


def workload_key(report: dict) -> tuple:
    metadata = report.get("metadata", {})
    return (
        metadata.get("model"),
        metadata.get("served_name"),
        metadata.get("prefix_caching"),
        metadata.get("prefix_chars"),
        metadata.get("turns"),
        metadata.get("max_tokens"),
        metadata.get("max_model_len"),
        metadata.get("attention_backend"),
    )


def ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return numerator / denominator


def build_comparisons(reports: list[dict]) -> list[dict]:
    comparisons = []
    groups: dict[tuple, dict[str, dict]] = {}
    for report in reports:
        if report.get("status") != "ok":
            continue
        kv_dtype = report.get("metadata", {}).get("kv_cache_dtype")
        if kv_dtype is None:
            continue
        groups.setdefault(workload_key(report), {})[kv_dtype] = report

    for key, rows in sorted(groups.items(), key=lambda item: repr(item[0])):
        baseline = rows.get("auto")
        fp8 = rows.get("fp8")
        if not baseline or not fp8:
            continue
        comparisons.append(
            {
                "workload_key": {
                    "model": key[0],
                    "served_name": key[1],
                    "prefix_caching": key[2],
                    "prefix_chars": key[3],
                    "turns": key[4],
                    "max_tokens": key[5],
                    "max_model_len": key[6],
                    "attention_backend": key[7],
                },
                "baseline_kv_cache_dtype": "auto",
                "candidate_kv_cache_dtype": "fp8",
                "median_ttft_speedup_fp8_over_auto": ratio(
                    baseline.get("median_ttft_ms"),
                    fp8.get("median_ttft_ms"),
                ),
                "median_e2e_speedup_fp8_over_auto": ratio(
                    baseline.get("median_e2e_ms"),
                    fp8.get("median_e2e_ms"),
                ),
                "last_turn_ttft_speedup_fp8_over_auto": ratio(
                    baseline.get("last_turn_ttft_ms"),
                    fp8.get("last_turn_ttft_ms"),
                ),
                "first_turn_ttft_speedup_fp8_over_auto": ratio(
                    baseline.get("first_turn_ttft_ms"),
                    fp8.get("first_turn_ttft_ms"),
                ),
            }
        )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for path in args.paths:
        if path.is_file():
            payload = load_json(path)
            metadata = {}
            reports.append(summarize_success(path.parent, payload, metadata))
        elif path.is_dir():
            child_runs = [
                child for child in sorted(path.iterdir())
                if child.is_dir() and (
                    (child / "kv-pressure-run.json").exists()
                    or (child / "kv-pressure-failure.json").exists()
                    or list(child.glob("kv-pressure-prefix-cache-*.json"))
                )
            ]
            if child_runs:
                reports.extend(summarize_run_dir(child) for child in child_runs)
            else:
                reports.append(summarize_run_dir(path))

    ok_reports = [row for row in reports if row["status"] == "ok"]
    comparisons = build_comparisons(reports)
    result = {
        "schema_version": 1,
        "reports": reports,
        "comparisons": comparisons,
        "summary": {
            "total_runs": len(reports),
            "ok_runs": len(ok_reports),
            "failed_runs": len(reports) - len(ok_reports),
            "comparison_count": len(comparisons),
            "best_median_ttft_ms": min(
                (row["median_ttft_ms"] for row in ok_reports if row["median_ttft_ms"] is not None),
                default=None,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
