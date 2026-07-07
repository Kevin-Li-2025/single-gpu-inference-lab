#!/usr/bin/env python3
import csv
import json
import statistics
import sys
from pathlib import Path


def fmt(value: str, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: summarize_csv.py RESULTS.csv", file=sys.stderr)
        return 2

    csv_path = Path(sys.argv[1])
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        raise SystemExit(f"no rows in {csv_path}")

    speedups = [float(row["speedup"]) for row in rows]
    max_diff = max(float(row["max_abs_diff"]) for row in rows)
    gpu = rows[0]["gpu"]
    compute_cap = rows[0]["compute_cap"]
    has_policy = "policy_provider" in rows[0]
    policy_speedups = (
        [float(row["policy_speedup"]) for row in rows] if has_policy else []
    )
    policy_regrets = [float(row["policy_regret"]) for row in rows] if has_policy else []
    sparse_policy_cases = (
        sum(1 for row in rows if row["policy_provider"] == "sparse")
        if has_policy
        else 0
    )

    summary = {
        "gpu": gpu,
        "compute_cap": compute_cap,
        "cases": len(rows),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "max_abs_diff": max_diff,
    }
    if has_policy:
        summary["policy"] = {
            "rule": (
                "Use sparse when vocab>=65536 and batch*vocab>=524288 "
                "and unique_tokens<=1024; otherwise use dense."
            ),
            "sparse_cases": sparse_policy_cases,
            "dense_cases": len(rows) - sparse_policy_cases,
            "median_speedup": statistics.median(policy_speedups),
            "min_speedup": min(policy_speedups),
            "max_speedup": max(policy_speedups),
            "max_regret": max(policy_regrets),
            "regression_cases": sum(1 for value in policy_speedups if value < 1.0),
        }
    json_path = csv_path.with_name("summary.json")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"# L20 sparse repetition-penalty benchmark")
    print()
    print(f"- GPU: `{gpu}` (`sm_{compute_cap.replace('.', '')}`)")
    print(f"- Cases: `{len(rows)}`")
    print(f"- Median speedup: `{summary['median_speedup']:.2f}x`")
    print(f"- Speedup range: `{summary['min_speedup']:.2f}x` to `{summary['max_speedup']:.2f}x`")
    print(f"- Max dense-vs-sparse diff: `{max_diff:.1f}`")
    if has_policy:
        policy = summary["policy"]
        print(f"- Policy sparse cases: `{policy['sparse_cases']} / {len(rows)}`")
        print(f"- Policy median speedup: `{policy['median_speedup']:.2f}x`")
        print(f"- Policy min speedup: `{policy['min_speedup']:.2f}x`")
        print(f"- Policy max regret: `{policy['max_regret']:.2f}x`")
        print(f"- Policy regression cases: `{policy['regression_cases']}`")
    print()
    if has_policy:
        print(
            "| batch | vocab | unique history | dense ms | sparse ms | "
            "speedup | policy | policy speedup | regret |"
        )
        print("|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    else:
        print("| batch | vocab | unique history | dense ms | sparse ms | speedup |")
        print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        base = (
            f"| {row['batch']} | {row['vocab']} | {row['unique_tokens']} | "
            f"{fmt(row['dense_ms'])} | {fmt(row['sparse_ms'])} | "
            f"{float(row['speedup']):.2f}x |"
        )
        if has_policy:
            print(
                f"{base} {row['policy_provider']} | "
                f"{float(row['policy_speedup']):.2f}x | "
                f"{float(row['policy_regret']):.2f}x |"
            )
        else:
            print(base)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
