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

    summary = {
        "gpu": gpu,
        "compute_cap": compute_cap,
        "cases": len(rows),
        "median_speedup": statistics.median(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "max_abs_diff": max_diff,
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
    print()
    print("| batch | vocab | unique history | dense ms | sparse ms | speedup |")
    print("|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['batch']} | {row['vocab']} | {row['unique_tokens']} | "
            f"{fmt(row['dense_ms'])} | {fmt(row['sparse_ms'])} | "
            f"{float(row['speedup']):.2f}x |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
