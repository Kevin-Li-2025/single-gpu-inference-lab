#!/usr/bin/env python3
"""Probe KleidiAI NEON and SME2 Q4 GEMV on Qwen2.5-Coder-3B shapes."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


SHAPES = (
    {"name": "ffn_up_gate", "m": 1, "n": 11008, "k": 2048, "bl": 32},
    {"name": "ffn_down", "m": 1, "n": 2048, "k": 11008, "bl": 32},
)
FILTER = (
    "qai8dxp1x4_qsi4c32p"
    "(4x4_1x4_neon_dotprod|4vlx4_1x4vl_sme2_dot)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def run_correctness(test_binary: str) -> dict:
    completed = subprocess.run(
        [test_binary, "--gtest_filter=*sme2_dot*", "--gtest_brief=1"],
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests from \d+ test suites ran", completed.stdout)
    passed = re.search(r"\[\s*PASSED\s*\]\s+(\d+) tests", completed.stdout)
    if match is None or passed is None:
        raise RuntimeError("unable to parse KleidiAI correctness output")
    return {"tests_run": int(match.group(1)), "tests_passed": int(passed.group(1))}


def run_shape(args: argparse.Namespace, shape: dict) -> dict:
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".log") as result_file:
        completed = subprocess.run(
            [
                args.benchmark,
                "matmul",
                "-m",
                str(shape["m"]),
                "-n",
                str(shape["n"]),
                "-k",
                str(shape["k"]),
                "-b",
                str(shape["bl"]),
                f"--benchmark_filter={FILTER}",
                "--benchmark_min_time=0.2s",
                f"--benchmark_repetitions={args.repetitions}",
                "--benchmark_report_aggregates_only=true",
            ],
            check=False,
            stdout=result_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        result_file.seek(0)
        output = result_file.read()

    if completed.returncode not in (0, -11, 139):
        raise RuntimeError(f"KleidiAI benchmark failed with {completed.returncode}")

    medians = {}
    pattern = re.compile(r"^(?P<name>\S+_median)\s+(?P<time>[0-9.]+) ns", re.MULTILINE)
    for match in pattern.finditer(output):
        name = match.group("name")
        value = float(match.group("time"))
        if "sme2_dot" in name:
            medians["sme2_median_ns"] = value
        elif "neon_dotprod" in name:
            medians["neon_median_ns"] = value
    if len(medians) != 2:
        raise RuntimeError(f"missing benchmark medians: {medians}")
    return {
        **shape,
        **medians,
        "benchmark_exit_code": completed.returncode,
        "sme2_speedup": medians["neon_median_ns"] / medians["sme2_median_ns"],
    }


def main() -> int:
    args = parse_args()
    correctness = run_correctness(args.test)
    rows = [run_shape(args, shape) for shape in SHAPES]
    payload = {
        "schema_version": 1,
        "implementation": "scripts/benchmark_m4_sme2_qwen3b.py",
        "scope": "external KleidiAI kernel probe; not integrated model inference",
        "format": "FP32 activation dynamically quantized to QAI8; QSI4C32 weights",
        "runner_note": (
            "KleidiAI v1.28.0 reports complete medians, then exits SIGSEGV on this "
            "macOS M4 host; each row records that external runner teardown status"
        ),
        "correctness": correctness,
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
