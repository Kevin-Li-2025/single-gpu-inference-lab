#!/usr/bin/env python3
"""Run a power-qualified, interleaved llama.cpp A/B for the M4 Q4_K SME2 path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from statistics import median


CUSTOM_ENV = (
    "GGML_M4_Q4K_SME2",
    "GGML_M4_Q4K_SME2_TENSORS",
    "GGML_M4_Q4K_SME2_SHARED_Q8",
    "GGML_M4_Q4K_SME2_SHARE_PERCENT",
    "GGML_M4_Q4K_SME2_PARALLEL_CORRECTION",
    "GGML_M4_Q4K_SME2_TRACE",
)
DEFAULT_PROMPT = (
    "Implement a Python function merge_intervals(intervals) that merges overlapping "
    "integer intervals. Include type hints and two assertions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-bench", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--llama-completion", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--cpu-mask", default="0x3c0")
    parser.add_argument("--n-gen", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--max-load-per-cpu", type=float, default=0.25)
    parser.add_argument("--min-speedup", type=float, default=1.0)
    parser.add_argument("--min-pair-speedup", type=float, default=0.98)
    parser.add_argument("--include-serial-control", action="store_true")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--n-predict", type=int, default=96)
    parser.add_argument("--allow-dirty-host", action="store_true")
    return parser.parse_args()


def run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)


def power_state() -> dict[str, object]:
    battery = run_text(["pmset", "-g", "batt"])
    custom = run_text(["pmset", "-g", "custom"])
    match = re.search(r"Now drawing from '([^']+)'", battery)
    source = match.group(1) if match else "unknown"
    section_name = "AC Power" if source == "AC Power" else "Battery Power"
    section_match = re.search(
        rf"^{re.escape(section_name)}:\n(?P<body>(?:^[ \t].*\n?)*)",
        custom,
        flags=re.MULTILINE,
    )
    section = section_match.group("body") if section_match else ""
    low_power_match = re.search(r"^\s*lowpowermode\s+(\d+)\s*$", section, re.MULTILINE)
    low_power_mode = int(low_power_match.group(1)) if low_power_match else None
    return {
        "source": source,
        "low_power_mode": low_power_mode,
        "battery_report": battery.strip(),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(args: argparse.Namespace) -> None:
    if not args.llama_bench.is_file() or not os.access(args.llama_bench, os.X_OK):
        raise SystemExit(f"llama-bench is not executable: {args.llama_bench}")
    if not args.model.is_file():
        raise SystemExit(f"model does not exist: {args.model}")
    with args.model.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise SystemExit(f"model is not GGUF: {args.model}")
    if args.llama_completion is not None and (
        not args.llama_completion.is_file() or not os.access(args.llama_completion, os.X_OK)
    ):
        raise SystemExit(f"llama-completion is not executable: {args.llama_completion}")
    if min(args.threads, args.n_gen, args.repetitions, args.pairs) <= 0:
        raise SystemExit("threads, n-gen, repetitions, and pairs must be positive")


def mode_env(mode: str) -> dict[str, str]:
    env = os.environ.copy()
    for name in CUSTOM_ENV:
        env.pop(name, None)
    if mode in {"candidate", "serial"}:
        # Shared candidate policy: down-only, one Q8 pack, and 25% SME rows.
        env["GGML_M4_Q4K_SME2"] = "1"
    if mode == "candidate":
        env["GGML_M4_Q4K_SME2_PARALLEL_CORRECTION"] = "1"
    if mode == "serial":
        env["GGML_M4_Q4K_SME2_PARALLEL_CORRECTION"] = "0"
    return env


def run_bench(
    args: argparse.Namespace, mode: str, pair: int, output_dir: Path
) -> dict[str, object]:
    stem = f"pair-{pair:02d}-{mode}"
    output_path = output_dir / f"{stem}.json"
    stderr_path = output_dir / f"{stem}.stderr"
    command = [
        str(args.llama_bench),
        "-m",
        str(args.model),
        "-p",
        "0",
        "-n",
        str(args.n_gen),
        "-t",
        str(args.threads),
        "-r",
        str(args.repetitions),
        "-ngl",
        "0",
        "-C",
        args.cpu_mask,
        "--cpu-strict",
        "1",
        "--poll",
        "0",
        "-o",
        "json",
    ]
    with (
        output_path.open("w", encoding="utf-8") as stdout,
        stderr_path.open("w", encoding="utf-8") as stderr,
    ):
        subprocess.run(
            command,
            env=mode_env(mode),
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    row = payload[0]
    samples = [float(value) for value in row["samples_ts"]]
    return {
        "pair": pair,
        "mode": mode,
        "samples_tps": samples,
        "median_tps": median(samples),
        "raw": output_path.name,
        "stderr": stderr_path.name,
    }


def run_completion(args: argparse.Namespace, mode: str, output_dir: Path) -> dict[str, object]:
    assert args.llama_completion is not None
    output_path = output_dir / f"completion-{mode}.txt"
    stderr_path = output_dir / f"completion-{mode}.stderr"
    command = [
        str(args.llama_completion),
        "-m",
        str(args.model),
        "-p",
        args.prompt,
        "-n",
        str(args.n_predict),
        "-t",
        str(args.threads),
        "-tb",
        str(args.threads),
        "-ngl",
        "0",
        "-C",
        args.cpu_mask,
        "--cpu-strict",
        "1",
        "--poll",
        "0",
        "--temp",
        "0",
        "--seed",
        "7",
        "--no-conversation",
        "--no-display-prompt",
        "--simple-io",
    ]
    with output_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        subprocess.run(
            command,
            env=mode_env(mode),
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    return {
        "mode": mode,
        "sha256": sha256_file(output_path),
        "bytes": output_path.stat().st_size,
        "output": output_path.name,
        "stderr": stderr_path.name,
    }


def main() -> int:
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    power = power_state()
    load_1m = os.getloadavg()[0]
    logical_cpus = os.cpu_count() or 1
    load_per_cpu = load_1m / logical_cpus
    host_clean = (
        power["source"] == "AC Power"
        and power["low_power_mode"] == 0
        and load_per_cpu <= args.max_load_per_cpu
    )
    if not host_clean and not args.allow_dirty_host:
        raise SystemExit(
            "refusing performance run on a contaminated host: "
            f"source={power['source']}, low_power_mode={power['low_power_mode']}, "
            f"load_per_cpu={load_per_cpu:.3f}; connect AC power, disable low-power "
            "mode, and wait for background load to settle"
        )

    rows: list[dict[str, object]] = []
    triangle_orders = (
        ("baseline", "serial", "candidate"),
        ("candidate", "baseline", "serial"),
        ("serial", "candidate", "baseline"),
    )
    for pair in range(1, args.pairs + 1):
        if args.include_serial_control:
            modes = triangle_orders[(pair - 1) % len(triangle_orders)]
        else:
            modes = ("baseline", "candidate") if pair % 2 else ("candidate", "baseline")
        for mode in modes:
            print(f"pair={pair}/{args.pairs} mode={mode}", flush=True)
            rows.append(run_bench(args, mode, pair, args.output_dir))
            if args.delay_seconds:
                time.sleep(args.delay_seconds)

    by_pair: list[dict[str, object]] = []
    measured_modes = ("baseline", "serial", "candidate") if args.include_serial_control else (
        "baseline",
        "candidate",
    )
    pooled: dict[str, list[float]] = {mode: [] for mode in measured_modes}
    for pair in range(1, args.pairs + 1):
        pair_rows = {row["mode"]: row for row in rows if row["pair"] == pair}
        baseline = float(pair_rows["baseline"]["median_tps"])
        candidate = float(pair_rows["candidate"]["median_tps"])
        by_pair.append(
            {
                "pair": pair,
                "baseline_median_tps": baseline,
                "candidate_median_tps": candidate,
                "speedup": candidate / baseline,
            }
        )
        for mode in pooled:
            pooled[mode].extend(pair_rows[mode]["samples_tps"])

    baseline_pooled = median(pooled["baseline"])
    candidate_pooled = median(pooled["candidate"])
    pair_speedups = [float(row["speedup"]) for row in by_pair]
    speedup = candidate_pooled / baseline_pooled
    performance_gate = speedup >= args.min_speedup and min(pair_speedups) >= args.min_pair_speedup

    completions: list[dict[str, object]] = []
    outputs_exact = None
    if args.llama_completion is not None:
        for mode in measured_modes:
            completions.append(run_completion(args, mode, args.output_dir))
        outputs_exact = len({row["sha256"] for row in completions}) == 1

    serial_control = None
    serial_control_gate = True
    if args.include_serial_control:
        serial_pair_results = []
        for pair in range(1, args.pairs + 1):
            pair_rows = {row["mode"]: row for row in rows if row["pair"] == pair}
            serial = float(pair_rows["serial"]["median_tps"])
            candidate = float(pair_rows["candidate"]["median_tps"])
            serial_pair_results.append(
                {
                    "pair": pair,
                    "serial_median_tps": serial,
                    "candidate_median_tps": candidate,
                    "speedup": candidate / serial,
                }
            )
        serial_pooled = median(pooled["serial"])
        serial_pair_speedups = [float(row["speedup"]) for row in serial_pair_results]
        serial_control_speedup = candidate_pooled / serial_pooled
        serial_control_gate = (
            serial_control_speedup >= args.min_speedup
            and min(serial_pair_speedups) >= args.min_pair_speedup
        )
        serial_control = {
            "serial_pooled_median_tps": serial_pooled,
            "candidate_pooled_median_tps": candidate_pooled,
            "speedup": serial_control_speedup,
            "median_pair_speedup": median(serial_pair_speedups),
            "minimum_pair_speedup": min(serial_pair_speedups),
            "gate_pass": serial_control_gate,
            "pair_results": serial_pair_results,
        }

    summary = {
        "schema_version": 1,
        "status": (
            "pass"
            if performance_gate and serial_control_gate and outputs_exact is not False
            else "fail"
        ),
        "host_qualified": host_clean,
        "power": power,
        "load": {
            "one_minute": load_1m,
            "logical_cpus": logical_cpus,
            "per_cpu": load_per_cpu,
            "maximum_per_cpu": args.max_load_per_cpu,
        },
        "model": {
            "path": str(args.model),
            "sha256": sha256_file(args.model),
            "real_gguf": True,
        },
        "candidate": {
            "enabled": True,
            "tensor_roles": "down",
            "shared_q8": True,
            "sme_share_percent": 25,
            "parallel_correction": True,
        },
        "benchmark": {
            "threads": args.threads,
            "cpu_mask": args.cpu_mask,
            "n_gen": args.n_gen,
            "repetitions": args.repetitions,
            "pairs": args.pairs,
            "baseline_pooled_median_tps": baseline_pooled,
            "candidate_pooled_median_tps": candidate_pooled,
            "speedup": speedup,
            "median_pair_speedup": median(pair_speedups),
            "minimum_pair_speedup": min(pair_speedups),
            "gate_pass": performance_gate,
            "pair_results": by_pair,
            "runs": rows,
        },
        "serial_control": serial_control,
        "correctness": {
            "checked": outputs_exact is not None,
            "outputs_byte_identical": outputs_exact,
            "runs": completions,
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
