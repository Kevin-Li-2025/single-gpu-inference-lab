"""Command line interface for Single-GPU Inference Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from l20_stack.artifacts import inspect_artifact_index
from l20_stack.experiment import ExperimentConfig
from l20_stack.memory import estimate_training_memory
from l20_stack.operators import OperatorTarget, l20_operator_summary, plan_operators


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="single-gpu-infer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="estimate memory for an experiment config")
    plan.add_argument("--config", required=True, help="path to a JSON experiment config")

    operator_plan = subparsers.add_parser(
        "operator-plan", help="rank L20 operator optimization targets"
    )
    operator_plan.add_argument("--config", required=True, help="path to operator target JSON")

    artifact_index = subparsers.add_parser(
        "artifact-index", help="validate the checked-in benchmark artifact index"
    )
    artifact_index.add_argument(
        "--index",
        default="benchmarks/results/README.md",
        help="path to the benchmark result index README",
    )
    artifact_index.add_argument(
        "--result-root",
        default="benchmarks/results",
        help="directory that contains checked-in benchmark result artifacts",
    )
    artifact_index.add_argument(
        "--strict-warnings",
        action="store_true",
        help="return a non-zero exit code when artifact warnings are present",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan":
        config = ExperimentConfig.from_file(args.config)
        estimate = estimate_training_memory(config.model, config.training)
        print(
            json.dumps(
                {
                    "task": config.task,
                    "dataset": config.dataset,
                    "output_dir": config.output_dir,
                    "estimate": estimate.to_dict(),
                    "note": (
                        "Planning estimate only; validate with real CUDA telemetry "
                        "before making performance claims."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "operator-plan":
        payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
        targets = [OperatorTarget.from_dict(item) for item in payload["operators"]]
        plans = plan_operators(targets)
        print(
            json.dumps(
                {
                    "workload": payload.get("workload", "unknown"),
                    "summary": l20_operator_summary(),
                    "plans": [plan.to_dict() for plan in plans],
                    "note": (
                        "This is an optimization plan, not a measured result. "
                        "Run the v2 RMSNorm benchmark on real L20 hardware before claiming speedup."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "artifact-index":
        report = inspect_artifact_index(args.index, result_root=args.result_root)
        print(report.to_json())
        if report.errors or (args.strict_warnings and report.warnings):
            return 1
        return 0

    parser.error("unknown command: " + str(args.command))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
