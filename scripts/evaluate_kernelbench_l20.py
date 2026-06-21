#!/usr/bin/env python3
"""Evaluate generated kernels with the official KernelBench execution core."""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernelbench-root", type=Path, required=True)
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--correctness-trials", type=int, default=5)
    parser.add_argument("--performance-trials", type=int, default=100)
    parser.add_argument("--chunked-allclose-elements", type=int, default=0)
    return parser.parse_args()


def install_chunked_allclose(torch, chunk_elements):
    original = torch.allclose

    def chunked(actual, expected, rtol=1e-5, atol=1e-8, equal_nan=False):
        if actual.numel() <= chunk_elements:
            return original(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
        actual_flat = actual.reshape(-1)
        expected_flat = expected.reshape(-1)
        for start in range(0, actual.numel(), chunk_elements):
            end = min(start + chunk_elements, actual.numel())
            if not original(
                actual_flat[start:end],
                expected_flat[start:end],
                rtol=rtol,
                atol=atol,
                equal_nan=equal_nan,
            ):
                return False
        return True

    torch.allclose = chunked


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.kernelbench_root / "src"))
    # KernelBench's utility module imports its optional API client at module load.
    # Local execution never calls it, so avoid pulling a serving dependency into
    # the isolated training environment.
    if "litellm" not in sys.modules:
        litellm_stub = types.ModuleType("litellm")

        def unavailable_completion(*_args, **_kwargs):
            raise RuntimeError("litellm is unavailable in local evaluation mode")

        litellm_stub.completion = unavailable_completion
        sys.modules["litellm"] = litellm_stub
    import torch
    from l20_stack.kernel_checks import validate_kernelbench_interface
    from kernelbench import eval as kernel_eval
    from kernelbench.kernel_static_checker import validate_kernel_static
    from kernelbench.utils import set_gpu_arch

    if torch.cuda.get_device_name() != "NVIDIA L20" or torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("evaluation requires NVIDIA L20 sm_89")
    if args.chunked_allclose_elements < 0:
        raise SystemExit("chunked-allclose-elements cannot be negative")
    if args.chunked_allclose_elements:
        install_chunked_allclose(torch, args.chunked_allclose_elements)
    set_gpu_arch(["Ada"])
    manifest = json.loads(
        (args.generations / "generation_manifest.json").read_text(encoding="utf-8")
    )
    results = []
    for task in manifest["tasks"]:
        reference = Path(task["reference"]).read_text(encoding="utf-8")
        candidate = Path(task["candidate"]).read_text(encoding="utf-8")
        interface = validate_kernelbench_interface(candidate)
        static = validate_kernel_static(candidate, backend="triton")
        if not interface.valid:
            payload = {
                "compiled": False,
                "correctness": False,
                "runtime": -1.0,
                "ref_runtime": -1.0,
                "metadata": {"interface_errors": interface.errors},
            }
        else:
            try:
                result = kernel_eval.eval_kernel_against_ref(
                    original_model_src=reference,
                    custom_model_src=candidate,
                    measure_performance=True,
                    timing_method="cuda_event",
                    verbose=False,
                    num_correct_trials=args.correctness_trials,
                    num_perf_trials=args.performance_trials,
                    device=torch.device("cuda:0"),
                    backend="triton",
                    precision=torch.float32,
                )
                payload = json.loads(json.dumps(result.model_dump(), default=str))
            except Exception as error:
                payload = {
                    "compiled": False,
                    "correctness": False,
                    "runtime": -1.0,
                    "ref_runtime": -1.0,
                    "metadata": {"exception": f"{type(error).__name__}: {error}"},
                }
        payload.update(
            {
                "level": task["level"],
                "problem_id": task["problem_id"],
                "interface_check": interface.to_dict(),
                "static_check": static,
            }
        )
        if payload.get("correctness") and payload.get("runtime", -1) > 0:
            payload["speedup"] = payload["ref_runtime"] / payload["runtime"]
        results.append(payload)
        print(json.dumps(payload, sort_keys=True, default=str), flush=True)
        torch.cuda.empty_cache()

    total = len(results)
    summary = {
        "tasks": total,
        "compile_rate": sum(bool(row.get("compiled")) for row in results) / total,
        "fast_0": sum(bool(row.get("correctness")) for row in results) / total,
        "fast_1": sum(row.get("speedup", 0) > 1 for row in results) / total,
        "fast_2": sum(row.get("speedup", 0) > 2 for row in results) / total,
    }
    report = {
        "schema_version": 1,
        "hardware": {"gpu": torch.cuda.get_device_name(), "compute_capability": "8.9"},
        "correctness_comparator": {
            "implementation": (
                "chunked_torch_allclose"
                if args.chunked_allclose_elements
                else "torch_allclose"
            ),
            "chunk_elements": args.chunked_allclose_elements or None,
            "note": (
                "Same torch.allclose tolerance and elementwise semantics; chunking "
                "reduces comparator peak memory but is not the default KernelBench path."
                if args.chunked_allclose_elements
                else None
            ),
        },
        "generation": manifest,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
