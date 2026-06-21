#!/usr/bin/env python3
"""Build a problem-disjoint kernel SFT dataset from executable sample records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from generate_kernelbench import build_prompt, find_problem


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--kernelbench-root", type=Path, required=True)
    parser.add_argument("--holdout-suite", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--triton-only", action="store_true")
    return parser.parse_args()


def problem_bucket(level: int, problem_id: int) -> int:
    digest = hashlib.sha256(f"{level}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def load_candidates(samples_root: Path):
    for path in sorted(samples_root.rglob("kernel.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["_source_path"] = str(path)
        yield payload


def prepare_records(samples_root, kernelbench_root, holdout, eval_fraction, triton_only):
    holdout_keys = {(int(item["level"]), int(item["problem_id"])) for item in holdout["tasks"]}
    records = {"train": [], "eval": []}
    seen_kernels = set()
    counts = {
        "scanned": 0,
        "incorrect": 0,
        "holdout": 0,
        "non_triton": 0,
        "duplicate": 0,
        "missing_reference": 0,
    }
    threshold = int(eval_fraction * 10_000)
    for sample in load_candidates(samples_root):
        counts["scanned"] += 1
        if not sample.get("correct"):
            counts["incorrect"] += 1
            continue
        level, problem_id = int(sample["level"]), int(sample["problem_id"])
        if (level, problem_id) in holdout_keys:
            counts["holdout"] += 1
            continue
        kernel = sample.get("kernel", "")
        if triton_only and not ("import triton" in kernel or "from triton" in kernel):
            counts["non_triton"] += 1
            continue
        kernel_hash = hashlib.sha256(kernel.encode()).hexdigest()
        if kernel_hash in seen_kernels:
            counts["duplicate"] += 1
            continue
        try:
            reference_path = find_problem(kernelbench_root, level, problem_id)
        except ValueError:
            counts["missing_reference"] += 1
            continue
        seen_kernels.add(kernel_hash)
        reference = reference_path.read_text(encoding="utf-8")
        split = "eval" if problem_bucket(level, problem_id) < threshold else "train"
        records[split].append(
            {
                "messages": [
                    {"role": "system", "content": "Generate correct, fast Triton kernels for NVIDIA L20."},
                    {"role": "user", "content": build_prompt(reference)},
                    {"role": "assistant", "content": kernel},
                ],
                "metadata": {
                    "level": level,
                    "problem_id": problem_id,
                    "source_model": sample.get("model_name"),
                    "source_hardware": sample.get("hardware"),
                    "kernel_sha256": kernel_hash,
                    "source_path": sample["_source_path"],
                },
            }
        )
    counts["train"] = len(records["train"])
    counts["eval"] = len(records["eval"])
    return records, counts


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    if not 0 < args.eval_fraction < 1:
        raise SystemExit("eval-fraction must be between zero and one")
    holdout = json.loads(args.holdout_suite.read_text(encoding="utf-8"))
    records, counts = prepare_records(
        args.samples_root,
        args.kernelbench_root,
        holdout,
        args.eval_fraction,
        args.triton_only,
    )
    if not records["train"] or not records["eval"]:
        raise SystemExit("dataset preparation produced an empty train or eval split")
    write_jsonl(args.train_output, records["train"])
    write_jsonl(args.eval_output, records["eval"])
    train_problems = {(r["metadata"]["level"], r["metadata"]["problem_id"]) for r in records["train"]}
    eval_problems = {(r["metadata"]["level"], r["metadata"]["problem_id"]) for r in records["eval"]}
    if train_problems.intersection(eval_problems):
        raise RuntimeError("problem-disjoint split invariant failed")
    manifest = {
        "schema_version": 1,
        "counts": counts,
        "triton_only": args.triton_only,
        "eval_fraction": args.eval_fraction,
        "holdout_suite": holdout.get("suite"),
        "train_problem_count": len(train_problems),
        "eval_problem_count": len(eval_problems),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
