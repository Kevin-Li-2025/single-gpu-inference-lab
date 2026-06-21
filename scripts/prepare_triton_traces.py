#!/usr/bin/env python3
"""Prepare problem-disjoint Triton SFT records with holdout source filtering."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from generate_kernelbench import build_prompt, find_problem
from l20_stack.kernel_checks import validate_kernelbench_interface
from prepare_kernelbench_sft import write_jsonl


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--kernelbench-root", type=Path, required=True)
    parser.add_argument("--holdout-suite", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--eval-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--eval-fraction", type=float, default=0.15)
    parser.add_argument("--require-fast-1", action="store_true")
    return parser.parse_args()


def source_fingerprint(source: str) -> str:
    normalized = re.sub(r"\s+", "", source).lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


def ensure_model_new(kernel: str, pytorch_source: str = "") -> str:
    if "class ModelNew" in kernel:
        return kernel
    if "def triton_kernel_wrapper" not in kernel:
        raise ValueError("kernel defines neither ModelNew nor triton_kernel_wrapper")
    adapter = build_modelnew_adapter(kernel, pytorch_source)
    if adapter:
        return kernel.rstrip() + "\n\n\n" + adapter.rstrip() + "\n"
    raise ValueError("could not build ModelNew adapter")


def build_modelnew_adapter(kernel: str, pytorch_source: str) -> str:
    """Build a KernelBench-shaped ModelNew wrapper from the reference Model."""
    if not pytorch_source:
        return ""
    try:
        kernel_tree = ast.parse(kernel)
        reference_tree = ast.parse(pytorch_source)
    except SyntaxError:
        return ""
    wrapper = next(
        (
            node
            for node in kernel_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "triton_kernel_wrapper"
        ),
        None,
    )
    reference_model = _find_reference_model(reference_tree)
    if wrapper is None or reference_model is None:
        return ""
    init_fn = _find_method(reference_model, "__init__")
    forward_fn = _find_method(reference_model, "forward")
    if init_fn is None or forward_fn is None:
        return ""

    init_source = _rewrite_super_model(_unparse_function(init_fn), reference_model.name)
    forward_args = _argument_names(forward_fn, drop_self=True)
    wrapper_args = _argument_names(wrapper, drop_self=False)
    self_attrs = _assigned_self_attributes(init_fn)

    call_args = []
    for name in wrapper_args:
        if name in forward_args:
            call_args.append(name)
        elif name in self_attrs:
            call_args.append(f"self.{name}")
        elif name in {"w", "weight"} and "linear" in self_attrs:
            call_args.append("self.linear.weight")
        elif name in {"b", "bias"} and "linear" in self_attrs:
            call_args.append("self.linear.bias")
        elif name == "scaling_factor" and "scaling_factor" in self_attrs:
            call_args.append("self.scaling_factor")
        else:
            return ""
    forward_signature = _signature_source(forward_fn)
    return (
        "class ModelNew(torch.nn.Module):\n"
        + _indent(init_source)
        + "\n\n"
        + f"    def forward({forward_signature}):\n"
        + f"        return triton_kernel_wrapper({', '.join(call_args)})\n"
    )


def _find_method(class_node: ast.ClassDef, name: str) -> Optional[ast.FunctionDef]:
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == name:
            return item
    return None


def _find_reference_model(tree: ast.Module) -> Optional[ast.ClassDef]:
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    for node in classes:
        if node.name == "Model" and _find_method(node, "forward") is not None:
            return node
    candidates = [node for node in classes if _find_method(node, "forward") is not None]
    if len(candidates) == 1:
        return candidates[0]
    module_like = [node for node in candidates if _inherits_module(node)]
    if len(module_like) == 1:
        return module_like[0]
    return None


def _inherits_module(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Attribute) and base.attr == "Module":
            return True
        if isinstance(base, ast.Name) and base.id in {"Module", "nn.Module"}:
            return True
    return False


def _unparse_function(node: ast.FunctionDef) -> str:
    return ast.unparse(node)


def _signature_source(node: ast.FunctionDef) -> str:
    args = ast.arguments(
        posonlyargs=node.args.posonlyargs,
        args=node.args.args,
        vararg=node.args.vararg,
        kwonlyargs=node.args.kwonlyargs,
        kw_defaults=node.args.kw_defaults,
        kwarg=node.args.kwarg,
        defaults=node.args.defaults,
    )
    return ast.unparse(args)


def _argument_names(node: ast.FunctionDef, drop_self: bool) -> list[str]:
    args = list(node.args.posonlyargs) + list(node.args.args)
    if drop_self and args and args[0].arg == "self":
        args = args[1:]
    if node.args.vararg is not None or node.args.kwarg is not None:
        return []
    return [arg.arg for arg in args]


def _assigned_self_attributes(node: ast.FunctionDef) -> set[str]:
    attrs = set()
    for item in ast.walk(node):
        if not isinstance(item, ast.Attribute):
            continue
        if isinstance(item.ctx, ast.Store) and isinstance(item.value, ast.Name) and item.value.id == "self":
            attrs.add(item.attr)
    return attrs


def _indent(source: str) -> str:
    return "\n".join(f"    {line}" if line else line for line in source.splitlines())


def _rewrite_super_model(source: str, class_name: str) -> str:
    pattern = rf"super\(\s*{re.escape(class_name)}\s*,\s*self\s*\)"
    return re.sub(pattern, "super(ModelNew, self)", source)


def build_records(rows, holdout_sources, holdout_keys, eval_fraction, require_fast_1):
    threshold = int(eval_fraction * 10_000)
    records = {"train": [], "eval": []}
    counts = {
        "scanned": 0,
        "incorrect": 0,
        "not_fast_1": 0,
        "holdout": 0,
        "duplicate": 0,
        "interface_invalid": 0,
    }
    seen = set()
    for row in rows:
        counts["scanned"] += 1
        if not row.get("result_correctness"):
            counts["incorrect"] += 1
            continue
        if require_fast_1 and not row.get("result_fast_1"):
            counts["not_fast_1"] += 1
            continue
        source = row.get("source") or "unknown"
        level = int(row.get("level") or 0)
        problem_id = int(row.get("problem_id") or 0)
        pytorch_code = row.get("pytorch_code") or ""
        triton_code = row.get("triton_code") or ""
        source_hash = source_fingerprint(pytorch_code)
        if (source == "kernelbench" and (level, problem_id) in holdout_keys) or source_hash in holdout_sources:
            counts["holdout"] += 1
            continue
        try:
            triton_code = ensure_model_new(triton_code, pytorch_code)
        except ValueError:
            counts["duplicate"] += 1
            continue
        interface_report = validate_kernelbench_interface(triton_code)
        if not interface_report.valid:
            counts["interface_invalid"] += 1
            continue
        kernel_hash = hashlib.sha256(triton_code.encode()).hexdigest()
        if not pytorch_code or not triton_code or kernel_hash in seen:
            counts["duplicate"] += 1
            continue
        seen.add(kernel_hash)
        split = "eval" if int(source_hash[:16], 16) % 10_000 < threshold else "train"
        records[split].append(
            {
                "messages": [
                    {"role": "system", "content": "Generate correct, fast Triton kernels for NVIDIA L20."},
                    {"role": "user", "content": build_prompt(pytorch_code)},
                    {"role": "assistant", "content": triton_code},
                ],
                "metadata": {
                    "source": source,
                    "level": level,
                    "problem_id": problem_id,
                    "source_sha256": source_hash,
                    "kernel_sha256": kernel_hash,
                    "family_sha256": source_hash,
                    "reported_speedup": row.get("result_speedup"),
                },
            }
        )
    counts["train"] = len(records["train"])
    counts["eval"] = len(records["eval"])
    return records, counts


def main() -> int:
    args = parse_args()
    from datasets import Dataset

    holdout = json.loads(args.holdout_suite.read_text(encoding="utf-8"))
    holdout_keys = {(int(item["level"]), int(item["problem_id"])) for item in holdout["tasks"]}
    holdout_sources = set()
    for level, problem_id in holdout_keys:
        source = find_problem(args.kernelbench_root, level, problem_id).read_text(encoding="utf-8")
        holdout_sources.add(source_fingerprint(source))
    dataset = Dataset.from_parquet(args.parquet)
    records, counts = build_records(
        dataset,
        holdout_sources,
        holdout_keys,
        args.eval_fraction,
        args.require_fast_1,
    )
    if not records["train"] or not records["eval"]:
        raise SystemExit("empty train or eval split")
    train_families = {r["metadata"]["family_sha256"] for r in records["train"]}
    eval_families = {r["metadata"]["family_sha256"] for r in records["eval"]}
    if train_families.intersection(eval_families):
        raise RuntimeError("train/eval family overlap")
    write_jsonl(args.train_output, records["train"])
    write_jsonl(args.eval_output, records["eval"])
    manifest = {
        "schema_version": 1,
        "counts": counts,
        "require_fast_1": args.require_fast_1,
        "holdout_suite": holdout["suite"],
        "holdout_source_fingerprints": len(holdout_sources),
        "train_families": len(train_families),
        "eval_families": len(eval_families),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
