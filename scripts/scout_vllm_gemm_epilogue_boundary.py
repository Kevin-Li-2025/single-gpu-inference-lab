#!/usr/bin/env python3
"""Scout the vLLM LM-head GEMM epilogue boundary.

This is the step after the standalone FlashSampling candidate: identify the
smallest upstream-shaped place to attach a future sampled-token epilogue while
preserving vLLM's optimized LM-head/quantization path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


PATCH_POINTS = (
    {
        "id": "gpu_runner_logits_boundary",
        "path": "vllm/v1/worker/gpu/model_runner.py",
        "required": True,
        "role": "decode callsite where sample hidden states become logits before sampler dispatch",
        "groups": (
            ("sample_hidden_states = hidden_states[input_batch.logits_indices]",),
            (
                "logits = self.model.compute_logits(sample_hidden_states)",
                "maybe_l20_flashsampling_compute_logits_or_sample(",
            ),
            ("self.sampler(logits, input_batch)",),
        ),
    },
    {
        "id": "legacy_gpu_runner_logits_boundary",
        "path": "vllm/v1/worker/gpu_model_runner.py",
        "required": False,
        "role": "older v1 runner callsite kept for source-tree compatibility",
        "groups": (
            ("sample_hidden_states",),
            ("self.model.compute_logits", "maybe_l20_flashsampling_compute_logits_or_sample("),
            ("self._sample(logits", "self.sampler(logits"),
        ),
    },
    {
        "id": "logits_processor_lm_head",
        "path": "vllm/model_executor/layers/logits_processor.py",
        "required": True,
        "role": "optimized LM-head logits producer; a real epilogue must preserve this path",
        "groups": (
            ("class LogitsProcessor",),
            ("def _get_logits",),
            ("lm_head.quant_method.apply(lm_head, hidden_states",),
            ("logits = self._gather_logits(logits)",),
            ("def get_top_tokens",),
        ),
    },
    {
        "id": "parallel_lm_head_contract",
        "path": "vllm/model_executor/layers/vocab_parallel_embedding.py",
        "required": True,
        "role": "LM-head weight, padding, shard, bias, and quantization abstraction",
        "groups": (
            ("class VocabParallelEmbedding",),
            ("class ParallelLMHead",),
            ("self.quant_config = quant_config",),
            ("def tie_weights",),
        ),
    },
    {
        "id": "sampler_output_contract",
        "path": "vllm/v1/sample/sampler.py",
        "required": True,
        "role": "SamplerOutput/logprobs semantics that any epilogue output must match",
        "groups": (
            ("class Sampler",),
            ("logits = logits.to(torch.float32)",),
            ("sampled, processed_logprobs = self.sample(logits, sampling_metadata)",),
            ("sampler_output = SamplerOutput(",),
        ),
    },
    {
        "id": "topk_topp_sampler_contract",
        "path": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "required": True,
        "role": "current production top-k/top-p backend and FlashInfer fallback contract",
        "groups": (
            ("class TopKTopPSampler",),
            ("def forward_cuda", "def forward_native"),
            ("flashinfer_sample", "top_k_top_p_sampling_from_logits"),
        ),
    },
    {
        "id": "lora_logits_processor",
        "path": "vllm/lora/layers/logits_processor.py",
        "required": False,
        "role": "LoRA-aware logits path; first epilogue gate should fallback when LoRA is active",
        "groups": (
            ("class LogitsProcessorWithLoRA",),
            ("actual_lm_head.quant_method.apply",),
        ),
    },
)


FIRST_SAFE_GATE = (
    "CUDA L20 / SM89 only",
    "tensor parallel size 1 for the first implementation",
    "decode path only, one scheduled token per active request",
    "no prompt logprobs, token logprobs, or raw/processed logits return",
    "no grammar or structured-output bitmask",
    "no speculative decoding or rejection sampler",
    "no LoRA or per-request adapter path in the first implementation",
    "no per-request torch.Generator semantics until RNG state is plumbed",
    "fallback to compute_logits plus sampler for every unsupported request",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--tile-summary", type=Path)
    parser.add_argument("--serving-summary", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args(argv)


def run_git(source: Path, args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def source_metadata(source: Path) -> dict[str, Any]:
    status = run_git(source, ["status", "--short"])
    return {
        "path": str(source),
        "branch": run_git(source, ["branch", "--show-current"]),
        "commit": run_git(source, ["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
        "l20_local_patch_present": "l20_" in status if status else False,
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_group(lines: list[str], alternatives: tuple[str, ...]) -> dict[str, Any] | None:
    for needle in alternatives:
        for line_number, line in enumerate(lines, start=1):
            if needle in line:
                return {"needle": needle, "line": line_number, "text": line.strip()}
    return None


def scan_patch_points(source: Path) -> list[dict[str, Any]]:
    scanned = []
    for point in PATCH_POINTS:
        target = source / point["path"]
        lines = read_text(target).splitlines() if target.exists() else []
        matches = []
        for group in point["groups"]:
            match = find_group(lines, group)
            if match is not None:
                matches.append(match)
        complete = target.exists() and len(matches) == len(point["groups"])
        scanned.append(
            {
                "id": point["id"],
                "path": point["path"],
                "role": point["role"],
                "required": point["required"],
                "exists": target.exists(),
                "matched_groups": len(matches),
                "expected_groups": len(point["groups"]),
                "complete": complete,
                "required_complete": complete or not point["required"],
                "matches": matches,
            }
        )
    return scanned


def load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_evidence(tile_summary: dict[str, Any], serving_summary: dict[str, Any]) -> dict[str, Any]:
    best = tile_summary.get("best_by_shape", {})
    serving_metrics = serving_summary.get("metrics", {})
    return {
        "tile_policy_decision": tile_summary.get("decision", {}),
        "tile_best_keys": sorted(best),
        "serving_decision": serving_summary.get("decision"),
        "serving_reason": serving_summary.get("reason"),
        "serving_delta_pct": serving_metrics.get("delta_pct", {}),
    }


def build_upstream_api(points: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    complete = {point["id"] for point in points if point["complete"]}
    ready = {
        "gpu_runner_logits_boundary",
        "logits_processor_lm_head",
        "parallel_lm_head_contract",
        "sampler_output_contract",
    }.issubset(complete)
    return {
        "ready_for_trace_pr": ready,
        "not_ready_for_perf_claim": True,
        "proposed_callsite": "GPUModelRunner.sample before model.compute_logits",
        "proposed_owner": "LogitsProcessor / ParallelLMHead, not TopKTopPSampler",
        "proposed_api": (
            "try_sample_from_lm_head(lm_head, hidden_states, sampling_metadata, "
            "embedding_bias=None) -> SamplerOutput | None"
        ),
        "fallback_contract": "return None and run existing compute_logits plus sampler",
        "why_not_sampler_only": (
            "TopKTopPSampler receives materialized logits; it is too late to avoid "
            "LM-head output traffic."
        ),
        "why_not_standalone_kernel": (
            "Standalone candidate fixed tile policy but still lost serving throughput/TTFT."
        ),
        "dirty_source_warning": (
            "The scanned vLLM tree contains local L20 patches; use a clean upstream "
            "checkout before producing a PR diff."
            if source["dirty"]
            else None
        ),
    }


def build_plan(points: list[dict[str, Any]], evidence: dict[str, Any], source: dict[str, Any]) -> list[dict[str, Any]]:
    complete = {point["id"] for point in points if point["complete"]}
    return [
        {
            "priority": "P0",
            "step": "open a clean upstream-shaped trace PR around the LM-head callsite",
            "ready": {
                "gpu_runner_logits_boundary",
                "logits_processor_lm_head",
                "sampler_output_contract",
            }.issubset(complete),
            "details": (
                "Add an opt-in API that can return None before compute_logits; do not "
                "ship a new CUDA kernel in the first upstream patch."
            ),
        },
        {
            "priority": "P0",
            "step": "keep standalone FlashSampling disabled",
            "ready": evidence.get("serving_decision") == "do_not_claim_serving_win",
            "details": "The policy-v2 serving smoke is negative for throughput and TTFT.",
        },
        {
            "priority": "P1",
            "step": "prototype the GEMM epilogue behind LogitsProcessor",
            "ready": "logits_processor_lm_head" in complete
            and "parallel_lm_head_contract" in complete,
            "details": (
                "The first kernel should preserve lm_head.quant_method.apply or add a "
                "quant-method epilogue hook; do not duplicate the LM-head matmul."
            ),
        },
        {
            "priority": "Blocker",
            "step": "rescan a clean vLLM checkout before publishing an upstream diff",
            "ready": not source["dirty"],
            "details": "The current remote source has local L20 patches and backup files.",
        },
    ]


def analyze(
    source: Path,
    tile_summary_path: Path | None = None,
    serving_summary_path: Path | None = None,
) -> dict[str, Any]:
    points = scan_patch_points(source)
    source_info = source_metadata(source)
    evidence = extract_evidence(load_json(tile_summary_path), load_json(serving_summary_path))
    return {
        "schema_version": 1,
        "complete": all(point["required_complete"] for point in points),
        "source": source_info,
        "patch_points": points,
        "first_safe_gate": list(FIRST_SAFE_GATE),
        "evidence": evidence,
        "upstream_api": build_upstream_api(points, source_info),
        "implementation_plan": build_plan(points, evidence, source_info),
    }


def render_markdown(result: dict[str, Any]) -> str:
    source = result["source"]
    api = result["upstream_api"]
    lines = [
        "# vLLM LM-Head GEMM Epilogue Boundary Scout",
        "",
        "This artifact is the handoff from standalone FlashSampling to a true "
        "LM-head GEMM epilogue integration.",
        "",
        "## Source",
        "",
        f"- Path: `{source['path']}`",
        f"- Branch: `{source.get('branch')}`",
        f"- Commit: `{source.get('commit')}`",
        f"- Dirty: `{source['dirty']}` ({source['status_line_count']} status lines)",
        f"- Local L20 patch present: `{source['l20_local_patch_present']}`",
        "",
        "## Patch Points",
        "",
        "| ID | Path | Required | Matched | Role |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for point in result["patch_points"]:
        lines.append(
            f"| `{point['id']}` | `{point['path']}` | `{point['required']}` | "
            f"{point['matched_groups']}/{point['expected_groups']} | {point['role']} |"
        )

    lines.extend(
        [
            "",
            "## Proposed Upstream API",
            "",
            f"- Ready for trace PR: `{api['ready_for_trace_pr']}`",
            f"- Owner: `{api['proposed_owner']}`",
            f"- Callsite: `{api['proposed_callsite']}`",
            f"- API: `{api['proposed_api']}`",
            f"- Fallback: {api['fallback_contract']}",
            f"- Why not sampler-only: {api['why_not_sampler_only']}",
            f"- Why not standalone kernel: {api['why_not_standalone_kernel']}",
        ]
    )
    if api.get("dirty_source_warning"):
        lines.append(f"- Warning: {api['dirty_source_warning']}")

    evidence = result.get("evidence", {})
    lines.extend(["", "## Evidence", ""])
    if evidence.get("tile_policy_decision"):
        decision = evidence["tile_policy_decision"]
        lines.append(f"- Tile policy decision: `{json.dumps(decision, sort_keys=True)}`")
    if evidence.get("serving_decision"):
        lines.append(f"- Serving decision: `{evidence['serving_decision']}`")
        lines.append(f"- Serving reason: {evidence.get('serving_reason')}")
    if evidence.get("serving_delta_pct"):
        lines.extend(["", "| Metric | Delta |", "| --- | ---: |"])
        for name, value in sorted(evidence["serving_delta_pct"].items()):
            lines.append(f"| `{name}` | {float(value):.2f}% |")

    lines.extend(["", "## First Safe Gate", ""])
    lines.extend(f"- {item}" for item in result["first_safe_gate"])

    lines.extend(["", "## Implementation Plan", "", "| Priority | Ready | Step |", "| --- | ---: | --- |"])
    for item in result["implementation_plan"]:
        lines.append(f"| {item['priority']} | `{item['ready']}` | {item['step']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze(args.vllm_source, args.tile_summary, args.serving_summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
