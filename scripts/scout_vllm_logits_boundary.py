#!/usr/bin/env python3
"""Scout vLLM logits/sampling patch points for an upstreamable boundary."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PATCH_POINTS = (
    {
        "id": "gpu_model_runner_logits_to_sampler",
        "path": "vllm/v1/worker/gpu/model_runner.py",
        "needles": (
            "sample_hidden_states = hidden_states[input_batch.logits_indices]",
            "logits = self.model.compute_logits(sample_hidden_states)",
            "sampler_output = self.sampler(logits, input_batch)",
        ),
        "role": "common decode boundary: hidden states become full logits, then sampler consumes logits",
    },
    {
        "id": "logits_processor_lm_head",
        "path": "vllm/model_executor/layers/logits_processor.py",
        "needles": (
            "class LogitsProcessor",
            "logits = lm_head.quant_method.apply(lm_head, hidden_states",
            "logits = self._gather_logits(logits)",
            "def get_top_tokens(",
        ),
        "role": "LM-head logits production and existing greedy-only local top-token shortcut",
    },
    {
        "id": "v1_gpu_sampler_full_logits",
        "path": "vllm/v1/worker/gpu/sample/sampler.py",
        "needles": (
            "logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)",
            "self.sampling_states.apply_temperature(",
            "self.sampling_states.apply_top_k_top_p(",
            "sampled = gumbel_sample(",
        ),
        "role": "newer GPU sampler path copies and mutates full logits before sampling",
    },
    {
        "id": "legacy_v1_sampler_full_logits",
        "path": "vllm/v1/sample/sampler.py",
        "needles": (
            "logits = logits.to(torch.float32)",
            "greedy_sampled = self.greedy_sample(logits)",
            "logits = self.apply_temperature(",
            "sampled, processed_logprobs = self.sample(logits, sampling_metadata)",
        ),
        "role": "legacy/common sampler path treats full logits as the required interface",
    },
    {
        "id": "topk_topp_backend_contract",
        "path": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "needles": (
            "class TopKTopPSampler",
            "Using FlashInfer for top-p & top-k sampling.",
            "return flashinfer_sample(logits.contiguous(), k, p, generators), None",
        ),
        "role": "FlashInfer sampler consumes contiguous full logits after LM-head materialization",
    },
    {
        "id": "parallel_lm_head_weight",
        "path": "vllm/model_executor/layers/vocab_parallel_embedding.py",
        "needles": (
            "class VocabParallelEmbedding(PluggableLayer):",
            "self.quant_method: QuantizeMethodBase = quant_method",
            "class ParallelLMHead(VocabParallelEmbedding)",
            "LMHead's weights should be used in the sampler.",
        ),
        "role": "LM-head weight and quantization abstraction that an epilogue hook must preserve",
    },
)


FIRST_GATE = (
    "SM89/L20 opt-in flag only",
    "tensor parallel size 1 before adding distributed reduction",
    "decode path only; no prompt logprobs",
    "no requested token logprobs and no raw/processed logits return",
    "no structured-output grammar bitmask",
    "no speculative rejection sampling",
    "no per-request generators",
    "simple top-k/top-p/temperature sampling before penalties/logit-bias support",
    "fallback to existing compute_logits plus sampler for every unsupported request",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--ceiling-summary", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


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


def source_metadata(source: Path) -> dict:
    status = run_git(source, ["status", "--short"])
    return {
        "path": str(source),
        "commit": run_git(source, ["rev-parse", "--short", "HEAD"]),
        "branch": run_git(source, ["branch", "--show-current"]),
        "dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()


def find_needles(file_path: Path, needles: tuple[str, ...]) -> list[dict]:
    if not file_path.exists():
        return []
    lines = read_lines(file_path)
    matches = []
    for needle in needles:
        for index, line in enumerate(lines, start=1):
            if needle in line:
                matches.append(
                    {
                        "needle": needle,
                        "line": index,
                        "text": line.strip(),
                    }
                )
                break
    return matches


def scan_patch_points(source: Path) -> list[dict]:
    points = []
    for point in PATCH_POINTS:
        target = source / point["path"]
        matches = find_needles(target, point["needles"])
        points.append(
            {
                "id": point["id"],
                "path": point["path"],
                "role": point["role"],
                "exists": target.exists(),
                "matched_needles": len(matches),
                "expected_needles": len(point["needles"]),
                "complete": len(matches) == len(point["needles"]),
                "matches": matches,
            }
        )
    return points


def load_ceiling(path: Path | None) -> dict | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ceiling_highlights(ceiling: dict | None) -> dict:
    if ceiling is None:
        return {}
    recommendations = ceiling.get("recommendations", [])
    p0 = [row for row in recommendations if row.get("priority") == "P0"]
    stop = [row for row in recommendations if row.get("priority") == "Stop"]
    return {
        "p0_targets": p0,
        "stop_targets": stop,
        "recommendation_count": len(recommendations),
    }


def build_plan(points: list[dict], ceiling: dict | None) -> list[dict]:
    complete_ids = {point["id"] for point in points if point["complete"]}
    plan = [
        {
            "priority": "P0",
            "step": "add a guarded logits-boundary API before writing a kernel",
            "details": (
                "Introduce an opt-in path around GPUModelRunner.sample that can "
                "ask the model/logits processor for sampled-token state directly "
                "and fall back to compute_logits plus sampler."
            ),
            "requires": ["gpu_model_runner_logits_to_sampler", "logits_processor_lm_head"],
        },
        {
            "priority": "P0",
            "step": "keep the first gate narrow",
            "details": (
                "Start with decode-only, TP=1, no logprobs, no grammar, no spec "
                "decode, no per-request generators, and simple top-k/top-p."
            ),
            "requires": [],
        },
        {
            "priority": "P1",
            "step": "prototype a trace-only vLLM patch",
            "details": (
                "Before CUTLASS work, land a patch that records when the safe "
                "gate would have fired and emits shapes/sampling params into a "
                "JSONL trace. This gives an upstreamable review surface."
            ),
            "requires": ["gpu_model_runner_logits_to_sampler"],
        },
        {
            "priority": "Stop",
            "step": "do not replace the full LM-head GEMM with standalone Triton",
            "details": (
                "Existing L20 boundary measurements show standalone chunked/top1 "
                "paths lose to full logits; the epilogue must preserve production "
                "GEMM/GEMV."
            ),
            "requires": [],
        },
    ]
    for item in plan:
        item["ready"] = all(req in complete_ids for req in item["requires"])
    if ceiling:
        item = {
            "priority": "Evidence",
            "step": "ceiling report is attached",
            "details": "Use the included Amdahl report as the numeric justification.",
            "requires": [],
            "ready": True,
        }
        plan.insert(0, item)
    return plan


def analyze(source: Path, ceiling_path: Path | None) -> dict:
    ceiling = load_ceiling(ceiling_path)
    points = scan_patch_points(source)
    complete = all(point["complete"] for point in points)
    return {
        "schema_version": 1,
        "complete": complete,
        "source": source_metadata(source),
        "patch_points": points,
        "first_gate": list(FIRST_GATE),
        "ceiling": ceiling_highlights(ceiling),
        "implementation_plan": build_plan(points, ceiling),
    }


def render_markdown(result: dict) -> str:
    source = result["source"]
    lines = [
        "# vLLM Logits Boundary Scout",
        "",
        "This is a static upstream-scouting artifact for the L20 serving P0 target: "
        "a production GEMM/GEMV epilogue or logits boundary.",
        "",
        "## Source",
        "",
        f"- Path: `{source['path']}`",
        f"- Branch: `{source.get('branch')}`",
        f"- Commit: `{source.get('commit')}`",
        f"- Dirty: `{source['dirty']}` ({source['status_line_count']} status lines)",
        f"- Complete: `{result['complete']}`",
        "",
        "## Patch Points",
        "",
        "| ID | File | Matches | Role |",
        "| --- | --- | ---: | --- |",
    ]
    for point in result["patch_points"]:
        lines.append(
            f"| `{point['id']}` | `{point['path']}` | "
            f"{point['matched_needles']}/{point['expected_needles']} | "
            f"{point['role']} |"
        )
    lines.extend(["", "## First Safe Gate", ""])
    for gate in result["first_gate"]:
        lines.append(f"- {gate}")
    lines.extend(
        [
            "",
            "## Implementation Plan",
            "",
            "| Priority | Ready | Step | Details |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in result["implementation_plan"]:
        ready = "yes" if item["ready"] else "no"
        lines.append(
            f"| `{item['priority']}` | {ready} | {item['step']} | {item['details']} |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append(
        "The source checkout may be dirty. Treat line matches as local static evidence, "
        "not as proof that a clean upstream PR already applies."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    result = analyze(args.vllm_source.expanduser().resolve(), args.ceiling_summary)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(serialized + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
