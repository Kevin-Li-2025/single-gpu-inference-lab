#!/usr/bin/env python3
"""Scout vLLM compiled-sampler/logits-epilogue integration boundaries.

The standalone L20 top-k/top-p hook loses real serving because it adds Python
gate work, random-uniform generation, and uncaptured Triton launches. This
script records the concrete source boundaries that must change before another
sampler kernel is worth benchmarking.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


PATCH_POINTS = (
    {
        "id": "gpu_runner_logits_sampler_boundary",
        "path": "vllm/v1/worker/gpu_model_runner.py",
        "needles": (
            "sample_hidden_states",
            "self.model.compute_logits",
            "self.sampler(",
        ),
        "role": "primary v1 boundary where logits are materialized before sampling",
    },
    {
        "id": "legacy_gpu_runner_logits_sampler_boundary",
        "path": "vllm/v1/worker/gpu/model_runner.py",
        "needles": (
            "sample_hidden_states = hidden_states[input_batch.logits_indices]",
            "logits = self.model.compute_logits(sample_hidden_states)",
            "sampler_output = self.sampler(logits, input_batch)",
        ),
        "role": "older GPU runner boundary kept for compatibility across vLLM checkouts",
        "optional": True,
    },
    {
        "id": "worker_gpu_sampler_flashinfer_branch",
        "path": "vllm/v1/worker/gpu/sample/sampler.py",
        "needles": (
            "processed_logits = self.apply_sampling_params(",
            "top_k, top_p = self.sampling_states.get_top_k_top_p(",
            "sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)",
            "sampled = gumbel_sample(",
        ),
        "role": "active GPU sampler path with FlashInfer and native fallback branches",
    },
    {
        "id": "active_v1_sampler_topk_topp_call",
        "path": "vllm/v1/sample/sampler.py",
        "needles": (
            "self.topk_topp_sampler(",
            "sampling_metadata.generators",
            "sampling_metadata.top_k",
            "sampling_metadata.top_p",
        ),
        "role": "current v1 sampler call path used by serving before TopKTopPSampler",
    },
    {
        "id": "active_v1_topk_topp_forward_cuda",
        "path": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "needles": (
            "def forward_cuda(",
            "return flashinfer_sample(logits.contiguous(), k, p, generators), None",
            "def flashinfer_sample(",
        ),
        "role": "current CUDA sampler path; it does not receive request position or seed tensors",
    },
    {
        "id": "active_v1_sampling_metadata_contract",
        "path": "vllm/v1/sample/metadata.py",
        "needles": (
            "class SamplingMetadata",
            "top_p: torch.Tensor | None",
            "top_k: torch.Tensor | None",
            "generators: dict[int, torch.Generator]",
        ),
        "role": "active sampler metadata contract; seed and position tensors are the missing extension",
    },
    {
        "id": "worker_gpu_sampler_full_logits_copy",
        "path": "vllm/v1/worker/gpu/sample/sampler.py",
        "needles": (
            "logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)",
            "self.sampling_states.apply_temperature(",
            "self.sampling_states.apply_min_p(",
        ),
        "role": "full-logits FP32 copy and mutation before top-k/top-p sampling",
    },
    {
        "id": "sampling_state_cpu_gpu_split",
        "path": "vllm/v1/worker/gpu/sample/states.py",
        "needles": (
            "np.any(self.top_k.np[idx_mapping_np]",
            "top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None",
            "def any_explicit_seed(",
        ),
        "role": "CPU-side gate plus GPU tensors for top-k/top-p/seed state",
    },
    {
        "id": "gumbel_rng_state_kernel",
        "path": "vllm/v1/worker/gpu/sample/gumbel.py",
        "needles": (
            "def gumbel_sample(",
            "seed: torch.Tensor",
            "_gumbel_sample_kernel[",
            "tl.randint(seed, pos)",
        ),
        "role": "existing graph-safe RNG/seed path that a custom sampler must reuse",
    },
    {
        "id": "flashinfer_sampler_contract",
        "path": "vllm/v1/sample/ops/topk_topp_sampler.py",
        "needles": (
            "def flashinfer_sample(",
            "top_k_top_p_sampling_from_logits",
            "deterministic=True",
        ),
        "role": "current fused sampler contract and seed/offset-compatible baseline",
    },
    {
        "id": "logits_processor_lm_head",
        "path": "vllm/model_executor/layers/logits_processor.py",
        "needles": (
            "class LogitsProcessor",
            "logits = lm_head.quant_method.apply(lm_head, hidden_states",
            "logits = self._gather_logits(logits)",
        ),
        "role": "LM-head/logits producer boundary; epilogue must preserve this optimized path",
    },
)


def detect_rng_metadata_gap(source: Path) -> dict:
    metadata = read_text(source / "vllm/v1/sample/metadata.py")
    sampler = read_text(source / "vllm/v1/sample/sampler.py")
    ops = read_text(source / "vllm/v1/sample/ops/topk_topp_sampler.py")
    metadata_has_seed_tensor = "seed" in metadata.lower()
    metadata_has_position_tensor = "position" in metadata.lower()
    sampler_passes_rng_state = (
        "sampling_metadata.seeds" in sampler
        or "sampling_metadata.seed" in sampler
        or "sampling_metadata.positions" in sampler
    )
    ops_accepts_rng_state = (
        "expanded_idx_mapping" in ops
        or "positions" in ops
        or "seeds" in ops
    )
    return {
        "metadata_has_seed_tensor": metadata_has_seed_tensor,
        "metadata_has_position_tensor": metadata_has_position_tensor,
        "sampler_passes_rng_state": sampler_passes_rng_state,
        "topk_topp_ops_accepts_rng_state": ops_accepts_rng_state,
        "stateful_sampler_ready": all(
            (
                metadata_has_seed_tensor,
                metadata_has_position_tensor,
                sampler_passes_rng_state,
                ops_accepts_rng_state,
            )
        ),
        "notes": [
            "The active v1 TopKTopPSampler path currently receives top_k, top_p, and generators only.",
            "A state-preserving custom sampler needs seed/position metadata plumbed before performance benchmarking.",
        ],
    }


BLOCKERS = (
    {
        "id": "standalone_triton_launches",
        "evidence": "The L20 hook uses two Triton kernels after logits are already produced.",
        "requirement": "Move work into vLLM's compiled sampler path or fuse with the logits producer.",
    },
    {
        "id": "rng_not_vllm_stateful",
        "evidence": "The prototype uses torch.rand uniforms instead of vLLM seed/position state.",
        "requirement": "Reuse vLLM's Philox/seed/offset path before any serving claim.",
    },
    {
        "id": "python_gate_hot_path",
        "evidence": "The prototype checks shape and top-k/top-p values in Python for every call.",
        "requirement": "Keep policy decisions in request metadata or a compiled dispatch path.",
    },
    {
        "id": "full_logits_materialization",
        "evidence": "vLLM materializes full logits before sampler dispatch.",
        "requirement": "A high-ceiling win must attach to LM-head/GEMM epilogue or avoid an extra logits copy.",
    },
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--serving-summary", type=Path)
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
        "branch": run_git(source, ["branch", "--show-current"]),
        "commit": run_git(source, ["rev-parse", "--short", "HEAD"]),
        "dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
    }


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def find_needles(file_path: Path, needles: tuple[str, ...]) -> list[dict]:
    if not file_path.exists():
        return []
    lines = read_text(file_path).splitlines()
    matches = []
    for needle in needles:
        for line_number, line in enumerate(lines, start=1):
            if needle in line:
                matches.append(
                    {
                        "needle": needle,
                        "line": line_number,
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
        optional = bool(point.get("optional"))
        complete = len(matches) == len(point["needles"])
        points.append(
            {
                "id": point["id"],
                "path": point["path"],
                "role": point["role"],
                "optional": optional,
                "exists": target.exists(),
                "matched_needles": len(matches),
                "expected_needles": len(point["needles"]),
                "complete": complete,
                "required_complete": complete or optional,
                "matches": matches,
            }
        )
    return points


def load_serving_summary(path: Path | None) -> dict:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    deltas = data.get("deltas", {}).get("l20_notrace", {})
    return {
        "path": str(path),
        "model": data.get("model"),
        "shape": data.get("shape", {}),
        "l20_notrace_deltas": deltas,
        "rejects_standalone_hook": any(
            float(row.get("median_itl_pct", 0.0)) > 0.0 for row in deltas.values()
        ),
    }


def build_plan(points: list[dict], serving: dict, rng_gap: dict) -> list[dict]:
    complete_ids = {point["id"] for point in points if point["complete"]}
    has_active_sampler_boundary = {
        "active_v1_sampler_topk_topp_call",
        "active_v1_topk_topp_forward_cuda",
        "active_v1_sampling_metadata_contract",
    }.issubset(complete_ids)
    can_patch_sampler = has_active_sampler_boundary and bool(
        rng_gap.get("stateful_sampler_ready")
    )
    can_patch_epilogue = {
        "gpu_runner_logits_sampler_boundary",
        "logits_processor_lm_head",
    }.issubset(complete_ids)
    return [
        {
            "priority": "P0",
            "step": "do not enable the standalone L20 sampler hook",
            "ready": bool(serving.get("rejects_standalone_hook")),
            "details": (
                "Real ITL regressed, so the hook is useful only as a negative "
                "artifact and path-proof harness."
            ),
        },
        {
            "priority": "P0",
            "step": "build a state-preserving compiled sampler prototype",
            "ready": can_patch_sampler,
            "details": (
                "First extend active SamplingMetadata with request seed/position "
                "state; then patch TopKTopPSampler so a custom path can reuse "
                "vLLM RNG semantics without torch.rand or a stale worker hook."
            ),
        },
        {
            "priority": "P0",
            "step": "measure CUDA graph membership before claiming speed",
            "ready": can_patch_sampler,
            "details": (
                "Run Nsight Systems with kernel-name matching and CUDA graph "
                "reports; a new sampler kernel outside graph capture is not a "
                "serving optimization."
            ),
        },
        {
            "priority": "P1",
            "step": "prototype a logits/LM-head epilogue boundary",
            "ready": can_patch_epilogue,
            "details": (
                "Attach to the optimized logits producer instead of replacing "
                "full LM-head GEMM/GEMV with standalone Triton."
            ),
        },
    ]


def analyze(source: Path, serving_summary: Path | None) -> dict:
    points = scan_patch_points(source)
    serving = load_serving_summary(serving_summary)
    rng_gap = detect_rng_metadata_gap(source)
    complete = all(point["required_complete"] for point in points)
    return {
        "schema_version": 1,
        "complete": complete,
        "source": source_metadata(source),
        "patch_points": points,
        "rng_metadata_gap": rng_gap,
        "serving_evidence": serving,
        "blockers": list(BLOCKERS),
        "implementation_plan": build_plan(points, serving, rng_gap),
    }


def render_markdown(result: dict) -> str:
    source = result["source"]
    lines = [
        "# vLLM Compiled Sampler Boundary Scout",
        "",
        "This artifact follows the negative L20 standalone-sampler serving result. "
        "It records the source boundaries required for a compiled sampler or "
        "logits/LM-head epilogue path.",
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
        "| ID | Path | Required | Matched | Role |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for point in result["patch_points"]:
        required = "no" if point["optional"] else "yes"
        matched = f"{point['matched_needles']}/{point['expected_needles']}"
        lines.append(
            f"| `{point['id']}` | `{point['path']}` | {required} | "
            f"{matched} | {point['role']} |"
        )

    serving = result.get("serving_evidence", {})
    if serving:
        lines.extend(
            [
                "",
                "## Serving Evidence",
                "",
                f"- Summary: `{serving.get('path')}`",
                f"- Model: `{serving.get('model')}`",
                f"- Rejects standalone hook: `{serving.get('rejects_standalone_hook')}`",
                "",
                "| Shape | Median ITL delta | Output throughput delta |",
                "| --- | ---: | ---: |",
            ]
        )
        for shape, row in serving.get("l20_notrace_deltas", {}).items():
            lines.append(
                f"| `{shape}` | {row.get('median_itl_pct', 0.0):.2f}% | "
                f"{row.get('output_throughput_pct', 0.0):.2f}% |"
            )

    rng_gap = result.get("rng_metadata_gap", {})
    if rng_gap:
        lines.extend(
            [
                "",
                "## Active Sampler RNG Gap",
                "",
                f"- Metadata has seed tensor: `{rng_gap['metadata_has_seed_tensor']}`",
                f"- Metadata has position tensor: `{rng_gap['metadata_has_position_tensor']}`",
                f"- Sampler passes RNG state: `{rng_gap['sampler_passes_rng_state']}`",
                f"- TopKTopP ops accepts RNG state: `{rng_gap['topk_topp_ops_accepts_rng_state']}`",
                f"- Stateful sampler ready: `{rng_gap['stateful_sampler_ready']}`",
            ]
        )

    lines.extend(["", "## Blockers", "", "| ID | Requirement |", "| --- | --- |"])
    for blocker in result["blockers"]:
        lines.append(f"| `{blocker['id']}` | {blocker['requirement']} |")

    lines.extend(["", "## Implementation Plan", "", "| Priority | Ready | Step |", "| --- | ---: | --- |"])
    for item in result["implementation_plan"]:
        lines.append(f"| {item['priority']} | `{item['ready']}` | {item['step']} |")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    result = analyze(args.vllm_source, args.serving_summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
