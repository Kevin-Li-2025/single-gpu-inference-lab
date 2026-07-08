#!/usr/bin/env python3
"""Build a compact CPU-vs-L20 break-even table from checked-in artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = (
    "benchmarks/results/cpu-l20-break-even/qwen-family-p512-o32-o128-v1"
)
SHAPE_RE = re.compile(r"c(?P<concurrency>\d+)-i(?P<input_tokens>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-o32",
        type=Path,
        default=Path(
            "benchmarks/results/cpu-real-model/"
            "qwen25-coder-0p5b-q4km-p512-o32-sweep/summary.json"
        ),
    )
    parser.add_argument(
        "--cpu-o128",
        type=Path,
        default=Path(
            "benchmarks/results/cpu-real-model/"
            "qwen25-coder-0p5b-q4km-p512-o128-sweep/summary.json"
        ),
    )
    parser.add_argument(
        "--l20-o32",
        type=Path,
        default=Path(
            "benchmarks/results/l20-vllm-sampling-winner-v2/"
            "qwen3-0p6b-c1c2c4c8-i512-o32-r5/summary.json"
        ),
    )
    parser.add_argument(
        "--l20-o128",
        type=Path,
        default=Path(
            "benchmarks/results/l20-vllm-sampling-winner-v2/"
            "qwen3-0p6b-c1-i512-o128-r3/summary.json"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--mode",
        default="cpu_l20_qwen_family_break_even",
        choices=(
            "cpu_l20_qwen_family_break_even",
            "cpu_l20_same_model_break_even",
        ),
    )
    parser.add_argument(
        "--title",
        default="CPU vs L20 Break-Even: Qwen-Family p512",
    )
    parser.add_argument(
        "--l20-model",
        default="Qwen3-0.6B",
    )
    parser.add_argument(
        "--l20-source",
        default="vLLM FlashInfer serving, NVIDIA L20",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cpu_shape(summary: dict[str, Any], prompt_tokens: int, output_tokens: int) -> dict[str, Any]:
    tests = summary["tests"]
    prefill = tests[f"pp{prompt_tokens}"]
    decode = tests[f"tg{output_tokens}"]
    combined = tests[f"pp{prompt_tokens}+tg{output_tokens}"]
    combined_ms = float(combined["avg_ms"])
    return {
        "model": "Qwen2.5-Coder-0.5B-Instruct Q4_K_M",
        "source": "llama-bench CPU-only, Apple M4 Accelerate",
        "model_filename": summary["model_filename"],
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "prefill_tokens_per_s": float(prefill["avg_tokens_per_s"]),
        "decode_tokens_per_s": float(decode["avg_tokens_per_s"]),
        "combined_tokens_per_s": float(combined["avg_tokens_per_s"]),
        "combined_ms": combined_ms,
        "serial_requests_per_s": 1000.0 / combined_ms,
        "prefill_threads": int(prefill["n_threads"]),
        "decode_threads": int(decode["n_threads"]),
        "combined_threads": int(combined["n_threads"]),
    }


def iter_l20_flashinfer_shapes(
    summary: dict[str, Any],
    output_tokens: int,
    *,
    model: str = "Qwen3-0.6B",
    source: str = "vLLM FlashInfer serving, NVIDIA L20",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in summary["pairs"]:
        for shape, result in sorted(pair["shapes"].items()):
            match = SHAPE_RE.match(shape)
            if not match:
                continue
            flashinfer = result["flashinfer"]
            output_throughput = float(flashinfer["output_throughput"])
            rows.append(
                {
                    "model": model,
                    "source": source,
                    "shape": shape,
                    "concurrency": int(match.group("concurrency")),
                    "prompt_tokens": int(match.group("input_tokens")),
                    "output_tokens": output_tokens,
                    "runs": int(flashinfer["runs"]),
                    "median_itl_ms": float(flashinfer["median_itl_ms"]),
                    "median_ttft_ms": float(flashinfer["median_ttft_ms"]),
                    "p99_itl_ms": float(flashinfer["p99_itl_ms"]),
                    "output_throughput": output_throughput,
                    "estimated_request_throughput": output_throughput / output_tokens,
                }
            )
    return rows


def attach_break_even(cpu: dict[str, Any], l20_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched = []
    for row in l20_rows:
        enriched_row = dict(row)
        enriched_row["vs_cpu_serial_request_throughput"] = (
            row["estimated_request_throughput"] / cpu["serial_requests_per_s"]
        )
        enriched_row["vs_cpu_decode_throughput"] = (
            row["output_throughput"] / cpu["decode_tokens_per_s"]
        )
        enriched.append(enriched_row)
    return enriched


def build_summary(
    cpu_o32_path: Path,
    cpu_o128_path: Path,
    l20_o32_path: Path,
    l20_o128_path: Path,
    *,
    mode: str = "cpu_l20_qwen_family_break_even",
    title: str = "CPU vs L20 Break-Even: Qwen-Family p512",
    l20_model: str = "Qwen3-0.6B",
    l20_source: str = "vLLM FlashInfer serving, NVIDIA L20",
) -> dict[str, Any]:
    cpu_o32 = cpu_shape(load_json(cpu_o32_path), prompt_tokens=512, output_tokens=32)
    cpu_o128 = cpu_shape(load_json(cpu_o128_path), prompt_tokens=512, output_tokens=128)
    l20_o32 = attach_break_even(
        cpu_o32,
        iter_l20_flashinfer_shapes(
            load_json(l20_o32_path),
            output_tokens=32,
            model=l20_model,
            source=l20_source,
        ),
    )
    l20_o128 = attach_break_even(
        cpu_o128,
        iter_l20_flashinfer_shapes(
            load_json(l20_o128_path),
            output_tokens=128,
            model=l20_model,
            source=l20_source,
        ),
    )
    if mode == "cpu_l20_same_model_break_even":
        claim_boundary = [
            "CPU and L20 rows target the same Qwen2.5-Coder-0.5B-Instruct model family.",
            "CPU rows use Qwen2.5-Coder-0.5B-Instruct Q4_K_M GGUF on Apple M4.",
            "L20 rows use Qwen2.5-Coder-0.5B-Instruct vLLM FlashInfer serving artifacts.",
            "CPU and L20 runtimes use different precision/runtime stacks, so this is a serving boundary comparison, not bit-identical math.",
            "CPU llama-bench excludes tokenization and sampling; use the C++ completion smoke for an output-producing CPU path proof.",
            "L20 request throughput is estimated as output_throughput / requested_output_tokens because these source summaries do not store request_throughput.",
        ]
        decision = {
            "single_local_request": "M4 CPU is usable for local single-user Qwen2.5-Coder-0.5B decode when the measured serial p512 request rate is acceptable.",
            "serving_boundary": "Use L20/vLLM once the p512 workload needs multi-request concurrency, stable tail latency, or more than one serial M4 process can provide.",
            "next_proof": "Add cost-per-1M-output-token and memory footprint columns after same-model latency is checked in.",
        }
    else:
        claim_boundary = [
            "CPU and L20 rows are Qwen-family controls, not identical-model comparisons.",
            "CPU rows use Qwen2.5-Coder-0.5B-Instruct Q4_K_M GGUF on Apple M4.",
            "L20 rows use checked-in Qwen3-0.6B vLLM FlashInfer serving artifacts.",
            "CPU llama-bench excludes tokenization and sampling; use the C++ completion smoke for an output-producing CPU path proof.",
            "L20 request throughput is estimated as output_throughput / requested_output_tokens because these source summaries do not store request_throughput.",
        ]
        decision = {
            "single_local_request": "M4 CPU is usable for local single-user Qwen-family decode when 0.35-0.57 serial req/s is acceptable.",
            "serving_boundary": "Use L20/vLLM once the p512 workload needs multi-request concurrency, stable tail latency, or more than one serial M4 process can provide.",
            "next_proof": "Replace the family-level L20 rows with same-model Qwen2.5-Coder-0.5B serving artifacts when available.",
        }
    return {
        "schema_version": 1,
        "mode": mode,
        "title": title,
        "claim_boundary": claim_boundary,
        "inputs": {
            "cpu_o32": str(cpu_o32_path),
            "cpu_o128": str(cpu_o128_path),
            "l20_o32": str(l20_o32_path),
            "l20_o128": str(l20_o128_path),
        },
        "cpu": {
            "p512_o32": cpu_o32,
            "p512_o128": cpu_o128,
        },
        "l20": {
            "p512_o32": l20_o32,
            "p512_o128": l20_o128,
        },
        "decision": decision,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    same_model = summary["mode"] == "cpu_l20_same_model_break_even"
    lines = [
        f"# {summary.get('title', 'CPU vs L20 Break-Even')}",
        "",
    ]
    if same_model:
        lines.extend(
            [
                "This artifact converts the checked-in M4 CPU Qwen2.5-Coder GGUF",
                "measurements and same-model L20 vLLM FlashInfer serving",
                "measurements into one boundary table.",
                "",
                "It is a serving-boundary comparison rather than a bit-identical",
                "runtime comparison: CPU uses Q4_K_M GGUF through llama.cpp, while",
                "L20 uses vLLM serving.",
            ]
        )
    else:
        lines.extend(
            [
                "This artifact converts the checked-in M4 CPU Qwen2.5-Coder GGUF",
                "measurements and L20 Qwen3 vLLM FlashInfer serving measurements into",
                "one boundary table. It is a Qwen-family control, not an identical-model",
                "comparison.",
            ]
        )
    lines.extend(
        [
            "",
            "## CPU Baseline",
            "",
            "| Shape | Combined ms | Serial req/s | Prefill tok/s | Decode tok/s | Threads |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for key, row in summary["cpu"].items():
        lines.append(
            f"| `{key}` | {row['combined_ms']:.3f} | "
            f"{row['serial_requests_per_s']:.3f} | "
            f"{row['prefill_tokens_per_s']:.3f} | "
            f"{row['decode_tokens_per_s']:.3f} | "
            f"prefill {row['prefill_threads']}, decode {row['decode_threads']}, combined {row['combined_threads']} |"
        )
    lines.extend(
        [
            "",
            "## L20 Serving Rows",
            "",
            "| Shape | Concurrency | Output tok/s | Est req/s | Median TTFT | Median ITL | M4 req/s equivalent | M4 decode equivalent |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for group_name, rows in summary["l20"].items():
        for row in rows:
            lines.append(
                f"| `{group_name}` / `{row['shape']}` | "
                f"{row['concurrency']} | "
                f"{row['output_throughput']:.3f} | "
                f"{row['estimated_request_throughput']:.3f} | "
                f"{row['median_ttft_ms']:.3f} ms | "
                f"{row['median_itl_ms']:.3f} ms | "
                f"{row['vs_cpu_serial_request_throughput']:.2f}x | "
                f"{row['vs_cpu_decode_throughput']:.2f}x |"
            )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- M4 CPU is credible for local single-user inference when roughly",
            "  0.35-0.57 serial p512 requests/s is acceptable.",
            "- L20/vLLM becomes the right tool for multi-request serving, tail-latency",
            "  control, or any workload that needs many serial M4 equivalents.",
            "- Keep this claim scoped: the CPU side is Qwen2.5-Coder-0.5B Q4_K_M,",
        ]
    )
    if same_model:
        lines.extend(
            [
                "  while the L20 side is Qwen2.5-Coder-0.5B vLLM FlashInfer serving.",
            ]
        )
    else:
        lines.extend(
            [
                "  while the checked-in L20 side is Qwen3-0.6B FlashInfer serving.",
            ]
        )
    lines.extend(
        [
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = build_summary(
        args.cpu_o32,
        args.cpu_o128,
        args.l20_o32,
        args.l20_o128,
        mode=args.mode,
        title=args.title,
        l20_model=args.l20_model,
        l20_source=args.l20_source,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "README.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
