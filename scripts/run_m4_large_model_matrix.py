#!/usr/bin/env python3
"""Run a real Qwen 3B CPU, Metal, and MLX matrix on Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import statistics
import subprocess
import tempfile
from pathlib import Path


DEFAULT_PROMPT = (
    "Implement a Python function merge_intervals(intervals) that merges "
    "overlapping integer intervals. Include type hints and two assertions."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-bench", required=True)
    parser.add_argument("--llama-completion", required=True)
    parser.add_argument("--mlx-python", required=True)
    parser.add_argument(
        "--mlx-model", default="mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    )
    parser.add_argument(
        "--gguf-source", default="Qwen/Qwen2.5-Coder-3B-Instruct-GGUF"
    )
    parser.add_argument(
        "--gguf-revision", default="f74adce6aa16316c625447af059dbebe4983757c"
    )
    parser.add_argument(
        "--mlx-revision", default="3dd939c621c08e5753d5b89f35a2642cd83b98ca"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--completion-runs", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--cpu-thread-sweep-json", required=True)
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results/cpu-m4-large-model/qwen25-coder-3b-v1",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_json(command: list[str]) -> object:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def run_llama_bench(
    args: argparse.Namespace, backend: str, prompt_tokens: int, decode_tokens: int
) -> dict:
    ngl = "0" if backend == "cpu" else "99"
    rows = run_json(
        [
            args.llama_bench,
            "-m",
            args.model,
            "-p",
            str(prompt_tokens),
            "-n",
            str(decode_tokens),
            "-t",
            str(args.threads),
            "-ngl",
            ngl,
            "-r",
            str(args.repetitions),
            "-o",
            "json",
        ]
    )
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError(f"unexpected llama-bench output for {backend}: {rows!r}")
    row = rows[0]
    return {
        "avg_tokens_per_s": row["avg_ts"],
        "stddev_tokens_per_s": row["stddev_ts"],
        "samples_tokens_per_s": row["samples_ts"],
        "build_commit": row["build_commit"],
        "model_size_bytes": row["model_size"],
        "backend": backend,
        "n_gpu_layers": row["n_gpu_layers"],
        "threads": row["n_threads"],
    }


def parse_completion_perf(log_text: str) -> dict[str, float]:
    pattern = re.compile(
        r"(?P<label>prompt eval|eval) time\s*=\s*(?P<ms>[0-9.]+) ms\s*/\s*"
        r"(?P<count>[0-9]+) (?:tokens|runs)\s*\(\s*(?P<per>[0-9.]+) ms per token,\s*"
        r"(?P<tps>[0-9.]+) tokens per second"
    )
    result: dict[str, float] = {}
    for match in pattern.finditer(log_text):
        prefix = "prompt" if match.group("label") == "prompt eval" else "decode"
        result[f"{prefix}_ms"] = float(match.group("ms"))
        result[f"{prefix}_tokens_per_s"] = float(match.group("tps"))
        result[f"{prefix}_count"] = int(match.group("count"))
    if "decode_tokens_per_s" not in result:
        raise RuntimeError("llama-completion performance counters were not found")
    return result


def run_completion(args: argparse.Namespace, backend: str, run_dir: Path) -> dict:
    log_path = run_dir / "runtime.log"
    ngl = "0" if backend == "cpu" else "99"
    completed = subprocess.run(
        [
            args.llama_completion,
            "-m",
            args.model,
            "-p",
            args.prompt,
            "-n",
            str(args.decode_tokens),
            "-c",
            "1024",
            "-b",
            "512",
            "-ub",
            "128",
            "-t",
            str(args.threads),
            "-tb",
            str(args.threads),
            "-ngl",
            ngl,
            "--mmap",
            "--temp",
            "0",
            "--top-k",
            "1",
            "--top-p",
            "1",
            "--repeat-penalty",
            "1",
            "--seed",
            "7",
            "--no-display-prompt",
            "--simple-io",
            "--log-file",
            str(log_path),
            "-no-cnv",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = parse_completion_perf(log_path.read_text(encoding="utf-8"))
    result["output_sha256"] = hashlib.sha256(completed.stdout.encode()).hexdigest()
    result["output"] = completed.stdout
    return result


def median(rows: list[dict], key: str) -> float:
    return statistics.median(row[key] for row in rows)


def load_thread_sweep(path: str | None) -> dict | None:
    if path is None:
        return None
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = [
        {
            "threads": row["n_threads"],
            "avg_tokens_per_s": row["avg_ts"],
            "stddev_tokens_per_s": row["stddev_ts"],
            "samples_tokens_per_s": row["samples_ts"],
        }
        for row in raw
    ]
    winner = max(rows, key=lambda row: row["avg_tokens_per_s"])
    return {"rows": rows, "selected_threads": winner["threads"]}


def render_readme(payload: dict) -> str:
    cpu = payload["llama"]["cpu"]
    metal = payload["llama"]["metal"]
    mlx = payload["mlx"]
    return f"""# Qwen2.5-Coder 3B On Apple M4

This artifact uses a real 3B model and a fixed code-generation prompt. The
llama.cpp rows use the exact same official GGUF Q4_K_M bytes. The MLX row uses
the same model architecture in MLX 4-bit format, so it is a system comparison,
not a bitwise-identical quantization comparison.

## Throughput

| Runtime | Prefill p{payload['settings']['prompt_tokens']} | Decode tg{payload['settings']['decode_tokens']} | Real completion decode |
| --- | ---: | ---: | ---: |
| llama.cpp CPU | {cpu['prefill']['avg_tokens_per_s']:.2f} tok/s | {cpu['decode']['avg_tokens_per_s']:.2f} tok/s | {cpu['completion_median_decode_tokens_per_s']:.2f} tok/s |
| llama.cpp Metal | {metal['prefill']['avg_tokens_per_s']:.2f} tok/s | {metal['decode']['avg_tokens_per_s']:.2f} tok/s | {metal['completion_median_decode_tokens_per_s']:.2f} tok/s |
| MLX Metal 4-bit | {mlx['median_prompt_tokens_per_s']:.2f} tok/s | {mlx['median_generation_tokens_per_s']:.2f} tok/s | {mlx['median_generation_tokens_per_s']:.2f} tok/s |

Metal/CPU llama decode speedup: **{payload['speedups']['llama_metal_over_cpu_decode']:.2f}x**.
MLX/CPU llama real-completion speedup: **{payload['speedups']['mlx_over_cpu_completion_decode']:.2f}x**.

The CPU thread sweep selected **{payload['cpu_thread_sweep']['selected_threads']} threads**;
using all ten M4 cores is slower for this memory-bound decode workload.

## Correctness Boundary

- llama.cpp CPU and Metal completion outputs exact: `{str(payload['llama']['cpu_metal_outputs_exact']).lower()}`;
- MLX repeated output stable: `{str(mlx['output_stable']).lower()}`;
- GGUF SHA-256: `{payload['model']['sha256']}`;
- no mock tensors or synthetic model weights are used in this artifact.

## SME2 Follow-up

The follow-up now parses and repacks real Q4_K tensors, restores the affine
minimum term, and reaches llama.cpp decode. Full FFN tensor kernels improve,
but the four-thread end-to-end gate regresses and remains disabled. See
`benchmarks/results/cpu-m4-q4k-sme2/qwen25-coder-3b-affine-v1/`.
"""


def main() -> int:
    args = parse_args()
    model_arg = Path(args.model)
    model = model_arg.resolve()
    with model.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"not a GGUF model: {model}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thread_sweep = load_thread_sweep(args.cpu_thread_sweep_json)
    if thread_sweep is None or thread_sweep["selected_threads"] != args.threads:
        raise RuntimeError(
            "--threads must match the winner in --cpu-thread-sweep-json"
        )

    llama: dict[str, dict] = {}
    completion_outputs: dict[str, list[str]] = {"cpu": [], "metal": []}
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        for backend in ("cpu", "metal"):
            prefill = run_llama_bench(args, backend, args.prompt_tokens, 0)
            decode = run_llama_bench(args, backend, 0, args.decode_tokens)
            completion_rows = []
            for iteration in range(args.completion_runs):
                run_dir = temp_root / f"{backend}-{iteration}"
                run_dir.mkdir()
                row = run_completion(args, backend, run_dir)
                completion_outputs[backend].append(row.pop("output"))
                completion_rows.append(row)
            llama[backend] = {
                "prefill": prefill,
                "decode": decode,
                "completion_rows": completion_rows,
                "completion_median_prompt_tokens_per_s": median(
                    completion_rows, "prompt_tokens_per_s"
                ),
                "completion_median_decode_tokens_per_s": median(
                    completion_rows, "decode_tokens_per_s"
                ),
            }

    mlx = run_json(
        [
            args.mlx_python,
            str(Path(__file__).with_name("benchmark_mlx_qwen.py")),
            "--model",
            args.mlx_model,
            "--prompt",
            args.prompt,
            "--max-tokens",
            str(args.decode_tokens),
            "--iterations",
            str(args.repetitions),
        ]
    )
    llama["cpu_metal_outputs_exact"] = (
        completion_outputs["cpu"] == completion_outputs["metal"]
    )

    payload = {
        "schema_version": 1,
        "implementation": "scripts/run_m4_large_model_matrix.py",
        "mode": "apple_m4_real_qwen3b_cpu_metal_mlx",
        "hardware": {
            "machine": platform.machine(),
            "macos": platform.mac_ver()[0],
            "processor": platform.processor(),
        },
        "model": {
            "filename": model_arg.name,
            "gguf_source": args.gguf_source,
            "gguf_revision": args.gguf_revision,
            "size_bytes": model.stat().st_size,
            "sha256": sha256_file(model),
            "llama_quantization": "official GGUF Q4_K_M",
            "mlx_model": args.mlx_model,
            "mlx_revision": args.mlx_revision,
            "comparison_boundary": (
                "llama CPU/Metal use identical GGUF bytes; MLX uses the same model "
                "architecture in a different 4-bit format"
            ),
        },
        "settings": {
            "threads": args.threads,
            "repetitions": args.repetitions,
            "completion_runs": args.completion_runs,
            "prompt_tokens": args.prompt_tokens,
            "decode_tokens": args.decode_tokens,
            "real_prompt": args.prompt,
        },
        "cpu_thread_sweep": thread_sweep,
        "llama": llama,
        "mlx": mlx,
        "speedups": {
            "llama_metal_over_cpu_decode": llama["metal"]["decode"]["avg_tokens_per_s"]
            / llama["cpu"]["decode"]["avg_tokens_per_s"],
            "mlx_over_cpu_completion_decode": mlx["median_generation_tokens_per_s"]
            / llama["cpu"]["completion_median_decode_tokens_per_s"],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(render_readme(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
