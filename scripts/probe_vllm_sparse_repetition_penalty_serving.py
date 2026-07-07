#!/usr/bin/env python3
"""Paired OpenAI-compatible serving probe for sparse repetition penalty.

This script exists because the candidate path must pass ``vllm_xargs`` into the
request body. It reports the same core serving metrics as vLLM benchmark runs
for a narrow A/B gate: TTFT, ITL, output throughput, request throughput, and
trace-policy coverage.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROCESSOR_FQCN = (
    "integrations.vllm.l20_sparse_repetition_penalty_logits_processor:"
    "L20SparseRepetitionPenaltyProcessor"
)


@dataclass(frozen=True)
class RequestResult:
    status: str
    total_ms: float
    ttft_ms: float
    itl_ms: float
    output_chunks: int
    completion_tokens: int
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="OpenAI completions endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["baseline", "candidate", "standalone", "fused"],
        required=True,
    )
    parser.add_argument("--input-tokens", type=int, default=512)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--trace-jsonl", type=Path)
    parser.add_argument("--processor-fqcn", default=PROCESSOR_FQCN)
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return float(ordered[index])


def synthetic_prompt(index: int, input_tokens: int) -> str:
    motif = (
        "CUDA kernel optimization L20 vLLM repetition penalty benchmark "
        f"request {index} "
    )
    approx_words = max(8, input_tokens)
    words = []
    while len(words) < approx_words:
        words.extend(motif.split())
    return " ".join(words[:approx_words])


def request_payload(args: argparse.Namespace, index: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": synthetic_prompt(index, args.input_tokens),
        "max_tokens": args.output_tokens,
        "stream": True,
        "ignore_eos": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }
    if args.variant in {"baseline", "fused"}:
        payload["repetition_penalty"] = args.repetition_penalty
    else:
        payload["repetition_penalty"] = 1.0
        payload["logits_processors"] = [args.processor_fqcn]
        payload["vllm_xargs"] = {
            "l20_sparse_repetition_penalty": True,
            "l20_repetition_penalty": args.repetition_penalty,
            "l20_penalty_include_prompt": False,
        }
    return payload


def stream_completion(
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> RequestResult:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    arrivals: list[float] = []
    completion_tokens = 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_line = line[len("data:") :].strip()
                if data_line == "[DONE]":
                    break
                event = json.loads(data_line)
                choices = event.get("choices") or []
                text = choices[0].get("text", "") if choices else ""
                usage = event.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = max(completion_tokens, int(usage["completion_tokens"]))
                if text:
                    arrivals.append(time.perf_counter())
        total_ms = 1000.0 * (time.perf_counter() - started)
        ttft_ms = 1000.0 * (arrivals[0] - started) if arrivals else total_ms
        intervals = [
            1000.0 * (arrivals[index] - arrivals[index - 1])
            for index in range(1, len(arrivals))
        ]
        itl_ms = statistics.median(intervals) if intervals else 0.0
        return RequestResult(
            status="ok",
            total_ms=total_ms,
            ttft_ms=ttft_ms,
            itl_ms=itl_ms,
            output_chunks=len(arrivals),
            completion_tokens=max(completion_tokens, len(arrivals)),
            error=None,
        )
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return RequestResult(
            status="error",
            total_ms=1000.0 * (time.perf_counter() - started),
            ttft_ms=0.0,
            itl_ms=0.0,
            output_chunks=0,
            completion_tokens=0,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_batch(args: argparse.Namespace, count: int, offset: int) -> tuple[list[RequestResult], float]:
    started = time.perf_counter()
    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futures = [
            pool.submit(
                stream_completion,
                args.url,
                request_payload(args, offset + index),
                args.timeout,
            )
            for index in range(count)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    elapsed_s = time.perf_counter() - started
    return results, elapsed_s


def load_trace_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"trace_exists": False}
    provider_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    rows = 0
    max_unique = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        rows += 1
        provider = str(event.get("provider", "unknown"))
        reason = str(event.get("reason", "unknown"))
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        max_unique = max(max_unique, int(event.get("max_unique_tokens") or 0))
    return {
        "trace_exists": True,
        "event_count": rows,
        "provider_counts": provider_counts,
        "reason_counts": reason_counts,
        "max_unique_tokens_seen": max_unique,
    }


def summarize(args: argparse.Namespace, results: list[RequestResult], elapsed_s: float) -> dict[str, Any]:
    errors = [result.error for result in results if result.status != "ok"]
    ok = [result for result in results if result.status == "ok"]
    completion_tokens = sum(result.completion_tokens for result in ok)
    ttft = [result.ttft_ms for result in ok]
    itl = [result.itl_ms for result in ok if result.itl_ms > 0]
    total = [result.total_ms for result in ok]
    return {
        "schema_version": 1,
        "variant": args.variant,
        "model": args.model,
        "completed": len(ok),
        "failed": len(errors),
        "errors": errors[:5],
        "input_tokens_requested": args.input_tokens,
        "output_tokens_requested": args.output_tokens,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "request_throughput": len(ok) / elapsed_s if elapsed_s > 0 else 0.0,
        "output_throughput": completion_tokens / elapsed_s if elapsed_s > 0 else 0.0,
        "median_ttft_ms": statistics.median(ttft) if ttft else 0.0,
        "p95_ttft_ms": percentile(ttft, 95),
        "median_itl_ms": statistics.median(itl) if itl else 0.0,
        "p95_itl_ms": percentile(itl, 95),
        "median_e2el_ms": statistics.median(total) if total else 0.0,
        "trace": load_trace_summary(args.trace_jsonl),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.warmup:
        run_batch(args, args.warmup, -args.warmup)
    results, elapsed_s = run_batch(args, args.num_prompts, 0)
    summary = summarize(args, results, elapsed_s)
    raw_path = args.output_dir / f"{args.variant}_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.__dict__, sort_keys=True) + "\n")
    summary_path = args.output_dir / f"{args.variant}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
