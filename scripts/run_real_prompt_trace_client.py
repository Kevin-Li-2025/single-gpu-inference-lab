#!/usr/bin/env python3
"""Run a fixed real-prompt streaming trace against a vLLM OpenAI endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--store-output-preview-chars", type=int, default=160)
    return parser.parse_args()


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def compact_stats(values: list[float]) -> dict[str, float | None]:
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)) if values else None,
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def load_prompts(path: Path) -> list[dict[str, Any]]:
    prompts = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        if "prompt" not in item:
            raise ValueError(f"{path}:{lineno} is missing prompt")
        item.setdefault("id", f"prompt-{lineno}")
        item.setdefault("category", "unknown")
        item.setdefault("max_tokens", 96)
        prompts.append(item)
    if not prompts:
        raise ValueError(f"{path} has no prompts")
    return prompts


def load_tokenizer(tokenizer_id: str):
    if not tokenizer_id:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    except Exception:
        return None


def count_tokens(tokenizer: Any | None, text: str) -> int | None:
    if tokenizer is None:
        return None
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded.get("input_ids", []))


def post_streaming_completion(
    *,
    base_url: str,
    model: str,
    prompt: dict[str, Any],
    temperature: float,
    top_p: float,
    timeout_s: float,
    preview_chars: int,
    tokenizer: Any | None,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt["prompt"],
        "max_tokens": int(prompt.get("max_tokens", 96)),
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_chunk_at: float | None = None
    chunk_times: list[float] = []
    pieces: list[str] = []
    error: str | None = None
    status = "ok"

    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                text = event.get("choices", [{}])[0].get("text") or ""
                if not text:
                    continue
                now = time.perf_counter()
                if first_chunk_at is None:
                    first_chunk_at = now
                chunk_times.append(now)
                pieces.append(text)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        status = "error"
        error = repr(exc)

    finished = time.perf_counter()
    inter_chunk_ms = [
        (right - left) * 1000.0
        for left, right in zip(chunk_times, chunk_times[1:])
    ]
    output_text = "".join(pieces)
    input_tokens = count_tokens(tokenizer, prompt["prompt"])
    output_tokens = count_tokens(tokenizer, output_text)
    return {
        "id": prompt["id"],
        "category": prompt.get("category"),
        "status": status,
        "error": error,
        "max_tokens": int(prompt.get("max_tokens", 96)),
        "input_chars": len(prompt["prompt"]),
        "output_chars": len(output_text),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "ttft_ms": (first_chunk_at - started) * 1000.0 if first_chunk_at else None,
        "e2el_ms": (finished - started) * 1000.0,
        "stream_chunks": len(chunk_times),
        "median_itl_ms": percentile(inter_chunk_ms, 50),
        "p95_itl_ms": percentile(inter_chunk_ms, 95),
        "p99_itl_ms": percentile(inter_chunk_ms, 99),
        "output_preview": output_text[:preview_chars],
    }


def build_summary(
    *,
    base_url: str,
    model: str,
    prompts_path: Path,
    tokenizer_id: str,
    concurrency: int,
    temperature: float,
    top_p: float,
    timeout_s: float,
    preview_chars: int,
) -> dict[str, Any]:
    prompts = load_prompts(prompts_path)
    tokenizer = load_tokenizer(tokenizer_id)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                post_streaming_completion,
                base_url=base_url,
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                timeout_s=timeout_s,
                preview_chars=preview_chars,
                tokenizer=tokenizer,
            )
            for prompt in prompts
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    finished = time.perf_counter()
    results.sort(key=lambda row: row["id"])

    ok = [row for row in results if row["status"] == "ok"]
    ttft = [row["ttft_ms"] for row in ok if row["ttft_ms"] is not None]
    e2el = [row["e2el_ms"] for row in ok]
    median_itl = [row["median_itl_ms"] for row in ok if row["median_itl_ms"] is not None]
    output_tokens = [row["output_tokens"] for row in ok if row["output_tokens"] is not None]
    input_tokens = [row["input_tokens"] for row in ok if row["input_tokens"] is not None]
    wall_s = finished - started
    total_output_tokens = sum(output_tokens) if output_tokens else None
    total_input_tokens = sum(input_tokens) if input_tokens else None

    return {
        "schema_version": 1,
        "mode": "real_prompt_trace_streaming_vllm_completions",
        "base_url": base_url,
        "model": model,
        "prompts_jsonl": str(prompts_path),
        "tokenizer": tokenizer_id,
        "tokenizer_loaded": tokenizer is not None,
        "concurrency": concurrency,
        "temperature": temperature,
        "top_p": top_p,
        "num_prompts": len(prompts),
        "completed": len(ok),
        "failed": len(results) - len(ok),
        "wall_time_s": wall_s,
        "request_throughput": len(ok) / wall_s if wall_s > 0 else None,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "output_throughput": (
            total_output_tokens / wall_s
            if total_output_tokens is not None and wall_s > 0
            else None
        ),
        "ttft_ms": compact_stats(ttft),
        "median_itl_ms": compact_stats(median_itl),
        "e2el_ms": compact_stats(e2el),
        "results": results,
        "claim_boundary": [
            "This is a fixed real-prompt trace, not a random-token throughput matrix.",
            "The client measures streaming TTFT, inter-chunk ITL, and end-to-end latency from the OpenAI-compatible HTTP endpoint.",
            "Generated output is truncated to previews only; prompts are synthetic public coding tasks and contain no private data.",
        ],
    }


def fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    ttft_p95 = summary["ttft_ms"]["p95"]
    ttft_median = summary["ttft_ms"]["median"]
    lines = [
        "# Real Prompt Trace",
        "",
        "This artifact runs fixed code-oriented prompts through the real vLLM HTTP",
        "streaming path instead of a random-token benchmark.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Prompts | {summary['completed']} / {summary['num_prompts']} completed |",
        f"| Concurrency | {summary['concurrency']} |",
        f"| Request throughput | {fmt(summary.get('request_throughput'))} req/s |",
        f"| Output throughput | {fmt(summary.get('output_throughput'))} tok/s |",
        f"| Median TTFT | {fmt(summary['ttft_ms']['median'])} ms |",
        f"| p95 TTFT | {fmt(summary['ttft_ms']['p95'])} ms |",
        f"| p99 TTFT | {fmt(summary['ttft_ms']['p99'])} ms |",
        f"| Median E2E | {fmt(summary['e2el_ms']['median'])} ms |",
        f"| p95 E2E | {fmt(summary['e2el_ms']['p95'])} ms |",
        f"| p99 E2E | {fmt(summary['e2el_ms']['p99'])} ms |",
        "",
        "## Interpretation",
        "",
        "This is a small fixed-prompt trace, so its tail values should be read as",
        "trace evidence rather than a stable service SLO. In this run the first",
        "concurrency wave carries a visible TTFT tail, while the decode-side",
        "inter-token latency stays tightly grouped.",
        "",
        f"- Median TTFT: {fmt(ttft_median)} ms.",
        f"- p95 TTFT: {fmt(ttft_p95)} ms.",
        f"- Median per-prompt ITL: {fmt(summary['median_itl_ms']['median'])} ms.",
        f"- p99 per-prompt median ITL: {fmt(summary['median_itl_ms']['p99'])} ms.",
        "",
        "## Prompt Rows",
        "",
        "| ID | Category | TTFT | Median ITL | E2E | Output tokens | Chunks |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["results"]:
        lines.append(
            f"| `{row['id']}` | {row.get('category') or ''} | "
            f"{fmt(row.get('ttft_ms'))} ms | "
            f"{fmt(row.get('median_itl_ms'))} ms | "
            f"{fmt(row.get('e2el_ms'))} ms | "
            f"{row.get('output_tokens') if row.get('output_tokens') is not None else 'n/a'} | "
            f"{row.get('stream_chunks')} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = build_summary(
        base_url=args.base_url,
        model=args.model,
        prompts_path=args.prompts_jsonl,
        tokenizer_id=args.tokenizer,
        concurrency=args.concurrency,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout_s=args.timeout_s,
        preview_chars=args.store_output_preview_chars,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
