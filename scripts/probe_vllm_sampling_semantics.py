#!/usr/bin/env python3
"""Probe serving latency by sampling semantics on an OpenAI-compatible vLLM API.

The goal is to separate the LM-head/logits boundary from sampling semantics.
Each case changes one sampling feature while keeping the prompt, model, output
length, and server fixed. Results are written as raw JSONL plus a compact
summary so the run can be used as a future optimization gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "List practical reasons batch-one LLM decode can be slow on a PCIe GPU. "
    "Use numbered points and keep each point concise."
)


@dataclass(frozen=True)
class ProbeCase:
    name: str
    description: str
    sampling: dict[str, Any]


def build_probe_cases(logprobs: int = 5) -> list[ProbeCase]:
    """Return a one-feature-at-a-time semantics matrix."""

    no_penalty = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
    }
    return [
        ProbeCase(
            "greedy_no_penalty",
            "Greedy decode with penalties disabled; this is the current narrow epilogue control.",
            dict(no_penalty),
        ),
        ProbeCase(
            "greedy_default_repetition",
            "Greedy decode with vLLM-style repetition penalty active.",
            {
                **no_penalty,
                "repetition_penalty": 1.05,
            },
        ),
        ProbeCase(
            "sample_topk_topp",
            "Stochastic top-k/top-p sampling without penalties.",
            {
                **no_penalty,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
            },
        ),
        ProbeCase(
            "sample_topk_topp_penalty",
            "Stochastic top-k/top-p sampling with repetition/presence/frequency penalties.",
            {
                **no_penalty,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "repetition_penalty": 1.05,
            },
        ),
        ProbeCase(
            "sample_topk_topp_penalty_logprobs",
            "Stochastic top-k/top-p sampling with penalties and token logprobs.",
            {
                **no_penalty,
                "temperature": 0.8,
                "top_p": 0.9,
                "top_k": 50,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "repetition_penalty": 1.05,
                "logprobs": logprobs,
            },
        ),
        ProbeCase(
            "greedy_token_logprobs",
            "Greedy decode requesting token logprobs.",
            {
                **no_penalty,
                "logprobs": logprobs,
            },
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="OpenAI completions endpoint")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--non-stream", action="store_true")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--logprobs", type=int, default=5)
    return parser.parse_args()


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_stream_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    return str(choice.get("text") or "")


def stream_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    sampling: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "ignore_eos": True,
        **sampling,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        chunks = 0
        parts: list[str] = []
        arrivals: list[float] = []
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_line = line[len("data:") :].strip()
                if data_line == "[DONE]":
                    break
                payload = json.loads(data_line)
                text = _extract_stream_text(payload)
                if not text:
                    continue
                now = time.perf_counter()
                arrivals.append(now)
                parts.append(text)
                chunks += 1
        total_ms = 1000.0 * (time.perf_counter() - started)
        ttft_ms = 1000.0 * (arrivals[0] - started) if arrivals else total_ms
        intervals = [
            1000.0 * (arrivals[index] - arrivals[index - 1])
            for index in range(1, len(arrivals))
        ]
        itl_ms = statistics.median(intervals) if intervals else 0.0
        return {
            "status": "ok",
            "error": None,
            "total_ms": total_ms,
            "ttft_ms": ttft_ms,
            "itl_ms": itl_ms,
            "completion_tokens": chunks,
            "output_chunks": chunks,
            "ms_per_output_token": total_ms / max(chunks, 1),
            "output_preview": "".join(parts)[:200],
            "usage": {},
        }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "total_ms": 1000.0 * (time.perf_counter() - started),
        }


def non_stream_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    sampling: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": False,
        "ignore_eos": True,
        **sampling,
    }
    started = time.perf_counter()
    try:
        response = post_json(url, payload, timeout)
        total_ms = 1000.0 * (time.perf_counter() - started)
        usage = response.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or max_tokens)
        choices = response.get("choices") or []
        text = choices[0].get("text", "") if choices else ""
        return {
            "status": "ok",
            "error": None,
            "total_ms": total_ms,
            "ttft_ms": total_ms,
            "itl_ms": 0.0,
            "completion_tokens": completion_tokens,
            "output_chunks": 1,
            "ms_per_output_token": total_ms / max(completion_tokens, 1),
            "output_preview": text[:200],
            "usage": usage,
        }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "total_ms": 1000.0 * (time.perf_counter() - started),
        }


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def run_case(args: argparse.Namespace, case: ProbeCase) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_runs = args.warmup + args.runs
    for index in range(total_runs):
        runner = non_stream_completion if args.non_stream else stream_completion
        result = runner(
            args.url,
            args.model,
            args.prompt,
            args.max_tokens,
            case.sampling,
            args.timeout,
        )
        row = {
            "case": case.name,
            "case_description": case.description,
            "sampling": case.sampling,
            "index": index,
            "warmup": index < args.warmup,
            **result,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)

    measured = [row for row in rows if not row["warmup"] and row["status"] == "ok"]
    return rows, {
        "case": case.name,
        "description": case.description,
        "sampling": case.sampling,
        "runs": args.runs,
        "warmup": args.warmup,
        "ok_runs": len(measured),
        "stream": not args.non_stream,
        "total_ms": summarize([float(row["total_ms"]) for row in measured]),
        "ttft_ms": summarize([float(row["ttft_ms"]) for row in measured]),
        "itl_ms": summarize([float(row["itl_ms"]) for row in measured]),
        "ms_per_output_token": summarize(
            [float(row["ms_per_output_token"]) for row in measured]
        ),
        "output_chunks": summarize(
            [float(row.get("output_chunks", 0)) for row in measured]
        ),
        "completion_tokens": summarize(
            [float(row["completion_tokens"]) for row in measured]
        ),
        "errors": [
            {"index": row["index"], "error": row["error"]}
            for row in rows
            if row["status"] != "ok"
        ],
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(args.cases or [])
    case_definitions = build_probe_cases(logprobs=args.logprobs)
    cases = [
        case
        for case in case_definitions
        if not selected or case.name in selected
    ]
    if selected and len(cases) != len(selected):
        known = {case.name for case in case_definitions}
        missing = sorted(selected - known)
        raise SystemExit(f"unknown probe case(s): {', '.join(missing)}")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case in cases:
        rows, summary = run_case(args, case)
        all_rows.extend(rows)
        summaries.append(summary)

    raw_path = args.output_dir / "sampling_semantics_raw.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "schema_version": 1,
        "url": args.url,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "stream": not args.non_stream,
        "prompt_preview": args.prompt[:200],
        "cases": summaries,
        "case_definitions": [asdict(case) for case in case_definitions],
    }
    (args.output_dir / "sampling_semantics_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
