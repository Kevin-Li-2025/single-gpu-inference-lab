#!/usr/bin/env python3
"""Summarize logits-boundary A/B intervention artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l20_stack.epilogue.intervention import (
    render_logits_boundary_ab_markdown,
    summarize_logits_boundary_ab,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument(
        "--min-runs-per-shape",
        type=int,
        default=2,
        help="Minimum baseline and candidate reports required per compared shape.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_logits_boundary_ab(
        args.input_dir,
        baseline_dir=args.baseline_dir,
        candidate_dir=args.candidate_dir,
        min_runs_per_shape=args.min_runs_per_shape,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(
            render_logits_boundary_ab_markdown(summary),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
