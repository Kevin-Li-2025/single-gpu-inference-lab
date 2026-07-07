#!/usr/bin/env python3
"""Emit a vLLM CompilationConfig for preserving the sparse-penalty op."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OP_NAME = "l20_stack::sparse_repetition_penalty_out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=int, default=3)
    parser.add_argument("--cudagraph-mode", default="FULL")
    parser.add_argument("--no-fuse-rope-kvcache", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_payload(args: argparse.Namespace) -> dict:
    pass_config = {}
    if args.no_fuse_rope_kvcache:
        pass_config["fuse_rope_kvcache"] = False
    return {
        "mode": args.mode,
        "cudagraph_mode": args.cudagraph_mode,
        "custom_ops": ["none", f"+{OP_NAME}"],
        "splitting_ops": [OP_NAME],
        "pass_config": pass_config,
    }


def main() -> int:
    args = parse_args()
    payload = build_payload(args)
    serialized = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
