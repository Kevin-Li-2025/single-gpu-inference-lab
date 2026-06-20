#!/usr/bin/env python3
"""Generate an L20 residual RMSNorm dispatch policy from benchmark reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l20_stack.ops.rmsnorm_policy import (
    build_residual_rmsnorm_policy,
    load_reports,
    policy_payload,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--minimum-margin-pct",
        type=float,
        default=2.0,
        help="mark a non-eager winner unstable unless it beats the next production provider by this margin",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = load_reports(args.reports)
    policies = build_residual_rmsnorm_policy(
        reports, minimum_margin_pct=args.minimum_margin_pct
    )
    payload = policy_payload(policies)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    mismatches = [
        policy
        for policy in policies
        if policy.stable and policy.recommended_backend != policy.current_backend
    ]
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
