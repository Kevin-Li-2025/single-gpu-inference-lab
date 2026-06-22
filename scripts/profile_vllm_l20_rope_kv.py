#!/usr/bin/env python3
"""Compile the vLLM L20 RoPE/KV kernel and report SM89 resource usage."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def metadata_value(metadata, *names):
    for name in names:
        value = getattr(metadata, name, None)
        if value is not None:
            return int(value)
    return None


def theoretical_occupancy(num_warps, num_regs, shared_bytes):
    limits = {"architectural_blocks": 24, "warp_blocks": 48 // num_warps}
    threads = num_warps * 32
    if num_regs:
        limits["register_blocks"] = 65536 // (num_regs * threads)
    if shared_bytes:
        limits["shared_memory_blocks"] = (100 * 1024) // shared_bytes
    resident_blocks = max(0, min(limits.values()))
    resident_warps = min(48, resident_blocks * num_warps)
    return {
        "limiting_resource": min(limits, key=limits.get),
        "resident_blocks_per_sm": resident_blocks,
        "resident_warps_per_sm": resident_warps,
        "theoretical_occupancy_pct": round(resident_warps / 48 * 100, 2),
        "block_limits": limits,
    }


def cubin_resource_usage(compiled):
    cubin = compiled.asm.get("cubin")
    if not cubin:
        return {}
    with tempfile.NamedTemporaryFile(suffix=".cubin") as handle:
        handle.write(cubin)
        handle.flush()
        output = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", handle.name],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    values = {}
    for label, value in re.findall(r"\b(REG|STACK|SHARED|LOCAL):(\d+)", output):
        values[label.lower()] = int(value)
    return values


def compile_shape(torch, triton, kernel, shape):
    tokens = shape["tokens"]
    q_heads = shape["q_heads"]
    kv_heads = shape["kv_heads"]
    head_dim = shape["head_dim"]
    block = triton.next_power_of_2(head_dim)
    num_warps = 4 if head_dim >= 128 else 2
    dtype = torch.float16
    query = torch.empty(tokens, q_heads, head_dim, device="cuda", dtype=dtype)
    key = torch.empty(tokens, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.empty_like(key)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cos_sin = torch.empty(4096, head_dim, device="cuda", dtype=torch.float32)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cache = torch.empty(16, 16, kv_heads, head_dim, device="cuda", dtype=dtype)
    compiled = kernel.warmup(
        query,
        key,
        value,
        positions,
        cos_sin,
        slots,
        cache,
        cache.clone(),
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cos_sin.stride(0),
        tokens,
        q_heads,
        kv_heads,
        head_dim,
        head_dim,
        cache.shape[1],
        True,
        BLOCK_SIZE=block,
        num_warps=num_warps,
        num_stages=1,
        grid=(tokens, q_heads),
    )
    metadata = compiled.metadata
    resources = cubin_resource_usage(compiled)
    num_regs = resources.get("reg") or metadata_value(metadata, "num_regs", "n_regs")
    shared = resources.get("shared") or metadata_value(metadata, "shared", "shared_memory") or 0
    spills = metadata_value(metadata, "num_spills", "n_spills")
    return shape | {
        "block_size": block,
        "num_warps": num_warps,
        "num_stages": 1,
        "registers_per_thread": num_regs,
        "shared_memory_bytes": shared,
        "stack_bytes": resources.get("stack"),
        "local_memory_bytes": resources.get("local"),
        "spills": spills,
        "occupancy_estimate": theoretical_occupancy(num_warps, num_regs, shared),
    }


def main() -> int:
    args = parse_args()
    import torch
    import triton
    from vllm.v1.attention.ops.l20_rope_kv import _l20_rope_kv_kernel

    if torch.cuda.get_device_name() != "NVIDIA L20":
        raise SystemExit("profile requires NVIDIA L20")
    shapes = [
        {"tokens": tokens, "q_heads": q_heads, "kv_heads": kv_heads, "head_dim": dim}
        for tokens in (1, 8, 16, 64)
        for q_heads, kv_heads, dim in ((14, 2, 64), (12, 2, 128), (32, 4, 128), (32, 8, 128), (16, 4, 256))
    ]
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "notes": [
            "Occupancy is an architectural upper-bound estimate, not measured active warps.",
            "DRAM, L2, coalescing, and stall metrics require Nsight Compute and are not inferred here.",
        ],
        "shapes": [compile_shape(torch, triton, _l20_rope_kv_kernel, shape) for shape in shapes],
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
