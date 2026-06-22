#!/usr/bin/env python3
"""Build and benchmark the CUDA SM89 paged-decode prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def latency_ms(function, warmup=10, iterations=50):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def l20_split_policy(batch, context):
    if batch <= 4:
        return 128
    return 512


def should_use_l20_cuda_paged_decode(batch, context):
    return batch == 1 or (batch <= 4 and context <= 512)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/l20-paged-cuda"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.build_dir.mkdir(parents=True, exist_ok=True)
    extension = load(
        "l20_paged_decode_cuda",
        [
            root / "integrations/vllm/cuda/l20_paged_decode.cpp",
            root / "integrations/vllm/cuda/l20_paged_decode.cu",
        ],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
        build_directory=args.build_dir,
    )
    import flashinfer

    reports = []
    for batch in (1, 4):
        for context in (512, 2048, 4096):
            page_size = 16
            pages = context // page_size
            num_pages = batch * pages
            block_table = torch.randperm(
                num_pages, device="cuda", dtype=torch.int32
            ).reshape(batch, pages)
            indptr = (
                torch.arange(batch + 1, device="cuda", dtype=torch.int32) * pages
            )
            last_page_len = torch.full(
                (batch,), page_size, device="cuda", dtype=torch.int32
            )
            seq_lens = torch.full(
                (batch,), context, device="cuda", dtype=torch.int32
            )
            query = torch.randn(
                batch, 16, 128, device="cuda", dtype=torch.float16
            )
            cache = (
                torch.randn(
                    num_pages, page_size, 8, 128, device="cuda", dtype=torch.float16
                ),
                torch.randn(
                    num_pages, page_size, 8, 128, device="cuda", dtype=torch.float16
                ),
            )
            workspace = torch.empty(
                128 * 1024 * 1024, device="cuda", dtype=torch.uint8
            )
            wrappers = {}
            for tensor_cores in (False, True):
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace, "NHD", use_tensor_cores=tensor_cores
                )
                wrapper.plan(
                    indptr,
                    block_table.flatten(),
                    last_page_len,
                    16,
                    8,
                    128,
                    page_size,
                    pos_encoding_mode="NONE",
                    q_data_type=query.dtype,
                    kv_data_type=query.dtype,
                )
                wrappers[tensor_cores] = wrapper
            expected = wrappers[False].run(query, cache)
            actual = extension.paged_decode(
                query, cache[0], cache[1], block_table, seq_lens
            )
            split_available = hasattr(extension, "paged_decode_split")
            flashinfer_ms = latency_ms(lambda: wrappers[False].run(query, cache))
            flashinfer_tensor_core_ms = latency_ms(
                lambda: wrappers[True].run(query, cache)
            )
            cuda_ms = latency_ms(
                lambda: extension.paged_decode(
                    query, cache[0], cache[1], block_table, seq_lens
                )
            )
            split_reports = []
            for split_size in (128, 256, 512, 1024):
                num_splits = (context + split_size - 1) // split_size
                partial_output = torch.empty(
                    batch,
                    16,
                    num_splits,
                    128,
                    device="cuda",
                    dtype=query.dtype,
                )
                partial_max = torch.empty(
                    batch, 16, num_splits, device="cuda", dtype=torch.float32
                )
                partial_sum = torch.empty_like(partial_max)
                split_output = torch.empty_like(query)
                split_actual = extension.paged_decode_split(
                    query,
                    cache[0],
                    cache[1],
                    block_table,
                    seq_lens,
                    context,
                    split_size,
                )
                split_ms = latency_ms(
                    lambda split_size=split_size: extension.paged_decode_split(
                        query,
                        cache[0],
                        cache[1],
                        block_table,
                        seq_lens,
                        context,
                        split_size,
                    )
                )
                workspace_ms = latency_ms(
                    lambda split_size=split_size: extension.paged_decode_split_out(
                        query,
                        cache[0],
                        cache[1],
                        block_table,
                        seq_lens,
                        partial_output,
                        partial_max,
                        partial_sum,
                        split_output,
                        context,
                        split_size,
                    )
                )
                indices_ms = latency_ms(
                    lambda split_size=split_size: extension.paged_decode_split_indices_out(
                        query,
                        cache[0],
                        cache[1],
                        indptr,
                        block_table.flatten(),
                        seq_lens,
                        partial_output,
                        partial_max,
                        partial_sum,
                        split_output,
                        context,
                        split_size,
                    )
                )
                split_reports.append(
                    {
                        "split_size": split_size,
                        "correct": bool(
                            torch.allclose(
                                split_actual,
                                expected,
                                rtol=2e-2,
                                atol=2e-2,
                            )
                        ),
                        "cuda_ms": split_ms,
                        "speedup": flashinfer_ms / split_ms,
                        "workspace_ms": workspace_ms,
                        "workspace_speedup": flashinfer_ms / workspace_ms,
                        "indices_ms": indices_ms,
                        "indices_vs_block_table": workspace_ms / indices_ms,
                    }
                )
            best_split = min(split_reports, key=lambda item: item["cuda_ms"])
            policy_split = l20_split_policy(batch, context)
            reports.append(
                {
                    "batch": batch,
                    "context": context,
                    "correct": bool(
                        torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
                    ),
                    "max_abs_error": float(
                        (actual.float() - expected.float()).abs().max()
                    ),
                    "flashinfer_ms": flashinfer_ms,
                    "flashinfer_tensor_core_ms": flashinfer_tensor_core_ms,
                    "flashinfer_tensor_core_vs_cuda_core": (
                        flashinfer_ms / flashinfer_tensor_core_ms
                    ),
                    "cuda_ms": cuda_ms,
                    "speedup": flashinfer_ms / cuda_ms,
                    "split_reports": split_reports,
                    "best_split_size": best_split["split_size"],
                    "policy_split_size": policy_split,
                    "split_correct": all(
                        report["correct"] for report in split_reports
                    ),
                    "split_cuda_ms": best_split["cuda_ms"],
                    "split_speedup": best_split["speedup"],
                }
            )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
