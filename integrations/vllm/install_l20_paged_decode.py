#!/usr/bin/env python3
"""Build and install the gated L20 CUDA paged-decode path into vLLM."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import vllm
from torch.utils.cpp_extension import load


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"cannot find patch point: {label}")
    return source.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    package = Path(inspect.getfile(vllm)).parent
    backend = package / "v1" / "attention" / "backends" / "flashinfer.py"
    backup = backend.with_suffix(".py.l20-paged-backup")
    if args.uninstall:
        if backup.exists():
            shutil.copy2(backup, backend)
        return 0
    if not backup.exists():
        shutil.copy2(backend, backup)

    root = Path(__file__).resolve().parents[2]
    build_dir = Path("/tmp/l20-paged-vllm-extension")
    build_dir.mkdir(parents=True, exist_ok=True)
    extension = load(
        "l20_paged_decode_cuda",
        [
            root / "integrations/vllm/cuda/l20_paged_decode.cpp",
            root / "integrations/vllm/cuda/l20_paged_decode.cu",
        ],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
        build_directory=build_dir,
    )
    shutil.copy2(extension.__file__, package.parent / "l20_paged_decode_cuda.so")

    source = backend.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "import torch\n",
        "import torch\nimport l20_paged_decode_cuda\n",
        "extension import",
    )
    source = replace_once(
        source,
        '''class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper
''',
        '''class FIDecode:
    """Metadata for the native FlashInfer decode pathway (non-TRTLLM)."""

    wrapper: BatchDecodeWithPagedKVCacheWrapper
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
''',
        "FIDecode metadata",
    )
    source = replace_once(
        source,
        "attn_metadata.decode = FIDecode(wrapper=decode_wrapper)",
        """attn_metadata.decode = FIDecode(
                    wrapper=decode_wrapper,
                    block_tables=block_table_tensor[:num_decodes],
                    seq_lens=seq_lens[:num_decodes],
                    max_seq_len=max_seq_len,
                )""",
        "FIDecode construction",
    )
    old_call = """                    decode_wrapper.run(
                        decode_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_decode,
                        kv_cache_sf=kv_cache_sf,
                    )
"""
    new_call = """                    l20_batch = decode_query.shape[0]
                    l20_max_seq = attn_metadata.decode.max_seq_len
                    l20_enabled = (
                        not torch.cuda.is_current_stream_capturing()
                        and decode_query.dtype == torch.float16
                        and decode_query.shape[-1] == 128
                        and decode_query.shape[1] in (12, 16)
                        and kv_cache_permute.shape[1] == 2
                        and kv_cache_permute.shape[2] == 16
                        and kv_cache_permute.shape[4] == 128
                        and (
                            (
                                decode_query.shape[1] == 16
                                and kv_cache_permute.shape[3] == 8
                            )
                            or (
                                decode_query.shape[1] == 12
                                and kv_cache_permute.shape[3] == 2
                            )
                        )
                        and (
                            (l20_batch == 1 and l20_max_seq <= 2304)
                            or (l20_batch <= 4 and l20_max_seq <= 640)
                        )
                    )
                    if l20_enabled:
                        l20_shape = (4, decode_query.shape[1], 64)
                        if getattr(self, "_l20_workspace_shape", None) != l20_shape:
                            self._l20_partial_output = torch.empty(
                                (*l20_shape, 128),
                                dtype=decode_query.dtype,
                                device=decode_query.device,
                            )
                            self._l20_partial_max = torch.empty(
                                l20_shape,
                                dtype=torch.float32,
                                device=decode_query.device,
                            )
                            self._l20_partial_sum = torch.empty_like(
                                self._l20_partial_max
                            )
                            self._l20_workspace_shape = l20_shape
                        key_cache, value_cache = kv_cache_permute.unbind(1)
                        l20_paged_decode_cuda.paged_decode_split_out(
                            decode_query,
                            key_cache,
                            value_cache,
                            attn_metadata.decode.block_tables,
                            attn_metadata.decode.seq_lens,
                            self._l20_partial_output,
                            self._l20_partial_max,
                            self._l20_partial_sum,
                            out_decode.view_as(decode_query),
                            l20_max_seq,
                            64,
                        )
                    else:
                        decode_wrapper.run(
                            decode_query,
                            kv_cache_permute,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            out=out_decode,
                            kv_cache_sf=kv_cache_sf,
                        )
"""
    source = replace_once(source, old_call, new_call, "decode dispatch")
    backend.write_text(source, encoding="utf-8")
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
