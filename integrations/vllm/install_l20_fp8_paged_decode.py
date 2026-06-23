#!/usr/bin/env python3
"""Install the gated L20 paged FP8 decode path into vLLM FlashInfer backend."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"cannot find patch point: {label}")
    return source.replace(old, new, 1)


def find_vllm_package(source_dir: str | None) -> Path:
    if source_dir:
        package = Path(source_dir).expanduser().resolve() / "vllm"
        if (package / "v1/attention/backends/flashinfer.py").exists():
            return package
    for candidate in (
        Path("/home/hhai/vllm-l20-upstream/vllm"),
        Path.cwd() / "vllm",
    ):
        if (candidate / "v1/attention/backends/flashinfer.py").exists():
            return candidate
    raise RuntimeError("cannot find vLLM source package")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-source-dir")
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()

    package = find_vllm_package(args.vllm_source_dir)
    backend = package / "v1/attention/backends/flashinfer.py"
    backup = backend.with_suffix(".py.l20-fp8-paged-backup")
    if args.uninstall:
        if backup.exists():
            shutil.copy2(backup, backend)
        return 0
    if not backup.exists():
        shutil.copy2(backend, backup)

    root = Path(__file__).resolve().parents[2]
    op_dir = package / "v1/attention/ops"
    shutil.copy2(root / "integrations/vllm/l20_paged_split_kv.py", op_dir)

    source = backend.read_text(encoding="utf-8")
    old_call = """                        decode_wrapper.run(
                            decode_query,
                            kv_cache_permute,
                            k_scale=layer._k_scale_float,
                            v_scale=layer._v_scale_float,
                            out=out_decode,
                            kv_cache_sf=kv_cache_sf,
                        )
"""
    new_call = """                        l20_fp8_batch = decode_query.shape[0]
                        l20_fp8_max_seq = attn_metadata.decode.max_seq_len
                        l20_fp8_enabled = (
                            os.getenv("VLLM_ENABLE_L20_FP8_PAGED_DECODE", "0") == "1"
                            and current_platform.get_device_capability()
                            == DeviceCapability(8, 9)
                            and not torch.cuda.is_current_stream_capturing()
                            and not self.is_kvcache_nvfp4
                            and is_quantized_kv_cache(self.kv_cache_dtype)
                            and decode_query.shape[-1] == 128
                            and decode_query.shape[1] == 16
                            and kv_cache_permute.shape[1] == 2
                            and kv_cache_permute.shape[2] == 16
                            and kv_cache_permute.shape[3] == 8
                            and kv_cache_permute.shape[4] == 128
                            and l20_fp8_batch >= 8
                            and l20_fp8_max_seq >= 4096
                        )
                        if l20_fp8_enabled:
                            from vllm.v1.attention.ops.l20_paged_split_kv import (
                                allocate_l20_paged_split_kv_workspace,
                                l20_paged_split_kv_attention_fp8,
                            )

                            l20_fp8_shape = (
                                l20_fp8_batch,
                                decode_query.shape[1],
                                (l20_fp8_max_seq + 511) // 512,
                            )
                            if getattr(self, "_l20_fp8_workspace_shape", None) != l20_fp8_shape:
                                self._l20_fp8_workspace = (
                                    allocate_l20_paged_split_kv_workspace(
                                        decode_query,
                                        l20_fp8_max_seq,
                                        split_size=512,
                                    )
                                )
                                self._l20_fp8_workspace_shape = l20_fp8_shape
                            key_cache, value_cache = kv_cache_permute.unbind(1)
                            l20_paged_split_kv_attention_fp8(
                                decode_query,
                                key_cache,
                                value_cache,
                                attn_metadata.decode.block_tables,
                                attn_metadata.decode.seq_lens,
                                k_scale=layer._k_scale_float,
                                v_scale=layer._v_scale_float,
                                max_seq_len=l20_fp8_max_seq,
                                split_size=512,
                                output=out_decode.view_as(decode_query),
                                workspace=self._l20_fp8_workspace,
                            )
                            trace_path = os.getenv("VLLM_L20_FP8_PAGED_TRACE")
                            if trace_path:
                                with open(trace_path, "a", encoding="utf-8") as trace_file:
                                    trace_file.write(
                                        json.dumps(
                                            {
                                                "event": "l20_fp8_paged_decode_run",
                                                "batch": int(l20_fp8_batch),
                                                "max_seq_len": int(l20_fp8_max_seq),
                                                "q_heads": int(decode_query.shape[1]),
                                                "kv_heads": int(kv_cache_permute.shape[3]),
                                            },
                                            sort_keys=True,
                                        )
                                        + "\\n"
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
    source = replace_once(source, old_call, new_call, "FlashInfer decode call")
    backend.write_text(source, encoding="utf-8")
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
