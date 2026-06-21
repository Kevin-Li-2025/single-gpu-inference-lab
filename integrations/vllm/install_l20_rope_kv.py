#!/usr/bin/env python3
"""Install the L20 RoPE/KV fusion experiment into a vLLM 0.23 environment."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import vllm


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"cannot find patch point: {label}")
    return text.replace(old, new, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package = Path(inspect.getfile(vllm)).parent
    integration = Path(__file__).resolve().parent
    targets = {
        "config": package / "config" / "compilation.py",
        "backend": package / "v1" / "attention" / "backends" / "flashinfer.py",
        "triton_backend": package / "v1" / "attention" / "backends" / "triton_attn.py",
        "kernel": package / "v1" / "attention" / "ops" / "l20_rope_kv.py",
    }
    if args.uninstall:
        for name in ("config", "backend", "triton_backend"):
            backup = targets[name].with_suffix(targets[name].suffix + ".l20-backup")
            if backup.exists():
                shutil.copy2(backup, targets[name])
        targets["kernel"].unlink(missing_ok=True)
        return 0

    for name in ("config", "backend", "triton_backend"):
        backup = targets[name].with_suffix(targets[name].suffix + ".l20-backup")
        if not backup.exists():
            shutil.copy2(targets[name], backup)
    shutil.copy2(integration / "l20_rope_kv.py", targets["kernel"])

    config = targets["config"].read_text(encoding="utf-8")
    config = replace_once(
        config,
        "if self.fuse_rope_kvcache and not current_platform.is_rocm():",
        (
            "if (self.fuse_rope_kvcache and not current_platform.is_rocm() "
            "and not current_platform.is_cuda()):"
        ),
        "CUDA config guard",
    )
    config = config.replace(
        "KV cache fusion currently only enabled on ROCm.",
        "KV cache fusion requires ROCm/AITER or the experimental CUDA backend.",
        1,
    )
    targets["config"].write_text(config, encoding="utf-8")

    backend = targets["backend"].read_text(encoding="utf-8")
    backend = replace_once(
        backend,
        "from vllm.v1.attention.ops.merge_attn_states import merge_attn_states\n",
        (
            "from vllm.v1.attention.ops.merge_attn_states import merge_attn_states\n"
            "from vllm.v1.attention.ops.l20_rope_kv import l20_rope_and_cache\n"
        ),
        "kernel import",
    )
    marker = "\n\ndef fast_plan_decode(\n"
    methods = '''

    def fused_rope_kvcache_supported(self):
        capability = current_platform.get_device_capability()
        return (
            capability is not None
            and capability.to_int() == 89
            and self.kv_cache_dtype == "auto"
            and self.kv_sharing_target_layer_name is None
            and self.head_size <= 256
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ) -> None:
        l20_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            kv_cache[:, 0],
            kv_cache[:, 1],
            layer_slot_mapping,
        )
'''
    backend = replace_once(backend, marker, methods + marker, "FlashInfer methods")
    targets["backend"].write_text(backend, encoding="utf-8")

    triton_backend = targets["triton_backend"].read_text(encoding="utf-8")
    triton_backend = replace_once(
        triton_backend,
        (
            "from vllm.v1.attention.ops.triton_prefill_attention import "
            "context_attention_fwd\n"
        ),
        (
            "from vllm.v1.attention.ops.triton_prefill_attention import "
            "context_attention_fwd\n"
            "from vllm.v1.attention.ops.l20_rope_kv import l20_rope_and_cache\n"
        ),
        "Triton backend kernel import",
    )
    old_methods = '''    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
'''
    new_methods = '''    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        capability = current_platform.get_device_capability()
        return (
            rocm_aiter_ops.is_enabled()
            or (
                capability is not None
                and capability.to_int() == 89
                and self.kv_cache_dtype == "auto"
                and self.head_size <= 256
            )
        )

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        key_cache, value_cache = kv_cache.unbind(1)
        if not rocm_aiter_ops.is_enabled():
            l20_rope_and_cache(
                query,
                key,
                value,
                positions,
                cos_sin_cache,
                is_neox,
                key_cache,
                value_cache,
                layer_slot_mapping,
            )
            return

        flash_layout = True
        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )
'''
    triton_backend = replace_once(
        triton_backend, old_methods, new_methods, "Triton backend methods"
    )
    targets["triton_backend"].write_text(triton_backend, encoding="utf-8")

    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
