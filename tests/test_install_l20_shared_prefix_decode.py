from pathlib import Path


def test_shared_prefix_decode_installer_copies_ops_into_vllm_namespace():
    source = Path("integrations/vllm/install_l20_shared_prefix_decode.py").read_text()
    assert '"l20_decode_attention.py"' in source
    assert '"l20_shared_prefix_decode_dispatch.py"' in source
    assert "src/l20_stack/ops/triton_decode_attention.py" in source
    assert "integrations/vllm/l20_shared_prefix_decode_dispatch.py" in source
    assert "VLLM_SOURCE_TREE" in source
    assert "--uninstall" in source
    assert ".l20-shared-prefix-backup" in source


def test_shared_prefix_decode_dispatch_is_env_gated_and_conservative():
    source = Path("integrations/vllm/l20_shared_prefix_decode_dispatch.py").read_text()
    assert "VLLM_ENABLE_L20_SHARED_PREFIX_DECODE" in source
    assert "torch.cuda.get_device_capability() != (8, 9)" in source
    assert "torch.cuda.is_current_stream_capturing()" in source
    assert "min_prefix_length: int = 4096" in source
    assert "min_batch: int = 8" in source
    assert "find_l20_shared_prefix_groups" in source
    assert "shared_paged_prefix_suffix_gqa_decode_attention" in source
    assert "len(groups) == 1 and len(groups[0]) == query.shape[0]" in source


def test_vllm_shared_prefix_smoke_uses_vllm_import_path():
    source = Path("scripts/smoke_vllm_l20_shared_prefix_decode.py").read_text()
    assert "from vllm.v1.attention.ops.l20_shared_prefix_decode_dispatch import" in source
    assert "maybe_l20_shared_prefix_decode" in source
    assert "should_dispatch_l20_shared_prefix_decode" in source
    assert "find_l20_shared_prefix_groups" in source
    assert "VLLM_ENABLE_L20_SHARED_PREFIX_DECODE" in source
    assert "l20_stack.ops" not in source
