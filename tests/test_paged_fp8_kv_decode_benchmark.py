from pathlib import Path


def test_paged_fp8_kv_decode_benchmark_uses_paged_fp8_operator():
    source = Path("scripts/benchmark_paged_fp8_kv_decode_attention.py").read_text()

    assert "torch.float8_e4m3fn" in source
    assert "l20_paged_split_kv_attention_fp8" in source
    assert "l20_fp8_materialize_dequant_then_paged" in source
    assert "l20_fp8_fused_dequant_paged" in source
    assert "fused_fp8_vs_materialized_fp8" in source
    assert "--q-heads" in source
    assert "--kv-heads" in source


def test_paged_fp8_operator_entrypoint_exists():
    source = Path("integrations/vllm/l20_paged_split_kv.py").read_text()

    assert "_paged_split_kv_fp8_partial" in source
    assert "def l20_paged_split_kv_attention_fp8" in source
    assert "key/value cache must use a torch FP8 dtype" in source
