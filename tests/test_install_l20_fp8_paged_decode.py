from pathlib import Path


def test_l20_fp8_paged_decode_installer_patches_flashinfer_decode():
    source = Path("integrations/vllm/install_l20_fp8_paged_decode.py").read_text()

    assert "l20_paged_split_kv_attention_fp8" in source
    assert "VLLM_ENABLE_L20_FP8_PAGED_DECODE" in source
    assert "VLLM_L20_FP8_PAGED_TRACE" in source
    assert "l20_fp8_batch >= 8" in source
    assert "l20_fp8_max_seq >= 4096" in source
    assert "is_quantized_kv_cache(self.kv_cache_dtype)" in source


def test_l20_fp8_paged_serving_campaign_uses_fp8_kv_cache():
    source = Path("scripts/run_vllm_l20_fp8_paged_campaign.sh").read_text()

    assert "--kv-cache-dtype fp8" in source
    assert "--calculate-kv-scales" in source
    assert "PYTHONPATH=\"$vllm_source_dir\"" in source
    assert "VLLM_ENABLE_L20_FP8_PAGED_DECODE" in source
