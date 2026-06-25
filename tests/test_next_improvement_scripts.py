from pathlib import Path


def test_spec_acceptance_campaign_is_version_tolerant():
    source = Path("scripts/run_vllm_l20_spec_acceptance_campaign.sh").read_text()
    assert "SPECULATIVE_ARGS" in source
    assert "summarize_spec_decode_acceptance.py" in source
    assert "spec-acceptance-summary.json" in source
    assert 'export PATH="$python_dir:$PATH"' in source


def test_spec_acceptance_summary_extracts_acceptance_evidence():
    source = Path("scripts/summarize_spec_decode_acceptance.py").read_text()
    assert "acceptance_rate" in source
    assert "accepted_tokens" in source
    assert "draft_tokens" in source
    assert "inferred_acceptance_rate_from_tokens" in source


def test_multiturn_kv_pressure_uses_streaming_openai_endpoint():
    source = Path("scripts/benchmark_multiturn_kv_pressure.py").read_text()
    runner = Path("scripts/run_vllm_l20_kv_pressure_campaign.sh").read_text()
    matrix = Path("scripts/run_vllm_l20_kv_pressure_matrix.sh").read_text()
    summary = Path("scripts/summarize_kv_pressure.py").read_text()
    assert '"/v1/completions"' in source
    assert '"stream": True' in source
    assert "ttft_ms" in source
    assert "prefix_chars" in source
    assert "PREFIX_CACHING" in runner
    assert "KV_CACHE_DTYPE" in runner
    assert "CALCULATE_KV_SCALES" in runner
    assert "FLASHINFER_SAMPLER" in runner
    assert "kv-pressure-failure.json" in runner
    assert "MAX_MODEL_LEN" in runner
    assert "--enforce-eager" in runner
    assert "kv-pressure-prefix-cache" in runner
    assert 'export PATH="$python_dir:$PATH"' in runner
    assert "KV_DTYPES" in matrix
    assert "auto fp8" in matrix
    assert "summarize_kv_pressure.py" in matrix
    assert "late_over_first_ttft" in summary
    assert "best_median_ttft_ms" in summary
    assert "build_comparisons" in summary
    assert "paired_run_count" in summary
    assert "median_ttft_speedup_range" in summary
    assert "median_ttft_speedup_fp8_over_auto" in summary
    assert "last_turn_ttft_speedup_fp8_over_auto" in summary


def test_next_improvement_doc_tracks_all_five_workstreams():
    doc = Path("docs/l20-next-improvements.md").read_text()
    assert "Q/K Norm + Q/K RoPE + KV Write Fusion" in doc
    assert "FP8 KV Fused Attention Kernel Boundary" in doc
    assert "vLLM FlashInfer Sampling Route Hardening" in doc
    assert "Spec Decode Acceptance-Rate Study" in doc
    assert "Multi-Turn KV Pressure Benchmark" in doc


def test_qk_norm_benchmark_can_import_repo_local_kernel():
    source = Path("scripts/benchmark_qk_norm_rope_kv.py").read_text()
    assert "import_source" in source
    assert "integrations/vllm" in source
