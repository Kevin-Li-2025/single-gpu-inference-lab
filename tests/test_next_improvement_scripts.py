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
    assert "NCU_OUTPUT_PREFIX" in runner
    assert "--section MemoryWorkloadAnalysis" in runner
    assert "ncu-status.json" in runner
    assert "ERR_NVGPUCTRPERM" in runner
    assert 'env PATH="$PATH" PYTHONPATH="$PYTHONPATH"' in runner
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


def test_top_tier_gap_doc_tracks_profiling_cuda_and_upstream():
    doc = Path("docs/l20-top-tier-kernel-gaps.md").read_text()
    readme = Path("README.md").read_text()
    assert "Complete Profiling Package" in doc
    assert "Nsight Systems timeline" in doc
    assert "Nsight Compute roofline" in doc
    assert "Occupancy report" in doc
    assert "Warp-stall breakdown" in doc
    assert "Shared-memory table" in doc
    assert "FlashAttention-style decode/prefill" in doc
    assert "PagedAttention" in doc
    assert "MoE routing" in doc
    assert "Grouped GEMM" in doc
    assert "Upstream Track" in doc
    assert "vLLM" in doc and "FlashInfer" in doc and "TensorRT-LLM" in doc
    assert "docs/l20-top-tier-kernel-gaps.md" in readme


def test_qk_norm_benchmark_can_import_repo_local_kernel():
    source = Path("scripts/benchmark_qk_norm_rope_kv.py").read_text()
    assert "import_source" in source
    assert "integrations/vllm" in source


def test_ncu_summary_accepts_cuda_13_stall_metric_names():
    source = Path("scripts/summarize_ncu_profile.py").read_text()
    assert "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio" in source
    assert "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio" in source
    assert "tensor_pipe_utilization_pct" in source
    assert "sm__pipe_tensor_cycles_active_v2" in source


def test_paged_decode_rfc_campaign_tracks_o2_and_flashinfer():
    campaign = Path("scripts/run_vllm_l20_paged_decode_rfc_campaign.sh").read_text()
    matrix = Path("scripts/run_vllm_l20_paged_decode_rfc_matrix.sh").read_text()
    summary = Path("scripts/summarize_l20_paged_decode_rfc_matrix.py").read_text()
    assert "EXECUTION_MODE: eager|o2" in campaign
    assert "--attention-backend" in campaign
    assert "FLASHINFER" in campaign
    assert "--enforce-eager" in campaign
    assert "cudagraph_mode" in campaign
    assert "VLLM_L20_PAGED_DECODE" in campaign
    assert "VLLM_L20_PAGED_DECODE_CUDAGRAPH" in campaign
    assert "l20-paged-decode-trace.txt" in campaign
    assert "run-config.json" in campaign
    assert "REQUEST_RATE" in campaign
    assert "--max-concurrency" in campaign
    assert "eager o2" in matrix
    assert "summarize_l20_paged_decode_rfc_matrix.py" in matrix
    assert "trace_hit_count" in summary
    assert "cudagraph_disabled" in summary
    assert "AttentionBackendEnum.FLASHINFER" in summary


def test_paged_decode_rfc_campaign_preserves_compilation_config_json():
    campaign = Path("scripts/run_vllm_l20_paged_decode_rfc_campaign.sh").read_text()
    assert "compilation_config=${COMPILATION_CONFIG:-" in campaign
    assert '--compilation-config "$compilation_config"' in campaign
    assert '--compilation-config "${COMPILATION_CONFIG:-' not in campaign


def test_qk_norm_serving_smoke_compares_o2_fusion_gate():
    source = Path("scripts/run_vllm_l20_qk_norm_rope_serving_smoke.sh").read_text()
    matrix = Path("scripts/run_vllm_l20_qk_norm_rope_serving_matrix.sh").read_text()
    assert "enable_qk_norm_rope_fusion" in source
    assert "qk-off" in source
    assert "qk-on" in source
    assert "cudagraph_mode" in source
    assert "qk-serving-summary.json" in source
    assert "flashinfer_sampling" in source
    assert "shape_summaries" in source
    assert "max_concurrency" in source
    assert "REQUEST_RATE" in matrix
    assert "run_vllm_l20_qk_norm_rope_serving_smoke.sh" in matrix


def test_decode_attention_has_isolated_tensor_core_candidate():
    op_source = Path("src/l20_stack/ops/triton_decode_attention.py").read_text()
    sweep_source = Path("scripts/benchmark_decode_attention_tile_sweep.py").read_text()
    profile_source = Path("scripts/profile_decode_attention_ncu.py").read_text()
    shared_source = Path("scripts/benchmark_shared_prefix_decode_attention.py").read_text()
    suffix_source = Path(
        "scripts/benchmark_shared_prefix_suffix_decode_attention.py"
    ).read_text()
    paged_suffix_source = Path(
        "scripts/benchmark_shared_paged_prefix_suffix_decode_attention.py"
    ).read_text()
    assert "gqa_decode_attention_split_kv_tensor_core_candidate" in op_source
    assert "gqa_decode_attention_split_kv_tensor_core_dsplit_candidate" in op_source
    assert "gqa_decode_attention_split_kv_bf16_partials_candidate" in op_source
    assert "tl.dot(query_values, tl.trans(keys))" in op_source
    assert "path\": \"tensor_core_candidate\"" in sweep_source
    assert "path\": \"tensor_core_dsplit_candidate\"" in sweep_source
    assert "path\": \"bf16_partials_candidate\"" in sweep_source
    assert "--include-bf16-partials" in sweep_source
    assert "--tensor-core-block-qs" in sweep_source
    assert "--tensor-core-dsplit-block-ds" in sweep_source
    assert '"tensor-core-candidate"' in profile_source
    assert '"tensor-core-dsplit-candidate"' in profile_source
    assert '"bf16-partials-candidate"' in profile_source
    assert "shared_prefix_gqa_decode_attention" in op_source
    assert "_shared_prefix_gqa_attention_kernel" in op_source
    assert "shared_prefix_packed" in shared_source
    assert "per_request_split_kv" in shared_source
    assert "shared_prefix_suffix_gqa_decode_attention" in op_source
    assert "_shared_prefix_gqa_partial_kernel" in op_source
    assert "_prefix_suffix_gqa_reduce_kernel" in op_source
    assert "shared_prefix_suffix_merge" in suffix_source
    assert "per_request_full_split_kv" in suffix_source
    assert "shared_paged_prefix_suffix_gqa_decode_attention" in op_source
    assert "_shared_paged_prefix_gqa_partial_kernel" in op_source
    assert "shared_paged_prefix_paged_suffix_gqa_decode_attention" in op_source
    assert "_paged_suffix_gqa_partial_kernel" in op_source
    assert "shared_prefix_suffix_paged" in paged_suffix_source
    assert "shared_prefix_paged_suffix_paged" in paged_suffix_source
    assert "paged_over_contiguous" in paged_suffix_source
    assert "paged_suffix_speedup_vs_baseline" in paged_suffix_source
