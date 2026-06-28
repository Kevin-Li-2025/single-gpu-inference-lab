from pathlib import Path


def test_sampling_campaign_switches_flashinfer_sampler():
    source = Path("scripts/run_vllm_l20_sampling_campaign.sh").read_text()

    assert "SAMPLER_MODE: flashinfer|torch|l20" in source
    assert "VLLM_USE_FLASHINFER_SAMPLER=\"$use_flashinfer_sampler\"" in source
    assert "VLLM_L20_TOPK_TOPP_SAMPLER" in source
    assert "install_l20_topk_topp_sampler.py" in source
    assert "--uninstall" in source
    assert "prewarm_l20_topk_topp_sampling.py" in source
    assert "L20_TRACE" in source
    assert "--batch 4" in source
    assert "l20-topk-topp-summary.json" in source
    assert "python_dir=$(dirname" in source
    assert '"$python_dir/ninja"' in source
    assert "export CUDA_HOME=" in source
    assert "export CUDACXX=" in source
    assert "scripts/prewarm_flashinfer_sampling.py" in source
    assert "flashinfer-prewarm.stderr" in source
    assert "prewarm_failed" in source
    assert "No FlashInfer stochastic serving ITL claim" in source
    assert "write_sampling_failure_report" in source
    assert "server_start_failed" in source
    assert "server_process_exited_before_health" in source
    assert "health_check_timeout" in source
    assert "--temperature \"$temperature\"" in source
    assert "--top-p \"$top_p\"" in source
    assert "--top-k \"$top_k\"" in source
    assert "--max-model-len \"$max_model_len\"" in source
    assert "--gpu-memory-utilization \"$gpu_memory_utilization\"" in source
    assert "--generation-config vllm" in source
    assert "--percentile-metrics ttft,tpot,itl,e2el" in source
    assert "scripts/inspect_vllm_sampling_path.py" in source
    assert "PYTHONPATH=\"$extra_vllm_pythonpath:$(pwd)/src" in source


def test_sampling_path_inspector_reports_cpu_fallback_evidence():
    source = Path("scripts/inspect_vllm_sampling_path.py").read_text()

    assert '"flashinfer"' in source
    assert '"fallback"' in source
    assert '"cpu"' in source
    assert "cpu_fallback_suspected" in source


def test_flashinfer_prewarm_reports_structured_errors():
    source = Path("scripts/prewarm_flashinfer_sampling.py").read_text()

    assert '"status": "error"' in source
    assert "traceback_tail" in source
    assert "return 1" in source


def test_l20_prewarm_covers_vllm_rng_sampler_kernel():
    source = Path("scripts/prewarm_l20_topk_topp_sampling.py").read_text()

    assert "topk_topp_sample_with_vllm_rng_out" in source
    assert "expanded_idx_mapping" in source
    assert "seeds" in source
    assert "positions" in source
    assert "vllm_rng_output_shape" in source
