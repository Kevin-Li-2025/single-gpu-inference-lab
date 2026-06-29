from pathlib import Path


def test_sampling_winner_matrix_runs_paired_production_modes():
    source = Path("scripts/run_vllm_l20_sampling_winner_matrix.sh").read_text()

    assert "torch flashinfer" in source
    assert "run_vllm_l20_sampling_campaign.sh" in source
    assert "summarize_l20_sampling_winner.py" in source
    assert "FlashInfer must reduce median ITL" in source
    assert "increase output throughput" in source


def test_sampling_winner_matrix_expands_l20_serving_shapes():
    source = Path("scripts/run_vllm_l20_sampling_winner_matrix.sh").read_text()

    assert 'inputs=${INPUTS:-"128 512 2048"}' in source
    assert 'concurrencies=${CONCURRENCIES:-"1 2 4 8"}' in source
    assert "Qwen2.5-Coder-1.5B-Instruct" in source
    assert "Qwen3-0.6B" in source
    assert "Qwen3-1.7B" in source


def test_sampling_winner_matrix_keeps_logs_out_of_artifacts_by_default():
    source = Path("scripts/run_vllm_l20_sampling_winner_matrix.sh").read_text()

    assert "KEEP_SERVER_LOGS" in source
    assert 'rm -f "$run_dir/server.log"' in source
    assert "matrix-config.json" in source
