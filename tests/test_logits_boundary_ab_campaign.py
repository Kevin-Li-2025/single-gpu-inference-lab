from pathlib import Path


SCRIPT = Path("scripts/run_vllm_l20_logits_boundary_ab_campaign.sh")


def read_source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_ab_campaign_accepts_only_output_dir_and_records_config():
    source = read_source()

    assert "if [[ $# -ne 1 ]]" in source
    assert 'echo "usage: $0 OUTPUT_DIR"' in source
    assert "ab-config.json" in source
    assert '"schema_version": 1' in source
    assert '"campaign": "l20_logits_boundary_ab"' in source
    assert '"baseline_trace_only"' in source
    assert '"sampler_boundary_candidate"' in source


def test_ab_campaign_uses_safe_l20_defaults_from_env():
    source = read_source()

    assert 'model=${MODEL:-"/home/hhai/models/Qwen2.5-Coder-1.5B-Instruct"}' in source
    assert 'served_name=${SERVED_NAME:-"qwen25-coder-1p5b"}' in source
    assert 'vllm_source_tree=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}' in source
    assert 'inputs=${INPUTS:-"512"}' in source
    assert 'concurrencies=${CONCURRENCIES:-"1 16"}' in source
    assert "runs=${RUNS:-1}" in source
    assert "num_prompts=${NUM_PROMPTS:-32}" in source
    assert "output_tokens=${OUTPUT_TOKENS:-32}" in source
    assert "port=${PORT:-8000}" in source
    assert "max_model_len=${MAX_MODEL_LEN:-2048}" in source
    assert "gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}" in source


def test_ab_campaign_reuses_existing_campaign_scripts_without_benchmark_logic():
    source = read_source()

    assert "run_vllm_l20_logits_boundary_trace_campaign.sh" in source
    assert "run_vllm_l20_sampling_campaign.sh" in source
    assert '"sampler_mode": "l20"' in source
    assert " vllm serve " not in source
    assert " bench serve " not in source
    assert "for concurrency in" not in source
    assert "for input_tokens in" not in source


def test_ab_campaign_passes_identical_shapes_to_both_modes():
    source = read_source()

    for assignment in (
        'PORT="$port"',
        'INPUTS="$inputs"',
        'CONCURRENCIES="$concurrencies"',
        'RUNS="$runs"',
        'NUM_PROMPTS="$num_prompts"',
        'OUTPUT_TOKENS="$output_tokens"',
        'MAX_MODEL_LEN="$max_model_len"',
        'GPU_MEMORY_UTILIZATION="$gpu_memory_utilization"',
    ):
        assert source.count(assignment) == 2

    assert '"$baseline_dir"' in source
    assert '"$candidate_dir"' in source
    assert "baseline-trace-only" in source
    assert "sampler-boundary-candidate" in source


def test_ab_campaign_writes_summary_outputs_after_both_modes():
    source = read_source()

    assert "summarize_l20_logits_boundary_ab.py" in source
    assert "--baseline-dir" in source
    assert "$baseline_dir" in source
    assert "--candidate-dir" in source
    assert "$candidate_dir" in source
    assert "--output-json" in source
    assert "summary.json" in source
    assert "--output-md" in source
    assert "README.md" in source
    assert "--min-runs-per-shape" in source
    assert "$min_runs_per_shape" in source


def test_ab_campaign_drops_server_logs_unless_requested():
    source = read_source()

    assert "keep_server_logs=${KEEP_SERVER_LOGS:-0}" in source
    assert "cleanup_server_logs()" in source
    assert 'if [[ "$keep_server_logs" != "1" ]]' in source
    assert 'rm -f "$baseline_dir/server.log"' in source
    assert 'rm -f "$candidate_dir/server.log"' in source
    assert "trap cleanup_server_logs EXIT" in source
