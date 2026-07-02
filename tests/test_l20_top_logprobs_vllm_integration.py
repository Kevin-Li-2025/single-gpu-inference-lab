import importlib.util
from pathlib import Path


def load_installer():
    path = Path("integrations/vllm/install_l20_top_logprobs.py")
    spec = importlib.util.spec_from_file_location("install_l20_top_logprobs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_l20_top_logprobs_installer_patches_sampler_forward():
    module = load_installer()
    sampler_source = """
from vllm.v1.sample.ops.logprobs import batched_count_greater_than
class Sampler:
    def forward(self, logits, sampling_metadata):
        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == LogprobsMode.RAW_LOGPROBS:
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == LogprobsMode.RAW_LOGITS:
                raw_logprobs = logits.clone()
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        logprobs_tensors = None if num_logprobs is None else \\
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)
"""

    patched = module.patch_sampler(sampler_source)
    patched_twice = module.patch_sampler(patched)

    assert patched_twice == patched
    assert "l20_top_logprobs_enabled" in patched
    assert "maybe_l20_gather_logprobs" in patched
    assert "l20_raw_logits_for_logprobs = logits.clone()" in patched
    assert "raw_logprobs = self.compute_logprobs(" in patched
    assert "l20_raw_logits_for_logprobs" in patched
    assert "self.gather_logprobs(" in patched
    assert "token_ids=sampled" in patched


def test_l20_top_logprobs_helper_is_guarded_and_vllm_shaped():
    source = Path("integrations/vllm/l20_top_logprobs.py").read_text()
    ops = Path("src/l20_stack/ops/triton_sampling.py").read_text()

    assert "VLLM_L20_TOP_LOGPROBS" in source
    assert "VLLM_L20_TOP_LOGPROBS_TRACE" in source
    assert "VLLM_L20_TOP_LOGPROBS_ALLOW_NON_L20" in source
    assert "should_use_l20_logprob_topk" in source
    assert "vllm_top_logprobs_out" in source
    assert "LogprobsTensors" in source
    assert "non_positive_num_logprobs" in source
    assert "output_token_ids" in source
    assert "output_ranks" in source
    assert "eligible" in source
    assert "_vllm_top_logprobs_partial_kernel" in ops
    assert "_vllm_top_logprobs_reduce_kernel" in ops
    assert "vLLM-compatible token logprobs" in ops


def test_a100_top_logprobs_ab_runner_targets_real_logprobs_workload():
    source = Path("scripts/run_vllm_a100_top_logprobs_ab.sh").read_text()
    readme = Path("README.md").read_text()
    status = Path("docs/experiment-status.md").read_text()
    results = Path("benchmarks/results/README.md").read_text()
    smoke = Path(
        "benchmarks/results/a100-vllm-top-logprobs-smoke/"
        "dirty-qwen25-05b-r2/README.md"
    ).read_text()
    clean = Path(
        "benchmarks/results/a100-vllm-top-logprobs-clean/"
        "qwen25-05b-r30/README.md"
    ).read_text()

    assert "install_l20_top_logprobs.py" in source
    assert "sample_topk_topp_penalty_logprobs" in source
    assert "VLLM_L20_TOP_LOGPROBS=1" in source
    assert "VLLM_L20_TOP_LOGPROBS_TRACE" in source
    assert "VLLM_L20_TOP_LOGPROBS_ALLOW_NON_L20=1" in source
    assert "baseline-flashinfer-logprobs" in source
    assert "candidate-fused-top-logprobs" in source
    assert "l20-top-logprobs-trace.jsonl" in source
    assert "configure_flashinfer_cuda13_env(required=True)" in source
    assert "check_port_free" in source
    assert "Port $port is already serving something" in source
    assert "GPU has active compute apps" in source
    assert "flashinfer_cuda_env" in source
    assert '"ports"' in source
    assert "Both paths keep FlashInfer top-k/top-p sampling enabled" in source
    assert "dirty and clean A100" in readme
    assert "a100-vllm-top-logprobs-clean" in readme
    assert "a100-vllm-top-logprobs-smoke" in status
    assert "a100-vllm-top-logprobs-clean" in status
    assert "a100-vllm-top-logprobs-smoke" in results
    assert "a100-vllm-top-logprobs-clean" in results
    assert "Total events | 8" in smoke
    assert "Eligible fused events | 8" in smoke
    assert "Total events | 80" in clean
    assert "Eligible fused events | 80" in clean
    assert "not a serving win" in clean
