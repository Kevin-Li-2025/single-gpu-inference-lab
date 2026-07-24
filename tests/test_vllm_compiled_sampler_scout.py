import importlib.util
import json
from pathlib import Path


def load_scout_script():
    path = Path("scripts/scout_vllm_compiled_sampler_boundary.py")
    spec = importlib.util.spec_from_file_location(
        "scout_vllm_compiled_sampler_boundary", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_fake_vllm(root: Path) -> Path:
    write(
        root / "vllm/v1/worker/gpu_model_runner.py",
        """
sample_hidden_states = hidden_states[self.input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
sampler_output = self.sampler(logits, self.input_batch)
""",
    )
    write(
        root / "vllm/v1/worker/gpu/sample/sampler.py",
        """
logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
self.sampling_states.apply_temperature(logits, expanded_idx_mapping, idx_mapping_np)
self.sampling_states.apply_min_p(logits, expanded_idx_mapping, idx_mapping_np)
processed_logits = self.apply_sampling_params(
top_k, top_p = self.sampling_states.get_top_k_top_p(
sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
sampled = gumbel_sample(
""",
    )
    write(
        root / "vllm/v1/sample/sampler.py",
        """
random_sampled, processed_logprobs = self.topk_topp_sampler(
    logits,
    sampling_metadata.generators,
    sampling_metadata.top_k,
    sampling_metadata.top_p,
)
""",
    )
    write(
        root / "vllm/v1/sample/metadata.py",
        """
class SamplingMetadata:
    top_p: torch.Tensor | None
    top_k: torch.Tensor | None
    generators: dict[int, torch.Generator]
""",
    )
    write(
        root / "vllm/v1/worker/gpu/sample/states.py",
        """
do_top_k = np.any(self.top_k.np[idx_mapping_np] != self.vocab_size)
top_k = self.top_k.gpu[expanded_idx_mapping] if do_top_k else None
def any_explicit_seed(self, idx_mapping_np):
    pass
""",
    )
    write(
        root / "vllm/v1/worker/gpu/sample/gumbel.py",
        """
def gumbel_sample(logits, expanded_idx_mapping, temperature, seed: torch.Tensor, pos):
    gumbel_seed = tl.randint(seed, pos)
    _gumbel_sample_kernel[(num_tokens, num_blocks)](
""",
    )
    write(
        root / "vllm/v1/sample/ops/topk_topp_sampler.py",
        """
def flashinfer_sample(logits, k, p, generators={}):
    return flashinfer_sample(logits.contiguous(), k, p, generators), None
def forward_cuda(self, logits, generators, k, p):
    return flashinfer_sample(logits.contiguous(), k, p, generators), None
def flashinfer_sample(logits, k, p, generators={}):
    return flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits, k, p, deterministic=True)
""",
    )
    write(
        root / "vllm/model_executor/layers/logits_processor.py",
        """
class LogitsProcessor:
    def _get_logits(self):
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = self._gather_logits(logits)
""",
    )
    return root


def write_serving_summary(path: Path) -> Path:
    payload = {
        "model": "Qwen2.5-Coder-1.5B-Instruct",
        "shape": {"input_tokens": 512, "output_tokens": 32},
        "deltas": {
            "l20_notrace": {
                "c1": {
                    "median_itl_pct": 32.36,
                    "output_throughput_pct": -21.70,
                }
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compiled_sampler_scout_finds_required_boundaries(tmp_path):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")
    summary = write_serving_summary(tmp_path / "serving-summary.json")

    result = module.analyze(source, summary)

    assert result["schema_version"] == 1
    assert result["complete"]
    assert result["serving_evidence"]["evidence_status"] == "superseded_semantics"
    assert not result["serving_evidence"]["performance_comparable"]
    assert not result["rng_metadata_gap"]["stateful_sampler_ready"]
    plan = {item["step"]: item for item in result["implementation_plan"]}
    assert plan["keep the standalone L20 sampler disabled pending revalidation"]["ready"]
    assert not plan["build a state-preserving compiled sampler prototype"]["ready"]
    assert plan["prototype a logits/LM-head epilogue boundary"]["ready"]


def test_compiled_sampler_scout_marks_missing_sampler_path_incomplete(tmp_path):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")
    (source / "vllm/v1/worker/gpu/sample/sampler.py").unlink()

    result = module.analyze(source, None)

    by_id = {point["id"]: point for point in result["patch_points"]}
    assert not by_id["worker_gpu_sampler_flashinfer_branch"]["exists"]
    assert not by_id["worker_gpu_sampler_flashinfer_branch"]["complete"]
    plan = {item["step"]: item for item in result["implementation_plan"]}
    assert not plan["build a state-preserving compiled sampler prototype"]["ready"]
