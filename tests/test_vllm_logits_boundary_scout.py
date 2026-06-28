import importlib.util
import json
from pathlib import Path


def load_scout_script():
    path = Path("scripts/scout_vllm_logits_boundary.py")
    spec = importlib.util.spec_from_file_location("scout_vllm_logits_boundary", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_fake_vllm(root: Path) -> Path:
    write(
        root / "vllm/v1/worker/gpu/model_runner.py",
        """
sample_hidden_states = hidden_states[input_batch.logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
sampler_output = self.sampler(logits, input_batch)
""",
    )
    write(
        root / "vllm/model_executor/layers/logits_processor.py",
        """
class LogitsProcessor:
    def _get_logits(self):
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = self._gather_logits(logits)
    def get_top_tokens(
        pass
""",
    )
    write(
        root / "vllm/v1/worker/gpu/sample/sampler.py",
        """
logits = torch.empty_like(logits, dtype=torch.float32).copy_(logits)
self.sampling_states.apply_temperature(logits, expanded_idx_mapping, idx_mapping_np)
self.sampling_states.apply_top_k_top_p(logits, expanded_idx_mapping, idx_mapping_np)
sampled = gumbel_sample(processed_logits)
""",
    )
    write(
        root / "vllm/v1/sample/sampler.py",
        """
logits = logits.to(torch.float32)
greedy_sampled = self.greedy_sample(logits)
logits = self.apply_temperature(logits, sampling_metadata.temperature, sampling_metadata.all_random)
sampled, processed_logprobs = self.sample(logits, sampling_metadata)
""",
    )
    write(
        root / "vllm/v1/sample/ops/topk_topp_sampler.py",
        """
class TopKTopPSampler:
    logger.info_once("Using FlashInfer for top-p & top-k sampling.")
    return flashinfer_sample(logits.contiguous(), k, p, generators), None
""",
    )
    write(
        root / "vllm/model_executor/layers/vocab_parallel_embedding.py",
        """
class VocabParallelEmbedding(PluggableLayer):
    self.quant_method: QuantizeMethodBase = quant_method
class ParallelLMHead(VocabParallelEmbedding):
    raise RuntimeError("LMHead's weights should be used in the sampler.")
""",
    )
    return root


def test_scout_finds_all_patch_points(tmp_path):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")
    result = module.analyze(source, None)
    assert result["schema_version"] == 1
    assert all(point["complete"] for point in result["patch_points"])
    plan = {item["step"]: item for item in result["implementation_plan"]}
    assert plan["add a guarded logits-boundary API before writing a kernel"]["ready"]
    assert plan["keep the first gate narrow"]["ready"]
    assert "no requested token logprobs" in "\n".join(result["first_gate"])


def test_scout_marks_missing_patch_point_incomplete(tmp_path):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")
    (source / "vllm/v1/worker/gpu/model_runner.py").unlink()
    result = module.analyze(source, None)
    by_id = {point["id"]: point for point in result["patch_points"]}
    assert not by_id["gpu_model_runner_logits_to_sampler"]["exists"]
    assert not by_id["gpu_model_runner_logits_to_sampler"]["complete"]
    plan = {item["step"]: item for item in result["implementation_plan"]}
    assert not plan["add a guarded logits-boundary API before writing a kernel"]["ready"]


def test_scout_loads_ceiling_summary(tmp_path):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text(
        json.dumps(
            {
                "recommendations": [
                    {"priority": "P0", "target": "production GEMM/GEMV epilogue"},
                    {"priority": "Stop", "target": "standalone sampling kernels"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = module.analyze(source, ceiling)
    assert result["ceiling"]["recommendation_count"] == 2
    assert result["ceiling"]["p0_targets"][0]["target"] == "production GEMM/GEMV epilogue"
