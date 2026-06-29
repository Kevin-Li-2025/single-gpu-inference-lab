import importlib.util
import json
from pathlib import Path


def load_scout_script():
    path = Path("scripts/scout_vllm_gemm_epilogue_boundary.py")
    spec = importlib.util.spec_from_file_location("scout_vllm_gemm_epilogue_boundary", path)
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
        root / "vllm/v1/worker/gpu_model_runner.py",
        """
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
sampler_output = self._sample(logits, spec_decode_metadata)
""",
    )
    write(
        root / "vllm/model_executor/layers/logits_processor.py",
        """
class LogitsProcessor:
    def _get_logits(self, hidden_states, lm_head, embedding_bias):
        logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
        logits = self._gather_logits(logits)
        return logits
    def get_top_tokens(self, lm_head, hidden_states, embedding_bias=None):
        pass
""",
    )
    write(
        root / "vllm/model_executor/layers/vocab_parallel_embedding.py",
        """
class VocabParallelEmbedding:
    pass
class ParallelLMHead:
    self.quant_config = quant_config
    def tie_weights(self, embed_tokens):
        pass
""",
    )
    write(
        root / "vllm/v1/sample/sampler.py",
        """
class Sampler:
    def forward(self, logits, sampling_metadata):
        logits = logits.to(torch.float32)
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        sampler_output = SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1))
        return sampler_output
""",
    )
    write(
        root / "vllm/v1/sample/ops/topk_topp_sampler.py",
        """
class TopKTopPSampler:
    def forward_cuda(self, logits, generators, k, p):
        return flashinfer_sample(logits.contiguous(), k, p, generators), None
""",
    )
    write(
        root / "vllm/lora/layers/logits_processor.py",
        """
class LogitsProcessorWithLoRA:
    logits = actual_lm_head.quant_method.apply(actual_lm_head, hidden_states)
""",
    )
    return root


def write_summaries(root: Path) -> tuple[Path, Path]:
    tile = root / "tile-summary.json"
    tile.write_text(
        json.dumps(
            {
                "decision": {
                    "batch1_default": {"block_vocab": 32, "block_hidden": 256},
                    "batched_default": {"block_vocab": 64, "block_hidden": 256},
                },
                "best_by_shape": {"b1-h1024-v151936-gumbel": {}},
            }
        ),
        encoding="utf-8",
    )
    serving = root / "serving-summary.json"
    serving.write_text(
        json.dumps(
            {
                "decision": "do_not_claim_serving_win",
                "reason": "standalone candidate regressed throughput",
                "metrics": {
                    "delta_pct": {
                        "output_throughput": -2.95,
                        "median_itl_ms": -1.11,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return tile, serving


def test_gemm_epilogue_scout_finds_upstream_boundaries(tmp_path):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")
    tile, serving = write_summaries(tmp_path)

    result = module.analyze(source, tile, serving)

    assert result["schema_version"] == 1
    assert result["complete"]
    assert result["upstream_api"]["ready_for_trace_pr"]
    assert result["upstream_api"]["proposed_owner"] == (
        "LogitsProcessor / ParallelLMHead, not TopKTopPSampler"
    )
    assert result["evidence"]["serving_decision"] == "do_not_claim_serving_win"
    plan = {item["step"]: item for item in result["implementation_plan"]}
    assert plan["keep standalone FlashSampling disabled"]["ready"]
    assert plan["prototype the GEMM epilogue behind LogitsProcessor"]["ready"]


def test_gemm_epilogue_scout_marks_missing_logits_processor_incomplete(tmp_path):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")
    (source / "vllm/model_executor/layers/logits_processor.py").unlink()

    result = module.analyze(source)

    by_id = {point["id"]: point for point in result["patch_points"]}
    assert not by_id["logits_processor_lm_head"]["exists"]
    assert not result["complete"]
    assert not result["upstream_api"]["ready_for_trace_pr"]


def test_gemm_epilogue_scout_markdown_contains_dirty_warning(tmp_path, monkeypatch):
    module = load_scout_script()
    source = make_fake_vllm(tmp_path / "vllm-src")

    monkeypatch.setattr(
        module,
        "source_metadata",
        lambda _: {
            "path": str(source),
            "branch": "main",
            "commit": "abc1234",
            "dirty": True,
            "status_line_count": 2,
            "l20_local_patch_present": True,
        },
    )
    result = module.analyze(source)
    rendered = module.render_markdown(result)

    assert "Local L20 patch present: `True`" in rendered
    assert "clean upstream checkout" in rendered
