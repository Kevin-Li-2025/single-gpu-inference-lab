import importlib.util
from pathlib import Path

import pytest


LOGITS_PROCESSOR_SOURCE = """import torch

from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

class LogitsProcessor:
    def forward(self, lm_head, hidden_states, embedding_bias=None):
        return self._get_logits(hidden_states, lm_head, embedding_bias)

    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits
"""


MODEL_RUNNER_SOURCE = """from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

class GPUModelRunner:
    def sample(self, hidden_states, input_batch, grammar_output):
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )

        return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected
"""


GPU_MODEL_RUNNER_SOURCE = """from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

class GPUModelRunner:
    def execute_model(self, hidden_states, logits_indices, scheduler_output, spec_decode_metadata):
        if True:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
        self.execute_model_state = (scheduler_output, logits, spec_decode_metadata, None, hidden_states, sample_hidden_states)

    def sample_tokens(self, grammar_output):
        scheduler_output, logits, spec_decode_metadata, _, hidden_states, sample_hidden_states = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )
        return sampler_output
"""

GPU_MODEL_RUNNER_0102_SOURCE = """from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

class GPUModelRunner:
    def execute_model(self, hidden_states, logits_indices, scheduler_output, spec_decode_metadata):
        if True:
                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        with record_function_or_nullcontext("Bookkeep"):
            self._bookkeeping_sync(scheduler_output, sampler_output, logits)
        return sampler_output
"""


HELPER_SOURCE = """def maybe_try_l20_gemm_epilogue(*args, **kwargs):
    return None

def maybe_take_l20_gemm_epilogue_sampler_output(*args, **kwargs):
    return None
"""


def load_installer():
    path = Path("integrations/vllm/install_l20_gemm_epilogue_trace.py")
    spec = importlib.util.spec_from_file_location("install_l20_gemm_epilogue_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_package(tmp_path: Path):
    package = tmp_path / "vllm"
    logits_processor = package / "model_executor/layers/logits_processor.py"
    model_runner = package / "v1/worker/gpu/model_runner.py"
    gpu_model_runner = package / "v1/worker/gpu_model_runner.py"
    logits_processor.parent.mkdir(parents=True, exist_ok=True)
    model_runner.parent.mkdir(parents=True, exist_ok=True)
    gpu_model_runner.parent.mkdir(parents=True, exist_ok=True)
    logits_processor.write_text(LOGITS_PROCESSOR_SOURCE, encoding="utf-8")
    model_runner.write_text(MODEL_RUNNER_SOURCE, encoding="utf-8")
    gpu_model_runner.write_text(GPU_MODEL_RUNNER_SOURCE, encoding="utf-8")
    return package, logits_processor, model_runner, gpu_model_runner


def test_gemm_epilogue_installer_patches_idempotently_and_uninstalls(tmp_path):
    installer = load_installer()
    package, logits_processor, model_runner, gpu_model_runner = write_package(tmp_path)
    helper_source = tmp_path / "l20_gemm_epilogue_trace.py"
    helper_source.write_text(HELPER_SOURCE, encoding="utf-8")
    installer.HELPER_SOURCE = helper_source

    installer.install(package)

    logits_text = logits_processor.read_text(encoding="utf-8")
    model_text = model_runner.read_text(encoding="utf-8")
    gpu_text = gpu_model_runner.read_text(encoding="utf-8")
    assert "def try_sample_from_lm_head(" in logits_text
    assert "maybe_try_l20_gemm_epilogue(" in model_text
    assert "if sampler_output is None:" in model_text
    assert "maybe_try_l20_gemm_epilogue(" in gpu_text
    assert "maybe_take_l20_gemm_epilogue_sampler_output(self)" in gpu_text
    assert (package / "v1/worker/gpu/l20_gemm_epilogue_trace.py").exists()

    installer.install(package)
    assert logits_processor.read_text(encoding="utf-8") == logits_text
    assert model_runner.read_text(encoding="utf-8") == model_text
    assert gpu_model_runner.read_text(encoding="utf-8") == gpu_text

    installer.uninstall(package)
    assert logits_processor.read_text(encoding="utf-8") == LOGITS_PROCESSOR_SOURCE
    assert model_runner.read_text(encoding="utf-8") == MODEL_RUNNER_SOURCE
    assert gpu_model_runner.read_text(encoding="utf-8") == GPU_MODEL_RUNNER_SOURCE
    assert not (package / "v1/worker/gpu/l20_gemm_epilogue_trace.py").exists()
    assert not list(package.rglob("*.l20-gemm-epilogue-trace-backup"))


def test_gemm_epilogue_installer_requires_helper(tmp_path):
    installer = load_installer()
    package, logits_processor, model_runner, gpu_model_runner = write_package(tmp_path)
    installer.HELPER_SOURCE = tmp_path / "missing.py"

    with pytest.raises(RuntimeError, match="missing helper source"):
        installer.install(package)

    assert logits_processor.read_text(encoding="utf-8") == LOGITS_PROCESSOR_SOURCE
    assert model_runner.read_text(encoding="utf-8") == MODEL_RUNNER_SOURCE
    assert gpu_model_runner.read_text(encoding="utf-8") == GPU_MODEL_RUNNER_SOURCE


def test_gemm_epilogue_installer_supports_vllm_0102_gpu_model_runner(tmp_path):
    installer = load_installer()
    package = tmp_path / "vllm"
    gpu_model_runner = package / "v1/worker/gpu_model_runner.py"
    gpu_model_runner.parent.mkdir(parents=True, exist_ok=True)
    gpu_model_runner.write_text(GPU_MODEL_RUNNER_0102_SOURCE, encoding="utf-8")
    helper_source = tmp_path / "l20_gemm_epilogue_trace.py"
    helper_source.write_text(HELPER_SOURCE, encoding="utf-8")
    installer.HELPER_SOURCE = helper_source

    installer.install(package)

    patched = gpu_model_runner.read_text(encoding="utf-8")
    assert "maybe_try_l20_gemm_epilogue(" in patched
    assert "self.model.compute_logits(sample_hidden_states, None)" in patched
    assert "maybe_take_l20_gemm_epilogue_sampler_output(self)" in patched
    assert "if sampler_output is None:" in patched
    assert (package / "v1/worker/gpu/__init__.py").exists()
    assert (package / "v1/worker/gpu/l20_gemm_epilogue_trace.py").exists()

    installer.install(package)
    assert gpu_model_runner.read_text(encoding="utf-8") == patched

    installer.uninstall(package)
    assert gpu_model_runner.read_text(encoding="utf-8") == GPU_MODEL_RUNNER_0102_SOURCE
    assert not (package / "v1/worker/gpu/l20_gemm_epilogue_trace.py").exists()
