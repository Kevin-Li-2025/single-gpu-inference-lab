import importlib.util
from pathlib import Path

import pytest


V2_MODEL_RUNNER_SOURCE = """from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

class GPUModelRunner:
    def execute(self, hidden_states, logits_indices, scheduler_output, spec_decode_metadata):
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
        return sampler_output
"""


HELPER_SOURCE = """def maybe_l20_flashsampling_compute_logits_or_sample(*args, **kwargs):
    return args[-1](args[4])

def maybe_take_l20_flashsampling_sampler_output(*args, **kwargs):
    return None
"""


def load_installer():
    path = Path("integrations/vllm/install_l20_flashsampling_epilogue_candidate.py")
    spec = importlib.util.spec_from_file_location(
        "install_l20_flashsampling_epilogue_candidate", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_package(tmp_path):
    package = tmp_path / "vllm"
    target = package / "v1/worker/gpu_model_runner.py"
    target.parent.mkdir(parents=True)
    target.write_text(V2_MODEL_RUNNER_SOURCE, encoding="utf-8")
    return package, target


def test_candidate_installer_patches_compute_and_sample_then_uninstalls(tmp_path):
    installer = load_installer()
    package, target = write_package(tmp_path)
    helper = tmp_path / "l20_flashsampling_candidate.py"
    helper.write_text(HELPER_SOURCE, encoding="utf-8")
    installer.HELPER_SOURCE = helper

    installer.install(package)

    patched = target.read_text(encoding="utf-8")
    assert "maybe_l20_flashsampling_compute_logits_or_sample" in patched
    assert "maybe_take_l20_flashsampling_sampler_output" in patched
    assert "self.model.compute_logits," in patched
    assert "if sampler_output is None:" in patched
    assert (package / "v1/worker/gpu/l20_flashsampling_candidate.py").exists()

    installer.install(package)
    assert target.read_text(encoding="utf-8") == patched

    installer.uninstall(package)
    assert target.read_text(encoding="utf-8") == V2_MODEL_RUNNER_SOURCE
    assert not (package / "v1/worker/gpu/l20_flashsampling_candidate.py").exists()


def test_candidate_installer_requires_helper(tmp_path):
    installer = load_installer()
    package, target = write_package(tmp_path)
    installer.HELPER_SOURCE = tmp_path / "missing.py"

    with pytest.raises(RuntimeError, match="missing helper source"):
        installer.install(package)

    assert target.read_text(encoding="utf-8") == V2_MODEL_RUNNER_SOURCE
