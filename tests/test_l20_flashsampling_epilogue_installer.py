import importlib.util
from pathlib import Path

import pytest


MODEL_RUNNER_SOURCE = """from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

class GPUModelRunner:
    def sample(self, hidden_states, input_batch, grammar_output):
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            pass
        return self.sampler(logits, input_batch)
"""

V2_MODEL_RUNNER_SOURCE = """from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

class GPUModelRunner:
    def sample_tokens(self, grammar_output):
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            pass
        return self._sample(logits, spec_decode_metadata)
"""

FLASH_HELPER_SOURCE = """def maybe_trace_l20_flashsampling_epilogue(*args, **kwargs):
    return {"eligible": False, "reasons": ["synthetic_test_helper"]}
"""


def load_installer():
    path = Path("integrations/vllm/install_l20_flashsampling_epilogue_trace.py")
    spec = importlib.util.spec_from_file_location(
        "install_l20_flashsampling_epilogue_trace",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_vllm_package(tmp_path):
    package = tmp_path / "vllm"
    target = package / "v1/worker/gpu/model_runner.py"
    target.parent.mkdir(parents=True)
    target.write_text(MODEL_RUNNER_SOURCE, encoding="utf-8")
    v2_target = package / "v1/worker/gpu_model_runner.py"
    v2_target.parent.mkdir(parents=True, exist_ok=True)
    v2_target.write_text(V2_MODEL_RUNNER_SOURCE, encoding="utf-8")
    return package, target, v2_target


def install_with_synthetic_flash_helper(installer, package, tmp_path):
    helper_source = tmp_path / "l20_flashsampling_epilogue.py"
    helper_source.write_text(FLASH_HELPER_SOURCE, encoding="utf-8")
    installer.FLASH_SAMPLING_HELPER_SOURCE = helper_source
    installer.install(package)
    return helper_source


def test_flashsampling_installer_patches_copies_and_uninstalls(tmp_path):
    installer = load_installer()
    package, target, v2_target = write_vllm_package(tmp_path)

    install_with_synthetic_flash_helper(installer, package, tmp_path)

    patched = target.read_text(encoding="utf-8")
    v2_patched = v2_target.read_text(encoding="utf-8")
    assert patched.count("maybe_trace_l20_logits_boundary(") == 1
    assert patched.count("maybe_trace_l20_flashsampling_epilogue(") == 1
    assert v2_patched.count("maybe_trace_l20_logits_boundary(") == 1
    assert v2_patched.count("maybe_trace_l20_flashsampling_epilogue(") == 1
    assert (
        "logits = self.model.compute_logits(sample_hidden_states)\n        maybe_trace"
        in patched
    )
    assert "maybe_trace_l20_flashsampling_epilogue(\n            self," in patched
    assert "scheduler_output,\n        )\n\n        maybe_trace" in v2_patched
    assert "        logits =" not in patched.split(
        "maybe_trace_l20_flashsampling_epilogue(",
        1,
    )[1]

    helper_dir = package / "v1/worker/gpu"
    assert (helper_dir / "l20_logits_boundary_trace.py").exists()
    assert (
        helper_dir / "l20_flashsampling_epilogue.py"
    ).read_text(encoding="utf-8") == FLASH_HELPER_SOURCE

    installer.install(package)
    assert target.read_text(encoding="utf-8") == patched
    assert v2_target.read_text(encoding="utf-8") == v2_patched

    installer.uninstall(package)
    assert target.read_text(encoding="utf-8") == MODEL_RUNNER_SOURCE
    assert v2_target.read_text(encoding="utf-8") == V2_MODEL_RUNNER_SOURCE
    assert not (helper_dir / "l20_logits_boundary_trace.py").exists()
    assert not (helper_dir / "l20_flashsampling_epilogue.py").exists()


def test_flashsampling_installer_composes_with_existing_logits_trace_patch():
    installer = load_installer()
    trace_model_runner = installer.patch_model_runner(MODEL_RUNNER_SOURCE).replace(
        installer.SAMPLE_FLASHSAMPLING_CALL,
        "",
    )
    trace_v2_runner = installer.patch_gpu_model_runner(V2_MODEL_RUNNER_SOURCE).replace(
        "\n" + installer.V2_FLASHSAMPLING_CALL,
        "",
    )

    patched = installer.patch_model_runner(trace_model_runner)
    v2_patched = installer.patch_gpu_model_runner(trace_v2_runner)

    assert patched.count("maybe_trace_l20_logits_boundary(") == 1
    assert patched.count("maybe_trace_l20_flashsampling_epilogue(") == 1
    assert v2_patched.count("maybe_trace_l20_logits_boundary(") == 1
    assert v2_patched.count("maybe_trace_l20_flashsampling_epilogue(") == 1
    assert installer.patch_model_runner(patched) == patched
    assert installer.patch_gpu_model_runner(v2_patched) == v2_patched


def test_flashsampling_installer_restores_preexisting_helpers(tmp_path):
    installer = load_installer()
    package, _, _ = write_vllm_package(tmp_path)
    helper_dir = package / "v1/worker/gpu"
    trace_helper = helper_dir / "l20_logits_boundary_trace.py"
    flash_helper = helper_dir / "l20_flashsampling_epilogue.py"
    trace_helper.write_text("# preexisting trace helper\n", encoding="utf-8")
    flash_helper.write_text("# preexisting flash helper\n", encoding="utf-8")

    install_with_synthetic_flash_helper(installer, package, tmp_path)
    assert trace_helper.read_text(encoding="utf-8") != "# preexisting trace helper\n"
    assert flash_helper.read_text(encoding="utf-8") == FLASH_HELPER_SOURCE

    installer.uninstall(package)
    assert trace_helper.read_text(encoding="utf-8") == "# preexisting trace helper\n"
    assert flash_helper.read_text(encoding="utf-8") == "# preexisting flash helper\n"


def test_flashsampling_installer_requires_helper_before_patching(tmp_path):
    installer = load_installer()
    package, target, v2_target = write_vllm_package(tmp_path)
    installer.FLASH_SAMPLING_HELPER_SOURCE = tmp_path / "missing_flash_helper.py"

    with pytest.raises(RuntimeError, match="missing helper source"):
        installer.install(package)

    assert target.read_text(encoding="utf-8") == MODEL_RUNNER_SOURCE
    assert v2_target.read_text(encoding="utf-8") == V2_MODEL_RUNNER_SOURCE
    assert not (package / "v1/worker/gpu/l20_logits_boundary_trace.py").exists()
    assert not (package / "v1/worker/gpu/l20_flashsampling_epilogue.py").exists()
