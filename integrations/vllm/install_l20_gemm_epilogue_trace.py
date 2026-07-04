#!/usr/bin/env python3
"""Install a fallback-first LM-head GEMM epilogue trace hook into vLLM."""

from __future__ import annotations

import argparse
import inspect
import shutil
from collections.abc import Callable
from pathlib import Path

HELPER_SOURCE = Path(__file__).with_name("l20_gemm_epilogue_trace.py")
HELPER_NAME = "l20_gemm_epilogue_trace.py"
BACKUP_SUFFIX = ".l20-gemm-epilogue-trace-backup"

GEMM_IMPORT_LINE = (
    "from vllm.v1.worker.gpu.l20_gemm_epilogue_trace import "
    "maybe_take_l20_gemm_epilogue_sampler_output, "
    "maybe_try_l20_gemm_epilogue\n"
)
MODEL_RUNNER_IMPORT_MARKER = (
    "from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker\n"
)
GPU_MODEL_RUNNER_IMPORT_MARKER = (
    "from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch\n"
)

LOGITS_PROCESSOR_PATCH_POINT = """    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
"""

LOGITS_PROCESSOR_METHOD = '''    def try_sample_from_lm_head(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: object,
        embedding_bias: torch.Tensor | None = None,
    ) -> object | None:
        """Fallback-first sampled-token epilogue hook.

        Returning ``None`` preserves the existing ``forward``/``_get_logits``
        path. Future hardware-specific epilogues can override this boundary
        without changing sampler call sites.
        """
        return None

'''

NATIVE_SAMPLE_BLOCK = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
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

NATIVE_SAMPLE_BLOCK_PATCHED = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        sampler_output = maybe_try_l20_gemm_epilogue(
            self,
            input_batch,
            grammar_output,
            sample_hidden_states,
        )
        if sampler_output is None:
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

V2_COMPUTE_PATCH_POINT = """                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
"""

V2_COMPUTE_PATCHED = """                sample_hidden_states = hidden_states[logits_indices]
                self._l20_gemm_epilogue_sampler_output = maybe_try_l20_gemm_epilogue(
                    self,
                    self.input_batch,
                    None,
                    sample_hidden_states,
                    scheduler_output,
                    spec_decode_metadata,
                )
                if self._l20_gemm_epilogue_sampler_output is None:
                    logits = self.model.compute_logits(sample_hidden_states)
                else:
                    logits = None
"""

V2_SAMPLE_BLOCK = """        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(
"""

VLLM_0102_COMPUTE_PATCH_POINT = """                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states, None)
"""

VLLM_0102_COMPUTE_PATCHED = """                sample_hidden_states = hidden_states[logits_indices]
                self._l20_gemm_epilogue_sampler_output = maybe_try_l20_gemm_epilogue(
                    self,
                    self.input_batch,
                    None,
                    sample_hidden_states,
                    scheduler_output,
                    spec_decode_metadata,
                )
                if self._l20_gemm_epilogue_sampler_output is None:
                    logits = self.model.compute_logits(sample_hidden_states, None)
                else:
                    logits = None
"""

VLLM_0102_SAMPLE_BLOCK = """            # Apply structured output bitmasks if present
            if scheduler_output.grammar_bitmask is not None:
                self.apply_grammar_bitmask(scheduler_output, logits)

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        with record_function_or_nullcontext("Bookkeep"):
"""

VLLM_0102_SAMPLE_BLOCK_PATCHED = """            sampler_output = maybe_take_l20_gemm_epilogue_sampler_output(self)
            if sampler_output is None:
                # Apply structured output bitmasks if present
                if scheduler_output.grammar_bitmask is not None:
                    self.apply_grammar_bitmask(scheduler_output, logits)

                with record_function_or_nullcontext("Sample"):
                    sampler_output = self._sample(logits, spec_decode_metadata)

        with record_function_or_nullcontext("Bookkeep"):
"""

V2_SAMPLE_BLOCK_PATCHED = """        # Clear ephemeral state.
        self.execute_model_state = None

        sampler_output = maybe_take_l20_gemm_epilogue_sampler_output(self)
        if sampler_output is None:
            # Apply structured output bitmasks if present.
            if grammar_output is not None:
                apply_grammar_bitmask(
                    scheduler_output, grammar_output, self.input_batch, logits
                )

            with record_function_or_nullcontext("gpu_model_runner: sample"):
                sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(
"""

TOPK_TOPP_NATIVE_SIGNATURE = """    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""

TOPK_TOPP_NATIVE_SIGNATURE_PATCHED = """    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        **_: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def resolve_package(vllm_source: Path | None) -> Path:
    if vllm_source is not None:
        return vllm_source.expanduser().resolve() / "vllm"
    import vllm

    return Path(inspect.getfile(vllm)).parent


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + BACKUP_SUFFIX)


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"cannot find patch point: {label}")
    return source.replace(old, new, 1)


def ensure_import(source: str, marker: str, import_line: str, label: str) -> str:
    if import_line in source:
        return source
    return replace_once(source, marker, marker + import_line, label)


def patch_logits_processor(source: str) -> str:
    if "def try_sample_from_lm_head(" in source:
        return source
    return replace_once(
        source,
        LOGITS_PROCESSOR_PATCH_POINT,
        LOGITS_PROCESSOR_METHOD + LOGITS_PROCESSOR_PATCH_POINT,
        "LogitsProcessor.try_sample_from_lm_head",
    )


def patch_model_runner(source: str) -> str:
    source = ensure_import(
        source,
        MODEL_RUNNER_IMPORT_MARKER,
        GEMM_IMPORT_LINE,
        "model_runner GEMM epilogue import",
    )
    if "maybe_try_l20_gemm_epilogue(" in source:
        return source
    source = replace_once(
        source,
        NATIVE_SAMPLE_BLOCK,
        NATIVE_SAMPLE_BLOCK_PATCHED,
        "GPUModelRunner.sample GEMM epilogue call",
    )
    return source


def patch_gpu_model_runner(source: str) -> str:
    source = ensure_import(
        source,
        GPU_MODEL_RUNNER_IMPORT_MARKER,
        GEMM_IMPORT_LINE,
        "gpu_model_runner GEMM epilogue import",
    )
    if "maybe_try_l20_gemm_epilogue(" not in source:
        try:
            source = replace_once(
                source,
                V2_COMPUTE_PATCH_POINT,
                V2_COMPUTE_PATCHED,
                "gpu_model_runner GEMM epilogue compute",
            )
        except RuntimeError:
            source = replace_once(
                source,
                VLLM_0102_COMPUTE_PATCH_POINT,
                VLLM_0102_COMPUTE_PATCHED,
                "gpu_model_runner GEMM epilogue compute vllm 0.10.2",
            )
    if "maybe_take_l20_gemm_epilogue_sampler_output(" in source:
        return source
    try:
        source = replace_once(
            source,
            V2_SAMPLE_BLOCK,
            V2_SAMPLE_BLOCK_PATCHED,
            "gpu_model_runner GEMM epilogue sample output",
        )
    except RuntimeError:
        source = replace_once(
            source,
            VLLM_0102_SAMPLE_BLOCK,
            VLLM_0102_SAMPLE_BLOCK_PATCHED,
            "gpu_model_runner GEMM epilogue sample output vllm 0.10.2",
        )
    return source


def patch_topk_topp_sampler(source: str) -> str:
    """Allow L20-only sampler metadata to reach the native fallback path.

    Some local vLLM RFC trees pass ``l20_expanded_idx_mapping`` and related
    kwargs through ``TopKTopPSampler``. When FlashInfer is disabled for a
    server smoke, the native fallback must ignore those kwargs instead of
    failing before the GEMM epilogue path can be exercised.
    """

    return replace_once(
        source,
        TOPK_TOPP_NATIVE_SIGNATURE,
        TOPK_TOPP_NATIVE_SIGNATURE_PATCHED,
        "TopKTopPSampler.forward_native L20 kwargs compatibility",
    )


def _patch_file(path: Path, patcher: Callable[[str], str]) -> None:
    if not path.exists():
        return
    original = path.read_text(encoding="utf-8")
    patched = patcher(original)
    if patched == original:
        return
    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")


def install(package: Path) -> None:
    if not HELPER_SOURCE.exists():
        raise RuntimeError(f"missing helper source: {HELPER_SOURCE}")
    helper_target = package / "v1" / "worker" / "gpu" / HELPER_NAME
    helper_target.parent.mkdir(parents=True, exist_ok=True)
    init_file = helper_target.parent / "__init__.py"
    init_file.touch(exist_ok=True)
    if helper_target.exists() and helper_target.read_bytes() != HELPER_SOURCE.read_bytes():
        backup = _backup_path(helper_target)
        if not backup.exists():
            shutil.copy2(helper_target, backup)
    shutil.copy2(HELPER_SOURCE, helper_target)
    _patch_file(
        package / "model_executor" / "layers" / "logits_processor.py",
        patch_logits_processor,
    )
    _patch_file(
        package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        patch_topk_topp_sampler,
    )
    _patch_file(package / "v1" / "worker" / "gpu" / "model_runner.py", patch_model_runner)
    _patch_file(package / "v1" / "worker" / "gpu_model_runner.py", patch_gpu_model_runner)


def uninstall(package: Path) -> None:
    for target in (
        package / "model_executor" / "layers" / "logits_processor.py",
        package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        package / "v1" / "worker" / "gpu" / "model_runner.py",
        package / "v1" / "worker" / "gpu_model_runner.py",
    ):
        backup = _backup_path(target)
        if backup.exists():
            shutil.copy2(backup, target)
            backup.unlink()
    helper_target = package / "v1" / "worker" / "gpu" / HELPER_NAME
    helper_backup = _backup_path(helper_target)
    if helper_backup.exists():
        shutil.copy2(helper_backup, helper_target)
        helper_backup.unlink()
    else:
        helper_target.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    package = resolve_package(args.vllm_source)
    if args.uninstall:
        uninstall(package)
    else:
        install(package)
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
