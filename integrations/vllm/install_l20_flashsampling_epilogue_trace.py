#!/usr/bin/env python3
"""Install a behavior-preserving L20 FlashSampling shadow hook into vLLM.

The installer composes with the logits-boundary trace patch points. It copies
the logits-boundary trace helper plus the FlashSampling epilogue planner helper
into the vLLM GPU worker package, then adds shadow-only calls after logits are
materialized. The inserted calls must not mutate logits, token ids, or sampler
state.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
from collections.abc import Callable
from pathlib import Path


LOGITS_TRACE_HELPER_SOURCE = Path(__file__).with_name("l20_logits_boundary_trace.py")
FLASH_SAMPLING_HELPER_SOURCE = Path(__file__).with_name(
    "l20_flashsampling_epilogue.py"
)

LOGITS_IMPORT_LINE = (
    "from vllm.v1.worker.gpu.l20_logits_boundary_trace import "
    "maybe_trace_l20_logits_boundary\n"
)
FLASH_IMPORT_LINE = (
    "from vllm.v1.worker.gpu.l20_flashsampling_epilogue import "
    "maybe_trace_l20_flashsampling_epilogue\n"
)

IMPORT_MARKER = (
    "from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker\n"
)
V2_IMPORT_MARKER = (
    "from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch\n"
)

SAMPLE_PATCH_POINT = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
"""

SAMPLE_LOGITS_TRACE_CALL = """        maybe_trace_l20_logits_boundary(
            self,
            input_batch,
            grammar_output,
            sample_hidden_states,
            logits,
        )
"""

SAMPLE_FLASHSAMPLING_CALL = """        maybe_trace_l20_flashsampling_epilogue(
            self,
            input_batch,
            grammar_output,
            sample_hidden_states,
            logits,
        )
"""

SAMPLE_PATCHED = (
    """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
"""
    + SAMPLE_LOGITS_TRACE_CALL
    + SAMPLE_FLASHSAMPLING_CALL
    + """        if grammar_output is not None:
"""
)

V2_SAMPLE_PATCH_POINT = """        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
"""

V2_LOGITS_TRACE_CALL = """        maybe_trace_l20_logits_boundary(
            self,
            self.input_batch,
            grammar_output,
            sample_hidden_states,
            logits,
            scheduler_output,
        )
"""

V2_FLASHSAMPLING_CALL = """        maybe_trace_l20_flashsampling_epilogue(
            self,
            self.input_batch,
            grammar_output,
            sample_hidden_states,
            logits,
            scheduler_output,
        )
"""

V2_SAMPLE_PATCHED = (
    """        # Clear ephemeral state.
        self.execute_model_state = None

"""
    + V2_LOGITS_TRACE_CALL
    + "\n"
    + V2_FLASHSAMPLING_CALL
    + """
        # Apply structured output bitmasks if present.
        if grammar_output is not None:
"""
)

BACKUP_SUFFIX = ".l20-flashsampling-epilogue-trace-backup"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="Path to a vLLM source checkout root. Defaults to the imported package.",
    )
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def resolve_package(vllm_source: Path | None) -> Path:
    if vllm_source is not None:
        return vllm_source.expanduser().resolve() / "vllm"
    import vllm

    return Path(inspect.getfile(vllm)).parent


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


def patch_model_runner(source: str) -> str:
    source = ensure_import(
        source,
        IMPORT_MARKER,
        FLASH_IMPORT_LINE,
        "model_runner FlashSampling import",
    )
    source = ensure_import(
        source,
        IMPORT_MARKER,
        LOGITS_IMPORT_LINE,
        "model_runner logits-boundary import",
    )
    if SAMPLE_FLASHSAMPLING_CALL in source:
        return source
    if SAMPLE_LOGITS_TRACE_CALL in source:
        return replace_once(
            source,
            SAMPLE_LOGITS_TRACE_CALL,
            SAMPLE_LOGITS_TRACE_CALL + SAMPLE_FLASHSAMPLING_CALL,
            "GPUModelRunner.sample FlashSampling shadow after logits trace",
        )
    return replace_once(
        source,
        SAMPLE_PATCH_POINT,
        SAMPLE_PATCHED,
        "GPUModelRunner.sample FlashSampling shadow",
    )


def patch_gpu_model_runner(source: str) -> str:
    source = ensure_import(
        source,
        V2_IMPORT_MARKER,
        FLASH_IMPORT_LINE,
        "v2 gpu_model_runner FlashSampling import",
    )
    source = ensure_import(
        source,
        V2_IMPORT_MARKER,
        LOGITS_IMPORT_LINE,
        "v2 gpu_model_runner logits-boundary import",
    )
    if V2_FLASHSAMPLING_CALL in source:
        return source
    if V2_LOGITS_TRACE_CALL in source:
        return replace_once(
            source,
            V2_LOGITS_TRACE_CALL,
            V2_LOGITS_TRACE_CALL + "\n" + V2_FLASHSAMPLING_CALL,
            "GPUModelRunner.sample_tokens FlashSampling shadow after logits trace",
        )
    return replace_once(
        source,
        V2_SAMPLE_PATCH_POINT,
        V2_SAMPLE_PATCHED,
        "GPUModelRunner.sample_tokens FlashSampling shadow",
    )


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + BACKUP_SUFFIX)


def _helper_targets(package: Path) -> list[tuple[Path, Path]]:
    helper_dir = package / "v1" / "worker" / "gpu"
    return [
        (LOGITS_TRACE_HELPER_SOURCE, helper_dir / "l20_logits_boundary_trace.py"),
        (FLASH_SAMPLING_HELPER_SOURCE, helper_dir / "l20_flashsampling_epilogue.py"),
    ]


def _check_helper_sources(package: Path) -> None:
    missing = [
        str(source)
        for source, _ in _helper_targets(package)
        if not source.exists()
    ]
    if missing:
        raise RuntimeError("missing helper source(s): " + ", ".join(missing))


def _copy_helper(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        source_bytes = source.read_bytes()
        if destination.read_bytes() == source_bytes:
            return
        backup = _backup_path(destination)
        if not backup.exists():
            shutil.copy2(destination, backup)
    shutil.copy2(source, destination)


def _restore_helper(destination: Path) -> None:
    backup = _backup_path(destination)
    if backup.exists():
        shutil.copy2(backup, destination)
    else:
        destination.unlink(missing_ok=True)


def _existing_targets(
    package: Path,
) -> list[tuple[Path, Callable[[str], str]]]:
    targets = [
        (package / "v1" / "worker" / "gpu" / "model_runner.py", patch_model_runner),
        (package / "v1" / "worker" / "gpu_model_runner.py", patch_gpu_model_runner),
    ]
    return [(path, patcher) for path, patcher in targets if path.exists()]


def _write_target(path: Path, original: str, patched: str) -> None:
    if patched == original:
        return
    backup = _backup_path(path)
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched, encoding="utf-8")


def _restore_target(path: Path) -> None:
    backup = _backup_path(path)
    if backup.exists():
        shutil.copy2(backup, path)


def install(package: Path) -> None:
    _check_helper_sources(package)
    targets = _existing_targets(package)
    if not targets:
        raise RuntimeError(f"missing supported vLLM model runner under: {package}")

    patched_targets = []
    for path, patcher in targets:
        original = path.read_text(encoding="utf-8")
        patched_targets.append((path, original, patcher(original)))

    for source, destination in _helper_targets(package):
        _copy_helper(source, destination)
    for path, original, patched in patched_targets:
        _write_target(path, original, patched)


def uninstall(package: Path) -> None:
    target_paths = [
        package / "v1" / "worker" / "gpu" / "model_runner.py",
        package / "v1" / "worker" / "gpu_model_runner.py",
    ]
    for target in target_paths:
        _restore_target(target)
    for _, destination in _helper_targets(package):
        _restore_helper(destination)


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
